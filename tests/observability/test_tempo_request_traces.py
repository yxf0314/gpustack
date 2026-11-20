import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.observability.request_traces import (
    TracePagePosition,
    get_request_trace,
    search_request_trace_page,
    search_request_traces,
)

_REQUEST = "a0000000-0000-4000-8000-000000000001"
_OTHER_REQUEST = "a0000000-0000-4000-8000-000000000002"
_TRACE = "ab000000000000000000000000000001"
_NS = 1_000_000_000


def _attributes(**values):
    keys = {
        str: "stringValue",
        int: "intValue",
        bool: "boolValue",
        float: "doubleValue",
    }
    return [{"key": k, "value": {keys[type(v)]: v}} for k, v in values.items()]


def _span(service, id, parent, name, start, duration, **attrs):
    span = {
        "spanId": base64.b64encode(id.to_bytes(8, "big")).decode(),
        "name": name,
        "startTimeUnixNano": str(start * _NS),
        "endTimeUnixNano": str((start + duration) * _NS),
        "attributes": _attributes(**attrs),
    }
    if parent:
        span["parentSpanId"] = base64.b64encode(parent.to_bytes(8, "big")).decode()
    return {
        "resource": {"attributes": _attributes(**{"service.name": service})},
        "scopeSpans": [{"spans": [span]}],
    }


def _client(body, status=200):
    async def read(_size):
        yield json.dumps(body).encode()

    client = MagicMock()
    client.get.return_value.__aenter__.return_value = SimpleNamespace(
        status=status, content=SimpleNamespace(iter_chunked=read)
    )
    return client


@pytest.mark.asyncio
async def test_detail_follows_only_the_selected_request_branch():
    batches = [
        _span("external", 1, None, "client", 90, 100),
        _span(
            "gpustack.higress",
            2,
            1,
            "egress",
            100,
            4,
            **{"gpustack.request_id": _REQUEST, "http.status_code": "200"},
        ),
        _span(
            "gpustack.higress",
            3,
            2,
            "router outbound|80||model-42-8.static egress",
            100,
            4,
            **{"gpustack.request_id": _REQUEST},
        ),
        _span(
            "gpustack.worker",
            4,
            3,
            "POST inference proxy",
            100,
            3,
            **{
                "gpustack.model_id": 42,
                "gpustack.model_instance_id": 8,
                "gpustack.model_instance_name": "removed-instance",
                "gpustack.request_id": _REQUEST,
                "http.route": "/v1/chat/completions",
                "gpustack.stream": True,
                "gpustack.cancelled": True,
                "gen_ai.prompt": "private prompt",
            },
        ),
        _span(
            "vllm",
            5,
            4,
            "llm_request",
            101,
            2,
            **{
                "gen_ai.usage.prompt_tokens": 10,
                "gen_ai.usage.completion_tokens": 25,
                "gen_ai.latency.time_in_model_decode": 1.2,
                "gen_ai.latency.time_to_first_token": 0.125,
            },
        ),
        _span(
            "gpustack.server",
            6,
            1,
            "POST /v1/chat/completions",
            110,
            2,
            **{"gpustack.model_id": 78, "gpustack.request_id": _OTHER_REQUEST},
        ),
    ]
    # Tempo can return the Worker and engine resource batches first.
    client = _client({"batches": [*batches[3:5], *batches[:3], batches[5]]})
    detail = await get_request_trace(client, "http://tempo", 42, _TRACE, _REQUEST)
    assert detail.span_count == 4
    assert {int(span.id, 16) for span in detail.spans} == {2, 3, 4, 5}
    assert detail.request.instance_name == "removed-instance"
    assert detail.request.outcome == "cancelled"
    assert detail.request.status_code == 200
    assert detail.request.observed_by == "gpustack.higress"
    assert detail.request.duration_ms == 4000
    assert detail.request.engine.ttft_ms == 125
    assert detail.request.engine.tpot_ms == 50
    assert detail.request.engine.queue_ms is None
    engine = next(span.engine for span in detail.spans if span.name == "llm_request")
    assert engine.decode_ms == 1200
    assert engine.tpot_ms == 50
    assert all(span.engine is None for span in detail.spans if span.role != "engine")
    assert "private prompt" not in detail.model_dump_json()
    assert (
        await get_request_trace(client, "http://tempo", 42, _TRACE, _OTHER_REQUEST)
        is None
    )
    assert await get_request_trace(client, "http://tempo", 43, _TRACE, _REQUEST) is None


@pytest.mark.asyncio
async def test_detail_exposes_pd_phases_and_explicit_span_errors():
    entry = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        10,
        **{"gpustack.model_id": 42, "gpustack.request_id": _REQUEST},
    )
    router = _span("vllm-router", 2, 1, "POST /v1/chat/completions", 100, 10)
    batches = [entry, router]
    for dispatch_id, phase in [(3, "prefill"), (5, "decode"), (7, "unknown")]:
        dispatch = _span(
            "vllm-router",
            dispatch_id,
            2,
            "POST /v1/chat/completions",
            101,
            4,
            **{"vllm.request_phase": phase},
        )
        engine = _span("vllm", dispatch_id + 1, dispatch_id, "llm_request", 102, 3)
        if phase == "decode":
            engine["scopeSpans"][0]["spans"][0]["status"] = {"code": 2}
        batches.extend([engine, dispatch])
    batches.append(
        _span(
            "custom",
            9,
            2,
            "custom-call",
            101,
            1,
            **{"vllm.request_phase": "decode"},
        )
    )

    detail = await get_request_trace(
        _client({"batches": batches}), "http://tempo", 42, _TRACE, _REQUEST
    )

    spans = {int(span.id, 16): span for span in detail.spans}
    assert {i: span.phase for i, span in spans.items() if span.phase} == {
        3: "prefill",
        4: "prefill",
        5: "decode",
        6: "decode",
    }
    assert spans[6].status == "error"
    assert spans[4].status == "unset"


@pytest.mark.asyncio
async def test_search_projects_request_data_with_one_bounded_query():
    root = {
        "spanID": "0000000000000001",
        "name": "egress",
        "startTimeUnixNano": str(100 * _NS),
        "durationNanos": str(4 * _NS),
        "attributes": _attributes(
            **{"service.name": "gpustack.higress", "gpustack.request_id": _REQUEST}
        ),
    }
    upstream = {
        "spanID": "0000000000000002",
        "name": "router outbound|80||model-42-8-w2.dns egress",
        "startTimeUnixNano": str(100 * _NS),
        "durationNanos": str(4 * _NS),
        "attributes": _attributes(
            **{
                "service.name": "gpustack.higress",
                "span:parentID": root["spanID"],
                "gpustack.request_id": _REQUEST,
            }
        ),
    }
    worker = {
        "spanID": "0000000000000003",
        "name": "POST inference proxy",
        "startTimeUnixNano": str(101 * _NS),
        "durationNanos": str(2 * _NS),
        "attributes": _attributes(
            **{
                "service.name": "gpustack.worker",
                "span:parentID": upstream["spanID"],
                "gpustack.model_id": 42,
                "gpustack.model_instance_id": 8,
                "gpustack.model_instance_name": "removed-instance",
                "gpustack.request_id": _REQUEST,
                "http.response.status_code": 504,
            }
        ),
    }
    client = _client(
        {
            "traces": [
                {"traceID": _TRACE, "spanSets": [{"spans": [worker, upstream, root]}]}
            ]
        }
    )
    rows = await search_request_traces(
        client,
        "http://tempo",
        42,
        start=1,
        end=200,
        identifier=_REQUEST,
        outcome="error",
        instance_id=8,
        no_engine=True,
    )
    (row,) = rows
    assert not row.has_engine
    assert row.instance_id == 8
    assert row.instance_name == "removed-instance"
    assert row.status_code == 504
    assert row.started_at_ms == 100_000
    client.get.assert_called_once()
    params = client.get.call_args.kwargs["params"]
    assert params["limit"] == 100
    assert "most_recent=true" in params["q"]
    assert f'span.gpustack.request_id="{_REQUEST}"' in params["q"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported", "module", "service", "raw"),
    [
        ("unknown_service", "vllm.llm_engine", "vllm", "unknown_service"),
        ("unknown_service:python", "vllm.llm_engine", "vllm", "unknown_service:python"),
        ("custom-engine", "vllm.llm_engine", "custom-engine", None),
        ("unknown_service", "other.engine", "unknown_service", None),
    ],
)
async def test_detail_resolves_unnamed_vllm_from_reported_resource_metadata(
    reported, module, service, raw
):
    entry = _span(
        "gpustack.worker",
        1,
        None,
        "POST inference proxy",
        100,
        3,
        **{"gpustack.model_id": 42, "gpustack.request_id": _REQUEST},
    )
    engine = _span(reported, 2, 1, "llm_request", 101, 2)
    engine["resource"]["attributes"].extend(
        _attributes(**{"vllm.instrumenting_module_name": module})
    )
    detail = await get_request_trace(
        _client({"batches": [entry, engine]}), "http://tempo", 42, _TRACE, _REQUEST
    )
    assert detail.spans[1].service == service
    assert detail.spans[1].reported_service == raw


@pytest.mark.asyncio
async def test_unavailable_tempo_does_not_report_empty_results():
    with pytest.raises(ServiceUnavailableException):
        await search_request_traces(
            _client({}, status=503),
            "http://tempo",
            42,
            start=1,
            end=200,
            identifier=None,
            outcome=None,
            instance_id=None,
            no_engine=False,
        )


@pytest.mark.asyncio
async def test_stream_error_preserves_the_committed_http_status():
    worker = _span(
        "gpustack.worker",
        2,
        1,
        "POST inference proxy",
        100,
        4,
        **{"gpustack.model_id": 42, "http.response.status_code": 200},
    )
    worker["scopeSpans"][0]["spans"][0]["status"] = {"code": "STATUS_CODE_ERROR"}
    gateway = _span(
        "gpustack.higress",
        1,
        None,
        "egress",
        99,
        6,
        **{"gpustack.request_id": _REQUEST, "http.status_code": "200"},
    )
    detail = await get_request_trace(
        _client({"batches": [worker, gateway]}), "http://tempo", 42, _TRACE, _REQUEST
    )
    assert detail.request.outcome == "error"
    assert detail.request.status_code == 200


@pytest.mark.asyncio
async def test_direct_trace_keeps_the_name_of_its_selected_engine():
    gateway = _span(
        "gpustack.higress",
        1,
        None,
        "router outbound|80||model-42-8.static egress",
        100,
        4,
        **{"gpustack.request_id": _REQUEST},
    )
    engine = _span("vllm", 2, 1, "llm_request", 101, 3)
    engine["resource"]["attributes"].extend(
        _attributes(
            **{
                "gpustack.engine_instance_id": "8",
                "gpustack.engine_instance_name": "removed-8",
            }
        )
    )
    other = _span("vllm", 3, 1, "llm_request", 101, 3)
    other["resource"]["attributes"].extend(
        _attributes(
            **{
                "gpustack.engine_instance_id": "90",
                "gpustack.engine_instance_name": "unrelated",
            }
        )
    )

    detail = await get_request_trace(
        _client({"batches": [gateway, engine, other]}),
        "http://tempo",
        42,
        _TRACE,
        _REQUEST,
    )

    assert detail.request.instance_id == 8
    assert detail.request.instance_name == "removed-8"


def _candidate(trace_number, started):
    return {
        "traceID": f"{trace_number:032x}",
        "startTimeUnixNano": str(started),
        "spanSets": [
            {
                "spans": [
                    {
                        "spanID": "0000000000000001",
                        "name": "POST inference proxy",
                        "startTimeUnixNano": str(started),
                        "durationNanos": str(_NS),
                        "attributes": _attributes(
                            **{
                                "service.name": "gpustack.worker",
                                "gpustack.model_id": 42,
                                "http.response.status_code": 200,
                            }
                        ),
                    }
                ]
            }
        ],
    }


@pytest.mark.asyncio
async def test_pages_preserve_traces_in_the_boundary_second():
    candidates = [_candidate(i, 102 * _NS + 10_000 * i) for i in range(101, 0, -1)]
    candidates.append(_candidate(102, 101 * _NS))
    position = TracePagePosition(start=1, end=200, before=200)
    seen = []
    for _ in range(11):
        client = _client(
            {
                "traces": [
                    trace
                    for trace in candidates
                    if trace["traceID"] not in position.seen
                    and int(trace["startTimeUnixNano"]) < position.before * _NS
                ][:100]
            }
        )
        page = await search_request_trace_page(
            client,
            "http://tempo",
            42,
            position=position,
            page_size=10,
            identifier=None,
            outcome="success",
            instance_id=None,
            no_engine=False,
        )
        seen.extend(row.trace_id for row in page.items)
        if not page.next_cursor:
            break
        position = TracePagePosition.model_validate_json(
            base64.urlsafe_b64decode(page.next_cursor)
        )
        assert position.end == 200
        assert position.before == 103

    expected = [f"{i:032x}" for i in range(101, 0, -1)] + [f"{102:032x}"]
    assert seen == expected
