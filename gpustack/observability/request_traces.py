"""Read bounded, deployment-scoped request observations from Tempo."""

import asyncio
import base64
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field, StringConstraints

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.observability.tracing import PUBLIC_INFERENCE_PATHS

TRACE_ID = r"[0-9a-fA-F]{32}"
REQUEST_ID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
RequestOutcome = Literal["success", "error", "cancelled", "unknown"]

_GATEWAY = "gpustack.higress"
_PLATFORM = {_GATEWAY, "gpustack.server", "gpustack.worker"}
_GATEWAY_ROUTE = re.compile(
    r"router outbound[|][0-9]+[|][|]model-(\d+)-(\d+)"
    r"(?:-dp\d+|-w\d+)?[.](?:static|dns) egress"
)
_ENGINE_TIMES = {
    "ttft_ms": "gen_ai.latency.time_to_first_token",
    "queue_ms": "gen_ai.latency.time_in_queue",
    "prefill_ms": "gen_ai.latency.time_in_model_prefill",
    "decode_ms": "gen_ai.latency.time_in_model_decode",
}
_ENGINE_TOKENS = {
    "input_tokens": "gen_ai.usage.prompt_tokens",
    "output_tokens": "gen_ai.usage.completion_tokens",
}
_ENGINE_FIELDS = {**_ENGINE_TIMES, **_ENGINE_TOKENS}
_PROJECTION = ",".join(
    [
        "span:name",
        "span:parentID",
        "span:status",
        "resource.service.name",
        "resource.gpustack.engine_instance_id",
        "resource.gpustack.engine_instance_name",
        *[
            f"span.{name}"
            for name in (
                "gpustack.request_id",
                "gpustack.model_id",
                "gpustack.model_instance_id",
                "gpustack.model_instance_name",
                "gpustack.cancelled",
                "gpustack.stream",
                "http.route",
                "http.response.status_code",
                "http.status_code",
                _ENGINE_TIMES["ttft_ms"],
                *_ENGINE_TOKENS.values(),
            )
        ],
    ]
)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_DETAIL_SPANS = 256
_SEARCH_BATCH_SIZE = 100
_CURSOR_ID_LIMIT = 240
_NANOSECONDS_PER_SECOND = 1_000_000_000


class RequestTraceCapabilities(BaseModel):
    tracing_enabled: bool
    trace_query_configured: bool
    log_collection_enabled: bool
    log_explore_url: Optional[str] = None
    log_datasource_uid: Optional[str] = None


class RequestEngineMetrics(BaseModel):
    ttft_ms: Optional[float] = None
    queue_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class RequestTraceRecord(BaseModel):
    trace_id: str
    request_id: Optional[str]
    started_at_ms: int
    duration_ms: float
    observed_by: str
    path: Optional[str]
    status_code: Optional[int]
    outcome: RequestOutcome
    stream: Optional[bool]
    instance_id: Optional[int]
    instance_name: Optional[str] = None
    has_engine: bool
    engine: Optional[RequestEngineMetrics]


class RequestTracePage(BaseModel):
    items: List[RequestTraceRecord]
    cursor: str = ""
    next_cursor: Optional[str] = None


_TraceId = Annotated[str, StringConstraints(pattern=f"^{TRACE_ID}$")]


class TracePagePosition(BaseModel):
    start: int = Field(ge=0)
    end: int
    before: int
    seen: List[_TraceId] = Field(default_factory=list, max_length=_CURSOR_ID_LIMIT)
    partial: Optional[_TraceId] = None
    offset: int = Field(default=0, ge=0)

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode()


class RequestTraceSpan(BaseModel):
    id: str
    parent_id: Optional[str]
    name: str
    service: str
    role: Literal["gateway", "proxy", "engine", "other"]
    start_offset_ms: float
    duration_ms: float
    status: str
    engine: Optional[RequestEngineMetrics] = None
    reported_service: Optional[str] = None
    phase: Optional[Literal["prefill", "decode"]] = None


class RequestTraceDetail(BaseModel):
    request: RequestTraceRecord
    spans: List[RequestTraceSpan]
    span_count: int


@dataclass(frozen=True)
class _Span:
    id: str
    parent_id: Optional[str]
    name: str
    service: str
    start: int
    duration: int
    status: str
    attrs: Dict[str, Any]
    reported_service: Optional[str] = None

    @property
    def target(self) -> Tuple[Optional[int], Optional[int]]:
        if self.service == _GATEWAY and (match := _GATEWAY_ROUTE.fullmatch(self.name)):
            return int(match[1]), int(match[2])
        if self.service in _PLATFORM:
            return (
                _positive_int(self.attrs.get("gpustack.model_id")),
                _positive_int(self.attrs.get("gpustack.model_instance_id")),
            )
        return None, None

    @property
    def request_id(self) -> Optional[str]:
        value = self.attrs.get("gpustack.request_id")
        if self.service in _PLATFORM and isinstance(value, str):
            if re.fullmatch(REQUEST_ID, value):
                return value
        return None

    @property
    def is_engine(self) -> bool:
        return self.service not in _PLATFORM and (
            self.name == "llm_request"
            or any(field in self.attrs for field in _ENGINE_FIELDS.values())
        )


def _positive_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value > 0 else None


def _number(value: Any) -> Optional[float]:
    if type(value) in (int, float) and 0 <= value < math.inf:
        return value
    return None


def _attributes(values: List[dict]) -> Dict[str, Any]:
    result = {}
    for item in values:
        name = item["key"]
        for kind, value in item["value"].items():
            if kind == "intValue":
                result[name] = int(value)
            elif kind == "stringValue" and value != "nil":
                result[name] = value
            elif kind in ("boolValue", "doubleValue"):
                result[name] = value
    return result


def _span_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if re.fullmatch(r"[0-9a-fA-F]{16}", value):
        value = value.lower()
    else:
        value = base64.b64decode(value, validate=True).hex()
        if len(value) != 16:
            raise ValueError("Invalid span ID")
    return value if value != "0000000000000000" else None


def _search_spans(trace: dict) -> List[_Span]:
    by_id = {}
    for group in trace.get("spanSets") or [trace.get("spanSet", {})]:
        for item in group.get("spans", []):
            attrs = _attributes(item.get("attributes", []))
            for key in ("gpustack.engine_instance_id", "gpustack.engine_instance_name"):
                if f"resource.{key}" in attrs:
                    attrs[key] = attrs[f"resource.{key}"]
            span = _Span(
                id=_span_id(item["spanID"]) or "",
                parent_id=_span_id(attrs.get("span:parentID")),
                name=item.get("name", ""),
                service=attrs.get("service.name", ""),
                start=int(item["startTimeUnixNano"]),
                duration=int(item["durationNanos"]),
                status=attrs.get("status", "unset"),
                attrs=attrs,
            )
            by_id[span.id] = span
    return list(by_id.values())


def _detail_spans(data: dict) -> List[_Span]:
    spans = []
    for batch in data.get("batches", data.get("resourceSpans", [])):
        resource = _attributes(batch.get("resource", {}).get("attributes", []))
        reported_service = resource.get("service.name", "")
        service = reported_service
        module = resource.get("vllm.instrumenting_module_name")
        if (
            (not service or re.match(r"^unknown_service(?::|$)", service))
            and isinstance(module, str)
            and module.startswith("vllm.")
        ):
            service = "vllm"
        for scope in batch.get("scopeSpans", []):
            for item in scope.get("spans", []):
                start = int(item["startTimeUnixNano"])
                status = (
                    str(item.get("status", {}).get("code", "unset"))
                    .lower()
                    .removeprefix("status_code_")
                )
                spans.append(
                    _Span(
                        id=_span_id(item["spanId"]) or "",
                        parent_id=_span_id(item.get("parentSpanId")),
                        name=item.get("name", ""),
                        service=service,
                        start=start,
                        duration=max(0, int(item["endTimeUnixNano"]) - start),
                        status={"0": "unset", "1": "ok", "2": "error"}.get(
                            status, status
                        ),
                        attrs={
                            **_attributes(item.get("attributes", [])),
                            **{
                                key: resource[key]
                                for key in (
                                    "gpustack.engine_instance_id",
                                    "gpustack.engine_instance_name",
                                )
                                if key in resource
                            },
                        },
                        reported_service=(
                            reported_service if service != reported_service else None
                        ),
                    )
                )
    return spans


def _entry_for(anchor: _Span, by_id: Dict[str, _Span]) -> _Span:
    # Stop at the first known entry. A caller can continue an existing trace
    # across requests, so the Tempo trace root can belong to another exchange.
    current = anchor
    request_id = anchor.request_id
    model_id = anchor.target[0]
    visited = {current.id}
    while current.parent_id and (parent := by_id.get(current.parent_id)):
        if (
            parent.id in visited
            or parent.service not in _PLATFORM
            or parent.target[0] not in (None, model_id)
            or (parent.request_id and request_id and parent.request_id != request_id)
        ):
            break
        current = parent
        visited.add(current.id)
        # Higress attaches the request ID to dispatch spans as well as ingress.
        if (
            current.request_id
            and current.service in (_GATEWAY, "gpustack.server")
            and not current.name.startswith("router ")
        ):
            break
    return current


def _requests(
    spans: List[_Span], model_id: int
) -> List[Tuple[_Span, List[_Span], List[_Span]]]:
    by_id = {span.id: span for span in spans}
    children = defaultdict(list)
    entries = {}
    for span in spans:
        children[span.parent_id].append(span)
        if span.target[0] == model_id:
            entry = _entry_for(span, by_id)
            entries.setdefault(entry.id, (entry, []))[1].append(span)
    requests = []
    for entry, anchors in entries.values():
        request_id = entry.request_id or next(
            (span.request_id for span in anchors if span.request_id), None
        )
        branch = {}
        pending = [entry, *anchors]
        while pending:
            span = pending.pop()
            if span.id in branch:
                continue
            if span.target[0] not in (None, model_id):
                continue
            if span.request_id and request_id and span.request_id != request_id:
                continue
            branch[span.id] = span
            pending.extend(children[span.id])
        requests.append(
            (entry, anchors, sorted(branch.values(), key=lambda s: s.start))
        )
    return requests


def _http_status(span: _Span) -> Optional[int]:
    value = span.attrs.get(
        "http.response.status_code", span.attrs.get("http.status_code")
    )
    if type(value) is int or (
        isinstance(value, str) and value.isascii() and value.isdigit()
    ):
        code = int(value)
        return code if 100 <= code <= 599 else None
    return None


def _request_outcome(
    status: Optional[int], entry: _Span, spans: List[_Span]
) -> RequestOutcome:
    if status == 499 or entry.attrs.get("gpustack.cancelled") is True:
        return "cancelled"
    if (status is not None and status >= 400) or entry.status in ("error", "2"):
        return "error"
    if any(
        span.attrs.get("gpustack.cancelled") is True
        for span in spans
        if span.service in _PLATFORM
    ):
        return "cancelled"
    if any(
        span.status in ("error", "2") and _http_status(span) == status
        for span in spans
        if span.service in _PLATFORM
    ):
        return "error"
    if status is None:
        return "unknown"
    return "success"


def _span_engine_metrics(span: _Span) -> Optional[RequestEngineMetrics]:
    if not span.is_engine:
        return None
    attrs = span.attrs
    values = {}
    for key, field in _ENGINE_TIMES.items():
        if (value := _number(attrs.get(field))) is not None:
            values[key] = value * 1000
    for key, field in _ENGINE_TOKENS.items():
        value = attrs.get(field)
        if type(value) is int and value >= 0:
            values[key] = value
    if "decode_ms" in values and values.get("output_tokens", 0) > 1:
        values["tpot_ms"] = values["decode_ms"] / (values["output_tokens"] - 1)
    return RequestEngineMetrics(**values) if values else None


def _engine_metrics(spans: List[_Span]) -> Optional[RequestEngineMetrics]:
    # Multiple reporting spans can represent prefill/decode, retries, or choices.
    observations = [
        metrics for span in spans if (metrics := _span_engine_metrics(span))
    ]
    return observations[0] if len(observations) == 1 else None


def _instance_name(
    instance_id: int, anchors: List[_Span], spans: List[_Span]
) -> Optional[str]:
    names = {
        name[:255]
        for span in anchors
        if span.service in _PLATFORM and span.target[1] == instance_id
        if isinstance(name := span.attrs.get("gpustack.model_instance_name"), str)
        and name
    }
    if not names:
        names = {
            name[:255]
            for span in spans
            if span.attrs.get("gpustack.engine_instance_id") == str(instance_id)
            if isinstance(name := span.attrs.get("gpustack.engine_instance_name"), str)
            and name
        }
    return next(iter(names)) if len(names) == 1 else None


def _summarize(
    trace_id: str, entry: _Span, anchors: List[_Span], spans: List[_Span]
) -> RequestTraceRecord:
    targets = [entry, *sorted(anchors, key=lambda span: span.start, reverse=True)]
    request_id = next((span.request_id for span in targets if span.request_id), None)
    path = next(
        (
            span.attrs["http.route"]
            for span in targets
            if span.attrs.get("http.route") in PUBLIC_INFERENCE_PATHS
        ),
        None,
    )
    status = next((code for span in targets if (code := _http_status(span))), None)
    # A gateway retry can select a different instance. Only show a single ID
    # when the observations agree.
    instances = {span.target[1] for span in anchors} - {None}
    instance_id = next(iter(instances)) if len(instances) == 1 else None
    stream = next(
        (
            value
            for span in targets
            if type(value := span.attrs.get("gpustack.stream")) is bool
        ),
        None,
    )
    return RequestTraceRecord(
        trace_id=trace_id,
        request_id=request_id,
        started_at_ms=entry.start // 1_000_000,
        duration_ms=entry.duration / 1_000_000,
        observed_by=entry.service,
        path=path,
        status_code=status,
        outcome=_request_outcome(status, entry, spans),
        stream=stream,
        instance_id=instance_id,
        instance_name=(
            _instance_name(instance_id, anchors, spans) if instance_id else None
        ),
        has_engine=any(span.is_engine for span in spans),
        engine=_engine_metrics(spans),
    )


def _search_query(
    model_id: int, identifier: Optional[str], seen: Tuple[str, ...] = ()
) -> str:
    if identifier and re.fullmatch(TRACE_ID, identifier):
        return f'{{trace:id="{identifier.lower()}"}} | select({_PROJECTION})'
    direct = f"router outbound.*[|][|]model-{model_id}-.* egress"
    scope = f'{{span.gpustack.model_id={model_id}||name=~"{direct}"}}'
    platform = '{resource.service.name=~"gpustack[.](higress|server|worker)"}'
    engine = '{name="llm_request"||span.gen_ai.usage.prompt_tokens>=0}'
    query = f"({scope}||({scope}<<{platform})||({scope}>>{engine}))"
    if identifier:
        query += '&&{span.gpustack.request_id=' + json.dumps(identifier) + "}"
    if seen:
        excluded = "&&".join(f'trace:id!="{trace_id}"' for trace_id in seen)
        query += "&&{" + excluded + "}"
    return f"{query} | select({_PROJECTION}) with (most_recent=true)"


async def _tempo_get(
    client: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
) -> Optional[dict]:
    try:
        async with client.get(
            url,
            params=params,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=8),
            headers={"Accept": "application/json"},
        ) as response:
            if response.status == 404 and params is None:
                return None
            if response.status != 200:
                raise ServiceUnavailableException("Trace query service is unavailable")
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > _MAX_RESPONSE_BYTES:
                    raise ServiceUnavailableException(
                        "Trace query response is too large"
                    )
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("Invalid trace query response")
            return payload
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise ServiceUnavailableException(
            "Trace query service is unavailable"
        ) from None


async def _search_candidates(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    start: int,
    end: int,
    identifier: Optional[str],
    seen: Tuple[str, ...] = (),
) -> List[dict]:
    payload = await _tempo_get(
        client,
        f"{base_url}/api/search",
        {
            "q": _search_query(model_id, identifier, seen),
            "start": start,
            "end": end,
            "limit": _SEARCH_BATCH_SIZE,
            "spss": 24,
        },
    )
    return payload.get("traces", [])


@dataclass
class _RowFilter:
    model_id: int
    start: int
    end: int
    identifier: Optional[str]
    outcome: Optional[RequestOutcome]
    instance_id: Optional[int]
    no_engine: bool

    def rows(self, trace: dict) -> List[RequestTraceRecord]:
        trace_id = _tempo_trace_id(trace["traceID"])
        if not trace_id:
            return []
        rows = (
            _summarize(trace_id, entry, anchors, spans)
            for entry, anchors, spans in _requests(_search_spans(trace), self.model_id)
        )
        return sorted(
            (
                row
                for row in rows
                if self.start * 1000 <= row.started_at_ms < self.end * 1000
                and (
                    not self.identifier
                    or self.identifier in (row.request_id, row.trace_id)
                )
                and (self.outcome is None or row.outcome == self.outcome)
                and (self.instance_id is None or row.instance_id == self.instance_id)
                and (not self.no_engine or not row.has_engine)
            ),
            key=lambda row: (row.started_at_ms, row.request_id or ""),
            reverse=True,
        )


async def search_request_traces(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    *,
    start: int,
    end: int,
    identifier: Optional[str],
    outcome: Optional[RequestOutcome],
    instance_id: Optional[int],
    no_engine: bool,
) -> List[RequestTraceRecord]:
    candidates = await _search_candidates(
        client, base_url, model_id, start, end, identifier
    )
    filters = _RowFilter(
        model_id, start, end, identifier, outcome, instance_id, no_engine
    )
    try:
        return sorted(
            (row for trace in candidates for row in filters.rows(trace)),
            key=lambda row: row.started_at_ms,
            reverse=True,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None


def _tempo_trace_id(value: str) -> str:
    # Tempo renders IDs without leading zeroes in search metadata.
    return value.lower().zfill(32) if re.fullmatch(r"[0-9a-fA-F]{1,32}", value) else ""


def _candidate_key(trace: dict) -> Tuple[int, str]:
    return int(trace["startTimeUnixNano"]), _tempo_trace_id(trace["traceID"])


def _page_from_candidates(
    candidates: List[dict],
    position: TracePagePosition,
    filters: _RowFilter,
    page_size: int,
) -> RequestTracePage:
    cursor = position.encode()
    items: List[RequestTraceRecord] = []
    # Tempo search takes second-resolution bounds. Exclude consumed candidates
    # in the overlapping second so advancing the window cannot skip ties.
    partial, offset = position.partial, position.offset
    for trace in sorted(candidates, key=_candidate_key, reverse=True):
        started, trace_id = _candidate_key(trace)
        if not trace_id or trace_id in position.seen:
            continue
        rows = filters.rows(trace)
        if trace_id == partial:
            rows = rows[offset:]
        before = max(
            position.start + 1,
            min(position.before, started // _NANOSECONDS_PER_SECOND + 1),
        )
        if before != position.before:
            position.before = before
            position.seen.clear()
        remaining = page_size - len(items)
        items.extend(rows[:remaining])
        if len(rows) > remaining:
            position.partial = trace_id
            position.offset = (offset if trace_id == partial else 0) + remaining
            return RequestTracePage(
                items=items, cursor=cursor, next_cursor=position.encode()
            )
        if len(position.seen) < _CURSOR_ID_LIMIT:
            position.seen.append(trace_id)
        else:
            raise ServiceUnavailableException(
                "Too many traces in a single second to page"
            )
    position.partial = None
    position.offset = 0
    return RequestTracePage(
        items=items,
        cursor=cursor,
        next_cursor=(
            position.encode() if len(candidates) >= _SEARCH_BATCH_SIZE else None
        ),
    )


async def search_request_trace_page(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    *,
    position: TracePagePosition,
    page_size: int,
    identifier: Optional[str],
    outcome: Optional[RequestOutcome],
    instance_id: Optional[int],
    no_engine: bool,
) -> RequestTracePage:
    filters = _RowFilter(
        model_id,
        position.start,
        position.end,
        identifier,
        outcome,
        instance_id,
        no_engine,
    )
    candidates = await _search_candidates(
        client,
        base_url,
        model_id,
        position.start,
        position.before,
        identifier,
        tuple(position.seen),
    )
    try:
        page = _page_from_candidates(candidates, position, filters, page_size)
        page.items.sort(key=lambda row: row.started_at_ms, reverse=True)
        return page
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None


def _display_span(span: _Span, start: int, parent: Optional[_Span]) -> RequestTraceSpan:
    role: Literal["gateway", "proxy", "engine", "other"] = "other"
    if span.service == _GATEWAY:
        role = "gateway"
    elif span.service in _PLATFORM:
        role = "proxy"
    elif span.is_engine:
        role = "engine"
    phase_span = parent if role == "engine" else span
    phase = (
        phase_span.attrs.get("vllm.request_phase")
        if phase_span and phase_span.service == "vllm-router"
        else None
    )
    return RequestTraceSpan(
        id=span.id,
        parent_id=span.parent_id,
        name=span.name[:160],
        service=span.service[:100],
        role=role,
        start_offset_ms=(span.start - start) / 1_000_000,
        duration_ms=span.duration / 1_000_000,
        status=span.status,
        engine=_span_engine_metrics(span),
        reported_service=span.reported_service,
        phase=phase if phase in ("prefill", "decode") else None,
    )


async def get_request_trace(
    client: aiohttp.ClientSession,
    base_url: str,
    model_id: int,
    trace_id: str,
    request_id: Optional[str],
) -> Optional[RequestTraceDetail]:
    payload = await _tempo_get(client, f"{base_url}/api/traces/{trace_id}")
    if payload is None:
        return None
    try:
        for entry, anchors, spans in _requests(_detail_spans(payload), model_id):
            row = _summarize(trace_id, entry, anchors, spans)
            if request_id and row.request_id != request_id:
                continue
            start = spans[0].start
            by_id = {span.id: span for span in spans}
            return RequestTraceDetail(
                request=row,
                spans=[
                    _display_span(span, start, by_id.get(span.parent_id or ""))
                    for span in spans[:_MAX_DETAIL_SPANS]
                ],
                span_count=len(spans),
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException("Trace query returned invalid data") from None
    return None
