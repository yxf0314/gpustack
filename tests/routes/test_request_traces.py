from types import SimpleNamespace
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes import request_traces
from gpustack.schemas.principals import PrincipalType
from gpustack.server.db import get_session


@pytest.fixture
def trace_client(monkeypatch):
    app = FastAPI()
    app.include_router(request_traces.router, prefix="/models")
    register_handlers(app)
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=True,
        current_principal_id=11,
        org_role=None,
    )
    cfg = SimpleNamespace(
        effective_trace_otlp_endpoint=None,
        get_trace_query_url=Mock(return_value=None),
        log_otlp_endpoint=None,
        log_grafana_datasource_uid=None,
        grafana_url="https://grafana.test/subpath",
    )
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    app.dependency_overrides[get_session] = lambda: None
    monkeypatch.setattr(request_traces, "get_global_config", lambda: cfg)
    monkeypatch.setattr(
        request_traces.Model,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(owner_principal_id=11)),
    )
    with TestClient(app) as client:
        yield client, ctx, cfg


@pytest.mark.parametrize("tracing", [False, True])
@pytest.mark.parametrize("logs", [False, True])
def test_capabilities_independently_report_sources(trace_client, tracing, logs):
    client, _, cfg = trace_client
    if tracing:
        cfg.effective_trace_otlp_endpoint = "tempo:4317"
        cfg.get_trace_query_url.return_value = "http://tempo:3200"
    if logs:
        cfg.log_otlp_endpoint = "https://loki.test/otlp/v1/logs"
        cfg.log_grafana_datasource_uid = "loki-uid"
    response = client.get("/models/42/request-traces/capabilities")
    assert response.status_code == 200
    assert response.json() == {
        "tracing_enabled": tracing,
        "trace_query_configured": tracing,
        "log_collection_enabled": logs,
        "log_explore_url": "https://grafana.test/subpath/explore" if logs else None,
        "log_datasource_uid": "loki-uid" if logs else None,
    }


@pytest.mark.parametrize(
    "suffix", ["/capabilities", "", "/abcd0000000000000000000000000000"]
)
def test_query_routes_check_platform_role_and_model_visibility(trace_client, suffix):
    client, ctx, _ = trace_client
    url = "/models/42/request-traces" + suffix
    ctx.is_platform_admin = False
    assert client.get(url).status_code == 403
    ctx.is_platform_admin = True
    ctx.current_principal_id = 12
    assert client.get(url).status_code == 404


def test_custom_trace_query_preserves_exact_time_bounds(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(request_traces, "search_request_traces", search)
    end = int(time.time()) - 300
    start = end - 23

    response = client.get(
        "/models/42/request-traces", params={"start": start, "end": end}
    )
    assert response.status_code == 200
    assert search.await_args.kwargs["start"] == start
    assert search.await_args.kwargs["end"] == end


@pytest.mark.parametrize(
    ("window", "seconds"), [("5m", 300), ("3h", 10800), ("7d", 604800)]
)
def test_trace_search_supports_the_monitoring_presets(
    trace_client, monkeypatch, window, seconds
):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(request_traces, "search_request_traces", search)

    response = client.get("/models/42/request-traces", params={"window": window})
    assert response.status_code == 200
    queried = search.await_args.kwargs
    assert queried["end"] - queried["start"] == seconds

    response = client.get("/models/42/request-traces", params={"window": "8d"})
    assert response.status_code == 400
    assert search.await_count == 1


@pytest.mark.parametrize(
    "offsets", [(None, -600), (-300, -600), (-604900, -10), (-10, 600)]
)
def test_invalid_trace_range_never_queries_tempo(trace_client, monkeypatch, offsets):
    client, _, _ = trace_client
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(request_traces, "search_request_traces", search)
    now = int(time.time())
    params = {
        key: now + offset
        for key, offset in zip(("start", "end"), offsets)
        if offset is not None
    }
    response = client.get("/models/42/request-traces", params=params)
    assert response.status_code == 400
    search.assert_not_awaited()


def test_cursor_pins_24_hour_window(trace_client, monkeypatch):
    client, _, cfg = trace_client
    cfg.effective_trace_otlp_endpoint = "tempo:4317"
    cfg.get_trace_query_url.return_value = "http://tempo:3200"
    client.app.state.http_client_no_proxy = object()
    query = AsyncMock(return_value=request_traces.RequestTracePage(items=[]))
    monkeypatch.setattr(request_traces, "search_request_trace_page", query)
    expected_now = int(time.time()) - 60
    url = "/models/42/request-traces"
    with monkeypatch.context() as patch:
        patch.setattr(
            request_traces, "time", SimpleNamespace(time=lambda: expected_now)
        )
        assert client.get(url, params={"page_size": 10}).status_code == 200
        first = query.await_args.kwargs["position"]
        assert first.end - first.start == 86400
        patch.setattr(
            request_traces, "time", SimpleNamespace(time=lambda: expected_now + 5)
        )
        assert (
            client.get(
                url, params={"page_size": 10, "cursor": first.encode()}
            ).status_code
            == 200
        )
        assert query.await_args.kwargs["position"] == first
    assert client.get(url, params={"cursor": "invalid"}).status_code == 400
    assert query.await_count == 2
