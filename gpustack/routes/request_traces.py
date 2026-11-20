import time
from typing import Annotated, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Path, Query, Request

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import assert_resource_visible
from gpustack.config.config import get_global_config
from gpustack.monitoring.prometheus import parse_window
from gpustack.monitoring.time_range import measured_interval
from gpustack.observability.request_traces import (
    REQUEST_ID,
    TRACE_ID,
    RequestOutcome,
    RequestTraceCapabilities,
    RequestTraceDetail,
    RequestTraceRecord,
    get_request_trace,
    search_request_traces,
)
from gpustack.schemas.models import Model
from gpustack.server.deps import PlatformAdminDep, SessionDep
from gpustack.utils.grafana import resolve_grafana_base_url


async def _trace_model(
    ctx: PlatformAdminDep, session: SessionDep, id: Annotated[int, Path(gt=0)]
) -> int:
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    return id


TraceModelDep = Annotated[int, Depends(_trace_model)]
router = APIRouter(prefix="/{id}/request-traces")


def _query_url() -> str:
    cfg = get_global_config()
    url = cfg.get_trace_query_url()
    if not url or not cfg.effective_trace_otlp_endpoint:
        raise ServiceUnavailableException("Trace query is not configured")
    return url


@router.get("/capabilities")
async def get_capabilities(
    request: Request, model_id: TraceModelDep
) -> RequestTraceCapabilities:
    cfg = get_global_config()
    tracing_enabled = bool(cfg.effective_trace_otlp_endpoint)
    log_collection_enabled = bool(cfg.log_otlp_endpoint)
    log_uid = (cfg.log_grafana_datasource_uid or "").strip()
    explore_url = None
    if log_collection_enabled and log_uid:
        base = resolve_grafana_base_url(cfg, request)
        parsed = urlparse(base)
        if (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and parsed.username is None
            and not parsed.query
            and not parsed.fragment
        ):
            explore_url = base + "/explore"
    return RequestTraceCapabilities(
        tracing_enabled=tracing_enabled,
        trace_query_configured=tracing_enabled and bool(cfg.get_trace_query_url()),
        log_collection_enabled=log_collection_enabled,
        log_explore_url=explore_url,
        log_datasource_uid=log_uid if explore_url else None,
    )


@router.get("")
async def list_traces(
    request: Request,
    model_id: TraceModelDep,
    window: str = "1h",
    q: Annotated[Optional[str], Query(pattern=f"^(?:{TRACE_ID}|{REQUEST_ID})$")] = None,
    outcome: Optional[RequestOutcome] = None,
    instance_id: Annotated[Optional[int], Query(gt=0)] = None,
    no_engine: bool = False,
    start: Annotated[Optional[int], Query(ge=0)] = None,
    end: Annotated[Optional[int], Query(ge=0)] = None,
) -> List[RequestTraceRecord]:
    if q and len(q) == 32:
        q = q.lower()
    now = int(time.time())
    if start is None and end is None:
        end = now + 1
        try:
            start = end - parse_window(window)
        except ValueError as exc:
            raise BadRequestException(message=str(exc)) from None
    else:
        try:
            interval = measured_interval(start, end, now=now)
        except ValueError:
            raise BadRequestException(
                message="Provide start and end in the past with a span of up to 7 days"
            ) from None
        start, end = int(interval.start), int(interval.end)
    return await search_request_traces(
        request.app.state.http_client_no_proxy,
        _query_url(),
        model_id,
        start=start,
        end=end,
        identifier=q,
        outcome=outcome,
        instance_id=instance_id,
        no_engine=no_engine,
    )


@router.get("/{trace_id}")
async def get_trace(
    request: Request,
    model_id: TraceModelDep,
    trace_id: Annotated[str, Path(pattern=f"^{TRACE_ID}$")],
    request_id: Annotated[Optional[str], Query(pattern=f"^{REQUEST_ID}$")] = None,
) -> RequestTraceDetail:
    detail = await get_request_trace(
        request.app.state.http_client_no_proxy,
        _query_url(),
        model_id,
        trace_id.lower(),
        request_id,
    )
    if detail is None:
        raise NotFoundException(message="Request trace not found")
    return detail
