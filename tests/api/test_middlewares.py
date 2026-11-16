"""Unit tests for usage-attribution helpers in ``api/middlewares.py``.

Focus: the direct (cookie-authed) path's ``X-Organization-Id`` validation,
which guards ``consumer_principal_id`` against a spoofed / non-existent
principal id that would otherwise violate the FK and roll back the whole
usage flush batch.
"""

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gpustack.api import middlewares
from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.app import create_app


class _FakeSessionCtx:
    """Minimal ``async with async_session() as session`` stand-in — the real
    session is never touched because ``resolve_tenant_context`` is stubbed."""

    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _stub_async_session(monkeypatch):
    monkeypatch.setattr("gpustack.server.db.async_session", lambda: _FakeSessionCtx())


def _stub_tenant_context(monkeypatch, *, result=None, exc=None):
    async def _fake(request, user, x_organization_id=None, session=None):
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr("gpustack.api.tenant.resolve_tenant_context", _fake)


_REQUEST = SimpleNamespace(state=SimpleNamespace())
_USER = SimpleNamespace(id=5, is_admin=False)


@pytest.mark.parametrize(
    ("method", "path", "has_request_id"),
    [
        ("POST", "/v1/chat/completions", True),
        ("POST", "/v1-openai/chat/completions", True),
        ("POST", "/v1/rerank", True),
        ("GET", "/v1/models", False),
        ("GET", "/v2/models", False),
        ("POST", "/v1/not-an-inference-route", False),
    ],
)
def test_native_inference_request_id_on_entry(config, method, path, has_request_id):
    app = create_app(
        config.model_copy(update={"gateway_mode": GatewayModeEnum.disabled})
    )

    async def reject():
        raise HTTPException(status_code=401, detail="Authentication required")

    app.dependency_overrides[get_current_user] = reject
    response = TestClient(app).request(method, path, headers={"X-Request-Id": "client"})

    request_ids = response.headers.get_list("X-GPUStack-Request-Id")
    if has_request_id:
        assert response.status_code == 401
        assert len(request_ids) == 1
        assert UUID(request_ids[0]).version == 4
    else:
        assert request_ids == []


@pytest.mark.parametrize(
    "gateway_mode", [GatewayModeEnum.disabled, GatewayModeEnum.embedded]
)
def test_native_inference_request_id_cors(config, gateway_mode):
    origin = "https://client.example.com"
    app = create_app(
        config.model_copy(
            update={
                "gateway_mode": gateway_mode,
                "enable_cors": True,
                "allow_origins": [origin],
            }
        )
    )

    async def reject():
        raise HTTPException(status_code=401, detail="Authentication required")

    app.dependency_overrides[get_current_user] = reject
    response = TestClient(app).post("/v1/chat/completions", headers={"Origin": origin})

    assert response.headers["access-control-allow-origin"] == origin
    if gateway_mode == GatewayModeEnum.disabled:
        assert (
            "x-gpustack-request-id"
            in response.headers["access-control-expose-headers"].lower()
        )
        assert "x-gpustack-request-id" in response.headers
    else:
        assert "x-gpustack-request-id" not in response.headers


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_returns_validated_id(monkeypatch):
    # A member acting in org 7 → validated current_principal_id → carried.
    _stub_tenant_context(monkeypatch, result=SimpleNamespace(current_principal_id=7))
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") == "7"


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_none_when_no_context(monkeypatch):
    # Admin with no effective org context → None → collector fallback applies.
    _stub_tenant_context(monkeypatch, result=SimpleNamespace(current_principal_id=None))
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") is None


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_nonexistent_id_dropped(monkeypatch):
    # A stale / spoofed id that doesn't resolve to a principal must NOT be
    # trusted — resolve_tenant_context raises NotFound, we swallow it and return
    # None so the FK can't be violated.
    _stub_tenant_context(
        monkeypatch, exc=NotFoundException(message="Organization 999999999 not found")
    )
    assert (
        await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "999999999")
        is None
    )


@pytest.mark.asyncio
async def test_resolve_direct_consumer_org_non_member_dropped(monkeypatch):
    # A real org the caller isn't a member of → Forbidden → dropped (no
    # mis-donation), falls back to the caller.
    _stub_tenant_context(
        monkeypatch, exc=ForbiddenException(message="Not a member of organization 7")
    )
    assert await middlewares._resolve_direct_consumer_org(_REQUEST, _USER, "7") is None
