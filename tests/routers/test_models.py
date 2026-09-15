from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import models as models_route
from gpustack.routes.models import (
    create_model,
    export_models,
    import_models,
    update_model,
)
from gpustack.routes.model_common import ModelStateFilterEnum
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.deployment_document import (
    DeploymentExportRequest,
    DeploymentImportRequest,
    dump_deployments,
    load_deployments,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteTarget,
)
from gpustack.schemas.models import (
    GPUSelector,
    LoraListEntry,
    Model,
    ModelCreate,
    ModelUpdate,
    SourceEnum,
)
from gpustack.schemas.principals import (
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.utils.export_limits import attachment_headers

DEFAULT_ORG_ID = platform_principal_id()
CUSTOM_ORG_ID = 5
OTHER_ORG_ID = 7
CLUSTER_ID = 101


def _ctx(current_principal_id, is_admin=False, accessible_cluster_ids=None):
    user = MagicMock()
    user.id = 99
    user.is_admin = is_admin
    # Tenant helpers compare user.kind against PrincipalType.SYSTEM; pin it
    # to a non-SYSTEM kind so a bare mock can't drift into the SYSTEM bypass.
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=current_principal_id,
        org_role=None,
        accessible_cluster_ids=set(accessible_cluster_ids or []),
    )


def _model_create(cluster_id=None):
    return ModelCreate(
        name="m1",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        cluster_id=cluster_id,
    )


def _cluster(owner_principal_id, cluster_id=CLUSTER_ID, deleted=False):
    cluster = MagicMock()
    cluster.id = cluster_id
    cluster.owner_principal_id = owner_principal_id
    cluster.deleted_at = object() if deleted else None
    return cluster


@pytest.mark.asyncio
async def test_create_model_rejects_default_org_cluster_for_custom_org(monkeypatch):
    """A custom org cannot deploy onto a visible cluster owned by another
    org (e.g. the Default org's shared cluster) — 403, not 404."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(DEFAULT_ORG_ID)),
    )

    with pytest.raises(ForbiddenException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A cluster the caller can't see is reported as missing (404), not
    forbidden (403), so cross-tenant cluster ids can't be probed."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_missing_cluster(monkeypatch):
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_rejects_deleted_cluster(monkeypatch):
    """A soft-deleted cluster is treated as missing (404)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID, deleted=True)),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_allows_own_org_cluster(monkeypatch):
    """An own-org cluster passes the org-alignment check and proceeds to
    the name-uniqueness check (signalled here by AlreadyExists)."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(CUSTOM_ORG_ID)),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        AsyncMock(return_value=MagicMock()),
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            _model_create(cluster_id=CLUSTER_ID),
        )


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_derives_owner_from_cluster(monkeypatch):
    """Admin in "All" mode (no principal context) derives the owning org
    from the chosen cluster; the ownership check then passes even for a
    non-default org's cluster, and the model is stamped with that owner."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=_cluster(OTHER_ORG_ID)),
    )
    one_by_fields = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_fields",
        one_by_fields,
    )

    with pytest.raises(AlreadyExistsException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=CLUSTER_ID),
        )

    # The uniqueness pre-check runs against the org derived from the
    # cluster, not the platform default.
    assert one_by_fields.await_args.args[1]["owner_principal_id"] == OTHER_ORG_ID


@pytest.mark.asyncio
async def test_create_model_admin_all_mode_rejects_missing_cluster(monkeypatch):
    """Admin "All" mode still rejects a non-existent cluster rather than
    stamping the model with the platform default."""
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(NotFoundException):
        await create_model(
            MagicMock(),
            _ctx(current_principal_id=None, is_admin=True),
            _model_create(cluster_id=999),
        )


async def _run_update(monkeypatch, ctx, cluster_return):
    """Drive update_model for an owned model pointed at ``cluster_return``."""
    model = MagicMock()
    model.owner_principal_id = CUSTOM_ORG_ID
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_resource_visible",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.Cluster.one_by_id",
        AsyncMock(return_value=cluster_return),
    )
    await update_model(
        MagicMock(),
        ctx,
        1,
        ModelUpdate(
            name="m1",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            cluster_id=CLUSTER_ID,
        ),
    )


@pytest.mark.asyncio
async def test_update_model_rejects_cross_org_cluster(monkeypatch):
    """A visible cluster owned by another org is a 403 on update."""
    with pytest.raises(ForbiddenException):
        await _run_update(
            monkeypatch,
            _ctx(CUSTOM_ORG_ID, accessible_cluster_ids=[CLUSTER_ID]),
            _cluster(DEFAULT_ORG_ID),
        )


@pytest.mark.asyncio
async def test_update_model_hides_non_visible_cluster_as_missing(monkeypatch):
    """A non-visible cluster is a 404 on update, not a 403 — no probing."""
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), _cluster(OTHER_ORG_ID))


@pytest.mark.asyncio
async def test_update_model_rejects_missing_cluster(monkeypatch):
    with pytest.raises(NotFoundException):
        await _run_update(monkeypatch, _ctx(CUSTOM_ORG_ID), None)


@pytest.mark.parametrize(
    "ready, replicas, state, expected",
    [
        (2, 3, ModelStateFilterEnum.READY, True),
        (0, 3, ModelStateFilterEnum.READY, False),
        (0, 3, ModelStateFilterEnum.NOT_READY, True),
        (2, 3, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.STOPPED, True),
        (0, 3, ModelStateFilterEnum.STOPPED, False),
        (0, 3, None, True),
    ],
)
def test_model_watch_filter_applies_state(
    monkeypatch, ready, replicas, state, expected
):
    """The /models watch stream honors ``state`` via replica counts."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=state
    )
    data = SimpleNamespace(ready_replicas=ready, replicas=replicas)
    assert visible(data) is expected


def test_model_watch_filter_passes_id_only_delete_events(monkeypatch):
    """ID-only DELETED payloads lack replica counts and must not be dropped
    by the state filter, else watch clients hold stale rows."""
    monkeypatch.setattr(models_route, "cluster_scoped_system", lambda ctx: False)

    visible = models_route._make_model_watch_filter(
        ctx=None, categories=None, state=ModelStateFilterEnum.READY
    )
    assert visible({"id": 7}) is True


@pytest.mark.asyncio
async def test_update_model_rejects_gpu_selector_on_vgpu_model(monkeypatch):
    """A sparse PUT setting gpu_selector on a model that already carries
    gpu_type_selector must fail mutual-exclusion validation against the
    merged (stored + request) state, not just the request payload."""
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.models import GPUSelector, GPUTypeSelector

    stored = MagicMock()
    stored.owner_principal_id = CUSTOM_ORG_ID
    stored.cluster_id = CLUSTER_ID
    stored.gpu_type_selector = GPUTypeSelector(
        type="pool-a100",
        accelerator_sliced_memory_percentage=50,
        accelerator_sliced_cores_percentage=50,
    )
    stored.gpu_selector = None
    monkeypatch.setattr(
        "gpustack.routes.models.Model.one_by_id",
        AsyncMock(return_value=stored),
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_resource_visible",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gpustack.routes.models.assert_cluster_belongs_to_org",
        AsyncMock(),
    )

    with pytest.raises(BadRequestException):
        await update_model(
            MagicMock(),
            _ctx(CUSTOM_ORG_ID),
            1,
            ModelUpdate(
                name="m1",
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/repo",
                gpu_selector=GPUSelector(
                    gpu_ids=["worker-1:nvidia:0"],
                    gpus_per_replica=1,
                ),
            ),
        )


# ==================== Deployment YAML export / import ====================
#
# The document schema (what an exported entry carries, how strictly it reads
# back) and the two routes, the latter against an in-memory SQLite database
# with the handlers driven directly through a ``TenantContext``.

EXPORTED_AT = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)

TABLES = (
    Principal.__table__,
    Cluster.__table__,
    Model.__table__,
    ModelRoute.__table__,
    ModelRouteTarget.__table__,
    ModelRoutePrincipalLink.__table__,
)

# What a user would hand-write: a routed base model with a LoRA, explicit GPU
# placement and a credential, plus a plain embedding model.
DOCUMENT = """
- name: qwen3-8b
  source: huggingface
  huggingface_repo_id: Qwen/Qwen3-8B
  backend: vLLM
  backend_parameters:
  - --max-model-len=32768
  env:
    HF_TOKEN: hf_xxx
  gpu_selector:
    gpu_ids:
    - worker-1:cuda:0
  lora_list:
  - lora_name: sql
    lora_repo_name: org/sql-lora
  enable_model_route: true
- name: bge-m3
  source: huggingface
  huggingface_repo_id: BAAI/bge-m3
  replicas: 2
"""


def _model_row(
    name, cluster_id=1, owner_principal_id=DEFAULT_ORG_ID, **fields
) -> Model:
    fields.setdefault("source", SourceEnum.HUGGING_FACE)
    fields.setdefault("huggingface_repo_id", f"org/{name}")
    return Model(
        name=name,
        cluster_id=cluster_id,
        owner_principal_id=owner_principal_id,
        **fields,
    )


def _exported_row(**overrides) -> Model:
    """A fully configured row with every server-derived field set, so the dump
    tests can check that none of those leak into the document."""
    fields = dict(
        id=7,
        name="qwen3-8b",
        huggingface_repo_id="Qwen/Qwen3-8B",
        backend="vLLM",
        backend_version="0.11.0",
        backend_parameters=["--max-model-len=32768"],
        env={"HF_TOKEN": "hf_xxx"},
        gpu_selector=GPUSelector(gpu_ids=["worker-1:cuda:0"]),
        lora_list=[
            LoraListEntry(
                lora_name="qwen3-8b:sql",
                lora_repo_name="org/sql-lora",
                path="/var/lib/gpustack/cache/sql",
                model_file_id=9,
            )
        ],
        meta={"n_params": 8_000_000_000},
        ready_replicas=1,
        cluster_id=3,
        owner_principal_id=CUSTOM_ORG_ID,
        access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
    )
    fields.update(overrides)
    return _model_row(fields.pop("name"), **fields)


def test_dump_keeps_user_input_and_drops_server_state():
    text = dump_deployments(
        [_exported_row()], route_backed_ids={7}, exported_at=EXPORTED_AT
    )

    assert text.startswith("# Exported from GPUStack v")
    assert "at 2026-09-07T10:00:00Z\n" in text
    (entry,) = yaml.safe_load(text)

    # ``name`` leads, everything else keeps the schema's declaration order.
    assert list(entry)[:3] == ["name", "source", "huggingface_repo_id"]
    assert list(entry)[-1] == "enable_model_route"
    assert entry["enable_model_route"] is True
    assert entry["backend_version"] == "0.11.0"
    assert entry["backend_parameters"] == ["--max-model-len=32768"]
    assert entry["env"] == {"HF_TOKEN": "hf_xxx"}
    assert entry["gpu_selector"] == {"gpu_ids": ["worker-1:cuda:0"]}
    assert entry["lora_list"] == [
        {"lora_name": "sql", "lora_repo_name": "org/sql-lora", "source": "huggingface"}
    ]

    for field in (
        "id",
        "created_at",
        "updated_at",
        "ready_replicas",
        "meta",
        "cluster_id",
        "owner_principal_id",
        "access_policy",
    ):
        assert field not in entry, field
    # ``None`` never round-trips into an explicit null.
    assert "description" not in entry
    assert "worker_selector" in entry  # an empty dict is user input, kept


def test_dump_is_byte_stable_and_infers_the_route_flag_per_model():
    models = [_exported_row(), _exported_row(id=8, name="second", lora_list=None)]

    first = dump_deployments(models, route_backed_ids={7}, exported_at=EXPORTED_AT)
    second = dump_deployments(models, route_backed_ids={7}, exported_at=EXPORTED_AT)
    assert first == second

    flags = [entry["enable_model_route"] for entry in yaml.safe_load(first)]
    assert flags == [True, False]


def test_dump_omits_scheduler_placement_for_an_auto_scheduled_model():
    """Where the scheduler put a deployment must never reach the document:
    those GPUs need not exist in the cluster it is imported into. Placement
    lives on ModelInstance, and only a user writes Model.gpu_selector --
    a sentinel for both, since neither is enforced by a type."""
    auto, manual = yaml.safe_load(
        dump_deployments(
            [
                _exported_row(gpu_selector=None, worker_selector={"zone": "a"}),
                _exported_row(
                    id=8,
                    name="manual",
                    gpu_selector=GPUSelector(
                        gpu_ids=["worker-1:cuda:0", "worker-1:cuda:1"],
                        gpus_per_replica=2,
                    ),
                ),
            ],
            route_backed_ids=set(),
            exported_at=EXPORTED_AT,
        )
    )

    assert "gpu_selector" not in auto
    assert "gpu_type_selector" not in auto
    assert auto["worker_selector"] == {"zone": "a"}
    # A manual pick is the user's own intent, and survives verbatim.
    assert manual["gpu_selector"] == {
        "gpu_ids": ["worker-1:cuda:0", "worker-1:cuda:1"],
        "gpus_per_replica": 2,
    }


@pytest.mark.parametrize(
    "text, message",
    [
        # The pre-release wrapper key is just another mapping now.
        ("deployments:\n- name: a\n", "must be a list of deployments"),
        ("just a string\n", "must be a list of deployments"),
        ("\n", "must be a list of deployments"),
        ("- [\n", "not valid YAML"),
    ],
)
def test_load_rejects_a_malformed_document_outright(text, message):
    with pytest.raises(ValueError) as raised:
        load_deployments(text)
    assert message in str(raised.value)


def test_load_reports_every_entry_problem_and_keeps_the_good_entries():
    text = """
- name: good
  source: huggingface
  huggingface_repo_id: org/good
  .anchor: &shared {}
- name: broken
  source: huggingface
  huggingface_repo_id: org/broken
  replicas: -1
  colour: red
  id: 12
  cluster_id: 3
  gpu_selector:
    gpu_ids: [worker-1:cuda:0]
    typo: 1
  lora_list:
  - lora_name: sql
    lora_repo_name: org/sql-lora
    colour: red
    path: /var/lib/gpustack/cache/sql
- name: good
  source: huggingface
  huggingface_repo_id: org/again
- not a mapping
"""
    loaded = load_deployments(text)

    # One entry per document item, in order, whether or not it parsed.
    assert [(item.index, item.name) for item in loaded] == [
        (0, "good"),
        (1, "broken"),
        (2, "good"),
        (3, None),
    ]
    assert [item.entry is not None for item in loaded] == [True, False, False, False]
    assert loaded[0].label == "deployment[0] (good)"
    assert loaded[3].label == "deployment[3]"

    # The schema error keeps pydantic's own text; the entry carries it
    # unlabelled, since the entry it belongs to is right there.
    broken = loaded[1].errors
    assert broken[2].startswith("replicas: ")
    assert broken[:2] + broken[3:] == [
        "server-managed field(s) are not allowed: cluster_id, id, lora_list[0].path",
        "unknown field(s): colour, gpu_selector.typo, lora_list[0].colour",
    ]
    assert loaded[2].errors == [
        "duplicate name, already used by deployment[0]",
    ]
    assert loaded[3].errors == ["must be a mapping"]


def test_attachment_header_fallback_stays_a_well_formed_quoted_string():
    header = attachment_headers('模型"x.yaml')["Content-Disposition"]
    assert header == (
        'attachment; filename="___x.yaml"; '
        "filename*=UTF-8''%E6%A8%A1%E5%9E%8B%22x.yaml"
    )


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        for table in TABLES:
            await conn.run_sync(table.create)
    yield e
    await e.dispose()


@pytest.fixture
def no_gpu_lookup(monkeypatch):
    """``gpu_selector`` validation needs live workers; placement is not under test."""
    monkeypatch.setattr("gpustack.routes.models.validate_gpu_ids", AsyncMock())


async def _seed(session: AsyncSession, *rows):
    session.add_all(rows)
    await session.commit()
    return rows


async def _count(session: AsyncSession, table) -> int:
    return (await session.exec(select(func.count()).select_from(table))).one()


async def _route_names(session: AsyncSession):
    return sorted(route.name for route in await ModelRoute.all_by_fields(session))


def _entries(response):
    assert response.media_type == "application/x-yaml"
    return yaml.safe_load(response.body)


def _body_without_header(response) -> str:
    return response.body.decode().split("\n", 1)[1]


async def _import(session, ctx, content, cluster_id=1, dry_run=False):
    return await import_models(
        session,
        ctx,
        DeploymentImportRequest(
            content=content, cluster_id=cluster_id, dry_run=dry_run
        ),
    )


def _plan(result) -> list:
    """The plan as (name, action, changed field names) per entry."""
    return [
        (
            entry.name,
            entry.action and entry.action.value,
            [change.field for change in entry.changes],
        )
        for entry in result.entries
    ]


@pytest.mark.asyncio
async def test_export_scopes_to_the_caller_and_names_the_file(engine):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        first, second, foreign = await _seed(
            session,
            _model_row("first", env={"HF_TOKEN": "hf_xxx"}),
            _model_row("second", cluster_id=2),
            _model_row("foreign", owner_principal_id=CUSTOM_ORG_ID),
        )
        await _seed(
            session,
            ModelRoute(name="first", created_model_id=first.id),
            # A LoRA child route also points at its base model; without the
            # primary route the flag must stay off.
            ModelRoute(name="second:sql", created_model_id=second.id),
        )

        # Everything the Org can see, in id order; the route flag is inferred.
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest()
        )
        entries = _entries(response)
        assert [entry["name"] for entry in entries] == ["first", "second"]
        assert [entry["enable_model_route"] for entry in entries] == [True, False]
        assert entries[0]["env"] == {"HF_TOKEN": "hf_xxx"}
        assert "cluster_id" not in entries[0]
        assert response.headers["content-disposition"].startswith(
            'attachment; filename="gpustack-deployments-'
        )

        # ``cluster_id`` narrows; a single model is named after itself.
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(cluster_id=2)
        )
        assert [entry["name"] for entry in _entries(response)] == ["second"]
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="second.yaml"'
        )

        # A cross-tenant or unknown id is 404 with no partial result; the
        # platform admin in "All" mode sees every Org.
        with pytest.raises(NotFoundException):
            await export_models(
                session,
                _ctx(DEFAULT_ORG_ID),
                DeploymentExportRequest(ids=[first.id, foreign.id]),
            )
        with pytest.raises(NotFoundException):
            await export_models(
                session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(ids=[9999])
            )
        response = await export_models(
            session,
            _ctx(None, is_admin=True),
            DeploymentExportRequest(ids=[foreign.id, second.id]),
        )
        assert [entry["name"] for entry in _entries(response)] == [
            "second",
            "foreign",
        ]

        # A name with shell/quote characters still yields a well-formed header.
        (odd,) = await _seed(session, _model_row('odd"name/v1'))
        response = await export_models(
            session, _ctx(DEFAULT_ORG_ID), DeploymentExportRequest(ids=[odd.id])
        )
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="odd_name_v1.yaml"'
        )


@pytest.mark.asyncio
async def test_import_round_trips_an_export(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )

        result = await _import(session, ctx, DOCUMENT)
        assert result.dry_run is False
        assert _plan(result) == [("qwen3-8b", "create", []), ("bge-m3", "create", [])]
        assert [item.name for item in result.items] == ["qwen3-8b", "bge-m3"]
        assert [item.cluster_id for item in result.items] == [1, 1]
        # Public form: the LoRA name is bare, as everywhere else in the API.
        assert result.items[0].model_dump()["lora_list"][0]["lora_name"] == "sql"
        # The route flag created the base route and the LoRA child route.
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]
        assert await _count(session, ModelRouteTarget.__table__) == 2

        exported = await export_models(session, ctx, DeploymentExportRequest())

        # Re-importing an export of rows that still exist changes nothing.
        result = await _import(session, ctx, exported.body.decode(), dry_run=True)
        assert _plan(result) == [
            ("qwen3-8b", "unchanged", []),
            ("bge-m3", "unchanged", []),
        ]

        for table in (ModelRouteTarget, ModelRoute, Model):
            await session.exec(delete(table))
        await session.commit()
        assert await _count(session, Model.__table__) == 0

        await _import(session, ctx, exported.body.decode())
        exported_again = await export_models(session, ctx, DeploymentExportRequest())
        assert _body_without_header(exported_again) == _body_without_header(exported)
        assert await _route_names(session) == ["qwen3-8b", "qwen3-8b:sql"]


@pytest.mark.asyncio
async def test_import_reports_every_problem_on_its_own_entry(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID),
            _model_row("running", replicas=1),
        )
        document = """
- name: running
  source: huggingface
  huggingface_repo_id: org/running
- name: fine
  source: huggingface
  huggingface_repo_id: org/fine
- name: broken
  source: huggingface
  huggingface_repo_id: org/broken
  replicas: -1
  colour: red
- name: fine
  source: huggingface
  huggingface_repo_id: org/fine-again
- name: bad-params
  source: huggingface
  huggingface_repo_id: org/bad
  backend_parameters:
  - --port=8000
"""
        result = await _import(session, ctx, document, dry_run=True)
        assert result.valid is False
        errors = [entry.errors for entry in result.entries]
        assert errors[0] == [
            "already exists and is running (replicas=1); stop it before overwriting"
        ]
        assert errors[1] == []
        assert errors[2][0] == "unknown field(s): colour"
        assert errors[2][1].startswith("replicas: ")
        assert errors[3] == ["duplicate name, already used by deployment[1]"]
        assert errors[4] == [
            "Setting the port using --port is not supported. Ports are "
            "automatically allocated by GPUStack."
        ]
        assert await _count(session, Model.__table__) == 1

        # Writing the same document is still a 400 — only the preview is
        # error-free — and it names each entry there.
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, document)
        assert "deployment[2] (broken): unknown field(s): colour" in (
            raised.value.message
        )
        assert await _count(session, Model.__table__) == 1

        # A cluster the caller cannot use is one 404, not one error per entry.
        with pytest.raises(NotFoundException):
            await _import(session, ctx, DOCUMENT, cluster_id=42)
        # A document that is not a list has no plan to show.
        with pytest.raises(BadRequestException):
            await _import(session, ctx, "deployments:\n- name: a\n", dry_run=True)
        assert await _count(session, Model.__table__) == 1

        # A LoRA route name owned by another model only conflicts while the
        # routes are created; the error still names the entry, nothing stays.
        await _seed(session, ModelRoute(name="qwen3-8b:sql", created_model_id=999))
        with pytest.raises(BadRequestException) as raised:
            await _import(session, ctx, DOCUMENT)
        assert raised.value.message.startswith("deployment[0] (qwen3-8b): LoRA route")
        assert await _count(session, Model.__table__) == 1


@pytest.mark.asyncio
async def test_dry_run_plans_without_writing(engine, no_gpu_lookup):
    ctx = _ctx(DEFAULT_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session, Cluster(id=1, name="c1", owner_principal_id=DEFAULT_ORG_ID)
        )

        result = await _import(session, ctx, DOCUMENT, dry_run=True)
        assert result.dry_run is True
        assert result.valid is True
        assert result.items == []
        assert _plan(result) == [("qwen3-8b", "create", []), ("bge-m3", "create", [])]
        assert await _count(session, Model.__table__) == 0
        assert await _count(session, ModelRoute.__table__) == 0

        await _import(session, ctx, DOCUMENT)
        assert await _count(session, Model.__table__) == 2

        # Edit the document and the diff names exactly what would change:
        # a raised replica count, and a GPU selector deleted outright.
        edited = DOCUMENT.replace("  replicas: 2", "  replicas: 5").replace(
            "  gpu_selector:\n    gpu_ids:\n    - worker-1:cuda:0\n", ""
        )
        result = await _import(session, ctx, edited, dry_run=True)
        assert _plan(result) == [
            ("qwen3-8b", "update", ["gpu_selector"]),
            ("bge-m3", "update", ["replicas"]),
        ]
        (change,) = result.entries[0].changes
        assert change.current == {"gpu_ids": ["worker-1:cuda:0"]}
        assert change.desired is None
        assert result.entries[1].changes[0].current == 2
        assert result.entries[1].changes[0].desired == 5


@pytest.mark.asyncio
async def test_import_stamps_the_callers_org_and_refuses_foreign_clusters(
    engine, no_gpu_lookup
):
    ctx = _ctx(CUSTOM_ORG_ID)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await _seed(
            session,
            Cluster(id=1, name="platform", owner_principal_id=DEFAULT_ORG_ID),
            Cluster(id=2, name="org", owner_principal_id=CUSTOM_ORG_ID),
        )

        # Another Org's cluster is reported as missing, and nothing is written.
        with pytest.raises(NotFoundException):
            await _import(session, ctx, DOCUMENT, cluster_id=1)
        assert await _count(session, Model.__table__) == 0

        # Own cluster: rows are stamped with the Org, scoped to it, and the
        # Org is granted on the route it asked for.
        result = await _import(session, ctx, DOCUMENT, cluster_id=2)
        assert {item.owner_principal_id for item in result.items} == {CUSTOM_ORG_ID}
        assert {item.access_policy for item in result.items} == {
            AccessPolicyEnum.ALLOWED_PRINCIPALS
        }
        route = await ModelRoute.one_by_field(session, "name", "qwen3-8b")
        granted = await session.exec(
            select(ModelRoutePrincipalLink.principal_id).where(
                ModelRoutePrincipalLink.route_id == route.id
            )
        )
        assert list(granted.all()) == [CUSTOM_ORG_ID]
