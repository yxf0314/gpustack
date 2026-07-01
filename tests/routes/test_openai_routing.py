import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.http_proxy.strategies import WorkerRoundRobinStrategy
from gpustack.routes import openai as openai_routes
from gpustack.routes.models import validate_model_in
from gpustack.schemas.models import (
    DistributedServers,
    ModelCreate,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from tests.utils.model import new_model


# ---------------------------------------------------------------------------
# _select_serving_worker_id — request-path worker selection (routes/openai.py)
# ---------------------------------------------------------------------------


def _distributed_instance(
    instance_id, leader_worker_id, subordinate_worker_ids, subordinate_states=None
):
    # Subordinates default to RUNNING, since only RUNNING nodes receive traffic.
    if subordinate_states is None:
        subordinate_states = [ModelInstanceStateEnum.RUNNING] * len(
            subordinate_worker_ids
        )
    instance = ModelInstance(id=instance_id, model_id=5, worker_id=leader_worker_id)
    instance.ports = [8000]
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=worker_id, state=state)
            for worker_id, state in zip(subordinate_worker_ids, subordinate_states)
        ]
    )
    return instance


def test_select_serving_worker_id_non_hybrid_returns_leader():
    """Without hybrid-LB the followers are headless, so every request must go to
    the single leader endpoint."""
    instance = _distributed_instance(
        101, leader_worker_id=1, subordinate_worker_ids=[2, 3]
    )
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=["--tensor-parallel-size=8", "--data-parallel-size=2"],
    )
    assert [
        openai_routes._select_serving_worker_id(instance, model) for _ in range(3)
    ] == [1, 1, 1]


def test_select_serving_worker_id_hybrid_lb_round_robins_nodes():
    """Hybrid-LB: each node serves its own API, so requests round-robin across
    the leader and follower workers."""
    instance = _distributed_instance(
        102, leader_worker_id=1, subordinate_worker_ids=[2, 3]
    )
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=[
            "--data-parallel-hybrid-lb",
            "--tensor-parallel-size=8",
            "--data-parallel-size=2",
        ],
    )
    picks = [openai_routes._select_serving_worker_id(instance, model) for _ in range(4)]
    assert picks == [1, 2, 3, 1]


def test_select_serving_worker_id_skips_non_running_subordinate():
    """A subordinate marked ERROR is excluded from the round-robin until it
    recovers; the leader and remaining RUNNING subordinate keep serving."""
    instance = _distributed_instance(
        104,
        leader_worker_id=1,
        subordinate_worker_ids=[2, 3],
        subordinate_states=[
            ModelInstanceStateEnum.ERROR,
            ModelInstanceStateEnum.RUNNING,
        ],
    )
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=[
            "--data-parallel-hybrid-lb",
            "--tensor-parallel-size=8",
            "--data-parallel-size=2",
        ],
    )
    picks = [openai_routes._select_serving_worker_id(instance, model) for _ in range(4)]
    # worker 2 (ERROR) is skipped: only leader 1 and RUNNING subordinate 3 cycle.
    assert picks == [1, 3, 1, 3]


def test_select_serving_worker_id_external_lb_round_robins_nodes():
    """External-LB: each rank serves its own API, so requests round-robin across
    the leader and subordinate workers, same as hybrid-LB."""
    instance = _distributed_instance(
        103, leader_worker_id=1, subordinate_worker_ids=[2, 3]
    )
    model = new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=[
            "--data-parallel-external-lb",
            "--tensor-parallel-size=8",
            "--data-parallel-size=4",
        ],
    )
    picks = [openai_routes._select_serving_worker_id(instance, model) for _ in range(4)]
    assert picks == [1, 2, 3, 1]


# ---------------------------------------------------------------------------
# WorkerRoundRobinStrategy — the primitive backing _select_serving_worker_id
# (http_proxy/strategies.py)
# ---------------------------------------------------------------------------


def test_worker_round_robin_cycles_and_resets_on_change():
    strategy = WorkerRoundRobinStrategy()
    assert [strategy.select_worker_id(1, [10, 20, 30]) for _ in range(4)] == [
        10,
        20,
        30,
        10,
    ]
    # Changing the worker set rebuilds the cycle from the start.
    assert strategy.select_worker_id(1, [40, 50]) == 40
    assert strategy.select_worker_id(1, [40, 50]) == 50


def test_worker_round_robin_evicts_least_recently_used():
    strategy = WorkerRoundRobinStrategy()
    strategy._max_instances = 2
    strategy.select_worker_id(1, [10])
    strategy.select_worker_id(2, [20])
    # Touch instance 1 so instance 2 becomes the least-recently-used.
    strategy.select_worker_id(1, [10])
    # Adding a third instance exceeds the cap and evicts the LRU entry (2).
    strategy.select_worker_id(3, [30])
    assert 2 not in strategy._iterators
    assert 2 not in strategy._worker_id_lists
    assert set(strategy._iterators) == {1, 3}


# ---------------------------------------------------------------------------
# validate_model_in — manual-distributed gate (routes/models.py)
# ---------------------------------------------------------------------------


def _manual_distributed_model_create(enable_model_route):
    base = new_model(
        1,
        "manual-dp",
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        env={"GPUSTACK_MANUAL_DISTRIBUTED": "1"},
    )
    model_in = ModelCreate.model_validate(base.model_dump())
    model_in.enable_model_route = enable_model_route
    return model_in


@pytest.mark.asyncio
async def test_validate_manual_distributed_requires_route_disabled():
    # enable_model_route defaults to None (not False): manual-distributed must be
    # rejected, since it registers no upstream and the route would be empty.
    # No gpu_selector, so validate_gpu_ids is skipped and session is unused.
    model_in = _manual_distributed_model_create(enable_model_route=None)
    with pytest.raises(BadRequestException) as exc_info:
        await validate_model_in(session=None, model_in=model_in)
    assert "enable_model_route" in exc_info.value.message


@pytest.mark.asyncio
async def test_validate_manual_distributed_route_disabled_ok():
    model_in = _manual_distributed_model_create(enable_model_route=False)
    # Passes the manual-distributed gate; remaining validation needs no session.
    await validate_model_in(session=None, model_in=model_in)


@pytest.mark.asyncio
async def test_validate_non_manual_distributed_route_enabled_ok():
    # A normal model with the route enabled is unaffected by the manual gate.
    base = new_model(1, "normal", huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model_in = ModelCreate.model_validate(base.model_dump())
    model_in.enable_model_route = True
    await validate_model_in(session=None, model_in=model_in)
