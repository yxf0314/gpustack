from gpustack.routes import openai as openai_routes
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstanceSubordinateWorker,
)
from tests.utils.model import new_model


def _distributed_instance(instance_id, leader_worker_id, subordinate_worker_ids):
    instance = ModelInstance(id=instance_id, model_id=5, worker_id=leader_worker_id)
    instance.ports = [8000]
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=worker_id)
            for worker_id in subordinate_worker_ids
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
