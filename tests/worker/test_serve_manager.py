from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance


def _build_serve_manager(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def test_sync_model_instances_state_marks_main_unreachable_when_subordinate_unreachable():
    manager, clientset = _build_serve_manager()

    model_instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.worker_ip = "127.0.0.1"
    model_instance.port = 8000
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.UNREACHABLE,
                state_message="Worker is unreachable from the server",
            )
        ],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.UNREACHABLE,
        state_message=(
            "Distributed serving unreachable in subordinate worker "
            "10.0.0.2: Worker is unreachable from the server."
        ),
    )


def test_restart_error_model_instance_uses_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.ERROR,
    )
    model_instance.restart_count = 20
    model_instance.last_restart_time = datetime.now(timezone.utc)

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
        patch("gpustack.worker.serve_manager.logger"),
    ):
        manager._restart_error_model_instance(model_instance)

    update_model_instance.assert_called_once_with(
        model_instance.id,
        restart_count=21,
        last_restart_time=ANY,
        state=ModelInstanceStateEnum.SCHEDULED,
        state_message="",
    )


def test_restart_model_instance_preserves_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.SCHEDULED,
    )
    manager._restart_backoff_counts[model_instance.id] = 1

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_start_model_instance"),
    ):
        manager._restart_model_instance(model_instance)

    assert manager._restart_backoff_counts[model_instance.id] == 1


def test_cleanup_old_logs_keeps_only_current_and_previous_restart(tmp_path: Path):
    """Keep main/container logs for R and R-1; delete older restart_count files."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid = 42
    for name in (
        f"{mid}.0.log",
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 2)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ]


def test_cleanup_old_logs_restart_zero_keeps_only_zero(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid = 7
    for name in (f"{mid}.0.log", f"{mid}.1.log", f"{mid}.container.1.log"):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 0)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [f"{mid}.0.log"]


def _build_distributed_follower_instance(backend_parameters):
    """2-node distributed instance: leader on worker 1, follower on worker 2."""
    model_instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    # _handle_model_instance_event re-validates the event payload, so source
    # (a required field) must be set.
    model_instance.source = SourceEnum.HUGGING_FACE
    model_instance.huggingface_repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_instance.worker_ip = "10.0.0.1"
    model_instance.port = 8000
    # ports[0] is the serving port, identical on every node (bound to each
    # worker's own IP); the rest are DP/master/connecting ports.
    model_instance.ports = [8000, 8001, 8002, 8003]
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.RUNNING,
            )
        ],
    )
    model = new_model(
        1,
        "test",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend_parameters=backend_parameters,
    )
    model.backend = BackendEnum.VLLM
    return model_instance, model


def _drive_follower_event(manager, model_instance, model):
    """Drive an UPDATED event on a follower worker, stubbing workload start."""
    event = Event(type=EventType.UPDATED, data=model_instance)
    with (
        # logger.trace is a custom level not registered in the bare test logger.
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_get_model", return_value=model),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch.object(manager, "_start_model_instance"),
    ):
        manager._handle_model_instance_event(event)


def test_hybrid_lb_follower_is_cached_for_routing():
    """A hybrid-LB follower serves its own API, so the follower worker must
    track the instance by id and resolve its local serving port (ports[0])."""
    manager, _clients = _build_serve_manager(worker_id=2)
    model_instance, model = _build_distributed_follower_instance(
        [
            "--data-parallel-hybrid-lb",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "2",
        ]
    )

    _drive_follower_event(manager, model_instance, model)

    assert model_instance.id in manager._model_instance_by_instance_id
    assert manager.get_instance_port_by_model_instance_id(model_instance.id) == 8000


def test_external_lb_follower_is_cached_for_routing():
    """An external-LB follower also serves its own API, so it must be tracked
    by id and resolve its local serving port, same as hybrid-LB."""
    manager, _clients = _build_serve_manager(worker_id=2)
    model_instance, model = _build_distributed_follower_instance(
        [
            "--data-parallel-external-lb",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "2",
        ]
    )

    _drive_follower_event(manager, model_instance, model)

    assert model_instance.id in manager._model_instance_by_instance_id
    assert manager.get_instance_port_by_model_instance_id(model_instance.id) == 8000


def test_headless_follower_is_not_cached_for_routing():
    """A non-hybrid (headless) follower does not serve an API, so it must stay
    out of the by-id map and never receive routed traffic."""
    manager, _clients = _build_serve_manager(worker_id=2)
    model_instance, model = _build_distributed_follower_instance(
        [
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "2",
            "--data-parallel-size-local",
            "1",
        ]
    )

    _drive_follower_event(manager, model_instance, model)

    assert model_instance.id not in manager._model_instance_by_instance_id
    assert manager.get_instance_port_by_model_instance_id(model_instance.id) is None
