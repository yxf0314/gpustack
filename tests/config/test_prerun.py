from pathlib import Path

import pytest
import yaml

from gpustack.cmd import prerun
from gpustack.cmd.prerun import check_ports_availability, prepare_s6_overlay
from gpustack.config.config import Config


def test_prepare_s6_overlay_enables_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)

    stale_migration = user_contents / "gpustack-migration"
    stale_migration.write_text("")

    # prepare_s6_overlay should cleanup the base dir and generate base on the input services
    prepare_s6_overlay(["postgres"], ["gpustack-migration"], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert (user_contents / "gpustack-migration").exists()


def test_prepare_s6_overlay_cleans_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)
    (user_contents / "gpustack-migration").write_text("")

    prepare_s6_overlay(["postgres"], [], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert not (user_contents / "gpustack-migration").exists()


def _one_gpustack_port(monkeypatch):
    monkeypatch.setattr(prerun, "ports_for_services", lambda cfg: {50000: "gpustack"})


def test_check_ports_availability_accepts_a_timewait_only_port(monkeypatch):
    """A recreated container leaves TIME_WAIT sockets on the fixed service
    ports; the check must not fail a port the services can rebind."""
    _one_gpustack_port(monkeypatch)
    monkeypatch.setattr(prerun, "is_port_available", lambda port, host=None: False)
    monkeypatch.setattr(prerun, "is_port_reusable", lambda port: True)

    check_ports_availability(None)


def test_check_ports_availability_fails_a_port_held_by_a_live_listener(monkeypatch):
    _one_gpustack_port(monkeypatch)
    monkeypatch.setattr(prerun, "is_port_available", lambda port, host=None: False)
    monkeypatch.setattr(prerun, "is_port_reusable", lambda port: False)

    with pytest.raises(Exception, match="not available"):
        check_ports_availability(None)


@pytest.fixture
def collector_path(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run))
    template = (
        Path(__file__).parents[2] / "pack/rootfs/etc/gpustack/logging/collector.yaml"
    )
    monkeypatch.setattr(prerun, "APPLICATION_LOG_COLLECTOR_TEMPLATE", template)
    return run / "logging" / "collector.yaml"


@pytest.mark.parametrize("worker", [False, True])
def test_log_collector_is_independent_of_builtin_observability(
    tmp_path, monkeypatch, collector_path, worker
):
    monkeypatch.setenv(
        "GPUSTACK_LOG_OTLP_ENDPOINT", "https://collector.example/v1/logs"
    )
    monkeypatch.setenv("GPUSTACK_LOG_OTLP_AUTHORIZATION", "Bearer literal$token")
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_dir=str(tmp_path / "logs"),
        disable_builtin_observability=True,
        gateway_mode="disabled",
        server_url="http://server" if worker else None,
        token="test" if worker else None,
    )

    assert prerun.prepare_log_collection(cfg)
    assert "otel-collector" in prerun.determine_enabled_services(cfg, logs_active=True)
    assert "prometheus" not in prerun.determine_enabled_services(cfg, logs_active=True)
    assert collector_path.stat().st_mode & 0o777 == 0o600
    rendered = yaml.safe_load(collector_path.read_text())
    include = rendered["receivers"]["filelog/application"]["include"]
    assert rendered["receivers"]["filelog/application"]["operators"][0]["parse_ints"]
    assert str(tmp_path / "logs/application/server.jsonl") in include
    assert str(tmp_path / "logs/application/worker.jsonl") in include
    exporter = rendered["exporters"]["otlp_http/logs"]
    assert exporter["logs_endpoint"] == "https://collector.example/v1/logs"
    assert exporter["headers"] == {"Authorization": "Bearer literal$$token"}
    assert exporter["sending_queue"]["queue_size"] == 16 * 1024**2
    assert exporter["retry_on_failure"]["max_elapsed_time"] == "300s"


def test_disabled_collector_removes_stale_config_and_service(tmp_path, collector_path):
    collector_path.parent.mkdir(parents=True)
    collector_path.write_text("stale")
    cfg = Config(data_dir=str(tmp_path / "data"), log_otlp_endpoint=None)

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()
    assert "otel-collector" not in prerun.determine_enabled_services(cfg)

    s6_base_path = tmp_path / "s6-rc.d"
    prepare_s6_overlay(["otel-collector"], [], s6_base_path)
    service = s6_base_path / "user/contents.d/otel-collector"
    assert service.exists()
    prepare_s6_overlay([], [], s6_base_path)
    assert not service.exists()


@pytest.mark.parametrize(
    ("endpoint", "authorization"),
    [
        ("http://user:private@collector/v1/logs", None),
        ("http://collector/v1/logs", "Bearer private"),
        ("https://collector/v1/logs", "Bearer private\nX-Other: invalid"),
        ("grpc://collector/v1/logs", None),
    ],
)
def test_invalid_otlp_settings_skip_collector_without_logging_secrets(
    tmp_path, collector_path, caplog, endpoint, authorization
):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_otlp_endpoint=endpoint,
        log_otlp_authorization=authorization,
    )

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()
    assert "private" not in caplog.text
