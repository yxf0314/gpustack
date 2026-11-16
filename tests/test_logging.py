import json
import logging
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import uvicorn

from gpustack import logging as app_logging


@pytest.fixture
def configure_logging(monkeypatch):
    root = logging.getLogger()
    level = root.level
    configured = False

    def configure(tmp_path, role="server", *, endpoint="http://collector/v1/logs"):
        nonlocal configured
        if not configured:
            monkeypatch.setattr(root, "handlers", [])
            configured = True
        cfg = SimpleNamespace(
            debug=True, log_dir=str(tmp_path), log_otlp_endpoint=endpoint
        )
        app_logging.setup_application_logging(cfg, role)
        return root.handlers[0]

    yield configure
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)


def test_logging_without_endpoint_keeps_text(configure_logging, tmp_path, capsys):
    configure_logging(tmp_path, endpoint=None)

    logging.getLogger("test.application").info("hello")

    assert " - test.application - INFO - hello" in capsys.readouterr().err
    assert not (tmp_path / "application").exists()


def test_uvicorn_startup_keeps_application_file_open(configure_logging, tmp_path):
    handler = configure_logging(tmp_path)
    uvicorn.Config(
        app="gpustack.server.app:create_app",
        log_config=app_logging.uvicorn_log_config(),
    )

    logging.getLogger("test.application").info("API started")
    handler.close()

    row = (tmp_path / "application/server.jsonl").read_text()
    assert json.loads(row)["message"] == "API started"


@pytest.mark.parametrize("role", ["server", "worker"])
def test_application_log_matches_console_and_preserves_context(
    configure_logging, tmp_path, capsys, role
):
    handler = configure_logging(tmp_path, role)
    assert configure_logging(tmp_path, role) is handler
    logger = logging.getLogger("test.application")
    logger.info("starting")
    if role == "worker":
        app_logging.set_worker_log_identity(42, 7, "qa-worker")

    try:
        raise ValueError("test failure")
    except ValueError:
        logger.exception(
            "failed\n继续",
            extra={
                "request_id": "request-1",
                "model_route_id": 11,
                "trace_id": "a" * 32,
                "span_id": "b" * 16,
                "trace_flags": 1,
                "authorization": "never copy this field",
            },
        )

    handler.close()
    console = capsys.readouterr().err
    archive = tmp_path / "application" / f"{role}.jsonl"
    assert archive.read_text() == console
    start, failure = [json.loads(line) for line in console.splitlines()]
    assert start["service"] == f"gpustack.{role}"
    assert "worker_id" not in start
    assert failure["message"] == "failed\n继续"
    assert failure["severity"] == "ERROR"
    assert "ValueError: test failure" in failure["exception"]
    assert failure["request_id"] == "request-1"
    assert failure["trace_id"] == "a" * 32
    assert failure["span_id"] == "b" * 16
    assert failure["trace_flags"] == 1
    assert failure["model_route_id"] == 11
    assert "authorization" not in failure
    if role == "worker":
        assert failure["worker_id"] == 42
        assert failure["worker_name"] == "qa-worker"
        assert failure["cluster_id"] == 7


def test_application_archive_rotation_is_bounded(
    configure_logging, tmp_path, monkeypatch
):
    monkeypatch.setattr(app_logging, "_APPLICATION_LOG_FILE_BYTES", 500)
    handler = configure_logging(tmp_path)
    for i in range(12):
        logging.getLogger("test.application").info("%d %s", i, "用" * 40)

    handler.close()

    directory = tmp_path / "application"
    assert {path.name for path in directory.iterdir()} == {
        "server.jsonl",
        "server.jsonl.1",
        "server.jsonl.2",
    }
    assert all(path.stat().st_size <= 500 for path in directory.iterdir())


def test_full_archive_queue_keeps_writing_console(
    configure_logging, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(app_logging, "_APPLICATION_LOG_QUEUE_BYTES", 256)
    started = Event()
    release = Event()
    emit = app_logging._ApplicationLogFile.emit

    def write_delayed(handler, record):
        started.set()
        if release.wait(timeout=3):
            emit(handler, record)

    monkeypatch.setattr(app_logging._ApplicationLogFile, "emit", write_delayed)
    handler = configure_logging(tmp_path)

    logger = logging.getLogger("test.application")
    try:
        logger.info("disk writing")
        assert started.wait(timeout=2)
        logger.info("queued")
        logger.info("stays on console")
        console = [
            json.loads(line)["message"] for line in capsys.readouterr().err.splitlines()
        ]
        assert console == ["disk writing", "queued", "stays on console"]
    finally:
        release.set()
    handler.close()

    archive = (tmp_path / "application" / "server.jsonl").read_text()
    assert [json.loads(line)["message"] for line in archive.splitlines()] == [
        "disk writing",
        "queued",
    ]


def test_archive_write_failure_does_not_disable_console(
    configure_logging, tmp_path, monkeypatch, capsys
):
    def deny_archive(*args, **kwargs):
        raise PermissionError("read-only directory")

    monkeypatch.setattr(app_logging._ApplicationLogFile, "_open", deny_archive)
    handler = configure_logging(tmp_path)

    logging.getLogger("test.application").info("still serving")
    handler.close()

    messages = [
        json.loads(line)["message"] for line in capsys.readouterr().err.splitlines()
    ]
    assert "still serving" in messages
    assert not (Path(tmp_path) / "application" / "server.jsonl").exists()
