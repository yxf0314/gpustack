from collections import deque
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from threading import Condition, Thread
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gpustack.config.config import Config


# Suppress warnings from transformers
# https://github.com/huggingface/transformers/issues/27214
# Note: This should be set before importing transformers
if "TRANSFORMERS_NO_ADVISORY_WARNINGS" not in os.environ:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

TRACE_LEVEL = 5
_TEXT_HANDLER_NAME = "gpustack-console"
_APPLICATION_LOG_FILE_BYTES = 10 * 1024**2
_APPLICATION_LOG_QUEUE_BYTES = 16 * 1024**2

logger = logging.getLogger(__name__)


class _ApplicationLogFormatter(logging.Formatter):
    _fields = (
        "request_id",
        "pd_coordinator_id",
        "trace_id",
        "span_id",
        "trace_flags",
        "cluster_id",
        "worker_id",
        "worker_name",
        "model_id",
        "model_route_id",
        "model_instance_id",
        "owner_principal_id",
        "restart_count",
        "method",
        "path",
        "response_code",
        "duration",
    )

    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.identity: dict = {}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "service": f"gpustack.{self.role}",
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for field in self._fields:
            value = getattr(record, field, None)
            if isinstance(value, (str, int)):
                payload[field] = value
        payload.update(self.identity)
        return json.dumps(payload, ensure_ascii=False)


class _ApplicationLogFile(RotatingFileHandler):
    def __init__(self, path: Path):
        super().__init__(
            path,
            maxBytes=_APPLICATION_LOG_FILE_BYTES,
            backupCount=2,
            encoding="utf-8",
            errors="backslashreplace",
            delay=True,
        )
        self._warned = False

    def _open(self):
        Path(self.baseFilename).parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        return super()._open()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.stream is None:
            self.stream = self._open()
        return self.stream.tell() + record.archive_bytes > self.maxBytes

    def handleError(self, record: logging.LogRecord) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("Unable to write application log copies to disk.")


class _ApplicationLogArchive:
    def __init__(self, path: Path):
        self._path = path
        self._pending: deque[tuple[str, int]] = deque()
        self._queued_bytes = 0
        self._ready = Condition()
        self._closed = False
        self._thread = Thread(
            target=self._write, name="application-log-archive", daemon=True
        )
        self._thread.start()

    def enqueue(self, line: str) -> None:
        size = len(line.encode("utf-8", errors="backslashreplace")) + 1
        with self._ready:
            if (
                self._closed
                or size > _APPLICATION_LOG_FILE_BYTES
                or self._queued_bytes + size > _APPLICATION_LOG_QUEUE_BYTES
            ):
                return
            self._pending.append((line, size))
            self._queued_bytes += size
            self._ready.notify()

    def _write(self) -> None:
        output = _ApplicationLogFile(self._path)
        try:
            while True:
                with self._ready:
                    self._ready.wait_for(lambda: self._pending or self._closed)
                    if not self._pending:
                        return
                    line, size = self._pending.popleft()
                    self._queued_bytes -= size
                record = logging.makeLogRecord({"msg": line, "archive_bytes": size})
                output.handle(record)
        finally:
            output.close()

    def close(self) -> None:
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._ready.notify()
        self._thread.join(timeout=1)


class _ApplicationLogHandler(logging.StreamHandler):
    def __init__(self, role: str, path: Path):
        super().__init__()
        self.formatter = _ApplicationLogFormatter(role)
        self._archive = _ApplicationLogArchive(path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            self.stream.write(line + self.terminator)
            self.flush()
            self._archive.enqueue(line)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._archive.close()
        super().close()


def setup_application_logging(cfg: "Config", role: str) -> None:
    """Configure the application console and file copies for this process."""
    setup_logging(cfg.debug)
    if not cfg.log_otlp_endpoint:
        return
    root = logging.getLogger()
    if any(isinstance(handler, _ApplicationLogHandler) for handler in root.handlers):
        return
    path = Path(cfg.log_dir) / "application" / f"{role}.jsonl"
    for handler in root.handlers[:]:
        if handler.name == _TEXT_HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()
    root.addHandler(_ApplicationLogHandler(role, path))
    root.setLevel(logging.DEBUG if cfg.debug else logging.INFO)


def set_worker_log_identity(
    worker_id: int, cluster_id: Optional[int], name: str
) -> None:
    """Set Worker identity once its registration has been confirmed."""
    identity = {"worker_id": worker_id, "worker_name": name}
    if cluster_id is not None:
        identity["cluster_id"] = cluster_id
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, _ApplicationLogHandler)
            and handler.formatter.role == "worker"
        ):
            handler.formatter.identity = identity


def uvicorn_log_config() -> Optional[dict]:
    """Keep the application archive open when an API listener starts."""
    # Uvicorn's default dictConfig closes previously installed handlers.
    if any(
        isinstance(handler, _ApplicationLogHandler)
        for handler in logging.getLogger().handlers
    ):
        return None
    from uvicorn.config import LOGGING_CONFIG

    return LOGGING_CONFIG


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO

    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.name = _TEXT_HANDLER_NAME
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[handler],
        )
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    logging.Logger.trace = trace

    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.fromtimestamp(
            record.created, timezone.utc
        ).astimezone()
    )

    # Third-party loggers to disable
    disable_logger_names = [
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.proxy",
        "httpx",
        "httpcore2.connection",
        "httpcore2.http11",
        "httpcore2.proxy",
        "httpx2",
        "asyncio",
        "aiocache.base",
        "urllib3.connectionpool",
        "multipart.multipart",
        "apscheduler.scheduler",
        "apscheduler.executors.default",
        "tzlocal",
        "alembic.runtime.migration",
        "python_multipart.multipart",
        "filelock",
        "fastapi-cdn-host",
        "huggingface_hub.file_download",
        "docker.auth",
        "kubernetes_asyncio.client.rest",
        "azure.core.pipeline.policies.http_logging_policy",
        "websockets.client",
    ]

    for logger_name in disable_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True

    # Third-party loggers to print on debug
    debug_logger_names = [
        "alembic.runtime.migration",
    ]

    for logger_name in debug_logger_names:
        logger = logging.getLogger(logger_name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.disabled = True


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


class RedirectStdoutStderr:
    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.target
        sys.stderr = self.target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
