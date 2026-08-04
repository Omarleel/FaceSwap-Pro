from __future__ import annotations

import atexit
import contextvars
import cProfile
import json
import logging
import os
import platform
import pstats
import queue
import sys
import threading
import time
import tracemalloc
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

_LOGGER_NAME = "faceswap_pro"
_PROFILE_SENTINEL = object()
_CURRENT_SPAN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "faceswap_pro_current_span",
    default=None,
)
_SESSION_LOCK = threading.RLock()
_SESSION: ObservabilitySession | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class _JsonLogFormatter(logging.Formatter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "process_id": record.process,
            "message": record.getMessage(),
        }
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            payload.update(event_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=_json_default)


class ObservabilitySession:
    """Escribe diagnósticos JSONL y perfiles detallados de una ejecución."""

    def __init__(self, log_dir: Path, *, command: str) -> None:
        self.log_dir = log_dir
        self.logs_path = log_dir / "logs.jsonl"
        self.profile_path = log_dir / "profile.jsonl"
        self.command = command
        self.run_id = uuid.uuid4().hex
        self._closed = False
        self._close_lock = threading.RLock()
        self._aggregate_lock = threading.Lock()
        self._span_aggregates: dict[str, dict[str, int]] = {}
        self._profile_queue: queue.SimpleQueue[dict[str, Any] | object] = queue.SimpleQueue()
        self._profile_thread: threading.Thread | None = None
        self._main_profiler = cProfile.Profile()
        self._main_profiler_enabled = False
        self._logger = logging.getLogger(_LOGGER_NAME)
        self._warning_logger = logging.getLogger("py.warnings")
        self._handler: logging.Handler | None = None
        self._previous_warning_level = self._warning_logger.level
        self._previous_warning_propagate = self._warning_logger.propagate
        self._previous_sys_hook = sys.excepthook
        self._previous_thread_hook = getattr(threading, "excepthook", None)
        self._started_ns = time.perf_counter_ns()
        self._owns_tracemalloc = False

    def start(self) -> ObservabilitySession:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path.touch(exist_ok=True)
        self.profile_path.touch(exist_ok=True)

        handler = logging.FileHandler(self.logs_path, encoding="utf-8")
        handler.setFormatter(_JsonLogFormatter(self.run_id))
        handler.setLevel(logging.INFO)
        self._handler = handler
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)
        self._logger.propagate = False
        self._warning_logger.setLevel(logging.WARNING)
        self._warning_logger.addHandler(handler)
        self._warning_logger.propagate = False

        self._profile_thread = threading.Thread(
            target=self._profile_writer,
            name="faceswap-profile-writer",
            daemon=True,
        )
        self._profile_thread.start()

        if not tracemalloc.is_tracing():
            tracemalloc.start(10)
            self._owns_tracemalloc = True

        sys.excepthook = self._sys_exception_hook
        if hasattr(threading, "excepthook"):
            threading.excepthook = self._thread_exception_hook
        logging.captureWarnings(True)

        self.write_profile(
            "session_start",
            command=self.command,
            python=platform.python_version(),
            platform=platform.platform(),
            process_id=os.getpid(),
            logical_cpu_count=os.cpu_count(),
            cwd=str(Path.cwd()),
            logs_path=str(self.logs_path),
            profile_path=str(self.profile_path),
        )
        self._logger.info(
            "Inicio de ejecución",
            extra={
                "event_data": {
                    "event": "session_start",
                    "command": self.command,
                    "profile_path": str(self.profile_path),
                }
            },
        )
        try:
            self._main_profiler.enable()
            self._main_profiler_enabled = True
        except ValueError as exc:
            self.write_profile(
                "cprofile_unavailable",
                scope="main",
                reason=str(exc),
            )
            self.log_problem(
                "cProfile no pudo activarse; se mantienen los spans detallados",
                scope="main",
                reason=str(exc),
            )
        atexit.register(self.close)
        return self

    def _profile_writer(self) -> None:
        with self.profile_path.open("a", encoding="utf-8", buffering=1) as stream:
            while True:
                item = self._profile_queue.get()
                if item is _PROFILE_SENTINEL:
                    return
                stream.write(
                    json.dumps(item, ensure_ascii=False, default=_json_default) + "\n"
                )

    def write_profile(self, event: str, **fields: Any) -> None:
        if self._closed:
            return
        if event == "span":
            name = str(fields.get("name", "unknown"))
            duration_ns = int(fields.get("duration_ns", 0))
            cpu_ns = int(fields.get("thread_cpu_ns", 0))
            with self._aggregate_lock:
                aggregate = self._span_aggregates.setdefault(
                    name,
                    {
                        "count": 0,
                        "total_duration_ns": 0,
                        "min_duration_ns": duration_ns,
                        "max_duration_ns": duration_ns,
                        "total_thread_cpu_ns": 0,
                    },
                )
                aggregate["count"] += 1
                aggregate["total_duration_ns"] += duration_ns
                aggregate["min_duration_ns"] = min(
                    aggregate["min_duration_ns"], duration_ns
                )
                aggregate["max_duration_ns"] = max(
                    aggregate["max_duration_ns"], duration_ns
                )
                aggregate["total_thread_cpu_ns"] += cpu_ns
        payload = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "event": event,
            "thread": threading.current_thread().name,
            "process_id": os.getpid(),
            **fields,
        }
        self._profile_queue.put(payload)

    def write_cprofile(self, profiler: cProfile.Profile, *, scope: str) -> None:
        stats = pstats.Stats(profiler)
        rows: list[tuple[float, tuple[str, int, str], tuple[Any, ...]]] = []
        for function, values in stats.stats.items():
            primitive_calls, total_calls, total_time, cumulative_time, callers = values
            del callers
            if total_time <= 0.0 and cumulative_time <= 0.0:
                continue
            rows.append(
                (
                    cumulative_time,
                    function,
                    (
                        primitive_calls,
                        total_calls,
                        total_time,
                        cumulative_time,
                    ),
                )
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        for _, function, values in rows:
            filename, line_number, function_name = function
            primitive_calls, total_calls, total_time, cumulative_time = values
            self.write_profile(
                "cprofile_function",
                scope=scope,
                filename=filename,
                line_number=line_number,
                function=function_name,
                primitive_calls=primitive_calls,
                total_calls=total_calls,
                total_seconds=total_time,
                cumulative_seconds=cumulative_time,
            )

    def log_problem(self, message: str, *, level: int = logging.WARNING, **fields: Any) -> None:
        self._logger.log(
            level,
            message,
            extra={"event_data": {"event": "problem", **fields}},
        )

    def log_exception(self, message: str, exc: BaseException, **fields: Any) -> None:
        if getattr(exc, "_faceswap_pro_logged", False):
            return
        try:
            setattr(exc, "_faceswap_pro_logged", True)
        except Exception:
            pass
        self._logger.error(
            message,
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "event_data": {
                    "event": "exception",
                    "exception_type": type(exc).__name__,
                    **fields,
                }
            },
        )

    def _sys_exception_hook(
        self,
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        self._logger.critical(
            "Excepción no controlada",
            exc_info=(exc_type, exc, traceback),
            extra={"event_data": {"event": "unhandled_exception"}},
        )
        if self._previous_sys_hook is not None:
            self._previous_sys_hook(exc_type, exc, traceback)

    def _thread_exception_hook(self, args: threading.ExceptHookArgs) -> None:
        self._logger.critical(
            "Excepción no controlada en un hilo",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            extra={
                "event_data": {
                    "event": "unhandled_thread_exception",
                    "failed_thread": args.thread.name if args.thread else None,
                }
            },
        )
        if self._previous_thread_hook is not None:
            self._previous_thread_hook(args)

    def close(self) -> None:
        global _SESSION
        with self._close_lock:
            if self._closed:
                return
            try:
                if self._main_profiler_enabled:
                    self._main_profiler.disable()
                    self.write_cprofile(self._main_profiler, scope="main")
                with self._aggregate_lock:
                    span_aggregates = {
                        name: dict(values)
                        for name, values in self._span_aggregates.items()
                    }
                for name, aggregate in sorted(
                    span_aggregates.items(),
                    key=lambda item: item[1]["total_duration_ns"],
                    reverse=True,
                ):
                    count = aggregate["count"]
                    total_duration_ns = aggregate["total_duration_ns"]
                    self.write_profile(
                        "span_summary",
                        name=name,
                        count=count,
                        total_duration_ns=total_duration_ns,
                        average_duration_ns=total_duration_ns / max(count, 1),
                        min_duration_ns=aggregate["min_duration_ns"],
                        max_duration_ns=aggregate["max_duration_ns"],
                        total_thread_cpu_ns=aggregate["total_thread_cpu_ns"],
                    )
                elapsed_ns = time.perf_counter_ns() - self._started_ns
                memory_current, memory_peak = (
                    tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
                )
                self.write_profile(
                    "session_end",
                    elapsed_ns=elapsed_ns,
                    elapsed_seconds=elapsed_ns / 1_000_000_000,
                    memory_current_bytes=memory_current,
                    memory_peak_bytes=memory_peak,
                )
                self._logger.info(
                    "Fin de ejecución",
                    extra={
                        "event_data": {
                            "event": "session_end",
                            "elapsed_seconds": elapsed_ns / 1_000_000_000,
                        }
                    },
                )
            finally:
                self._closed = True
                self._profile_queue.put(_PROFILE_SENTINEL)
                if self._profile_thread is not None:
                    self._profile_thread.join()
                sys.excepthook = self._previous_sys_hook
                if hasattr(threading, "excepthook") and self._previous_thread_hook is not None:
                    threading.excepthook = self._previous_thread_hook
                logging.captureWarnings(False)
                if self._handler is not None:
                    self._logger.removeHandler(self._handler)
                    self._warning_logger.removeHandler(self._handler)
                    self._warning_logger.setLevel(self._previous_warning_level)
                    self._warning_logger.propagate = self._previous_warning_propagate
                    self._handler.close()
                if self._owns_tracemalloc and tracemalloc.is_tracing():
                    tracemalloc.stop()
                try:
                    atexit.unregister(self.close)
                except Exception:
                    pass
                with _SESSION_LOCK:
                    if _SESSION is self:
                        _SESSION = None


def configure_observability(
    log_dir: Path = Path("logs"),
    *,
    command: str = "cli",
) -> ObservabilitySession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None and not _SESSION._closed:
            return _SESSION
        _SESSION = ObservabilitySession(Path(log_dir), command=command).start()
        return _SESSION


def current_session() -> ObservabilitySession | None:
    with _SESSION_LOCK:
        if _SESSION is None or _SESSION._closed:
            return None
        return _SESSION


def profile_event(event: str, **fields: Any) -> None:
    session = current_session()
    if session is not None:
        session.write_profile(event, **fields)


def log_problem(message: str, **fields: Any) -> None:
    session = current_session()
    if session is not None:
        session.log_problem(message, **fields)


def log_exception(message: str, exc: BaseException, **fields: Any) -> None:
    session = current_session()
    if session is not None:
        session.log_exception(message, exc, **fields)


@contextmanager
def profile_span(name: str, **fields: Any) -> Iterator[None]:
    session = current_session()
    if session is None:
        yield
        return

    span_id = uuid.uuid4().hex[:16]
    parent_span_id = _CURRENT_SPAN.get()
    token = _CURRENT_SPAN.set(span_id)
    started_ns = time.perf_counter_ns()
    cpu_started_ns = time.thread_time_ns()
    memory_started = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
    status = "ok"
    error_type: str | None = None
    try:
        yield
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        session.log_exception(
            f"Falló la operación perfilada: {name}",
            exc,
            span=name,
            span_id=span_id,
            **fields,
        )
        raise
    finally:
        ended_ns = time.perf_counter_ns()
        cpu_ended_ns = time.thread_time_ns()
        memory_current, memory_peak = (
            tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        )
        session.write_profile(
            "span",
            name=name,
            span_id=span_id,
            parent_span_id=parent_span_id,
            status=status,
            error_type=error_type,
            started_ns=started_ns,
            duration_ns=ended_ns - started_ns,
            duration_seconds=(ended_ns - started_ns) / 1_000_000_000,
            thread_cpu_ns=cpu_ended_ns - cpu_started_ns,
            memory_delta_bytes=memory_current - memory_started,
            memory_current_bytes=memory_current,
            memory_peak_bytes=memory_peak,
            **fields,
        )
        _CURRENT_SPAN.reset(token)


@contextmanager
def function_profile(scope: str) -> Iterator[None]:
    session = current_session()
    if session is None:
        yield
        return

    profiler = cProfile.Profile()
    enabled = False
    try:
        profiler.enable()
        enabled = True
    except ValueError as exc:
        session.write_profile(
            "cprofile_unavailable",
            scope=scope,
            reason=str(exc),
        )
    try:
        yield
    finally:
        if enabled:
            profiler.disable()
            session.write_cprofile(profiler, scope=scope)


__all__ = [
    "ObservabilitySession",
    "configure_observability",
    "current_session",
    "function_profile",
    "log_exception",
    "log_problem",
    "profile_event",
    "profile_span",
]
