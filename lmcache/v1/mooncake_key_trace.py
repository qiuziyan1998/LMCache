# SPDX-License-Identifier: Apache-2.0
"""Mooncake zero-lookup retry and opt-in exact-key tracing."""

# Standard
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO
import json
import os
import socket
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_TRACE_DIR_ENV = "LMCACHE_MOONCAKE_KEY_TRACE_DIR"


@dataclass(frozen=True, slots=True)
class _MooncakeTraceContext:
    request_id: str | None
    phase: str
    lookup_attempt: int | None = None
    retry_delay_ms: int | None = None
    kv_group: int | None = None
    layer_id: int | None = None


_TRACE_CONTEXT: ContextVar[_MooncakeTraceContext | None] = ContextVar(
    "lmcache_mooncake_trace_context",
    default=None,
)


@contextmanager
def mooncake_trace_context(
    request_id: str | None,
    phase: str,
    *,
    lookup_attempt: int | None = None,
    retry_delay_ms: int | None = None,
    kv_group: int | None = None,
    layer_id: int | None = None,
) -> Iterator[None]:
    """Attach request context to native Mooncake trace records.

    Args:
        request_id: LMCache or vLLM request identifier.
        phase: Request phase, such as ``store`` or ``lookup``.
        lookup_attempt: Zero-based request-level lookup attempt.
        retry_delay_ms: Delay applied immediately before this attempt.
        kv_group: Optional MLA/DSA KV group.
        layer_id: Optional layer being stored.

    Yields:
        Control while the supplied context is active.
    """
    token = _TRACE_CONTEXT.set(
        _MooncakeTraceContext(
            request_id=request_id,
            phase=phase,
            lookup_attempt=lookup_attempt,
            retry_delay_ms=retry_delay_ms,
            kv_group=kv_group,
            layer_id=layer_id,
        )
    )
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def run_mooncake_zero_lookup_retries(
    config: Any,
    request_id: str | None,
    lookup_once: Callable[[], int],
) -> int:
    """Retry a complete Mooncake lookup only when it returns zero tokens.

    Args:
        config: LMCache configuration carrying retry delays in milliseconds.
        request_id: Request identifier included in logs and trace records.
        lookup_once: Complete request-level lookup operation.

    Returns:
        The first positive hit count, or the final zero result.

    Raises:
        ValueError: If a configured retry delay is negative.
    """
    configured = getattr(
        config,
        "mooncake_lookup_retry_delays_ms",
        [],
    )
    delays = [] if configured is None else [int(delay) for delay in configured]
    if any(delay < 0 for delay in delays):
        raise ValueError("mooncake_lookup_retry_delays_ms must be non-negative")

    result = 0
    for attempt, delay_ms in enumerate([0, *delays]):
        if delay_ms:
            time.sleep(delay_ms / 1000)
        with mooncake_trace_context(
            request_id,
            "lookup",
            lookup_attempt=attempt,
            retry_delay_ms=delay_ms,
        ):
            result = lookup_once()
        if result > 0 or attempt == len(delays):
            return result
        logger.warning(
            "Mooncake lookup returned zero tokens; retrying: "
            "request_id=%s attempt=%d next_attempt=%d delay_ms=%d",
            request_id,
            attempt,
            attempt + 1,
            delays[attempt],
        )
    return result


class MooncakeKeyTracingStore:
    """Proxy a Mooncake store and record exact store and lookup key events.

    Records are written only when ``LMCACHE_MOONCAKE_KEY_TRACE_DIR`` is set.
    Each compact JSONL record represents one native call and includes its
    exact keys, raw statuses, outcomes, timestamps, host, and process.
    """

    def __init__(
        self,
        store: Any,
        trace_dir: Path,
        component: str,
        metadata: Any = None,
    ) -> None:
        """Initialize a tracing proxy.

        Args:
            store: Initialized native Mooncake store.
            trace_dir: Directory in which trace files are written.
            component: Short name identifying the LMCache caller.
            metadata: Optional LMCache metadata used for process context.
        """
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._store = store
        self._trace_dir = trace_dir
        self._component = component
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._role = getattr(metadata, "role", None)
        self._worker_id = getattr(metadata, "worker_id", None)
        self._lock = threading.Lock()
        self._sequence = 0
        self._files: dict[str, TextIO] = {}
        self._failed = False
        logger.warning(
            "Mooncake key tracing enabled: component=%s directory=%s",
            component,
            trace_dir,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def batch_is_exist(self, keys: Sequence[str]) -> Any:
        """Trace and run a native batch existence query.

        Args:
            keys: Exact serialized keys passed to Mooncake.

        Returns:
            The native Mooncake result without modification.
        """
        key_list = list(keys)
        return self._call(
            "lookup",
            "batch_is_exist",
            key_list,
            self._store.batch_is_exist,
            key_list,
        )

    def is_exist(self, key: str) -> Any:
        """Trace and run a native single-key existence query.

        Args:
            key: Exact serialized key passed to Mooncake.

        Returns:
            The native Mooncake result without modification.
        """
        return self._call(
            "lookup",
            "is_exist",
            [key],
            self._store.is_exist,
            key,
        )

    def batch_put_from(self, keys: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Trace and run a native batched zero-copy store.

        Args:
            keys: Exact serialized keys passed to Mooncake.
            *args: Remaining native positional arguments.
            **kwargs: Remaining native keyword arguments.

        Returns:
            The native Mooncake result without modification.
        """
        key_list = list(keys)
        return self._call(
            "store",
            "batch_put_from",
            key_list,
            self._store.batch_put_from,
            key_list,
            *args,
            **kwargs,
        )

    def batch_put_from_multi_buffers(
        self,
        keys: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Trace and run a native page-oriented batched store.

        Args:
            keys: Exact serialized page keys passed to Mooncake.
            *args: Remaining native positional arguments.
            **kwargs: Remaining native keyword arguments.

        Returns:
            The native Mooncake result without modification.
        """
        key_list = list(keys)
        return self._call(
            "store",
            "batch_put_from_multi_buffers",
            key_list,
            self._store.batch_put_from_multi_buffers,
            key_list,
            *args,
            **kwargs,
        )

    def put_from(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Trace and run a native single-key zero-copy store.

        Args:
            key: Exact serialized key passed to Mooncake.
            *args: Remaining native positional arguments.
            **kwargs: Remaining native keyword arguments.

        Returns:
            The native Mooncake result without modification.
        """
        return self._call(
            "store",
            "put_from",
            [key],
            self._store.put_from,
            key,
            *args,
            **kwargs,
        )

    def put_parts(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """Trace and run a native single-key metadata store.

        Args:
            key: Exact serialized key passed to Mooncake.
            *args: Remaining native positional arguments.
            **kwargs: Remaining native keyword arguments.

        Returns:
            The native Mooncake result without modification.
        """
        return self._call(
            "store",
            "put_parts",
            [key],
            self._store.put_parts,
            key,
            *args,
            **kwargs,
        )

    def close(self) -> Any:
        """Close the native store and trace files.

        Returns:
            The native Mooncake close result.
        """
        try:
            return self._store.close()
        finally:
            with self._lock:
                for trace_file in self._files.values():
                    trace_file.close()
                self._files.clear()

    def _call(
        self,
        kind: str,
        operation: str,
        keys: list[str],
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        started_at_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            self._record(
                kind,
                operation,
                keys,
                None,
                started_at_ns,
                started_monotonic_ns,
                f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record(
            kind,
            operation,
            keys,
            result,
            started_at_ns,
            started_monotonic_ns,
            None,
        )
        return result

    def _record(
        self,
        kind: str,
        operation: str,
        keys: list[str],
        result: Any,
        started_at_ns: int,
        started_monotonic_ns: int,
        error: str | None,
    ) -> None:
        if self._failed:
            return
        completed_at_ns = time.time_ns()
        duration_ns = time.monotonic_ns() - started_monotonic_ns
        statuses, response_count = self._statuses(result, len(keys), error)
        context = _TRACE_CONTEXT.get()
        try:
            with self._lock:
                sequence = self._sequence
                self._sequence += 1
                serialized_statuses = [self._json_value(status) for status in statuses]
                outcomes = [
                    self._outcome(
                        kind,
                        statuses[index] if index < len(statuses) else None,
                        index < len(statuses),
                        error,
                    )
                    for index in range(len(keys))
                ]
                record = {
                    "completed_at": datetime.fromtimestamp(
                        completed_at_ns / 1e9,
                        tz=timezone.utc,
                    ).isoformat(timespec="microseconds"),
                    "started_at_ns": started_at_ns,
                    "completed_at_ns": completed_at_ns,
                    "duration_ns": duration_ns,
                    "hostname": self._hostname,
                    "pid": self._pid,
                    "component": self._component,
                    "role": self._role,
                    "worker_id": self._worker_id,
                    "request_id": context.request_id if context else None,
                    "phase": context.phase if context else None,
                    "lookup_attempt": (context.lookup_attempt if context else None),
                    "retry_delay_ms": (context.retry_delay_ms if context else None),
                    "kv_group": context.kv_group if context else None,
                    "layer_id": context.layer_id if context else None,
                    "sequence": sequence,
                    "operation": operation,
                    "key_count": len(keys),
                    "response_count": response_count,
                    "keys": keys,
                    "statuses": serialized_statuses,
                    "outcomes": outcomes,
                    "error": error,
                }
                trace_file = self._files.get(kind)
                if trace_file is None:
                    path = self._trace_dir / (
                        f"mooncake-{kind}-{self._hostname}-"
                        f"pid{self._pid}-{self._component}.jsonl"
                    )
                    trace_file = path.open("a", encoding="utf-8")
                    self._files[kind] = trace_file
                trace_file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                trace_file.flush()
        except Exception as exc:
            self._failed = True
            logger.error(
                "Disabling Mooncake key tracing after write failure: %s",
                exc,
            )

    @staticmethod
    def _statuses(
        result: Any,
        key_count: int,
        error: str | None,
    ) -> tuple[list[Any], int | None]:
        if error is not None:
            return [], 0
        if result is None:
            return [None] * key_count, None
        if isinstance(result, Sequence) and not isinstance(
            result,
            (str, bytes, bytearray),
        ):
            statuses = list(result)
            return statuses, len(statuses)
        return [result], 1

    @staticmethod
    def _outcome(
        kind: str,
        status: Any,
        status_present: bool,
        error: str | None,
    ) -> str:
        if error is not None:
            return "error"
        if not status_present:
            return "status_missing"
        if kind == "store":
            return "stored" if status is None or status == 0 else "error"
        if status == 1:
            return "found"
        if status == 0:
            return "missing"
        return "error" if status == -1 else "unexpected_status"

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)


def maybe_trace_mooncake_store(
    store: Any,
    component: str,
    metadata: Any = None,
) -> Any:
    """Wrap a Mooncake store when key tracing is explicitly enabled.

    Args:
        store: Initialized native Mooncake store.
        component: Short name identifying the LMCache caller.
        metadata: Optional LMCache metadata included in trace records.

    Returns:
        The original store when tracing is disabled or cannot initialize;
        otherwise a tracing proxy.
    """
    trace_dir = os.getenv(_TRACE_DIR_ENV, "").strip()
    if not trace_dir:
        return store
    try:
        return MooncakeKeyTracingStore(
            store,
            Path(trace_dir),
            component,
            metadata,
        )
    except Exception as exc:
        logger.error(
            "Mooncake key tracing requested but could not initialize: %s",
            exc,
        )
        return store
