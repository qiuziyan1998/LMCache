# SPDX-License-Identifier: Apache-2.0
"""Opt-in PD serving timing, configured once before worker startup.

Use PD_SERVING_PERF=1 for production anomalies, detail for full host tracing, or
device for explicit device timing in supported components. Content diagnostics
and operational failures have separate controls. Log schemas remain stable.
"""

# Standard
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from typing import Any
import json
import os
import socket
import time

SERVING_PERF_ENV = "PD_SERVING_PERF"
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_MODE = os.environ.get(SERVING_PERF_ENV, "0").strip().lower()
# Bound recent request/transfer IDs even when every read is slow.
_REPORT_TIMES: OrderedDict[str, float] = OrderedDict()
_REPORT_LOCK = Lock()
SLOW_OPERATION_MS = 100.0


def serving_perf_report_due(key: str) -> bool:
    """Allow the first and one subsequent summary per minute for this event or transfer."""
    now = time.monotonic()
    with _REPORT_LOCK:
        if now - _REPORT_TIMES.get(key, -float("inf")) < 60:
            return False
        _REPORT_TIMES[key] = now
        _REPORT_TIMES.move_to_end(key)
        if len(_REPORT_TIMES) > 256:
            _REPORT_TIMES.popitem(last=False)
        return True


_PERF_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "lmcache_serving_perf_context", default=None
)
_PERF_TIMINGS: ContextVar[dict[str, Any] | None] = ContextVar(
    "lmcache_serving_perf_timings", default=None
)


def serving_perf_context() -> dict[str, Any]:
    """Return a snapshot for explicit diagnostic correlation across IPC."""
    return dict(_PERF_CONTEXT.get() or {}) if serving_perf_enabled() else {}


@contextmanager
def serving_perf_measure(*, rollup_prefix: str = "") -> Iterator[dict[str, Any]]:
    """Collect host timings for one operation, isolating nested operations."""
    values: dict[str, Any] = {}
    token = _PERF_TIMINGS.set(values)
    try:
        yield values
    finally:
        _PERF_TIMINGS.reset(token)
        if rollup_prefix:
            for name, value in values.items():
                serving_perf_record(rollup_prefix + name, value)


def serving_perf_record(name: str, value: float) -> None:
    """Accumulate a host measurement only in an active operation summary."""
    values = _PERF_TIMINGS.get()
    if values is not None:
        values[name] = round(values.get(name, 0) + value, 3)


@contextmanager
def serving_perf_phase(name: str) -> Iterator[None]:
    """Measure a host phase without device events or synchronization."""
    if _PERF_TIMINGS.get() is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        serving_perf_record(name, (time.perf_counter() - started) * 1000)


def _clock_domain() -> tuple[str, str]:
    host = socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = str(round(time.time() - time.monotonic()))
    return host, f"{host}:{boot}"


_HOST, _CLOCK_DOMAIN = _clock_domain()


def serving_perf_enabled() -> bool:
    return _MODE not in _FALSE_VALUES


def serving_perf_detailed_enabled() -> bool:
    """Enable per-layer CPU diagnostics only for explicit detail/device modes.

    Ordinary values such as ``1`` retain coarse host logs. This function only
    reads configuration; it does not inspect tensors or call a device runtime.
    """
    return _MODE in ("detail", "device")


def _non_json_field(_value: Any) -> str:
    # Do not stringify arbitrary objects: a Tensor repr may read device memory.
    return "<non-JSON value>"


def serving_perf_now() -> float:
    return time.perf_counter()


@contextmanager
def serving_perf_scope(**fields: Any) -> Iterator[None]:
    """Attach correlation fields to nested cold-perf events in this task."""
    if not serving_perf_enabled():
        yield
        return
    current = _PERF_CONTEXT.get() or {}
    token = _PERF_CONTEXT.set({**current, **fields})
    try:
        yield
    finally:
        _PERF_CONTEXT.reset(token)


def serving_perf_log(
    logger,
    event: str,
    *,
    started: float | None = None,
    production: bool = False,
    **fields: Any,
) -> None:
    if not serving_perf_enabled():
        return
    detailed = serving_perf_detailed_enabled()
    transfer = event in (
        "mooncake_page_lookup",
        "mooncake_page_get",
        "mooncake_page_put",
        "mooncake_legacy_get",
        "direct_npu_page_get",
        "direct_npu_page_put",
        "remote_resolver",
    )
    if not detailed and not production and not transfer:
        anomaly = event.endswith(
            ("_slow", "_failure", "_abort", "_fallback")
        ) or event in (
            "sparse_h2d_python_stall",
            "remote_fill_fatal_restart",
            "remote_fill_direct_abandoned",
            "remote_fill_local_prefix_missing_at_load",
            "live_source_missing_producer_event",
        )
        if not anomaly or not serving_perf_report_due(event):
            return
    if started is not None:
        fields["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if not detailed and not production and transfer:
        duration = fields.get("elapsed_ms")
        if fields.get("status") not in (
            "error",
            "failed",
            "timeout",
            "allocation_failed",
            "no_buffers",
        ) and (
            type(duration) not in (float, int)
            or duration < SLOW_OPERATION_MS
            or not serving_perf_report_due(event)
        ):
            return
    payload = {
        "schema": 1,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        "wall_time_ns": time.time_ns(),
        "host": _HOST,
        "clock_domain": _CLOCK_DOMAIN,
        **(_PERF_CONTEXT.get() or {}),
        **fields,
    }
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(payload, default=_non_json_field, separators=(",", ":")),
    )
