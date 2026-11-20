# SPDX-License-Identifier: Apache-2.0
"""Opt-in host timing for decoder storage operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
import json
import os
import socket
import time

_MODE = os.getenv("PD_SERVING_PERF", "0").strip().lower()
_FALSE_VALUES = {"", "0", "false", "no", "off"}
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


def _non_json_field(_value: Any) -> str:
    # Do not stringify arbitrary objects: a Tensor repr may read device memory.
    return "<non-JSON value>"


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
    **fields: Any,
) -> None:
    if not serving_perf_enabled():
        return
    if started is not None:
        fields["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
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
