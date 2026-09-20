# SPDX-License-Identifier: Apache-2.0
"""Opt-in PD serving timing, configured once before worker startup.

Use PD_SERVING_PERF=1 for host timing, detail for additional host detail, or
device for explicit device timing in supported components. Content diagnostics
and operational failures have separate controls. Log schemas remain stable.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
import json
import os
import socket
import time

SERVING_PERF_ENV = "PD_SERVING_PERF"
PREFILL_START_TIMING_ENV = "LMCACHE_PREFILL_START_TIMING"
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_MODE = os.environ.get(SERVING_PERF_ENV, "0").strip().lower()
_PREFILL_START_TIMING = os.environ.get(PREFILL_START_TIMING_ENV, "0") == "1"
_PERF_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "lmcache_serving_perf_context", default=None
)


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


def prefill_start_timing_enabled() -> bool:
    """Whether opt-in, per-chunk P-node setup timing is enabled."""
    return _PREFILL_START_TIMING


def prefill_start_timing_log(
    logger: Any, stage: str, started: float, **fields: Any
) -> None:
    """Log one host-stage duration without inspecting or syncing NPU tensors."""
    if not _PREFILL_START_TIMING:
        return
    payload = {
        "stage": stage,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "pid": os.getpid(),
        **fields,
    }
    logger.info("[PREFILL_START] %s", json.dumps(payload, separators=(",", ":")))


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
