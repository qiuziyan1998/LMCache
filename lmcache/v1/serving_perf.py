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
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import Any
from zlib import crc32
import json
import logging
import os
import socket
import sys
import time

SERVING_PERF_ENV = "PD_SERVING_PERF"
PREFILL_START_TIMING_ENV = "LMCACHE_PREFILL_START_TIMING"
PREFILL_REUSE_DEBUG_RANK_ENV = "LMCACHE_PREFILL_REUSE_DEBUG_RANK"
PREFILL_REUSE_DEBUG_FILE_ENV = "LMCACHE_PREFILL_REUSE_DEBUG_FILE"
PREFILL_REUSE_DEBUG_MAX_LINES = 360
PREFILL_REUSE_DEBUG_MAX_WIDTH = 96
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_MODE = os.environ.get(SERVING_PERF_ENV, "0").strip().lower()
_PREFILL_START_TIMING = os.environ.get(PREFILL_START_TIMING_ENV, "0") == "1"
_REUSE_RANK_TEXT = os.environ.get(PREFILL_REUSE_DEBUG_RANK_ENV, "").strip()
_PREFILL_REUSE_RANK = int(_REUSE_RANK_TEXT) if _REUSE_RANK_TEXT.isdigit() else -1
_PREFILL_REUSE_FILE = os.environ.get(PREFILL_REUSE_DEBUG_FILE_ENV, "")
_PREFILL_REUSE_LINES = count()
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


def prefill_reuse_debug_enabled(rank: int) -> bool:
    """Return whether compact reuse diagnostics select this worker rank.

    The worker ID is supplied by LMCache metadata. The environment is read
    once at import; an unset or invalid rank disables these diagnostics.
    """
    return rank >= 0 and rank == _PREFILL_REUSE_RANK


@lru_cache(maxsize=1)
def _prefill_reuse_logger() -> logging.Logger:
    logger = logging.getLogger("lmcache.prefill_reuse_debug")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers[:] = [handler]
    if _PREFILL_REUSE_FILE:
        try:
            file_handler = logging.FileHandler(_PREFILL_REUSE_FILE, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)
        except OSError:
            logger.warning(
                "[PFR] r%d file_unavailable; use server.log", _PREFILL_REUSE_RANK
            )
    return logger


def prefill_reuse_debug_log(
    rank: int,
    stage: str,
    *,
    req_id: str = "",
    p: int = 0,
    g: int | None = None,
    **fields: int | float | str | bool,
) -> None:
    """Emit a bounded, scalar-only INFO record for the selected worker.

    Args:
        rank: LMCache worker ID, not a device query.
        stage: Short diagnostic stage name.
        req_id: Request ID, represented by a stable six-digit checksum.
        p: Prefix token count used to correlate stages within a chunk.
        g: Optional KV group number.
        fields: Short scalar counters or labels; never tensors or MemoryObjs.

    At most 360 data lines and one limit notice are emitted per process.
    Message bodies contain at most 96 characters, without timestamps or
    source locations. No caller object is stringified and no device is read.
    """
    if not prefill_reuse_debug_enabled(rank):
        return
    index = next(_PREFILL_REUSE_LINES)
    if index > PREFILL_REUSE_DEBUG_MAX_LINES:
        return
    logger = _prefill_reuse_logger()
    if index == PREFILL_REUSE_DEBUG_MAX_LINES:
        logger.info("[PFR] r%d limit=%d", rank, PREFILL_REUSE_DEBUG_MAX_LINES)
        return
    request_tag = f"{crc32(req_id.encode('utf-8')):08x}"[:6]
    parts = ["[PFR]", f"r{rank}", f"q{request_tag}", f"p{p}"]
    if g is not None:
        parts.append(f"g{g}")
    parts.append(stage)
    for key, value in fields.items():
        if type(value) not in (int, float, str, bool):
            encoded = "?"
        elif isinstance(value, float):
            encoded = f"{value:.1f}"
        elif isinstance(value, bool):
            encoded = str(int(value))
        else:
            encoded = str(value)
        parts.append(f"{key}={encoded}")
    line = " ".join(parts).replace("\n", "_").replace("\r", "_")
    if len(line) > PREFILL_REUSE_DEBUG_MAX_WIDTH:
        line = line[: PREFILL_REUSE_DEBUG_MAX_WIDTH - 3] + "..."
    logger.info("%s", line)


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
