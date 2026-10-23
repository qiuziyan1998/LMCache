# SPDX-License-Identifier: Apache-2.0
"""Opt-in, scalar-only P-prefill store diagnostics.

Set LMCACHE_PREFILL_STORE_DEBUG=1 before worker startup. Each request/group
prints the first occurrence of each stage; the terminal record counts all
occurrences, including suppressed repeats. Errors are never suppressed.
No payload, device state, or allocator free-list is inspected.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
from typing import Any
import logging
import os
import sys
import threading

_ENABLED = os.environ.get("LMCACHE_PREFILL_STORE_DEBUG", "0") == "1"
_MAX_REQUESTS = 1024
_LOCK = threading.Lock()
_IDS = count(1)
_LOGGER = logging.getLogger("lmcache.prefill_store_debug")


@dataclass
class _Request:
    tag: int
    stages: dict[tuple[int, str], int] = field(default_factory=dict)


_REQUESTS: OrderedDict[str, _Request] = OrderedDict()


def _emit(tag: int, message: str) -> None:
    if not _LOGGER.handlers:
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.info("[PFS] p%d q%d %s", os.getpid(), tag, message)


def prefill_store_debug_enabled() -> bool:
    """Return the diagnostic switch read once before worker startup."""
    return _ENABLED


def prefill_store_allocator_snapshot(allocator: Any) -> dict[str, int]:
    """Return constant-time CPU pool capacity in MiB, or -1 if unavailable.

    Only call on the allocation-failure path. This uses the allocator's public
    capacity API, without scanning objects or querying the device. A snapshot
    cannot distinguish fragmentation from non-evictable ownership.
    """
    try:
        free, total = allocator.get_allocator_capacity_bytes()
        return {"free_mb": free // 2**20, "pool_mb": total // 2**20}
    except Exception:
        # Diagnostics must not change unsupported allocators' fallback behavior.
        return {"free_mb": -1, "pool_mb": -1}


def prefill_store_debug_log(
    stage: str,
    req_id: str,
    kv_group: int = -1,
    *,
    error: bool = False,
    **fields: int | str | bool,
) -> None:
    """Log the first stage occurrence and count repeats for a terminal summary.

    Args:
        stage: Short store phase name; one counter per request and KV group.
        req_id: Full producer ID, emitted once with a process-local short tag.
        kv_group: Group ID, or -1 for a request-wide phase.
        error: Emit every error occurrence, even after earlier successes.
        fields: Scalar metadata only; tensors and events are never formatted.

    State is limited to 1024 requests per process. Eviction only removes debug
    counters; a later event re-emits the full ID. No serving object is retained.
    """
    if not _ENABLED:
        return
    with _LOCK:
        state = _REQUESTS.get(req_id)
        if state is None:
            state = _Request(next(_IDS))
            _REQUESTS[req_id] = state
            _emit(state.tag, "req=" + req_id.replace("\n", "_").replace("\r", "_"))
            if len(_REQUESTS) > _MAX_REQUESTS:
                _REQUESTS.popitem(last=False)
        _REQUESTS.move_to_end(req_id)
        key = (kv_group, stage)
        state.stages[key] = state.stages.get(key, 0) + 1
        if state.stages[key] > 1 and not error:
            return
        parts = [f"g{kv_group}", stage]
        if error:
            parts.append("ERROR")
        for name, value in fields.items():
            # Reject arbitrary repr/str calls, especially Tensor and Event.
            if type(value) in (int, str, bool):
                parts.append(f"{name}={int(value) if type(value) is bool else value}")
        _emit(state.tag, " ".join(parts).replace("\n", "_").replace("\r", "_"))


def prefill_store_debug_finish(req_id: str) -> None:
    """Print stage call counts after existing persistence barriers succeed.

    Counts represent store submissions/callbacks, not layers or token chunks.
    This function does not wait for futures or device events itself.
    """
    if not _ENABLED:
        return
    with _LOCK:
        state = _REQUESTS.pop(req_id, None)
        if state is None:
            return
        for (group, stage), calls in sorted(state.stages.items()):
            _emit(state.tag, f"g{group} total {stage}={calls}")
        _emit(state.tag, "persist_done")
