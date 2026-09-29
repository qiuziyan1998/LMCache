# SPDX-License-Identifier: Apache-2.0
"""Flushed phase markers for initialization only, never request execution."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import os
import socket
import sys
import time


def _emit_phase(
    stage: str,
    state: str,
    details: dict[str, object],
    elapsed_ms: float | None = None,
    error: BaseException | None = None,
) -> None:
    """Best-effort diagnostics must never change initialization behavior."""
    try:
        host = socket.gethostname()[:24]
        fields = [f"[LMCACHE_INIT] host={host}", f"pid={os.getpid()}"]
        if "rank" in details:
            fields.append(f"rank={details['rank']}")
        fields.extend((f"stage={stage}", f"state={state}"))
        if elapsed_ms is not None:
            fields.append(f"ms={elapsed_ms:.1f}")
        if error is not None:
            # Native errors may embed a remote URL or credentials.
            fields.append(f"error={type(error).__name__}")
        for key, value in details.items():
            if key != "rank":
                text = " ".join(str(value).split())
                fields.append(f"{key}={text[:32]}")
        print(" ".join(fields), file=sys.stderr, flush=True)
    except Exception:
        # A closed stderr pipe or diagnostic formatting error must not mask
        # a backend exception, or stop an otherwise successful startup.
        pass


@contextmanager
def startup_phase(stage: str, **details: object) -> Iterator[None]:
    """Log before a startup call, then its duration or unchanged exception.

    Hostname and PID correlate native allocation markers with the outer
    rank-specific phase. Details should contain only safe scalar metadata,
    never endpoints or configuration dumps. Output failures are ignored;
    exceptions from the wrapped call propagate unchanged. No watchdog,
    collective, device synchronization or retry is added.
    """
    _emit_phase(stage, "begin", details)
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _emit_phase(stage, "error", details, elapsed_ms, exc)
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _emit_phase(stage, "end", details, elapsed_ms)
