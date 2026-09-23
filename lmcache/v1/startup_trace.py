# SPDX-License-Identifier: Apache-2.0
"""Flushed phase markers for initialization only, never request execution."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
import os
import sys
import time


@contextmanager
def startup_phase(stage: str, **details: object) -> Iterator[None]:
    """Log before a startup call, then its duration or unchanged exception.

    PID correlates native allocation markers with the outer rank-specific
    phase. No watchdog, collective, device synchronization or retry is added.
    """
    fields = " ".join(f"{key}={value!r}" for key, value in details.items())
    prefix = f"[LMCACHE_INIT] pid={os.getpid()} stage={stage}"
    print(f"{prefix} state=begin {fields}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"{prefix} state=error elapsed_ms={elapsed_ms:.3f} {fields} "
            f"error={type(exc).__name__}:{str(exc)!r}",
            file=sys.stderr,
            flush=True,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"{prefix} state=end elapsed_ms={elapsed_ms:.3f} {fields}",
            file=sys.stderr,
            flush=True,
        )
