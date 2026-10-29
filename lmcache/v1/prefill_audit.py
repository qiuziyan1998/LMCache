# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU-only bank protocol and sampled transport diagnostics.

No device calls, payload tensor conversion, worker thread, queue, or serving
assertion is used here. Transport samples must only be taken from owned CPU
buffers after the caller's existing producer/receive fence.

Set LMCACHE_PREFILL_AUDIT=1 and LMCACHE_PREFILL_STORE_DEBUG=0 before startup.
LMCACHE_PREFILL_AUDIT_DIR defaults to ./prefill-audit. Only the file path and
first errors appear as [PFA] lines; full scalar evidence stays in JSONL files.
"""

# Standard
from collections import Counter, deque
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Sequence
import ctypes
import json
import logging
import os
import socket
import time
import zlib

_ENABLED = os.environ.get("LMCACHE_PREFILL_AUDIT", "0") == "1"
_PATH = Path(os.environ.get("LMCACHE_PREFILL_AUDIT_DIR", "prefill-audit"))
_LOCK = Lock()
_LOGGER = logging.getLogger("lmcache.prefill_audit")
_HOST = socket.gethostname()
_OPENED = False
_ERRORS = 0


def audit_enabled() -> bool:
    """Return the startup-only lightweight audit flag."""
    return _ENABLED


def cpu_backed(objects: Sequence[Any]) -> bool:
    """Check storage device metadata only; never dereference an NPU address."""
    try:
        for obj in objects:
            if not obj.is_valid():
                return False
            storage = getattr(getattr(obj, "parent_allocator", None), "buffer", None)
            if storage is None:
                storage = obj.raw_data
            if storage.device.type != "cpu":
                return False
        return bool(objects)
    except Exception:
        return False


def audit_record(kind: str, **fields: Any) -> None:
    """Append scalar audit evidence without propagating diagnostic failures.

    Files are process-private and written only at transport or request end,
    never per layer on the model's submission path. No fsync is performed.
    """
    global _OPENED, _ERRORS
    if not _ENABLED:
        return
    try:
        record = dict(
            v=1, kind=kind, host=_HOST, pid=os.getpid(), ns=time.time_ns(), **fields
        )
        encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        with _LOCK:
            _PATH.mkdir(parents=True, exist_ok=True)
            path = _PATH / f"audit-{os.getpid()}.jsonl"
            with path.open("ab") as output:
                output.write(encoded)
            if not _OPENED:
                _OPENED = True
                _LOGGER.warning("[PFA] file=%s", path.absolute())
            if fields.get("error") and _ERRORS < 8:
                _ERRORS += 1
                _LOGGER.warning("[PFA] %s", encoded.decode().strip())
    except Exception as error:
        # An audit I/O problem must not change model or cache execution.
        if _ERRORS < 8:
            _ERRORS += 1
            _LOGGER.warning("[PFA] diagnostic_io_error=%s", type(error).__name__)


def cpu_payload_samples(
    pointers: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
) -> list[tuple[int, str]]:
    """Sample up to 64 CPU bytes per logical object in canonical byte order.

    Args:
        pointers: Owned, accessible CPU addresses, one row per logical key.
        sizes: Matching buffer lengths. NPU addresses must never be passed.
    Returns:
        Logical byte length and CRC32 of eight evenly spaced 8-byte windows.
    Raises:
        ValueError: Descriptor shapes or CPU addresses are invalid.
    """
    result = []
    for ptrs, lengths in zip(pointers, sizes, strict=True):
        checksum = 0
        for pointer, size in sample_windows(ptrs, lengths):
            checksum = zlib.crc32(ctypes.string_at(pointer, size), checksum)
        result.append((sum(lengths), f"{checksum:08x}"))
    return result


def sample_windows(
    pointers: Sequence[int], sizes: Sequence[int]
) -> list[tuple[int, int]]:
    """Return ordered sample addresses without reading host or device memory."""
    if len(pointers) != len(sizes) or any(p <= 0 for p in pointers):
        raise ValueError("invalid sample pointers")
    if any(n < 0 for n in sizes) or not sizes or sum(sizes) <= 0:
        raise ValueError("invalid sample lengths")
    total, result = sum(sizes), []
    width = min(8, total)
    for offset in sorted({(total - width) * index // 7 for index in range(8)}):
        remaining, skip = width, offset
        for pointer, size in zip(pointers, sizes, strict=True):
            if skip >= size:
                skip -= size
                continue
            take = min(remaining, size - skip)
            result.append((pointer + skip, take))
            remaining -= take
            skip = 0
            if remaining == 0:
                break
    return result


def sampled_key_indices(keys: Sequence[str], limit: int | None = None) -> list[int]:
    """Select the same deterministic keys for CPU samples and bounded readback."""
    selected = [i for i, key in enumerate(keys) if zlib.crc32(key.encode()) % 8 == 0]
    if not selected and keys:
        selected = [0]
    if limit is not None and limit < 1:
        raise ValueError("sample key limit must be positive")
    if limit == 1:
        return selected[:1]
    if limit is not None and len(selected) > limit:
        selected = [
            selected[(len(selected) - 1) * i // (limit - 1)] for i in range(limit)
        ]
    return selected


def audit_payload(
    stage: str,
    keys: Sequence[str],
    pointers: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
    *,
    req_id: str | None = None,
    before: list[tuple[int, str]] | None = None,
    emit: bool = True,
) -> list[tuple[int, str]] | None:
    """Record CPU transport samples; report source mutation during a put."""
    if not _ENABLED:
        return None
    try:
        selected = sampled_key_indices(keys)
        samples = cpu_payload_samples(
            [pointers[i] for i in selected], [sizes[i] for i in selected]
        )
        if len(samples) != len(selected):
            raise ValueError("sample/key count mismatch")
        changed = (
            []
            if before is None
            else [
                i
                for i, (a, b) in enumerate(zip(before, samples, strict=True))
                if a != b
            ]
        )
        if emit:
            audit_record(
                "payload",
                stage=stage,
                req=req_id,
                keys=[keys[i] for i in selected],
                available=len(keys),
                samples=samples,
                before=before,
                changed=changed,
                error="source_changed_during_put" if changed else None,
            )
        return samples
    except Exception as error:
        audit_record(
            "coverage",
            stage=stage,
            req=req_id,
            error="sample_unavailable",
            detail=type(error).__name__,
        )
        return None


def audit_cpu_put(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    keys: Sequence[str],
    pointers: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
) -> Any:
    """Sample an existing CPU transfer call without adding a fence or thread."""
    before = audit_payload("put_source", keys, pointers, sizes, emit=False)
    result = function(*args)
    audit_payload("put", keys, pointers, sizes, before=before)
    return result


def audit_sources(
    req_id: str,
    keys: Sequence[Sequence[Any]],
    sources: Sequence[Sequence[Any]],
) -> None:
    """Sample completed CPU source pages at the existing sharing boundary."""
    if not _ENABLED or not keys or not sources:
        return
    try:
        # First Party
        from lmcache.v1.mooncake_layout import mooncake_page_key

        layers = len(keys)
        labels = [mooncake_page_key(key, layers) for key in keys[0]]
        selected = sampled_key_indices(labels)
        pointers, sizes = [], []
        for index in selected:
            first = sources[0][index]
            if hasattr(first, "layer_data_ptr"):
                if not cpu_backed([first]):
                    raise ValueError("source storage is not CPU")
                pointers.append([first.layer_data_ptr(0)])
                sizes.append([first.get_size()])
            else:
                objects = [row[index] for row in sources]
                if not cpu_backed(objects):
                    raise ValueError("source storage is not CPU")
                pointers.append([obj.data_ptr for obj in objects])
                sizes.append([obj.get_size() for obj in objects])
        audit_payload(
            "cache_read", [labels[i] for i in selected], pointers, sizes, req_id=req_id
        )
    except Exception as error:
        audit_record(
            "coverage",
            stage="cache_read",
            req=req_id,
            error="sample_unavailable",
            detail=type(error).__name__,
        )


def _diagnostic_only(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            audit_record(
                "coverage",
                stage=function.__name__,
                error="audit_failure",
                detail=type(error).__name__,
            )
            return None

    return wrapped


class BankAudit:
    """Observe bank handoff invariants using integer event identities only.

    The tracker neither queries nor retains device events or payload objects.
    Normal layer hooks update bounded in-memory scalar state. Only errors and
    request-end summaries write files. Errors do not abort serving.
    """

    def __init__(self, rank: int, role: str) -> None:
        self.rank, self.role = rank, role
        self.step = 0
        self.requests: tuple[str, ...] = ()
        self.loads: dict[tuple[int, int, int, int], int] = {}
        self.load_generations: dict[tuple[int, int, int], int] = {}
        self.consumed: set[tuple[int, int, int, int]] = set()
        self.uses: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.saves: dict[tuple[int, int], tuple[int, int]] = {}
        self.counts: Counter[str] = Counter()
        self.recent: deque[dict[str, Any]] = deque(maxlen=16)
        self.first_error: str | None = None

    @_diagnostic_only
    def begin(self, requests: Sequence[str]) -> None:
        """Start a model step, retaining no request or cache objects."""
        self.step += 1
        self.requests = tuple(str(req) for req in requests)
        self.loads.clear()
        self.load_generations.clear()
        self.consumed.clear()
        self.uses.clear()
        self.saves.clear()
        self._note("begin", requests=len(requests))

    @_diagnostic_only
    def load(
        self, group: int, layer: int, bank: int, generation: int, event: int
    ) -> None:
        """Remember the final FIFO load event for a layer's physical bank."""
        self.loads[group, layer, bank, generation] = event
        self.load_generations[group, layer, bank] = generation
        self._note("load", g=group, l=layer, b=bank, gen=generation, e=event)

    @_diagnostic_only
    def wait(
        self, group: int, layer: int, bank: int, generation: int, event: int | None
    ) -> None:
        """Check the event actually passed to the compute stream wait call."""
        key = (group, layer, bank, generation)
        expected = self.loads.pop(key, None)
        self._note("wait", g=group, l=layer, b=bank, gen=generation, e=event)
        if self.load_generations.get((group, layer, bank), generation) != generation:
            self._error("load_generation_mismatch")
        if expected is not None and event != expected:
            self._error("wrong_load_wait", expected=expected, actual=event)
        self.consumed.add(key)

    @_diagnostic_only
    def use(
        self, group: int, layer: int, bank: int, generation: int, event: int
    ) -> None:
        """Check that bank use follows the corresponding consumer wait hook."""
        self._note("use", g=group, l=layer, b=bank, gen=generation, e=event)
        if (group, layer, bank, generation) not in self.consumed:
            self._error("bank_used_without_wait_hook")
        self.uses[group, bank] = (generation, layer, event)

    @_diagnostic_only
    def save(
        self, group: int, layer: int, bank: int, generation: int, event: int
    ) -> None:
        """Check that the saved layer handed off the same physical bank."""
        self._note("save", g=group, l=layer, b=bank, gen=generation, e=event)
        if self.uses.get((group, bank), ())[:2] != (generation, layer):
            self._error("save_without_current_bank_handoff")
        self.saves[group, bank] = (generation, event)

    @_diagnostic_only
    def publish(self, group: int, generation: int, events: Sequence[int]) -> None:
        """Check that publication includes every last bank save event."""
        expected = {
            event
            for (g, _), (gen, event) in self.saves.copy().items()
            if g == group and gen == generation
        }
        self._note("publish", g=group, gen=generation, events=list(events))
        missing = expected.difference(events)
        if missing:
            self._error("publication_missing_bank_tail", missing=sorted(missing))

    @_diagnostic_only
    def finish(self, req_id: str) -> None:
        """Write a compact request summary; no device completion is inferred."""
        audit_record(
            "flow",
            req=req_id,
            rank=self.rank,
            role=self.role,
            step=self.step,
            worker_counts=dict(self.counts),
            first_worker_error=self.first_error,
            recent=list(self.recent),
        )

    def _note(self, stage: str, **fields: Any) -> None:
        self.counts[stage] += 1
        self.recent.append(dict(stage=stage, step=self.step, **fields))

    def _error(self, error: str, **fields: Any) -> None:
        if self.first_error is not None:
            return
        self.first_error = error
        audit_record(
            "flow",
            reqs=list(self.requests),
            rank=self.rank,
            role=self.role,
            error=error,
            recent=list(self.recent),
            **fields,
        )
