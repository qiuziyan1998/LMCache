# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in CPU payload hashes for P-save/P-get/D-get diagnostics.

LMCACHE_PREFILL_STORE_DEBUG=1 selects the first and last physical owner layer in
each group. Complete cache chunks are hashed in logical order, excluding
allocator padding. SHA-256 is truncated to 64 bits. Each stage/group prints
at most eight 65536-token intervals (first four and last four), two per line.
This is a transfer diagnostic, not an all-layer correctness verdict.
"""

from __future__ import annotations

# Standard
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Sequence
import ctypes
import hashlib
import logging
import os
import queue
import struct
import sys
import threading

SEGMENT_TOKENS = 65536
_ENABLED = os.environ.get("LMCACHE_PREFILL_STORE_DEBUG", "0") == "1"
_LOGGER = logging.getLogger("lmcache.prefill_kv_hash")
_INSTANCE_LOCK = threading.Lock()
_INSTANCE: "KVHashRecorder | None" = None


def _log(tag: int, message: str) -> None:
    if not _LOGGER.handlers:
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOGGER.addHandler(handler)
    _LOGGER.info("[KVH] p%d q%d %s", os.getpid(), tag, message)


def kv_hash_enabled() -> bool:
    """Return the content-diagnostic switch read before worker startup."""
    return _ENABLED


def _recorder() -> "KVHashRecorder":
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = KVHashRecorder()
        return _INSTANCE


def submit_kv_hash(*args: Any, **kwargs: Any) -> "CPUHashFence | None":
    """Submit ready CPU sources, or sources fenced by supplied producer events.

    Uses ``KVHashRecorder.submit``'s arguments. Disabled by default; disabled
    calls retain nothing. Submission never waits for queue space or the NPU.
    """
    if _ENABLED:
        try:
            return _recorder().submit(*args, **kwargs)
        except Exception as error:
            # An unavailable diagnostic must not fail a serving request.
            if len(args) >= 3:
                _recorder().note(args[0], args[1], args[2], type(error).__name__)
            else:
                _LOGGER.warning("[KVH] submit_failed %s", type(error).__name__)
    return None


def finish_kv_hash(req_id: str) -> None:
    """Queue final partial-interval output without waiting on the serving thread."""
    if _ENABLED and _INSTANCE is not None:
        _INSTANCE.finish(req_id)


def skip_kv_hash(stage: str, req_id: str, group: int, reason: str) -> None:
    """Mark missing coverage explicitly without reading an unfenced payload."""
    if _ENABLED:
        _recorder().note(stage, req_id, group, reason)


def drain_kv_hash(timeout: float = 60.0) -> bool:
    """Flush and drain at engine shutdown, before CPU slabs are unmapped.

    Returns False on timeout; the caller must keep the slabs alive in that
    case. This shutdown-only API is never called at a model-layer boundary.
    """
    return _INSTANCE is None or _INSTANCE.drain(timeout)


class CPUHashFence:
    """CPU-only readiness fence consumed by the background persistent writer.

    S hashes finish before persistence reads their CPU sources. Hash failures
    release the fence and mark coverage incomplete, without failing serving.
    This adds no NPU event or compute-stream wait.
    """

    cpu_hash_fence = True

    def __init__(self) -> None:
        self.ready = threading.Event()

    def synchronize(self) -> None:
        """Wait on the hash worker, only from a background writer thread."""
        self.ready.wait()


@dataclass
class _Group:
    tag: int
    stage: str
    group: int
    layers: tuple[int, ...]
    digest: Any = field(default_factory=hashlib.sha256)
    start: int | None = None
    end: int = 0
    total: int = 0
    tail: deque = field(default_factory=lambda: deque(maxlen=4))
    output: list[str] = field(default_factory=list)
    error: str = ""


class KVHashRecorder:
    """Hash immutable CPU chunks on one bounded background worker.

    At most max_jobs data jobs retain source objects at once. Queue pressure
    marks the diagnostic incomplete instead of blocking serving or growing
    ownership without bound. The worker releases every acquired reference.
    """

    def __init__(self, max_jobs: int = 16, emit: Any = _log) -> None:
        self.emit = emit
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.slots = threading.BoundedSemaphore(max_jobs)
        self.lock = threading.Lock()
        self.submitted: dict[tuple[str, str, int], int] = {}
        self.tags: dict[str, int] = {}
        self.groups: dict[tuple[str, str, int], _Group] = {}
        self.ids = count(1)
        self.worker = threading.Thread(target=self._run, daemon=True, name="kv-hash")
        self.worker.start()

    def submit(
        self,
        stage: str,
        req_id: str,
        group: int,
        starts: Sequence[int],
        ends: Sequence[int],
        rows: Sequence[Sequence[Any]],
        chunk_size: int,
        *,
        producer_events: tuple[Any, ...] = (),
        layer_ids: tuple[int, ...] | None = None,
        keepers: tuple[Any, ...] = (),
    ) -> CPUHashFence | None:
        """Retain only newly observed full chunks and enqueue a nonblocking job.

        stage is S (P before persistence), P (P cache sources), or D (D cache
        sources). rows is layer-major; explicit layer_ids names preselected rows.
        producer_events fence asynchronous D2H snapshots. CPU sources must be
        immutable after those events. keepers optionally owns extra CPU storage.
        Duplicate prefixes are skipped, but separate requests/stages never share
        progress. The output states actual intervals, including shorter tails.
        """
        if not rows or not req_id:
            return
        selected = tuple(sorted({0, len(rows) - 1}))
        layers = layer_ids or selected
        if layer_ids is not None:
            selected = tuple(range(len(rows)))
        key = (req_id, stage, group)
        with self.lock:
            # D records its initial prompt history, not later generated KV.
            if stage == "D" and key in self.submitted:
                return None
            old = self.submitted.get(key, -1)
            begin = bisect_right(ends, old)
            records = [
                (
                    int(starts[i]),
                    int(ends[i]),
                    tuple(rows[layer][i] for layer in selected),
                )
                for i in range(begin, len(ends))
                if int(ends[i]) - int(starts[i]) == chunk_size
            ]
            if not records:
                return
            self.submitted[key] = records[-1][1]
            if not self.slots.acquire(blocking=False):
                self.queue.put(("error", key, layers, "queue_full"))
                return
            owners = {id(obj): obj for _, _, row in records for obj in row}
            held = []
            try:
                if chunk_size <= 0 or SEGMENT_TOKENS % chunk_size:
                    raise ValueError("chunk_alignment")
                for obj in owners.values():
                    obj.ref_count_up()
                    held.append(obj)
                    if obj.raw_tensor.device.type != "cpu":
                        raise ValueError("non_cpu_source")
                completion = CPUHashFence()
                self.queue.put(
                    (
                        "data",
                        key,
                        layers,
                        records,
                        tuple(held),
                        producer_events,
                        keepers,
                        completion,
                    )
                )
                return completion
            except Exception as error:
                for obj in held:
                    obj.ref_count_down()
                self.slots.release()
                self.queue.put(("error", key, layers, type(error).__name__))
        return None

    def finish(self, req_id: str) -> None:
        """Enqueue request finalization after its last submitted source."""
        with self.lock:
            for key in tuple(self.submitted):
                if key[0] == req_id:
                    self.submitted.pop(key)
            self.queue.put(("finish", req_id))

    def note(self, stage: str, req_id: str, group: int, reason: str) -> None:
        """Queue one incomplete-coverage reason for the request summary."""
        self.queue.put(("error", (req_id, stage, group), (), reason))

    def drain(self, timeout: float = 60.0) -> bool:
        """Flush all completed requests and wait at shutdown or in CPU tests."""
        done = threading.Event()
        with self.lock:
            self.submitted.clear()
            self.queue.put(("drain", done))
        return done.wait(timeout)

    def _group(self, key: tuple[str, str, int], layers: tuple[int, ...]) -> _Group:
        if key not in self.groups:
            req, stage, group = key
            if req not in self.tags:
                self.tags[req] = next(self.ids)
                self.emit(
                    self.tags[req], "req=" + req.replace("\n", "_").replace("\r", "_")
                )
            state = _Group(self.tags[req], stage, group, layers)
            self.groups[key] = state
            self.emit(
                state.tag,
                f"{stage} g{group} layers={','.join(map(str, layers))} "
                f"seg={SEGMENT_TOKENS} sha256/64 full_chunks_only",
            )
        return self.groups[key]

    def _output(self, state: _Group, record: tuple[int, int, str]) -> None:
        start, end, digest = record
        state.output.append(f"{start}-{end}={digest}")
        if len(state.output) == 2:
            self._flush_output(state)

    def _flush_output(self, state: _Group) -> None:
        if state.output:
            self.emit(
                state.tag, f"{state.stage} g{state.group} " + " ".join(state.output)
            )
            state.output.clear()

    def _seal(self, state: _Group) -> None:
        if state.start is None:
            return
        record = (state.start, state.end, state.digest.hexdigest()[:16])
        state.total += 1
        if state.total <= 4:
            self._output(state, record)
        else:
            state.tail.append(record)
        state.digest = hashlib.sha256()
        state.start = None

    def _finish(self, req: str) -> None:
        for key in tuple(self.groups):
            if key[0] != req:
                continue
            state = self.groups.pop(key)
            self._seal(state)
            for record in state.tail:
                self._output(state, record)
            self._flush_output(state)
            self.emit(
                state.tag,
                f"{state.stage} g{state.group} done "
                f"total={state.total} shown={min(state.total, 8)} "
                f"ok={int(not state.error)} err={state.error or '-'}",
            )
        self.tags.pop(req, None)

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            kind = job[0]
            if kind == "finish":
                self._finish(job[1])
            elif kind == "drain":
                for req in tuple(self.tags):
                    self._finish(req)
                job[1].set()
            elif kind == "error":
                self._group(job[1], job[2]).error = job[3]
            else:
                _, key, layers, records, held, events, keepers, completion = job
                state = self._group(key, layers)
                row = obj = view = base = None
                try:
                    for event in events:
                        # Background only; never the serving thread.
                        event.synchronize()
                    if state.error:
                        continue
                    for start, end, row in records:
                        if state.start is not None and start != state.end:
                            state.error = "gap"
                            self._seal(state)
                        if state.start is None:
                            state.start = start
                        for layer, obj in zip(layers, row, strict=True):
                            is_page = callable(getattr(obj, "layer_data_ptr", None))
                            pointer = (
                                obj.layer_data_ptr(layer) if is_page else obj.data_ptr
                            )
                            size = (
                                obj.layer_size_bytes(layer)
                                if is_page
                                else obj.get_size()
                            )
                            base = obj.raw_tensor
                            base_ptr = int(base.data_ptr())
                            base_bytes = int(base.numel() * base.element_size())
                            if (
                                size <= 0
                                or pointer < base_ptr
                                or pointer + size > base_ptr + base_bytes
                            ):
                                raise ValueError("payload_bounds")
                            state.digest.update(
                                struct.pack("<QQIQ", start, end, layer, size)
                            )
                            view = (ctypes.c_ubyte * size).from_address(pointer)
                            state.digest.update(memoryview(view).cast("B"))
                        state.end = end
                        if end % SEGMENT_TOKENS == 0:
                            self._seal(state)
                except Exception as error:
                    state.error = type(error).__name__
                finally:
                    for obj in held:
                        try:
                            obj.ref_count_down()
                        except Exception as error:
                            state.error = type(error).__name__
                    self.slots.release()
                    completion.ready.set()
                    # Drop source owners before waiting for another queue job.
                    del (
                        records,
                        held,
                        events,
                        keepers,
                        job,
                        row,
                        obj,
                        view,
                        base,
                        completion,
                    )
