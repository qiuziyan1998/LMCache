# SPDX-License-Identifier: Apache-2.0
"""Stage 4 layerwise-prefill transfer-window protocol tests."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

# Third Party
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    LayerwisePrefillCallbackMetadata,
)
from vllm.v1.kv_cache_interface import DSAExecutionRow, DSAKVRow

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.integration.vllm.vllm_v1_adapter import (
    LayerwisePrefillSavePhase,
    LayerwisePrefillWindowCoordinator,
    LMCacheConnectorV1Impl,
    _DSAKVTopologyCache,
)

# Local
from tests.v1.test_vllm_kv_cache_config_cardinality import (
    _kv_cache_config,
    _patch_connector_startup,
)

PRODUCER_EXECUTIONS = (0, 1, 2, *(6 + 4 * index for index in range(18)), 78)


def _topology_cache(cardinalities: tuple[int, int] = (79, 22)) -> _DSAKVTopologyCache:
    config = _kv_cache_config(*cardinalities)
    topology = config.dsa_kv_topology
    rows = tuple(tuple(rows) for rows in topology.rows_by_group)
    executions = tuple(topology.executions)
    layer_name_to_row = {row.layer_name: row for rows_ in rows for row in rows_}
    execution_to_entry = {entry.execution_ordinal: entry for entry in executions}
    group_layer_names = tuple(
        tuple(row.layer_name for row in rows_) for rows_ in rows
    )
    return _DSAKVTopologyCache(
        descriptor=topology,
        layer_name_to_row=layer_name_to_row,
        execution_to_entry=execution_to_entry,
        group_layer_names=group_layer_names,
        group_cardinalities=cardinalities,
    )


class RecordingBackend:
    """Transfer backend double with protocol-observable events."""

    def __init__(
        self,
        *,
        supports_sync: bool = True,
        supports_window: bool = True,
        persists_indexer: bool = True,
        persist_futures: Optional[dict[Any, Future]] = None,
    ):
        self.supports_sync_callbacks = supports_sync
        self.supports_transfer_window = supports_window
        self.persists_indexer_group = persists_indexer
        self.events: list[str] = []
        self.persist_futures = persist_futures or {}
        self.aborted: list[str] = []

    def wait_for_load(self, metadata: Any) -> None:
        self.events.append(f"wait-{metadata.row.kv_group}")

    def submit_save(self, metadata: Any, kv_layer: Any, attn_metadata: Any) -> None:
        self.events.append(f"submit-save-{metadata.row.kv_group}")

    def submit_load(self, metadata: Any) -> None:
        self.events.append("submit-load")

    def finish_save(self, metadata: Any) -> Optional[Future]:
        self.events.append(f"finish-{metadata.row.kv_group}")
        return self.persist_futures.get(metadata.row)

    def sync_save(
        self, metadata: Any, kv_layer: Any, attn_metadata: Any
    ) -> Optional[Future]:
        self.events.append(f"sync-save-{metadata.row.kv_group}")
        return self.persist_futures.get(metadata.row)

    def abort_request(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _callback(
    cache: _DSAKVTopologyCache,
    execution_ordinal: int,
    generation: int = 7,
    *,
    request_generations: Optional[tuple[tuple[str, int], ...]] = None,
) -> Any:
    entry = cache.execution_to_entry[execution_ordinal]
    latent_row = entry.latent
    latent = DSAKVRow(
        latent_row.layer_name,
        latent_row.execution_ordinal,
        latent_row.kv_group,
        latent_row.row_ordinal,
        latent_row.bank,
    )
    indexer = None
    if entry.indexer is not None:
        indexer_row = entry.indexer
        indexer = DSAKVRow(
            indexer_row.layer_name,
            indexer_row.execution_ordinal,
            indexer_row.kv_group,
            indexer_row.row_ordinal,
            indexer_row.bank,
        )
    execution = DSAExecutionRow(entry.execution_ordinal, latent, indexer)
    return LayerwisePrefillCallbackMetadata.for_execution(
        execution,
        request_generations
        if request_generations is not None
        else (("req-1", generation),),
    )


def _kv_layer() -> list[torch.Tensor]:
    return [torch.empty(4)]


def _drive_full_request(
    coordinator: LayerwisePrefillWindowCoordinator,
    cache: _DSAKVTopologyCache,
) -> None:
    """Wait, submit, and finish every LATENT row and producer INDEXER row."""

    for execution_ordinal, entry in cache.execution_to_entry.items():
        callbacks = _callback(cache, execution_ordinal)
        for metadata in callbacks:
            coordinator.wait_for_load(metadata)
        for metadata in callbacks:
            coordinator.submit_save(metadata, _kv_layer())
        coordinator.submit_load(callbacks[0])
        for metadata in callbacks:
            coordinator.finish_save(metadata)


def test_generator_counts_are_exact_79_and_22() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    _drive_full_request(coordinator, cache)

    assert backend.events.count("submit-save-0") == 79
    assert backend.events.count("submit-save-1") == 22
    assert backend.events.count("finish-0") == 79
    assert backend.events.count("finish-1") == 22
    # One load submit per execution with a next row; execution 78 is the
    # final row of both groups and has nothing left to prefetch. The 57
    # shared consumers never generate a group-1 submission.
    assert backend.events.count("submit-load") == 78
    assert coordinator.request_persist_done("req-1") is True


def test_execution_six_indexer_is_row_three_bank_one() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for execution_ordinal in range(7):
        for metadata in _callback(cache, execution_ordinal):
            coordinator.submit_save(metadata, _kv_layer())
            coordinator.finish_save(metadata)

    execution = cache.execution_to_entry[6]
    assert execution.indexer is not None
    assert execution.indexer.row_ordinal == 3
    assert execution.indexer.bank == 1
    # Execution 6's tail events: LATENT row 6, then INDEXER row 3, each
    # submitted and finished before the next row starts.
    assert backend.events[-4:] == [
        "submit-save-0",
        "finish-0",
        "submit-save-1",
        "finish-1",
    ]
    # Row identity in the job map is group-local, not the model layer id.
    arena = coordinator._arenas["req-1"]
    assert (1, 3) in arena.jobs
    assert (1, 6) not in arena.jobs


def test_shared_consumers_never_touch_group_one() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for execution_ordinal in (3, 4, 5):
        entry = cache.execution_to_entry[execution_ordinal]
        assert entry.indexer is None
        metadata = _callback(cache, execution_ordinal)[0]
        coordinator.wait_for_load(metadata)
        coordinator.submit_load(metadata)

    assert all("-1" not in event for event in backend.events)


def test_finish_before_submit_and_out_of_order_rows_fail_closed() -> None:
    cache = _topology_cache()
    coordinator = LayerwisePrefillWindowCoordinator(cache, RecordingBackend())

    # The wait is the request's first callback and creates its arena.
    coordinator.wait_for_load(_callback(cache, 0)[0])
    with pytest.raises(RuntimeError, match="before its submit"):
        coordinator.finish_save(_callback(cache, 0)[0])

    with pytest.raises(RuntimeError, match="per-group row order"):
        coordinator.submit_save(_callback(cache, 1)[0], _kv_layer())


def test_duplicate_finish_is_idempotent() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    metadata = _callback(cache, 0)[0]

    coordinator.submit_save(metadata, _kv_layer())
    coordinator.finish_save(metadata)
    coordinator.finish_save(metadata)

    assert backend.events.count("finish-0") == 1
    arena = coordinator._arenas["req-1"]
    assert arena.jobs[(0, 0)].phase is LayerwisePrefillSavePhase.PERSIST_DONE


def test_delayed_persistence_future_blocks_request_completion() -> None:
    cache = _topology_cache()
    metadata = _callback(cache, 0)[0]
    future: Future = Future()
    backend = RecordingBackend(persist_futures={metadata.row: future})
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    coordinator.submit_save(metadata, _kv_layer())
    coordinator.finish_save(metadata)
    coordinator.finish_save(metadata)
    assert backend.events.count("finish-0") == 1

    arena = coordinator._arenas["req-1"]
    assert arena.jobs[(0, 0)].phase is LayerwisePrefillSavePhase.SOURCE_DONE
    assert coordinator.request_persist_done("req-1") is False

    future.set_result(None)
    coordinator.poll_completed_persists()
    coordinator.finish_save(metadata)
    assert backend.events.count("finish-0") == 1
    assert arena.jobs[(0, 0)].phase is LayerwisePrefillSavePhase.PERSIST_DONE


def test_stale_generation_cannot_touch_current_arena() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    old = _callback(cache, 0, generation=7)[0]
    coordinator.submit_save(old, _kv_layer())
    coordinator.finish_save(old)

    new = _callback(cache, 0, generation=8)[0]
    coordinator.submit_save(new, _kv_layer())

    # A late old-generation finish may only clean its own resources.
    coordinator.finish_save(old)
    arena = coordinator._arenas["req-1"]
    assert arena.allocation_generation == 8
    assert (0, 0) in arena.jobs
    with pytest.raises(RuntimeError, match="superseded"):
        coordinator.submit_save(old, _kv_layer())
    assert coordinator.request_persist_done("req-1") is False


def test_request_id_reuse_uses_generation_arena_isolation() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    _drive_full_request(coordinator, cache)
    assert coordinator.request_persist_done("req-1") is True

    reused = _callback(cache, 0, generation=9)[0]
    coordinator.submit_save(reused, _kv_layer())

    arena = coordinator._arenas["req-1"]
    assert arena.allocation_generation == 9
    assert len(arena.jobs) == 1
    assert coordinator.request_persist_done("req-1") is False


def test_bounded_pending_jobs_fail_closed_instead_of_dropping() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(
        cache,
        backend,
        max_pending_jobs=2,
    )

    # Execution 0 is a producer: two rows fit exactly inside the bound.
    for metadata in _callback(cache, 0):
        coordinator.submit_save(metadata, _kv_layer())

    with pytest.raises(RuntimeError, match="bounded queue"):
        coordinator.submit_save(_callback(cache, 1)[0], _kv_layer())

    # Nothing was dropped: both accepted submissions are intact.
    arena = coordinator._arenas["req-1"]
    assert len(arena.jobs) == 2


def test_completion_barrier_requires_both_groups() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for execution_ordinal, entry in cache.execution_to_entry.items():
        metadata = _callback(cache, execution_ordinal)[0]
        coordinator.submit_save(metadata, _kv_layer())
        coordinator.finish_save(metadata)

    assert coordinator.request_persist_done("req-1") is False

    for execution_ordinal in PRODUCER_EXECUTIONS:
        metadata = _callback(cache, execution_ordinal)[1]
        coordinator.submit_save(metadata, _kv_layer())
        coordinator.finish_save(metadata)

    assert coordinator.request_persist_done("req-1") is True


def test_release_drops_arenas_and_aborts_backend() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    _drive_full_request(coordinator, cache)
    coordinator.release_request("req-1")

    assert coordinator.has_request("req-1") is False
    assert backend.aborted == ["req-1"]
    events = backend.events.copy()
    coordinator.finish_save(_callback(cache, 0)[0])
    assert backend.events == events


def test_blocking_barrier_waits_for_outstanding_futures() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    futures: dict[Any, Future] = {}

    def finish_with_future(metadata: Any) -> Future:
        future: Future = Future()
        futures[metadata.row] = future
        return future

    backend.finish_save = finish_with_future
    coordinator = LayerwisePrefillWindowCoordinator(
        cache,
        backend,
        max_pending_jobs=102,
        max_pending_bytes=1 << 30,
    )

    _drive_full_request(coordinator, cache)
    assert coordinator.request_persist_done("req-1") is False

    for future in futures.values():
        future.set_result(None)
    coordinator.wait_for_request_persist_done("req-1")
    assert coordinator.request_persist_done("req-1") is True


def test_blocking_barrier_fails_closed_when_rows_are_missing() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for metadata in _callback(cache, 0):
        coordinator.submit_save(metadata, _kv_layer())
        coordinator.finish_save(metadata)

    with pytest.raises(RuntimeError, match="missing"):
        coordinator.wait_for_request_persist_done("req-1")


def test_multi_request_callback_counts_host_resources_once() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    entry = cache.execution_to_entry[0]
    latent_row = entry.latent
    latent = DSAKVRow(
        latent_row.layer_name,
        latent_row.execution_ordinal,
        latent_row.kv_group,
        latent_row.row_ordinal,
        latent_row.bank,
    )
    indexer_row = entry.indexer
    assert indexer_row is not None
    indexer = DSAKVRow(
        indexer_row.layer_name,
        indexer_row.execution_ordinal,
        indexer_row.kv_group,
        indexer_row.row_ordinal,
        indexer_row.bank,
    )
    metadata = LayerwisePrefillCallbackMetadata(
        DSAExecutionRow(entry.execution_ordinal, latent, indexer),
        latent,
        (("req-a", 1), ("req-b", 2)),
    )

    coordinator.submit_save(metadata, _kv_layer())
    # One host allocation, counted once even though two requests share it.
    assert coordinator.pending_bytes() == 16
    assert coordinator.pending_jobs() == 1
    coordinator.finish_save(metadata)

    assert coordinator.pending_bytes() == 0
    assert coordinator.pending_jobs() == 0
    for req_id in ("req-a", "req-b"):
        arena = coordinator._arenas[req_id]
        job = arena.jobs[(0, 0)]
        assert job.phase is LayerwisePrefillSavePhase.PERSIST_DONE
        assert coordinator.request_persist_done(req_id) is False
    assert backend.events == ["submit-save-0", "finish-0"]


@pytest.mark.parametrize("backend_present", [False, True])
def test_unsupported_backend_fails_closed_without_protocol_mutation(
    backend_present: bool,
) -> None:
    cache = _topology_cache()
    backend = (
        RecordingBackend(supports_sync=False, supports_window=False)
        if backend_present
        else None
    )
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    assert coordinator.supports_sync_callbacks is False
    # Full-resident INDEXER support is independent of P-node callbacks.
    assert coordinator.persists_indexer_group is True
    assert coordinator.supports_transfer_window is False

    for metadata in _callback(cache, 0):
        for callback, args in (
            (coordinator.wait_for_load, (metadata,)),
            (coordinator.save, (metadata, _kv_layer())),
            (coordinator.submit_save, (metadata, _kv_layer())),
            (coordinator.submit_load, (metadata,)),
            (coordinator.finish_save, (metadata,)),
        ):
            with pytest.raises(RuntimeError, match="row-aware.*backend"):
                callback(*args)
            assert coordinator.has_request("req-1") is False
            assert coordinator.request_persist_done("req-1") is False
            assert coordinator.pending_jobs() == 0
            assert coordinator.pending_bytes() == 0

    with pytest.raises(RuntimeError, match="unknown request"):
        coordinator.wait_for_request_persist_done("req-1")
    if backend is not None:
        assert backend.events == []


def test_sync_contract_saves_one_row_to_persist_done() -> None:
    cache = _topology_cache()
    backend = RecordingBackend(supports_window=False)
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    assert coordinator.supports_sync_callbacks is True
    assert coordinator.supports_transfer_window is False

    metadata = _callback(cache, 0)[0]
    coordinator.save(metadata, _kv_layer())

    arena = coordinator._arenas["req-1"]
    assert arena.jobs[(0, 0)].phase is LayerwisePrefillSavePhase.PERSIST_DONE
    assert backend.events == ["sync-save-0"]

    with pytest.raises(RuntimeError, match="per-group row order"):
        coordinator.save(metadata, _kv_layer())
    assert backend.events == ["sync-save-0"]


@pytest.mark.parametrize("failure", [False, True])
def test_sync_source_completion_waits_for_shared_page_commit(failure: bool) -> None:
    cache = _topology_cache()
    future = Future()
    backend = RecordingBackend(supports_window=False)
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    for execution in cache.execution_to_entry:
        for metadata in _callback(cache, execution):
            backend.persist_futures[metadata.row] = future
            coordinator.save(metadata, _kv_layer())
    assert not coordinator.request_persist_done("req-1")
    assert coordinator.pending_jobs() == 101
    assert coordinator.pending_bytes() == 0  # CPU manifests, not live NPU sources.
    if failure:
        future.set_exception(RuntimeError("page commit failed"))
        with pytest.raises(RuntimeError, match="page commit failed"):
            coordinator.wait_for_request_persist_done("req-1")
        assert not coordinator.request_persist_done("req-1")
    else:
        future.set_result(None)
        coordinator.wait_for_request_persist_done("req-1")
        assert coordinator.request_persist_done("req-1")
        assert coordinator.pending_jobs() == 0
    coordinator.release_request("req-1")
    assert not coordinator.request_persist_done("req-1")


def test_sync_rejects_invalid_persistence_result() -> None:
    cache = _topology_cache()
    backend = RecordingBackend(supports_window=False)
    backend.sync_save = Mock(return_value=True)
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    with pytest.raises(TypeError, match="Future or None"):
        coordinator.save(_callback(cache, 0)[0], _kv_layer())
    assert not coordinator.has_request("req-1")


@pytest.mark.parametrize("backend_present", [False, True])
def test_sync_persistence_requires_row_aware_backend(
    backend_present: bool,
) -> None:
    cache = _topology_cache((1, 1))
    backend = RecordingBackend(supports_window=False) if backend_present else None
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for metadata in _callback(cache, 0):
        if backend is None:
            with pytest.raises(RuntimeError, match="row-aware load/save backend"):
                coordinator.save(metadata, _kv_layer())
            assert coordinator.has_request("req-1") is False
            assert coordinator.request_persist_done("req-1") is False
        else:
            coordinator.save(metadata, _kv_layer())

    assert coordinator.pending_jobs() == 0
    assert coordinator.pending_bytes() == 0
    assert coordinator.request_persist_done("req-1") is backend_present
    if backend is not None:
        assert backend.events == ["sync-save-0", "sync-save-1"]


def test_sync_contract_repeats_rows_for_chunked_prefill() -> None:
    cache = _topology_cache()
    backend = RecordingBackend(supports_window=False)
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    for _ in range(2):
        for execution_ordinal in cache.execution_to_entry:
            for metadata in _callback(cache, execution_ordinal):
                coordinator.wait_for_load(metadata)
                coordinator.save(metadata, _kv_layer())
        assert coordinator.request_persist_done("req-1") is True

    assert backend.events.count("sync-save-0") == 2 * 79
    assert backend.events.count("sync-save-1") == 2 * 22
    assert backend.events.count("wait-0") == 2 * 79
    assert backend.events.count("wait-1") == 2 * 22


def test_transfer_window_repeats_rows_after_previous_chunk_persists() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)

    _drive_full_request(coordinator, cache)
    assert coordinator.request_persist_done("req-1") is True
    _drive_full_request(coordinator, cache)

    assert coordinator.request_persist_done("req-1") is True
    assert backend.events.count("submit-save-0") == 2 * 79
    assert backend.events.count("submit-save-1") == 2 * 22
    assert backend.events.count("wait-0") == 2 * 79
    assert backend.events.count("wait-1") == 2 * 22


@pytest.mark.parametrize(
    "callback", ["wait_for_load", "save", "submit_save", "submit_load"]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_lower_generation_batch_is_rejected_atomically(
    callback: str,
    mixed: bool,
) -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    current = _callback(cache, 0, generation=9)[0]
    coordinator.wait_for_load(current)
    generations = (("req-1", 8),)
    if mixed:
        generations = (("fresh", 10), *generations)
    stale = _callback(cache, 0, request_generations=generations)[0]
    events = backend.events.copy()

    args = (stale, _kv_layer()) if callback in ("save", "submit_save") else (stale,)
    with pytest.raises(RuntimeError, match="superseded|inactive"):
        getattr(coordinator, callback)(*args)
    assert backend.events == events
    assert coordinator.has_request("fresh") is False
    # The unseen lower generation must not replace the active arena.
    coordinator.save(current, _kv_layer())


@pytest.mark.parametrize("callback", ["wait_for_load", "save", "submit_save"])
def test_invalid_batch_does_not_supersede_an_earlier_member(callback: str) -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    current = _callback(cache, 0, request_generations=(("req-1", 9), ("req-2", 9)))[0]
    coordinator.wait_for_load(current)
    mixed = _callback(cache, 0, request_generations=(("req-1", 10), ("req-2", 8)))[0]
    events = backend.events.copy()
    args = (mixed, _kv_layer()) if callback != "wait_for_load" else (mixed,)

    with pytest.raises(RuntimeError, match="superseded"):
        getattr(coordinator, callback)(*args)
    assert backend.events == events
    coordinator.save(current, _kv_layer())


def test_failed_bootstrap_wait_does_not_create_an_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    wait = Mock(side_effect=[RuntimeError("missing restored prefix"), None])
    monkeypatch.setattr(backend, "wait_for_load", wait)
    metadata = _callback(cache, 0)[0]

    with pytest.raises(RuntimeError, match="missing restored prefix"):
        coordinator.wait_for_load(metadata)
    assert coordinator.has_request("req-1") is False
    coordinator.wait_for_load(metadata)
    assert wait.call_count == 2
    assert coordinator.has_request("req-1") is True


def test_failed_load_submit_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    submit = Mock(side_effect=[RuntimeError("load failed"), None])
    monkeypatch.setattr(backend, "submit_load", submit)
    metadata = _callback(cache, 0)[0]
    coordinator.wait_for_load(metadata)

    with pytest.raises(RuntimeError, match="load failed"):
        coordinator.submit_load(metadata)
    coordinator.submit_load(metadata)
    assert submit.call_count == 2


def test_release_blocks_old_and_equal_generations_but_allows_newer() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    coordinator.wait_for_load(_callback(cache, 0, generation=9)[0])
    coordinator.release_request("req-1")
    coordinator.release_request("req-1")
    events = backend.events.copy()

    for generation in (8, 9):
        old = _callback(cache, 0, generation=generation)[0]
        with pytest.raises(RuntimeError, match="released"):
            coordinator.wait_for_load(old)
        with pytest.raises(RuntimeError, match="released"):
            coordinator.submit_save(old, _kv_layer())
        coordinator.finish_save(old)
    assert backend.events == events
    assert coordinator.has_request("req-1") is False
    new = _callback(cache, 0, generation=10)[0]
    coordinator.wait_for_load(new)
    coordinator.submit_save(new, _kv_layer())
    coordinator.finish_save(new)
    assert coordinator.pending_jobs() == 0


def test_finish_requires_exact_batch_and_never_activates_a_generation() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    metadata = _callback(cache, 0, request_generations=(("req-1", 7), ("req-2", 7)))[0]
    coordinator.submit_save(metadata, _kv_layer())
    subset = _callback(cache, 0)[0]

    with pytest.raises(RuntimeError, match="identity"):
        coordinator.finish_save(subset)
    unknown = _callback(cache, 0, request_generations=(("fresh", 8),))[0]
    with pytest.raises(RuntimeError, match="unknown request"):
        coordinator.finish_save(unknown)
    assert coordinator.has_request("fresh") is False
    coordinator.finish_save(metadata)
    assert backend.events == ["submit-save-0", "finish-0"]


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("failure", ["later_cursor", "backend", "rollover"])
def test_save_failure_preserves_arena_generation_and_chunk_state(
    monkeypatch: pytest.MonkeyPatch, window: bool, failure: str
) -> None:
    cache = _topology_cache((1, 1) if failure == "rollover" else (2, 1))
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    generations = (("req-1", 7), ("req-2", 7))
    metadata = _callback(cache, 0, request_generations=generations)[0]
    save = coordinator.submit_save if window else coordinator.save
    save(metadata, _kv_layer())
    if window:
        coordinator.finish_save(metadata)
    arenas = dict(coordinator._arenas)
    jobs = {req: dict(arena.jobs) for req, arena in arenas.items()}
    cursors = {req: dict(arena.save_cursors) for req, arena in arenas.items()}

    if failure != "later_cursor":
        monkeypatch.setattr(
            backend, "submit_save" if window else "sync_save",
            Mock(side_effect=RuntimeError("submission failed")),
        )
    if failure == "later_cursor":
        generations = (("req-1", 9), ("req-2", 7))
    elif failure == "backend":
        generations = (("req-1", 9), ("req-2", 9))
    metadata = _callback(cache, 0, request_generations=generations)[0]
    with pytest.raises(RuntimeError, match="per-group row order|submission failed"):
        save(metadata, _kv_layer())
    for req, arena in arenas.items():
        assert coordinator._arenas[req] is arena
        assert arena.allocation_generation == 7
        assert arena.jobs == jobs[req]
        assert arena.save_cursors == cursors[req]
    assert coordinator._stale_arenas == {}


def _window_connector(
    monkeypatch,
    backend: Optional[Any],
) -> LMCacheConnectorV1Impl:
    _config, vllm_config, _observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=101,
    )
    impl = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(),
        kv_cache_config=_kv_cache_config(79, 22),
    )
    cache = impl._dsa_kv_topology_cache
    assert cache is not None
    impl._layerwise_prefill_window = LayerwisePrefillWindowCoordinator(
        cache,
        backend,
    )
    return impl


def test_connector_capabilities_and_delegation(monkeypatch) -> None:
    backend = RecordingBackend()
    impl = _window_connector(monkeypatch, backend)

    assert impl.supports_layerwise_prefill_eager_callbacks is True
    assert impl.supports_dsa_index_lmcache is True
    assert impl.supports_layerwise_prefill_transfer_window is True

    cache = impl._dsa_kv_topology_cache
    indexer_metadata = _callback(cache, 0)[1]
    impl.submit_layerwise_prefill_save(indexer_metadata, _kv_layer())
    impl.submit_layerwise_prefill_load(_callback(cache, 0)[0])
    impl.finish_layerwise_prefill_save(indexer_metadata)

    assert backend.events == [
        "submit-save-1",
        "submit-load",
        "finish-1",
    ]
    assert impl.layerwise_prefill_request_persist_done("req-1") is False


@pytest.mark.parametrize("faulty_property", ["backend", "engine"])
def test_p_worker_startup_preserves_property_attribute_error(
    monkeypatch: pytest.MonkeyPatch, faulty_property: str
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch, dsa_two_groups=True, model_num_layers=79
    )
    monkeypatch.setattr(
        adapter_module.VllmServiceFactory, "get_or_create_metadata", Mock()
    )
    error = AttributeError("original startup construction failure")

    class Engine:
        @property
        def layerwise_prefill_window_backend(self) -> RecordingBackend:
            raise error

    engine_getter = Mock(return_value=Engine())
    if faulty_property == "engine":
        engine_getter.side_effect = error
    monkeypatch.setattr(
        LMCacheConnectorV1Impl, "lmcache_engine", property(engine_getter)
    )
    create_window = Mock(wraps=LayerwisePrefillWindowCoordinator)
    monkeypatch.setattr(
        adapter_module, "LayerwisePrefillWindowCoordinator", create_window
    )

    with pytest.raises(AttributeError, match=str(error)) as exc_info:
        LMCacheConnectorV1Impl(
            vllm_config,
            KVConnectorRole.WORKER,
            SimpleNamespace(),
            kv_cache_config=_kv_cache_config(79, 22),
        )

    assert exc_info.value is error
    create_window.assert_not_called()
    assert observed == ["manager", "services"]


@pytest.mark.parametrize("p_node", [False, True])
def test_worker_startup_missing_backend_is_optional_only_when_feature_off(
    monkeypatch: pytest.MonkeyPatch, p_node: bool
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", str(p_node).lower())
    _, vllm_config, _ = _patch_connector_startup(
        monkeypatch, dsa_two_groups=True, model_num_layers=79
    )
    monkeypatch.setattr(
        adapter_module.VllmServiceFactory, "get_or_create_metadata", Mock()
    )
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "lmcache_engine",
        property(lambda self: SimpleNamespace()),
    )

    if p_node:
        with pytest.raises(AttributeError, match="layerwise_prefill_window_backend"):
            LMCacheConnectorV1Impl(
                vllm_config,
                KVConnectorRole.WORKER,
                SimpleNamespace(),
                kv_cache_config=_kv_cache_config(79, 22),
            )
    else:
        impl = LMCacheConnectorV1Impl(
            vllm_config,
            KVConnectorRole.WORKER,
            SimpleNamespace(),
            kv_cache_config=_kv_cache_config(79, 22),
        )
        assert impl.supports_layerwise_prefill_eager_callbacks is False
        assert impl.supports_layerwise_prefill_transfer_window is False
        assert impl.supports_dsa_index_lmcache is True


def test_p_worker_startup_with_sync_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch, dsa_two_groups=True, model_num_layers=79
    )
    monkeypatch.setattr(
        adapter_module.VllmServiceFactory, "get_or_create_metadata", Mock()
    )
    backend = RecordingBackend(supports_window=False)
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "lmcache_engine",
        property(
            lambda self: SimpleNamespace(layerwise_prefill_window_backend=backend)
        ),
    )

    impl = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.WORKER,
        SimpleNamespace(),
        kv_cache_config=_kv_cache_config(79, 22),
    )

    assert impl.supports_layerwise_prefill_eager_callbacks is True
    assert impl.supports_layerwise_prefill_transfer_window is False
    assert impl.supports_dsa_index_lmcache is True
    assert observed == ["manager", "services", "layerwise", "metrics"]


def test_p_scheduler_startup_does_not_access_backend_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch, dsa_two_groups=True, model_num_layers=79
    )
    backend_getter = Mock(side_effect=AssertionError("scheduler accessed backend"))

    class Engine:
        layerwise_prefill_window_backend = property(backend_getter)

    monkeypatch.setattr(
        LMCacheConnectorV1Impl, "lmcache_engine", property(lambda self: Engine())
    )

    impl = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(),
        kv_cache_config=_kv_cache_config(79, 22),
    )

    backend_getter.assert_not_called()
    assert impl.supports_layerwise_prefill_eager_callbacks is False
    assert impl.supports_layerwise_prefill_transfer_window is False
    assert observed == ["manager", "services", "layerwise", "metrics"]


def test_connector_without_backend_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _window_connector(monkeypatch, None)

    assert impl.supports_layerwise_prefill_eager_callbacks is False
    assert impl.supports_dsa_index_lmcache is True
    assert impl.supports_layerwise_prefill_transfer_window is False

    cache = impl._dsa_kv_topology_cache
    legacy_saver = Mock(side_effect=AssertionError("legacy saver must not run"))
    legacy_waiter = Mock(side_effect=AssertionError("legacy load wait must not run"))
    monkeypatch.setattr(impl, "save_kv_layer", legacy_saver)
    monkeypatch.setattr(impl, "wait_for_layer_load", legacy_waiter)
    metadata = _callback(cache, 0)[0]

    for callback, args in (
        (impl.wait_for_layerwise_prefill_load, (metadata,)),
        (impl.save_layerwise_prefill_kv_layer, (metadata, _kv_layer())),
        (impl.submit_layerwise_prefill_save, (metadata, _kv_layer())),
        (impl.submit_layerwise_prefill_load, (metadata,)),
        (impl.finish_layerwise_prefill_save, (metadata,)),
    ):
        with pytest.raises(RuntimeError, match="backend"):
            callback(*args)
        assert impl.layerwise_prefill_request_persist_done("req-1") is False

    with pytest.raises(RuntimeError, match="unknown request"):
        impl.wait_for_layerwise_prefill_request_persist_done("req-1")
    legacy_saver.assert_not_called()
    legacy_waiter.assert_not_called()


def test_dsa_long_request_admission_gates_sparse_decode() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._dsa_kv_policy_threshold = 256
    impl._dsa_kv_topology_cache = _topology_cache()
    tracker = SimpleNamespace(
        req_id="req-long",
        token_ids=list(range(1000)),
        prompt_len=1000,
        num_lmcache_cached_tokens=768,
    )

    # A long request without a full remote hit must not dense prefill.
    with pytest.raises(RuntimeError, match="no exact full remote hit"):
        impl._dsa_long_request_admission_check(tracker)

    # The exact lookup count, not the chunk-aligned sparse working frontier,
    # proves that the remote cache covers the complete prompt.
    tracker.num_lmcache_cached_tokens = 1000
    impl._dsa_long_request_admission_check(tracker)

    # A short prompt may grow past the policy threshold during decode. That is
    # a policy transition, not a new long-prompt admission decision.
    short_tracker = SimpleNamespace(
        req_id="req-short",
        token_ids=list(range(300)),
        prompt_len=200,
        num_lmcache_cached_tokens=0,
    )
    impl._dsa_long_request_admission_check(short_tracker)

    # Non-canonical group cardinalities fail closed.
    impl._dsa_kv_topology_cache = _topology_cache((79, 79))
    with pytest.raises(RuntimeError, match="canonical 79/22"):
        impl._dsa_long_request_admission_check(tracker)


@pytest.mark.parametrize("backend_present", [False, True])
def test_connector_logs_p_node_observability(
    monkeypatch: pytest.MonkeyPatch,
    backend_present: bool,
) -> None:
    _config, vllm_config, _observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=101,
    )
    impl = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(),
        kv_cache_config=_kv_cache_config(79, 22),
    )
    impl._role = KVConnectorRole.WORKER
    impl._dsa_kv_topology_cache = _topology_cache()
    impl.config = SimpleNamespace(
        max_local_cpu_size=120.0,
        extra_config={"global_segment_size": 137_438_953_472},
    )
    if backend_present:
        monkeypatch.setattr(
            LMCacheConnectorV1Impl,
            "lmcache_engine",
            property(
                lambda self: SimpleNamespace(
                    layerwise_prefill_window_backend=RecordingBackend()
                )
            ),
        )
    logged = []
    warnings = []
    monkeypatch.setattr(
        adapter_module.logger,
        "info",
        lambda message, *args, **_kwargs: logged.append(
            message % args if args else message
        ),
    )
    monkeypatch.setattr(
        adapter_module.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(
            message % args if args else message
        ),
    )

    window = impl._build_layerwise_prefill_window()

    assert window is not None
    message = next(
        message for message in logged if "Layerwise-prefill P node" in message
    )
    assert "cpu_cache_bytes=128849018880" in message
    assert "mooncake_segment_bytes=137438953472" in message
    assert f"connector_transfer_window={backend_present}" in message
    assert f"connector_sync_callbacks={backend_present}" in message
    if backend_present:
        assert warnings == []
    else:
        diagnostic = " ".join(warnings)
        assert "Layerwise-prefill P mode is unavailable" in diagnostic
        assert "row-aware load/save backend unavailable" in diagnostic
        assert (
            "If enabling P mode, disable VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"
            in diagnostic
        )
        assert "set it to false or unset it" in diagnostic
        assert "full-resident fallback" in diagnostic
        assert "cannot start" not in diagnostic
        assert "override" not in diagnostic
        assert "138" not in diagnostic
