# SPDX-License-Identifier: Apache-2.0
"""Stage 4 layerwise-prefill transfer-window protocol tests."""

# Standard
from concurrent.futures import Future
from dataclasses import replace
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
from lmcache.integration.vllm.layerwise_prefill import LayerwisePrefillRequest
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.vllm_v1_adapter import (
    LayerwisePrefillSavePhase,
    LayerwisePrefillWindowCoordinator,
    LMCacheConnectorV1Impl,
    LMCacheConnectorMetadata,
    ReqMeta,
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


class ManagedBackend(RecordingBackend):
    """Managed public contract double; pre-HCOM failures surface only at finish."""

    manages_pending_work = True
    accepts_coordinator_validation_errors = True

    def __init__(self) -> None:
        super().__init__()
        self.future: Future = Future()
        self.error: Optional[Exception] = None
        self.validations: list[tuple[str, Any, Optional[Exception]]] = []
        self.counts = (0, 0, 0)
        self.bindings: list[Any] = []
        self.releases: list[tuple[str, Optional[int]]] = []

    def configure_window_limits(
        self, max_jobs: int, max_bytes: int, max_futures: int
    ) -> None:
        self.limits = (max_jobs, max_bytes, max_futures)
        self.events.append("configure")

    def pending_jobs(self) -> int:
        return self.counts[0]

    def pending_bytes(self) -> int:
        return self.counts[1]

    def pending_futures(self) -> int:
        return self.counts[2]

    def window_stats(self) -> dict[str, int]:
        return dict(
            zip(("max_jobs", "max_bytes", "max_futures"), self.limits, strict=True)
        )

    def bind_step(
        self,
        requests: list,
        caches: dict,
        *,
        callbacks: Optional[tuple] = None,
        validation_error: Optional[Exception] = None,
    ) -> None:
        self.events.append("bind")
        self.bindings.append((requests, caches, callbacks, validation_error))
        if validation_error is not None:
            raise ValueError(f"bind ACK: {validation_error}")
        self.future = Future()

    def wait_for_load(
        self, metadata: Any, *, validation_error: Optional[Exception] = None
    ) -> None:
        self.validations.append(("wait", metadata, validation_error))
        if validation_error is not None:
            raise ValueError(f"entry ACK: {validation_error}")

    def submit_save(
        self,
        metadata: Any,
        kv_layer: Any,
        attn_metadata: Any = None,
        *,
        validation_error: Optional[Exception] = None,
    ) -> None:
        self.validations.append(("save", metadata, validation_error))
        self.error = self.error or validation_error
        self.events.append("submit-save")

    def submit_load(
        self, metadata: Any, *, validation_error: Optional[Exception] = None
    ) -> None:
        self.validations.append(("load", metadata, validation_error))
        self.error = self.error or validation_error
        self.events.append("submit-load")

    def finish_save(
        self, metadata: Any, *, validation_error: Optional[Exception] = None
    ) -> Future:
        self.validations.append(("finish", metadata, validation_error))
        self.events.append("finish-ACK")
        error = self.error or validation_error
        if error is not None:
            raise ValueError(f"source ACK: {error}")
        return self.future

    def finish_step(self) -> None:
        self.events.append("finish-step")
        self.future.set_result(None)

    def abort_step(self) -> None:
        self.events.append("abort-step")

    def abort_request(
        self, request_id: str, *, allocation_generation: Optional[int] = None
    ) -> None:
        self.releases.append((request_id, allocation_generation))


def _binding(request_id: str = "req-1", generation: int = 7) -> LayerwisePrefillRequest:
    return LayerwisePrefillRequest(
        request_id=request_id,
        allocation_generation=generation,
        token_ids=(1,),
        compute_start=0,
        compute_end=1,
        restore_end=0,
        block_ids_by_bank=(((1,), (2,)), ((3,), (4,))),
        block_size=1,
    )


@pytest.mark.parametrize("flag", [False, 1, "true", Mock()])
def test_managed_flag_requires_exact_true(flag: Any) -> None:
    backend = RecordingBackend()
    backend.manages_pending_work = flag
    coordinator = LayerwisePrefillWindowCoordinator(_topology_cache(), backend)
    assert coordinator.manages_pending_work is False
    assert coordinator.supports_transfer_window is True
    assert coordinator.pending_futures() == 0


@pytest.mark.parametrize(
    "hook",
    [
        "configure_window_limits",
        "pending_jobs",
        "pending_bytes",
        "pending_futures",
        "window_stats",
        "bind_step",
        "wait_for_load",
        "submit_save",
        "submit_load",
        "finish_save",
        "finish_step",
        "abort_step",
        "abort_request",
        "sync_save",
    ],
)
def test_managed_capability_rejects_missing_hooks(
    monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    backend = ManagedBackend()
    monkeypatch.setattr(backend, hook, None)
    with pytest.raises(ValueError, match=hook):
        LayerwisePrefillWindowCoordinator(_topology_cache(), backend)
    assert backend.events == []


def test_managed_capability_rejects_dynamic_placeholder_hooks() -> None:
    backend = Mock(
        manages_pending_work=True,
        accepts_coordinator_validation_errors=True,
        supports_transfer_window=True,
    )
    with pytest.raises(ValueError, match="configure_window_limits"):
        LayerwisePrefillWindowCoordinator(_topology_cache(), backend)


@pytest.mark.parametrize("flag", [False, 1, "true"])
def test_managed_capability_requires_collective_validation(flag: Any) -> None:
    backend = ManagedBackend()
    backend.accepts_coordinator_validation_errors = flag
    with pytest.raises(ValueError, match="collective validation"):
        LayerwisePrefillWindowCoordinator(_topology_cache(), backend)


@pytest.mark.parametrize("source", ["default", "env", "constructor"])
def test_managed_limits_configured_before_bind_and_counters_delegated(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    names = ("JOBS", "BYTES", "FUTURES")
    for name in names:
        monkeypatch.delenv(
            f"LMCACHE_LAYERWISE_PREFILL_MAX_PENDING_{name}", raising=False
        )
    limits = (8, 64 << 20, 8) if source == "default" else (3, 2048, 2)
    if source != "default":
        for name, value in zip(names, limits, strict=True):
            monkeypatch.setenv(
                f"LMCACHE_LAYERWISE_PREFILL_MAX_PENDING_{name}",
                str(value) if source == "env" else "invalid",
            )
    kwargs = (
        dict(
            zip(
                ("max_pending_jobs", "max_pending_bytes", "max_pending_futures"),
                limits,
                strict=True,
            )
        )
        if source == "constructor"
        else {}
    )
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(
        _topology_cache(), backend, **kwargs
    )
    assert backend.limits == limits
    assert (
        coordinator.max_pending_jobs,
        coordinator.max_pending_bytes,
        coordinator.max_pending_futures,
    ) == limits
    coordinator.bind_step([_binding()], {}, callbacks=())
    assert backend.events == ["configure", "bind"]
    backend.counts = (2, 1234, 1)
    assert (
        coordinator.pending_jobs(),
        coordinator.pending_bytes(),
        coordinator.pending_futures(),
    ) == backend.counts


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_future_limit_constructor_requires_positive_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LayerwisePrefillWindowCoordinator(
            _topology_cache(), ManagedBackend(), max_pending_futures=value
        )


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5", ""])
def test_future_limit_env_is_strict_and_managed_only(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LMCACHE_LAYERWISE_PREFILL_MAX_PENDING_FUTURES", value)
    with pytest.raises(ValueError, match="MAX_PENDING_FUTURES"):
        LayerwisePrefillWindowCoordinator(_topology_cache(), ManagedBackend())
    assert (
        LayerwisePrefillWindowCoordinator(
            _topology_cache(), RecordingBackend()
        ).max_pending_futures
        == 8
    )


def test_generic_pending_futures_deduplicates_shared_batch_and_rows() -> None:
    cache = _topology_cache()
    backend = RecordingBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    assert coordinator.pending_futures() == 0
    future = Future()
    for metadata in _callback(cache, 0, request_generations=(("a", 1), ("b", 2))):
        backend.persist_futures[metadata.row] = future
        coordinator.submit_save(metadata, _kv_layer())
        coordinator.finish_save(metadata)
    assert coordinator.pending_futures() == 1
    future.set_result(None)
    assert coordinator.pending_futures() == 0


def test_managed_101_source_rows_do_not_consume_assembly_future_credits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _topology_cache()
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    monkeypatch.setattr(
        backend.future,
        "result",
        Mock(
            side_effect=AssertionError("blocked on assembly before all rows executed")
        ),
    )
    _drive_full_request(coordinator, cache)
    assert backend.events.count("submit-save") == 101
    assert backend.events.count("submit-load") == 79
    assert backend.events.count("finish-ACK") == 101
    assert (
        coordinator.pending_jobs(),
        coordinator.pending_bytes(),
        coordinator.pending_futures(),
    ) == (0, 0, 0)
    assert not coordinator.request_persist_done("req-1")
    # Only protocol records remain; the backend owns all source resource credit.
    assert all(
        job.phase is LayerwisePrefillSavePhase.SOURCE_DONE and job.bytes == 0
        for job in coordinator._arenas["req-1"].jobs.values()
    )
    monkeypatch.undo()
    backend.finish_step()
    coordinator.wait_for_request_persist_done("req-1")
    assert coordinator.request_persist_done("req-1")


@pytest.mark.parametrize("phase", ["save", "load", "finish"])
@pytest.mark.parametrize("invalid", ["metadata", "generation", "row", "batch"])
def test_managed_validation_is_reported_by_post_hcom_ack(
    phase: str, invalid: str
) -> None:
    cache = _topology_cache()
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    metadata = _callback(cache, 0)[0]
    bad = {
        "metadata": None,
        "generation": replace(metadata, request_generations=(("req-1", 6),)),
        "row": SimpleNamespace(
            execution=metadata.execution,
            row=replace(metadata.row, bank=99),
            request_generations=metadata.request_generations,
        ),
        "batch": replace(metadata, request_generations=(("req-1", 7), ("new", 7))),
    }[invalid]
    coordinator.wait_for_load(metadata)
    coordinator.submit_save(bad if phase == "save" else metadata, _kv_layer())
    coordinator.submit_load(bad if phase == "load" else metadata)
    assert "finish-ACK" not in backend.events
    if phase == "save":
        assert coordinator._arenas["req-1"].jobs == {}
    with pytest.raises(ValueError, match="source ACK"):
        coordinator.finish_save(bad if phase == "finish" else metadata)
    validation = next(item for item in backend.validations if item[0] == phase)
    assert validation[1] is bad and isinstance(validation[2], Exception)
    assert backend.events[-1] == "finish-ACK"
    assert not coordinator.has_request("new")
    assert not coordinator.request_persist_done("req-1")


@pytest.mark.parametrize("failure", ["missing", "duplicate", "batch"])
def test_managed_finish_state_rejections_still_reach_backend(failure: str) -> None:
    cache = _topology_cache()
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    metadata = _callback(cache, 0, request_generations=(("req-1", 7), ("req-2", 7)))[0]
    coordinator.wait_for_load(metadata)
    if failure != "missing":
        coordinator.submit_save(metadata, _kv_layer())
    if failure == "duplicate":
        coordinator.finish_save(metadata)
    if failure == "batch":
        metadata = _callback(cache, 0)[0]
    with pytest.raises(ValueError, match="source ACK"):
        coordinator.finish_save(metadata)
    assert backend.validations[-1][0] == "finish"
    assert isinstance(backend.validations[-1][2], RuntimeError)


def test_managed_reused_callback_identity_is_checked_by_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _topology_cache()
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    metadata = _callback(cache, 0)[0]
    coordinator.wait_for_load(metadata)
    coordinator.submit_save(metadata, _kv_layer())
    old = replace(metadata)
    finish = Mock(side_effect=ValueError("callback object is not registered"))
    monkeypatch.setattr(backend, "finish_save", finish)
    with pytest.raises(ValueError, match="not registered"):
        coordinator.finish_save(old)
    finish.assert_called_once_with(old, validation_error=None)
    assert coordinator._arenas["req-1"].jobs[(0, 0)].phase is (
        LayerwisePrefillSavePhase.SAVE_SUBMITTED
    )


def test_managed_wait_reports_validation_at_row_entry() -> None:
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(_topology_cache(), backend)
    with pytest.raises(ValueError, match="entry ACK"):
        coordinator.wait_for_load(None)
    assert backend.validations[-1][0] == "wait"


def test_managed_unbound_chunk_rollover_never_waits_pre_hcom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _topology_cache((1, 1))
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    _drive_full_request(coordinator, cache)
    result = Mock(side_effect=AssertionError("waited on unfinished assembly"))
    monkeypatch.setattr(backend.future, "result", result)
    metadata = _callback(cache, 0)[0]
    coordinator.submit_save(metadata, _kv_layer())
    result.assert_not_called()
    assert isinstance(backend.validations[-1][2], RuntimeError)
    with pytest.raises(ValueError, match="source ACK.*per-group row order"):
        coordinator.finish_save(metadata)


def test_managed_release_keeps_arena_until_backend_generation_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _topology_cache()
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    coordinator.wait_for_load(_callback(cache, 0, generation=8)[0])
    abort = Mock(side_effect=RuntimeError("cannot release source yet"))
    monkeypatch.setattr(backend, "abort_request", abort)
    with pytest.raises(RuntimeError, match="cannot release"):
        coordinator.release_request("req-1")
    abort.assert_called_once_with("req-1", allocation_generation=8)
    assert coordinator.has_request("req-1")


@pytest.mark.parametrize("failure", ["none", "bind", "abort"])
def test_managed_abort_and_release_are_exact_generation_scoped(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cache = _topology_cache((1, 1))
    backend = ManagedBackend()
    coordinator = LayerwisePrefillWindowCoordinator(cache, backend)
    coordinator.bind_step([_binding()], {}, callbacks=())
    _drive_full_request(coordinator, cache)
    backend.finish_step()
    coordinator.wait_for_request_persist_done("req-1")
    if failure == "bind":
        with pytest.raises(ValueError, match="bind ACK"):
            coordinator.bind_step(
                [_binding("req-2", 8)], {}, callbacks=(),
                validation_error=ValueError("malformed next forward"),
            )
        coordinator.abort_step()
        assert coordinator.request_persist_done("req-1")
        assert not coordinator.has_request("req-2")
        return
    coordinator.bind_step([_binding("req-2", 8)], {}, callbacks=())
    metadata = _callback(cache, 0, request_generations=(("req-2", 8),))[0]
    coordinator.wait_for_load(metadata)
    coordinator.submit_save(metadata, _kv_layer())
    if failure == "abort":
        monkeypatch.setattr(
            backend, "abort_step", Mock(side_effect=RuntimeError("unsafe drain"))
        )
        with pytest.raises(RuntimeError, match="unsafe drain"):
            coordinator.abort_step()
        assert coordinator.has_request("req-2")
    else:
        coordinator.abort_step()
        assert not coordinator.has_request("req-2")
        with pytest.raises(ValueError, match="released generation"):
            coordinator.wait_for_load(metadata)
    assert coordinator.request_persist_done("req-1")
    coordinator.release_request("req-1")
    assert backend.releases == [("req-1", 7)]
    assert not coordinator.has_request("req-1")


def _start_managed_adapter(
    monkeypatch: pytest.MonkeyPatch, *, malformed: bool = False
) -> tuple[LMCacheConnectorV1Impl, ManagedBackend, tuple, dict]:
    backend = ManagedBackend()
    impl = _window_connector(monkeypatch, backend)
    impl._layerwise_prefill_p_node = True
    impl.kv_caches = {"registered": _kv_layer()}
    bindings = [_binding(), _binding("req-2", 8)]
    callbacks = tuple(
        metadata
        for execution in range(79)
        for metadata in _callback(
            impl._dsa_kv_topology_cache,
            execution,
            request_generations=(("req-2", 8), ("req-1", 7)),
        )
    )
    attention = {"unrelated": SimpleNamespace()}
    for metadata in callbacks:
        attention[metadata.row.layer_name] = SimpleNamespace(
            layerwise_prefill_callback_metadata=(metadata,)
        )
    attention["alias"] = attention[callbacks[-1].row.layer_name]
    if malformed:
        attention["bad"] = SimpleNamespace(layerwise_prefill_callback_metadata=1)
    connector_metadata = LMCacheConnectorMetadata(
        requests=[ReqMeta(req_id=req.request_id, token_ids=[1]) for req in bindings],
        layerwise_prefill_requests=bindings,
    )
    impl._parent = SimpleNamespace(_get_connector_metadata=lambda: connector_metadata)
    # These are worker-only fields; the startup helper constructs scheduler state.
    impl._layerwise_save_storers = {}
    impl._deferred_latent_pending = set()
    impl._decode_window_save_completed_groups = set()
    impl._prefill_save_completed_groups = {}
    impl._completed_decode_window_saves = {}
    impl._finished_req_ids_waiting_for_save = set()
    impl._late_finished_sending = set()
    impl._cold_perf_dense_load_started = {}
    impl._cold_perf_dense_load_completed = {}
    impl._cold_perf_load_started = {}
    monkeypatch.setattr(impl, "_drop_worker_retrieve_state", Mock())
    return impl, backend, callbacks, attention


@pytest.mark.parametrize("malformed", [False, True])
def test_adapter_binds_all_actual_callbacks_and_forwards_initial_errors(
    monkeypatch: pytest.MonkeyPatch, malformed: bool
) -> None:
    impl, backend, callbacks, attention = _start_managed_adapter(
        monkeypatch, malformed=malformed
    )
    context = SimpleNamespace(attn_metadata=attention)
    if malformed:
        with pytest.raises(ValueError, match="bind ACK"):
            impl.start_load_kv(context)
        assert backend.events.count("abort-step") == 1
        assert isinstance(backend.bindings[-1][-1], TypeError)
    else:
        impl.start_load_kv(context)
        bindings, caches, bound, error = backend.bindings[-1]
        assert [req.request_id for req in bindings] == ["req-2", "req-1"]
        assert caches is impl.kv_caches
        assert error is None
        assert len(bound) == 102  # Identical-object aliases are backend-validated.
        assert all(
            actual is expected
            for actual, expected in zip(bound[:-1], callbacks, strict=True)
        )
        assert bound[-1] is callbacks[-1]
        assert backend.validations == []  # No row entry or transfer during bind.


@pytest.mark.parametrize("failure", [False, True])
def test_adapter_abort_is_once_and_never_reports_completion(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    impl, backend, callbacks, attention = _start_managed_adapter(monkeypatch)
    impl.start_load_kv(SimpleNamespace(attn_metadata=attention))
    impl.wait_for_layerwise_prefill_load(callbacks[0])
    impl.submit_layerwise_prefill_save(callbacks[0], _kv_layer())
    if failure:
        monkeypatch.setattr(
            backend,
            "abort_step",
            Mock(side_effect=RuntimeError("unknown device fence")),
        )
        with pytest.raises(RuntimeError, match="unknown device fence"):
            impl.abort_layerwise_prefill_step()
    else:
        impl.abort_layerwise_prefill_step()
    impl._abort_save_step(())
    impl._abort_layerwise_retrieve_step(())
    impl.abort_layerwise_prefill_step()
    if failure:
        backend.abort_step.assert_called_once_with()
        assert impl._layerwise_prefill_window.has_request("req-1")
    else:
        assert backend.events.count("abort-step") == 1
        assert not impl._layerwise_prefill_window.has_request("req-1")
    assert backend.releases == []
    assert not impl.layerwise_prefill_request_persist_done("req-1")
    assert impl.get_completed_decode_window_saves() == {}


@pytest.mark.parametrize(
    "managed,p_node,active",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_adapter_abort_is_noop_without_active_managed_p_step(
    monkeypatch: pytest.MonkeyPatch, managed: bool, p_node: bool, active: bool
) -> None:
    backend = ManagedBackend() if managed else RecordingBackend()
    impl = _window_connector(monkeypatch, backend)
    impl._layerwise_prefill_p_node = p_node
    impl._layerwise_prefill_step_active = active
    events = backend.events.copy()
    impl.abort_layerwise_prefill_step()
    assert backend.events == events
    assert backend.aborted == []


def test_dynamic_connector_delegates_step_abort() -> None:
    connector = object.__new__(LMCacheConnectorV1Dynamic)
    connector._lmcache_engine = Mock()
    connector.abort_layerwise_prefill_step()
    connector._lmcache_engine.abort_layerwise_prefill_step.assert_called_once_with()
