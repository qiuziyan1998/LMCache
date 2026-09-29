# SPDX-License-Identifier: Apache-2.0
"""Exercise consecutive passive prefill loads with real shared slab views."""

# Standard
from collections import deque
from types import SimpleNamespace
from typing import Any, Generator

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1 import cache_engine as engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import LayerPageSource, MemoryFormat
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedHandleBatch,
    SharedHandleEnvelope,
)


class HostDMAConsumer:
    """Record source preparation and DMA submissions without device execution."""

    def __init__(self) -> None:
        self.append_counts: list[list[int]] = []
        self.submissions: list[Any] = []

    def supports_dense_sparse_cache_retention(self) -> bool:
        return True

    def prepare_layerwise_prefill_source_pointers(
        self, sources: list, host_rows: list, device_rows: list, **kwargs: Any
    ) -> None:
        assert kwargs["prefill_dma"]
        self.append_counts.append(
            [len(source.pages) + len(source.suffix) for source in sources]
        )
        if not host_rows:
            host_rows.extend([] for _ in sources)
        for layer, source in enumerate(sources):
            assert isinstance(source, LayerPageSource)
            host_rows[layer].extend(page.layer_data_ptr(layer) for page in source.pages)
            host_rows[layer].extend(obj.data_ptr for obj in source.suffix)
        device_rows[:] = [None] * len(sources)

    def batched_to_gpu(self, *args: Any, **kwargs: Any) -> Generator:
        while True:
            source = yield
            if source is not None:
                self.submissions.append(source)

    def synchronize_dense_load_stream(self) -> None:
        pytest.fail("A successful retained prefill load must not synchronize")


@pytest.mark.parametrize("kv_group,layers", [(0, 3), (1, 2)])
def test_consecutive_chunks_reuse_views_and_release_only_at_request_end(
    monkeypatch: pytest.MonkeyPatch, kv_group: int, layers: int
) -> None:
    # The real generator, descriptor validation, allocator and request lease run;
    # only the collective transport and device consumer are replaced.
    monkeypatch.setattr(engine_module, "assert_layerwise_gpu_connector", lambda _: None)
    consumer = HostDMAConsumer()
    allocator = PassiveSharedViewAllocator(
        reuse_prefill=True,
        slab_tensor=torch.zeros(1024, dtype=torch.uint8),
        shm_name="/prefill-test",
        generation=7,
    )
    created: list[Any] = []
    for name in ("create_page_view", "create_batch_view"):
        original = getattr(allocator, name)

        def track(*args: Any, factory=original, **kwargs: Any) -> Any:
            view = factory(*args, **kwargs)
            if view is not kwargs.get("previous"):
                created.append(view)
            return view

        monkeypatch.setattr(allocator, name, track)

    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace()
    engine.num_layers = layers
    engine.gpu_connector = consumer
    engine.shared_cpu_cache_generation = 7
    engine.shared_cpu_cache_passive_allocator = allocator
    engine.metadata = SimpleNamespace(
        first_rank=0, worker_id=1, is_first_rank=lambda: False
    )
    engine.stats_monitor = SimpleNamespace(on_retrieve_finished=lambda *_: None)
    engine._shared_cpu_request_leases = {}
    engine._expected_shared_cpu_chunk_metadata = lambda **kwargs: (
        torch.Size([kwargs["num_tokens"]]),
        torch.float16,
        MemoryFormat.KV_MLA_LATENT_FMT,
    )
    incoming: deque = deque()
    engine._receive_shared_envelope = incoming.popleft
    caches = {
        name: []
        for name in (
            "cached_keys",
            "cached_starts",
            "cached_ends",
            "cached_memory_objs",
            "cached_chunk_dev_ptrs",
            "cached_chunk_ptrs_npu",
            "cached_shared_handles",
        )
    }
    kwargs = dict(
        caches,
        _retain_shared_dense_cache=True,
        _reuse_shared_dense_prefix=True,
        deferred_layerwise_get=True,
        prefill_dma_block_ids_by_bank=((0,), (1,)),
    )
    initial_rows = None
    for step, total in enumerate((6, 8, 10, 10)):
        kwargs["layerwise_prefill_bank_offset"] = step & 1
        starts = list(range(0, total, 4))
        ends = [min(start + 4, total) for start in starts]
        chunks = len(starts)
        hashes = [100 + index for index in range(chunks)]
        batch = SharedHandleBatch(
            shm_name=allocator.shm_name,
            producer_rank=0,
            num_layers=layers,
            num_chunks=chunks,
            physical_sizes=[8] + [16] * (chunks - 1),
            chunk_hashes=hashes,
            offsets=[
                128 + layer * 128 + chunk * 16
                for layer in range(layers)
                for chunk in range(chunks - 1)
            ],
            page_offsets=[0],
            page_physical_sizes=[8 * layers],
        )
        incoming.append(
            SharedHandleEnvelope(
                request_id="r",
                phase="dense_prefix",
                request_ordinal=0,
                layer_id=0,
                kv_group=kv_group,
                status="ok",
                generation=7,
                handles=[],
                batch=batch,
            )
        )
        keys = [
            CacheEngineKey(
                model_name="model",
                world_size=8,
                worker_id=0,
                chunk_hash=value,
                dtype=torch.float16,
                kv_group=kv_group,
            ).split_layers(layers)
            for value in hashes
        ]
        results = list(
            engine._retrieve_layer_shared_passive(
                starts_all=starts,
                ends_all=ends,
                keys_layer_major=[list(row) for row in zip(*keys, strict=True)],
                ret_mask=torch.zeros(total, dtype=torch.bool),
                monitor_req_id=step,
                req_id="r",
                kv_group=kv_group,
                kwargs=kwargs,
            )
        )
        assert results[-1].tolist() == [True] * total
        assert not incoming
        assert all(obj.is_valid() and obj.get_ref_count() == 1 for obj in created)
        assert caches["cached_chunk_ptrs_npu"] == [None] * layers
        rows = caches["cached_memory_objs"]
        assert all(len(row) == chunks for row in rows)
        if initial_rows is None:
            initial_rows = [list(row) for row in rows]
        else:
            assert all(
                rows[layer][0] is initial_rows[layer][0] for layer in range(layers)
            )
            assert all(
                rows[layer][1] is not initial_rows[layer][1] for layer in range(layers)
            )
        assert len(created) == (1 + layers * min(step + 1, 3))

    assert consumer.append_counts == [
        [2] * layers,
        [1] * layers,
        [1] * layers,
        [0] * layers,
    ]
    assert len(consumer.submissions) == layers * 4
    engine.release_shared_cpu_sparse_request("r")
    assert all(not obj.is_valid() for obj in created)
