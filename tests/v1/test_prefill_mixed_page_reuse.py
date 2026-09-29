# SPDX-License-Identifier: Apache-2.0
"""Mixed packets keep exact layer offsets when passive P views are reused."""

# Standard
from dataclasses import replace
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedCPUCacheValidationError,
    SharedHandleBatch,
    SharedHandleEnvelope,
)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("tokens", [1, 127, 443, 1024])
def test_mixed_page_offsets_and_reuse(reuse: bool, tokens: int) -> None:
    policy = IndexerC8Layout(128, (True, False, True))
    shapes = [torch.Size([policy.layer_bytes(tokens, i)]) for i in range(3)]
    size = sum(s.numel() for s in shapes)
    slab = torch.zeros(size * 2, dtype=torch.uint8)
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab, shm_name="mixed", generation=3, reuse_prefill=reuse
    )
    batch = SharedHandleBatch(
        shm_name="mixed",
        producer_rank=0,
        num_layers=3,
        num_chunks=1,
        physical_sizes=[shapes[0].numel()],
        chunk_hashes=[101],
        offsets=[],
        page_offsets=[size],
        page_physical_sizes=[size],
    )
    args = dict(
        chunk_index=0,
        shape=shapes[0],
        dtype=torch.uint8,
        fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        cached_positions=range(tokens),
        layer_shapes=shapes,
    )
    page = allocator.create_page_view(batch, **args)
    try:
        cursor = size
        for layer, shape in enumerate(shapes):
            tensor = page.layer_tensor(layer)
            assert tensor.shape == shape
            assert page.layer_size_bytes(layer) == tensor.numel() * tensor.element_size()
            assert tensor.data_ptr() == slab.data_ptr() + cursor
            tensor.fill_(layer + 1)
            cursor += shape.numel()
        again = allocator.create_page_view(batch, previous=page, **args)
        assert (again is page) is reuse
        for layer in range(3):
            assert bool((again.layer_tensor(layer) == layer + 1).all())
        assert page.metadata.ref_count == 1
        if again is not page:
            again.ref_count_down()
        # Identical total byte count but different layer boundaries is not
        # an interchangeable cached view.
        changed = allocator.create_page_view(
            batch,
            previous=page,
            **{**args, "layer_shapes": [shapes[0], shapes[2], shapes[1]]},
        )
        assert changed is not page
        assert (
            changed.layer_tensor(2).data_ptr()
            == slab.data_ptr() + size + 2 * shapes[0].numel()
        )
        changed.ref_count_down()
        with pytest.raises(SharedCPUCacheValidationError):
            allocator.create_page_view(
                replace(batch, page_physical_sizes=[size - 1]), previous=page, **args
            )
    finally:
        page.ref_count_down()


def test_engine_forwards_mixed_shapes_and_previous_page_together() -> None:
    policy = IndexerC8Layout(128, (True, False))
    shapes = [torch.Size([policy.layer_bytes(3, i)]) for i in range(2)]
    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.zeros(4096, dtype=torch.uint8),
        shm_name="mixed",
        generation=1,
        reuse_prefill=True,
    )
    batch = SharedHandleBatch(
        shm_name="mixed",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[shapes[0].numel()],
        chunk_hashes=[101],
        offsets=[],
        page_offsets=[0],
        page_physical_sizes=[sum(s.numel() for s in shapes)],
    )
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_passive_allocator = allocator
    engine.metadata = SimpleNamespace(
        worker_id=1,
        first_rank=0,
        indexer_c8_layout=policy,
        indexer_layer_shapes=lambda tokens: shapes,
    )
    engine.num_layers_for_group = lambda group: 2
    engine._expected_shared_cpu_chunk_metadata = lambda **kw: (
        shapes[0],
        torch.uint8,
        MemoryFormat.KV_DSA_INDEX_FMT,
    )
    keys = [
        [key]
        for key in CacheEngineKey(
            "model", 4, 0, 101, torch.uint8, kv_group=1
        ).split_layers(2)
    ]
    args = dict(starts=[0], ends=[3], keys_layer_major=keys, kv_group=1)
    (first,) = engine._make_passive_layer_page_views(batch, **args)
    try:
        (reused,) = engine._make_passive_layer_page_views(
            batch,
            **args,
            reuse_kwargs={
                "_reuse_shared_dense_prefix": True,
                "cached_starts": [0],
                "cached_ends": [3],
                "cached_keys": keys,
                "cached_memory_objs": [[first], [first]],
            },
        )
        assert reused is first
        assert reused.layer_tensor(1).numel() == 3 * 256
        assert reused.metadata.ref_count == 1
    finally:
        first.ref_count_down()


def test_passive_c8_prepares_all_sources_before_first_yield(monkeypatch) -> None:
    from lmcache.v1 import cache_engine as engine_module
    from tests.v1.test_shared_prefill_reuse_integration import HostDMAConsumer

    monkeypatch.setattr(engine_module, "assert_layerwise_gpu_connector", lambda _: None)
    policy = IndexerC8Layout(128, (True, False))
    shapes = [torch.Size([policy.layer_bytes(3, i)]) for i in range(2)]
    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.zeros(4096, dtype=torch.uint8),
        shm_name="mixed",
        generation=1,
        reuse_prefill=True,
    )
    order = []

    class Consumer(HostDMAConsumer):
        def prepare_layerwise_prefill_source_pointers(self, *args, **kwargs):
            order.append("sources")
            return super().prepare_layerwise_prefill_source_pointers(*args, **kwargs)

        def batched_to_gpu(self, *args, **kwargs):
            assert order == ["sources"]
            rows = kwargs["prefill_c8_memory_objs"]
            assert len(rows) == 2 and all(len(row) == 1 for row in rows)
            assert [rows[i][0].layer_tensor(i).numel() for i in range(2)] == [390, 768]
            order.append("consumer")
            yield from super().batched_to_gpu(*args, **kwargs)

    engine = object.__new__(LMCacheEngine)
    engine.config = SimpleNamespace()
    engine.num_layers = 2
    engine.gpu_connector = Consumer()
    engine.shared_cpu_cache_generation = 1
    engine.shared_cpu_cache_passive_allocator = allocator
    engine.metadata = SimpleNamespace(
        worker_id=1,
        first_rank=0,
        is_first_rank=lambda: False,
        indexer_c8_layout=policy,
        indexer_layer_shapes=lambda tokens: shapes,
    )
    engine._shared_cpu_request_leases = {}
    engine.stats_monitor = SimpleNamespace(on_retrieve_finished=lambda *_: None)
    engine._expected_shared_cpu_chunk_metadata = lambda **kw: (
        shapes[kw.get("layer_id", 0)],
        torch.uint8,
        MemoryFormat.KV_DSA_INDEX_FMT,
    )
    batch = SharedHandleBatch(
        shm_name="mixed",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[390],
        chunk_hashes=[101],
        offsets=[],
        page_offsets=[0],
        page_physical_sizes=[1158],
    )
    envelope = SharedHandleEnvelope(
        request_id="r",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=1,
        status="ok",
        generation=1,
        handles=[],
        batch=batch,
    )
    engine._receive_shared_envelope = lambda: envelope
    keys = [
        [key]
        for key in CacheEngineKey(
            "model", 4, 0, 101, torch.uint8, kv_group=1
        ).split_layers(2)
    ]
    generator = engine._retrieve_layer_shared_passive(
        starts_all=[0],
        ends_all=[3],
        keys_layer_major=keys,
        ret_mask=torch.zeros(3, dtype=torch.bool),
        monitor_req_id=0,
        req_id="r",
        kv_group=1,
        kwargs={
            "deferred_layerwise_get": True,
            "prefill_dma_block_ids_by_bank": ((1,), (2,)),
        },
    )
    try:
        assert next(generator).item() == 3
        assert order == ["sources", "consumer"]
        assert not engine.gpu_connector.submissions
        result = list(generator)
        assert len(engine.gpu_connector.submissions) == 2
        assert bool(result[-1].all())
    finally:
        generator.close()


def test_promoted_mixed_flat_sources_use_each_layer_shape() -> None:
    policy = IndexerC8Layout(128, (True, False))
    allocator = TensorMemoryAllocator(torch.empty(16384, dtype=torch.uint8))
    objects = [
        allocator.allocate(
            torch.Size([policy.layer_bytes(3, layer)]),
            torch.uint8,
            fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        )
        for layer in range(2)
    ]
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(indexer_c8_layout=policy)
    engine._memory_format_for_kv_group = lambda group: MemoryFormat.KV_DSA_INDEX_FMT
    engine.get_shared_cpu_request_lease = lambda req: SimpleNamespace(
        owns=lambda obj: obj in objects
    )
    engine._shared_prefill_source_state = lambda *args: None
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None
    try:
        rows, _, owned, _ = engine._resolve_shared_rank0_prefill_suffix(
            req_id="r",
            phase="dense_prefix",
            kv_group=1,
            starts=[0],
            ends=[3],
            keys_layer_major=[["c8"], ["bf16"]],
            chunk_locations_layer_major=[[], []],
            planned_page_chunks=0,
            page_first=False,
            prefix=1,
            kwargs={"cached_memory_objs": [[obj] for obj in objects]},
        )
        assert rows == [[obj] for obj in objects] and not owned
    finally:
        for obj in objects:
            obj.ref_count_down()
