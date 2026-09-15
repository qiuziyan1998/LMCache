# SPDX-License-Identifier: Apache-2.0
"""All-layer pages must produce one-layer views on the individual-handle path."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import (
    MemoryFormat,
    TensorMemoryAllocator,
)
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedChunkHandle,
    SharedCPUCacheValidationError,
    SharedHandleEnvelope,
)


@pytest.mark.parametrize(
    "layers,tokens,width,group",
    [(79, 1024, 128, 1), (3, 7, 128, 1), (3, 7, 576, 0), (1, 7, 128, 1)],
)
def test_layer_page_individual_handles_preserve_layer_bytes(
    layers: int, tokens: int, width: int, group: int
) -> None:
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_DSA_INDEX_FMT if group else MemoryFormat.KV_MLA_LATENT_FMT
    shape = torch.Size([tokens * width])
    slab = torch.empty(
        2 * (layers * shape.numel() * dtype.itemsize + 4096), dtype=torch.uint8
    )
    owner = TensorMemoryAllocator(slab)
    pages = owner.batched_allocate_layer_pages(
        [shape], [dtype], 2, layers, fmt, valid_tokens=tokens, full_tokens=tokens
    )
    assert pages is not None
    page = pages[1]  # A nonzero slab offset also detects missing layer offsets.
    for layer in range(layers):
        page.layer_tensor(layer).fill_(layer + 1)
    passive = PassiveSharedViewAllocator(
        slab_tensor=slab, shm_name="test", generation=9
    )
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "test"
    engine.shared_cpu_cache_generation = 9
    engine.metadata = SimpleNamespace(worker_id=0)
    try:
        for req in ("a", "b", "c", "d"):
            for layer in sorted({0, min(4, layers - 1), layers - 1}):
                key = CacheEngineKey(
                    "model", 1, 0, 7, dtype, kv_group=group
                ).split_layers(layers)[layer]
                handles = engine._make_shared_handles_for_layer(
                    req_id=req,
                    phase="sparse_decode_bootstrap",
                    keys_layer=[key],
                    mem_objs_layer=[page],
                    layer_id=layer,
                    kv_group=group,
                    validate_memory_objs=False,
                )
                envelope = SharedHandleEnvelope(
                    req,
                    "sparse_decode_bootstrap",
                    0,
                    layer,
                    group,
                    "ok",
                    9,
                    handles=handles,
                )
                handle = SharedHandleEnvelope.from_dict(envelope.to_dict()).handles[0]
                view = passive.create_view(
                    handle,
                    expected_request_id=req,
                    expected_phase="sparse_decode_bootstrap",
                    expected_layer_id=layer,
                    expected_kv_group=group,
                    expected_chunk_index=0,
                    expected_key=key,
                    expected_shape=shape,
                    expected_dtype=dtype,
                    expected_fmt=fmt,
                    expected_cached_positions=range(tokens),
                    expected_producer_rank=0,
                )
                try:
                    tensor = view.tensor
                    assert tensor is not None
                    assert tensor.shape == shape
                    assert tensor.data_ptr() == page.layer_data_ptr(layer)
                    assert torch.all(tensor == layer + 1)
                    assert handle.logical_size == page.layer_size
                    assert handle.physical_size >= handle.logical_size
                    assert handle.shapes == [shape] and handle.dtypes == [dtype]
                finally:
                    view.ref_count_down()
                assert page.is_valid() and page.get_ref_count() == 1
    finally:
        for page in pages:
            page.ref_count_down()
    assert owner.total_allocated_size == 0


def test_mixed_legacy_and_page_cache_uses_safe_individual_handles() -> None:
    layers, tokens = 3, 7
    shape, dtype, fmt = (
        torch.Size([tokens * 128]),
        torch.bfloat16,
        MemoryFormat.KV_DSA_INDEX_FMT,
    )
    slab = torch.empty(1 << 18, dtype=torch.uint8)
    owner = TensorMemoryAllocator(slab)
    legacy = owner.batched_allocate(shape, dtype, layers, fmt)
    pages = owner.batched_allocate_layer_pages(
        [shape], [dtype], 1, layers, fmt, valid_tokens=tokens, full_tokens=tokens
    )
    assert legacy is not None and pages is not None
    page = pages[0]
    keys = [
        CacheEngineKey("model", 1, 0, h, dtype, kv_group=1).split_layers(layers)
        for h in (11, 22)
    ]
    rows = [[legacy[layer], page] for layer in range(layers)]
    layer_keys = [[chunk[layer] for chunk in keys] for layer in range(layers)]
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name, engine.num_layers = "mixed", layers
    engine.shared_cpu_cache_generation = 3
    engine.metadata = SimpleNamespace(worker_id=0)
    passive = PassiveSharedViewAllocator(
        slab_tensor=slab, shm_name="mixed", generation=3
    )
    try:
        # A merged page after a legacy chunk cannot use the compact page-prefix format.
        assert engine._make_shared_handle_batch(rows, layer_keys) is None
        for layer in range(layers):
            legacy[layer].tensor.fill_(layer + 10)
            page.layer_tensor(layer).fill_(layer + 20)
            handles = engine._make_shared_handles_for_layer(
                req_id="mixed",
                phase="sparse_decode_bootstrap",
                keys_layer=layer_keys[layer],
                mem_objs_layer=rows[layer],
                layer_id=layer,
                kv_group=1,
                validate_memory_objs=False,
            )
            for chunk, handle in enumerate(handles):
                view = passive.create_view(
                    handle,
                    expected_request_id="mixed",
                    expected_phase="sparse_decode_bootstrap",
                    expected_layer_id=layer,
                    expected_kv_group=1,
                    expected_chunk_index=chunk,
                    expected_key=layer_keys[layer][chunk],
                    expected_shape=shape,
                    expected_dtype=dtype,
                    expected_fmt=fmt,
                )
                try:
                    assert torch.all(view.tensor == layer + 10 * (chunk + 1))
                    if chunk == 0:
                        assert handle.offset == legacy[layer].metadata.address
                        assert handle.physical_size == legacy[layer].metadata.phy_size
                finally:
                    view.ref_count_down()
    finally:
        for obj in [*legacy, page]:
            obj.ref_count_down()
    assert owner.total_allocated_size == 0


@pytest.mark.parametrize("layer", [-1, 3])
def test_page_handle_rejects_invalid_layer_without_materializing_storage(
    layer: int,
) -> None:
    shape, dtype = torch.Size([16]), torch.float16
    owner = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    pages = owner.batched_allocate_layer_pages(
        [shape],
        [dtype],
        1,
        3,
        MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=1,
        full_tokens=1,
    )
    assert pages is not None
    page = pages[0]
    try:
        assert "raw_data" not in page.__dict__
        with pytest.raises(SharedCPUCacheValidationError, match="handle row"):
            SharedChunkHandle.from_memory_obj(
                request_id="r",
                phase="sparse_decode_bootstrap",
                key=CacheEngineKey("model", 1, 0, 1, dtype, kv_group=1),
                layer_id=layer,
                kv_group=1,
                chunk_index=0,
                shm_name="test",
                memory_obj=page,
                generation=1,
                producer_rank=0,
            )
        assert "raw_data" not in page.__dict__
    finally:
        page.ref_count_down()
