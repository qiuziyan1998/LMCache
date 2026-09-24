# SPDX-License-Identifier: Apache-2.0
"""Mixed physical indexer layers keep exact, unpadded byte ranges."""

import pytest
import torch

from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator


def test_glm53_policy_has_six_unquantized_and_sixteen_c8_owners():
    policy = IndexerC8Layout(c8_layers=(False,) * 3 + (True,) * 16 + (False,) * 3)
    assert policy.mixed
    assert sum(policy.token_bytes_for(i) for i in range(22)) == 3616
    assert policy.plane_ranges(7, 0) == ((0, 7 * 256),)
    assert policy.plane_ranges(7, 3) == ((0, 7 * 128), (7 * 128, 7 * 2))
    with pytest.raises(ValueError, match="physical layer ID"):
        policy.layer_bytes(7)
    assert policy.descriptor() != IndexerC8Layout().descriptor()


def test_mixed_pages_round_trip_tails_without_padding_or_layer_overlap():
    policy = IndexerC8Layout(c8_layers=(False, True, False, True))
    counts = (1, 7, 129, 1024)
    allocator = TensorMemoryAllocator(torch.zeros(2**21, dtype=torch.uint8))
    shapes = [torch.Size([policy.layer_bytes(1024, i)]) for i in range(4)]
    pages = allocator.batched_allocate_layer_pages(
        shapes, [torch.uint8] * 4, len(counts), 4,
        MemoryFormat.KV_DSA_INDEX_FMT, valid_tokens=list(counts), full_tokens=1024,
    )
    assert pages is not None
    for page, count in zip(pages, counts, strict=True):
        assert page.valid_tokens == count
        assert page.layer_size is None
        offset = 0
        expected = []
        for layer in range(4):
            view = page.layer_tensor(layer)
            size = policy.layer_bytes(count, layer)
            assert view.numel() == size == page.layer_size_bytes(layer)
            assert view.data_ptr() == page.layer_data_ptr(layer) == page.data_ptr + offset
            if policy.is_c8(layer):
                bits = torch.arange(count * 128).to(torch.int8).view(torch.uint8)
                scales = (torch.arange(count, dtype=torch.int32) * 97).to(torch.int16).view(torch.uint8)
                payload = torch.cat((bits, scales))
            else:
                payload = torch.full((count * 128,), layer + 1, dtype=torch.bfloat16).view(torch.uint8)
            view.copy_(payload)
            expected.append(payload)
            offset += size
        assert page.get_size() == offset == count * (256 + 130) * 2
        for layer, payload in enumerate(expected):
            assert torch.equal(page.layer_tensor(layer), payload)
    allocator.batched_free(pages)
    assert allocator.memcheck()


def test_mixed_page_allocation_failure_releases_earlier_tail_batches():
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        [torch.Size([256 * 1024]), torch.Size([130 * 1024])],
        [torch.uint8, torch.uint8], 2, 2, MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=[1, 1024], full_tokens=1024,
    )
    assert pages is None
    assert allocator.memcheck()
