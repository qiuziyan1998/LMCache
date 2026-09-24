# SPDX-License-Identifier: Apache-2.0
"""Request-owned source descriptors for incremental shared prefill retrieval."""

# Standard
from dataclasses import dataclass
from typing import Any, Sequence

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.shared_cpu_cache import SharedHandleBatch


def same_chunk_key(left: CacheEngineKey, right: CacheEngineKey) -> bool:
    """Compare base identities without allocating stripped or per-layer keys."""
    return (
        left.model_name == right.model_name
        and left.world_size == right.world_size
        and left.worker_id == right.worker_id
        and left.chunk_hash == right.chunk_hash
        and left.dtype == right.dtype
        and left.tags == right.tags
        and left.kv_group == right.kv_group
    )


@dataclass
class SharedPrefillSources:
    """Descriptor proof whose sources remain owned by its enclosing lease."""

    context: tuple[Any, ...]
    batch: SharedHandleBatch
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    keys: tuple[CacheEngineKey, ...]
    valid_chunks: int

    def matching_prefix(
        self,
        starts: Sequence[int],
        ends: Sequence[int],
        keys: Sequence[CacheEngineKey],
        limit: int,
    ) -> int:
        """Check scalar request identities, without touching historical views."""
        limit = min(limit, self.valid_chunks, len(self.starts), len(starts))
        for index in range(limit):
            if (
                self.starts[index] != starts[index]
                or self.ends[index] != ends[index]
                or not same_chunk_key(self.keys[index], keys[index])
            ):
                return index
        return limit

    def wire_prefix(self, batch: SharedHandleBatch, limit: int) -> int:
        """Match the existing full wire protocol using only scalar descriptors.

        View shape/dtype/format and ranges were validated when the old batch was
        installed. This check binds that proof to the current slab offsets and
        hashes; callers still validate the new envelope and slab bounds.
        """
        old = self.batch
        if (
            old.shm_name != batch.shm_name
            or old.producer_rank != batch.producer_rank
            or old.num_layers != batch.num_layers
        ):
            return 0
        old_pages, pages = len(old.page_offsets), len(batch.page_offsets)
        old_tail, tail = old.num_chunks - old_pages, batch.num_chunks - pages
        limit = min(limit, old.num_chunks, batch.num_chunks)
        for index in range(limit):
            if (
                old.physical_sizes[index] != batch.physical_sizes[index]
                or old.chunk_hashes[index] != batch.chunk_hashes[index]
                or (index < old_pages) != (index < pages)
            ):
                limit = index
                break
            if index < pages and (
                old.page_offsets[index] != batch.page_offsets[index]
                or old.page_physical_sizes[index] != batch.page_physical_sizes[index]
            ):
                limit = index
                break
        # Tail offsets remain layer-major on the wire. Slice comparisons avoid
        # calling MemoryObj metadata accessors for every historical layer.
        for layer in range(batch.num_layers):
            begin = min(pages, limit)
            old_offsets = old.offsets[
                layer * old_tail : layer * old_tail + limit - begin
            ]
            offsets = batch.offsets[layer * tail : layer * tail + limit - begin]
            if old_offsets != offsets:
                for index, (left, right) in enumerate(
                    zip(old_offsets, offsets, strict=True)
                ):
                    if left != right:
                        limit = min(limit, begin + index)
                        break
        return limit
