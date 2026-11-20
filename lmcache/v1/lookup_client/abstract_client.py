# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Optional, Union
import abc

# Third Party
import torch


LookupChunkHashesEntry = tuple[
    tuple[Any, ...],
    tuple[int, ...],
    int,
    Optional[int],
    Optional[int],
    tuple[int, ...],
]


class LookupClientInterface(metaclass=abc.ABCMeta):
    """Abstract interface for lookup clients."""

    _lookup_chunk_hashes: Optional[dict[str, LookupChunkHashesEntry]] = None

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        Lookup the cache for the given lookup ID.

        Args:
            lookup_id: The lookup ID to lookup

        Returns:
            -1 means not found;
            None means ongoing; (this semantic is not supported in sync lookup clients)
            int >= 0 means number of hit tokens
        """
        return None

    @abc.abstractmethod
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Perform lookup for the given token IDs.
        Should be called for first lookup and pinning. Subsequent lookups for the same
        request should call lookup_cache instead.

        Caller should handle overlaps between tokens that exist in LMCache
        and tokens that are already computed by the caller.

        Args:
            token_ids: The token IDs to lookup

            lookup_id: The lookup ID to associate with the lookup

            request_configs: The configs of the request,
            includes tags and the other configs

        Returns:
            The number of tokens that exist inside LMCache.
            None indicates the lookup/prefetch is in progress.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the lookup client and clean up resources."""
        raise NotImplementedError

    def supports_producer_reuse(self) -> bool:
        """
        Return whether this lookup client supports producer KV cache reuse.

        Returns:
            True if producer reuse is supported, False otherwise
        """
        return False

    def clear_lookup_status(self, lookup_id: str) -> None:
        """
        Clear temporary lookup status for a given lookup ID.

        Args:
            lookup_id: The lookup ID whose status needs to be cleared.
        """
        return

    def _stash_lookup_chunk_hashes(
        self,
        lookup_id: Optional[str],
        token_ids: Union[torch.Tensor, list[int]],
        hashes: Sequence[Any],
        offsets: Sequence[int],
        chunk_end_tokens: Sequence[int],
    ) -> None:
        if not lookup_id:
            return
        lock = getattr(self, "lock", None)
        with lock if lock is not None else nullcontext():
            store: Optional[dict[str, LookupChunkHashesEntry]] = getattr(
                self, "_lookup_chunk_hashes", None
            )
            if store is None:
                store = {}
                self._lookup_chunk_hashes = store
            token_len = len(token_ids)
            if not hashes or token_len <= 0:
                store.pop(lookup_id, None)
                return
            if lookup_id not in store and len(store) >= 4096:
                store.pop(next(iter(store)), None)
            store[lookup_id] = (
                tuple(hashes),
                tuple(offsets),
                token_len,
                int(token_ids[0]),
                int(token_ids[-1]),
                tuple(chunk_end_tokens),
            )

    def _drop_lookup_chunk_hashes(self, lookup_id: Optional[str]) -> None:
        if not lookup_id:
            return
        lock = getattr(self, "lock", None)
        with lock if lock is not None else nullcontext():
            store = getattr(self, "_lookup_chunk_hashes", None)
            if store is not None:
                store.pop(lookup_id, None)

    def pop_lookup_chunk_hashes(
        self, lookup_id: str
    ) -> Optional[LookupChunkHashesEntry]:
        """Pop precomputed chunk hashes/offsets recorded during lookup(), if any."""
        lock = getattr(self, "lock", None)
        with lock if lock is not None else nullcontext():
            store = getattr(self, "_lookup_chunk_hashes", None)
            return store.pop(lookup_id, None) if store is not None else None

    def cleanup_lookup(self, lookup_id: str) -> None:
        """Release worker-side lookup state, if any, and local status."""
        self.clear_lookup_status(lookup_id)
