# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence
from typing import Any, Optional, Union
import abc
import threading
import zlib

# Third Party
import numpy as np
import torch


_STASH_LOCK_GUARD = threading.Lock()


def _compute_chunk_crc32s(
    token_ids: Union[torch.Tensor, list[int]],
    offsets: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    arr = (
        token_ids.detach().to(device="cpu", dtype=torch.int64, copy=False).numpy()
        if isinstance(token_ids, torch.Tensor)
        else np.asarray(token_ids, dtype=np.int64)
    )
    mv = memoryview(np.ascontiguousarray(arr)).cast("B")
    crc = 0
    pos = 0
    chunk_crcs: list[int] = []
    for off in offsets:
        end = pos + int(off)
        crc = zlib.crc32(mv[pos * 8 : end * 8], crc)
        chunk_crcs.append(crc)
        pos = end
    if pos < len(arr):
        crc = zlib.crc32(mv[pos * 8 :], crc)
    return crc, tuple(chunk_crcs)


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
        self._drop_lookup_chunk_hashes(lookup_id)

    def _get_lookup_stash_lock(self) -> Any:
        stash_lock = getattr(self, "_stash_lock", None)
        if stash_lock is None:
            with _STASH_LOCK_GUARD:
                stash_lock = getattr(self, "_stash_lock", None)
                if stash_lock is None:
                    stash_lock = threading.RLock()
                    self._stash_lock = stash_lock
        return stash_lock

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
        token_len = len(token_ids)
        if not hashes or token_len <= 0:
            self._drop_lookup_chunk_hashes(lookup_id)
            return
        offsets_tup = tuple(offsets)
        digests = _compute_chunk_crc32s(token_ids, offsets_tup)
        with self._get_lookup_stash_lock():
            store: Optional[dict[str, LookupChunkHashesEntry]] = getattr(
                self, "_lookup_chunk_hashes", None
            )
            if store is None:
                store = {}
                self._lookup_chunk_hashes = store
            digest_store: Optional[dict[str, tuple[int, tuple[int, ...]]]] = getattr(
                self, "_lookup_chunk_digests", None
            )
            if digest_store is None:
                digest_store = {}
                self._lookup_chunk_digests = digest_store
            if lookup_id not in store and len(store) >= 4096:
                evicted = next(iter(store))
                store.pop(evicted, None)
                digest_store.pop(evicted, None)
            store[lookup_id] = (
                tuple(hashes),
                offsets_tup,
                token_len,
                int(token_ids[0]),
                int(token_ids[-1]),
                tuple(chunk_end_tokens),
            )
            digest_store[lookup_id] = digests

    def _drop_lookup_chunk_hashes(self, lookup_id: Optional[str]) -> None:
        if not lookup_id:
            return
        with self._get_lookup_stash_lock():
            store = getattr(self, "_lookup_chunk_hashes", None)
            if store is not None:
                store.pop(lookup_id, None)
            digest_store = getattr(self, "_lookup_chunk_digests", None)
            if digest_store is not None:
                digest_store.pop(lookup_id, None)

    def _clear_all_lookup_stashes(self) -> None:
        with self._get_lookup_stash_lock():
            store = getattr(self, "_lookup_chunk_hashes", None)
            if store is not None:
                store.clear()
            digest_store = getattr(self, "_lookup_chunk_digests", None)
            if digest_store is not None:
                digest_store.clear()

    def pop_lookup_chunk_hashes(
        self, lookup_id: str
    ) -> Optional[LookupChunkHashesEntry]:
        """Pop precomputed chunk hashes/offsets recorded during lookup(), if any."""
        with self._get_lookup_stash_lock():
            store = getattr(self, "_lookup_chunk_hashes", None)
            return store.pop(lookup_id, None) if store is not None else None

    def pop_lookup_chunk_digests(
        self, lookup_id: str
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        """Pop precomputed cumulative CRC32 digests recorded during lookup()."""
        with self._get_lookup_stash_lock():
            digest_store = getattr(self, "_lookup_chunk_digests", None)
            return (
                digest_store.pop(lookup_id, None)
                if digest_store is not None
                else None
            )

    def cleanup_lookup(self, lookup_id: str) -> None:
        """Release worker-side lookup state, if any, and local status."""
        self.clear_lookup_status(lookup_id)
