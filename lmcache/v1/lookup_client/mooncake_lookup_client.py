# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_key_trace import trace_mooncake_keys
from lmcache.v1.mooncake_layout import (
    mooncake_legacy_key,
    mooncake_page_key,
    mooncake_page_layout_enabled,
)
from lmcache.v1.sampled_lookup import (
    find_last_sampled_hit,
    first_last_layer_keys,
)

class MooncakeLookupClient(LookupClientInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        master_addr: str,
    ):
        # Third Party
        from mooncake.store import MooncakeDistributedStore

        self.config = config
        self.metadata = metadata
        self.store = MooncakeDistributedStore()
        status = self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            0,
            "tcp",
            "",
            master_addr,
        )
        if status not in (None, 0):
            self.store.close()
            raise RuntimeError(f"Mooncake lookup setup failed: status={status}")

        # Initialize token database for processing tokens
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed."
        )

        # First Party
        from lmcache.v1.token_database import ChunkedTokenDatabase

        assert not config.enable_blending, (
            "LMCache v1 blending is not supported in MooncakeLookupClient yet."
        )
        self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: Optional[str] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """Return the cached prefix length.

        In experimental sampled mode the result is only a candidate under the
        contiguous-prefix contract. Unlike the production lookup-server path,
        this standalone client does not pin and revalidate intermediate chunks.
        """
        # process token_ids to cacheengine keys
        ends = []
        chunk_group_keys: list[list[CacheEngineKey]] = []
        use_layerwise = bool(
            getattr(getattr(self, "config", None), "use_layerwise", False)
        )
        dsa_two_groups = bool(
            getattr(getattr(self, "config", None), "dsa_two_groups", False)
        )
        num_layers = int(
            getattr(getattr(self, "metadata", None), "kv_shape", (1,))[0]
        )
        sampled_lookup = bool(
            use_layerwise
            and getattr(self.config, "experimental_sampled_layerwise_lookup", False)
        )
        page_first = use_layerwise and mooncake_page_layout_enabled(self.config)

        base_chunks = list(
            self.token_database.process_tokens(
                token_ids, request_configs=request_configs
            )
        )
        group1_keys: list[CacheEngineKey] = []
        if dsa_two_groups and base_chunks:
            group1_chunks = list(
                self.token_database.process_tokens(
                    hashes=[key.chunk_hash for _, _, key in base_chunks],
                    offsets=[end - start for start, end, _ in base_chunks],
                    request_configs=request_configs,
                    kv_group=1,
                )
            )
            if [
                (start, end, key.chunk_hash)
                for start, end, key in group1_chunks
            ] != [
                (start, end, key.chunk_hash)
                for start, end, key in base_chunks
            ]:
                raise ValueError(
                    "KV groups produced inconsistent chunk metadata"
                )
            group1_keys = [key for _, _, key in group1_chunks]

        for chunk_index, (start, end, key) in enumerate(base_chunks):
            assert isinstance(key, CacheEngineKey)
            group_keys = [key]
            if dsa_two_groups:
                group_keys.append(group1_keys[chunk_index])
            chunk_group_keys.append(group_keys)
            ends.append(end)

        def string_keys(
            group_keys: list[CacheEngineKey], *, sampled: bool
        ) -> list[str]:
            def serialize(key: CacheEngineKey) -> str:
                return mooncake_legacy_key(key) if page_first else key.to_string()

            if sampled:
                return [
                    serialize(key)
                    for key in first_last_layer_keys(group_keys, num_layers)
                ]
            if use_layerwise:
                return [
                    serialize(layer_key)
                    for group_key in group_keys
                    for layer_key in group_key.split_layers(num_layers)
                ]
            return [serialize(group_key) for group_key in group_keys]

        def batch_exists(keys: list[str]) -> bool:
            if not keys:
                return False
            results = self.store.batch_is_exist(keys)
            trace_mooncake_keys(
                "lookup",
                keys,
                results,
                api="MooncakeLookupClient.batch_is_exist",
                lookup_id=lookup_id,
            )
            return len(results) == len(keys) and all(
                result == 1 for result in results
            )

        if page_first:
            page_keys_by_chunk = [
                [mooncake_page_key(key, num_layers) for key in group_keys]
                for group_keys in chunk_group_keys
            ]

            def page_or_legacy_exists(index: int) -> bool:
                if batch_exists(page_keys_by_chunk[index]):
                    return True
                return batch_exists(
                    string_keys(
                        chunk_group_keys[index], sampled=sampled_lookup
                    )
                )

            if sampled_lookup:
                winner = find_last_sampled_hit(
                    len(chunk_group_keys), page_or_legacy_exists
                )
                return 0 if winner is None else ends[winner]

            flat_page_keys = [key for chunk in page_keys_by_chunk for key in chunk]
            page_results = self.store.batch_is_exist(flat_page_keys)
            trace_mooncake_keys(
                "lookup",
                flat_page_keys,
                page_results,
                api="MooncakeLookupClient.batch_is_exist",
                lookup_id=lookup_id,
            )
            offset = 0
            for index, page_keys in enumerate(page_keys_by_chunk):
                chunk_results = page_results[offset : offset + len(page_keys)]
                offset += len(page_keys)
                page_hit = len(chunk_results) == len(page_keys) and all(
                    result == 1 for result in chunk_results
                )
                if not page_hit and not batch_exists(
                    string_keys(chunk_group_keys[index], sampled=False)
                ):
                    return ends[index - 1] if index else 0
            return ends[-1] if ends else 0

        if sampled_lookup:
            winner = find_last_sampled_hit(
                len(chunk_group_keys),
                lambda index: batch_exists(
                    string_keys(chunk_group_keys[index], sampled=True)
                ),
            )
            return 0 if winner is None else ends[winner]

        # Use batch_is_exist to check all keys at once
        # rets is list of int: 1 = found, 0 = not found, -1 = error
        chunk_keys_by_chunk = [
            string_keys(group_keys, sampled=False)
            for group_keys in chunk_group_keys
        ]
        keys = [key for chunk_keys in chunk_keys_by_chunk for key in chunk_keys]
        rets = self.store.batch_is_exist(keys)
        trace_mooncake_keys(
            "lookup",
            keys,
            rets,
            api="MooncakeLookupClient.batch_is_exist",
            lookup_id=lookup_id,
        )

        # Find the first key that doesn't exist (ret != 1)
        # This follows the same logic as cache engine's lookup method
        offset = 0
        for chunk_idx, chunk_keys in enumerate(chunk_keys_by_chunk):
            key_count = len(chunk_keys)
            chunk_rets = rets[offset : offset + key_count]
            offset += key_count
            if len(chunk_rets) < key_count or any(
                ret != 1 for ret in chunk_rets
            ):
                # Return the end position of the previous chunk
                # If chunk_idx == 0, no chunks were found, return 0
                return ends[chunk_idx - 1] if chunk_idx > 0 else 0

        # All keys were found, return the last end position
        return ends[-1] if ends else 0

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.store.close()
