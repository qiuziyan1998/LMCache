# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_key_trace import (
    maybe_trace_mooncake_store,
    run_mooncake_zero_lookup_retries,
)
from lmcache.v1.mooncake_layout import (
    mooncake_page_key,
    mooncake_page_layout_enabled,
)
from lmcache.v1.sampled_lookup import (
    find_last_sampled_hit,
    first_last_layer_keys,
)

logger = init_logger(__name__)


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
        self.store.setup(
            "localhost",
            "P2PHANDSHAKE",
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )
        self.store = maybe_trace_mooncake_store(
            self.store,
            "scheduler-lookup",
            metadata,
        )

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
        # process token_ids to cacheengine keys
        ends = []
        chunk_keys_by_chunk: list[list[str]] = []
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

        for start, end, key in self.token_database.process_tokens(
            token_ids, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            group_keys = [key]
            if dsa_two_groups:
                make_key = self.token_database._make_key_by_hash
                index_key = make_key(
                    key.chunk_hash,
                    request_configs,
                    kv_group=1,
                )
                group_keys.append(index_key)

            if page_first and end - start == self.config.chunk_size:
                chunk_keys = [
                    mooncake_page_key(group_key, num_layers)
                    for group_key in group_keys
                ]
            elif sampled_lookup:
                chunk_keys = [
                    key.to_string()
                    for key in first_last_layer_keys(group_keys, num_layers)
                ]
            elif use_layerwise:
                chunk_keys = [
                    layer_key.to_string()
                    for group_key in group_keys
                    for layer_key in group_key.split_layers(num_layers)
                ]
            else:
                chunk_keys = [group_key.to_string() for group_key in group_keys]
            chunk_keys_by_chunk.append(chunk_keys)
            ends.append(end)

        if not chunk_keys_by_chunk:
            return 0

        if sampled_lookup:

            def sampled_lookup_once() -> int:
                def batch_exists(keys: list[str]) -> bool:
                    if not keys:
                        return False
                    results = self.store.batch_is_exist(keys)
                    return len(results) == len(keys) and all(
                        result == 1 for result in results
                    )

                winner = find_last_sampled_hit(
                    len(chunk_keys_by_chunk),
                    lambda index: batch_exists(chunk_keys_by_chunk[index]),
                )
                return 0 if winner is None else ends[winner]

            return run_mooncake_zero_lookup_retries(
                getattr(self, "config", None),
                lookup_id,
                sampled_lookup_once,
            )

        keys = [key for chunk_keys in chunk_keys_by_chunk for key in chunk_keys]

        def lookup_once() -> int:
            # Use batch_is_exist to check all keys at once.
            # Results: 1 = found, 0 = not found, -1 = error.
            rets = self.store.batch_is_exist(keys)

            offset = 0
            for chunk_idx, chunk_keys in enumerate(chunk_keys_by_chunk):
                key_count = len(chunk_keys)
                chunk_rets = rets[offset : offset + key_count]
                offset += key_count
                if len(chunk_rets) < key_count or any(
                    ret != 1 for ret in chunk_rets
                ):
                    return ends[chunk_idx - 1] if chunk_idx > 0 else 0
            return ends[-1] if ends else 0

        return run_mooncake_zero_lookup_retries(
            getattr(self, "config", None),
            lookup_id,
            lookup_once,
        )

    def supports_producer_reuse(self) -> bool:
        """Return True as MooncakeLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        # nothing here
        pass
