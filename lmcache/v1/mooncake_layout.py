# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.utils import CacheEngineKey


def mooncake_page_key(key: CacheEngineKey, num_layers: int) -> str:
    """Return the versioned Mooncake key for one all-layer token page.

    The key intentionally excludes ``layer_id`` while retaining the model,
    worker, chunk hash, dtype, KV group, and request tags of the source key.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    chunk_key = CacheEngineKey(
        model_name=key.model_name,
        world_size=key.world_size,
        worker_id=key.worker_id,
        chunk_hash=key.chunk_hash,
        dtype=key.dtype,
        request_configs=key.request_configs,
        kv_group=key.kv_group,
    )
    return f"__lmcache_page_v1__@{num_layers}@{chunk_key.to_string()}"


def mooncake_page_layout_enabled(config: object) -> bool:
    """Return whether page-first Mooncake multi-buffer storage is enabled."""
    extra_config = getattr(config, "extra_config", None) or {}
    return bool(extra_config.get("mooncake_page_first_multi_buffer", False))
