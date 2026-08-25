# SPDX-License-Identifier: Apache-2.0

# Standard
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata


MOONCAKE_PAYLOAD_LAYOUT_TAG = "lmcache.tag.payload_v3"
MOONCAKE_VALID_TOKENS_TAG = "lmcache.tag.internal.valid_tokens"
REMOTE_FILL_PAGE_ABI_VERSION = "lmcache-layer-page-v3"
REMOTE_FILL_AUTO_ARTIFACT_VERSION = "sampled-serving-bundle-v1"
_AUTO_IDENTITY_EXTENSIONS = {
    ".bin",
    ".json",
    ".model",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".tiktoken",
    ".txt",
}
_AUTO_IDENTITY_SAMPLE_BYTES = 64 * 1024


def _parse_dsa_raw_token_dims(value: Any) -> dict[int, int]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = dict(item.split(":", 1) for item in value.split(",") if item)
    else:
        parsed = value
    if isinstance(parsed, dict):
        return {int(key): int(dimension) for key, dimension in parsed.items()}
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return {0: int(parsed[0]), 1: int(parsed[1])}
    raise ValueError(
        "mooncake_dsa_raw_token_dims must be a dict, list, or "
        "'0:latent,1:indexer' string"
    )


def _first_int_from_nested(config_dict: dict[str, Any], names: set[str]) -> int | None:
    stack: list[Any] = [config_dict]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in names and isinstance(value, int):
                    return value
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return None


def _load_model_config_for_dsa_dims(
    metadata: LMCacheMetadata,
) -> tuple[dict[str, Any], str]:
    model_name = getattr(metadata, "model_name", "")
    if not model_name:
        return {}, "metadata.model_name unavailable"
    config_path = os.path.join(model_name, "config.json")
    if not os.path.isfile(config_path):
        return {}, f"{config_path} unavailable"
    try:
        with open(config_path, encoding="utf-8") as stream:
            loaded = json.load(stream)
    except Exception as exc:
        return {}, f"{config_path} load failed: {exc}"
    if not isinstance(loaded, dict):
        return {}, f"{config_path} is not a JSON object"
    return loaded, config_path


def _infer_dsa_raw_token_dims(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
) -> tuple[dict[int, int], str]:
    if not config.dsa_two_groups:
        return {}, "dsa_two_groups disabled"

    inferred: dict[int, int] = {}
    shapes = metadata.get_shapes()
    if metadata.use_mla and shapes:
        for kv_group, shape in enumerate(shapes[:2]):
            if len(shape) >= 1:
                inferred[kv_group] = int(shape[-1])

    model_config, source = _load_model_config_for_dsa_dims(metadata)
    if not model_config:
        return inferred, source if inferred else "no inferable local metadata"

    kv_lora_rank = _first_int_from_nested(
        model_config,
        {"kv_lora_rank", "k_head_dim", "k_hidden_dims"},
    )
    qk_rope_head_dim = _first_int_from_nested(
        model_config,
        {"qk_rope_head_dim", "rope_head_dim", "v_head_dim"},
    )
    dsa_head_dim = _first_int_from_nested(
        model_config,
        {
            "dsa_head_dim",
            "dsa_hidden_dim",
            "dsa_hidden_dims",
            "index_head_dim",
            "indexer_head_dim",
        },
    )
    if dsa_head_dim is None:
        dsa_head_dim = _first_int_from_nested(
            model_config,
            {"head_dim", "hidden_size_per_attention_head"},
        )
    if kv_lora_rank is not None and qk_rope_head_dim is not None:
        inferred[0] = kv_lora_rank + qk_rope_head_dim
    if dsa_head_dim is not None:
        inferred[1] = dsa_head_dim
    return inferred, source


def resolve_mooncake_dsa_raw_token_dims(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
) -> tuple[dict[int, int], str]:
    """Resolve the immutable per-token widths for both DSA cache groups.

    Explicit configuration remains an override for unusual model layouts. The
    normal path derives dimensions from model metadata at startup so Mooncake
    persistence and RemoteFill negotiate one identical payload identity.

    Args:
        config: Resolved LMCache engine configuration.
        metadata: Model and cache topology metadata.

    Returns:
        A mapping from cache group to raw elements per token and a diagnostic
        description of the source used for the mapping.
    """

    if not config.dsa_two_groups:
        return {}, "dsa_two_groups disabled"

    override = (config.extra_config or {}).get("mooncake_dsa_raw_token_dims")
    if override is not None:
        return (
            _parse_dsa_raw_token_dims(override),
            "extra_config.mooncake_dsa_raw_token_dims",
        )

    inferred, source = _infer_dsa_raw_token_dims(config, metadata)
    if inferred.get(0, 0) > 0 and inferred.get(1, 0) > 0:
        return inferred, f"model config inference: {source}"

    model_name = getattr(metadata, "model_name", "")
    world_size = getattr(metadata, "world_size", None)
    if "GLM-5.1-w4a8" in model_name and world_size == 8:
        return {0: 576, 1: 128}, "hardcoded GLM-5.1-w4a8 TP8"
    return {}, "no raw DSA dims rule matched"


def _update_sampled_file_identity(digest: Any, path: Path) -> None:
    """Add bounded content evidence for one serving-bundle file."""

    size = path.stat().st_size
    digest.update(str(size).encode())
    with path.open("rb") as stream:
        digest.update(stream.read(_AUTO_IDENTITY_SAMPLE_BYTES))
        if size > _AUTO_IDENTITY_SAMPLE_BYTES:
            stream.seek(max(0, size - _AUTO_IDENTITY_SAMPLE_BYTES))
            digest.update(stream.read(_AUTO_IDENTITY_SAMPLE_BYTES))


@lru_cache(maxsize=16)
def _derive_remote_fill_artifact_id(
    model_name: str,
    served_model_name: str,
) -> str:
    """Derive a stable, bounded-cost serving-bundle fingerprint at startup."""

    digest = hashlib.sha256()
    digest.update(REMOTE_FILL_AUTO_ARTIFACT_VERSION.encode())
    digest.update(served_model_name.encode())
    model_path = Path(model_name)
    if not model_path.is_dir():
        digest.update(model_name.encode())
        return f"auto-{REMOTE_FILL_AUTO_ARTIFACT_VERSION}-{digest.hexdigest()}"

    candidates = sorted(
        (
            path
            for path in model_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _AUTO_IDENTITY_EXTENSIONS
        ),
        key=lambda path: path.relative_to(model_path).as_posix(),
    )
    if not candidates:
        digest.update(model_path.name.encode())
    for path in candidates:
        digest.update(path.relative_to(model_path).as_posix().encode())
        _update_sampled_file_identity(digest, path)
    return f"auto-{REMOTE_FILL_AUTO_ARTIFACT_VERSION}-{digest.hexdigest()}"


def resolve_remote_fill_identity(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
) -> tuple[str, str]:
    """Resolve optional overrides into deterministic RemoteFill identities.

    The automatic artifact fingerprint samples every model, tokenizer, and
    custom-code file at bounded cost. An explicit immutable build/revision ID
    remains available for artifact stores that can replace unsampled content
    in place.
    """

    artifact_id = str(config.remote_fill_model_artifact_id or "").strip()
    namespace = str(config.remote_fill_cache_namespace or "").strip()
    remote_fill_active = bool(config.enable_remote_lmcache_store)
    if not remote_fill_active:
        # Preserve the disabled path exactly: no bundle walk, no key namespace
        # change, and no new startup work unless the operator explicitly set
        # an identity for another purpose.
        return namespace, artifact_id
    if not artifact_id:
        artifact_id = _derive_remote_fill_artifact_id(
            str(metadata.model_name),
            str(metadata.served_model_name or metadata.model_name),
        )
    if not namespace:
        namespace_digest = hashlib.sha256(
            f"{artifact_id}\0{REMOTE_FILL_PAGE_ABI_VERSION}".encode()
        ).hexdigest()[:24]
        namespace = f"remote-fill-{namespace_digest}"
    return namespace, artifact_id


def mooncake_valid_tokens(key: CacheEngineKey, chunk_size: int) -> int:
    """Return the authoritative token count encoded in a page key.

    Full-page keys intentionally omit the internal tag and therefore retain
    their historical identity. Partial-page keys must carry a positive count
    smaller than ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    tag_name = MOONCAKE_VALID_TOKENS_TAG.removeprefix("lmcache.tag.")
    tagged = dict(key.tags or ()).get(tag_name)
    if tagged is None:
        return chunk_size
    try:
        valid_tokens = int(tagged)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid partial-page token count: {tagged!r}") from error
    if not 0 < valid_tokens < chunk_size:
        raise ValueError(
            "Partial-page token count must be between 1 and chunk_size - 1: "
            f"valid_tokens={valid_tokens}, chunk_size={chunk_size}"
        )
    return valid_tokens


def mooncake_legacy_key(key: CacheEngineKey) -> str:
    """Return the pre-partial-page serialization for legacy tail reads."""
    marker = "@internal.valid_tokens%"
    return "@".join(
        part for part in key.to_string().split("@") if not part.startswith(marker[1:])
    )


def mooncake_payload_layout(
    config: LMCacheEngineConfig, metadata: LMCacheMetadata
) -> tuple[str, dict[str, object]]:
    """Return a stable, scheduler-visible Mooncake payload schema identity."""
    # Local import avoids coupling the lightweight page-key helpers to backend
    # initialization while reusing the connector's authoritative defaulting.
    from lmcache.v1.storage_backend.connector.base_connector import (
        resolve_save_chunk_meta,
    )

    chunk_size = int(config.chunk_size)
    kv_shape = tuple(int(value) for value in metadata.kv_shape)
    dtype = metadata.kv_dtype
    extra_config = config.extra_config or {}
    model_config_hash = ""
    model_config_path = os.path.join(metadata.model_name, "config.json")
    try:
        with open(model_config_path, encoding="utf-8") as model_config:
            canonical_model_config = json.dumps(
                json.load(model_config), sort_keys=True, separators=(",", ":")
            )
            model_config_hash = hashlib.blake2b(
                canonical_model_config.encode(), digest_size=8
            ).hexdigest()
    except (OSError, TypeError, ValueError):
        pass
    resolved_dims, _dims_source = resolve_mooncake_dsa_raw_token_dims(
        config, metadata
    )
    normalized_dims = {
        str(key): int(dimension) for key, dimension in resolved_dims.items()
    }
    raw_token_dims: str | tuple[tuple[str, int], ...] = (
        tuple(sorted(normalized_dims.items())) if normalized_dims else ""
    )
    deployment_namespace, model_artifact_id = resolve_remote_fill_identity(
        config, metadata
    )
    descriptor = {
        "version": 3,
        "deployment_namespace": deployment_namespace,
        "model_artifact_id": model_artifact_id,
        # The deployment-supplied artifact is the immutable serving-bundle ID:
        # weights, quantization output, tokenizer, and custom model code.  The
        # page ABI remains code-owned so a storage-layout change also changes
        # every canonical page key without another operator knob.
        "model_artifact_scope": "serving-bundle-v1",
        "page_abi_version": REMOTE_FILL_PAGE_ABI_VERSION,
        "chunk_size": chunk_size,
        "kv_shape": kv_shape,
        "dtype": str(dtype),
        "model_config_digest": model_config_hash,
        "cache_bearing_layers": extra_config.get("mooncake_cache_bearing_layers", ""),
        "use_mla": metadata.use_mla,
        "use_layerwise": config.use_layerwise,
        "dsa_two_groups": config.dsa_two_groups,
        "save_chunk_meta": resolve_save_chunk_meta(config),
        "raw_token_dims": raw_token_dims,
        "group0_raw_token_dim": normalized_dims.get("0", 0),
        "group1_raw_token_dim": normalized_dims.get("1", 0),
        "group1_schema_version": str(
            extra_config.get("mooncake_group1_schema_version", "")
        ),
        "mtp_layout_version": str(extra_config.get("mooncake_mtp_layout_version", "")),
        "page_first_multi_buffer": bool(
            extra_config.get("mooncake_page_first_multi_buffer", False)
        ),
        "layer_merged_pages": bool(
            extra_config.get("mooncake_layer_merged_page_objects", False)
        ),
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode(), digest_size=8).hexdigest(), descriptor


def mooncake_page_key(key: CacheEngineKey, num_layers: int) -> str:
    """Return the versioned Mooncake key for one all-layer token page.

    The key intentionally excludes ``layer_id`` while retaining the model,
    worker, chunk hash, dtype, KV group, and request tags of the source key.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    # Explicit base dispatch omits LayerCacheEngineKey.layer_id without
    # rebuilding and revalidating an equivalent chunk key.
    chunk_key = CacheEngineKey.to_string(key)
    return f"__lmcache_page_v1__@{num_layers}@{chunk_key}"


def mooncake_page_layout_enabled(config: object) -> bool:
    """Return whether page-first Mooncake multi-buffer storage is enabled."""
    extra_config = getattr(config, "extra_config", None) or {}
    return bool(extra_config.get("mooncake_page_first_multi_buffer", False))


def mooncake_layer_pages_enabled(config: object) -> bool:
    """Return whether the experimental LocalCPU layer-page layout is enabled."""
    extra_config = getattr(config, "extra_config", None) or {}
    shared = bool(
        extra_config.get(
            "enable_shared_cpu_cache",
            getattr(config, "enable_shared_cpu_cache", False),
        )
    )
    return (
        shared
        and bool(getattr(config, "use_layerwise", False))
        and bool(extra_config.get("save_only_first_rank", False))
        and str(getattr(config, "remote_url", "")).startswith("mooncakestore://")
        and mooncake_page_layout_enabled(config)
        and bool(extra_config.get("mooncake_layer_merged_page_objects", False))
    )
