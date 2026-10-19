# SPDX-License-Identifier: Apache-2.0
"""Freeze registered source geometry without serializing device destinations."""

# Standard
from collections.abc import Mapping
from typing import Any

# Third Party
import msgspec

# First Party
from lmcache.v1.rpc.storage_protocol import GroupLayout, LayoutSpec


def freeze_storage_layout(
    config: Any,
    metadata: Any,
    token_database: Any,
    *,
    allocator_alignment: int = 64,
    plane_token_bytes: Mapping[int, tuple[int, ...] | tuple[tuple[int, ...], ...]]
    | None = None,
) -> LayoutSpec:
    """Build the storage ABI from authoritative registered physical owners.

    HBM block maps, pointers, streams and callbacks deliberately remain local.
    The existing key namespace and precision descriptor define source identity.
    """
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.mooncake_layout import resolve_remote_fill_identity

    counts = metadata.runtime_kv_group_layer_counts
    names = metadata.runtime_kv_group_layer_names
    groups = metadata.kv_layer_groups_manager.kv_layer_groups
    if counts is None or names is None or len(counts) != 2 or len(groups) != 2:
        raise ValueError("Decoder storage requires both registered physical KV groups")
    if tuple(len(group_names) for group_names in names) != tuple(counts):
        raise ValueError("Registered physical source names disagree with group counts")
    if tuple(group.num_layers for group in groups) != tuple(counts):
        raise ValueError("Registered source groups disagree with physical layer counts")
    policy = metadata.indexer_c8_layout
    supplied_planes = plane_token_bytes or {}
    if set(supplied_planes) - {0, 1}:
        raise ValueError("Unknown registered source plane group")
    specs = []
    for group_id, group in enumerate(groups):
        count = counts[group_id]
        if group_id == 1 and policy is not None:
            widths = tuple(
                (policy.head_dim, 2) if policy.is_c8(layer) else (policy.head_dim * 2,)
                for layer in range(count)
            )
            dtypes = ("uint8",) * count
        else:
            widths = ((int(group.hidden_dim_size) * group.dtype.itemsize,),) * count
            dtypes = (str(group.dtype).removeprefix("torch."),) * count
        if group_id in supplied_planes:
            supplied = tuple(supplied_planes[group_id])
            if supplied and all(isinstance(width, int) for width in supplied):
                supplied = (supplied,) * count
            else:
                try:
                    supplied = tuple(tuple(planes) for planes in supplied)
                except TypeError as error:
                    raise ValueError("Invalid registered source plane shape") from error
            if len(supplied) != count or any(
                not planes
                or any(type(width) is not int or width <= 0 for width in planes)
                for planes in supplied
            ):
                raise ValueError("Invalid registered source plane cardinality")
            if group_id == 1 and policy is not None:
                if supplied != widths:
                    raise ValueError(
                        "Registered indexer planes disagree with precision"
                    )
            elif any(
                sum(row) != sum(expected)
                for row, expected in zip(supplied, widths, strict=True)
            ):
                raise ValueError(
                    "Registered source planes disagree with dtype byte geometry"
                )
            widths = supplied
        specs.append(
            GroupLayout(
                group_id,
                tuple(names[group_id]),
                dtypes,
                widths,
                (
                    MemoryFormat.KV_MLA_LATENT_FMT
                    if group_id == 0
                    else MemoryFormat.KV_DSA_INDEX_FMT
                ).value,
            )
        )
    kind = "bytes" if token_database.chunk_hash_type is bytes else "int"
    algorithm = str(config.pre_caching_hash_algorithm)
    width = (
        int(token_database.chunk_hash_bytes) * 8
        if kind == "bytes"
        else 256
        if "sha256" in algorithm
        else 64
    )
    namespace, artifact = resolve_remote_fill_identity(config, metadata)
    layout = LayoutSpec(
        artifact or metadata.model_name,
        namespace or token_database.mooncake_payload_layout or metadata.model_name,
        algorithm,
        kind,
        width,
        1 if token_database.save_only_first_rank else int(metadata.world_size),
        int(config.chunk_size),
        tuple(specs),
        msgspec.msgpack.encode(policy.descriptor()) if policy is not None else b"",
        allocator_alignment=allocator_alignment,
    )
    layout.validate()
    return layout
