# SPDX-License-Identifier: Apache-2.0
"""Shared CPU cache handle and passive-view primitives.

This module intentionally contains no storage-tier policy. Rank0 resolves real
MemoryObjs through the existing StorageManager/LocalCPUBackend path, publishes
metadata handles, and passive ranks build view-only MemoryObjs from those
handles after strict validation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

logger = init_logger(__name__)


class SharedCPUCacheError(RuntimeError):
    """Base error for shared CPU cache contract violations."""


class SharedCPUCacheValidationError(SharedCPUCacheError):
    """Raised before view creation or pointer install when metadata is unsafe."""


def _dtype_to_str(dtype: Optional[torch.dtype]) -> Optional[str]:
    return str(dtype) if dtype is not None else None


def _dtype_from_str(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    try:
        resolved = getattr(torch, dtype.replace("torch.", ""))
    except AttributeError as exc:
        raise SharedCPUCacheValidationError(
            f"Unknown shared CPU cache tensor dtype {dtype!r}"
        ) from exc
    if not isinstance(resolved, torch.dtype):
        raise SharedCPUCacheValidationError(
            f"Invalid shared CPU cache tensor dtype {dtype!r}"
        )
    return resolved


def _positions_to_list(
    cached_positions: Optional[Union[torch.Tensor, list[int]]],
) -> Optional[list[int]]:
    if cached_positions is None:
        return None
    if isinstance(cached_positions, torch.Tensor):
        return [int(x) for x in cached_positions.detach().cpu().flatten().tolist()]
    return [int(x) for x in cached_positions]


def _expected_positions_match(
    cached_positions: list[int],
    expected_positions: Iterable[int],
) -> tuple[bool, Any]:
    if isinstance(expected_positions, range):
        if len(cached_positions) != len(expected_positions):
            return (
                False,
                f"range({expected_positions.start}, "
                f"{expected_positions.stop}, {expected_positions.step})",
            )
        for cached, expected in zip(cached_positions, expected_positions):
            if int(cached) != int(expected):
                return (
                    False,
                    f"range({expected_positions.start}, "
                    f"{expected_positions.stop}, {expected_positions.step})",
                )
        return True, None

    expected_list = [int(pos) for pos in expected_positions]
    return cached_positions == expected_list, expected_list


def _shape_nbytes(shape: torch.Size, dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        if int(dim) < 0:
            raise ValueError(f"negative dimension {dim}")
        numel *= int(dim)
    return int(numel * dtype.itemsize)


def _handle_logical_nbytes(handle: "SharedChunkHandle") -> int:
    if handle.shapes is not None or handle.dtypes is not None:
        if not handle.shapes or not handle.dtypes:
            raise ValueError("shapes and dtypes must both be present")
        if len(handle.shapes) != len(handle.dtypes):
            raise ValueError(
                f"shapes/dtypes length mismatch: "
                f"{len(handle.shapes)} != {len(handle.dtypes)}"
            )
        return sum(
            _shape_nbytes(shape, dtype)
            for shape, dtype in zip(handle.shapes, handle.dtypes, strict=True)
        )
    return _shape_nbytes(handle.shape, handle.dtype)


def _shared_key_matches_expected(
    handle_key: CacheEngineKey,
    expected_key: CacheEngineKey,
    expected_producer_rank: Optional[int],
) -> bool:
    if handle_key == expected_key:
        return True
    if expected_producer_rank is None:
        return False
    if type(handle_key) is not type(expected_key):
        return False
    if int(handle_key.worker_id) != int(expected_producer_rank):
        return False

    comparable_fields = (
        "model_name",
        "world_size",
        "chunk_hash",
        "dtype",
        "tags",
        "kv_group",
    )
    for field in comparable_fields:
        if getattr(handle_key, field) != getattr(expected_key, field):
            return False
    if hasattr(expected_key, "layer_id") and (
        getattr(handle_key, "layer_id", None)
        != getattr(expected_key, "layer_id", None)
    ):
        return False
    return True


def _load_lmc_ops(*, purpose: str):
    try:
        import lmcache.c_ops as lmc_ops

        return lmc_ops
    except ImportError as c_ops_exc:
        try:
            import lmcache.non_cuda_equivalents as lmc_ops

            return lmc_ops
        except ImportError as fallback_exc:
            raise SharedCPUCacheError(
                f"Shared CPU cache {purpose} requires lmcache.c_ops or "
                "lmcache.non_cuda_equivalents. On Ascend, import/build "
                "lmcache_ascend so lmcache.c_ops is patched to "
                "lmcache_ascend.c_ops."
            ) from fallback_exc


def _require_fields(data: dict[str, Any], fields: set[str], owner: str) -> None:
    missing = sorted(fields - set(data))
    if missing:
        raise SharedCPUCacheValidationError(
            f"{owner} missing required fields: {missing}"
        )


def _reject_private_fields(
    data: dict[str, Any],
    fields: set[str],
    owner: str,
) -> None:
    present = sorted(fields & set(data))
    if present:
        raise SharedCPUCacheValidationError(
            f"{owner} contains forbidden pointer/allocator fields: {present}"
        )


_FORBIDDEN_TRANSPORT_FIELDS = {
    "address_manager",
    "allocator",
    "allocator_state",
    "device_ptr",
    "host_ptr",
    "parent_allocator",
    "ptr",
    "python_object_id",
    "raw_data",
    "storage",
    "tensor",
}

_COMPACT_ENVELOPE_TYPE = "SharedHandleEnvelopeColumnarV1"


def _require_exact_fields(
    data: dict[str, Any],
    fields: set[str],
    owner: str,
) -> None:
    _require_fields(data, fields, owner)
    unexpected = sorted(set(data) - fields)
    if unexpected:
        raise SharedCPUCacheValidationError(
            f"{owner} contains unexpected fields: {unexpected}"
        )


def _values_exactly_equal(left: Any, right: Any) -> bool:
    """Compare serialized values without Python's cross-type equality."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if len(left) != len(right):
            return False
        return all(
            _values_exactly_equal(left_key, right_key)
            and _values_exactly_equal(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left.items(), right.items(), strict=True
            )
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_exactly_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if left is None or isinstance(left, (bool, int, float, str, bytes)):
        return bool(left == right)
    # Unknown objects are only common when they are literally the same value.
    # This deliberately sacrifices compression rather than trusting a custom
    # equality implementation that may ignore serialized state.
    return left is right


def _encode_exact_column(values: list[Any]) -> dict[str, Any]:
    """Encode a column using only representations proven exactly reversible."""
    if not values:
        return {"kind": "values", "values": []}

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        step = 0 if len(values) == 1 else int(values[1]) - int(values[0])
        if all(
            int(value) == int(values[0]) + index * step
            for index, value in enumerate(values)
        ):
            return {
                "kind": "arithmetic",
                "start": int(values[0]),
                "step": step,
            }

    default = values[0]
    overrides = [
        [index, value]
        for index, value in enumerate(values[1:], start=1)
        if not _values_exactly_equal(value, default)
    ]
    if len(overrides) * 2 < len(values):
        return {
            "kind": "default_overrides",
            "default": default,
            "overrides": overrides,
        }
    return {"kind": "values", "values": values}


def _decode_exact_column(
    encoded: dict[str, Any],
    count: int,
    owner: str,
) -> list[Any]:
    if not isinstance(encoded, dict):
        raise SharedCPUCacheValidationError(
            f"{owner} expected dict column, got {type(encoded)!r}"
        )
    kind = encoded.get("kind")
    if kind == "arithmetic":
        _require_exact_fields(encoded, {"kind", "start", "step"}, owner)
        start = int(encoded["start"])
        step = int(encoded["step"])
        return [start + index * step for index in range(count)]
    if kind == "default_overrides":
        _require_exact_fields(
            encoded,
            {"kind", "default", "overrides"},
            owner,
        )
        overrides = encoded["overrides"]
        if not isinstance(overrides, list):
            raise SharedCPUCacheValidationError(
                f"{owner} overrides must be a list"
            )
        values = [encoded["default"] for _ in range(count)]
        seen: set[int] = set()
        for item in overrides:
            if not isinstance(item, list) or len(item) != 2:
                raise SharedCPUCacheValidationError(
                    f"{owner} override must be [index, value]"
                )
            index = int(item[0])
            if index < 0 or index >= count or index in seen:
                raise SharedCPUCacheValidationError(
                    f"{owner} has invalid override index {index}"
                )
            seen.add(index)
            values[index] = item[1]
        return values
    if kind == "values":
        _require_exact_fields(encoded, {"kind", "values"}, owner)
        values = encoded["values"]
        if not isinstance(values, list) or len(values) != count:
            raise SharedCPUCacheValidationError(
                f"{owner} value count mismatch: "
                f"{len(values) if isinstance(values, list) else type(values)!r} "
                f"!= {count}"
            )
        return values
    raise SharedCPUCacheValidationError(
        f"{owner} has unsupported column encoding {kind!r}"
    )


def _positions_descriptor(positions: Optional[list[int]]) -> Any:
    if positions is None:
        return None
    if not all(type(value) is int for value in positions):
        return ["values", positions]
    if len(positions) <= 1:
        return [
            "range",
            int(positions[0]) if positions else 0,
            1,
            len(positions),
        ]
    start = int(positions[0])
    step = int(positions[1]) - start
    if all(
        int(value) == start + index * step
        for index, value in enumerate(positions)
    ):
        return ["range", start, step, len(positions)]
    # Preserve the original explicit list. The compact path must not mutate it,
    # and avoiding a second copy is important for irregular long contexts.
    return ["values", positions]


def _encode_positions_column(
    values: list[Optional[list[int]]],
) -> dict[str, Any]:
    descriptors = [_positions_descriptor(value) for value in values]
    if descriptors and all(
        isinstance(descriptor, list)
        and len(descriptor) == 4
        and descriptor[0] == "range"
        for descriptor in descriptors
    ):
        return {
            "kind": "ranges",
            "starts": _encode_exact_column(
                [int(descriptor[1]) for descriptor in descriptors]
            ),
            "steps": _encode_exact_column(
                [int(descriptor[2]) for descriptor in descriptors]
            ),
            "counts": _encode_exact_column(
                [int(descriptor[3]) for descriptor in descriptors]
            ),
        }
    return {
        "kind": "descriptors",
        "descriptors": _encode_exact_column(descriptors),
    }


def _decode_positions_column(
    encoded: dict[str, Any],
    count: int,
) -> list[Optional[list[int]]]:
    if not isinstance(encoded, dict):
        raise SharedCPUCacheValidationError(
            "Compact shared handle cached_positions must be a dict"
        )
    kind = encoded.get("kind")
    if kind == "ranges":
        _require_exact_fields(
            encoded,
            {"kind", "starts", "steps", "counts"},
            "Compact shared handle cached_positions",
        )
        starts = _decode_exact_column(encoded["starts"], count, "position starts")
        steps = _decode_exact_column(encoded["steps"], count, "position steps")
        lengths = _decode_exact_column(encoded["counts"], count, "position counts")
        decoded: list[Optional[list[int]]] = []
        for start, step, length in zip(starts, steps, lengths, strict=True):
            start = int(start)
            step = int(step)
            length = int(length)
            if length < 0:
                raise SharedCPUCacheValidationError(
                    "Compact shared handle position count must be non-negative"
                )
            decoded.append([start + index * step for index in range(length)])
        return decoded
    if kind != "descriptors":
        raise SharedCPUCacheValidationError(
            f"Unsupported cached_positions encoding {kind!r}"
        )
    _require_exact_fields(
        encoded,
        {"kind", "descriptors"},
        "Compact shared handle cached_positions",
    )
    descriptors = _decode_exact_column(
        encoded["descriptors"],
        count,
        "position descriptors",
    )
    decoded = []
    for descriptor in descriptors:
        if descriptor is None:
            decoded.append(None)
        elif (
            isinstance(descriptor, list)
            and len(descriptor) == 4
            and descriptor[0] == "range"
        ):
            start = int(descriptor[1])
            step = int(descriptor[2])
            length = int(descriptor[3])
            if length < 0:
                raise SharedCPUCacheValidationError(
                    "Compact shared handle position count must be non-negative"
                )
            decoded.append([start + index * step for index in range(length)])
        elif (
            isinstance(descriptor, list)
            and len(descriptor) == 2
            and descriptor[0] == "values"
            and isinstance(descriptor[1], list)
        ):
            decoded.append(list(descriptor[1]))
        else:
            raise SharedCPUCacheValidationError(
                f"Invalid compact cached_positions descriptor {descriptor!r}"
            )
    return decoded


def _encode_key_column(keys: list[CacheEngineKey]) -> dict[str, Any]:
    if not keys:
        return {"kind": "objects", "values": []}
    key_type = type(keys[0])
    supported = key_type in (CacheEngineKey, LayerCacheEngineKey)
    common_fields = (
        "model_name",
        "world_size",
        "worker_id",
        "dtype",
        "request_configs",
        "kv_group",
    )
    if supported and all(
        type(key) is key_type
        and all(
            _values_exactly_equal(
                getattr(key, field), getattr(keys[0], field)
            )
            for field in common_fields
        )
        and (
            key_type is CacheEngineKey
            or int(getattr(key, "layer_id")) == int(getattr(keys[0], "layer_id"))
        )
        for key in keys
    ):
        template = {
            "key_type": (
                "layer" if key_type is LayerCacheEngineKey else "chunk"
            ),
            "model_name": keys[0].model_name,
            "world_size": int(keys[0].world_size),
            "worker_id": int(keys[0].worker_id),
            "dtype": _dtype_to_str(keys[0].dtype),
            "request_configs": keys[0].request_configs,
            "kv_group": int(keys[0].kv_group),
            "layer_id": (
                int(keys[0].layer_id)
                if key_type is LayerCacheEngineKey
                else None
            ),
        }
        return {
            "kind": "template_hashes",
            "template": template,
            "chunk_hashes": [key.chunk_hash for key in keys],
        }
    return {"kind": "objects", "values": keys}


def _decode_key_column(
    encoded: dict[str, Any],
    count: int,
) -> list[CacheEngineKey]:
    if not isinstance(encoded, dict):
        raise SharedCPUCacheValidationError(
            "Compact shared handle keys must be a dict"
        )
    kind = encoded.get("kind")
    if kind == "objects":
        _require_exact_fields(
            encoded, {"kind", "values"}, "Compact shared handle keys"
        )
        values = encoded["values"]
        if (
            not isinstance(values, list)
            or len(values) != count
            or not all(isinstance(value, CacheEngineKey) for value in values)
        ):
            raise SharedCPUCacheValidationError(
                "Compact shared handle key object count/type mismatch"
            )
        return values
    if kind != "template_hashes":
        raise SharedCPUCacheValidationError(
            f"Unsupported compact shared handle key encoding {kind!r}"
        )
    _require_exact_fields(
        encoded,
        {"kind", "template", "chunk_hashes"},
        "Compact shared handle keys",
    )
    template = encoded["template"]
    if not isinstance(template, dict):
        raise SharedCPUCacheValidationError(
            "Compact shared handle key template must be a dict"
        )
    _require_exact_fields(
        template,
        {
            "key_type",
            "model_name",
            "world_size",
            "worker_id",
            "dtype",
            "request_configs",
            "kv_group",
            "layer_id",
        },
        "Compact shared handle key template",
    )
    hashes = encoded["chunk_hashes"]
    if not isinstance(hashes, list) or len(hashes) != count:
        raise SharedCPUCacheValidationError(
            "Compact shared handle key hash count mismatch"
        )
    dtype = _dtype_from_str(template["dtype"])
    if dtype is None:
        raise SharedCPUCacheValidationError(
            "Compact shared handle key dtype cannot be None"
        )
    key_type = template["key_type"]
    if key_type not in ("chunk", "layer"):
        raise SharedCPUCacheValidationError(
            f"Unsupported compact shared handle key type {key_type!r}"
        )
    keys: list[CacheEngineKey] = []
    for chunk_hash in hashes:
        kwargs = {
            "model_name": template["model_name"],
            "world_size": int(template["world_size"]),
            "worker_id": int(template["worker_id"]),
            "chunk_hash": chunk_hash,
            "dtype": dtype,
            "request_configs": template["request_configs"],
            "kv_group": int(template["kv_group"]),
        }
        if key_type == "layer":
            keys.append(
                LayerCacheEngineKey(
                    **kwargs,
                    layer_id=int(template["layer_id"]),
                )
            )
        else:
            if template["layer_id"] is not None:
                raise SharedCPUCacheValidationError(
                    "Compact chunk key cannot carry layer_id"
                )
            keys.append(CacheEngineKey(**kwargs))
    return keys


@dataclass(frozen=True)
class SharedChunkHandle:
    """Serializable metadata for one rank0-published shared CPU chunk.

    The handle carries slab-relative offsets and logical tensor metadata only.
    Raw host pointers, device pointers, Python object identity, and allocator
    internals must never be published.
    """

    request_id: str
    phase: str
    key: CacheEngineKey
    layer_id: int
    kv_group: int
    chunk_index: int
    shm_name: str
    offset: int
    physical_size: int
    logical_size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    generation: int
    producer_rank: int
    status: str = "ok"
    shapes: Optional[list[torch.Size]] = None
    dtypes: Optional[list[torch.dtype]] = None
    cached_positions: Optional[list[int]] = None

    @classmethod
    def from_memory_obj(
        cls,
        *,
        request_id: str,
        phase: str,
        key: CacheEngineKey,
        layer_id: int,
        kv_group: int,
        chunk_index: int,
        shm_name: str,
        memory_obj: MemoryObj,
        generation: int,
        producer_rank: int,
    ) -> "SharedChunkHandle":
        dtype = memory_obj.get_dtype()
        if dtype is None:
            raise SharedCPUCacheValidationError(
                "SharedChunkHandle requires tensor dtype; "
                f"request_id={request_id}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        meta = memory_obj.metadata
        return cls(
            request_id=request_id,
            phase=phase,
            key=key,
            layer_id=layer_id,
            kv_group=kv_group,
            chunk_index=chunk_index,
            shm_name=shm_name,
            offset=int(meta.address),
            physical_size=int(meta.phy_size),
            logical_size=int(memory_obj.get_size()),
            shape=torch.Size(memory_obj.get_shape()),
            dtype=dtype,
            fmt=memory_obj.get_memory_format(),
            generation=int(generation),
            producer_rank=int(producer_rank),
            shapes=[torch.Size(s) for s in meta.shapes] if meta.shapes else None,
            dtypes=list(meta.dtypes) if meta.dtypes else None,
            cached_positions=_positions_to_list(meta.cached_positions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "key": self.key,
            "layer_id": self.layer_id,
            "kv_group": self.kv_group,
            "chunk_index": self.chunk_index,
            "shm_name": self.shm_name,
            "offset": self.offset,
            "physical_size": self.physical_size,
            "logical_size": self.logical_size,
            "shape": list(self.shape),
            "dtype": _dtype_to_str(self.dtype),
            "shapes": [list(shape) for shape in self.shapes]
            if self.shapes
            else None,
            "dtypes": [_dtype_to_str(dtype) for dtype in self.dtypes]
            if self.dtypes
            else None,
            "fmt": self.fmt.value,
            "cached_positions": self.cached_positions,
            "generation": self.generation,
            "producer_rank": self.producer_rank,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedChunkHandle":
        if not isinstance(data, dict):
            raise SharedCPUCacheValidationError(
                "SharedChunkHandle expected dict payload, "
                f"got {type(data)!r}"
            )
        _reject_private_fields(
            data,
            _FORBIDDEN_TRANSPORT_FIELDS,
            "SharedChunkHandle",
        )
        _require_fields(
            data,
            {
                "request_id",
                "phase",
                "key",
                "layer_id",
                "kv_group",
                "chunk_index",
                "shm_name",
                "offset",
                "physical_size",
                "logical_size",
                "shape",
                "dtype",
                "shapes",
                "dtypes",
                "fmt",
                "cached_positions",
                "generation",
                "producer_rank",
                "status",
            },
            "SharedChunkHandle",
        )
        shapes_data = data.get("shapes")
        dtypes_data = data.get("dtypes")
        return cls(
            request_id=data["request_id"],
            phase=data["phase"],
            key=data["key"],
            layer_id=int(data["layer_id"]),
            kv_group=int(data["kv_group"]),
            chunk_index=int(data["chunk_index"]),
            shm_name=data["shm_name"],
            offset=int(data["offset"]),
            physical_size=int(data["physical_size"]),
            logical_size=int(data["logical_size"]),
            shape=torch.Size(data["shape"]),
            dtype=_dtype_from_str(data["dtype"]),  # type: ignore[arg-type]
            shapes=[torch.Size(shape) for shape in shapes_data]
            if shapes_data
            else None,
            dtypes=[_dtype_from_str(dtype) for dtype in dtypes_data]
            if dtypes_data
            else None,
            fmt=MemoryFormat(data["fmt"]),
            cached_positions=data["cached_positions"],
            generation=int(data["generation"]),
            producer_rank=int(data["producer_rank"]),
            status=data["status"],
        )


@dataclass(frozen=True)
class SharedHandleEnvelope:
    """Ordered rank0 broadcast envelope for one request/layer/group."""

    request_id: str
    phase: str
    request_ordinal: int
    layer_id: int
    kv_group: int
    status: str
    generation: int
    handles: list[SharedChunkHandle]
    message: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None

    def to_compact_dict(self) -> dict[str, Any]:
        """Return an exact columnar wire representation of this envelope.

        Fields common to the envelope are validated and omitted from each
        handle. Repeated and arithmetic columns are compressed only after an
        exact equality check; arbitrary values remain explicit. Decoding
        recreates the same ``SharedChunkHandle`` values and order.
        """
        handles = self.handles
        for handle in handles:
            common_failures = []
            if handle.request_id != self.request_id:
                common_failures.append("request_id")
            if handle.phase != self.phase:
                common_failures.append("phase")
            if int(handle.layer_id) != int(self.layer_id):
                common_failures.append("layer_id")
            if int(handle.kv_group) != int(self.kv_group):
                common_failures.append("kv_group")
            if int(handle.generation) != int(self.generation):
                common_failures.append("generation")
            if common_failures:
                raise SharedCPUCacheValidationError(
                    "Compact shared handle envelope has non-common fields: "
                    f"{common_failures}; request_id={self.request_id}, "
                    f"layer_id={self.layer_id}, kv_group={self.kv_group}, "
                    f"chunk_index={handle.chunk_index}"
                )

        columns = {
            "keys": _encode_key_column([handle.key for handle in handles]),
            "chunk_indices": _encode_exact_column(
                [int(handle.chunk_index) for handle in handles]
            ),
            "shm_names": _encode_exact_column(
                [handle.shm_name for handle in handles]
            ),
            "offsets": _encode_exact_column(
                [int(handle.offset) for handle in handles]
            ),
            "physical_sizes": _encode_exact_column(
                [int(handle.physical_size) for handle in handles]
            ),
            "logical_sizes": _encode_exact_column(
                [int(handle.logical_size) for handle in handles]
            ),
            "shapes": _encode_exact_column(
                [tuple(int(dim) for dim in handle.shape) for handle in handles]
            ),
            "dtypes": _encode_exact_column(
                [_dtype_to_str(handle.dtype) for handle in handles]
            ),
            "formats": _encode_exact_column(
                [handle.fmt.value for handle in handles]
            ),
            "nested_shapes": _encode_exact_column(
                [
                    (
                        tuple(
                            tuple(int(dim) for dim in shape)
                            for shape in handle.shapes
                        )
                        if handle.shapes is not None
                        else None
                    )
                    for handle in handles
                ]
            ),
            "nested_dtypes": _encode_exact_column(
                [
                    (
                        tuple(_dtype_to_str(dtype) for dtype in handle.dtypes)
                        if handle.dtypes is not None
                        else None
                    )
                    for handle in handles
                ]
            ),
            "cached_positions": _encode_positions_column(
                [handle.cached_positions for handle in handles]
            ),
            "producer_ranks": _encode_exact_column(
                [int(handle.producer_rank) for handle in handles]
            ),
            "statuses": _encode_exact_column(
                [handle.status for handle in handles]
            ),
        }
        return {
            "__type__": _COMPACT_ENVELOPE_TYPE,
            "request_id": self.request_id,
            "phase": self.phase,
            "request_ordinal": int(self.request_ordinal),
            "layer_id": int(self.layer_id),
            "kv_group": int(self.kv_group),
            "status": self.status,
            "generation": int(self.generation),
            "handle_count": len(handles),
            "handle_columns": columns,
            "message": self.message,
            "error_details": self.error_details,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "request_ordinal": self.request_ordinal,
            "layer_id": self.layer_id,
            "kv_group": self.kv_group,
            "status": self.status,
            "generation": self.generation,
            "handles": [handle.to_dict() for handle in self.handles],
            "message": self.message,
            "error_details": self.error_details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedHandleEnvelope":
        if not isinstance(data, dict):
            raise SharedCPUCacheValidationError(
                "SharedHandleEnvelope expected dict payload, "
                f"got {type(data)!r}"
            )
        _reject_private_fields(
            data,
            _FORBIDDEN_TRANSPORT_FIELDS,
            "SharedHandleEnvelope",
        )
        _require_fields(
            data,
            {
                "request_id",
                "phase",
                "request_ordinal",
                "layer_id",
                "kv_group",
                "status",
                "generation",
                "handles",
                "message",
                "error_details",
            },
            "SharedHandleEnvelope",
        )
        if data["status"] not in ("ok", "miss", "skipped", "error"):
            raise SharedCPUCacheValidationError(
                "SharedHandleEnvelope has unsupported status "
                f"{data['status']!r}"
            )
        if not isinstance(data["handles"], list):
            raise SharedCPUCacheValidationError(
                "SharedHandleEnvelope handles must be a list, "
                f"got {type(data['handles'])!r}"
            )
        return cls(
            request_id=data["request_id"],
            phase=data["phase"],
            request_ordinal=int(data["request_ordinal"]),
            layer_id=int(data["layer_id"]),
            kv_group=int(data["kv_group"]),
            status=data["status"],
            generation=int(data["generation"]),
            handles=[
                SharedChunkHandle.from_dict(handle)
                for handle in data["handles"]
            ],
            message=data["message"],
            error_details=data["error_details"],
        )

    @classmethod
    def from_wire_dict(cls, data: dict[str, Any]) -> "SharedHandleEnvelope":
        """Decode either the legacy row format or exact compact format."""
        if not isinstance(data, dict):
            raise SharedCPUCacheValidationError(
                "SharedHandleEnvelope expected dict payload, "
                f"got {type(data)!r}"
            )
        wire_type = data.get("__type__")
        if wire_type is None:
            return cls.from_dict(data)
        if wire_type != _COMPACT_ENVELOPE_TYPE:
            raise SharedCPUCacheValidationError(
                f"Unsupported shared handle envelope wire type {wire_type!r}"
            )
        _require_exact_fields(
            data,
            {
                "__type__",
                "request_id",
                "phase",
                "request_ordinal",
                "layer_id",
                "kv_group",
                "status",
                "generation",
                "handle_count",
                "handle_columns",
                "message",
                "error_details",
            },
            "Compact SharedHandleEnvelope",
        )
        _reject_private_fields(
            data,
            _FORBIDDEN_TRANSPORT_FIELDS,
            "Compact SharedHandleEnvelope",
        )
        status = data["status"]
        if status not in ("ok", "miss", "skipped", "error"):
            raise SharedCPUCacheValidationError(
                "Compact SharedHandleEnvelope has unsupported status "
                f"{status!r}"
            )
        count = int(data["handle_count"])
        if count < 0:
            raise SharedCPUCacheValidationError(
                "Compact SharedHandleEnvelope handle_count must be non-negative"
            )
        columns = data["handle_columns"]
        if not isinstance(columns, dict):
            raise SharedCPUCacheValidationError(
                "Compact SharedHandleEnvelope handle_columns must be a dict"
            )
        _require_exact_fields(
            columns,
            {
                "keys",
                "chunk_indices",
                "shm_names",
                "offsets",
                "physical_sizes",
                "logical_sizes",
                "shapes",
                "dtypes",
                "formats",
                "nested_shapes",
                "nested_dtypes",
                "cached_positions",
                "producer_ranks",
                "statuses",
            },
            "Compact SharedHandleEnvelope columns",
        )
        keys = _decode_key_column(columns["keys"], count)
        chunk_indices = _decode_exact_column(
            columns["chunk_indices"], count, "handle chunk_indices"
        )
        shm_names = _decode_exact_column(
            columns["shm_names"], count, "handle shm_names"
        )
        offsets = _decode_exact_column(
            columns["offsets"], count, "handle offsets"
        )
        physical_sizes = _decode_exact_column(
            columns["physical_sizes"], count, "handle physical_sizes"
        )
        logical_sizes = _decode_exact_column(
            columns["logical_sizes"], count, "handle logical_sizes"
        )
        shapes = _decode_exact_column(
            columns["shapes"], count, "handle shapes"
        )
        dtypes = _decode_exact_column(
            columns["dtypes"], count, "handle dtypes"
        )
        formats = _decode_exact_column(
            columns["formats"], count, "handle formats"
        )
        nested_shapes = _decode_exact_column(
            columns["nested_shapes"], count, "handle nested_shapes"
        )
        nested_dtypes = _decode_exact_column(
            columns["nested_dtypes"], count, "handle nested_dtypes"
        )
        cached_positions = _decode_positions_column(
            columns["cached_positions"], count
        )
        producer_ranks = _decode_exact_column(
            columns["producer_ranks"], count, "handle producer_ranks"
        )
        handle_statuses = _decode_exact_column(
            columns["statuses"], count, "handle statuses"
        )

        handles: list[SharedChunkHandle] = []
        for index in range(count):
            dtype = _dtype_from_str(dtypes[index])
            if dtype is None:
                raise SharedCPUCacheValidationError(
                    "Compact shared handle dtype cannot be None"
                )
            encoded_nested_shapes = nested_shapes[index]
            encoded_nested_dtypes = nested_dtypes[index]
            decoded_nested_shapes = (
                [torch.Size(shape) for shape in encoded_nested_shapes]
                if encoded_nested_shapes is not None
                else None
            )
            decoded_nested_dtypes = None
            if encoded_nested_dtypes is not None:
                decoded_nested_dtypes = []
                for encoded_dtype in encoded_nested_dtypes:
                    nested_dtype = _dtype_from_str(encoded_dtype)
                    if nested_dtype is None:
                        raise SharedCPUCacheValidationError(
                            "Compact nested handle dtype cannot be None"
                        )
                    decoded_nested_dtypes.append(nested_dtype)
            handles.append(
                SharedChunkHandle(
                    request_id=data["request_id"],
                    phase=data["phase"],
                    key=keys[index],
                    layer_id=int(data["layer_id"]),
                    kv_group=int(data["kv_group"]),
                    chunk_index=int(chunk_indices[index]),
                    shm_name=shm_names[index],
                    offset=int(offsets[index]),
                    physical_size=int(physical_sizes[index]),
                    logical_size=int(logical_sizes[index]),
                    shape=torch.Size(shapes[index]),
                    dtype=dtype,
                    fmt=MemoryFormat(formats[index]),
                    generation=int(data["generation"]),
                    producer_rank=int(producer_ranks[index]),
                    status=handle_statuses[index],
                    shapes=decoded_nested_shapes,
                    dtypes=decoded_nested_dtypes,
                    cached_positions=cached_positions[index],
                )
            )
        return cls(
            request_id=data["request_id"],
            phase=data["phase"],
            request_ordinal=int(data["request_ordinal"]),
            layer_id=int(data["layer_id"]),
            kv_group=int(data["kv_group"]),
            status=status,
            generation=int(data["generation"]),
            handles=handles,
            message=data["message"],
            error_details=data["error_details"],
        )


def validate_shared_handle(
    handle: SharedChunkHandle,
    *,
    expected_request_id: str,
    expected_phase: str,
    expected_layer_id: int,
    expected_kv_group: int,
    expected_shm_name: str,
    expected_generation: int,
    expected_chunk_index: Optional[int],
    slab_size: int,
    expected_key: Optional[CacheEngineKey] = None,
    expected_shape: Optional[torch.Size] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_fmt: Optional[MemoryFormat] = None,
    expected_cached_positions: Optional[Iterable[int]] = None,
    expected_producer_rank: Optional[int] = None,
) -> None:
    """Validate a handle before passive view creation."""

    failures: list[str] = []
    if handle.status != "ok":
        failures.append(f"status={handle.status!r}")
    if handle.request_id != expected_request_id:
        failures.append(
            f"request_id={handle.request_id!r}, expected={expected_request_id!r}"
        )
    if handle.phase != expected_phase:
        failures.append(f"phase={handle.phase!r}, expected={expected_phase!r}")
    if handle.layer_id != expected_layer_id:
        failures.append(
            f"layer_id={handle.layer_id}, expected={expected_layer_id}"
        )
    if handle.kv_group != expected_kv_group:
        failures.append(
            f"kv_group={handle.kv_group}, expected={expected_kv_group}"
        )
    if handle.shm_name != expected_shm_name:
        failures.append(
            f"shm_name={handle.shm_name!r}, expected={expected_shm_name!r}"
        )
    if handle.generation != expected_generation:
        failures.append(
            f"generation={handle.generation}, expected={expected_generation}"
        )
    if (
        expected_producer_rank is not None
        and handle.producer_rank != int(expected_producer_rank)
    ):
        failures.append(
            f"producer_rank={handle.producer_rank}, "
            f"expected={int(expected_producer_rank)}"
        )
    if (
        expected_chunk_index is not None
        and handle.chunk_index != expected_chunk_index
    ):
        failures.append(
            f"chunk_index={handle.chunk_index}, expected={expected_chunk_index}"
        )
    if (
        expected_key is not None
        and not _shared_key_matches_expected(
            handle.key,
            expected_key,
            expected_producer_rank,
        )
    ):
        failures.append(f"key={handle.key!r}, expected={expected_key!r}")
    if expected_shape is not None and handle.shape != torch.Size(expected_shape):
        failures.append(f"shape={handle.shape}, expected={torch.Size(expected_shape)}")
    if expected_dtype is not None and handle.dtype != expected_dtype:
        failures.append(f"dtype={handle.dtype}, expected={expected_dtype}")
    if expected_fmt is not None and handle.fmt != expected_fmt:
        failures.append(f"fmt={handle.fmt}, expected={expected_fmt}")
    if handle.offset < 0:
        failures.append(f"offset={handle.offset} must be non-negative")
    if handle.logical_size <= 0:
        failures.append(f"logical_size={handle.logical_size} must be positive")
    if handle.physical_size <= 0:
        failures.append(f"physical_size={handle.physical_size} must be positive")
    if handle.logical_size > handle.physical_size:
        failures.append(
            f"logical_size={handle.logical_size} exceeds "
            f"physical_size={handle.physical_size}"
        )
    if handle.offset + handle.physical_size > slab_size:
        failures.append(
            f"bounds [{handle.offset}, {handle.offset + handle.physical_size}) "
            f"exceed slab_size={slab_size}"
        )
    if handle.dtype is None:
        failures.append("dtype is None")
    if handle.fmt == MemoryFormat.UNDEFINED:
        failures.append("fmt is UNDEFINED")
    if handle.cached_positions is not None:
        try:
            cached_positions = [int(pos) for pos in handle.cached_positions]
            if any(pos < 0 for pos in cached_positions):
                failures.append("cached_positions contains negative offsets")
            if expected_cached_positions is not None:
                positions_match, expected_for_log = _expected_positions_match(
                    cached_positions,
                    expected_cached_positions,
                )
            else:
                positions_match = True
                expected_for_log = None
            if not positions_match:
                failures.append(
                    f"cached_positions={cached_positions}, "
                    f"expected={expected_for_log}"
                )
        except Exception as exc:
            failures.append(f"invalid cached_positions metadata: {exc}")
    try:
        shape_bytes = _handle_logical_nbytes(handle)
        if shape_bytes != handle.logical_size:
            failures.append(
                f"shape/dtype bytes={shape_bytes} do not match "
                f"logical_size={handle.logical_size}"
            )
    except Exception as exc:
        failures.append(f"invalid shape/dtype metadata: {exc}")

    if failures:
        raise SharedCPUCacheValidationError(
            "Invalid shared CPU cache handle before passive view creation: "
            + "; ".join(failures)
        )


class PassiveSharedViewAllocator(MemoryAllocatorInterface):
    """Allocator for passive-rank shm views.

    It never owns the backing address space. Freeing a passive view invalidates
    the local MemoryObj only; it must not return offsets to any AddressManager.
    """

    def __init__(
        self,
        *,
        slab_tensor: torch.Tensor,
        shm_name: str,
        generation: int,
    ) -> None:
        self.slab_tensor = slab_tensor.view(torch.uint8).flatten()
        self.shm_name = shm_name
        self.generation = int(generation)

    @property
    def slab_size(self) -> int:
        return int(self.slab_tensor.numel())

    def create_view(
        self,
        handle: SharedChunkHandle,
        *,
        expected_request_id: str,
        expected_phase: str,
        expected_layer_id: int,
        expected_kv_group: int,
        expected_chunk_index: Optional[int] = None,
        expected_key: Optional[CacheEngineKey] = None,
        expected_shape: Optional[torch.Size] = None,
        expected_dtype: Optional[torch.dtype] = None,
        expected_fmt: Optional[MemoryFormat] = None,
        expected_cached_positions: Optional[Iterable[int]] = None,
        expected_producer_rank: Optional[int] = None,
    ) -> TensorMemoryObj:
        validate_shared_handle(
            handle,
            expected_request_id=expected_request_id,
            expected_phase=expected_phase,
            expected_layer_id=expected_layer_id,
            expected_kv_group=expected_kv_group,
            expected_shm_name=self.shm_name,
            expected_generation=self.generation,
            expected_chunk_index=expected_chunk_index,
            expected_key=expected_key,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            expected_fmt=expected_fmt,
            expected_cached_positions=expected_cached_positions,
            expected_producer_rank=expected_producer_rank,
            slab_size=self.slab_size,
        )
        raw_data = self.slab_tensor[
            handle.offset : handle.offset + handle.logical_size
        ]
        cached_positions = (
            torch.tensor(handle.cached_positions, dtype=torch.int64)
            if handle.cached_positions is not None
            else None
        )
        metadata = MemoryObjMetadata(
            shape=handle.shape,
            dtype=handle.dtype,
            address=handle.offset,
            phy_size=handle.physical_size,
            ref_count=1,
            pin_count=0,
            fmt=handle.fmt,
            cached_positions=cached_positions,
            shapes=handle.shapes,
            dtypes=handle.dtypes,
        )
        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=metadata,
            parent_allocator=self,
        )

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        raise SharedCPUCacheError("PassiveSharedViewAllocator cannot allocate")

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[list[MemoryObj]]:
        raise SharedCPUCacheError(
            "PassiveSharedViewAllocator cannot allocate batches"
        )

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        if not memory_obj.is_valid():
            logger.warning(
                "Double-free of passive shared CPU view ignored: "
                "shm_name=%s address=%s size=%s generation=%s",
                self.shm_name,
                memory_obj.metadata.address,
                memory_obj.metadata.phy_size,
                self.generation,
            )
            return
        memory_obj.invalidate()

    def batched_free(
        self,
        memory_objs: list[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        for memory_obj in memory_objs:
            self.free(memory_obj, allocator_type)


class SharedSlabMapping:
    """Native shared slab mapping for one process.

    Rank0 normally owns the mapping through LocalCPUBackend's shm-backed
    allocator. Passive ranks use this class to attach/register their local
    view and then build PassiveSharedViewAllocator on top of it.
    """

    def __init__(
        self,
        *,
        shm_name: str,
        size: int,
        ptr: int,
        tensor: torch.Tensor,
        generation: int,
        owner: bool,
        backing_buffer: Optional[Any] = None,
    ) -> None:
        self.shm_name = shm_name
        self.size = int(size)
        self.ptr = int(ptr)
        self.tensor = tensor
        self.generation = int(generation)
        self.owner = owner
        self._backing_buffer = backing_buffer
        self._closed = False

    @staticmethod
    def _tensor_from_ptr(ptr: int, size: int) -> tuple[torch.Tensor, Any]:
        array_type = ctypes.c_uint8 * size
        buf = array_type.from_address(ptr)
        return torch.frombuffer(buf, dtype=torch.uint8), buf

    @classmethod
    def attach(
        cls,
        *,
        shm_name: str,
        size: int,
        generation: int,
        writable: bool = True,
    ) -> "SharedSlabMapping":
        lmc_ops = _load_lmc_ops(purpose="attach")

        if not hasattr(lmc_ops, "attach_shm_pinned_ptr"):
            raise SharedCPUCacheError(
                "lmcache.c_ops.attach_shm_pinned_ptr is unavailable; "
                "rebuild LMCache/LMCache-Ascend with shared CPU cache hooks."
            )
        size = int(size)
        if size <= 0:
            raise SharedCPUCacheError(
                "Shared CPU cache attach received invalid slab size "
                f"{size} for shm_name={shm_name}, generation={generation}."
            )
        ptr = int(lmc_ops.attach_shm_pinned_ptr(size, shm_name, writable))
        if ptr == 0:
            raise SharedCPUCacheError(
                "Shared CPU cache attach failed: attach_shm_pinned_ptr "
                f"returned 0 for shm_name={shm_name}, size={size}, "
                f"generation={generation}, writable={writable}."
            )
        if ptr < 0:
            raise SharedCPUCacheError(
                "Shared CPU cache attach failed: attach_shm_pinned_ptr "
                f"returned invalid host pointer {ptr} for shm_name={shm_name}, "
                f"size={size}, generation={generation}, writable={writable}."
            )
        try:
            tensor, backing_buffer = cls._tensor_from_ptr(ptr, size)
        except Exception:
            try:
                lmc_ops.detach_shm_pinned_ptr(ptr, size)
            except Exception:
                logger.exception(
                    "Failed to detach shared CPU cache mapping after tensor "
                    "view creation failure: shm_name=%s, size=%s, "
                    "generation=%s, writable=%s",
                    shm_name,
                    size,
                    generation,
                    writable,
                )
            raise
        return cls(
            shm_name=shm_name,
            size=size,
            ptr=ptr,
            tensor=tensor,
            generation=generation,
            owner=False,
            backing_buffer=backing_buffer,
        )

    @classmethod
    def from_rank0_allocator(
        cls,
        *,
        shm_name: str,
        allocator_tensor: torch.Tensor,
        generation: int,
    ) -> "SharedSlabMapping":
        tensor = allocator_tensor.view(torch.uint8).flatten()
        size = int(tensor.numel())
        ptr = int(tensor.data_ptr())
        if size <= 0:
            raise SharedCPUCacheError(
                "Shared CPU cache rank0 allocator has invalid buffer size "
                f"{size} for shm_name={shm_name}, generation={generation}."
            )
        if ptr == 0:
            raise SharedCPUCacheError(
                "Shared CPU cache rank0 allocator has invalid buffer pointer "
                f"0 for shm_name={shm_name}, size={size}, "
                f"generation={generation}."
            )
        return cls(
            shm_name=shm_name,
            size=size,
            ptr=ptr,
            tensor=tensor,
            generation=generation,
            owner=True,
        )

    def passive_allocator(self) -> PassiveSharedViewAllocator:
        return PassiveSharedViewAllocator(
            slab_tensor=self.tensor,
            shm_name=self.shm_name,
            generation=self.generation,
        )

    def preflight_device_ptr(self) -> int:
        lmc_ops = _load_lmc_ops(purpose="preflight")

        if not hasattr(lmc_ops, "get_device_ptr"):
            raise SharedCPUCacheError(
                "lmcache.c_ops.get_device_ptr is unavailable for shared CPU "
                "cache preflight."
            )
        raw_dev_ptr = lmc_ops.get_device_ptr(self.ptr)
        if raw_dev_ptr is None:
            raise SharedCPUCacheError(
                "Shared CPU cache preflight failed: get_device_ptr returned "
                f"None for shm_name={self.shm_name}, size={self.size}, "
                f"generation={self.generation}."
            )
        dev_ptr = int(raw_dev_ptr)
        if dev_ptr == 0:
            raise SharedCPUCacheError(
                "Shared CPU cache preflight failed: get_device_ptr returned 0 "
                f"for shm_name={self.shm_name}, size={self.size}, "
                f"generation={self.generation}."
            )
        return dev_ptr

    def close(self) -> None:
        if self._closed:
            return
        if self.owner:
            self.unlink(self.shm_name)
        else:
            lmc_ops = _load_lmc_ops(purpose="detach")
            if hasattr(lmc_ops, "detach_shm_pinned_ptr"):
                lmc_ops.detach_shm_pinned_ptr(self.ptr, self.size)
            else:
                raise SharedCPUCacheError(
                    "lmcache.c_ops.detach_shm_pinned_ptr is unavailable; "
                    "cannot safely detach passive shared CPU cache mapping."
                )
        self._closed = True

    @staticmethod
    def unlink(shm_name: str) -> None:
        lmc_ops = _load_lmc_ops(purpose="unlink")
        if hasattr(lmc_ops, "unlink_shm"):
            lmc_ops.unlink_shm(shm_name)
            return
        raise SharedCPUCacheError(
            "lmcache.c_ops.unlink_shm is unavailable; rebuild native hooks."
        )
