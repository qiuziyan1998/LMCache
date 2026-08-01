# SPDX-License-Identifier: Apache-2.0
"""Shared CPU cache request ownership, handles, and passive-view primitives.

This module intentionally contains no storage-tier policy. Rank0 resolves real
MemoryObjs through the existing StorageManager/LocalCPUBackend path, publishes
metadata handles, and passive ranks build view-only MemoryObjs from those
handles after strict validation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
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


@dataclass
class SharedCPURequestLease:
    """Own the shared MemoryObjs retained for one live request."""

    request_id: str
    generation: int
    is_rank0: bool
    active: bool = False
    groups: dict[int, list[list[MemoryObj]]] = field(default_factory=dict)

    @staticmethod
    def _unique_objects(
        groups: dict[int, list[list[MemoryObj]]],
    ) -> list[MemoryObj]:
        seen: set[int] = set()
        objects: list[MemoryObj] = []
        for layers in groups.values():
            for layer in layers:
                for memory_obj in layer:
                    identity = id(memory_obj)
                    if identity not in seen:
                        seen.add(identity)
                        objects.append(memory_obj)
        return objects

    @staticmethod
    def _is_valid(memory_obj: MemoryObj) -> bool:
        is_valid = getattr(memory_obj, "is_valid", None)
        return bool(is_valid()) if callable(is_valid) else True

    def _release(self, objects: Iterable[MemoryObj]) -> None:
        for memory_obj in objects:
            if self.is_rank0 and getattr(memory_obj, "is_pinned", False):
                try:
                    memory_obj.unpin()
                except Exception as exc:
                    logger.warning(
                        "Failed to unpin shared CPU request object: "
                        "request_id=%s error=%s",
                        self.request_id,
                        exc,
                    )
            try:
                if self._is_valid(memory_obj):
                    memory_obj.ref_count_down()
            except Exception as exc:
                logger.warning(
                    "Failed to release shared CPU request reference: "
                    "request_id=%s error=%s",
                    self.request_id,
                    exc,
                )

    def replace_groups(
        self,
        groups: dict[int, list[list[MemoryObj]]],
        *,
        retain: bool,
    ) -> None:
        """Replace selected groups, retaining borrowed objects when requested."""

        replacement = dict(self.groups)
        for kv_group, layers in groups.items():
            copied_layers = [list(layer) for layer in layers]
            if any(copied_layers):
                replacement[kv_group] = copied_layers
            else:
                replacement.pop(kv_group, None)

        old_objects = self._unique_objects(self.groups)
        old_ids = {id(memory_obj) for memory_obj in old_objects}
        new_objects = self._unique_objects(replacement)
        new_ids = {id(memory_obj) for memory_obj in new_objects}

        retained: list[MemoryObj] = []
        if retain:
            try:
                for memory_obj in new_objects:
                    if id(memory_obj) in old_ids:
                        continue
                    memory_obj.ref_count_up()
                    try:
                        if not self._is_valid(memory_obj):
                            raise RuntimeError(
                                "Cannot retain an invalid shared CPU "
                                "MemoryObj"
                            )
                        if self.is_rank0 and memory_obj.pin() is False:
                            raise RuntimeError("MemoryObj.pin() returned False")
                    except Exception:
                        if self._is_valid(memory_obj):
                            memory_obj.ref_count_down()
                        raise
                    retained.append(memory_obj)
            except Exception:
                self._release(reversed(retained))
                raise

        self.groups = replacement
        self._release(
            memory_obj for memory_obj in old_objects if id(memory_obj) not in new_ids
        )

    def append_groups(
        self,
        groups: dict[int, list[list[MemoryObj]]],
        chunk_index_base: dict[int, int],
    ) -> None:
        """Adopt suffixes without rescanning or retaining the live prefix.

        Args:
            groups: Complete per-layer object lists after the append.
            chunk_index_base: Existing chunk count for each owned group.

        Raises:
            ValueError: If a live group is not aligned with its append point.
        """

        updates: list[tuple[list[list[MemoryObj]], list[list[MemoryObj]]]] = []
        additions: list[tuple[int, list[list[MemoryObj]]]] = []
        for kv_group, layers in groups.items():
            current = self.groups.get(kv_group)
            if current is None:
                additions.append((kv_group, [list(layer) for layer in layers]))
                continue
            if len(current) != len(layers):
                raise ValueError(
                    "Shared CPU request lease suffix has an incompatible "
                    f"layer count: kv_group={kv_group}, "
                    f"current={len(current)}, new={len(layers)}"
                )
            append_at = chunk_index_base[kv_group]
            if any(len(layer) != append_at for layer in current):
                raise ValueError(
                    "Shared CPU request lease is not append-aligned: "
                    f"kv_group={kv_group}, append_at={append_at}"
                )
            updates.append(
                (current, [list(layer[append_at:]) for layer in layers])
            )

        for current, suffix in updates:
            for layer, layer_suffix in zip(current, suffix, strict=True):
                layer.extend(layer_suffix)
        for kv_group, suffix in additions:
            self.groups[kv_group] = suffix

    def object_ids(self, kv_group: Optional[int] = None) -> set[int]:
        groups = (
            self.groups.values()
            if kv_group is None
            else (self.groups.get(kv_group, []),)
        )
        return {
            id(memory_obj)
            for layers in groups
            for layer in layers
            for memory_obj in layer
        }

    def close(self) -> None:
        objects = self._unique_objects(self.groups)
        self.groups.clear()
        self.active = False
        self._release(objects)


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
        for cached, expected in zip(
            cached_positions,
            expected_positions,
            strict=True,
        ):
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
    if type(handle_key) is type(expected_key) and handle_key == expected_key:
        return True
    if expected_producer_rank is None:
        return False
    if type(handle_key) is not type(expected_key):
        # A compact passive lookup may retain one base key per chunk instead
        # of materializing one LayerCacheEngineKey per layer. Layer identity
        # is validated independently by expected_layer_id.
        if not (
            hasattr(handle_key, 'layer_id')
            and not hasattr(expected_key, 'layer_id')
        ):
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
    for field_name in comparable_fields:
        if getattr(handle_key, field_name) != getattr(
            expected_key,
            field_name,
        ):
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
    except ImportError:
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
