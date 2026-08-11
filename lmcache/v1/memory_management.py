# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, wraps
from typing import Any, List, Optional, Tuple, Union
import abc
import ctypes
import os
import threading

# Third Party
from sortedcontainers import SortedList
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.system_detection import NUMAMapping

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops
else:
    # First Party
    import lmcache.non_cuda_equivalents as lmc_ops


logger = init_logger(__name__)


def _group_prefix_sums(
    shapes: Optional[list[torch.Size]],
    dtypes: Optional[list[torch.dtype]],
    size: int,
) -> tuple[int, ...]:
    if shapes is None or dtypes is None:
        return (0, size)
    prefix = [0]
    for shape, dtype in zip(shapes, dtypes, strict=True):
        prefix.append(prefix[-1] + shape.numel() * dtype.itemsize)
    return tuple(prefix)


# Helper functions for thread safety
def synchronized(lock_attr_name):
    """
    Decorator to make a method thread-safe by acquiring the lock
    specified by lock_attr_name on the instance.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr_name)
            with lock:
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    # KV_BLOB = 1
    KV_2LTD = auto()
    """[num_tokens, 2, hidden_dim]
    """
    # LAYER_KV_BLOB = 2
    KV_T2D = auto()
    """[2, num_tokens, hidden_dim]
    """

    KV_2TD = auto()
    """Compressed binary array format
    """
    BINARY = auto()

    BINARY_BUFFER = auto()

    KV_MLA_FMT = auto()
    """MLA plane-major layout: 1D stacked planes
    [k_nope plane: N*kH | k_pe plane: N*vH] per layer-chunk.

    Deprecated name — use :attr:`KV_MLA_LATENT_FMT` (same enum value).
    """

    KV_MLA_LATENT_FMT = KV_MLA_FMT
    """Canonical tag for MLA latent plane-major storage (identical to ``KV_MLA_FMT``).
    """

    KV_DSA_INDEX_FMT = auto()
    """DSA indexer-only (two-group DSA path): 1D single plane
    [indexer plane: N*dsaH] per layer-chunk.
    """

    def token_dim(self) -> int:
        if self == MemoryFormat.KV_2LTD:
            return 2
        elif self == MemoryFormat.KV_T2D:
            return 1
        elif self == MemoryFormat.KV_2TD:
            return 0
        elif self == MemoryFormat.BINARY:
            return 0
        elif self == MemoryFormat.BINARY_BUFFER:
            return 0
        elif self == MemoryFormat.KV_MLA_FMT:
            return 2
        elif self == MemoryFormat.KV_DSA_INDEX_FMT:
            return 2
        return 0


def _layer_page_shape(
    shape: torch.Size,
    fmt: MemoryFormat,
    valid_tokens: int,
    full_tokens: Optional[int] = None,
) -> torch.Size:
    """Resize one layer-page shape without changing its byte layout."""
    if valid_tokens < 1:
        raise ValueError("Layer-page valid_tokens must be positive")
    token_dim = fmt.token_dim()
    if token_dim < len(shape):
        source_tokens = int(shape[token_dim])
        if full_tokens is not None and source_tokens != full_tokens:
            raise ValueError("Layer-page full_tokens does not match its shape")
        if valid_tokens > source_tokens:
            raise ValueError("Layer-page valid_tokens exceeds its full shape")
        resized = list(shape)
        resized[token_dim] = valid_tokens
        return torch.Size(resized)
    if fmt not in (
        MemoryFormat.KV_MLA_LATENT_FMT,
        MemoryFormat.KV_DSA_INDEX_FMT,
    ):
        raise ValueError("Layer-page shape has no token dimension")
    source_tokens = full_tokens or int(shape[0])
    if valid_tokens > source_tokens or shape.numel() % source_tokens:
        raise ValueError("Flat layer-page shape is not token divisible")
    target_numel = shape.numel() // source_tokens * valid_tokens
    if len(shape) > 1 and int(shape[0]) == source_tokens:
        return torch.Size([valid_tokens, *shape[1:]])
    return torch.Size([target_numel])


@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators"""

    start: int
    size: int

    def can_be_coalesced(self, succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start


@dataclass
class MemoryObjMetadata:
    # TODO(chunxiaozheng): use shapes and dtypes to replace shape and dtype
    # The 'logical' shape of the tensor
    shape: torch.Size

    # The 'logical' dtype of the tensor
    dtype: Optional[torch.dtype]

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # Whether the object is pinned and cannot be evicted
    # lookup pins are temporary
    # cache controller pins are persistent
    pin_count: int = 0

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    # Positions when the cache is stored
    cached_positions: Optional[torch.Tensor] = None

    # shapes and dtypes should be used in the future
    shapes: Optional[list[torch.Size]] = None
    dtypes: Optional[list[torch.dtype]] = None

    # Authoritative logical token count for merged layer pages.
    valid_tokens: Optional[int] = None

    def to_dict(self):
        # Note(Kuntai): this is used for serializing MemoryObjMetadata via
        # msgpack.
        return {
            "__type__": "MemoryObjMetadata",
            "shape": list(self.shape),  # torch.Size -> list
            "dtype": str(self.dtype) if self.dtype else None,
            "address": self.address,
            "phy_size": self.phy_size,
            "ref_count": self.ref_count,
            "fmt": self.fmt.value,
            "shapes": [list(shape) for shape in self.shapes] if self.shapes else None,
            "dtypes": [str(dtype) for dtype in self.dtypes] if self.dtypes else None,
            "valid_tokens": self.valid_tokens,
        }

    @staticmethod
    def from_dict(d):
        dtype_str = d["dtype"]
        dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None
        shapes_list = d["shapes"]
        shapes = [torch.Size(s) for s in shapes_list] if shapes_list else None
        dtypes_list = d["dtypes"]
        dtypes = (
            [getattr(torch, d_str.replace("torch.", "")) for d_str in dtypes_list]
            if dtypes_list
            else None
        )
        return MemoryObjMetadata(
            shape=torch.Size(d["shape"]),
            dtype=dtype,
            address=d["address"],
            phy_size=d["phy_size"],
            ref_count=d["ref_count"],
            fmt=MemoryFormat(d["fmt"]),
            shapes=shapes,
            dtypes=dtypes,
            valid_tokens=d.get("valid_tokens"),
        )

    def get_size(self) -> int:
        if self.shapes is not None and self.dtypes is not None:
            return get_size_bytes(self.shapes, self.dtypes)
        return self.shape.numel() * self.dtype.itemsize  # type: ignore


@dataclass(frozen=True, slots=True)
class _TensorAllocationBatch:
    """Immutable provenance for one homogeneous tensor allocation."""

    parent: Any
    slab_size: int
    dtype: torch.dtype
    fmt: MemoryFormat
    addresses: tuple[int, ...]
    physical_size: int
    member_ids: tuple[int, ...]

    def matches(
        self,
        parent: Any,
        slab_size: int,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> bool:
        return (
            self.parent is parent
            and self.slab_size == slab_size
            and self.dtype == dtype
            and self.fmt == fmt
            and len(self.addresses) == len(self.member_ids)
            and self.physical_size > 0
            and all(
                address >= 0
                and address + self.physical_size <= slab_size
                for address in self.addresses
            )
        )


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    # subclasses should expose raw_data differently
    raw_data: Any

    def __init__(self, metadata: MemoryObjMetadata):
        self.meta = metadata

    @abc.abstractmethod
    def invalidate(self):
        """
        Invalidate the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self):
        """
        Check if the MemoryObj is valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the MemoryObj in bytes.
        Note that this number could be smaller than the physical size.
        The physical size is aligned to the allocator's alignment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        """
        Get the shape of the MemoryObj.
        """
        raise NotImplementedError

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Get the dtype of the MemoryObj.
        """
        return None

    @abc.abstractmethod
    def get_shapes(self) -> list[torch.Size]:
        """
        Get the shapes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dtypes(self) -> list[torch.dtype]:
        """
        Get the dtypes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_format(self) -> MemoryFormat:
        """
        Get the memory format of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_size(self) -> int:
        """
        Get the physical size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pin(self) -> bool:
        """
        Pin the memory obj so that it will not be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self):
        """
        Increase ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpin(self) -> bool:
        """
        Unpin the memory obj so that it can be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self):
        """
        Decrease ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self) -> int:
        """
        Get ref count for the given MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_tokens(self) -> int:
        """
        Get token number for the given MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def metadata(self) -> MemoryObjMetadata:
        """
        Get the metada of the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor(self) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def byte_array(self) -> bytes:
        """
        Get the byte array from the MemoryObj.
        The size is will be the physical size instead of the unaligned size.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_ptr(self) -> int:
        """
        Get the data pointer of the MemoryObj.
        This is used to access the raw data in the memory.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_pinned(self) -> bool:
        """
        Check whether the memory obj is pinned.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def raw_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the raw tensor from the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj at the given index(group).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        """
        Get the allocator that allocates this memory object
        """
        raise NotImplementedError


def _resolve_pinned_alloc_free(
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    size: Optional[int] = None,
    shm_interleave_nodes: Optional[tuple[int, ...]] = None,
) -> Tuple[
    tuple,  # (alloc_fn, *alloc_args)
    tuple,  # (free_fn, *free_args_after_ptr)
]:
    """Resolve the alloc/free function pair based on memory type.

    Returns:
        A tuple of (alloc_info, free_info) where:
        - alloc_info: (alloc_fn, *args) to call as alloc_fn(size, *args)
        - free_info: (free_fn, *args) to call as free_fn(ptr, *args)
    """
    if shm_name:
        alloc_args = (
            (shm_name, list(shm_interleave_nodes))
            if shm_interleave_nodes
            else (shm_name,)
        )
        return (
            (lmc_ops.alloc_shm_pinned_ptr, *alloc_args),
            (lmc_ops.free_shm_pinned_ptr, size, shm_name),
        )
    elif numa_mapping:
        if torch.cuda.is_available():
            current_device_id = torch.cuda.current_device()
        else:
            current_device_id = 0
        gpu_to_numa_mapping = numa_mapping.gpu_to_numa_mapping
        assert current_device_id in gpu_to_numa_mapping, (
            f"Current device {current_device_id} is not in the GPU NUMA mapping."
        )
        numa_id = gpu_to_numa_mapping[current_device_id]
        return (
            (lmc_ops.alloc_pinned_numa_ptr, numa_id),
            (lmc_ops.free_pinned_numa_ptr, size),
        )
    else:
        return (
            (lmc_ops.alloc_pinned_ptr, 0),
            (lmc_ops.free_pinned_ptr,),
        )


def _allocate_cpu_memory(
    size: int,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    shm_interleave_nodes: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    if size == 0:
        return torch.empty(0, dtype=torch.uint8)

    alloc_info, _ = _resolve_pinned_alloc_free(
        numa_mapping,
        shm_name,
        shm_interleave_nodes=shm_interleave_nodes,
    )
    alloc_fn, *alloc_args = alloc_info
    ptr = alloc_fn(size, *alloc_args)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_address(ptr)
    buffer = torch.frombuffer(buf, dtype=torch.uint8)

    return buffer


def _free_cpu_memory(
    buffer: torch.Tensor,
    size: int | None = None,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    _, free_info = _resolve_pinned_alloc_free(
        numa_mapping,
        shm_name,
        size=size,
    )
    free_fn, *free_args = free_info
    free_fn(buffer.data_ptr(), *free_args)


def _allocate_gpu_memory(
    size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    page_size = os.sysconf("SC_PAGESIZE")

    # Over-allocate
    base_buffer = torch.empty(size + page_size, dtype=torch.uint8, device=device)
    offset = -base_buffer.data_ptr() % page_size

    # Make aligned view
    aligned_buffer = base_buffer[offset : offset + size]

    # Need to return the base buffer as well in order to prevent GC
    return base_buffer, aligned_buffer


class TensorMemoryObj(MemoryObj):
    """
    Wraps eager or address-backed tensor storage with metadata.
    """

    monitor = LMCStatsMonitor.GetOrCreate()

    def __init__(
        self,
        raw_data: Optional[torch.Tensor],
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
        group_prefix_sum: Optional[tuple[int, ...]] = None,
        raw_view_size: Optional[int] = None,
    ):
        assert metadata.dtype is not None, "dtype must be specified for TensorMemoryObj"
        super().__init__(metadata)
        if raw_data is not None:
            self.raw_data = raw_data
        self._raw_view_size = (
            raw_view_size
            if raw_view_size is not None
            else raw_data.numel() * raw_data.element_size()
            if raw_data is not None
            else metadata.phy_size
        )
        self.valid = True
        self.lock = threading.Lock()
        self.parent_allocator = parent_allocator
        self.group_prefix_sum: tuple[int, ...] = (
            group_prefix_sum if group_prefix_sum is not None else ()
        )
        self._allocation_batch: Optional[_TensorAllocationBatch] = None
        self._allocation_batch_index = -1
        if group_prefix_sum is None:
            self.refresh_metadata_view()

    def refresh_metadata_view(self) -> None:
        # Calculate the prefix sum of the group sizes. If there are two
        # groups, the prefix sum will be:
        # [0, size_of_group_1, size_of_group_1 + size_of_group_2].
        fallback_size = (
            self.meta.get_size()
            if self.meta.shapes is None or self.meta.dtypes is None
            else 0
        )
        self.group_prefix_sum = _group_prefix_sums(
            self.meta.shapes, self.meta.dtypes, fallback_size
        )

    @cached_property
    def raw_data(self) -> torch.Tensor:
        """Return the raw view, materializing address-backed storage on demand."""
        buffer = getattr(self.parent_allocator, "buffer", None)
        start = self.meta.address
        end = start + self._raw_view_size
        if (
            not isinstance(buffer, torch.Tensor)
            or start < 0
            or end > buffer.numel()
        ):
            raise RuntimeError("Address-backed tensor storage is unavailable")
        return buffer[start:end]

    def resize_raw_view(self, size: int) -> None:
        """Shrink the logical raw view to ``size`` bytes.

        Raises:
            ValueError: If ``size`` exceeds the current view or is not aligned
                to an already-materialized tensor's element size.
        """
        raw_data = self.__dict__.get("raw_data")
        current_size = (
            raw_data.numel() * raw_data.element_size()
            if raw_data is not None
            else self._raw_view_size
        )
        if size < 0 or size > current_size:
            raise ValueError(f"Invalid raw tensor view size: {size}")
        if raw_data is not None:
            element_size = raw_data.element_size()
            if size % element_size:
                raise ValueError(f"Raw tensor view size is unaligned: {size}")
            self.raw_data = raw_data[: size // element_size]
        self._raw_view_size = size

    @property
    def has_tensor_storage(self) -> bool:
        """Return whether this valid object has accessible backing storage."""
        if not self.valid:
            return False
        if "raw_data" in self.__dict__:
            return True
        buffer = getattr(self.parent_allocator, "buffer", None)
        start = self.meta.address
        return (
            isinstance(buffer, torch.Tensor)
            and start >= 0
            and start + self._raw_view_size <= buffer.numel()
        )

    def __del__(self):
        """
        Destructor to ensure memory is released when the object is garbage collected.
        This acts as a safety net to prevent memory leaks if ref_count_down() is not
        called properly somewhere in the code path.
        """
        if self.parent_allocator is not None and self.is_valid():
            if self.meta.ref_count > 0 or self.meta.pin_count > 0:
                logger.warning(
                    "MemoryObj at %s is being garbage collected "
                    "with ref_count=%d, pin_count=%d. "
                    "This indicates ref_count_down()/unpin() was not called properly.",
                    self.meta.address,
                    self.meta.ref_count,
                    self.meta.pin_count,
                )
            self.parent_allocator.free(self)

    def invalidate(self):
        self.valid = False
        self._allocation_batch = None
        self._allocation_batch_index = -1

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return self.group_prefix_sum[-1]

    def is_trusted_allocation_batch_member(
        self,
        validation_cache: dict[int, bool],
        *,
        parent: Any,
        slab_size: int,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> bool:
        """Validate batch invariants once, then only this object's membership."""
        batch = self._allocation_batch
        if batch is None:
            return False
        batch_id = id(batch)
        if batch_id not in validation_cache:
            validation_cache[batch_id] = batch.matches(
                parent, slab_size, dtype, fmt
            )
        index = self._allocation_batch_index
        metadata = self.meta
        logical_size = self.get_size()
        return (
            self.valid
            and validation_cache[batch_id]
            and 0 <= index < len(batch.member_ids)
            and batch.member_ids[index] == id(self)
            and self.parent_allocator is batch.parent
            and metadata.address == batch.addresses[index]
            and metadata.phy_size == batch.physical_size
            and metadata.dtype == batch.dtype
            and metadata.fmt == batch.fmt
            and 0 < logical_size <= batch.physical_size
        )

    # TODO(chunxiaozheng): use get_shapes and get_dtypes to replace
    #  get_shape and get_dtype
    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        assert self.meta.shapes is not None
        return self.meta.shapes

    def get_dtypes(self) -> list[torch.dtype]:
        assert self.meta.dtypes is not None
        return self.meta.dtypes

    def get_memory_format(self) -> MemoryFormat:
        with self.lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        with self.lock:
            self.meta.ref_count += 1

    def ref_count_down(self):
        with self.lock:
            self.meta.ref_count -= 1
            if self.meta.ref_count < 0:
                logger.warning(
                    f"Ref count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.ref_count}."
                    "Double free occurred somewhere."
                    "Setting ref count back to 0 as a hack but please find the bug."
                )
                self.meta.ref_count = 0
            if (
                self.meta.ref_count == 0
                and self.parent_allocator is not None
                and self.meta.pin_count == 0
            ):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count

    def get_num_tokens(self) -> int:
        with self.lock:
            token_dim = self.meta.fmt.token_dim()
            return self.meta.shape[token_dim]

    def pin(self) -> bool:
        with self.lock:
            # if pin_count is 0, indicates that the object is pinned for the first time
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)

            self.meta.pin_count += 1

            # Register/update with PinMonitor for timeout tracking on every pin
            pin_monitor = PinMonitor.GetOrCreate()
            pin_monitor.on_pin(self)
            return True

    @classmethod
    def pin_many(cls, memory_objs: list["TensorMemoryObj"]) -> bool:
        """Pin a homogeneous batch with one monitor update."""
        if not memory_objs:
            return True

        unique = sorted({id(obj): obj for obj in memory_objs}.values(), key=id)
        locked: list[TensorMemoryObj] = []
        newly_pinned = 0
        stats_updated = False
        try:
            for obj in unique:
                obj.lock.acquire()
                locked.append(obj)
            newly_pinned = sum(obj.meta.pin_count == 0 for obj in unique)
            for obj in memory_objs:
                obj.meta.pin_count += 1
            try:
                if newly_pinned:
                    cls.monitor.update_pinned_memory_objs_count(newly_pinned)
                    stats_updated = True
                PinMonitor.GetOrCreate().on_pin_many(unique)
            except Exception:
                for obj in memory_objs:
                    obj.meta.pin_count -= 1
                if stats_updated:
                    cls.monitor.update_pinned_memory_objs_count(-newly_pinned)
                raise
            return True
        finally:
            for obj in reversed(locked):
                obj.lock.release()

    def unpin(self) -> bool:
        with self.lock:
            self.meta.pin_count -= 1

            # if pin_count is 0, indicates that the object is unpinned
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(-1)
                # Unregister from PinMonitor when fully unpinned
                pin_monitor = PinMonitor.GetOrCreate()
                pin_monitor.on_unpin(self)

            if self.meta.pin_count <= 0 and self.meta.ref_count <= 0:
                if self.parent_allocator is None:
                    logger.error(
                        "Parent allocator is None when trying to free MemoryObj."
                        "This could cause memory leak"
                    )
                else:
                    self.parent_allocator.free(self)

            if self.meta.pin_count < 0:
                logger.warning(
                    f"Pin count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.pin_count}."
                    "Double unpin occurred somewhere."
                    "Setting pin count back to 0 as a hack but please find the bug."
                )
                self.meta.pin_count = 0
            return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        with self.lock:
            return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.dtype is not None
        # TODO(Jiayi): consider caching the `get_size()`
        return self.raw_data[: self.get_size()].view(self.meta.dtype).view(
            self.meta.shape
        )

    @property
    def byte_array(self) -> memoryview:
        # TODO: consider using one of the alternatives

        # Alternative 1:
        # # PyTorch tensors support buffer protocol directly for CPU tensors
        # return memoryview(self.raw_data)

        # Alternative 2:
        # assert self.raw_data.device.type == 'cpu',
        #   "byte_array only works with CPU tensors"
        # return memoryview(self.raw_data.contiguous().numpy())

        raw_data = self.raw_data
        num_bytes = raw_data.numel() * raw_data.element_size()
        ptr = raw_data.data_ptr()
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents)
        )
        return memoryview(byte_array)

    @property
    def data_ptr(self) -> int:
        raw_data = self.__dict__.get("raw_data")
        if raw_data is not None:
            return raw_data.data_ptr()
        if not self.has_tensor_storage:
            raise RuntimeError("Address-backed tensor storage is unavailable")
        assert self.parent_allocator is not None
        return (
            self.parent_allocator.buffer.data_ptr()  # type: ignore[attr-defined]
            + self.meta.address
        )

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A memory obj can be evicted if it is not pinned and ref_count=1.
        """
        return not self.is_pinned and self.get_ref_count() == 1

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return self.raw_data

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.shapes is not None
        assert self.meta.dtypes is not None
        begin = self.group_prefix_sum[index]
        end = self.group_prefix_sum[index + 1]
        return (
            self.raw_data[begin:end]
            .view(self.meta.dtypes[index])
            .view(self.meta.shapes[index])
        )

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        return self.parent_allocator


class LayerPageMemoryObj(TensorMemoryObj):
    """One allocator-owned chunk containing the same layout for every layer."""

    def __init__(
        self,
        raw_data: Optional[torch.Tensor],
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
        *,
        num_layers: int,
        group_prefix_sum: Optional[tuple[int, ...]] = None,
        raw_view_size: Optional[int] = None,
        valid_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(
            raw_data,
            metadata,
            parent_allocator,
            group_prefix_sum,
            raw_view_size,
        )
        if num_layers < 1 or len(self.group_prefix_sum) != num_layers + 1:
            raise ValueError("Layer page requires one tensor group per layer")
        sizes = {
            right - left
            for left, right in zip(
                self.group_prefix_sum, self.group_prefix_sum[1:], strict=False
            )
        }
        if len(sizes) != 1:
            raise ValueError("Layer page requires a homogeneous layer layout")
        self.num_layers = num_layers
        self.layer_size = sizes.pop()
        if valid_tokens is None:
            token_dim = metadata.fmt.token_dim()
            valid_tokens = int(
                metadata.shape[token_dim]
                if token_dim < len(metadata.shape)
                else metadata.shape[0]
            )
        self.valid_tokens = valid_tokens
        if (
            _layer_page_shape(
                metadata.shape,
                metadata.fmt,
                valid_tokens,
                full_tokens=valid_tokens,
            )
            != metadata.shape
        ):
            raise ValueError(
                "Layer-page valid_tokens must match its physical layer shape"
            )
        self.meta.valid_tokens = self.valid_tokens
        self._base_data_ptr = self.data_ptr

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        raise RuntimeError("Layer page requires layer_tensor(layer_id)")

    def layer_tensor(self, layer_id: int) -> torch.Tensor:
        """Return the typed tensor view for one layer."""
        if not self.valid:
            raise RuntimeError("Layer page storage is no longer valid")
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"Invalid layer page index: {layer_id}")
        tensor = self.get_tensor(layer_id)
        assert tensor is not None
        return tensor

    def layer_data_ptr(self, layer_id: int) -> int:
        """Return the host address of one layer without constructing a view."""
        if not self.valid:
            raise RuntimeError("Layer page storage is no longer valid")
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"Invalid layer page index: {layer_id}")
        return self._base_data_ptr + self.group_prefix_sum[layer_id]


@dataclass(frozen=True, slots=True)
class LayerPageSource:
    """Layer selection over request-owned page objects."""

    pages: tuple[LayerPageMemoryObj, ...]
    layer_id: int
    suffix: tuple[MemoryObj, ...] = ()


class BytesBufferMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_bytes: bytes, metadata: Optional[MemoryObjMetadata] = None):
        self.raw_data = raw_bytes
        if metadata is None:
            bytes_shape = torch.Size([len(self.raw_data), 0, 0, 0])
            metadata = MemoryObjMetadata(
                shape=bytes_shape,
                dtype=None,
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=MemoryFormat.BINARY_BUFFER,
            )
        super().__init__(metadata)
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return len(self.raw_data)

    def get_shape(self) -> torch.Size:
        return torch.Size([len(self.raw_data), 0, 0, 0])

    def get_dtype(self) -> Optional[torch.dtype]:
        return None

    def get_shapes(self) -> list[torch.Size]:
        return [self.get_shape()]

    def get_dtypes(self) -> list[torch.dtype]:
        return []

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    def pin(self) -> bool:
        self.metadata.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.metadata.pin_count -= 1
        if self.metadata.pin_count < 0:
            logger.warning(
                f"Pin count of MemoryObj {self.meta.address}"
                f"is negative: {self.meta.pin_count}."
                "Double unpin occurred somewhere."
                "Setting pin count back to 0 as a hack but please find the bug."
            )
            self.metadata.pin_count = 0
        return True

    def ref_count_up(self):
        pass

    def ref_count_down(self):
        pass

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        # TODO(Jiayi): record the number of tokens somehow
        return 1

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    @property
    def byte_array(self) -> bytes:
        return self.raw_data

    @property
    def data_ptr(self) -> int:
        mv = memoryview(self.raw_data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        return addr

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A buffer memory obj can be evicted if it is not pinned.
        """
        return not self.is_pinned

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        # NOTE: BytesBufferMemoryObj may not be allocated by any allocator,
        # so just return None here
        return None


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param int batch_size: The number of tensors to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A list of MemoryObjs wrapping the allocated memory.
            Returns None if the allocation failed.

        :rtype: Optional[List[MemoryObj]]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        """
        Frees the memory allocated for the given MemoryObj.
        Note that this function shouldn't be explicitly called.
        Instead, use `ref_count_down` to decrease ref count.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Frees the memory allocated for the given list of MemoryObjs.

        :param List[MemoryObj] memory_objs: The list of MemoryObjs
            to free.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the memory allocator.
        This is called when the LMCacheEngine is closed.
        """
        return

    def memcheck(self) -> bool:
        """
        Checks the memory allocator for consistency.

        Returns:
            True if everything is fine otherwise False
        """
        return True

    # TODO(chunxiaozheng): remove if after all params replaced by shapes/dtypes
    def _adapt_shapes_and_dtypes(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> Tuple[list[torch.Size], list[torch.dtype]]:
        if isinstance(shapes, torch.Size):
            shapes = [shapes]

        if isinstance(dtypes, torch.dtype):
            dtypes = [dtypes]

        assert len(shapes) == len(dtypes), (
            f"shapes and dtypes must have the same length, "
            f"got {len(shapes)} and {len(dtypes)}, "
            f"shapes: {shapes}, dtypes: {dtypes}"
        )
        return shapes, dtypes


class AddressManager:
    """
    Manages a virtual address space starting from 0 for memory allocation.

    Key interfaces:
    - allocate(size): Allocate a block of memory of the given size. The starting
      address and the actual allocated size will be aligned.

    - free(address, size): Free a previously allocated region. Note that if the
      region is not "allocated" before, it may have internal errors.

    - sbrk(size): Expand the virtual address space by the given size. The size
      will be aligned internally.

    Core assumptions:
    - The allocated size should be aligned with ALIGN_BYTES.
    """

    ALIGN_BYTES = 4096

    def __init__(self, size: int, align_bytes: int = ALIGN_BYTES):
        """
        Initializes the AddressManager with a given size.

        Args:
            size: The initial size of the virtual address space.
            align_bytes: The alignment requirement for allocations.
        """
        self._size = size
        self._align = align_bytes

        # Current implementation: explicit list
        self._explicit_list: SortedList[FreeBlock] = SortedList(key=lambda x: x.start)
        self._explicit_list.add(FreeBlock(start=0, size=size))

        # thread safe lock
        self._lock = threading.Lock()

        # For debugging purposes
        self.total_allocated_size = 0

    def compute_aligned_size(self, raw_size: int) -> int:
        """
        Helper function to compute the aligned size for a given raw size.

        Args:
            raw_size: The raw size to be aligned.

        Returns:
            The aligned size.
        """
        return (raw_size + self._align - 1) & ~(self._align - 1)

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        """Hook: Check if curr_block can merge with prev_block."""
        return prev_block.can_be_coalesced(curr_block)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        """Hook: Check if curr_block can merge with succ_block."""
        return curr_block.can_be_coalesced(succ_block)

    @_lmcache_nvtx_annotate
    def _coalesce(
        self,
        curr_block: FreeBlock,
        prev_block: Optional[FreeBlock],
        succ_block: Optional[FreeBlock],
    ):
        """
        Coalesces the current block with the previous and/or successor block.
        This assumes the curr_block is NOT in self._explicit_list

        Returns True if the current block was coalesced, otherwise False.
        """
        merge_prev = prev_block is not None and self._can_merge_with_prev(
            curr_block, prev_block
        )
        merge_succ = succ_block is not None and self._can_merge_with_succ(
            curr_block, succ_block
        )

        if merge_prev and merge_succ:
            prev_block.size += curr_block.size + succ_block.size  # type: ignore
            self._explicit_list.remove(succ_block)
        elif merge_prev:
            prev_block.size += curr_block.size  # type: ignore
        elif merge_succ:
            # NOTE: logically, this won't change the order of the succ_block,
            #       so we don't need to do a "remove" and "reinsert" here
            self._explicit_list.remove(succ_block)
            succ_block.start -= curr_block.size  # type: ignore
            succ_block.size += curr_block.size  # type: ignore
            self._explicit_list.add(succ_block)

        return merge_prev or merge_succ

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def allocate(self, size: int) -> tuple[int, int]:
        """
        Allocate a block of memory from the virtual address space of a given
        size. The actual allocated size could be larger than the requested size
        in order to satisfy alignment requirements.

        Args:
            size: The requested size of the memory block. Should be greater
                than 0.

        Returns:
            A tuple (address, allocated_size) where address is the starting
            address of the allocated block and allocated_size is the actual
            size of the allocated block.

        Raises:
            RuntimeError: If no memory is available to allocate.
        """
        aligned_size = self.compute_aligned_size(size)
        for block in self._explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.warning(
                "Failed to allocate memory block of size %d "
                "because no memory is available",
                size,
            )
            raise RuntimeError(
                f"Failed to allocate memory block of size {size} "
                "because no memory is available"
            )

        self._explicit_list.remove(block)
        if block.size > aligned_size:
            self._explicit_list.add(
                FreeBlock(
                    start=block.start + aligned_size,
                    size=block.size - aligned_size,
                )
            )

        # For debug
        self.total_allocated_size += aligned_size

        return block.start, aligned_size

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def batched_allocate(self, size: int, batch_size: int) -> list[tuple[int, int]]:
        """
        Allocate blocks of memory from the virtual address space of a given
        size and batch size. The actual allocated size could be larger than
        the requested size in order to satisfy alignment requirements.

        Args:
            size: The requested size of the memory block. Should be greater
                than 0.
            batch_size: The number of memory blocks to allocate.

        Returns:
            A list of tuple (address, allocated_size) where address is the starting
            address of the allocated block and allocated_size is the actual size of
            the allocated block.
            Note: the length of the return list is the same as the batch_size.

        Raises:
            RuntimeError: If no memory is available to allocate.
        """
        aligned_size = self.compute_aligned_size(size)
        remaining = batch_size
        allocate_result: list[tuple[int, int]] = []

        blocks_to_remove: list[FreeBlock] = []
        blocks_to_add: list[FreeBlock] = []

        for block in self._explicit_list:
            if remaining <= 0:
                break
            if block.size < aligned_size:
                continue

            # Greedily carve out as many aligned_size chunks as possible
            num_from_block = min(remaining, block.size // aligned_size)
            start = block.start
            for i in range(num_from_block):
                allocate_result.append((start + i * aligned_size, aligned_size))
            remaining -= num_from_block

            # Mark the original block for removal
            blocks_to_remove.append(block)

            # Keep the remaining tail as a new free block if any space is left
            used = num_from_block * aligned_size
            if block.size > used:
                blocks_to_add.append(
                    FreeBlock(start=block.start + used, size=block.size - used)
                )

        if remaining > 0:
            # Not enough memory; free list is untouched, no rollback needed
            logger.warning(
                "Failed to batched allocate %d memory blocks of size %d "
                "because no enough memory is available (short by %d blocks)",
                batch_size,
                size,
                remaining,
            )
            raise RuntimeError(
                f"Failed to batched allocate {batch_size} memory blocks "
                f"of size {size} because no enough memory is available"
            )
        if len(allocate_result) != batch_size:
            # The length of allocate_result is not equal to batch_size;
            # free list is untouched, no rollback needed
            logger.warning(
                "Failed to batched allocate %d memory blocks of size %d "
                "because the length of allocate_result %d is not equal to batch_size",
                batch_size,
                size,
                len(allocate_result),
            )
            raise RuntimeError(
                f"Failed to batched allocate {batch_size} memory blocks "
                f"of size {size} because the length of allocate_result "
                f"{len(allocate_result)} is not equal to batch_size"
            )

        # Allocation succeeded; batch-update the free list
        for block in blocks_to_remove:
            self._explicit_list.remove(block)
        for block in blocks_to_add:
            self._explicit_list.add(block)

        # Update debug statistics
        total_allocated = aligned_size * batch_size
        self.total_allocated_size += total_allocated

        return allocate_result

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def free(self, address: int, size: int):
        """
        Free a previously allocated block of memory.

        Args:
            address: The starting address of the block to free.
            size: The size of the block to free. Should be greater than 0.
        """
        new_free_block = FreeBlock(start=address, size=size)
        index = self._explicit_list.bisect_left(new_free_block)
        prev_block = self._explicit_list[index - 1] if index > 0 else None
        succ_block = (
            self._explicit_list[index] if index < len(self._explicit_list) else None
        )

        coalesced = self._coalesce(new_free_block, prev_block, succ_block)
        if not coalesced:
            self._explicit_list.add(new_free_block)

        # For debug
        self.total_allocated_size -= size

    @synchronized("_lock")
    def sbrk(self, size: int):
        """
        Expand the virtual address space by a given size.

        Args:
            size: The size to expand the address space. Will be aligned internally
                with the ALIGN_BYTES
        """
        size = self.compute_aligned_size(size)
        new_block = FreeBlock(start=self._size, size=size)
        prev_block = self._explicit_list[-1] if len(self._explicit_list) > 0 else None
        succ_block = None
        coalesced = self._coalesce(new_block, prev_block, succ_block)
        if not coalesced:
            self._explicit_list.add(new_block)

        self._size += size

    def get_heap_size(self) -> int:
        """
        Get the total size of the address space.

        Returns:
            The total size in bytes.
        """
        return self._size

    def get_free_size(self) -> int:
        """
        Get the total free size in the address space.

        Returns:
            The total free size in bytes.
        """
        return self._size - self.total_allocated_size

    def check_consistency(self) -> bool:
        """
        Check if the address manager is consistent.

        Returns:
            True if consistent, False otherwise.
        """
        # Check if free blocks are properly coalesced
        for prev, succ in zip(
            self._explicit_list[:-1], self._explicit_list[1:], strict=False
        ):
            if prev.can_be_coalesced(succ):
                return False

        # Check if total size matches
        total_free_size = sum(block.size for block in self._explicit_list)
        if total_free_size + self.total_allocated_size != self._size:
            return False

        return True


class TensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a "explicit list" memory allocator.
    Uses AddressManager for address space management.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        init_address_space: int | None = None,
    ):
        """
        Args:
            tensor: The pre-allocated flat tensor to use as the memory pool.
            align_bytes: The alignment requirement for allocations.
            init_address_space: Initial size of the address space. If None,
                use the size of the provided tensor.

        Note:
            The `init_address_space` is used for lazy memory allocation.
            We probably want to have a better way to make sure that the
            LazyMemoryAllocator can be decoupled from TensorMemoryAllocator.
        """
        self.buffer = tensor.view(torch.uint8).flatten()

        # Use AddressManager for address space management
        self.address_manager = AddressManager(
            self.buffer.numel() if init_address_space is None else init_address_space,
            align_bytes,
        )

        # For debugging purposes
        self.num_active_allocations = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @property
    def total_allocated_size(self) -> int:
        return self.address_manager.total_allocated_size

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[TensorMemoryObj]:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        # Calculate the size of the tensor
        raw_size = get_size_bytes(shapes, dtypes)

        # Allocate from address manager
        try:
            block_start, aligned_size = self.address_manager.allocate(raw_size)
        except RuntimeError:
            # No block found
            return None

        # For debug
        self.num_active_allocations += 1

        # Update stats
        self.stats_monitor.update_local_cache_usage(
            self.address_manager.total_allocated_size
        )
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        raw_data = self._get_buffer_slice(block_start, raw_size)
        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=MemoryObjMetadata(
                shapes[0],
                dtypes[0],
                block_start,
                aligned_size,
                1,
                0,
                fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )

    def _get_buffer_slice(self, start: int, size: int) -> torch.Tensor:
        """Hook: Get buffer slice. Override for custom buffer access."""
        return self.buffer[start : start + size]

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """
        Batched allocate tensor memory objs with equal sizes.
        """
        return self._batched_allocate(
            shapes, dtypes, batch_size, fmt, materialize_views=True
        )

    @_lmcache_nvtx_annotate
    def batched_allocate_address_backed(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """Allocate a homogeneous batch without constructing tensor views.

        Arguments and return semantics match :meth:`batched_allocate`. Each
        returned object computes its pointer from the backing buffer and
        materializes a compatible raw tensor view only when requested.
        """
        return self._batched_allocate(
            shapes, dtypes, batch_size, fmt, materialize_views=False
        )

    def batched_allocate_layer_pages(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        num_layers: int,
        fmt: MemoryFormat,
        valid_tokens: Optional[Union[int, list[int]]] = None,
        full_tokens: Optional[int] = None,
    ) -> Optional[List[LayerPageMemoryObj]]:
        """Allocate one exact-size all-layer object per token chunk."""
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        if len(shapes) != 1 or len(dtypes) != 1:
            return None
        token_dim = fmt.token_dim()
        if full_tokens is None:
            full_tokens = int(
                shapes[0][token_dim]
                if token_dim < len(shapes[0])
                else shapes[0][0]
            )
        full_shape = _layer_page_shape(
            shapes[0], fmt, full_tokens, full_tokens=full_tokens
        )
        token_counts = (
            [full_tokens] * batch_size
            if valid_tokens is None
            else [valid_tokens] * batch_size
            if isinstance(valid_tokens, int)
            else list(valid_tokens)
        )
        if len(token_counts) != batch_size or any(
            not 0 < count <= full_tokens for count in token_counts
        ):
            raise ValueError("Invalid layer-page valid_tokens batch")
        if len(set(token_counts)) != 1:
            positions_by_count: dict[int, list[int]] = {}
            for index, count in enumerate(token_counts):
                positions_by_count.setdefault(count, []).append(index)
            pages_by_position: list[Optional[LayerPageMemoryObj]] = [
                None
            ] * batch_size
            allocated_pages: list[LayerPageMemoryObj] = []
            for count, positions in positions_by_count.items():
                allocated = self.batched_allocate_layer_pages(
                    [full_shape],
                    dtypes,
                    len(positions),
                    num_layers,
                    fmt,
                    count,
                    full_tokens,
                )
                if allocated is None:
                    for page in allocated_pages:
                        page.ref_count_down()
                    return None
                allocated_pages.extend(allocated)
                for position, page in zip(positions, allocated, strict=True):
                    pages_by_position[position] = page
            return [
                page
                for page in pages_by_position
                if page is not None
            ]
        count = token_counts[0]
        shapes = [
            _layer_page_shape(
                full_shape, fmt, count, full_tokens=full_tokens
            )
        ]
        return self._batched_allocate(
            shapes * num_layers,
            dtypes * num_layers,
            batch_size,
            fmt,
            materialize_views=False,
            page_layers=num_layers,
            page_valid_tokens=count,
        )  # type: ignore[return-value]

    def _batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat,
        *,
        materialize_views: bool,
        page_layers: int = 0,
        page_valid_tokens: Optional[int] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        # Calculate the size of the tensor
        unit_raw_size = get_size_bytes(shapes, dtypes)
        unit_aligned_size = self.address_manager.compute_aligned_size(unit_raw_size)

        try:
            alloc_results = self.address_manager.batched_allocate(
                unit_aligned_size, batch_size
            )
        except RuntimeError:
            return None
        addresses = [addr for addr, _ in alloc_results]
        raw_datas: list[Optional[torch.Tensor]]
        if materialize_views:
            contiguous = len(addresses) > 1 and all(
                curr == prev + unit_aligned_size
                for prev, curr in zip(addresses, addresses[1:], strict=False)
            )
            if contiguous:
                raw_datas = list(
                    self._get_buffer_slice(
                        addresses[0], unit_aligned_size * batch_size
                    ).split(unit_aligned_size)
                )
            else:
                raw_datas = [
                    self._get_buffer_slice(addr, unit_aligned_size)
                    for addr in addresses
                ]
        else:
            raw_datas = [None] * batch_size

        # For debug
        self.num_active_allocations += batch_size

        # Update stats
        self.stats_monitor.update_local_cache_usage(
            self.address_manager.total_allocated_size
        )
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        group_prefix_sum = _group_prefix_sums(shapes, dtypes, unit_raw_size)
        tensor_mem_objs = []
        for raw_data, address in zip(raw_datas, addresses, strict=True):
            cls = LayerPageMemoryObj if page_layers else TensorMemoryObj
            tensor_mem_objs.append(
                cls(
                    raw_data=raw_data,
                    metadata=MemoryObjMetadata(
                        shapes[0],
                        dtypes[0],
                        address,
                        unit_aligned_size,
                        1,
                        0,
                        fmt,
                        shapes=shapes,
                        dtypes=dtypes,
                    ),
                    parent_allocator=self,
                    group_prefix_sum=group_prefix_sum,
                    raw_view_size=unit_aligned_size,
                    **({"num_layers": page_layers} if page_layers else {}),
                    **(
                        {"valid_tokens": page_valid_tokens}
                        if page_layers
                        else {}
                    ),
                )
            )

        batch = _TensorAllocationBatch(
            self,
            int(self.buffer.numel()),
            dtypes[0],
            fmt,
            tuple(addresses),
            unit_aligned_size,
            tuple(map(id, tensor_mem_objs)),
        )
        for index, memory_obj in enumerate(tensor_mem_objs):
            memory_obj._allocation_batch = batch
            memory_obj._allocation_batch_index = index

        return tensor_mem_objs

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        if not memory_obj.is_valid():
            return

        self.address_manager.free(memory_obj.meta.address, memory_obj.meta.phy_size)
        memory_obj.invalidate()

        # For debug
        self.num_active_allocations -= 1

        # Update stats
        self.stats_monitor.update_local_cache_usage(
            self.address_manager.total_allocated_size
        )
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Batched free memory objs.
        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.
        """
        if not memory_objs:
            return

        # Coalesce adjacent memory objects before freeing to reduce
        # the number of free operations
        coalesced_blocks: list[tuple[int, int, int]] = []  # (address, size, count)
        curr_start = None
        curr_size = 0
        curr_count = 0

        memory_objs.sort(key=lambda x: x.meta.address)
        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            memory_obj.invalidate()

            if curr_start is None:
                curr_start = memory_obj.meta.address
                curr_size = memory_obj.meta.phy_size
                curr_count = 1
            elif curr_start + curr_size == memory_obj.meta.address:
                # Adjacent block, extend current
                curr_size += memory_obj.meta.phy_size
                curr_count += 1
            else:
                # Non-adjacent, save current and start new
                coalesced_blocks.append((curr_start, curr_size, curr_count))
                curr_start = memory_obj.meta.address
                curr_size = memory_obj.meta.phy_size
                curr_count = 1

        if curr_start is not None:
            coalesced_blocks.append((curr_start, curr_size, curr_count))

        # Free all coalesced blocks
        total_count = 0
        for address, size, count in coalesced_blocks:
            self.address_manager.free(address, size)
            total_count += count

        # For debug
        self.num_active_allocations -= total_count

        if update_stats:
            self.stats_monitor.update_local_cache_usage(
                self.address_manager.total_allocated_size
            )
            self.stats_monitor.update_active_memory_objs_count(
                self.num_active_allocations
            )

    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """
        clear = True
        logger.info("Checking memory allocator consistency")
        logger.info(f" - Total active allocations: {self.num_active_allocations}")
        logger.info(
            f" - Total allocated size: "
            f"{self.address_manager.total_allocated_size / 1048576} MB"
        )

        # Check the real total free size
        total_free_size = self.address_manager.get_free_size()
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if (
            total_free_size + self.address_manager.total_allocated_size
            != self.address_manager.get_heap_size()
        ):
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        # Check if the blocks are coalesced
        if not self.address_manager.check_consistency():
            logger.error("Memory allocator has non-coalesced blocks")
            logger.error("This implies a bug in the memory allocator")
            clear = False

        return clear

    def __str__(self):
        return "TensorMemoryAllocator"


class PagedAddressManager:
    """
    A lightweight address manager for PagedTensorMemoryAllocator.
    Provides get_free_size() and get_heap_size() by reading the
    paged allocator's state.
    """

    def __init__(self, paged_allocator: "PagedTensorMemoryAllocator"):
        self._allocator = paged_allocator

    def get_heap_size(self) -> int:
        """Get the total size of the paged address space in bytes."""
        return self._allocator.buffer_size

    def get_free_size(self) -> int:
        """Get the total free size in bytes."""
        return len(self._allocator.free_blocks) * self._allocator.align_bytes


class PagedTensorMemoryAllocator(MemoryAllocatorInterface):
    """
    Implements a paged memory allocator.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ):
        self.buffer = tensor.view(torch.uint8).flatten()
        self.buffer_size = self.buffer.numel() * self.buffer.element_size()
        self.buffer_ptr = self.buffer.data_ptr()

        self.shapes = shapes
        self.dtypes = dtypes
        self.fmt = fmt

        # full chunk size bytes
        self.align_bytes = get_size_bytes(shapes, dtypes)

        assert self.buffer_size % self.align_bytes == 0, (
            f"Buffer size {self.buffer_size} must be a"
            f" multiple of align bytes {self.align_bytes}"
            " in paged memory allocator."
        )

        self.paged_buffers = torch.split(self.buffer, self.align_bytes, dim=0)

        # NOTE: deque is used since thread-safety is not a concern here as
        # is implemented in C under the hood (in CPython), and operations
        # on deque are atomic.
        self.free_blocks: deque[TensorMemoryObj] = deque()

        for idx, buf in enumerate(self.paged_buffers):
            # NOTE: idx is the paged index
            # NOTE: the last unfull chunk's shape needs to be
            # adjusted during allocation.
            metadata = MemoryObjMetadata(
                self.shapes[0],
                self.dtypes[0],
                idx,
                self.align_bytes,  # 1 page
                1,  # ref_count=1
                0,  # pin_count=0
                self.fmt,
                shapes=self.shapes,
                dtypes=self.dtypes,
            )
            mem_obj = TensorMemoryObj(
                raw_data=buf,
                metadata=metadata,
                parent_allocator=self,
            )
            self.free_blocks.append(mem_obj)

        # Address manager for memory usage tracking
        self.address_manager = PagedAddressManager(self)

        # For debugging purposes
        self.num_active_allocations = 0
        self.total_allocated_size = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        logger.info(
            "Paged tensor memory allocator initialized, "
            "shapes: %s, dtypes: %s, align bytes: %s",
            self.shapes,
            self.dtypes,
            self.align_bytes,
        )

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[TensorMemoryObj]:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        try:
            free_block = self.free_blocks.popleft()
        except IndexError:
            logger.debug(
                f"Failed to allocate memory for "
                f"tensor({shapes}, {dtypes}) because "
                "no free blocks is available"
            )
            return None

        # TODO (Jiayi): This is a bit redundant.
        free_block.meta.shape = shapes[0]
        free_block.meta.dtype = dtypes[0]
        free_block.meta.shapes = shapes
        free_block.meta.dtypes = dtypes
        free_block.meta.fmt = fmt
        free_block.meta.ref_count = 1

        if shapes != self.shapes:
            size_in_bytes = get_size_bytes(shapes, dtypes)
            free_block.raw_data = free_block.raw_data[:size_in_bytes]
        free_block.refresh_metadata_view()

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += 1
        self.total_allocated_size += self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return free_block

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[TensorMemoryObj]]:
        """
        Batched allocate tensor memory objs with pre-defined equal sizes.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)

        allocated_blocks: list[TensorMemoryObj] = []
        for i in range(batch_size):
            try:
                free_block = self.free_blocks.popleft()
            except IndexError:
                logger.debug(
                    f"Failed to allocate memory for "
                    f"tensor({shapes}, {dtypes}) because "
                    "no free blocks is available"
                )
                self.batched_free(allocated_blocks, update_stats=False)
                return None

            # FIXME: think about whether pareant_allocator
            # should be updated here.
            free_block.meta.shape = shapes[0]
            free_block.meta.dtype = dtypes[0]
            free_block.meta.shapes = shapes
            free_block.meta.dtypes = dtypes
            free_block.meta.fmt = fmt
            free_block.meta.ref_count = 1

            if shapes != self.shapes:
                size_in_bytes = get_size_bytes(shapes, dtypes)
                free_block.raw_data = free_block.raw_data[:size_in_bytes]
            free_block.refresh_metadata_view()

            allocated_blocks.append(free_block)

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.num_active_allocations += batch_size
        self.total_allocated_size = self.num_active_allocations * self.align_bytes
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

        # Allocate the block
        return allocated_blocks

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: TensorMemoryObj, allocator_type: Optional[str] = None):
        if not memory_obj.is_valid():
            return
        if memory_obj.meta.shapes != self.shapes:
            page_idx = memory_obj.meta.address
            memory_obj.raw_data = self.paged_buffers[page_idx]

        self.free_blocks.append(memory_obj)

        # memory_obj.invalidate()

        # TODO (Jiayi): need a flag to drop these debug ops
        # NOTE (Jiayi): the following code is not thread-safe but
        # is tolerable as this is only used for debugging purposes.
        # Update debug status
        self.total_allocated_size -= self.align_bytes
        self.num_active_allocations -= 1
        self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[TensorMemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Batched free memory objs.
        Unlike `batched_allocate`, this function does not
        assume that the memory objs are equal-sized.
        """
        if not memory_objs:
            return

        for memory_obj in memory_objs:
            if not memory_obj.is_valid():
                logger.warning("Trying to free an invalidated MemoryObj")
                continue
            # memory_obj.invalidate()
            if memory_obj.meta.shapes != self.shapes:
                page_idx = memory_obj.meta.address
                memory_obj.raw_data = self.paged_buffers[page_idx]

            self.free_blocks.append(memory_obj)

        if update_stats:
            num_freed_blocks = len(memory_objs)
            # TODO (Jiayi): need a flag to drop these debug ops
            # NOTE (Jiayi): the following code is not thread-safe but
            # is tolerable as this is only used for debugging purposes.
            # Update debug status
            self.total_allocated_size -= self.align_bytes * num_freed_blocks
            self.num_active_allocations -= num_freed_blocks
            self.stats_monitor.update_local_cache_usage(self.total_allocated_size)
            self.stats_monitor.update_active_memory_objs_count(
                self.num_active_allocations
            )

    def memcheck(self):
        """For debug purposes.
        Returns True is everything is fine, otherwise False.
        """

        logger.info("Checking memory allocator consistency")
        logger.info(f" - Total active allocations: {self.num_active_allocations}")
        logger.info(
            f" - Total allocated size: {self.total_allocated_size / 1048576} MB"
        )

        # Check the real total free size
        total_free_size = len(self.free_blocks) * self.align_bytes
        logger.info(f" - Total free size: {total_free_size / 1048576} MB")

        # Check if the numbers are consistent
        if total_free_size + self.total_allocated_size != self.buffer.numel():
            logger.error("Memory allocator size is inconsistent")
            logger.error("This implies a bug in the memory allocator")
            return False

        return True

    def __str__(self):
        return "PagedTensorMemoryAllocator"

    def __del__(self):
        # FIXME: NIXL-related memory leak should be handled somewhere (else).
        del self.buffer


class BufferAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, device="cpu"):
        """
        :param str device: The device of the buffer memory.
        """
        self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
        allocator_type: Optional[str] = None,
    ) -> BytesBufferMemoryObj:
        if isinstance(shapes, list):
            n = shapes[0][0]
        else:
            n = shapes[0]
        byte_array = bytearray(n)
        return BytesBufferMemoryObj(byte_array)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER,
        allocator_type: Optional[str] = None,
    ) -> List[BytesBufferMemoryObj]:
        if isinstance(shapes, list):
            n = shapes[0][0]
        else:
            n = shapes[0]
        # TODO(Jiayi): Optimize the following loop.
        byte_arrays = [bytearray(n) for _ in range(batch_size)]
        return [BytesBufferMemoryObj(byte_array) for byte_array in byte_arrays]

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        return

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        return

    def __str__(self):
        return "BufferAllocator"

    def memcheck(self):
        return True


class HostMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated Host memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def __str__(self):
        return "HostMemoryAllocator"


class PinMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated pinned memory."""

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """
        self.size = size
        self.buffer = _allocate_cpu_memory(size)
        self._unregistered = False

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.host_mem_lock:
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.host_mem_lock:
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.host_mem_lock:
            self.allocator.free(memory_obj)

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.host_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.host_mem_lock:
            return self.allocator.memcheck()

    def close(self):
        if not self._unregistered:
            if self.buffer.numel() == 0:
                return
            _free_cpu_memory(self.buffer, self.size)
            self._unregistered = True

    def __str__(self):
        return "PinMemoryAllocator"


class MixedMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates (1) memory in the pre-allocated pinned memory.
              (2) byte_array buffer memory.
    """

    def __init__(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        self.numa_mapping = kwargs.get("numa_mapping", None)
        self.align_bytes = kwargs.get("align_bytes", AddressManager.ALIGN_BYTES)
        if self.align_bytes <= 0 or self.align_bytes & (self.align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # Extract shm_name from config.extra_config if available
        config = kwargs.get("config", None)
        if config is not None:
            self.shm_name: Optional[str] = config.get_extra_config_value(
                "shm_name", None
            )
        else:
            self.shm_name = kwargs.get("shm_name", None)

        self.size = size

        self.buffer = _allocate_cpu_memory(
            size,
            self.numa_mapping,
            self.shm_name,
            kwargs.get("shm_interleave_nodes", None),
        )

        self._unregistered = False

        self.pin_allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.pin_allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.pin_allocator = TensorMemoryAllocator(
                self.buffer, align_bytes=self.align_bytes
            )

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

        self.buffer_allocator = BufferAllocator("cpu")

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shapes, dtypes, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.KV_MLA_LATENT_FMT,
            MemoryFormat.KV_DSA_INDEX_FMT,
        ]:
            with self.host_mem_lock:
                return self.pin_allocator.allocate(shapes, dtypes, fmt, str(self))
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.KV_MLA_LATENT_FMT,
            MemoryFormat.KV_DSA_INDEX_FMT,
        ]:
            with self.host_mem_lock:
                return self.pin_allocator.batched_allocate(
                    shapes, dtypes, batch_size, fmt, str(self)
                )
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def batched_allocate_address_backed(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Allocate an address-backed batch from the pinned tensor allocator.

        Arguments and return semantics match :meth:`batched_allocate`.
        Unsupported formats and allocators retain the normal allocation path.
        """
        allocate = getattr(
            self.pin_allocator, "batched_allocate_address_backed", None
        )
        if fmt not in (
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.KV_MLA_LATENT_FMT,
            MemoryFormat.KV_DSA_INDEX_FMT,
        ) or not callable(allocate):
            return self.batched_allocate(shapes, dtypes, batch_size, fmt)
        with self.host_mem_lock:
            return allocate(shapes, dtypes, batch_size, fmt, str(self))

    def batched_allocate_layer_pages(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        num_layers: int,
        fmt: MemoryFormat,
        valid_tokens: Optional[Union[int, list[int]]] = None,
        full_tokens: Optional[int] = None,
    ) -> Optional[List[LayerPageMemoryObj]]:
        """Allocate layer pages from the pinned allocator."""
        allocate = getattr(self.pin_allocator, "batched_allocate_layer_pages", None)
        if not callable(allocate):
            return None
        with self.host_mem_lock:
            return allocate(
                shapes,
                dtypes,
                batch_size,
                num_layers,
                fmt,
                valid_tokens,
                full_tokens,
            )

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.KV_MLA_LATENT_FMT,
            MemoryFormat.KV_DSA_INDEX_FMT,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        if not memory_objs:
            return

        # NOTE: fmts of all memory_objs should be the same
        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.KV_MLA_LATENT_FMT,
            MemoryFormat.KV_DSA_INDEX_FMT,
        ]:
            with self.host_mem_lock:
                self.pin_allocator.batched_free(memory_objs)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def memcheck(self):
        with self.host_mem_lock:
            return self.pin_allocator.memcheck()

    def close(self):
        if not self._unregistered:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self.buffer.numel() == 0:
                return
            _free_cpu_memory(
                self.buffer,
                self.size,
                self.numa_mapping,
                self.shm_name,
            )
            self._unregistered = True

    def __str__(self):
        return "MixedMemoryAllocator"


class GPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated GPU memory."""

    def __init__(
        self,
        size: int,
        device="cuda",
        align_bytes: Optional[int] = None,
        use_paging: bool = False,
        **kwargs,
    ):
        """
        :param int size: The size of the GPU memory in bytes.
        :param Optional[int] align_bytes: The byte alignment for allocations.
        """
        if not torch.cuda.is_available():
            device = "cpu"

        self.tensor = torch.empty(size, dtype=torch.uint8, device=device)

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.tensor,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            kwargs = {}
            if align_bytes is not None:
                kwargs["align_bytes"] = align_bytes
            self.allocator = TensorMemoryAllocator(self.tensor, **kwargs)

        self.device_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.device_mem_lock:
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.device_mem_lock:
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.device_mem_lock:
            self.allocator.free(memory_obj)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.device_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.device_mem_lock:
            return self.allocator.memcheck()

    def __str__(self):
        return "GPUMemoryAllocator"


class AdHocMemoryAllocator(MemoryAllocatorInterface):
    """
    AdHocMemoryAllocator is a simple allocator that does not actually
    allocate memory. It is used for testing purposes only.
    """

    def __init__(self, device: str = "cpu"):
        """
        :param str device: The device of the ad hoc memory allocator.
        """
        if not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Returns a dummy MemoryObj for testing purposes.
        """
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        size = get_size_bytes(shapes, dtypes)

        # Return a dummy object with no actual memory allocation
        return TensorMemoryObj(
            raw_data=torch.empty(
                torch.Size([size]), dtype=torch.uint8, device=self.device
            ),
            metadata=MemoryObjMetadata(
                shape=shapes[0],
                dtype=dtypes[0],
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        raise NotImplementedError(
            "Batched allocation is not supported in AdHocMemoryAllocator"
        )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        pass

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        pass

    def ref_count_up(self, memory_obj: MemoryObj):
        pass

    def ref_count_down(self, memory_obj: MemoryObj):
        pass

    def get_ref_count(self, memory_obj: MemoryObj):
        return 0

    def memcheck(self):
        return True

    def __str__(self):
        return "AdHocMemoryAllocator"


class CuFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None):
        # HACK(Jiayi): cufile import is buggy on some hardware
        # (e.g., without GPUDirect), so it's temporarily put here.
        # Third Party
        from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

        self.cuFileBufDeregister = cuFileBufDeregister
        if device is None:
            # TODO(Serapheim): Ideally we'd get the device from the upper
            # layer - for now just use the current device.
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu:0"
        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        cuFileBufRegister(ctypes.c_void_p(self.base_pointer), size, flags=0)

    def __del__(self):
        self.cuFileBufDeregister(ctypes.c_void_p(self.base_pointer))

    def __str__(self):
        return "CuFileMemoryAllocator"


class HipFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None):
        # HACK: hipfile import is placed here to avoid import errors on
        # hardware without GPUDirect Storage / hipFile support.
        # Third Party
        from hipfile.bindings import hipFileBufDeregister, hipFileBufRegister

        self.hipFileBufDeregister = hipFileBufDeregister
        if device is None:
            if torch.cuda.is_available():
                # TODO: On ROCm, PyTorch still uses the CUDA API internally
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu:0"

        super().__init__(size, device, align_bytes=4096)
        self.base_pointer = self.tensor.data_ptr()
        hipFileBufRegister(ctypes.c_void_p(self.base_pointer), size, flags=0)

    def __del__(self):
        self.hipFileBufDeregister(ctypes.c_void_p(self.base_pointer))

    def __str__(self):
        return "HipFileMemoryAllocator"


class PagedCpuGpuMemoryAllocator(MemoryAllocatorInterface):
    """
    Paged Memory Allocator for both CPU and GPU memory.
    This is a paged memory allocator for PD and P2P sharing
    when NIXL is enabled as NIXL relies on the paging abstraction.
    """

    def __init__(self):
        pass

    def init_gpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        device: str = "cuda",
    ):
        self.gpu_buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )
        self.gpu_allocator = PagedTensorMemoryAllocator(
            self.gpu_buffer,
            shapes,
            dtypes,
            fmt,
        )

    def init_cpu_memory_allocator(
        self,
        size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        numa_mapping: Optional[NUMAMapping] = None,
    ):
        self.cpu_buffer = _allocate_cpu_memory(size, numa_mapping)
        self.cpu_allocator = PagedTensorMemoryAllocator(
            self.cpu_buffer,
            shapes,
            dtypes,
            fmt,
        )
        self.align_bytes = self.cpu_allocator.align_bytes

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "cpu",
    ) -> Optional[MemoryObj]:
        if allocator_type == "gpu":
            return self.gpu_allocator.allocate(shapes, dtypes, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.allocate(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = "gpu",
    ) -> Optional[List[MemoryObj]]:
        if allocator_type == "gpu":
            return self.gpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        elif allocator_type == "cpu":
            return self.cpu_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = "cpu"):
        if allocator_type == "gpu":
            self.gpu_allocator.free(memory_obj)
        elif allocator_type == "cpu":
            self.cpu_allocator.free(memory_obj)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        if allocator_type == "gpu":
            self.gpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        elif allocator_type == "cpu":
            self.cpu_allocator.batched_free(memory_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported allocator type: {allocator_type}")

    def __str__(self):
        return "PDMemoryAllocator"


class XPUMemoryAllocator(MemoryAllocatorInterface):
    """Allocates memory in the pre-allocated XPU memory."""

    def __init__(
        self,
        size: int,
        device="xpu",
        align_bytes: Optional[int] = None,
        use_paging: bool = False,
        **kwargs,
    ):
        self.tensor = torch.empty((size,), dtype=torch.uint8, device=device)

        self.allocator: MemoryAllocatorInterface
        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.tensor,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            alloc_kwargs = {}
            if align_bytes is not None:
                alloc_kwargs["align_bytes"] = align_bytes
            self.allocator = TensorMemoryAllocator(self.tensor, **alloc_kwargs)

        self.device_mem_lock = threading.Lock() if not use_paging else nullcontext()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        with self.device_mem_lock:
            return self.allocator.allocate(shapes, dtypes, fmt, str(self))

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        with self.device_mem_lock:
            return self.allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt, str(self)
            )

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None):
        with self.device_mem_lock:
            self.allocator.free(memory_obj)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        with self.device_mem_lock:
            self.allocator.batched_free(memory_objs)

    def memcheck(self):
        with self.device_mem_lock:
            return self.allocator.memcheck()

    def close(self):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    def __str__(self):
        return "XPUMemoryAllocator"
