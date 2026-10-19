# SPDX-License-Identifier: Apache-2.0
"""Shared CPU source operations, independent of device engines and collectives."""

# Standard
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Union
import os
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.mooncake_layout import mooncake_valid_tokens
from lmcache.v1.remote_fill.security import content_digest
from lmcache.v1.serving_perf import (
    serving_perf_enabled,
    serving_perf_log,
    serving_perf_now,
    serving_perf_scope,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class _RemoteFillChunkLookupPlan:
    start: int
    end: int
    # Raw chunk hash: int for builtin/64-bit algorithms, digest bytes for
    # digest-based algorithms (e.g. sha256_cbor in this vLLM fork).
    chunk_hash: Union[int, bytes]
    locations_by_group: tuple[str, str]
    page_by_group: tuple[bool, bool]


@dataclass(frozen=True, slots=True)
class _RemoteFillLookupPlan:
    chunks: tuple[_RemoteFillChunkLookupPlan, ...]


class _RemoteFillMaterializationError(RuntimeError):
    """A retained RemoteFill plan cannot be safely materialized for loading."""


@dataclass(frozen=True, slots=True)
class _SharedRank0ObjectContext:
    parent_allocator: MemoryAllocatorInterface
    slab_size: int
    dtype: torch.dtype
    fmt: MemoryFormat


class CPUCacheOperations:
    """CPU algorithms shared by local engines and the independent storage owner.

    Instances provide config, registered metadata/layout, token database,
    StorageManager, shared-slab identity and lookup/ownership registries. This
    mixin never creates an engine, connector, device stream or TP collective.
    """

    def _expected_shared_cpu_chunk_metadata(
        self,
        *,
        kv_group: int,
        num_tokens: int,
        layer_id: int = 0,
    ) -> tuple[torch.Size, torch.dtype, MemoryFormat]:
        """Read exact flattened storage geometry from the frozen source layout."""
        layout = self.layout
        if layout is None:
            raise ValueError("CPU source layout has not been registered")
        if not 0 <= kv_group < len(layout.groups) or num_tokens <= 0:
            raise ValueError("Invalid CPU source group or token count")
        group = layout.groups[kv_group]
        if not 0 <= layer_id < len(group.layer_names):
            raise ValueError("CPU source layer exceeds registered physical owners")
        dtype = getattr(torch, group.layer_dtypes[layer_id].removeprefix("torch."))
        size = num_tokens * sum(group.plane_token_bytes[layer_id])
        if size % dtype.itemsize:
            raise ValueError("CPU source byte geometry is not dtype-aligned")
        return (
            torch.Size([size // dtype.itemsize]),
            dtype,
            MemoryFormat(group.memory_format),
        )

    def num_layers_for_group(self, kv_group: int = 0) -> int:
        """Read immutable transfer counts; worker registration validates them."""
        if not getattr(self, "dsa_two_groups", False):
            return self.num_layers
        if kv_group not in (0, 1):
            raise ValueError(f"KV group is out of range: {kv_group}")
        return self._transfer_layer_counts[kv_group]

    def _shared_cpu_dtype_for_kv_group(self, kv_group: int) -> torch.dtype:
        dtypes = self.metadata.get_dtypes()
        if 0 <= kv_group < len(dtypes):
            return dtypes[kv_group]
        if kv_group != 0:
            if len(dtypes) == 1 and getattr(self, "dsa_two_groups", False):
                return dtypes[0]
            raise ValueError(
                "KV group dtype metadata is unavailable: "
                f"kv_group={kv_group}, num_dtypes={len(dtypes)}"
            )
        return self.metadata.kv_dtype

    def _metadata_shape_for_kv_group(
        self,
        kv_group: int,
        num_tokens: int,
    ) -> Optional[torch.Size]:
        shapes = self.metadata.get_shapes(num_tokens)
        if 0 <= kv_group < len(shapes):
            return torch.Size(shapes[kv_group])
        if kv_group == 0 and shapes:
            return torch.Size(shapes[0])
        return None

    def _lookup_key_for_kv_group(
        self,
        base_key: CacheEngineKey,
        *,
        kv_group: int,
        request_configs: Optional[dict],
    ) -> CacheEngineKey:
        if kv_group == base_key.kv_group:
            return base_key
        make_key = self.token_database._make_key_by_hash
        return make_key(
            base_key.chunk_hash,
            request_configs,
            kv_group=kv_group,
            valid_tokens=mooncake_valid_tokens(base_key, int(self.config.chunk_size)),
        )

    def _persistent_direct_hbm_split_group_enabled(self) -> bool:
        return self.config.dsa_group1_load_mode == "persistent_direct_hbm"

    def _lookup_persistent_direct_hbm_prefix(
        self,
        chunks: list[tuple[int, int, CacheEngineKey]],
        group_keys: dict[int, list[LayerCacheEngineKey]],
        *,
        search_range: list[str],
        lookup_id: Optional[str],
        pin: bool,
    ) -> int:
        """Prove a Mooncake pair prefix, then overlay role-appropriate CPU hits."""

        assert self.storage_manager is not None
        if "RemoteBackend" not in search_range:
            return 0

        local_mapping: dict[str, list[CacheEngineKey]] = defaultdict(list)
        pins_registered = False
        try:
            pair_count, persistent_mapping = (
                self.storage_manager.batched_contains_two_group_layer_pages(
                    group_keys[0],
                    group_keys[1],
                    ["RemoteBackend"],
                    False,
                )
            )
            if not 0 <= pair_count <= len(chunks):
                raise RuntimeError(
                    "Persistent two-group lookup returned an invalid prefix"
                )
            persistent_keys = persistent_mapping.get("RemoteBackend", [])
            if (
                set(persistent_mapping) - {"RemoteBackend"}
                or len(persistent_keys) != 2 * pair_count
            ):
                raise RuntimeError(
                    "Persistent two-group lookup returned invalid backend mapping"
                )
            if pair_count == 0:
                return 0

            local_counts = [0, 0]
            if "LocalCPUBackend" in search_range:
                # P loads both groups through shared CPU pages. D must keep
                # Group 1 on the persistent direct-to-HBM path.
                local_groups = (
                    (0, 1)
                    if pin and getattr(self.config, "pd_role", None) == "sender"
                    else (0,)
                )
                for group in local_groups:
                    local_count, group_mapping = (
                        self.storage_manager.batched_contains_layer_pages(
                            group_keys[group][:pair_count],
                            ["LocalCPUBackend"],
                            pin,
                        )
                    )
                    # Retain each returned pin before validation so a later
                    # group failure also rolls back earlier successful pins.
                    for location, keys in group_mapping.items():
                        local_mapping[location].extend(keys)
                    local_keys = group_mapping.get("LocalCPUBackend", [])
                    if (
                        not 0 <= local_count <= pair_count
                        or set(group_mapping) - {"LocalCPUBackend"}
                        or len(local_keys) != local_count
                    ):
                        raise RuntimeError(
                            f"Group-{group} LocalCPU overlay returned invalid "
                            "prefix mapping"
                        )
                    local_counts[group] = local_count

            plan = tuple(
                _RemoteFillChunkLookupPlan(
                    start=start,
                    end=end,
                    chunk_hash=key.chunk_hash,
                    locations_by_group=(
                        "LocalCPUBackend"
                        if index < local_counts[0]
                        else "RemoteBackend",
                        "LocalCPUBackend"
                        if index < local_counts[1]
                        else "RemoteBackend",
                    ),
                    page_by_group=(True, True),
                )
                for index, (start, end, key) in enumerate(chunks[:pair_count])
            )
            if pin:
                assert lookup_id is not None
                for location, keys in local_mapping.items():
                    self.lookup_pins[lookup_id][location].extend(keys)
                pins_registered = True
                self._remote_fill_lookup_plans[lookup_id] = _RemoteFillLookupPlan(plan)
            return plan[-1].end
        except Exception:
            if pin:
                if pins_registered:
                    # The request registry now owns the pins; remove it before
                    # the caller's error cleanup can attempt another release.
                    assert lookup_id is not None
                    self._release_lookup_pins(lookup_id)
                else:
                    for location, keys in local_mapping.items():
                        if keys:
                            self.storage_manager.batched_unpin(keys, [location])
            raise

    def _lookup_remote_fill_two_group_prefix(
        self,
        chunks: list[tuple[int, int, CacheEngineKey]],
        *,
        search_range: list[str],
        lookup_id: Optional[str],
        pin: bool,
        request_configs: Optional[dict],
        diagnostics: Optional[dict[str, object]] = None,
    ) -> int:
        """Locate and retain a same-tier two-group prefix for actual load."""
        if not chunks:
            return 0
        if pin:
            assert lookup_id is not None, "lookup_id is required when pin is True"
        assert self.storage_manager is not None

        group_keys = {
            group: [
                self._lookup_key_for_kv_group(
                    key,
                    kv_group=group,
                    request_configs=request_configs,
                ).get_first_layer()
                for _, _, key in chunks
            ]
            for group in (0, 1)
        }
        if self._persistent_direct_hbm_split_group_enabled():
            return self._lookup_persistent_direct_hbm_prefix(
                chunks,
                group_keys,
                search_range=search_range,
                lookup_id=lookup_id,
                pin=pin,
            )
        retained: dict[str, list[CacheEngineKey]] = defaultdict(list)
        plan: list[_RemoteFillChunkLookupPlan] = []

        def release(mapping: dict[str, list[CacheEngineKey]]) -> None:
            if not pin:
                return
            for location, keys in mapping.items():
                if keys:
                    self.storage_manager.batched_unpin(keys, [location])

        def append_mapping(mapping: dict[str, list[CacheEngineKey]]) -> None:
            for location, keys in mapping.items():
                retained[location].extend(keys)

        def mapping_locations(
            mapping: dict[str, list[CacheEngineKey]], expected_pairs: int
        ) -> list[str]:
            locations: list[str] = []
            for location, keys in mapping.items():
                if len(keys) % 2:
                    raise RuntimeError(
                        "Two-group page lookup returned an unpaired mapping"
                    )
                locations.extend([location] * (len(keys) // 2))
            if len(locations) != expected_pairs:
                raise RuntimeError(
                    "Two-group page lookup returned incomplete backend mapping"
                )
            return locations

        def add_page_plans(start_index: int, locations: list[str]) -> None:
            for offset, location in enumerate(locations):
                start, end, key = chunks[start_index + offset]
                plan.append(
                    _RemoteFillChunkLookupPlan(
                        start=start,
                        end=end,
                        chunk_hash=key.chunk_hash,
                        locations_by_group=(location, location),
                        page_by_group=(True, True),
                    )
                )

        def finalize() -> int:
            if pin and plan:
                assert lookup_id is not None
                for location, keys in retained.items():
                    self.lookup_pins[lookup_id][location].extend(keys)
                self._remote_fill_lookup_plans[lookup_id] = _RemoteFillLookupPlan(
                    tuple(plan)
                )
            return plan[-1].end if plan else 0

        try:
            covered = 0
            local_pairs = 0
            local_lookup_attempted = False
            if "LocalCPUBackend" in search_range:
                local_lookup_attempted = True
                local_pairs, local_mapping = (
                    self.storage_manager.batched_contains_two_group_layer_pages(
                        group_keys[0],
                        group_keys[1],
                        ["LocalCPUBackend"],
                        pin,
                        diagnostics=diagnostics,
                    )
                )
                append_mapping(local_mapping)
                add_page_plans(0, mapping_locations(local_mapping, local_pairs))
                covered = local_pairs

            if diagnostics is not None:
                hint = self._remote_fill_local_full_hint(request_configs)
                if hint is not None:
                    required_store_end, _ = hint
                    required_pairs = sum(
                        start < required_store_end for start, _, _ in chunks
                    )
                    diagnostics.update(
                        local_lookup_attempted=local_lookup_attempted,
                        local_pairs=min(local_pairs, required_pairs),
                        required_pairs=required_pairs,
                    )
                    try:
                        diagnostics["required_key_digest"] = content_digest(
                            (
                                tuple(
                                    key.without_layer().to_string()
                                    for key in group_keys[0][:required_pairs]
                                ),
                                tuple(
                                    key.without_layer().to_string()
                                    for key in group_keys[1][:required_pairs]
                                ),
                            )
                        )
                    except Exception:
                        # Diagnostics must not invalidate an already-completed
                        # paired lookup or alter its pin ownership.
                        diagnostics["key_digest_error"] = True
                        logger.warning(
                            "Failed to fingerprint remote-fill lookup keys",
                            exc_info=True,
                        )

            persistent_range = [
                location for location in search_range if location != "LocalCPUBackend"
            ]
            fallback_range = (
                search_range
                if getattr(self, "_cpu_runtime_local_flat_sources", False)
                else persistent_range
            )
            if covered == len(chunks) or not fallback_range:
                return finalize()

            page_pairs, page_mapping = (
                (
                    self.storage_manager.batched_contains_two_group_layer_pages(
                        group_keys[0][covered:],
                        group_keys[1][covered:],
                        persistent_range,
                        pin,
                    )
                )
                if persistent_range
                else (0, {})
            )
            append_mapping(page_mapping)
            add_page_plans(
                covered,
                mapping_locations(page_mapping, page_pairs),
            )
            covered += page_pairs

            def locate_group(
                chunk_index: int,
                kv_group: int,
            ) -> tuple[str, dict[str, list[CacheEngineKey]], bool] | None:
                page_key = group_keys[kv_group][chunk_index]
                hits, mapping = self.storage_manager.batched_contains_layer_pages(
                    [page_key],
                    fallback_range,
                    pin,
                )
                if hits == 1 and len(mapping) == 1:
                    return next(iter(mapping)), mapping, True
                release(mapping)

                layer_keys = self._lookup_key_for_kv_group(
                    chunks[chunk_index][2],
                    kv_group=kv_group,
                    request_configs=request_configs,
                ).split_layers(self.num_layers_for_group(kv_group))
                hits, mapping = self.storage_manager.batched_contains(
                    layer_keys,
                    fallback_range,
                    pin,
                )
                if hits == len(layer_keys) and len(mapping) == 1:
                    return next(iter(mapping)), mapping, False
                release(mapping)
                return None

            while covered < len(chunks):
                group0 = locate_group(covered, 0)
                if group0 is None:
                    break
                try:
                    group1 = locate_group(covered, 1)
                except Exception:
                    release(group0[1])
                    raise
                if group1 is None or group0[0] != group1[0]:
                    release(group0[1])
                    if group1 is not None:
                        release(group1[1])
                    break
                append_mapping(group0[1])
                append_mapping(group1[1])
                start, end, key = chunks[covered]
                plan.append(
                    _RemoteFillChunkLookupPlan(
                        start=start,
                        end=end,
                        chunk_hash=key.chunk_hash,
                        locations_by_group=(group0[0], group0[0]),
                        page_by_group=(group0[2], group1[2]),
                    )
                )
                covered += 1
            return finalize()
        except Exception:
            release(dict(retained))
            raise

    @staticmethod
    def _remote_fill_local_full_hint(
        request_configs: Optional[dict],
    ) -> Optional[tuple[int, int]]:
        """Return the sanitized LOCAL_FULL handoff hint, if present.

        This metadata is diagnostic only.  In particular, callers must still
        perform the ordinary paired lookup-and-pin before consulting it.
        """

        if not isinstance(request_configs, dict):
            return None
        hint = request_configs.get("lmcache.remote_fill_result")
        if not isinstance(hint, dict) or hint.get("outcome") != "LOCAL_FULL":
            return None
        required_store_end = hint.get("required_store_end")
        destination_engine_epoch = hint.get("destination_engine_epoch")
        if (
            isinstance(required_store_end, bool)
            or not isinstance(required_store_end, int)
            or required_store_end <= 0
            or isinstance(destination_engine_epoch, bool)
            or not isinstance(destination_engine_epoch, int)
            or destination_engine_epoch <= 0
        ):
            return None
        return required_store_end, destination_engine_epoch

    def _remote_fill_retrieve_plan(
        self,
        req_id: str,
        candidates: list[tuple[int, int, CacheEngineKey]],
        kv_group: int,
    ) -> Optional[list[tuple[str, bool]]]:
        plan = getattr(self, "_remote_fill_lookup_plans", {}).get(req_id)
        if plan is None or kv_group not in (0, 1):
            return None
        by_chunk = {
            (chunk.start, chunk.end, chunk.chunk_hash): chunk for chunk in plan.chunks
        }
        selected: list[tuple[str, bool]] = []
        for start, end, key in candidates:
            chunk = by_chunk.get((start, end, key.chunk_hash))
            if chunk is None:
                raise _RemoteFillMaterializationError(
                    "Remote-fill actual-load candidates do not match the "
                    f"retained paired lookup plan: req_id={req_id}, "
                    f"kv_group={kv_group}, start={start}, end={end}"
                )
            selected.append(
                (
                    chunk.locations_by_group[kv_group],
                    chunk.page_by_group[kv_group],
                )
            )
        return selected

    def _shared_local_cpu_backend(self):
        if self.storage_manager is None:
            raise ValueError("StorageManager is required for shared CPU cache")
        local_cpu_backend = getattr(self.storage_manager, "local_cpu_backend", None)
        if local_cpu_backend is None:
            raise ValueError(
                "Shared CPU cache requires LocalCPUBackend on rank0. "
                "Check local_cpu=true and enable_shared_cpu_cache=true."
            )
        return local_cpu_backend

    @staticmethod
    def _shared_cpu_allocator_address_manager(root_allocator: Any) -> Any:
        address_manager = getattr(root_allocator, "address_manager", None)
        if address_manager is not None:
            return address_manager
        pin_allocator = getattr(root_allocator, "pin_allocator", None)
        return getattr(pin_allocator, "address_manager", None)

    def _shared_cpu_capacity_snapshot(self) -> dict[str, Any]:
        try:
            local_cpu_backend = self._shared_local_cpu_backend()
            allocator = getattr(local_cpu_backend, "memory_allocator", None)
            root_allocator = getattr(allocator, "_allocator", allocator)
            snapshot: dict[str, Any] = {}
            buffer = getattr(root_allocator, "buffer", None)
            if buffer is not None:
                snapshot["slab_bytes"] = int(buffer.numel())
            address_manager = self._shared_cpu_allocator_address_manager(root_allocator)
            if address_manager is not None:
                get_free_size = getattr(address_manager, "get_free_size", None)
                if callable(get_free_size):
                    snapshot["free_bytes"] = int(get_free_size())
                get_heap_size = getattr(address_manager, "get_heap_size", None)
                if callable(get_heap_size):
                    snapshot["heap_bytes"] = int(get_heap_size())
                snapshot["allocated_bytes"] = int(
                    getattr(address_manager, "total_allocated_size", 0)
                )
            hot_cache = getattr(local_cpu_backend, "hot_cache", {})
            pinned_bytes = 0
            evictable_bytes = 0
            for mem_obj in hot_cache.values():
                physical_size = self._shared_cpu_mem_obj_physical_size(mem_obj)
                if getattr(mem_obj, "is_pinned", False):
                    pinned_bytes += physical_size
                if getattr(mem_obj, "can_evict", False):
                    evictable_bytes += physical_size
            snapshot["pinned_bytes"] = pinned_bytes
            snapshot["evictable_bytes"] = evictable_bytes
            snapshot["active_sparse_requests"] = sum(
                lease.active for lease in self._shared_cpu_request_leases.values()
            )
            return snapshot
        except Exception as exc:
            return {"capacity_snapshot_error": str(exc)}

    def _shared_cpu_mem_obj_physical_size(self, mem_obj: MemoryObj) -> int:
        metadata = getattr(mem_obj, "metadata", None)
        if metadata is not None and getattr(metadata, "phy_size", None) is not None:
            return int(metadata.phy_size)
        return int(mem_obj.get_physical_size())

    def _is_rank0_shared_mem_obj(self, mem_obj: MemoryObj) -> bool:
        try:
            local_cpu_backend = self._shared_local_cpu_backend()
            allocator = getattr(local_cpu_backend, "memory_allocator", None)
            if allocator is None:
                return False
            if getattr(allocator, "shm_name", None) != self.shared_cpu_cache_name:
                return False
            expected_parent = getattr(allocator, "pin_allocator", None)
            if mem_obj.parent() is not expected_parent:
                return False
            buffer = getattr(allocator, "buffer", None)
            if buffer is None:
                return False
            offset = int(mem_obj.metadata.address)
            physical_size = int(mem_obj.metadata.phy_size)
            return (
                offset >= 0
                and physical_size > 0
                and offset + physical_size <= int(buffer.numel())
            )
        except Exception:
            return False

    def _shared_rank0_object_context(
        self,
        kv_group: int,
    ) -> Optional[_SharedRank0ObjectContext]:
        """Cache stable shared-slab invariants for one remote batch."""
        try:
            allocator = self._shared_local_cpu_backend().memory_allocator
            parent = allocator.pin_allocator
            if (
                allocator.shm_name != self.shared_cpu_cache_name
                or parent is None
                or allocator.buffer is None
            ):
                return None
            return _SharedRank0ObjectContext(
                parent,
                int(allocator.buffer.numel()),
                self._shared_cpu_dtype_for_kv_group(kv_group),
                self._memory_format_for_kv_group(kv_group),
            )
        except Exception:
            return None

    @staticmethod
    def _shared_rank0_object_metadata(
        mem_obj: MemoryObj,
        context: _SharedRank0ObjectContext,
    ) -> Optional[MemoryObjMetadata]:
        """Return metadata only for an object contained by the expected slab."""
        try:
            metadata = mem_obj.metadata
            offset = int(metadata.address)
            physical_size = int(metadata.phy_size)
            if (
                mem_obj.parent() is not context.parent_allocator
                or offset < 0
                or physical_size <= 0
                or offset + physical_size > context.slab_size
            ):
                return None
            return metadata
        except Exception:
            return None

    def _copy_memory_obj_bytes(
        self,
        dst: MemoryObj,
        src: MemoryObj,
    ) -> None:
        dst_tensor = dst.tensor
        src_tensor = src.tensor
        if dst_tensor is not None and src_tensor is not None:
            dst_tensor.copy_(src_tensor, non_blocking=True)
            return
        dst_view = dst.byte_array
        src_view = src.byte_array
        logical_size = int(src.get_size())
        dst_view[:logical_size] = src_view[:logical_size]

    def _materialize_shared_rank0_copy(
        self,
        *,
        key: CacheEngineKey,
        src_obj: MemoryObj,
        req_id: str,
        phase: str,
        layer_id: int,
        kv_group: int,
        chunk_index: int,
    ) -> MemoryObj:
        local_cpu_backend = self._shared_local_cpu_backend()
        hot_obj = local_cpu_backend.get_blocking(key)
        if hot_obj is not None:
            if self._is_rank0_shared_mem_obj(hot_obj):
                return hot_obj
            hot_obj.ref_count_down()
            local_cpu_backend.remove(key, force=True)

        materialized = local_cpu_backend.allocate(
            src_obj.get_shapes(),
            src_obj.get_dtypes(),
            fmt=src_obj.get_memory_format(),
            eviction=True,
            busy_loop=False,
        )
        if materialized is None:
            raise ValueError(
                "Shared CPU cache capacity error while materializing fetched "
                "chunk into rank0 shm-backed LocalCPU: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"key={key}, capacity={self._shared_cpu_capacity_snapshot()}"
            )

        try:
            self._copy_memory_obj_bytes(materialized, src_obj)
            local_cpu_backend.submit_put_task(key, materialized)
            hot_obj = local_cpu_backend.get_blocking(key)
            materialized.ref_count_down()
            if hot_obj is None:
                raise ValueError(
                    "Shared CPU cache failed to install materialized chunk "
                    "into rank0 hot cache: "
                    f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                    f"kv_group={kv_group}, chunk_index={chunk_index}, key={key}"
                )
            if not self._is_rank0_shared_mem_obj(hot_obj):
                hot_obj.ref_count_down()
                raise ValueError(
                    "Shared CPU cache materialized chunk is not shm-backed "
                    "after LocalCPU install: "
                    f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                    f"kv_group={kv_group}, chunk_index={chunk_index}, key={key}"
                )
            return hot_obj
        except Exception:
            try:
                local_cpu_backend.remove(key, force=True)
            except Exception:
                logger.debug(
                    "Failed to remove materialized shared CPU cache key after "
                    "copy/install error",
                    exc_info=True,
                )
            if materialized.is_valid():
                materialized.ref_count_down()
            raise

    def _validate_rank0_shared_mem_obj(
        self,
        mem_obj: MemoryObj,
        *,
        req_id: str,
        phase: str,
        layer_id: int,
        kv_group: int,
        chunk_index: int,
        _context: Optional[_SharedRank0ObjectContext] = None,
        _metadata: Optional[MemoryObjMetadata] = None,
        _require_pinned: bool = True,
    ) -> None:
        context = _context or self._shared_rank0_object_context(kv_group)
        if context is None:
            raise ValueError(
                "Shared CPU cache object is not backed by the resolved shm slab: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        metadata = _metadata or mem_obj.metadata
        if _metadata is None and mem_obj.parent() is not context.parent_allocator:
            raise ValueError(
                "Shared CPU cache object does not belong to rank0's shm-backed "
                "LocalCPUBackend allocator: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        if metadata.dtype is None:
            raise ValueError(
                "Shared CPU cache cannot publish dtype-less MemoryObj: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        if metadata.dtype != context.dtype:
            raise ValueError(
                "Shared CPU cache object dtype does not match KV group: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"dtype={metadata.dtype}, expected={context.dtype}"
            )
        if metadata.fmt == MemoryFormat.UNDEFINED:
            raise ValueError(
                "Shared CPU cache cannot publish MemoryObj with undefined format: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        if metadata.fmt != context.fmt:
            raise ValueError(
                "Shared CPU cache object format does not match KV group: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"fmt={metadata.fmt}, expected={context.fmt}"
            )
        offset = int(metadata.address)
        physical_size = int(metadata.phy_size)
        logical_size = int(mem_obj.get_size())
        if (
            logical_size <= 0
            or logical_size > physical_size
            or (
                _metadata is None
                and (
                    offset < 0
                    or physical_size <= 0
                    or offset + physical_size > context.slab_size
                )
            )
        ):
            raise ValueError(
                "Shared CPU cache object has invalid slab bounds: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"offset={offset}, logical_size={logical_size}, "
                f"physical_size={physical_size}, slab_size={context.slab_size}"
            )
        if _require_pinned and metadata.pin_count <= 0:
            raise ValueError(
                "Shared CPU cache object must be pinned before handle publication: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )

    def _resolve_shared_rank0_layer_mem_objs(
        self,
        *,
        req_id: str,
        phase: str,
        layer_id: int,
        kv_group: int,
        keys_layer: list[CacheEngineKey],
        chunk_locations: Optional[list[str]] = None,
        local_prefix: Optional[LocalCPUPrefixGetResult] = None,
        enforce_planned_location: bool = False,
    ) -> list[MemoryObj]:
        """Resolve one layer into rank0 shm-backed LocalCPU MemoryObjs.

        The resolver is intentionally rank0-only. Its normal path checks every
        chunk in LocalCPU before using the planned backend. A supplied
        ``local_prefix`` instead preserves one LocalCPU-prefix/remote-suffix
        probe. Returned objects carry one caller reference and one request pin.
        """
        try:
            assert self.storage_manager is not None
            local_cpu_backend = self._shared_local_cpu_backend()
            if local_prefix is not None:
                if chunk_locations is not None:
                    raise ValueError(
                        "Shared CPU resolver cannot combine location metadata "
                        "with a LocalCPU prefix result"
                    )
                local_prefix.validate(keys_layer)
                local_count = len(local_prefix.local_memory_objs)
                resolved_chunk_locations = ["LocalCPUBackend"] * local_count + [
                    "RemoteBackend"
                ] * len(local_prefix.remote_positions)
            elif chunk_locations is None or len(keys_layer) != len(chunk_locations):
                raise ValueError(
                    "Shared CPU cache location metadata length mismatch: "
                    f"layer_id={layer_id}, kv_group={kv_group}, "
                    f"keys={len(keys_layer)}, "
                    f"locations={len(chunk_locations or [])}"
                )
            else:
                resolved_chunk_locations = chunk_locations
        except Exception:
            if local_prefix is not None:
                local_prefix.release()
            raise

        resolved: list[Optional[MemoryObj]] = [None] * len(keys_layer)
        grouped_positions: dict[str, list[int]] = defaultdict(list)
        acquired: list[MemoryObj] = []
        pinned: list[MemoryObj] = []

        try:
            for chunk_index, (key, planned_location) in enumerate(
                zip(keys_layer, resolved_chunk_locations, strict=True)
            ):
                hot_obj = (
                    local_prefix.take_local(chunk_index)
                    if local_prefix is not None
                    else local_cpu_backend.get_blocking(key)
                    if not enforce_planned_location
                    or planned_location == "LocalCPUBackend"
                    else None
                )
                if hot_obj is not None:
                    if self._is_rank0_shared_mem_obj(hot_obj):
                        logger.debug(
                            "[req_id=%s kv_group=%s layer=%s chunk=%s] "
                            "Shared CPU rank0 hot-cache hit",
                            req_id,
                            kv_group,
                            layer_id,
                            chunk_index,
                        )
                        resolved[chunk_index] = hot_obj
                        acquired.append(hot_obj)
                    else:
                        logger.warning(
                            "[req_id=%s kv_group=%s layer=%s chunk=%s] "
                            "Shared CPU rank0 hot-cache object is not "
                            "shm-backed; rematerializing into the shared "
                            "LocalCPU slab before handle publication.",
                            req_id,
                            kv_group,
                            layer_id,
                            chunk_index,
                        )
                        try:
                            materialized_obj = self._materialize_shared_rank0_copy(
                                key=key,
                                src_obj=hot_obj,
                                req_id=req_id,
                                phase=phase,
                                layer_id=layer_id,
                                kv_group=kv_group,
                                chunk_index=chunk_index,
                            )
                        finally:
                            hot_obj.ref_count_down()
                        resolved[chunk_index] = materialized_obj
                        acquired.append(materialized_obj)
                    continue
                grouped_positions[planned_location].append(chunk_index)

            for location, positions in grouped_positions.items():
                if location == "LocalCPUBackend":
                    raise ValueError(
                        "Shared CPU cache hot-cache metadata was stale: "
                        f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                        f"kv_group={kv_group}, positions={positions}"
                    )
                fetch_keys = [keys_layer[pos] for pos in positions]
                fetched = self.storage_manager.batched_get(
                    fetch_keys,
                    location=location,
                )
                if len(fetched) != len(fetch_keys):
                    for fetched_obj in fetched:
                        if fetched_obj is not None:
                            fetched_obj.ref_count_down()
                    raise ValueError(
                        "Shared CPU cache backend returned unexpected result count: "
                        f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                        f"kv_group={kv_group}, location={location}, "
                        f"expected={len(fetch_keys)}, got={len(fetched)}"
                    )
                missing_positions: list[int] = []
                for pos, fetched_obj in zip(positions, fetched, strict=True):
                    key = keys_layer[pos]
                    if fetched_obj is None:
                        missing_positions.append(pos)
                        continue

                    if self._is_rank0_shared_mem_obj(fetched_obj):
                        # RemoteBackend receives Mooncake data through the same
                        # shm-backed LocalCPU allocator used for publication.
                        # Keep the returned caller reference directly instead
                        # of taking another hot-cache lookup/ref for every
                        # chunk. StorageManager has already installed a cache
                        # reference when the complete batch succeeded.
                        acquired.append(fetched_obj)
                        resolved[pos] = fetched_obj
                        continue

                    hot_obj = (
                        None
                        if enforce_planned_location
                        else local_cpu_backend.get_blocking(key)
                    )
                    if hot_obj is not None and self._is_rank0_shared_mem_obj(hot_obj):
                        acquired.append(hot_obj)
                        fetched_obj.ref_count_down()
                        resolved[pos] = hot_obj
                    else:
                        if hot_obj is not None:
                            hot_obj.ref_count_down()
                        try:
                            materialized_obj = self._materialize_shared_rank0_copy(
                                key=key,
                                src_obj=fetched_obj,
                                req_id=req_id,
                                phase=phase,
                                layer_id=layer_id,
                                kv_group=kv_group,
                                chunk_index=pos,
                            )
                        finally:
                            fetched_obj.ref_count_down()
                        acquired.append(materialized_obj)
                        resolved[pos] = materialized_obj

                if missing_positions:
                    raise ValueError(
                        "Shared CPU cache missing required chunks during rank0 "
                        "materialization. The chunks were absent from the "
                        "requested backend or could not be staged into the "
                        "shared LocalCPU slab under current capacity/eviction "
                        "state: "
                        f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                        f"kv_group={kv_group}, location={location}, "
                        f"positions={missing_positions}, "
                        f"capacity={self._shared_cpu_capacity_snapshot()}"
                    )

            missing = [i for i, obj in enumerate(resolved) if obj is None]
            if missing:
                raise ValueError(
                    "Shared CPU cache failed to resolve all required chunks: "
                    f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                    f"kv_group={kv_group}, missing_positions={missing}, "
                    f"capacity={self._shared_cpu_capacity_snapshot()}"
                )

            mem_objs: list[MemoryObj] = []
            for obj in resolved:
                assert obj is not None
                mem_objs.append(obj)

            for chunk_index, mem_obj in enumerate(mem_objs):
                mem_obj.pin()
                pinned.append(mem_obj)
                self._validate_rank0_shared_mem_obj(
                    mem_obj,
                    req_id=req_id,
                    phase=phase,
                    layer_id=layer_id,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                )
            return mem_objs
        except Exception:
            for mem_obj in reversed(pinned):
                mem_obj.unpin()
            for mem_obj in reversed(acquired):
                mem_obj.ref_count_down()
            if local_prefix is not None:
                local_prefix.release()
            raise

    def _resolve_shared_rank0_page_first_layers(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        local_prefix_layers: Optional[list[LocalCPUPrefixGetResult]] = None,
    ) -> list[list[MemoryObj]]:
        """Join the common LocalCPU prefix with one all-layer remote suffix.

        Supplied prefix results are consumed on both success and failure.
        """
        num_layers = self.num_layers_for_group(kv_group)
        if len(keys_layer_major) != num_layers or not keys_layer_major:
            raise ValueError("Page-first retrieval requires every model layer")
        chunks = len(keys_layer_major[0])
        if any(len(keys) != chunks for keys in keys_layer_major):
            raise ValueError("Page-first retrieval requires equal layer chunk counts")

        prefixes = (
            local_prefix_layers
            if local_prefix_layers is not None
            else self._shared_local_cpu_backend().batched_get_prefixes_with_misses(
                keys_layer_major
            )
        )
        if len(prefixes) != num_layers:
            for prefix in prefixes:
                prefix.release()
            raise ValueError("LocalCPU prefix lookup returned the wrong layer count")

        resolved: list[list[MemoryObj]] = [[] for _ in keys_layer_major]
        owned: list[MemoryObj] = []
        try:
            for prefix, keys in zip(prefixes, keys_layer_major, strict=True):
                prefix.validate(keys)
            local_chunks = min(len(prefix.local_memory_objs) for prefix in prefixes)

            if local_chunks:
                for layer_id, (prefix, keys) in enumerate(
                    zip(prefixes, keys_layer_major, strict=True)
                ):
                    local_objs = [prefix.take_local(i) for i in range(local_chunks)]
                    prefix.release()
                    local = self._resolve_shared_rank0_layer_mem_objs(
                        req_id=req_id,
                        phase=phase,
                        layer_id=layer_id,
                        kv_group=kv_group,
                        keys_layer=keys[:local_chunks],
                        local_prefix=LocalCPUPrefixGetResult(local_objs, [], []),
                    )
                    resolved[layer_id].extend(local)
                    owned.extend(local)
            else:
                for prefix in prefixes:
                    prefix.release()

            if local_chunks < chunks:
                remote = self._resolve_shared_rank0_remote_layers_windowed(
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    keys_layer_major=[keys[local_chunks:] for keys in keys_layer_major],
                    layers_per_batch=num_layers,
                )
                owned.extend(obj for layer in remote for obj in layer)
                if len(remote) != num_layers or any(
                    len(layer) != chunks - local_chunks for layer in remote
                ):
                    raise ValueError(
                        "Page-first remote suffix returned the wrong shape"
                    )
                for layer, suffix in zip(resolved, remote, strict=True):
                    layer.extend(suffix)
            return resolved
        except Exception:
            for prefix in prefixes:
                prefix.release()
            self._release_shared_retrieve_objs(owned, unpin=True)
            raise

    def _resolve_shared_rank0_exact_layer_pages(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        page_chunks: int,
        base_page_keys: list[CacheEngineKey],
        chunk_locations: list[str],
    ) -> tuple[list[list[MemoryObj]], int]:
        """Resolve a retained RemoteFill plan without reselecting its tier."""
        chunks = len(keys_layer_major[0])
        if (
            len(keys_layer_major) != self.num_layers_for_group(kv_group)
            or any(len(keys) != chunks for keys in keys_layer_major)
            or len(base_page_keys) < page_chunks
            or len(chunk_locations) != chunks
            or not 0 <= page_chunks <= chunks
        ):
            raise ValueError("RemoteFill exact layer-page plan is malformed")

        pages: list[MemoryObj] = []
        owned: list[MemoryObj] = []
        pinned: list[MemoryObj] = []
        try:
            offset = 0
            while offset < page_chunks:
                location = chunk_locations[offset]
                run_end = offset + 1
                while run_end < page_chunks and chunk_locations[run_end] == location:
                    run_end += 1
                run_keys = base_page_keys[offset:run_end]
                if location == "LocalCPUBackend":
                    fetched, count = (
                        self._shared_local_cpu_backend().batched_get_layer_page_prefix(
                            run_keys
                        )
                    )
                    if count != len(run_keys):
                        owned.extend(fetched)
                        raise ValueError(
                            "Pinned RemoteFill LocalCPU page plan became unavailable"
                        )
                else:
                    assert self.storage_manager is not None
                    backend = self.storage_manager.storage_backends.get(location)
                    retrieve = getattr(backend, "batched_get_layer_pages", None)
                    if not callable(retrieve):
                        raise RuntimeError(
                            f"RemoteFill backend {location!r} cannot retrieve pages"
                        )
                    fetched = retrieve(run_keys)
                if len(fetched) != len(run_keys):
                    owned.extend(fetched)
                    raise ValueError(
                        "RemoteFill exact layer-page result count is inconsistent"
                    )
                pages.extend(fetched)
                owned.extend(fetched)
                offset = run_end

            tail: list[list[MemoryObj]] = []
            tail_locations = chunk_locations[page_chunks:]
            for layer_id, layer_keys in enumerate(keys_layer_major):
                layer = self._resolve_shared_rank0_layer_mem_objs(
                    req_id=req_id,
                    phase=phase,
                    layer_id=layer_id,
                    kv_group=kv_group,
                    keys_layer=layer_keys[page_chunks:],
                    chunk_locations=tail_locations,
                    enforce_planned_location=True,
                )
                tail.append(layer)
                owned.extend(layer)
                pinned.extend(layer)

            invalid_page = any(
                not isinstance(page, LayerPageMemoryObj)
                or page.num_layers != self.num_layers_for_group(kv_group)
                or page.get_shape()
                != self._expected_shared_cpu_chunk_metadata(
                    kv_group=kv_group,
                    num_tokens=mooncake_valid_tokens(
                        base_page_keys[index], self.config.chunk_size
                    ),
                )[0]
                for index, page in enumerate(pages)
            )
            if invalid_page or not LayerPageMemoryObj.pin_many(pages):
                raise ValueError("Invalid or unpinnable RemoteFill layer page")
            pinned.extend(pages)
            for chunk_index, page in enumerate(pages):
                self._validate_rank0_shared_mem_obj(
                    page,
                    req_id=req_id,
                    phase=phase,
                    layer_id=0,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                )
            return [list(pages) + layer for layer in tail], page_chunks
        except Exception:
            for obj in reversed(pinned):
                obj.unpin()
            self._release_shared_retrieve_objs(owned, unpin=False)
            raise

    def _resolve_shared_rank0_layer_pages(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        page_chunks: int,
        base_page_keys: Optional[list[CacheEngineKey]] = None,
        exact_chunk_locations: Optional[list[str]] = None,
    ) -> tuple[list[list[MemoryObj]], int]:
        """Resolve merged pages and expand layer keys only for a legacy suffix."""
        if not 0 < page_chunks <= len(keys_layer_major[0]):
            raise ValueError("Layer-page retrieval requires at least one page")
        page_keys = (
            list(base_page_keys[:page_chunks])
            if base_page_keys is not None
            else [
                key.without_layer()
                for key in keys_layer_major[0][:page_chunks]
                if isinstance(key, LayerCacheEngineKey)
            ]
        )
        if len(page_keys) != page_chunks:
            raise ValueError("Layer-page retrieval requires one base key per page")
        if exact_chunk_locations is not None:
            return self._resolve_shared_rank0_exact_layer_pages(
                req_id=req_id,
                phase=phase,
                kv_group=kv_group,
                keys_layer_major=keys_layer_major,
                page_chunks=page_chunks,
                base_page_keys=page_keys,
                chunk_locations=exact_chunk_locations,
            )

        local = self._shared_local_cpu_backend()
        pages, local_count = local.batched_get_layer_page_prefix(page_keys)
        owned: list[MemoryObj] = list(pages)
        pinned: list[MemoryObj] = []
        try:
            tail_start = local_count
            while tail_start < page_chunks:
                if local.contains_all_exact(
                    page_keys[tail_start].split_layers(
                        self.num_layers_for_group(kv_group)
                    )
                ):
                    break
                remote = self.storage_manager.storage_backends.get("RemoteBackend")
                contains = getattr(remote, "batched_contains_layer_pages", None)
                retrieve = getattr(remote, "batched_get_layer_pages", None)
                if not callable(contains) or not callable(retrieve):
                    raise RuntimeError("RemoteBackend does not support layer pages")
                remote_page_keys = page_keys[tail_start:page_chunks]
                resolver_call_id = (
                    f"{os.getpid()}:{time.monotonic_ns()}"
                    if serving_perf_enabled()
                    else None
                )
                with (
                    serving_perf_scope(
                        req_id=req_id,
                        rank=self.metadata.worker_id,
                        resolver_call_id=resolver_call_id,
                    )
                    if resolver_call_id is not None
                    else nullcontext()
                ):
                    remote_count = contains(remote_page_keys)
                    if not 0 <= remote_count <= len(remote_page_keys):
                        raise ValueError(
                            "Remote layer-page lookup returned invalid count"
                        )
                    fetched = (
                        retrieve(remote_page_keys[:remote_count])
                        if remote_count
                        else []
                    )
                if remote_count:
                    if len(fetched) != remote_count:
                        raise ValueError(
                            "Remote layer-page result count is inconsistent"
                        )
                    pages.extend(fetched)
                    owned.extend(fetched)
                tail_start += remote_count
                if tail_start < page_chunks:
                    # Eviction can leave arbitrarily alternating tier runs.
                    fetched, count = local.batched_get_layer_page_prefix(
                        page_keys[tail_start:page_chunks]
                    )
                    pages.extend(fetched)
                    owned.extend(fetched)
                    tail_start += count
                    if not count:
                        # Neither tier has a merged page at this boundary.
                        break
            legacy_page_layers = (
                [
                    list(layer)
                    for layer in zip(
                        *(
                            key.split_layers(self.num_layers_for_group(kv_group))
                            for key in page_keys[tail_start:page_chunks]
                        ),
                        strict=True,
                    )
                ]
                if tail_start < page_chunks
                else [[] for _ in range(self.num_layers_for_group(kv_group))]
            )
            tail_keys_layer_major = [
                legacy_page_layers[layer_id]
                + list(keys_layer_major[layer_id][page_chunks:])
                for layer_id in range(self.num_layers_for_group(kv_group))
            ]
            tail = (
                self._resolve_shared_rank0_page_first_layers(
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    keys_layer_major=tail_keys_layer_major,
                )
                if tail_keys_layer_major[0]
                else [[] for _ in keys_layer_major]
            )
            tail_objects = [obj for layer in tail for obj in layer]
            owned.extend(tail_objects)
            # The legacy resolver returns request-pinned objects. Track them
            # alongside pages so failures after tail resolution undo both.
            pinned.extend(tail_objects)
            expected_pages = tail_start
            if len(pages) != expected_pages:
                raise ValueError(
                    f"Layer-page result count {len(pages)} != {expected_pages}"
                )

            invalid_page = any(
                page.num_layers != self.num_layers_for_group(kv_group)
                or page.get_shape()
                != self._expected_shared_cpu_chunk_metadata(
                    kv_group=kv_group,
                    num_tokens=mooncake_valid_tokens(
                        page_keys[index], self.config.chunk_size
                    ),
                )[0]
                for index, page in enumerate(pages)
            )
            if invalid_page or not LayerPageMemoryObj.pin_many(pages):
                raise ValueError("Invalid or unpinnable layer-page object")
            pinned.extend(pages)
            for chunk_index, page in enumerate(pages):
                self._validate_rank0_shared_mem_obj(
                    page,
                    req_id=req_id,
                    phase=phase,
                    layer_id=0,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                )

            return [list(pages) + layer for layer in tail], expected_pages
        except Exception:
            for page in reversed(pinned):
                page.unpin()
            self._release_shared_retrieve_objs(owned, unpin=False)
            raise

    def _resolve_shared_rank0_remote_layers_windowed(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        layers_per_batch: int,
    ) -> list[list[MemoryObj]]:
        """Resolve remote-only layers with fewer synchronous backend calls.

        Mooncake allocates receive buffers from rank0's shm-backed
        ``LocalCPUBackend``. Each returned object is therefore already the
        final publication object; this method preserves that caller reference,
        pins it, and scatters a multi-layer batch back into layer-major order.
        Callers must only use this path after proving every requested chunk is
        in ``RemoteBackend``.
        """
        assert self.storage_manager is not None
        if layers_per_batch < 1:
            raise ValueError("layers_per_batch must be at least 1")
        if not keys_layer_major:
            return []

        resolved_layers: list[list[MemoryObj]] = [[] for _ in keys_layer_major]
        acquired: list[MemoryObj] = []
        pinned: list[MemoryObj] = []
        context = self._shared_rank0_object_context(kv_group)
        timing_hook = getattr(
            self,
            "_shared_rank0_resolver_timing_hook",
            None,
        )
        if not callable(timing_hook):
            timing_hook = None
        perf_enabled = serving_perf_enabled()
        perf_started = serving_perf_now() if perf_enabled else 0.0
        resolver_call_id = (
            f"{os.getpid()}:{time.monotonic_ns()}" if perf_enabled else None
        )
        stage_times: dict[str, float] = defaultdict(float)

        def start_stage() -> float:
            return (
                time.perf_counter() if timing_hook is not None or perf_enabled else 0.0
            )

        def finish_stage(stage: str, started: float) -> None:
            if not started:
                return
            elapsed = time.perf_counter() - started
            if timing_hook is not None:
                timing_hook(stage, elapsed)
            if perf_enabled:
                stage_times[stage] += elapsed

        started = start_stage()
        windows: list[tuple[int, int, list[tuple[int, int]], list[CacheEngineKey]]] = []
        for window_start in range(0, len(keys_layer_major), layers_per_batch):
            window_end = min(
                window_start + layers_per_batch,
                len(keys_layer_major),
            )
            coordinates: list[tuple[int, int]] = []
            fetch_keys: list[CacheEngineKey] = []
            for layer_id in range(window_start, window_end):
                for chunk_index, key in enumerate(keys_layer_major[layer_id]):
                    coordinates.append((layer_id, chunk_index))
                    fetch_keys.append(key)
            windows.append((window_start, window_end, coordinates, fetch_keys))
        finish_stage("windows", started)

        try:
            for (
                window_start,
                window_end,
                coordinates,
                fetch_keys,
            ) in windows:
                started = start_stage()
                try:
                    with (
                        serving_perf_scope(
                            req_id=req_id,
                            rank=self.metadata.worker_id,
                            resolver_call_id=resolver_call_id,
                        )
                        if perf_enabled
                        else nullcontext()
                    ):
                        fetched = self.storage_manager.batched_get(
                            fetch_keys,
                            location="RemoteBackend",
                        )
                finally:
                    finish_stage("remote_get", started)
                started = start_stage()
                if len(fetched) != len(fetch_keys):
                    for fetched_obj in fetched:
                        if fetched_obj is not None:
                            fetched_obj.ref_count_down()
                    raise ValueError(
                        "Shared CPU windowed remote retrieve returned an "
                        "unexpected result count: "
                        f"request_id={req_id}, phase={phase}, "
                        f"kv_group={kv_group}, layers=[{window_start}, "
                        f"{window_end}), expected={len(fetch_keys)}, "
                        f"got={len(fetched)}"
                    )

                missing = [
                    coordinates[index]
                    for index, fetched_obj in enumerate(fetched)
                    if fetched_obj is None
                ]
                if missing:
                    for fetched_obj in fetched:
                        if fetched_obj is not None:
                            fetched_obj.ref_count_down()
                    raise ValueError(
                        "Shared CPU windowed remote retrieve missed required "
                        "chunks: "
                        f"request_id={req_id}, phase={phase}, "
                        f"kv_group={kv_group}, missing={missing}"
                    )
                finish_stage("results", started)

                pending_fetched = list(fetched)
                try:
                    started = start_stage()
                    batch_validation: dict[int, bool] = {}
                    window_objects: list[
                        tuple[int, int, MemoryObj, Optional[MemoryObjMetadata]]
                    ] = []
                    for index, (layer_id, chunk_index) in enumerate(coordinates):
                        fetched_obj = pending_fetched[index]
                        assert fetched_obj is not None
                        key = fetch_keys[index]
                        trusted_batch_member = (
                            context is not None
                            and type(fetched_obj) is TensorMemoryObj
                            and fetched_obj.is_trusted_allocation_batch_member(
                                batch_validation,
                                parent=context.parent_allocator,
                                slab_size=context.slab_size,
                                dtype=context.dtype,
                                fmt=context.fmt,
                            )
                        )
                        metadata = (
                            self._shared_rank0_object_metadata(fetched_obj, context)
                            if context is not None and not trusted_batch_member
                            else None
                        )
                        if (
                            trusted_batch_member
                            or metadata is not None
                            or (
                                context is None
                                and self._is_rank0_shared_mem_obj(fetched_obj)
                            )
                        ):
                            mem_obj = fetched_obj
                            pending_fetched[index] = None
                        else:
                            try:
                                mem_obj = self._materialize_shared_rank0_copy(
                                    key=key,
                                    src_obj=fetched_obj,
                                    req_id=req_id,
                                    phase=phase,
                                    layer_id=layer_id,
                                    kv_group=kv_group,
                                    chunk_index=chunk_index,
                                )
                            finally:
                                fetched_obj.ref_count_down()
                                pending_fetched[index] = None
                            if context is not None:
                                metadata = self._shared_rank0_object_metadata(
                                    mem_obj,
                                    context,
                                )

                        acquired.append(mem_obj)
                        if not trusted_batch_member:
                            self._validate_rank0_shared_mem_obj(
                                mem_obj,
                                req_id=req_id,
                                phase=phase,
                                layer_id=layer_id,
                                kv_group=kv_group,
                                chunk_index=chunk_index,
                                _context=context,
                                _metadata=metadata,
                                _require_pinned=False,
                            )
                        window_objects.append(
                            (layer_id, chunk_index, mem_obj, metadata)
                        )
                    finish_stage("classification", started)

                    started = start_stage()
                    window_mem_objs = [item[2] for item in window_objects]
                    if all(
                        type(mem_obj) is TensorMemoryObj for mem_obj in window_mem_objs
                    ):
                        TensorMemoryObj.pin_many(window_mem_objs)
                        pinned.extend(window_mem_objs)
                    else:
                        for layer_id, chunk_index, mem_obj, metadata in window_objects:
                            if mem_obj.pin() is False:
                                raise ValueError(
                                    "Shared CPU cache failed to pin a resolved object: "
                                    f"request_id={req_id}, phase={phase}, "
                                    f"layer_id={layer_id}, kv_group={kv_group}, "
                                    f"chunk_index={chunk_index}"
                                )
                            pinned.append(mem_obj)
                            self._validate_rank0_shared_mem_obj(
                                mem_obj,
                                req_id=req_id,
                                phase=phase,
                                layer_id=layer_id,
                                kv_group=kv_group,
                                chunk_index=chunk_index,
                                _context=context,
                                _metadata=metadata,
                            )
                    finish_stage("pinning", started)

                    started = start_stage()
                    for layer_id, _chunk_index, mem_obj, _metadata in window_objects:
                        resolved_layers[layer_id].append(mem_obj)
                    finish_stage("scatter", started)
                finally:
                    for fetched_obj in pending_fetched:
                        if fetched_obj is not None:
                            fetched_obj.ref_count_down()

            if perf_enabled:
                serving_perf_log(
                    logger,
                    "remote_resolver",
                    started=perf_started,
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    layers=len(keys_layer_major),
                    chunks=(len(keys_layer_major[0]) if keys_layer_major else 0),
                    objects=sum(len(layer) for layer in resolved_layers),
                    windows=len(windows),
                    status="ok",
                    rank=self.metadata.worker_id,
                    resolver_call_id=resolver_call_id,
                    exclusive_ms=round(
                        max(
                            0.0,
                            serving_perf_now()
                            - perf_started
                            - stage_times.get("remote_get", 0.0),
                        )
                        * 1000,
                        3,
                    ),
                    **{
                        f"{stage}_ms": round(elapsed * 1000, 3)
                        for stage, elapsed in stage_times.items()
                    },
                )
            return resolved_layers
        except Exception as exc:
            started = start_stage()
            for mem_obj in reversed(pinned):
                if mem_obj.is_pinned:
                    mem_obj.unpin()
            for mem_obj in reversed(acquired):
                if mem_obj.is_valid():
                    mem_obj.ref_count_down()
            finish_stage("rollback", started)
            if perf_enabled:
                serving_perf_log(
                    logger,
                    "remote_resolver",
                    started=perf_started,
                    req_id=req_id,
                    phase=phase,
                    kv_group=kv_group,
                    layers=len(keys_layer_major),
                    chunks=(len(keys_layer_major[0]) if keys_layer_major else 0),
                    windows=len(windows),
                    status="error",
                    error=type(exc).__name__,
                    rank=self.metadata.worker_id,
                    resolver_call_id=resolver_call_id,
                    exclusive_ms=round(
                        max(
                            0.0,
                            serving_perf_now()
                            - perf_started
                            - stage_times.get("remote_get", 0.0),
                        )
                        * 1000,
                        3,
                    ),
                    **{
                        f"{stage}_ms": round(elapsed * 1000, 3)
                        for stage, elapsed in stage_times.items()
                    },
                )
            raise

    @staticmethod
    def _release_shared_retrieve_objs(
        memory_objs: list[MemoryObj], *, unpin: bool
    ) -> None:
        if unpin:
            for mem_obj in memory_objs:
                if mem_obj.is_pinned:
                    mem_obj.unpin()
        while memory_objs:
            mem_obj = memory_objs.pop()
            if mem_obj.is_valid():
                mem_obj.ref_count_down()

    def _release_lookup_pins(self, lookup_id: str) -> bool:
        """Release retained lookup pins and any paired retrieval plan."""
        getattr(self, "_remote_fill_lookup_plans", {}).pop(lookup_id, None)
        block_mapping = self.lookup_pins.pop(lookup_id, None)
        if block_mapping is None:
            return False
        assert self.storage_manager is not None
        for location, keys in block_mapping.items():
            self.storage_manager.batched_unpin(keys, [location])
        return True

    def _memory_format_for_kv_group(self, kv_group: int = 0) -> MemoryFormat:
        """Return the CPU chunk format tag for a KV cache group."""
        if kv_group == 1 and getattr(self.config, "dsa_two_groups", False):
            return MemoryFormat.KV_DSA_INDEX_FMT
        if self.metadata.use_mla:
            return MemoryFormat.KV_MLA_LATENT_FMT
        assert self.fmt is not None, "Memory format is not initialized"
        return self.fmt
