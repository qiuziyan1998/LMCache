# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
import threading
import time
from weakref import ReferenceType, ref

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.cold_start_perf import cold_start_perf_enabled, cold_start_perf_log
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MixedMemoryAllocator,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import (
    mooncake_layer_pages_enabled,
    mooncake_valid_tokens,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.system_detection import NUMADetector, SystemMemoryDetector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

_MAX_EXTERNAL_RETENTION_TRACES = 16
_MAX_EXTERNAL_RETENTION_MUTATIONS = 128


@dataclass
class LocalCPUPrefixGetResult:
    local_memory_objs: List[Optional[MemoryObj]]
    remote_positions: List[int]
    remote_keys: List[CacheEngineKey]

    def validate(self, keys: Sequence[CacheEngineKey]) -> None:
        local_count = len(self.local_memory_objs)
        expected_remote_positions = list(range(local_count, len(keys)))
        if (
            local_count > len(keys)
            or any(memory_obj is None for memory_obj in self.local_memory_objs)
            or self.remote_positions != expected_remote_positions
            or self.remote_keys != list(keys[local_count:])
        ):
            raise ValueError("LocalCPU prefix result is not aligned with its keys")

    def take_local(self, index: int) -> Optional[MemoryObj]:
        if index >= len(self.local_memory_objs):
            return None
        memory_obj = self.local_memory_objs[index]
        self.local_memory_objs[index] = None
        return memory_obj

    def release(self) -> None:
        for index in range(len(self.local_memory_objs) - 1, -1, -1):
            memory_obj = self.take_local(index)
            if memory_obj is not None:
                memory_obj.ref_count_down()


@dataclass(frozen=True)
class LayerPageBatchPutResult:
    """Result of an immutable layer-page batch admission.

    Attributes:
        inserted_keys: Keys newly admitted by this call, in input order.
        existing_keys: Keys whose previously committed entries won, in input
            order.
    """

    inserted_keys: tuple[CacheEngineKey, ...]
    existing_keys: tuple[CacheEngineKey, ...]


@dataclass(frozen=True)
class ExternalTwoGroupCommitResult:
    """Result of an atomic external two-group prefix publication.

    Attributes:
        committed: Whether complete required coverage was published.
        inserted_keys: Required keys newly admitted by this call.
        existing_keys: Required keys won by an earlier committed entry.
        redundant_pages: Ready pages not admitted and safe for the caller to
            release according to its reservation lifecycle.
        missing_keys: Required keys with neither a committed entry nor a ready
            reservation. This is empty whenever ``committed`` is true.
        lock_wait_seconds: Time spent waiting for the LocalCPU cache lock.
        lock_hold_seconds: Time spent holding the lock for validation and
            atomic admission.
        retention_trace_id: Opt-in diagnostic trace armed atomically with a
            successful commit. ``None`` when cold-start diagnostics are off.
    """

    committed: bool
    inserted_keys: tuple[CacheEngineKey, ...]
    existing_keys: tuple[CacheEngineKey, ...]
    redundant_pages: tuple[LayerPageMemoryObj, ...]
    missing_keys: tuple[CacheEngineKey, ...] = ()
    lock_wait_seconds: float = field(default=0.0, compare=False)
    lock_hold_seconds: float = field(default=0.0, compare=False)
    retention_trace_id: Optional[int] = field(default=None, compare=False)


@dataclass(frozen=True)
class _ExternalRetentionMutation:
    monotonic_ns: int
    cause: str
    operation: str
    pair_index: int
    kv_group: int
    old_type: str
    new_type: str
    removed_committed_page: bool
    callsite: str


@dataclass
class _ExternalTwoGroupRetentionTrace:
    trace_id: int
    started_ns: int
    required_keys: tuple[CacheEngineKey, ...]
    positions: dict[CacheEngineKey, tuple[int, int]]
    committed_pages: dict[CacheEngineKey, ReferenceType[MemoryObj]]
    mutations: list[_ExternalRetentionMutation] = field(default_factory=list)
    dropped_mutations: int = 0


class LayerPageAdmissionRollbackError(RuntimeError):
    """Admission failed and rollback could not restore ownership safely."""


class LocalCPUBackend(AllocatorBackendInterface):
    """
    Even if local_cpu is False (the hot_cache is not used), contains(),
    insert_key(), remove(), get_blocking(), get_keys(), and clear()
    are still callable by the storage manager.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.use_hot = config.local_cpu
        # NOTE: we keep the memory allocator argument for temporary
        # test compatibility
        # TODO: fix the tests to get rid the memory allocator
        assert metadata is not None or memory_allocator is not None
        self.memory_allocator = (
            self.initialize_allocator(config, metadata)  # type: ignore
            if memory_allocator is None
            else memory_allocator
        )
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cpu_lock = threading.Lock()

        # Cold-start diagnostics are armed only by a successful external
        # two-group commit while LMCACHE_COLD_START_PERF is enabled. Keeping an
        # empty list in ordinary operation makes every mutation hook a single
        # predictable branch, with no logging or cache inspection.
        self._external_retention_trace_sequence = 0
        self._external_retention_traces: list[_ExternalTwoGroupRetentionTrace] = []

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.layerwise = config.use_layerwise
        self.enable_blending = config.enable_blending
        self.layer_page_objects = bool(
            mooncake_layer_pages_enabled(config)
            and metadata is not None
            and metadata.use_mla
        )

        # Store config and metadata for chunk budget calculation
        self.config = config
        self.metadata = metadata

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),  # Backend location
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

        self._setup_metrics()

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.local_cpu_hot_cache_count.set_function(
                lambda: len(self.hot_cache)
            )
            prometheus_logger.local_cpu_keys_in_request_count.set_function(
                lambda: len(self.keys_in_request)
            )

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def batched_contains(
        self, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        hits: list[CacheEngineKey] = []
        with self.cpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    break
                hits.append(key)
            if pin:
                for cache_key in hits:
                    self.hot_cache[cache_key].pin()
                    self.keys_in_request.append(cache_key)
        return len(hits)

    def batched_contains_layer_pages(
        self, keys: Sequence[LayerCacheEngineKey], pin: bool = False
    ) -> int:
        """Check physical page objects using one representative layer key."""
        pages: list[tuple[CacheEngineKey, LayerPageMemoryObj]] = []
        with self.cpu_lock:
            for key in keys:
                page_key = key.without_layer()
                page = self.hot_cache.get(page_key)
                if not isinstance(page, LayerPageMemoryObj):
                    break
                pages.append((page_key, page))
            if pin:
                LayerPageMemoryObj.pin_many([page for _, page in pages])
                self.keys_in_request.extend(page_key for page_key, _ in pages)
        return len(pages)

    def batched_contains_two_group_prefix(
        self,
        group0_keys: Sequence[CacheEngineKey],
        group1_keys: Sequence[CacheEngineKey],
        *,
        pin: bool = False,
        diagnostics: Optional[dict[str, object]] = None,
    ) -> int:
        """Return the contiguous prefix present in both page keyspaces.

        Args:
            group0_keys: Canonical Group 0 physical-page keys.
            group1_keys: Canonical Group 1 physical-page keys aligned by
                logical page with ``group0_keys``.
            pin: Atomically pin both groups for every prefix page when true.
            diagnostics: Optional request-local mapping populated from the
                already-held cache lock. This never probes the allocator or
                emits a log from the mutation path.

        Returns:
            The number of complete two-group page pairs in the contiguous
            prefix. If the batch pin primitive declines the operation, returns
            zero without recording request keys.

        Raises:
            ValueError: If the two key sequences have different lengths.

        Notes:
            Lookup, the first-hole decision, pinning, and request-key tracking
            occur under one LocalCPU cache lock. A later complete pair cannot
            extend the result past an earlier hole.
        """
        self._validate_two_group_page_keys(
            group0_keys,
            group1_keys,
            context="Two-group lookup",
        )

        pages: list[LayerPageMemoryObj] = []
        page_keys: list[CacheEngineKey] = []
        with self.cpu_lock:
            for group0_key, group1_key in zip(group0_keys, group1_keys, strict=True):
                group0_page = self.hot_cache.get(group0_key)
                group1_page = self.hot_cache.get(group1_key)
                if not isinstance(group0_page, LayerPageMemoryObj) or not isinstance(
                    group1_page, LayerPageMemoryObj
                ):
                    break
                pages.extend((group0_page, group1_page))
                page_keys.extend((group0_key, group1_key))

            if pin and not LayerPageMemoryObj.pin_many(pages):
                return 0
            if pin:
                self.keys_in_request.extend(page_keys)
            if diagnostics is not None:
                diagnostics.update(
                    self._consume_external_retention_trace_locked(
                        group0_keys,
                        group1_keys,
                        local_pairs=len(pages) // 2,
                    )
                )

        return len(pages) // 2

    def batched_unpin(self, keys: Sequence[CacheEngineKey]) -> None:
        """Unpin each stored key occurrence."""
        with self.cpu_lock:
            for key in keys:
                if key not in self.hot_cache:
                    continue
                self.hot_cache[key].unpin()

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cpu_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        contains() and exists_in_put_tasks() should be checked together
        """
        return False

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[Future]:
        """
        Synchronously put the MemoryObj into the local cpu backend.

        :param on_complete_callback: Optional callback invoked after the
            synchronous put completes. Callback exceptions are caught and logged.
        """
        stored = False
        with self.cpu_lock:
            if key in self.hot_cache:
                return None

            memory_obj.ref_count_up()
            self.hot_cache[key] = memory_obj
            self._record_external_retention_mutation_locked(
                key,
                cause="ordinary_put",
                operation="admit",
                old_obj=None,
                new_obj=memory_obj,
            )

            self.cache_policy.update_on_put(key)

            # Push kv admit msg with batching
            if self.batched_msg_sender is not None:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT,
                    key=key.chunk_hash,
                )
            stored = True

        # Call callback after put completes (outside lock)
        if stored and on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

        return None

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Synchronously put the MemoryObjs into the local cpu backend.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's put completes (not once per batch).
        """
        if not self.use_hot:
            return

        stored_keys: list[CacheEngineKey] = []
        with self.cpu_lock:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                if key in self.hot_cache:
                    continue
                memory_obj.ref_count_up()
                self.hot_cache[key] = memory_obj
                self._record_external_retention_mutation_locked(
                    key,
                    cause="ordinary_batched_put",
                    operation="admit",
                    old_obj=None,
                    new_obj=memory_obj,
                )
                stored_keys.append(key)
            if stored_keys:
                self.cache_policy.update_on_put_many(stored_keys)
            for key in stored_keys:
                if self.batched_msg_sender is not None:
                    self.batched_msg_sender.add_kv_op(
                        op_type=OpType.ADMIT,
                        key=key.chunk_hash,
                    )

        if on_complete_callback is not None:
            for key in stored_keys:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(
                        f"on_complete_callback failed for key {key}: {e}"
                    )

    def batched_submit_layer_pages(
        self,
        keys: Sequence[CacheEngineKey],
        pages: List[LayerPageMemoryObj],
    ) -> None:
        """Install allocator-owned pages under their layer-independent keys."""
        if (
            len(keys) != len(pages)
            or len(set(keys)) != len(keys)
            or len({id(page) for page in pages}) != len(pages)
        ):
            raise ValueError("Layer-page admission requires one unique key per page")
        if any(not page.is_valid() for page in pages):
            raise ValueError("Layer-page admission cannot store invalid pages")
        if not self.use_hot:
            return
        stored_keys: list[CacheEngineKey] = []
        with self.cpu_lock:
            for key, page in zip(keys, pages, strict=True):
                existing = self.hot_cache.get(key)
                if self._compatible_layer_page(existing, page):
                    continue
                page.ref_count_up()
                self.hot_cache[key] = page
                self._record_external_retention_mutation_locked(
                    key,
                    cause=(
                        "layer_page_replace"
                        if existing is not None
                        else "layer_page_put"
                    ),
                    operation="replace" if existing is not None else "admit",
                    old_obj=existing,
                    new_obj=page,
                )
                stored_keys.append(key)
                if existing is not None:
                    # Other request owners retain their own references. Only
                    # the cache mapping's reference is transferred here.
                    self.cache_policy.update_on_force_evict(key)
                    existing.ref_count_down()
            for key in stored_keys:
                if self.batched_msg_sender is not None:
                    self.batched_msg_sender.add_kv_op(
                        op_type=OpType.ADMIT,
                        key=key.chunk_hash,
                    )
            if stored_keys:
                self.cache_policy.update_on_put_many(stored_keys)

    def batched_submit_layer_pages_if_absent(
        self,
        keys: Sequence[CacheEngineKey],
        pages: Sequence[LayerPageMemoryObj],
    ) -> LayerPageBatchPutResult:
        """Atomically admit immutable physical pages without overwriting keys.

        Args:
            keys: Distinct canonical, layer-independent physical-page keys.
            pages: Distinct, valid allocator-owned pages aligned with ``keys``.

        Returns:
            A result separating newly inserted keys from keys already present.
            Existing cache entries always win and are never compared or
            replaced. If LocalCPU storage is disabled, both result tuples are
            empty and page ownership is unchanged.

        Raises:
            ValueError: If counts differ, a key or page object is repeated, a
                key is layer-specific, or a page is not a valid
                ``LayerPageMemoryObj``.
            Exception: Re-raises an unexpected cache or policy exception after
                rolling back mappings and cache-owned page references. Raises
                ``RuntimeError`` instead if rollback itself cannot restore
                policy state or page ownership safely.

        Notes:
            Validation happens before the cache lock. Coverage validation,
            reference acquisition, insertion, and policy updates share one
            lock, so readers observe either the complete admitted batch or its
            complete predecessor. Controller notifications are best effort and
            are emitted only after the cache commit becomes visible.
        """
        self._validate_layer_page_batch(keys, pages)
        if not self.use_hot:
            return LayerPageBatchPutResult((), ())

        with self.cpu_lock:
            pending: list[tuple[CacheEngineKey, LayerPageMemoryObj]] = []
            existing_keys: list[CacheEngineKey] = []
            for key, page in zip(keys, pages, strict=True):
                if key in self.hot_cache:
                    existing_keys.append(key)
                else:
                    pending.append((key, page))
            result = self._admit_layer_pages_locked(pending, existing_keys)

        self._notify_layer_page_admissions(result.inserted_keys)
        return result

    def commit_external_two_group_prefix_if_absent(
        self,
        required_group0_keys: Sequence[CacheEngineKey],
        required_group1_keys: Sequence[CacheEngineKey],
        ready_reservations: Mapping[CacheEngineKey, LayerPageMemoryObj],
    ) -> ExternalTwoGroupCommitResult:
        """Atomically publish complete externally filled two-group coverage.

        Args:
            required_group0_keys: Canonical Group 0 keys in logical page order.
            required_group1_keys: Canonical Group 1 keys aligned by logical
                page with ``required_group0_keys``.
            ready_reservations: Terminal-success reservation pages keyed by
                canonical required key. Reservations may also be supplied for
                keys that won an insertion race.

        Returns:
            An immutable result describing whether the complete prefix was
            committed, newly inserted and pre-existing keys, redundant ready
            pages, and any missing required keys. A failed coverage check never
            inserts a page; all supplied ready pages are then redundant.

        Raises:
            ValueError: If groups are misaligned, keys are noncanonical,
                duplicated, assigned to the wrong group, reservation keys are
                outside the required prefix, or reservation pages are invalid
                or aliased.
            Exception: Re-raises an unexpected cache or policy exception after
                rolling back mappings and cache-owned references. Raises
                ``RuntimeError`` if rollback itself cannot safely restore
                state.

        Notes:
            Input validation occurs before the cache lock. Under one lock, the
            first pass proves every required key has an existing winner or a
            ready page; only then does the shared admission path insert all
            absent pages and update policy state. Notifications occur after
            the lock. The caller retains its reservation references for
            admitted pages and remains responsible for releasing them.
        """
        self._validate_two_group_page_keys(
            required_group0_keys,
            required_group1_keys,
            context="External two-group commit",
        )

        required_keys = [
            key
            for pair in zip(
                required_group0_keys, required_group1_keys, strict=True
            )
            for key in pair
        ]
        return self._commit_external_prefix_if_absent(
            required_keys,
            ready_reservations,
            context="External two-group commit",
            arm_retention_trace=cold_start_perf_enabled(),
        )

    def commit_external_group0_prefix_if_absent(
        self,
        required_group0_keys: Sequence[CacheEngineKey],
        ready_reservations: Mapping[CacheEngineKey, LayerPageMemoryObj],
    ) -> ExternalTwoGroupCommitResult:
        """Atomically publish an exact externally filled Group-0 prefix.

        This is the one-group counterpart of
        :meth:`commit_external_two_group_prefix_if_absent`.  It deliberately
        returns the existing immutable result type so lifecycle callers can
        share release and rollback handling.  Pair-specific retention tracing
        remains on the legacy wrapper.
        """

        self._validate_canonical_page_keys(required_group0_keys)
        if any(key.kv_group != 0 for key in required_group0_keys):
            raise ValueError("External Group-0 commit requires Group 0 keys")
        return self._commit_external_prefix_if_absent(
            required_group0_keys,
            ready_reservations,
            context="External Group-0 commit",
            arm_retention_trace=False,
        )

    def _commit_external_prefix_if_absent(
        self,
        required_keys: Sequence[CacheEngineKey],
        ready_reservations: Mapping[CacheEngineKey, LayerPageMemoryObj],
        *,
        context: str,
        arm_retention_trace: bool,
    ) -> ExternalTwoGroupCommitResult:
        """Atomically admit one prevalidated canonical page prefix."""

        ready = dict(ready_reservations)
        ready_keys = list(ready)
        ready_pages = list(ready.values())
        self._validate_layer_page_batch(ready_keys, ready_pages)
        required_key_set = set(required_keys)
        if any(key not in required_key_set for key in ready_keys):
            raise ValueError(
                f"{context} reservations must belong to the required prefix"
            )

        if not self.use_hot:
            return ExternalTwoGroupCommitResult(
                committed=False,
                inserted_keys=(),
                existing_keys=(),
                redundant_pages=tuple(ready_pages),
                missing_keys=tuple(required_keys),
            )

        lock_wait_started = time.perf_counter()
        with self.cpu_lock:
            lock_acquired = time.perf_counter()
            lock_wait_seconds = lock_acquired - lock_wait_started
            pending: list[tuple[CacheEngineKey, LayerPageMemoryObj]] = []
            existing_keys: list[CacheEngineKey] = []
            missing_keys: list[CacheEngineKey] = []
            redundant_pages: list[LayerPageMemoryObj] = []
            for key in required_keys:
                existing_page = self.hot_cache.get(key)
                if self._existing_layer_page_matches_key(key, existing_page):
                    existing_keys.append(key)
                    ready_page = ready.get(key)
                    if ready_page is not None:
                        redundant_pages.append(ready_page)
                    continue
                # Canonical keys are immutable. An invalid or wrong-kind
                # winner must not be overwritten by a remote fill, but it
                # also cannot satisfy exact two-group publication coverage.
                if key in self.hot_cache:
                    missing_keys.append(key)
                    continue
                ready_page = ready.get(key)
                if ready_page is None:
                    missing_keys.append(key)
                else:
                    pending.append((key, ready_page))

            if missing_keys:
                return ExternalTwoGroupCommitResult(
                    committed=False,
                    inserted_keys=(),
                    existing_keys=tuple(existing_keys),
                    redundant_pages=tuple(ready_pages),
                    missing_keys=tuple(missing_keys),
                    lock_wait_seconds=lock_wait_seconds,
                    lock_hold_seconds=time.perf_counter() - lock_acquired,
                )

            put_result = self._admit_layer_pages_locked(pending, existing_keys)
            retention_trace_id = self._arm_external_retention_trace_locked(
                required_keys,
                enabled=arm_retention_trace,
            )
            lock_hold_seconds = time.perf_counter() - lock_acquired

        self._notify_layer_page_admissions(put_result.inserted_keys)
        return ExternalTwoGroupCommitResult(
            committed=True,
            inserted_keys=put_result.inserted_keys,
            existing_keys=put_result.existing_keys,
            redundant_pages=tuple(redundant_pages),
            lock_wait_seconds=lock_wait_seconds,
            lock_hold_seconds=lock_hold_seconds,
            retention_trace_id=retention_trace_id,
        )

    def _existing_layer_page_matches_key(
        self,
        key: CacheEngineKey,
        page: Optional[MemoryObj],
    ) -> bool:
        """Cheap static validation for an immutable existing-key winner."""

        if not isinstance(page, LayerPageMemoryObj) or not page.is_valid():
            return False
        try:
            dtypes = tuple(page.meta.dtypes or ())
            expected_layers = (
                int(self.metadata.kv_shape[0])
                if self.metadata is not None
                else page.num_layers
            )
            expected_format = (
                MemoryFormat.KV_MLA_LATENT_FMT
                if key.kv_group == 0
                else MemoryFormat.KV_DSA_INDEX_FMT
            )
            return bool(
                page.num_layers == expected_layers
                and page.get_size() == page.layer_size * page.num_layers
                and len(dtypes) == page.num_layers
                and all(dtype == key.dtype for dtype in dtypes)
                and (
                    not bool(getattr(self.config, "dsa_two_groups", False))
                    or page.meta.fmt == expected_format
                )
                and page.valid_tokens
                == mooncake_valid_tokens(key, int(self.config.chunk_size))
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def batched_get_layer_page_prefix(
        self, keys: Sequence[CacheEngineKey]
    ) -> tuple[list[LayerPageMemoryObj], int]:
        """Retain and return the contiguous prefix of cached layer pages."""
        pages: list[LayerPageMemoryObj] = []
        with self.cpu_lock:
            for key in keys:
                page = self.hot_cache.get(key)
                if not isinstance(page, LayerPageMemoryObj):
                    break
                page.ref_count_up()
                pages.append(page)
        return pages, len(pages)

    def contains_any_exact(self, keys: Sequence[CacheEngineKey]) -> bool:
        """Return whether any key exists without resolving page aliases."""
        with self.cpu_lock:
            return any(key in self.hot_cache for key in keys)

    def contains_all_exact(self, keys: Iterable[CacheEngineKey]) -> bool:
        """Return whether every exact key exists under one cache lock."""
        with self.cpu_lock:
            return all(key in self.hot_cache for key in keys)

    def contains_compatible_layer_pages_exact(
        self,
        keys: Sequence[CacheEngineKey],
        expected_pages: Sequence[LayerPageMemoryObj],
    ) -> bool:
        """Validate exact-key winners before publishing a live page import."""
        if len(keys) != len(expected_pages):
            return False
        with self.cpu_lock:
            for key, expected in zip(keys, expected_pages, strict=True):
                actual = self.hot_cache.get(key)
                if not self._compatible_layer_page(actual, expected):
                    return False
        return True

    @staticmethod
    def _compatible_layer_page(
        actual: Optional[MemoryObj], expected: LayerPageMemoryObj
    ) -> bool:
        return bool(
            isinstance(actual, LayerPageMemoryObj)
            and actual.is_valid()
            and actual.get_size() == expected.get_size()
            and actual.valid_tokens == expected.valid_tokens
            and actual.num_layers == expected.num_layers
            and actual.layer_size == expected.layer_size
            and actual.meta.fmt == expected.meta.fmt
            and actual.meta.shape == expected.meta.shape
            and actual.meta.dtype == expected.meta.dtype
            and actual.meta.shapes == expected.meta.shapes
            and actual.meta.dtypes == expected.meta.dtypes
        )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return None
            memory_obj = self.hot_cache[key]
            # ref count up for caller to avoid situation where the memory_obj
            # is evicted from the local cpu backend before the caller calls
            # ref count up themselves
            memory_obj.ref_count_up()
            return memory_obj

    def batched_get_prefix_with_misses(
        self,
        keys: List[CacheEngineKey],
    ) -> LocalCPUPrefixGetResult:
        """Get the local prefix and report the remaining remote suffix."""
        local_memory_objs: List[Optional[MemoryObj]] = []
        with self.cpu_lock:
            for position, key in enumerate(keys):
                memory_obj = self.hot_cache.get(key)
                if memory_obj is None:
                    return LocalCPUPrefixGetResult(
                        local_memory_objs,
                        list(range(position, len(keys))),
                        list(keys[position:]),
                    )
                memory_obj.ref_count_up()
                local_memory_objs.append(memory_obj)
        return LocalCPUPrefixGetResult(local_memory_objs, [], [])

    def batched_get_prefixes_with_misses(
        self,
        keys_layer_major: List[List[CacheEngineKey]],
    ) -> List[LocalCPUPrefixGetResult]:
        """Get several layer prefixes while holding the cache lock once.

        Args:
            keys_layer_major: Cache keys grouped by model layer. Each layer is
                resolved only up to its first local miss, matching
                :meth:`batched_get_prefix_with_misses`.

        Returns:
            One prefix result per input layer, in the same order. Every local
            object in a returned result owns a caller reference that must be
            released by the consumer.

        Notes:
            Sparse cold bootstrap probes every layer before issuing its
            Mooncake request. Acquiring the same lock once avoids one Python
            call and one lock round trip per model layer without changing
            prefix or reference-count semantics.
        """
        results: List[LocalCPUPrefixGetResult] = []
        retained: List[MemoryObj] = []
        try:
            with self.cpu_lock:
                for keys in keys_layer_major:
                    local_memory_objs: List[Optional[MemoryObj]] = []
                    first_miss = len(keys)
                    for position, key in enumerate(keys):
                        memory_obj = self.hot_cache.get(key)
                        if memory_obj is None:
                            first_miss = position
                            break
                        memory_obj.ref_count_up()
                        retained.append(memory_obj)
                        local_memory_objs.append(memory_obj)
                    results.append(
                        LocalCPUPrefixGetResult(
                            local_memory_objs,
                            list(range(first_miss, len(keys))),
                            list(keys[first_miss:]),
                        )
                    )
            return results
        except Exception:
            for memory_obj in reversed(retained):
                if memory_obj.is_valid():
                    memory_obj.ref_count_down()
            raise

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs = []
        with self.cpu_lock:
            for key in keys:
                mem_obj = self.hot_cache[key]
                mem_obj.ref_count_up()
                mem_objs.append(mem_obj)
        return mem_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return self.batched_contains(keys, pin)

    def pin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.cpu_lock:
            if key not in self.hot_cache:
                return False
            memory_obj = self.hot_cache[key]
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.cpu_lock.acquire()
        if key not in self.hot_cache:
            if force:
                self.cpu_lock.release()
            return False

        memory_obj = self.hot_cache[key]
        self.hot_cache.pop(key)
        self._record_external_retention_mutation_locked(
            key,
            cause="explicit_remove" if force else "allocation_evict",
            operation="remove",
            old_obj=memory_obj,
            new_obj=None,
        )
        memory_obj.ref_count_down()

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.cpu_lock.release()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )
        # NOTE (Jiayi): This `return True` might not accurately reflect
        # whether the key is removed from the actual memory because
        # other backends might still (temporarily) hold the memory object.
        return True

    def _calculate_effective_cpu_size(
        self,
        configured_cpu_size: float,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> float:
        """
        Calculate the effective CPU memory size based on system available memory
        and reserve memory configuration.

        Args:
            configured_cpu_size: The configured CPU memory size in GB
            config: The LMCache engine configuration
            metadata: Optional metadata for first rank handling

        Returns:
            The effective CPU memory size in GB
        """

        save_only_first_rank = (
            metadata is not None
            and config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if not save_only_first_rank:
            # Do not adjust cpu_size if save_only_first_rank is False for now
            return configured_cpu_size

        # Get the system available memory and calculate effective cpu_size
        system_available_memory_gb = SystemMemoryDetector.get_available_memory_gb()
        # Get reserve memory size from config
        reserve_cpu_size = config.reserve_local_cpu_size

        # TODO(baoloongmao): For disable save_only_first_rank case,
        #  we need to avoid multi-rank race condition in future.
        #  But for enable save_only_first_rank case,
        #  we can handle reserve memory simply since non-first ranks
        #  do not allocate memory.
        # Effective memory: min(configured_size, available_memory - reserve_size)
        if system_available_memory_gb > 0:
            max_usable_memory = max(0, system_available_memory_gb - reserve_cpu_size)
            effective_cpu_size = min(configured_cpu_size, max_usable_memory)
            logger.info(
                f"Adjusted CPU memory size from {configured_cpu_size:.2f} GB "
                f"to {effective_cpu_size:.2f} GB "
                f"(system available: {system_available_memory_gb:.2f} GB, "
                f"reserve: {reserve_cpu_size:.2f} GB)"
            )
            assert effective_cpu_size > 0
            return effective_cpu_size
        else:
            logger.warning(
                "Could not determine system available memory, using configured cpu_size"
            )
            return configured_cpu_size

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> MemoryAllocatorInterface:
        cpu_size = config.max_local_cpu_size

        if metadata is not None:
            # save_only_first_rank only works when use mla
            save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )

            if save_only_first_rank and metadata.is_first_rank():
                # Only the first rank will save the cache,
                # so we need to set it larger than other ranks
                cpu_size = config.get_extra_config_value(
                    "first_rank_max_local_cpu_size", cpu_size
                )

        # Detect the numa mapping
        numa_mapping = NUMADetector.get_numa_mapping(config)
        logger.info(f"NUMA mapping {numa_mapping}")

        # Calculate effective CPU memory size
        cpu_size = self._calculate_effective_cpu_size(cpu_size, config, metadata)
        cpu_size_bytes = int(cpu_size * 1024**3)

        allocator_align_bytes = self._resolve_local_cpu_allocator_alignment(config)
        if allocator_align_bytes is not None:
            logger.info(
                "LocalCPUBackend: using pinned allocation alignment=%d bytes",
                allocator_align_bytes,
            )

        if config.enable_p2p:
            # TODO(baoloongmao): Add lazy memory allocator support for P2P mode
            # For now, keep the original P2P implementation
            assert metadata is not None
            shapes = metadata.get_shapes()
            dtypes = metadata.get_dtypes()

            paged_mem_allocator = PagedCpuGpuMemoryAllocator()
            chunk_size_bytes = get_size_bytes(shapes, dtypes)
            origin_cpu_size_bytes = cpu_size_bytes
            align_cpu_size_bytes = (
                origin_cpu_size_bytes // chunk_size_bytes * chunk_size_bytes
            )
            logger.info(
                f"Auto align cpu size bytes, origin: {origin_cpu_size_bytes}, "
                f"aligned: {align_cpu_size_bytes}, chunk size: {chunk_size_bytes}"
            )
            paged_mem_allocator.init_cpu_memory_allocator(
                align_cpu_size_bytes,
                shapes=shapes,
                dtypes=dtypes,
                fmt=MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
                numa_mapping=numa_mapping,
            )
            return paged_mem_allocator
        else:
            # Check if lazy memory allocator should be enabled
            use_lazy = (
                config.enable_lazy_memory_allocator
                and cpu_size > config.lazy_memory_safe_size
            )
            if use_lazy:
                logger.warning(
                    "LazyMixedMemoryAllocator is temporarily unavailable; "
                    "falling back to MixedMemoryAllocator with full allocation. "
                    "Disable enable_lazy_memory_allocator or reduce "
                    "max_local_cpu_size to avoid large pinned allocations."
                )
            elif config.enable_lazy_memory_allocator:
                logger.info(
                    f"LazyMixedMemoryAllocator is disabled because "
                    f"cpu_size ({cpu_size:.2f} GB) does not exceed "
                    f"lazy_memory_safe_size "
                    f"({config.lazy_memory_safe_size:.2f} GB). "
                    f"Using MixedMemoryAllocator instead."
                )
            shared_cpu_cache = config.get_extra_config_value(
                "enable_shared_cpu_cache",
                getattr(config, "enable_shared_cpu_cache", False),
            )
            shm_interleave_nodes = (
                NUMADetector.get_shared_cpu_interleave_nodes(config)
                if (
                    shared_cpu_cache
                    and config.get_extra_config_value("shm_name", None)
                )
                else None
            )
            if shm_interleave_nodes:
                logger.info(
                    "Shared CPU cache NUMA interleave nodes %s",
                    shm_interleave_nodes,
                )
            if allocator_align_bytes is not None:
                return MixedMemoryAllocator(
                    cpu_size_bytes,
                    numa_mapping=numa_mapping,
                    align_bytes=allocator_align_bytes,
                    config=config,
                    shm_interleave_nodes=shm_interleave_nodes,
                )
            return MixedMemoryAllocator(
                cpu_size_bytes,
                numa_mapping=numa_mapping,
                config=config,
                shm_interleave_nodes=shm_interleave_nodes,
            )

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def _resolve_local_cpu_allocator_alignment(
        self, config: LMCacheEngineConfig
    ) -> Optional[int]:
        """
        Determine pinned-memory alignment for LocalCPUBackend allocator.

        Precedence:
        1) explicit override: extra_config["local_cpu.pinned_align_bytes"]
        2) rust raw block auto mode:
           - rust_raw_block.device_path is set
           - rust_raw_block.use_odirect is true
           - rust_raw_block.align_local_cpu_allocator is true (default)
           -> use rust_raw_block.block_align
        3) None (use allocator default)
        """
        extra = config.extra_config or {}

        explicit_align = extra.get("local_cpu.pinned_align_bytes")
        if explicit_align is not None:
            align = int(explicit_align)
            if not self._is_power_of_two(align):
                raise ValueError(
                    "extra_config['local_cpu.pinned_align_bytes'] must be "
                    "a positive power of two"
                )
            return align

        rust_device_path = extra.get("rust_raw_block.device_path")
        rust_use_odirect = bool(extra.get("rust_raw_block.use_odirect", False))
        rust_auto_align = bool(
            extra.get("rust_raw_block.align_local_cpu_allocator", True)
        )

        if not rust_device_path or not rust_use_odirect or not rust_auto_align:
            return None

        rust_block_align = int(extra.get("rust_raw_block.block_align", 4096))
        if not self._is_power_of_two(rust_block_align):
            raise ValueError(
                "extra_config['rust_raw_block.block_align'] must be a positive "
                "power of two when O_DIRECT alignment is enabled"
            )
        return rust_block_align

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            f"Allocating memory in local cpu backend with busy loop: {busy_loop}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if memory_obj is not None or not eviction:
            return memory_obj

        evict_keys_count = 0
        num_attempts = 0
        while True:
            # whether or not this request needs to wait or other requests
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )
                    if evict_keys:
                        # we can continue trying to evict from the hot_cache
                        # and don't need to wait for other requests yet
                        wait_other_requests = False
                        logger.debug(
                            f"Evicting {len(evict_keys)} chunks from cpu memory"
                        )
                        # remove
                        self.batched_remove(evict_keys, force=False)
                        evict_keys_count += len(evict_keys)
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend allocate()"
            )

        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: Optional[MemoryFormat] = None,
        eviction: bool = True,
        busy_loop: bool = True,
        address_backed: bool = False,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        evict if necessary. Storage manager should always call
        local_cpu_backend.allocate() to get memory objects
        regardless of whether local_cpu is True or False

        ``address_backed`` requests lazy tensor-view construction when the
        configured allocator supports it; otherwise allocation remains eager.

        busy_loop should only be used for retrieve
        the reasoning is that:

        1. synchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - one retrieve at a time (okay to busy loop because stores will clear)

        2. asynchronous case
        - many stores happen concurrently (if they busy_loop, deadlock happens)
        - many retrieves happen concurrently
        (we use the async serializer to handle this)
        """
        logger.debug(
            f"Batched allocating memory in local cpu backend"
            f" with busy loop: {busy_loop}"
        )
        if fmt is None:
            if self.layerwise:
                if self.enable_blending:
                    fmt = MemoryFormat.KV_2TD
                else:
                    fmt = MemoryFormat.KV_T2D
            else:
                fmt = MemoryFormat.KV_2LTD

        allocate_batch = self.memory_allocator.batched_allocate
        if address_backed:
            allocate_batch = getattr(
                self.memory_allocator,
                "batched_allocate_address_backed",
                allocate_batch,
            )
        memory_objs = allocate_batch(shapes, dtypes, batch_size, fmt)

        if memory_objs is not None or not eviction:
            return memory_objs

        assert isinstance(self.memory_allocator, MixedMemoryAllocator)

        evict_keys_count = 0
        num_attempts = 0
        while True:
            wait_other_requests = True
            if self.use_hot:
                # TODO(Jiayi): optimize `num_candidates` with estimation.
                # Accurate estimation is hard due to fragmentation
                num_candidates = 1
                evict_keys = None
                with self.cpu_lock:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.hot_cache, num_candidates=num_candidates
                    )

                    # HACK: We assume batch_size=num_layers here.
                    # FIXME: We also assume if the one layer's ref_count > 1 or pinned,
                    # then the other layers are also ref_count > 1 or
                    # pinned in the cpu memory. This might not be true.
                    if evict_keys:
                        evict_keys_count += len(evict_keys)
                        wait_other_requests = False
                        for evict_key in evict_keys:
                            evict_key_all_layer = (
                                [evict_key]
                                if isinstance(
                                    self.hot_cache.get(evict_key),
                                    LayerPageMemoryObj,
                                )
                                else evict_key.split_layers(batch_size)
                            )

                            # TODO(Jiayi): batched allocate is not supported through
                            # `batched_remove`. Therefore, features like usage tracking
                            # is not supported.
                            old_mem_objs = []
                            for key in evict_key_all_layer:
                                old_mem_obj = self.hot_cache[key]
                                self.cache_policy.update_on_force_evict(key)
                                self.hot_cache.pop(key, None)
                                self._record_external_retention_mutation_locked(
                                    key,
                                    cause="batched_allocate_evict",
                                    operation="remove",
                                    old_obj=old_mem_obj,
                                    new_obj=None,
                                )
                                old_mem_objs.append(old_mem_obj)

                            for old_mem_obj in old_mem_objs:
                                old_mem_obj.ref_count_down()

                            logger.debug(
                                f"Evicting {len(old_mem_objs)} chunks from cpu memory"
                            )
                    else:
                        self.stats_monitor.update_local_cpu_evict_failed_count(
                            num_candidates
                        )

            if wait_other_requests:
                if not busy_loop:
                    logger.debug(
                        "Not busy looping because we are not immediately able to evict"
                    )
                    break

                # TODO: make time_to_wait a config
                time_to_wait = 0.1
                logger.warning(
                    "No eviction candidates found in local cpu backend. "
                    "Local cpu memory is under pressure. "
                    f"Waiting for {time_to_wait} seconds before retrying."
                )
                # self.memory_allocator.memcheck()
                # do not hold the lock during sleep
                time.sleep(time_to_wait)

            memory_objs = allocate_batch(shapes, dtypes, batch_size, fmt)
            if memory_objs:
                break

            num_attempts += 1
            logger.debug(
                f"Unable to allocate memory object after {num_attempts}"
                " attempts of local cpu backend batched_allocate()"
            )
        self.stats_monitor.update_local_cpu_evict_metrics(evict_keys_count)
        return memory_objs

    def batched_allocate_layer_pages(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        batch_size: int,
        num_layers: int,
        fmt: MemoryFormat,
        busy_loop: bool = True,
        valid_tokens: Optional[Union[int, list[int]]] = None,
        full_tokens: Optional[int] = None,
        eviction: bool = True,
    ) -> Optional[list[LayerPageMemoryObj]]:
        """Allocate exact-size layer pages, evicting entries when required."""
        allocate = getattr(
            self.memory_allocator, "batched_allocate_layer_pages", None
        )
        if not callable(allocate):
            return None
        allocation_args = (shapes, dtypes, batch_size, num_layers, fmt)
        pages = allocate(
            *allocation_args,
            valid_tokens,
            full_tokens,
        )
        if pages is None and not eviction:
            return None
        evicted = 0
        deadline = time.monotonic() + float(
            getattr(self.config, "blocking_timeout_secs", 60)
        )
        while pages is None and self.use_hot:
            with self.cpu_lock:
                keys, objects = self._pop_layer_page_evict_candidate_locked(
                    num_layers,
                    cause="layer_page_allocate_evict",
                )
            for memory_obj in objects:
                memory_obj.ref_count_down()
            evicted += len(keys)
            if not objects:
                self.stats_monitor.update_local_cpu_evict_failed_count(1)
                if not busy_loop or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            pages = allocate(
                *allocation_args,
                valid_tokens,
                full_tokens,
            )
        self.stats_monitor.update_local_cpu_evict_metrics(evicted)
        return pages

    def get_full_chunk_size_bytes(self) -> int:
        logger.info("Calculating the size of a single LMCache chunk")
        assert self.metadata is not None, (
            "metadata required for chunk budget calculation"
        )

        chunk_tokens = self.config.chunk_size
        # already accounted for parallelism
        kv_shape = (
            self.metadata.kv_shape
        )  # [num_layers, kv_size, chunk_size, num_heads, head_size]
        num_layers = kv_shape[0]
        kv_size = kv_shape[1]  # 1 for MLA, 2 for regular
        # per gpu
        num_heads = kv_shape[3]
        head_size = kv_shape[4]
        hidden_dim = num_heads * head_size
        dtype_size = self.metadata.kv_dtype.itemsize

        if self.layerwise:
            # layerwise: [chunk_tokens, kv_size, hidden_dim]
            chunk_bytes = chunk_tokens * kv_size * hidden_dim * dtype_size
        else:
            # full: [kv_size, num_layers, chunk_tokens, hidden_dim]
            chunk_bytes = kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size
        logger.debug(
            f"Stats received: num_layers={num_layers}, kv_size={kv_size}, "
            f"chunk_tokens={chunk_tokens}, head_dim={head_size}, "
            f"dtype_size={dtype_size}, "
            f"hidden_dim={hidden_dim}"
        )
        logger.debug(f"Calculated bytes per chunk per rank: {chunk_bytes}")
        return chunk_bytes

    def calculate_chunk_budget(self) -> int:
        """
        Calculate the maximum number of chunks that can be allocated concurrently
        without causing memory deadlocks in the async loading system.

        Returns:
            int: The estimated chunk budget for concurrent allocations
        """
        total_memory = int(self.config.max_local_cpu_size * 1024**3)
        chunk_bytes = self.get_full_chunk_size_bytes()
        # add alignment overhead
        # (MixedMemoryAllocator uses TensorMemoryAllocator with 4KB alignment)
        assert hasattr(self.memory_allocator, "align_bytes")
        alignment = self.memory_allocator.align_bytes
        aligned_chunk_bytes = ((chunk_bytes + alignment - 1) // alignment) * alignment

        # calculate budget with safety margin
        max_chunks = total_memory // aligned_chunk_bytes

        return max_chunks

    def get_keys(self) -> List[CacheEngineKey]:
        """
        array ordering of keys from LRU to MRU
        """
        with self.cpu_lock:
            return list(self.hot_cache.keys())

    def clear(self) -> int:
        """
        counts the number of memory objects removed
        """
        if not self.use_hot:
            return 0
        clear_keys = []
        num_cleared_tokens = 0
        with self.cpu_lock:
            for key in self.hot_cache:
                memory_obj = self.hot_cache[key]
                if not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()

        # TODO(Jiayi): might not be accurate if we don't calculate
        # `num_cleared_token` and remove the keys in an atomic way.
        self.batched_remove(clear_keys)

        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def get_allocator_capacity_bytes(self) -> tuple[int, int]:
        """Return free and total allocator bytes without scanning cache keys.

        Returns:
            A constant-time ``(free_bytes, heap_bytes)`` snapshot synchronized
            with allocator allocation and free operations.

        Raises:
            NotImplementedError: If the configured allocator cannot provide a
                synchronized constant-time snapshot.
            RuntimeError: If an allocator violates the capacity contract.
        """

        free_bytes, heap_bytes = self.memory_allocator.get_capacity_bytes()
        if (
            not isinstance(free_bytes, int)
            or not isinstance(heap_bytes, int)
            or free_bytes < 0
            or heap_bytes <= 0
            or free_bytes > heap_bytes
        ):
            raise RuntimeError("allocator returned invalid capacity bytes")
        return free_bytes, heap_bytes

    def reclaim_evictable_capacity(
        self,
        required_bytes: int,
        *,
        min_free_bytes: int,
        min_free_ratio: float,
        num_layers: int,
        cause: str,
    ) -> bool:
        """Evict enough eligible entries to accommodate one allocation.

        The method scans once and never waits or allocates. If the pressure
        snapshot proves all eligible entries insufficient, it changes nothing.

        Args:
            required_bytes: Incoming allocation size to accommodate.
            min_free_bytes: Absolute free-capacity floor after allocation.
            min_free_ratio: Heap-relative free-capacity floor after allocation.
            num_layers: Layer count used to expand legacy layerwise keys.
            cause: Retention-trace cause recorded for removed entries.

        Returns:
            Whether the allocator reached the required free-capacity target.

        Raises:
            ValueError: If an argument is invalid.
        """

        if (
            required_bytes < 0
            or min_free_bytes < 0
            or not 0 <= min_free_ratio <= 1
            or num_layers <= 0
            or not cause
        ):
            raise ValueError("invalid LocalCPU capacity-reclaim request")
        started = time.perf_counter() if cold_start_perf_enabled() else None
        free_before, heap_bytes = self.get_allocator_capacity_bytes()
        target_free_bytes = required_bytes + max(
            min_free_bytes,
            int(heap_bytes * min_free_ratio),
        )
        evictable_bytes = 0
        evicted_bytes = 0
        evicted_keys = 0
        removed: list[MemoryObj] = []
        if free_before < target_free_bytes:
            with self.cpu_lock:
                if self.use_hot:
                    evictable_bytes = sum(
                        memory_obj.get_physical_size()
                        for memory_obj in self.hot_cache.values()
                        if memory_obj.can_evict
                    )
                if free_before + evictable_bytes >= target_free_bytes:
                    while free_before + evicted_bytes < target_free_bytes:
                        keys, objects = self._pop_layer_page_evict_candidate_locked(
                            num_layers,
                            cause=cause,
                        )
                        if not objects:
                            break
                        evicted_keys += len(keys)
                        evicted_bytes += sum(
                            memory_obj.get_physical_size() for memory_obj in objects
                        )
                        removed.extend(objects)

        for memory_obj in removed:
            memory_obj.ref_count_down()
        free_after = (
            self.get_allocator_capacity_bytes()[0] if removed else free_before
        )
        sufficient = free_after >= target_free_bytes
        self.stats_monitor.update_local_cpu_evict_metrics(evicted_keys)
        if not sufficient:
            self.stats_monitor.update_local_cpu_evict_failed_count(1)
        if started is not None:
            cold_start_perf_log(
                logger,
                cause,
                started=started,
                target_free_bytes=target_free_bytes,
                free_before=free_before,
                free_after=free_after,
                evictable_bytes=evictable_bytes,
                evicted_bytes=evicted_bytes,
                evicted_keys=evicted_keys,
                outcome=(
                    "reclaimed"
                    if sufficient and evicted_keys
                    else "not_needed"
                    if sufficient
                    else "raced"
                    if evicted_keys
                    else "insufficient"
                ),
            )
        return sufficient

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.memory_allocator.close()
        self.clear()

    @staticmethod
    def _validate_canonical_page_keys(
        keys: Sequence[CacheEngineKey],
    ) -> None:
        if any(type(key) is not CacheEngineKey for key in keys):
            raise ValueError(
                "Layer-page admission requires canonical layer-independent keys"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("Layer-page admission requires unique keys")

    @classmethod
    def _validate_two_group_page_keys(
        cls,
        group0_keys: Sequence[CacheEngineKey],
        group1_keys: Sequence[CacheEngineKey],
        *,
        context: str,
    ) -> None:
        """Validate aligned canonical page identities for the two DSA groups."""
        if len(group0_keys) != len(group1_keys):
            raise ValueError(f"{context} requires aligned key counts")
        cls._validate_canonical_page_keys([*group0_keys, *group1_keys])
        if any(key.kv_group != 0 for key in group0_keys) or any(
            key.kv_group != 1 for key in group1_keys
        ):
            raise ValueError(f"{context} requires Group 0/1 keys")

        def shared_tags(key: CacheEngineKey) -> tuple:
            return tuple(tag for tag in (key.tags or ()) if tag[0] != "dsa_idx")

        for group0_key, group1_key in zip(group0_keys, group1_keys, strict=True):
            group0_identity = (
                group0_key.model_name,
                group0_key.world_size,
                group0_key.worker_id,
                group0_key.chunk_hash,
                shared_tags(group0_key),
            )
            group1_identity = (
                group1_key.model_name,
                group1_key.world_size,
                group1_key.worker_id,
                group1_key.chunk_hash,
                shared_tags(group1_key),
            )
            if group0_identity != group1_identity:
                raise ValueError(f"{context} requires identical logical page pairs")

    @classmethod
    def _validate_layer_page_batch(
        cls,
        keys: Sequence[CacheEngineKey],
        pages: Sequence[LayerPageMemoryObj],
    ) -> None:
        if len(keys) != len(pages):
            raise ValueError("Layer-page admission requires aligned key counts")
        cls._validate_canonical_page_keys(keys)
        if any(not isinstance(page, LayerPageMemoryObj) for page in pages):
            raise ValueError("Layer-page admission requires layer-page objects")
        if len({id(page) for page in pages}) != len(pages):
            raise ValueError("Layer-page admission requires unique pages")
        if any(not page.is_valid() for page in pages):
            raise ValueError("Layer-page admission cannot store invalid pages")

    def _admit_layer_pages_locked(
        self,
        pending: Sequence[tuple[CacheEngineKey, LayerPageMemoryObj]],
        existing_keys: Sequence[CacheEngineKey],
    ) -> LayerPageBatchPutResult:
        inserted_keys: list[CacheEngineKey] = []
        inserted_pages: list[LayerPageMemoryObj] = []
        try:
            for key, page in pending:
                page.ref_count_up()
                inserted_pages.append(page)
                inserted_keys.append(key)
                self.hot_cache[key] = page
                self._record_external_retention_mutation_locked(
                    key,
                    cause="layer_page_admit_if_absent",
                    operation="admit",
                    old_obj=None,
                    new_obj=page,
                )
            if inserted_keys:
                self.cache_policy.update_on_put_many(inserted_keys)
        except BaseException as admission_error:
            rollback_errors: list[BaseException] = []
            for key, page in reversed(
                list(zip(inserted_keys, inserted_pages, strict=True))
            ):
                try:
                    if self.hot_cache.get(key) is page:
                        self.hot_cache.pop(key)
                        self._record_external_retention_mutation_locked(
                            key,
                            cause="layer_page_admission_rollback",
                            operation="remove",
                            old_obj=page,
                            new_obj=None,
                        )
                except BaseException as error:
                    rollback_errors.append(error)
                try:
                    self.cache_policy.update_on_force_evict(key)
                except BaseException as error:
                    rollback_errors.append(error)
                try:
                    page.ref_count_down()
                except BaseException as error:
                    rollback_errors.append(error)
            if rollback_errors:
                raise LayerPageAdmissionRollbackError(
                    "Layer-page admission rollback could not restore state"
                ) from admission_error
            raise

        return LayerPageBatchPutResult(
            tuple(inserted_keys), tuple(existing_keys)
        )

    def _arm_external_retention_trace_locked(
        self,
        required_keys: Sequence[CacheEngineKey],
        *,
        enabled: bool,
    ) -> Optional[int]:
        """Arm bounded host-only diagnostics while the commit lock is held."""

        if not enabled:
            return None
        self._external_retention_trace_sequence += 1
        trace_id = self._external_retention_trace_sequence
        required = tuple(required_keys)
        positions = {
            key: (pair_index, key.kv_group)
            for pair_index, pair in enumerate(
                zip(required[::2], required[1::2], strict=True)
            )
            for key in pair
        }
        trace = _ExternalTwoGroupRetentionTrace(
            trace_id=trace_id,
            started_ns=time.monotonic_ns(),
            required_keys=required,
            positions=positions,
            committed_pages={key: ref(self.hot_cache[key]) for key in required},
        )
        if len(self._external_retention_traces) >= _MAX_EXTERNAL_RETENTION_TRACES:
            self._external_retention_traces.pop(0)
        self._external_retention_traces.append(trace)
        return trace_id

    def _record_external_retention_mutation_locked(
        self,
        key: CacheEngineKey,
        *,
        cause: str,
        operation: str,
        old_obj: Optional[MemoryObj],
        new_obj: Optional[MemoryObj],
    ) -> None:
        """Record a watched mapping mutation without I/O or extra locking."""

        if not self._external_retention_traces:
            return
        watched = [
            trace for trace in self._external_retention_traces if key in trace.positions
        ]
        if not watched:
            return
        now_ns = time.monotonic_ns()
        callsite = self._external_retention_callsite()
        for trace in watched:
            position = trace.positions[key]
            if len(trace.mutations) >= _MAX_EXTERNAL_RETENTION_MUTATIONS:
                trace.dropped_mutations += 1
                continue
            pair_index, kv_group = position
            trace.mutations.append(
                _ExternalRetentionMutation(
                    monotonic_ns=now_ns,
                    cause=cause,
                    operation=operation,
                    pair_index=pair_index,
                    kv_group=kv_group,
                    old_type=type(old_obj).__name__
                    if old_obj is not None
                    else "absent",
                    new_type=type(new_obj).__name__
                    if new_obj is not None
                    else "absent",
                    removed_committed_page=(
                        old_obj is not None
                        and (committed_page := trace.committed_pages.get(key))
                        is not None
                        and committed_page() is old_obj
                        and operation in {"remove", "replace"}
                    ),
                    callsite=callsite,
                )
            )

    @staticmethod
    def _external_retention_callsite() -> str:
        """Return a short host-Python call chain for a watched explicit remove."""

        names: list[str] = []
        try:
            frame = sys._getframe(3)
        except ValueError:
            return ""
        for _ in range(3):
            if frame is None:
                break
            names.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
            frame = frame.f_back
        return "<-".join(names)

    def _consume_external_retention_trace_locked(
        self,
        group0_keys: Sequence[CacheEngineKey],
        group1_keys: Sequence[CacheEngineKey],
        *,
        local_pairs: int,
    ) -> dict[str, object]:
        """Summarize one trace at the exact lookup observation point."""

        required = tuple(
            key for pair in zip(group0_keys, group1_keys, strict=True) for key in pair
        )
        candidates = [
            trace
            for trace in self._external_retention_traces
            if trace.required_keys == required
        ]
        trace = candidates[0] if candidates else None
        if trace is not None:
            self._external_retention_traces.remove(trace)

        fields: dict[str, object] = {
            "retention_trace_status": (
                "matched" if trace is not None else "no_exact_key_match"
            ),
            "retention_trace_candidates": len(candidates),
        }
        first_hole = local_pairs if local_pairs < len(group0_keys) else None
        if first_hole is None:
            fields["local_first_hole_pair"] = None
        else:
            group0_key = group0_keys[first_hole]
            group1_key = group1_keys[first_hole]
            fields.update(
                local_first_hole_pair=first_hole,
                local_first_hole_group0_state=self._external_retention_page_state(
                    group0_key, trace
                ),
                local_first_hole_group1_state=self._external_retention_page_state(
                    group1_key, trace
                ),
            )

        if trace is None:
            if first_hole is not None:
                fields["retention_attribution_status"] = "trace_not_matched"
            else:
                fields["retention_attribution_status"] = "not_needed"
            return fields

        now_ns = time.monotonic_ns()
        cause_counts: dict[str, int] = {}
        for mutation in trace.mutations:
            cause_counts[mutation.cause] = cause_counts.get(mutation.cause, 0) + 1
        fields.update(
            retention_trace_id=trace.trace_id,
            retention_trace_age_ms=round((now_ns - trace.started_ns) / 1_000_000, 3),
            retention_mutation_count=len(trace.mutations),
            retention_mutation_dropped=trace.dropped_mutations,
            retention_mutation_causes=cause_counts,
        )

        if first_hole is None:
            fields["retention_attribution_status"] = "not_needed"
            return fields
        if trace.dropped_mutations:
            fields["retention_attribution_status"] = "journal_overflow"
            return fields

        bad_keys = [
            key
            for key in (group0_keys[first_hole], group1_keys[first_hole])
            if not isinstance(self.hot_cache.get(key), LayerPageMemoryObj)
        ]
        last_by_key = {
            key: next(
                (
                    mutation
                    for mutation in reversed(trace.mutations)
                    if (mutation.pair_index, mutation.kv_group) == trace.positions[key]
                ),
                None,
            )
            for key in bad_keys
        }
        roots = [mutation for mutation in last_by_key.values() if mutation is not None]
        if not roots:
            fields["retention_attribution_status"] = "no_matching_mutation"
            return fields
        root = min(roots, key=lambda mutation: mutation.monotonic_ns)
        fields.update(
            retention_attribution_status="attributed",
            retention_attributed_cause=root.cause,
            retention_attributed_operation=root.operation,
            retention_attributed_pair=root.pair_index,
            retention_attributed_group=root.kv_group,
            retention_attributed_elapsed_ms=round(
                (root.monotonic_ns - trace.started_ns) / 1_000_000,
                3,
            ),
            retention_attributed_old_type=root.old_type,
            retention_attributed_new_type=root.new_type,
            retention_attributed_removed_committed_page=root.removed_committed_page,
        )
        if root.callsite:
            fields["retention_attributed_callsite"] = root.callsite
        return fields

    def _external_retention_page_state(
        self,
        key: CacheEngineKey,
        trace: Optional[_ExternalTwoGroupRetentionTrace],
    ) -> str:
        """Describe one watched mapping without allocating or probing storage."""

        page = self.hot_cache.get(key)
        if page is None:
            return "absent"
        if not isinstance(page, LayerPageMemoryObj):
            return f"wrong_type:{type(page).__name__}"
        if trace is not None:
            committed_page = trace.committed_pages.get(key)
            if committed_page is not None and committed_page() is page:
                return "committed_page"
        return "replacement_page"

    def _notify_layer_page_admissions(
        self, inserted_keys: Sequence[CacheEngineKey]
    ) -> None:
        if self.batched_msg_sender is None:
            return
        for key in inserted_keys:
            try:
                self.batched_msg_sender.add_kv_op(
                    op_type=OpType.ADMIT,
                    key=key.chunk_hash,
                )
            except Exception:
                logger.exception(
                    "Failed to notify controller of admitted layer page %s",
                    key,
                )

    def _pop_layer_page_evict_candidate_locked(
        self,
        num_layers: int,
        *,
        cause: str,
    ) -> tuple[list[CacheEngineKey], list[MemoryObj]]:
        candidates = self.cache_policy.get_evict_candidates(
            self.hot_cache, num_candidates=1
        )
        if not candidates:
            return [], []
        key = candidates[0]
        keys = (
            [key]
            if isinstance(self.hot_cache.get(key), LayerPageMemoryObj)
            or not isinstance(key, LayerCacheEngineKey)
            else [
                item
                for item in key.split_layers(num_layers)
                if item in self.hot_cache
            ]
        )
        objects = []
        for item in keys:
            memory_obj = self.hot_cache.pop(item)
            self._record_external_retention_mutation_locked(
                item,
                cause=cause,
                operation="remove",
                old_obj=memory_obj,
                new_obj=None,
            )
            objects.append(memory_obj)
        for item in keys:
            self.cache_policy.update_on_force_evict(item)
        return keys, objects
