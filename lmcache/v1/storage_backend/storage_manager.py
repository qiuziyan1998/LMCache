# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
import asyncio
import functools
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.utils import (
    CacheEngineKey,
    LayerCacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_valid_tokens
from lmcache.v1.storage_backend import CreateStorageBackends, is_cuda_worker
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

logger = init_logger(__name__)


# Helper function to get the class name of the backend
def get_backend_cname(backend: StorageBackendInterface) -> str:
    return backend.__class__.__name__


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: torch.cuda.Stream,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """
    Allocate the memory objects and copy the data from src_memory_objs to
    the newly allocated memory objects

    Args:
        allocator_backend: the allocator backend to allocate the new memory
          objects
        keys: the cache engine keys corresponding to the memory objects
        src_memory_objs: the memory objects to copy from
        stream: the cuda stream to run the copy in

    Returns:
        - list of cache engine keys that corresponds to the memory objects
          that has been successfully allocated
        - list of the memory objects that has been successfully allocated
    """
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            src_memory_obj.get_shape(),
            src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None:
            break

        if memory_obj.tensor is None:
            # This should not happen with current implementation,
            # but handle it defensively to avoid memory leak
            logger.warning(
                "Allocated MemoryObj has None tensor, this is unexpected. "
                "Releasing the memory object."
            )
            memory_obj.ref_count_down()
            break

        with torch.cuda.stream(stream):
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_objects.append(memory_obj)

    if stream is not None:
        stream.synchronize()
    return keys[: len(allocated_objects)], allocated_objects


class WeightedSemaphore:
    def __init__(self, chunk_budget: int):
        # it is physically impossible to have more fragmentation than 50%
        # when all of the chunks are of the same size (save_unfull_chunk=False)
        # so we can safely allocate half of the chunk budget for concurrent requests
        self._concurrent_budget_cap = chunk_budget // 2
        self._chunk_budget_cap = chunk_budget
        self._current_chunks = self._concurrent_budget_cap
        self._cond = asyncio.Condition()

    async def acquire(self, n: int = 1) -> None:
        if n > self._chunk_budget_cap:
            raise ValueError(
                f"Trying to acquire {n} chunks, "
                f"Cannot acquire more than {self._chunk_budget_cap} chunks"
                "Please set the max local cpu size to a larger value"
            )

        async with self._cond:
            logger.debug(f"WeightedSemaphore: Attempting to acquire {n} chunks")
            if n <= self._concurrent_budget_cap:
                await self._cond.wait_for(lambda: self._current_chunks >= n)
                self._current_chunks -= n
            else:
                # Oversized case: require exclusive access
                await self._cond.wait_for(
                    lambda: self._current_chunks == self._concurrent_budget_cap
                )
                # Reserve everything
                self._current_chunks = 0
            logger.debug(
                f"WeightedSemaphore: Acquired {n} chunks, "
                f"remaining chunks: {self._current_chunks}"
            )

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            if n <= self._concurrent_budget_cap:
                self._current_chunks += n
            else:
                self._current_chunks = self._concurrent_budget_cap
            self._cond.notify_all()


class AsyncMultiSerializer:
    """
    Prevent race conditions where multiple batched_get's cause the local CPU
    backend to allocate memory objects in parallel and get deadlocked.
    Make the assumption that the save_unfull_chunk is False so that we
    can assume that we can always use 50% of the given memory
    """

    def __init__(
        self,
        allocator_backend: AllocatorBackendInterface,
        loop: asyncio.AbstractEventLoop,
    ):
        self.chunk_budget = allocator_backend.calculate_chunk_budget()
        self._sem = WeightedSemaphore(self.chunk_budget)
        self.loop = loop

    async def run(
        self,
        coro_fn: Coroutine[Any, Any, Any],
        num_chunks: int,
    ) -> Any:
        await self._sem.acquire(num_chunks)
        try:
            return await coro_fn
        finally:
            await self._sem.release(num_chunks)


class AsyncSingleSerializer:
    """
    Prevent race conditions in a naive way by forcing each request that
    is passed through to be serialized
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        # lazy init in run
        self.lock: Optional[asyncio.Lock] = None

    async def run(self, coro_fn: Coroutine[Any, Any, Any], *args, **kwargs) -> Any:
        # we need to lazily initialize the lock to
        # place it on the calling event loop
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:  # type: ignore
            return await coro_fn


AsyncSerializer = Union[AsyncSingleSerializer, AsyncMultiSerializer]


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None,
    ):
        self.config = config
        self.metadata = metadata
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(self.loop,),
            name="storage-manger-event-loop",
        )
        self.thread.start()

        self.storage_backends: OrderedDict[str, StorageBackendInterface] = OrderedDict()
        self.manager_lock = threading.Lock()
        self.lmcache_worker = lmcache_worker

        # Use the unified create path so that init and
        # dynamic creation share the same logic.
        self.create_backends()

        # the backend used for actual storage
        self.non_allocator_backends = self.get_non_allocator_backends()

        self.enable_pd = config.enable_pd

        self.allocator_backend = None
        if metadata.role != "scheduler":
            self.allocator_backend = self._get_allocator_backend(config)

        self.local_cpu_backend = self.storage_backends.get("LocalCPUBackend", None)

        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.event_manager = event_manager

        self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = (
            async_lookup_server
        )
        self.async_serializer: Optional[AsyncSerializer] = None

        # The cuda stream for internal copies during put
        if is_cuda_worker(metadata):
            self.internal_copy_stream = torch.cuda.Stream()
        else:
            self.internal_copy_stream = None

        # freeze mode: only use local_cpu backend for retrieval
        self._freeze = False
        self._freeze_lock = threading.RLock()

        # Backend bypass mode: skip specific backends during health check failures
        self._bypassed_backends: set[str] = set()
        self._bypass_lock = threading.RLock()

        if not self.enable_pd and self.config.enable_async_loading:
            assert self.allocator_backend is not None
            self.async_serializer = AsyncSingleSerializer(self.loop)

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "event metrics will not be collected"
            )
            return

        metric_map = {
            "storage_events_ongoing_count": EventStatus.ONGOING,
            "storage_events_done_count": EventStatus.DONE,
            "storage_events_not_found_count": EventStatus.NOT_FOUND,
        }

        for metric_name, status in metric_map.items():
            metric = getattr(prometheus_logger, metric_name)
            metric.set_function(
                lambda s=status: self.event_manager.get_events_count_by_status(
                    EventType.LOADING, s
                )
            )

    def _get_allocator_backend(
        self, config: LMCacheEngineConfig
    ) -> AllocatorBackendInterface:
        if self.enable_pd:
            allocator_backend = self.storage_backends["PDBackend"]
        elif "MaruBackend" in self.storage_backends:
            if "LocalCPUBackend" in self.storage_backends:
                allocator_backend = self.storage_backends["LocalCPUBackend"]
            else:
                allocator_backend = self.storage_backends["MaruBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        assert self.allocator_backend is not None
        return self.allocator_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        if self.allocator_backend is None:
            raise RuntimeError("Allocator backend not available for scheduler role")
        return self.allocator_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        raise RuntimeError(
            "StorageManager.put is deprecated and should not be called anymore"
        )

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> list[Future]:
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        Returns futures for backends that require persistence completion.
        """
        # The dictionary from backend cname to objects and keys
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        if self.allocator_backend is None:
            # For scheduler role, no allocator backend available
            raise RuntimeError("Batched put not available for scheduler role")
        obj_dict[get_backend_cname(self.allocator_backend)] = (
            keys,
            memory_objs,
        )

        required_futures: list[Future] = []
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            # Skip bypassed backends
            with self._bypass_lock:
                if backend_name in self._bypassed_backends:
                    continue

            allocator_backend = backend.get_allocator_backend()
            cname = get_backend_cname(allocator_backend)
            if cname not in obj_dict:
                new_keys, new_objs = allocate_and_copy_objects(
                    allocator_backend, keys, memory_objs, self.internal_copy_stream
                )
                obj_dict[cname] = (new_keys, new_objs)

            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            ks, objs = obj_dict[cname]
            requires_completion = backend.requires_put_completion()
            submitted = backend.batched_submit_put_task(
                ks, objs, transfer_spec=transfer_spec
            )
            if requires_completion and submitted:
                required_futures.extend(submitted)

        for _, objs in obj_dict.values():
            for memory_obj in objs:
                memory_obj.ref_count_down()

        return required_futures

    @staticmethod
    def _supports_layer_page_backends(
        selected: Sequence[StorageBackendInterface],
    ) -> bool:
        return any(
            isinstance(backend, LocalCPUBackend)
            and bool(getattr(backend, "use_hot", False))
            for backend in selected
        ) and all(
            isinstance(backend, LocalCPUBackend)
            or (
                isinstance(backend, RemoteBackend)
                and backend.connection is not None
                and callable(
                    getattr(backend.connection, "batched_put_external_pages", None)
                )
            )
            for backend in selected
        )

    def supports_batched_put_layer_pages(
        self, location: Optional[str] = None
    ) -> bool:
        """Whether selected backends accept one physical page for all layers."""
        return self._supports_layer_page_backends(
            [
                backend
                for _, backend in self.get_active_storage_backends(
                    location=location
                )
            ]
        )

    def batched_put_layer_pages(
        self,
        keys: Sequence[CacheEngineKey],
        pages: List[LayerPageMemoryObj],
        location: Optional[str] = None,
        req_id: str = "",
    ) -> list[Future]:
        """Store each physical page once locally and remotely."""
        if (
            len(keys) != len(pages)
            or len(set(keys)) != len(keys)
            or len({id(page) for page in pages}) != len(pages)
        ):
            raise ValueError("Layer-page put requires one unique key and page")
        if not pages:
            return []
        selected = list(self.get_active_storage_backends(location=location))
        if not self._supports_layer_page_backends(
            [backend for _, backend in selected]
        ):
            raise RuntimeError("Selected storage backends do not support layer pages")

        layer_count = pages[0].num_layers
        if any(
            not page.is_valid()
            or page.num_layers != layer_count
            or page.get_size() != page.layer_size * layer_count
            or len(page.metadata.dtypes or ()) != layer_count
            or any(dtype != key.dtype for dtype in page.metadata.dtypes or ())
            or page.valid_tokens
            != mooncake_valid_tokens(key, int(self.config.chunk_size))
            for key, page in zip(keys, pages, strict=True)
        ):
            raise ValueError("Layer-page put metadata does not match its key")
        local = next(
            backend
            for _, backend in selected
            if isinstance(backend, LocalCPUBackend)
            and bool(getattr(backend, "use_hot", False))
        )
        remote = next(
            (
                backend
                for _, backend in selected
                if isinstance(backend, RemoteBackend)
            ),
            None,
        )
        required_futures: list[Future] = []
        if remote is None:
            local.batched_submit_layer_pages(keys, pages)
        else:
            owner_by_storage = {
                int(page.raw_data.untyped_storage().data_ptr()): page.raw_data
                for page in pages
            }
            for page in pages:
                page.ref_count_up()
            try:
                future = self.batched_put_external_pages(
                    keys,
                    [
                        [page.layer_data_ptr(layer) for layer in range(layer_count)]
                        for page in pages
                    ],
                    [[page.layer_size] * layer_count for page in pages],
                    tuple(owner_by_storage.values()),
                    None,
                    req_id,
                )
            except Exception:
                for page in pages:
                    page.ref_count_down()
                raise

            completion: Future = Future()

            def finish_pages(
                completed_remote: Future,
                held=tuple(pages),
                page_keys=tuple(keys),
            ) -> None:
                error = None
                try:
                    completed_remote.result()
                    # A local page hit must also represent a durable remote page.
                    local.batched_submit_layer_pages(page_keys, list(held))
                except Exception as caught:
                    error = caught
                finally:
                    for page in held:
                        page.ref_count_down()
                if not completion.done():
                    if error is None:
                        completion.set_result(None)
                    else:
                        completion.set_exception(error)

            future.add_done_callback(finish_pages)
            required_futures.append(completion)
        for page in pages:
            page.ref_count_down()
        return required_futures

    def batched_put_external_pages(
        self,
        keys: Sequence[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        producer_events: tuple[Any, ...] | Any,
        req_id: str,
    ) -> Future:
        """Forward registered external page or legacy buffers to RemoteBackend."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            raise RuntimeError("Direct page store requires RemoteBackend")
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                raise RuntimeError("RemoteBackend is bypassed")
        return backend.batched_submit_external_pages(
            keys,
            buffer_ptrs,
            buffer_sizes,
            owners,
            producer_events,
            req_id,
        )

    def batched_get_external_pages(
        self,
        keys: Sequence[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        req_id: str,
    ) -> None:
        """Load remote pages directly into externally owned buffers."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            raise RuntimeError("Direct page load requires RemoteBackend")
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                raise RuntimeError("RemoteBackend is bypassed")
        backend.batched_get_external_pages(
            keys, buffer_ptrs, buffer_sizes, owners, req_id
        )

    def batched_external_pages_exist(
        self, keys: Sequence[CacheEngineKey]
    ) -> List[bool]:
        """Check arbitrary remote page keys without consulting LocalCPU."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            return [False] * len(keys)
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                return [False] * len(keys)
        return backend.batched_external_pages_exist(keys)

    def submit_remote_fill_direct_push(
        self,
        *,
        remote_session: str,
        source_plan: Any,
        destination_descriptors: tuple[Any, ...],
        activation: Any,
    ) -> Future:
        """Submit one armed remote fill through the active RemoteBackend."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            raise RuntimeError("Remote fill requires RemoteBackend")
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                raise RuntimeError("RemoteBackend is bypassed")
        return backend.submit_remote_fill_direct_push(
            remote_session=remote_session,
            source_plan=source_plan,
            destination_descriptors=destination_descriptors,
            activation=activation,
        )

    def prepare_remote_fill_source(self, source_plan: Any) -> Future:
        """Fence and register direct-push sources before destination ARM."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            raise RuntimeError("Remote fill requires RemoteBackend")
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                raise RuntimeError("RemoteBackend is bypassed")
        return backend.prepare_remote_fill_source(source_plan)

    def get_remote_fill_destination_session(self) -> str | None:
        """Return TP0's registered native destination session, if enabled."""
        backend = self.storage_backends.get("RemoteBackend")
        if not isinstance(backend, RemoteBackend):
            return None
        with self._bypass_lock:
            if "RemoteBackend" in self._bypassed_backends:
                return None
        return backend.get_remote_fill_destination_session()

    def get(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """

        # Search all backends for blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # TODO(Jiayi): need to make sure all memory_objs returned
            # are allocated by the allocator backend.
            memory_obj = backend.get_blocking(key)
            if memory_obj:
                if (
                    backend_name not in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
                    and "LocalCPUBackend" in self.storage_backends
                ):
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    local_cpu_backend.submit_put_task(key, memory_obj)
                return memory_obj

        return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def batched_get(
        self,
        keys: List[CacheEngineKey],
        location: Optional[str] = None,
    ) -> List[Optional[MemoryObj]]:
        """
        Blocking function to get the memory objects from the storages.
        """
        # TODO (ApostaC): remove the nested optional here
        for backend_name, storage_backend in self.get_active_storage_backends(location):
            memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                if None not in memory_objs:
                    self._write_back_to_local_cpu(
                        backend_name,
                        keys,
                        cast(List[MemoryObj], memory_objs),
                    )
                return memory_objs
        return [None] * len(keys)

    def _write_back_to_local_cpu(
        self,
        backend_name: str,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        """Write remote-tier hits into LocalCPUBackend hot cache."""
        if (
            backend_name in ["LocalCPUBackend", "PDBackend", "MaruBackend"]
            or "LocalCPUBackend" not in self.storage_backends
            or not self.config.local_cpu
            or not memory_objs
        ):
            return

        logger.debug(
            "Storing %s objects from %s to LocalCPUBackend",
            len(memory_objs),
            backend_name,
        )
        local_cpu_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(local_cpu_backend, LocalCPUBackend)
        local_cpu_backend.batched_submit_put_task(
            list(keys[: len(memory_objs)]),
            memory_objs,
        )

    def _normalize_layerwise_prefix(
        self,
        objs: List[Optional[MemoryObj]],
    ) -> List[MemoryObj]:
        """
        Return the consecutive prefix of valid memory objects.
        Release objects after the first None to match remote prefix semantics.
        """
        memory_objs: List[MemoryObj] = []
        found_failure = False
        for obj in objs:
            if found_failure:
                if obj is not None:
                    obj.ref_count_down()
                continue
            if obj is None:
                found_failure = True
                continue
            memory_objs.append(obj)
        return memory_objs

    def _layerwise_get_prefers_blocking(
        self,
        backend_name: str,
        backend: StorageBackendInterface,
    ) -> bool:
        if backend_name != "RemoteBackend":
            return False

        if self.config.get_extra_config_value("layerwise_use_blocking_get", False):
            return True

        connection = getattr(backend, "connection", None)
        if connection is None:
            return False

        if (
            connection.support_batched_get()
            and not connection.support_batched_get_non_blocking()
        ):
            return True
        return False

    async def _layerwise_batched_get_blocking(
        self,
        backend: StorageBackendInterface,
        backend_name: str,
        keys_multi_chunk: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        objs_optional = await asyncio.to_thread(
            backend.batched_get_blocking, keys_multi_chunk
        )
        objs = self._normalize_layerwise_prefix(objs_optional)
        self._write_back_to_local_cpu(backend_name, keys_multi_chunk, objs)
        return objs

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str] = None,
    ) -> Generator[Future, None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a future for each layer.
        """
        if location is None:
            location = "LocalCPUBackend"
        elif (
            location != "LocalCPUBackend"
            and "LocalCPUBackend" in self.storage_backends
            and keys
            and keys[0]
        ):
            local_backend = cast(
                LocalCPUBackend,
                self.storage_backends["LocalCPUBackend"],
            )
            if local_backend.contains_all_exact(
                key for layer_keys in keys for key in layer_keys
            ):
                location = "LocalCPUBackend"
        backend = self.storage_backends[location]
        use_blocking = self._layerwise_get_prefers_blocking(location, backend)
        for keys_multi_chunk in keys:
            if use_blocking:
                coro = self._layerwise_batched_get_blocking(
                    backend, location, keys_multi_chunk
                )
            else:
                coro = backend.batched_get_non_blocking(
                    "fake_lookup_id", keys_multi_chunk
                )
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            yield task

    def prefetch_single_done_callback(
        self,
        future: asyncio.Future,
        keys: list[CacheEngineKey],
        backend_name: str,
    ) -> None:
        """
        Callback function when a single prefetch task
        (i.e., prefetching from a single backend) is done.
        """
        # TODO(Jiayi): support write-back policy here
        pass

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_chunk_lengths_total: list[int],
        tier_expected_chunks: list[int],
    ) -> None:
        """
        Callback function when all prefetch tasks
        (i.e., prefetching from all backends for the entire request) are done.
        """
        assert self.async_lookup_server is not None
        self.event_manager.update_event_status(
            EventType.LOADING, lookup_id, status=EventStatus.DONE
        )
        res = task.result()

        # Calculate total retrieved chunks across all tiers based on actual results
        # from batched_get_non_blocking, not the batched_async_contains results.
        # This handles the case where chunks may be evicted between contains check
        # and actual retrieval.
        #
        # Example: chunk_size=256, 7 chunks total (1792 tokens) across 3 tiers
        #   cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        #   tier_expected_chunks = [3, 2, 2]  # Tier 0: 3, Tier 1: 2, Tier 2: 2
        #
        #   Chunks:
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- stored in Tier0, tier_expected_chunks[0]==3
        #         |---|      <--- stored in Tier1, tier_expected_chunks[1]==2
        #             |---|  <--- stored in Tier2, tier_expected_chunks[2]==2
        #
        # Case 1: All chunks retrieved successfully
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4)
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6)
        #   res = [[obj0, obj1, obj2], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 7
        #   retrieved_length = cum_chunk_lengths_total[7] = 1792
        #
        # Case 2: Tier 1 only got 1 chunk (eviction), Tier 2 got all 2 chunks
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |-|X|      <--- Tier1: retrieved 1 chunk (obj3), missing obj4
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1, obj2], [obj3], [obj5, obj6]]
        #   total_retrieved_chunks = 4 (stop at tier 1, tier 2 chunks ignored)
        #   retrieved_length = cum_chunk_lengths_total[4] = 1024
        #   Note: Even though tier 2 successfully retrieved 2 chunks, they are
        #   not counted because tier 1 has a gap, breaking prefix continuity.
        #
        # Case 3: Tier 0 only got 2 chunks (eviction), other tiers got all
        #   [0 1 2 3 4 5 6]
        #   |---|X|          <--- Tier0: retrieved 2 chunks (obj0, obj1), missing obj2
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4) - IGNORED
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 2 (stop at tier 0, all subsequent ignored)
        #   retrieved_length = cum_chunk_lengths_total[2] = 512
        total_retrieved_chunks = 0
        for tier_idx, tier_result in enumerate(res):
            actual_chunks = len(tier_result)
            expected_chunks = tier_expected_chunks[tier_idx]
            total_retrieved_chunks += actual_chunks

            # If a tier retrieved fewer chunks than expected, we stop counting
            # because subsequent chunks are not contiguous
            if actual_chunks < expected_chunks:
                # Release all chunks in subsequent tiers since they won't be used
                for subsequent_tier in res[tier_idx + 1 :]:
                    for mem_obj in subsequent_tier:
                        mem_obj.ref_count_down()
                break

        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
        logger.info(
            f"Responding to scheduler for lookup id {lookup_id}"
            f" with retrieved length {retrieved_length}"
        )
        self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)

    async def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
        search_range: Optional[list[str]] = None,
        pin: bool = False,
    ) -> None:
        """
        Perform asynchronous lookup and prefetching across all storage backends.

        :param str lookup_id: The unique id (e.g., request id) for the request.
        :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
        :param list[int] cum_chunk_lengths: The cumulative token lengths of the chunks.
            This is a list where cum_chunk_lengths[i] represents the total number of
            tokens from chunk 0 to chunk i-1 (inclusive).
            Example: If chunk_size=256 and we have 3 chunks:
                - chunk 0: 256 tokens (tokens 0-255)
                - chunk 1: 256 tokens (tokens 256-511)
                - chunk 2: 128 tokens (tokens 512-639)
            Then cum_chunk_lengths = [0, 256, 512, 640]
            Note: len(cum_chunk_lengths) = len(keys) + 1
        :param Optional[list[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now. If None, search in all backends.
        :param bool pin: Whether to pin the keys.
        """

        # NOTE(Jiayi): Currently, the retrieval pattern is always
        # prefix-based. That is, we retrieve 0-t1 tokens from backend 1
        # and retrieve t1-t2 tokens from backend 2, etc. The assumption
        # here is that the suffix chunks are more likely to be evicted
        # than the prefix chunks.
        # TODO(Jiayi): We need to change/optimize this for non-prefix
        # based retrieval patterns or cases where middle chunks are missing.

        # NOTE(Jiayi): We can tolerate the last tier to have fewer loaded
        # chunks than its lookup result indicated. This is especially helpful
        # for P2PBackend.

        num_total_chunks = len(keys)
        num_total_hit_chunks = 0
        # cum_chunk_lengths_total: A copy of the original cumulative chunk lengths
        # for all chunks. This is preserved to calculate the final token count
        # based on the actual retrieved chunks.
        # Example: If chunk_size=256 and we have 3 chunks with total 640 tokens:
        #     cum_chunk_lengths_total = [0, 256, 512, 640]
        # If we retrieve 2 chunks, the retrieved token count is:
        #     cum_chunk_lengths_total[2] = 512 tokens
        cum_chunk_lengths_total = cum_chunk_lengths[:]
        loading_tasks = []
        tier_expected_chunks = []
        # we also keep track of the keys for each tier and each chunk
        loading_task_keys: list[list[CacheEngineKey]] = []
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            num_hit_chunks = await backend.batched_async_contains(lookup_id, keys, pin)

            if num_hit_chunks == 0:
                continue

            num_total_hit_chunks += num_hit_chunks
            tier_expected_chunks.append(num_hit_chunks)

            backend_keys = keys[:num_hit_chunks]
            loading_task_keys.append(backend_keys)

            assert self.async_serializer is not None, (
                "Async serializer must be initialized via post_init before using "
                "async_lookup_and_prefetch."
            )
            # num_hit_chunks is only used for the multi serializer
            get_coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    lookup_id,
                    backend_keys,
                    {"cum_chunk_lengths": cum_chunk_lengths[: num_hit_chunks + 1]},
                ),
                num_hit_chunks,
            )
            loading_task = asyncio.create_task(get_coro)
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=keys,
                    backend_name=backend_name,
                )
            )

            loading_tasks.append(loading_task)

            cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]

            if num_total_hit_chunks == num_total_chunks:
                break
            keys = keys[num_hit_chunks:]

        # If no chunks were hit across all backends, respond immediately and return.
        if num_total_hit_chunks == 0:
            if self.async_lookup_server is not None:
                self.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            return

        # gather_with_keys() here make a pair of (key, memory_obj) for each chunk
        # in each tier. The all_done result's layout is like following and
        # will be processed in _async_process_tokens_internal()
        # Tier 0:
        #  Tuple(loading_task_keys[0][0] : MemoryObj0)
        #  Tuple(loading_task_keys[0][1] : MemoryObj1)
        # Tier 1:
        #  Tuple(loading_task_keys[1][0] : MemoryObj2)
        #  Tuple(loading_task_keys[1][1] : MemoryObj3)
        async def gather_with_keys() -> list[list[tuple[CacheEngineKey, MemoryObj]]]:
            loading_results = await asyncio.gather(*loading_tasks)
            return [
                list(zip(keys, results, strict=False))
                for keys, results in zip(
                    loading_task_keys, loading_results, strict=False
                )
            ]

        all_done = asyncio.create_task(gather_with_keys())
        # Register the event before adding the callback to avoid race conditions
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )

        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
            )
        )

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the hot cache on LocalCPUBackend.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled: True to enable hot cache, False to disable
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            logger.warning("Cannot set hot_cache: LocalCPUBackend not available")
            return

        if not enabled:
            backend.clear()
        backend.use_hot = enabled
        logger.info("LocalCPUBackend hot_cache set to %s", enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        backend = self.local_cpu_backend
        if not isinstance(backend, LocalCPUBackend):
            return False
        return backend.use_hot

    def set_freeze(self, enabled: bool) -> None:
        """
        Set freeze mode.

        When enabled, only local_cpu backend will be used for retrieval.
        """
        with self._freeze_lock:
            self._freeze = enabled
        logger.info("StorageManager freeze mode set to %s", enabled)

    def is_frozen(self) -> bool:
        """
        Get freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        with self._freeze_lock:
            return self._freeze

    def set_backend_bypass(self, backend_name: str, bypassed: bool) -> None:
        """
        Set bypass mode for a specific backend.

        When a backend is bypassed:
        - It will be skipped during contains/put/get operations
        - This is typically used when a health check fails with LOCAL_CPU fallback

        Args:
            backend_name: The name of the backend to bypass (e.g., "RemoteBackend")
            bypassed: True to bypass, False to restore normal operation
        """
        with self._bypass_lock:
            if bypassed:
                self._bypassed_backends.add(backend_name)
                logger.info(f"StorageManager: Backend {backend_name} is now bypassed")
            else:
                self._bypassed_backends.discard(backend_name)
                logger.info(
                    f"StorageManager: Backend {backend_name} bypass removed, "
                    "restored to normal operation"
                )

    def is_backend_bypassed(self, backend_name: str) -> bool:
        """
        Check if a backend is currently bypassed.

        Args:
            backend_name: The name of the backend to check

        Returns:
            bool: True if the backend is bypassed, False otherwise
        """
        with self._bypass_lock:
            return backend_name in self._bypassed_backends

    def get_bypassed_backends(self) -> List[str]:
        """
        Get the list of currently bypassed backend names.

        Returns:
            List[str]: List of bypassed backend names
        """
        with self._bypass_lock:
            return list(self._bypassed_backends)

    def get_all_backend_names(self) -> List[str]:
        """
        Get the list of all registered backend names.

        Returns:
            List[str]: List of all backend names
        """
        return list(self.storage_backends.keys())

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            if backend.contains(key, pin_in_backend):
                return backend_name

        return None

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> tuple[int, dict]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: Return hit chunks and block mapping by prefix match.
        """
        total_keys = len(keys)
        total_hit_chunks = 0
        block_mapping = {}
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            hit_chunks = backend.batched_contains(keys, pin_in_backend)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = keys[:hit_chunks]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]

        return total_hit_chunks, block_mapping

    def batched_contains_layer_pages(
        self,
        keys: List[LayerCacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> tuple[int, dict]:
        """Prefix lookup using one physical page key per logical chunk."""
        total = 0
        mapping = {}
        remaining: List[LayerCacheEngineKey] = keys
        for name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            backend_keys: list[CacheEngineKey]
            if name == "LocalCPUBackend":
                backend_keys = [key.without_layer() for key in remaining]
                hits = backend.batched_contains_layer_pages(remaining, pin)
            else:
                contains = getattr(backend, "batched_contains_layer_pages", None)
                if not callable(contains):
                    continue
                backend_keys = list(remaining)
                hits = contains(backend_keys, pin)
            if not hits:
                continue
            mapping[name] = backend_keys[:hits]
            total += hits
            if total == len(keys):
                break
            remaining = remaining[hits:]
        return total, mapping

    def batched_contains_two_group_layer_pages(
        self,
        group0_keys: Sequence[LayerCacheEngineKey],
        group1_keys: Sequence[LayerCacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> tuple[int, dict[str, list[CacheEngineKey]]]:
        """Locate a contiguous prefix of complete two-group physical pages.

        Args:
            group0_keys: Group 0 representative layer keys in logical order.
            group1_keys: Aligned Group 1 representative layer keys.
            search_range: Optional ordered backend filter.
            pin: Atomically pin complete pairs in pin-capable backends.

        Returns:
            The number of complete page pairs and their backend mapping. Each
            mapping value is interleaved ``g0, g1`` by logical page.

        Raises:
            ValueError: If the key sequences are not aligned two-group page
                representatives.

        Notes:
            A backend can contribute only whole pairs. An isolated Group 0 or
            Group 1 hit never advances the prefix and is never retained. The
            LocalCPU backend performs lookup and pair pinning under its single
            cache lock; persistent backends are rechecked on the exact even
            prefix before any backend-specific pin is recorded.
        """
        if len(group0_keys) != len(group1_keys):
            raise ValueError("Two-group page lookup requires aligned key counts")
        for group0, group1 in zip(group0_keys, group1_keys, strict=True):
            if (
                group0.kv_group != 0
                or group1.kv_group != 1
                or group0.layer_id != group1.layer_id
                or group0.chunk_hash != group1.chunk_hash
                or group0.model_name != group1.model_name
                or group0.world_size != group1.world_size
                or group0.worker_id != group1.worker_id
            ):
                raise ValueError(
                    "Two-group page lookup requires identical logical page pairs"
                )

        total_pairs = 0
        mapping: dict[str, list[CacheEngineKey]] = {}
        pinned: list[tuple[StorageBackendInterface, list[CacheEngineKey]]] = []

        def rollback() -> None:
            for backend, keys in reversed(pinned):
                batched_unpin = getattr(backend, "batched_unpin", None)
                if callable(batched_unpin):
                    batched_unpin(keys)
                else:
                    for key in keys:
                        backend.unpin(key)

        try:
            for name, backend in self.get_active_storage_backends(
                search_range=search_range
            ):
                if total_pairs == len(group0_keys):
                    break
                remaining0 = group0_keys[total_pairs:]
                remaining1 = group1_keys[total_pairs:]
                if name == "LocalCPUBackend":
                    physical0 = [key.without_layer() for key in remaining0]
                    physical1 = [key.without_layer() for key in remaining1]
                    pair_hits = backend.batched_contains_two_group_prefix(
                        physical0,
                        physical1,
                        pin=pin,
                    )
                    if not 0 <= pair_hits <= len(remaining0):
                        raise RuntimeError(
                            "LocalCPU returned an invalid two-group page prefix"
                        )
                    backend_keys = [
                        key
                        for pair in zip(
                            physical0[:pair_hits],
                            physical1[:pair_hits],
                            strict=True,
                        )
                        for key in pair
                    ]
                else:
                    contains = getattr(backend, "batched_contains_layer_pages", None)
                    if not callable(contains):
                        continue
                    interleaved = [
                        key
                        for pair in zip(remaining0, remaining1, strict=True)
                        for key in pair
                    ]
                    raw_hits = int(contains(interleaved, False))
                    if not 0 <= raw_hits <= len(interleaved):
                        raise RuntimeError(
                            f"{name} returned an invalid layer-page prefix"
                        )
                    pair_hits = raw_hits // 2
                    backend_keys = interleaved[: 2 * pair_hits]
                    if pin and backend_keys:
                        retained = int(contains(backend_keys, True))
                        if retained != len(backend_keys):
                            if retained > 0:
                                partial = backend_keys[:retained]
                                batched_unpin = getattr(backend, "batched_unpin", None)
                                if callable(batched_unpin):
                                    batched_unpin(partial)
                                else:
                                    for key in partial:
                                        backend.unpin(key)
                            return total_pairs, mapping

                if pair_hits <= 0:
                    continue
                mapping[name] = backend_keys
                if pin:
                    pinned.append((backend, backend_keys))
                total_pairs += pair_hits
        except Exception:
            if pin:
                rollback()
            raise

        return total_pairs, mapping

    def get_block_mapping(
        self, chunk_infos: List[Tuple[CacheEngineKey, int, int]]
    ) -> Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        """
        Get block mapping for the given chunk infos, works by prefix match.

        :param List[Tuple[CacheEngineKey, int, int]] chunk_infos:
        List of chunk infos, each tuple contains (key, begin, end)

        :return: Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        Block mapping for the given chunk infos, each key is the backend name,
        each value is a list of chunk infos in the backend.
        """
        keys = [chunk_info[0] for chunk_info in chunk_infos]
        total_keys = len(keys)
        block_mapping = {}
        total_hit_chunks = 0
        for backend_name, backend in self.get_active_storage_backends():
            hit_chunks = backend.batched_contains(keys)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = chunk_infos[
                total_hit_chunks : total_hit_chunks + hit_chunks
            ]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]
        return block_mapping

    def touch_cache(self) -> None:
        for backend_name, backend in self.storage_backends.items():
            if backend_name == "LocalCPUBackend" or backend_name == "LocalDiskBackend":
                backend.touch_cache()

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                num_removed += backend.remove(key)

        return num_removed

    def batched_remove(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Batched remove the keys and the corresponding cache in the specified
        locations.

        :param List[CacheEngineKey] keys: The keys to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """
        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                num_removed += backend.batched_remove(keys)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                batched = getattr(backend, "batched_unpin", None)
                if callable(batched):
                    batched(keys)
                else:
                    for key in keys:
                        backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared tokens in the specified
        storage backends.
        """

        num_cleared_tokens = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared_tokens += backend.clear()
                else:
                    logger.warning(
                        f"Storage backend {backend_name} does not support "
                        "clear operation. Skipping."
                    )

        return num_cleared_tokens

    def memcheck(self) -> bool:
        """
        Check the integrity of the underlying storage backend's
        memory allocators

        Returns:
            True if everything is good otherwise False
        """
        for backend in self.storage_backends.values():
            if not isinstance(backend, AllocatorBackendInterface):
                continue
            if not backend.get_memory_allocator().memcheck():
                return False
        return True

    def get_active_storage_backends(
        self,
        location: Optional[str] = None,
        search_range: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, StorageBackendInterface], None, None]:
        """
        Get the active storage backends based on freeze mode, bypass mode, and filters.

        :param Optional[str] location: If specified, only yield backends
            matching this exact name.
        :param Optional[List[str]] search_range: If specified, only yield
            backends whose names are in this list.

        :return: Generator of (backend_name, backend) tuples.
        """
        for backend_name, backend in self.storage_backends.items():
            # In freeze mode, only use local_cpu backend
            with self._freeze_lock:
                if self._freeze and backend_name != "LocalCPUBackend":
                    continue
            # Skip bypassed backends
            with self._bypass_lock:
                if backend_name in self._bypassed_backends:
                    continue
            if location and backend_name != location:
                continue
            if search_range and backend_name not in search_range:
                continue
            yield backend_name, backend

    def get_non_allocator_backends(self) -> List[str]:
        """
        Get the names of the actual storage backends. Some backends,
        such as LocalCPUBackend and PDBackend, in some cases, only
        serve as a backend for allocation.
        """
        storage_names = []
        for backend_name, backend in self.storage_backends.items():
            if "LocalCPUBackend" == backend_name and not self.config.local_cpu:
                # if local_cpu is False, means LocalCPUBackend is only a allocator
                continue
            if "PDBackend" == backend_name and backend.pd_config.role == "sender":  # type: ignore
                # if pd_config.role is sender, means PDBackend is only a allocator
                continue
            storage_names.append(backend_name)
        return storage_names

    def list_backends(self) -> Dict[str, str]:
        """
        List all active storage backends.

        Returns:
            Dict mapping backend name to its class name.
        """
        with self.manager_lock:
            return {
                name: type(backend).__name__
                for name, backend in self.storage_backends.items()
            }

    def close_backend(self, backend_name: str) -> bool:
        """
        Close and remove a specific storage backend by name.

        The backend will be closed and removed from the internal
        dict so that no stale references remain.

        Args:
            backend_name: The name of the backend to close.

        Returns:
            True if the backend was found and closed, False
            otherwise.
        """
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                logger.warning(
                    "Backend %s not found, cannot close",
                    backend_name,
                )
                return False

            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)

            del self.storage_backends[backend_name]

            # Update derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            if backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None
            logger.info("Backend %s closed and removed", backend_name)
            return True

    def create_backends(self) -> Dict[str, str]:
        """
        Create new storage backends based on current config.

        Backends that are already present will be skipped
        **before** instantiation so that no unnecessary
        resources are allocated.  This allows callers to close
        a subset of backends, update config via ``/conf``,
        and then call this method to bring up only the missing
        backends.

        Returns:
            Dict mapping newly created backend name to its
            class name.
        """
        with self.manager_lock:
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=("cuda" if is_cuda_worker(self.metadata) else "cpu"),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
            )

            created: Dict[str, str] = {}
            for name, backend in new_backends.items():
                self.storage_backends[name] = backend
                created[name] = type(backend).__name__
                logger.info(
                    "Created backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu

            return created

    def recreate_backend(self, backend_name: str) -> Dict[str, str]:
        """
        Close a backend and recreate it from current config.

        This is an atomic close-then-create operation that
        combines :meth:`close_backend` and :meth:`create_backends`
        into a single step.

        Args:
            backend_name: Name of the backend to recreate
                (e.g. ``RemoteBackend``).

        Returns:
            Dict mapping newly created backend name to its
            class name.

        Raises:
            KeyError: If *backend_name* does not exist.
        """
        with self.manager_lock:
            backend = self.storage_backends.get(backend_name)
            if backend is None:
                raise KeyError("Backend %s not found" % backend_name)

            # --- close ---
            try:
                logger.info("Closing backend: %s", backend_name)
                backend.close()
            except Exception:
                logger.exception("Error closing backend %s", backend_name)
            del self.storage_backends[backend_name]

            # --- create ---
            existing_names = set(self.storage_backends)
            new_backends = CreateStorageBackends(
                self.config,
                self.metadata,
                self.loop,
                dst_device=("cuda" if is_cuda_worker(self.metadata) else "cpu"),
                lmcache_worker=self.lmcache_worker,
                skip_backends=existing_names,
                existing_backends=self.storage_backends,
            )

            created: Dict[str, str] = {}
            for name, be in new_backends.items():
                self.storage_backends[name] = be
                created[name] = type(be).__name__
                logger.info(
                    "Recreated backend: %s (%s)",
                    name,
                    created[name],
                )

            # Refresh derived references
            self.non_allocator_backends = self.get_non_allocator_backends()
            cpu = self.storage_backends.get("LocalCPUBackend")
            if cpu is not None:
                self.local_cpu_backend = cpu
            elif backend_name == "LocalCPUBackend":
                self.local_cpu_backend = None

            return created

    def close(self):
        logger.info("Closing StorageManager...")

        # Close all backends
        for name, backend in self.storage_backends.items():
            try:
                logger.info(f"Closing storage backend: {name}")
                backend.close()
                logger.info(f"Storage backend {name} closed successfully")
            except Exception as e:
                logger.error(f"Error closing backend {name}: {e}")

        # Stop event loop
        try:
            if self.loop.is_running():
                logger.info("Stopping event loop...")
                self.loop.call_soon_threadsafe(self.loop.stop)
                logger.info("Event loop stop signaled")
        except Exception as e:
            logger.error(f"Error stopping event loop: {e}")

        # Wait for thread with timeout
        if self.thread.is_alive():
            logger.info("Waiting for storage manager thread to finish...")
            self.thread.join(timeout=10.0)

            if self.thread.is_alive():
                logger.warning(
                    "Storage manager thread did not terminate within 10s timeout. "
                    "Proceeding with shutdown anyway."
                )
            else:
                logger.info("Storage manager thread terminated successfully")
        else:
            logger.info("Storage manager thread already stopped")

        logger.info("Storage manager closed.")
