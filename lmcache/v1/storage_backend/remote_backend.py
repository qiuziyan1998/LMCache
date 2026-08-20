# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, List, Optional, Sequence, Set
import asyncio
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self.put_tasks: Set[CacheEngineKey] = set()
        # Single-key callers must observe the completion of the writer that
        # actually owns a pending key.  ``put_tasks`` remains a set because it
        # is also the public in-flight gauge used by batched operations.
        self._single_put_futures: dict[CacheEngineKey, Future] = {}
        self._single_put_callbacks: dict[
            CacheEngineKey, list[Callable[[CacheEngineKey], None]]
        ] = {}
        self.lock = threading.Lock()

        assert config.remote_url is not None

        self.remote_url = config.remote_url

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self.init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status.  When save_only_first_rank is disabled,
        # workers store distinct per-rank keys, so remote must not collapse
        # non-zero worker ids to rank 0 or skip their remote writes.
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        remote_worker_id_as0 = config.get_extra_config_value(
            "remote_enable_mla_worker_id_as0", save_only_first_rank
        )
        if (
            remote_worker_id_as0
            and metadata.use_mla
            and metadata.world_size > 1
            and not save_only_first_rank
        ):
            raise ValueError(
                "remote_enable_mla_worker_id_as0=True with "
                "save_only_first_rank=False would replace per-rank MLA cache "
                "data with rank-0 data. Disable worker-ID collapsing or "
                "enable save_only_first_rank."
            )

        self._mla_worker_id_as0_mode = (
            remote_worker_id_as0
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info(f"metadata={metadata}")
        logger.info(
            f"Connected to remote storage at {config.remote_url}, "
            f"remote_mla_worker_id_as_0 mode: {self._mla_worker_id_as0_mode}"
        )

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # NOTE: Health monitoring is now handled at the LMCacheEngine level
        # through HealthMonitor. RemoteBackend no longer manages its own
        # health monitoring. The HealthMonitor in LMCacheEngine will
        # register RemoteBackendHealthCheck for each RemoteBackend.

        self._setup_metrics()

        self._get_blocking_failed_count = 0
        self._put_failed_count = 0

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.remote_put_task_num.set_function(
                lambda: len(self.put_tasks)
            )
            prometheus_logger.get_blocking_failed_count.set_function(
                lambda: self._get_blocking_failed_count
            )
            prometheus_logger.put_failed_count.set_function(
                lambda: self._put_failed_count
            )

    def __str__(self):
        return self.__class__.__name__

    def init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            assert self.config.remote_url is not None
            self.connection = CreateConnector(
                self.config.remote_url,
                self.loop,
                self.local_cpu_backend,
                self.config,
                self.metadata,
            )
            logger.info(
                f"Connection initialized/re-established at {self.config.remote_url}"
            )
        except IrrecoverableException:
            logger.error("Irrecoverable error during connection initialization")
            raise
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(f"Failed to initialize/re-establish remote connection: {e}")
            self.connection = None

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        try:
            if self.config.extra_config is not None and self.config.extra_config.get(
                "use_exists_sync", False
            ):
                return self.connection.exists_sync(key)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.exists(key), self.loop
                )
                res = future.result()
                return res
        except Exception as e:
            logger.warning(f"Remote connection failed in contains: {e}")
            logger.warning("Returning False")
            return False

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_contains, returning 0")
            return 0

        if not self.connection.support_batched_contains():
            return super().batched_contains(keys, pin)

        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            return self.connection.batched_contains(keys)
        except Exception as e:
            logger.warning(f"Remote connection failed in batched_contains: {e}")
            return 0

    def batched_contains_layer_pages(
        self, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        """Check Mooncake page keys without accepting legacy layer objects."""
        if self.connection is None:
            return 0
        contains = getattr(self.connection, "batched_contains_layer_pages", None)
        if not callable(contains):
            return 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
        try:
            return contains(keys)
        except Exception as error:
            logger.warning(f"Remote layer-page lookup failed: {error}")
            return 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def requires_put_completion(self) -> bool:
        return (
            self.connection is not None
            and self.connection.requires_put_completion()
        )

    def put_callback(self, future: Future, key: CacheEngineKey):
        with self.lock:
            self.put_tasks.discard(key)
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error(f"Put task failed for key {key}: {e}")

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to remote storage asynchronously.

        :param on_complete_callback: Optional callback invoked after the remote
            write completes. Callback exceptions are caught and logged.
        """

        def create_immediate_empty_future() -> Future:
            f: Future = Future()
            f.set_result(None)
            return f

        if self.connection is None:
            logger.warning("Connection is None in submit_put_task, returning None")
            return create_immediate_empty_future()

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            return create_immediate_empty_future()

        completion: Future = Future()
        with self.lock:
            # Do not translate "another writer is pending" into "this put is
            # complete".  Returning the same completion future preserves the
            # persistent-store durability barrier for tails and repair paths.
            pending = getattr(self, "_single_put_futures", {}).get(key)
            if pending is not None:
                if on_complete_callback is not None:
                    self._single_put_callbacks.setdefault(key, []).append(
                        on_complete_callback
                    )
                return pending
            if not hasattr(self, "_single_put_futures"):
                # Some focused tests construct RemoteBackend without running
                # __init__; keep that supported without weakening production.
                self._single_put_futures = {}
                self._single_put_callbacks = {}
            self._single_put_futures[key] = completion
            self._single_put_callbacks[key] = (
                [on_complete_callback] if on_complete_callback is not None else []
            )
            self.put_tasks.add(key)

        memory_obj.ref_count_up()
        try:
            compressed_memory_obj = self.serializer.serialize(memory_obj)
        except BaseException as error:
            with self.lock:
                if self._single_put_futures.get(key) is completion:
                    self._single_put_futures.pop(key, None)
                    self._single_put_callbacks.pop(key, None)
                    self.put_tasks.discard(key)
            if not completion.done():
                completion.set_exception(error)
            raise
        finally:
            memory_obj.ref_count_down()

        def put_done_callback(f: Future) -> None:
            self.put_callback(f, key)
            try:
                result = f.result()
            except BaseException as error:
                if not completion.done():
                    completion.set_exception(error)
            else:
                if not completion.done():
                    completion.set_result(result)
            with self.lock:
                if self._single_put_futures.get(key) is completion:
                    self._single_put_futures.pop(key, None)
                    callbacks = self._single_put_callbacks.pop(key, ())
                else:
                    callbacks = ()
            for callback in callbacks:
                self._invoke_put_complete_callback(callback, key)

        # Submission failure must unwind the shared completion registration.
        coroutine = self.connection.put(key, compressed_memory_obj)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except BaseException as error:
            coroutine.close()
            with self.lock:
                if self._single_put_futures.get(key) is completion:
                    self._single_put_futures.pop(key, None)
                    self._single_put_callbacks.pop(key, None)
                    self.put_tasks.discard(key)
            if not completion.done():
                completion.set_exception(error)
            raise
        future.add_done_callback(put_done_callback)
        return completion

    @staticmethod
    def _invoke_put_complete_callback(
        callback: Callable[[CacheEngineKey], None],
        key: CacheEngineKey,
    ) -> None:
        try:
            callback(key)
        except Exception as error:
            logger.warning("on_complete_callback failed for key %s: %s", key, error)

    def batched_put_callback(self, future: Future, keys: List[CacheEngineKey]):
        """
        Callback function for batched put tasks.
        """
        with self.lock:
            self.put_tasks.difference_update(keys)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """
        Submit batched put tasks to store KV caches to remote storage.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        if self.connection is None:
            logger.warning(
                "Connection is None in batched_submit_put_task, returning None"
            )
            return None
        if self.connection.support_batched_put():
            if self._mla_worker_id_as0_mode:
                return None

            # First, increment reference counts for all objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

            compressed_memory_objs = []
            try:
                for memory_obj in memory_objs:
                    compressed_memory_objs.append(self.serializer.serialize(memory_obj))
            finally:
                # Always decrement reference counts for all objects,
                # regardless of whether serialization succeeded or failed
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

            def batched_done_callback(f: Future) -> None:
                self.batched_put_callback(f, list(keys))
                # Invoke per-key callback for each key in the batch
                if on_complete_callback is not None:
                    for key in keys:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                f"on_complete_callback failed for key {key}: {e}"
                            )

            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_put(keys, compressed_memory_objs),  # type: ignore
                self.loop,
            )
            future.add_done_callback(batched_done_callback)
            return [future] if self.requires_put_completion() else None
        else:
            futures: List[Future] = []
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                futures.append(
                    self.submit_put_task(
                        key,
                        memory_obj,
                        on_complete_callback=on_complete_callback,
                    )
                )
            return futures if self.requires_put_completion() else None

    def batched_submit_external_pages(
        self,
        keys: Sequence[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        ready_event: Any,
        req_id: str,
    ) -> Future:
        """Submit registered external buffers to the remote connector."""
        if self.connection is None:
            raise RuntimeError("Remote connection is unavailable")
        if self._mla_worker_id_as0_mode:
            raise RuntimeError("External page puts are only valid on the saving rank")
        normalized = list(keys)
        future = asyncio.run_coroutine_threadsafe(
            self.connection.batched_put_external_pages(
                normalized,
                buffer_ptrs,
                buffer_sizes,
                owners,
                ready_event,
                req_id,
            ),
            self.loop,
        )
        return future

    def batched_get_external_pages(
        self,
        keys: Sequence[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        req_id: str,
    ) -> None:
        """Retrieve remote pages directly into registered external buffers."""
        if self.connection is None:
            raise RuntimeError("Remote connection is unavailable")
        normalized = list(keys)
        if self._mla_worker_id_as0_mode:
            normalized = [key.with_new_worker_id(0) for key in normalized]
        future = asyncio.run_coroutine_threadsafe(
            self.connection.batched_get_external_pages(
                normalized, buffer_ptrs, buffer_sizes, owners, req_id
            ),
            self.loop,
        )
        try:
            future.result(self.config.blocking_timeout_secs)
        except TimeoutError as timeout_error:
            # The connector may be running an uncancellable native write into
            # externally owned destinations. Drain it so its serialization lock
            # and buffer registrations remain valid before surfacing timeout.
            try:
                future.result()
            except BaseException as native_error:
                raise TimeoutError(
                    "Direct external page load failed after timing out"
                ) from native_error
            raise timeout_error

    def batched_external_pages_exist(
        self, keys: Sequence[CacheEngineKey]
    ) -> List[bool]:
        """Check arbitrary Mooncake pages through the storage event loop."""
        if self.connection is None:
            return [False] * len(keys)
        normalized = list(keys)
        if self._mla_worker_id_as0_mode:
            normalized = [key.with_new_worker_id(0) for key in normalized]
        return self.connection.batched_external_pages_exist(normalized)

    def submit_remote_fill_direct_push(
        self,
        *,
        remote_session: str,
        source_plan: Any,
        destination_descriptors: tuple[Any, ...],
        activation: Any,
    ) -> Future:
        """Submit one armed native direct push on the storage event loop."""
        if self.connection is None:
            raise RuntimeError("Remote connection is unavailable")
        return asyncio.run_coroutine_threadsafe(
            self.connection.push_external_pages(
                remote_session=remote_session,
                source_plan=source_plan,
                destination_descriptors=destination_descriptors,
                activation=activation,
            ),
            self.loop,
        )

    def get_remote_fill_destination_session(self) -> str | None:
        """Return the active connector's native destination session."""
        if self.connection is None:
            return None
        return self.connection.get_remote_fill_destination_session()

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in get_blocking "
                "(likely scheduler role), returning None"
            )
            return None

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.config.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            logger.warning("Error occurred in get_blocking: %s, return None", e)
            memory_obj = None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            self._get_blocking_failed_count += 1
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            "Get takes %.6f msec, deserialization takes %.6f msec",
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
        )
        return decompressed_memory_obj

    @property
    def get_blocking_failed_count(self):
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self):
        return self._put_failed_count

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_blocking "
                "(likely scheduler role), returning None list"
            )
            return [None] * len(keys)

        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)

        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        t1 = time.perf_counter()
        # batched get
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_get(keys), self.loop
            )
            try:
                memory_objs = future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning(
                        "batched get blocking timeout, trigger cancel the future task"
                    )
                    future.cancel()
                else:
                    logger.warning(
                        f"Error occurred in batched_get_blocking: {e}, "
                        f"returning None list"
                    )
                memory_objs = [None] * len(keys)
        else:
            remote_backend_individual_get_stats: dict[
                CacheEngineKey, dict[str, float]
            ] = {}
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_individual_get_stats"
                ] = remote_backend_individual_get_stats

            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.config.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning(
                                "get blocking timeout, trigger cancel the future task"
                            )
                            fut.cancel()
                        else:
                            logger.warning(
                                f"Error occurred in get_blocking: {e}, returning None"
                            )
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None)
                    fut.cancel()

        t2 = time.perf_counter()
        duration = t2 - t1
        self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)

        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "remote_backend_batched_get_blocking_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "remote_backend_batched_get_blocking_time", 0.0
                )
                + duration
            )
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        error_happened = False
        for memory_obj in memory_objs:
            if memory_obj is None:
                error_happened = True
                decompressed_memory_objs.append(None)
            else:
                decompressed_memory_objs.append(
                    self.deserializer.deserialize(memory_obj)
                )
        if error_happened:
            self._get_blocking_failed_count += 1

        assert len(decompressed_memory_objs) == len(keys), (
            f"keys length: {len(keys)}, "
            f"decompressed memory objs length: {len(decompressed_memory_objs)}"
        )
        return decompressed_memory_objs

    def batched_get_layer_pages(
        self, keys: List[CacheEngineKey]
    ) -> list[LayerPageMemoryObj]:
        """Retrieve pages identified by one representative key per chunk."""
        if self.connection is None:
            raise RuntimeError("Remote connection is unavailable")
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
        retrieve = getattr(self.connection, "batched_get_layer_pages", None)
        if not callable(retrieve):
            raise RuntimeError("Remote connector does not support layer pages")
        future = asyncio.run_coroutine_threadsafe(retrieve(keys), self.loop)
        try:
            return future.result(self.config.blocking_timeout_secs)
        except TimeoutError:
            def release_late_result(done: Future) -> None:
                try:
                    pages = done.result()
                except BaseException:
                    return
                for page in pages:
                    try:
                        if page.is_valid():
                            page.ref_count_down()
                    except Exception:
                        logger.exception("Failed to release a late layer-page result")

            future.add_done_callback(release_late_result)
            raise
        except BaseException:
            future.cancel()
            raise

    async def support_batched_async_contains(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_async_contains()
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_async_contains, returning 0")
            return 0
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]

        try:
            assert self.connection.support_batched_async_contains(), (
                f"Connector {self.connection} does not support batched async contains"
            )
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_async_contains(lookup_id, keys, pin),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_async_contains timed out")
            return 0
        except Exception as e:
            logger.warning(f"Error occurred in batched_async_contains: {e}")
            return 0

    async def support_batched_get_non_blocking(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_get_non_blocking()
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_non_blocking "
                "(likely scheduler role), returning empty list"
            )
            return []

        if self.connection is None:
            logger.warning(
                "Connection is None in batched_get_non_blocking, returning empty list"
            )
            return []
        try:
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_get_non_blocking(lookup_id, keys),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_get_non_blocking timed out")
            return []
        except Exception as e:
            logger.warning(f"Error occurred in batched_get_non_blocking: {e}")
            return []

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def remove(self, key, force=True):
        if self.connection is None:
            logger.warning("Connection is None in remove, returning False")
            return False

        try:
            return self.connection.remove_sync(key)
        except Exception as e:
            logger.exception(
                f"Failed to remove key {key} from remote backend, error: {e}"
            )
            return False

    def get_allocator_backend(self):
        assert self.local_cpu_backend is not None, (
            "local_cpu_backend is required for get_allocator_backend, "
            "should not be called in scheduler role"
        )
        return self.local_cpu_backend

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning(f"Error occurred when closing remote connection: {e}")
