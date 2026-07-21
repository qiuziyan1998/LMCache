# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import gc
import math
import multiprocessing
import os
import socket
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.sampled_lookup import (
    find_last_sampled_hit,
    first_last_layer_keys,
)
from lmcache.v1.shared_cpu_cache import (
    SharedCPURequestLease,
    SharedChunkHandle,
    SharedHandleEnvelope,
    SharedSlabMapping,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUPrefixGetResult
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]
# (location, start indices, end indices, chunk-major layer keys)
LayerwiseRetrieveSegment = Tuple[
    str,
    List[int],
    List[int],
    List[List[CacheEngineKey]],
]


@dataclass
class LayerwiseStoreResult:
    """Cache objects produced by one completed layerwise store generator.

    Storage backends own the memory objects after a successful put. A consumer
    that needs lifetime independent of backend residency must acquire its own
    reference.
    """

    request_id: str
    kv_group: int = 0
    starts: List[int] = field(default_factory=list)
    ends: List[int] = field(default_factory=list)
    keys: List[List[CacheEngineKey]] = field(default_factory=list)
    memory_objs: List[List[MemoryObj]] = field(default_factory=list)
    tensors: List[List[torch.Tensor]] = field(default_factory=list)
    chunk_dev_ptrs: List[List[int]] = field(default_factory=list)
    chunk_ptrs: List[Optional[torch.Tensor]] = field(default_factory=list)

    def has_cache(self) -> bool:
        """Return whether the completed store produced reusable cache data."""
        return bool(self.starts and self.ends and self.keys and self.memory_objs)


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        self.dsa_two_groups = getattr(self.config, "dsa_two_groups", False)
        self.enable_shared_cpu_cache = bool(
            self._get_shared_config_value("enable_shared_cpu_cache", False)
        )
        self.save_indexer_only_first_rank = (
            self._resolve_save_indexer_only_first_rank(
                self.config,
                metadata,
                self.save_only_first_rank,
                self.dsa_two_groups,
            )
        )

        self.shared_cpu_cache_strict = bool(
            self._get_shared_config_value("shared_cpu_cache_strict", True)
        )
        self.shared_cpu_cache_name: Optional[str] = None
        self.shared_cpu_cache_slab_size: Optional[int] = None
        self.shared_cpu_cache_generation = 0
        self.shared_cpu_cache_mapping: Optional[SharedSlabMapping] = None
        self.shared_cpu_cache_passive_allocator = None
        self._shared_cpu_request_leases: dict[str, SharedCPURequestLease] = {}
        self._sampled_lookup_local_fallback_logged = False
        self._validate_shared_cpu_cache_contract()
        self._prepare_shared_cpu_cache_name()

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise

        # save_only_first_rank + layerwise: store_layer now has _is_passive()
        # guard (see store_layer). When save_only_first_rank is True, only the
        # first rank and lookup server workers initialize the storage_manager.
        # When False, all ranks initialize the storage_manager.
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_LATENT_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_LATENT_FMT
        self._report_shared_cpu_sparse_capacity_sanity()

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

    def _get_shared_config_value(self, key: str, default: Any = None) -> Any:
        get_extra = getattr(self.config, "get_extra_config_value", None)
        value = get_extra(key, None) if callable(get_extra) else None
        if value is not None:
            return value
        return getattr(self.config, key, default)

    @staticmethod
    def _legacy_indexer_policy_configured(config: LMCacheEngineConfig) -> bool:
        extra_config = getattr(config, "extra_config", None) or {}
        user_set_keys = getattr(config, "_user_set_keys", set())
        return (
            "save_indexer_only_first_rank" in extra_config
            or "save_indexer_only_first_rank" in user_set_keys
        )

    @classmethod
    def _resolve_save_indexer_only_first_rank(
        cls,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        save_only_first_rank: bool,
        dsa_two_groups: bool,
    ) -> bool:
        legacy_indexer_policy = config.get_extra_config_value(
            "save_indexer_only_first_rank",
            getattr(config, "save_indexer_only_first_rank", False),
        )
        if dsa_two_groups:
            if cls._legacy_indexer_policy_configured(config):
                logger.warning(
                    "save_indexer_only_first_rank is deprecated for "
                    "dsa_two_groups=true. Use save_only_first_rank=%s; it "
                    "controls both MLA latent and DSA index storage policy.",
                    save_only_first_rank,
                )
            return bool(save_only_first_rank)

        return bool(legacy_indexer_policy) and metadata.use_mla

    def _shared_cpu_contract_context(self) -> str:
        return (
            " shared_cpu_config={"
            f"enable_shared_cpu_cache={self.enable_shared_cpu_cache}, "
            f"enable_sparse_attention={self.config.enable_sparse_attention}, "
            f"save_only_first_rank={self.save_only_first_rank}, "
            f"use_layerwise={self.config.use_layerwise}, "
            f"TP/world_size={self.metadata.world_size}, "
            f"local_cpu={self.config.local_cpu}, "
            f"max_local_cpu_size={self.config.max_local_cpu_size}, "
            "shared_cpu_cache_name="
            f"{self._get_shared_config_value('shared_cpu_cache_name', None)!r}, "
            "shared_cpu_cache_size_gb="
            f"{self._get_shared_config_value('shared_cpu_cache_size_gb', None)!r}, "
            f"shared_cpu_cache_strict={self.shared_cpu_cache_strict}"
            "}"
        )

    def _validate_shared_cpu_cache_contract(self) -> None:
        if not self.metadata.use_mla:
            return

        tp_gt_one = self.metadata.world_size > 1
        context = self._shared_cpu_contract_context()
        needs_dense_layerwise_shared = (
            self.config.use_layerwise
            and self.save_only_first_rank
            and tp_gt_one
        )
        needs_sparse_shared = (
            self.config.enable_sparse_attention
            and self.save_only_first_rank
            and tp_gt_one
        )
        if needs_dense_layerwise_shared and not self.enable_shared_cpu_cache:
            raise ValueError(
                "use_layerwise=true with save_only_first_rank=true and "
                "TP/world_size>1 requires enable_shared_cpu_cache=true. "
                "Passive ranks cannot perform dense prefix layerwise load from "
                "rank0-only MLA/DSA chunks without shared CPU cache handles."
                + context
            )
        if needs_sparse_shared and not self.enable_shared_cpu_cache:
            raise ValueError(
                "enable_sparse_attention=true with save_only_first_rank=true "
                "and TP/world_size>1 requires enable_shared_cpu_cache=true. "
                "Passive decode ranks cannot read rank0-only MLA/DSA chunks "
                "without shared CPU cache handles."
                + context
            )

        if not self.enable_shared_cpu_cache:
            return

        if (
            self.metadata.world_size > 1
            and not callable(getattr(self, "broadcast_object_fn", None))
        ):
            raise ValueError(
                "enable_shared_cpu_cache with TP/world_size>1 requires a "
                "callable broadcast_object_fn for ordered shared CPU cache "
                "startup and per-layer handle envelopes."
                + context
            )

        if not self.config.local_cpu or self.config.max_local_cpu_size <= 0:
            raise ValueError(
                "enable_shared_cpu_cache requires local_cpu=true and "
                "max_local_cpu_size > 0 on rank0."
                + context
            )

        if (
            self.config.enable_sparse_attention
            and self.dsa_two_groups
            and self.shared_cpu_cache_strict
            and not bool(
                self._get_shared_config_value(
                    "shared_cpu_materialize_index_on_decode_cold",
                    True,
                )
            )
        ):
            raise ValueError(
                "Strict shared-cache sparse decode with dsa_two_groups=true "
                "must materialize DSA index on decode cold path. "
                "shared_cpu_materialize_index_on_decode_cold=false is only "
                "valid for a non-strict debug/proven-resident path."
                + context
            )

    def _prepare_shared_cpu_cache_name(self) -> None:
        if not self.enable_shared_cpu_cache:
            return

        explicit_name = self._get_shared_config_value("shared_cpu_cache_name", None)
        legacy_name = self.config.get_extra_config_value("shm_name", None)
        if explicit_name and legacy_name and explicit_name != legacy_name:
            raise ValueError(
                "shared_cpu_cache_name and shm_name must not differ in "
                "shared CPU cache mode."
            )

        resolved_name = explicit_name or legacy_name
        if not resolved_name:
            def shm_component(value: Any, default: str) -> str:
                raw = str(value or default)
                safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
                return safe[:48] or default

            instance = shm_component(self.config.lmcache_instance_id, "default")
            role = shm_component(self.metadata.role, "worker")
            node = shm_component(socket.gethostname(), "node")
            resolved_name = (
                f"/lmcache_shared_{instance}_{node}_{role}"
                f"_tp{self.metadata.world_size}_rank0_{os.getpid()}"
                f"_{int(time.time() * 1000)}"
            )
        elif not str(resolved_name).startswith("/"):
            resolved_name = "/" + str(resolved_name)

        self.shared_cpu_cache_name = str(resolved_name)
        if self.config.extra_config is None:
            self.config.extra_config = {}
        self.config.extra_config["shared_cpu_cache_resolved_name"] = (
            self.shared_cpu_cache_name
        )
        if self.metadata.is_first_rank():
            size_gb = self._get_shared_config_value(
                "shared_cpu_cache_size_gb",
                None,
            )
            if size_gb is not None:
                self.config.max_local_cpu_size = float(size_gb)
            self.config.extra_config["shm_name"] = self.shared_cpu_cache_name

    def _effective_shared_cpu_cache_size_bytes(self) -> int:
        size_gb = self._get_shared_config_value(
            "shared_cpu_cache_size_gb",
            None,
        )
        if size_gb is None:
            size_gb = self._get_shared_config_value("max_local_cpu_size", 0)
        return int(float(size_gb) * 1024**3)

    def _preflight_shared_cpu_shm_capacity(self) -> None:
        if (
            not self.enable_shared_cpu_cache
            or not self.metadata.is_first_rank()
            or os.name != "posix"
        ):
            return
        shm_dir = "/dev/shm"
        if not os.path.isdir(shm_dir):
            return
        try:
            stat = os.statvfs(shm_dir)
        except OSError as exc:
            logger.warning(
                "Shared CPU cache could not stat %s before shm allocation: %s",
                shm_dir,
                exc,
            )
            return
        required_bytes = self._effective_shared_cpu_cache_size_bytes()
        available_bytes = int(stat.f_bavail) * int(stat.f_frsize)
        if required_bytes > available_bytes:
            raise ValueError(
                "Shared CPU cache rank0 slab does not fit in /dev/shm before "
                "native shm allocation. This would otherwise crash the worker "
                "with SIGBUS during first_touch. "
                f"required_bytes={required_bytes}, "
                f"available_bytes={available_bytes}, "
                f"shm_name={self.shared_cpu_cache_name!r}. "
                "Increase /dev/shm or reduce max_local_cpu_size/"
                "shared_cpu_cache_size_gb."
            )

    def _shared_cpu_dtype_for_kv_group(self, kv_group: int) -> torch.dtype:
        dtypes = self.metadata.get_dtypes()
        if 0 <= kv_group < len(dtypes):
            return dtypes[kv_group]
        if kv_group != 0:
            if (
                len(dtypes) == 1
                and getattr(self, "dsa_two_groups", False)
            ):
                return dtypes[0]
            raise ValueError(
                "KV group dtype metadata is unavailable: "
                f"kv_group={kv_group}, num_dtypes={len(dtypes)}"
            )
        return self.metadata.kv_dtype

    def _metadata_shapes_dtypes_for_kv_group(
        self,
        *,
        kv_group: int,
        num_tokens: int,
    ):
        shapes = self.metadata.get_shapes(num_tokens)
        dtypes = self.metadata.get_dtypes()
        if len(shapes) <= 1:
            if kv_group != 0:
                raise ValueError(
                    "KV group shape metadata is unavailable: "
                    f"kv_group={kv_group}, num_groups={len(shapes)}"
                )
            return shapes, dtypes
        if kv_group < 0 or kv_group >= len(shapes):
            raise ValueError(
                "KV group is out of range for LMCache metadata shapes: "
                f"kv_group={kv_group}, num_groups={len(shapes)}"
            )
        if kv_group >= len(dtypes):
            raise ValueError(
                "KV group is out of range for LMCache metadata dtypes: "
                f"kv_group={kv_group}, num_dtypes={len(dtypes)}"
            )
        return shapes[kv_group], dtypes[kv_group]

    def _shape_numel_without_layer_dim(self, shape: torch.Size) -> int:
        dims = [int(dim) for dim in shape]
        if (
            len(dims) >= 3
            and self.num_layers > 0
            and dims[1] == self.num_layers
        ):
            dims = dims[:1] + dims[2:]
        elif (
            len(dims) >= 3
            and self.num_layers > 0
            and dims[0] == self.num_layers
        ):
            dims = dims[1:]
        numel = 1
        for dim in dims:
            numel *= dim
        return numel

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

    def _estimate_shared_cpu_bytes_per_layer(
        self,
        kv_group: int,
        num_tokens: int,
    ) -> int:
        shape: Optional[torch.Size] = None
        if kv_group != 0:
            shape = self._metadata_shape_for_kv_group(kv_group, num_tokens)
        get_shape = getattr(self.gpu_connector, "get_shape", None)
        if shape is None and callable(get_shape):
            try:
                shape = get_shape(num_tokens, kv_group=kv_group)
            except TypeError:
                if kv_group == 0:
                    shape = get_shape(num_tokens)
            except Exception as exc:
                logger.warning(
                    "Unable to query gpu_connector.get_shape for shared CPU "
                    "capacity estimate: kv_group=%s, error=%s",
                    kv_group,
                    exc,
                )

        if shape is None:
            shape = self._metadata_shape_for_kv_group(kv_group, num_tokens)
            if shape is None:
                raise ValueError(
                    "KV group shape metadata is unavailable for shared CPU "
                    "capacity estimate: "
                    f"kv_group={kv_group}, "
                    f"num_shapes={len(self.metadata.get_shapes(num_tokens))}"
                )

        dtype = self._shared_cpu_dtype_for_kv_group(kv_group)
        return self._shape_numel_without_layer_dim(shape) * dtype.itemsize

    def _expected_shared_cpu_chunk_metadata(
        self,
        *,
        kv_group: int,
        num_tokens: int,
    ) -> tuple[torch.Size, torch.dtype, MemoryFormat]:
        shape: Optional[torch.Size] = None
        if kv_group != 0:
            shape = self._metadata_shape_for_kv_group(kv_group, num_tokens)
        get_shape = getattr(self.gpu_connector, "get_shape", None)
        if shape is None and callable(get_shape):
            try:
                shape = torch.Size(get_shape(num_tokens, kv_group=kv_group))
            except TypeError:
                if kv_group == 0:
                    shape = torch.Size(get_shape(num_tokens))
        if shape is None:
            shape = self._metadata_shape_for_kv_group(kv_group, num_tokens)
            if shape is None:
                raise ValueError(
                    "KV group shape metadata is unavailable for shared CPU "
                    "chunk metadata: "
                    f"kv_group={kv_group}, "
                    f"num_shapes={len(self.metadata.get_shapes(num_tokens))}"
                )
        return (
            shape,
            self._shared_cpu_dtype_for_kv_group(kv_group),
            self._memory_format_for_kv_group(kv_group),
        )

    def _estimate_shared_cpu_chunk_bytes_per_layer(self, kv_group: int) -> int:
        return self._estimate_shared_cpu_bytes_per_layer(
            kv_group,
            int(self.config.chunk_size),
        )

    def _report_shared_cpu_sparse_capacity_sanity(self) -> None:
        if not (
            self.config.enable_sparse_attention
            and self.enable_shared_cpu_cache
            and self.save_only_first_rank
            and self.metadata.world_size > 1
            and self.metadata.is_first_rank()
        ):
            return

        max_model_len = self._get_shared_config_value(
            "vllm_max_model_len",
            getattr(self.metadata, "max_model_len", None),
        )
        if max_model_len is None:
            logger.warning(
                "Shared CPU sparse capacity sanity skipped because vLLM "
                "max_model_len is unavailable."
            )
            return

        max_num_seqs = self._get_shared_config_value("vllm_max_num_seqs", None)
        chunk_size = int(self.config.chunk_size)
        max_model_len = int(max_model_len)
        chunks_per_seq = int(math.ceil(max_model_len / chunk_size))
        kv_groups = [0]
        materialize_index = bool(
            self._get_shared_config_value(
                "shared_cpu_materialize_index_on_decode_cold",
                True,
            )
        )
        if self.dsa_two_groups and materialize_index:
            kv_groups.append(1)

        bytes_per_chunk_all_layers = sum(
            self._estimate_shared_cpu_chunk_bytes_per_layer(kv_group)
            * self.num_layers
            for kv_group in kv_groups
        )
        one_request_bytes = bytes_per_chunk_all_layers * chunks_per_seq
        slab_bytes = self._effective_shared_cpu_cache_size_bytes()
        estimate: dict[str, Any] = {
            "slab_bytes": slab_bytes,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "chunk_size": chunk_size,
            "chunks_per_seq": chunks_per_seq,
            "kv_groups": kv_groups,
            "bytes_per_chunk_all_layers": bytes_per_chunk_all_layers,
            "one_max_request_bytes": one_request_bytes,
        }
        if max_num_seqs is not None:
            estimate["configured_worst_case_bytes"] = (
                one_request_bytes * int(max_num_seqs)
            )
        if self.config.extra_config is None:
            self.config.extra_config = {}
        self.config.extra_config[
            "shared_cpu_sparse_startup_capacity_estimate"
        ] = estimate

        if one_request_bytes > slab_bytes:
            raise ValueError(
                "Shared CPU sparse capacity preflight failed: one maximum "
                "request cannot fit in the rank0 shared CPU slab. "
                f"estimate={estimate}. Increase max_local_cpu_size or "
                "shared_cpu_cache_size_gb, or reduce max_model_len."
            )
        configured_worst_case = estimate.get("configured_worst_case_bytes")
        if configured_worst_case is not None and configured_worst_case > slab_bytes:
            logger.warning(
                "Shared CPU sparse configured worst-case active set exceeds "
                "the rank0 slab: estimate=%s. Runtime admission uses actual "
                "active prompt lengths, but this configuration can still hit "
                "strict capacity failures at decode.",
                estimate,
            )
        else:
            logger.info(
                "Shared CPU sparse capacity sanity passed: estimate=%s",
                estimate,
            )

    def _shared_cpu_cache_startup_envelope(
        self,
        status: str,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "shm_name": self.shared_cpu_cache_name,
            "slab_size": self.shared_cpu_cache_slab_size,
            "generation": self.shared_cpu_cache_generation,
            "rank": self.metadata.worker_id,
        }

    def _post_init_shared_cpu_cache(self) -> None:
        if not self.enable_shared_cpu_cache or self.metadata.world_size <= 1:
            return

        if self.metadata.is_first_rank():
            try:
                if self.storage_manager is None:
                    raise ValueError(
                        "Shared CPU cache rank0 preflight requires StorageManager."
                    )
                local_cpu_backend = getattr(
                    self.storage_manager,
                    "local_cpu_backend",
                    None,
                )
                if local_cpu_backend is None:
                    raise ValueError(
                        "Shared CPU cache rank0 preflight requires "
                        "LocalCPUBackend."
                    )
                allocator = getattr(local_cpu_backend, "memory_allocator", None)
                allocator_buffer = getattr(allocator, "buffer", None)
                allocator_shm_name = getattr(allocator, "shm_name", None)
                if allocator_buffer is None or allocator_shm_name is None:
                    raise ValueError(
                        "Shared CPU cache rank0 LocalCPUBackend must be "
                        "shm-backed and expose buffer/shm_name."
                    )
                self.shared_cpu_cache_name = allocator_shm_name
                self.shared_cpu_cache_slab_size = int(allocator_buffer.numel())
                self.shared_cpu_cache_generation = int(time.time() * 1000)
                self.shared_cpu_cache_mapping = (
                    SharedSlabMapping.from_rank0_allocator(
                        shm_name=self.shared_cpu_cache_name,
                        allocator_tensor=allocator_buffer,
                        generation=self.shared_cpu_cache_generation,
                    )
                )
                if self.shared_cpu_cache_strict:
                    self.shared_cpu_cache_mapping.preflight_device_ptr()
                self.broadcast_object_fn(
                    self._shared_cpu_cache_startup_envelope("ok"),
                    self.metadata.first_rank,
                )
            except Exception as exc:
                mapping = getattr(self, "shared_cpu_cache_mapping", None)
                if mapping is not None:
                    try:
                        mapping.close()
                        self.shared_cpu_cache_mapping = None
                    except Exception:
                        logger.exception(
                            "Failed to clean up rank0 shared CPU cache mapping "
                            "after startup preflight failure"
                        )
                try:
                    self.broadcast_object_fn(
                        self._shared_cpu_cache_startup_envelope(
                            "error",
                            str(exc),
                        ),
                        self.metadata.first_rank,
                    )
                except Exception:
                    logger.exception(
                        "Failed to broadcast shared CPU cache rank0 startup "
                        "error envelope"
                    )
                raise
            logger.info(
                "Shared CPU cache rank0 preflight ok: shm_name=%s, "
                "slab_size=%s, generation=%s",
                self.shared_cpu_cache_name,
                self.shared_cpu_cache_slab_size,
                self.shared_cpu_cache_generation,
            )
            return

        envelope = self.broadcast_object_fn(None, self.metadata.first_rank)
        if not isinstance(envelope, dict):
            raise ValueError(
                "Shared CPU cache passive preflight expected dict envelope, "
                f"got {type(envelope)!r}"
            )
        if envelope.get("status") != "ok":
            raise ValueError(
                "Shared CPU cache passive preflight failed from rank0: "
                f"{envelope.get('message')}"
            )
        shm_name = envelope.get("shm_name")
        slab_size = envelope.get("slab_size")
        generation = envelope.get("generation")
        if not shm_name or not slab_size or generation is None:
            raise ValueError(
                "Shared CPU cache passive preflight envelope missing shm_name, "
                f"slab_size, or generation: {envelope}"
            )
        self.shared_cpu_cache_name = str(shm_name)
        self.shared_cpu_cache_slab_size = int(slab_size)
        self.shared_cpu_cache_generation = int(generation)
        explicit_writable = self._get_shared_config_value(
            "shared_cpu_cache_passive_writable",
            None,
        )
        writable_attempts = (
            [bool(explicit_writable)]
            if explicit_writable is not None
            else [False, True]
        )
        for writable in writable_attempts:
            mapping = None
            try:
                mapping = SharedSlabMapping.attach(
                    shm_name=self.shared_cpu_cache_name,
                    size=self.shared_cpu_cache_slab_size,
                    generation=self.shared_cpu_cache_generation,
                    writable=writable,
                )
                if self.shared_cpu_cache_strict:
                    mapping.preflight_device_ptr()
                self.shared_cpu_cache_mapping = mapping
                if self.config.extra_config is None:
                    self.config.extra_config = {}
                self.config.extra_config[
                    "shared_cpu_cache_passive_writable_resolved"
                ] = writable
                break
            except Exception as exc:
                if mapping is not None:
                    try:
                        mapping.close()
                    except Exception:
                        logger.exception(
                            "Failed to detach shared CPU cache mapping after "
                            "passive preflight failure"
                        )
                if explicit_writable is None and not writable:
                    logger.warning(
                        "Shared CPU cache passive read-only attach/preflight "
                        "failed for shm_name=%s; retrying read-write: %s",
                        self.shared_cpu_cache_name,
                        exc,
                    )
                    continue
                raise ValueError(
                    "Shared CPU cache passive attach/preflight failed: "
                    f"shm_name={self.shared_cpu_cache_name}, "
                    f"slab_size={self.shared_cpu_cache_slab_size}, "
                    f"generation={self.shared_cpu_cache_generation}, "
                    f"writable={writable}"
                ) from exc
        if self.shared_cpu_cache_mapping is None:
            raise ValueError(
                "Shared CPU cache passive attach/preflight did not produce "
                "a usable mapping."
            )
        self.shared_cpu_cache_passive_allocator = (
            self.shared_cpu_cache_mapping.passive_allocator()
        )
        logger.info(
            "Shared CPU cache passive preflight ok: rank=%s, shm_name=%s, "
            "slab_size=%s, generation=%s",
            self.metadata.worker_id,
            self.shared_cpu_cache_name,
            self.shared_cpu_cache_slab_size,
            self.shared_cpu_cache_generation,
        )

    def _should_use_shared_layerwise_retrieve(self, kv_group: int) -> bool:
        if not getattr(self, "enable_shared_cpu_cache", False):
            return False
        if not getattr(self, "use_layerwise", False):
            return False
        metadata = getattr(self, "metadata", None)
        if not getattr(metadata, "use_mla", False):
            return False
        if getattr(metadata, "world_size", 1) <= 1:
            return False
        if kv_group == 1:
            return bool(getattr(self, "save_indexer_only_first_rank", False))
        return bool(getattr(self, "save_only_first_rank", False))

    def _shared_layerwise_error_envelope(
        self,
        *,
        req_id: str,
        phase: str,
        request_ordinal: int,
        layer_id: int,
        kv_group: int,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> SharedHandleEnvelope:
        return SharedHandleEnvelope(
            request_id=req_id,
            phase=phase,
            request_ordinal=request_ordinal,
            layer_id=layer_id,
            kv_group=kv_group,
            status="error",
            generation=self.shared_cpu_cache_generation,
            handles=[],
            message=message,
            error_details=details,
        )

    def _broadcast_shared_envelope(self, envelope: SharedHandleEnvelope) -> None:
        self.broadcast_object_fn(envelope.to_dict(), self.metadata.first_rank)

    def _receive_shared_envelope(self) -> SharedHandleEnvelope:
        raw = self.broadcast_object_fn(None, self.metadata.first_rank)
        if not isinstance(raw, dict):
            raise ValueError(
                "Shared CPU cache expected dict envelope from rank0, "
                f"got {type(raw)!r}"
            )
        try:
            return SharedHandleEnvelope.from_dict(raw)
        except Exception as exc:
            raise ValueError(
                "Shared CPU cache received corrupt envelope before view "
                f"creation: error={exc}, raw={raw!r}"
            ) from exc

    def _validate_shared_layerwise_envelope(
        self,
        envelope: SharedHandleEnvelope,
        *,
        req_id: str,
        phase: str,
        request_ordinal: int,
        layer_id: int,
        kv_group: int,
    ) -> None:
        failures = []
        if envelope.request_id != req_id:
            failures.append(
                f"request_id={envelope.request_id!r}, expected={req_id!r}"
            )
        if envelope.phase != phase:
            failures.append(f"phase={envelope.phase!r}, expected={phase!r}")
        if envelope.request_ordinal != int(request_ordinal):
            failures.append(
                f"request_ordinal={envelope.request_ordinal}, "
                f"expected={int(request_ordinal)}"
            )
        if envelope.layer_id != layer_id:
            failures.append(
                f"layer_id={envelope.layer_id}, expected={layer_id}"
            )
        if envelope.kv_group != kv_group:
            failures.append(
                f"kv_group={envelope.kv_group}, expected={kv_group}"
            )
        if envelope.generation != self.shared_cpu_cache_generation:
            failures.append(
                f"generation={envelope.generation}, "
                f"expected={self.shared_cpu_cache_generation}"
            )
        if failures:
            raise ValueError(
                "Invalid shared CPU cache layerwise envelope: "
                + "; ".join(failures)
            )
        if envelope.status == "error":
            raise ValueError(
                "Shared CPU cache rank0 error envelope: "
                f"{envelope.message}; details={envelope.error_details}"
            )
        if envelope.status == "miss" and self.shared_cpu_cache_strict:
            raise ValueError(
                "Shared CPU cache strict mode received miss envelope before "
                "view creation: "
                f"request_id={envelope.request_id}, phase={envelope.phase}, "
                f"layer_id={envelope.layer_id}, kv_group={envelope.kv_group}, "
                f"message={envelope.message}, details={envelope.error_details}"
            )
        if envelope.status not in ("ok", "miss", "skipped"):
            raise ValueError(
                "Shared CPU cache envelope has unsupported status "
                f"{envelope.status!r}"
            )
        if envelope.status == "ok" and not envelope.handles:
            raise ValueError(
                "Shared CPU cache ok envelope must carry at least one handle: "
                f"request_id={envelope.request_id}, phase={envelope.phase}, "
                f"layer_id={envelope.layer_id}, kv_group={envelope.kv_group}"
            )
        if envelope.status in ("miss", "skipped") and envelope.handles:
            raise ValueError(
                "Shared CPU cache non-present envelope must not carry handles: "
                f"status={envelope.status!r}, request_id={envelope.request_id}, "
                f"phase={envelope.phase}, layer_id={envelope.layer_id}, "
                f"kv_group={envelope.kv_group}, handles={len(envelope.handles)}"
            )

    def _make_shared_handles_for_layer(
        self,
        *,
        req_id: str,
        phase: str,
        keys_layer: list[CacheEngineKey],
        mem_objs_layer: list[MemoryObj],
        layer_id: int,
        kv_group: int,
        chunk_index_base: int = 0,
    ) -> list[SharedChunkHandle]:
        if self.shared_cpu_cache_name is None:
            raise ValueError("Shared CPU cache name is not initialized")
        if len(keys_layer) != len(mem_objs_layer):
            raise ValueError(
                "Shared CPU cache refuses to publish partial layer handles: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, keys={len(keys_layer)}, "
                f"memory_objs={len(mem_objs_layer)}"
            )
        handles: list[SharedChunkHandle] = []
        for chunk_offset, (key, mem_obj) in enumerate(
            zip(keys_layer, mem_objs_layer, strict=True)
        ):
            chunk_index = chunk_index_base + chunk_offset
            self._validate_rank0_shared_mem_obj(
                mem_obj,
                req_id=req_id,
                phase=phase,
                layer_id=layer_id,
                kv_group=kv_group,
                chunk_index=chunk_index,
            )
            handles.append(
                SharedChunkHandle.from_memory_obj(
                    request_id=req_id,
                    phase=phase,
                    key=key,
                    layer_id=layer_id,
                    kv_group=kv_group,
                    chunk_index=chunk_index,
                    shm_name=self.shared_cpu_cache_name,
                    memory_obj=mem_obj,
                    generation=self.shared_cpu_cache_generation,
                    producer_rank=self.metadata.worker_id,
                )
            )
        return handles

    def _layerwise_chunk_location_if_fully_stored(
        self,
        keys_multi_layer: list[CacheEngineKey],
        *,
        req_id: str,
        kv_group: int,
        start: int,
        end: int,
        repair_partial: bool = False,
    ) -> Optional[str]:
        """Return the backend location only when all layers already exist."""
        assert self.storage_manager is not None
        hit_layers, block_mapping = self.storage_manager.batched_contains(
            keys_multi_layer,
            self.retrieve_locations,
        )

        if hit_layers == len(keys_multi_layer) and len(block_mapping) == 1:
            return next(iter(block_mapping.keys()))
        if hit_layers == 0 and repair_partial:
            local_removed = self._remove_local_cpu_nonprefix_layerwise_hits(
                keys_multi_layer,
                req_id=req_id,
                kv_group=kv_group,
                start=start,
                end=end,
            )
            if local_removed > 0:
                return None
        if hit_layers > 0:
            present_layers = list(range(hit_layers))
            missing_layers = list(range(hit_layers, len(keys_multi_layer)))
            action = "removing existing layers before re-store" if repair_partial else (
                "treating chunk as a miss"
            )
            logger.warning(
                "Layerwise cache found incomplete or mixed-location chunk; "
                "%s: "
                "req_id=%s, kv_group=%s, start=%d, end=%d, "
                "present_layers=%s, missing_layers=%s, locations=%s",
                action,
                req_id,
                kv_group,
                start,
                end,
                present_layers,
                missing_layers,
                list(block_mapping.keys()),
            )
            if repair_partial:
                removed = self.storage_manager.batched_remove(
                    keys_multi_layer,
                    locations=self.retrieve_locations,
                )
                if removed < hit_layers:
                    raise ValueError(
                        "Layerwise cache found partial or mixed-location "
                        "chunk but could not remove all existing layers "
                        "before re-store. Clear the stale backend cache or "
                        "use a fresh lmcache.tag.* namespace: "
                        f"req_id={req_id}, kv_group={kv_group}, "
                        f"start={start}, end={end}, hit_layers={hit_layers}, "
                        f"removed_layers={removed}, "
                        f"locations={list(block_mapping.keys())}"
                    )
        return None

    def _remove_local_cpu_nonprefix_layerwise_hits(
        self,
        keys_multi_layer: list[CacheEngineKey],
        *,
        req_id: str,
        kv_group: int,
        start: int,
        end: int,
    ) -> int:
        """Remove stale LocalCPU layer keys when layer0 is absent.

        StorageManager.batched_contains intentionally reports prefix hits only.
        That is correct for retrieve, but store must not leave stale later-layer
        hot-cache entries because LocalCPU put skips keys that already exist.
        """
        assert self.storage_manager is not None
        if (
            self.retrieve_locations is not None
            and "LocalCPUBackend" not in self.retrieve_locations
        ):
            return 0

        storage_backends = getattr(self.storage_manager, "storage_backends", {})
        local_cpu_backend = storage_backends.get("LocalCPUBackend")
        if local_cpu_backend is None:
            return 0

        present_layers = []
        for layer_id, key in enumerate(keys_multi_layer):
            if local_cpu_backend.contains(key):
                present_layers.append(layer_id)
        if not present_layers:
            return 0

        logger.warning(
            "Layerwise cache found non-prefix LocalCPU stale layers; "
            "removing before re-store: req_id=%s, kv_group=%s, "
            "start=%d, end=%d, present_layers=%s",
            req_id,
            kv_group,
            start,
            end,
            present_layers,
        )
        removed = local_cpu_backend.batched_remove(keys_multi_layer)
        if removed < len(present_layers):
            raise ValueError(
                "Layerwise cache found non-prefix LocalCPU stale layers but "
                "could not remove all of them before re-store. Clear the "
                "stale local CPU cache or use a fresh lmcache.tag.* namespace: "
                f"req_id={req_id}, kv_group={kv_group}, start={start}, "
                f"end={end}, present_layers={present_layers}, "
                f"removed_layers={removed}"
            )
        return removed

    def _layerwise_chunk_fully_stored(
        self,
        keys_multi_layer: list[CacheEngineKey],
        *,
        req_id: str,
        kv_group: int,
        start: int,
        end: int,
    ) -> bool:
        return (
            self._layerwise_chunk_location_if_fully_stored(
                keys_multi_layer,
                req_id=req_id,
                kv_group=kv_group,
                start=start,
                end=end,
                repair_partial=True,
            )
            is not None
        )

    def _layerwise_lookup_kv_groups(self) -> list[int]:
        if (
            getattr(self, "use_layerwise", False)
            and getattr(self.config, "dsa_two_groups", False)
        ):
            return [0, 1]
        return [0]

    def _sampled_scheduler_lookup_requested(self) -> bool:
        config = getattr(self, "config", None)
        return bool(
            getattr(config, "experimental_sampled_layerwise_lookup", False)
            and getattr(self, "use_layerwise", False)
        )

    def _use_sampled_scheduler_lookup(self) -> bool:
        if not self._sampled_scheduler_lookup_requested():
            return False

        assert self.storage_manager is not None
        has_remote_backend = any(
            backend_name == "RemoteBackend"
            for backend_name, _ in self.storage_manager.get_active_storage_backends(
                search_range=["RemoteBackend"]
            )
        )
        if has_remote_backend:
            return True

        if not getattr(self, "_sampled_lookup_local_fallback_logged", False):
            logger.warning(
                "experimental_sampled_layerwise_lookup is enabled without an "
                "active RemoteBackend; falling back to regular layerwise "
                "LocalCPU lookup."
            )
            self._sampled_lookup_local_fallback_logged = True
        return False

    def _sampled_scheduler_keys(
        self,
        base_key: CacheEngineKey,
        *,
        request_configs: Optional[dict],
    ) -> list[CacheEngineKey]:
        group_keys = [
            self._lookup_key_for_kv_group(
                base_key,
                kv_group=kv_group,
                request_configs=request_configs,
            )
            for kv_group in self._layerwise_lookup_kv_groups()
        ]
        return first_last_layer_keys(group_keys, self.num_layers)

    def _sampled_scheduler_lookup(
        self,
        chunks: list[tuple[int, CacheEngineKey]],
        *,
        lookup_id: Optional[str],
        pin: bool,
        request_configs: Optional[dict],
    ) -> int:
        if not chunks:
            return 0
        assert self.storage_manager is not None

        def remote_exists(keys: list[CacheEngineKey]) -> bool:
            if not keys:
                return False
            hits, _ = self.storage_manager.batched_contains(
                keys, ["RemoteBackend"], False
            )
            return hits == len(keys)

        def chunk_exists(index: int) -> bool:
            return remote_exists(
                self._sampled_scheduler_keys(
                    chunks[index][1],
                    request_configs=request_configs,
                )
            )

        winner_index = find_last_sampled_hit(
            len(chunks),
            chunk_exists,
        )
        if winner_index is None:
            return 0

        if pin:
            assert lookup_id is not None, "lookup_id is required when pin is True"
            first_keys = self._sampled_scheduler_keys(
                chunks[0][1],
                request_configs=request_configs,
            )
            winner_keys = self._sampled_scheduler_keys(
                chunks[winner_index][1],
                request_configs=request_configs,
            )
            pin_keys = list(dict.fromkeys([*first_keys, *winner_keys]))
            hits, block_mapping = self.storage_manager.batched_contains(
                pin_keys,
                ["RemoteBackend"],
                True,
            )
            if hits != len(pin_keys):
                for location, location_keys in block_mapping.items():
                    self.storage_manager.batched_unpin(
                        location_keys,
                        [location],
                    )
                return 0
            for location, location_keys in block_mapping.items():
                self.lookup_pins[lookup_id][location].extend(location_keys)

        return chunks[winner_index][0]

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
        )

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
            address_manager = self._shared_cpu_allocator_address_manager(
                root_allocator
            )
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
                lease.active
                for lease in self._shared_cpu_request_leases.values()
            )
            return snapshot
        except Exception as exc:
            return {"capacity_snapshot_error": str(exc)}

    def _get_shared_cpu_request_lease(
        self,
        req_id: str,
    ) -> tuple[SharedCPURequestLease, bool]:
        lease = self._shared_cpu_request_leases.get(req_id)
        if (
            lease is not None
            and lease.generation != self.shared_cpu_cache_generation
        ):
            self._shared_cpu_request_leases.pop(req_id, None)
            lease.close()
            lease = None
        if lease is not None:
            return lease, False
        return (
            SharedCPURequestLease(
                request_id=req_id,
                generation=self.shared_cpu_cache_generation,
                is_rank0=self.metadata.is_first_rank(),
            ),
            True,
        )

    def register_shared_cpu_sparse_request(
        self,
        req_id: str,
        *,
        owned_groups: Optional[dict[int, list[list[MemoryObj]]]] = None,
        append_from: Optional[dict[int, int]] = None,
    ) -> None:
        """Adopt complete groups or append-only suffixes for a live request.

        Args:
            req_id: Request whose shared objects remain live.
            owned_groups: Complete per-layer object lists to adopt.
            append_from: Existing chunk count per group for suffix adoption.

        Raises:
            ValueError: If suffix adoption is not append-aligned.
        """
        if not req_id:
            return
        lease, created = self._get_shared_cpu_request_lease(req_id)
        try:
            if append_from is not None and not created and owned_groups:
                lease.append_groups(owned_groups, append_from)
            elif owned_groups:
                lease.replace_groups(owned_groups, retain=False)
            lease.active = True
            self._shared_cpu_request_leases[req_id] = lease
        except Exception:
            if created:
                lease.close()
            raise

    def retain_shared_cpu_store_seed(
        self,
        req_id: str,
        groups: dict[int, list[list[MemoryObj]]],
    ) -> None:
        if not req_id or not groups or not self.metadata.is_first_rank():
            return
        lease, created = self._get_shared_cpu_request_lease(req_id)
        try:
            lease.replace_groups(groups, retain=True)
            self._shared_cpu_request_leases[req_id] = lease
        except Exception:
            if created:
                lease.close()
            raise

    def shared_cpu_rank0_request_object_ids(
        self,
        req_id: Optional[str],
        kv_group: int,
    ) -> set[int]:
        if not req_id:
            return set()
        lease = self._shared_cpu_request_leases.get(req_id)
        if (
            lease is None
            or lease.generation != self.shared_cpu_cache_generation
            or not lease.is_rank0
        ):
            return set()
        return lease.object_ids(kv_group)

    def release_shared_cpu_unowned_objects(
        self,
        req_id: Optional[str],
        groups: dict[int, list[list[MemoryObj]]],
    ) -> None:
        """Release retrieved objects not adopted by the request lease."""

        lease = self._shared_cpu_request_leases.get(req_id or "")
        owned_ids = (
            lease.object_ids()
            if lease is not None
            and lease.generation == self.shared_cpu_cache_generation
            else set()
        )
        unowned_objects = [
            memory_obj
            for layers in groups.values()
            for layer in layers
            for memory_obj in layer
            if id(memory_obj) not in owned_ids
        ]
        if not unowned_objects:
            return
        SharedCPURequestLease(
            request_id=req_id or "unowned",
            generation=self.shared_cpu_cache_generation,
            is_rank0=self.metadata.is_first_rank(),
            groups={0: [unowned_objects]},
        ).close()

    def release_shared_cpu_sparse_request(self, req_id: Optional[str]) -> None:
        if not req_id:
            return
        lease = self._shared_cpu_request_leases.pop(req_id, None)
        if lease is not None:
            lease.close()

    def _release_all_shared_cpu_request_leases(self) -> None:
        leases = list(self._shared_cpu_request_leases.values())
        self._shared_cpu_request_leases.clear()
        for lease in leases:
            lease.close()

    def _shared_cpu_estimated_physical_chunk_bytes(
        self,
        kv_group: int,
        num_tokens: Optional[int] = None,
    ) -> int:
        logical_bytes = self._estimate_shared_cpu_bytes_per_layer(
            kv_group,
            int(num_tokens or self.config.chunk_size),
        )
        try:
            allocator = getattr(
                self._shared_local_cpu_backend(),
                "memory_allocator",
                None,
            )
            align_bytes = int(getattr(allocator, "align_bytes", 4096) or 4096)
        except Exception:
            align_bytes = 4096
        return ((logical_bytes + align_bytes - 1) // align_bytes) * align_bytes

    def _shared_cpu_mem_obj_physical_size(self, mem_obj: MemoryObj) -> int:
        metadata = getattr(mem_obj, "metadata", None)
        if metadata is not None and getattr(metadata, "phy_size", None) is not None:
            return int(metadata.phy_size)
        return int(mem_obj.get_physical_size())

    def _shared_cpu_runtime_capacity_details(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        chunk_locations_layer_major: list[list[str]],
        token_count: int = 0,
        chunk_token_lengths: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        local_cpu_backend = self._shared_local_cpu_backend()
        required_local_keys = {
            key
            for layer_keys, layer_locations in zip(
                keys_layer_major,
                chunk_locations_layer_major,
                strict=False,
            )
            for key, location in zip(layer_keys, layer_locations, strict=False)
            if location == "LocalCPUBackend"
        }
        hot_cache = getattr(local_cpu_backend, "hot_cache", {})
        cpu_lock = getattr(local_cpu_backend, "cpu_lock", None)
        lock_cm = cpu_lock if cpu_lock is not None else None
        rank0_shared_hot_keys = set()
        non_shm_hot_keys = set()
        if lock_cm is None:
            required_local_items = [
                (key, hot_cache.get(key)) for key in required_local_keys
            ]
        else:
            with lock_cm:
                required_local_items = [
                    (key, hot_cache.get(key)) for key in required_local_keys
                ]
        for key, mem_obj in required_local_items:
            if mem_obj is not None and self._is_rank0_shared_mem_obj(mem_obj):
                rank0_shared_hot_keys.add(key)
            else:
                # A LocalCPU hit that is not backed by the resolved shm slab
                # still needs rank0 rematerialization before handle publication.
                non_shm_hot_keys.add(key)

        missing_chunk_count = 0
        required_bytes = 0
        default_chunk_bytes = self._shared_cpu_estimated_physical_chunk_bytes(
            kv_group
        )
        for layer_keys, layer_locations in zip(
            keys_layer_major,
            chunk_locations_layer_major,
            strict=False,
        ):
            for chunk_index, (key, location) in enumerate(
                zip(layer_keys, layer_locations, strict=False)
            ):
                if not location:
                    continue
                if (
                    location == "LocalCPUBackend"
                    and key in rank0_shared_hot_keys
                ):
                    continue
                missing_chunk_count += 1
                if (
                    chunk_token_lengths is not None
                    and chunk_index < len(chunk_token_lengths)
                    and chunk_token_lengths[chunk_index] > 0
                ):
                    required_bytes += self._shared_cpu_estimated_physical_chunk_bytes(
                        kv_group,
                        num_tokens=chunk_token_lengths[chunk_index],
                    )
                else:
                    required_bytes += default_chunk_bytes

        allocator = getattr(local_cpu_backend, "memory_allocator", None)
        root_allocator = getattr(allocator, "_allocator", allocator)
        buffer = getattr(root_allocator, "buffer", None)
        slab_size = int(buffer.numel()) if buffer is not None else None
        free_bytes = 0
        address_manager = self._shared_cpu_allocator_address_manager(root_allocator)
        if address_manager is not None:
            get_free_size = getattr(address_manager, "get_free_size", None)
            if callable(get_free_size):
                free_bytes = int(get_free_size())

        evictable_bytes = 0
        pinned_bytes = 0
        protected_hot_bytes = 0
        if lock_cm is None:
            cache_items = list(hot_cache.items())
        else:
            with lock_cm:
                cache_items = list(hot_cache.items())
        for key, mem_obj in cache_items:
            is_shared_hot_obj = self._is_rank0_shared_mem_obj(mem_obj)
            physical_size = self._shared_cpu_mem_obj_physical_size(mem_obj)
            if is_shared_hot_obj and getattr(mem_obj, "is_pinned", False):
                pinned_bytes += physical_size
            if key in rank0_shared_hot_keys:
                protected_hot_bytes += physical_size
                continue
            if is_shared_hot_obj and getattr(mem_obj, "can_evict", False):
                evictable_bytes += physical_size

        available_after_eviction = free_bytes + evictable_bytes
        active_sparse_requests = {
            req_id
            for req_id, lease in self._shared_cpu_request_leases.items()
            if lease.active
        }
        if req_id:
            active_sparse_requests.add(req_id)
        details = {
            "request_id": req_id,
            "phase": phase,
            "kv_group": kv_group,
            "token_count": int(token_count or 0),
            "chunk_count": sum(len(layer) for layer in keys_layer_major),
            "missing_chunk_count": missing_chunk_count,
            "hot_chunk_count": len(rank0_shared_hot_keys),
            "non_shm_hot_chunk_count": len(non_shm_hot_keys),
            "required_bytes": required_bytes,
            "per_chunk_physical_bytes_estimate": default_chunk_bytes,
            "available_after_eviction": available_after_eviction,
            "free_bytes": free_bytes,
            "evictable_bytes": evictable_bytes,
            "pinned_bytes": pinned_bytes,
            "protected_hot_bytes": protected_hot_bytes,
            "active_sparse_requests": len(active_sparse_requests),
            "slab_size": slab_size,
            "fits": required_bytes <= available_after_eviction,
        }
        return details

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
    ) -> None:
        local_cpu_backend = self._shared_local_cpu_backend()
        allocator = getattr(local_cpu_backend, "memory_allocator", None)
        if allocator is None:
            raise ValueError("LocalCPUBackend has no memory allocator")
        if getattr(allocator, "shm_name", None) != self.shared_cpu_cache_name:
            raise ValueError(
                "Shared CPU cache object is not backed by the resolved shm slab: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"allocator_shm={getattr(allocator, 'shm_name', None)!r}, "
                f"expected={self.shared_cpu_cache_name!r}"
            )
        expected_parent = getattr(allocator, "pin_allocator", None)
        if mem_obj.parent() is not expected_parent:
            raise ValueError(
                "Shared CPU cache object does not belong to rank0's shm-backed "
                "LocalCPUBackend allocator: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        if mem_obj.get_dtype() is None:
            raise ValueError(
                "Shared CPU cache cannot publish dtype-less MemoryObj: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        expected_dtype = self._shared_cpu_dtype_for_kv_group(kv_group)
        if mem_obj.get_dtype() != expected_dtype:
            raise ValueError(
                "Shared CPU cache object dtype does not match KV group: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"dtype={mem_obj.get_dtype()}, expected={expected_dtype}"
            )
        expected_fmt = self._memory_format_for_kv_group(kv_group)
        if mem_obj.get_memory_format() == MemoryFormat.UNDEFINED:
            raise ValueError(
                "Shared CPU cache cannot publish MemoryObj with undefined format: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )
        if mem_obj.get_memory_format() != expected_fmt:
            raise ValueError(
                "Shared CPU cache object format does not match KV group: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"fmt={mem_obj.get_memory_format()}, expected={expected_fmt}"
            )
        slab_size = int(allocator.buffer.numel())
        offset = int(mem_obj.metadata.address)
        physical_size = int(mem_obj.metadata.phy_size)
        logical_size = int(mem_obj.get_size())
        if (
            offset < 0
            or physical_size <= 0
            or logical_size <= 0
            or logical_size > physical_size
            or offset + physical_size > slab_size
        ):
            raise ValueError(
                "Shared CPU cache object has invalid slab bounds: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}, "
                f"offset={offset}, logical_size={logical_size}, "
                f"physical_size={physical_size}, slab_size={slab_size}"
            )
        if not mem_obj.is_pinned:
            raise ValueError(
                "Shared CPU cache object must be pinned before handle publication: "
                f"request_id={req_id}, phase={phase}, layer_id={layer_id}, "
                f"kv_group={kv_group}, chunk_index={chunk_index}"
            )

    def _find_shared_rank0_chunk_location(
        self,
        key: CacheEngineKey,
    ) -> Optional[str]:
        assert self.storage_manager is not None
        local_cpu_backend = self._shared_local_cpu_backend()
        if local_cpu_backend.contains(key):
            return "LocalCPUBackend"
        return self.storage_manager.contains(key, self.retrieve_locations)

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
                            materialized_obj = (
                                self._materialize_shared_rank0_copy(
                                    key=key,
                                    src_obj=hot_obj,
                                    req_id=req_id,
                                    phase=phase,
                                    layer_id=layer_id,
                                    kv_group=kv_group,
                                    chunk_index=chunk_index,
                                )
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

                    hot_obj = local_cpu_backend.get_blocking(key)
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

    def _resolve_shared_rank0_remote_layers_windowed(
        self,
        *,
        req_id: str,
        phase: str,
        kv_group: int,
        keys_layer_major: list[list[CacheEngineKey]],
        layers_per_batch: int,
        max_inflight_batches: int = 1,
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
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be at least 1")
        if not keys_layer_major:
            return []

        resolved_layers: list[list[MemoryObj]] = [
            [] for _ in keys_layer_major
        ]
        acquired: list[MemoryObj] = []
        pinned: list[MemoryObj] = []
        backend_calls = 0
        direct_shared_objects = 0
        fallback_materializations = 0
        started = time.perf_counter()

        windows: list[
            tuple[int, int, list[tuple[int, int]], list[CacheEngineKey]]
        ] = []
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

        def fetch_window(
            fetch_keys: list[CacheEngineKey],
        ) -> list[Optional[MemoryObj]]:
            return self.storage_manager.batched_get(
                fetch_keys,
                location="RemoteBackend",
            )

        executor: Optional[ThreadPoolExecutor] = None
        futures: list[Future[list[Optional[MemoryObj]]]] = []
        next_future_index = 0
        if max_inflight_batches > 1 and len(windows) > 1:
            executor = ThreadPoolExecutor(
                max_workers=min(max_inflight_batches, len(windows)),
                thread_name_prefix="lmcache-mooncake-get",
            )
            futures = [
                executor.submit(fetch_window, fetch_keys)
                for _, _, _, fetch_keys in windows
            ]

        try:
            for window_index, (
                window_start,
                window_end,
                coordinates,
                fetch_keys,
            ) in enumerate(windows):
                if futures:
                    fetched = futures[window_index].result()
                    next_future_index = window_index + 1
                else:
                    fetched = fetch_window(fetch_keys)
                backend_calls += 1
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

                pending_fetched = list(fetched)
                try:
                    for index, (layer_id, chunk_index) in enumerate(coordinates):
                        fetched_obj = pending_fetched[index]
                        assert fetched_obj is not None
                        key = fetch_keys[index]
                        if self._is_rank0_shared_mem_obj(fetched_obj):
                            mem_obj = fetched_obj
                            pending_fetched[index] = None
                            direct_shared_objects += 1
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
                            fallback_materializations += 1

                        acquired.append(mem_obj)
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
                        resolved_layers[layer_id].append(mem_obj)
                finally:
                    for fetched_obj in pending_fetched:
                        if fetched_obj is not None:
                            fetched_obj.ref_count_down()

            logger.info(
                "[P2D_SHARED_CPU_WINDOWED_GET] req=%s phase=%s kv_group=%d "
                "layers=%d layers_per_batch=%d max_inflight_batches=%d "
                "backend_calls=%d chunks=%d "
                "direct_shared_objects=%d fallback_materializations=%d "
                "total_ms=%.3f",
                req_id,
                phase,
                kv_group,
                len(keys_layer_major),
                layers_per_batch,
                min(max_inflight_batches, len(windows)),
                backend_calls,
                sum(len(layer) for layer in keys_layer_major),
                direct_shared_objects,
                fallback_materializations,
                (time.perf_counter() - started) * 1000,
            )
            return resolved_layers
        except Exception:
            for future in futures[next_future_index:]:
                if future.cancel():
                    continue
                try:
                    unfetched = future.result()
                except Exception:
                    continue
                for mem_obj in unfetched:
                    if mem_obj is not None:
                        mem_obj.ref_count_down()
            for mem_obj in reversed(pinned):
                if mem_obj.is_pinned:
                    mem_obj.unpin()
            for mem_obj in reversed(acquired):
                if mem_obj.is_valid():
                    mem_obj.ref_count_down()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _close_shared_retrieve_consumer(
        consumer: Optional[Generator[Any, Any, Any]],
    ) -> None:
        if consumer is not None:
            consumer.close()

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

    def _retrieve_layer_shared_rank0(
        self,
        *,
        starts: list[int],
        ends: list[int],
        keys_layer_major: list[list[CacheEngineKey]],
        chunk_locations_layer_major: list[list[str]],
        location: Optional[str],
        ret_mask: torch.Tensor,
        monitor_req_id: int,
        req_id: str,
        kv_group: int,
        kwargs: dict[str, Any],
    ) -> Generator[Optional[torch.Tensor], None, None]:
        assert self.storage_manager is not None
        assert self.gpu_connector is not None

        phase = kwargs.get("shared_cpu_phase", "dense_prefix")
        request_ordinal = int(kwargs.get("shared_cpu_request_ordinal", 0))
        if not keys_layer_major:
            for layer_id in range(self.num_layers):
                self._broadcast_shared_envelope(
                    SharedHandleEnvelope(
                        request_id=req_id,
                        phase=phase,
                        request_ordinal=request_ordinal,
                        layer_id=layer_id,
                        kv_group=kv_group,
                        status="skipped",
                        generation=self.shared_cpu_cache_generation,
                        handles=[],
                        message="no dense prefix shared CPU cache chunks selected",
                    )
                )
                yield None
            yield None
            self.stats_monitor.on_retrieve_finished(monitor_req_id, 0)
            yield ret_mask
            return

        assert_layerwise_gpu_connector(self.gpu_connector)
        mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
        next(mem_obj_consumer)

        to_release: list[MemoryObj] = []
        try:
            for layer_id in range(self.num_layers):
                try:
                    mem_objs_layer = self._resolve_shared_rank0_layer_mem_objs(
                        req_id=req_id,
                        phase=phase,
                        layer_id=layer_id,
                        kv_group=kv_group,
                        keys_layer=keys_layer_major[layer_id],
                        chunk_locations=chunk_locations_layer_major[layer_id],
                    )
                except Exception as exc:
                    message = (
                        "Shared CPU cache rank0 materialization failed before "
                        "handle publication."
                    )
                    self._broadcast_shared_envelope(
                        self._shared_layerwise_error_envelope(
                            req_id=req_id,
                            phase=phase,
                            request_ordinal=request_ordinal,
                            layer_id=layer_id,
                            kv_group=kv_group,
                            message=message,
                            details={
                                "error": str(exc),
                                "location": location,
                            },
                        )
                    )
                    raise
                to_release.extend(mem_objs_layer)

                handles = self._make_shared_handles_for_layer(
                    req_id=req_id,
                    phase=phase,
                    keys_layer=keys_layer_major[layer_id],
                    mem_objs_layer=mem_objs_layer,
                    layer_id=layer_id,
                    kv_group=kv_group,
                )
                self._broadcast_shared_envelope(
                    SharedHandleEnvelope(
                        request_id=req_id,
                        phase=phase,
                        request_ordinal=request_ordinal,
                        layer_id=layer_id,
                        kv_group=kv_group,
                        status="ok",
                        generation=self.shared_cpu_cache_generation,
                        handles=handles,
                    )
                )

                if layer_id == 0:
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_obj_consumer.send(mem_objs_layer)

            next(mem_obj_consumer)
            self._close_shared_retrieve_consumer(mem_obj_consumer)
            mem_obj_consumer = None
            retrieved_tokens = torch.sum(ret_mask)
            self.stats_monitor.on_retrieve_finished(
                monitor_req_id,
                retrieved_tokens,
            )
            logger.info(
                "[req_id=%s kv_group=%s] Shared CPU rank0 retrieved %d tokens",
                req_id,
                kv_group,
                retrieved_tokens,
            )
            yield None
            # Keep request-owned shared objects through the final layer wait,
            # but release them before the result yield can remain suspended.
            self._release_shared_retrieve_objs(to_release, unpin=True)
            yield ret_mask
        finally:
            try:
                self._close_shared_retrieve_consumer(mem_obj_consumer)
            finally:
                self._release_shared_retrieve_objs(to_release, unpin=True)

    def _retrieve_layer_shared_passive(
        self,
        *,
        starts_all: list[int],
        ends_all: list[int],
        keys_layer_major: list[list[CacheEngineKey]],
        ret_mask: torch.Tensor,
        monitor_req_id: int,
        req_id: str,
        kv_group: int,
        kwargs: dict[str, Any],
    ) -> Generator[Optional[torch.Tensor], None, None]:
        assert self.gpu_connector is not None
        if self.shared_cpu_cache_passive_allocator is None:
            raise ValueError(
                "Shared CPU cache passive allocator is not initialized. "
                "Startup preflight did not complete."
            )

        phase = kwargs.get("shared_cpu_phase", "dense_prefix")
        request_ordinal = int(kwargs.get("shared_cpu_request_ordinal", 0))
        assert_layerwise_gpu_connector(self.gpu_connector)
        mem_obj_consumer = None
        to_release: list[MemoryObj] = []
        expected_handle_count: Optional[int] = None

        try:
            for layer_id in range(self.num_layers):
                envelope = self._receive_shared_envelope()
                self._validate_shared_layerwise_envelope(
                    envelope,
                    req_id=req_id,
                    phase=phase,
                    request_ordinal=request_ordinal,
                    layer_id=layer_id,
                    kv_group=kv_group,
                )

                if envelope.status in ("miss", "skipped"):
                    if layer_id == 0:
                        yield torch.sum(ret_mask)
                    else:
                        yield None
                    continue

                if expected_handle_count is None:
                    expected_handle_count = len(envelope.handles)
                    starts = starts_all[:expected_handle_count]
                    ends = ends_all[:expected_handle_count]
                    for start, end in zip(starts, ends, strict=False):
                        ret_mask[start:end] = True
                    mem_obj_consumer = self.gpu_connector.batched_to_gpu(
                        starts,
                        ends,
                        **kwargs,
                    )
                    next(mem_obj_consumer)
                elif len(envelope.handles) != expected_handle_count:
                    raise ValueError(
                        "Shared CPU cache passive received inconsistent handle "
                        f"count at layer {layer_id}: {len(envelope.handles)} != "
                        f"{expected_handle_count}"
                    )

                mem_objs_layer: list[MemoryObj] = []
                for chunk_index, handle in enumerate(envelope.handles):
                    expected_shape, expected_dtype, expected_fmt = (
                        self._expected_shared_cpu_chunk_metadata(
                            kv_group=kv_group,
                            num_tokens=int(
                                ends_all[chunk_index] - starts_all[chunk_index]
                            ),
                        )
                    )
                    mem_obj = self.shared_cpu_cache_passive_allocator.create_view(
                        handle,
                        expected_request_id=req_id,
                        expected_phase=phase,
                        expected_layer_id=layer_id,
                        expected_kv_group=kv_group,
                        expected_chunk_index=chunk_index,
                        expected_key=keys_layer_major[layer_id][chunk_index],
                        expected_shape=expected_shape,
                        expected_dtype=expected_dtype,
                        expected_fmt=expected_fmt,
                        expected_cached_positions=range(
                            int(starts_all[chunk_index]),
                            int(ends_all[chunk_index]),
                        ),
                        expected_producer_rank=self.metadata.first_rank,
                    )
                    mem_objs_layer.append(mem_obj)
                    to_release.append(mem_obj)

                if layer_id == 0:
                    yield torch.sum(ret_mask)
                else:
                    yield None

                assert mem_obj_consumer is not None
                mem_obj_consumer.send(mem_objs_layer)

            if mem_obj_consumer is not None:
                next(mem_obj_consumer)
                self._close_shared_retrieve_consumer(mem_obj_consumer)
                mem_obj_consumer = None

            retrieved_tokens = torch.sum(ret_mask)
            self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
            logger.info(
                "[req_id=%s kv_group=%s] Shared CPU passive retrieved %d tokens",
                req_id,
                kv_group,
                retrieved_tokens,
            )
            yield None
            # Keep request-owned shared objects through the final layer wait,
            # but release them before the result yield can remain suspended.
            self._release_shared_retrieve_objs(to_release, unpin=False)
            yield ret_mask
        finally:
            try:
                self._close_shared_retrieve_consumer(mem_obj_consumer)
            finally:
                self._release_shared_retrieve_objs(to_release, unpin=False)

    def skip_shared_layerwise_retrieve(
        self,
        *,
        req_id: str,
        phase: str,
        request_ordinal: int = 0,
        kv_group: int,
        message: str,
        num_tokens: int = 0,
    ) -> Generator[Optional[torch.Tensor], Any, None]:
        """Ordered no-op shared retrieve for intentionally skipped groups."""
        ret_mask = torch.zeros(num_tokens, dtype=torch.bool, device="cpu")
        if not self.enable_shared_cpu_cache or self.metadata.world_size <= 1:
            for _ in range(self.num_layers):
                yield ret_mask
            yield ret_mask
            return

        for layer_id in range(self.num_layers):
            yield ret_mask
            if self.metadata.is_first_rank():
                self._broadcast_shared_envelope(
                    SharedHandleEnvelope(
                        request_id=req_id,
                        phase=phase,
                        request_ordinal=int(request_ordinal),
                        layer_id=layer_id,
                        kv_group=kv_group,
                        status="skipped",
                        generation=self.shared_cpu_cache_generation,
                        handles=[],
                        message=message,
                    )
                )
            else:
                envelope = self._receive_shared_envelope()
                self._validate_shared_layerwise_envelope(
                    envelope,
                    req_id=req_id,
                    phase=phase,
                    request_ordinal=int(request_ordinal),
                    layer_id=layer_id,
                    kv_group=kv_group,
                )
                if envelope.status != "skipped":
                    raise ValueError(
                        "Shared CPU cache expected skipped envelope for "
                        f"req_id={req_id}, phase={phase}, layer_id={layer_id}, "
                        f"kv_group={kv_group}, got status={envelope.status!r}"
                    )
        yield ret_mask

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            self._preflight_shared_cpu_shm_capacity()
            shared_passive_rank = (
                self.enable_shared_cpu_cache
                and self._is_passive()
                and self.metadata.world_size > 1
            )
            need_layerwise_storage = self.use_layerwise and not shared_passive_rank
            if (
                self.lmcache_worker is not None
                or need_layerwise_storage
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    f"Initialize storage manager on rank {self.metadata.worker_id}, "
                    f"use layerwise: {self.use_layerwise},"
                    f"save only first rank: {self.save_only_first_rank}"
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                try:
                    self.storage_manager = StorageManager(
                        self.config,
                        self.metadata,
                        event_manager=self.event_manager,
                        lmcache_worker=self.lmcache_worker,
                        async_lookup_server=async_lookup_server,
                    )
                except Exception as exc:
                    if (
                        self.enable_shared_cpu_cache
                        and self.metadata.world_size > 1
                        and self.metadata.is_first_rank()
                        and callable(getattr(self, "broadcast_object_fn", None))
                    ):
                        try:
                            self.broadcast_object_fn(
                                self._shared_cpu_cache_startup_envelope(
                                    "error",
                                    "Shared CPU cache rank0 failed while "
                                    "initializing shm-backed StorageManager: "
                                    f"{exc}",
                                ),
                                self.metadata.first_rank,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to broadcast shared CPU cache startup "
                                "error after rank0 StorageManager failure"
                            )
                    raise
            self._post_init_shared_cpu_cache()
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        kv_group = kwargs.get("kv_group", 0)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
                kv_group=kv_group,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes, kv_dtypes = self._metadata_shapes_dtypes_for_kv_group(
                    kv_group=kv_group,
                    num_tokens=num_tokens,
                )

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        with store_stats.profile_from_gpu():
            self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

        with store_stats.profile_put():
            transfer_spec = kwargs.get("transfer_spec", None)
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s kv_group=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            kv_group,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[LayerwiseStoreResult], None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None for each layer and a
            LayerwiseStoreResult after the final layer. In the first iteration,
            the generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends and yields the completed
            store output.
        """
        store_result = LayerwiseStoreResult(
            request_id=str(kwargs.get("req_id", "unspecified")),
            kv_group=int(kwargs.get("kv_group", 0) or 0),
        )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        # Passive rank guard: when save_only_first_rank is enabled, only rank 0
        # stores. This closes the known TODO at cache_engine.py:160-165 -
        # previously store_layer had no _is_passive() check, causing duplicate
        # stores on non-rank-0 workers under MLA + layerwise.
        if self._is_passive():
            logger.debug(
                "Passive rank (save_only_first_rank), skipping store_layer"
            )
            for layer_id in range(self.num_layers):
                yield
            # Extra yield consumed by wait_for_save() after the last layer.
            yield store_result
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)
        store_result.request_id = req_id

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            yield store_result
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        kv_group = kwargs.get("kv_group", 0)
        kv_dtype = self._shared_cpu_dtype_for_kv_group(kv_group)
        store_fmt = self._memory_format_for_kv_group(kv_group)
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            if self._layerwise_chunk_fully_stored(
                keys_multi_layer,
                req_id=req_id,
                kv_group=kv_group,
                start=start,
                end=end,
            ):
                continue

            # Allocate the memory object
            num_tokens = end - start
            try:
                kv_shape_single_layer = self.gpu_connector.get_shape(
                    num_tokens,
                    kv_group=kv_group,
                )
            except TypeError as exc:
                if kv_group != 0:
                    raise TypeError(
                        "Layerwise store for kv_group=1 requires "
                        "gpu_connector.get_shape(num_tokens, kv_group=...) "
                        "or an engine-specific store_layer override."
                    ) from exc
                kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=store_fmt,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]
            store_result.starts = starts
            store_result.ends = ends
            store_result.keys = keys
            store_result.memory_objs = memory_objs
            pending_store_release: dict[int, MemoryObj] = {
                id(mem_obj): mem_obj
                for layer_objs in memory_objs
                for mem_obj in layer_objs
            }
            mem_obj_generator = None

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            try:
                t_start = time.perf_counter()
                mem_obj_generator = self.gpu_connector.batched_from_gpu(
                    memory_objs, starts, ends, **kwargs
                )

                next(mem_obj_generator)

                for layer_id in range(self.num_layers):
                    yield
                    next(mem_obj_generator)
                    self.storage_manager.batched_put(
                        keys[layer_id],
                        memory_objs[layer_id],
                        location=self.store_location,
                    )
                    for mem_obj in memory_objs[layer_id]:
                        pending_store_release.pop(id(mem_obj), None)

                tot_time = time.perf_counter() - t_start
                logger.info(
                    "[req_id=%s kv_group=%s] Stored %d out of total %d tokens. "
                    "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                    req_id,
                    kv_group,
                    tot_token_num,
                    len(tokens),
                    tot_kv_size / 1024**3,
                    tot_time * 1000,
                    tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
                )
            finally:
                if mem_obj_generator is not None:
                    close_fn = getattr(mem_obj_generator, "close", None)
                    if close_fn is not None:
                        try:
                            close_fn()
                        except (GeneratorExit, RuntimeError, ValueError):
                            pass
                for mem_obj in list(pending_store_release.values()):
                    if mem_obj.is_valid():
                        mem_obj.ref_count_down()
                pending_store_release.clear()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield store_result

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                with torch.cuda.stream(self.broadcast_stream):
                    self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                    )

                # if self.gpu_connector has load_stream, self.broadcast_stream is equals
                # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
                # will execute sequentially within the stream.
                # if self.gpu_connector does not have load_stream, self.broadcast_stream
                # is created by torch.cuda.Stream(), we need to synchronize broadcast
                # operation, and then process to_cpu operation.
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
            if not self.async_loading:
                memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            kv_group = kwargs.get("kv_group", 0)
            logger.info(
                "[req_id=%s kv_group=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                kv_group,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        kv_group = kwargs.get("kv_group", 0)
        shared_layerwise_retrieve = self._should_use_shared_layerwise_retrieve(
            kv_group
        )
        if not (shared_layerwise_retrieve and self._is_passive()):
            assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        segments: List[LayerwiseRetrieveSegment] = []
        segment_location: Optional[str] = None
        segment_starts: List[int] = []
        segment_ends: List[int] = []
        segment_keys: List[List[CacheEngineKey]] = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        if shared_layerwise_retrieve and self._is_passive():
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
                kv_group=kv_group,
            ):
                assert isinstance(key, CacheEngineKey)
                starts.append(start)
                ends.append(end)
                keys.append(key.split_layers(self.num_layers))

            yield from self._retrieve_layer_shared_passive(
                starts_all=starts,
                ends_all=ends,
                keys_layer_major=[list(row) for row in zip(*keys, strict=False)]
                if keys
                else [],
                ret_mask=ret_mask,
                monitor_req_id=monitor_req_id,
                req_id=req_id,
                kv_group=kv_group,
                kwargs=kwargs,
            )
            return

        if shared_layerwise_retrieve:
            location = None
            chunk_locations: list[list[str]] = []
            missing_shared_chunks: list[dict[str, Any]] = []
            phase = kwargs.get("shared_cpu_phase", "dense_prefix")
            request_ordinal = int(kwargs.get("shared_cpu_request_ordinal", 0))
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
                kv_group=kv_group,
            ):
                assert isinstance(key, CacheEngineKey)

                keys_multi_layer = key.split_layers(self.num_layers)
                locations_multi_layer: list[str] = []
                missing_layer = False
                for layer_idx, layer_key in enumerate(keys_multi_layer):
                    current_location = self._find_shared_rank0_chunk_location(
                        layer_key
                    )
                    if current_location is None:
                        # A missing layer0 is the normal dense-prefix miss
                        # boundary. If layer0 exists but a later layer is
                        # absent, the selected chunk is inconsistent and must
                        # fail before handle publication in strict mode.
                        if layer_idx != 0:
                            missing_shared_chunks.append(
                                {
                                    "chunk_index": len(keys),
                                    "layer_id": layer_idx,
                                    "start": int(start),
                                    "end": int(end),
                                    "key": repr(layer_key),
                                }
                            )
                        missing_layer = True
                        break
                    locations_multi_layer.append(current_location)
                if missing_layer:
                    break

                starts.append(start)
                ends.append(end)
                keys.append(keys_multi_layer)
                chunk_locations.append(locations_multi_layer)
                ret_mask[start:end] = True

            keys_layer_major = (
                [list(row) for row in zip(*keys, strict=False)]
                if keys
                else []
            )
            chunk_locations_layer_major = (
                [list(row) for row in zip(*chunk_locations, strict=False)]
                if chunk_locations
                else []
            )
            unique_locations = {
                location
                for locations_multi_layer in chunk_locations
                for location in locations_multi_layer
            }
            location = (
                next(iter(unique_locations))
                if len(unique_locations) == 1
                else "mixed"
                if unique_locations
                else None
            )
            if missing_shared_chunks and self.shared_cpu_cache_strict:
                message = (
                    "Shared CPU dense prefix layerwise retrieve missing required "
                    "chunks before rank0 handle publication."
                )
                self._broadcast_shared_envelope(
                    self._shared_layerwise_error_envelope(
                        req_id=req_id,
                        phase=phase,
                        request_ordinal=request_ordinal,
                        layer_id=0,
                        kv_group=kv_group,
                        message=message,
                        details={
                            "missing_chunks": missing_shared_chunks,
                            "resolved_chunk_count": len(keys),
                            "token_count": len(tokens),
                        },
                    )
                )
                raise ValueError(
                    f"{message} req_id={req_id}, kv_group={kv_group}, "
                    f"missing_chunks={missing_shared_chunks}"
                )
            yield from self._retrieve_layer_shared_rank0(
                starts=starts,
                ends=ends,
                keys_layer_major=keys_layer_major,
                chunk_locations_layer_major=chunk_locations_layer_major,
                location=location,
                ret_mask=ret_mask,
                monitor_req_id=monitor_req_id,
                req_id=req_id,
                kv_group=kv_group,
                kwargs=kwargs,
            )
            return
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                if segment_location is None:
                    segment_location = current_location
                elif segment_location != current_location:
                    segments.append(
                        (
                            segment_location,
                            segment_starts,
                            segment_ends,
                            segment_keys,
                        )
                    )
                    segment_location = current_location
                    segment_starts = []
                    segment_ends = []
                    segment_keys = []
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            segment_starts.append(start)
            segment_ends.append(end)
            segment_keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if segment_location is not None and segment_keys:
            segments.append(
                (segment_location, segment_starts, segment_ends, segment_keys)
            )

        if keys:
            get_generators = []
            for segment in segments:
                segment_keys_layer_major = [
                    list(row) for row in zip(*segment[3], strict=False)
                ]
                get_generators.append(
                    self.storage_manager.layerwise_batched_get(
                        segment_keys_layer_major,
                        location=segment[0],
                    )
                )

            assert_layerwise_gpu_connector(self.gpu_connector)

            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            retrieved_by_location: dict[str, list[MemoryObj]] = defaultdict(list)
            for layer_id in range(self.num_layers):
                tasks = [next(get_generator) for get_generator in get_generators]
                for task in tasks:
                    assert task is not None

                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = []
                for segment, task in zip(segments, tasks, strict=True):
                    segment_mem_objs = task.result()
                    mem_objs_layer.extend(segment_mem_objs)
                    retrieved_by_location[segment[0]].extend(segment_mem_objs)
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()

            next(mem_obj_consumer)

            # Unpin disk-loaded staging objects after device-side sync is enqueued.
            for location, mem_objs in retrieved_by_location.items():
                self._maybe_unpin_retrieved_objs(mem_objs, location)
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s kv_group=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                kv_group,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield ret_mask


    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                if self._use_sampled_scheduler_lookup():
                    sampled_chunks: list[tuple[int, CacheEngineKey]] = []
                    for _, end, key in chunk_info_iterator:
                        assert isinstance(key, CacheEngineKey)
                        sampled_chunks.append((end, key))
                    res = self._sampled_scheduler_lookup(
                        sampled_chunks,
                        lookup_id=lookup_id,
                        pin=pin,
                        request_configs=request_configs,
                    )
                    return res

                lookup_kv_groups = self._layerwise_lookup_kv_groups()
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    chunk_group_layers: list[list[CacheEngineKey]] = []
                    for kv_group in lookup_kv_groups:
                        group_key = self._lookup_key_for_kv_group(
                            key,
                            kv_group=kv_group,
                            request_configs=request_configs,
                        )
                        key_all_layers = group_key.split_layers(self.num_layers)
                        hit_chunks, block_mapping = (
                            self.storage_manager.batched_contains(
                                key_all_layers,  # type: ignore
                                search_range,
                                False,
                            )
                        )
                        # Only all layers are hit and hit in one location,
                        # we consider this group complete for the chunk.
                        if (
                            hit_chunks != self.num_layers
                            or len(block_mapping) != 1
                        ):
                            return res
                        chunk_group_layers.append(key_all_layers)

                    if pin:
                        assert lookup_id is not None, (
                            "lookup_id is required when pin is True"
                        )
                        pinned_mappings: list[tuple[str, list[CacheEngineKey]]] = []
                        try:
                            for key_all_layers in chunk_group_layers:
                                hit_chunks, block_mapping = (
                                    self.storage_manager.batched_contains(
                                        key_all_layers,  # type: ignore
                                        search_range,
                                        True,
                                    )
                                )
                                if (
                                    hit_chunks != self.num_layers
                                    or len(block_mapping) != 1
                                ):
                                    for location, pinned_keys in block_mapping.items():
                                        self.storage_manager.batched_unpin(
                                            pinned_keys,
                                            [location],
                                        )
                                    for location, pinned_keys in pinned_mappings:
                                        self.storage_manager.batched_unpin(
                                            pinned_keys,
                                            [location],
                                        )
                                    return res
                                location = next(iter(block_mapping.keys()))
                                pinned_keys = block_mapping[location]
                                pinned_mappings.append((location, pinned_keys))
                            for location, pinned_keys in pinned_mappings:
                                self.lookup_pins[lookup_id][location].extend(
                                    pinned_keys
                                )
                        except Exception:
                            for location, pinned_keys in pinned_mappings:
                                self.storage_manager.batched_unpin(
                                    pinned_keys,
                                    [location],
                                )
                            raise
                    res = end
                    continue
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        if self.use_layerwise:
            experimental_lookup_enabled = self._sampled_scheduler_lookup_requested()
            if experimental_lookup_enabled:
                use_remote_sampling = self._use_sampled_scheduler_lookup()
                lookup_search_range = (
                    ["RemoteBackend"] if use_remote_sampling else search_range
                )
                async_lookup_server = getattr(
                    self.storage_manager,
                    "async_lookup_server",
                    None,
                )
                if async_lookup_server is None:
                    logger.error(
                        "Layerwise async lookup has no response server: lookup_id=%s",
                        lookup_id,
                    )
                    return

                async def run_layerwise_lookup() -> None:
                    try:
                        result = await asyncio.to_thread(
                            self.lookup,
                            tokens=tokens,
                            hashes=hashes,
                            offsets=offsets,
                            search_range=lookup_search_range,
                            lookup_id=lookup_id,
                            pin=pin,
                            request_configs=request_configs,
                        )
                    except Exception:
                        logger.exception(
                            "Layerwise async lookup failed: lookup_id=%s",
                            lookup_id,
                        )
                        result = 0
                    async_lookup_server.send_response_to_scheduler(
                        lookup_id,
                        result,
                    )

                asyncio.run_coroutine_threadsafe(
                    run_layerwise_lookup(),
                    self.storage_manager.loop,
                )
                return

            message = (
                "Async lookup/prefetch is not supported with layerwise cache "
                "keys. Falling back to recompute for this request instead of "
                "admitting a potentially incomplete layerwise hit."
            )
            logger.error("%s lookup_id=%s", message, lookup_id)
            async_lookup_server = getattr(
                self.storage_manager,
                "async_lookup_server",
                None,
            )
            if async_lookup_server is not None:
                async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            return

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        self._release_all_shared_cpu_request_leases()
        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        try:
            if self.shared_cpu_cache_mapping is not None:
                self.shared_cpu_cache_mapping.close()
                self.shared_cpu_cache_mapping = None
        except Exception as e:
            logger.error(f"Error closing shared CPU cache mapping: {e}")

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        kv_group = kwargs.get("kv_group", 0)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        kv_group = kwargs.get("kv_group", 0)

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            kv_group=kv_group,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(f"cuda:{self.metadata.worker_id}")
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _memory_format_for_kv_group(self, kv_group: int = 0) -> MemoryFormat:
        """Return the CPU chunk format tag for a KV cache group."""
        if kv_group == 1 and getattr(self.config, "dsa_two_groups", False):
            return MemoryFormat.KV_DSA_INDEX_FMT
        if self.metadata.use_mla:
            return MemoryFormat.KV_MLA_LATENT_FMT
        assert self.fmt is not None, "Memory format is not initialized"
        return self.fmt

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _is_indexer_passive(self):
        """Same as _is_passive but for the DSA indexer group (kv_group=1)."""
        return (
            self.save_indexer_only_first_rank
            and not self.metadata.is_first_rank()
        )

    def _maybe_unpin_retrieved_objs(
        self,
        mem_objs: List[MemoryObj],
        location: Optional[str],
    ) -> None:
        for mem_obj in mem_objs:
            if mem_obj.is_pinned:
                mem_obj.unpin()

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            shared_cpu_size_gb = (
                config.get_extra_config_value(
                    "shared_cpu_cache_size_gb",
                    getattr(config, "shared_cpu_cache_size_gb", None),
                )
                if config.get_extra_config_value(
                    "enable_shared_cpu_cache",
                    getattr(config, "enable_shared_cpu_cache", False),
                )
                else None
            )
            if shared_cpu_size_gb is not None:
                first_rank_max_local_cpu_size = float(shared_cpu_size_gb)
            else:
                first_rank_max_local_cpu_size = (
                    config.extra_config.get(
                        "first_rank_max_local_cpu_size", max_local_cpu_size
                    )
                    if config.extra_config
                    else max_local_cpu_size
                )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
                config=config,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
            config=config,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
