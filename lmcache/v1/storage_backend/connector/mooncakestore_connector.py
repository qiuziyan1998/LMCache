# SPDX-License-Identifier: Apache-2.0
# Standard
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, cast, no_type_check
import asyncio
import json
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.cold_start_perf import (
    cold_start_perf_enabled,
    cold_start_perf_log,
    cold_start_perf_now,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    LayerPageMemoryObj,
    MemoryFormat,
    MemoryObj,
    _layer_page_shape,
)
from lmcache.v1.mooncake_key_trace import trace_mooncake_keys
from lmcache.v1.mooncake_layout import (
    mooncake_legacy_key,
    mooncake_layer_pages_enabled,
    mooncake_page_key,
    mooncake_valid_tokens,
)
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.system_detection import NUMADetector

logger = init_logger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    transfer_timeout: int
    storage_root_dir: str
    prefer_local_alloc: bool = False
    page_first_multi_buffer: bool = False

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        # Read Mooncake-specific knob
        prefer_local_alloc = config.get("mooncake_prefer_local_alloc", False)

        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            transfer_timeout=config.get("transfer_timeout", 1),
            storage_root_dir=config.get("storage_root_dir", ""),
            prefer_local_alloc=prefer_local_alloc,
            page_first_multi_buffer=config.get(
                "mooncake_page_first_multi_buffer", False
            ),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)

    @staticmethod
    def load_from_lmcache_config(
        config: "LMCacheEngineConfig",
    ) -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra config is not set.")
        # Read Mooncake-specific knob
        prefer_local_alloc = extra_config.get("mooncake_prefer_local_alloc", False)

        return MooncakeStoreConfig(
            local_hostname=extra_config["local_hostname"],
            metadata_server=extra_config["metadata_server"],
            global_segment_size=extra_config.get("global_segment_size", 3355443200),
            local_buffer_size=extra_config.get("local_buffer_size", 1073741824),
            protocol=extra_config.get("protocol", "tcp"),
            device_name=extra_config.get("device_name", ""),
            master_server_address=extra_config["master_server_address"],
            transfer_timeout=extra_config.get("transfer_timeout", 1),
            storage_root_dir=extra_config.get("storage_root_dir", ""),
            prefer_local_alloc=prefer_local_alloc,
            page_first_multi_buffer=extra_config.get(
                "mooncake_page_first_multi_buffer", False
            ),
        )


class MooncakestoreConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        dev_name,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_config: Optional[LMCacheEngineConfig],
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)
        self._sampled_layerwise_lookup = bool(
            getattr(
                local_cpu_backend.config,
                "experimental_sampled_layerwise_lookup",
                False,
            )
        )
        logger.info(
            "Mooncake connector save_chunk_meta=%s, meta_shapes=%s",
            self.save_chunk_meta,
            self.meta_shapes,
        )
        self._dsa_raw_token_dims, dsa_raw_token_dims_source = (
            self._resolve_dsa_raw_token_dims(
                local_cpu_backend.config,
                local_cpu_backend.metadata,
            )
        )
        if self._dsa_raw_token_dims:
            logger.info(
                "Mooncake raw DSA group metadata enabled: %s (source=%s)",
                self._dsa_raw_token_dims,
                dsa_raw_token_dims_source,
            )

        try:
            # Third Party
            from mooncake.store import (
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
            if config_file_path is not None:
                self.config = MooncakeStoreConfig.from_file(config_file_path)
            elif lmcache_config is not None:
                self.config = MooncakeStoreConfig.load_from_lmcache_config(
                    lmcache_config
                )
            else:
                raise ValueError("MOONCAKE_CONFIG_PATH/lmcache_config must be provided")

            if not self.config.master_server_address:
                if host != "" and port != 0:
                    self.config.master_server_address = host + ":" + str(port)
            if dev_name != "":
                self.config.device_name = dev_name
            logger.info("Mooncake Configuration loaded. config: %s", self.config)

            # Check if storage_root_dir exists and set environment variable
            if (
                self.config.storage_root_dir is not None
                and self.config.storage_root_dir != ""
            ):
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s", self.config.storage_root_dir
                )

            logger.info("Setting up Mooncake store with parameters:")
            logger.info(f"  local_hostname: {self.config.local_hostname}")
            logger.info(f"  metadata_server: {self.config.metadata_server}")
            logger.info(f"  global_segment_size: {self.config.global_segment_size}")
            logger.info(f"  local_buffer_size: {self.config.local_buffer_size}")
            logger.info(f"  protocol: {self.config.protocol}")
            logger.info(f"  device_name: {self.config.device_name}")
            logger.info(f"  master_server_address: {self.config.master_server_address}")

            try:
                numa_mapping = getattr(
                    local_cpu_backend.memory_allocator, "numa_mapping", None
                )
                if numa_mapping is None and lmcache_config is not None:
                    numa_mapping = NUMADetector.get_numa_mapping(lmcache_config)

                if numa_mapping:
                    current_device_id = torch.cuda.current_device()
                    gpu_to_numa = getattr(numa_mapping, "gpu_to_numa_mapping", {})
                    numa_id = gpu_to_numa.get(current_device_id)
                    logger.info(
                        f"NUMA mapping detected (pre-Mooncake setup): {gpu_to_numa}"
                    )
                    try:
                        # Third Party
                        from mooncake.store import bind_to_numa_node

                        if numa_id is not None:
                            bind_to_numa_node(numa_id)
                            logger.info(
                                f"GPU {current_device_id}, "
                                f"NUMA node {numa_id} binding done"
                            )
                        else:
                            logger.info(
                                f"NUMA mapping not found for GPU {current_device_id}"
                            )
                    except ImportError:
                        logger.warning(
                            "unable to import bind_to_numa_node from mooncake.store"
                        )
                else:
                    logger.info("NUMA mapping unavailable or disabled")
            except Exception as e:
                logger.warning(
                    f"Failed to determine NUMA mapping before Mooncake setup: {e}"
                )

            status = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            if status not in (None, 0):
                raise RuntimeError(f"Mooncake setup failed: status={status}")

            logger.info("Mooncake store setup completed successfully")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self._page_first_multi_buffer = self.config.page_first_multi_buffer
        self._layer_merged_pages = bool(
            lmcache_config is not None
            and mooncake_layer_pages_enabled(lmcache_config)
            and getattr(local_cpu_backend, "layer_page_objects", False)
        )
        self._page_num_layers = int(
            getattr(local_cpu_backend.metadata, "kv_shape", (1,))[0]
        )
        if getattr(self, "_page_first_multi_buffer", False):
            if self.save_chunk_meta:
                raise ValueError(
                    "mooncake_page_first_multi_buffer requires "
                    "save_chunk_meta=False"
                )
            required_methods = (
                "batch_get_into_multi_buffers",
                "batch_put_from_multi_buffers",
                "batch_is_exist",
            )
            missing_methods = [
                name
                for name in required_methods
                if not callable(getattr(self.store, name, None))
            ]
            if missing_methods:
                raise RuntimeError(
                    "Installed Mooncake lacks page-first APIs: "
                    f"{missing_methods}"
                )
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.registered_buffer_ptr = None
        self.registered_buffer_size = 0
        self._external_buffers: dict[int, int] = {}
        self._external_put_lock = asyncio.Lock()
        self._inflight_put_tasks: set[asyncio.Task[Any]] = set()
        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1

        # Set preferred_segment based on configuration
        if self.config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()

        # Register CPU buffer for zero-copy operations
        self._register_cpu_buffer()

        logger.info("MooncakeConnector initialized successfully.")

    @staticmethod
    def _parse_dsa_raw_token_dims(value: Any) -> dict[int, int]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = dict(item.split(":", 1) for item in value.split(",") if item)
        else:
            parsed = value
        if isinstance(parsed, dict):
            return {int(k): int(v) for k, v in parsed.items()}
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return {0: int(parsed[0]), 1: int(parsed[1])}
        raise ValueError(
            "mooncake_dsa_raw_token_dims must be a dict, list, or "
            "'0:latent,1:indexer' string"
        )

    @staticmethod
    def _first_int_from_nested(
        config_dict: dict[str, Any], names: set[str]
    ) -> int | None:
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

    @staticmethod
    def _load_model_config_for_dsa_dims(metadata) -> tuple[dict[str, Any], str]:
        model_name = getattr(metadata, "model_name", "")
        if not model_name:
            return {}, "metadata.model_name unavailable"
        config_path = os.path.join(model_name, "config.json")
        if not os.path.isfile(config_path):
            return {}, f"{config_path} unavailable"
        try:
            with open(config_path, encoding="utf-8") as fin:
                loaded = json.load(fin)
        except Exception as exc:
            return {}, f"{config_path} load failed: {exc}"
        if not isinstance(loaded, dict):
            return {}, f"{config_path} is not a JSON object"
        return loaded, config_path

    @staticmethod
    def _has_complete_dsa_raw_token_dims(dims: dict[int, int]) -> bool:
        return dims.get(0, 0) > 0 and dims.get(1, 0) > 0

    @classmethod
    def _infer_dsa_raw_token_dims(
        cls,
        config: LMCacheEngineConfig,
        metadata,
    ) -> tuple[dict[int, int], str]:
        if not getattr(config, "dsa_two_groups", False):
            return {}, "dsa_two_groups disabled"

        inferred: dict[int, int] = {}

        shapes = getattr(metadata, "get_shapes", lambda: [])()
        if getattr(metadata, "use_mla", False) and shapes:
            shape0 = shapes[0]
            if len(shape0) >= 1:
                inferred[0] = int(shape0[-1])

        model_config, source = cls._load_model_config_for_dsa_dims(metadata)
        if not model_config:
            return inferred, source if inferred else "no inferable local metadata"

        kv_lora_rank = cls._first_int_from_nested(
            model_config,
            {"kv_lora_rank", "k_head_dim", "k_hidden_dims"},
        )
        qk_rope_head_dim = cls._first_int_from_nested(
            model_config,
            {"qk_rope_head_dim", "rope_head_dim", "v_head_dim"},
        )
        dsa_head_dim = cls._first_int_from_nested(
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
            dsa_head_dim = cls._first_int_from_nested(
                model_config,
                {"head_dim", "hidden_size_per_attention_head"},
            )
        if kv_lora_rank is not None and qk_rope_head_dim is not None:
            inferred[0] = kv_lora_rank + qk_rope_head_dim
        if dsa_head_dim is not None:
            inferred[1] = dsa_head_dim
        return inferred, source

    @classmethod
    def _resolve_dsa_raw_token_dims(
        cls,
        config: LMCacheEngineConfig,
        metadata,
    ) -> tuple[dict[int, int], str]:
        if not getattr(config, "dsa_two_groups", False):
            return {}, "dsa_two_groups disabled"

        extra = config.extra_config or {}
        override = extra.get("mooncake_dsa_raw_token_dims")
        if override is not None:
            return (
                cls._parse_dsa_raw_token_dims(override),
                "extra_config.mooncake_dsa_raw_token_dims",
            )

        inferred, source = cls._infer_dsa_raw_token_dims(config, metadata)
        if cls._has_complete_dsa_raw_token_dims(inferred):
            return inferred, f"model config inference: {source}"

        model_name = getattr(metadata, "model_name", "")
        world_size = getattr(metadata, "world_size", None)
        if "GLM-5.1-w4a8" in model_name and world_size == 8:
            # Last-resort shortcut for the current GLM DSA two-group layout:
            # kv_group=0 latent payload is (k=512 + v=64) bf16 elements/token;
            # kv_group=1 indexer payload is 128 bf16 elements/token.
            return {0: 576, 1: 128}, "hardcoded GLM-5.1-w4a8 TP8"
        return {}, "no raw DSA dims rule matched"

    def _metadata_for_raw_key(
        self,
        key: CacheEngineKey,
    ) -> tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]:
        token_dims = self._dsa_raw_token_dims.get(key.kv_group)
        if token_dims is None:
            return (
                self.meta_shapes,
                self.meta_dtypes,
                self.meta_fmt,
                self.single_token_size,
            )

        dtype = self.meta_dtypes[0]
        element_size = torch.empty((), dtype=dtype).element_size()
        chunk_size = self.local_cpu_backend.metadata.chunk_size
        fmt = (
            MemoryFormat.KV_DSA_INDEX_FMT
            if key.kv_group == 1
            else MemoryFormat.KV_MLA_LATENT_FMT
        )
        return (
            [torch.Size([chunk_size * token_dims])],
            [dtype],
            fmt,
            token_dims * element_size,
        )

    @staticmethod
    def _reshape_partial_chunk_with_token_size(
        memory_obj: MemoryObj,
        bytes_read: int,
        single_token_size: int,
    ) -> MemoryObj:
        full_chunk_size = memory_obj.get_size()
        if (
            bytes_read == 0
            or bytes_read % single_token_size != 0
            or bytes_read > full_chunk_size
        ):
            raise ValueError(
                f"bytes_read: {bytes_read} is illegal, "
                f"single_token_size: {single_token_size}, "
                f"full_chunk_size_bytes: {full_chunk_size}"
            )

        if bytes_read == full_chunk_size:
            return memory_obj

        shape_list = list(memory_obj.meta.shape)
        if len(shape_list) == 1:
            dtype = memory_obj.meta.dtype
            if dtype is None and memory_obj.meta.dtypes:
                dtype = memory_obj.meta.dtypes[0]
            if dtype is None:
                raise ValueError(
                    "Cannot reshape 1D partial chunk without dtype metadata"
                )
            element_size = torch.empty((), dtype=dtype).element_size()
            if bytes_read % element_size != 0:
                raise ValueError(
                    f"bytes_read: {bytes_read} is not aligned to element size: "
                    f"{element_size}"
                )
            shape_list[0] = bytes_read // element_size
        else:
            token_dim = memory_obj.meta.fmt.token_dim()
            if token_dim >= len(shape_list):
                raise ValueError(
                    f"Cannot reshape partial chunk with shape={memory_obj.meta.shape} "
                    f"and fmt={memory_obj.meta.fmt}"
                )
            shape_list[token_dim] = bytes_read // single_token_size

        actual_shape = torch.Size(shape_list)
        resize_raw_view = getattr(memory_obj, "resize_raw_view", None)
        if callable(resize_raw_view):
            resize_raw_view(bytes_read)
        else:
            memory_obj.raw_data = memory_obj.raw_data[:bytes_read]
        memory_obj.meta.shape = actual_shape
        if memory_obj.meta.shapes:
            memory_obj.meta.shapes = [actual_shape]
        refresh_metadata_view = getattr(memory_obj, "refresh_metadata_view", None)
        if callable(refresh_metadata_view):
            refresh_metadata_view()

        return memory_obj

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                buffer = allocator.pin_allocator.buffer
                self.registered_buffer_ptr = buffer.data_ptr()
                result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())
                if result == 0:
                    self.registered_buffer_size = buffer.numel()
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
                    self.registered_buffer_size = 0
            else:
                self.registered_buffer_ptr = None
                self.registered_buffer_size = 0
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None
            self.registered_buffer_size = 0

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None
            self.registered_buffer_size = 0

    def _register_external_owners(self, owners: tuple[Any, ...]) -> None:
        """Register tensor storages once in this Mooncake transport context."""
        active: dict[int, int] = {}
        cpu_ptr = getattr(self, "registered_buffer_ptr", None)
        cpu_size = int(getattr(self, "registered_buffer_size", 0))
        for owner in owners:
            storage = owner.untyped_storage()
            ptr, size = int(storage.data_ptr()), int(storage.nbytes())
            if (
                getattr(getattr(owner, "device", None), "type", None) == "cpu"
                and ptr == cpu_ptr
                and size <= cpu_size
            ):
                continue
            active[ptr] = size
        for ptr in self._external_buffers.keys() - active.keys():
            self.store.unregister_buffer(ptr)
            self._external_buffers.pop(ptr, None)
        for ptr, size in active.items():
            if self._external_buffers.get(ptr) == size:
                continue
            if ptr in self._external_buffers:
                self.store.unregister_buffer(ptr)
            status = self.store.register_buffer(ptr, size)
            if status not in (None, 0):
                raise RuntimeError(
                    f"Mooncake NPU buffer registration failed: ptr={ptr:#x} "
                    f"size={size} status={status}"
                )
            self._external_buffers[ptr] = size

    def _page_keys_for(self, keys: List[CacheEngineKey]) -> list[Optional[str]]:
        """Resolve page keys from canonical chunk or representative layer keys."""
        if not getattr(self, "_page_first_multi_buffer", False):
            return [None] * len(keys)

        page_keys_by_identity: dict[tuple[Any, ...], str] = {}
        page_keys: list[Optional[str]] = []
        for key in keys:
            if not isinstance(key, LayerCacheEngineKey):
                page_keys.append(mooncake_page_key(key, self._page_num_layers))
                continue
            identity = (
                key.model_name,
                key.world_size,
                key.worker_id,
                key.chunk_hash,
                key.dtype,
                key.tags,
                key.kv_group,
            )
            page_key = page_keys_by_identity.get(identity)
            if page_key is None:
                page_key = mooncake_page_key(key, self._page_num_layers)
                page_keys_by_identity[identity] = page_key
            page_keys.append(page_key)
        return page_keys

    def _page_aware_exists_many(self, keys: List[CacheEngineKey]) -> list[bool]:
        page_results = [0] * len(keys)
        complete, _ = self._complete_page_groups(keys)
        page_keys: list[Optional[str]] = [None] * len(keys)
        page_positions = []
        for page_key, indices in complete:
            for index in indices:
                page_keys[index] = page_key
                page_positions.append(index)
        if page_positions:
            queried_page_keys: list[str] = list(
                dict.fromkeys(key for key in page_keys if key is not None)
            )
            raw_page_results = self.store.batch_is_exist(queried_page_keys)
            trace_mooncake_keys(
                "lookup",
                queried_page_keys,
                raw_page_results,
                api="connector.page_aware_exists",
            )
            result_by_page = dict(
                zip(queried_page_keys, raw_page_results, strict=False)
            )
            for index in page_positions:
                page_key = page_keys[index]
                assert page_key is not None
                page_results[index] = result_by_page.get(page_key, 0)

        legacy_positions = [
            index for index, result in enumerate(page_results) if result != 1
        ]
        if legacy_positions:
            legacy_keys = [
                mooncake_legacy_key(keys[index]) for index in legacy_positions
            ]
            legacy_results = self.store.batch_is_exist(legacy_keys)
            trace_mooncake_keys(
                "lookup",
                legacy_keys,
                legacy_results,
                api="connector.page_aware_exists_legacy",
            )
            for index, result in zip(
                legacy_positions, legacy_results, strict=False
            ):
                page_results[index] = result
        return [result == 1 for result in page_results]

    def _complete_page_groups(
        self, keys: List[CacheEngineKey]
    ) -> tuple[list[tuple[str, list[int]]], list[int]]:
        grouped: dict[str, list[int]] = {}
        legacy_indices: list[int] = []
        for index, page_key in enumerate(self._page_keys_for(keys)):
            if page_key is None:
                legacy_indices.append(index)
                continue
            grouped.setdefault(page_key, []).append(index)

        complete: list[tuple[str, list[int]]] = []
        expected_layers = list(range(self._page_num_layers))
        for page_key, indices in grouped.items():
            ordered = sorted(
                indices,
                key=lambda index: keys[index].layer_id,  # type: ignore[attr-defined]
            )
            layer_ids = [
                keys[index].layer_id  # type: ignore[attr-defined]
                for index in ordered
            ]
            if layer_ids == expected_layers:
                complete.append((page_key, ordered))
            else:
                legacy_indices.extend(indices)
        legacy_indices.sort()
        return complete, legacy_indices

    @staticmethod
    def _has_zero_copy_storage(memory_obj: MemoryObj) -> bool:
        available = getattr(memory_obj, "has_tensor_storage", None)
        return bool(
            memory_obj.raw_tensor is not None if available is None else available
        )

    @staticmethod
    def _zero_copy_buffer(
        key: CacheEngineKey, memory_obj: MemoryObj
    ) -> tuple[int, int]:
        if isinstance(memory_obj, LayerPageMemoryObj):
            if not isinstance(key, LayerCacheEngineKey):
                raise ValueError("Layer page requires a layer cache key")
            return memory_obj.layer_data_ptr(key.layer_id), memory_obj.layer_size
        return memory_obj.data_ptr, memory_obj.get_size()

    def _allocate_zero_copy_buffers(
        self, keys: List[CacheEngineKey]
    ) -> tuple[
        list[Optional[MemoryObj]],
        list[tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]],
        str,
    ]:
        chunk_size = self.local_cpu_backend.metadata.chunk_size
        base_metadata = {}
        scaled_metadata = {}
        key_metadata = []
        for key in keys:
            valid_tokens = mooncake_valid_tokens(key, chunk_size)
            base_key = self._raw_layout_cache_key(key)
            metadata_key = (*base_key, valid_tokens)
            metadata = scaled_metadata.get(metadata_key)
            if metadata is None:
                base = base_metadata.get(base_key)
                if base is None:
                    base = self._metadata_for_raw_key(key)
                    base_metadata[base_key] = base
                shapes, dtypes, fmt, single_token_size = base
                if valid_tokens != chunk_size:
                    shapes = [
                        _layer_page_shape(
                            shape,
                            fmt,
                            valid_tokens,
                            full_tokens=chunk_size,
                        )
                        for shape in shapes
                    ]
                metadata = (shapes, dtypes, fmt, single_token_size)
                scaled_metadata[metadata_key] = metadata
            key_metadata.append(metadata)
        memory_objs: list[Optional[MemoryObj]] = []
        allocation_mode = "individual"
        first_shapes, first_dtypes, first_fmt, _ = key_metadata[0]
        uniform_metadata = all(
            shapes == first_shapes and dtypes == first_dtypes and fmt == first_fmt
            for shapes, dtypes, fmt, _ in key_metadata
        )
        if uniform_metadata:
            batched = self.local_cpu_backend.batched_allocate(
                first_shapes,
                first_dtypes,
                batch_size=len(keys),
                fmt=first_fmt,
                eviction=False,
                busy_loop=False,
                address_backed=getattr(self, "_page_first_multi_buffer", False),
            )
            if batched is not None:
                if len(batched) == len(keys) and all(
                    self._has_zero_copy_storage(obj) for obj in batched
                ):
                    memory_objs = list(batched)
                    allocation_mode = "batched"
                else:
                    for obj in batched:
                        if obj.is_valid():
                            obj.ref_count_down()
        if not memory_objs:
            memory_objs = [
                self.local_cpu_backend.allocate(shapes, dtypes, fmt)
                for shapes, dtypes, fmt, _ in key_metadata
            ]
        return memory_objs, key_metadata, allocation_mode

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return True

    def support_batched_contains(self) -> bool:
        return bool(
            getattr(
                self,
                "_sampled_layerwise_lookup",
                getattr(self.config, "experimental_sampled_layerwise_lookup", False),
            )
            and callable(getattr(self.store, "batch_is_exist", None))
        )

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0

        if getattr(self, "_page_first_multi_buffer", False):
            results = self._page_aware_exists_many(keys)
        else:
            key_strings = [key.to_string() for key in keys]
            raw_results = self.store.batch_is_exist(key_strings)
            trace_mooncake_keys(
                "lookup",
                key_strings,
                raw_results,
                api="connector.batched_contains",
            )
            results = [result == 1 for result in raw_results]
        hit_count = 0
        for result in results:
            if not result:
                break
            hit_count += 1
        return hit_count

    def batched_contains_layer_pages(self, keys: List[CacheEngineKey]) -> int:
        """Check only layer-merged page keys, without legacy-key fallback."""
        started = cold_start_perf_now() if cold_start_perf_enabled() else None
        page_keys = self._page_keys_for(keys)
        if any(key is None for key in page_keys):
            return 0
        batch_exists = getattr(self.store, "batch_is_exist", None)
        results = (
            batch_exists(page_keys)
            if callable(batch_exists)
            else [self.store.is_exist(key) for key in page_keys]
        )
        trace_mooncake_keys(
            "lookup",
            cast(list[str], page_keys),
            results,
            api="connector.batched_contains_layer_pages",
        )
        found = next(
            (index for index, result in enumerate(results) if result != 1),
            len(results),
        )
        cold_start_perf_log(
            logger,
            "mooncake_page_lookup",
            started=started,
            kv_groups=sorted({int(key.kv_group) for key in keys}),
            keys=len(keys),
            complete_pages=len(page_keys),
            found_pages=found,
            legacy_keys=0,
            status="ok" if found == len(page_keys) else "partial",
        )
        return found

    def support_batched_get_non_blocking(self) -> bool:
        """
        Mooncake only supports batched_get / batch_get_into, not per-key get().
        Layerwise retrieval uses StorageManager's blocking batched_get path.
        """
        return False

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """
        Fallback for callers that still invoke non-blocking get on Mooncake.
        Delegates to batched_get and applies prefix semantics.
        """
        if not keys:
            return []

        results = await self.batched_get(keys)
        memory_objs: list[MemoryObj] = []
        found_failure = False
        for result in results:
            if found_failure:
                if result is not None:
                    result.ref_count_down()
                continue
            if result is None:
                found_failure = True
                continue
            memory_objs.append(result)
        return memory_objs

    async def exists(self, key: CacheEngineKey) -> bool:
        key_string = key.to_string()
        result = self.store.is_exist(key_string)
        trace_mooncake_keys(
            "lookup", [key_string], result, api="connector.exists"
        )
        return bool(result)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        key_string = key.to_string()
        result = self.store.is_exist(key_string)
        trace_mooncake_keys(
            "lookup", [key_string], result, api="connector.exists_sync"
        )
        return bool(result)

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get operation - the only supported get method.
        Uses batch_get_into (with metadata) or batch_get_buffer (without metadata).
        """
        if not keys:
            return []

        # Check if we have metadata for zero-copy operations
        if self.save_chunk_meta:
            # Use legacy mode with metadata stored in remote
            return await self._batch_get_buffer(keys)
        else:
            # Use optimized mode with local metadata
            return await self._batch_get_into(keys)

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if getattr(self, "_page_first_multi_buffer", False):
            return self.batched_contains(keys)
        num_hit_counts = 0
        for key in keys:
            key_string = key.to_string()
            result = self.store.is_exist(key_string)
            trace_mooncake_keys(
                "lookup",
                [key_string],
                result,
                api="connector.batched_async_contains",
                lookup_id=lookup_id,
            )
            if not result:
                break
            num_hit_counts += 1
        return num_hit_counts

    async def _batch_get_pages(
        self,
        keys: List[CacheEngineKey],
        page_groups: list[tuple[str, list[int]]],
    ) -> List[Optional[MemoryObj]]:
        perf_enabled = cold_start_perf_enabled()
        perf_started = cold_start_perf_now() if perf_enabled else 0.0
        results: List[Optional[MemoryObj]] = [None] * len(keys)
        page_keys = [keys[index] for _, indices in page_groups for index in indices]
        allocation_started = cold_start_perf_now() if perf_enabled else 0.0
        memory_objs, _, _ = self._allocate_zero_copy_buffers(page_keys)
        allocation_ms = (
            (cold_start_perf_now() - allocation_started) * 1000
            if perf_enabled
            else 0.0
        )

        submission_started = cold_start_perf_now() if perf_enabled else 0.0
        submitted_groups: list[tuple[str, list[int], int]] = []
        all_buffer_ptrs: list[list[int]] = []
        all_buffer_sizes: list[list[int]] = []
        offset = 0
        for page_key, indices in page_groups:
            end = offset + len(indices)
            page_objects = [
                obj
                for obj in memory_objs[offset:end]
                if obj is not None
                and self._has_zero_copy_storage(obj)
            ]
            if len(page_objects) != len(indices):
                offset = end
                continue
            submitted_groups.append((page_key, indices, offset))
            all_buffer_ptrs.append([obj.data_ptr for obj in page_objects])
            all_buffer_sizes.append([obj.get_size() for obj in page_objects])
            offset = end
        submission_ms = (
            (cold_start_perf_now() - submission_started) * 1000
            if perf_enabled
            else 0.0
        )
        transfer_ms = 0.0
        completed_pages = 0
        result_status = "empty"

        try:
            if not submitted_groups:
                return results
            transfer_started = cold_start_perf_now() if perf_enabled else 0.0
            statuses = await asyncio.to_thread(
                self.store.batch_get_into_multi_buffers,
                [page_key for page_key, _, _ in submitted_groups],
                all_buffer_ptrs,
                all_buffer_sizes,
            )
            transfer_ms = (
                (cold_start_perf_now() - transfer_started) * 1000
                if perf_enabled
                else 0.0
            )
            result_status = (
                "ok"
                if len(submitted_groups) == len(page_groups)
                else "partial"
            )
            for group_index, (page_key, indices, offset) in enumerate(
                submitted_groups
            ):
                if group_index >= len(statuses):
                    result_status = "partial"
                    logger.warning(
                        "Mooncake page get omitted status for page=%s", page_key
                    )
                    continue
                expected_bytes = sum(all_buffer_sizes[group_index])
                page_status = statuses[group_index]
                if page_status != expected_bytes:
                    logger.warning(
                        "Mooncake page get failed or was short: page=%s "
                        "status=%s expected_bytes=%d",
                        page_key,
                        page_status,
                        expected_bytes,
                    )
                    result_status = "partial"
                    continue
                completed_pages += 1
                for position, index in enumerate(indices, start=offset):
                    memory_obj = memory_objs[position]
                    assert memory_obj is not None
                    results[index] = memory_obj
                    memory_objs[position] = None

            return results
        except Exception as exc:
            result_status = "error"
            logger.error("Mooncake page-first get failed: %s", exc)
            return results
        finally:
            for memory_obj in memory_objs:
                if memory_obj is not None and memory_obj.is_valid():
                    memory_obj.ref_count_down()
            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_page_get",
                    started=perf_started,
                    kv_groups=sorted(
                        {int(getattr(key, "kv_group", 0)) for key in page_keys}
                    ),
                    pages=len(page_groups),
                    submitted_pages=len(submitted_groups),
                    completed_pages=completed_pages,
                    buffers=sum(len(indices) for _, indices in page_groups),
                    bytes=sum(sum(sizes) for sizes in all_buffer_sizes),
                    allocation_ms=round(allocation_ms, 3),
                    submission_ms=round(submission_ms, 3),
                    transfer_ms=round(transfer_ms, 3),
                    status=result_status,
                )

    async def batched_get_layer_pages(
        self, keys: List[CacheEngineKey]
    ) -> list[LayerPageMemoryObj]:
        """Load pages identified by canonical chunk or representative layer keys."""
        perf_enabled = cold_start_perf_enabled()
        perf_started = cold_start_perf_now() if perf_enabled else 0.0
        if not self._layer_merged_pages:
            raise RuntimeError("Layer-merged page objects are not enabled")
        if not keys:
            raise ValueError("Layer-page retrieval requires page keys")
        base_keys = [
            key.without_layer() if isinstance(key, LayerCacheEngineKey) else key
            for key in keys
        ]
        resolved_page_keys = self._page_keys_for(base_keys)
        if any(page_key is None for page_key in resolved_page_keys):
            raise ValueError("Layer-page retrieval requires page-first storage")
        page_keys = cast(list[str], resolved_page_keys)
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("Layer-page retrieval requires unique chunk keys")

        first_key = base_keys[0]
        first = self._metadata_for_raw_key(first_key)
        shapes, dtypes, fmt, _ = first
        if len(shapes) != 1 or len(dtypes) != 1 or any(
            key.kv_group != first_key.kv_group or key.dtype != first_key.dtype
            for key in base_keys[1:]
        ):
            raise ValueError("Layer-page retrieval requires one homogeneous tensor")
        metadata_ms = (
            (cold_start_perf_now() - perf_started) * 1000 if perf_enabled else 0.0
        )
        allocation_started = cold_start_perf_now() if perf_enabled else 0.0
        pages = self.local_cpu_backend.batched_allocate_layer_pages(
            shapes,
            dtypes,
            len(page_keys),
            self._page_num_layers,
            fmt,
            valid_tokens=[
                mooncake_valid_tokens(key, self.local_cpu_backend.metadata.chunk_size)
                for key in base_keys
            ],
            full_tokens=self.local_cpu_backend.metadata.chunk_size,
        )
        allocation_ms = (
            (cold_start_perf_now() - allocation_started) * 1000
            if perf_enabled
            else 0.0
        )
        if pages is None:
            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_page_get",
                    started=perf_started,
                    layout="layer_merged",
                    kv_group=int(first_key.kv_group),
                    kv_groups=[int(first_key.kv_group)],
                    layers=self._page_num_layers,
                    pages=len(page_keys),
                    submitted_pages=0,
                    completed_pages=0,
                    buffers=0,
                    bytes=0,
                    metadata_ms=round(metadata_ms, 3),
                    allocation_ms=round(allocation_ms, 3),
                    buffer_setup_ms=0.0,
                    transfer_ms=0.0,
                    publish_ms=0.0,
                    status="allocation_failed",
                )
            return []

        setup_started = cold_start_perf_now() if perf_enabled else 0.0
        sizes = [[page.layer_size] * self._page_num_layers for page in pages]
        ptrs = [
            [page.layer_data_ptr(layer) for layer in range(self._page_num_layers)]
            for page in pages
        ]
        expected = [sum(page_sizes) for page_sizes in sizes]
        buffer_setup_ms = (
            (cold_start_perf_now() - setup_started) * 1000
            if perf_enabled
            else 0.0
        )
        transfer_started = cold_start_perf_now() if perf_enabled else 0.0
        transfer = asyncio.create_task(
            asyncio.to_thread(
                self.store.batch_get_into_multi_buffers,
                page_keys,
                ptrs,
                sizes,
            )
        )
        transfer_ms: float | None = None
        publish_ms = 0.0
        status = "error"
        try:
            statuses = await asyncio.shield(transfer)
            transfer_ms = (
                (cold_start_perf_now() - transfer_started) * 1000
                if perf_enabled
                else 0.0
            )
            if list(statuses) != expected:
                raise RuntimeError(
                    f"Mooncake layer-page get returned {list(statuses)}, "
                    f"expected {expected}"
                )
            publish_started = cold_start_perf_now() if perf_enabled else 0.0
            self.local_cpu_backend.batched_submit_layer_pages(
                base_keys, pages
            )
            publish_ms = (
                (cold_start_perf_now() - publish_started) * 1000
                if perf_enabled
                else 0.0
            )
            status = "ok"
            return pages
        except asyncio.CancelledError:
            status = "cancelled"
            try:
                await transfer
            finally:
                for page in pages:
                    if page.is_valid():
                        page.ref_count_down()
            raise
        except Exception:
            for page in pages:
                if page.is_valid():
                    page.ref_count_down()
            raise
        finally:
            if perf_enabled:
                if transfer_ms is None:
                    transfer_ms = (cold_start_perf_now() - transfer_started) * 1000
                cold_start_perf_log(
                    logger,
                    "mooncake_page_get",
                    started=perf_started,
                    layout="layer_merged",
                    kv_group=int(first_key.kv_group),
                    kv_groups=[int(first_key.kv_group)],
                    layers=self._page_num_layers,
                    pages=len(page_keys),
                    submitted_pages=len(page_keys),
                    completed_pages=len(page_keys) if status == "ok" else 0,
                    buffers=len(page_keys) * self._page_num_layers,
                    bytes=sum(expected),
                    metadata_ms=round(metadata_ms, 3),
                    allocation_ms=round(allocation_ms, 3),
                    buffer_setup_ms=round(buffer_setup_ms, 3),
                    transfer_ms=round(transfer_ms, 3),
                    publish_ms=round(publish_ms, 3),
                    status=status,
                )

    async def _batch_get_into(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not getattr(self, "_page_first_multi_buffer", False):
            return await self._batch_get_into_legacy(keys)

        perf_enabled = cold_start_perf_enabled()
        lookup_started = cold_start_perf_now() if perf_enabled else 0.0
        complete_groups, legacy_indices = self._complete_page_groups(keys)
        if not complete_groups:
            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_page_lookup",
                    started=lookup_started,
                    kv_groups=sorted(
                        {int(getattr(key, "kv_group", 0)) for key in keys}
                    ),
                    keys=len(keys),
                    complete_pages=0,
                    found_pages=0,
                    legacy_keys=len(keys),
                    status="legacy",
                )
            return await self._batch_get_into_legacy(keys)

        page_keys = [page_key for page_key, _ in complete_groups]
        page_exists = await asyncio.to_thread(
            self.store.batch_is_exist,
            page_keys,
        )
        trace_mooncake_keys(
            "lookup",
            page_keys,
            page_exists,
            api="connector.batch_get_page_lookup",
        )
        page_groups: list[tuple[str, list[int]]] = []
        for group, exists in zip(complete_groups, page_exists, strict=False):
            if exists == 1:
                page_groups.append(group)
            else:
                legacy_indices.extend(group[1])
        if len(page_exists) < len(complete_groups):
            for group in complete_groups[len(page_exists) :]:
                legacy_indices.extend(group[1])
        if perf_enabled:
            cold_start_perf_log(
                logger,
                "mooncake_page_lookup",
                started=lookup_started,
                kv_groups=sorted(
                    {int(getattr(key, "kv_group", 0)) for key in keys}
                ),
                keys=len(keys),
                complete_pages=len(complete_groups),
                found_pages=len(page_groups),
                legacy_keys=len(set(legacy_indices)),
                status="ok" if not legacy_indices else "partial",
            )

        results = (
            await self._batch_get_pages(keys, page_groups)
            if page_groups
            else [None] * len(keys)
        )

        if legacy_indices:
            legacy_indices = sorted(set(legacy_indices))
            legacy_results = await self._batch_get_into_legacy(
                [keys[index] for index in legacy_indices]
            )
            for index, memory_obj in zip(
                legacy_indices, legacy_results, strict=False
            ):
                results[index] = memory_obj
        return results

    async def _batch_get_into_legacy(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Zero-copy batch get using batch_get_into when metadata is available locally.
        This is used when save_chunk_meta=False (metadata not stored remotely).
        """
        if not self.meta_shapes or not self.meta_dtypes or not self.meta_fmt:
            logger.error(
                f"Metadata required for batch_get_into but not available: "
                f"meta_shapes={self.meta_shapes}, "
                f"meta_dtypes={self.meta_dtypes}, "
                f"meta_fmt={self.meta_fmt}"
            )
            return [None] * len(keys)

        logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")

        perf_enabled = cold_start_perf_enabled()
        perf_started = cold_start_perf_now() if perf_enabled else 0.0
        valid_idx: list[int] = []
        key_strs: list[str] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        single_token_sizes: dict[int, int] = {}
        allocation_started = cold_start_perf_now() if perf_enabled else 0.0
        memory_objs, key_metadata, allocation_mode = (
            self._allocate_zero_copy_buffers(keys)
        )
        allocation_ms = (
            (cold_start_perf_now() - allocation_started) * 1000
            if perf_enabled
            else 0.0
        )

        for i, (key, metadata_entry, obj) in enumerate(
            zip(keys, key_metadata, memory_objs, strict=True)
        ):
            single_token_size = metadata_entry[3]
            if obj is not None and self._has_zero_copy_storage(obj):
                valid_idx.append(i)
                single_token_sizes[i] = single_token_size

                # Prepare the argument lists for the C++ call
                key_strs.append(mooncake_legacy_key(key))
                buffer_ptrs.append(obj.data_ptr)
                buffer_sizes.append(obj.get_size())

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_legacy_get",
                    started=perf_started,
                    kv_groups=sorted(
                        {int(getattr(key, "kv_group", 0)) for key in keys}
                    ),
                    keys=len(keys),
                    valid_buffers=0,
                    bytes=0,
                    allocation_mode=allocation_mode,
                    allocation_ms=round(allocation_ms, 3),
                    transfer_ms=0.0,
                    status="no_buffers",
                )
            return [None] * len(keys)

        try:
            # Single RPC call for multiple chunks
            logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")
            transfer_started = cold_start_perf_now() if perf_enabled else 0.0
            bytes_read_list = await asyncio.to_thread(
                self.store.batch_get_into, key_strs, buffer_ptrs, buffer_sizes
            )
            transfer_ms = (
                (cold_start_perf_now() - transfer_started) * 1000
                if perf_enabled
                else 0.0
            )
            logger.debug(f"batch_get_into returned: {bytes_read_list}")
            if bytes_read_list is None or len(bytes_read_list) != len(valid_idx):
                raise RuntimeError(
                    "Mooncake batch_get_into returned an invalid result count: "
                    f"expected={len(valid_idx)}, actual="
                    f"{0 if bytes_read_list is None else len(bytes_read_list)}"
                )

            # Assemble the final result list
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=True):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()  # type: ignore
                    memory_objs[i] = None
                    continue

                try:
                    results[i] = self._reshape_partial_chunk_with_token_size(
                        memory_objs[i],  # type: ignore
                        n_read,
                        single_token_sizes[i],
                    )
                except Exception as exc:
                    logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                    memory_objs[i].ref_count_down()  # type: ignore
                    memory_objs[i] = None

            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_legacy_get",
                    started=perf_started,
                    kv_groups=sorted(
                        {int(getattr(key, "kv_group", 0)) for key in keys}
                    ),
                    keys=len(keys),
                    valid_buffers=len(valid_idx),
                    bytes=sum(buffer_sizes),
                    bytes_read=sum(
                        max(int(value), 0) for value in bytes_read_list
                    ),
                    allocation_mode=allocation_mode,
                    allocation_ms=round(allocation_ms, 3),
                    transfer_ms=round(transfer_ms, 3),
                    status=(
                        "ok"
                        if all(results[index] is not None for index in valid_idx)
                        else "partial"
                    ),
                )
            return results

        except Exception as exc:
            logger.error(f"batch_get_into threw exception: {str(exc)}")
            # Release any buffers we successfully allocated
            for i in valid_idx:
                if memory_objs[i] is not None:
                    memory_objs[i].ref_count_down()
            if perf_enabled:
                cold_start_perf_log(
                    logger,
                    "mooncake_legacy_get",
                    started=perf_started,
                    kv_groups=sorted(
                        {int(getattr(key, "kv_group", 0)) for key in keys}
                    ),
                    keys=len(keys),
                    valid_buffers=len(valid_idx),
                    bytes=sum(buffer_sizes),
                    allocation_mode=allocation_mode,
                    allocation_ms=round(allocation_ms, 3),
                    status="error",
                    error=type(exc).__name__,
                )
            return [None] * len(keys)

    async def _batch_get_buffer(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get using batch_get_buffer when metadata is stored remotely.
        This is used when save_chunk_meta=True (metadata stored with data).
        """
        key_strs = [key.to_string() for key in keys]

        try:
            buffers = await asyncio.to_thread(self.store.batch_get_buffer, key_strs)
        except Exception as e:
            logger.error(f"batch_get_buffer failed: {str(e)}")
            return [None] * len(keys)

        results: list[Optional[MemoryObj]] = []
        for i, buffer in enumerate(buffers):
            if buffer is None:
                logger.warning(f"Buffer {i} is None for key {key_strs[i]}")
                results.append(None)
                continue
            try:
                memory_obj = self._process_buffer_with_metadata(buffer)
                results.append(memory_obj)
            except Exception as e:
                logger.error(
                    f"Failed to process buffer {i} for key {key_strs[i]}: {str(e)}"
                )
                results.append(None)
        return results

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Single get method - NOT SUPPORTED.
        Use batched_get instead for all operations.
        """
        logger.error("Single get operation is not supported. Use batched_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batched_get([key]) instead."
        )

    def _process_buffer_with_metadata(self, buffer: bytes) -> Optional[MemoryObj]:
        """
        Process buffer that contains metadata + data.
        Used when save_chunk_meta=True (metadata stored remotely).
        """
        retrieved_view = memoryview(buffer)
        metadata_bytes = retrieved_view[: self.remote_metadata_bytes]
        if metadata_bytes is None or len(metadata_bytes) != self.remote_metadata_bytes:
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        assert len(retrieved_view) == metadata.length + self.remote_metadata_bytes

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if memory_obj.raw_tensor is not None:
            temp_tensor = torch.frombuffer(
                buffer,
                dtype=torch.uint8,
                offset=self.remote_metadata_bytes,
                count=metadata.length,
            )

            memory_obj.raw_tensor.copy_(temp_tensor)
            return memory_obj
        else:
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Put operation with metadata-consistent handling.
        Uses put_from (without metadata) or
        put_parts (with metadata) to match get behavior.
        """
        key_str = key.to_string()

        # Check metadata handling mode to match get behavior
        if self.save_chunk_meta:
            # Use put_parts with metadata stored remotely
            await self._put_with_metadata(key_str, memory_obj)
        else:
            # Use put_from without metadata (zero-copy)
            await self._put_without_metadata(key, memory_obj)

    def support_batched_put(self) -> bool:
        return True

    def requires_put_completion(self) -> bool:
        return True

    async def _run_blocking_put(
        self,
        operation: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        memory_objs: List[MemoryObj],
    ) -> Any:
        """Run a Mooncake put without releasing its source buffers early."""
        for memory_obj in memory_objs:
            memory_obj.ref_count_up()

        task = asyncio.create_task(asyncio.to_thread(func, *args))
        self._inflight_put_tasks.add(task)

        def release_buffers(done: asyncio.Task[Any]) -> None:
            self._inflight_put_tasks.discard(done)
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()
            if not done.cancelled():
                done.exception()

        task.add_done_callback(release_buffers)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self.config.transfer_timeout
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Mooncake {operation} timed out after "
                f"{self.config.transfer_timeout}s"
            ) from e

    @staticmethod
    def _check_put_status(operation: str, status: Any) -> None:
        if status is not None and status != 0:
            raise RuntimeError(f"Mooncake {operation} failed with status {status}")

    @staticmethod
    def _check_batched_put_status(
        keys: List[CacheEngineKey], statuses: Any
    ) -> None:
        if statuses is None:
            return
        if len(statuses) != len(keys):
            raise RuntimeError(
                "Mooncake batch_put_from returned "
                f"{len(statuses)} statuses for {len(keys)} keys"
            )
        for key, status in zip(keys, statuses, strict=True):
            if status != 0:
                raise RuntimeError(
                    f"Mooncake batch_put_from failed for {key}: status {status}"
                )

    @staticmethod
    def _validate_external_buffer_owners(
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
    ) -> None:
        if any(
            not ptrs or len(ptrs) != len(sizes) or any(size <= 0 for size in sizes)
            for ptrs, sizes in zip(buffer_ptrs, buffer_sizes, strict=True)
        ):
            raise ValueError("Direct page pointer/size layout is invalid")
        ranges = sorted(
            (
                int(owner.untyped_storage().data_ptr()),
                int(owner.untyped_storage().nbytes()),
            )
            for owner in owners
        )
        starts = [base for base, _ in ranges]
        prefix_ends: list[int] = []
        for base, capacity in ranges:
            prefix_ends.append(
                max(prefix_ends[-1] if prefix_ends else 0, base + capacity)
            )
        for ptrs, sizes in zip(buffer_ptrs, buffer_sizes, strict=True):
            for ptr, size in zip(ptrs, sizes, strict=True):
                index = bisect_right(starts, ptr) - 1
                if index < 0 or ptr + size > prefix_ends[index]:
                    raise ValueError(
                        "Direct page buffer lies outside registered storage"
                    )

    @staticmethod
    def _raw_layout_cache_key(key: CacheEngineKey) -> tuple:
        return (
            int(key.kv_group),
            key.dtype,
            key.tags,
        )

    def _external_page_key(
        self, key: CacheEngineKey, sizes: List[int]
    ) -> str:
        valid_tokens = mooncake_valid_tokens(
            key, self.local_cpu_backend.metadata.chunk_size
        )
        layer_key = isinstance(key, LayerCacheEngineKey)
        cache = getattr(self, "_external_page_bytes", None)
        if cache is None:
            cache = self._external_page_bytes = {}
        cache_key = (
            *self._raw_layout_cache_key(key),
            key.layer_id if layer_key else None,
            valid_tokens,
        )
        expected = cache.get(cache_key)
        if expected is None:
            expected = self._metadata_for_raw_key(key)[3] * valid_tokens
            if not layer_key:
                expected *= self._page_num_layers
            cache[cache_key] = expected
        actual = sum(sizes)
        if actual != expected:
            raise ValueError(
                "Direct page buffer byte count mismatch: "
                f"key={key} expected={expected} actual={actual}"
            )
        return (
            key.to_string()
            if layer_key
            else mooncake_page_key(key, self._page_num_layers)
        )

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ):
        """
        Batched put with clear split by metadata mode.
        - save_chunk_meta False: use Mooncake's batch_put_from (zero-copy).
        - save_chunk_meta True: no batch API; fall back to sequential put_parts.
        """
        if not keys:
            return

        if self.save_chunk_meta:
            await self._batched_put_with_metadata(keys, memory_objs)
        else:
            await self._batched_put_zero_copy(keys, memory_objs)

    async def batched_put_external_pages(
        self,
        keys: List[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        ready_event: Any,
        req_id: str,
    ) -> None:
        """Write registered accelerator buffers with page or legacy keys."""
        if self.save_chunk_meta or not self._page_first_multi_buffer:
            raise RuntimeError("Direct page store requires metadata-free page mode")
        if not keys:
            return
        if not (len(keys) == len(buffer_ptrs) == len(buffer_sizes)):
            raise ValueError("Direct page keys and buffers have different lengths")
        if any(
            len(ptrs) != len(sizes)
            for ptrs, sizes in zip(buffer_ptrs, buffer_sizes, strict=True)
        ):
            raise ValueError("Direct page pointer and size counts differ")
        self._validate_external_buffer_owners(buffer_ptrs, buffer_sizes, owners)
        legacy_objects = sum(isinstance(key, LayerCacheEngineKey) for key in keys)
        page_keys = [
            self._external_page_key(key, sizes)
            for key, sizes in zip(keys, buffer_sizes, strict=True)
        ]
        started = cold_start_perf_now() if cold_start_perf_enabled() else None

        def put() -> Any:
            if owners and owners[0].device.type == "npu":
                torch.npu.set_device(owners[0].device)
            wait_started = cold_start_perf_now() if started is not None else None
            if ready_event is not None:
                ready_event.synchronize()
            wait_ms = (
                (cold_start_perf_now() - wait_started) * 1000
                if wait_started is not None
                else 0.0
            )
            self._register_external_owners(owners)
            transfer_started = cold_start_perf_now() if started is not None else None
            statuses = self.store.batch_put_from_multi_buffers(
                page_keys, buffer_ptrs, buffer_sizes, self.replica_config
            )
            transfer_ms = (
                (cold_start_perf_now() - transfer_started) * 1000
                if transfer_started is not None
                else 0.0
            )
            return statuses, wait_ms, transfer_ms

        async with self._external_put_lock:
            task = asyncio.create_task(asyncio.to_thread(put))
            self._inflight_put_tasks.add(task)
            task.add_done_callback(self._inflight_put_tasks.discard)
            try:
                statuses, wait_ms, transfer_ms = await asyncio.wait_for(
                    asyncio.shield(task), timeout=self.config.transfer_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Mooncake direct page put exceeded %ss; waiting for the "
                    "uncancellable native read before releasing source buffers",
                    self.config.transfer_timeout,
                )
                try:
                    statuses, wait_ms, transfer_ms = await task
                except BaseException as native_error:
                    raise TimeoutError(
                        "Mooncake direct page put failed after timing out"
                    ) from native_error
        trace_mooncake_keys(
            "put",
            page_keys,
            statuses if statuses is not None else 0,
            api="connector.direct_npu_page_put",
        )
        if statuses is not None:
            if len(statuses) != len(page_keys):
                raise RuntimeError(
                    "Mooncake direct page put returned invalid status count"
                )
            failed = [
                page_key
                for page_key, status in zip(page_keys, statuses, strict=True)
                if status != 0
            ]
            if failed:
                error = RuntimeError(
                    f"Mooncake direct page put failed: {failed[:4]}"
                )
                error.failed_pages = failed  # type: ignore[attr-defined]
                raise error
        if started is not None:
            cold_start_perf_log(
                logger,
                "direct_npu_page_put",
                started=started,
                req_id=req_id,
                pages=len(page_keys) - legacy_objects,
                legacy_objects=legacy_objects,
                buffers=sum(map(len, buffer_ptrs)),
                bytes=sum(map(sum, buffer_sizes)),
                format=(
                    "legacy_tail"
                    if legacy_objects == len(keys)
                    else "mixed" if legacy_objects else "page"
                ),
                event_wait_ms=wait_ms,
                transfer_ms=transfer_ms,
                source_device=(
                    getattr(getattr(owners[0], "device", None), "type", "unknown")
                    if owners
                    else "unknown"
                ),
                status="ok",
            )

    async def batched_get_external_pages(
        self,
        keys: List[CacheEngineKey],
        buffer_ptrs: List[List[int]],
        buffer_sizes: List[List[int]],
        owners: tuple[Any, ...],
        req_id: str,
    ) -> None:
        """Read exact Mooncake pages directly into accelerator tensor storage."""
        if self.save_chunk_meta or not self._page_first_multi_buffer:
            raise RuntimeError("Direct page load requires metadata-free page mode")
        if not keys:
            return
        if not (len(keys) == len(buffer_ptrs) == len(buffer_sizes)):
            raise ValueError("Direct page keys and buffers have different lengths")
        self._validate_external_buffer_owners(buffer_ptrs, buffer_sizes, owners)

        page_keys = [
            self._external_page_key(key, sizes)
            for key, sizes in zip(keys, buffer_sizes, strict=True)
        ]

        started = cold_start_perf_now() if cold_start_perf_enabled() else None

        def get() -> Any:
            if owners and owners[0].device.type == "npu":
                torch.npu.set_device(owners[0].device)
            self._register_external_owners(owners)
            transfer_started = cold_start_perf_now() if started is not None else None
            statuses = self.store.batch_get_into_multi_buffers(
                page_keys, buffer_ptrs, buffer_sizes
            )
            transfer_ms = (
                (cold_start_perf_now() - transfer_started) * 1000
                if transfer_started is not None
                else 0.0
            )
            return statuses, transfer_ms

        async with self._external_put_lock:
            task = asyncio.create_task(asyncio.to_thread(get))
            self._inflight_put_tasks.add(task)
            task.add_done_callback(self._inflight_put_tasks.discard)
            try:
                statuses, transfer_ms = await asyncio.wait_for(
                    asyncio.shield(task), timeout=self.config.transfer_timeout
                )
            except asyncio.TimeoutError:
                # Native transfers cannot be cancelled safely while their destination
                # tensors are live. Drain the call, then force the caller to fallback.
                try:
                    await task
                except BaseException as native_error:
                    raise TimeoutError(
                        "Mooncake direct page load failed after timing out"
                    ) from native_error
                raise TimeoutError(
                    "Mooncake direct page load timed out"
                ) from None

        if statuses is None or len(statuses) != len(page_keys):
            raise RuntimeError(
                "Mooncake direct page load returned invalid status count"
            )
        failed = [
            (page_key, status, sum(sizes))
            for page_key, status, sizes in zip(
                page_keys, statuses, buffer_sizes, strict=True
            )
            if status != sum(sizes)
        ]
        if failed:
            error = RuntimeError(
                f"Mooncake direct page load failed or was short: {failed[:4]}"
            )
            error.failed_pages = failed  # type: ignore[attr-defined]
            raise error
        trace_mooncake_keys(
            "get", page_keys, statuses, api="connector.direct_npu_page_get"
        )
        if started is not None:
            cold_start_perf_log(
                logger,
                "direct_npu_page_get",
                started=started,
                req_id=req_id,
                pages=len(page_keys),
                buffers=sum(map(len, buffer_ptrs)),
                bytes=sum(map(sum, buffer_sizes)),
                transfer_ms=transfer_ms,
                status="ok",
            )

    def batched_external_pages_exist(
        self, keys: List[CacheEngineKey]
    ) -> List[bool]:
        """Check arbitrary existing-format page keys in one Mooncake call."""
        page_keys = [mooncake_page_key(key, self._page_num_layers) for key in keys]
        results = self.store.batch_is_exist(page_keys)
        trace_mooncake_keys(
            "lookup",
            page_keys,
            results,
            api="connector.direct_npu_page_exists",
        )
        if results is None or len(results) != len(keys):
            raise RuntimeError("Mooncake direct page lookup returned invalid results")
        return [result == 1 for result in results]

    async def _batched_put_zero_copy(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        if not getattr(self, "_page_first_multi_buffer", False):
            await self._batched_put_zero_copy_legacy(keys, memory_objs)
            return

        complete_groups, legacy_indices = self._complete_page_groups(keys)
        page_groups: list[tuple[str, list[int]]] = []
        for page_key, indices in complete_groups:
            valid_tokens = mooncake_valid_tokens(
                keys[indices[0]], self.local_cpu_backend.metadata.chunk_size
            )
            expected_layer_bytes = (
                self._metadata_for_raw_key(keys[indices[0]])[3] * valid_tokens
            )
            if all(
                self._zero_copy_buffer(keys[index], memory_objs[index])[1]
                == expected_layer_bytes
                for index in indices
            ):
                page_groups.append((page_key, indices))
            else:
                legacy_indices.extend(indices)

        if page_groups:
            page_keys = [page_key for page_key, _ in page_groups]
            page_buffers = [
                [
                    self._zero_copy_buffer(keys[index], memory_objs[index])
                    for index in indices
                ]
                for _, indices in page_groups
            ]
            all_buffer_ptrs = [[ptr for ptr, _ in page] for page in page_buffers]
            all_buffer_sizes = [[size for _, size in page] for page in page_buffers]
            page_memory_objs = list(
                {
                    id(memory_objs[index]): memory_objs[index]
                    for _, indices in page_groups
                    for index in indices
                }.values()
            )
            put_started = (
                cold_start_perf_now() if cold_start_perf_enabled() else None
            )
            statuses = await self._run_blocking_put(
                "batch_put_from_multi_buffers",
                self.store.batch_put_from_multi_buffers,
                (
                    page_keys,
                    all_buffer_ptrs,
                    all_buffer_sizes,
                    self.replica_config,
                ),
                page_memory_objs,
            )
            trace_mooncake_keys(
                "put",
                page_keys,
                statuses if statuses is not None else 0,
                api="connector.batch_put_from_multi_buffers",
            )
            if statuses is not None:
                if len(statuses) != len(page_keys):
                    raise RuntimeError(
                        "Mooncake page put returned "
                        f"{len(statuses)} statuses for {len(page_keys)} pages"
                    )
                for page_key, status in zip(page_keys, statuses, strict=True):
                    if status != 0:
                        raise RuntimeError(
                            "Mooncake page put failed for "
                            f"{page_key}: status {status}"
                        )
            if put_started is not None:
                cold_start_perf_log(
                    logger,
                    "mooncake_page_put",
                    started=put_started,
                    pages=len(page_keys),
                    buffers=sum(map(len, all_buffer_ptrs)),
                    bytes=sum(map(sum, all_buffer_sizes)),
                    kv_groups=sorted(
                        {
                            int(keys[indices[0]].kv_group)
                            for _, indices in page_groups
                        }
                    ),
                    first_page_key=page_keys[0],
                    last_page_key=page_keys[-1],
                    legacy_objects=len(legacy_indices),
                    status="ok",
                )
        if legacy_indices:
            legacy_indices = sorted(set(legacy_indices))
            await self._batched_put_zero_copy_legacy(
                [keys[index] for index in legacy_indices],
                [memory_objs[index] for index in legacy_indices],
            )

    async def _batched_put_zero_copy_legacy(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        key_strs = [mooncake_legacy_key(key) for key in keys]
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []
        for key, obj in zip(keys, memory_objs, strict=True):
            if not self._has_zero_copy_storage(obj):
                raise ValueError("Mooncake zero-copy put requires tensor storage")
            ptr, size = self._zero_copy_buffer(key, obj)
            buffer_ptrs.append(ptr)
            buffer_sizes.append(size)

        statuses = await self._run_blocking_put(
            "batch_put_from",
            self.store.batch_put_from,
            (key_strs, buffer_ptrs, buffer_sizes, self.replica_config),
            memory_objs,
        )
        trace_mooncake_keys(
            "put",
            key_strs,
            statuses if statuses is not None else 0,
            api="connector.batch_put_from",
        )
        self._check_batched_put_status(keys, statuses)

    async def _batched_put_with_metadata(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        for key, obj in zip(keys, memory_objs, strict=False):
            await self._put_with_metadata(key.to_string(), obj)

    async def _put_without_metadata(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ):
        """
        Zero-copy put using put_from when metadata is not stored remotely.
        This is used when save_chunk_meta=False (matches _batch_get_into).
        """
        try:
            if not self._has_zero_copy_storage(memory_obj):
                raise ValueError("Mooncake zero-copy put requires tensor storage")
            buffer_ptr, buffer_size = self._zero_copy_buffer(key, memory_obj)

            status = await self._run_blocking_put(
                "put_from",
                self.store.put_from,
                (
                    mooncake_legacy_key(key),
                    buffer_ptr,
                    buffer_size,
                    self.replica_config,
                ),
                [memory_obj],
            )
            trace_mooncake_keys(
                "put",
                [key.to_string()],
                0 if status is None else status,
                api="connector.put_from",
            )
            self._check_put_status("put_from", status)
        except Exception as e:
            logger.error(
                f"Failed to put key {key} using put_from: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    async def _put_with_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Put using put_parts when metadata is stored remotely.
        This is used when save_chunk_meta=True (matches _batch_get_buffer).
        """
        try:
            # Serialize data and metadata
            kv_bytes = memory_obj.byte_array
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()
            assert len(metadata_bytes) == self.remote_metadata_bytes

            status = await self._run_blocking_put(
                "put_parts",
                self.store.put_parts,
                (key_str, metadata_bytes, kv_bytes),
                [memory_obj],
            )
            trace_mooncake_keys(
                "put",
                [key_str],
                0 if status is None else status,
                api="connector.put_parts",
            )
            self._check_put_status("put_parts", status)
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_parts: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        if self._inflight_put_tasks:
            await asyncio.gather(
                *tuple(self._inflight_put_tasks), return_exceptions=True
            )

        # Unregister buffer before closing the store
        self._unregister_cpu_buffer()
        for ptr in tuple(self._external_buffers):
            self.store.unregister_buffer(ptr)
        self._external_buffers.clear()

        self.store.close()
        logger.info("Closed the mooncake store connection")
