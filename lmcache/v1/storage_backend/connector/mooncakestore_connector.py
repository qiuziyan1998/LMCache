# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, no_type_check
import asyncio
import json
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.mooncake_layout import mooncake_page_key
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
    protocol_fallback: Optional[str] = None
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
            protocol_fallback=config.get("protocol_fallback"),
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
            protocol_fallback=extra_config.get("protocol_fallback"),
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

            requested_protocol = self.config.protocol

            def setup_store(protocol: str) -> None:
                result = self.store.setup(
                    self.config.local_hostname,
                    self.config.metadata_server,
                    self.config.global_segment_size,
                    self.config.local_buffer_size,
                    protocol,
                    self.config.device_name,
                    self.config.master_server_address,
                )
                if isinstance(result, int) and result != 0:
                    raise RuntimeError(
                        "Mooncake store setup returned error "
                        f"{result} for protocol={protocol}"
                    )

            try:
                setup_store(requested_protocol)
                self.effective_protocol = requested_protocol
            except Exception as primary_exc:
                fallback_protocol = self.config.protocol_fallback
                if (
                    not fallback_protocol
                    or fallback_protocol == requested_protocol
                ):
                    raise
                logger.warning(
                    "Mooncake protocol setup failed; retrying configured "
                    "fallback: requested=%s fallback=%s error=%s",
                    requested_protocol,
                    fallback_protocol,
                    primary_exc,
                )
                close_store = getattr(self.store, "close", None)
                if callable(close_store):
                    try:
                        close_store()
                    except Exception:
                        logger.debug(
                            "Failed to close Mooncake store after setup error",
                            exc_info=True,
                        )
                self.store = MooncakeDistributedStore()
                try:
                    setup_store(fallback_protocol)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "Mooncake requested and fallback protocol setup both "
                        "failed: "
                        f"requested={requested_protocol}, "
                        f"fallback={fallback_protocol}"
                    ) from fallback_exc
                self.effective_protocol = fallback_protocol

            logger.info(
                "[P2D_MOONCAKE_TRANSPORT] requested_protocol=%s "
                "effective_protocol=%s fallback_protocol=%s device_name=%s",
                requested_protocol,
                self.effective_protocol,
                self.config.protocol_fallback,
                self.config.device_name,
            )
            logger.info("Mooncake store setup completed successfully")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self._page_first_multi_buffer = self.config.page_first_multi_buffer
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
            logger.info(
                "[P2D_MOONCAKE_PAGE_LAYOUT] enabled=True num_layers=%d",
                self._page_num_layers,
            )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.registered_buffer_ptr = None
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
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
            else:
                self.registered_buffer_ptr = None
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None

    def _page_key_for(self, key: CacheEngineKey) -> Optional[str]:
        if not getattr(self, "_page_first_multi_buffer", False) or not isinstance(
            key, LayerCacheEngineKey
        ):
            return None
        return mooncake_page_key(key, self._page_num_layers)

    def _page_aware_exists_many(self, keys: List[CacheEngineKey]) -> list[bool]:
        page_keys = [self._page_key_for(key) for key in keys]
        page_results = [0] * len(keys)
        page_positions = [index for index, key in enumerate(page_keys) if key]
        if page_positions:
            queried_page_keys: list[str] = list(
                dict.fromkeys(key for key in page_keys if key is not None)
            )
            raw_page_results = self.store.batch_is_exist(queried_page_keys)
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
            legacy_results = self.store.batch_is_exist(
                [keys[index].to_string() for index in legacy_positions]
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
        for index, key in enumerate(keys):
            page_key = self._page_key_for(key)
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

    def _allocate_zero_copy_buffers(
        self, keys: List[CacheEngineKey]
    ) -> tuple[
        list[Optional[MemoryObj]],
        list[tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]],
        str,
    ]:
        key_metadata = [self._metadata_for_raw_key(key) for key in keys]
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
            )
            if batched is not None:
                if len(batched) == len(keys) and all(
                    obj.raw_tensor is not None for obj in batched
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
            getattr(self.config, "experimental_sampled_layerwise_lookup", False)
            and callable(getattr(self.store, "batch_is_exist", None))
        )

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        if not keys:
            return 0

        if getattr(self, "_page_first_multi_buffer", False):
            results = self._page_aware_exists_many(keys)
        else:
            results = [
                result == 1
                for result in self.store.batch_is_exist(
                    [key.to_string() for key in keys]
                )
            ]
        hit_count = 0
        for result in results:
            if not result:
                break
            hit_count += 1
        return hit_count

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
        page_key = self._page_key_for(key)
        return bool(
            (page_key is not None and self.store.is_exist(page_key))
            or self.store.is_exist(key.to_string())
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        page_key = self._page_key_for(key)
        return bool(
            (page_key is not None and self.store.is_exist(page_key))
            or self.store.is_exist(key.to_string())
        )

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
            if not self.store.is_exist(key.to_string()):
                break
            num_hit_counts += 1
        return num_hit_counts

    async def _batch_get_pages(
        self,
        keys: List[CacheEngineKey],
        page_groups: list[tuple[str, list[int]]],
    ) -> dict[int, MemoryObj]:
        total_started = time.perf_counter()
        flat_indices = [index for _, indices in page_groups for index in indices]
        page_keys = [keys[index] for index in flat_indices]
        allocation_started = time.perf_counter()
        memory_objs, _, allocation_mode = self._allocate_zero_copy_buffers(page_keys)
        allocation_ms = (time.perf_counter() - allocation_started) * 1000
        object_by_index = {
            original_index: memory_objs[position]
            for position, original_index in enumerate(flat_indices)
        }

        submitted_groups: list[tuple[str, list[int]]] = []
        all_buffer_ptrs: list[list[int]] = []
        all_buffer_sizes: list[list[int]] = []
        for page_key, indices in page_groups:
            group_objects = [object_by_index[index] for index in indices]
            if any(
                obj is None or obj.raw_tensor is None for obj in group_objects
            ):
                continue
            submitted_groups.append((page_key, indices))
            all_buffer_ptrs.append(
                [obj.data_ptr for obj in group_objects if obj is not None]
            )
            all_buffer_sizes.append(
                [obj.get_size() for obj in group_objects if obj is not None]
            )

        loaded: dict[int, MemoryObj] = {}
        transfer_ms = 0.0
        bytes_read = 0
        try:
            if not submitted_groups:
                return loaded
            transfer_started = time.perf_counter()
            statuses = await asyncio.to_thread(
                self.store.batch_get_into_multi_buffers,
                [page_key for page_key, _ in submitted_groups],
                all_buffer_ptrs,
                all_buffer_sizes,
            )
            transfer_ms = (time.perf_counter() - transfer_started) * 1000
            for group_index, (page_key, indices) in enumerate(submitted_groups):
                if group_index >= len(statuses):
                    logger.warning(
                        "Mooncake page get omitted status for page=%s", page_key
                    )
                    continue
                expected_bytes = sum(all_buffer_sizes[group_index])
                status = statuses[group_index]
                if status != expected_bytes:
                    logger.warning(
                        "Mooncake page get failed or was short: page=%s "
                        "status=%s expected_bytes=%d",
                        page_key,
                        status,
                        expected_bytes,
                    )
                    continue
                bytes_read += status
                for index in indices:
                    memory_obj = object_by_index[index]
                    assert memory_obj is not None
                    loaded[index] = memory_obj
                    object_by_index[index] = None

            first_key = keys[flat_indices[0]]
            gib_s = (
                bytes_read / (1024**3) / (transfer_ms / 1000)
                if transfer_ms > 0
                else 0.0
            )
            logger.info(
                "[P2D_MOONCAKE_GET] mode=batch_get_into_multi_buffers "
                "worker_id=%s kv_group=%s page_objects=%d logical_keys=%d "
                "keys_hit=%d requested_bytes=%d transferred_bytes=%d "
                "allocation_ms=%.3f transfer_ms=%.3f total_ms=%.3f "
                "effective_gib_s=%.3f allocation_mode=%s buffers_per_page=%d",
                first_key.worker_id,
                first_key.kv_group,
                len(submitted_groups),
                len(flat_indices),
                len(loaded),
                sum(sum(sizes) for sizes in all_buffer_sizes),
                bytes_read,
                allocation_ms,
                transfer_ms,
                (time.perf_counter() - total_started) * 1000,
                gib_s,
                allocation_mode,
                self._page_num_layers,
            )
            return loaded
        except Exception as exc:
            logger.error("Mooncake page-first get failed: %s", exc)
            return loaded
        finally:
            for memory_obj in object_by_index.values():
                if memory_obj is not None and memory_obj.is_valid():
                    memory_obj.ref_count_down()

    async def _batch_get_into(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not getattr(self, "_page_first_multi_buffer", False):
            return await self._batch_get_into_legacy(keys)

        complete_groups, legacy_indices = self._complete_page_groups(keys)
        if not complete_groups:
            return await self._batch_get_into_legacy(keys)

        page_exists = await asyncio.to_thread(
            self.store.batch_is_exist,
            [page_key for page_key, _ in complete_groups],
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

        results: list[Optional[MemoryObj]] = [None] * len(keys)
        if page_groups:
            page_results = await self._batch_get_pages(keys, page_groups)
            for index, memory_obj in page_results.items():
                results[index] = memory_obj

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

        valid_idx: list[int] = []
        key_strs: list[str] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        single_token_sizes: dict[int, int] = {}
        memory_objs, key_metadata, allocation_mode = (
            self._allocate_zero_copy_buffers(keys)
        )

        for i, (key, metadata_entry, obj) in enumerate(
            zip(keys, key_metadata, memory_objs, strict=True)
        ):
            single_token_size = metadata_entry[3]
            if obj is not None and obj.raw_tensor is not None:
                valid_idx.append(i)
                single_token_sizes[i] = single_token_size

                # Prepare the argument lists for the C++ call
                key_strs.append(key.to_string())
                buffer_ptrs.append(obj.data_ptr)
                buffer_sizes.append(obj.get_size())

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            return [None] * len(keys)

        try:
            # Single RPC call for multiple chunks
            logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")
            bytes_read_list = await asyncio.to_thread(
                self.store.batch_get_into, key_strs, buffer_ptrs, buffer_sizes
            )
            logger.debug(f"batch_get_into returned: {bytes_read_list}")

            # Assemble the final result list
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()  # type: ignore
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

            for i in valid_idx[len(bytes_read_list) :]:
                memory_objs[i].ref_count_down()  # type: ignore

            return results

        except Exception as exc:
            logger.error(f"batch_get_into threw exception: {str(exc)}")
            # Release any buffers we successfully allocated
            for i in valid_idx:
                memory_objs[i].ref_count_down()  # type: ignore
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
            await self._put_without_metadata(key_str, memory_obj)

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
        chunk_size = self.local_cpu_backend.metadata.chunk_size
        for page_key, indices in complete_groups:
            is_full_page = all(
                memory_objs[index].get_size()
                == self._metadata_for_raw_key(keys[index])[3] * chunk_size
                for index in indices
            )
            if is_full_page:
                page_groups.append((page_key, indices))
            else:
                legacy_indices.extend(indices)

        if page_groups:
            page_keys = [page_key for page_key, _ in page_groups]
            all_buffer_ptrs = [
                [memory_objs[index].data_ptr for index in indices]
                for _, indices in page_groups
            ]
            all_buffer_sizes = [
                [memory_objs[index].get_size() for index in indices]
                for _, indices in page_groups
            ]
            page_memory_objs = [
                memory_objs[index] for _, indices in page_groups for index in indices
            ]
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
            logger.info(
                "[P2D_MOONCAKE_PUT] mode=batch_put_from_multi_buffers "
                "page_objects=%d logical_keys=%d bytes=%d buffers_per_page=%d",
                len(page_groups),
                len(page_memory_objs),
                sum(sum(sizes) for sizes in all_buffer_sizes),
                self._page_num_layers,
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
        key_strs = [k.to_string() for k in keys]
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []
        for obj in memory_objs:
            assert obj.raw_tensor is not None
            buffer_ptrs.append(obj.data_ptr)
            buffer_sizes.append(obj.get_size())

        statuses = await self._run_blocking_put(
            "batch_put_from",
            self.store.batch_put_from,
            (key_strs, buffer_ptrs, buffer_sizes, self.replica_config),
            memory_objs,
        )
        self._check_batched_put_status(keys, statuses)

    async def _batched_put_with_metadata(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        for key, obj in zip(keys, memory_objs, strict=False):
            await self._put_with_metadata(key.to_string(), obj)

    async def _put_without_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Zero-copy put using put_from when metadata is not stored remotely.
        This is used when save_chunk_meta=False (matches _batch_get_into).
        """
        try:
            assert memory_obj.raw_tensor is not None
            buffer_ptr = memory_obj.data_ptr
            buffer_size = memory_obj.get_size()

            status = await self._run_blocking_put(
                "put_from",
                self.store.put_from,
                (key_str, buffer_ptr, buffer_size, self.replica_config),
                [memory_obj],
            )
            self._check_put_status("put_from", status)
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_from: "
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

        self.store.close()
        logger.info("Closed the mooncake store connection")
