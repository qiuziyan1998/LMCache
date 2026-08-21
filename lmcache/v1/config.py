# SPDX-License-Identifier: Apache-2.0
"""
LMCache Engine Configuration

Configuration system for LMCache Engine that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Provides convenient access to configuration values
"""

# Standard
from typing import Any, Dict, Optional
import json
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config_base import (
    _parse_local_disk,
    _parse_quoted_string,
    _resolve_config_aliases,
    _to_bool,
    _to_float_list,
    _to_int_list,
    _to_str_list,
    create_config_class,
    load_config_with_overrides,
)

logger = init_logger(__name__)
_REMOTE_FILL_H0_QUALIFICATION_ENV = "LMCACHE_REMOTE_FILL_H0_QUALIFICATION"
_REMOTE_FILL_H0_QUALIFICATION_V1 = "mooncake-sync-write-visible-v1"


# Configuration aliases and deprecated mappings
_CONFIG_ALIASES = {
    # Maps deprecated names to current names
    "enable_xpyd": "enable_pd",
    "nixl_peer_host": "pd_peer_host",
    "nixl_peer_init_port": "pd_peer_init_port",
    "nixl_peer_alloc_port": "pd_peer_alloc_port",
    "nixl_proxy_host": "pd_proxy_host",
    "nixl_proxy_port": "pd_proxy_port",
    "nixl_buffer_size": "pd_buffer_size",
    "nixl_role": "pd_role",
    "controller_url": "controller_pull_url",
    "lmcache_worker_port": "lmcache_worker_ports",
    "plugin_locations": "runtime_plugin_locations",
    "external_backends": "storage_plugins",
}

_DEPRECATED_CONFIGS = {
    # Maps deprecated names to warning messages
    "nixl_peer_port": "nixl_peer_port is deprecated, use nixl_receiver_port instead",
    "plugin_locations": (
        "plugin_locations is deprecated, use runtime_plugin_locations instead"
    ),
    "external_backends": (
        "external_backends is deprecated, use storage_plugins instead"
    ),
    "save_indexer_only_first_rank": (
        "save_indexer_only_first_rank is deprecated; use "
        "extra_config.save_only_first_rank to control both MLA latent and "
        "DSA index first-rank storage policy"
    ),
    "remote_fill_transport_mode": (
        "remote_fill_transport_mode is internal and ignored"
    ),
    "remote_fill_publish_mode": "remote_fill_publish_mode is internal and ignored",
    "remote_fill_require_both_groups": (
        "remote_fill_require_both_groups is internal and ignored"
    ),
    "remote_fill_persistent_placement": (
        "remote_fill_persistent_placement is internal and ignored"
    ),
    "remote_fill_allow_evict": "remote_fill_allow_evict is internal and ignored",
    "remote_fill_busy_loop_alloc": (
        "remote_fill_busy_loop_alloc is internal and ignored"
    ),
}

# Single configuration definition center - add new config items only here
_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Basic configurations
    "chunk_size": {"type": int, "default": 256, "env_converter": int},
    "local_cpu": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "max_local_cpu_size": {"type": float, "default": 5.0, "env_converter": float},
    "reserve_local_cpu_size": {"type": float, "default": 0.0, "env_converter": float},
    "local_disk": {
        "type": Optional[str],
        "default": None,
        "env_converter": _parse_local_disk,
    },
    "max_local_disk_size": {"type": float, "default": 0.0, "env_converter": float},
    "remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_serde": {"type": Optional[str], "default": "naive", "env_converter": str},
    # Feature toggles
    "use_layerwise": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "save_decode_cache": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pre_caching_hash_algorithm": {
        "type": str,
        "default": "builtin",
        "env_converter": str,
    },
    "enable_sparse_attention": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "experimental_sampled_layerwise_lookup": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Blending configurations
    "enable_blending": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "blend_recompute_ratios": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_thresholds": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_check_layers": {
        "type": list[int],
        "default": None,
        "env_converter": _to_int_list,
    },
    "blend_min_tokens": {"type": int, "default": 256, "env_converter": int},
    "blend_special_str": {"type": str, "default": " # # ", "env_converter": str},
    "retrieve_locations": {"type": Optional[list[str]], "default": None},
    "store_location": {"type": Optional[str], "default": None},
    # P2P configurations
    "enable_p2p": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "p2p_host": {"type": Optional[str], "default": None, "env_converter": str},
    "p2p_init_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "p2p_lookup_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # Controller configurations
    "enable_controller": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "lmcache_instance_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_pull_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_reply_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "lmcache_worker_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "lmcache_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # LMCache Worker heartbeat
    # the lmcache_worker_heartbeat_delay_time means that delay a period of time
    # before starting, ensures that the heartbeat starts working only after the
    # service is fully ready(such as, waiting register).
    "lmcache_worker_heartbeat_delay_time": {
        "type": int,
        "default": 10,
        "env_converter": int,
    },
    # the lmcache_worker_heartbeat_time means that sending heartbeat periodically.
    "lmcache_worker_heartbeat_time": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # PD-related configurations
    "enable_pd": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pd_role": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_buffer_size": {"type": Optional[int], "default": None, "env_converter": int},
    "pd_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "pd_peer_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_peer_init_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_peer_alloc_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_proxy_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_proxy_port": {"type": Optional[int], "default": None, "env_converter": int},
    # Transfer-related configurations
    "transfer_channel": {"type": Optional[str], "default": None, "env_converter": str},
    # Direct remote LocalCPU fill (default-off; native activation is hardware-gated)
    "enable_remote_lmcache_store": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "remote_fill_cache_namespace": {
        "type": str,
        "default": "",
        "env_converter": str,
    },
    "remote_fill_model_artifact_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_fill_max_active_transactions": {
        "type": int,
        "default": 8,
        "env_converter": int,
    },
    "remote_fill_max_inflight_windows_per_request": {
        "type": int,
        "default": 2,
        "env_converter": int,
    },
    "remote_fill_max_inflight_bytes": {
        "type": int,
        "default": 2 * 1024**3,
        "env_converter": int,
    },
    "remote_fill_max_reserved_bytes": {
        "type": int,
        "default": 16 * 1024**3,
        "env_converter": int,
    },
    "remote_fill_max_bytes_per_request": {
        "type": int,
        "default": 64 * 1024**3,
        "env_converter": int,
    },
    "remote_fill_min_free_bytes": {
        "type": int,
        "default": 8 * 1024**3,
        "env_converter": int,
    },
    "remote_fill_min_free_ratio": {
        "type": float,
        "default": 0.05,
        "env_converter": float,
    },
    "remote_fill_max_native_operations": {
        "type": int,
        "default": 2,
        "env_converter": int,
    },
    "remote_fill_direct_worker_count": {
        "type": int,
        "default": 2,
        "env_converter": int,
    },
    "remote_fill_window_tokens": {
        "type": int,
        "default": 4096,
        "env_converter": int,
    },
    "remote_fill_max_control_pages_per_window": {
        "type": int,
        # Zero derives the exact two-group bound from window/chunk size.
        "default": 0,
        "env_converter": int,
    },
    "remote_fill_max_rpc_message_bytes": {
        "type": int,
        "default": 64 * 1024,
        "env_converter": int,
    },
    "remote_fill_open_timeout_ms": {
        "type": int,
        "default": 5000,
        "env_converter": int,
    },
    "remote_fill_reserve_timeout_ms": {
        "type": int,
        "default": 5000,
        "env_converter": int,
    },
    "remote_fill_arm_timeout_ms": {
        "type": int,
        "default": 5000,
        "env_converter": int,
    },
    "remote_fill_transfer_timeout_ms": {
        "type": int,
        "default": 30000,
        "env_converter": int,
    },
    "remote_fill_finish_timeout_ms": {
        "type": int,
        "default": 30000,
        "env_converter": int,
    },
    "remote_fill_reservation_ttl_sec": {
        "type": int,
        "default": 30,
        "env_converter": int,
    },
    "remote_fill_terminal_record_ttl_sec": {
        "type": int,
        "default": 300,
        "env_converter": int,
    },
    "remote_fill_native_hard_timeout_ms": {
        "type": int,
        "default": 120000,
        "env_converter": int,
    },
    "remote_fill_control_host": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_fill_control_advertise_host": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_fill_control_port_start": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "remote_fill_descriptor_ttl_sec": {
        "type": int,
        "default": 10,
        "env_converter": int,
    },
    "remote_fill_circuit_breaker_enabled": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "remote_fill_circuit_breaker_failure_threshold": {
        "type": int,
        "default": 3,
        "env_converter": int,
    },
    "remote_fill_circuit_breaker_cooldown_sec": {
        "type": int,
        "default": 60,
        "env_converter": int,
    },
    # Nixl-related configurations
    "nixl_backends": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "nixl_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "nixl_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    # Storage paths
    "gds_path": {"type": Optional[str], "default": None, "env_converter": str},
    "gds_path_sharding": {
        "type": str,
        "default": "by_gpu",
        "env_converter": str,
    },
    "cufile_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # Maru CXL shared memory backend
    "maru_path": {"type": Optional[str], "default": None, "env_converter": str},
    "maru_pool_size": {
        "type": float,
        "default": 4.0,
        "env_converter": float,
    },
    # Other configurations
    # (Deprecated) The url of the actual remote lmcache instance for auditing.
    # Please use extra_config['audit_actual_remote_url'] instead.
    "audit_actual_remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "internal_api_server_host": {
        "type": str,
        "default": "0.0.0.0",
        "env_converter": str,
    },
    "extra_config": {
        "type": Optional[dict],
        "default": None,
        "env_converter": lambda x: x
        if isinstance(x, dict)
        else json.loads(x)
        if x
        else None,
    },
    "save_unfull_chunk": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "save_full_chunk_in_decode": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Deprecated compatibility only. With dsa_two_groups=true, the DSA index
    # first-rank policy follows extra_config.save_only_first_rank.
    "save_indexer_only_first_rank": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "dsa_two_groups": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "enable_dsa_cold_compact_load": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "dsa_group1_load_mode": {
        "type": str,
        "default": "p2p_preferred",
        "env_converter": str,
        "description": (
            "Group-1 cold-load policy: prefer live P2P, force serial "
            "persistent load, or overlap persistent prefetch with Group 0."
        ),
    },
    "enable_npu_transfer_validation": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
        "description": "Validate NPU transfer slots, cached destinations, and "
        "registered source spans before native kernel launch.",
    },
    "enable_npu_content_diagnostics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Fingerprint sampled Group-1 source, P2P destination, first "
            "decoder-consume, and selected top-k tensors. Diagnostic NPU-to-CPU "
            "readback may synchronize device work; keep disabled outside "
            "correctness investigations."
        ),
    },
    "enable_shared_cpu_cache": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "shared_cpu_cache_strict": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "shared_cpu_cache_name": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "shared_cpu_cache_size_gb": {
        "type": Optional[float],
        "default": None,
        "env_converter": float,
    },
    "shared_cpu_cache_numa_policy": {
        "type": str,
        "default": "first_touch",
        "env_converter": str,
    },
    "shared_cpu_cache_numa_nodes": {
        "type": str | int | list[int] | None,
        "default": None,
        "env_converter": lambda value: value,
    },
    "shared_cpu_materialize_index_on_decode_cold": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "shared_cpu_cache_passive_writable": {
        "type": Optional[bool],
        "default": None,
        "env_converter": _to_bool,
    },
    "blocking_timeout_secs": {"type": int, "default": 10, "env_converter": int},
    "external_lookup_client": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "py_enable_gc": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "cache_policy": {
        "type": str,
        "default": "LRU",
        "env_converter": str,
    },
    "numa_mode": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "enable_async_loading": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_port_start": {
        "type": int,
        "default": 6999,
        "env_converter": int,
    },
    "priority_limit": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "internal_api_server_include_index_list": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "internal_api_server_socket_path_prefix": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "runtime_plugin_locations": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": lambda x: x if isinstance(x, list) else [x] if x else [],
    },
    "storage_plugins": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "remote_storage_plugins": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lookup client configurations
    "lookup_timeout_ms": {
        "type": int,
        "default": 3000,
        "env_converter": int,
    },
    "min_retrieve_tokens": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": (
            "Minimum number of hit tokens required to perform retrieve. "
            "If hit tokens < min_retrieve_tokens, skip retrieve but the "
            "actual hit count is still used for skip_leading_tokens to avoid "
            "re-storing existing chunks. Default is 0 (disabled)."
        ),
    },
    "hit_miss_ratio": {
        "type": Optional[float],
        "default": None,
        "env_converter": float,
    },
    "lookup_server_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "enable_scheduler_bypass_lookup": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "script_allowed_imports": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lazy memory allocator configurations
    "enable_lazy_memory_allocator": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable lazy memory allocator to reduce initial memory footprint. "
            "Memory is allocated on-demand and expanded automatically when needed."
        ),
    },
    "lazy_memory_initial_ratio": {
        "type": float,
        "default": 0.2,
        "env_converter": float,
        "description": (
            "Initial memory allocation ratio (0.0-1.0). "
            "Determines the percentage of target memory size to allocate at startup. "
            "Default is 0.2 (20%)."
        ),
    },
    "lazy_memory_expand_trigger_ratio": {
        "type": float,
        "default": 0.5,
        "env_converter": float,
        "description": (
            "Memory usage ratio (0.0-1.0) that triggers automatic expansion. "
            "When memory usage exceeds this threshold, expansion is triggered. "
            "Default is 0.5 (50%)."
        ),
    },
    "lazy_memory_step_ratio": {
        "type": float,
        "default": 0.1,
        "env_converter": float,
        "description": (
            "Memory expansion step ratio (0.0-1.0). "
            "Determines the percentage of target memory size to add in each expansion. "
            "Default is 0.1 (10%)."
        ),
    },
    "lazy_memory_safe_size": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": (
            "Safe threshold size in GB. Lazy allocator is only enabled when "
            "max_local_cpu_size exceeds this value. Default is 0.0 GB (always enabled)."
        ),
    },
    # Chunk statistics configurations
    "enable_chunk_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable chunk statistics tracking.",
    },
    "chunk_statistics_auto_start_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Auto-start statistics on init.",
    },
    "chunk_statistics_auto_exit_timeout_hours": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": "Auto-stop timeout in hours (0=disabled).",
    },
    "chunk_statistics_auto_exit_target_unique_chunks": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": "Auto-stop at target unique chunks.",
    },
    "chunk_statistics_strategy": {
        "type": str,
        "default": "memory_bloom_filter",
        "env_converter": str,
        "description": "Recording strategy: memory_bloom_filter or file_hash.",
    },
    # KV events configuration
    "enable_kv_events": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # TODO(chunxiaozheng): remove this after VLLMPagedMemGPUConnectorV3 is stable
    "use_gpu_connector_v3": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Memory management configurations
    "pin_timeout_sec": {
        "type": int,
        "default": 300,
        "env_converter": int,
        "description": (
            "Maximum duration in seconds that a memory object can remain pinned. "
            "If a pinned object exceeds this timeout, it will be forcibly unpinned "
            "by the PinMonitor to prevent memory leaks. Default is 300 seconds."
        ),
    },
    "pin_check_interval_sec": {
        "type": int,
        "default": 30,
        "env_converter": int,
        "description": (
            "Interval in seconds between PinMonitor timeout checks. "
            "The background thread periodically scans all pinned objects at this "
            "interval to detect and handle timeouts. Default is 30 seconds."
        ),
    },
    # Remote configuration service
    "remote_config_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
        "description": (
            "URL of the remote configuration service. When set, LMCache will "
            "fetch additional configuration from this URL at startup."
        ),
    },
    "app_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
        "description": (
            "Application ID to send to the remote configuration service. "
            "If not set, the remote service may infer it from current config "
            "and environment variables."
        ),
    },
}


# Specialized methods that are unique to LMCacheEngineConfig
def _validate_config(self):
    """Validate configuration"""

    # needed for the old async serializer implementation
    # # auto-adjust save_unfull_chunk for async loading to prevent CPU fragmentation
    # if self.enable_async_loading:
    #     logger.warning(
    #         "Automatically setting save_unfull_chunk=False because "
    #         "enable_async_loading=True or use_layerwise=True to prevent "
    #         "CPU memory fragmentation"
    #     )
    #     self.save_unfull_chunk = False

    if self.min_retrieve_tokens < 0:
        raise ValueError(
            "min_retrieve_tokens must be >= 0, got %d" % self.min_retrieve_tokens
        )

    group1_load_modes = {
        "p2p_preferred",
        "persistent_serial",
        "persistent_parallel_prefetch",
    }
    if self.dsa_group1_load_mode not in group1_load_modes:
        raise ValueError(
            "dsa_group1_load_mode must be one of "
            f"{sorted(group1_load_modes)}, got {self.dsa_group1_load_mode!r}"
        )

    if self.dsa_two_groups and not self.use_layerwise:
        raise ValueError(
            "dsa_two_groups=true requires use_layerwise=true. The dense "
            "non-layerwise retrieve path only materializes KV group 0 and "
            "must not run with an uninitialized Group-1 index cache."
        )

    if self.enable_blending:
        if not self.save_unfull_chunk:
            logger.warning(
                "Automatically setting save_unfull_chunk=True because "
                "enable_blending=True"
            )
            self.save_unfull_chunk = True

    if (
        self.use_layerwise
        and self.enable_sparse_attention
        and not self.save_unfull_chunk
    ):
        raise ValueError(
            "use_layerwise=true with enable_sparse_attention=true requires "
            "save_unfull_chunk=true. Chunked-prefill tail KV must remain "
            "retrievable until a longer partial or full LMCache chunk replaces it."
        )

    extra_config = self.extra_config or {}
    enable_shared_cpu_cache = bool(
        extra_config.get(
            "enable_shared_cpu_cache",
            getattr(self, "enable_shared_cpu_cache", False),
        )
    )
    shared_cpu_cache_name = extra_config.get(
        "shared_cpu_cache_name",
        getattr(self, "shared_cpu_cache_name", None),
    )
    shared_cpu_cache_size_gb = extra_config.get(
        "shared_cpu_cache_size_gb",
        getattr(self, "shared_cpu_cache_size_gb", None),
    )
    shared_cpu_config_context = (
        " shared_cpu_config={"
        f"enable_shared_cpu_cache={enable_shared_cpu_cache}, "
        f"local_cpu={self.local_cpu}, "
        f"max_local_cpu_size={self.max_local_cpu_size}, "
        f"shared_cpu_cache_name={shared_cpu_cache_name!r}, "
        f"shared_cpu_cache_size_gb={shared_cpu_cache_size_gb!r}, "
        f"shm_name={extra_config.get('shm_name')!r}"
        "}"
    )
    if enable_shared_cpu_cache:
        if not self.local_cpu:
            raise ValueError(
                "enable_shared_cpu_cache requires local_cpu=true so rank0 "
                "LocalCPUBackend can be the shm-backed publication store."
                + shared_cpu_config_context
            )
        if self.max_local_cpu_size <= 0:
            raise ValueError(
                "enable_shared_cpu_cache requires max_local_cpu_size > 0 "
                "on rank0."
                + shared_cpu_config_context
            )
        shared_size_gb = shared_cpu_cache_size_gb
        if shared_size_gb is not None and float(shared_size_gb) <= 0:
            raise ValueError(
                "shared_cpu_cache_size_gb must be positive when set."
                + shared_cpu_config_context
            )
        shared_name = shared_cpu_cache_name
        shm_name = extra_config.get("shm_name")
        if shared_name and shm_name and shared_name != shm_name:
            raise ValueError(
                "shared_cpu_cache_name and shm_name refer to the same shared "
                "slab and must not conflict."
                + shared_cpu_config_context
            )

    remote_fill_active = bool(
        self.enable_remote_lmcache_store
        and os.getenv(_REMOTE_FILL_H0_QUALIFICATION_ENV)
        == _REMOTE_FILL_H0_QUALIFICATION_V1
    )
    if remote_fill_active:
        # These are invariants of the only implemented RemoteFill protocol,
        # not deployment choices: layerwise DSA two-group pages, immutable
        # final-only publication, non-evicting reservations, and prefiller-
        # local persistence.  Enabling the feature selects that contract.
        self.use_layerwise = True
        self.dsa_two_groups = True
        self.enable_sparse_attention = True
        self.save_unfull_chunk = True
        extra_config = dict(extra_config)
        extra_config.update(
            {
                "save_only_first_rank": True,
                "mooncake_page_first_multi_buffer": True,
                "mooncake_layer_merged_page_objects": True,
                "save_chunk_meta": False,
            }
        )
        self.extra_config = extra_config
        required_remote_fill = {
            "remote_url=mooncakestore://...": str(self.remote_url).startswith(
                "mooncakestore://"
            ),
        }
        missing_remote_fill = [
            name for name, enabled in required_remote_fill.items() if not enabled
        ]
        if missing_remote_fill:
            raise ValueError(
                "enable_remote_lmcache_store requires " + ", ".join(missing_remote_fill)
            )
        if self.pre_caching_hash_algorithm == "builtin":
            # RemoteFill always crosses process/host boundaries. Select the
            # existing deterministic vLLM hash instead of requiring operators
            # to configure Python's process-global hash seed before startup.
            self.pre_caching_hash_algorithm = "sha256_cbor"
        if self.remote_fill_window_tokens <= 0 or (
            self.remote_fill_window_tokens % self.chunk_size
        ):
            raise ValueError(
                "remote_fill_window_tokens must be a positive multiple of chunk_size"
            )
        required_control_pages = (self.remote_fill_window_tokens // self.chunk_size) * 2
        if self.remote_fill_max_control_pages_per_window == 0:
            self.remote_fill_max_control_pages_per_window = required_control_pages
        elif self.remote_fill_max_control_pages_per_window < required_control_pages:
            raise ValueError(
                "remote_fill_max_control_pages_per_window is too small for one "
                "two-group window"
            )
        positive_remote_fill_values = {
            "remote_fill_max_active_transactions": (
                self.remote_fill_max_active_transactions
            ),
            "remote_fill_max_inflight_windows_per_request": (
                self.remote_fill_max_inflight_windows_per_request
            ),
            "remote_fill_max_inflight_bytes": self.remote_fill_max_inflight_bytes,
            "remote_fill_max_reserved_bytes": self.remote_fill_max_reserved_bytes,
            "remote_fill_max_bytes_per_request": (
                self.remote_fill_max_bytes_per_request
            ),
            "remote_fill_max_native_operations": (
                self.remote_fill_max_native_operations
            ),
            "remote_fill_direct_worker_count": self.remote_fill_direct_worker_count,
            "remote_fill_max_rpc_message_bytes": (
                self.remote_fill_max_rpc_message_bytes
            ),
            "remote_fill_open_timeout_ms": self.remote_fill_open_timeout_ms,
            "remote_fill_reserve_timeout_ms": self.remote_fill_reserve_timeout_ms,
            "remote_fill_arm_timeout_ms": self.remote_fill_arm_timeout_ms,
            "remote_fill_transfer_timeout_ms": self.remote_fill_transfer_timeout_ms,
            "remote_fill_finish_timeout_ms": self.remote_fill_finish_timeout_ms,
            "remote_fill_reservation_ttl_sec": self.remote_fill_reservation_ttl_sec,
            "remote_fill_terminal_record_ttl_sec": (
                self.remote_fill_terminal_record_ttl_sec
            ),
            "remote_fill_native_hard_timeout_ms": (
                self.remote_fill_native_hard_timeout_ms
            ),
            "remote_fill_descriptor_ttl_sec": self.remote_fill_descriptor_ttl_sec,
        }
        nonpositive = [
            name for name, value in positive_remote_fill_values.items() if value <= 0
        ]
        if nonpositive:
            raise ValueError(
                "remote fill requires positive values for " + ", ".join(nonpositive)
            )
        minimum_pin_timeout = (
            float(self.remote_fill_native_hard_timeout_ms) / 1000.0 + 60.0
        )
        if float(self.pin_timeout_sec) <= minimum_pin_timeout:
            raise ValueError(
                "remote fill requires pin_timeout_sec to exceed the native hard "
                "timeout by more than 60 seconds"
            )
        if self.remote_fill_min_free_bytes < 0:
            raise ValueError("remote_fill_min_free_bytes must be non-negative")
        if not 0 <= self.remote_fill_min_free_ratio < 1:
            raise ValueError("remote_fill_min_free_ratio must be in [0, 1)")
        control_port = self.remote_fill_control_port_start
        if control_port is not None and not 0 < control_port < 65536:
            raise ValueError(
                "remote_fill_control_port_start must be between 1 and 65535"
            )
        advertise_host = self.remote_fill_control_advertise_host
        if advertise_host is not None and not advertise_host.strip():
            raise ValueError(
                "remote_fill_control_advertise_host must be non-empty when set"
            )
        if self.remote_fill_circuit_breaker_failure_threshold <= 0:
            raise ValueError(
                "remote_fill_circuit_breaker_failure_threshold must be positive"
            )
        if self.remote_fill_circuit_breaker_cooldown_sec <= 0:
            raise ValueError(
                "remote_fill_circuit_breaker_cooldown_sec must be positive"
            )

    if self.enable_dsa_cold_compact_load:
        required_flags = {
            "use_layerwise": self.use_layerwise,
            "enable_sparse_attention": self.enable_sparse_attention,
            "dsa_two_groups": self.dsa_two_groups,
            "enable_shared_cpu_cache": enable_shared_cpu_cache,
        }
        missing_flags = [
            name for name, enabled in required_flags.items() if not enabled
        ]
        if missing_flags:
            raise ValueError(
                "enable_dsa_cold_compact_load requires "
                + ", ".join(f"{name}=true" for name in missing_flags)
            )

    if self.dsa_group1_load_mode == "persistent_parallel_prefetch":
        prefetch_requirements = {
            "enable_dsa_cold_compact_load": self.enable_dsa_cold_compact_load,
            "use_layerwise": self.use_layerwise,
            "enable_sparse_attention": self.enable_sparse_attention,
            "dsa_two_groups": self.dsa_two_groups,
            "enable_shared_cpu_cache": enable_shared_cpu_cache,
            "remote_url=mooncakestore://...": str(self.remote_url).startswith(
                "mooncakestore://"
            ),
            "extra_config.save_only_first_rank": bool(
                extra_config.get("save_only_first_rank", False)
            ),
            "extra_config.mooncake_page_first_multi_buffer": bool(
                extra_config.get("mooncake_page_first_multi_buffer", False)
            ),
            "extra_config.mooncake_layer_merged_page_objects": bool(
                extra_config.get("mooncake_layer_merged_page_objects", False)
            ),
        }
        missing_prefetch_requirements = [
            name
            for name, enabled in prefetch_requirements.items()
            if not enabled
        ]
        if missing_prefetch_requirements:
            raise ValueError(
                "dsa_group1_load_mode=persistent_parallel_prefetch requires "
                + ", ".join(missing_prefetch_requirements)
            )

    if self.experimental_sampled_layerwise_lookup:
        required_flags = {
            "use_layerwise": self.use_layerwise,
            "enable_sparse_attention": self.enable_sparse_attention,
            "dsa_two_groups": self.dsa_two_groups,
            "enable_shared_cpu_cache": enable_shared_cpu_cache,
        }
        missing_flags = [
            name for name, enabled in required_flags.items() if not enabled
        ]
        if missing_flags:
            raise ValueError(
                "experimental_sampled_layerwise_lookup requires "
                + ", ".join(f"{name}=true" for name in missing_flags)
            )

    if self.enable_p2p:
        assert self.enable_controller
        assert self.controller_pull_url is not None
        assert self.controller_reply_url is not None
        assert self.lmcache_worker_ports is not None
        assert self.p2p_host is not None
        assert self.p2p_init_ports is not None
        assert self.p2p_lookup_ports is not None
        assert self.transfer_channel is not None

    enable_nixl_storage = self.extra_config is not None and self.extra_config.get(
        "enable_nixl_storage"
    )
    if self.enable_pd:
        assert self.pd_role is not None
        assert self.pd_buffer_size is not None
        assert self.pd_buffer_device is not None
        assert self.enable_p2p is False, "PD only supports enable_p2p=False"

        # PD requires save_unfull_chunk=True for complete KV cache transfer
        # from prefill node to decode node. Without this, partial chunks would
        # be discarded, causing incomplete KV cache transfer and wrong results
        # on the decode node.
        if not self.save_unfull_chunk:
            logger.warning(
                "PD (Peer-to-Peer Disaggregation) requires save_unfull_chunk=True "
                "for complete KV cache transfer. Automatically setting "
                "save_unfull_chunk=True."
            )
            self.save_unfull_chunk = True
        else:
            logger.info(
                "PD mode enabled with save_unfull_chunk=True - all KV cache "
                "including partial chunks will be transferred to decode node"
            )

        # for receiver, PDBackend is for retrieve location
        # can't take PDBackend as store location
        # as PDBackend is now one way from producer to receiver only
        if self.pd_role == "receiver":
            assert self.store_location != "PDBackend", (
                "store_location cannot be PDBackend for receiver"
            )
            assert self.retrieve_locations in (None, ["PDBackend"]), (
                "for pd receiver, "
                'retrieve_locations are expected to be ["PDBackend"], '
                f"now, it is {self.retrieve_locations}"
            )

    if enable_nixl_storage:
        assert self.extra_config.get("nixl_backend") is not None
        assert self.extra_config.get("nixl_pool_size") is not None
        assert self.nixl_buffer_size is not None
        assert self.nixl_buffer_device is not None

    return self


def _log_config(self):
    """Log configuration"""
    config_dict = {}
    for name in _CONFIG_DEFINITIONS:
        value = getattr(self, name)
        if name in ["max_local_cpu_size", "max_local_disk_size"]:
            value = f"{value} GB"
        config_dict[name] = value

    logger.info(f"LMCache Configuration: {config_dict}")
    return self


def _get_extra_config_value(self, key, default_value=None):
    if hasattr(self, "extra_config") and self.extra_config is not None:
        return self.extra_config.get(key, default_value)
    else:
        return default_value


def _get_lmcache_worker_ids(self, use_mla, world_size):
    if not self.lmcache_worker_ids:
        # if mla is not enabled, return all worker ids, which means start
        # lmcache worker on all ranks as default;
        # if mla is enabled, return [0], which means start lmcache
        # worker on worker 0 as default.
        return [0] if use_mla else list(range(world_size))

    # check the input
    for worker_id in self.lmcache_worker_ids:
        assert -1 < worker_id < world_size
    return self.lmcache_worker_ids


def _get_lookup_server_worker_ids(self, use_mla, world_size):
    if not self.lookup_server_worker_ids:
        # Non-MLA: lookup server on every worker (scheduler takes min hit count).
        if not use_mla:
            return list(range(world_size))
        # MLA + save_only_first_rank (default): only rank 0 stores/lookup keys.
        save_only_first_rank = self.get_extra_config_value(
            "save_only_first_rank", use_mla
        )
        if save_only_first_rank:
            return [0]
        # MLA + per-rank store: each TP rank has distinct worker_id keys; the
        # scheduler must query every rank and use min(hit) to stay consistent.
        return list(range(world_size))

    # check the input
    for worker_id in self.lookup_server_worker_ids:
        assert -1 < worker_id < world_size
    return self.lookup_server_worker_ids


def _from_legacy(cls, **kwargs):
    """Create configuration from legacy format"""
    backend = kwargs.pop("backend", "cpu")

    # Define backend mappings
    backend_configs = {
        "cpu": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
            "max_local_disk_size": 0,
            "remote_url": None,
        },
        "local_disk": {
            "local_cpu": False,
            "max_local_cpu_size": 3,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 2,
            "remote_url": None,
        },
        "local_cpu_disk": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
            "remote_url": None,
        },
        "remote": {"local_cpu": False, "max_local_cpu_size": 2, "local_disk": None},
        "local_cpu_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
        },
        "local_disk_remote": {
            "local_cpu": False,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
        "local_cpu_disk_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
    }

    if backend not in backend_configs:
        raise ValueError(f"Invalid backend: {backend}")

    # Merge configurations
    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in backend_configs[backend]:
            config_values[name] = backend_configs[backend][name]
        elif name in kwargs:
            config_values[name] = kwargs[name]
        else:
            config_values[name] = config["default"]

    instance = cls(**config_values)
    instance.validate()
    return instance


def _update_config_from_env(self):
    """Update an existing config object with environment variable configurations."""

    def get_env_name(attr_name: str) -> str:
        return f"LMCACHE_{attr_name.upper()}"

    # Collect environment variables
    env_config = {}
    for name in _CONFIG_DEFINITIONS:
        env_name = get_env_name(name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[name] = env_value

    # Handle deprecated environment variables
    for deprecated_name, new_name in _CONFIG_ALIASES.items():
        env_name = get_env_name(deprecated_name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[deprecated_name] = env_value

    # Resolve aliases and handle deprecated configurations
    resolved_config = _resolve_config_aliases(
        env_config,
        "environment variables",
        _CONFIG_DEFINITIONS,
        _CONFIG_ALIASES,
        _DEPRECATED_CONFIGS,
    )

    # Ensure _user_set_keys exists
    if not hasattr(self, "_user_set_keys"):
        object.__setattr__(self, "_user_set_keys", set())

    # Update config object with environment values
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in resolved_config:
            try:
                # Parse quoted strings and handle escape characters
                raw_value = resolved_config[name]  # Keep original value for logging
                value = _parse_quoted_string(raw_value)
                converted_value = config["env_converter"](value)
                setattr(self, name, converted_value)
                # Mark as user-set
                self._user_set_keys.add(name)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to parse {get_env_name(name)}={raw_value!r}: {e}"
                )
                # Keep existing value if conversion fails
    self.validate()
    return self


# Create configuration class using the base utility
LMCacheEngineConfig = create_config_class(
    config_name="LMCacheEngineConfig",
    config_definitions=_CONFIG_DEFINITIONS,
    config_aliases=_CONFIG_ALIASES,
    deprecated_configs=_DEPRECATED_CONFIGS,
    namespace_extras={
        "validate": _validate_config,
        "log_config": _log_config,
        "get_extra_config_value": _get_extra_config_value,
        "get_lmcache_worker_ids": _get_lmcache_worker_ids,
        "get_lookup_server_worker_ids": _get_lookup_server_worker_ids,
        "from_legacy": classmethod(_from_legacy),
        "update_config_from_env": _update_config_from_env,
    },
)


def load_engine_config_with_overrides(
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "LMCacheEngineConfig":  # type: ignore[valid-type]
    """
    Load engine configuration with support for file, env vars, and overrides.

    This function uses the generic load_config_with_overrides utility from
    config_base.py to reduce code duplication.

    Args:
        config_file_path: Optional direct path to config file
        overrides: Optional dictionary of configuration overrides

    Returns:
        Loaded and validated LMCacheEngineConfig instance
    """

    return load_config_with_overrides(
        config_class=LMCacheEngineConfig,
        config_file_env_var="LMCACHE_CONFIG_FILE",
        config_file_path=config_file_path,
        overrides=overrides,
    )
