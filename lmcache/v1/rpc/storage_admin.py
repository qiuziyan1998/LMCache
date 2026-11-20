# SPDX-License-Identifier: Apache-2.0
"""Explicit administrative calls to the sole CPU owner; no owner construction."""

# Standard
from copy import copy, deepcopy
from threading import RLock
from typing import Any
from uuid import uuid4


# These values establish live allocation ownership, physical interpretation, or
# native endpoint resources. Backend timeouts, URLs, cache policy and diagnostics
# continue through the same existing config converters and management paths.
_BOOTSTRAP_FIELDS = frozenset(
    {
        "decoder_storage_mode",
        "chunk_size",
        "local_cpu",
        "max_local_cpu_size",
        "reserve_local_cpu_size",
        "pre_caching_hash_algorithm",
        "use_layerwise",
        "dsa_two_groups",
        "enable_sparse_attention",
        "save_indexer_only_first_rank",
        "enable_shared_cpu_cache",
        "shared_cpu_cache_name",
        "shared_cpu_cache_size_gb",
        "shared_cpu_cache_numa_policy",
        "shared_cpu_cache_numa_nodes",
        "shared_cpu_cache_passive_writable",
        "numa_mode",
        "enable_lazy_memory_allocator",
        "lazy_memory_initial_ratio",
        "lazy_memory_expand_trigger_ratio",
        "lazy_memory_step_ratio",
        "lazy_memory_safe_size",
        "pd_role",
        "enable_pd",
        "enable_p2p",
        "enable_controller",
        "enable_blending",
        "local_disk",
        "max_local_disk_size",
        "gds_path",
        "maru_path",
        "storage_plugins",
        "remote_storage_plugins",
        "enable_remote_lmcache_store",
        "dsa_group1_load_mode",
        "remote_fill_cache_namespace",
        "remote_fill_model_artifact_id",
        "remote_fill_direct_worker_count",
        "remote_fill_max_native_operations",
        "remote_fill_control_host",
        "remote_fill_control_advertise_host",
        "remote_fill_control_port_start",
        "remote_serde",
    }
)
_NATIVE_EXTRA_FIELDS = _BOOTSTRAP_FIELDS | frozenset(
    {
        "local_hostname",
        "metadata_server",
        "global_segment_size",
        "local_buffer_size",
        "protocol",
        "device_name",
        "master_server_address",
        "storage_root_dir",
        "mooncake_prefer_local_alloc",
        "use_ascend_direct",
        "save_only_first_rank",
        "remote_enable_mla_worker_id_as0",
        "enable_nixl_storage",
    }
)
_INTERNAL_SLAB_FIELDS = frozenset({"shm_name", "shared_cpu_cache_resolved_name"})

# These existing families are updated by the CPU allocator/LocalCPU backend and
# StorageManager callbacks. Native/request/store counters continue to belong to
# W and must not be replaced by M's independently initialized zero counters.
CPU_STORAGE_METRIC_NAMES = frozenset(
    "lmcache:" + name
    for name in (
        "local_cache_usage",
        "remote_cache_usage",
        "local_storage_usage",
        "active_memory_objs_count",
        "pinned_memory_objs_count",
        "local_cpu_hot_cache_count",
        "local_cpu_keys_in_request_count",
        "storage_events_ongoing_count",
        "storage_events_done_count",
        "storage_events_not_found_count",
    )
)


def validate_process_config_update(
    config: Any,
    key: str,
    value: Any,
    metadata: Any,
) -> tuple[Any, str | None]:
    """Normalize an explicit config edit and reject changes to live resources.

    The local-mode API retains its existing behavior. In process mode M owns the
    authoritative config; a successful edit must be applied there before W's
    mirror is changed. Internal resolved slab names survive an ordinary public
    ``extra_config`` replacement but cannot be explicitly replaced.
    """
    from lmcache.v1.config import _CONFIG_DEFINITIONS
    from lmcache.v1.config_base import validate_and_set_config_value
    from lmcache.v1.mooncake_layout import mooncake_payload_layout

    if key not in _CONFIG_DEFINITIONS:
        return None, "Unknown config"
    if not _CONFIG_DEFINITIONS[key].get("mutable", True):
        return None, "Config is not mutable at runtime"
    candidate = copy(config)
    candidate.extra_config = deepcopy(config.extra_config)
    if not validate_and_set_config_value(candidate, key, value):
        return None, "Failed to set config value"
    normalized = getattr(candidate, key)
    previous = getattr(config, key)
    if key in _BOOTSTRAP_FIELDS and normalized != previous:
        return (
            None,
            "Changing storage ownership, layout or native resources requires a restart",
        )
    if key == "remote_url" and bool(normalized) != bool(previous):
        return None, "Changing the native remote session topology requires a restart"
    if (
        key == "remote_url"
        and normalized
        and not normalized.startswith("mooncakestore://")
    ):
        return None, "Process storage supports only Mooncake remote storage"
    if key == "extra_config":
        if normalized is not None and not isinstance(normalized, dict):
            return None, "extra_config must be a dictionary or null"
        old, new = config.extra_config or {}, dict(normalized or {})
        for name in _INTERNAL_SLAB_FIELDS:
            if name not in new and name in old:
                new[name] = old[name]
        for name in _NATIVE_EXTRA_FIELDS | _INTERNAL_SLAB_FIELDS:
            if new.get(name) != old.get(name):
                return None, f"Changing extra_config.{name} requires a restart"
        candidate.extra_config = new or None
        normalized = candidate.extra_config
    try:
        if (
            mooncake_payload_layout(config, metadata)[0]
            != mooncake_payload_layout(candidate, metadata)[0]
        ):
            return (
                None,
                "Changing the canonical cache-key or payload layout requires a restart",
            )
    except (TypeError, ValueError) as error:
        return None, str(error)
    return normalized, None


class StorageAdminClient:
    """Expose existing CPU management operations over coarse, acknowledged RPCs."""

    def __init__(self, client: Any, config: Any) -> None:
        self.client, self.config = client, config
        self.cached_frozen = False
        self.cached_hot_cache = bool(config.local_cpu)
        self._pending: dict[tuple, Any] = {}
        self._lock = RLock()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Run one explicit admin operation, preserving known exception classes."""
        # Cancellation may also be refused while M's history is full. Retry
        # those original IDs on later explicit API calls, outside warm decode.
        with self._lock:
            previous = tuple(self._pending.values())
        for pending in previous:
            if pending.done() and pending.exception() is None:
                if pending.result().status == "BUSY":
                    self._cancel_unadmitted(pending)
        future = self.client.request(
            "admin",
            {"method": method, "args": args, "kwargs": kwargs},
            request_id="storage-admin",
            request_attempt=uuid4().hex,
            load_generation=0,
            purpose="administration",
        )
        with self._lock:
            self._pending[future.identity.key] = future
        # A deadline does not cancel, replace, or release the operation. Its
        # callback still acknowledges the same eventual terminal result.
        future.add_done_callback(lambda done: self._completed(done, method, args))
        result = future.result(float(self.config.blocking_timeout_secs))
        if result.status != "COMPLETED" or result.error:
            detail = result.value if isinstance(result.value, dict) else {}
            if detail.get("admin_error") == "HTTPException":
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=detail["status_code"], detail=detail.get("detail")
                )
            error_type = {
                "KeyError": KeyError,
                "ValueError": ValueError,
                "RuntimeError": RuntimeError,
                "TypeError": TypeError,
            }.get(detail.get("admin_error"), RuntimeError)
            raise error_type(detail.get("message") or result.error or result.status)
        return result.value

    def _completed(self, future: Any, method: str, args: tuple) -> None:
        try:
            result = future.result()
            if result.status == "BUSY":
                # Admission can refuse work before M creates a registry record.
                # Reconcile that original ID on the independent lifecycle lane;
                # BUSY's digest cannot acknowledge the cancellation tombstone.
                self._cancel_unadmitted(future)
                return
            if result.status not in (
                "COMPLETED",
                "FAILED_SAFE",
                "CANCELLED_NOT_STARTED",
            ):
                return
            if result.status == "COMPLETED" and not result.error:
                if method == "set_freeze":
                    self.cached_frozen = bool(args[0])
                elif method == "set_hot_cache":
                    self.cached_hot_cache = bool(args[0])
            self._acknowledge(future.identity)
        except BaseException:
            return  # Retain unknown administration until supervised shutdown.

    def _cancel_unadmitted(self, future: Any) -> None:
        """Reconcile a refused original operation without replacing its work."""
        try:
            cancelled = self.client.cancel(future.identity)
        except (BufferError, RuntimeError, ValueError):
            return  # Keep its existing record until control admission recovers.
        with self._lock:
            self._pending[future.identity.key] = cancelled

        def cancelled_busy(done: Any) -> None:
            try:
                cancellation = done.result()
                if cancellation.status == "CANCELLED_NOT_STARTED":
                    self._acknowledge(
                        future.identity,
                        {
                            "terminal_result_digest": cancellation.value[
                                "terminal_result_digest"
                            ],
                        },
                    )
            except BaseException:
                return  # No cancellation proof: retain the original ID.

        cancelled.add_done_callback(cancelled_busy)

    def _acknowledge(self, identity: Any, payload: dict | None = None) -> None:
        ack = (
            self.client.acknowledge_terminal(identity)
            if payload is None
            else self.client.acknowledge_terminal(identity, payload)
        )
        with self._lock:
            self._pending[identity.key] = ack

        def acknowledged(done: Any) -> None:
            if done.result().status == "ACKNOWLEDGED":
                with self._lock:
                    self._pending.pop(identity.key, None)

        ack.add_done_callback(acknowledged)

    def is_healthy(self) -> bool:
        """Read cached client health without an RPC or native query."""
        return bool(self.client.is_healthy())

    def list_backends(self) -> dict[str, str]:
        """Return active M backend names and existing class names."""
        return self.call("list_backends")

    def close_backend(self, backend_name: str) -> bool:
        """Close an existing remote backend; M enforces owner protection."""
        return self.call("close_backend", backend_name)

    def create_backends(self) -> dict[str, str]:
        """Create missing configured CPU backends in M."""
        return self.call("create_backends")

    def recreate_backend(self, backend_name: str) -> dict[str, str]:
        """Reuse M's existing atomic close/recreate path."""
        return self.call("recreate_backend", backend_name)

    def get_bypassed_backends(self) -> list[str]:
        """Return existing bypass policy in M."""
        return self.call("get_bypassed_backends")

    def get_all_backend_names(self) -> list[str]:
        """Return existing configured backend names in M."""
        return self.call("get_all_backend_names")

    def is_backend_bypassed(self, backend_name: str) -> bool:
        """Return one backend's bypass policy."""
        return self.call("is_backend_bypassed", backend_name)

    def set_backend_bypass(self, backend_name: str, bypassed: bool) -> None:
        """Apply existing M bypass policy without replacing CPU ownership."""
        self.call("set_backend_bypass", backend_name, bypassed)

    def set_freeze(self, enabled: bool) -> None:
        """Change M freeze policy and cache the successful local control state."""
        self.call("set_freeze", enabled)
        self.cached_frozen = enabled

    def is_frozen(self) -> bool:
        """Refresh freeze status for an explicit administration request."""
        self.cached_frozen = bool(self.call("is_frozen"))
        return self.cached_frozen

    def set_hot_cache(self, enabled: bool) -> None:
        """Apply existing hot-cache/clear behavior only in M."""
        self.call("set_hot_cache", enabled)
        self.cached_hot_cache = enabled

    def is_hot_cache_enabled(self) -> bool:
        """Refresh hot-cache state for an explicit administration request."""
        self.cached_hot_cache = bool(self.call("is_hot_cache_enabled"))
        return self.cached_hot_cache

    def clear(self, locations: Any = None) -> int:
        """Reuse M's ordinary eviction semantics with live leases retained."""
        return self.call("clear", locations)

    def clear_tokens(
        self,
        hashes: tuple,
        offsets: tuple[int, ...],
        *,
        request_configs: dict | None = None,
        locations: Any = None,
    ) -> int:
        """Remove the existing canonical keys without sending prompt tokens."""
        return self.call(
            "clear_tokens",
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
            locations=locations,
        )

    def memcheck(self) -> bool:
        """Run the existing CPU allocator diagnostic only on explicit request."""
        return self.call("memcheck")

    def snapshot(self) -> dict[str, Any]:
        """Return M's operation/lease counters without scanning cache objects."""
        return self.call("snapshot")

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Validate/apply explicit mutable config changes in the sole CPU owner."""
        return self.call("update_config", updates)

    def load_fs_chunks(
        self,
        config_path: str,
        max_chunks: int | None,
        max_failed_keys: int,
    ) -> dict[str, Any]:
        """Load FS objects through M's existing allocator and async helpers."""
        return self.call(
            "load_fs_chunks",
            config_path=config_path,
            max_chunks=max_chunks,
            max_failed_keys=max_failed_keys,
        )

    def metrics(self) -> bytes:
        """Export M-owned CPU gauges only for an explicit metrics scrape."""
        return self.call("metrics")

    def reset_metrics(self) -> None:
        """Reset M's existing observability counters on explicit request."""
        self.call("reset_metrics")


def merge_storage_metrics(worker: bytes, storage: bytes) -> bytes:
    """Merge CPU-owned samples by exact metric name/labels without collisions."""
    from prometheus_client import generate_latest
    from prometheus_client.parser import text_string_to_metric_families

    families = {
        family.name: family
        for family in text_string_to_metric_families(worker.decode())
    }
    for family in text_string_to_metric_families(storage.decode()):
        if family.name not in CPU_STORAGE_METRIC_NAMES:
            continue
        current = families.get(family.name)
        if current is None:
            families[family.name] = family
            continue

        def sample_key(sample: Any) -> tuple:
            return sample.name, tuple(sorted(sample.labels.items()))

        replacements = {sample_key(sample): sample for sample in family.samples}
        current.samples = [
            sample
            for sample in current.samples
            if sample_key(sample) not in replacements
        ]
        current.samples.extend(replacements.values())

    class Collector:
        def collect(self):
            yield from families.values()

    return generate_latest(Collector())


def get_engine_storage_admin(engine: Any) -> Any:
    """Return a local manager or one cached facade for an existing W context."""
    context = getattr(engine, "storage_process", None)
    if context is None:
        return getattr(engine, "storage_manager", None)
    admin = getattr(engine, "storage_admin", None)
    if admin is None:
        admin = StorageAdminClient(context.client, engine.config)
        engine.storage_admin = admin
    return admin
