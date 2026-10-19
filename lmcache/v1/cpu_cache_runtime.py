# SPDX-License-Identifier: Apache-2.0
"""Independent decoder CPU source ownership and coarse storage operations."""

# Standard
from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
from uuid import uuid4
import time

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.cpu_cache import (
    StorageFailedSafe,
    StorageOperationRegistry,
    ValidatedStorageWork,
)
from lmcache.v1.cpu_cache_operations import CPUCacheOperations
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.memory_management import LayerPageMemoryObj, MemoryFormat
from lmcache.v1.mooncake_layout import mooncake_payload_layout
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.rpc.storage_protocol import (
    GroupManifest,
    LayoutSpec,
    LookupReply,
    OperationIdentity,
    PreparedReply,
    PrepareSpec,
    SourceGeometry,
    StorageRequest,
    StorageResult,
    content_digest,
    pack_uints,
    restore_hash,
    wire_hash,
)
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError


@dataclass
class _LookupTicket:
    identity: OperationIdentity
    reply: LookupReply
    claimed: OperationIdentity | None = None


class CPUCacheRuntime(CPUCacheOperations):
    """Own CPU storage without constructing a device cache engine.

    ``storage_manager`` owns the sole pre-profile allocator/session. ``layout``
    may be absent only during native bootstrap; lookup and export require a
    registered layout and all intended worker registrations. Sessions are
    authenticated by the operation registry, independently of request IDs.
    """

    def __init__(
        self,
        config: Any,
        metadata: Any,
        token_database: Any,
        *,
        storage_manager: Any,
        layout: LayoutSpec | None,
        cache_instance_id: str,
        service_epoch: str,
        slab_name: str,
        slab_size: int,
        sessions: dict[str, bytes],
        slab_generation: int = 1,
        max_operations: int = 256,
    ) -> None:
        if not slab_name or slab_size <= 0 or slab_generation <= 0:
            raise ValueError("CPU storage requires a live registered slab")
        self.config, self.metadata = config, metadata
        self.token_database, self.storage_manager = token_database, storage_manager
        self.layout: LayoutSpec | None = None
        self.cache_instance_id, self.service_epoch = cache_instance_id, service_epoch
        self.shared_cpu_cache_name = slab_name
        self.shared_cpu_cache_slab_size = slab_size
        self.shared_cpu_cache_generation = slab_generation
        self.num_layers = metadata.kv_shape[0]
        self.dsa_two_groups = bool(config.dsa_two_groups)
        self._transfer_layer_counts: tuple[int, ...] = ()
        self._cpu_runtime_local_flat_sources = True
        self.fmt = MemoryFormat.KV_MLA_LATENT_FMT if metadata.use_mla else None
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._remote_fill_lookup_plans: dict[str, Any] = {}
        self._lookup_request_ids: dict[str, str] = {}
        self._shared_cpu_request_leases: dict[str, Any] = {}
        self.registry = StorageOperationRegistry(
            cache_instance_id,
            service_epoch,
            sessions,
            max_operations=max_operations,
        )
        self._lock = RLock()
        self._tickets: dict[str, _LookupTicket] = {}
        self.source_key_plans: dict[
            str, tuple[OperationIdentity, LookupReply, int]
        ] = {}
        self.source_payload_digests: dict[str, bytes] = {}
        self.source_manifests: dict[str, GroupManifest] = {}
        self._lookup_generation = 0
        self._workers = frozenset(
            session for session in sessions if session.startswith("worker-")
        )
        self._registrations: dict[str, dict[str, Any]] = {}
        self._layout_received: set[str] = set()
        self._layout_starting = False
        self._serving_ready = False
        self._registration_error: str | None = None
        self._closed = False
        self._closing = False
        self._stopping = False
        PinMonitor.GetOrCreate(config)
        if layout is not None:
            self.bind_layout(layout)

    def bootstrap_descriptor(self) -> dict[str, Any]:
        """Return pointer-free slab identity for independently attaching workers."""
        return {
            "cache_instance_id": self.cache_instance_id,
            "service_epoch": self.service_epoch,
            "slab_name": self.shared_cpu_cache_name,
            "slab_size": self.shared_cpu_cache_slab_size,
            "slab_generation": self.shared_cpu_cache_generation,
            "allocator_alignment": self.allocator_alignment,
            "layout_bound": self.layout is not None,
            "serving_ready": self._serving_ready,
            "participants": tuple(sorted(self._workers)),
        }

    def memory_ready(self) -> dict[str, Any]:
        """Report the pre-profile native resources already owned by this service."""
        return self.bootstrap_descriptor()

    @property
    def allocator_alignment(self) -> int:
        """Read the existing slab allocator's physical alignment requirement."""
        allocator = self._shared_local_cpu_backend().get_memory_allocator()
        alignment = getattr(allocator, "align_bytes", None)
        if alignment is None:
            alignment = allocator.address_manager.compute_aligned_size(1)
        return int(alignment)

    def register_participant(
        self, session: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Record one authenticated worker mapping or fail the complete startup gate."""
        if session not in self._workers or payload.get("session") != session:
            raise ValueError("Registration cannot impersonate another participant")
        error = payload.get("error")
        resource = payload.get("resource", {})
        rank = payload.get("tp_rank")
        if error is None and (
            type(rank) is not int
            or session != f"worker-{rank}"
            or resource.get("slab_name") != self.shared_cpu_cache_name
            or resource.get("slab_size") != self.shared_cpu_cache_slab_size
            or resource.get("slab_generation") != self.shared_cpu_cache_generation
            or resource.get("native_registered") is not True
            or resource.get("reader_ready") is not True
            or type(resource.get("device_base")) is not int
            or resource["device_base"] <= 0
        ):
            error = (
                "Worker registration does not cover the shared slab and native reader"
            )
        with self._lock:
            if error:
                self._registration_error = f"{session}: {error}"
            elif (
                session in self._registrations
                and self._registrations[session] != payload
            ):
                self._registration_error = f"{session}: native registration changed"
            else:
                self._registrations[session] = dict(payload)
        return self.registration_status()

    def registration_status(self) -> dict[str, Any]:
        """Read the gate without waiting, collecting or querying native resources."""
        with self._lock:
            if self._registration_error:
                return {
                    "status": "FAILED",
                    "layout_status": "FAILED",
                    "error": self._registration_error,
                }
            return {
                "status": "READY"
                if self._workers <= self._registrations.keys()
                else "WAITING",
                "registered": tuple(sorted(self._registrations)),
                "layout_status": "READY" if self._serving_ready else "WAITING",
            }

    def unregister_participant(self, session: str) -> dict[str, Any]:
        """Record shutdown without treating process loss as a DMA completion proof."""
        with self._lock:
            self._registrations.pop(session, None)
        return self.registration_status()

    def bind_layout(
        self, layout: LayoutSpec, session: str | None = None
    ) -> dict[str, Any]:
        """Finalize registered metadata using the existing native session."""
        layout.validate()
        if session is not None and session not in self._workers:
            raise ValueError("Only an inference worker may bind its source layout")
        with self._lock:
            if session is not None and session not in self._registrations:
                raise StorageFailedSafe(
                    "Worker native registration must precede layout binding"
                )
            if self.layout is not None:
                if self.layout != layout:
                    self._registration_error = "Worker source layouts disagree"
                    raise ValueError("CPU layout cannot change during a service epoch")
                if session is not None:
                    self._layout_received.add(session)
                apply = False
            else:
                apply = True
        if apply:
            self._apply_layout(layout)
        with self._lock:
            if session is not None:
                self._layout_received.add(session)
            start = (
                session is not None
                and self._workers <= self._layout_received
                and self._workers <= self._registrations.keys()
                and not self._layout_starting
                and not self._serving_ready
            )
            if start:
                self._layout_starting = True
        if start:
            try:
                self.on_layout_bound()
            except BaseException:
                with self._lock:
                    self._registration_error = "CPU layout-bound initialization failed"
                raise
            with self._lock:
                self._serving_ready = True
        return {
            "layout_digest": layout.digest,
            "status": "READY" if self._serving_ready else "WAITING",
        }

    def on_layout_bound(self) -> None:
        """Initialize CPU services after every worker validates the same layout."""

    def _apply_layout(self, layout: LayoutSpec) -> None:
        with self._lock:
            if self.layout is not None:
                if self.layout != layout:
                    raise ValueError("CPU layout cannot change during a service epoch")
                return
            if layout.chunk_size != self.config.chunk_size:
                raise ValueError("Registered source chunk size disagrees with storage")
            counts = tuple(len(group.layer_names) for group in layout.groups)
            if self.dsa_two_groups and len(counts) != 2:
                raise ValueError(
                    "Decoder CPU storage requires both registered KV groups"
                )
            policy = None
            if layout.precision_descriptor:
                descriptor = msgspec.msgpack.decode(layout.precision_descriptor)
                policy = IndexerC8Layout(
                    int(descriptor["head_dim"]),
                    tuple(descriptor.get("c8_layers", ())),
                )
                if policy.c8_layers and len(policy.c8_layers) != counts[1]:
                    raise ValueError(
                        "Registered indexer precision has the wrong owner count"
                    )
            groups = []
            for group in layout.groups:
                dtype = getattr(torch, group.layer_dtypes[0].removeprefix("torch."))
                width = sum(group.plane_token_bytes[0]) // dtype.itemsize
                if group.group_id == 1 and policy is not None:
                    dtype, width = self.metadata.kv_dtype, policy.head_dim
                    for layer, widths in enumerate(group.plane_token_bytes):
                        if sum(widths) != policy.token_bytes_for(layer):
                            raise ValueError(
                                "Indexer byte geometry disagrees with precision"
                            )
                groups.append(
                    KVLayerGroupInfo(
                        list(group.layer_names),
                        list(range(len(group.layer_names))),
                        torch.Size([1, layout.chunk_size, width]),
                        dtype,
                    )
                )
            self.metadata.runtime_kv_group_layer_counts = counts
            self.metadata.runtime_kv_group_layer_names = tuple(
                g.layer_names for g in layout.groups
            )
            self.metadata.indexer_c8_layout = policy
            self.metadata.kv_layer_groups_manager = KVLayerGroupsManager(groups)
            self._transfer_layer_counts = counts
            self.num_layers = counts[0]
            local = self._shared_local_cpu_backend()
            local.metadata = self.metadata
            factory = getattr(self.storage_manager, "backend_factory", None)
            bind = getattr(factory, "bind_layout", None)
            if callable(bind):
                bind(self.metadata)
            else:
                remote = self.storage_manager.storage_backends.get("RemoteBackend")
                remote = getattr(remote, "real_backend", remote)
                if remote is not None and getattr(remote, "defer_layout", False):
                    remote.bind_layout(self.metadata)
            if self.token_database is not None:
                self.token_database.metadata = self.metadata
                if layout.precision_descriptor or self.config.get_extra_config_value(
                    "mooncake_page_first_multi_buffer",
                    False,
                ):
                    self.token_database.mooncake_payload_layout = (
                        mooncake_payload_layout(
                            self.config,
                            self.metadata,
                        )[0]
                    )
            self.layout = layout

    def lookup(
        self, plan_input: dict[str, Any], identity: OperationIdentity
    ) -> LookupReply:
        """Retain a proved paired prefix from already-hashed canonical chunks."""
        self._require_ready()
        chunks, hashes = self._chunks(plan_input)
        request_configs = dict(plan_input.get("request_configs") or {})
        search = plan_input.get("search_range")
        search = (
            list(
                self.config.retrieve_locations or self.storage_manager.storage_backends
            )
            if search is None
            else list(search)
        )
        ticket_id = uuid4().hex
        with self._lock:
            self._lookup_request_ids[ticket_id] = identity.request_id
        try:
            matched = self._lookup_remote_fill_two_group_prefix(
                chunks,
                search_range=search,
                lookup_id=ticket_id,
                pin=True,
                request_configs=request_configs,
            )
            if self.registry.cancellation_requested(identity):
                raise StorageFailedSafe("CPU lookup was cancelled")
            retained = self._remote_fill_lookup_plans.get(ticket_id)
            selected = tuple(chunk for chunk in chunks if chunk[0] < matched)
            digest = content_digest(
                (
                    self.layout.digest,
                    request_configs,
                    tuple(
                        (a, b, wire_hash(key.chunk_hash, self.layout.hash_width))
                        for a, b, key in selected
                    ),
                    tuple(
                        (c.locations_by_group, c.page_by_group) for c in retained.chunks
                    )
                    if retained
                    else (),
                )
            )
            with self._lock:
                self._lookup_generation += 1
                reply = LookupReply(
                    matched,
                    ticket_id if matched else "",
                    self._lookup_generation,
                    digest,
                    tuple(a for a, _, _ in selected),
                    tuple(b for _, b, _ in selected),
                    hashes[: len(selected)],
                    request_configs,
                )
                if matched:
                    self._tickets[ticket_id] = _LookupTicket(identity, reply)
                    self.registry.set_ticket(identity, ticket_id)
            return reply
        except BaseException:
            self._release_lookup_pins(ticket_id)
            raise
        finally:
            with self._lock:
                self._lookup_request_ids.pop(ticket_id, None)

    def drop_lookup(self, ticket_id: str, identity: OperationIdentity) -> None:
        """Drop an unclaimed lookup hold, preserving prepared source leases."""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                return
            if (ticket.identity.request_id, ticket.identity.request_attempt) != (
                identity.request_id,
                identity.request_attempt,
            ):
                raise ValueError("Lookup cleanup targets another request lifetime")
            if ticket.claimed is not None:
                return
            del self._tickets[ticket_id]
        self._release_lookup_pins(ticket_id)
        self.registry.set_ticket(ticket.identity, None)

    def prepare(
        self,
        spec: PrepareSpec,
        identity: OperationIdentity,
        *,
        payload_digest: bytes | None = None,
    ) -> PreparedReply:
        """Resolve exact sources once and export independent physical group leases."""
        self._require_ready()
        with self._lock:
            ticket = self._tickets.get(spec.ticket_id)
            if ticket is None or ticket.claimed is not None:
                raise StorageFailedSafe(
                    "CPU lookup ticket is missing or already claimed"
                )
            if (ticket.identity.request_id, ticket.identity.request_attempt) != (
                identity.request_id,
                identity.request_attempt,
            ):
                raise StorageFailedSafe(
                    "CPU lookup ticket belongs to another request attempt"
                )
            if (
                spec.key_plan_digest != ticket.reply.key_plan_digest
                or spec.layout_digest != self.layout.digest
                or not 0 < spec.source_end <= ticket.reply.matched_tokens
                or not spec.participants
                or not set(spec.participants) <= self._workers
            ):
                raise StorageFailedSafe(
                    "CPU prepare identity, extent or participants are invalid"
                )
            ticket.claimed = identity
        leases: list[str] = []
        producer_leases: set[str] = set()
        restore_lease = None
        checkpoint_base = 0
        started = time.monotonic()
        try:
            if spec.mode == "checkpoint_restore":
                checkpoint_base, restore_owners = (
                    self.prepare_checkpoint_restore_sources(
                        spec,
                        identity,
                        ticket.reply,
                    )
                )
                if restore_owners:
                    try:
                        restore_lease = self.registry.add_lease(
                            identity,
                            restore_owners,
                            participants=spec.participants,
                            release=lambda objects: self._release_shared_retrieve_objs(
                                list(objects),
                                unpin=False,
                            ),
                            producer_holds=1,
                        )
                    except BaseException:
                        self._release_shared_retrieve_objs(restore_owners, unpin=False)
                        raise
                    leases.append(restore_lease)
                    producer_leases.add(restore_lease)
            count = sum(start < spec.source_end for start in ticket.reply.starts)
            data = {
                "starts": ticket.reply.starts[:count],
                "ends": ticket.reply.ends[:count],
                "hashes": ticket.reply.hashes[:count],
                "request_configs": ticket.reply.request_configs,
            }
            candidates, _ = self._chunks(data)
            manifests = []
            actions = []
            if spec.mode == "checkpoint_restore":
                actions.append("checkpoint_group1_tail")
                if checkpoint_base:
                    actions.append("load_group1_direct_prefix")
            for group in range(len(self.layout.groups)):
                if self.registry.cancellation_requested(identity):
                    raise StorageFailedSafe(
                        "CPU preparation was cancelled before export"
                    )
                if (
                    group == 1
                    and self._persistent_direct_hbm_split_group_enabled()
                    and spec.mode not in ("checkpoint_restore", "dense")
                ):
                    actions.append("load_group1_direct")
                    continue
                first = (
                    checkpoint_base // self.layout.chunk_size
                    if group == 1 and spec.mode == "checkpoint_restore"
                    else 0
                )
                selected = candidates[first:]
                group_reply = msgspec.structs.replace(
                    ticket.reply,
                    starts=ticket.reply.starts[first:count],
                    ends=ticket.reply.ends[first:count],
                    hashes=ticket.reply.hashes[first:count],
                )
                plan = self._remote_fill_retrieve_plan(spec.ticket_id, selected, group)
                if plan is None:
                    raise StorageFailedSafe(
                        "CPU preparation has no authoritative paired plan"
                    )
                page_count = next(
                    (i for i, (_, page) in enumerate(plan) if not page), len(plan)
                )
                mixed_order = any(page for _, page in plan[page_count:])
                layer_keys = [
                    [
                        self._lookup_key_for_kv_group(
                            key, kv_group=group, request_configs=data["request_configs"]
                        ).split_layers(self.num_layers_for_group(group))[layer]
                        for _, _, key in selected
                    ]
                    for layer in range(self.num_layers_for_group(group))
                ]
                page_positions = None
                if mixed_order:
                    rows, acquisitions, page_positions = self._resolve_mixed_sources(
                        spec.ticket_id,
                        group,
                        selected,
                        layer_keys,
                        plan,
                        data["request_configs"],
                    )
                    page_count = len(page_positions)
                elif page_count:
                    rows, page_count = self._resolve_shared_rank0_layer_pages(
                        req_id=spec.ticket_id,
                        phase="cpu_prepare",
                        kv_group=group,
                        keys_layer_major=layer_keys,
                        page_chunks=page_count,
                        base_page_keys=[
                            self._lookup_key_for_kv_group(
                                key,
                                kv_group=group,
                                request_configs=data["request_configs"],
                            )
                            for _, _, key in selected
                        ],
                        exact_chunk_locations=[location for location, _ in plan],
                    )
                else:
                    rows = []
                    try:
                        for layer, keys in enumerate(layer_keys):
                            rows.append(
                                self._resolve_shared_rank0_layer_mem_objs(
                                    req_id=spec.ticket_id,
                                    phase="cpu_prepare",
                                    layer_id=layer,
                                    kv_group=group,
                                    keys_layer=keys,
                                    chunk_locations=[location for location, _ in plan],
                                    enforce_planned_location=True,
                                )
                            )
                    except BaseException:
                        self._release_shared_retrieve_objs(
                            [obj for row in rows for obj in row], unpin=True
                        )
                        raise
                if not mixed_order:
                    acquisitions = list(rows[0][:page_count]) + [
                        obj for row in rows for obj in row[page_count:]
                    ]
                try:
                    holder: list[str] = []

                    def release_source(objects: tuple[Any, ...], holder=holder) -> None:
                        self._release_shared_retrieve_objs(list(objects), unpin=True)
                        with self._lock:
                            self.source_key_plans.pop(holder[0], None)
                            self.source_payload_digests.pop(holder[0], None)
                            self.source_manifests.pop(holder[0], None)

                    lease = self.registry.add_lease(
                        identity,
                        acquisitions,
                        participants=spec.participants,
                        release=release_source,
                        producer_holds=1,
                    )
                except BaseException:
                    self._release_shared_retrieve_objs(acquisitions, unpin=True)
                    raise
                leases.append(lease)
                producer_leases.add(lease)
                holder.append(lease)
                with self._lock:
                    self.source_key_plans[lease] = (identity, group_reply, group)
                    self.source_payload_digests[lease] = (
                        payload_digest
                        if payload_digest is not None
                        else content_digest(("prepare", spec))
                    )
                manifest = self._manifest(
                    group,
                    rows,
                    page_count,
                    group_reply,
                    spec.source_end,
                    lease,
                    page_positions=page_positions,
                )
                manifests.append(manifest)
                with self._lock:
                    self.source_manifests[lease] = manifest
            reply = PreparedReply(
                identity,
                spec.source_end,
                self.shared_cpu_cache_name,
                self.shared_cpu_cache_slab_size,
                self.shared_cpu_cache_generation,
                tuple(manifests),
                {"prepare_ms": (time.monotonic() - started) * 1000},
                tuple(actions),
                restore_lease_id=restore_lease,
            )
            self.registry.publish_leases(identity, leases)
            producer_leases.clear()
            return reply
        except BaseException as error:
            for lease in tuple(producer_leases):
                producer_leases.remove(lease)
                self.registry.producer_done(lease, known_complete=True)
            for lease in reversed(leases):
                self.registry.discard_unexported_lease(identity, lease)
            if isinstance(error, Exception) and not isinstance(
                error,
                (StorageFailedSafe, NativeExternalPageTransferUnknownError),
            ):
                raise StorageFailedSafe(str(error)) from error
            raise
        finally:
            self._release_lookup_pins(spec.ticket_id)
            self.registry.set_ticket(ticket.identity, None)
            with self._lock:
                self._tickets.pop(spec.ticket_id, None)

    def _resolve_mixed_sources(
        self,
        ticket_id: str,
        group: int,
        candidates: list,
        layer_keys: list[list],
        plan: list,
        configs: dict,
    ) -> tuple[list[list[Any]], list[Any], tuple[int, ...]]:
        """Reuse exact resolvers for ordered page/flat runs without CPU copies."""
        rows = [[] for _ in range(self.num_layers_for_group(group))]
        acquisitions, page_positions = [], []
        begin = 0
        try:
            while begin < len(plan):
                is_page = plan[begin][1]
                stop = begin + 1
                while stop < len(plan) and plan[stop][1] == is_page:
                    stop += 1
                locations = [location for location, _ in plan[begin:stop]]
                keys = [layer[begin:stop] for layer in layer_keys]
                if is_page:
                    resolved, count = self._resolve_shared_rank0_layer_pages(
                        req_id=ticket_id,
                        phase="cpu_prepare",
                        kv_group=group,
                        keys_layer_major=keys,
                        page_chunks=stop - begin,
                        base_page_keys=[
                            self._lookup_key_for_kv_group(
                                key,
                                kv_group=group,
                                request_configs=configs,
                            )
                            for _, _, key in candidates[begin:stop]
                        ],
                        exact_chunk_locations=locations,
                    )
                    acquisitions.extend(resolved[0][:count])
                    if count != stop - begin:
                        raise StorageFailedSafe(
                            "Exact merged source run changed representation"
                        )
                    for layer, values in enumerate(resolved):
                        rows[layer].extend(values)
                    page_positions.extend(range(begin, stop))
                else:
                    for layer, keys_layer in enumerate(keys):
                        values = self._resolve_shared_rank0_layer_mem_objs(
                            req_id=ticket_id,
                            phase="cpu_prepare",
                            layer_id=layer,
                            kv_group=group,
                            keys_layer=keys_layer,
                            chunk_locations=locations,
                            enforce_planned_location=True,
                        )
                        acquisitions.extend(values)
                        rows[layer].extend(values)
                begin = stop
            return rows, acquisitions, tuple(page_positions)
        except BaseException:
            self._release_shared_retrieve_objs(acquisitions, unpin=True)
            raise

    def prepare_checkpoint_restore_sources(
        self,
        spec: PrepareSpec,
        identity: OperationIdentity,
        reply: LookupReply,
    ) -> tuple[int, list[Any]]:
        """Normalize checkpoint CPU sources in the platform runtime extension."""
        raise StorageFailedSafe(
            "Checkpoint normalization requires the Ascend CPU runtime"
        )

    def validate_source_operation(
        self,
        descriptor: dict[str, Any],
        identity: OperationIdentity,
        *,
        group: int = 0,
    ) -> tuple[OperationIdentity, LookupReply, int]:
        """Resolve a source proof only while its exact exported lease remains live."""
        try:
            manifest = msgspec.convert(descriptor["manifest"], type=GroupManifest)
            origin = msgspec.convert(descriptor["identity"], type=OperationIdentity)
        except (KeyError, TypeError, ValueError) as error:
            raise StorageFailedSafe(
                "Invalid CPU source operation descriptor"
            ) from error
        with self._lock:
            source = self.source_key_plans.get(manifest.lease_id)
            if source is None or (
                origin != source[0]
                or manifest != self.source_manifests.get(manifest.lease_id)
                or descriptor.get("payload_digest")
                != self.source_payload_digests.get(manifest.lease_id)
                or source[2] != group
                or (origin.request_id, origin.request_attempt)
                != (
                    identity.request_id,
                    identity.request_attempt,
                )
            ):
                raise StorageFailedSafe(
                    "CPU source descriptor disagrees with live ownership"
                )
            return source

    def prepare_store(self, request: StorageRequest) -> PreparedReply:
        """Promote a newly persisted paired frontier through the existing resolver."""
        payload, identity = request.payload, request.identity
        if payload.get("source_operation") is not None:
            self.validate_source_operation(payload["source_operation"], identity)
        if payload["layout_digest"] != self.layout.digest:
            raise StorageFailedSafe("CPU store promotion has a stale layout")
        hit = self.lookup(payload["plan"], identity)
        try:
            if hit.matched_tokens < payload["source_end"]:
                raise StorageFailedSafe(
                    "CPU store promotion has no complete paired prefix",
                    details={"promotion_refused": True},
                )
            return self.prepare(
                PrepareSpec(
                    hit.ticket_id,
                    hit.key_plan_digest,
                    payload["source_end"],
                    payload["layout_digest"],
                    tuple(payload["participants"]),
                    mode=payload.get("mode", "resident"),
                ),
                identity,
                payload_digest=request.payload_digest,
            )
        finally:
            if hit.ticket_id:
                self.drop_lookup(hit.ticket_id, identity)

    def handle(self, request: StorageRequest) -> StorageResult:
        """Authenticate and dispatch work or original-operation lifecycle controls."""
        result = self.registry.handle(request, self._dispatch)
        if request.method == "cancel":
            with self._lock:
                tickets = [
                    key
                    for key, ticket in self._tickets.items()
                    if ticket.identity == request.identity
                ]
            for key in tickets:
                self.drop_lookup(key, request.identity)
        return result

    def handle_validated(
        self, request: StorageRequest, proof: ValidatedStorageWork
    ) -> StorageResult:
        """Dispatch service-validated WORK while retaining registry safeguards.

        Direct callers should use handle(), which always validates the body.
        The proof is in-process only and bound to this exact decoded request.
        """
        return self.registry.handle_validated(request, self._dispatch, proof)

    def set_transport_notifier(self, notifier: Callable[[], None] | None) -> bool:
        """Forward terminal/fatal wakeups without constructing transport state."""
        return self.registry.set_transport_notifier(notifier)

    def is_healthy(self) -> bool:
        """Return cached CPU ownership/registration health without remote queries."""
        return (
            not self._closed
            and not self._registration_error
            and self.registry.is_healthy()
        )

    def drain_terminal_notifications(self) -> list[StorageResult]:
        """Forward resource-free completion acknowledgements to the lifecycle loop."""
        return self.registry.drain_terminal_notifications()

    def check_close_ready(self) -> None:
        """Validate drain before any backend or native service is stopped."""
        snapshot = self.registry.snapshot()
        if (
            snapshot["leases"]
            or snapshot["fatal"]
            or snapshot["quarantined"]
            or snapshot["running"] > int(self._stopping)
        ):
            raise RuntimeError(
                "CPU storage still owns live or quarantined source leases"
            )
        with self._lock:
            if self._tickets:
                raise RuntimeError("CPU storage lookup tickets have not been retired")

    def close(self) -> None:
        """Close storage only after every exported source has a retirement proof."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
        try:
            self.check_close_ready()
            self.storage_manager.close()
            self._closed = True
        finally:
            with self._lock:
                self._closing = False

    def _dispatch(self, request: StorageRequest) -> Any:
        if self._closing:
            raise StorageFailedSafe("CPU storage shutdown is in progress")
        payload, identity = request.payload, request.identity
        if request.method in ("memory_ready", "bootstrap_descriptor"):
            return self.bootstrap_descriptor()
        if request.method == "register_participant":
            return self.register_participant(identity.client_session, payload)
        if request.method == "registration_status":
            return self.registration_status()
        if request.method == "unregister_participant":
            if payload.get("session") != identity.client_session:
                raise ValueError("Unregister cannot impersonate another participant")
            return self.unregister_participant(identity.client_session)
        if request.method == "bind_layout":
            if payload.get("session") != identity.client_session:
                raise ValueError(
                    "Layout binding cannot impersonate another participant"
                )
            return self.bind_layout(
                msgspec.convert(payload["layout"], type=LayoutSpec),
                identity.client_session,
            )
        if request.method == "lookup":
            return self.lookup(payload, identity)
        if request.method == "drop_lookup":
            return self.drop_lookup(payload["ticket_id"], identity)
        if request.method == "prepare":
            return self.prepare(
                msgspec.convert(payload, type=PrepareSpec),
                identity,
                payload_digest=request.payload_digest,
            )
        if request.method == "prepare_store":
            return self.prepare_store(request)
        if request.method == "admin":
            return self.admin(payload)
        if request.method == "snapshot":
            return self.registry.snapshot()
        if request.method == "stop":
            with self._lock:
                if self._registrations:
                    return {
                        "status": "BUSY",
                        "registered": tuple(sorted(self._registrations)),
                    }
            self._stopping = True
            try:
                self.close()
            except RuntimeError as error:
                raise StorageFailedSafe(str(error)) from error
            finally:
                self._stopping = False
            return {"status": "STOPPED"}
        raise StorageFailedSafe(f"Unknown CPU storage operation: {request.method}")

    def admin(self, payload: dict[str, Any]) -> Any:
        """Delegate explicit management operations to the sole storage manager."""
        if payload.get("method") == "snapshot":
            return self.registry.snapshot()
        if payload.get("method") == "load_fs_chunks":
            if payload.get("args"):
                raise StorageFailedSafe("Invalid FS cache import arguments")
            return self.load_fs_chunks(**dict(payload.get("kwargs", {})))
        if payload.get("method") == "clear_tokens":
            return self.clear_tokens(**dict(payload.get("kwargs", {})))
        if payload.get("method") == "metrics":
            return self.metrics()
        if payload.get("method") == "reset_metrics":
            from lmcache.observability import reset_observability_metrics

            reset_observability_metrics()
            return None
        if payload.get("method") == "update_config":
            from lmcache.v1.rpc.storage_admin import validate_process_config_update

            args, kwargs = tuple(payload.get("args", ())), payload.get("kwargs", {})
            if len(args) != 1 or kwargs or not isinstance(args[0], dict):
                raise StorageFailedSafe("Invalid CPU configuration update")
            updated, errors = {}, {}
            for key, value in args[0].items():
                normalized, error = validate_process_config_update(
                    self.config,
                    key,
                    value,
                    self.metadata,
                )
                if error is not None:
                    errors[key] = error
                else:
                    setattr(self.config, key, normalized)
                    updated[key] = normalized
            return {"updated": updated, "errors": errors}
        allowed = {
            "list_backends",
            "close_backend",
            "create_backends",
            "recreate_backend",
            "get_bypassed_backends",
            "get_all_backend_names",
            "is_backend_bypassed",
            "set_backend_bypass",
            "set_freeze",
            "is_frozen",
            "set_hot_cache",
            "is_hot_cache_enabled",
            "clear",
            "memcheck",
        }
        method = payload.get("method")
        if method not in allowed:
            raise StorageFailedSafe("Unknown CPU storage management operation")
        args, kwargs = tuple(payload.get("args", ())), dict(payload.get("kwargs", {}))
        if method in ("close_backend", "recreate_backend"):
            name = args[0] if args else kwargs.get("backend_name")
            if name == "LocalCPUBackend":
                raise StorageFailedSafe(
                    "Registered LocalCPU storage requires service restart"
                )
        try:
            return getattr(self.storage_manager, method)(*args, **kwargs)
        except (KeyError, ValueError, RuntimeError, TypeError) as error:
            if method in ("close_backend", "recreate_backend") and isinstance(
                error, RuntimeError
            ):
                # Native teardown can refuse because transfer completion is
                # unknown. Preserve the owner and latch the registry failure.
                raise
            raise StorageFailedSafe(
                str(error),
                details={"admin_error": type(error).__name__, "message": str(error)},
            ) from error

    def clear_tokens(
        self,
        hashes: tuple[int | bytes, ...],
        offsets: tuple[int, ...],
        request_configs: dict[str, Any] | None = None,
        locations: list[str] | None = None,
    ) -> int:
        """Remove the same base keys as local clear without transferring raw tokens."""
        self._require_ready()
        if len(hashes) != len(offsets) or any(
            type(count) is not int or not 0 < count <= self.config.chunk_size
            for count in offsets
        ):
            raise StorageFailedSafe("Invalid canonical cache-clear ranges")
        keys = self.token_database.process_tokens(
            hashes=[
                restore_hash(h, self.layout.hash_width, self.layout.hash_kind)
                for h in hashes
            ],
            offsets=offsets,
            request_configs=request_configs,
        )
        return sum(self.storage_manager.remove(key, locations) for _, _, key in keys)

    def metrics(self) -> bytes:
        """Export CPU/storage metrics, preserving worker request counters."""
        from prometheus_client import REGISTRY, generate_latest
        from lmcache.observability import LMCStatsMonitor, PrometheusLogger
        from lmcache.v1.rpc.storage_admin import CPU_STORAGE_METRIC_NAMES

        logger = PrometheusLogger.GetOrCreate(self.metadata, config=self.config)
        logger.log_prometheus(LMCStatsMonitor.GetOrCreate().get_stats_and_clear())

        class Collector:
            def collect(self):
                yield from (
                    family
                    for family in REGISTRY.collect()
                    if family.name in CPU_STORAGE_METRIC_NAMES
                )

        return generate_latest(Collector())

    def load_fs_chunks(
        self,
        config_path: str,
        max_chunks: int | None = None,
        max_failed_keys: int = 10,
    ) -> dict[str, Any]:
        """Warm the sole LocalCPU owner through the existing FS import helpers."""
        import asyncio
        from fastapi import HTTPException
        from lmcache.v1.internal_api_server.vllm.load_fs_chunks_api import (
            _initialize_remote_backend,
            _load_chunks_from_fs_connector,
            _load_config_from_file,
        )

        remote = None
        try:
            config = asyncio.run_coroutine_threadsafe(
                _load_config_from_file(config_path),
                self.storage_manager.loop,
            ).result()
            if not str(config.remote_url).startswith("fs://"):
                raise HTTPException(
                    status_code=400, detail="FS import requires fs:// storage"
                )
            local = self.storage_manager.allocator_backend
            remote = asyncio.run(
                _initialize_remote_backend(
                    config,
                    self.metadata,
                    local,
                    self.storage_manager.loop,
                )
            )
            result = asyncio.run_coroutine_threadsafe(
                _load_chunks_from_fs_connector(
                    remote, local, max_chunks, max_failed_keys
                ),
                self.storage_manager.loop,
            ).result()
            return dict(result, status="success", config_path=config_path)
        except HTTPException as error:
            raise StorageFailedSafe(
                str(error.detail),
                details={
                    "admin_error": "HTTPException",
                    "status_code": error.status_code,
                    "detail": error.detail,
                },
            ) from error
        finally:
            if remote is not None:
                remote.close()

    def _require_ready(self) -> None:
        if (
            self._closed
            or self._closing
            or self.layout is None
            or self.token_database is None
        ):
            raise StorageFailedSafe("CPU source layout is not ready")
        status = self.registration_status()
        if status["status"] != "READY":
            raise StorageFailedSafe(f"CPU worker registration is {status['status']}")
        if self._workers and not self._serving_ready:
            raise StorageFailedSafe(
                "CPU source layout validation is not complete on every worker"
            )

    def _chunks(
        self, value: dict[str, Any]
    ) -> tuple[list[Any], tuple[int | bytes, ...]]:
        starts, ends, hashes = (
            tuple(value["starts"]),
            tuple(value["ends"]),
            tuple(value["hashes"]),
        )
        if not len(starts) == len(ends) == len(hashes):
            raise StorageFailedSafe("CPU canonical chunk cardinalities disagree")
        if (
            value.get("hash_kind", self.layout.hash_kind) != self.layout.hash_kind
            or value.get("hash_width", self.layout.hash_width) != self.layout.hash_width
        ):
            raise StorageFailedSafe(
                "CPU hash representation disagrees with registered layout"
            )
        cursor = 0
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if (
                type(start) is not int
                or type(end) is not int
                or start != cursor
                or not start < end <= start + self.layout.chunk_size
                or index < len(starts) - 1
                and end - start != self.layout.chunk_size
            ):
                raise StorageFailedSafe(
                    "CPU chunk ranges have a hole, overlap or invalid tail"
                )
            cursor = end
        restored = [
            restore_hash(h, self.layout.hash_width, self.layout.hash_kind)
            for h in hashes
        ]
        chunks = list(
            self.token_database.process_tokens(
                hashes=restored,
                offsets=[b - a for a, b in zip(starts, ends, strict=True)],
                request_configs=value.get("request_configs") or {},
                kv_group=0,
            )
        )
        if tuple((a, b) for a, b, _ in chunks) != tuple(zip(starts, ends, strict=True)):
            raise StorageFailedSafe(
                "Token database changed the canonical source ranges"
            )
        return chunks, tuple(wire_hash(h, self.layout.hash_width) for h in restored)

    def source_geometry(self, group: int, page: Any) -> SourceGeometry:
        """Read actual allocation boundaries without constructing tensor views."""
        starts = tuple(page.group_prefix_sum[:-1])
        sizes = tuple(
            b - a
            for a, b in zip(
                page.group_prefix_sum,
                page.group_prefix_sum[1:],
                strict=False,
            )
        )
        shapes, dtypes = page.get_shapes(), page.get_dtypes()
        planes = []
        for widths in self.layout.groups[group].plane_token_bytes:
            cursor, ranges = 0, []
            for width in widths:
                length = page.valid_tokens * width
                ranges.append((cursor, length))
                cursor += length
            planes.append(tuple(ranges))
        geometry = SourceGeometry(
            page.valid_tokens,
            starts,
            sizes,
            tuple(tuple(shape) for shape in shapes),
            tuple(str(dtype).removeprefix("torch.") for dtype in dtypes),
            tuple(planes),
        )
        return geometry

    def _manifest(
        self,
        group: int,
        rows: list[list[Any]],
        page_count: int,
        reply: LookupReply,
        source_end: int,
        lease_id: str,
        *,
        page_positions: tuple[int, ...] | None = None,
    ) -> GroupManifest:
        registered = self.layout.groups[group]
        if page_positions is None:
            page_positions = tuple(range(page_count))
        page_set = set(page_positions)
        flat_positions = tuple(
            index for index in range(len(rows[0])) if index not in page_set
        )
        geometries, indices, offsets, extents = [], [], [], []
        for position in page_positions:
            page = rows[0][position]
            if not isinstance(page, LayerPageMemoryObj):
                raise ValueError("Merged source manifest requires actual layer pages")
            geometry = self.source_geometry(group, page)
            if geometry not in geometries:
                geometries.append(geometry)
            indices.append(geometries.index(geometry))
            offsets.append(int(page.metadata.address))
            extents.append(int(page.metadata.phy_size))
        flat = [row[index] for row in rows for index in flat_positions]
        capacities = []
        tail = len(rows[0]) - page_count
        for index, obj in enumerate(flat):
            width = sum(registered.plane_token_bytes[index // tail])
            logical = obj.get_size()
            if logical % width:
                raise ValueError(
                    "Flat source bytes are not a complete registered packet"
                )
            capacities.append(logical // width)
        manifest = GroupManifest(
            group,
            self.shared_cpu_cache_generation,
            self.layout.digest,
            reply.key_plan_digest,
            lease_id,
            reply.starts[: len(rows[0])],
            reply.ends[: len(rows[0])],
            pack_uints(offsets),
            pack_uints(extents),
            pack_uints(indices, 32),
            tuple(geometries),
            pack_uints(int(obj.metadata.address) for obj in flat),
            pack_uints(int(obj.metadata.phy_size) for obj in flat),
            pack_uints(obj.get_size() for obj in flat),
            pack_uints(capacities, 32),
            source_start=reply.starts[0],
            page_chunk_indices=(
                pack_uints(page_positions, 32)
                if page_positions != tuple(range(page_count))
                else b""
            ),
        )
        manifest.validate(
            self.layout,
            self.shared_cpu_cache_slab_size,
            source_end,
            key_plan_digest=reply.key_plan_digest,
            allow_checkpoint_tail=manifest.source_start > 0,
        )
        return manifest
