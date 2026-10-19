# SPDX-License-Identifier: Apache-2.0
"""Worker-local compact source import; no allocator or steady-state RPC."""

# Standard
from concurrent.futures import Future
from threading import Lock
from typing import Any, Callable
from weakref import proxy
import time

# Third Party
import numpy as np
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_compact_prepared_sparse_source
from lmcache.v1.rpc.storage_protocol import (
    GroupManifest,
    LayoutSpec,
    OperationIdentity,
    PreparedReply,
    checked_uint,
)
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle


def initialize_worker_storage(engine: Any) -> None:
    """Bind resources registered before profiling to final source geometry."""
    from lmcache.v1.rpc.storage_layout import freeze_storage_layout
    from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime

    context = engine.storage_process
    if context is None or engine.gpu_connector is None:
        raise ValueError(
            "Process storage requires an existing worker bootstrap and connector"
        )
    alignment = context.descriptor.get("allocator_alignment", 64)
    layout = freeze_storage_layout(
        engine.config,
        engine.metadata,
        engine.token_database,
        allocator_alignment=alignment,
        plane_token_bytes=engine.storage_plane_token_bytes,
    )
    if context.reader is not None:
        context.reader.bind_layout(engine.metadata)
    context.bind_layout(layout)
    engine.storage_layout = layout
    engine.shared_cpu_cache_mapping = context.mapping
    engine.shared_cpu_cache_passive_allocator = context.mapping.passive_allocator()
    receiver = proxy(engine)
    engine.compact_source_importer = CompactSourceImporter(
        layout=layout,
        mapping=context.mapping,
        local_device_base=context.device_base,
        client=context.client,
        upload=lambda table: receiver.gpu_connector.stage_dense_load_tensor(
            table,
            dtype=torch.long,
        ),
        fence_upload=receiver.gpu_connector.synchronize_dense_load_stream,
    )
    engine.gpu_connector.compact_source_importer = engine.compact_source_importer
    context.drain_guards.append(engine.compact_source_importer.sources_drained)
    engine.native_source_lifetime = NativeSourceLifetime(
        torch.npu.Event,
        lambda reason: receiver.storage_process.mark_failed(reason),
    )
    engine.gpu_connector.native_source_lifetime = engine.native_source_lifetime
    context.drain_guards.append(engine.native_source_lifetime.sources_drained)


def source_pointer_table(
    manifest: GroupManifest, local_device_base: int, layers: int
) -> torch.Tensor:
    """Rebase checked byte frames in bulk, preserving all uint64 pointer bits."""
    pages = len(manifest.page_offsets) // 8
    chunks = len(manifest.chunk_starts)
    base = checked_uint(local_device_base)
    offsets = np.frombuffer(manifest.page_offsets, dtype="<u8")
    indices = np.frombuffer(manifest.geometry_ids, dtype="<u4")
    table = np.empty((layers, chunks), dtype=np.uint64)
    page_positions = (
        np.frombuffer(manifest.page_chunk_indices, dtype="<u4")
        if manifest.page_chunk_indices
        else None
    )
    if pages:
        starts = np.asarray(
            [geometry.layer_starts for geometry in manifest.geometries], dtype=np.uint64
        )
        if page_positions is None:
            table[:, :pages] = starts[indices].T + offsets
        else:
            table[:, page_positions] = starts[indices].T + offsets
    if chunks > pages:
        flat = np.frombuffer(manifest.flat_offsets, dtype="<u8").reshape(
            layers, chunks - pages
        )
        if page_positions is None:
            table[:, pages:] = flat
        else:
            table[:, manifest.flat_positions()] = flat
    if table.size and int(table.max()) > (1 << 64) - 1 - base:
        raise ValueError("Worker mapped source pointer overflows uint64")
    table += np.uint64(base)
    return torch.from_numpy(table.view(np.int64))


def source_physical_token_counts(
    manifest: GroupManifest, layers: int
) -> tuple[tuple[int, ...], ...]:
    """Read certified packet capacities once, in the canonical chunk order."""
    indices = np.frombuffer(manifest.geometry_ids, dtype="<u4")
    capacities = [[0] * len(manifest.chunk_starts) for _ in range(layers)]
    for page, position in enumerate(manifest.page_positions()):
        capacity = manifest.geometries[int(indices[page])].token_capacity
        for row in capacities:
            row[position] = capacity
    flat_positions = manifest.flat_positions()
    if flat_positions:
        flat = np.frombuffer(manifest.flat_token_capacities, dtype="<u4").reshape(
            layers, -1
        )
        for layer, row in enumerate(capacities):
            for index, position in enumerate(flat_positions):
                row[position] = int(flat[layer, index])
    return tuple(tuple(row) for row in capacities)


class CompactSourceImporter:
    """Retain local mappings until physical-retirement acknowledgement.

    ``upload`` reuses the connector's existing pointer-table staging path.
    Consumers retain local child leases; only their last release sends RETIRE.
    Failed or unknown retirement keeps its mapping in ``pending_retirements``.
    """

    def __init__(
        self,
        *,
        layout: LayoutSpec,
        mapping: Any,
        local_device_base: int,
        client: Any,
        upload: Callable[[torch.Tensor], torch.Tensor],
        fence_upload: Callable[[], None],
    ) -> None:
        layout.validate()
        self.layout = layout
        self.mapping = mapping
        self.local_device_base = checked_uint(local_device_base)
        self.client = client
        self.upload = upload
        self.fence_upload = fence_upload
        self.pending_retirements: dict[str, tuple[Any, Any]] = {}
        self._live_leases: set[str] = set()
        self._restore_holds: dict[tuple[str, int], LocalLeaseHandle] = {}
        self._quarantined_imports: list[tuple[Any, ...]] = []
        self._certificates: dict[
            int, tuple[Any, OperationIdentity, GroupManifest, bytes]
        ] = {}
        self._lock = Lock()

    def sources_drained(self) -> bool:
        """Permit unregistration only after all local consumers and retire ACKs."""
        with self._lock:
            return not self._live_leases and not self.pending_retirements

    def release_restore(self, request_id: str, load_generation: int) -> None:
        """Retire normalization owners at the existing scheduler all-rank ACK."""
        with self._lock:
            hold = self._restore_holds.pop((request_id, load_generation), None)
        if hold is not None:
            hold.release("fenced")

    def is_certified_source(
        self, source: Any, request_id: str, load_generation: int
    ) -> bool:
        """Validate the immutable import handoff independently of sealing alone."""
        if source is None or source.owner_lease is None:
            return False
        with self._lock:
            record = self._certificates.get(id(source.binding_token))
        return bool(
            record is not None
            and record[0] is source.binding_token
            and record[1].request_id == request_id
            and record[1].load_generation == load_generation
            and record[2].lease_id == source.owner_lease.lease_id
            and source.owner_lease.is_valid()
        )

    def borrow_source_tensors(
        self, source: Any, *, layer_id: int | None = None
    ) -> tuple[tuple[tuple[torch.Tensor, ...], ...], LocalLeaseHandle]:
        """Lazily expose real mapped CPU tensors for existing dense capture APIs.

        The caller owns the returned child, marks it before native submission,
        and releases it only at that consumer's final completion event.
        """
        with self._lock:
            record = self._certificates.get(id(source.binding_token))
        if record is None or record[0] is not source.binding_token:
            raise ValueError("Dense source was not certified by this importer")
        hold = source.owner_lease.retain()
        manifest = record[2]
        try:
            pages = len(manifest.page_offsets) // 8
            layers = len(self.layout.groups[manifest.group_id].layer_names)
            offsets = np.frombuffer(manifest.page_offsets, dtype="<u8")
            indices = np.frombuffer(manifest.geometry_ids, dtype="<u4")
            suffix = np.frombuffer(manifest.flat_offsets, dtype="<u8").reshape(
                layers, -1
            )
            logical = np.frombuffer(manifest.flat_logical_sizes, dtype="<u8").reshape(
                layers, -1
            )
            page_lookup = (
                dict(zip(manifest.page_positions(), range(pages), strict=True))
                if manifest.page_chunk_indices
                else None
            )
            flat_lookup = (
                dict(
                    zip(
                        manifest.flat_positions(),
                        range(len(manifest.chunk_starts) - pages),
                        strict=True,
                    )
                )
                if manifest.page_chunk_indices
                else None
            )
            rows = []
            selected_layers = range(layers) if layer_id is None else (layer_id,)
            if layer_id is not None and not 0 <= layer_id < layers:
                raise ValueError("Source layer is outside the registered group")
            for layer in selected_layers:
                tensors = []
                for chunk in range(len(manifest.chunk_starts)):
                    page_index = (
                        page_lookup.get(chunk)
                        if page_lookup is not None
                        else chunk
                        if chunk < pages
                        else None
                    )
                    if page_index is not None:
                        geometry = manifest.geometries[int(indices[page_index])]
                        offset = int(offsets[page_index]) + geometry.layer_starts[layer]
                        size = geometry.layer_sizes[layer]
                        dtype = getattr(
                            torch, geometry.layer_dtypes[layer].removeprefix("torch.")
                        )
                        shape = geometry.layer_shapes[layer]
                    else:
                        flat_index = (
                            flat_lookup[chunk]
                            if flat_lookup is not None
                            else chunk - pages
                        )
                        offset, size = (
                            int(suffix[layer, flat_index]),
                            int(logical[layer, flat_index]),
                        )
                        dtype = getattr(
                            torch,
                            self.layout.groups[manifest.group_id]
                            .layer_dtypes[layer]
                            .removeprefix("torch."),
                        )
                        shape = (size // dtype.itemsize,)
                    tensors.append(
                        self.mapping.tensor.narrow(0, offset, size)
                        .view(dtype)
                        .reshape(shape)
                    )
                rows.append(tuple(tensors))
            return tuple(rows), hold
        except BaseException:
            hold.release("never_imported")
            raise

    def source_layer_tensors(
        self, source: Any, layer_id: int
    ) -> tuple[torch.Tensor, ...]:
        """Borrow diagnostic CPU views while the caller holds the source lease.

        Deferred consumers must retain their own child before returning. This
        method performs no device access and does no work on disabled paths.
        """
        rows, hold = self.borrow_source_tensors(source, layer_id=layer_id)
        try:
            return rows[0]
        finally:
            hold.release("never_imported")

    def source_descriptor(self, source: Any) -> dict[str, Any]:
        """Name an existing certified source for a coarse checkpoint reservation."""
        with self._lock:
            record = self._certificates.get(id(source.binding_token))
        if (
            record is None
            or record[0] is not source.binding_token
            or source.owner_lease is None
            or not source.owner_lease.is_valid()
        ):
            raise ValueError("Source descriptor no longer belongs to this importer")
        return {
            "identity": record[1],
            "manifest": record[2],
            "payload_digest": record[3],
        }

    def source_manifest(self, source: Any) -> GroupManifest:
        """Return the immutable certified allocation geometry for local consumers."""
        return self.source_descriptor(source)["manifest"]

    def _retire(
        self, identity: OperationIdentity, digest: bytes, lease_id: str, proof: str
    ) -> None:
        try:
            future = self.client.retire(identity, digest, lease_id, proof)
        except Exception as error:
            future = Future()
            error.__traceback__ = None
            future.set_exception(error)
        with self._lock:
            self.pending_retirements[lease_id] = (future, self.mapping)

        def acknowledged(done: Any) -> None:
            try:
                result = done.result()
            except BaseException:
                return
            if result.status == "RETIRED":
                self._ack_retirement(lease_id)

        future.add_done_callback(acknowledged)

    def _ack_retirement(self, lease_id: str) -> None:
        with self._lock:
            self.pending_retirements.pop(lease_id, None)
            self._live_leases.discard(lease_id)
            for token in [
                key
                for key, value in self._certificates.items()
                if value[2].lease_id == lease_id
            ]:
                self._certificates.pop(token)

    def wait_for_retirements(self, timeout_s: float) -> None:
        """Drain known local lifecycle acknowledgements during orderly shutdown."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                pending = tuple(self.pending_retirements.items())
            if not pending:
                return
            for lease_id, (future, _) in pending:
                result = future.result(timeout=max(0, deadline - time.monotonic()))
                if result.status != "RETIRED":
                    raise RuntimeError("Storage retirement was not acknowledged")
                self._ack_retirement(lease_id)

    def retire_unimported(
        self, identity: OperationIdentity, payload_digest: bytes, lease_ids: Any
    ) -> None:
        """Prove this participant never imported the rejected source leases."""
        for lease_id in lease_ids:
            with self._lock:
                if lease_id in self._live_leases:
                    # Imported and already-retiring holds need their own proof.
                    continue
                self._live_leases.add(lease_id)
            self._retire(identity, payload_digest, lease_id, "never_imported")

    def import_reply(
        self,
        reply: PreparedReply,
        *,
        identity: OperationIdentity,
        payload_digest: bytes,
        key_plan_digest: bytes,
        source_end: int,
    ) -> dict[int, Any]:
        """Validate once, upload bounded pointer tables and create local holds."""
        if reply.identity != identity:
            raise ValueError("Prepared source operation identity is stale")
        try:
            self._validate_reply(reply, source_end, key_plan_digest)
        except BaseException:
            leases = [group.lease_id for group in reply.groups if group.lease_id]
            if reply.restore_lease_id:
                leases.append(reply.restore_lease_id)
            self.retire_unimported(identity, payload_digest, leases)
            raise
        return self._import_validated(reply, identity, payload_digest, source_end)

    def _validate_reply(
        self, reply: PreparedReply, source_end: int, key_plan_digest: bytes
    ) -> None:
        if (
            reply.source_end != source_end
            or reply.slab_name != self.mapping.shm_name
            or reply.slab_size != self.mapping.size
            or reply.slab_generation != self.mapping.generation
        ):
            raise ValueError(
                "Prepared source identity or registered slab generation is stale"
            )
        group_ids = tuple(group.group_id for group in reply.groups)
        if not group_ids or 0 not in group_ids or len(set(group_ids)) != len(group_ids):
            raise ValueError("Prepared source requires unique groups including Group 0")
        for manifest in reply.groups:
            manifest.validate(
                self.layout,
                reply.slab_size,
                source_end,
                key_plan_digest=key_plan_digest,
                allow_checkpoint_tail="checkpoint_group1_tail" in reply.actions,
            )

    def _import_validated(
        self,
        reply: PreparedReply,
        identity: OperationIdentity,
        payload_digest: bytes,
        source_end: int,
    ) -> dict[int, Any]:
        sources = {}
        holds = []
        tables = []
        restore_hold = None
        try:

            def retire(lease_id: str, proof: str) -> None:
                self._retire(identity, payload_digest, lease_id, proof)

            if reply.restore_lease_id:
                restore_hold = LocalLeaseHandle(
                    lease_id=reply.restore_lease_id,
                    mapping=self.mapping,
                    retire=retire,
                )
                with self._lock:
                    self._live_leases.add(reply.restore_lease_id)
            for manifest in reply.groups:
                hold = LocalLeaseHandle(
                    lease_id=manifest.lease_id, mapping=self.mapping, retire=retire
                )
                with self._lock:
                    self._live_leases.add(manifest.lease_id)
                holds.append(hold)
                layers = len(self.layout.groups[manifest.group_id].layer_names)
                hold.mark_imported()
                table = self.upload(
                    source_pointer_table(manifest, self.local_device_base, layers)
                )
                tables.append(table)
                source = build_compact_prepared_sparse_source(
                    tuple(table.unbind(0)),
                    num_layers=layers,
                    total_tokens=source_end - manifest.source_start,
                    chunk_token_counts=tuple(
                        min(b, source_end) - a
                        for a, b in zip(
                            manifest.chunk_starts, manifest.chunk_ends, strict=True
                        )
                    ),
                    owner_lease=hold,
                    chunk_size=self.layout.chunk_size,
                    expected_pointer_device=table.device,
                    physical_chunk_token_counts=source_physical_token_counts(
                        manifest, layers
                    ),
                )
                sources[manifest.group_id] = source
                with self._lock:
                    self._certificates[id(source.binding_token)] = (
                        source.binding_token,
                        identity,
                        manifest,
                        payload_digest,
                    )
            if restore_hold is not None:
                restore_hold.mark_imported()
                with self._lock:
                    key = (identity.request_id, identity.load_generation)
                    if key in self._restore_holds:
                        raise ValueError(
                            "Checkpoint restore generation was imported twice"
                        )
                    self._restore_holds[key] = restore_hold
            return sources
        except BaseException:
            try:
                self.fence_upload()
            except BaseException:
                self._quarantined_imports.append(
                    (tuple(holds), tuple(tables), restore_hold)
                )
                raise
            if restore_hold is not None:
                # Only the successful return transfers this hold to an ACK.
                restore_hold.release("fenced")
            for hold in holds:
                hold.release("fenced")
            # Every other intended group also needs this participant's proof.
            for manifest in reply.groups[len(holds) :]:
                self._retire(
                    identity, payload_digest, manifest.lease_id, "never_imported"
                )
            raise
