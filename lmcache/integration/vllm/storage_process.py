# SPDX-License-Identifier: Apache-2.0
"""Worker-only execution of service-prepared decoder sources."""

# Standard
from concurrent.futures import Future, TimeoutError
from dataclasses import replace
from typing import Any, Generator
import time

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.rpc.storage_protocol import (
    PrepareSpec,
    PreparedReply,
    OperationIdentity,
    StorageLookupTicket,
    StorageResult,
    decode_message,
    encode_message,
    restore_hash,
)
from lmcache.integration.vllm.preemption_checkpoint import CheckpointRestoreMiss


def cancel_unadmitted_operation(client: Any, future: Any, timeout: float) -> Any:
    """Resolve BUSY against its original identity; only cancellation proves no work."""
    cancelled = client.cancel(future.identity).result(timeout)
    summary = cancelled.value
    if cancelled.status == "CANCELLED_NOT_STARTED":
        if not isinstance(summary, dict) or not isinstance(
            summary.get("terminal_result_digest"), bytes
        ):
            raise RuntimeError("CPU cancellation omitted its canonical terminal proof")
        acknowledgement = client.acknowledge_terminal(
            future.identity,
            {"terminal_result_digest": summary["terminal_result_digest"]},
        ).result(timeout)
        if acknowledgement.status != "ACKNOWLEDGED":
            raise RuntimeError("CPU cancellation terminal proof was not acknowledged")
    elif isinstance(summary, dict) and summary.get("result") is not None:
        return msgspec.convert(summary["result"], type=StorageResult)
    return cancelled


def submit_prepare(adapter: Any, plan: dict[str, Any]) -> None:
    """Overlap one TP0 CPU operation while retaining the scheduler's exact ticket."""
    engine = adapter.lmcache_engine
    request = plan["request"]
    spec = request.load_spec
    future = None
    try:
        ticket = msgspec.convert(spec.cpu_source_ticket, type=StorageLookupTicket)
        if (
            ticket.identity.request_id != request.req_id
            or not 0 < plan["token_count"] <= ticket.reply.matched_tokens
        ):
            raise ValueError("Cold source ticket does not cover this request")
        layout = engine.storage_layout
        chunks = tuple(
            (start, end, restore_hash(digest, layout.hash_width, layout.hash_kind))
            for start, end, digest in zip(
                ticket.reply.starts, ticket.reply.ends, ticket.reply.hashes, strict=True
            )
        )
        plan["chunk_plan"] = lambda: chunks
        plan["storage_ticket"] = ticket
        if not engine._is_passive():
            mode = plan.get("storage_mode") or (
                "checkpoint_restore"
                if hasattr(spec, "checkpoint_generation")
                else "resident"
                if getattr(spec, "dsa_cold_resident_load", False)
                else "compact"
            )
            prepared = PrepareSpec(
                ticket.reply.ticket_id,
                ticket.reply.key_plan_digest,
                plan["token_count"],
                layout.digest,
                tuple(
                    name
                    for name in engine.storage_process.descriptor["sessions"]
                    if name.startswith("worker-")
                ),
                mode=mode,
                checkpoint_generation=getattr(spec, "checkpoint_generation", None),
            )
            future = engine.storage_process.client.request(
                "prepare",
                prepared,
                request_id=request.req_id,
                request_attempt=ticket.identity.request_attempt,
                load_generation=plan.get(
                    "storage_generation", getattr(spec, "dsa_cold_load_generation", 0)
                ),
                purpose=mode,
            )
    except BaseException as error:
        # Priming failure still occupies this request's ordered publication slot.
        future = Future()
        future.set_exception(error)
    plan["storage_prepare"] = future


def _exchange_reply(adapter: Any, plan: dict[str, Any]) -> StorageResult | None:
    engine = adapter.lmcache_engine
    request = plan["request"]
    identity = dict(
        req_id=request.req_id,
        phase=plan.get("storage_phase", "storage_process_prepare"),
        request_ordinal=plan.get(
            "storage_generation",
            getattr(request.load_spec, "dsa_cold_load_generation", 0),
        ),
        layer_id=0,
        kv_group=0,
    )
    if not engine._is_passive():
        result, error = None, None
        details = {}
        if plan.get("storage_source_plan") is not None:
            details["source_plan"] = plan["storage_source_plan"]
        future = plan["storage_prepare"]
        timeout = 0.0
        not_started = False
        try:
            timeout = max(
                float(engine.config.blocking_timeout_secs),
                float(engine.config.lookup_timeout_ms) / 1000,
                float(engine.config.remote_fill_native_hard_timeout_ms) / 1000,
            )
            try:
                result = future.result(timeout) if future is not None else None
                if result is not None and result.status == "BUSY":
                    result = cancel_unadmitted_operation(
                        engine.storage_process.client, future, timeout
                    )
                    not_started = result.status == "CANCELLED_NOT_STARTED"
            except TimeoutError as timeout_error:
                engine.storage_process.client.cancel(future.identity).result(timeout)
                raise RuntimeError(
                    "CPU preparation timed out; the original operation was cancelled"
                ) from timeout_error
            if result is not None and (result.status != "COMPLETED" or result.error):
                raise RuntimeError(result.error or result.status)
        except BaseException as exc:
            error = exc
            if hasattr(future, "identity") and not not_started:
                try:
                    status = engine.storage_process.client.operation_status(
                        future.identity
                    ).result(timeout)
                    if status.status == "UNKNOWN" or not isinstance(status.value, dict):
                        raise RuntimeError("CPU operation ownership remains unknown")
                    summary = status.value
                    details.update(
                        {
                            "operation_identity": msgspec.msgpack.encode(
                                future.identity
                            ),
                            "payload_digest": future.payload_digest,
                            "lease_ids": tuple((summary or {}).get("leases", {})),
                        }
                    )
                except BaseException:
                    # Failure to reconcile never permits allocating again or freeing.
                    # Publish this failure in the same TP slot before fail-stop,
                    # so passive ranks do not wait for a success envelope.
                    details["ownership_unknown"] = True
        if result is not None:
            details["storage_result"] = encode_message(result)
        elif error is None:
            details["storage_skip"] = True
        envelope = engine._shared_layerwise_error_envelope(
            **identity,
            message=str(error) if error else "",
            details=details,
        )
        if error is None:
            envelope = replace(envelope, status="skipped")
        engine._broadcast_shared_envelope(envelope)
    else:
        envelope = engine._receive_matching_shared_envelope(**identity)
    engine._validate_shared_layerwise_envelope(
        replace(envelope, status="skipped"), **identity
    )
    if "source_plan" in (envelope.error_details or {}):
        plan["storage_source_plan"] = envelope.error_details["source_plan"]
    if envelope.status != "skipped":
        details = envelope.error_details or {}
        if details.get("ownership_unknown"):
            engine.storage_process.mark_failed(
                "CPU preparation ownership could not be reconciled"
            )
            raise RuntimeError("CPU preparation ownership could not be reconciled")
        if details.get("operation_identity"):
            original = msgspec.msgpack.decode(
                details["operation_identity"], type=OperationIdentity
            )
            if (
                original.request_id != request.req_id
                or original.load_generation != identity["request_ordinal"]
                or original.request_attempt
                != (
                    plan.get("storage_attempt")
                    or plan["storage_ticket"].identity.request_attempt
                )
            ):
                raise ValueError(
                    "Failed CPU operation identity does not match this load"
                )
            engine.compact_source_importer.retire_unimported(
                original,
                details["payload_digest"],
                details["lease_ids"],
            )
        if details.get("storage_result"):
            failed = decode_message(details["storage_result"], StorageResult)
            if failed.status == "FAILED_SAFE" and plan.get("storage_allow_safe_miss"):
                return failed
            value = failed.value
            if isinstance(value, dict) and value.get("promotion_refused"):
                return failed
            if isinstance(value, dict) and "checkpoint_miss_end" in value:
                end = value["checkpoint_miss_end"]
                if type(end) is not int or not 0 <= end < plan["token_count"]:
                    raise ValueError("Invalid checkpoint restore-miss frontier")
                raise CheckpointRestoreMiss(end, envelope.message)
        raise RuntimeError(envelope.message or "CPU source preparation failed")
    if envelope.error_details.get("storage_skip"):
        return None
    return decode_message(envelope.error_details["storage_result"], StorageResult)


def import_ordinary_sources(
    adapter: Any, plan: dict[str, Any], state: Any, *, request_ordinal: int = 0
) -> bool:
    """Import a paired dense prefix into private state after ordered preparation.

    The caller owns publication and cleanup. Return False for a proven safe
    preparation miss; failed imports retain their adopted leases on ``state``.
    """
    engine, request = adapter.lmcache_engine, plan["request"]
    connector = engine.gpu_connector
    result = _exchange_reply(adapter, plan)
    if result is None or result.status != "COMPLETED":
        if not engine._is_passive():
            engine.storage_process.client.acknowledge_terminal(
                plan["storage_prepare"].identity
            )
        return False
    ticket = plan["storage_ticket"]
    generation = plan["storage_generation"]
    sources, error = {}, None
    try:
        reply = msgspec.convert(result.value, type=PreparedReply)
        if (
            result.identity.request_id != request.req_id
            or result.identity.request_attempt != ticket.identity.request_attempt
            or result.identity.load_generation != generation
        ):
            raise ValueError("Ordinary source belongs to another request lifetime")
        with torch.npu.stream(connector.load_stream):
            sources = engine.compact_source_importer.import_reply(
                reply,
                identity=result.identity,
                payload_digest=result.payload_digest,
                key_plan_digest=ticket.reply.key_plan_digest,
                source_end=plan["token_count"],
            )
            readiness = connector.record_dense_load_readiness()
        if set(sources) != {0, 1}:
            raise ValueError("Ordinary prefix preparation requires both KV groups")
    except BaseException as exc:
        error = exc
    state.prepared_sparse_sources = sources
    ready = engine._ordered_checkpoint_agreement(
        request.req_id,
        "storage_dense_import",
        generation,
        request_ordinal,
        error is None,
    )
    if error is not None:
        raise error
    if not ready:
        raise RuntimeError("Ordinary source import failed on another TP rank")
    state.process_load_generation = generation
    state.process_source_plan = {
        "starts": tuple(ticket.reply.starts),
        "ends": tuple(ticket.reply.ends),
        "hashes": tuple(ticket.reply.hashes),
    }
    state.token_count = plan["token_count"]
    state.metadata_token_ids = list(plan["tokens"])
    state.location, state.metadata_warm = "LocalCPU", True
    state.dense_load_readiness = readiness
    state.dense_prefix_seed = not request.is_sparse_decode
    return True


def ordinary_layerwise_load(
    adapter: Any,
    request: Any,
    tokens: list[int],
    mask: torch.Tensor,
    slot_mapping: torch.Tensor,
    indexer_slots: torch.Tensor,
    *,
    request_ordinal: int,
) -> tuple[Any, Any]:
    """Prepare an ordinary paired prefix without a worker storage manager.

    The scheduler's cold/graph admission decisions remain unchanged. Only source
    preparation is coarse; the existing per-layer generator still runs at the
    original model callbacks, with excluded vLLM-cached rows left untouched.
    """
    from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState

    engine, connector = adapter.lmcache_engine, adapter.lmcache_engine.gpu_connector
    ticket = msgspec.convert(
        request.load_spec.cpu_source_ticket, type=StorageLookupTicket
    )
    count = len(tokens)
    generation = ticket.reply.lookup_generation
    plan = {
        "request": request,
        "tokens": tokens,
        "token_mask": mask,
        "token_count": count,
        "storage_mode": "dense",
        "storage_generation": generation,
        "storage_phase": "storage_process_dense",
        "storage_allow_safe_miss": True,
    }
    coordinator = adapter._get_cold_load_coordinator()
    previous, terminal = coordinator.last_latent_future, Future()
    coordinator.last_latent_future = terminal
    state = WorkerRetrieveState(req_id=request.req_id)
    stats = getattr(engine, "stats_monitor", None)
    monitor_requests = (
        {group: stats.on_retrieve_request(int(mask.sum().item())) for group in (0, 1)}
        if stats is not None
        else {}
    )
    adopted = False
    try:
        submit_prepare(adapter, plan)
        if previous is not None:
            previous.exception()  # Ordered publication includes the whole predecessor.
        prepared = import_ordinary_sources(
            adapter, plan, state, request_ordinal=request_ordinal
        )
        if not prepared:
            # An explicit safe CPU refusal has no exported sources. Keep the
            # ordinary missing-block validation and recomputation fallback.

            def missing(group: int) -> Generator[Any, Any, None]:
                empty = torch.zeros(count, dtype=torch.bool)
                for _ in range(
                    adapter._num_layers_for_group(group)
                    + (not request.is_sparse_decode)
                ):
                    yield None
                if stats is not None:
                    stats.on_retrieve_finished(monitor_requests[group], 0)
                yield empty

            return missing(0), missing(1)
        sources = state.prepared_sparse_sources
        readiness = state.dense_load_readiness
        state.slot_mapping, state.indexer_slot_mapping = slot_mapping, indexer_slots
        state.dense_prefix_seed = not request.is_sparse_decode
        completed = set()

        def retrieve(group: int, mapping: torch.Tensor) -> Generator[Any, Any, None]:
            # Preserve the mask without changing physical source coverage or
            # scheduler computed tokens. This is admission-time metadata only.
            if request.is_sparse_decode and group == 0:
                slots = mapping  # Dynamic selected/target rows own sparse slots.
            else:
                slots = mapping[:count]
                if slots.numel() != count:
                    raise ValueError("Ordinary prefix destination slots are incomplete")
                slots = slots.clone()
                slots.masked_fill_(~mask.to(device=slots.device, dtype=torch.bool), -1)
            consumer = None
            try:
                source = sources[group]
                operands = source.compact_transfer_metadata
                if operands is None:
                    operands = {}
                    object.__setattr__(source, "compact_transfer_metadata", operands)
                # NativeSourceLifetime retains the source through its exact
                # final stream event, including generator close/unknown DMA.
                operands["ordinary_host_slots"] = slots
                group_readiness = readiness
                if slots.device.type == "cpu":
                    with torch.npu.stream(connector.load_stream):
                        slots = connector.stage_dense_load_tensor(
                            slots, dtype=torch.long
                        )
                        group_readiness = connector.record_dense_load_readiness()
                        state.dense_load_readiness = group_readiness
                operands["ordinary_slots"] = slots
                operands["ordinary_readiness"] = group_readiness
                # Each actual consuming stream receives its own dependency.
                connector.consume_dense_load_readiness(group_readiness)
                consumer = engine._retrieve_prepared_sparse_layers(
                    tokens,
                    {
                        "prepared_sparse_source": sources[group],
                        "kvcaches": adapter._kvcaches_for_group(group),
                        "slot_mapping": slots,
                        "kv_group": group,
                        "req_id": request.req_id,
                        "ret_mask": mask.clone(),
                    },
                )
                value = next(consumer)
                payload = yield value
                for layer in range(len(sources[group].layers)):
                    value = consumer.send(payload)
                    if layer == len(sources[group].layers) - 1:
                        completed.add(group)
                        if group == 1:
                            state.indexer_npu_resident = True
                        if completed == {0, 1}:
                            if not request.is_sparse_decode:
                                state.dense_prefix_resident_tokens = count
                            record_state(adapter, state, request)
                        if request.is_sparse_decode and stats is not None:
                            stats.on_retrieve_finished(
                                monitor_requests[group], mask.sum()
                            )
                    payload = yield (
                        None
                        if not request.is_sparse_decode
                        and layer == len(sources[group].layers) - 1
                        else value
                    )
                if stats is not None and not request.is_sparse_decode:
                    stats.on_retrieve_finished(monitor_requests[group], mask.sum())
                if not request.is_sparse_decode:
                    yield mask.clone()
            finally:
                if consumer is not None:
                    consumer.close()

        adapter._drop_worker_retrieve_state(request.req_id)
        adapter._set_worker_retrieve_state(request.req_id, state)
        adopted = True
        return retrieve(0, slot_mapping), retrieve(1, indexer_slots)
    finally:
        if not adopted and state.prepared_sparse_sources:
            try:
                connector.synchronize_dense_load_stream()
                release_sources(state, engine)
            except BaseException:
                engine.__dict__.setdefault("_failed_process_source_reads", []).append(
                    state
                )
        terminal.set_result(None)


def _dense_source(
    adapter: Any, source: Any, plan: dict[str, Any], group: int, start: int = 0
) -> Any:
    """Use the existing prepared direct kernel's implicit dense selection."""
    engine = adapter.lmcache_engine
    connector = engine.gpu_connector
    held = source.owner_lease.retain()
    retriever = None
    try:
        held.mark_imported()
        with torch.npu.stream(connector.load_stream):
            slots = (
                plan["request"].slot_mapping[0]
                if group == 0
                else plan["indexer_slots_cpu"][start:]
            )
            if plan.get("storage_mode") == "dense":
                slots = slots[: plan["token_count"]].clone()
                if slots.numel() != plan["token_count"]:
                    raise ValueError("Ordinary prefix destination slots are incomplete")
                slots.masked_fill_(
                    ~plan["token_mask"].to(device=slots.device, dtype=torch.bool), -1
                )
            slots = connector.stage_dense_load_tensor(slots, dtype=torch.long)
            retriever = engine._retrieve_prepared_sparse_layers(
                plan["tokens"],
                {
                    "prepared_sparse_source": source,
                    "kvcaches": plan[
                        "latent_kvcaches" if group == 0 else "indexer_kvcaches"
                    ],
                    "slot_mapping": slots,
                    "kv_group": group,
                    "sync": True,
                    "req_id": plan["request"].req_id,
                },
            )
            next(retriever)
            for _ in source.layers:
                retriever.send(None)
            readiness = connector.record_dense_load_readiness()
        connector.synchronize_dense_load_readiness(readiness)
        held.release("fenced")
        return readiness
    except BaseException:
        try:
            connector.synchronize_dense_load_stream()
        except BaseException:
            engine.__dict__.setdefault("_failed_process_source_reads", []).append(
                (held, source, retriever)
            )
            raise
        held.release("fenced")
        raise
    finally:
        if retriever is not None:
            retriever.close()


def run_ordinary(
    adapter: Any,
    plan: dict[str, Any],
    device: int | None,
    previous: Future | None,
) -> Any:
    """Load both ordinary groups off-thread and return unpublished private state.

    This job participates in the same TP publication FIFO as compact loads.
    All device waits and source preparation stay on the existing loader pool.
    """
    from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState

    if device is not None:
        torch.npu.set_device(device)
    engine = adapter.lmcache_engine
    request = plan["request"]
    state = WorkerRetrieveState(req_id=request.req_id)
    stats = getattr(engine, "stats_monitor", None)
    monitors = (
        [stats.on_retrieve_request(int(plan["token_mask"].sum())) for _ in (0, 1)]
        if stats is not None
        else []
    )
    finished_groups: set[int] = set()
    try:
        if previous is not None:
            previous.exception()
        if not import_ordinary_sources(adapter, plan, state):
            raise RuntimeError("Ordinary CPU prefix preparation missed safely")
        for group in (0, 1):
            _dense_source(adapter, state.prepared_sparse_sources[group], plan, group)
            if stats is not None:
                stats.on_retrieve_finished(monitors[group], plan["token_mask"].sum())
                finished_groups.add(group)
        state.slot_mapping = request.slot_mapping[0]
        state.indexer_slot_mapping = plan["indexer_slots_cpu"]
        state.indexer_npu_resident = True
        state.dense_prefix_resident_tokens = plan["token_count"]
        adapter._record_dsa_cold_dense_load_readiness(state)
        state._dsa_cold_load_completed_at = time.perf_counter()
        return state
    except BaseException as error:
        if stats is not None:
            for group in (0, 1):
                if group not in finished_groups:
                    stats.on_retrieve_finished(monitors[group], 0)
        try:
            adapter._synchronize_dsa_cold_dense_load()
            release_sources(state, engine)
        except BaseException:
            error._lmcache_dsa_cold_state = state
            engine.__dict__.setdefault("_failed_process_source_reads", []).append(state)
        raise


def run_indexer(adapter: Any, plan: dict[str, Any], device: int | None) -> Any:
    """Keep native Group-1 I/O local; checkpoint tails use separate CPU leases."""
    if device is not None:
        torch.npu.set_device(device)
    request = plan["request"]
    spec = request.load_spec
    engine = adapter.lmcache_engine
    checkpoint = hasattr(spec, "checkpoint_generation")
    if not checkpoint and spec.dsa_group1_direct_hbm:
        engine.load_group1_pages_direct(
            plan["tokens"],
            plan["indexer_slots_cpu"],
            plan["indexer_kvcaches"],
            request.request_configs,
            request.req_id,
            chunk_plan=plan["chunk_plan"],
        )
        return plan["token_mask"], None, 0.0, 0.0
    plan["latent_shared_ready"].result()
    sources = plan["storage_sources"]
    base = 0
    if checkpoint:
        base = (
            spec.checkpoint_prefix_end
            // adapter._lmcache_chunk_size
            * adapter._lmcache_chunk_size
        )
        error = None
        try:
            if base:
                chunks = tuple(
                    chunk for chunk in plan["chunk_plan"]() if chunk[1] <= base
                )
                engine.load_group1_pages_direct(
                    plan["tokens"][:base],
                    plan["indexer_slots_cpu"][:base],
                    plan["indexer_kvcaches"],
                    request.request_configs,
                    request.req_id,
                    chunk_plan=lambda: chunks,
                )
        except BaseException as exc:
            error = exc
        ready = engine.finish_checkpoint_prefix(request, error is None)
        if error is not None:
            raise error
        if not ready:
            raise RuntimeError("Checkpoint prefix failed on another TP rank")
    source = sources.get(1)
    readiness = None
    if source is not None:
        readiness = _dense_source(adapter, source, plan, 1, base)
        engine.native_source_lifetime.retire_source(source)
        source.owner_lease.release("fenced")
        sources.pop(1)
    elif base != plan["token_count"]:
        raise RuntimeError("CPU preparation omitted the required Group-1 source")
    return plan["token_mask"], readiness, 0.0, 0.0


def run_cold(
    adapter: Any,
    plan: dict[str, Any],
    device: int | None,
    indexer: Future,
    previous: Future | None = None,
    live_state: Any = None,
) -> Any:
    """Publish manifests in the existing complete predecessor-Future order."""
    from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState

    if device is not None:
        torch.npu.set_device(device)
    engine = adapter.lmcache_engine
    request = plan["request"]
    state = live_state or WorkerRetrieveState(req_id=request.req_id)
    try:
        if previous is not None:
            try:
                previous.exception()
            except BaseException:
                pass
        result = _exchange_reply(adapter, plan)
        if result is None:
            raise RuntimeError("Cold CPU preparation cannot skip publication")
        reply = msgspec.convert(result.value, type=PreparedReply)
        error = None
        sources = {}
        try:
            ticket = plan["storage_ticket"]
            expected = request.load_spec.dsa_cold_load_generation
            if (
                result.identity.request_attempt != ticket.identity.request_attempt
                or result.identity.request_id != request.req_id
                or result.identity.load_generation != expected
            ):
                raise ValueError("CPU source result belongs to another request attempt")
            coordinator = getattr(adapter, "_cold_load_coordinator", None)
            if coordinator is not None and request.req_id in (
                coordinator.aborted or ()
            ):
                engine.compact_source_importer.retire_unimported(
                    result.identity,
                    result.payload_digest,
                    tuple(group.lease_id for group in reply.groups)
                    + ((reply.restore_lease_id,) if reply.restore_lease_id else ()),
                )
                raise RuntimeError("Cancelled CPU source generation cannot be imported")
            sources = engine.compact_source_importer.import_reply(
                reply,
                identity=result.identity,
                payload_digest=result.payload_digest,
                key_plan_digest=plan["storage_ticket"].reply.key_plan_digest,
                source_end=plan["token_count"],
            )
        except BaseException as exc:
            error = exc
        state.prepared_sparse_sources = sources
        ready = engine._ordered_checkpoint_agreement(
            request.req_id,
            "storage_source_import",
            request.load_spec.dsa_cold_load_generation,
            0,
            error is None,
        )
        if error is not None:
            raise error
        if not ready:
            raise RuntimeError("CPU source import failed on another TP rank")
        plan["storage_sources"] = sources
        plan["latent_shared_ready"].set_result(None)
        if getattr(request.load_spec, "dsa_cold_resident_load", False):
            _dense_source(adapter, sources[0], plan, 0)
            state.dense_prefix_resident_tokens = plan["token_count"]
        indexer.result()
        state.indexer_npu_resident = True
        state.location = "LocalCPU"
        state.metadata_warm = True
        state.token_count = plan["token_count"]
        state.metadata_token_ids = plan["tokens"]
        state.process_load_generation = request.load_spec.dsa_cold_load_generation
        ticket = plan["storage_ticket"].reply
        state.process_source_plan = {
            "starts": tuple(ticket.starts),
            "ends": tuple(ticket.ends),
            "hashes": tuple(ticket.hashes),
        }
        state.cold_publication_proof = (
            request.req_id,
            request.load_spec.dsa_cold_load_generation,
            engine.shared_cpu_cache_generation,
            sources[0],
        )
        adapter._record_dsa_cold_dense_load_readiness(state)
        state._dsa_cold_load_completed_at = time.perf_counter()
        return state
    except BaseException as error:
        gate = plan["latent_shared_ready"]
        if not gate.done():
            gate.set_exception(error)
        try:
            try:
                indexer.result()
            except BaseException:
                pass  # Terminal sibling errors do not skip exact stream cleanup.
            adapter._synchronize_dsa_cold_dense_load()
            release_sources(state, engine)
        except BaseException:
            error._lmcache_dsa_cold_state = state
            engine.__dict__.setdefault("_failed_process_source_reads", []).append(state)
        raise


def record_state(adapter: Any, state: Any, request: Any) -> None:
    """Adopt only importer-certified ownership; no allocator proxy registration."""
    engine = adapter.lmcache_engine
    generation = getattr(state, "process_load_generation", 0)
    sources = state.prepared_sparse_sources
    if (
        0 not in sources
        or not state.indexer_npu_resident
        or any(
            not engine.compact_source_importer.is_certified_source(
                source, request.req_id, generation
            )
            or source.total_tokens != state.token_count
            or len(source.layers) != adapter._num_layers_for_group(group)
            for group, source in sources.items()
        )
    ):
        raise RuntimeError(
            "Process publication requires certified sources and both groups ready"
        )
    state.shared_generation = engine.shared_cpu_cache_generation
    state.pointer_cache_generation = state.shared_generation
    state.shared_request_active = True
    state.shared_latent_status = state.shared_index_status = "present"


def release_sources(state: Any, engine: Any) -> None:
    """Release request holds after the caller proves its existing native fence."""
    for source in state.prepared_sparse_sources.values():
        if source.owner_lease is not None:
            engine.native_source_lifetime.retire_source(source)
            source.owner_lease.release("fenced")


def release_shutdown_states(adapter: Any) -> None:
    """Drain completed cold and foreground request holds before stopping M."""
    states = dict(getattr(adapter, "_worker_retrieve_state", {}))
    coordinator = getattr(adapter, "_cold_load_coordinator", None)
    if coordinator is not None:
        for entry in tuple(coordinator.futures.values()):
            try:
                state = entry[1].result()
            except BaseException:
                continue  # Uncertain owners stay with their failed future/recovery.
            states.setdefault(id(state), state)
    released = set()
    for state in states.values():
        if id(state) in released:
            continue
        adapter._release_shared_worker_retrieve_state(state, adapter.lmcache_engine)
        released.add(id(state))


def reserve_store_step(adapter: Any) -> Future | None:
    """Insert one foreground save boundary into the existing publication chain."""
    slot = getattr(adapter, "_process_store_step_slot", None)
    if slot is None:
        coordinator = adapter._get_cold_load_coordinator()
        prior = coordinator.last_latent_future
        terminal = Future()
        adapter._process_store_step_slot = (prior, terminal)
        coordinator.last_latent_future = terminal
        return prior
    return slot[0]


def finish_store_step(adapter: Any) -> None:
    """Release later cold publications only after every foreground save slot."""
    slot = getattr(adapter, "_process_store_step_slot", None)
    if slot is not None:
        if not slot[1].done():
            slot[1].set_result(None)
        adapter._process_store_step_slot = None


def promote_store(adapter: Any, request: Any, result: Any) -> None:
    """Replace a completed paired store with one newly proved all-rank source."""
    from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState

    engine = adapter.lmcache_engine
    window = (
        request.req_id,
        getattr(request, "decode_window_start", None),
        getattr(request, "decode_window_end", None),
        len(request.token_ids),
    )
    pending = adapter.__dict__.setdefault("_process_store_promotions", {})
    parts = pending.setdefault(window, {})
    parts[result.kv_group] = result
    if set(parts) != {0, 1}:
        return
    pending.pop(window)
    if adapter._is_decode_window_save_request(
        request
    ) and not adapter._decode_window_save_is_next_expected(request):
        return
    old = adapter._worker_retrieve_state.get(request.req_id)
    generation = (
        max(
            int(getattr(old, "process_load_generation", 0)),
            int(getattr(request.load_spec, "dsa_cold_load_generation", 0)),
        )
        + 1
    )
    source = None if old is None else old.prepared_sparse_sources.get(0)
    descriptor = (
        engine.compact_source_importer.source_descriptor(source)
        if source is not None
        else None
    )
    ticket = getattr(request.load_spec, "cpu_source_ticket", None)
    attempt = (
        descriptor["identity"].request_attempt
        if descriptor is not None
        else ticket.identity.request_attempt
        if ticket is not None
        else f"store-{request.req_id}-{generation}"
    )
    job = {
        "request": request,
        "storage_phase": "storage_process_store",
        "storage_generation": generation,
        "storage_attempt": attempt,
        "storage_prepare": None,
        "token_count": len(request.token_ids),
    }
    source_plan = None
    if not engine._is_passive():
        try:
            first, second = parts[0].process_store_plan, parts[1].process_store_plan
            if first and second:
                suffix = tuple(
                    zip(first["starts"], first["ends"], first["hashes"], strict=True)
                )
                peer = tuple(
                    zip(second["starts"], second["ends"], second["hashes"], strict=True)
                )
                stop = min(parts[0].committed_end, parts[1].committed_end)
                suffix = tuple(item for item in suffix if item[1] <= stop)
                peer = tuple(item for item in peer if item[1] <= stop)
                if suffix != peer:
                    raise ValueError(
                        "Paired store groups disagree on their canonical chunks"
                    )
                prior = getattr(old, "process_source_plan", None)
                prefix = (
                    tuple(
                        zip(
                            prior["starts"], prior["ends"], prior["hashes"], strict=True
                        )
                    )
                    if prior
                    else ()
                )
                combined = (
                    tuple(item for item in prefix if suffix and item[1] <= suffix[0][0])
                    + suffix
                )
                cursor = 0
                for start, end, _ in combined:
                    if start != cursor or not start < end:
                        raise ValueError(
                            "Store promotion has a hole in its complete prefix"
                        )
                    cursor = end
                if (
                    combined
                    and cursor > int(getattr(old, "token_count", 0))
                    and cursor <= len(request.token_ids)
                ):
                    source_plan = {
                        "starts": tuple(a for a, _, _ in combined),
                        "ends": tuple(b for _, b, _ in combined),
                        "hashes": tuple(h for _, _, h in combined),
                    }
                    payload = dict(
                        plan=dict(
                            source_plan,
                            request_configs=dict(request.request_configs or {}),
                        ),
                        source_end=cursor,
                        layout_digest=engine.storage_layout.digest,
                        participants=tuple(
                            name
                            for name in engine.storage_process.descriptor["sessions"]
                            if name.startswith("worker-")
                        ),
                        mode="compact",
                    )
                    if descriptor is not None:
                        payload["source_operation"] = descriptor
                    job["storage_prepare"] = engine.storage_process.client.request(
                        "prepare_store",
                        payload,
                        request_id=request.req_id,
                        request_attempt=attempt,
                        load_generation=generation,
                        purpose="store_promotion",
                    )
        except BaseException as error:
            job["storage_prepare"] = Future()
            job["storage_prepare"].set_exception(error)
    # Each store completion already waited for the save step's predecessor.
    # Later cold publications depend on its foreground slot until wait_for_save.
    job["storage_source_plan"] = source_plan
    prepared = _exchange_reply(adapter, job)
    if prepared is None or prepared.status == "FAILED_SAFE":
        return
    reply = msgspec.convert(prepared.value, type=PreparedReply)
    sources, error = {}, None
    state = WorkerRetrieveState(req_id=request.req_id)
    try:
        if (
            prepared.identity.request_id != request.req_id
            or prepared.identity.request_attempt != attempt
            or prepared.identity.load_generation != generation
        ):
            raise ValueError("Store promotion belongs to a stale request lifetime")
        # This coarse operation performs the authoritative paired lookup itself;
        # its authenticated original payload binds the new complete key plan.
        sources = engine.compact_source_importer.import_reply(
            reply,
            identity=prepared.identity,
            payload_digest=prepared.payload_digest,
            key_plan_digest=reply.groups[0].key_plan_digest,
            source_end=reply.source_end,
        )
        state.prepared_sparse_sources = sources
        state.location, state.metadata_warm = "LocalCPU", True
        state.token_count = reply.source_end
        state.metadata_token_ids = request.token_ids[: reply.source_end]
        state.process_load_generation = generation
        state.process_source_plan = job["storage_source_plan"]
        state.indexer_npu_resident = True
        if old is not None:
            state.slot_mapping = old.slot_mapping
            state.indexer_slot_mapping = old.indexer_slot_mapping
        adapter._record_dsa_cold_dense_load_readiness(state)
        record_state(adapter, state, request)
    except BaseException as exc:
        error = exc
    ready = engine._ordered_checkpoint_agreement(
        request.req_id,
        "storage_store_import",
        generation,
        reply.source_end,
        error is None,
    )
    if error is not None or not ready:
        try:
            adapter._synchronize_dsa_cold_dense_load()
            adapter._release_shared_worker_retrieve_state(state, engine)
        except BaseException:
            engine.__dict__.setdefault("_failed_process_source_reads", []).append(state)
            engine.storage_process.mark_failed(
                "Store source publication completion is unknown"
            )
            raise
        if error is not None:
            raise error
        raise RuntimeError("Store source import failed on another TP rank")
    adapter._set_worker_retrieve_state(request.req_id, state)
    if old is not None:
        adapter._release_shared_worker_retrieve_state(old, engine)
    for group, item in parts.items():
        item.process_store_plan["promoted_end"] = reply.source_end
        adapter._record_decode_window_save_group_completed(request, group, item)
