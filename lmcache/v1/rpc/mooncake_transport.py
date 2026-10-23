# SPDX-License-Identifier: Apache-2.0
"""Bounded Mooncake I/O into an existing shared slab; no KV payload IPC."""

# Standard
from threading import BoundedSemaphore, Lock, local
from types import SimpleNamespace
from typing import Any, Callable
from weakref import ref
import ctypes
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cpu_cache import StorageFailedSafe, StorageOperationRegistry
from lmcache.v1.serving_perf import (
    serving_perf_enabled,
    serving_perf_log,
    serving_perf_measure,
    serving_perf_phase,
    serving_perf_record,
    serving_perf_context,
    serving_perf_scope,
)
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import (
    StorageResult,
    checked_uint,
    content_digest,
    pack_uints,
    unpack_uints,
)

MAX_KEYS = 256
logger = init_logger(__name__)
MAX_RANGES = 1024
MAX_CONTROL_BYTES = 256 * 1024
_READS = frozenset(("batch_get_into", "batch_get_into_multi_buffers"))
_WRITES = frozenset(
    ("put_from", "batch_put_from", "batch_put_from_multi_buffers", "put_parts")
)


def start_worker_transport(
    config: Any, metadata: Any, mapping: Any, descriptor: dict[str, Any]
) -> Any:
    """Start W0's full Store on its existing GlobalTE, before memory profiling."""
    from lmcache.v1.storage_backend.connector import parse_remote_url
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakestoreConnector,
    )
    from lmcache.v1.storage_service import StorageService
    import torch

    transport = descriptor["transport"]
    parsed = parse_remote_url(config.remote_url)
    loop = asyncio.new_event_loop()
    connector = MooncakestoreConnector(
        parsed.host,
        parsed.port,
        parsed.query_params.get("device", [""])[0],
        loop,
        None,
        config,
        metadata,
        defer_layout=True,
        native_owner_mapping=mapping,
    )
    device = torch.npu.current_device()

    def close_native() -> None:
        torch.npu.set_device(device)
        asyncio.run(connector.close())
        loop.close()

    try:
        if (
            connector.registered_buffer_ptr != mapping.ptr
            or connector.registered_buffer_size != mapping.size
            or not connector.get_registered_transport_session()
        ):
            raise RuntimeError(
                "Worker Mooncake registration does not cover its shared host mapping"
            )
        runtime = WorkerMooncakeTransport(
            connector.store,
            mapping,
            instance=descriptor["cache_instance_id"],
            epoch=descriptor["service_epoch"],
            nonce=bytes.fromhex(transport["nonce"]),
            session=connector.get_registered_transport_session(),
            replica_factory=connector.replica_config.__class__,
            max_bytes=config.remote_fill_max_inflight_bytes,
            close_native=close_native,
            initialize_thread=lambda: torch.npu.set_device(device),
        )
        service = StorageService(
            runtime,
            transport["work"],
            transport["lifecycle"],
            cache_instance_id=descriptor["cache_instance_id"],
            service_epoch=descriptor["service_epoch"],
            sessions={"storage-owner": bytes.fromhex(transport["nonce"])},
        )
        service.start()
        return service
    except BaseException:
        close_native()
        raise


class WorkerMooncakeTransport:
    """Execute explicit native jobs, independently of the model and G1 reader."""

    work_lanes = {
        "native_exists": "lookup",
        "native_get": "bulk",
        "native_put": "persist",
    }

    def __init__(
        self,
        store: Any,
        mapping: Any,
        *,
        instance: str,
        epoch: str,
        nonce: bytes,
        session: str,
        replica_factory: Callable[[], Any],
        max_bytes: int,
        close_native: Callable[[], None],
        initialize_thread: Callable[[], None] = lambda: None,
    ) -> None:
        self.store, self.mapping = store, mapping
        self.instance, self.epoch = instance, epoch
        self.session, self.replica_factory = session, replica_factory
        self.max_bytes, self.close_native = max_bytes, close_native
        self.registry = StorageOperationRegistry(
            instance, epoch, {"storage-owner": nonce}
        )
        self._closed = False
        self._thread = local()
        self._initialize_thread = initialize_thread

    def descriptor(self) -> dict[str, Any]:
        """Describe the worker HOST mapping, distinct from its NPU alias."""
        return dict(
            epoch=self.epoch,
            session=self.session,
            hostname=self.store.get_hostname(),
            slab_name=self.mapping.shm_name,
            slab_size=self.mapping.size,
            slab_generation=self.mapping.generation,
            host_base=self.mapping.ptr,
        )

    def handle(self, request: Any) -> StorageResult:
        return self.registry.handle(request, self._execute)

    def handle_validated(self, request: Any, proof: Any) -> StorageResult:
        return self.registry.handle_validated(request, self._execute, proof)

    def set_transport_notifier(self, notify: Any) -> bool:
        return self.registry.set_transport_notifier(notify)

    def drain_terminal_notifications(self) -> list:
        return self.registry.drain_terminal_notifications()

    def is_healthy(self) -> bool:
        return not self._closed and self.registry.is_healthy()

    def close(self) -> None:
        """Keep the registration alive if any native operation is uncertain."""
        if self._closed:
            return
        state = self.registry.snapshot()
        if state["running"] or state["fatal"] or state["quarantined"]:
            raise RuntimeError("Worker transport still owns native work")
        self.close_native()
        self._closed = True

    def _execute(self, request: Any) -> Any:
        try:
            p = request.payload
            keys = p["keys"]
            if (
                not isinstance(keys, (list, tuple))
                or not 0 < len(keys) <= MAX_KEYS
                or any(not isinstance(k, str) or not k for k in keys)
                or sum(len(k.encode()) for k in keys) > MAX_CONTROL_BYTES
                or p["generation"] != self.mapping.generation
                or self._closed
            ):
                raise ValueError("Invalid native transport identity or keys")
            if request.method == "native_exists":
                self._prepare_thread()
                return self.store.batch_is_exist(keys)
            api = p["api"]
            allowed = (
                _READS
                if request.method == "native_get"
                else _WRITES
                if request.method == "native_put"
                else ()
            )
            if api not in allowed:
                raise ValueError("Unknown native transport operation")
            counts = p["counts"]
            if len(counts) != len(keys) or any(
                type(n) is not int or n < 1 for n in counts
            ):
                raise ValueError("Invalid per-key range counts")
            count = sum(counts)
            if count > MAX_RANGES:
                raise ValueError("Too many native ranges")
            offsets = unpack_uints(p["offsets"], count)
            sizes = unpack_uints(p["sizes"], count)
            if sum(sizes) > self.max_bytes:
                raise ValueError("Native job exceeds byte budget")
            ptrs, lengths, begin = [], [], 0
            for n in counts:
                row, span = [], []
                for offset, size in zip(
                    offsets[begin : begin + n], sizes[begin : begin + n], strict=True
                ):
                    if (
                        not size
                        or offset > self.mapping.size
                        or size > self.mapping.size - offset
                    ):
                        raise ValueError("Native range exceeds registered host slab")
                    row.append(checked_uint(self.mapping.ptr + offset))
                    span.append(size)
                ptrs.append(row)
                lengths.append(span)
                begin += n
            header = p.get("header", b"")
            if not isinstance(header, bytes) or len(header) > MAX_CONTROL_BYTES:
                raise ValueError("Invalid metadata header")
            segment = p.get("segment")
            if segment is not None and not isinstance(segment, str):
                raise ValueError("Invalid placement segment")
            if api in ("put_from", "put_parts") and (len(keys) != 1 or counts != [1]):
                raise ValueError("Single-key operation requires one range")
            if api in ("batch_get_into", "batch_put_from") and any(
                n != 1 for n in counts
            ):
                raise ValueError("Flat operation requires one range per key")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise StorageFailedSafe(str(error)) from error
        # No native call occurred before validation. Exceptions after this point
        # are UNKNOWN: registry retains the result and refuses normal shutdown.
        with serving_perf_phase("native_thread_setup_ms"):
            self._prepare_thread()
        if api in _READS:
            with serving_perf_phase("native_call_ms"):
                if api == "batch_get_into":
                    return self.store.batch_get_into(
                        keys, [r[0] for r in ptrs], [r[0] for r in lengths]
                    )
                return self.store.batch_get_into_multi_buffers(keys, ptrs, lengths)
        replica = self.replica_factory()
        replica.replica_num = 1
        if segment:
            replica.preferred_segment = segment
        if api == "put_parts":
            view = memoryview(
                (ctypes.c_ubyte * lengths[0][0]).from_address(ptrs[0][0])
            ).cast("B")
            return self.store.put_parts(keys[0], header, view)
        if api == "put_from":
            return self.store.put_from(keys[0], ptrs[0][0], lengths[0][0], replica)
        if api == "batch_put_from":
            return self.store.batch_put_from(
                keys, [r[0] for r in ptrs], [r[0] for r in lengths], replica
            )
        return self.store.batch_put_from_multi_buffers(keys, ptrs, lengths, replica)

    def _prepare_thread(self) -> None:
        if not getattr(self._thread, "ready", False):
            self._initialize_thread()
            self._thread.ready = True


class RemoteMooncakeStore:
    """M-side native-call facade. Allocations and their quarantine remain in M."""

    replica_factory = SimpleNamespace

    def __init__(
        self,
        transport: dict[str, Any],
        *,
        instance: str,
        epoch: str,
        base: int,
        size: int,
        generation: int,
        timeout: float,
        max_bytes: int,
        max_operations: int = 2,
        client: Any = None,
    ) -> None:
        self.epoch, self.base, self.size, self.generation = (
            epoch,
            base,
            size,
            generation,
        )
        self.timeout, self.max_bytes = timeout, max_bytes
        self.registration: dict[str, Any] | None = None
        self._fatal: str | None = None
        self._quarantine: list[Any] = []
        self._notify: Callable[[], None] | None = None
        self._lock = Lock()
        self._slots = BoundedSemaphore(max_operations)
        owner = ref(self)

        def failed(reason: str) -> None:
            target = owner()
            if target is not None:
                target._fail(reason)

        self.client = client or StorageClient(
            transport["work"],
            transport["lifecycle"],
            cache_instance_id=instance,
            service_epoch=epoch,
            client_session="storage-owner",
            session_nonce=bytes.fromhex(transport["nonce"]),
            max_pending=32,
            failure_callback=failed,
        )

    def bind(self, registration: dict[str, Any], slab_name: str) -> None:
        """Bind only a matching, already registered worker host mapping."""
        if (
            registration.get("epoch") != self.epoch
            or registration.get("slab_name") != slab_name
            or registration.get("slab_size") != self.size
            or registration.get("slab_generation") != self.generation
            or type(registration.get("host_base")) is not int
            or registration["host_base"] <= 0
            or not registration.get("session")
            or not registration.get("hostname")
        ):
            raise ValueError("Worker native registration does not match M's slab")
        if self.registration is not None and self.registration != registration:
            raise ValueError("Worker transport registration changed within an epoch")
        self.registration = dict(registration)

    def export_address(self, ptr: int, size: int) -> int:
        """Translate M's host pointer for a signed RemoteFill grant."""
        offset = self._offset(ptr, size)
        if self.registration is None:
            raise RuntimeError("Worker native transport is not registered")
        return checked_uint(self.registration["host_base"] + offset)

    def get_hostname(self) -> str:
        return self.registration["hostname"] if self.registration else ""

    def get_session(self) -> str:
        return self.registration["session"] if self.registration else ""

    def is_healthy(self) -> bool:
        return self._fatal is None and self.client.is_healthy()

    def set_transport_notifier(self, notify: Callable[[], None] | None) -> None:
        self._notify = notify

    def close(self) -> None:
        """A connector close never destroys the shared transport owner."""
        if not self.is_healthy() or self._quarantine:
            raise RuntimeError("Native transport completion is unknown")

    def shutdown(self) -> None:
        self.close()
        if not self.client.close():
            raise RuntimeError("Native transport client has not drained")

    def invoke(self, api: str, args: tuple, owners: Any = ()) -> Any:
        """Keep independent physical references until known native completion."""
        held = tuple({id(o): o for o in owners if o is not None}.values())
        refs = tuple(o for o in held if callable(getattr(o, "ref_count_up", None)))
        for obj in refs:
            obj.ref_count_up()
        unknown = False
        try:
            if api not in _READS or not serving_perf_enabled():
                return self._invoke(api, args, held)
            started = time.perf_counter()
            status = "error"
            # One summary across every split batch; never log per-page transfers.
            with (
                serving_perf_measure(rollup_prefix="remote_get_") as phases,
                serving_perf_scope(transfer_id=f"{id(self):x}:{time.monotonic_ns()}"),
            ):
                try:
                    result = self._invoke(api, args, held)
                    status = "ok"
                    return result
                finally:
                    if status != "ok" or time.perf_counter() - started >= 0.1:
                        serving_perf_log(
                            logger,
                            "storage_native_transfer",
                            started=started,
                            api=api,
                            status=status,
                            **phases,
                        )
        except NativeExternalPageTransferUnknownError:
            unknown = True
            with self._lock:
                self._quarantine.append(held)
            raise
        finally:
            if not unknown:
                for obj in refs:
                    obj.ref_count_down()

    def batch_is_exist(self, keys: Any) -> Any:
        return self.invoke("batch_is_exist", (keys,))

    def is_exist(self, key: str) -> Any:
        return self.batch_is_exist([key])[0]

    def batch_get_into_multi_buffers(self, *args: Any) -> Any:
        return self.invoke("batch_get_into_multi_buffers", args)

    def batch_get_into(self, *args: Any) -> Any:
        return self.invoke("batch_get_into", args)

    def batch_put_from_multi_buffers(self, *args: Any) -> Any:
        return self.invoke("batch_put_from_multi_buffers", args)

    def batch_put_from(self, *args: Any) -> Any:
        return self.invoke("batch_put_from", args)

    def put_from(self, *args: Any) -> Any:
        return self.invoke("put_from", args)

    def put_parts(self, *args: Any) -> Any:
        return self.invoke("put_parts", args)

    def _offset(self, ptr: int, size: int) -> int:
        offset = checked_uint(ptr - self.base)
        if not checked_uint(size) or offset > self.size or size > self.size - offset:
            raise ValueError("Transport operand is outside M's shared slab")
        return offset

    def _invoke(self, api: str, args: tuple, owners: tuple) -> Any:
        if not self.is_healthy() or self.registration is None:
            raise RuntimeError("Worker native transport is unavailable")
        perf = api in _READS and serving_perf_enabled()
        planning_started = time.perf_counter() if perf else None
        header, segment = b"", None
        scalar = api in ("put_from", "put_parts")
        exists = api == "batch_is_exist"
        keys = [args[0]] if scalar else list(args[0])
        rows = []
        if not exists:
            if api not in _READS | _WRITES:
                raise ValueError("Unsupported native transport API")
            if api == "put_parts":
                if len(owners) != 1:
                    raise ValueError("Metadata PUT requires one retained shared object")
                header = bytes(args[1])
                ptrs, sizes = [[owners[0].data_ptr]], [[len(args[2])]]
                replica = None
            else:
                ptrs, sizes = ([args[1]], [args[2]]) if scalar else (args[1], args[2])
                if api in ("put_from", "batch_put_from", "batch_get_into"):
                    ptrs, sizes = [[p] for p in ptrs], [[s] for s in sizes]
                replica = args[3] if api in _WRITES else None
            segment = getattr(replica, "preferred_segment", None)
            for pointers, lengths in zip(ptrs, sizes, strict=True):
                offsets, spans = [], []
                for ptr, size in zip(pointers, lengths, strict=True):
                    offset = self._offset(ptr, size)
                    if offsets and offsets[-1] + spans[-1] == offset:
                        spans[-1] += size
                    else:
                        offsets.append(offset)
                        spans.append(size)
                if (
                    not offsets
                    or len(offsets) > MAX_RANGES
                    or sum(spans) > self.max_bytes
                ):
                    raise ValueError("One native object exceeds transport bounds")
                rows.append((offsets, spans))
            if len(rows) != len(keys):
                raise ValueError("Native key/range cardinality mismatch")
        if perf:
            serving_perf_record(
                "operand_plan_ms", (time.perf_counter() - planning_started) * 1000
            )
        results = []
        begin = 0
        while begin < len(keys):
            batch_started = time.perf_counter() if perf else None
            end, count, size, key_bytes = begin, 0, 0, len(header)
            while end < len(keys) and end - begin < MAX_KEYS:
                n, b = (0, 0) if exists else (len(rows[end][0]), sum(rows[end][1]))
                k = len(keys[end].encode())
                if (
                    count + n > MAX_RANGES
                    or size + b > self.max_bytes
                    or key_bytes + k > MAX_CONTROL_BYTES
                ):
                    break
                count, size, key_bytes = count + n, size + b, key_bytes + k
                end += 1
            if end == begin:
                raise ValueError("One native key exceeds control bounds")
            payload = dict(
                keys=keys[begin:end],
                generation=self.generation,
            )
            if not exists:
                batch_rows = rows[begin:end]
                payload.update(
                    api=api,
                    counts=[len(r[0]) for r in batch_rows],
                    offsets=pack_uints(o for r in batch_rows for o in r[0]),
                    sizes=pack_uints(s for r in batch_rows for s in r[1]),
                    segment=segment,
                    header=header,
                )
            method = (
                "native_exists"
                if exists
                else "native_get"
                if api in _READS
                else "native_put"
            )
            if perf:
                correlation = serving_perf_context()
                payload["perf_context"] = {
                    k: correlation[k]
                    for k in ("source_req_id", "prepare_operation_id", "transfer_id")
                    if k in correlation
                }
                serving_perf_record(
                    "batch_plan_ms", (time.perf_counter() - batch_started) * 1000
                )
                serving_perf_record("batches", 1)
                serving_perf_record("keys", end - begin)
                serving_perf_record("bytes", size)
            # Only M background jobs wait here. The worker model thread never
            # services this queue. Separate lookup credit remains available.
            if method == "native_exists":
                values = self._request(method, payload)
            else:
                slot_started = time.perf_counter() if perf else None
                with self._slots:
                    if perf:
                        serving_perf_record(
                            "credit_wait_ms",
                            (time.perf_counter() - slot_started) * 1000,
                        )
                    values = self._request(method, payload)
            if scalar:
                results.append(values)
            else:
                if values is None and api in _WRITES:
                    values = [0] * (end - begin)
                # An omitted status must not shift the next batch onto another key.
                if not isinstance(values, (list, tuple)) or len(values) != end - begin:
                    raise ValueError("Native batch returned an invalid result count")
                results.extend(values)
            begin = end
        return results[0] if scalar else results

    def _request(self, method: str, payload: dict) -> Any:
        # Recheck after waiting for bulk credit and between split batches.
        if not self.is_healthy():
            raise RuntimeError("Worker native transport is unavailable")
        started = (
            time.perf_counter()
            if method == "native_get" and serving_perf_enabled()
            else None
        )
        submitted = replied = ack_started = ack_finished = None
        future = self.client.request(
            method,
            payload,
            request_id="native-io",
            request_attempt=self.epoch,
            load_generation=self.generation,
            purpose=method,
        )
        terminal = False
        if started is not None:
            submitted = time.perf_counter()
        try:
            result = future.result(timeout=self.timeout)
            if started is not None:
                replied = time.perf_counter()
            if result.status in ("BUSY", "REJECTED"):
                cancelled = self.client.cancel(future.identity).result(
                    timeout=self.timeout
                )
                if cancelled.status != "CANCELLED_NOT_STARTED":
                    raise RuntimeError("Refused native job has ambiguous state")
                terminal = True
                digest = cancelled.value["terminal_result_digest"]
            elif result.status in ("COMPLETED", "FAILED_SAFE", "CANCELLED_NOT_STARTED"):
                terminal = True
                digest = content_digest((result.status, result.value, result.error))
            else:
                raise RuntimeError(f"Native transport status {result.status}")
            if started is not None:
                ack_started = time.perf_counter()
            ack = self.client.acknowledge_terminal(
                future.identity, {"terminal_result_digest": digest}
            ).result(timeout=self.timeout)
            if started is not None:
                ack_finished = time.perf_counter()
            if ack.status != "ACKNOWLEDGED":
                raise RuntimeError("Native completion acknowledgement failed")
            if result.status != "COMPLETED":
                raise ValueError(result.error or result.status)
            return result.value
        except Exception as error:
            if terminal and isinstance(error, ValueError):
                raise
            self._fail(str(error))
            if terminal:
                raise RuntimeError(
                    "Native work completed but its control channel failed"
                ) from error
            raise NativeExternalPageTransferUnknownError(method, future) from error
        finally:
            if started is not None:
                serving_perf_record("submit_ms", (submitted - started) * 1000)
                serving_perf_record(
                    "reply_wait_ms",
                    ((replied or time.perf_counter()) - submitted) * 1000,
                )
                if ack_started is not None:
                    serving_perf_record(
                        "ack_ms",
                        ((ack_finished or time.perf_counter()) - ack_started) * 1000,
                    )
                if replied is not None and ack_started is not None:
                    serving_perf_record(
                        "terminal_processing_ms", (ack_started - replied) * 1000
                    )
                serving_perf_record(
                    "ack_completed_batches", int(ack_finished is not None)
                )

    def _fail(self, reason: str) -> None:
        self._fatal = reason
        if self._notify is not None:
            self._notify()
