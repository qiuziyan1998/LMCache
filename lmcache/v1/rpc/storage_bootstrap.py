# SPDX-License-Identifier: Apache-2.0
"""Early, noncollective storage-process bootstrap and worker supervision."""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable
from uuid import uuid4
from weakref import ref
import json
import os
import signal
import subprocess
import sys
import threading
import time

# Third Party
import psutil

# First Party
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import content_digest


_BOOTSTRAPS: dict[str, "WorkerStorageBootstrap"] = {}


def storage_discovery_path(
    engine_id: str, rpc_port: int = 0, base_url: str | None = None
) -> Path:
    """Use the existing instance namespace, including Ascend's short-ID wrapper."""
    from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

    return Path(
        get_zmq_rpc_path_lmcache(
            engine_id, "storage_bootstrap", rpc_port, 0, base_url=base_url
        )
        + ".json"
    )


def _pid_alive(pid: int) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    return psutil.pid_exists(pid)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def discover_storage_service(
    engine_id: str,
    rpc_port: int = 0,
    *,
    timeout_s: float = 600,
    base_url: str | None = None,
    launcher_pid: int | None = None,
) -> dict[str, Any]:
    """Wait for a live MEMORY_READY instance; never adopt a stale service epoch."""
    path = storage_discovery_path(engine_id, rpc_port, base_url)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            time.sleep(0.02)
            continue
        if value.get("cache_instance_id") != engine_id:
            raise RuntimeError("Storage discovery belongs to another instance")
        if launcher_pid is not None and value.get("launcher_pid") != launcher_pid:
            raise RuntimeError("Storage discovery belongs to another executor launch")
        if value.get("state") == "FAILED":
            raise RuntimeError(value.get("error", "Storage bootstrap failed"))
        if not _pid_alive(value.get("owner_pid")):
            raise RuntimeError(
                "Storage discovery names a dead owner; recovery required"
            )
        return value
    raise TimeoutError("Storage MEMORY_READY discovery deadline expired")


def get_worker_storage_bootstrap(engine_id: str) -> "WorkerStorageBootstrap":
    """Return resources registered before profiling; never initialize them late."""
    try:
        return _BOOTSTRAPS[engine_id]
    except KeyError as error:
        raise RuntimeError(
            "Decoder storage was not bootstrapped before profiling"
        ) from error


def _terminate_worker(reason: str) -> None:
    # Executor supervision notices worker death even when the model is idle.
    os.kill(os.getpid(), signal.SIGTERM)


@dataclass
class WorkerStorageBootstrap:
    """Already registered native worker resources, reused by final layout binding."""

    client: Any
    descriptor: dict[str, Any]
    session: str
    timeout_s: float
    mapping: Any = None
    reader: Any = None
    device_base: int = 0
    process: Any = None
    discovery_path: Path | None = None
    failure_callback: Callable[[str], None] = _terminate_worker
    closing: threading.Event = field(default_factory=threading.Event)
    drain_guards: list[Callable[[], bool]] = field(default_factory=list)
    failure_lock: Any = field(default_factory=threading.Lock)
    failed: bool = False

    def call(self, method: str, payload: Any, timeout_s: float | None = None) -> Any:
        """Complete one startup control and acknowledge its resource-free result."""
        future = self.client.request(
            method,
            payload,
            request_id="bootstrap",
            request_attempt=self.session,
            load_generation=0,
            purpose="bootstrap",
        )
        result = future.result(
            timeout=self.timeout_s if timeout_s is None else timeout_s
        )
        if result.error:
            raise RuntimeError(result.error)
        self.client.acknowledge_terminal(
            future.identity,
            {
                "terminal_result_digest": content_digest(
                    (result.status, result.value, result.error)
                )
            },
        ).result(timeout=self.timeout_s if timeout_s is None else timeout_s)
        return result.value if result.value is not None else {"status": result.status}

    def bind_layout(self, layout: Any) -> Any:
        """Bind immutable layout without registering native resources again."""
        import msgspec

        self.call(
            "bind_layout",
            {
                "session": self.session,
                "layout": msgspec.to_builtins(layout, builtin_types=(bytes,)),
            },
        )
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Storage all-worker layout deadline expired")
            status = self.call("registration_status", {}, remaining)
            if (
                status.get("status") == "FAILED"
                or status.get("layout_status") == "FAILED"
            ):
                raise RuntimeError(status.get("error", "Storage layout binding failed"))
            if status.get("layout_status") == "READY":
                return status
            time.sleep(min(0.02, remaining))

    def mark_failed(self, reason: str) -> None:
        """Fail pending work and route service loss through executor supervision."""
        with self.failure_lock:
            if self.failed or self.closing.is_set():
                return
            self.failed = True
        self.client.mark_fatal(reason)
        self.failure_callback(reason)

    def close(self) -> None:
        """Close drained worker resources, retaining readers of unknown completion."""
        if self.closing.is_set():
            return
        if any(not guard() for guard in self.drain_guards):
            raise RuntimeError(
                "Storage source leases have not drained; mapping retained"
            )
        if self.reader is not None:
            self.reader.close()  # A failed native close must keep the mapping attached.
        if self.mapping is not None:
            self.mapping.close()
        self.closing.set()
        try:
            self.call("unregister_participant", {"session": self.session})
            stopped = self.call("stop", {})
            # Only a successful, acknowledged drain permits namespace cleanup.
            if stopped.get("status") == "STOPPED":
                deadline = time.monotonic() + min(self.timeout_s, 5)
                if self.process is not None:
                    if self.process.stdin is not None:
                        self.process.stdin.close()
                    self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
                else:
                    while _pid_alive(self.descriptor["owner_pid"]):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Storage STOP did not exit after drain")
                        time.sleep(0.02)
                self.remove_discovery()
        finally:
            self.client.close()
            if self.process is not None and self.process.stdin is not None:
                # EOF quiesces; it never proves peers drained.
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            if _BOOTSTRAPS.get(self.descriptor["cache_instance_id"]) is self:
                _BOOTSTRAPS.pop(self.descriptor["cache_instance_id"])

    def remove_discovery(self) -> None:
        """Remove only this drained epoch's descriptor, never a slab or new owner."""
        if self.discovery_path is None:
            return
        try:
            current = json.loads(self.discovery_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if current.get("service_epoch") == self.descriptor["service_epoch"]:
            self.discovery_path.unlink()
            self.discovery_path.with_suffix(".owner").unlink(missing_ok=True)


def start_worker_storage(
    *,
    engine_id: str,
    rpc_port: int,
    tp_rank: int,
    tp_size: int,
    runtime_config: dict[str, Any],
    mapping_factory: Callable[..., Any],
    reader_factory: Callable[[], Any] | None,
    timeout_s: float = 600,
    base_url: str | None = None,
    launcher_pid: int | None = None,
    spawn: Callable[..., Any] = subprocess.Popen,
    client_factory: Callable[..., Any] = StorageClient,
    failure_callback: Callable[[str], None] = _terminate_worker,
    endpoint_factory: Callable[[str, str], str] | None = None,
) -> WorkerStorageBootstrap:
    """Launch M once, register every W, then pass a bounded noncollective gate.

    Factories allow CPU lifecycle tests without replacing the production owner
    allocator. The native mapping and external reader are created on TP0 too.
    No inference process group is used or initialized here.
    """
    if not engine_id or not 0 <= tp_rank < tp_size or timeout_s <= 0:
        raise ValueError("Invalid decoder storage bootstrap identity or deadline")
    if engine_id in _BOOTSTRAPS:
        raise RuntimeError("Decoder storage bootstrap already exists for this worker")
    launcher_pid = os.getppid() if launcher_pid is None else launcher_pid
    path = storage_discovery_path(engine_id, rpc_port, base_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    if tp_rank == 0:
        marker = path.with_suffix(".owner")
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise RuntimeError(
                "Storage owner marker exists; explicit epoch recovery required"
            ) from error
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"worker_pid": os.getpid(), "launcher_pid": launcher_pid}, stream)
        epoch = uuid4().hex
        sessions = {f"worker-{rank}": os.urandom(32).hex() for rank in range(tp_size)}
        sessions["scheduler"] = os.urandom(32).hex()
        if endpoint_factory is None:
            from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

            def endpoint_factory(kind: str, generation: str) -> str:
                return "ipc://" + get_zmq_rpc_path_lmcache(
                    engine_id + "-" + generation, kind, rpc_port, 0, base_url=base_url
                )

        endpoints = {
            kind: endpoint_factory(kind, epoch)
            for kind in ("storage_work", "storage_lifecycle")
        }
        runtime_config = dict(
            runtime_config,
            cache_instance_id=engine_id,
            service_epoch=epoch,
            slab_name="/lmcache-storage-" + epoch,
            participants=sessions,
            expected_workers=[f"worker-{rank}" for rank in range(tp_size)],
        )
        bootstrap = dict(
            runtime_factory="lmcache_ascend.v1.cpu_cache:create_storage_runtime",
            runtime_config=runtime_config,
            work_endpoint=endpoints["storage_work"],
            lifecycle_endpoint=endpoints["storage_lifecycle"],
            cache_instance_id=engine_id,
            service_epoch=epoch,
            sessions=sessions,
        )
        payload_path = path.with_name(path.name + ".launch-" + epoch)
        _write_json(payload_path, bootstrap)
        try:
            process = spawn(
                [
                    sys.executable,
                    "-m",
                    "lmcache.v1.storage_service",
                    "--bootstrap",
                    str(payload_path),
                    "--watch-parent-stdin",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                close_fds=True,
            )
            ready: Queue = Queue()

            def read_ready() -> None:
                try:
                    reported = False
                    for line in process.stdout:
                        try:
                            reply = json.loads(line)
                        except ValueError:
                            continue
                        if reply.get("status") == "READY" and not reported:
                            ready.put(reply)
                            reported = True
                    if not reported:
                        ready.put(
                            RuntimeError("Storage service exited before MEMORY_READY")
                        )
                except BaseException as error:
                    ready.put(error)

            threading.Thread(
                target=read_ready, name="storage-bootstrap-ready", daemon=True
            ).start()
            try:
                reply = ready.get(timeout=timeout_s)
            except Empty as error:
                raise TimeoutError(
                    "Storage service MEMORY_READY deadline expired"
                ) from error
            if isinstance(reply, BaseException):
                raise reply
            descriptor = dict(
                reply["bootstrap_descriptor"],
                cache_instance_id=engine_id,
                service_epoch=epoch,
                owner_pid=process.pid,
                launcher_pid=launcher_pid,
                work_endpoint=bootstrap["work_endpoint"],
                lifecycle_endpoint=bootstrap["lifecycle_endpoint"],
                sessions=sessions,
                state="MEMORY_READY",
            )
            _write_json(path, descriptor)
        except BaseException as error:
            _write_json(
                path,
                {
                    "cache_instance_id": engine_id,
                    "launcher_pid": launcher_pid,
                    "state": "FAILED",
                    "error": str(error),
                },
            )
            if process is not None and process.stdin is not None:
                process.stdin.close()
            raise
        finally:
            payload_path.unlink(missing_ok=True)
    descriptor = discover_storage_service(
        engine_id,
        rpc_port,
        timeout_s=timeout_s,
        base_url=base_url,
        launcher_pid=launcher_pid,
    )
    session = f"worker-{tp_rank}"
    client = client_factory(
        descriptor["work_endpoint"],
        descriptor["lifecycle_endpoint"],
        cache_instance_id=engine_id,
        service_epoch=descriptor["service_epoch"],
        client_session=session,
        session_nonce=bytes.fromhex(descriptor["sessions"][session]),
    )
    context = WorkerStorageBootstrap(
        client,
        descriptor,
        session,
        timeout_s,
        process=process,
        discovery_path=path,
        failure_callback=failure_callback,
    )
    context_ref = ref(context)

    def notify_failure(reason: str) -> None:
        target = context_ref()
        if target is not None:
            target.mark_failed(reason)

    client.set_failure_callback(notify_failure)

    def watch_service() -> None:
        if process is not None:
            process.wait()
        else:
            while not context.closing.wait(0.1) and _pid_alive(descriptor["owner_pid"]):
                pass
        if not context.closing.is_set():
            context.mark_failed(
                "Decoder storage process exited; coordinated worker recovery required"
            )

    threading.Thread(
        target=watch_service, name="storage-process-watch", daemon=True
    ).start()
    deadline = time.monotonic() + timeout_s
    try:
        context.mapping = mapping_factory(
            shm_name=descriptor["slab_name"],
            size=descriptor["slab_size"],
            generation=descriptor["slab_generation"],
            writable=True,
        )
        context.device_base = int(context.mapping.preflight_device_ptr())
        if context.device_base <= 0:
            raise ValueError("Storage mapping has no valid local device base")
        context.reader = reader_factory() if reader_factory is not None else None
        resource = dict(
            pid=os.getpid(),
            device_base=context.device_base,
            slab_name=descriptor["slab_name"],
            slab_size=descriptor["slab_size"],
            slab_generation=descriptor["slab_generation"],
            native_registered=True,
            reader_ready=True,
        )
        context.call(
            "register_participant",
            {"session": session, "tp_rank": tp_rank, "resource": resource},
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Storage all-worker registration deadline expired")
            status = context.call("registration_status", {}, remaining)
            if status.get("status") == "READY":
                break
            if status.get("status") == "FAILED":
                raise RuntimeError(
                    status.get("error", "Storage participant registration failed")
                )
            time.sleep(min(0.02, remaining))
    except BaseException as error:
        try:
            context.call(
                "register_participant",
                {"session": session, "tp_rank": tp_rank, "error": str(error)},
            )
        except BaseException:
            pass
        context.mark_failed(str(error))
        raise
    _BOOTSTRAPS[engine_id] = context
    return context


def early_bootstrap_decoder_storage(worker: Any) -> WorkerStorageBootstrap | None:
    """Initialize supported process-mode resources before the worker memory snapshot."""
    transfer = worker.vllm_config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "LMCacheAscendConnectorV1Dynamic":
        return None
    from lmcache.integration.vllm.utils import (
        apply_vllm_config_overrides,
        create_lmcache_metadata,
        lmcache_get_or_create_config,
    )

    config = lmcache_get_or_create_config()
    apply_vllm_config_overrides(config, worker.vllm_config)
    if config.decoder_storage_mode != "process":
        return None
    import lmcache_ascend  # noqa: F401

    parallel = worker.parallel_config
    if (
        config.pd_role != "receiver"
        or not config.enable_shared_cpu_cache
        or parallel.pipeline_parallel_size != 1
        or parallel.tensor_parallel_size != parallel.local_world_size
    ):
        raise ValueError(
            "Decoder process storage requires local TP/shared-CPU receiver topology"
        )
    from lmcache.v1.shared_cpu_cache import SharedSlabMapping
    from lmcache.v1.storage_backend.remote_backend import RemoteExternalPageReader
    import vllm.envs

    metadata, _ = create_lmcache_metadata(worker.vllm_config, role="worker")
    metadata.chunk_size = config.chunk_size
    # Match the native dynamic Ascend connector's routing before it is built.
    extra = dict(transfer.kv_connector_extra_config or {})
    dp_rank = getattr(parallel, "data_parallel_index", None)
    if dp_rank is None:
        dp_rank = getattr(parallel, "data_parallel_rank_local", 0)
    extra["lmcache_remote_fill_destination_dp_rank"] = int(dp_rank or 0)
    extra["lmcache_remote_fill_destination_dp_size"] = int(
        getattr(parallel, "data_parallel_size", 1) or 1
    )
    transfer.kv_connector_extra_config = extra
    metadata.kv_connector_extra_config = extra
    metadata.mla_cache_tp_replicated = (
        metadata.use_mla
        and worker.vllm_config.model_config.get_num_kv_heads(parallel) == 1
        and parallel.world_size == parallel.tensor_parallel_size
        and all(
            getattr(parallel, name, 1) == 1
            for name in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        and not getattr(parallel, "enable_elastic_ep", False)
    )
    values = {
        name: getattr(metadata, name)
        for name in (
            "model_name",
            "world_size",
            "local_world_size",
            "worker_id",
            "local_worker_id",
            "use_mla",
            "role",
            "served_model_name",
            "engine_id",
            "kv_connector_extra_config",
            "mla_cache_tp_replicated",
            "chunk_size",
            "max_model_len",
        )
    }
    values.update(kv_dtype=str(metadata.kv_dtype), kv_shape=list(metadata.kv_shape))
    tp_rank = worker.rank % parallel.tensor_parallel_size
    return start_worker_storage(
        engine_id=transfer.engine_id,
        rpc_port=(transfer.kv_connector_extra_config or {}).get("lmcache_rpc_port", 0),
        tp_rank=tp_rank,
        tp_size=parallel.tensor_parallel_size,
        runtime_config={
            "config": config.to_dict(),
            "metadata": values,
            "owner_device": worker.local_rank - tp_rank,
        },
        mapping_factory=SharedSlabMapping.attach,
        reader_factory=(
            (lambda: RemoteExternalPageReader(config, metadata, defer_layout=True))
            if config.remote_url is not None
            else None
        ),
        timeout_s=vllm.envs.VLLM_ENGINE_READY_TIMEOUT_S,
    )
