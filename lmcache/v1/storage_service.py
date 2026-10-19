# SPDX-License-Identifier: Apache-2.0
"""Fresh-process CPU storage service with independent bounded lifecycle traffic."""

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol
import argparse
import hmac
import importlib
import json
import os
import sys
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.cpu_cache import StorageBusy
from lmcache.v1.rpc.storage_client import LIFECYCLE_METHODS
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
    decode_message,
    encode_message,
)


class StorageRuntime(Protocol):
    """The transport never constructs an engine, allocator or device context."""

    def handle(self, request: StorageRequest) -> StorageResult: ...
    def close(self) -> None: ...
    def is_healthy(self) -> bool: ...


class StorageService:
    """Serve authenticated value-only work and short lifecycle transitions.

    Runtime lifecycle handlers must be nonblocking registry transitions. Bulk and
    metadata work run in independent bounded lanes. Completed results retain their
    admission slots while a client is backpressured, instead of being dropped.
    """

    def __init__(
        self,
        runtime: StorageRuntime,
        work_endpoint: str,
        lifecycle_endpoint: str,
        *,
        cache_instance_id: str,
        service_epoch: str,
        sessions: dict[str, bytes],
        max_work: int = 16,
        socket_hwm: int = 64,
    ) -> None:
        if (
            max_work <= 0
            or socket_hwm <= 0
            or not sessions
            or not all(sessions.values())
        ):
            raise ValueError(
                "Storage service requires positive bounds and authenticated sessions"
            )
        self.runtime = runtime
        self.cache_instance_id = cache_instance_id
        self.service_epoch = service_epoch
        self.sessions = dict(sessions)
        self.max_work = max_work
        self.socket_hwm = socket_hwm
        self._endpoints = (work_endpoint, lifecycle_endpoint)
        self._quiescing = threading.Event()
        self._stop_requested = threading.Event()
        self._stop_operation: tuple | None = None
        self.ready = threading.Event()
        self.failure: BaseException | None = None
        self.drain_error: BaseException | None = None
        self.send_backpressure = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the sole socket owner and wait for endpoint binding or failure."""
        if self._thread is not None:
            raise RuntimeError("Storage service has already started")
        self._thread = threading.Thread(
            target=self.serve_forever, name="storage-service-io", daemon=True
        )
        self._thread.start()
        if not self.ready.wait(10):
            raise TimeoutError("Storage service endpoint startup timed out")
        if self.failure is not None:
            raise RuntimeError(
                "Storage service endpoint startup failed"
            ) from self.failure

    def close(self, timeout: float = 5.0) -> None:
        """Quiesce admission and retain runtime ownership until admitted jobs drain."""
        self._quiescing.set()
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(
                    "Storage service still owns unresolved work or replies"
                )
        if self.failure is not None:
            raise RuntimeError("Storage service failed") from self.failure

    def quiesce(self) -> None:
        """Reject new work on parent loss while retaining lifecycle/lease ownership."""
        self._quiescing.set()
        quiesce = getattr(self.runtime, "quiesce", None)
        if callable(quiesce):
            quiesce()

    def join(self) -> None:
        """Wait for the running service and propagate its failure."""
        if self._thread is None:
            raise RuntimeError("Storage service has not started")
        self._thread.join()
        if self.failure is not None:
            raise RuntimeError("Storage service failed") from self.failure

    def serve_forever(self) -> None:
        """Run nonblocking sockets; no Future or executor queue waits in this loop."""
        context = zmq.Context()
        sockets = []
        executors = {
            "bulk": ThreadPoolExecutor(1, thread_name_prefix="storage-prepare"),
            "lookup": ThreadPoolExecutor(1, thread_name_prefix="storage-lookup"),
            "persist": ThreadPoolExecutor(1, thread_name_prefix="storage-persist"),
        }
        jobs: dict[tuple[bytes, tuple, int], tuple[Future, StorageRequest, str]] = {}
        # Per-channel outboxes are bounded by admitted work plus each session's
        # lifecycle credits. A blocked route does not block another route's sends.
        outboxes: tuple[OrderedDict, OrderedDict] = (OrderedDict(), OrderedDict())
        routes: dict[tuple[str, int], bytes] = {}
        fatal_notified: set[str] = set()
        terminal_notices: dict[tuple, StorageResult] = {}
        try:
            poller = zmq.Poller()
            for endpoint in self._endpoints:
                socket = context.socket(zmq.ROUTER)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                socket.setsockopt(zmq.SNDHWM, self.socket_hwm)
                socket.setsockopt(zmq.RCVHWM, self.socket_hwm)
                socket.bind(endpoint)
                sockets.append(socket)
                poller.register(socket, zmq.POLLIN)
            self.ready.set()
            while True:
                events = dict(poller.poll(5))
                drain_notices = getattr(
                    self.runtime, "drain_terminal_notifications", None
                )
                if drain_notices is not None:
                    for notice in drain_notices():
                        terminal_notices[notice.identity.key] = notice
                for key, notice in tuple(terminal_notices.items()):
                    route = routes.get((notice.identity.client_session, 1))
                    if route is not None:
                        outboxes[1][(route, key, "terminal")] = encode_message(notice)
                        del terminal_notices[key]
                for channel in (1, 0):
                    socket = sockets[channel]
                    # Reserve response space before consuming another request.
                    limit = self.max_work * (8 * len(self.sessions) if channel else 2)
                    if (
                        events.get(socket, 0) & zmq.POLLIN
                        and len(outboxes[channel]) < limit
                    ):
                        for _ in range(min(16, limit - len(outboxes[channel]))):
                            try:
                                route, *frames = socket.recv_multipart(zmq.NOBLOCK)
                            except zmq.Again:
                                break
                            try:
                                request = decode_message(frames, StorageRequest)
                            except (ValueError, msgspec.DecodeError):
                                continue  # No trustworthy identity to acknowledge.
                            key = (route, request.identity.key, request.correlation_id)
                            try:
                                self._validate(request, bool(channel))
                                session = (
                                    request.participant_session
                                    or request.identity.client_session
                                )
                                routes[(session, channel)] = route
                                if key in jobs or key in outboxes[channel]:
                                    continue  # Existing reply/registry owns this retry.
                                if channel:
                                    result = self._handle(request)
                                    outboxes[channel][key] = encode_message(result)
                                    if request.method.lower() == "stop" and (
                                        result.status == "STOPPED"
                                        or (
                                            result.status == "COMPLETED"
                                            and isinstance(result.value, dict)
                                            and result.value.get("status") == "STOPPED"
                                        )
                                    ):
                                        self._quiescing.set()
                                        self._stop_operation = request.identity.key
                                    if (
                                        request.method == "acknowledge_terminal"
                                        and request.identity.key == self._stop_operation
                                        and result.status == "ACKNOWLEDGED"
                                    ):
                                        self._stop_requested.set()
                                else:
                                    method = request.method.lower()
                                    lane = (
                                        "lookup"
                                        if method == "lookup"
                                        else (
                                            "persist"
                                            if method
                                            in (
                                                "commit_write",
                                                "commit_checkpoint",
                                                "persist",
                                            )
                                            else "bulk"
                                        )
                                    )
                                    if (
                                        self._quiescing.is_set()
                                        or sum(job[2] == lane for job in jobs.values())
                                        >= self.max_work
                                    ):
                                        outboxes[channel][key] = encode_message(
                                            self._reply(request, "BUSY")
                                        )
                                    else:
                                        jobs[key] = (
                                            executors[lane].submit(
                                                self._handle, request
                                            ),
                                            request,
                                            lane,
                                        )
                            except (ValueError, PermissionError) as error:
                                outboxes[channel][key] = encode_message(
                                    self._reply(request, "REJECTED", error=str(error))
                                )
                    if not channel:
                        for key, (future, request, _) in tuple(jobs.items()):
                            if future.done():
                                # _handle converts runtime failures into an unknown
                                # completion result, never an allocator-free promise.
                                outboxes[0][key] = encode_message(future.result())
                                del jobs[key]
                    for key in tuple(outboxes[channel])[:16]:
                        route = key[0]
                        try:
                            socket.send_multipart(
                                [route, *outboxes[channel][key]],
                                zmq.NOBLOCK,
                                copy=False,
                            )
                        except (zmq.Again, zmq.ZMQError):
                            self.send_backpressure += 1
                            outboxes[channel].move_to_end(key)
                            continue
                        del outboxes[channel][key]
                if not self.runtime.is_healthy():
                    self._quiescing.set()
                    for session, nonce in self.sessions.items():
                        channel = 1 if (session, 1) in routes else 0
                        route = routes.get((session, channel))
                        if route is not None and session not in fatal_notified:
                            identity = OperationIdentity(
                                self.cache_instance_id,
                                self.service_epoch,
                                session,
                                0,
                                "session-health",
                                session,
                                0,
                                "fatal",
                            )
                            result = StorageResult(
                                identity,
                                "FATAL",
                                error=(
                                    "Storage runtime is unhealthy; "
                                    "coordinated recovery required"
                                ),
                                session_nonce=nonce,
                            )
                            outboxes[channel][(route, identity.key, 0)] = (
                                encode_message(result)
                            )
                            fatal_notified.add(session)
                if self._stop_requested.is_set() and not jobs and not any(outboxes):
                    # Runtime.close owns its lease/quarantine drain contract.
                    try:
                        self.runtime.close()
                    except Exception as error:
                        # A refused drain cannot become process exit: M must keep
                        # its mappings alive while native access is uncertain.
                        self.drain_error = error
                        self._stop_requested.clear()
                    else:
                        break
        except BaseException as error:
            self.failure = error
            self.ready.set()
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=False)
            for socket in sockets:
                socket.close(linger=1000 if self._stop_requested.is_set() else 0)
            context.term()

    def _validate(self, request: StorageRequest, lifecycle: bool) -> None:
        identity = request.identity
        if (
            identity.cache_instance_id != self.cache_instance_id
            or identity.service_epoch != self.service_epoch
        ):
            raise PermissionError("Stale storage instance or epoch")
        session = request.participant_session or identity.client_session
        nonce = self.sessions.get(session)
        if nonce is None or not hmac.compare_digest(nonce, request.session_nonce):
            raise PermissionError("Unknown storage session or invalid nonce")
        if session != identity.client_session and request.method != "retire":
            raise PermissionError("Only the original session can control its operation")
        if lifecycle != (request.method.lower() in LIFECYCLE_METHODS):
            raise ValueError("Storage method sent on the wrong channel")
        if not lifecycle and request.payload_digest != content_digest(
            (request.method, request.payload)
        ):
            raise ValueError("Storage work payload digest mismatch")

    def _handle(self, request: StorageRequest) -> StorageResult:
        try:
            result = self.runtime.handle(request)
            if (
                not isinstance(result, StorageResult)
                or result.identity != request.identity
            ):
                raise ValueError("Runtime returned a different operation identity")
            return msgspec.structs.replace(
                result,
                payload_digest=request.payload_digest,
                correlation_id=request.correlation_id,
            )
        except StorageBusy as error:
            return self._reply(request, "BUSY", error=str(error))
        except BaseException as error:
            return self._reply(request, "FAILED_UNKNOWN", error=str(error))

    @staticmethod
    def _reply(
        request: StorageRequest, status: str, *, error: str | None = None
    ) -> StorageResult:
        return StorageResult(
            request.identity,
            status,
            error=error,
            payload_digest=request.payload_digest,
            correlation_id=request.correlation_id,
        )


def main() -> None:
    """Start a runtime from an explicit trusted JSON bootstrap, without pickle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap", required=True, help="JSON path, or '-' for stdin"
    )
    parser.add_argument("--watch-parent-stdin", action="store_true")
    args = parser.parse_args()
    if args.bootstrap == "-":
        bootstrap = json.load(sys.stdin)
    else:
        with open(args.bootstrap, encoding="utf-8") as source:
            bootstrap = json.load(source)
    module_name, symbol = bootstrap.pop("runtime_factory").split(":", 1)
    if not module_name or not symbol.isidentifier():
        raise ValueError("Runtime factory must be an explicit module:symbol")
    runtime_kwargs = bootstrap.pop("runtime_kwargs", {})
    if "runtime_config" in bootstrap:
        if runtime_kwargs:
            raise ValueError("Specify runtime_config or runtime_kwargs, not both")
        runtime_kwargs = {"runtime_config": bootstrap.pop("runtime_config")}
    runtime = getattr(importlib.import_module(module_name), symbol)(**runtime_kwargs)
    bootstrap["sessions"] = {
        name: bytes.fromhex(nonce) for name, nonce in bootstrap["sessions"].items()
    }
    service = StorageService(runtime, **bootstrap)
    service.start()
    descriptor = getattr(runtime, "bootstrap_descriptor", None)
    print(
        json.dumps(
            {
                "status": "READY",
                "service_epoch": service.service_epoch,
                "bootstrap_descriptor": descriptor() if callable(descriptor) else {},
            }
        ),
        flush=True,
    )
    if args.watch_parent_stdin:
        if args.bootstrap == "-":
            raise ValueError("Parent liveness requires a separate bootstrap file")

        def watch_parent() -> None:
            while os.read(sys.stdin.fileno(), 1):
                pass
            service.quiesce()

        threading.Thread(
            target=watch_parent, name="storage-parent-watch", daemon=True
        ).start()
    service.join()


if __name__ == "__main__":
    main()
