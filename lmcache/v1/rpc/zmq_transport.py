# SPDX-License-Identifier: Apache-2.0
"""ZMQ-based transport implementations for RPC communication.

TODO: Implement ZmqPushTransport and ZmqPullTransport for
LMCacheAsyncLookupClient/Server.
"""

# Standard
from collections import namedtuple
from datetime import datetime, timezone
from typing import Any
import os
import threading
import time
import traceback

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc.transport import (
    RpcClientTransport,
    RpcServerTransport,
)
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_socket,
)

logger = init_logger(__name__)

SocketParams = namedtuple(
    "SocketParams", ["socket_path", "rank", "protocol"], defaults=["ipc"]
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _caller_location() -> str:
    caller = traceback.extract_stack(limit=3)[0]
    return f"{caller.filename}:{caller.lineno}:{caller.name}"


class ZmqReqRepClientTransport(RpcClientTransport):
    """ZMQ REQ socket transport for synchronous RPC clients.

    Manages multiple REQ sockets (one per server rank) and
    provides send/recv with timeout + automatic socket
    recreation on failure.
    """

    def __init__(
        self,
        socket_params: list[SocketParams],
        timeout_ms: int,
    ):
        self.ctx = get_zmq_context(use_asyncio=False)
        self.socket_params = socket_params
        self.timeout_ms = timeout_ms
        self._request_lock = threading.Lock()
        self._world_size = len(socket_params)
        self.encoder = msgspec.msgpack.Encoder()

        self.sockets: list[zmq.Socket] = []
        for params in self.socket_params:
            logger.info(
                "Transport connecting to rank %s with socket path %s",
                params.rank,
                params.socket_path,
            )
            socket = self._create_socket(params)
            self.sockets.append(socket)

    def _create_socket(self, params: SocketParams) -> zmq.Socket:
        """Create and configure a REQ socket."""
        socket = get_zmq_socket(
            self.ctx,
            params.socket_path,
            params.protocol,
            zmq.REQ,
            "connect",
        )
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        return socket

    def _recreate_all_sockets(self) -> None:
        """Recreate all sockets after a failure."""
        for rank_idx in range(self._world_size):
            params = self.socket_params[rank_idx]
            old_socket = self.sockets[rank_idx]
            if old_socket is not None:
                try:
                    old_socket.close(linger=0)
                except zmq.ZMQError as e:
                    logger.warning(
                        "ZMQ error closing old socket: timestamp=%s "
                        "socket_index=%s rank=%s endpoint=%s error=%s",
                        _utc_timestamp(),
                        rank_idx,
                        params.rank,
                        params.socket_path,
                        e,
                    )
                except AttributeError:
                    pass

            logger.info(
                "Recreating socket: timestamp=%s socket_index=%s "
                "rank=%s endpoint=%s",
                _utc_timestamp(),
                rank_idx,
                params.rank,
                params.socket_path,
            )
            self.sockets[rank_idx] = self._create_socket(params)

    def send_and_recv_all(
        self,
        msg: list[Any],
        timeout_ms: int | None = None,
    ) -> list[bytes]:
        """Send msg to all ranks and collect responses.

        Each element of msg is serialized via msgpack before
        sending. On timeout or ZMQ error, recreates all
        sockets and returns an empty list.
        """
        effective_timeout = self.timeout_ms if timeout_ms is None else timeout_ms
        if effective_timeout <= 0:
            raise ValueError("RPC operation timeout must be positive")
        request_lock = getattr(self, "_request_lock", None)
        if request_lock is None:
            request_lock = self._request_lock = threading.Lock()
        with request_lock:
            for socket in self.sockets:
                set_option = getattr(socket, "setsockopt", None)
                if callable(set_option):
                    set_option(zmq.RCVTIMEO, effective_timeout)
                    set_option(zmq.SNDTIMEO, effective_timeout)
            started_at = _utc_timestamp()
            started = time.perf_counter()
            encoded = [self.encoder.encode(m) for m in msg]
            results: list[bytes] = []
            failed_socket_idx = -1
            sent_count = 0
            phase = "send"
            try:
                for i in range(self._world_size):
                    failed_socket_idx = i
                    self.sockets[i].send_multipart(encoded, copy=False)
                    sent_count += 1

                phase = "recv"
                for i in range(self._world_size):
                    failed_socket_idx = i
                    resp = self.sockets[i].recv()
                    results.append(resp)
            except zmq.ZMQError as e:
                params = (
                    self.socket_params[failed_socket_idx]
                    if 0 <= failed_socket_idx < self._world_size
                    else None
                )
                failure = (
                    "Timeout occurred" if isinstance(e, zmq.Again) else "ZMQ error"
                )
                logger.exception(
                    "%s for rank %s; recreating all sockets: failed_at=%s "
                    "started_at=%s elapsed_ms=%.3f phase=%s socket_index=%s "
                    "endpoint=%s timeout_ms=%s world_size=%s sent=%s received=%s "
                    "pid=%s thread=%s caller=%s error=%s",
                    failure,
                    params.rank if params is not None else "unknown",
                    _utc_timestamp(),
                    started_at,
                    (time.perf_counter() - started) * 1000,
                    phase,
                    failed_socket_idx,
                    params.socket_path if params is not None else "unknown",
                    effective_timeout,
                    self._world_size,
                    sent_count,
                    len(results),
                    os.getpid(),
                    threading.current_thread().name,
                    _caller_location(),
                    e,
                    stack_info=True,
                )
                self._recreate_all_sockets()
                return []
            finally:
                for socket in self.sockets:
                    set_option = getattr(socket, "setsockopt", None)
                    if callable(set_option):
                        set_option(zmq.RCVTIMEO, self.timeout_ms)
                        set_option(zmq.SNDTIMEO, self.timeout_ms)

            return results

    @property
    def world_size(self) -> int:
        return self._world_size

    def close(self) -> None:
        for socket in self.sockets:
            try:
                socket.close(linger=0)
            except Exception as e:
                logger.warning("Error closing socket: %s", e)
        self.sockets.clear()
        # get_zmq_context() returns the process-wide singleton.  This transport
        # owns its sockets, but not that shared context; terminating it here
        # would invalidate unrelated and subsequently-created RPC clients.


class ZmqRouterServerTransport(RpcServerTransport):
    """ZMQ ROUTER socket transport for synchronous RPC servers.

    Listens for incoming requests and sends responses back
    using ROUTER socket identity-based routing.
    """

    def __init__(
        self,
        socket_path: str,
        recv_timeout_ms: int = 1000,
        protocol: str = "ipc",
        max_frame_bytes: int | None = None,
        max_frame_count: int | None = None,
        send_timeout_ms: int | None = None,
        high_water_mark: int = 1024,
    ):
        if max_frame_bytes is not None and max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive when configured")
        if max_frame_count is not None and max_frame_count <= 0:
            raise ValueError("max_frame_count must be positive when configured")
        if send_timeout_ms is not None and send_timeout_ms <= 0:
            raise ValueError("send_timeout_ms must be positive when configured")
        if high_water_mark <= 0:
            raise ValueError("high_water_mark must be positive")
        self.decoder = msgspec.msgpack.Decoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket = get_zmq_socket(
            self.ctx,
            socket_path,
            protocol,
            zmq.ROUTER,  # type: ignore[attr-defined]
            "bind",
        )
        self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.socket.setsockopt(
            zmq.SNDTIMEO,
            recv_timeout_ms if send_timeout_ms is None else send_timeout_ms,
        )
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, high_water_mark)
        self.socket.setsockopt(zmq.RCVHWM, high_water_mark)
        self.socket_path = socket_path
        self.max_frame_bytes = max_frame_bytes
        self.max_frame_count = max_frame_count

    def recv_request(
        self,
    ) -> tuple[bytes, list[Any]] | None:
        """Receive a request.

        Returns (identity, data_frames) or None on timeout.
        Each data frame is deserialized via msgpack.
        ROUTER socket frames:
          [0] = identity, [1] = empty delimiter, [2:] = data
        """
        try:
            frames = self.socket.recv_multipart(copy=False)
        except zmq.Again:
            return None

        if not frames:
            logger.warning("Malformed request received: missing routing identity.")
            return None
        identity = frames[0].bytes
        if len(frames) < 2 or frames[1].bytes != b"":
            logger.warning("Malformed request received: invalid REQ delimiter.")
            return (identity, [])
        # frames[1] is the empty delimiter from REQ socket
        raw_frames = frames[2:]
        if len(raw_frames) < 3:
            logger.warning("Malformed request received: not enough frames.")
            return (identity, [])
        if (
            self.max_frame_count is not None
            and len(raw_frames) > self.max_frame_count
        ):
            logger.warning("Malformed request received: too many frames.")
            return (identity, [])
        if self.max_frame_bytes is not None and any(
            len(frame) > self.max_frame_bytes for frame in raw_frames
        ):
            logger.warning("Malformed request received: frame exceeds byte limit.")
            return (identity, [])
        try:
            data_frames = [self.decoder.decode(f) for f in raw_frames]
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning("Malformed request received: invalid msgpack frame.")
            return (identity, [])
        return (identity, data_frames)

    def send_response(
        self,
        identity: bytes,
        response: bytes,
    ) -> None:
        """Send response back via ROUTER socket."""
        try:
            self.socket.send_multipart([identity, b"", response])
        except zmq.Again:
            # The state transition and its idempotent result are already
            # recorded.  Treat this as a lost reply so the client can retry or
            # resolve the operation through STATUS; never stall maintenance.
            logger.warning("Dropping an RPC response after the bounded send timeout")

    def close(self) -> None:
        self.socket.close(linger=0)
