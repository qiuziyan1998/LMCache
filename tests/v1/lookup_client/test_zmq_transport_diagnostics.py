# SPDX-License-Identifier: Apache-2.0
"""Diagnostics emitted when synchronous lookup RPCs fail or run slowly."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, call
import queue
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.v1.lookup_client import lmcache_lookup_client
from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
from lmcache.v1.rpc import zmq_transport
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
    ZmqRouterServerTransport,
)


class _RecvTimeoutSocket:
    def send_multipart(self, frames, copy=False) -> None:
        pass

    def recv(self) -> bytes:
        raise zmq.Again()


class _RouterSocket:
    def __init__(self, frames) -> None:
        self.frames = frames

    def recv_multipart(self, copy=False):
        return self.frames


def _call_transport(transport: ZmqReqRepClientTransport) -> list[bytes]:
    return transport.send_and_recv_all([[1, 2], "request-id", ""])


def test_zmq_timeout_log_contains_transport_and_caller_context(monkeypatch) -> None:
    transport = object.__new__(ZmqReqRepClientTransport)
    transport.socket_params = [SocketParams("/tmp/lookup-rank-7", 7)]
    transport.timeout_ms = 3000
    transport._world_size = 1
    transport.encoder = msgspec.msgpack.Encoder()
    transport.sockets = [_RecvTimeoutSocket()]
    transport._recreate_all_sockets = MagicMock()

    logger = MagicMock()
    monkeypatch.setattr(zmq_transport, "logger", logger)

    assert _call_transport(transport) == []

    args, kwargs = logger.exception.call_args
    message = args[0] % args[1:]
    assert "Timeout occurred for rank 7" in message
    assert "phase=recv" in message
    assert "socket_index=0" in message
    assert "endpoint=/tmp/lookup-rank-7" in message
    assert "timeout_ms=3000" in message
    assert "sent=1 received=0" in message
    assert "started_at=" in message and "failed_at=" in message
    assert "caller=" in message and "_call_transport" in message
    assert kwargs["stack_info"] is True
    transport._recreate_all_sockets.assert_called_once_with()


def test_zmq_client_transport_accepts_explicit_tcp_endpoint(monkeypatch) -> None:
    socket = MagicMock()
    create_socket = MagicMock(return_value=socket)
    monkeypatch.setattr(zmq_transport, "get_zmq_socket", create_socket)
    transport = object.__new__(ZmqReqRepClientTransport)
    transport.ctx = object()
    transport.timeout_ms = 1000

    result = transport._create_socket(
        SocketParams("10.0.0.8:19001", 0, "tcp")
    )

    assert result is socket
    create_socket.assert_called_once_with(
        transport.ctx,
        "10.0.0.8:19001",
        "tcp",
        zmq.REQ,
        "connect",
    )
    assert socket.setsockopt.call_args_list == [
        call(zmq.RCVTIMEO, 1000),
        call(zmq.SNDTIMEO, 1000),
    ]


def test_zmq_server_returns_identity_for_malformed_request() -> None:
    transport = object.__new__(ZmqRouterServerTransport)
    transport.decoder = msgspec.msgpack.Decoder()
    transport.max_frame_bytes = 1024
    transport.max_frame_count = 3
    transport.socket = _RouterSocket([zmq.Frame(b"client"), zmq.Frame(b"")])

    assert transport.recv_request() == (b"client", [])


def test_zmq_server_drops_timed_out_response_without_stopping() -> None:
    transport = object.__new__(ZmqRouterServerTransport)
    transport.socket = MagicMock()
    transport.socket.send_multipart.side_effect = zmq.Again()

    transport.send_response(b"client", b"response")

    transport.socket.send_multipart.assert_called_once_with(
        [b"client", b"", b"response"]
    )


def test_zmq_server_rejects_oversized_frame_before_decode() -> None:
    transport = object.__new__(ZmqRouterServerTransport)
    transport.decoder = MagicMock()
    transport.max_frame_bytes = 8
    transport.max_frame_count = 3
    transport.socket = _RouterSocket(
        [
            zmq.Frame(b"client"),
            zmq.Frame(b""),
            zmq.Frame(b"x" * 9),
            zmq.Frame(b"1"),
            zmq.Frame(b"2"),
        ]
    )

    assert transport.recv_request() == (b"client", [])
    transport.decoder.decode.assert_not_called()


def test_zmq_server_contains_invalid_outer_msgpack() -> None:
    transport = object.__new__(ZmqRouterServerTransport)
    transport.decoder = msgspec.msgpack.Decoder()
    transport.max_frame_bytes = 1024
    transport.max_frame_count = 3
    transport.socket = _RouterSocket(
        [
            zmq.Frame(b"client"),
            zmq.Frame(b""),
            zmq.Frame(b"\xc1"),
            zmq.Frame(msgspec.msgpack.encode(1)),
            zmq.Frame(msgspec.msgpack.encode(b"request")),
        ]
    )

    assert transport.recv_request() == (b"client", [])


def test_zmq_server_rejects_excess_frames_before_decode() -> None:
    transport = object.__new__(ZmqRouterServerTransport)
    transport.decoder = MagicMock()
    transport.max_frame_bytes = 1024
    transport.max_frame_count = 3
    transport.socket = _RouterSocket(
        [
            zmq.Frame(b"client"),
            zmq.Frame(b""),
            zmq.Frame(b"1"),
            zmq.Frame(b"2"),
            zmq.Frame(b"3"),
            zmq.Frame(b"4"),
        ]
    )

    assert transport.recv_request() == (b"client", [])
    transport.decoder.decode.assert_not_called()


def test_closing_client_does_not_terminate_shared_context(monkeypatch) -> None:
    context = MagicMock()
    first_socket = MagicMock()
    second_socket = MagicMock()
    second_socket.recv.return_value = b"response"
    sockets = iter((first_socket, second_socket))

    monkeypatch.setattr(
        zmq_transport,
        "get_zmq_context",
        MagicMock(return_value=context),
    )
    monkeypatch.setattr(
        zmq_transport,
        "get_zmq_socket",
        MagicMock(side_effect=lambda *_args: next(sockets)),
    )

    params = [SocketParams("127.0.0.1:19001", 0, "tcp")]
    first = ZmqReqRepClientTransport(params, timeout_ms=1000)
    second = ZmqReqRepClientTransport(params, timeout_ms=1000)

    first.close()

    first_socket.close.assert_called_once_with(linger=0)
    context.term.assert_not_called()
    second_socket.close.assert_not_called()
    assert second.send_and_recv_all(["service", "version", b"request"]) == [
        b"response"
    ]

    second.close()
    context.term.assert_not_called()


class _TokenDatabase:
    @staticmethod
    def process_tokens(token_ids, make_key=False):
        yield 0, len(token_ids), 123


class _FailedTransport:
    @staticmethod
    def send_and_recv_all(msg) -> list[bytes]:
        return []


def test_lookup_failure_log_contains_request_context(monkeypatch) -> None:
    client = object.__new__(LMCacheLookupClient)
    client.config = SimpleNamespace(lookup_timeout_ms=3000)
    client.transport = _FailedTransport()
    client.reqs_status = {}
    client.enable_blending = False
    client.token_database = _TokenDatabase()

    logger = MagicMock()
    monkeypatch.setattr(lmcache_lookup_client, "logger", logger)

    assert client.lookup(list(range(512)), "request-42") == 0

    args = logger.error.call_args.args
    message = args[0] % args[1:]
    assert "caller=LMCacheLookupClient.lookup" in message
    assert "lookup_id=request-42" in message
    assert "mode=hashes input_count=1 token_count=512" in message
    assert "transport=_FailedTransport timeout_ms=3000" in message
    assert "started_at=" in message and "failed_at=" in message


class _ServerTransport:
    def __init__(self) -> None:
        self.requests: queue.Queue = queue.Queue()
        self.requests.put((b"client", [[123], [256], "request-7", ""]))
        self.response_sent = threading.Event()

    def recv_request(self):
        try:
            return self.requests.get(timeout=0.01)
        except queue.Empty:
            return None

    def send_response(self, identity: bytes, response: bytes) -> None:
        self.response_sent.set()

    def close(self) -> None:
        pass


class _LookupEngine:
    config = SimpleNamespace(enable_blending=False, lookup_timeout_ms=0)

    @staticmethod
    def lookup(**kwargs) -> int:
        return 256


def test_lookup_server_logs_start_and_slow_completion(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(lmcache_lookup_client, "logger", logger)
    transport = _ServerTransport()
    server = lmcache_lookup_client.LMCacheLookupServer(
        _LookupEngine(), SimpleNamespace(), transport
    )
    try:
        assert transport.response_sent.wait(timeout=1)
    finally:
        server.close()

    info_messages = [
        call.args[0] % call.args[1:] for call in logger.info.call_args_list
    ]
    warning_messages = [
        call.args[0] % call.args[1:] for call in logger.warning.call_args_list
    ]
    started = next(msg for msg in info_messages if "processing started" in msg)
    completed = next(msg for msg in warning_messages if "processing completed" in msg)

    assert "lookup_id=request-7 mode=hashes input_count=1" in started
    assert "client_timeout_ms=0" in started
    assert "lookup_id=request-7 mode=hashes input_count=1" in completed
    assert "result_tokens=256 client_timeout_ms=0" in completed
