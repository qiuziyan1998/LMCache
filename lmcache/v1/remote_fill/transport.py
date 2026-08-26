# SPDX-License-Identifier: Apache-2.0
"""In-process client and fault-injectable transport for remote-fill tests."""

# Standard
from collections import Counter
from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol

# Local
from .codec import (
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    response_for_invalid_message,
)
from .protocol import (
    OperationKind,
    ProtocolLimits,
    RemoteFillRequest,
    RemoteFillResponse,
    operation_kind,
)
from .security import seal_request
from .service import RemoteFillService


REMOTE_FILL_SERVICE_HEADER = "lmcache.remote_fill"
REMOTE_FILL_ENVELOPE_VERSION = 1


class ReplyLostError(ConnectionError):
    """Raised after the service commits an operation but its reply is dropped."""


class RemoteFillTransportError(ConnectionError):
    """Raised when an existing RPC transport violates the fixed envelope."""


class ByteRoundTripTransport(Protocol):
    """Minimal byte transport adaptable to in-process or existing ZMQ RPC."""

    limits: ProtocolLimits

    def round_trip(
        self, encoded: bytes, *, timeout_ms: int | None = None
    ) -> bytes:
        """Send one request and return one response."""


class ExistingRpcClientTransport(Protocol):
    """Structural subset implemented by ``lmcache.v1.rpc.RpcClientTransport``."""

    @property
    def world_size(self) -> int:
        """Return the number of destination ranks."""

    def send_and_recv_all(
        self, msg: list[Any], timeout_ms: int | None = None
    ) -> list[bytes]:
        """Send a multi-frame message and collect raw responses."""

    def close(self) -> None:
        """Close the transport."""


class ExistingRpcServerTransport(Protocol):
    """Structural subset implemented by ``lmcache.v1.rpc.RpcServerTransport``."""

    def recv_request(self) -> tuple[bytes, list[Any]] | None:
        """Receive one decoded multi-frame request."""

    def send_response(self, identity: bytes, response: bytes) -> None:
        """Send one raw response."""

    def close(self) -> None:
        """Close the transport."""


class InProcessRemoteFillTransport:
    """Synchronous byte transport with deterministic lost-reply injection."""

    def __init__(
        self,
        service: RemoteFillService,
        limits: ProtocolLimits | None = None,
    ) -> None:
        """Create an in-process transport.

        Args:
            service: Destination service to invoke.
            limits: Codec bounds. Defaults to the service's bounds.
        """

        self._service = service
        self.limits = limits or service.limits
        self._drop_replies: Counter[OperationKind] = Counter()
        self._lock = Lock()

    def drop_next_reply(self, operation: OperationKind) -> None:
        """Drop one reply after the selected operation reaches the service.

        Args:
            operation: Operation whose next reply should be lost.
        """

        with self._lock:
            self._drop_replies[operation] += 1

    def round_trip(
        self, encoded: bytes, *, timeout_ms: int | None = None
    ) -> bytes:
        """Send encoded bytes to the service and return encoded response bytes.

        Args:
            encoded: Bounded request bytes.

        Returns:
            Bounded response bytes.

        Raises:
            ReplyLostError: If fault injection drops the committed reply.
        """

        request = decode_request(encoded, self.limits)
        response = self._service.handle_bytes(encoded)
        kind = operation_kind(request)
        with self._lock:
            if self._drop_replies[kind] > 0:
                self._drop_replies[kind] -= 1
                raise ReplyLostError(f"{kind.value} reply was lost")
        return response

    def close(self) -> None:
        """Close the in-process transport (a no-op)."""


class RpcRemoteFillByteTransport:
    """One-rank adapter over the existing LMCache RPC client transport."""

    def __init__(
        self,
        transport: ExistingRpcClientTransport,
        limits: ProtocolLimits | None = None,
    ) -> None:
        """Create a byte round-trip adapter.

        Args:
            transport: Existing msgspec/ZMQ-capable LMCache RPC transport.
            limits: Protocol bounds applied by ``RemoteFillClient``.

        Raises:
            ValueError: If the transport targets anything other than TP0 only.
        """

        if transport.world_size != 1:
            raise ValueError("remote fill requires a one-rank TP0 RPC transport")
        self._transport = transport
        self.limits = limits or ProtocolLimits()

    def round_trip(
        self, encoded: bytes, *, timeout_ms: int | None = None
    ) -> bytes:
        """Send the fixed three-frame envelope through existing RPC.

        Args:
            encoded: Bounded remote-fill request bytes.

        Returns:
            The single raw response frame.

        Raises:
            RemoteFillTransportError: If RPC does not return exactly one byte
                response.
        """

        responses = self._transport.send_and_recv_all(
            [
                REMOTE_FILL_SERVICE_HEADER,
                REMOTE_FILL_ENVELOPE_VERSION,
                encoded,
            ],
            timeout_ms=timeout_ms,
        )
        if len(responses) != 1 or not isinstance(responses[0], bytes):
            raise RemoteFillTransportError(
                "remote-fill RPC requires exactly one byte response"
            )
        return responses[0]

    def close(self) -> None:
        """Close the underlying existing RPC transport."""

        self._transport.close()


class RemoteFillRpcServer:
    """Serve the fixed remote-fill envelope on an existing RPC server."""

    def __init__(
        self,
        transport: ExistingRpcServerTransport,
        service: RemoteFillService,
    ) -> None:
        """Create a server-loop adapter.

        Args:
            transport: Existing LMCache ROUTER/server transport.
            service: Bounded remote-fill byte service.
        """

        self._transport = transport
        self._service = service

    def serve_once(self) -> bool:
        """Receive and handle at most one request.

        Returns:
            ``True`` when a request was received, including malformed
            envelopes, or ``False`` when the transport timed out.
        """

        received = self._transport.recv_request()
        if received is None:
            return False
        identity, frames = received
        if not self._valid_envelope(frames):
            response = encode_response(
                response_for_invalid_message("invalid remote-fill RPC envelope"),
                self._service.limits,
            )
        else:
            response = self._service.handle_bytes(frames[2])
        self._transport.send_response(identity, response)
        return True

    def serve_until(self, should_stop: Callable[[], bool]) -> None:
        """Serve until ``should_stop`` returns true.

        Args:
            should_stop: Callback evaluated between receive attempts.
        """

        while not should_stop():
            self.serve_once()

    def close(self) -> None:
        """Close the underlying existing RPC server transport."""

        self._transport.close()

    @staticmethod
    def _valid_envelope(frames: list[object]) -> bool:
        return (
            len(frames) == 3
            and frames[0] == REMOTE_FILL_SERVICE_HEADER
            and frames[1] == REMOTE_FILL_ENVELOPE_VERSION
            and isinstance(frames[2], bytes)
        )


class RemoteFillClient:
    """Small client that seals and bounds every in-process protocol request."""

    def __init__(
        self,
        transport: ByteRoundTripTransport,
        limits: ProtocolLimits | None = None,
        operation_timeouts_ms: dict[OperationKind, int] | None = None,
    ) -> None:
        """Create a mock-capable protocol client.

        Args:
            transport: Any bounded byte round-trip adapter, including an
                adapter over the existing ``lmcache.v1.rpc`` transport.
            limits: Codec bounds. Defaults to the transport's bounds.
        """

        self._transport = transport
        self.limits = limits or transport.limits
        self._operation_timeouts_ms = dict(operation_timeouts_ms or {})
        if any(timeout <= 0 for timeout in self._operation_timeouts_ms.values()):
            raise ValueError("remote-fill operation timeouts must be positive")

    def execute(self, request: RemoteFillRequest) -> RemoteFillResponse:
        """Seal, encode, execute, and decode one operation.

        Args:
            request: Request whose payload digest may be blank or stale.

        Returns:
            The decoded service response.

        Raises:
            ReplyLostError: If the transport loses the committed reply.
            ProtocolValidationError: If local bounds or validation fail.
        """

        sealed = seal_request(request)
        encoded = encode_request(sealed, self.limits)
        timeout_ms = self._operation_timeouts_ms.get(operation_kind(sealed))
        if timeout_ms is None:
            response = self._transport.round_trip(encoded)
        else:
            response = self._transport.round_trip(
                encoded, timeout_ms=timeout_ms
            )
        return decode_response(response, self.limits)

    def close(self) -> None:
        """Close the underlying transport when it exposes a close method."""

        close = getattr(self._transport, "close", None)
        if close is not None:
            close()
