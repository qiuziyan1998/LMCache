# SPDX-License-Identifier: Apache-2.0
"""Tests for adapting the existing LMCache RPC transport."""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.remote_fill import (
    REMOTE_FILL_ENVELOPE_VERSION,
    REMOTE_FILL_SERVICE_HEADER,
    RemoteFillClient,
    RemoteFillRpcServer,
    RemoteFillTransportError,
    ResultCode,
    RpcRemoteFillByteTransport,
    OperationKind,
    decode_response,
)


class FakeRpcClientTransport:
    """Existing-client-shaped transport that loops back to a service."""

    def __init__(self, service, world_size: int = 1) -> None:
        self.service = service
        self._world_size = world_size
        self.last_message: list[Any] | None = None
        self.closed = False
        self.override_responses: list[bytes] | None = None
        self.last_timeout_ms: int | None = None

    def send_and_recv_all(
        self, msg: list[Any], timeout_ms: int | None = None
    ) -> list[bytes]:
        """Capture the envelope and loop back its request payload."""

        self.last_message = msg
        self.last_timeout_ms = timeout_ms
        if self.override_responses is not None:
            return self.override_responses
        return [self.service.handle_bytes(msg[2])]

    @property
    def world_size(self) -> int:
        """Return the configured fake world size."""

        return self._world_size

    def close(self) -> None:
        """Record close."""

        self.closed = True


class FakeRpcServerTransport:
    """Existing-server-shaped transport with a one-item receive queue."""

    def __init__(self, frames: list[Any] | None) -> None:
        self.frames = frames
        self.response: bytes | None = None
        self.closed = False

    def recv_request(self) -> tuple[bytes, list[Any]] | None:
        """Return one queued request."""

        if self.frames is None:
            return None
        frames, self.frames = self.frames, None
        return b"identity", frames

    def send_response(self, identity: bytes, response: bytes) -> None:
        """Capture the response for the expected identity."""

        assert identity == b"identity"
        self.response = response

    def close(self) -> None:
        """Record close."""

        self.closed = True


def test_rpc_client_adapter_uses_fixed_three_frame_envelope(harness) -> None:
    """Production adaptation reuses existing RPC without another socket stack."""

    rpc = FakeRpcClientTransport(harness.service)
    client = RemoteFillClient(RpcRemoteFillByteTransport(rpc))

    response = client.execute(harness.requests.open())
    assert response.code is ResultCode.ACCEPTED
    assert rpc.last_message is not None
    assert rpc.last_message[:2] == [
        REMOTE_FILL_SERVICE_HEADER,
        REMOTE_FILL_ENVELOPE_VERSION,
    ]
    assert isinstance(rpc.last_message[2], bytes)


def test_rpc_client_adapter_applies_operation_specific_timeout(harness) -> None:
    rpc = FakeRpcClientTransport(harness.service)
    client = RemoteFillClient(
        RpcRemoteFillByteTransport(rpc),
        operation_timeouts_ms={OperationKind.OPEN: 321},
    )

    response = client.execute(harness.requests.open())

    assert response.code is ResultCode.ACCEPTED
    assert rpc.last_timeout_ms == 321


def test_rpc_client_adapter_requires_one_tp0_response(harness) -> None:
    """Missing or multi-rank replies fail closed."""

    rpc = FakeRpcClientTransport(harness.service)
    rpc.override_responses = []
    client = RemoteFillClient(RpcRemoteFillByteTransport(rpc))

    with pytest.raises(RemoteFillTransportError, match="exactly one"):
        client.execute(harness.requests.open())
    with pytest.raises(ValueError, match="one-rank TP0"):
        RpcRemoteFillByteTransport(
            FakeRpcClientTransport(harness.service, world_size=2)
        )


def test_rpc_server_helper_accepts_only_fixed_envelope(harness) -> None:
    """The server dispatches the exact service/version/payload envelope."""

    in_process = harness.requests.open()
    from lmcache.v1.remote_fill import encode_request, seal_request

    encoded = encode_request(seal_request(in_process), harness.state.limits)
    rpc = FakeRpcServerTransport(
        [REMOTE_FILL_SERVICE_HEADER, REMOTE_FILL_ENVELOPE_VERSION, encoded]
    )
    server = RemoteFillRpcServer(rpc, harness.service)

    assert server.serve_once()
    assert rpc.response is not None
    assert (
        decode_response(rpc.response, harness.state.limits).code is ResultCode.ACCEPTED
    )


def test_rpc_server_helper_rejects_malformed_envelope(harness) -> None:
    """Wrong routing headers never reach transaction state."""

    rpc = FakeRpcServerTransport(["another-service", 1, b"payload"])
    server = RemoteFillRpcServer(rpc, harness.service)

    assert server.serve_once()
    assert rpc.response is not None
    response = decode_response(rpc.response, harness.state.limits)
    assert response.code is ResultCode.INVALID_MESSAGE
