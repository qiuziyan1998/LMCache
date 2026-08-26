# SPDX-License-Identifier: Apache-2.0
"""Byte-oriented service facade for the remote-fill state core."""

# Local
from .codec import (
    ProtocolValidationError,
    decode_request,
    encode_response,
    response_for_invalid_message,
)
from .protocol import ProtocolLimits, RemoteFillResponse, ResultCode, operation_kind
from .state import RemoteFillMetricsSnapshot, RemoteFillStateCore


class RemoteFillService:
    """Decode bounded messages and dispatch them to ``RemoteFillStateCore``."""

    def __init__(
        self,
        state: RemoteFillStateCore,
        limits: ProtocolLimits | None = None,
    ) -> None:
        """Create a service facade.

        Args:
            state: Thread-safe transaction state core.
            limits: Codec bounds. Defaults to the state's bounds.
        """

        self._state = state
        self.limits = limits or state.limits

    def handle_bytes(self, encoded: bytes) -> bytes:
        """Handle one encoded request and return one encoded response.

        Args:
            encoded: Untrusted request bytes. The size is checked before decode.

        Returns:
            A bounded encoded response. Validation failures never echo payload
            content or destination addresses.
        """

        try:
            request = decode_request(encoded, self.limits)
        except ProtocolValidationError as exc:
            self._state.record_protocol_error()
            return encode_response(
                response_for_invalid_message(str(exc)),
                self.limits,
            )
        try:
            self._state.validate_response_capacity(request, self.limits)
        except ProtocolValidationError:
            self._state.record_capacity_rejection(request)
            bounded_error = RemoteFillResponse(
                operation=operation_kind(request),
                operation_id=request.common.operation_id,
                code=ResultCode.RESOURCE_EXHAUSTED,
                message="remote-fill response exceeds byte limit",
                destination_engine_epoch=self._state.destination_engine_epoch,
                shared_cache_generation=self._state.shared_cache_generation,
            )
            return encode_response(bounded_error, self.limits)
        # decode_request() is the single untrusted-wire validation boundary.
        # Direct in-process users may still call state.handle(), which retains
        # its own validation contract.
        response = self._state.handle_validated(request)
        try:
            return encode_response(response, self.limits)
        except ProtocolValidationError:
            bounded_error = RemoteFillResponse(
                operation=operation_kind(request),
                operation_id=request.common.operation_id,
                code=ResultCode.RESOURCE_EXHAUSTED,
                message="remote-fill response exceeds byte limit",
                destination_engine_epoch=self._state.destination_engine_epoch,
                shared_cache_generation=self._state.shared_cache_generation,
            )
            return encode_response(bounded_error, self.limits)

    def run_maintenance(self) -> tuple[str, ...]:
        """Run TTL and hard-timeout checks.

        Returns:
            Transfer IDs requiring paired restart.
        """

        return self._state.run_maintenance()

    def metrics_snapshot(self) -> RemoteFillMetricsSnapshot:
        """Return fixed-cardinality service metrics without enabling export."""

        return self._state.metrics_snapshot()

    def shutdown(self) -> tuple[str, ...]:
        """Stop admission and release only destinations safe to reuse.

        Returns:
            Transfer IDs whose armed destinations require paired restart.
        """

        return self._state.shutdown()
