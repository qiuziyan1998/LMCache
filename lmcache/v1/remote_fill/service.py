# SPDX-License-Identifier: Apache-2.0
"""Byte-oriented service facade for the remote-fill state core."""

# Standard
import time

# Local
from lmcache.logging import init_logger
from lmcache.v1.cold_start_perf import cold_start_perf_enabled, cold_start_perf_log

from .codec import (
    ProtocolValidationError,
    decode_request,
    encode_response,
    response_for_invalid_message,
)
from .protocol import ProtocolLimits, RemoteFillResponse, ResultCode, operation_kind
from .state import RemoteFillMetricsSnapshot, RemoteFillStateCore


logger = init_logger(__name__)


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
        self._perf_transfers: dict[str, dict[str, object]] = {}

    def handle_bytes(self, encoded: bytes) -> bytes:
        """Handle one encoded request and return one encoded response.

        Args:
            encoded: Untrusted request bytes. The size is checked before decode.

        Returns:
            A bounded encoded response. Validation failures never echo payload
            content or destination addresses.
        """

        diagnose = cold_start_perf_enabled()
        started = time.perf_counter() if diagnose else 0.0
        thread_started = time.thread_time_ns() if diagnose else 0
        phase_started = started
        operation = "INVALID"
        transfer_id = ""
        decode_ms = capacity_ms = dispatch_ms = 0.0

        def complete(response: bytes) -> bytes:
            if not diagnose:
                return response
            completed = time.perf_counter()
            elapsed_ms = (completed - started) * 1000
            encode_ms = (completed - phase_started) * 1000
            thread_cpu_ms = (time.thread_time_ns() - thread_started) / 1_000_000
            if transfer_id:
                if (
                    transfer_id not in self._perf_transfers
                    and len(self._perf_transfers)
                    >= self.limits.max_active_transactions * 2
                ):
                    self._perf_transfers.pop(next(iter(self._perf_transfers)))
                stats = self._perf_transfers.setdefault(
                    transfer_id,
                    {"rpcs": 0, "wall_ms": 0.0, "cpu_ms": 0.0, "ops": {}},
                )
                stats["rpcs"] = int(stats["rpcs"]) + 1
                stats["wall_ms"] = float(stats["wall_ms"]) + elapsed_ms
                stats["cpu_ms"] = float(stats["cpu_ms"]) + thread_cpu_ms
                ops = stats["ops"]
                assert isinstance(ops, dict)
                ops[operation] = int(ops.get(operation, 0)) + 1
                if operation in {"FINISH", "ABORT"}:
                    if float(stats["wall_ms"]) >= 100.0:
                        cold_start_perf_log(
                            logger,
                            "remote_fill_decoder_control_summary",
                            transfer_id=transfer_id,
                            rpc_count=stats["rpcs"],
                            wall_ms=round(float(stats["wall_ms"]), 3),
                            thread_cpu_ms=round(float(stats["cpu_ms"]), 3),
                            operation_counts=ops,
                        )
                    self._perf_transfers.pop(transfer_id, None)
            if elapsed_ms >= 100.0:
                cold_start_perf_log(
                    logger,
                    "remote_fill_decoder_control_slow",
                    operation=operation,
                    transfer_id=transfer_id,
                    request_bytes=len(encoded),
                    response_bytes=len(response),
                    elapsed_ms=round(elapsed_ms, 3),
                    thread_cpu_ms=round(thread_cpu_ms, 3),
                    decode_ms=round(decode_ms, 3),
                    capacity_ms=round(capacity_ms, 3),
                    dispatch_ms=round(dispatch_ms, 3),
                    encode_ms=round(encode_ms, 3),
                )
            return response

        try:
            request = decode_request(encoded, self.limits)
        except ProtocolValidationError as exc:
            if diagnose:
                decode_ms = (time.perf_counter() - phase_started) * 1000
                phase_started = time.perf_counter()
            self._state.record_protocol_error()
            return complete(encode_response(
                response_for_invalid_message(str(exc)),
                self.limits,
            ))
        operation = operation_kind(request).value
        transfer_id = request.common.transfer_id
        if diagnose:
            decode_ms = (time.perf_counter() - phase_started) * 1000
            phase_started = time.perf_counter()
        try:
            self._state.validate_response_capacity(request, self.limits)
        except ProtocolValidationError:
            if diagnose:
                capacity_ms = (time.perf_counter() - phase_started) * 1000
                phase_started = time.perf_counter()
            self._state.record_capacity_rejection(request)
            bounded_error = RemoteFillResponse(
                operation=operation_kind(request),
                operation_id=request.common.operation_id,
                code=ResultCode.RESOURCE_EXHAUSTED,
                message="remote-fill response exceeds byte limit",
                destination_engine_epoch=self._state.destination_engine_epoch,
                shared_cache_generation=self._state.shared_cache_generation,
            )
            return complete(encode_response(bounded_error, self.limits))
        if diagnose:
            capacity_ms = (time.perf_counter() - phase_started) * 1000
            phase_started = time.perf_counter()
        # decode_request() is the single untrusted-wire validation boundary.
        # Direct in-process users may still call state.handle(), which retains
        # its own validation contract.
        response = self._state.handle_validated(request)
        if diagnose:
            dispatch_ms = (time.perf_counter() - phase_started) * 1000
            phase_started = time.perf_counter()
        try:
            return complete(encode_response(response, self.limits))
        except ProtocolValidationError:
            bounded_error = RemoteFillResponse(
                operation=operation_kind(request),
                operation_id=request.common.operation_id,
                code=ResultCode.RESOURCE_EXHAUSTED,
                message="remote-fill response exceeds byte limit",
                destination_engine_epoch=self._state.destination_engine_epoch,
                shared_cache_generation=self._state.shared_cache_generation,
            )
            return complete(encode_response(bounded_error, self.limits))

    def run_maintenance(self) -> tuple[str, ...]:
        """Run TTL and hard-timeout checks.

        Returns:
            Transfer IDs requiring paired restart.
        """

        if not cold_start_perf_enabled():
            return self._state.run_maintenance()
        started = time.perf_counter()
        thread_started = time.thread_time_ns()
        result = self._state.run_maintenance()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms >= 100.0:
            cold_start_perf_log(
                logger,
                "remote_fill_decoder_maintenance_slow",
                elapsed_ms=round(elapsed_ms, 3),
                thread_cpu_ms=round(
                    (time.thread_time_ns() - thread_started) / 1_000_000, 3
                ),
                fatal_transfers=len(result),
            )
        return result

    def metrics_snapshot(self) -> RemoteFillMetricsSnapshot:
        """Return fixed-cardinality service metrics without enabling export."""

        return self._state.metrics_snapshot()

    def shutdown(self) -> tuple[str, ...]:
        """Stop admission and release only destinations safe to reuse.

        Returns:
            Transfer IDs whose armed destinations require paired restart.
        """

        return self._state.shutdown()
