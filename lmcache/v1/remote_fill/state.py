# SPDX-License-Identifier: Apache-2.0
"""Thread-safe, hardware-independent remote-fill transaction core."""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

# Third Party
import msgspec

# Local
from .codec import encode_response, validate_request
from .protocol import (
    AbortRequest,
    ArmWindowRequest,
    ControlPage,
    DestinationNativeState,
    DestinationPageDescriptor,
    FinishRequest,
    NegotiateRequest,
    OpenRequest,
    OperationKind,
    PageDisposition,
    PagePreparationStatus,
    ProtocolLimits,
    RemoteFillRequest,
    RemoteFillResponse,
    ReportTransferCompleteRequest,
    ReserveWindowRequest,
    ResultCode,
    StatusRequest,
    TerminalOutcome,
    TransactionState,
    WindowState,
    WindowStatus,
    operation_kind,
)
from .security import (
    destination_descriptor_digest,
    seal_descriptor,
    transaction_manifest_digest,
)


def _new_token() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class NegotiationSpec:
    """Static decoder properties that a prefiller must match exactly."""

    cache_namespace_tag: str
    layout_tag: str
    model_artifact_id: str
    chunk_size: int
    model_layout: str
    group_dimensions: tuple[int, ...]
    layer_count: int
    save_only_first_rank: bool
    shared_group1: bool
    tp_size: int
    dp_size: int
    global_te_push: bool
    destination_engine_id: str
    destination_dp_rank: int
    destination_remote_session: str
    token_hash_algorithm: str
    python_hash_seed: str = ""


@dataclass(frozen=True)
class PreparedPage:
    """Atomic decoder preparation result for one control page.

    Existing pages are snapshots and therefore carry no handle or address.
    Missing pages also carry no handle. Only ``ALLOCATED`` pages own an opaque
    hidden destination and expose bounds for a private capability.
    """

    handle: Any | None
    destination_ptr: int = 0
    destination_length: int = 0
    reservation_base: int = 0
    reservation_length: int = 0
    disposition: PageDisposition = PageDisposition.ALLOCATED


AllocatedPage = PreparedPage


@dataclass(frozen=True)
class ReservedPageView:
    """Opaque page view passed back to the decoder-owned lifecycle callback."""

    control_page: ControlPage
    prepared: PreparedPage
    reservation_id: str


class PageLifecycle(Protocol):
    """Decoder-owned page allocation, release, and admission boundary."""

    def prepare_pages(
        self,
        transfer_id: str,
        window_id: int,
        pages: tuple[ControlPage, ...],
        reserve_missing: bool,
    ) -> Sequence[PreparedPage]:
        """Snapshot existing pages and optionally allocate absent pages."""

    def release_prepared_pages(
        self,
        pages: tuple[PreparedPage, ...],
        reason: str,
    ) -> None:
        """Release raw hidden allocations rejected before page association."""

    def release_pages(
        self,
        pages: tuple[ReservedPageView, ...],
        reason: str,
    ) -> None:
        """Release hidden pages after native access is known to have stopped."""

    def commit_pages(
        self,
        transfer_id: str,
        required_pages: tuple[ControlPage, ...],
        pages: tuple[ReservedPageView, ...],
        finish: FinishRequest,
    ) -> bool:
        """Recheck exact keys, atomically admit, and report complete coverage."""


class UnsafePageLifecycleError(RuntimeError):
    """A lifecycle mutation failed without safely restoring page ownership."""


@dataclass(frozen=True)
class _ReservationKey:
    transfer_id: str
    window_id: int
    kv_group: int
    chunk_index: int


@dataclass
class _Reservation:
    key: _ReservationKey
    page: ControlPage
    prepared: PreparedPage
    reservation_id: str
    state: WindowState
    created_at: float
    expires_at: float
    armed_at: float | None = None
    released: bool = False

    def view(self) -> ReservedPageView:
        return ReservedPageView(
            control_page=self.page,
            prepared=self.prepared,
            reservation_id=self.reservation_id,
        )


@dataclass
class _Window:
    window_id: int
    source_generation: int
    manifest_digest: str
    control_pages: tuple[ControlPage, ...]
    native_transfer_attempt_id: str
    descriptor_digest: str
    descriptors: tuple[DestinationPageDescriptor, ...]
    reservations: tuple[_Reservation, ...]
    page_results: tuple[PagePreparationStatus, ...]
    descriptor_expires_at: float = 0.0
    state: WindowState = WindowState.RESERVED
    native_state: DestinationNativeState = DestinationNativeState.NOT_ARMED
    armed_at: float | None = None
    expired_unarmed: bool = False
    coverage_missing: bool = False
    native_return_code: int | None = None
    completed_bytes: int | None = None

    @property
    def total_bytes(self) -> int:
        return sum(
            item.page.expected_bytes
            for item in self.reservations
            if item.prepared.disposition is PageDisposition.ALLOCATED
        )


@dataclass
class _Transaction:
    transfer_id: str
    request_attempt: int
    request_id: str
    source_engine_id: str
    remote_session: str
    required_store_end_hint: int
    planned_window_count_hint: int
    d_local_end_open: int
    destination_dp_rank: int
    manifest_digest_seed: str
    state: TransactionState = TransactionState.ACTIVE
    terminal_outcome: TerminalOutcome | None = None
    source_generation: int | None = None
    windows: dict[int, _Window] = field(default_factory=dict)
    registered_selectors: set[tuple[int, int]] = field(default_factory=set)
    registered_keys: set[str] = field(default_factory=set)
    inflight_window_count: int = 0
    reserved_bytes: int = 0
    publication_ineligible: bool = False
    coverage_missing: bool = False
    terminal_at: float | None = None


@dataclass(frozen=True)
class _OperationRecord:
    payload_digest: str
    transfer_id: str
    kind: OperationKind
    response: RemoteFillResponse
    window_id: int | None
    created_at: float


@dataclass(frozen=True)
class RemoteFillMetricsSnapshot:
    """Fixed-cardinality, label-free remote-fill metrics snapshot."""

    active_transactions: int
    active_windows: int
    active_bytes: int
    reserved_bytes_total: int
    submitted_bytes_total: int
    published_bytes_total: int
    discarded_bytes_total: int
    protocol_errors_total: int
    stale_requests_total: int
    capacity_rejections_total: int
    transfer_failures_total: int
    fatal_restarts_total: int
    terminal_local_full_total: int
    terminal_persistent_only_total: int
    terminal_persistence_failed_total: int
    terminal_cancelled_total: int
    terminal_fatal_restart_total: int


@dataclass
class _RemoteFillMetricCounters:
    active_transactions: int = 0
    active_windows: int = 0
    active_bytes: int = 0
    reserved_bytes: int = 0
    submitted_bytes: int = 0
    published_bytes: int = 0
    discarded_bytes: int = 0
    protocol_errors: int = 0
    stale_requests: int = 0
    capacity_rejections: int = 0
    transfer_failures: int = 0
    fatal_restarts: int = 0
    terminal_local_full: int = 0
    terminal_persistent_only: int = 0
    terminal_persistence_failed: int = 0
    terminal_cancelled: int = 0
    terminal_fatal_restart: int = 0


_TERMINAL_STATES = {
    TransactionState.LOCAL_FULL,
    TransactionState.GROUP0_LOCAL,
    TransactionState.PERSISTENT_ONLY,
    TransactionState.PERSISTENCE_FAILED,
    TransactionState.CANCELLED,
    TransactionState.FATAL_RESTART,
}

_GENERATED_TOKEN_MAX_BYTES = 64


class RemoteFillStateCore:
    """Coordinate remote-fill control state without allocating cache pages.

    Allocation and LocalCPU admission are delegated to ``PageLifecycle``. The
    core serializes transitions under one lock, retains armed destinations, and
    never infers that a prefiller submitted or completed native DMA.
    """

    def __init__(
        self,
        *,
        destination_engine_epoch: int,
        shared_cache_generation: int,
        descriptor_verification_key: bytes,
        negotiation: NegotiationSpec,
        page_lifecycle: PageLifecycle,
        limits: ProtocolLimits | None = None,
        reservation_ttl_sec: float = 30.0,
        descriptor_ttl_sec: float | None = None,
        native_hard_timeout_sec: float = 300.0,
        terminal_record_ttl_sec: float = 300.0,
        clock: Callable[[], float] = monotonic,
        token_factory: Callable[[], str] = _new_token,
    ) -> None:
        """Create a state core.

        Args:
            destination_engine_epoch: Epoch changed on each decoder restart.
            shared_cache_generation: Generation of the registered shared pool.
            descriptor_verification_key: Decoder-incarnation key used only to
                authenticate destination descriptors to the selected prefiller.
            negotiation: Exact static layout expected from prefiller peers.
            page_lifecycle: Decoder-owned opaque page callback.
            limits: Protocol bounds, or defaults when omitted.
            reservation_ttl_sec: Lifetime of unarmed reservations only.
            descriptor_ttl_sec: D-local deadline for accepting ARM. Defaults
                to the reservation lifetime and is capped by that lifetime.
            native_hard_timeout_sec: Bound after arm before restart is required.
            terminal_record_ttl_sec: Idempotent replay lifetime after safe terminal
                outcomes. Fatal records are retained until process restart.
            clock: Monotonic clock, injectable for deterministic tests.
            token_factory: Unique ID source for sessions, attempts, and pages.

        Raises:
            ValueError: If a safety-critical configuration value is invalid.
        """

        if destination_engine_epoch <= 0 or shared_cache_generation < 0:
            raise ValueError("engine epoch must be positive; generation nonnegative")
        if not descriptor_verification_key:
            raise ValueError("descriptor_verification_key must not be empty")
        self._validate_negotiation_spec(negotiation)
        descriptor_ttl = (
            reservation_ttl_sec
            if descriptor_ttl_sec is None
            else descriptor_ttl_sec
        )
        if (
            reservation_ttl_sec <= 0
            or descriptor_ttl <= 0
            or native_hard_timeout_sec <= 0
            or terminal_record_ttl_sec <= 0
        ):
            raise ValueError("timeout values must be positive")
        self.destination_engine_epoch = destination_engine_epoch
        self.shared_cache_generation = shared_cache_generation
        self.limits = limits or ProtocolLimits()
        self._validate_limits(self.limits)
        self._descriptor_verification_key = descriptor_verification_key
        self._negotiation = negotiation
        self._lifecycle = page_lifecycle
        self._reservation_ttl_sec = reservation_ttl_sec
        self._descriptor_ttl_sec = min(descriptor_ttl, reservation_ttl_sec)
        self._native_hard_timeout_sec = native_hard_timeout_sec
        self._terminal_record_ttl_sec = terminal_record_ttl_sec
        self._clock = clock
        self._terminal_prune_interval_sec = min(1.0, terminal_record_ttl_sec)
        self._next_terminal_prune_at = self._clock() + self._terminal_prune_interval_sec
        self._token_factory = token_factory
        self._lock = RLock()
        self._transactions: dict[str, _Transaction] = {}
        self._reservations: dict[_ReservationKey, _Reservation] = {}
        self._operations: OrderedDict[str, _OperationRecord] = OrderedDict()
        self._expired_reservation_operations: set[str] = set()
        self._metrics = _RemoteFillMetricCounters()
        self._closed = False

    @property
    def _direct_groups(self) -> tuple[int, ...]:
        """Return the fixed group set selected by exact negotiation."""

        return (0, 1) if self._negotiation.shared_group1 else (0,)

    @staticmethod
    def _validate_negotiation_spec(negotiation: NegotiationSpec) -> None:
        required_text = (
            negotiation.cache_namespace_tag,
            negotiation.layout_tag,
            negotiation.model_artifact_id,
            negotiation.model_layout,
            negotiation.destination_engine_id,
            negotiation.destination_remote_session,
            negotiation.token_hash_algorithm,
        )
        if any(not value for value in required_text):
            raise ValueError("negotiation identity fields must not be empty")
        if (
            negotiation.chunk_size <= 0
            or negotiation.layer_count <= 0
            or negotiation.tp_size <= 0
            or negotiation.dp_size <= 0
            or not negotiation.group_dimensions
            or any(value <= 0 for value in negotiation.group_dimensions)
            or negotiation.destination_dp_rank < 0
            or negotiation.destination_dp_rank >= negotiation.dp_size
        ):
            raise ValueError("negotiation dimensions or rank are invalid")
        if negotiation.token_hash_algorithm == "builtin":
            if not negotiation.python_hash_seed:
                raise ValueError("builtin token hashing requires PYTHONHASHSEED")
        elif negotiation.python_hash_seed:
            raise ValueError(
                "PYTHONHASHSEED identity is valid only for builtin token hashing"
            )

    @staticmethod
    def _validate_limits(limits: ProtocolLimits) -> None:
        values = msgspec.structs.asdict(limits).values()
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("all remote-fill protocol limits must be positive")

    def handle(self, request: RemoteFillRequest) -> RemoteFillResponse:
        """Validate and atomically execute one control operation.

        Args:
            request: A sealed request from the bounded protocol codec.

        Returns:
            A pointer-free result, except successful reservation replies whose
            descriptor field intentionally carries private capabilities.

        Raises:
            ProtocolValidationError: If the request schema or digest is invalid.
        """

        validate_request(request, self.limits)
        return self.handle_validated(request)

    def handle_validated(
        self,
        request: RemoteFillRequest,
    ) -> RemoteFillResponse:
        """Execute a request already validated by the byte-service boundary."""

        with self._lock:
            if self._closed:
                return self._response(
                    request,
                    ResultCode.TERMINAL,
                    "remote-fill state core is shut down",
                )
            self._refresh_timeouts_locked(self._clock())
            if self._restart_required_locked() and not isinstance(
                request,
                (ReportTransferCompleteRequest, AbortRequest, StatusRequest),
            ):
                return self._response(
                    request,
                    ResultCode.FATAL_RESTART_REQUIRED,
                    "paired prefiller and decoder restart is required",
                    transaction_state=TransactionState.FATAL_RESTART,
                    terminal_outcome=TerminalOutcome.FATAL_RESTART,
                    fatal_restart_required=True,
                )
            conflict = self._operation_conflict_locked(request)
            if conflict is not None:
                self._observe_response_locked(conflict)
                return conflict
            terminal = self._terminal_replay_locked(request)
            if terminal is not None:
                return terminal
            cached = self._cached_response_locked(request)
            if cached is not None:
                return cached
            if len(self._operations) >= self.limits.max_operation_records:
                self._evict_reconstructible_operation_locked()
            kind = operation_kind(request)
            requires_record_before_mutation = kind in (
                OperationKind.OPEN,
                OperationKind.RESERVE_WINDOW,
            )
            if (
                len(self._operations) >= self.limits.max_operation_records
                and requires_record_before_mutation
            ):
                response = self._response(
                    request,
                    ResultCode.RESOURCE_EXHAUSTED,
                    "operation record capacity exhausted",
                )
                self._observe_response_locked(response)
                return response
            response = self._dispatch_locked(request)
            self._observe_response_locked(response)
            if len(self._operations) >= self.limits.max_operation_records:
                self._evict_reconstructible_operation_locked()
            if len(self._operations) < self.limits.max_operation_records:
                self._operations[request.common.operation_id] = _OperationRecord(
                    payload_digest=request.common.payload_digest,
                    transfer_id=request.common.transfer_id,
                    kind=kind,
                    response=response,
                    window_id=getattr(request, "window_id", None),
                    created_at=self._clock(),
                )
            return response

    def record_protocol_error(self) -> None:
        """Count a rejected undecodable request without retaining its payload."""

        with self._lock:
            self._metrics.protocol_errors += 1

    def record_capacity_rejection(
        self,
        request: RemoteFillRequest | None = None,
    ) -> None:
        """Count pre-dispatch exhaustion and abandon a required direct hole."""

        with self._lock:
            self._metrics.capacity_rejections += 1
            if not isinstance(request, ReserveWindowRequest):
                return
            transaction = self._transactions.get(request.common.transfer_id)
            if not request.reserve_missing or transaction is None:
                return
            if (
                self._closed
                or self._restart_required_locked()
                or request.common.destination_engine_epoch
                != self.destination_engine_epoch
                or request.common.shared_cache_generation
                != self.shared_cache_generation
                or transaction.state in _TERMINAL_STATES
                or transaction.request_attempt != request.common.request_attempt
                or request.common.operation_id in self._operations
                or (
                    transaction.source_generation is not None
                    and transaction.source_generation != request.source_generation
                )
                or not self._control_pages_match_negotiation(request.control_pages)
            ):
                return
            existing = transaction.windows.get(request.window_id)
            if existing is not None:
                return
            if transaction.registered_selectors & {
                (page.kv_group, page.chunk_index) for page in request.control_pages
            } or transaction.registered_keys & {
                page.canonical_key for page in request.control_pages
            }:
                return
            if (
                transaction.inflight_window_count
                >= self.limits.max_inflight_windows_per_transaction
            ):
                return
            self._mark_direct_abandoned_locked(transaction)

    def validate_response_capacity(
        self,
        request: RemoteFillRequest,
        limits: ProtocolLimits,
    ) -> None:
        """Reject a reservation whose largest valid reply cannot be encoded.

        The check runs before ``PageLifecycle.prepare_pages``. It models every
        requested page as newly allocated, which is the largest possible
        reservation response because it includes both a destination descriptor
        and a page result for each control page.

        Args:
            request: Validated request about to be dispatched.
            limits: Wire bounds enforced by the service facade.

        Raises:
            ProtocolValidationError: If the success response could exceed the
                configured message bound.
        """

        if not isinstance(request, ReserveWindowRequest):
            return
        generated_token = "f" * _GENERATED_TOKEN_MAX_BYTES
        max_address = (1 << 64) - 1
        max_finite_expiry = float.fromhex("0x1.fffffffffffffp+1023")
        descriptors = tuple(
            DestinationPageDescriptor(
                canonical_key=page.canonical_key,
                chunk_index=page.chunk_index,
                remote_session=self._negotiation.destination_remote_session,
                destination_ptr=max_address - page.expected_bytes,
                destination_length=page.expected_bytes,
                reservation_id=generated_token,
                window_id=request.window_id,
                kv_group=page.kv_group,
                transfer_id=request.common.transfer_id,
                request_attempt=request.common.request_attempt,
                destination_dp_rank=self._negotiation.destination_dp_rank,
                destination_tp_rank=page.destination_tp_rank,
                destination_engine_epoch=self.destination_engine_epoch,
                shared_cache_generation=self.shared_cache_generation,
                manifest_digest=request.manifest_digest,
                native_transfer_attempt_id=generated_token,
                expires_at=max_finite_expiry,
                capability_mac="0" * 64,
            )
            for page in request.control_pages
        )
        page_results = tuple(
            PagePreparationStatus(
                canonical_key=page.canonical_key,
                kv_group=page.kv_group,
                chunk_index=page.chunk_index,
                disposition=PageDisposition.ALLOCATED,
            )
            for page in request.control_pages
        )
        response = self._response(
            request,
            ResultCode.OK,
            transaction_state=TransactionState.ACTIVE,
            native_transfer_attempt_id=generated_token,
            destination_descriptor_digest=(
                destination_descriptor_digest(descriptors)
            ),
            descriptors=descriptors,
            page_results=page_results,
        )
        encode_response(response, limits)

    def metrics_snapshot(self) -> RemoteFillMetricsSnapshot:
        """Return an O(1), lock-consistent fixed-cardinality snapshot."""

        with self._lock:
            counters = self._metrics
            return RemoteFillMetricsSnapshot(
                active_transactions=counters.active_transactions,
                active_windows=counters.active_windows,
                active_bytes=counters.active_bytes,
                reserved_bytes_total=counters.reserved_bytes,
                submitted_bytes_total=counters.submitted_bytes,
                published_bytes_total=counters.published_bytes,
                discarded_bytes_total=counters.discarded_bytes,
                protocol_errors_total=counters.protocol_errors,
                stale_requests_total=counters.stale_requests,
                capacity_rejections_total=counters.capacity_rejections,
                transfer_failures_total=counters.transfer_failures,
                fatal_restarts_total=counters.fatal_restarts,
                terminal_local_full_total=counters.terminal_local_full,
                terminal_persistent_only_total=(counters.terminal_persistent_only),
                terminal_persistence_failed_total=(
                    counters.terminal_persistence_failed
                ),
                terminal_cancelled_total=counters.terminal_cancelled,
                terminal_fatal_restart_total=(counters.terminal_fatal_restart),
            )

    def run_maintenance(self) -> tuple[str, ...]:
        """Expire safe reservations and identify fatal armed attempts.

        Returns:
            Transfer IDs newly or already requiring a paired P+D restart.

        Notes:
            Armed pages are retained even after the hard timeout. A caller must
            restart the paired deployment before their addresses can be reused.
        """

        with self._lock:
            self._refresh_timeouts_locked(self._clock())
            return tuple(
                sorted(
                    transfer_id
                    for transfer_id, transaction in self._transactions.items()
                    if transaction.state is TransactionState.FATAL_RESTART
                )
            )

    def shutdown(self) -> tuple[str, ...]:
        """Stop admission and release only destinations proven safe to reuse.

        Returns:
            Transfer IDs with armed or fatal-unknown native attempts. Their
            pages remain retained and require paired P+D restart before reuse.

        Notes:
            The operation is idempotent. Existing LocalCPU snapshots carry no
            pin and require no release.
        """

        with self._lock:
            self._closed = True
            fatal: set[str] = set()
            for transaction in self._transactions.values():
                if transaction.state is TransactionState.FATAL_RESTART:
                    fatal.add(transaction.transfer_id)
                for window in transaction.windows.values():
                    if window.native_state in (
                        DestinationNativeState.ARMED,
                        DestinationNativeState.FATAL_UNKNOWN,
                    ):
                        fatal.add(transaction.transfer_id)
                        self._mark_fatal_locked(transaction)
                        continue
                    if window.state not in (
                        WindowState.COMMITTED,
                        WindowState.RELEASED,
                    ):
                        if not self._release_window_locked(
                            transaction,
                            window,
                            "state shutdown",
                        ):
                            fatal.add(transaction.transfer_id)
                            continue
                        self._set_window_state_locked(window, WindowState.RELEASED)
                if (
                    transaction.transfer_id not in fatal
                    and transaction.state not in _TERMINAL_STATES
                ):
                    self._set_terminal_locked(
                        transaction,
                        TransactionState.CANCELLED,
                        TerminalOutcome.CANCELLED,
                    )
            return tuple(sorted(fatal))

    def _dispatch_locked(
        self,
        request: RemoteFillRequest,
    ) -> RemoteFillResponse:
        if request.common.destination_engine_epoch != self.destination_engine_epoch:
            return self._response(
                request,
                ResultCode.STALE_ENGINE_EPOCH,
                "destination engine epoch is stale",
            )
        if request.common.shared_cache_generation != self.shared_cache_generation:
            return self._response(
                request,
                ResultCode.STALE_CACHE_GENERATION,
                "shared cache generation is stale",
            )
        if self._restart_required_locked() and not isinstance(
            request,
            (ReportTransferCompleteRequest, AbortRequest, StatusRequest),
        ):
            return self._response(
                request,
                ResultCode.FATAL_RESTART_REQUIRED,
                "paired prefiller and decoder restart is required",
                transaction_state=TransactionState.FATAL_RESTART,
                terminal_outcome=TerminalOutcome.FATAL_RESTART,
                fatal_restart_required=True,
            )
        if isinstance(request, NegotiateRequest):
            return self._negotiate_locked(request)
        if isinstance(request, OpenRequest):
            return self._open_locked(request)
        transaction = self._transactions.get(request.common.transfer_id)
        if transaction is None:
            return self._response(
                request, ResultCode.NOT_FOUND, "transaction not found"
            )
        if transaction.request_attempt != request.common.request_attempt:
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "request attempt does not match transaction",
            )
        if isinstance(request, ReserveWindowRequest):
            return self._reserve_locked(request, transaction)
        if isinstance(request, ArmWindowRequest):
            return self._arm_locked(request, transaction)
        if isinstance(request, ReportTransferCompleteRequest):
            return self._report_locked(request, transaction)
        if isinstance(request, FinishRequest):
            return self._finish_locked(request, transaction)
        if isinstance(request, AbortRequest):
            return self._abort_locked(request, transaction)
        if isinstance(request, StatusRequest):
            return self._status_locked(request, transaction)
        return self._response(
            request,
            ResultCode.INVALID_MESSAGE,
            "unsupported operation",
        )

    def _negotiate_locked(self, request: NegotiateRequest) -> RemoteFillResponse:
        supplied = NegotiationSpec(
            cache_namespace_tag=request.cache_namespace_tag,
            layout_tag=request.layout_tag,
            model_artifact_id=request.model_artifact_id,
            chunk_size=request.chunk_size,
            model_layout=request.model_layout,
            group_dimensions=request.group_dimensions,
            layer_count=request.layer_count,
            save_only_first_rank=request.save_only_first_rank,
            shared_group1=request.shared_group1,
            tp_size=request.tp_size,
            dp_size=request.dp_size,
            global_te_push=request.global_te_push,
            destination_engine_id=self._negotiation.destination_engine_id,
            destination_dp_rank=self._negotiation.destination_dp_rank,
            destination_remote_session=(self._negotiation.destination_remote_session),
            token_hash_algorithm=request.token_hash_algorithm,
            python_hash_seed=request.python_hash_seed,
        )
        if supplied != self._negotiation:
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "remote-fill layout negotiation failed",
            )
        return self._response(request, ResultCode.OK)

    def _open_locked(self, request: OpenRequest) -> RemoteFillResponse:
        if (
            request.cache_namespace_tag != self._negotiation.cache_namespace_tag
            or request.layout_tag != self._negotiation.layout_tag
            or request.model_artifact_id != self._negotiation.model_artifact_id
            or request.destination_engine_id != self._negotiation.destination_engine_id
            or request.destination_dp_rank != self._negotiation.destination_dp_rank
        ):
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "open namespace or model artifact mismatch",
            )
        existing = self._transactions.get(request.common.transfer_id)
        if existing is not None:
            if (
                existing.request_attempt != request.common.request_attempt
                or existing.request_id != request.request_id
                or existing.source_engine_id != request.source_engine_id
                or existing.destination_dp_rank != request.destination_dp_rank
                or existing.required_store_end_hint != request.required_store_end_hint
                or existing.planned_window_count_hint
                != request.planned_window_count_hint
                or existing.manifest_digest_seed != request.manifest_digest_seed
            ):
                return self._response(
                    request,
                    ResultCode.WINDOW_CONFLICT,
                    "transfer ID belongs to another request attempt",
                )
            return self._response(
                request,
                ResultCode.ALREADY_EXISTS,
                remote_session=existing.remote_session,
                transaction_state=existing.state,
                d_local_end_open=existing.d_local_end_open,
            )
        if self._metrics.active_transactions >= self.limits.max_active_transactions:
            return self._response(
                request,
                ResultCode.RESOURCE_EXHAUSTED,
                "active transaction capacity exhausted",
            )
        # OPEN is intentionally keyless and therefore cannot prove a local
        # prefix. Exact pages arrive in bounded RESERVE_WINDOW manifests.
        d_local_end_open = 0
        transaction = _Transaction(
            transfer_id=request.common.transfer_id,
            request_attempt=request.common.request_attempt,
            request_id=request.request_id,
            source_engine_id=request.source_engine_id,
            remote_session=self._negotiation.destination_remote_session,
            required_store_end_hint=request.required_store_end_hint,
            planned_window_count_hint=request.planned_window_count_hint,
            d_local_end_open=d_local_end_open,
            destination_dp_rank=request.destination_dp_rank,
            manifest_digest_seed=request.manifest_digest_seed,
        )
        self._transactions[request.common.transfer_id] = transaction
        self._metrics.active_transactions += 1
        return self._response(
            request,
            ResultCode.ACCEPTED,
            remote_session=transaction.remote_session,
            transaction_state=transaction.state,
            d_local_end_open=d_local_end_open,
        )

    def _reserve_locked(
        self,
        request: ReserveWindowRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        if transaction.publication_ineligible and request.reserve_missing:
            return self._response(
                request,
                ResultCode.TERMINAL,
                "transaction is no longer eligible for new allocations",
                transaction_state=transaction.state,
            )
        if transaction.source_generation is None:
            transaction.source_generation = request.source_generation
        elif transaction.source_generation != request.source_generation:
            return self._response(
                request,
                ResultCode.STALE_SOURCE_GENERATION,
                "source generation changed within transaction",
            )
        existing = transaction.windows.get(request.window_id)
        if existing is not None:
            if existing.expired_unarmed:
                transaction.windows.pop(request.window_id)
                transaction.registered_selectors.difference_update(
                    (page.kv_group, page.chunk_index)
                    for page in existing.control_pages
                )
                transaction.registered_keys.difference_update(
                    page.canonical_key for page in existing.control_pages
                )
                transaction.coverage_missing = any(
                    window.coverage_missing for window in transaction.windows.values()
                )
            else:
                return self._response(
                    request,
                    ResultCode.WINDOW_CONFLICT,
                    "window already has a reservation attempt",
                    transaction_state=transaction.state,
                )
        if len(transaction.windows) >= self.limits.max_windows_per_transaction:
            if request.reserve_missing:
                self._mark_direct_abandoned_locked(transaction)
            return self._response(
                request,
                ResultCode.RESOURCE_EXHAUSTED,
                "transaction window capacity exhausted",
                transaction_state=transaction.state,
            )
        requested_selectors = {
            (page.kv_group, page.chunk_index) for page in request.control_pages
        }
        if transaction.registered_selectors & requested_selectors:
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "group/chunk page was already registered in another window",
            )
        requested_keys = {page.canonical_key for page in request.control_pages}
        if transaction.registered_keys & requested_keys:
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "canonical page key was already registered in another window",
            )
        if not self._control_pages_match_negotiation(request.control_pages):
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "control pages do not match negotiated layout",
            )
        if (
            transaction.inflight_window_count
            >= self.limits.max_inflight_windows_per_transaction
        ):
            return self._response(
                request,
                ResultCode.RESOURCE_EXHAUSTED,
                "transaction inflight window capacity exhausted",
            )
        requested_bytes = (
            sum(page.expected_bytes for page in request.control_pages)
            if request.reserve_missing
            else 0
        )
        global_reserved_bytes = self._metrics.active_bytes
        if (
            transaction.reserved_bytes + requested_bytes
            > self.limits.max_bytes_per_transaction
        ):
            if request.reserve_missing:
                self._mark_direct_abandoned_locked(transaction)
            return self._response(
                request,
                ResultCode.RESOURCE_EXHAUSTED,
                "transaction byte capacity exhausted",
                transaction_state=transaction.state,
            )
        if global_reserved_bytes + requested_bytes > self.limits.max_reserved_bytes:
            if request.reserve_missing:
                self._mark_direct_abandoned_locked(transaction)
            return self._response(
                request,
                ResultCode.RESOURCE_EXHAUSTED,
                "global reserved byte capacity exhausted",
                transaction_state=transaction.state,
            )

        generated_tokens: tuple[str, ...] = ()
        if request.reserve_missing:
            try:
                generated_tokens = tuple(
                    self._token_factory()
                    for _ in range(len(request.control_pages) + 1)
                )
                if any(
                    not token
                    or len(token.encode("utf-8")) > _GENERATED_TOKEN_MAX_BYTES
                    for token in generated_tokens
                ):
                    raise ValueError("generated token exceeds its bound")
            except Exception:
                self._mark_direct_abandoned_locked(transaction)
                return self._response(
                    request,
                    ResultCode.RESERVATION_REJECTED,
                    "destination token allocation failed",
                    transaction_state=transaction.state,
                )

        try:
            prepared_pages = tuple(
                self._lifecycle.prepare_pages(
                    transaction.transfer_id,
                    request.window_id,
                    request.control_pages,
                    request.reserve_missing,
                )
            )
        except Exception:
            if request.reserve_missing:
                self._mark_direct_abandoned_locked(transaction)
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "page allocator rejected the window",
                transaction_state=transaction.state,
            )
        if not self._prepared_pages_valid(
            request.control_pages,
            prepared_pages,
            request.reserve_missing,
        ):
            self._release_raw_prepared_locked(
                transaction,
                prepared_pages,
                "invalid allocation result",
            )
            if transaction.state is TransactionState.FATAL_RESTART:
                return self._fatal_response(request, transaction)
            if request.reserve_missing:
                self._mark_direct_abandoned_locked(transaction)
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "allocator returned invalid page preparation results",
                transaction_state=transaction.state,
            )

        now = self._clock()
        reservation_expires_at = now + self._reservation_ttl_sec
        descriptor_expires_at = now + self._descriptor_ttl_sec
        has_allocated = any(
            prepared.disposition is PageDisposition.ALLOCATED
            for prepared in prepared_pages
        )
        has_missing = any(
            prepared.disposition is PageDisposition.MISSING
            for prepared in prepared_pages
        )
        self._metrics.reserved_bytes += sum(
            prepared.destination_length
            for prepared in prepared_pages
            if prepared.disposition is PageDisposition.ALLOCATED
        )
        native_attempt_id = generated_tokens[0] if has_allocated else ""
        reservation_tokens = iter(generated_tokens[1:])
        reservations: list[_Reservation] = []
        descriptors: list[DestinationPageDescriptor] = []
        page_results: list[PagePreparationStatus] = []
        for page, prepared in zip(
            request.control_pages,
            prepared_pages,
            strict=True,
        ):
            page_results.append(
                PagePreparationStatus(
                    canonical_key=page.canonical_key,
                    kv_group=page.kv_group,
                    chunk_index=page.chunk_index,
                    disposition=prepared.disposition,
                )
            )
            if prepared.disposition is PageDisposition.MISSING:
                continue
            key = _ReservationKey(
                transfer_id=transaction.transfer_id,
                window_id=request.window_id,
                kv_group=page.kv_group,
                chunk_index=page.chunk_index,
            )
            if (
                prepared.disposition is PageDisposition.ALLOCATED
                and key in self._reservations
            ):
                self._release_raw_prepared_locked(
                    transaction,
                    prepared_pages,
                    "reservation key collision",
                )
                if transaction.state is TransactionState.FATAL_RESTART:
                    return self._fatal_response(request, transaction)
                self._mark_direct_abandoned_locked(transaction)
                return self._response(
                    request,
                    ResultCode.WINDOW_CONFLICT,
                    "reservation key already exists",
                    transaction_state=transaction.state,
                )
            reservation_id = (
                next(reservation_tokens)
                if prepared.disposition is PageDisposition.ALLOCATED
                else ""
            )
            reservation = _Reservation(
                key=key,
                page=page,
                prepared=prepared,
                reservation_id=reservation_id,
                state=(
                    WindowState.RESERVED
                    if prepared.disposition is PageDisposition.ALLOCATED
                    else WindowState.READY_HIDDEN
                ),
                created_at=now,
                expires_at=(
                    reservation_expires_at
                    if prepared.disposition is PageDisposition.ALLOCATED
                    else 0.0
                ),
            )
            reservations.append(reservation)
            if prepared.disposition is PageDisposition.ALLOCATED:
                descriptors.append(
                    seal_descriptor(
                        self._descriptor_verification_key,
                        DestinationPageDescriptor(
                            canonical_key=page.canonical_key,
                            chunk_index=page.chunk_index,
                            remote_session=transaction.remote_session,
                            destination_ptr=prepared.destination_ptr,
                            destination_length=prepared.destination_length,
                            reservation_id=reservation_id,
                            window_id=request.window_id,
                            kv_group=page.kv_group,
                            transfer_id=transaction.transfer_id,
                            request_attempt=transaction.request_attempt,
                            destination_dp_rank=transaction.destination_dp_rank,
                            destination_tp_rank=page.destination_tp_rank,
                            destination_engine_epoch=self.destination_engine_epoch,
                            shared_cache_generation=self.shared_cache_generation,
                            manifest_digest=request.manifest_digest,
                            native_transfer_attempt_id=native_attempt_id,
                            expires_at=descriptor_expires_at,
                            capability_mac="",
                        ),
                    )
                )

        descriptor_tuple = tuple(descriptors)
        page_result_tuple = tuple(page_results)
        if has_missing:
            self._release_raw_prepared_locked(
                transaction,
                prepared_pages,
                "incomplete page preparation",
            )
            if transaction.state is TransactionState.FATAL_RESTART:
                return self._fatal_response(request, transaction)
            self._mark_direct_abandoned_locked(transaction)
        window = _Window(
            window_id=request.window_id,
            source_generation=request.source_generation,
            manifest_digest=request.manifest_digest,
            control_pages=request.control_pages,
            native_transfer_attempt_id=native_attempt_id,
            descriptor_digest=(
                destination_descriptor_digest(descriptor_tuple)
                if descriptor_tuple
                else ""
            ),
            descriptors=descriptor_tuple if not has_missing else (),
            reservations=tuple(reservations),
            page_results=page_result_tuple,
            descriptor_expires_at=(
                descriptor_expires_at if has_allocated else 0.0
            ),
            state=(
                WindowState.RELEASED
                if has_missing
                else (
                    WindowState.RESERVED if has_allocated else WindowState.READY_HIDDEN
                )
            ),
            native_state=(
                DestinationNativeState.TERMINAL_FAILURE
                if has_missing
                else (
                    DestinationNativeState.NOT_ARMED
                    if has_allocated
                    else DestinationNativeState.TERMINAL_SUCCESS
                )
            ),
            coverage_missing=has_missing,
        )
        transaction.windows[request.window_id] = window
        transaction.registered_selectors.update(requested_selectors)
        transaction.registered_keys.update(requested_keys)
        if window.state not in (WindowState.RELEASED, WindowState.COMMITTED):
            allocated_bytes = sum(
                reservation.prepared.destination_length
                for reservation in window.reservations
                if reservation.prepared.disposition is PageDisposition.ALLOCATED
                and not reservation.released
            )
            self._metrics.active_windows += 1
            self._metrics.active_bytes += allocated_bytes
            transaction.reserved_bytes += allocated_bytes
        if window.native_state in (
            DestinationNativeState.NOT_ARMED,
            DestinationNativeState.ARMED,
        ) and window.state not in (WindowState.RELEASED, WindowState.COMMITTED):
            transaction.inflight_window_count += 1
        if not has_missing:
            for reservation in reservations:
                if reservation.prepared.disposition is PageDisposition.ALLOCATED:
                    self._reservations[reservation.key] = reservation
        if has_missing:
            return self._response(
                request,
                ResultCode.RESERVATION_REJECTED,
                "one or more authoritative pages are not local and were not allocated",
                transaction_state=transaction.state,
                page_results=page_result_tuple,
            )
        return self._response(
            request,
            ResultCode.OK,
            transaction_state=transaction.state,
            native_transfer_attempt_id=native_attempt_id,
            destination_descriptor_digest=window.descriptor_digest,
            descriptors=descriptor_tuple,
            page_results=page_result_tuple,
        )

    def _arm_locked(
        self,
        request: ArmWindowRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        window = transaction.windows.get(request.window_id)
        if window is None:
            return self._response(request, ResultCode.NOT_FOUND, "window not found")
        if window.expired_unarmed:
            return self._response(
                request,
                ResultCode.RESERVATION_EXPIRED,
                "unarmed reservation expired",
                transaction_state=transaction.state,
            )
        if not window.descriptors:
            return self._response(
                request,
                ResultCode.TERMINAL,
                "window has no absent pages requiring native transfer",
                transaction_state=transaction.state,
            )
        if (
            request.native_transfer_attempt_id != window.native_transfer_attempt_id
            or request.manifest_digest != window.manifest_digest
            or request.destination_descriptor_digest != window.descriptor_digest
        ):
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "arm evidence does not match the reservation",
            )
        if window.native_state is DestinationNativeState.ARMED:
            return self._response(
                request,
                ResultCode.OK,
                transaction_state=transaction.state,
                native_transfer_attempt_id=window.native_transfer_attempt_id,
            )
        if window.native_state in (
            DestinationNativeState.TERMINAL_SUCCESS,
            DestinationNativeState.TERMINAL_FAILURE,
        ):
            return self._response(
                request,
                ResultCode.TERMINAL,
                "native attempt is already terminal",
                transaction_state=transaction.state,
            )
        if window.state is not WindowState.RESERVED:
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "window is not armable",
            )
        now = self._clock()
        if window.descriptor_expires_at <= now:
            return self._response(
                request,
                ResultCode.RESERVATION_EXPIRED,
                "destination descriptor expired before arm",
                transaction_state=transaction.state,
            )
        if any(
            reservation.prepared.disposition is PageDisposition.ALLOCATED
            and reservation.expires_at <= now
            for reservation in window.reservations
        ):
            self._expire_unarmed_window_locked(transaction, window)
            return self._response(
                request,
                ResultCode.RESERVATION_EXPIRED,
                "unarmed reservation expired",
                transaction_state=transaction.state,
            )
        # Whole-window atomic transition: validate every reservation first.
        if any(
            reservation.prepared.disposition is PageDisposition.ALLOCATED
            and (reservation.state is not WindowState.RESERVED or reservation.released)
            for reservation in window.reservations
        ):
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "reservation set is incomplete",
            )
        self._set_window_state_locked(window, WindowState.ARMED)
        window.native_state = DestinationNativeState.ARMED
        window.armed_at = now
        for reservation in window.reservations:
            if reservation.prepared.disposition is PageDisposition.ALLOCATED:
                reservation.state = WindowState.ARMED
                reservation.armed_at = now
        self._metrics.submitted_bytes += window.total_bytes
        return self._response(
            request,
            ResultCode.OK,
            transaction_state=transaction.state,
            native_transfer_attempt_id=window.native_transfer_attempt_id,
        )

    def _report_locked(
        self,
        request: ReportTransferCompleteRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        window = transaction.windows.get(request.window_id)
        if window is None:
            return self._response(request, ResultCode.NOT_FOUND, "window not found")
        if transaction.state is TransactionState.FATAL_RESTART:
            return self._fatal_response(request, transaction)
        if (
            request.native_transfer_attempt_id != window.native_transfer_attempt_id
            or request.manifest_digest != window.manifest_digest
        ):
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "completion evidence does not match the armed attempt",
            )
        if window.native_state in (
            DestinationNativeState.TERMINAL_SUCCESS,
            DestinationNativeState.TERMINAL_FAILURE,
        ):
            if (
                request.native_return_code == window.native_return_code
                and request.completed_bytes == window.completed_bytes
            ):
                code = (
                    ResultCode.OK
                    if window.native_state is DestinationNativeState.TERMINAL_SUCCESS
                    else ResultCode.TRANSFER_FAILED
                )
                return self._response(
                    request,
                    code,
                    transaction_state=transaction.state,
                )
            return self._response(
                request,
                ResultCode.WINDOW_CONFLICT,
                "terminal completion evidence changed",
            )
        if window.native_state is not DestinationNativeState.ARMED:
            return self._response(
                request,
                ResultCode.WINDOW_NOT_ARMED,
                "completion requires a matching armed attempt",
            )
        window.native_return_code = request.native_return_code
        window.completed_bytes = request.completed_bytes
        transfer_ok = (
            request.native_return_code == 0
            and request.completed_bytes == window.total_bytes
        )
        if transfer_ok:
            self._set_window_native_terminal_locked(
                transaction,
                window,
                DestinationNativeState.TERMINAL_SUCCESS,
            )
            if transaction.publication_ineligible:
                if not self._release_window_locked(transaction, window, "aborted"):
                    return self._fatal_response(request, transaction)
                self._set_window_state_locked(window, WindowState.RELEASED)
                self._finish_abort_if_drained_locked(transaction)
            else:
                self._set_window_state_locked(window, WindowState.READY_HIDDEN)
                for reservation in window.reservations:
                    if reservation.prepared.disposition is PageDisposition.ALLOCATED:
                        reservation.state = WindowState.READY_HIDDEN
            return self._response(
                request,
                ResultCode.OK,
                transaction_state=transaction.state,
            )

        self._set_window_native_terminal_locked(
            transaction,
            window,
            DestinationNativeState.TERMINAL_FAILURE,
        )
        self._metrics.transfer_failures += 1
        if not self._release_window_locked(transaction, window, "native failure"):
            return self._fatal_response(request, transaction)
        self._set_window_state_locked(window, WindowState.RELEASED)
        transaction.publication_ineligible = True
        if transaction.state is TransactionState.ABORT_REQUESTED:
            self._finish_abort_if_drained_locked(transaction)
        else:
            transaction.state = TransactionState.DIRECT_ABANDONED
        return self._response(
            request,
            ResultCode.TRANSFER_FAILED,
            "native transfer did not satisfy completion contract",
            transaction_state=transaction.state,
        )

    def _finish_locked(
        self,
        request: FinishRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        expected_partial = request.required_store_end % self._negotiation.chunk_size
        if request.final_partial_valid_tokens != expected_partial:
            return self._response(
                request,
                ResultCode.INVALID_MESSAGE,
                "final partial-page metadata does not match required store end",
                transaction_state=transaction.state,
            )
        expected_manifest_digest = transaction_manifest_digest(
            transaction.manifest_digest_seed,
            tuple(
                (window.window_id, window.manifest_digest)
                for window in sorted(
                    transaction.windows.values(),
                    key=lambda item: item.window_id,
                )
            ),
            request.required_store_end,
            request.final_partial_valid_tokens,
        )
        if request.final_manifest_digest != expected_manifest_digest:
            return self._response(
                request,
                ResultCode.INVALID_DIGEST,
                "final transaction manifest digest mismatch",
                transaction_state=transaction.state,
            )
        for window in transaction.windows.values():
            if window.native_state is DestinationNativeState.ARMED:
                return self._response(
                    request,
                    ResultCode.WINDOW_NOT_TERMINAL,
                    "an armed native attempt has no terminal report",
                    transaction_state=transaction.state,
                )
            if (
                window.native_state is DestinationNativeState.NOT_ARMED
                and window.state is WindowState.RESERVED
            ):
                if not self._release_window_locked(
                    transaction,
                    window,
                    "finish before arm",
                ):
                    return self._fatal_response(request, transaction)
                self._set_window_state_locked(window, WindowState.RELEASED)
                window.coverage_missing = True
                transaction.coverage_missing = True

        # Persistent durability is mandatory. Direct placement accelerates
        # decoder handoff but must never become the only discoverable copy.
        if request.persistent_common_end < request.required_store_end:
            return self._finish_without_commit_locked(request, transaction)

        required_pages = tuple(
            page
            for window in sorted(
                transaction.windows.values(),
                key=lambda item: item.window_id,
            )
            for page in window.control_pages
        )
        exact_coverage = self._has_exact_direct_group_coverage(
            required_pages,
            request.required_store_end,
            self._direct_groups,
        )
        if (
            transaction.publication_ineligible
            or transaction.coverage_missing
            or not exact_coverage
        ):
            return self._finish_without_commit_locked(request, transaction)
        transaction.state = TransactionState.FINALIZING
        prepared_pages = tuple(
            reservation.view()
            for window in transaction.windows.values()
            if window.state is WindowState.READY_HIDDEN
            for reservation in window.reservations
            if reservation.prepared.disposition
            in (PageDisposition.EXISTING, PageDisposition.ALLOCATED)
        )
        try:
            local_complete = self._lifecycle.commit_pages(
                transaction.transfer_id,
                required_pages,
                prepared_pages,
                request,
            )
        except UnsafePageLifecycleError:
            self._mark_fatal_locked(transaction)
            return self._fatal_response(request, transaction)
        except Exception:
            local_complete = False
        if not isinstance(local_complete, bool):
            self._mark_fatal_locked(transaction)
            return self._fatal_response(request, transaction)
        if local_complete:
            self._metrics.published_bytes += sum(
                window.total_bytes
                for window in transaction.windows.values()
                if window.state is WindowState.READY_HIDDEN
            )
            for window in transaction.windows.values():
                if window.state is WindowState.READY_HIDDEN:
                    self._metrics.active_bytes = max(
                        0,
                        self._metrics.active_bytes - window.total_bytes,
                    )
                    transaction.reserved_bytes = max(
                        0,
                        transaction.reserved_bytes - window.total_bytes,
                    )
                    self._set_window_state_locked(window, WindowState.COMMITTED)
                    for reservation in window.reservations:
                        reservation.state = WindowState.COMMITTED
            if self._direct_groups == (0, 1):
                terminal_state = TransactionState.LOCAL_FULL
                terminal_outcome = TerminalOutcome.LOCAL_FULL
            else:
                terminal_state = TransactionState.GROUP0_LOCAL
                terminal_outcome = TerminalOutcome.PERSISTENT_ONLY
            self._set_terminal_locked(
                transaction,
                terminal_state,
                terminal_outcome,
            )
            for window in transaction.windows.values():
                if window.state is WindowState.COMMITTED:
                    for reservation in window.reservations:
                        self._reservations.pop(reservation.key, None)
        else:
            for window in transaction.windows.values():
                if (
                    window.state is WindowState.READY_HIDDEN
                    and not self._release_window_locked(
                        transaction,
                        window,
                        "finish without local full",
                    )
                ):
                    return self._fatal_response(request, transaction)
                if window.state is WindowState.READY_HIDDEN:
                    self._set_window_state_locked(window, WindowState.RELEASED)
            if request.persistent_common_end >= request.required_store_end:
                self._set_terminal_locked(
                    transaction,
                    TransactionState.PERSISTENT_ONLY,
                    TerminalOutcome.PERSISTENT_ONLY,
                )
            else:
                self._set_terminal_locked(
                    transaction,
                    TransactionState.PERSISTENCE_FAILED,
                    TerminalOutcome.PERSISTENCE_FAILED,
                )
        return self._response(
            request,
            ResultCode.TERMINAL,
            transaction_state=transaction.state,
            terminal_outcome=transaction.terminal_outcome,
        )

    def _abort_locked(
        self,
        request: AbortRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        transaction.publication_ineligible = True
        transaction.state = TransactionState.ABORT_REQUESTED
        for window in transaction.windows.values():
            if window.native_state is DestinationNativeState.ARMED:
                self._set_window_state_locked(window, WindowState.ABORT_REQUESTED)
                for reservation in window.reservations:
                    reservation.state = WindowState.ABORT_REQUESTED
                continue
            if window.state not in (WindowState.COMMITTED, WindowState.RELEASED):
                if not self._release_window_locked(transaction, window, "abort"):
                    return self._fatal_response(request, transaction)
                self._set_window_state_locked(window, WindowState.RELEASED)
        self._finish_abort_if_drained_locked(transaction)
        code = (
            ResultCode.TERMINAL
            if transaction.state is TransactionState.CANCELLED
            else ResultCode.ACCEPTED
        )
        return self._response(
            request,
            code,
            transaction_state=transaction.state,
            terminal_outcome=transaction.terminal_outcome,
        )

    def _status_locked(
        self,
        request: StatusRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        windows = self._window_statuses(transaction, request.window_id)
        if request.window_id >= 0 and not windows:
            return self._response(request, ResultCode.NOT_FOUND, "window not found")
        return self._response(
            request,
            ResultCode.OK,
            transaction_state=transaction.state,
            terminal_outcome=transaction.terminal_outcome,
            windows=windows,
            fatal_restart_required=(
                transaction.state is TransactionState.FATAL_RESTART
            ),
        )

    def _refresh_timeouts_locked(self, now: float) -> None:
        for transaction in self._transactions.values():
            if transaction.state in _TERMINAL_STATES:
                continue
            for window in transaction.windows.values():
                if (
                    window.native_state is DestinationNativeState.NOT_ARMED
                    and window.state is WindowState.RESERVED
                    and any(
                        item.prepared.disposition is PageDisposition.ALLOCATED
                        and item.expires_at <= now
                        for item in window.reservations
                    )
                ):
                    self._expire_unarmed_window_locked(transaction, window)
                elif (
                    window.native_state is DestinationNativeState.ARMED
                    and window.armed_at is not None
                    and now - window.armed_at >= self._native_hard_timeout_sec
                ):
                    self._set_window_native_terminal_locked(
                        transaction,
                        window,
                        DestinationNativeState.FATAL_UNKNOWN,
                    )
                    self._set_window_state_locked(window, WindowState.FATAL_RESTART)
                    for reservation in window.reservations:
                        reservation.state = WindowState.FATAL_RESTART
                    self._mark_fatal_locked(transaction)
            if transaction.state is TransactionState.ABORT_REQUESTED:
                self._finish_abort_if_drained_locked(transaction)
        if now >= self._next_terminal_prune_at:
            self._next_terminal_prune_at = now + self._terminal_prune_interval_sec
            self._prune_terminal_records_locked(now)

    def _expire_unarmed_window_locked(
        self,
        transaction: _Transaction,
        window: _Window,
    ) -> None:
        if window.native_state is not DestinationNativeState.NOT_ARMED:
            return
        if not self._release_window_locked(transaction, window, "unarmed TTL expiry"):
            return
        window.expired_unarmed = True
        window.coverage_missing = True
        transaction.coverage_missing = True
        self._set_window_state_locked(window, WindowState.RELEASED)
        for operation_id, record in tuple(self._operations.items()):
            if (
                record.kind is OperationKind.RESERVE_WINDOW
                and record.transfer_id == transaction.transfer_id
                and record.window_id == window.window_id
            ):
                self._expired_reservation_operations.add(operation_id)
        if transaction.state is TransactionState.ABORT_REQUESTED:
            self._finish_abort_if_drained_locked(transaction)

    def _release_window_locked(
        self,
        transaction: _Transaction,
        window: _Window,
        reason: str,
    ) -> bool:
        pages = tuple(
            reservation.view()
            for reservation in window.reservations
            if reservation.prepared.disposition is PageDisposition.ALLOCATED
            and not reservation.released
        )
        if not pages:
            return True
        try:
            self._lifecycle.release_pages(pages, reason)
        except Exception:
            self._mark_fatal_locked(transaction)
            return False
        if window.native_state in (
            DestinationNativeState.NOT_ARMED,
            DestinationNativeState.ARMED,
        ):
            transaction.inflight_window_count = max(
                0,
                transaction.inflight_window_count - 1,
            )
        released_bytes = sum(page.prepared.destination_length for page in pages)
        self._metrics.discarded_bytes += released_bytes
        self._metrics.active_bytes = max(
            0,
            self._metrics.active_bytes - released_bytes,
        )
        transaction.reserved_bytes = max(
            0,
            transaction.reserved_bytes - released_bytes,
        )
        for reservation in window.reservations:
            if reservation.prepared.disposition is PageDisposition.ALLOCATED:
                reservation.released = True
                reservation.state = WindowState.RELEASED
                self._reservations.pop(reservation.key, None)
        return True

    def _release_raw_prepared_locked(
        self,
        transaction: _Transaction,
        prepared_pages: tuple[PreparedPage, ...],
        reason: str,
    ) -> None:
        pages = tuple(
            prepared
            for prepared in prepared_pages
            if isinstance(prepared, PreparedPage)
            and prepared.disposition is PageDisposition.ALLOCATED
        )
        if not pages:
            return
        try:
            self._lifecycle.release_prepared_pages(pages, reason)
        except Exception:
            self._mark_fatal_locked(transaction)
            return
        self._metrics.discarded_bytes += sum(page.destination_length for page in pages)

    @staticmethod
    def _mark_direct_abandoned_locked(transaction: _Transaction) -> None:
        """Make a transaction permanently ineligible after a required hole."""

        transaction.coverage_missing = True
        transaction.publication_ineligible = True
        if transaction.state not in _TERMINAL_STATES:
            transaction.state = TransactionState.DIRECT_ABANDONED

    def _prepared_pages_valid(
        self,
        pages: tuple[ControlPage, ...],
        prepared_pages: tuple[PreparedPage, ...],
        reserve_missing: bool,
    ) -> bool:
        if len(pages) != len(prepared_pages):
            return False
        ranges: list[tuple[int, int]] = []
        max_address = (1 << 64) - 1
        for page, prepared in zip(pages, prepared_pages, strict=True):
            if not isinstance(prepared, PreparedPage):
                return False
            if prepared.disposition in (
                PageDisposition.EXISTING,
                PageDisposition.MISSING,
            ):
                if (
                    prepared.handle is not None
                    or prepared.destination_ptr != 0
                    or prepared.destination_length != 0
                    or prepared.reservation_base != 0
                    or prepared.reservation_length != 0
                ):
                    return False
                continue
            if prepared.disposition is not PageDisposition.ALLOCATED:
                return False
            if not reserve_missing or prepared.handle is None:
                return False
            if (
                prepared.reservation_base < 0
                or prepared.destination_ptr < prepared.reservation_base
                or prepared.destination_length != page.expected_bytes
                or prepared.reservation_length < prepared.destination_length
                or prepared.destination_ptr > max_address
                or prepared.reservation_base > max_address
            ):
                return False
            end = prepared.destination_ptr + prepared.destination_length
            reservation_end = prepared.reservation_base + prepared.reservation_length
            if end > reservation_end or end > max_address:
                return False
            ranges.append((prepared.destination_ptr, end))
        ranges.sort()
        if not all(
            left[1] <= right[0] for left, right in zip(ranges, ranges[1:], strict=False)
        ):
            return False
        active_ranges = sorted(
            (
                reservation.prepared.destination_ptr,
                reservation.prepared.destination_ptr
                + reservation.prepared.destination_length,
            )
            for reservation in self._reservations.values()
            if not reservation.released
            and reservation.prepared.disposition is PageDisposition.ALLOCATED
        )
        return all(
            new_end <= active_start or active_end <= new_start
            for new_start, new_end in ranges
            for active_start, active_end in active_ranges
        )

    @staticmethod
    def _has_exact_direct_group_coverage(
        pages: tuple[ControlPage, ...],
        required_store_end: int,
        direct_groups: tuple[int, ...],
    ) -> bool:
        if required_store_end == 0:
            return not pages
        expected_groups = set(direct_groups)
        if not expected_groups or not expected_groups.issubset({0, 1}):
            return False
        by_chunk: dict[int, list[ControlPage]] = {}
        for page in pages:
            by_chunk.setdefault(page.chunk_index, []).append(page)
        intervals: list[tuple[int, int]] = []
        for chunk_pages in by_chunk.values():
            if len(chunk_pages) != len(direct_groups) or {
                page.kv_group for page in chunk_pages
            } != expected_groups:
                return False
            first = chunk_pages[0]
            metadata = {
                (
                    page.chunk_start,
                    page.chunk_end,
                    page.valid_tokens,
                    page.layer_count,
                    page.layout_tag,
                )
                for page in chunk_pages
            }
            if len(metadata) != 1:
                return False
            intervals.append(
                (first.chunk_start, first.chunk_start + first.valid_tokens)
            )
        intervals.sort()
        expected_start = 0
        for start, end in intervals:
            if start != expected_start or end <= start:
                return False
            expected_start = end
        return expected_start == required_store_end

    def _control_pages_match_negotiation(
        self,
        pages: tuple[ControlPage, ...],
    ) -> bool:
        chunk_size = self._negotiation.chunk_size
        groups_by_chunk: dict[int, set[int]] = {}
        for page in pages:
            groups_by_chunk.setdefault(page.chunk_index, set()).add(page.kv_group)
        expected_groups = set(self._direct_groups)
        return bool(groups_by_chunk) and all(
            page.layout_tag == self._negotiation.layout_tag
            and page.layer_count == self._negotiation.layer_count
            and page.chunk_end - page.chunk_start == page.valid_tokens
            and 0 < page.valid_tokens <= chunk_size
            and page.chunk_start == page.chunk_index * chunk_size
            for page in pages
        ) and all(groups == expected_groups for groups in groups_by_chunk.values())

    def _finish_abort_if_drained_locked(self, transaction: _Transaction) -> None:
        if transaction.state is not TransactionState.ABORT_REQUESTED:
            return
        if any(
            window.native_state is DestinationNativeState.ARMED
            for window in transaction.windows.values()
        ):
            return
        self._set_terminal_locked(
            transaction,
            TransactionState.CANCELLED,
            TerminalOutcome.CANCELLED,
        )

    def _mark_fatal_locked(self, transaction: _Transaction) -> None:
        transaction.publication_ineligible = True
        if transaction.state is not TransactionState.FATAL_RESTART:
            self._metrics.fatal_restarts += 1
        self._set_terminal_locked(
            transaction,
            TransactionState.FATAL_RESTART,
            TerminalOutcome.FATAL_RESTART,
        )

    def _set_terminal_locked(
        self,
        transaction: _Transaction,
        state: TransactionState,
        outcome: TerminalOutcome,
    ) -> None:
        was_active = transaction.state not in _TERMINAL_STATES
        prior = transaction.terminal_outcome
        transaction.state = state
        transaction.terminal_outcome = outcome
        transaction.terminal_at = self._clock()
        if was_active and state in _TERMINAL_STATES:
            self._metrics.active_transactions = max(
                0,
                self._metrics.active_transactions - 1,
            )
        if prior is outcome:
            return
        counters = self._metrics
        if outcome is TerminalOutcome.LOCAL_FULL:
            counters.terminal_local_full += 1
        elif outcome is TerminalOutcome.PERSISTENT_ONLY:
            counters.terminal_persistent_only += 1
        elif outcome is TerminalOutcome.PERSISTENCE_FAILED:
            counters.terminal_persistence_failed += 1
        elif outcome is TerminalOutcome.CANCELLED:
            counters.terminal_cancelled += 1
        elif outcome is TerminalOutcome.FATAL_RESTART:
            counters.terminal_fatal_restart += 1

    def _set_window_state_locked(
        self,
        window: _Window,
        state: WindowState,
    ) -> None:
        """Update one window and its O(1) active gauge under the state lock."""

        was_active = window.state not in (WindowState.COMMITTED, WindowState.RELEASED)
        is_active = state not in (WindowState.COMMITTED, WindowState.RELEASED)
        window.state = state
        if was_active == is_active:
            return
        if is_active:
            self._metrics.active_windows += 1
        else:
            self._metrics.active_windows = max(
                0,
                self._metrics.active_windows - 1,
            )

    @staticmethod
    def _set_window_native_terminal_locked(
        transaction: _Transaction,
        window: _Window,
        state: DestinationNativeState,
    ) -> None:
        if window.native_state in (
            DestinationNativeState.NOT_ARMED,
            DestinationNativeState.ARMED,
        ):
            transaction.inflight_window_count = max(
                0,
                transaction.inflight_window_count - 1,
            )
        window.native_state = state

    def _observe_response_locked(self, response: RemoteFillResponse) -> None:
        if response.code in (
            ResultCode.INVALID_MESSAGE,
            ResultCode.INVALID_DIGEST,
            ResultCode.OPERATION_CONFLICT,
            ResultCode.WINDOW_CONFLICT,
            ResultCode.WINDOW_NOT_ARMED,
        ):
            self._metrics.protocol_errors += 1
        elif response.code in (
            ResultCode.STALE_ENGINE_EPOCH,
            ResultCode.STALE_CACHE_GENERATION,
            ResultCode.STALE_SOURCE_GENERATION,
        ):
            self._metrics.stale_requests += 1
        elif response.code is ResultCode.RESOURCE_EXHAUSTED:
            self._metrics.capacity_rejections += 1

    def _restart_required_locked(self) -> bool:
        return any(
            transaction.state is TransactionState.FATAL_RESTART
            for transaction in self._transactions.values()
        )

    def _finish_without_commit_locked(
        self,
        request: FinishRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        for window in transaction.windows.values():
            if window.state is WindowState.READY_HIDDEN:
                if not self._release_window_locked(
                    transaction,
                    window,
                    "direct-abandoned finish",
                ):
                    return self._fatal_response(request, transaction)
                self._set_window_state_locked(window, WindowState.RELEASED)
        if request.persistent_common_end >= request.required_store_end:
            self._set_terminal_locked(
                transaction,
                TransactionState.PERSISTENT_ONLY,
                TerminalOutcome.PERSISTENT_ONLY,
            )
        else:
            self._set_terminal_locked(
                transaction,
                TransactionState.PERSISTENCE_FAILED,
                TerminalOutcome.PERSISTENCE_FAILED,
            )
        return self._response(
            request,
            ResultCode.TERMINAL,
            transaction_state=transaction.state,
            terminal_outcome=transaction.terminal_outcome,
        )

    def _prune_terminal_records_locked(self, now: float) -> None:
        expired_transfers = {
            transfer_id
            for transfer_id, transaction in self._transactions.items()
            if transaction.state in _TERMINAL_STATES
            and transaction.state is not TransactionState.FATAL_RESTART
            and transaction.terminal_at is not None
            and now - transaction.terminal_at >= self._terminal_record_ttl_sec
        }
        for transfer_id in expired_transfers:
            self._transactions.pop(transfer_id, None)
        protected_transfers = {
            transfer_id
            for transfer_id, transaction in self._transactions.items()
            if transaction.state not in _TERMINAL_STATES
            or transaction.state is TransactionState.FATAL_RESTART
        }
        for operation_id, record in tuple(self._operations.items()):
            if record.transfer_id in expired_transfers or (
                record.transfer_id not in protected_transfers
                and now - record.created_at >= self._terminal_record_ttl_sec
            ):
                self._operations.pop(operation_id, None)
                self._expired_reservation_operations.discard(operation_id)

    def _operation_conflict_locked(
        self,
        request: RemoteFillRequest,
    ) -> RemoteFillResponse | None:
        record = self._operations.get(request.common.operation_id)
        if record is None or record.payload_digest == request.common.payload_digest:
            return None
        return self._response(
            request,
            ResultCode.OPERATION_CONFLICT,
            "operation ID was reused with a different payload",
        )

    def _evict_reconstructible_operation_locked(self) -> bool:
        for operation_id, record in self._operations.items():
            if record.kind is not OperationKind.RESERVE_WINDOW:
                self._operations.pop(operation_id)
                self._expired_reservation_operations.discard(operation_id)
                return True
            transaction = self._transactions.get(record.transfer_id)
            window = (
                transaction.windows.get(record.window_id)
                if transaction is not None and record.window_id is not None
                else None
            )
            live_capability = (
                window is not None
                and window.native_state is DestinationNativeState.NOT_ARMED
                and bool(window.descriptors)
                and not window.expired_unarmed
            )
            if not live_capability:
                self._operations.pop(operation_id)
                self._expired_reservation_operations.discard(operation_id)
                return True
        return False

    def _cached_response_locked(
        self,
        request: RemoteFillRequest,
    ) -> RemoteFillResponse | None:
        record = self._operations.get(request.common.operation_id)
        if record is None:
            return None
        if request.common.operation_id in self._expired_reservation_operations:
            return msgspec.structs.replace(
                record.response,
                code=ResultCode.RESERVATION_EXPIRED,
                message="unarmed reservation expired",
                descriptors=(),
                destination_descriptor_digest="",
            )
        if record.kind is OperationKind.RESERVE_WINDOW and record.window_id is not None:
            transaction = self._transactions.get(record.transfer_id)
            window = (
                transaction.windows.get(record.window_id)
                if transaction is not None
                else None
            )
            if window is not None and window.expired_unarmed:
                return msgspec.structs.replace(
                    record.response,
                    code=ResultCode.RESERVATION_EXPIRED,
                    message="unarmed reservation expired",
                    descriptors=(),
                    destination_descriptor_digest="",
                )
        return record.response

    def _terminal_replay_locked(
        self,
        request: RemoteFillRequest,
    ) -> RemoteFillResponse | None:
        if isinstance(request, (NegotiateRequest, StatusRequest)):
            return None
        transaction = self._transactions.get(request.common.transfer_id)
        if transaction is None or transaction.state not in _TERMINAL_STATES:
            return None
        return self._response(
            request,
            ResultCode.TERMINAL,
            transaction_state=transaction.state,
            terminal_outcome=transaction.terminal_outcome,
            fatal_restart_required=(
                transaction.state is TransactionState.FATAL_RESTART
            ),
        )

    def _window_statuses(
        self,
        transaction: _Transaction,
        window_id: int,
    ) -> tuple[WindowStatus, ...]:
        windows = (
            [transaction.windows[window_id]]
            if window_id in transaction.windows
            else ([] if window_id >= 0 else list(transaction.windows.values()))
        )
        return tuple(
            WindowStatus(
                window_id=window.window_id,
                state=window.state,
                native_state=window.native_state,
                native_transfer_attempt_id=window.native_transfer_attempt_id,
                page_count=len(window.reservations),
                total_bytes=window.total_bytes,
                expired_unarmed=window.expired_unarmed,
                fatal_restart_required=(
                    window.native_state is DestinationNativeState.FATAL_UNKNOWN
                ),
            )
            for window in sorted(windows, key=lambda item: item.window_id)
        )

    def _fatal_response(
        self,
        request: RemoteFillRequest,
        transaction: _Transaction,
    ) -> RemoteFillResponse:
        return self._response(
            request,
            ResultCode.FATAL_RESTART_REQUIRED,
            "paired prefiller and decoder restart is required",
            transaction_state=TransactionState.FATAL_RESTART,
            terminal_outcome=TerminalOutcome.FATAL_RESTART,
            fatal_restart_required=True,
        )

    def _response(
        self,
        request: RemoteFillRequest,
        code: ResultCode,
        message: str = "",
        **kwargs: Any,
    ) -> RemoteFillResponse:
        return RemoteFillResponse(
            operation=operation_kind(request),
            operation_id=request.common.operation_id,
            code=code,
            message=message,
            destination_engine_epoch=self.destination_engine_epoch,
            shared_cache_generation=self.shared_cache_generation,
            **kwargs,
        )
