# SPDX-License-Identifier: Apache-2.0
"""Bounded, pointer-free control protocol for direct remote LMCache fill."""

# Standard
from enum import Enum
from typing import Union

# Third Party
import msgspec


PROTOCOL_VERSION = 1


class OperationKind(str, Enum):
    """Operations supported by the remote-fill control service."""

    NEGOTIATE = "NEGOTIATE"
    OPEN = "OPEN"
    RESERVE_WINDOW = "RESERVE_WINDOW"
    ARM_WINDOW = "ARM_WINDOW"
    REPORT_TRANSFER_COMPLETE = "REPORT_TRANSFER_COMPLETE"
    FINISH = "FINISH"
    ABORT = "ABORT"
    STATUS = "STATUS"


class ResultCode(str, Enum):
    """Stable result codes returned by the control service."""

    OK = "OK"
    ACCEPTED = "ACCEPTED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    NOT_FOUND = "NOT_FOUND"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_DIGEST = "INVALID_DIGEST"
    OPERATION_CONFLICT = "OPERATION_CONFLICT"
    STALE_ENGINE_EPOCH = "STALE_ENGINE_EPOCH"
    STALE_CACHE_GENERATION = "STALE_CACHE_GENERATION"
    STALE_SOURCE_GENERATION = "STALE_SOURCE_GENERATION"
    RESERVATION_REJECTED = "RESERVATION_REJECTED"
    RESERVATION_EXPIRED = "RESERVATION_EXPIRED"
    WINDOW_CONFLICT = "WINDOW_CONFLICT"
    WINDOW_NOT_ARMED = "WINDOW_NOT_ARMED"
    WINDOW_NOT_TERMINAL = "WINDOW_NOT_TERMINAL"
    TRANSFER_FAILED = "TRANSFER_FAILED"
    TERMINAL = "TERMINAL"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FATAL_RESTART_REQUIRED = "FATAL_RESTART_REQUIRED"


class TransactionState(str, Enum):
    """Decoder-owned remote-fill transaction states."""

    ACTIVE = "ACTIVE"
    DIRECT_ABANDONED = "DIRECT_ABANDONED"
    FINALIZING = "FINALIZING"
    ABORT_REQUESTED = "ABORT_REQUESTED"
    LOCAL_FULL = "LOCAL_FULL"
    PERSISTENT_ONLY = "PERSISTENT_ONLY"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    CANCELLED = "CANCELLED"
    FATAL_RESTART = "FATAL_RESTART"


class WindowState(str, Enum):
    """Visibility and lifetime states for a destination window."""

    RESERVED = "RESERVED"
    ARMED = "ARMED"
    READY_HIDDEN = "READY_HIDDEN"
    ABORT_REQUESTED = "ABORT_REQUESTED"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"
    FATAL_RESTART = "FATAL_RESTART"


class DestinationNativeState(str, Enum):
    """Native attempt knowledge available to the decoder.

    There is intentionally no ``SUBMITTED`` value. Submission happens on the
    prefiller and cannot be inferred by the destination.
    """

    NOT_ARMED = "NOT_ARMED"
    ARMED = "ARMED"
    TERMINAL_SUCCESS = "TERMINAL_SUCCESS"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    FATAL_UNKNOWN = "FATAL_UNKNOWN"


class TerminalOutcome(str, Enum):
    """Final externally visible transaction outcomes."""

    LOCAL_FULL = "LOCAL_FULL"
    PERSISTENT_ONLY = "PERSISTENT_ONLY"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    CANCELLED = "CANCELLED"
    FATAL_RESTART = "FATAL_RESTART"


class PageDisposition(str, Enum):
    """Decoder result for one authoritative control page."""

    EXISTING = "EXISTING"
    ALLOCATED = "ALLOCATED"
    MISSING = "MISSING"


class ProtocolLimits(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Hard bounds applied before a request can affect service state."""

    max_rpc_message_bytes: int = 64 * 1024
    max_control_pages_per_window: int = 8
    max_string_bytes: int = 4096
    max_key_bytes: int = 8192
    max_window_bytes: int = 64 * 1024 * 1024 * 1024
    max_active_transactions: int = 1024
    max_inflight_windows_per_transaction: int = 2
    max_windows_per_transaction: int = 4096
    max_reserved_bytes: int = 1024 * 1024 * 1024 * 1024
    max_bytes_per_transaction: int = 1024 * 1024 * 1024 * 1024
    max_operation_records: int = 65536


class OperationIdentity(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Identity and idempotency fields common to every operation."""

    protocol_version: int
    operation_id: str
    operation_sequence: int
    payload_digest: str
    transfer_id: str
    request_attempt: int
    destination_engine_epoch: int
    shared_cache_generation: int


class ControlPage(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Pointer-free identity and allocation requirements for one cache page."""

    canonical_key: str
    kv_group: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    valid_tokens: int
    destination_tp_rank: int
    expected_bytes: int
    layer_count: int
    layout_tag: str


class DestinationPageDescriptor(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Private capability for one nonnecessarily-contiguous destination page."""

    canonical_key: str
    chunk_index: int
    remote_session: str
    destination_ptr: int
    destination_length: int
    reservation_id: str
    window_id: int
    kv_group: int
    transfer_id: str
    request_attempt: int
    destination_dp_rank: int
    destination_tp_rank: int
    destination_engine_epoch: int
    shared_cache_generation: int
    manifest_digest: str
    native_transfer_attempt_id: str
    expires_at: float
    capability_mac: str

    @property
    def length(self) -> int:
        """Return ``destination_length`` for native planner compatibility."""

        return self.destination_length


class RemoteFillRequestBase(
    msgspec.Struct,
    tag_field="operation",
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Base type for tagged remote-fill requests."""

    common: OperationIdentity


class NegotiateRequest(RemoteFillRequestBase, tag="NEGOTIATE"):
    """Validate static source and destination layout compatibility."""

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
    token_hash_algorithm: str
    python_hash_seed: str = ""


class OpenRequest(RemoteFillRequestBase, tag="OPEN"):
    """Open a bounded remote-fill transaction."""

    request_id: str
    source_engine_id: str
    destination_engine_id: str
    destination_dp_rank: int
    required_store_end_hint: int
    planned_window_count_hint: int
    cache_namespace_tag: str
    layout_tag: str
    model_artifact_id: str
    manifest_digest_seed: str


class ReserveWindowRequest(RemoteFillRequestBase, tag="RESERVE_WINDOW"):
    """Reserve hidden destination pages for one incremental prefill window."""

    window_id: int
    source_generation: int
    manifest_digest: str
    control_pages: tuple[ControlPage, ...]
    reserve_missing: bool = True


class ArmWindowRequest(RemoteFillRequestBase, tag="ARM_WINDOW"):
    """Atomically freeze a complete reservation before native submission."""

    window_id: int
    native_transfer_attempt_id: str
    manifest_digest: str
    destination_descriptor_digest: str


class ReportTransferCompleteRequest(
    RemoteFillRequestBase,
    tag="REPORT_TRANSFER_COMPLETE",
):
    """Report the terminal return of the prefiller-owned native operation."""

    window_id: int
    native_transfer_attempt_id: str
    native_return_code: int
    completed_bytes: int
    manifest_digest: str


class FinishRequest(RemoteFillRequestBase, tag="FINISH"):
    """Finalize and, through a callback, atomically publish a transaction."""

    required_store_end: int
    persistent_common_end: int
    final_manifest_digest: str
    final_partial_valid_tokens: int = 0


class AbortRequest(RemoteFillRequestBase, tag="ABORT"):
    """Permanently make a transaction ineligible for publication."""

    reason: str


class StatusRequest(RemoteFillRequestBase, tag="STATUS"):
    """Query transaction and native-attempt state without mutating it."""

    window_id: int = -1


RemoteFillRequest = Union[
    NegotiateRequest,
    OpenRequest,
    ReserveWindowRequest,
    ArmWindowRequest,
    ReportTransferCompleteRequest,
    FinishRequest,
    AbortRequest,
    StatusRequest,
]


class WindowStatus(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Pointer-free status for one window."""

    window_id: int
    state: WindowState
    native_state: DestinationNativeState
    native_transfer_attempt_id: str
    page_count: int
    total_bytes: int
    expired_unarmed: bool
    fatal_restart_required: bool


class PagePreparationStatus(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Pointer-free preparation result aligned with one control page."""

    canonical_key: str
    kv_group: int
    chunk_index: int
    disposition: PageDisposition


class RemoteFillResponse(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    """Uniform bounded response for all protocol operations."""

    operation: OperationKind
    operation_id: str
    code: ResultCode
    message: str = ""
    remote_session: str = ""
    destination_engine_epoch: int = 0
    shared_cache_generation: int = 0
    transaction_state: TransactionState | None = None
    terminal_outcome: TerminalOutcome | None = None
    d_local_end_open: int = 0
    native_transfer_attempt_id: str = ""
    destination_descriptor_digest: str = ""
    descriptors: tuple[DestinationPageDescriptor, ...] = ()
    page_results: tuple[PagePreparationStatus, ...] = ()
    windows: tuple[WindowStatus, ...] = ()
    fatal_restart_required: bool = False


REQUEST_KIND_BY_TYPE: dict[type[RemoteFillRequestBase], OperationKind] = {
    NegotiateRequest: OperationKind.NEGOTIATE,
    OpenRequest: OperationKind.OPEN,
    ReserveWindowRequest: OperationKind.RESERVE_WINDOW,
    ArmWindowRequest: OperationKind.ARM_WINDOW,
    ReportTransferCompleteRequest: OperationKind.REPORT_TRANSFER_COMPLETE,
    FinishRequest: OperationKind.FINISH,
    AbortRequest: OperationKind.ABORT,
    StatusRequest: OperationKind.STATUS,
}


def operation_kind(request: RemoteFillRequest) -> OperationKind:
    """Return the stable operation kind for a tagged request.

    Args:
        request: Decoded remote-fill request.

    Returns:
        The operation enum corresponding to the concrete request type.

    Raises:
        TypeError: If ``request`` is not a supported request type.
    """

    try:
        return REQUEST_KIND_BY_TYPE[type(request)]
    except KeyError as exc:
        raise TypeError("unsupported remote-fill request type") from exc
