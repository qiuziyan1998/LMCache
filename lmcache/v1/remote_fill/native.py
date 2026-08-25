# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
from dataclasses import dataclass
from typing import Any


DIRECT_PUSH_H0_QUALIFICATION_V1 = "mooncake-sync-write-visible-v1"


@dataclass(frozen=True)
class NativeDirectPushActivation:
    """Compatibility contract and ARM proof for one native submission."""

    # Kept for constructor compatibility; runtime activation is controlled by
    # enable_remote_lmcache_store and the request-scoped ARM acknowledgement.
    h0_qualification: str
    native_transfer_attempt_id: str
    arm_acknowledged: bool


@dataclass(frozen=True)
class DirectPushPageSource:
    """P-local registered source runs for one canonical cache page."""

    canonical_key: str
    kv_group: int
    source_ptrs: tuple[int, ...]
    source_lengths: tuple[int, ...]


@dataclass(frozen=True)
class DirectPushSourcePlan:
    """P-local page runs and ownership fences for one direct push."""

    pages: tuple[DirectPushPageSource, ...]
    owners: tuple[Any, ...]
    producer_events: tuple[Any, ...]


@dataclass(frozen=True)
class PreparedDirectPushSource:
    """Producer-fenced and registered source state prepared before ARM."""

    source_plan: DirectPushSourcePlan
    source_event_wait_ms: float
    source_fences_ready_monotonic: float
    source_registration_ms: float


@dataclass(frozen=True)
class NativeDirectPushResult:
    """Terminal aggregate result of one native direct push."""

    native_transfer_attempt_id: str
    return_code: int
    vector_count: int
    transferred_bytes: int
    elapsed_ms: float
    source_event_wait_ms: float = 0.0
    source_fences_ready_monotonic: float = 0.0
    source_registration_ms: float = 0.0
    native_slot_wait_ms: float = 0.0
    native_started_monotonic: float = 0.0
    native_ended_monotonic: float = 0.0


class NativeDirectPushTerminalError(RuntimeError):
    """A native direct push returned a known nonzero terminal status."""

    def __init__(self, result: NativeDirectPushResult) -> None:
        super().__init__(
            "Mooncake native direct push failed: "
            f"attempt={result.native_transfer_attempt_id} "
            f"return_code={result.return_code}"
        )
        self.result = result


class NativeDirectPushPreSubmitError(RuntimeError):
    """Producer fencing or source registration failed before native submission."""


class NativeExternalPageTransferUnknownError(RuntimeError):
    """A persistent external-page DMA did not terminate by its hard deadline."""

    def __init__(self, operation: str, terminal_future: asyncio.Future[Any]) -> None:
        super().__init__(
            f"Mooncake external-page {operation} has unknown native completion"
        )
        self.operation = operation
        self.terminal_future = terminal_future


class NativeDirectPushAmbiguousError(RuntimeError):
    """A native direct push did not reach a terminal state by its deadline."""

    def __init__(
        self,
        native_transfer_attempt_id: str,
        terminal_future: asyncio.Future[NativeDirectPushResult],
    ) -> None:
        super().__init__(
            "Mooncake native direct push has ambiguous completion: "
            f"attempt={native_transfer_attempt_id}"
        )
        self.native_transfer_attempt_id = native_transfer_attempt_id
        self.terminal_future = terminal_future

    async def wait_for_terminal(self) -> NativeDirectPushResult:
        """Wait for the original native call without resubmitting it."""
        return await asyncio.shield(self.terminal_future)
