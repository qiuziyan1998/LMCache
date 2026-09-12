# SPDX-License-Identifier: Apache-2.0
"""Checkpoint control messages. These contain no device pointers or owners."""

# Standard
from dataclasses import dataclass, field
from typing import Any
import time

LOCAL_CHECKPOINT_CONFIG = "lmcache.local_checkpoint_generation"


@dataclass(frozen=True)
class CaptureSpec:
    """Capture [resident_start, end); base is the aligned store boundary."""

    req_id: str
    generation: int
    base: int
    end: int
    resident_start: int
    blocks: tuple[tuple[int, ...], ...]
    request_configs: dict[str, Any] | None = None
    prefix_end: int = 0


@dataclass(frozen=True)
class SealSpec:
    """Bind captured KV to accepted tokens after older outputs are consumed."""

    req_id: str
    generation: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointResult:
    """Writer result; only READY authorizes generated-prefix lookup."""

    req_id: str
    generation: int
    status: str
    end: int = 0
    reason: str = ""
    timings_ms: dict[str, float] | None = None


def choose_checkpoint_end(
    history_length: int, capture: CaptureSpec, captured_end: int | None = None
) -> int:
    """Limit a decode checkpoint to accepted/computed, captured positions.

    Only decode victims are admitted to CaptureSpec. Their final sampled token
    has not been computed; speculative slots above that token are never stored.
    """
    end = min(history_length - 1, capture.end, captured_end or capture.end)
    return end if end > max(capture.base, capture.prefix_end) else 0


@dataclass
class PendingCheckpoint:
    """Scheduler-owned state for one preemption generation."""

    capture: CaptureSpec
    status: str = "capturing"
    end: int = 0
    started_at: float = field(default_factory=time.monotonic)
    cancel_pending: bool = False
    captured_end: int = 0
    restore_retries: int = 0

    def retry_shorter(self, chunk_size: int, failed_end: int | None = None) -> None:
        """Permit one strictly shorter local restore, then use ordinary recovery."""
        end = self.end if failed_end is None else min(self.end, failed_end)
        shorter = (end - 1) // chunk_size * chunk_size
        if not self.restore_retries and shorter > max(
            self.capture.base, self.capture.prefix_end
        ):
            self.end = shorter
            self.status = "ready"
        else:
            self.status = "failed"
        self.restore_retries += 1

    def expire(self, now: float, timeout: float) -> bool:
        """Bound scheduler waiting even if the worker never acknowledges capture."""
        if self.status in ("ready", "failed") or now - self.started_at < timeout:
            return False
        self.status = "failed"
        self.cancel_pending = True
        return True

    def accept(self, result: CheckpointResult) -> bool:
        """Return whether a matching reply changed this attempt's state."""
        if (
            result.req_id != self.capture.req_id
            or result.generation != self.capture.generation
            or self.status in ("failed", "ready")
        ):
            return False
        if result.status == "failed":
            self.status = "failed"
        elif result.status == "captured" and self.status == "capturing":
            self.status = (
                "captured"
                if max(self.capture.base, self.capture.prefix_end)
                < result.end
                <= self.capture.end
                else "failed"
            )
            self.captured_end = result.end
            self.cancel_pending = self.status == "failed"
        elif result.status == "ready" and self.status == "persisting":
            if (
                result.end != self.end
                or not self.capture.base < result.end <= self.capture.end
            ):
                self.status = "failed"
            else:
                self.status = "ready"
        else:
            return False
        return True
