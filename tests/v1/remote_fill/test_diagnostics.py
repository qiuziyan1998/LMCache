# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded RemoteFill diagnostic records."""

# Standard
import json

# First Party
from lmcache.v1.remote_fill_diagnostics import (
    REMOTE_FILL_DIAGNOSTIC_MARKER,
    log_remote_fill_diagnostic,
    log_remote_fill_validation_failure,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def warning(self, template: str, marker: str, payload: str) -> None:
        self.calls.append(("warning", marker, payload))

    def error(self, template: str, marker: str, payload: str) -> None:
        self.calls.append(("error", marker, payload))

    def critical(self, template: str, marker: str, payload: str) -> None:
        self.calls.append(("critical", marker, payload))


def test_diagnostic_record_is_bounded_structured_and_single_line() -> None:
    logger = _RecordingLogger()
    error = RuntimeError("bad\nvalue source_ptr=123456 0x1234abcd " + "x" * 1000)

    log_remote_fill_diagnostic(
        logger,
        event="remote_fill_materialization_failure",
        code="RF-D-004",
        stage="rank0_materialization",
        action="RECOMPUTE",
        req_id="req-1",
        reason="retained plan mismatch",
        error=error,
        severity="warning",
    )

    assert len(logger.calls) == 1
    severity, marker, encoded = logger.calls[0]
    assert severity == "warning"
    assert marker == REMOTE_FILL_DIAGNOSTIC_MARKER
    assert "\n" not in encoded
    fields = json.loads(encoded)
    assert fields["code"] == "RF-D-004"
    assert fields["diagnostic_name"] == "decoder_materialization_failure"
    assert fields["event"] == "remote_fill_materialization_failure"
    assert fields["action"] == "RECOMPUTE"
    assert fields["error_type"] == "RuntimeError"
    assert fields["error"].endswith("...")
    assert len(fields["error"]) == 512
    assert "123456" not in fields["error"]
    assert "1234abcd" not in fields["error"]


def test_fatal_record_states_memory_safety_and_restart_action() -> None:
    logger = _RecordingLogger()

    log_remote_fill_diagnostic(
        logger,
        event="remote_fill_fatal_restart",
        code="RF-D-900",
        stage="armed_native_lifecycle",
        action="PAIRED_RESTART_REQUIRED",
        transfer_id="transfer-1",
        memory_safety_uncertain=True,
        severity="critical",
    )

    severity, _, encoded = logger.calls[0]
    fields = json.loads(encoded)
    assert severity == "critical"
    assert fields["memory_safety_uncertain"] is True
    assert fields["action"] == "PAIRED_RESTART_REQUIRED"


def test_former_validation_helper_remains_import_compatible() -> None:
    logger = _RecordingLogger()

    log_remote_fill_validation_failure(
        logger,
        code="RF-P-004",
        stage="producer_admission",
        action="PERSISTENT_ONLY",
    )

    _, marker, encoded = logger.calls[0]
    fields = json.loads(encoded)
    assert marker == REMOTE_FILL_DIAGNOSTIC_MARKER
    assert fields["event"] == "remote_fill_diagnostic"
    assert fields["diagnostic_name"] == "producer_persistent_fallback"
