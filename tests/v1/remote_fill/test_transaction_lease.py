# SPDX-License-Identifier: Apache-2.0
"""Idle owner recovery through the real codec, service and state interfaces."""

# Standard
from typing import Any
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.remote_fill import (
    AbortRequest,
    DestinationNativeState,
    InProcessRemoteFillTransport,
    ProtocolLimits,
    RemoteFillClient,
    RemoteFillService,
    RemoteFillStateCore,
    ResultCode,
    TerminalOutcome,
    TransactionState,
)


@pytest.fixture
def lease_harness(harness: Any, request: pytest.FixtureRequest) -> Any:
    """Use six admission slots and a ten-second progress lease."""
    harness.state = RemoteFillStateCore(
        destination_engine_epoch=7,
        shared_cache_generation=11,
        descriptor_verification_key=b"test-key",
        negotiation=harness.negotiation,
        page_lifecycle=harness.lifecycle,
        limits=ProtocolLimits(max_active_transactions=6),
        transaction_idle_timeout_sec=10,
        reservation_ttl_sec=getattr(request, "param", 5),
        native_hard_timeout_sec=20,
        terminal_record_ttl_sec=30,
        clock=harness.clock,
    )
    harness.client = RemoteFillClient(
        InProcessRemoteFillTransport(RemoteFillService(harness.state))
    )
    return harness


@pytest.mark.parametrize("mode", ["open_only", "reserved_unarmed", "ready_hidden"])
@pytest.mark.parametrize("maintenance", [True, False])
def test_six_orphans_recover_admission_and_bytes(
    lease_harness: Any, mode: str, maintenance: bool
) -> None:
    """The incident's three scenarios recover even on the next OPEN alone."""
    h = lease_harness
    for index in range(6):
        h.requests.transfer_id = f"orphan-{index}"
        assert h.client.execute(h.requests.open()).code is ResultCode.ACCEPTED
        if mode == "open_only":
            continue
        reserve = h.requests.reserve()
        response = h.client.execute(reserve)
        assert response.code is ResultCode.OK
        if mode == "ready_hidden":
            h.client.execute(h.requests.arm(reserve, response))
            h.client.execute(h.requests.report(reserve, response))
    h.requests.transfer_id = "next"
    assert h.client.execute(h.requests.open()).code is ResultCode.RESOURCE_EXHAUSTED
    h.clock.advance(10)
    if maintenance:
        assert h.state.run_maintenance() == ()
        assert h.state.metrics_snapshot().active_transactions == 0
    assert h.client.execute(h.requests.open()).code is ResultCode.ACCEPTED
    metrics = h.state.metrics_snapshot()
    assert metrics.active_transactions == 1
    assert metrics.active_bytes == metrics.active_windows == 0
    assert metrics.expired_transactions_total == 6
    assert len(h.lifecycle.released) == (0 if mode == "open_only" else 6)


def test_long_prefill_renews_only_on_new_progress(lease_harness: Any) -> None:
    """Total duration may exceed the lease many times while windows advance."""
    h = lease_harness
    h.client.execute(h.requests.open())
    for chunk in range(8):
        h.clock.advance(9)
        reserve = h.requests.reserve(
            h.requests.pages(start_chunk=chunk), window_id=chunk
        )
        response = h.client.execute(reserve)
        assert response.code is ResultCode.OK
        h.client.execute(h.requests.arm(reserve, response))
        h.client.execute(h.requests.report(reserve, response))
        assert h.state.run_maintenance() == ()
        assert h.state.metrics_snapshot().active_transactions == 1
    result = h.client.execute(h.requests.finish(required_store_end=8192))
    assert result.terminal_outcome is TerminalOutcome.LOCAL_FULL
    assert h.state.metrics_snapshot().active_transactions == 0
    assert h.state.metrics_snapshot().expired_transactions_total == 0
    h.clock.advance(10)
    h.state.run_maintenance()
    assert not h.lifecycle.released  # Published pages are not lease-owned.


@pytest.mark.parametrize("replay", ["open", "status", "reserve", "report", "invalid"])
def test_polling_or_replay_cannot_keep_orphans_alive(
    lease_harness: Any, replay: str
) -> None:
    h = lease_harness
    opened = h.requests.open()
    h.client.execute(opened)
    reserve = h.requests.reserve()
    response = h.client.execute(reserve)
    h.client.execute(h.requests.arm(reserve, response))
    report = h.requests.report(reserve, response)
    h.client.execute(report)
    for _ in range(3):
        h.clock.advance(3)
        if replay == "open":
            h.client.execute(opened)
        elif replay == "reserve":
            h.client.execute(reserve)
        elif replay == "report":
            h.client.execute(h.requests.report(reserve, response))
        elif replay == "invalid":
            h.client.execute(h.requests.report(reserve, response, completed_bytes=1))
        else:
            h.client.execute(h.requests.status())
    h.clock.advance(1)
    assert h.state.run_maintenance() == ()
    assert h.state.metrics_snapshot().active_transactions == 0
    assert h.client.execute(reserve).descriptors == ()
    for request in (
        opened,
        report,
        h.requests.finish(),
        AbortRequest(common=h.requests.common(), reason="late cancel"),
    ):
        response = h.client.execute(request)
        assert response.terminal_outcome is TerminalOutcome.CANCELLED
    h.state.run_maintenance()
    assert len(h.lifecycle.released) == 1
    assert h.state.metrics_snapshot().terminal_cancelled_total == 1


@pytest.mark.parametrize("report_arrives", [True, False])
def test_expired_lease_keeps_armed_memory_until_evidence(
    lease_harness: Any, report_arrives: bool
) -> None:
    h = lease_harness
    h.client.execute(h.requests.open())
    reserve = h.requests.reserve()
    response = h.client.execute(reserve)
    h.client.execute(h.requests.arm(reserve, response))
    h.clock.advance(10)
    assert h.state.run_maintenance() == ()
    status = h.client.execute(h.requests.status())
    assert status.transaction_state is TransactionState.ABORT_REQUESTED
    assert status.windows[0].native_state is DestinationNativeState.ARMED
    assert not h.lifecycle.released
    assert h.state.metrics_snapshot().active_transactions == 1
    assert h.state.metrics_snapshot().active_bytes > 0
    if report_arrives:
        h.client.execute(h.requests.report(reserve, response))
        assert h.state.metrics_snapshot().active_transactions == 0
        assert h.state.metrics_snapshot().active_bytes == 0
        assert len(h.lifecycle.released) == 1
    else:
        h.clock.advance(10)
        assert h.state.run_maintenance() == (h.requests.transfer_id,)
        h.clock.advance(86400)
        h.state.run_maintenance()
        assert not h.lifecycle.released
        h.requests.transfer_id = "after-fatal"
        assert h.client.execute(h.requests.open()).fatal_restart_required
        assert h.state.shutdown() == ("transfer-1",)


def test_lease_release_failure_is_fatal(lease_harness: Any) -> None:
    h = lease_harness
    h.client.execute(h.requests.open())
    reserve = h.requests.reserve()
    response = h.client.execute(reserve)
    h.client.execute(h.requests.arm(reserve, response))
    h.client.execute(h.requests.report(reserve, response))
    h.lifecycle.fail_release = True
    h.clock.advance(10)
    assert h.state.run_maintenance() == (h.requests.transfer_id,)
    assert h.state.metrics_snapshot().active_bytes > 0


@pytest.mark.parametrize("lease_harness", [30], indirect=True)
def test_expiry_draining_one_window_revokes_other_window_capabilities(
    lease_harness: Any,
) -> None:
    h = lease_harness
    h.client.execute(h.requests.open())
    first = h.requests.reserve()
    first_reply = h.client.execute(first)
    h.client.execute(h.requests.arm(first, first_reply))
    h.clock.advance(8)
    second = h.requests.reserve(h.requests.pages(start_chunk=1), window_id=1)
    second_reply = h.client.execute(second)
    h.clock.advance(10)
    h.state.run_maintenance()
    assert h.client.execute(second).descriptors == ()
    assert (
        h.client.execute(h.requests.arm(second, second_reply)).code
        is ResultCode.RESERVATION_EXPIRED
    )
    assert (
        h.client.execute(
            h.requests.reserve(
                h.requests.pages(start_chunk=2), window_id=2, reserve_missing=False
            )
        ).code
        is ResultCode.TERMINAL
    )
    h.client.execute(h.requests.report(first, first_reply))
    assert h.state.metrics_snapshot().active_transactions == 0
    assert h.state.metrics_snapshot().active_bytes == 0
    assert len(h.lifecycle.released) == 2


def test_restarted_decoder_rejects_old_owner_without_touching_new_state(
    lease_harness: Any,
) -> None:
    h = lease_harness
    h.client.execute(h.requests.open())
    old_reserve = h.requests.reserve()
    h.client.execute(old_reserve)
    assert h.state.shutdown() == ()
    assert h.state.metrics_snapshot().active_transactions == 0
    restarted = RemoteFillStateCore(
        destination_engine_epoch=8,
        shared_cache_generation=12,
        descriptor_verification_key=b"new-incarnation-key",
        negotiation=h.negotiation,
        page_lifecycle=h.lifecycle,
        clock=h.clock,
    )
    client = RemoteFillClient(
        InProcessRemoteFillTransport(RemoteFillService(restarted))
    )
    for request in (
        old_reserve,
        AbortRequest(common=h.requests.common(), reason="old owner"),
    ):
        assert client.execute(request).code is ResultCode.STALE_ENGINE_EPOCH
    assert restarted.metrics_snapshot().active_transactions == 0
    assert len(h.lifecycle.released) == 1


def test_capacity_diagnostics_show_stalled_owners(
    lease_harness: Any, caplog: pytest.LogCaptureFixture
) -> None:
    h = lease_harness
    for index in range(6):
        h.requests.transfer_id = f"owner-{index}"
        h.client.execute(h.requests.open())
    h.clock.advance(4)
    h.requests.transfer_id = "rejected"
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            h.client.execute(h.requests.open())
    records = [
        r.message for r in caplog.records if "active_capacity_exhausted" in r.message
    ]
    assert len(records) == 1
    for index in range(6):
        assert f"owner-{index}" in records[0]
    for field in (
        "source_engine_id",
        "last_progress_at",
        "idle_sec",
        "reserved_bytes",
        "windows",
    ):
        assert field in records[0]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_lease_timeout_must_be_finite_positive(harness: Any, timeout: float) -> None:
    with pytest.raises(ValueError, match="transaction idle timeout"):
        RemoteFillStateCore(
            destination_engine_epoch=7,
            shared_cache_generation=11,
            descriptor_verification_key=b"test-key",
            negotiation=harness.negotiation,
            page_lifecycle=harness.lifecycle,
            transaction_idle_timeout_sec=timeout,
        )
