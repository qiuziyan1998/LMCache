# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and safety tests for the decoder-owned remote-fill state core."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.remote_fill import (
    AbortRequest,
    DestinationNativeState,
    OperationKind,
    PageDisposition,
    PreparedPage,
    ReplyLostError,
    RemoteFillStateCore,
    ResultCode,
    TerminalOutcome,
    TransactionState,
    UnsafePageLifecycleError,
    WindowState,
)


def _open_and_reserve(harness, *, pages=None):
    open_response = harness.client.execute(harness.requests.open())
    reserve = harness.requests.reserve(pages)
    reserve_response = harness.client.execute(reserve)
    return open_response, reserve, reserve_response


def _arm_and_report(harness, reserve, reserve_response) -> None:
    harness.client.execute(harness.requests.arm(reserve, reserve_response))
    harness.client.execute(harness.requests.report(reserve, reserve_response))


def _select_group0_only(harness) -> None:
    negotiation = replace(harness.negotiation, shared_group1=False)
    harness.negotiation = negotiation
    harness.state._negotiation = negotiation


def test_open_conservatively_reports_zero_until_exact_keys_arrive(harness) -> None:
    """Keyless OPEN cannot claim a LocalCPU prefix or create a lease."""

    response = harness.client.execute(harness.requests.open())

    assert response.code is ResultCode.ACCEPTED
    assert response.d_local_end_open == 0
    assert response.remote_session == "decoder-host:19001"
    assert harness.lifecycle.prepare_calls == 0


def test_reservation_key_supports_multiple_chunks_per_group(harness) -> None:
    """One window can reserve four chunks for each of two DSA groups."""

    _, _, response = _open_and_reserve(
        harness,
        pages=harness.requests.pages(4),
    )

    selectors = {
        (descriptor.kv_group, descriptor.chunk_index)
        for descriptor in response.descriptors
    }
    assert selectors == {(group, chunk) for chunk in range(4) for group in (0, 1)}


def test_paired_negotiation_rejects_group0_only_window_before_allocation(
    harness,
) -> None:
    """Codec-valid Group-0 windows still require matching negotiation."""

    harness.client.execute(harness.requests.open())
    response = harness.client.execute(
        harness.requests.reserve(harness.requests.pages(groups=(0,)))
    )

    assert response.code is ResultCode.RESERVATION_REJECTED
    assert harness.lifecycle.prepare_calls == 0


def test_group0_negotiation_rejects_paired_window_before_allocation(harness) -> None:
    """A Group-0-only decoder never allocates an unexpected Group-1 page."""

    _select_group0_only(harness)
    assert (
        harness.client.execute(
            harness.requests.negotiate(shared_group1=False)
        ).code
        is ResultCode.OK
    )
    harness.client.execute(harness.requests.open())
    response = harness.client.execute(harness.requests.reserve())

    assert response.code is ResultCode.RESERVATION_REJECTED
    assert harness.lifecycle.prepare_calls == 0


def test_group0_commit_is_internal_local_and_public_persistent_only(harness) -> None:
    """G0 publication stays distinguishable without weakening LOCAL_FULL."""

    _select_group0_only(harness)
    pages = harness.requests.pages(groups=(0,))
    _, reserve, response = _open_and_reserve(harness, pages=pages)
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(harness.requests.finish())
    status = harness.client.execute(harness.requests.status())
    metrics = harness.service.metrics_snapshot()

    assert finished.transaction_state is TransactionState.GROUP0_LOCAL
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert status.transaction_state is TransactionState.GROUP0_LOCAL
    assert status.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert status.windows[0].state is WindowState.COMMITTED
    assert harness.lifecycle.commit_calls == 1
    assert harness.lifecycle.committed[0][0] == (pages[0].canonical_key,)
    assert harness.lifecycle.released == []
    assert metrics.published_bytes_total == response.descriptors[0].destination_length
    assert metrics.terminal_local_full_total == 0
    assert metrics.terminal_persistent_only_total == 1


def test_group0_partial_tail_commits_with_exact_finish_metadata(harness) -> None:
    """The one-group coverage validator preserves exact partial tails."""

    _select_group0_only(harness)
    page = msgspec.structs.replace(
        harness.requests.pages(groups=(0,))[0],
        chunk_end=512,
        valid_tokens=512,
    )
    _, reserve, response = _open_and_reserve(harness, pages=(page,))
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(
        harness.requests.finish(
            required_store_end=512,
            final_partial_valid_tokens=512,
        )
    )

    assert finished.transaction_state is TransactionState.GROUP0_LOCAL
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY


def test_out_of_order_windows_finalize_deterministically(harness) -> None:
    """Concurrent window arrival order does not affect the final manifest."""

    harness.client.execute(harness.requests.open())
    second = harness.requests.reserve(
        harness.requests.pages(start_chunk=1),
        window_id=1,
    )
    first = harness.requests.reserve(
        harness.requests.pages(start_chunk=0),
        window_id=0,
    )
    second_response = harness.client.execute(second)
    first_response = harness.client.execute(first)
    _arm_and_report(harness, second, second_response)
    _arm_and_report(harness, first, first_response)

    result = harness.client.execute(harness.requests.finish(required_store_end=2048))
    assert result.terminal_outcome is TerminalOutcome.LOCAL_FULL


def test_lost_arm_reply_is_idempotently_retried(harness) -> None:
    """A lost acknowledgement never creates or re-arms a native attempt."""

    _, reserve, response = _open_and_reserve(harness)
    arm = harness.requests.arm(reserve, response, operation_id="arm-op")
    harness.transport.drop_next_reply(OperationKind.ARM_WINDOW)

    try:
        harness.client.execute(arm)
    except ReplyLostError:
        pass
    else:
        raise AssertionError("injected ARM reply was not lost")

    retried = harness.client.execute(arm)
    status = harness.client.execute(harness.requests.status(window_id=0))
    assert retried.code is ResultCode.OK
    assert status.windows[0].native_state is DestinationNativeState.ARMED
    assert (
        status.windows[0].native_transfer_attempt_id
        == response.native_transfer_attempt_id
    )


def test_every_lost_control_reply_is_exactly_replayable(harness) -> None:
    """All eight operations replay the same committed result after reply loss."""

    def execute_after_lost_reply(kind, request):
        harness.transport.drop_next_reply(kind)
        with pytest.raises(ReplyLostError):
            harness.client.execute(request)
        return harness.client.execute(request)

    negotiate = harness.requests.negotiate(operation_id="lost-negotiate")
    assert (
        execute_after_lost_reply(OperationKind.NEGOTIATE, negotiate).code
        is ResultCode.OK
    )
    open_request = harness.requests.open(operation_id="lost-open")
    assert (
        execute_after_lost_reply(OperationKind.OPEN, open_request).code
        is ResultCode.ACCEPTED
    )
    reserve = harness.requests.reserve(operation_id="lost-reserve")
    reserve_response = execute_after_lost_reply(
        OperationKind.RESERVE_WINDOW, reserve
    )
    assert reserve_response.code is ResultCode.OK
    arm = harness.requests.arm(
        reserve, reserve_response, operation_id="lost-arm"
    )
    assert (
        execute_after_lost_reply(OperationKind.ARM_WINDOW, arm).code
        is ResultCode.OK
    )
    report = harness.requests.report(
        reserve, reserve_response, operation_id="lost-report"
    )
    assert (
        execute_after_lost_reply(
            OperationKind.REPORT_TRANSFER_COMPLETE, report
        ).code
        is ResultCode.OK
    )
    status = harness.requests.status(window_id=0, operation_id="lost-status")
    assert (
        execute_after_lost_reply(OperationKind.STATUS, status).code
        is ResultCode.OK
    )
    finish = harness.requests.finish(operation_id="lost-finish")
    finish_response = execute_after_lost_reply(OperationKind.FINISH, finish)
    assert finish_response.terminal_outcome is TerminalOutcome.LOCAL_FULL

    harness.requests.transfer_id = "transfer-abort"
    abort_open = harness.requests.open(operation_id="lost-abort-open")
    assert harness.client.execute(abort_open).code is ResultCode.ACCEPTED
    abort = AbortRequest(
        common=harness.requests.common("lost-abort"),
        reason="qualification",
    )
    abort_response = execute_after_lost_reply(OperationKind.ABORT, abort)
    assert abort_response.code is ResultCode.TERMINAL
    assert abort_response.terminal_outcome is TerminalOutcome.CANCELLED


def test_concurrent_identical_reserve_allocates_once(harness) -> None:
    """The operation cache and allocator transition share one state lock."""

    harness.client.execute(harness.requests.open())
    reserve = harness.requests.reserve(operation_id="reserve-once")
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = tuple(executor.map(harness.client.execute, [reserve] * 8))

    assert harness.lifecycle.prepare_calls == 1
    assert all(
        response.descriptors == responses[0].descriptors for response in responses
    )


def test_unarmed_expiry_releases_and_retry_returns_no_pointer(harness) -> None:
    """Expired unarmed reservations are reclaimable but never replay pointers."""

    _, reserve, response = _open_and_reserve(harness)
    assert response.descriptors
    harness.clock.advance(6.0)
    harness.service.run_maintenance()

    replay = harness.client.execute(reserve)
    status = harness.client.execute(harness.requests.status(window_id=0))
    assert replay.code is ResultCode.RESERVATION_EXPIRED
    assert replay.descriptors == ()
    assert status.windows[0].expired_unarmed
    assert status.windows[0].state is WindowState.RELEASED
    assert len(harness.lifecycle.released) == 1


def test_arm_disables_ordinary_ttl_cleanup(harness) -> None:
    """An armed destination stays allocated beyond its descriptor TTL."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.clock.advance(6.0)
    assert harness.service.run_maintenance() == ()

    status = harness.client.execute(harness.requests.status(window_id=0))
    assert status.windows[0].state is WindowState.ARMED
    assert harness.lifecycle.released == []


def test_descriptor_ttl_rejects_late_arm_without_early_page_reuse(harness) -> None:
    """Descriptor expiry is D-local and distinct from reservation cleanup."""

    _, reserve, response = _open_and_reserve(harness)
    descriptor_deadline = response.descriptors[0].expires_at
    assert descriptor_deadline == harness.clock.now + 2.0

    harness.clock.advance(3.0)
    late_arm = harness.client.execute(harness.requests.arm(reserve, response))
    assert late_arm.code is ResultCode.RESERVATION_EXPIRED
    assert harness.lifecycle.released == []

    harness.clock.advance(3.0)
    harness.service.run_maintenance()
    assert len(harness.lifecycle.released) == 1


def test_missing_armed_completion_requires_paired_restart(harness) -> None:
    """The hard timeout retains pages and exposes fatal restart explicitly."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.clock.advance(21.0)

    assert harness.service.run_maintenance() == ("transfer-1",)
    status = harness.client.execute(harness.requests.status(window_id=0))
    assert status.code is ResultCode.OK
    assert status.transaction_state is TransactionState.FATAL_RESTART
    assert status.fatal_restart_required
    assert status.windows[0].native_state is DestinationNativeState.FATAL_UNKNOWN
    assert harness.lifecycle.released == []
    metrics = harness.service.metrics_snapshot()
    assert metrics.fatal_restarts_total == 1
    assert metrics.terminal_fatal_restart_total == 1
    assert metrics.active_bytes > 0


def test_abort_after_hard_timeout_replays_fatal_without_releasing(harness) -> None:
    """Terminal replay protects FATAL_UNKNOWN destinations from ABORT."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.clock.advance(21.0)
    assert harness.service.run_maintenance() == ("transfer-1",)

    abort = AbortRequest(common=harness.requests.common(), reason="too late")
    result = harness.client.execute(abort)

    assert result.code is ResultCode.TERMINAL
    assert result.transaction_state is TransactionState.FATAL_RESTART
    assert result.fatal_restart_required is True
    assert harness.lifecycle.released == []


def test_completion_requires_arm(harness) -> None:
    """The decoder does not infer native submission from a completion report."""

    _, reserve, response = _open_and_reserve(harness)
    report = harness.requests.report(reserve, response)

    result = harness.client.execute(report)
    assert result.code is ResultCode.WINDOW_NOT_ARMED


def test_successful_report_remains_hidden_until_finish(harness) -> None:
    """Terminal DMA success does not publish ordinary LocalCPU entries."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.client.execute(harness.requests.report(reserve, response))
    status = harness.client.execute(harness.requests.status(window_id=0))

    assert status.windows[0].state is WindowState.READY_HIDDEN
    assert harness.lifecycle.committed == []
    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.LOCAL_FULL
    assert harness.lifecycle.commit_calls == 1


def test_metrics_snapshot_tracks_lifecycle_without_labels(harness) -> None:
    """Metrics expose fixed counters and lock-consistent active gauges."""

    initial = harness.service.metrics_snapshot()
    assert initial.active_transactions == 0
    assert initial.reserved_bytes_total == 0

    _, reserve, response = _open_and_reserve(harness)
    reserved_bytes = sum(
        descriptor.destination_length for descriptor in response.descriptors
    )
    reserved = harness.service.metrics_snapshot()
    assert reserved.active_transactions == 1
    assert reserved.active_windows == 1
    assert reserved.active_bytes == reserved_bytes
    assert reserved.reserved_bytes_total == reserved_bytes

    harness.client.execute(harness.requests.arm(reserve, response))
    harness.client.execute(harness.requests.report(reserve, response))
    assert harness.service.metrics_snapshot().submitted_bytes_total == reserved_bytes

    harness.client.execute(harness.requests.finish())
    finished = harness.service.metrics_snapshot()
    assert finished.active_transactions == 0
    assert finished.active_windows == 0
    assert finished.active_bytes == 0
    assert finished.published_bytes_total == reserved_bytes
    assert finished.terminal_local_full_total == 1


def test_metrics_count_validation_stale_and_transfer_failure(harness) -> None:
    """Low-cardinality failure counters do not retain request identities."""

    harness.service.handle_bytes(b"not-msgpack")
    stale_open = harness.requests.open()
    stale_open = msgspec.structs.replace(
        stale_open,
        common=msgspec.structs.replace(
            stale_open.common,
            destination_engine_epoch=6,
        ),
    )
    assert harness.client.execute(stale_open).code is ResultCode.STALE_ENGINE_EPOCH

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.client.execute(harness.requests.report(reserve, response, return_code=9))
    metrics = harness.service.metrics_snapshot()
    discarded_bytes = sum(
        descriptor.destination_length for descriptor in response.descriptors
    )
    assert metrics.protocol_errors_total == 1
    assert metrics.stale_requests_total == 1
    assert metrics.transfer_failures_total == 1
    assert metrics.discarded_bytes_total == discarded_bytes


def test_metrics_count_capacity_rejection(harness) -> None:
    """Resource exhaustion increments one fixed counter."""

    limits = msgspec.structs.replace(
        harness.state.limits,
        max_active_transactions=1,
    )
    harness.state.limits = limits
    harness.service.limits = limits
    harness.client.limits = limits
    harness.client.execute(harness.requests.open())
    other = type(harness.requests)(transfer_id="transfer-2")

    assert (
        harness.client.execute(other.open(operation_id="other-open")).code
        is ResultCode.RESOURCE_EXHAUSTED
    )
    assert harness.service.metrics_snapshot().capacity_rejections_total == 1


def test_required_byte_capacity_hole_rejects_later_allocations(harness) -> None:
    """A nonretryable required hole abandons direct fill but permits probes."""

    limits = msgspec.structs.replace(
        harness.state.limits,
        max_bytes_per_transaction=1,
    )
    harness.state.limits = limits
    harness.service.limits = limits
    harness.client.limits = limits
    harness.client.execute(harness.requests.open())

    first = harness.client.execute(harness.requests.reserve(window_id=0))
    later = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=1),
            window_id=1,
        )
    )
    probe = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=2),
            window_id=2,
            reserve_missing=False,
        )
    )

    assert first.code is ResultCode.RESOURCE_EXHAUSTED
    assert first.transaction_state is TransactionState.DIRECT_ABANDONED
    assert later.code is ResultCode.TERMINAL
    assert probe.code is ResultCode.RESERVATION_REJECTED
    assert harness.lifecycle.prepare_calls == 1


def test_existing_pages_are_snapshots_without_descriptors_or_pins(harness) -> None:
    """Already-local pages register exact keys without allocation or a lease."""

    pages = harness.requests.pages()
    harness.lifecycle.existing_keys.update(page.canonical_key for page in pages)
    _, _, response = _open_and_reserve(harness, pages=pages)

    assert response.code is ResultCode.OK
    assert response.descriptors == ()
    assert {item.disposition for item in response.page_results} == {
        PageDisposition.EXISTING
    }
    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.LOCAL_FULL
    assert harness.lifecycle.commit_calls == 1
    assert harness.lifecycle.released == []
    assert harness.lifecycle.raw_released == []


def test_mixed_existing_and_absent_pages_describe_only_allocations(harness) -> None:
    """A mixed window transfers only absent pages and rechecks all at FINISH."""

    pages = harness.requests.pages()
    harness.lifecycle.existing_keys.add(pages[0].canonical_key)
    _, reserve, response = _open_and_reserve(harness, pages=pages)

    assert [item.disposition for item in response.page_results] == [
        PageDisposition.EXISTING,
        PageDisposition.ALLOCATED,
    ]
    assert len(response.descriptors) == 1
    assert response.descriptors[0].canonical_key == pages[1].canonical_key
    _arm_and_report(harness, reserve, response)
    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.LOCAL_FULL
    required_keys, handles = harness.lifecycle.committed[0]
    assert required_keys == tuple(page.canonical_key for page in pages)
    assert handles[0] is None
    assert handles[1] is not None


def test_probe_only_missing_page_prevents_local_publication(harness) -> None:
    """A retained authoritative manifest with a hole degrades persistently."""

    harness.client.execute(harness.requests.open())
    reserve = harness.requests.reserve(reserve_missing=False)
    response = harness.client.execute(reserve)

    assert response.code is ResultCode.RESERVATION_REJECTED
    assert response.descriptors == ()
    assert {item.disposition for item in response.page_results} == {
        PageDisposition.MISSING
    }
    assert response.transaction_state is TransactionState.DIRECT_ABANDONED
    later = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=1),
            window_id=1,
            reserve_missing=True,
        )
    )
    assert later.code is ResultCode.TERMINAL
    assert harness.lifecycle.prepare_calls == 1
    # RequestFactory records every constructed manifest; rejected windows are
    # not part of the destination transaction's authoritative final digest.
    harness.requests.window_manifests.pop(1)
    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert harness.lifecycle.commit_calls == 0


def test_missing_manifest_chunk_prevents_commit(harness) -> None:
    """FINISH requires a hole-free aligned two-group manifest from token zero."""

    harness.client.execute(harness.requests.open())
    first = harness.requests.reserve(
        harness.requests.pages(start_chunk=0),
        window_id=0,
    )
    first_response = harness.client.execute(first)
    _arm_and_report(harness, first, first_response)
    third = harness.requests.reserve(
        harness.requests.pages(start_chunk=2),
        window_id=1,
    )
    third_response = harness.client.execute(third)
    _arm_and_report(harness, third, third_response)

    finished = harness.client.execute(harness.requests.finish(required_store_end=3072))
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert harness.lifecycle.commit_calls == 0
    assert len(harness.lifecycle.released) == 2


def test_incomplete_persistence_fails_before_local_commit(harness) -> None:
    """Direct placement never replaces the required persistent copy."""

    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(harness.requests.finish(persistent_common_end=0))
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENCE_FAILED
    assert harness.lifecycle.commit_calls == 0
    assert len(harness.lifecycle.released) == 1
    metrics = harness.service.metrics_snapshot()
    assert metrics.terminal_persistence_failed_total == 1
    assert metrics.discarded_bytes_total > 0


def test_final_commit_rechecks_existing_winners_and_can_fall_back(harness) -> None:
    """Eviction between RESERVE and FINISH yields persistent-only fallback."""

    harness.lifecycle.commit_result = False
    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert harness.lifecycle.commit_calls == 1
    assert len(harness.lifecycle.released) == 1


def test_generic_commit_exception_falls_back_to_persistent_only(harness) -> None:
    """A safe admission failure preserves the durable Mooncake fallback."""

    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)

    def reject_commit(*_args) -> bool:
        raise RuntimeError("injected safe commit rejection")

    harness.lifecycle.commit_pages = reject_commit
    finished = harness.client.execute(harness.requests.finish())

    assert finished.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert finished.fatal_restart_required is False
    assert len(harness.lifecycle.released) == 1


def test_unsafe_commit_exception_requires_paired_restart(harness) -> None:
    """Unknown admission ownership never releases decoder destinations."""

    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)

    def reject_commit(*_args) -> bool:
        raise UnsafePageLifecycleError("injected unsafe admission failure")

    harness.lifecycle.commit_pages = reject_commit
    finished = harness.client.execute(harness.requests.finish())

    assert finished.terminal_outcome is TerminalOutcome.FATAL_RESTART
    assert finished.fatal_restart_required is True
    assert harness.lifecycle.released == []


def test_partial_final_page_uses_authoritative_valid_token_count(harness) -> None:
    """A partial final chunk can commit only with matching FINISH metadata."""

    pages = tuple(
        msgspec.structs.replace(
            page,
            chunk_end=page.chunk_start + 512,
            valid_tokens=512,
        )
        for page in harness.requests.pages()
    )
    _, reserve, response = _open_and_reserve(harness, pages=pages)
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(
        harness.requests.finish(
            required_store_end=512,
            final_partial_valid_tokens=512,
        )
    )
    assert finished.terminal_outcome is TerminalOutcome.LOCAL_FULL


def test_inconsistent_final_partial_metadata_is_rejected(harness) -> None:
    """FINISH cannot relabel a full page as a partial store-plan page."""

    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)

    finished = harness.client.execute(
        harness.requests.finish(final_partial_valid_tokens=1)
    )
    assert finished.code is ResultCode.INVALID_MESSAGE
    assert harness.lifecycle.commit_calls == 0


def test_native_failure_releases_only_after_terminal_report(harness) -> None:
    """A nonzero native return safely abandons direct fill after the call ends."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    assert harness.lifecycle.released == []

    result = harness.client.execute(
        harness.requests.report(reserve, response, return_code=9)
    )
    assert result.code is ResultCode.TRANSFER_FAILED
    assert result.transaction_state is TransactionState.DIRECT_ABANDONED
    assert len(harness.lifecycle.released) == 1


def test_one_failed_window_does_not_turn_fallback_into_cancellation(harness) -> None:
    """Direct abandonment remains eligible for required persistent fallback."""

    harness.client.execute(harness.requests.open())
    first = harness.requests.reserve(
        harness.requests.pages(start_chunk=0),
        window_id=0,
    )
    second = harness.requests.reserve(
        harness.requests.pages(start_chunk=1),
        window_id=1,
    )
    first_response = harness.client.execute(first)
    second_response = harness.client.execute(second)
    harness.client.execute(harness.requests.arm(first, first_response))
    harness.client.execute(harness.requests.arm(second, second_response))
    harness.client.execute(
        harness.requests.report(first, first_response, return_code=1)
    )
    harness.client.execute(harness.requests.report(second, second_response))

    result = harness.client.execute(harness.requests.finish(required_store_end=2048))
    assert result.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY


def test_direct_abandoned_finish_skips_commit_callback(harness) -> None:
    """Persistent fallback cannot accidentally publish failed direct pages."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.client.execute(harness.requests.report(reserve, response, return_code=1))

    result = harness.client.execute(harness.requests.finish())
    assert result.terminal_outcome is TerminalOutcome.PERSISTENT_ONLY
    assert harness.lifecycle.commit_calls == 0


def test_abort_does_not_release_armed_destination_early(harness) -> None:
    """Logical cancellation waits for the original native terminal return."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    abort = AbortRequest(common=harness.requests.common(), reason="client cancelled")

    accepted = harness.client.execute(abort)
    assert accepted.transaction_state is TransactionState.ABORT_REQUESTED
    assert harness.lifecycle.released == []

    harness.client.execute(harness.requests.report(reserve, response))
    status = harness.client.execute(harness.requests.status())
    assert status.transaction_state is TransactionState.CANCELLED
    assert len(harness.lifecycle.released) == 1


def test_same_operation_id_different_payload_is_rejected(harness) -> None:
    """An operation ID cannot be replayed with a changed payload digest."""

    first = harness.requests.open(operation_id="same-op")
    assert harness.client.execute(first).code is ResultCode.ACCEPTED
    changed = msgspec.structs.replace(first, request_id="changed")

    result = harness.client.execute(changed)
    assert result.code is ResultCode.OPERATION_CONFLICT


def test_reopened_transfer_cannot_change_source_identity(harness) -> None:
    """A transfer ID remains bound to the source that opened it."""

    assert harness.client.execute(harness.requests.open()).code is ResultCode.ACCEPTED
    changed = msgspec.structs.replace(
        harness.requests.open(),
        source_engine_id="another-prefiller",
    )

    assert harness.client.execute(changed).code is ResultCode.WINDOW_CONFLICT


def test_control_pages_must_match_negotiated_layout(harness) -> None:
    """Syntactically valid pages cannot change startup layout identity."""

    harness.client.execute(harness.requests.open())
    pages = harness.requests.pages()
    changed = tuple(msgspec.structs.replace(page, layer_count=78) for page in pages)

    result = harness.client.execute(harness.requests.reserve(changed))
    assert result.code is ResultCode.RESERVATION_REJECTED
    assert harness.lifecycle.prepare_calls == 0


def test_invalid_allocator_batch_releases_every_raw_allocation(harness) -> None:
    """Malformed callback output is rolled back without page association."""

    harness.client.execute(harness.requests.open())
    harness.lifecycle.prepared_override = tuple(
        PreparedPage(
            handle=f"raw-{index}",
            destination_ptr=0x100000 + index * 0x10000,
            destination_length=4096,
            reservation_base=0x100000 + index * 0x10000,
            reservation_length=4096,
        )
        for index in range(3)
    )

    response = harness.client.execute(harness.requests.reserve())
    assert response.code is ResultCode.RESERVATION_REJECTED
    assert harness.lifecycle.raw_released[0][0] == (
        "raw-0",
        "raw-1",
        "raw-2",
    )
    later = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=1),
            window_id=1,
        )
    )
    assert response.transaction_state is TransactionState.DIRECT_ABANDONED
    assert later.code is ResultCode.TERMINAL
    assert harness.lifecycle.prepare_calls == 1


def test_incomplete_batch_release_failure_returns_fatal(harness) -> None:
    """A failed rollback cannot be reported as an ordinary reservation miss."""

    harness.client.execute(harness.requests.open())
    pages = harness.requests.pages()
    harness.lifecycle.missing_keys.add(pages[0].canonical_key)
    harness.lifecycle.fail_release = True

    response = harness.client.execute(harness.requests.reserve(pages))
    assert response.code is ResultCode.FATAL_RESTART_REQUIRED
    assert response.fatal_restart_required


def test_allocator_cannot_reuse_an_active_destination_range(harness) -> None:
    """A faulty allocator cannot alias two simultaneously hidden windows."""

    harness.client.execute(harness.requests.open())
    first = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=0),
            window_id=0,
        )
    )
    assert first.code is ResultCode.OK
    harness.lifecycle.next_address = 0x100000

    second = harness.client.execute(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=1),
            window_id=1,
        )
    )
    assert second.code is ResultCode.RESERVATION_REJECTED
    assert len(harness.lifecycle.raw_released) == 1


def test_canonical_key_cannot_be_registered_in_two_windows(harness) -> None:
    """Cross-window manifests cannot alias one immutable cache object."""

    harness.client.execute(harness.requests.open())
    first_pages = harness.requests.pages(start_chunk=0)
    harness.client.execute(harness.requests.reserve(first_pages, window_id=0))
    second_pages = tuple(
        msgspec.structs.replace(
            page,
            canonical_key=first_pages[index].canonical_key,
        )
        for index, page in enumerate(harness.requests.pages(start_chunk=1))
    )

    result = harness.client.execute(harness.requests.reserve(second_pages, window_id=1))
    assert result.code is ResultCode.WINDOW_CONFLICT


def test_final_manifest_digest_must_bind_all_registered_windows(harness) -> None:
    """A changed final extent cannot bypass retained window evidence."""

    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)
    finish = harness.requests.finish()
    finish = msgspec.structs.replace(finish, final_manifest_digest="0" * 64)

    result = harness.client.execute(finish)
    assert result.code is ResultCode.INVALID_DIGEST
    assert harness.lifecycle.commit_calls == 0


def test_stale_epoch_and_cache_generation_are_rejected(harness) -> None:
    """Restart and shared-pool generations fence stale control messages."""

    stale_epoch = harness.requests.open()
    stale_epoch = msgspec.structs.replace(
        stale_epoch,
        common=msgspec.structs.replace(
            stale_epoch.common,
            destination_engine_epoch=6,
        ),
    )
    stale_generation = harness.requests.open()
    stale_generation = msgspec.structs.replace(
        stale_generation,
        common=msgspec.structs.replace(
            stale_generation.common,
            shared_cache_generation=10,
        ),
    )

    assert harness.client.execute(stale_epoch).code is ResultCode.STALE_ENGINE_EPOCH
    assert (
        harness.client.execute(stale_generation).code
        is ResultCode.STALE_CACHE_GENERATION
    )


def test_source_generation_is_stable_across_windows(harness) -> None:
    """A transaction rejects a later window from another source generation."""

    harness.client.execute(harness.requests.open())
    harness.client.execute(harness.requests.reserve(window_id=0, source_generation=3))
    changed = harness.requests.reserve(window_id=1, source_generation=4)

    assert harness.client.execute(changed).code is ResultCode.STALE_SOURCE_GENERATION


def test_terminal_records_expire_but_fatal_records_do_not(harness) -> None:
    """Safe terminal replay is bounded while fatal ownership survives forever."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.client.execute(harness.requests.report(reserve, response))
    finish = harness.requests.finish(operation_id="finish-op")
    harness.client.execute(finish)
    assert harness.client.execute(finish).terminal_outcome is TerminalOutcome.LOCAL_FULL

    harness.clock.advance(11.0)
    harness.service.run_maintenance()
    assert (
        harness.client.execute(harness.requests.status()).code is ResultCode.NOT_FOUND
    )


def test_terminal_pruning_restores_operation_record_capacity(harness) -> None:
    """Safe completed transactions cannot exhaust idempotency storage forever."""

    harness.state.limits = msgspec.structs.replace(
        harness.state.limits,
        max_operation_records=5,
    )
    _, reserve, response = _open_and_reserve(harness)
    _arm_and_report(harness, reserve, response)
    harness.client.execute(harness.requests.finish())
    harness.clock.advance(11.0)
    harness.service.run_maintenance()
    second_factory = type(harness.requests)("transfer-2")

    assert harness.client.execute(second_factory.open()).code is ResultCode.ACCEPTED


def test_terminal_pruning_is_rate_limited(harness, monkeypatch) -> None:
    """Frequent safety refreshes do not repeatedly scan terminal history."""

    prune = Mock(wraps=harness.state._prune_terminal_records_locked)
    monkeypatch.setattr(harness.state, "_prune_terminal_records_locked", prune)

    for _ in range(100):
        harness.service.run_maintenance()
    assert prune.call_count == 0

    harness.clock.advance(1.0)
    for _ in range(100):
        harness.service.run_maintenance()
    assert prune.call_count == 1


def test_terminal_pruning_interval_does_not_exceed_record_ttl(
    harness,
    monkeypatch,
) -> None:
    """A subsecond replay TTL is not extended by pruning throttling."""

    state = RemoteFillStateCore(
        destination_engine_epoch=7,
        shared_cache_generation=11,
        descriptor_verification_key=b"test descriptor verification key",
        negotiation=harness.negotiation,
        page_lifecycle=harness.lifecycle,
        terminal_record_ttl_sec=0.1,
        clock=harness.clock,
    )

    prune = Mock(wraps=state._prune_terminal_records_locked)
    monkeypatch.setattr(state, "_prune_terminal_records_locked", prune)

    harness.clock.advance(0.1)
    state.run_maintenance()

    assert prune.call_count == 1


def test_one_record_capacity_still_allows_arm_report_and_finish(harness) -> None:
    """Idempotency pressure cannot strand an armed native destination."""

    harness.state.limits = msgspec.structs.replace(
        harness.state.limits,
        max_operation_records=1,
    )
    _, reserve, response = _open_and_reserve(harness)
    assert (
        harness.client.execute(harness.requests.arm(reserve, response)).code
        is ResultCode.OK
    )
    assert (
        harness.client.execute(harness.requests.report(reserve, response)).code
        is ResultCode.OK
    )

    finished = harness.client.execute(harness.requests.finish())
    assert finished.terminal_outcome is TerminalOutcome.LOCAL_FULL


def test_fatal_records_survive_terminal_record_ttl(harness) -> None:
    """Fatal armed ownership is never pruned by ordinary maintenance."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.clock.advance(21.0)
    harness.service.run_maintenance()
    harness.clock.advance(100.0)

    assert harness.service.run_maintenance() == ("transfer-1",)
    status = harness.client.execute(harness.requests.status())
    assert status.transaction_state is TransactionState.FATAL_RESTART


def test_fatal_transaction_stops_new_admission(harness) -> None:
    """A paired-restart requirement fails closed for later transactions."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))
    harness.clock.advance(21.0)
    harness.service.run_maintenance()
    second_factory = type(harness.requests)("transfer-2")

    result = harness.client.execute(second_factory.open(operation_id="transfer-2-open"))
    assert result.code is ResultCode.FATAL_RESTART_REQUIRED


def test_expired_window_can_be_reserved_with_new_operation(harness) -> None:
    """TTL permits a fresh allocation but old idempotent replay stays expired."""

    _, original, original_response = _open_and_reserve(harness)
    assert original_response.descriptors
    harness.clock.advance(6.0)
    harness.service.run_maintenance()

    replacement = harness.requests.reserve(window_id=0, operation_id="reserve-new")
    replacement_response = harness.client.execute(replacement)
    stale_replay = harness.client.execute(original)
    assert replacement_response.code is ResultCode.OK
    assert replacement_response.descriptors
    assert stale_replay.code is ResultCode.RESERVATION_EXPIRED
    assert stale_replay.descriptors == ()


def test_reserved_byte_quota_rejects_before_allocation(harness) -> None:
    """State-level byte credit bounds hidden destination ownership."""

    harness.state.limits = msgspec.structs.replace(
        harness.state.limits,
        max_reserved_bytes=1,
    )
    harness.client.execute(harness.requests.open())
    response = harness.client.execute(harness.requests.reserve())

    assert response.code is ResultCode.RESOURCE_EXHAUSTED
    assert harness.lifecycle.prepare_calls == 0


def test_shutdown_releases_unarmed_pages_and_stops_admission(harness) -> None:
    """Clean shutdown drains safe hidden allocations idempotently."""

    _open_and_reserve(harness)
    assert harness.service.shutdown() == ()
    assert harness.service.shutdown() == ()
    assert len(harness.lifecycle.released) == 1
    assert harness.client.execute(harness.requests.status()).code is ResultCode.TERMINAL


def test_shutdown_retains_armed_pages_and_reports_restart(harness) -> None:
    """Shutdown cannot release destinations that native code may still target."""

    _, reserve, response = _open_and_reserve(harness)
    harness.client.execute(harness.requests.arm(reserve, response))

    assert harness.service.shutdown() == ("transfer-1",)
    assert harness.lifecycle.released == []


def test_release_failure_escalates_instead_of_reusing_memory(harness) -> None:
    """Allocator release uncertainty becomes a paired-restart requirement."""

    _open_and_reserve(harness)
    harness.lifecycle.fail_release = True
    harness.clock.advance(6.0)

    assert harness.service.run_maintenance() == ("transfer-1",)
    assert harness.lifecycle.released == []


def test_inflight_window_quota_is_enforced(harness) -> None:
    """Reservations fail before allocation when inflight credit is exhausted."""

    harness.client.execute(harness.requests.open())
    harness.client.execute(
        harness.requests.reserve(harness.requests.pages(start_chunk=0), window_id=0)
    )
    harness.client.execute(
        harness.requests.reserve(harness.requests.pages(start_chunk=1), window_id=1)
    )
    third = harness.client.execute(
        harness.requests.reserve(harness.requests.pages(start_chunk=2), window_id=2)
    )

    assert third.code is ResultCode.RESOURCE_EXHAUSTED


def test_state_requires_positive_engine_epoch_and_limits(harness) -> None:
    """A decoder cannot advertise an unusable epoch or unbounded state limit."""

    kwargs = {
        "shared_cache_generation": 11,
        "descriptor_verification_key": b"secret",
        "negotiation": harness.negotiation,
        "page_lifecycle": harness.lifecycle,
        "clock": harness.clock,
    }
    with pytest.raises(ValueError, match="epoch must be positive"):
        type(harness.state)(destination_engine_epoch=0, **kwargs)
    with pytest.raises(ValueError, match="limits must be positive"):
        type(harness.state)(
            destination_engine_epoch=7,
            limits=msgspec.structs.replace(
                harness.state.limits,
                max_operation_records=0,
            ),
            **kwargs,
        )
