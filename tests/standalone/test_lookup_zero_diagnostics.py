# SPDX-License-Identifier: Apache-2.0
"""Zero-lookup evidence uses existing results and never adds storage probes."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import json

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1 import serving_perf
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupBusyTimeout
from lmcache.v1.lookup_client.storage_lookup_client import StorageLookupClient
from lmcache.v1.lookup_diagnostics import (
    collect_lookup_failure,
    lookup_failure_pending,
    page_key_fingerprint,
    record_lookup_failure,
)
from lmcache.v1.rpc.storage_client import StorageFuture
from lmcache.v1.rpc.storage_protocol import (
    LookupReply,
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend

# Local
import test_cpu_cache_runtime as cpu_runtime

setup = cpu_runtime.setup


def records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage().split("] ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("[LMCACHE_LOOKUP_ZERO]")
    ]


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(serving_perf, "_MODE", "0")
    clients = []

    def make(
        status: str | tuple[str, ...] = "COMPLETED",
        *,
        hit: bool = False,
        async_loading: bool = True,
        timeout_ms: int = 1,
    ) -> tuple[StorageLookupClient, Mock]:
        transport = Mock()

        def request(method: str, payload: dict, **fields: Any) -> StorageFuture:
            identity = OperationIdentity(
                "cache",
                "epoch",
                "scheduler",
                16 + transport.request.call_count,
                **fields,
            )
            work = StorageRequest(
                identity, method, payload, content_digest(payload), b"s"
            )
            future = StorageFuture(work)
            outcome = (
                status[min(transport.request.call_count, len(status)) - 1]
                if isinstance(status, tuple)
                else status
            )
            if outcome != "PENDING":
                reply = LookupReply(
                    4 if hit else 0,
                    "ticket" if hit else "",
                    1,
                    content_digest(payload),
                    (0,) if hit else (),
                    (4,) if hit else (),
                    (7,) if hit else (),
                    payload["request_configs"],
                    {"backend_reason": "page_unavailable"} if not hit else {},
                )
                future.set_result(
                    StorageResult(
                        identity,
                        outcome,
                        msgspec.to_builtins(reply),
                        error="not ready" if outcome == "FAILED_SAFE" else None,
                    )
                )
            return future

        def cancel(identity: OperationIdentity) -> Future:
            future = Future()
            future.set_result(StorageResult(identity, "ACKNOWLEDGED"))
            return future

        transport.request.side_effect = request
        transport.cancel.side_effect = cancel
        database = SimpleNamespace(
            chunk_hash_type=int,
            process_tokens=lambda tokens, **kw: [(0, 4, 7)] if tokens else [],
        )
        client = StorageLookupClient(
            LMCacheEngineConfig.from_defaults(
                pd_role="receiver",
                enable_async_loading=async_loading,
                lookup_timeout_ms=timeout_ms,
                blocking_timeout_secs=0,
            ),
            SimpleNamespace(use_mla=True, mla_cache_tp_replicated=True),
            storage_client=transport,
            token_database=database,
        )
        clients.append(client)
        return client, transport

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize(
    "status,reason",
    [
        ("COMPLETED", "no_complete_prefix"),
        ("CANCELLED_NOT_STARTED", "storage_cancelled_not_started"),
        ("FAILED_SAFE", "storage_failed_safe"),
    ],
)
@pytest.mark.parametrize("async_loading", [False, True])
def test_terminal_zero_logs_once_without_perf_logs(
    make_client: Any,
    caplog: pytest.LogCaptureFixture,
    status: str,
    reason: str,
    async_loading: bool,
) -> None:
    client, transport = make_client(status, async_loading=async_loading)
    client.lookup([1, 2, 3, 4], "request")
    for _ in range(20):
        assert client.lookup_cache("request") == 0
    [record] = records(caplog)
    assert record["reason"] == reason
    assert record["status"] == status
    assert record["operation_id"] == 17
    assert record["req_id"] == "request"
    assert record["query_tokens"] == 4
    assert record["elapsed_ms"] >= 0
    assert record["error"] == ("not ready" if status == "FAILED_SAFE" else None)
    if status == "COMPLETED":
        assert record["diagnostics"]["backend_reason"] == "page_unavailable"
        assert client.peek_ticket("request") is not None
        transport.cancel.assert_not_called()
    else:
        transport.cancel.assert_called_once()
    transport.request.assert_called_once()


@pytest.mark.parametrize("async_loading", [False, True])
def test_timeout_logs_once_and_preserves_cached_miss_and_cancellation(
    make_client: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    async_loading: bool,
) -> None:
    client, transport = make_client("PENDING", async_loading=async_loading)
    client.lookup([1, 2, 3, 4], "request")
    if async_loading:
        # Advance only the lookup clock; no sleep or extra service request.
        from lmcache.v1.lookup_client import storage_lookup_client as module

        now = module.time.monotonic()
        monkeypatch.setattr(module.time, "monotonic", lambda: now + 1)
        assert client.lookup_cache("request") is None
    for _ in range(20):
        assert client.lookup_cache("request") == 0
    [record] = records(caplog)
    assert record["reason"] == "timeout"
    assert record["status"] == "PENDING"
    transport.request.assert_called_once()
    transport.cancel.assert_called_once()


def test_empty_query_reports_local_cause(
    make_client: Any, caplog: pytest.LogCaptureFixture
) -> None:
    client, transport = make_client()
    assert client.lookup([], "request") == 0
    for _ in range(20):
        assert client.lookup_cache("request") == 0
    [record] = records(caplog)
    assert record["status"] == "NOT_SUBMITTED"
    assert record["reason"] == "no_query_chunks"
    assert "operation_id" not in record
    transport.request.assert_not_called()


@pytest.mark.parametrize("async_loading", [False, True])
def test_busy_retries_only_after_cancel_ack_and_backoff(
    make_client: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    async_loading: bool,
) -> None:
    from lmcache.v1.lookup_client import storage_lookup_client as module

    clock = Mock(return_value=10.0)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=clock, sleep=module.time.sleep)
    )
    client, transport = make_client(
        ("BUSY", "COMPLETED"), hit=True, async_loading=async_loading, timeout_ms=1000
    )
    cancelled, ack = Future(), Future()
    transport.cancel.side_effect = lambda _: cancelled
    transport.acknowledge_terminal.return_value = ack
    configs = {"tags": [1]}
    assert client.lookup([1, 2, 3, 4], "r", configs) is None
    assert client.lookup_cache("r") is None
    configs["tags"].append(2)
    clock.return_value += 0.2
    for _ in range(100):
        assert client.lookup_cache("r") is None
    transport.cancel.assert_called_once()
    old = transport.cancel.call_args.args[0]
    proof = {"terminal_result_digest": b"proof"}
    cancelled.set_result(StorageResult(old, "CANCELLED_NOT_STARTED", proof))
    for _ in range(100):
        assert client.lookup([7], "r") is None
    transport.acknowledge_terminal.assert_called_once_with(old, proof)
    transport.request.assert_called_once()
    ack.set_result(StorageResult(old, "ACKNOWLEDGED"))
    # The replacement must use the first plan, not the caller's changed data.
    assert client.lookup_cache("r") is None
    assert client.lookup_cache("r") == 4
    ticket = client.peek_ticket("r")
    assert ticket.identity.operation_id > old.operation_id
    assert ticket.identity.request_attempt != old.request_attempt
    assert ticket.reply.request_configs == {"tags": [1]}
    assert transport.request.call_count == 2
    assert not records(caplog)


@pytest.mark.parametrize(
    "phase", ["busy", "ack_pending", "retry_inflight", "not_submitted"]
)
def test_busy_deadline_is_from_first_submission(
    make_client: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    phase: str,
) -> None:
    from lmcache.v1.lookup_client import storage_lookup_client as module

    clock = Mock(return_value=10.0)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=clock, sleep=module.time.sleep)
    )
    client, transport = make_client(
        ("BUSY", "PENDING") if phase == "retry_inflight" else "BUSY", timeout_ms=1000
    )
    if phase == "not_submitted":
        transport.request.side_effect = BufferError("transport full")
    cancel = Future()
    if phase == "ack_pending":
        transport.cancel.side_effect = lambda _: cancel
    assert client.lookup(range(4), "r") is None
    for _ in range(100):
        assert client.lookup_cache("r") is None
    transport.request.assert_called_once()
    clock.return_value += 0.7
    for _ in range(10):
        assert client.lookup_cache("r") is None
    attempts = transport.request.call_count
    assert attempts == (1 if phase == "ack_pending" else 2)
    clock.return_value += 0.31
    for _ in range(5):
        with pytest.raises(LookupBusyTimeout):
            client.lookup_cache("r")
    assert client.peek_ticket("r") is None and not records(caplog)
    assert transport.request.call_count == attempts
    if phase == "ack_pending":
        cancel.set_result(
            StorageResult(transport.cancel.call_args.args[0], "ACKNOWLEDGED")
        )
    client.cleanup_lookup("r")
    for _ in range(10):
        if client.lookup_cache("r") == -1:
            break
    assert client.lookup_cache("r") == -1


@pytest.mark.parametrize("action", ["clear_lookup_status", "cleanup_lookup", "close"])
def test_discarded_busy_request_cannot_retry_after_ack(
    make_client: Any, action: str
) -> None:
    client, transport = make_client(("COMPLETED", "BUSY"), hit=True, timeout_ms=1000)
    assert client.lookup(range(4), "r") is None
    assert client.lookup_cache("r") == 4
    prior = client.peek_ticket("r")
    client.clear_lookup_status("r")  # This ticket now belongs to the worker.
    assert client.lookup(range(4), "r") is None
    assert client.lookup_cache("r") is None
    # Poll until cleanup has started, but do not advance past the retry delay.
    assert client.lookup_cache("r") is None
    transport.cancel.assert_called_once()
    assert transport.cancel.call_args.args[0] != prior.identity
    getattr(client, action)(*(() if action == "close" else ("r",)))
    if action == "close":
        assert not client.is_healthy()
    else:
        assert client.lookup_cache("r") == -1
        client.cleanup_lookup("r")
    assert transport.request.call_count == 2


def test_hits_remain_silent(make_client: Any, caplog: pytest.LogCaptureFixture) -> None:
    client, transport = make_client(hit=True)
    client.lookup([1, 2, 3, 4], "request")
    assert client.lookup_cache("request") == 4
    assert not records(caplog)
    transport.request.assert_called_once()


@pytest.mark.parametrize("results", [[0, 1, 1, 1], [1, 0, 1, 1], [-7, 1, 1, 1], []])
def test_runtime_reports_exact_first_remote_page_without_an_extra_query(
    setup: Any,
    results: list[int],
) -> None:
    owner, runtime, work, bootstrap, _ = setup
    owner.config.dsa_group1_load_mode = "persistent_direct_hbm"
    connector = MooncakestoreConnector.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_group_layer_counts = (2, 3)
    connector.store = SimpleNamespace(batch_is_exist=Mock(return_value=results))
    remote = RemoteBackend.__new__(RemoteBackend)
    remote.connection = connector
    remote._mla_worker_id_as0_mode = True
    owner.storage_manager.storage_backends["RemoteBackend"] = remote
    bootstrap()
    result, _ = cpu_runtime.lookup(work)
    assert result.status == "COMPLETED", result.error
    assert result.value.matched_tokens == 0
    evidence = result.value.diagnostics
    index = 1 if results and results[0] == 1 else 0
    assert evidence["exists_status"] == (results[index] if results else None)
    assert evidence["kv_group"] == index
    assert evidence["page_index"] == index
    [args] = connector.store.batch_is_exist.call_args_list
    assert evidence["first_unavailable_page_fp"] == page_key_fingerprint(
        args.args[0][index]
    )
    assert not lookup_failure_pending()


@pytest.mark.parametrize(
    "connection,reason",
    [
        (None, "remote_connection_unavailable"),
        (SimpleNamespace(), "remote_page_lookup_unsupported"),
        (
            SimpleNamespace(
                batched_contains_layer_pages=Mock(side_effect=OSError("offline"))
            ),
            "remote_page_lookup_exception",
        ),
    ],
)
def test_swallowed_backend_errors_remain_distinguishable(
    connection: Any, reason: str
) -> None:
    backend = RemoteBackend.__new__(RemoteBackend)
    backend.connection = connection
    backend._mla_worker_id_as0_mode = False
    with collect_lookup_failure() as evidence:
        assert backend.batched_contains_layer_pages([]) == 0
    assert evidence["backend_reason"] == reason


def test_failure_scope_does_not_leak_between_threads_or_lookups() -> None:
    barrier = Barrier(2)

    def record(name: str) -> dict:
        with collect_lookup_failure() as evidence:
            barrier.wait(2)
            record_lookup_failure(name)
            record_lookup_failure("later failure must not replace the first")
        assert not lookup_failure_pending()
        return evidence

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(record, ["a", "b"])) == [
            {"backend_reason": "a"},
            {"backend_reason": "b"},
        ]
    with pytest.raises(RuntimeError), collect_lookup_failure():
        raise RuntimeError("lookup failure")
    assert not lookup_failure_pending()


def test_diagnostic_field_is_optional_on_old_lookup_replies() -> None:
    reply = LookupReply(0, "", 1, b"x" * 32, (), (), (), {})
    old = msgspec.to_builtins(reply)
    assert "diagnostics" not in old
    assert msgspec.convert(old, type=LookupReply).diagnostics == {}
