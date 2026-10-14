# SPDX-License-Identifier: Apache-2.0
"""Ordering tests for asynchronous scheduler/worker lookup cleanup."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupResponseMsg,
    LookupCompleteMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
    LMCacheAsyncLookupServer,
)


def _server(state: str) -> LMCacheAsyncLookupServer:
    server = object.__new__(LMCacheAsyncLookupServer)
    server._cleanup_lock = threading.Lock()
    server._cleanup_state = {"request": state}
    server.lmcache_engine = MagicMock()
    server.lmcache_engine.storage_manager.loop.call_soon_threadsafe.side_effect = (
        lambda callback, *args: callback(*args)
    )
    server.worker_id = 3
    server.push_socket = MagicMock()
    return server


def test_cleanup_before_response_waits_for_terminal_lookup() -> None:
    server = _server("pending")

    server._request_cleanup("request")

    assert server._cleanup_state == {"request": "cleanup_requested"}
    server.lmcache_engine.cleanup_memory_objs.assert_not_called()

    server.send_response_to_scheduler("request", 1024)

    assert server._cleanup_state == {}
    server.lmcache_engine.cleanup_memory_objs.assert_called_once_with("request")
    assert server.push_socket.send.call_count == 2
    sent = [
        msgspec.msgpack.decode(
            call.args[0], type=LookupResponseMsg | LookupCleanupResponseMsg
        )
        for call in server.push_socket.send.call_args_list
    ]
    assert isinstance(sent[0], LookupResponseMsg)
    assert isinstance(sent[1], LookupCleanupResponseMsg)


def test_off_loop_response_and_cleanup_share_storage_loop_socket_owner() -> None:
    server = _server("pending")
    loop = server.lmcache_engine.storage_manager.loop
    loop.call_soon_threadsafe.side_effect = None

    server.send_response_to_scheduler("request", 1024)

    server.push_socket.send.assert_not_called()
    assert server._cleanup_state == {"request": "pending"}
    callback, *args = loop.call_soon_threadsafe.call_args.args

    # FIFO delivers cleanup after lookup submission but before the marshalled
    # terminal response executes. It must remain deferred until that response.
    server._request_cleanup("request")
    assert server._cleanup_state == {"request": "cleanup_requested"}
    callback(*args)

    assert server._cleanup_state == {}
    server.lmcache_engine.cleanup_memory_objs.assert_called_once_with("request")
    sent = [
        msgspec.msgpack.decode(
            call.args[0], type=LookupResponseMsg | LookupCleanupResponseMsg
        )
        for call in server.push_socket.send.call_args_list
    ]
    assert isinstance(sent[0], LookupResponseMsg)
    assert isinstance(sent[1], LookupCleanupResponseMsg)


def test_synchronous_lookup_setup_failure_sends_terminal_miss() -> None:
    server = _server("pending")
    server.running = True
    server.lmcache_engine.async_lookup_and_prefetch.side_effect = RuntimeError(
        "setup failed"
    )
    payload = msgspec.msgpack.encode(
        LookupRequestMsg(
            lookup_id="request",
            hashes=[],
            offsets=[],
            request_configs=None,
        )
    )

    class _OneMessageSocket:
        def recv(self, copy: bool = False) -> bytes:
            server.running = False
            return payload

    server.pull_socket = _OneMessageSocket()
    server.process_requests_from_scheduler()

    server.push_socket.send.assert_called_once()
    response = msgspec.msgpack.decode(
        server.push_socket.send.call_args.args[0], type=LookupResponseMsg
    )
    assert response.lookup_id == "request"
    assert response.num_hit_tokens == 0
    assert server._cleanup_state == {}


def test_cleanup_after_response_releases_immediately() -> None:
    server = _server("pending")

    server.send_response_to_scheduler("request", 1024)
    assert server._cleanup_state == {}

    server._request_cleanup("request")

    assert server._cleanup_state == {}
    server.lmcache_engine.cleanup_memory_objs.assert_called_once_with("request")
    assert server.push_socket.send.call_count == 2


def test_cancelled_lookup_id_is_not_reused_before_all_responses() -> None:
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.RLock()
    client.aborted_lookups = {"request"}
    client.reqs_status = {}
    client.first_lookup_time = {}

    assert client.lookup_cache("request") is None
    assert client.reqs_status == {}


def test_timeout_can_queue_cleanup_without_reentrant_lock_deadlock() -> None:
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.RLock()
    client.aborted_lookups = set()
    client.reqs_status = {"request": None}
    client.res_for_each_worker = {}
    client.first_lookup_time = {"request": time.time() - 1}
    client.config = SimpleNamespace(lookup_timeout_ms=0)
    client.lookup_backoff_time = 0
    client.world_size = 1
    client.push_sockets = [MagicMock()]

    assert client.lookup_cache("request") == 0
    assert client.reqs_status == {}
    assert client.aborted_lookups == {"request"}
    client.push_sockets[0].send.assert_called_once()


def test_all_cleanup_acks_clear_partial_old_generation_state() -> None:
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.RLock()
    client.running = True
    client.world_size = 1
    client.aborted_lookups = {"request"}
    client.cleanup_res_for_each_worker = {}
    client.res_for_each_worker = {"request": [512]}
    client.reqs_status = {"request": None}
    client.first_lookup_time = {"request": time.time()}
    payload = msgspec.msgpack.encode(
        LookupCleanupResponseMsg(lookup_id="request", worker_id=0)
    )

    class _OneMessageSocket:
        def recv(self, copy: bool = False) -> bytes:
            client.running = False
            return payload

    client.pull_socket = _OneMessageSocket()
    client.process_responses_from_workers()

    assert client.aborted_lookups == set()
    assert client.cleanup_res_for_each_worker == {}
    assert client.res_for_each_worker == {}
    assert client.reqs_status == {}
    assert client.first_lookup_time == {}


def _prefetch_client():
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.RLock()
    client.aborted_lookups = set()
    client.reqs_status = {}
    client.first_lookup_time = {}
    client.prefetch_timeouts = set()
    client.lookup_complete_workers = {}
    client.prefill_cpu_prefetch = True
    client.cleanup_res_for_each_worker = {}
    client.res_for_each_worker = {}
    client.config = SimpleNamespace(
        enable_async_loading=True,
        enable_prefill_cpu_prefetch=True,
        pd_role="sender",
        use_layerwise=True,
        enable_shared_cpu_cache=True,
        dsa_two_groups=True,
        enable_remote_lmcache_store=True,
        lookup_timeout_ms=1000,
    )
    client.lookup_backoff_time = 0.01
    client.world_size = 1
    client.push_sockets = [MagicMock()]
    return client


def test_prefetch_bounds_pending_and_ready_jobs_without_sleep(monkeypatch):
    client = _prefetch_client()
    monkeypatch.setattr(
        time, "sleep", lambda _: (_ for _ in ()).throw(AssertionError("sleep"))
    )
    assert client.lookup_cache("first") == -1
    assert client.lookup_cache("second") is None
    assert "second" not in client.first_lookup_time
    assert client.lookup_cache("first") is None
    client.reqs_status["first"] = 1024
    assert client.lookup_cache("second") is None
    client.clear_lookup_status("first")
    assert client.lookup_cache("second") == -1


def test_prefetch_timeout_waits_for_native_cleanup_before_miss():
    client = _prefetch_client()
    assert client.lookup_cache("request") == -1
    client.first_lookup_time["request"] = time.time() - 2
    assert client.lookup_cache("request") is None
    assert client.lookup_cache("other") is None
    assert client.lookup_cache("request") is None
    client.running = True

    class Ack:
        def recv(self, copy=False):
            client.running = False
            return msgspec.msgpack.encode(
                LookupCleanupResponseMsg(lookup_id="request", worker_id=0)
            )

    client.pull_socket = Ack()
    client.process_responses_from_workers()
    assert client.lookup_cache("request") == 0
    assert client.lookup_cache("other") is None
    client.clear_lookup_status("request")
    assert client.lookup_cache("other") == -1


def test_abort_after_timeout_does_not_resurrect_miss():
    client = _prefetch_client()
    client.lookup_cache("request")
    client.first_lookup_time["request"] = time.time() - 2
    assert client.lookup_cache("request") is None
    client.cleanup_lookup("request")
    assert client.prefetch_timeouts == set()


def _deliver(client, message):
    client.running = True

    class Socket:
        def recv(self, copy=False):
            client.running = False
            return msgspec.msgpack.encode(message)

    client.pull_socket = Socket()
    client.process_responses_from_workers()


def test_payload_preparation_outlives_lookup_deadline_without_recompute():
    client = _prefetch_client()
    client.lookup_cache("request")
    _deliver(client, LookupCompleteMsg(lookup_id="request", worker_id=0))
    client.config.lookup_timeout_ms = -1
    assert client.lookup_cache("request") is None
    assert not client.aborted_lookups
    client.push_sockets[0].send.assert_not_called()
    assert client.lookup_cache("next") is None
    _deliver(client, LookupResponseMsg(lookup_id="request", num_hit_tokens=1024))
    assert client.lookup_cache("request") == 1024


def test_lookup_deadline_requires_all_workers_to_finish_lookup():
    client = _prefetch_client()
    client.world_size = 2
    client.push_sockets.append(MagicMock())
    client.lookup_cache("request")
    _deliver(client, LookupCompleteMsg(lookup_id="request", worker_id=0))
    _deliver(client, LookupCompleteMsg(lookup_id="request", worker_id=0))
    assert "request" in client.first_lookup_time
    client.first_lookup_time["request"] = time.time() - 2
    assert client.lookup_cache("request") is None
    assert client.aborted_lookups == {"request"}
    _deliver(client, LookupCompleteMsg(lookup_id="request", worker_id=1))
    assert client.lookup_complete_workers == {}


def test_preparation_progress_does_not_finish_worker_cleanup():
    server = _server("cleanup_requested")
    server.send_lookup_complete("request")
    assert server._cleanup_state == {"request": "cleanup_requested"}
    server.lmcache_engine.cleanup_memory_objs.assert_not_called()
    msg = msgspec.msgpack.decode(
        server.push_socket.send.call_args.args[0], type=LookupCompleteMsg
    )
    assert msg.worker_id == 3


@pytest.mark.parametrize("chunk_hash", [17, bytes(range(32))])
@pytest.mark.parametrize(
    "configs",
    [
        None,
        {
            "lmcache.remote_fill_result": {
                "outcome": "LOCAL_FULL",
                "required_store_end": 1024,
                "destination_engine_epoch": 9,
            }
        },
    ],
)
def test_async_request_wire_preserves_supported_hash_and_config(chunk_hash, configs):
    client = _prefetch_client()
    client.token_database = SimpleNamespace(
        process_tokens=lambda *a, **kw: iter([(0, 1024, chunk_hash)])
    )
    client.lookup([1], "request", configs)
    payload = client.push_sockets[0].send.call_args.args[0]
    server = _server("pending")
    server.running = True

    class Socket:
        def recv(self, copy=False):
            server.running = False
            return payload

    server.pull_socket = Socket()
    server.process_requests_from_scheduler()
    server.lmcache_engine.async_lookup_and_prefetch.assert_called_once_with(
        lookup_id="request",
        hashes=[chunk_hash],
        offsets=[1024],
        pin=True,
        request_configs=configs,
    )
