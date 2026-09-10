# SPDX-License-Identifier: Apache-2.0
"""Allocator-free passive Mooncake page-reader behavior."""

# Standard
from types import SimpleNamespace
from typing import Any, Sequence
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.remote_fill.native import NativeExternalPageTransferUnknownError
from lmcache.v1.storage_backend import remote_backend
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.remote_backend import (
    RemoteBackend,
    RemoteExternalPageReader,
)


class _ExternalPageConnector:
    def __init__(self) -> None:
        self.get_calls: list[tuple[Any, ...]] = []
        self.unknown_get = False
        self.close_errors: list[BaseException | None] = []
        self.close_calls = 0
        self.close_started: threading.Event | None = None
        self.close_release: threading.Event | None = None
        self.close_exited: threading.Event | None = None
        self.close_unknown_error: BaseException | None = None
        self.close_teardown_calls = 0

    async def batched_get_external_pages(
        self,
        keys: Sequence[CacheEngineKey],
        buffer_ptrs: list[list[int]],
        buffer_sizes: list[list[int]],
        owners: tuple[Any, ...],
        req_id: str,
    ) -> None:
        self.get_calls.append((list(keys), buffer_ptrs, buffer_sizes, owners, req_id))
        if self.unknown_get:
            terminal = asyncio.get_running_loop().create_future()
            terminal.set_result(None)
            raise NativeExternalPageTransferUnknownError("get", terminal)

    async def close(self) -> None:
        self.close_calls += 1
        try:
            if self.close_started is not None:
                self.close_started.set()
            if self.close_release is not None:
                await asyncio.to_thread(self.close_release.wait)
            if self.close_unknown_error is not None:
                raise RuntimeError("close ownership unknown") from (
                    self.close_unknown_error
                )
            if self.close_errors:
                error = self.close_errors.pop(0)
                if error is not None:
                    raise error
            self.close_teardown_calls += 1
        finally:
            if self.close_exited is not None:
                self.close_exited.set()

    def mark_external_close_unknown(self, error: BaseException) -> None:
        if self.close_unknown_error is None:
            self.close_unknown_error = error


def _config() -> LMCacheEngineConfig:
    return LMCacheEngineConfig.from_defaults(
        remote_url="mooncakestore://127.0.0.1:50051",
        dsa_group1_load_mode="persistent_direct_hbm",
        extra_config={
            "save_only_first_rank": True,
            "remote_enable_mla_worker_id_as0": True,
        },
    )


def _metadata(worker_id: int = 3) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test-model",
        world_size=8,
        local_world_size=8,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.float16,
        kv_shape=(4, 1, 8, 1, 1),
        use_mla=True,
        chunk_size=8,
    )


def _key(worker_id: int = 3) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=worker_id,
        chunk_hash=17,
        dtype=torch.float16,
        kv_group=1,
    )


def _read(reader: RemoteExternalPageReader) -> None:
    reader.batched_get_external_pages(
        [_key()], [[100]], [[16]], (object(),), "request"
    )


def test_reader_keeps_local_cpu_absent_and_normalizes_passive_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ExternalPageConnector()
    create_calls: list[tuple[Any, ...]] = []

    def create_connector(
        url,
        loop,
        local_cpu_backend,
        config,
        metadata,
        external_page_only=False,
    ):
        create_calls.append(
            (
                url,
                loop,
                local_cpu_backend,
                config,
                metadata,
                external_page_only,
            )
        )
        return connector

    monkeypatch.setattr(remote_backend, "CreateConnector", create_connector)
    monkeypatch.setattr(
        remote_backend,
        "CreateSerde",
        lambda *_args, **_kwargs: pytest.fail(
            "external-page reader must not construct serde"
        ),
    )

    reader = RemoteExternalPageReader(_config(), _metadata())
    owner = object()
    try:
        reader.batched_get_external_pages(
            [_key()], [[100]], [[16]], (owner,), "request-1"
        )
    finally:
        reader.close()

    assert len(create_calls) == 1
    assert create_calls[0][2] is None
    assert create_calls[0][5] is True
    get_keys = connector.get_calls[0][0]
    assert [key.worker_id for key in get_keys] == [0]
    assert connector.get_calls[0][-1] == "request-1"

    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        _read(reader)


def test_reader_close_failure_is_fatal_and_never_resubmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ExternalPageConnector()
    connector.close_errors = [RuntimeError("native completion unknown")]
    monkeypatch.setattr(
        remote_backend,
        "CreateConnector",
        lambda *_args, **_kwargs: connector,
    )

    reader = RemoteExternalPageReader(_config(), _metadata())
    with pytest.raises(RuntimeError, match="native completion unknown"):
        reader.close()

    with pytest.raises(RuntimeError, match="state=FATAL"):
        _read(reader)
    with pytest.raises(RuntimeError, match="fatal"):
        reader.close()
    assert connector.close_calls == 1


def test_reader_reuses_one_close_future_and_rejects_io_while_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ExternalPageConnector()
    connector.close_started = threading.Event()
    connector.close_release = threading.Event()
    monkeypatch.setattr(
        remote_backend,
        "CreateConnector",
        lambda *_args, **_kwargs: connector,
    )
    reader = RemoteExternalPageReader(_config(), _metadata())
    errors: list[BaseException] = []

    def close_reader() -> None:
        try:
            reader.close()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=close_reader)
    second = threading.Thread(target=close_reader)
    first.start()
    assert connector.close_started.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="state=CLOSING"):
        _read(reader)
    second.start()
    connector.close_release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert connector.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        _read(reader)


def test_reader_close_timeout_latches_fail_stop_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ExternalPageConnector()
    connector.close_started = threading.Event()
    connector.close_release = threading.Event()
    connector.close_exited = threading.Event()
    monkeypatch.setattr(
        remote_backend,
        "CreateConnector",
        lambda *_args, **_kwargs: connector,
    )
    config = _config()
    config.blocking_timeout_secs = 0.1
    config.remote_fill_native_hard_timeout_ms = 100
    reader = RemoteExternalPageReader(config, _metadata())

    try:
        with pytest.raises(TimeoutError):
            reader.close()
        assert connector.close_started.wait(timeout=1.0)
        assert isinstance(connector.close_unknown_error, TimeoutError)
        assert connector.close_teardown_calls == 0
        with pytest.raises(RuntimeError, match="fatal"):
            reader.close()
    finally:
        connector.close_release.set()
        assert connector.close_exited.wait(timeout=1.0)
        reader._stop_loop()


def test_reader_latches_unknown_direct_get_as_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _ExternalPageConnector()
    connector.unknown_get = True
    monkeypatch.setattr(
        remote_backend,
        "CreateConnector",
        lambda *_args, **_kwargs: connector,
    )
    reader = RemoteExternalPageReader(_config(), _metadata())
    owner = object()

    with pytest.raises(NativeExternalPageTransferUnknownError):
        reader.batched_get_external_pages(
            [_key()], [[100]], [[16]], (owner,), "request"
        )
    with pytest.raises(RuntimeError, match="state=FATAL"):
        _read(reader)
    with pytest.raises(RuntimeError, match="fatal"):
        reader.close()
    assert connector.close_calls == 0


def test_remote_backend_outer_direct_get_deadline_is_bounded() -> None:
    cancelled = threading.Event()

    class _WedgedConnection:
        @staticmethod
        async def batched_get_external_pages(*_args: Any) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    backend = object.__new__(RemoteBackend)
    backend.connection = _WedgedConnection()
    backend.loop = loop
    backend.config = SimpleNamespace(
        blocking_timeout_secs=0.01,
        remote_fill_native_hard_timeout_ms=10,
    )
    backend._mla_worker_id_as0_mode = False
    owner = torch.empty(32, dtype=torch.uint8)
    started = time.monotonic()
    try:
        with pytest.raises(NativeExternalPageTransferUnknownError) as error_info:
            backend.batched_get_external_pages(
                [_key()], [[owner.data_ptr()]], [[32]], (owner,), "request"
            )
        assert time.monotonic() - started < 0.5
        with pytest.raises(NativeExternalPageTransferUnknownError) as repeated_error:
            backend.batched_get_external_pages(
                [_key()], [[owner.data_ptr()]], [[32]], (owner,), "request-2"
            )
        assert repeated_error.value is error_info.value
        error_info.value.terminal_future.cancel()
        assert cancelled.wait(timeout=1.0)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()


def test_remote_backend_missing_connection_does_not_enable_strict_close() -> None:
    backend = object.__new__(RemoteBackend)
    backend.connection = None
    backend._external_page_fatal_error = None
    backend._strict_external_destination_ownership = False

    with pytest.raises(RuntimeError, match="unavailable"):
        backend.batched_get_external_pages([], [], [], (), "request")

    assert not backend.requires_strict_external_close()


def test_remote_backend_rejected_submission_does_not_enable_strict_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_submission(coroutine, _loop) -> None:
        coroutine.close()
        raise RuntimeError("submission rejected")

    monkeypatch.setattr(
        remote_backend.asyncio,
        "run_coroutine_threadsafe",
        reject_submission,
    )
    backend = object.__new__(RemoteBackend)
    backend.connection = _ExternalPageConnector()
    backend.loop = object()
    backend._external_page_fatal_error = None
    backend._strict_external_destination_ownership = False
    backend._mla_worker_id_as0_mode = False

    with pytest.raises(RuntimeError, match="submission rejected"):
        backend.batched_get_external_pages([], [], [], (), "request")

    assert not backend.requires_strict_external_close()


def test_remote_backend_accepted_submission_enables_strict_close_on_failure() -> None:
    class _FailingConnection:
        @staticmethod
        async def batched_get_external_pages(*_args: Any) -> None:
            raise ValueError("known terminal failure")

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    backend = object.__new__(RemoteBackend)
    backend.connection = _FailingConnection()
    backend.loop = loop
    backend.config = SimpleNamespace(
        blocking_timeout_secs=0.1,
        remote_fill_native_hard_timeout_ms=100,
    )
    backend._external_page_fatal_error = None
    backend._strict_external_destination_ownership = False
    backend._mla_worker_id_as0_mode = False
    try:
        with pytest.raises(ValueError, match="known terminal failure"):
            backend.batched_get_external_pages([], [], [], (), "request")
        assert backend.requires_strict_external_close()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()


def test_remote_backend_outer_deadline_uses_wrapped_connector_timeout() -> None:
    wrapped = object.__new__(InstrumentedRemoteConnector)
    wrapped._connector = SimpleNamespace(
        config=SimpleNamespace(transfer_timeout=0.2)
    )
    backend = object.__new__(RemoteBackend)
    backend.connection = wrapped
    backend.config = SimpleNamespace(
        blocking_timeout_secs=0.01,
        remote_fill_native_hard_timeout_ms=10,
    )

    timeout = backend._external_page_outer_timeout_secs()

    assert 0.2 < timeout < 0.3


def test_reader_rejects_nonzero_writer_rank() -> None:
    metadata = _metadata()
    metadata.first_rank = 1

    with pytest.raises(ValueError, match="first_rank == 0"):
        RemoteExternalPageReader(_config(), metadata)
