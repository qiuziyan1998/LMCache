# SPDX-License-Identifier: Apache-2.0
"""Ownership of serialized batches before and after remote submission."""

# Standard
from collections.abc import Iterator
from concurrent.futures import Future
from unittest.mock import Mock
import asyncio
import inspect

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.naive_serde.naive_serde import NaiveSerializer
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


@pytest.fixture
def remote_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[RemoteBackend]:
    """Build a real backend and instrumented connector without remote I/O."""
    connector = Mock(spec=RemoteConnector)
    connector.support_batched_put.return_value = True
    connector.requires_put_completion.return_value = True
    connection = InstrumentedRemoteConnector(connector)
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.remote_backend.CreateConnector",
        Mock(return_value=connection),
    )
    loop = asyncio.new_event_loop()
    try:
        backend = RemoteBackend(
            LMCacheEngineConfig(
                remote_url="lm://localhost:12345", remote_serde="naive"
            ),
            LMCacheMetadata(
                model_name="test",
                world_size=1,
                local_world_size=1,
                worker_id=0,
                local_worker_id=0,
                kv_dtype=torch.float16,
                kv_shape=(1, 2, 256, 1, 8),
            ),
            loop,
            local_cpu_backend=None,
            dst_device="cpu",
        )
        assert isinstance(backend.serializer, NaiveSerializer)
        yield backend
    finally:
        if not loop.is_closed():
            loop.run_until_complete(connection.close())
            loop.close()


@pytest.fixture
def memory_batch() -> Iterator[tuple[list[CacheEngineKey], list[MemoryObj]]]:
    """Allocate caller-owned CPU objects with real reference counting."""
    allocator = TensorMemoryAllocator(torch.empty(16384, dtype=torch.uint8))
    memory_objs = allocator.batched_allocate(torch.Size([8]), torch.float16, 3)
    assert memory_objs is not None
    keys = [CacheEngineKey("test", 1, 0, i, torch.float16) for i in range(3)]
    try:
        yield keys, list(memory_objs)
    finally:
        for memory_obj in memory_objs:
            while memory_obj.get_ref_count() > 0:
                memory_obj.ref_count_down()


def test_closed_loop_releases_serialized_batch_and_closes_coroutine(
    remote_backend: RemoteBackend,
    memory_batch: tuple[list[CacheEngineKey], list[MemoryObj]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduling failure must preserve only the caller's references."""
    keys, memory_objs = memory_batch
    submit = Mock(wraps=asyncio.run_coroutine_threadsafe)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    callback = Mock()
    remote_backend.loop.close()

    with pytest.raises(RuntimeError, match="Event loop is closed"):
        remote_backend.batched_submit_put_task(
            keys, memory_objs, on_complete_callback=callback
        )

    submit.assert_called_once()
    coroutine = submit.call_args.args[0]
    try:
        assert [obj.get_ref_count() for obj in memory_objs] == [1, 1, 1]
        assert all(obj.is_valid() for obj in memory_objs)
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        callback.assert_not_called()
    finally:
        coroutine.close()


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_partial_serialization_releases_owned_prefix(
    remote_backend: RemoteBackend,
    memory_batch: tuple[list[CacheEngineKey], list[MemoryObj]],
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    """Release successful serializer outputs as well as temporary input refs."""
    keys, memory_objs = memory_batch
    serialize = remote_backend.serializer.serialize
    failure = RuntimeError("serialization failed")

    def fail_serialization(memory_obj: MemoryObj) -> MemoryObj:
        if memory_obj is memory_objs[fail_at]:
            raise failure
        return serialize(memory_obj)

    monkeypatch.setattr(remote_backend.serializer, "serialize", fail_serialization)
    submit = Mock(wraps=asyncio.run_coroutine_threadsafe)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    callback = Mock()

    with pytest.raises(RuntimeError, match="serialization failed") as exc_info:
        remote_backend.batched_submit_put_task(
            keys, memory_objs, on_complete_callback=callback
        )

    assert exc_info.value is failure
    assert [obj.get_ref_count() for obj in memory_objs] == [1, 1, 1]
    assert all(obj.is_valid() for obj in memory_objs)
    submit.assert_not_called()
    callback.assert_not_called()


@pytest.mark.parametrize("write_fails", [False, True])
def test_scheduled_future_leaves_cleanup_to_instrumented_connector(
    remote_backend: RemoteBackend,
    memory_batch: tuple[list[CacheEngineKey], list[MemoryObj]],
    write_fails: bool,
) -> None:
    """Successful handoff transfers ownership even if the remote write fails."""
    keys, memory_objs = memory_batch
    connection = remote_backend.connection
    assert isinstance(connection, InstrumentedRemoteConnector)
    connector = connection.getWrappedConnector()
    assert isinstance(connector, Mock)

    async def put(actual_keys: list[CacheEngineKey], objs: list[MemoryObj]) -> None:
        assert actual_keys == keys
        assert objs == memory_objs
        assert [obj.get_ref_count() for obj in objs] == [2, 2, 2]
        if write_fails:
            raise RuntimeError("remote write failed")

    connector.batched_put.side_effect = put
    callback = Mock()
    futures = remote_backend.batched_submit_put_task(
        keys, memory_objs, on_complete_callback=callback
    )

    assert futures is not None and len(futures) == 1
    future = futures[0]
    assert isinstance(future, Future)
    assert not future.done()
    assert [obj.get_ref_count() for obj in memory_objs] == [2, 2, 2]
    connector.batched_put.assert_not_awaited()
    callback.assert_not_called()

    completion = asyncio.wrap_future(future, loop=remote_backend.loop)
    if write_fails:
        with pytest.raises(RuntimeError, match="remote write failed"):
            remote_backend.loop.run_until_complete(completion)
    else:
        assert remote_backend.loop.run_until_complete(completion) is None
    assert future.done()
    connector.batched_put.assert_awaited_once_with(keys, memory_objs)
    assert [obj.get_ref_count() for obj in memory_objs] == [1, 1, 1]
    assert all(obj.is_valid() for obj in memory_objs)
    assert [call.args[0] for call in callback.call_args_list] == keys
