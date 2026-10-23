# SPDX-License-Identifier: Apache-2.0
"""Exercise fenced flat-object publication without CUDA or Mooncake hardware."""

# Standard
import ast
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import threading

# Third Party
import pytest
import torch


@dataclass(frozen=True)
class Key:
    chunk: int
    layer_id: int | None = None

    def without_layer(self) -> "Key":
        return replace(self, layer_id=None)


class Object:
    def __init__(self, size: int) -> None:
        self.raw_tensor = torch.full((size,), -1, dtype=torch.int8)
        self.refs = 2  # Allocation and the request's existing lease.

    @property
    def data_ptr(self) -> int:
        return self.raw_tensor.data_ptr()

    def get_size(self) -> int:
        return self.raw_tensor.numel()

    def is_valid(self) -> bool:
        return self.refs > 0

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        self.refs -= 1
        assert self.refs >= 0


class Page(Object):
    pass


class Local:
    use_hot = True

    def __init__(self) -> None:
        self.objects = {}

    def batched_submit_put_task(self, keys: list[Key], objects: list[Object]) -> None:
        for key, obj in zip(keys, objects, strict=True):
            obj.ref_count_up()
            self.objects[key] = obj


class Remote:
    connection = NS(batched_put_external_pages=lambda: None)


class UnknownCompletion(RuntimeError):
    pass


def manager(remote: bool = True) -> tuple[Any, Local]:
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/v1/storage_backend/storage_manager.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if getattr(node, "name", "") == "StorageManager"
    )
    methods = [
        node
        for node in cls.body
        if getattr(node, "name", "")
        in {
            "batched_put_layerwise_objects",
            "_supports_layer_page_backends",
        }
    ]
    module = ast.parse("from __future__ import annotations\nclass Manager: pass")
    module.body[1].body = methods
    namespace = dict(
        torch=torch,
        Future=Future,
        LayerCacheEngineKey=Key,
        LayerPageMemoryObj=Page,
        LocalCPUBackend=Local,
        RemoteBackend=Remote,
        NativeExternalPageTransferUnknownError=UnknownCompletion,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    obj, local = namespace["Manager"](), Local()
    selected = [("LocalCPUBackend", local)]
    if remote:
        selected.append(("RemoteBackend", Remote()))
    obj.get_active_storage_backends = lambda **kwargs: iter(selected)
    return obj, local


def sources(
    layers: int = 3, chunks: int = 2
) -> tuple[list[list[Key]], list[list[Object]]]:
    keys = [[Key(chunk, layer) for chunk in range(chunks)] for layer in range(layers)]
    objects = [
        [Object(4 + layer * 2 + chunk) for chunk in range(chunks)]
        for layer in range(layers)
    ]
    return keys, objects


def test_background_writer_observes_d2h_before_reading_every_layer() -> None:
    obj, local = manager()
    keys, objects = sources()
    flattened = [item for row in objects for item in row]
    release, waiting = threading.Event(), threading.Event()
    host_thread, wait_threads = threading.get_ident(), []
    archived = []

    class Producer:
        def synchronize(self) -> None:
            wait_threads.append(threading.get_ident())
            waiting.set()
            assert release.wait(5), "test did not release the pending D2H"
            for row in objects:
                for item in row:
                    item.raw_tensor.fill_(37)

    event = Producer()
    with ThreadPoolExecutor(max_workers=1) as executor:

        def submit(
            base_keys: list[Key],
            pointers: list[list[int]],
            sizes: list[list[int]],
            owners: tuple[torch.Tensor, ...],
            events: tuple[Producer, ...],
            request_id: str,
        ) -> Future:
            assert base_keys == [Key(0), Key(1)]
            assert pointers == [
                [row[chunk].data_ptr for row in objects] for chunk in range(2)
            ]
            assert sizes == [
                [row[chunk].get_size() for row in objects] for chunk in range(2)
            ]
            assert tuple(tensor.data_ptr() for tensor in owners) == tuple(
                item.data_ptr for item in flattened
            )
            assert events == (event,) and request_id == "prefill"

            def write() -> None:
                for ready in events:
                    ready.synchronize()
                archived.extend(item.raw_tensor.clone() for item in flattened)

            return executor.submit(write)

        obj.batched_put_external_pages = submit
        try:
            (future,) = obj.batched_put_layerwise_objects(
                keys, objects, producer_events=(event,), req_id="prefill"
            )
            assert waiting.wait(5)
            assert not future.done() and not archived
            assert all(item.refs == 3 for item in flattened)
            assert len(local.objects) == 6
            # Eviction and request retirement cannot free a pending writer's
            # source, even after the caller's allocation reference was consumed.
            for item in local.objects.values():
                item.ref_count_down()
            local.objects.clear()
            for item in flattened:
                item.ref_count_down()
                assert item.refs == 1
        finally:
            release.set()
        future.result(timeout=5)
    assert len(wait_threads) == 1 and wait_threads[0] != host_thread
    assert all(torch.all(tensor == 37) for tensor in archived)
    assert all(item.refs == 0 for item in flattened)


@pytest.mark.parametrize("unknown", [False, True])
def test_remote_failure_propagates_and_unknown_completion_keeps_sources(
    unknown: bool,
) -> None:
    obj, _ = manager()
    keys, objects = sources()
    remote = Future()
    obj.batched_put_external_pages = lambda *args: remote
    (future,) = obj.batched_put_layerwise_objects(
        keys, objects, producer_events=(object(),)
    )
    error = UnknownCompletion("unknown") if unknown else RuntimeError("failed")
    remote.set_exception(error)
    with pytest.raises(type(error), match=str(error)):
        future.result()
    flattened = tuple(item for row in objects for item in row)
    assert all(item.refs == (3 if unknown else 2) for item in flattened)
    if unknown:
        assert error.layerwise_source_pages == flattened


def test_submission_failure_does_not_consume_the_callers_allocation() -> None:
    obj, _ = manager()
    keys, objects = sources()

    def fail(*args: Any) -> None:
        raise RuntimeError("submit failed")

    obj.batched_put_external_pages = fail
    with pytest.raises(RuntimeError, match="submit failed"):
        obj.batched_put_layerwise_objects(keys, objects, producer_events=(object(),))
    assert all(item.refs == 3 for row in objects for item in row)


def test_cpu_only_put_installs_original_layer_keys_without_a_host_wait() -> None:
    obj, local = manager(remote=False)
    keys, objects = sources()

    class Event:
        def synchronize(self) -> None:
            pytest.fail("publication must not wait on the host")

    assert (
        obj.batched_put_layerwise_objects(keys, objects, producer_events=(Event(),))
        == []
    )
    assert list(local.objects) == [key for row in keys for key in row]
    assert all(item.refs == 2 for row in objects for item in row)


@pytest.mark.parametrize("invalid", ["events", "objects", "keys", "layer"])
def test_invalid_input_is_rejected_before_local_publication(invalid: str) -> None:
    obj, local = manager()
    keys, objects = sources()
    events = (object(),)
    if invalid == "events":
        events = ()
    elif invalid == "objects":
        objects[1][0] = objects[0][0]
    elif invalid == "keys":
        keys[1][0] = Key(9, 1)
    else:
        keys[1][0] = Key(0, 2)
    with pytest.raises(ValueError):
        obj.batched_put_layerwise_objects(keys, objects, producer_events=events)
    assert not local.objects
