# SPDX-License-Identifier: Apache-2.0
"""CPU ordering tests; fake events cannot qualify actual NPU stream behavior."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import gc

import pytest

from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle


class Fence:
    def __init__(self, ready=False):
        self.ready = ready
        self.stream = None
        self.fail = False

    def record(self, stream):
        self.stream = stream

    def query(self):
        if self.fail:
            raise RuntimeError("query failed")
        return self.ready

    def synchronize(self):
        if self.fail:
            raise RuntimeError("sync failed")
        self.ready = True


def fixture():
    retired, events, errors = [], [], []
    source = SimpleNamespace(
        binding_token=object(),
        owner_lease=LocalLeaseHandle(
            lease_id="lease",
            mapping=object(),
            retire=lambda *args: retired.append(args),
        ),
    )

    def factory():
        fence = Fence()
        events.append(fence)
        return fence

    return NativeSourceLifetime(factory, errors.append), source, retired, events, errors


def test_no_per_step_events_or_duplicate_stream_wrappers_and_gc_off_cleanup():
    lifetime, source, retired, events, _ = fixture()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(1000):
            lifetime.note(source, SimpleNamespace(npu_stream=123, device_index=0))
        assert not events
        lifetime.retire_source(source)
        assert len(events) == 1
        source.owner_lease.release("never_imported")
        assert not retired
        events[0].ready = True
        lifetime.drain()
        assert retired == [("lease", "fenced")]
        assert lifetime.sources_drained()
    finally:
        if enabled:
            gc.enable()


def test_every_participating_stream_must_complete():
    lifetime, source, retired, events, _ = fixture()
    lifetime.note(source, SimpleNamespace(npu_stream=1))
    lifetime.note(source, SimpleNamespace(npu_stream=2))
    lifetime.retire_source(source)
    source.owner_lease.release("never_imported")
    events[0].ready = True
    lifetime.drain()
    assert not retired
    events[1].ready = True
    lifetime.drain()
    assert len(retired) == 1


def test_event_query_failure_retains_owner_and_refuses_shutdown():
    lifetime, source, retired, events, errors = fixture()
    lifetime.note(source, SimpleNamespace(npu_stream=1))
    lifetime.retire_source(source)
    source.owner_lease.release("never_imported")
    events[0].fail = True
    with pytest.raises(RuntimeError):
        lifetime.drain()
    with pytest.raises(RuntimeError):
        lifetime.close()
    assert not retired and errors


def test_partial_event_record_is_not_an_empty_completed_fence_set():
    lifetime, source, retired, _, _ = fixture()
    started, finish = Event(), Event()

    class BlockingFence(Fence):
        def record(self, stream):
            started.set()
            assert finish.wait(5)
            super().record(stream)

    lifetime._event_factory = BlockingFence
    lifetime.note(source, SimpleNamespace(npu_stream=1))
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(lifetime.retire_source, source)
        assert started.wait(5)
        source.owner_lease.release("never_imported")
        lifetime.drain()
        assert not retired
        finish.set()
        future.result(5)
    lifetime.close()
    assert len(retired) == 1
