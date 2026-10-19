# SPDX-License-Identifier: Apache-2.0
"""Independent CPU failure injection against production engine cleanup methods."""

from types import SimpleNamespace as NS
import ast
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import gc
import weakref

import pytest
import torch
import msgspec

from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.gpu_connector.sparse import build_compact_prepared_sparse_source
from lmcache.v1.rpc.storage_lifetime import NativeSourceLifetime
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle
from lmcache.v1.cpu_cache import StorageBusy
from lmcache.v1.rpc.storage_protocol import SourceGeometry, StorageResult, pack_uints

from test_storage_registry import registry, request, control, ack
from test_storage_import import fixture as import_fixture, imported


def test_close_after_process_layout_bind_failure_unregisters_early_resources():
    closed = []
    engine = NS(
        storage_process=NS(timeout_s=1, close=lambda: closed.append(True)),
        shared_cpu_cache_mapping=object(),
    )
    # initialize_worker_storage has not created importer/lifetime if native
    # reader.bind_layout or the service's final layout agreement fails.
    LMCacheEngine.close(engine)
    assert closed == [True]
    assert engine.shared_cpu_cache_mapping is None


def test_close_with_unknown_native_completion_keeps_context_registered():
    closed = []

    def unknown():
        raise RuntimeError("unknown native completion")

    engine = NS(
        storage_process=NS(timeout_s=1, close=lambda: closed.append(True)),
        native_source_lifetime=NS(close=unknown),
        shared_cpu_cache_mapping=object(),
    )
    with pytest.raises(RuntimeError, match="unknown native"):
        LMCacheEngine.close(engine)
    assert not closed
    assert engine.shared_cpu_cache_mapping is not None


def test_sparse_ack_bound_cannot_reexport_freed_source():
    reg = registry(max_operations=4)
    freed, works = [], []
    for operation in (3, 5, 7, 9, 11, 13, 15, 17):
        work = request(operation)
        try:
            result = reg.handle(
                work,
                lambda _, identity=work.identity: reg.add_lease(
                    identity,
                    (object(),),
                    participants=("tp0",),
                    release=lambda objects: freed.extend(objects),
                ),
            )
        except StorageBusy:
            break
        reg.handle(
            control(work, "retire", {"lease_id": result.value, "proof": "fenced"}),
            lambda _: None,
        )
        assert ack(reg, work, result).status == "ACKNOWLEDGED"
        works.append(work)
        state = reg.snapshot()
        assert len(state["acknowledged_ranges"]["tp0"]) <= 4
        assert state["operations"] <= 4
    assert reg.snapshot()["leases"] == 0
    assert len(freed) == len(works)
    for work in works:
        # A full sparse-proof budget may retain an ACKed descriptor rather than
        # drop safety evidence; WORK must not expose its already-freed pointers.
        with pytest.raises(ValueError, match="acknowledged"):
            reg.handle(work, lambda _: pytest.fail("Old work reallocated"))


@pytest.mark.parametrize("page_first", [False, True])
def test_import_capacity_order_preserves_per_layer_planes_and_optional_fast_path(
    page_first,
):
    importer, reply, client = import_fixture()
    original = reply.groups[0]
    if page_first:
        manifest = msgspec.structs.replace(
            original,
            flat_extents=pack_uints([8, 12]),
            flat_logical_sizes=pack_uints([8, 12]),
            flat_token_capacities=pack_uints([2, 3], 32),
        )
        expected = ((4, 2), (4, 3))
    else:
        manifest = msgspec.structs.replace(
            original,
            geometries=(
                SourceGeometry(
                    1,
                    (0, 4),
                    (4, 4),
                    ((1, 2),) * 2,
                    ("bfloat16",) * 2,
                    (((0, 4),),) * 2,
                ),
            ),
            page_chunk_indices=pack_uints([1], 32),
            flat_extents=pack_uints([16, 16]),
            flat_logical_sizes=pack_uints([16, 16]),
            flat_token_capacities=pack_uints([4, 4], 32),
        )
        expected = ((4, 1), (4, 1))
    source = imported(importer, msgspec.structs.replace(reply, groups=(manifest,)))[0]
    assert source.total_tokens == 5 and source.chunk_token_counts == (4, 1)
    assert source.physical_chunk_token_counts == expected
    assert (source.compact_transfer_metadata is not None) == page_first
    source.owner_lease.release("fenced")
    client.calls[0][1].set_result(StorageResult(reply.identity, "RETIRED"))
    assert importer.sources_drained()


@pytest.mark.parametrize("unknown_completion", [False, True])
def test_gc_off_generator_close_retains_compact_metadata_until_final_fence(
    unknown_completion,
):
    """Actual compact builder/lifetime retain metadata outside a closed generator."""
    retired, fatal = [], []
    stream = NS(device_index=0, done=False)

    class Event:
        def record(self, value):
            self.stream = value

        def query(self):
            if unknown_completion:
                raise RuntimeError("unknown native completion")
            return self.stream.done

        def synchronize(self):
            assert not unknown_completion
            self.stream.done = True

    lifetime = NativeSourceLifetime(Event, fatal.append)
    source = build_compact_prepared_sparse_source(
        (torch.tensor([123], dtype=torch.int64),),
        num_layers=1,
        total_tokens=1,
        chunk_token_counts=(1,),
        chunk_size=4,
        physical_chunk_token_counts=((4,),),
        owner_lease=LocalLeaseHandle(
            lease_id="source",
            mapping=object(),
            retire=lambda *args: retired.append(args),
        ),
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        operand = torch.tensor([4, -1], dtype=torch.int64)
        weak_operand = weakref.ref(operand)
        source.compact_transfer_metadata["queued"] = (operand,)
        del operand

        def prepared_generator(value):
            cache = value.compact_transfer_metadata
            assert cache
            yield

        retriever = prepared_generator(source)
        next(retriever)
        lifetime.note(source, stream)
        if unknown_completion:
            with pytest.raises(RuntimeError, match="unknown native"):
                lifetime.retire_source(source)
        else:
            lifetime.retire_source(source)
        source.owner_lease.release("fenced")
        # This is the actual _dense_source failure cleanup shape: retaining a
        # closed generator alone cannot protect its cleared local variables.
        retriever.close()
        del retriever, source
        assert weak_operand() is not None and not retired
        if unknown_completion:
            with pytest.raises(RuntimeError, match="completion is unknown"):
                lifetime.close()
            assert weak_operand() is not None and fatal and not retired
        else:
            stream.done = True
            lifetime.drain()
            assert weak_operand() is None and len(retired) == 1
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("join_on_current", [False, True])
def test_async_checkpoint_c8_copy_does_not_join_another_threads_warm_step(
    join_on_current,
):
    """Execute the actual native launch planner; CPU events model stream ordering."""
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/npu_connector/npu_connectors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_c8_indexer_bootstrap"
    )

    class Stream:
        device_index = 0

        def __init__(self):
            self.pending = False
            self.waits = []

        def wait_stream(self, other):
            pass

        def wait_event(self, event):
            self.waits.append(event)

    cold, warm, joined = Stream(), Stream(), Stream()
    if join_on_current:
        warm = cold
    current = [cold]
    launches, retired = [], []

    class Event:
        def record(self, stream):
            self.stream = stream

        def query(self):
            return not self.stream.pending and all(
                event.query() for event in self.stream.waits
            )

        def synchronize(self):
            self.stream.pending = False

    class Vector:
        def __init__(self, length=0):
            self.length = length

        def numel(self):
            return self.length

        def record_stream(self, stream):
            pass

        def __getitem__(self, item):
            return self

    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous

    def launch(*args, **kw):
        launches.append(current[0])
        current[0].pending = True

    namespace = {
        "torch": NS(Tensor=Vector, npu=NS(current_stream=lambda: current[0])),
        "Optional": Optional,
        "_SparseDestinationPlan": object,
        "lmc_ops": NS(indexer_c8_transfer_prepared=launch),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    connector = NS(
        load_stream_list=[joined],
        _active_sparse_load_join=NS(compute_stream=warm, used_stream_indices=set()),
        _validate_sparse_fixed_chunk_coverage=lambda *args: None,
        _dense_direct_dummy_metadata_tensor=Vector,
        _stream_context_or_null=stream_context,
    )
    source = NS(
        binding_token=object(),
        owner_lease=LocalLeaseHandle(
            lease_id="source",
            mapping=object(),
            retire=lambda *args: retired.append(joined.pending),
        ),
    )
    lifetime = NativeSourceLifetime(Event, lambda reason: pytest.fail(reason))
    lifetime.note(source, cold)
    namespace["_run_c8_indexer_bootstrap"](
        connector, NS(states=[None]), Vector(2), 0, Vector(5), 5, 4, False
    )
    if join_on_current:
        assert launches == [joined]
        assert connector._active_sparse_load_join.used_stream_indices == {0}
        # The existing warm join records the transfer stream's completion and
        # inserts its reverse dependency on the owning compute stream.
        joined_done = Event()
        joined_done.record(joined)
        cold.wait_event(joined_done)
        lifetime.retire_source(source)
        source.owner_lease.release("fenced")
        assert not retired
        joined.pending = False
        lifetime.drain()
        assert retired == [False]
        return
    # _dense_source records and waits its own connector.load_stream readiness.
    readiness = Event()
    readiness.record(cold)
    readiness.synchronize()
    lifetime.retire_source(source)
    source.owner_lease.release("fenced")
    assert launches == [cold]
    assert retired == [False]
