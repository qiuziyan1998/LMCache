# SPDX-License-Identifier: Apache-2.0
"""Check event activation and derived dispatch, with no device imports."""

import ast
from dataclasses import dataclass, field, fields
import gc
from types import SimpleNamespace as NS
import weakref

from test_checkpoint_adapter import SOURCE, control, method, request


def metadata_types():
    names = {"LMCacheConnectorMetadata", "PreemptionConnectorMetadata"}
    nodes = [
        n
        for n in ast.parse(SOURCE.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    ns = dict(
        vars(control),
        dataclass=dataclass,
        field=field,
        KVConnectorMetadata=object,
        _lmcache_nvtx_annotate=lambda f: f,
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, *nodes], type_ignores=[])
            ),
            "metadata",
            "exec",
        ),
        ns,
    )
    return ns["LMCacheConnectorMetadata"], ns["PreemptionConnectorMetadata"]


def fixture():
    plain, checkpoint = metadata_types()
    calls = []

    class Base:
        def build_connector_meta(self, output):
            calls.append("ordinary")
            metadata = plain(requests=["ordinary-request-metadata"])
            if getattr(output, "cold_load_pending", False):
                metadata.dsa_cold_compact_load_pending = True
            return metadata

    for name in (
        "prepare_preemption_checkpoint",
        "_arm_preemption_controls",
        "_build_checkpoint_connector_meta",
        "_build_preemption_controls",
        "accept_preemption_result",
    ):
        setattr(Base, name, method(name, PreemptionConnectorMetadata=checkpoint))

    class Derived(Base):
        def build_connector_meta(self, output):
            calls.append("derived")
            return super().build_connector_meta(output)

    adapter = Derived()
    adapter.__dict__.update(
        _checkpoint_snapshots=[],
        _checkpoint_cancels=[],
        _preemption_checkpoints={},
        _lmcache_chunk_size=4,
        _decode_window_save_window_size=8,
        _request_trackers={
            "r": NS(
                prompt_len=4,
                decode_window_save_committed_end=4,
                dsa_nonresident_frontier=6,
                skip_save=False,
                request_configs=None,
            )
        },
        _unfinished_requests={"r": request()},
        _resume_lookup_queries={},
        kv_role="kv_both",
        config=NS(dsa_two_groups=True),
        lookup_client=None,
    )
    return adapter, calls, plain, checkpoint


def test_controls_are_armed_by_events_and_restore_derived_method():
    adapter, calls, plain, checkpoint = fixture()
    output = NS(finished_req_ids=set())
    original = type(adapter).build_connector_meta
    for _ in range(20):
        assert type(adapter.build_connector_meta(output)) is plain
    assert "build_connector_meta" not in adapter.__dict__
    assert calls == ["derived", "ordinary"] * 20
    adapter.prepare_preemption_checkpoint(("r", 1, ((1, 2, 3), (4, 5, 6)), 14))
    meta = adapter.build_connector_meta(output)
    assert isinstance(meta, checkpoint) and len(meta.preemption_captures) == 1
    assert meta.requests == ["ordinary-request-metadata"]
    assert adapter.build_connector_meta.__func__ is original
    assert "_checkpoint_build_original" not in adapter.__dict__
    # Awaiting a reply does not run a control scan each scheduler step.
    assert type(adapter.build_connector_meta(output)) is plain
    adapter.accept_preemption_result(control.CheckpointResult("r", 1, "captured", 14))
    meta = adapter.build_connector_meta(output)
    assert meta.preemption_seals[0].tokens == tuple(range(11))
    assert adapter.build_connector_meta.__func__ is original
    adapter.accept_preemption_result(control.CheckpointResult("r", 1, "ready", 11))
    assert type(adapter.build_connector_meta(output)) is plain


def test_ordinary_metadata_keeps_its_original_wire_fields():
    plain, checkpoint = metadata_types()
    assert [f.name for f in fields(plain)] == ["requests"]
    assert {f.name for f in fields(checkpoint)} == {
        "requests",
        "preemption_captures",
        "preemption_seals",
        "preemption_cancels",
        "preemption_releases",
    }


def test_checkpoint_controls_preserve_simultaneous_cold_load_trigger(monkeypatch):
    import pickle
    import sys
    from types import ModuleType

    adapter, _, plain, checkpoint = fixture()
    adapter.prepare_preemption_checkpoint(("r", 1, ((1, 2, 3), (4, 5, 6)), 14))
    # The ordinary builder sets this flag when dispatching an async prefix load,
    # including a zero-model-token batch. Dropping it prevents _start_load_kv
    # from submitting the load before its attn_metadata=None early return.
    metadata = adapter.build_connector_meta(
        NS(finished_req_ids=set(), cold_load_pending=True)
    )
    assert metadata.dsa_cold_compact_load_pending is True
    assert len(metadata.preemption_captures) == 1
    # The multiprocess executor uses pickle for scheduler output broadcasts.
    module = ModuleType("checkpoint_metadata_wire_test")
    for cls in (plain, checkpoint):
        cls.__module__ = module.__name__
        setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    metadata = pickle.loads(pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
    assert metadata.dsa_cold_compact_load_pending is True
    cold_request = NS(load_spec=NS(can_load=True, dsa_cold_compact_load=True))
    metadata.requests = [cold_request]
    fn = next(
        n
        for n in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "_start_load_kv"
    )
    # Exercise the actual no-forward prefix, stopping before NPU attention work.
    stop = next(
        i
        for i, node in enumerate(fn.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "attn_metadata is None"
    )
    fn.body = fn.body[: stop + 1]
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"LMCacheConnectorMetadata": plain, "logger": NS(debug=lambda *a: None)}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, fn], type_ignores=[])),
            "no_forward",
            "exec",
        ),
        ns,
    )
    submitted = []
    worker = NS(
        _parent=NS(_get_connector_metadata=lambda: metadata),
        kv_caches={"layer": object()},
        _try_prepare_dsa_live_split=lambda req: False,
        _submit_dsa_cold_compact_load=submitted.append,
    )
    ns[fn.name](worker, NS(attn_metadata=None))
    assert submitted == [cold_request]


def test_abandoned_armed_scheduler_does_not_require_cyclic_gc():
    adapter, _, _, _ = fixture()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        adapter.prepare_preemption_checkpoint(("r", 1, ((1,), (2,)), 14))
        owner = weakref.ref(adapter)
        del adapter
        assert owner() is None
    finally:
        if gc_was_enabled:
            gc.enable()
