# SPDX-License-Identifier: Apache-2.0
"""Startup markers can identify a stuck call without a traceback or NPU."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import ctypes
import importlib.util
import io
import logging
import time

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "startup_trace_under_test", ROOT / "lmcache/v1/startup_trace.py"
)
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


def test_begin_is_flushed_before_call_and_end_reports_duration(monkeypatch):
    stream = FlushedStream()
    monkeypatch.setattr(trace.sys, "stderr", stream)
    ticks = iter((1.0, 1.125))
    monkeypatch.setattr(trace.time, "perf_counter", lambda: next(ticks))
    with trace.startup_phase("shared_startup_receive", rank=3, src=0):
        assert stream.flushes == 1
        assert "state=begin rank=3 src=0" in stream.getvalue()
        assert "state=end" not in stream.getvalue()
    assert stream.flushes == 2
    assert "elapsed_ms=125.000" in stream.getvalue()
    assert f"pid={trace.os.getpid()}" in stream.getvalue()


@pytest.mark.parametrize("error", [ValueError("allocation failed"), SystemExit(7)])
def test_failure_is_logged_and_the_same_exception_is_reraised(error, capsys):
    with pytest.raises(type(error)) as caught:
        with trace.startup_phase("native_pinned_alloc", bytes=16 << 30):
            raise error
    assert caught.value is error
    output = capsys.readouterr().err
    assert "state=error" in output and "state=end" not in output
    assert type(error).__name__ in output


def load_engine():
    path = ROOT / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheEngine"
    )
    methods = {
        "post_init",
        "_post_init_shared_cpu_cache",
        "_shared_cpu_cache_startup_envelope",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
    cls.bases, cls.decorator_list = [], []
    ns = dict(
        startup_phase=trace.startup_phase, logger=logging.getLogger(__name__), time=time
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), ns)
    return ns, ns[cls.name]


@pytest.mark.parametrize("rank", [0, 3])
def test_post_init_marks_calls_before_entering_them_and_keeps_order(rank, capsys):
    ns, engine_type = load_engine()
    engine = engine_type()
    events = []

    def observe(stage):
        output = capsys.readouterr().err
        assert f"stage={stage} state=begin" in output
        assert f"rank={rank}" in output
        events.append(stage)

    def storage(*args, **kwargs):
        observe("storage_manager")
        return NS(
            local_cpu_backend=NS(
                memory_allocator=NS(
                    buffer=NS(numel=lambda: 16 << 30), shm_name="/test-shm"
                )
            )
        )

    def device_ptr():
        observe("shared_device_ptr")
        return 1234

    mapping = NS(preflight_device_ptr=device_ptr, passive_allocator=lambda: object())

    def attach(**kwargs):
        observe("shared_slab_attach")
        return mapping

    ns["StorageManager"] = storage
    ns["SharedSlabMapping"] = NS(
        from_rank0_allocator=lambda **kw: mapping, attach=attach
    )
    engine.post_inited = False
    engine.config = NS(
        get_lookup_server_worker_ids=lambda *a: [0],
        max_local_cpu_size=16,
        extra_config={},
    )
    engine.metadata = NS(
        worker_id=rank,
        first_rank=0,
        world_size=8,
        use_mla=True,
        is_first_rank=lambda: rank == 0,
    )
    engine._shared_cpu_sparse_capacity_sanity_pending = True
    engine._report_shared_cpu_sparse_capacity_sanity = lambda: observe("capacity_check")
    engine._preflight_shared_cpu_shm_capacity = lambda: observe("shm_space_check")
    engine.enable_shared_cpu_cache = True
    engine._is_passive = lambda: rank != 0
    engine.use_layerwise = engine.save_only_first_rank = True
    engine.lmcache_worker = engine.event_manager = engine.storage_manager = None
    engine.shared_cpu_cache_strict = True
    engine.shared_cpu_cache_name = "/test-shm"
    engine.shared_cpu_cache_slab_size = 0
    engine.shared_cpu_cache_generation = 0
    engine._get_shared_config_value = lambda *a: True

    def broadcast(payload, src):
        assert src == 0
        observe("shared_startup_broadcast" if rank == 0 else "shared_startup_receive")
        return payload or dict(
            status="ok", shm_name="/test-shm", slab_size=16 << 30, generation=1
        )

    engine.broadcast_object_fn = broadcast
    engine.post_init()
    assert engine.post_inited
    expected = ["capacity_check", "shm_space_check"]
    expected += (
        ["storage_manager", "shared_device_ptr", "shared_startup_broadcast"]
        if rank == 0
        else ["shared_startup_receive", "shared_slab_attach", "shared_device_ptr"]
    )
    assert events == expected
    capsys.readouterr()
    engine.post_init()
    assert capsys.readouterr().err == ""  # no repeated startup markers


def test_early_preflight_failure_has_a_phase_marker_without_changing_failure(capsys):
    ns, engine_type = load_engine()
    engine = engine_type()
    failure = ValueError("slab does not fit")
    engine.post_inited = False
    engine.config = NS(get_lookup_server_worker_ids=lambda *a: [0])
    engine.metadata = NS(worker_id=0, use_mla=True, world_size=8)
    engine._shared_cpu_sparse_capacity_sanity_pending = False

    def preflight():
        assert "stage=shm_space_check state=begin" in capsys.readouterr().err
        raise failure

    engine._preflight_shared_cpu_shm_capacity = preflight
    with pytest.raises(ValueError) as caught:
        engine.post_init()
    assert caught.value is failure and not engine.post_inited
    assert "stage=shm_space_check state=error" in capsys.readouterr().err


def test_native_allocation_has_flushed_marker_before_native_call(capsys):
    path = ROOT / "lmcache/v1/memory_management.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_allocate_cpu_memory"
    )
    backing = ctypes.create_string_buffer(16)

    def allocate(size, name):
        assert (size, name) == (16, "/test-shm")
        output = capsys.readouterr().err
        assert "stage=native_pinned_alloc state=begin" in output
        assert "bytes=16 shm_name='/test-shm'" in output
        return ctypes.addressof(backing)

    ns = dict(
        startup_phase=trace.startup_phase,
        ctypes=ctypes,
        torch=NS(uint8="uint8", frombuffer=lambda b, **kw: b),
        _resolve_pinned_alloc_free=lambda *a, **kw: ((allocate, "/test-shm"), ()),
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), ns)
    buffer = ns[fn.name](16, shm_name="/test-shm")
    assert len(buffer) == 16
    assert "stage=native_pinned_alloc state=end" in capsys.readouterr().err


def test_registration_marks_groups_manager_and_final_preflight(capsys):
    path = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl"
    )
    fn = next(n for n in cls.body if getattr(n, "name", None) == "register_kv_caches")
    fn.decorator_list = []
    calls = []

    def observe(stage):
        assert f"stage={stage} state=begin" in capsys.readouterr().err
        calls.append(stage)

    adapter = NS(
        kv_caches={},
        _refresh_kvcaches_list=lambda: None,
        _build_kv_layer_groups=lambda: observe("kv_layer_groups"),
        _manager=NS(post_init=lambda: observe("manager_post_init")),
        lmcache_engine=NS(
            preflight_group1_direct_hbm=lambda cache: observe(
                "group1_direct_hbm_preflight"
            )
        ),
        _kvcaches_for_group=lambda group: [group],
    )
    ns = dict(startup_phase=trace.startup_phase, logger=logging.getLogger(__name__))
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), ns)
    caches = {"layer0": object()}
    ns[fn.name](adapter, caches)
    assert adapter.kv_caches is caches
    assert calls == [
        "kv_layer_groups",
        "manager_post_init",
        "group1_direct_hbm_preflight",
    ]
