# SPDX-License-Identifier: Apache-2.0
"""Admission CPU-work regressions; Ascend methods execute without NPU imports."""

import ast
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace as NS
import weakref

import pytest
import torch

from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.token_database import ChunkedTokenDatabase
from lmcache.utils import CacheEngineKey

from .test_shared_cpu_cache import (
    _FakeLocalCPUBackend,
    _make_engine_for_sparse_capacity,
    _make_key,
)


def ascend_method(name):
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/cache_engine.py"
    )
    if not path.exists():
        pytest.skip("requires sibling Ascend checkout")
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendLMCacheEngine"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    scope = dict(
        threading=threading,
        torch=torch,
        CacheEngineKey=CacheEngineKey,
        ChunkedTokenDatabase=ChunkedTokenDatabase,
        _SHARED_CPU_CHUNK_PLAN_KEY="_shared_cpu_chunk_hash_plan",
        _REMOTE_FILL_EXACT_LOCATIONS_KEY="_remote_fill_exact_locations",
        serving_perf_enabled=lambda: False,
        NativeExternalPageTransferUnknownError=RuntimeError,
    )
    exec(compile(tree, str(path), "exec"), scope)
    return scope[name], method


@pytest.mark.parametrize("free", [0, 1 << 30])
def test_mixed_capacity_checks_each_remote_page_once_and_skips_unneeded_scan(free):
    class Cache(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    class Locations(list):
        checks = 0

        def __getitem__(self, index):
            type(self).checks += 1
            return super().__getitem__(index)

    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    engine.config.chunk_size = 4
    engine.config.enable_shared_cpu_cache = engine.config.use_layerwise = True
    engine.config.remote_url = "mooncakestore://test"
    engine.config.extra_config.update(
        mooncake_layer_merged_page_objects=True,
        mooncake_page_first_multi_buffer=True,
        save_only_first_rank=True,
    )
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        torch.Size([8]),
        torch.float16,
        batch_size=1,
        num_layers=4,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        valid_tokens=4,
        full_tokens=4,
    )
    a = _make_key()
    b = CacheEngineKey("model", 8, 0, 5678, torch.float16, kv_group=0)
    cache = Cache({a: pages[0]})
    backend = _FakeLocalCPUBackend(free_bytes=free, hot_cache=cache)
    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda obj: obj is pages[0]
    try:
        details = engine._shared_cpu_runtime_capacity_details(
            req_id="r",
            phase="cold",
            kv_group=0,
            keys_layer_major=[[a.get_layer(i), b.get_layer(i)] for i in range(4)],
            chunk_locations_layer_major=[
                Locations(["LocalCPUBackend", "RemoteBackend"]) for _ in range(4)
            ],
            chunk_token_lengths=[4, 3],
        )
        assert Locations.checks == 4
        assert cache.scans == int(not free)
        assert details["fits"] == bool(free)
        assert details["hot_chunk_count"] == 1
        assert details[
            "required_bytes"
        ] == engine._shared_cpu_estimated_physical_page_bytes(0, num_tokens=3)
    finally:
        pages[0].ref_count_down()


def token_database():
    db = object.__new__(ChunkedTokenDatabase)
    db.config = NS(chunk_size=4, save_unfull_chunk=True, dsa_two_groups=True)
    db.chunk_size, db.save_only_first_rank, db.mooncake_payload_layout = (
        4,
        True,
        "mixed-c8",
    )
    db.metadata = NS(
        model_name="model",
        world_size=4,
        worker_id=0,
        get_dtypes=lambda: [torch.bfloat16, torch.int8],
    )
    db.hash_func = hash
    return db


@pytest.mark.parametrize("total", [1, 4, 5, 11, 12])
def test_shared_hash_plan_is_lazy_concurrent_and_keeps_group1_identity(total):
    factory, _ = ascend_method("prepare_cold_chunk_plan")
    direct, _ = ascend_method("load_group1_pages_direct")
    db = token_database()
    tokens = list(range(total))
    configs = {"lmcache.tag.test": "request-a"}
    expected = list(
        db.process_tokens(tokens=tokens, request_configs=configs, kv_group=1)
    )
    hashes = []
    db.hash_func = lambda data: hashes.append(data) or hash(data)
    engine = NS(token_database=db)
    get = factory(engine, tokens)
    assert not hashes
    with ThreadPoolExecutor(2) as pool:
        plans = list(pool.map(lambda _: get(), range(8)))
    assert all(plan is plans[0] for plan in plans)
    assert len(hashes) == (total + 3) // 4
    loaded = []
    engine._persistent_direct_hbm_split_group_enabled = lambda: True
    engine._ensure_layerwise_connector_layout = lambda **kw: None
    engine.gpu_connector = NS(
        plan_direct_page_destinations=lambda caches, slots, starts, ends, group: (
            [[1]] * len(starts),
            [[2]] * len(starts),
            (),
        )
    )
    engine._group1_external_page_load = lambda: lambda keys, *args: loaded.extend(keys)
    direct(engine, tokens, torch.arange(total), [], configs, "r", chunk_plan=get)
    assert loaded == [key for _, _, key in expected]
    assert len(hashes) == len(expected)  # Group 1 did not hash again.
    # Run the actual active-TP metadata builder with the same shared plan.
    ensure, _ = ascend_method("_ensure_retrieve_chunk_metadata")
    engine._needs_retrieve_metadata_refresh = lambda *a: True
    engine._num_layers_for_kv_group = lambda group: 3
    engine._should_use_shared_layerwise_retrieve = lambda group: True
    engine._is_shared_retrieve_passive = lambda group: False
    engine._use_sampled_worker_retrieve = lambda group: True
    engine._remote_fill_local_full_hint = lambda cfg: None
    engine._force_layerwise_prefill_store = False
    engine._fill_retrieve_mask = lambda mask, starts, ends: mask.fill_(True)
    mask = torch.zeros(total, dtype=torch.bool)
    _, starts, ends, keys = ensure(
        engine,
        tokens=tokens,
        mask=mask,
        request_configs=configs,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        ret_mask=mask,
        retrieve_kwargs={
            "kv_group": 0,
            "_cold_chunk_plan": get,
            "shared_cpu_request_preflight_state": {},
        },
    )
    assert starts == [a for a, _, _ in expected]
    assert ends == [b for _, b, _ in expected]
    assert all(key.kv_group == 0 for row in keys for key in row)
    assert [key.chunk_hash for key in keys[0]] == [key.chunk_hash for key in loaded]
    assert len(hashes) == len(expected) and mask.all()


def test_hash_failure_releases_lock_and_segmented_database_keeps_old_path():
    factory, _ = ascend_method("prepare_cold_chunk_plan")
    assert factory(NS(token_database=object()), [1]) is None
    db = token_database()
    get = factory(NS(token_database=db), [1, 2])

    def fail(data):
        raise ValueError("hash failed")

    db.hash_func = fail
    with pytest.raises(ValueError, match="hash failed"):
        get()
    db.hash_func = hash
    assert get()[0][:2] == (0, 2)


def test_retained_remote_fill_plan_does_not_wait_for_shared_hashing():
    ensure, _ = ascend_method("_ensure_retrieve_chunk_metadata")
    db = token_database()
    tokens = list(range(11))
    retained = tuple(db.process_tokens(tokens=tokens, make_key=False))

    def unexpected_hash():
        raise AssertionError("retained plan waited for unrelated shared hashing")

    engine = NS(
        token_database=db,
        _needs_retrieve_metadata_refresh=lambda *args: True,
        _num_layers_for_kv_group=lambda group: 3,
        _should_use_shared_layerwise_retrieve=lambda group: True,
        _is_shared_retrieve_passive=lambda group: False,
        _use_sampled_worker_retrieve=lambda group: True,
        _remote_fill_local_full_hint=lambda cfg: (11,),
        _remote_fill_retained_local_page_plan=lambda *args: (
            retained,
            ["LocalCPUBackend"] * len(retained),
        ),
        _get_req_id=lambda kwargs: "r",
        _force_layerwise_prefill_store=False,
        _fill_retrieve_mask=lambda mask, starts, ends: mask.fill_(True),
    )
    kwargs = {
        "kv_group": 0,
        "_cold_chunk_plan": unexpected_hash,
        "shared_cpu_request_preflight_state": {},
    }
    mask = torch.zeros(11, dtype=torch.bool)
    location, starts, ends, keys = ensure(
        engine,
        tokens=tokens,
        mask=torch.ones(11, dtype=torch.bool),
        request_configs={},
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        ret_mask=mask,
        retrieve_kwargs=kwargs,
    )
    assert location == "LocalCPUBackend" and mask.all()
    assert (
        tuple(zip(starts, ends, (key.chunk_hash for key in keys[0]), strict=True))
        == retained
    )
    assert kwargs["_retrieve_metadata_mode"] == "remote_fill_retained"


def test_passive_tp_uses_shared_hashes_and_preserves_partial_tail():
    factory, _ = ascend_method("prepare_cold_chunk_plan")
    _, passive = ascend_method("_retrieve_layer_head_token_wise_shared_passive")
    # Execute the real passive metadata prefix before transport/consumer setup.
    stop = next(
        i
        for i, n in enumerate(passive.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "required_chunks"
    )
    passive.body = (
        passive.body[:stop] + ast.parse("return starts, ends, keys_layer_major").body
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(passive)
    scope = dict(
        torch=torch, CacheEngineKey=CacheEngineKey, serving_perf_enabled=lambda: False
    )
    exec(compile(tree, "actual_passive_metadata", "exec"), scope)
    db = token_database()
    tokens = list(range(11))
    expected = list(db.process_tokens(tokens=tokens, kv_group=0))
    engine = NS(
        token_database=db,
        gpu_connector=object(),
        shared_cpu_cache_passive_allocator=object(),
        _num_transfer_layers_for_call=lambda *args: 3,
        _build_retrieve_metadata_extension=lambda **kwargs: None,
    )
    plan = factory(engine, tokens)
    plan()
    db.hash_func = lambda data: pytest.fail("passive TP rehashed an existing plan")
    starts, ends, keys = scope[passive.name](
        engine,
        tokens,
        torch.ones(11, dtype=torch.bool),
        torch.zeros(11, dtype=torch.bool),
        kv_group=0,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        _cold_chunk_plan=plan,
        _sparse_cache_append=object(),
    )
    assert list(zip(starts, ends, strict=True)) == [(a, b) for a, b, _ in expected]
    assert keys == [[key.get_layer(i) for _, _, key in expected] for i in range(3)]


def test_hash_plan_is_request_scoped_and_released_without_cyclic_gc():
    factory, _ = ascend_method("prepare_cold_chunk_plan")

    class Engine:
        pass

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        engine = Engine()
        engine.token_database = token_database()
        ref = weakref.ref(engine)
        first = factory(engine, [1, 2, 3])
        second = factory(engine, [1, 2, 4])
        assert first() != second()
        del first, second, engine
        assert ref() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("page_prefix", [0, 2, 3])
def test_legacy_probing_only_visits_unproven_suffix(page_prefix):
    _, bootstrap = ascend_method("_retrieve_layer_head_token_wise_bootstrap_impl")
    branch = next(
        n
        for n in ast.walk(bootstrap)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "sampled_worker_retrieve"
        and n.body
        and isinstance(n.body[0], ast.Assign)
        and ast.unparse(n.body[0].targets[0]) == "local_cpu_backend"
    )
    tree = ast.parse(
        "def probe(self, missing_keys):\n local_prefix_layers=[]\n"
        " shared_chunk_locations_layer_major=[]\n"
    )
    method = tree.body[0]
    method.body.extend(branch.body)
    method.body.extend(
        ast.parse("return local_prefix_layers, shared_chunk_locations_layer_major").body
    )
    ast.fix_missing_locations(tree)
    scope = dict(
        mooncake_layer_pages_enabled=lambda config: True,
        LOCAL_CPU_BACKEND_NAME="LocalCPUBackend",
    )
    exec(compile(tree, "actual_bootstrap_probe", "exec"), scope)

    class Cache(dict):
        reads = 0

        def get(self, key, default=None):
            self.reads += 1
            return super().get(key, default)

    cache = Cache()
    backend = NS(cpu_lock=threading.Lock(), hot_cache=cache)
    backend.batched_get_prefixes_with_misses = MethodType(
        LocalCPUBackend.batched_get_prefixes_with_misses, backend
    )
    engine = NS(
        config=object(),
        _shared_local_cpu_backend=lambda: backend,
        storage_manager=NS(
            batched_contains_layer_pages=lambda *args: (page_prefix, {})
        ),
    )
    rows = [[_make_key().get_layer(i)] * 3 for i in range(4)]
    results, locations = scope["probe"](engine, rows)
    assert cache.reads == (4 if page_prefix < 3 else 0)
    assert sum(len(result.remote_positions) for result in results) == 4 * (
        3 - page_prefix
    )
    assert (
        locations
        == [["LocalCPUBackend"] * page_prefix + ["RemoteBackend"] * (3 - page_prefix)]
        * 4
    )
