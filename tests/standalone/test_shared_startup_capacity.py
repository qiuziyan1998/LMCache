# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for MLA startup sizing and failed-rank notification."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event
from types import ModuleType, SimpleNamespace as NS
import ast
import logging
import math
import sys
import time
import traceback

# Third Party
import pytest

# Local
from test_startup_trace import trace


ROOT = Path(__file__).resolve().parents[2]


def load_class(relative_path, name, methods, **ns):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
    cls.bases, cls.decorator_list = [], []
    ns.update(
        logger=logging.getLogger(__name__),
        startup_phase=trace.startup_phase,
        math=math,
        time=time,
        traceback=traceback,
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
    return ns, ns[name]


@pytest.fixture
def engine_api():
    return load_class(
        "lmcache/v1/cache_engine.py",
        "LMCacheEngine",
        {
            "post_init",
            "_post_init_shared_cpu_cache",
            "_shared_cpu_cache_startup_envelope",
            "_report_shared_cpu_sparse_capacity_sanity",
        },
    )


def capacity_engine(
    engine_api,
    *,
    pool_gb=16,
    length=100352,
    seqs=1,
    registered=True,
    group0_bytes=2,
    group1_bytes=2,
):
    _, cls = engine_api
    engine = cls()
    extra = {"vllm_max_model_len": length, "vllm_max_num_seqs": seqs}
    engine.config = NS(
        enable_sparse_attention=True,
        chunk_size=1024,
        extra_config=extra,
        max_local_cpu_size=pool_gb,
    )
    engine.enable_shared_cpu_cache = engine.save_only_first_rank = True
    engine.dsa_two_groups = True
    engine.metadata = NS(
        world_size=8,
        worker_id=0,
        is_first_rank=lambda: True,
        max_model_len=length,
        kv_layer_groups_manager=NS(
            kv_layer_groups=[object(), object()] if registered else []
        ),
        get_shapes=lambda tokens: [(1, 79, tokens, 576), (1, 22, tokens, 128)],
    )
    engine._get_shared_config_value = lambda key, default=None: extra.get(key, default)
    engine._persistent_direct_hbm_split_group_enabled = lambda: False
    engine._shared_cpu_dtype_for_kv_group = lambda g: NS(
        itemsize=(group0_bytes, group1_bytes)[g]
    )
    engine._effective_shared_cpu_cache_size_bytes = lambda: int(pool_gb * 2**30)
    engine.num_layers_for_group = lambda group: (79, 22)[group]
    return engine


def test_100k_uses_registered_mla_shape_without_querying_lazy_connector(
    engine_api, caplog
):
    engine = capacity_engine(engine_api)

    def uninitialized_connector_shape(group):
        pytest.fail("Startup must not use the connector's generic K/V fallback")

    engine._estimate_shared_cpu_chunk_bytes_per_layer = uninitialized_connector_shape
    with caplog.at_level(logging.INFO):
        engine._report_shared_cpu_sparse_capacity_sanity()
    report = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert report["bytes_per_chunk_all_layers"] == 98959360
    assert report["one_max_request_bytes"] == 9698017280  # not 18830852096
    assert report["max_single_request_tokens"] == 177152
    assert report["max_full_length_requests"] == 1
    assert report["max_tokens_per_request_at_max_num_seqs"] == 177152
    assert "[LMCACHE_CAPACITY]" in caplog.text
    assert "not runtime admission guarantees" in caplog.text


@pytest.mark.parametrize("length, fits", [(177152, True), (177153, False)])
def test_chunk_rounded_limit_and_actionable_failure(engine_api, length, fits, caplog):
    engine = capacity_engine(engine_api, length=length)
    with caplog.at_level(logging.INFO):
        if fits:
            engine._report_shared_cpu_sparse_capacity_sanity()
        else:
            with pytest.raises(ValueError) as caught:
                engine._report_shared_cpu_sparse_capacity_sanity()
            assert "one maximum request cannot fit" in str(caught.value)
            assert "max_single_request_tokens': 177152" in str(caught.value)
            assert "max_full_length_requests': 0" in str(caught.value)
            assert "Increase max_local_cpu_size" in str(caught.value)
    assert "max_single_request_tokens" in caplog.text


def test_concurrency_limit_is_a_report_not_a_new_runtime_admission_check(
    engine_api, caplog
):
    engine = capacity_engine(engine_api, seqs=4)
    with caplog.at_level(logging.WARNING):
        engine._report_shared_cpu_sparse_capacity_sanity()
    report = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert report["configured_worst_case_bytes"] == 9698017280 * 4
    assert report["max_tokens_per_request_at_max_num_seqs"] == 44032
    assert "does not reserve capacity" in caplog.text


def test_startup_uses_per_group_dtypes_and_keeps_direct_hbm_scope(engine_api):
    engine = capacity_engine(engine_api, group1_bytes=1)
    engine._report_shared_cpu_sparse_capacity_sanity()
    report = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert report["bytes_per_chunk_all_layers"] == 93192192 + 2883584
    engine._persistent_direct_hbm_split_group_enabled = lambda: True
    engine._report_shared_cpu_sparse_capacity_sanity()
    report = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert report["kv_groups"] == [0]
    assert report["bytes_per_chunk_all_layers"] == 93192192


def test_unregistered_legacy_metadata_keeps_existing_estimate(engine_api):
    engine = capacity_engine(engine_api, registered=False)
    engine._estimate_shared_cpu_chunk_bytes_per_layer = lambda group: (
        1024 * (576, 128)[group] * 2
    )
    engine._report_shared_cpu_sparse_capacity_sanity()
    assert (
        engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"][
            "one_max_request_bytes"
        ]
        == 9698017280
    )


def initialize_stub(engine_api, rank, broadcast):
    engine = capacity_engine(engine_api)
    engine.post_inited = False
    engine.metadata.worker_id = rank
    engine.metadata.first_rank = 0
    engine.metadata.use_mla = True
    engine.metadata.is_first_rank = lambda: rank == 0
    engine.config.get_lookup_server_worker_ids = lambda *args: [0]
    engine._shared_cpu_sparse_capacity_sanity_pending = True
    engine._preflight_shared_cpu_shm_capacity = lambda: None
    engine._is_passive = lambda: rank != 0
    engine.use_layerwise = True
    engine.lmcache_worker = engine.event_manager = engine.storage_manager = None
    engine.shared_cpu_cache_name = "/test-capacity"
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.broadcast_object_fn = broadcast
    return engine


@pytest.mark.parametrize(
    "phase", ["capacity_check", "shm_space_check", "StorageManager"]
)
def test_rank0_failure_releases_waiting_peer_exactly_once(engine_api, phase, caplog):
    ns, _ = engine_api
    messages = Queue()
    receiving = Event()
    sent = []

    def send(payload, src):
        assert src == 0
        sent.append(payload)
        messages.put(payload)

    def receive(payload, src):
        assert payload is None and src == 0
        receiving.set()
        return messages.get(timeout=3)

    rank0 = initialize_stub(engine_api, 0, send)
    peer = initialize_stub(engine_api, 3, receive)
    failure = ValueError(f"{phase}: insufficient memory; max_supported_tokens=177152")

    def fail(*args, **kwargs):
        raise failure

    def storage(*args, **kwargs):
        pytest.fail("No allocation should run after a preflight failure")

    ns["StorageManager"] = fail if phase == "StorageManager" else storage
    if phase == "capacity_check":
        rank0._report_shared_cpu_sparse_capacity_sanity = fail
    elif phase == "shm_space_check":
        rank0._preflight_shared_cpu_shm_capacity = fail

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(peer.post_init)
        assert receiving.wait(timeout=3)
        with pytest.raises(ValueError) as caught:
            rank0.post_init()
        assert caught.value is failure
        with pytest.raises(
            ValueError, match="passive preflight failed from rank0"
        ) as peer_error:
            waiting.result(timeout=3)
    assert str(failure) in str(peer_error.value)
    assert len(sent) == 1 and sent[0]["status"] == "error"
    assert phase in sent[0]["message"]
    assert not rank0.post_inited and not peer.post_inited
    assert "[LMCACHE_INIT_FAILED]" in caplog.text


def test_broadcast_failure_does_not_replace_original_capacity_error(engine_api, caplog):
    def failed_broadcast(*args):
        raise RuntimeError("broadcast unavailable")

    engine = initialize_stub(engine_api, 0, failed_broadcast)
    failure = ValueError("original capacity failure")

    def fail():
        raise failure

    engine._report_shared_cpu_sparse_capacity_sanity = fail
    with pytest.raises(ValueError) as caught:
        engine.post_init()
    assert caught.value is failure
    assert "Failed to broadcast" in caplog.text


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("earlier_failure", [False, True])
def test_manager_fails_shared_startup_but_preserves_optional_cache_fallback(
    monkeypatch, shared, earlier_failure, caplog
):
    module = ModuleType("lmcache.v1.lookup_client.lmcache_async_lookup_client")
    module.LMCacheAsyncLookupServer = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _, cls = load_class(
        "lmcache/v1/manager.py",
        "LMCacheManager",
        {
            "post_init",
            "_handle_post_init_failure",
            "_shared_cpu_startup_required",
        },
    )
    manager = cls()
    calls = []
    failure = ValueError("capacity too small; max_single_request_tokens=177152")

    def post_init(**kwargs):
        calls.append("engine")
        raise failure

    manager._config = NS(
        enable_async_loading=False,
        enable_shared_cpu_cache=shared,
        get_extra_config_value=lambda key, default=None: default,
    )
    manager._lmcache_engine = NS(
        post_init=post_init, mark_init_failed=lambda reason: calls.append(reason)
    )
    manager._lookup_server = None
    manager._init_failed = earlier_failure
    manager._init_failed_reason = str(failure) if earlier_failure else ""
    manager._init_health_monitor = lambda: pytest.fail(
        "failed engine cannot become healthy"
    )
    with caplog.at_level(logging.WARNING):
        if shared:
            with pytest.raises((ValueError, RuntimeError)) as caught:
                manager.post_init()
            if not earlier_failure:
                assert caught.value is failure
            assert "max_single_request_tokens=177152" in str(caught.value)
        else:
            manager.post_init()
    assert manager._init_failed
    assert ("engine" in calls) == (not earlier_failure)
    if shared:
        assert "[LMCACHE_INIT_FAILED]" in caplog.text
        assert "System will operate in degraded mode" not in caplog.text
