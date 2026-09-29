# SPDX-License-Identifier: Apache-2.0
"""Exercise D-path contracts with real CPU tensors and no vLLM installation."""

# Standard
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator, SharedHandleBatch


def adapter_type() -> type:
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl"
    )
    names = {
        "_merge_cache_group_by_ranges",
        "_ensure_layer_cache_shape",
        "_merge_store_result_into_worker_state",
        "_layerwise_storer_drain_limit",
        "wait_for_layer_load",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", "") in names]
    namespace: dict[str, Any] = {
        "torch": torch,
        "logger": logging.getLogger(__name__),
        "LayerwisePointerTable": Mock(
            side_effect=AssertionError("D allocated P pointer table")
        ),
        "_lmcache_nvtx_annotate": lambda fn: fn,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["LMCacheConnectorV1Impl"]


Adapter = adapter_type()


def test_decoder_appends_existing_pointer_tensor_without_prefill_table() -> None:
    adapter = Adapter()
    adapter._layerwise_prefill_p_node = False
    adapter._is_dsa_two_groups = lambda: False
    adapter._is_decode_window_save_request = lambda _: True
    adapter.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=True)
    cache = {
        "cached_starts": [0],
        "cached_ends": [4],
        "cached_keys": [["a"]],
        "cached_memory_objs": [[object()]],
        "cached_tensors": [[torch.ones(4)]],
        "cached_chunk_dev_ptrs": [[101]],
        "cached_chunk_ptrs_npu": [torch.tensor([101])],
        "cached_shared_handles": [],
    }
    prior_pointer_tensor = cache["cached_chunk_ptrs_npu"][0]
    state = SimpleNamespace(
        cache_kwargs=lambda *args, **kwargs: cache,
        pointer_tables={},
        dense_prefix_generation=None,
    )
    result = SimpleNamespace(
        kv_group=0,
        starts=[4],
        ends=[8],
        keys=[["b"]],
        memory_objs=[[object()]],
        tensors=[[torch.ones(4)]],
        chunk_dev_ptrs=[[202]],
        chunk_ptrs=[torch.tensor([202])],
    )
    assert adapter._merge_store_result_into_worker_state(state, result, object()) == 1
    assert cache["cached_starts"] == [0, 4] and cache["cached_ends"] == [4, 8]
    assert cache["cached_chunk_ptrs_npu"][0].tolist() == [101, 202]
    assert prior_pointer_tensor.tolist() == [101]
    assert state.pointer_tables == {}


@pytest.mark.parametrize("prefill,expected", [(False, 6), (True, 10)])
def test_storer_drain_budget_only_doubles_for_prefill(
    prefill: bool, expected: int
) -> None:
    adapter = Adapter()
    adapter.lmcache_engine = SimpleNamespace(num_layers=4)
    adapter._layerwise_prefill_p_node = prefill
    assert adapter._layerwise_storer_drain_limit() == expected


def test_decoder_without_retrievers_does_not_resolve_a_prefill_bank() -> None:
    adapter = Adapter()
    adapter.supports_layerwise_prefill_transfer_window = False
    adapter.layerwise_retrievers = []
    adapter._layerwise_wait_group = Mock(
        side_effect=AssertionError("inactive D wait resolved layer")
    )
    adapter.wait_for_layer_load("not-a-prefill-layer")
    adapter._layerwise_wait_group.assert_not_called()


@pytest.mark.parametrize("deferred,expected_yields", [(False, 0), (True, 6)])
def test_unhealthy_store_keeps_ordinary_stopiteration_contract(
    deferred: bool, expected_yields: int
) -> None:
    engine = object.__new__(LMCacheEngine)
    engine._num_transfer_layers_for_call = lambda *args: 2
    engine.is_healthy = lambda: False
    assert (
        len(list(engine.store_layer([1, 2], deferred_layerwise_put=deferred)))
        == expected_yields
    )


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("page", [False, True])
def test_passive_view_reuse_is_explicit_and_keeps_real_layer_addresses(
    prefill: bool, page: bool
) -> None:
    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.arange(256, dtype=torch.uint8),
        shm_name="isolation",
        generation=1,
        reuse_prefill=prefill,
    )
    batch = SharedHandleBatch(
        shm_name="isolation",
        producer_rank=0,
        num_layers=2,
        num_chunks=1,
        physical_sizes=[8],
        chunk_hashes=[1],
        offsets=[] if page else [32, 40],
        page_offsets=[32] if page else [],
        page_physical_sizes=[16] if page else [],
    )
    kwargs = dict(
        chunk_index=0,
        shape=torch.Size([4]),
        dtype=torch.float16,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
        cached_positions=range(4),
    )
    create = allocator.create_page_view if page else allocator.create_batch_view
    if not page:
        kwargs["layer_id"] = 1
    original = create(batch, **kwargs)
    second = create(batch, previous=original, **kwargs)
    assert (second is original) is prefill
    if page:
        assert second.layer_data_ptr(0) == allocator.buffer.data_ptr() + 32
        assert second.layer_data_ptr(1) == allocator.buffer.data_ptr() + 40
    else:
        assert second.data_ptr == allocator.buffer.data_ptr() + 40
    if second is not original:
        second.ref_count_down()
    original.ref_count_down()
    assert not original.is_valid()


@pytest.mark.parametrize("registration_error", [False, True])
def test_decoder_adopts_shared_sources_before_registration(
    registration_error: bool,
) -> None:
    engine = object.__new__(LMCacheEngine)
    engine.supports_dense_sparse_cache_retention = lambda: True
    engine.num_layers_for_group = lambda _: 1
    engine.gpu_connector = SimpleNamespace(release_sparse_chunk_ptr_cache=Mock())
    caches = {
        name: []
        for name in (
            "cached_keys",
            "cached_starts",
            "cached_ends",
            "cached_memory_objs",
            "cached_chunk_dev_ptrs",
            "cached_chunk_ptrs_npu",
            "cached_shared_handles",
        )
    }
    caches["cached_chunk_dev_ptrs"] = [[101]]
    caches["cached_chunk_ptrs_npu"] = [torch.tensor([101])]

    def register(*args: Any, **kwargs: Any) -> None:
        assert caches["cached_starts"] == [0]
        assert caches["cached_ends"] == [4]
        assert "preserve_replaced" not in kwargs and "replace_from" not in kwargs
        if registration_error:
            raise RuntimeError("registration failed")

    engine.register_shared_cpu_sparse_request = register
    arguments = dict(
        req_id="D",
        starts=[0],
        ends=[4],
        keys_layer_major=[["key"]],
        memory_objs=[[object()]],
        handles=[[object()]],
        kv_group=0,
        kwargs={**caches, "_retain_shared_dense_cache": True},
    )
    if registration_error:
        with pytest.raises(RuntimeError, match="registration failed"):
            engine._adopt_dense_shared_retrieve_cache(**arguments)
        assert all(not values for values in caches.values())
    else:
        assert engine._adopt_dense_shared_retrieve_cache(**arguments)
    engine.gpu_connector.release_sparse_chunk_ptr_cache.assert_not_called()


@pytest.mark.parametrize("prefill", [False, True])
def test_extra_startup_failure_collective_only_enabled_on_prefill(
    prefill: bool,
) -> None:
    engine = object.__new__(LMCacheEngine)
    engine.post_inited = False
    engine._layerwise_prefill_p_node = prefill
    engine.enable_shared_cpu_cache = True
    engine.config = SimpleNamespace(get_lookup_server_worker_ids=lambda *args: [])
    engine.metadata = SimpleNamespace(
        use_mla=True,
        worker_id=0,
        world_size=2,
        first_rank=0,
        is_first_rank=lambda: True,
    )
    engine._preflight_shared_cpu_shm_capacity = Mock(
        side_effect=RuntimeError("capacity failed")
    )
    engine._shared_cpu_cache_startup_envelope = lambda *args: {"status": "error"}
    engine.broadcast_object_fn = Mock()
    with pytest.raises(RuntimeError, match="capacity failed"):
        engine.post_init()
    assert engine.broadcast_object_fn.call_count == int(prefill)


@pytest.mark.parametrize("prefill", [False, True])
def test_shared_startup_failure_policy_keeps_decoder_recompute(
    prefill: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First Party
    from lmcache.v1.manager import LMCacheManager

    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", str(prefill).lower())
    manager = object.__new__(LMCacheManager)
    manager._init_failed = True
    manager._init_failed_reason = "earlier startup failure"
    manager._lmcache_engine = SimpleNamespace(mark_init_failed=Mock())
    manager._config = SimpleNamespace(get_extra_config_value=lambda *args: True)
    if prefill:
        with pytest.raises(RuntimeError, match="earlier startup failure"):
            manager.post_init()
    else:
        manager.post_init()
    manager._lmcache_engine.mark_init_failed.assert_called_once_with(
        "earlier startup failure"
    )
