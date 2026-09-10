# SPDX-License-Identifier: Apache-2.0
"""Real adapter dispatch and per-load Group-1 source ownership."""

# Standard
from typing import Any
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState
from lmcache.v1.cache_engine import LMCacheEngine
from tests.v1.connector_test_utils import make_sparse_req_meta, make_worker_connector


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("old_state", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_prefill_group0_adapter_dispatch_keeps_group1_cache_path(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, old_state: bool, fail: bool
) -> None:
    request = make_sparse_req_meta("request", token_count=11)
    request.is_sparse_decode = False
    request.indexer_slot_mapping = [torch.arange(11)]
    request.load_spec.vllm_cached_tokens = 6
    if enabled:
        monkeypatch.setattr(
            request.slot_mapping[0],
            "to",
            Mock(side_effect=AssertionError("G0 CPU slots were uploaded")),
        )
    impl, _, _ = make_worker_connector([request], use_layerwise=True)
    impl._prefill_group0_direct_hbm = enabled
    impl._lmcache_chunk_size = 4
    impl.num_layers = 2
    impl._block_size = 4
    impl._kvcaches_list = [object()]
    impl._kvcaches_for_group = lambda group: [object()]
    impl._indexer_layer_names = ["indexer"]
    impl._layerwise_requests = []
    impl.layerwise_retrievers = []
    impl._layerwise_retriever_is_sparse = []
    impl._layerwise_sparse_shared_ordered = []
    impl.enable_sparse_attention = True
    impl.config = SimpleNamespace(dsa_two_groups=True)
    impl._is_dsa_two_groups = lambda: True
    impl._prune_worker_retrieve_state = Mock()
    impl._stats_monitor = Mock()
    impl._warm_request_retrieve_metadata = Mock(return_value=(None, False))
    impl._publish_worker_retrieve_state = Mock()
    impl._release_shared_worker_retrieve_state = Mock()
    impl._set_worker_retrieve_state = Mock()
    impl._mark_worker_retrieve_registry_changed = Mock()
    if old_state:
        impl._worker_retrieve_state["request"] = WorkerRetrieveState(req_id="request")
    calls = []

    def load(tokens: Any, mask: Any, **kwargs: Any) -> Any:
        calls.append(("shared", tokens, mask, kwargs))
        if fail and kwargs["kv_group"] == 0:
            raise RuntimeError("group0 failed")
        yield None
        yield None
        yield None
        yield mask

    def direct(tokens: Any, mask: Any, **kwargs: Any) -> Any:
        calls.append(("direct", tokens, mask, kwargs))
        if fail:
            raise RuntimeError("group0 failed")
        assert kwargs["slot_mapping"].device.type == "cpu"
        assert "cached_memory_objs" not in kwargs
        yield None
        yield None
        yield None
        yield mask

    engine = SimpleNamespace(
        enable_shared_cpu_cache=True,
        supports_dense_sparse_cache_retention=lambda: True,
        retrieve_layer=load,
        retrieve_prefill_group0_direct=direct,
        release_shared_cpu_sparse_request=Mock(),
        lookup_unpin=Mock(),
        gpu_connector=SimpleNamespace(set_layerwise_staging_concurrency=Mock()),
    )
    impl.lmcache_engine = engine
    monkeypatch.setattr(adapter_mod, "serving_perf_enabled", lambda: False)
    if fail:
        with pytest.raises(RuntimeError, match="group0 failed"):
            impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))
        assert len(calls) == 1  # The Group-1 generator was never advanced.
        assert impl.layerwise_retrievers == []
        assert impl._worker_retrieve_state == {}
        engine.lookup_unpin.assert_called_once_with("request")
        return
    if enabled:
        monkeypatch.setattr(
            adapter_mod,
            "WorkerRetrieveState",
            Mock(side_effect=AssertionError("Direct P must not build a sparse seed")),
        )
    impl.start_load_kv(SimpleNamespace(attn_metadata=SimpleNamespace()))
    assert [call[0] for call in calls] == ["direct" if enabled else "shared", "shared"]
    assert calls[0][2].tolist() == [False] * 4 + [True] * 7
    assert len(calls[0][1]) == 11  # Keep the complete final partial page.
    group1 = calls[1][3]
    assert group1["kv_group"] == 1
    assert group1["_retain_shared_dense_cache"] is (not enabled)
    assert group1.get("_retain_dense_sources_until_save", False) is enabled
    if not enabled:
        assert "_retain_dense_sources_until_save" not in group1
    assert ("cached_memory_objs" in group1) is (not enabled)
    if enabled:
        impl._warm_request_retrieve_metadata.assert_not_called()
        impl._publish_worker_retrieve_state.assert_not_called()
        impl._set_worker_retrieve_state.assert_not_called()
        engine.lookup_unpin.assert_not_called()
        if not old_state:
            engine.release_shared_cpu_sparse_request.assert_called_once_with("request")
    impl._drain_layerwise_retrievers()
    assert impl.layerwise_retrievers == []


def test_prefill_group1_source_lease_survives_until_request_cleanup() -> None:
    class Page:
        is_pinned = True
        refs = 2  # Hot-cache reference and the current load reference.

        def unpin(self) -> None:
            self.is_pinned = False

        def ref_count_down(self) -> None:
            self.refs -= 1

        def is_valid(self) -> bool:
            return self.refs > 0

    page = Page()
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(is_first_rank=lambda: True)
    engine.shared_cpu_cache_generation = 1
    engine._shared_cpu_request_leases = {}
    assert engine._adopt_dense_shared_retrieve_cache(
        req_id="request",
        starts=[0],
        ends=[4],
        keys_layer_major=[[], []],
        memory_objs=[[page], [page]],
        handles=[[], []],
        kv_group=1,
        kwargs={"_retain_dense_sources_until_save": True},
    )
    assert page.refs == 2 and page.is_pinned
    assert set(engine._shared_cpu_request_leases["request"].groups) == {1}
    engine.release_shared_cpu_sparse_request("request")
    assert page.refs == 1 and not page.is_pinned
    assert engine._shared_cpu_request_leases == {}
