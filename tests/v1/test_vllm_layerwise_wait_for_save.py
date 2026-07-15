# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
)
from lmcache.v1.cache_engine import LayerwiseStoreResult


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeEngine:
    def __init__(self):
        self.unpinned: list[str] = []
        self.store_steps: dict[str, int] = {}
        self.store_calls: list[str] = []
        self.store_kwargs: list[dict] = []
        self.store_results: dict[str, LayerwiseStoreResult | None] = {}
        self.passive = False

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_kwargs.append(dict(kwargs))
        self.store_steps.setdefault(req_id, 0)
        num_layers = max(1, len(kwargs.get("kvcaches", []) or []))

        def _storer():
            for _ in range(num_layers):
                self.store_steps[req_id] += 1
                yield None
            self.store_steps[req_id] += 1
            yield self.store_results.get(req_id)

        return _storer()

    def _is_passive(self) -> bool:
        return self.passive


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def _make_req(
    req_id: str,
    can_save: bool = True,
    request_configs: dict | None = None,
):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=[torch.arange(4, dtype=torch.long)],
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_last_prefill=False,
        is_sparse_decode=False,
        load_spec=None,
        request_configs=request_configs,
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.enable_sparse_attention = False
    connector.config = SimpleNamespace(dsa_two_groups=False)
    connector.device = "cpu"
    connector.config = SimpleNamespace(dsa_two_groups=False)
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._kvcaches_list = []
    connector._latent_layer_names = []
    connector._indexer_layer_names = []
    connector._latent_kvcaches = []
    connector._indexer_kvcaches = []
    connector._layerwise_save_storers = {}
    connector._deferred_latent_pending = set()
    # lmcache_ascend patches LMCacheConnectorV1Impl; __new__ skips generic and
    # Ascend worker initialization.
    connector.store_async = False
    connector._wait_for_save_done = True
    connector._finished_req_ids_waiting_for_save = set()
    connector._late_finished_sending = set()
    connector._completed_decode_window_saves = {}
    connector._decode_window_save_completed_groups = set()
    connector._prefill_save_completed_groups = {}
    connector._decode_window_save_expected_start = {}
    return connector, metadata, engine


def test_layerwise_storer_is_request_scoped_across_interleaved_finalize() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1"), _make_req("req-2")]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1", "req-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 1

    metadata.requests = [_make_req("req-1")]
    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert engine.store_steps["req-2"] == 1
    assert engine.unpinned == ["req-1"]
    assert set(connector._layerwise_save_storers.keys()) == {
        ("req-2", "normal_save", 0, 0, 4)
    }

    metadata.requests = [_make_req("req-2")]
    connector.wait_for_save()
    assert engine.store_steps["req-2"] == 2
    assert engine.unpinned == ["req-1", "req-2"]
    assert connector._layerwise_save_storers == {}


def test_wait_for_save_repeated_call_does_not_readvance_finalized_storer() -> None:
    connector, metadata, engine = _make_connector([_make_req("req-1")])
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_steps["req-1"] == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2


def test_multi_layer_save_and_finalize() -> None:
    connector, _, engine = _make_connector([_make_req("req-1"), _make_req("req-2")])
    num_layers = 4
    connector.kv_caches = {
        f"layer{i}": torch.zeros(1) for i in range(num_layers)
    }
    connector._refresh_kvcaches_list()

    for layer_name in connector.kv_caches:
        connector.save_kv_layer(layer_name, torch.zeros(1), None)

    assert engine.store_steps["req-1"] == num_layers
    assert engine.store_steps["req-2"] == num_layers

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == num_layers + 1
    assert engine.store_steps["req-2"] == num_layers + 1
    assert connector._layerwise_save_storers == {}


def test_decode_window_save_completion_is_drained_after_wait() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, _ = _make_connector([request])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert connector.get_completed_decode_window_saves() == {}

    connector.wait_for_save()
    assert connector.get_completed_decode_window_saves() == {"req-window": 512}
    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_completion_waits_for_final_store_fence() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, _ = _make_connector([request])
    connector._decode_window_save_expected_start = {"req-window": 256}

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector._wait_for_save_done = False

    def fail_store_fence(_context) -> None:
        raise RuntimeError("store fence failed")

    connector._finish_save_batch = fail_store_fence

    with pytest.raises(RuntimeError, match="store fence failed"):
        connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}
    assert connector._decode_window_save_completed_groups == set()
    assert connector._decode_window_save_expected_start == {"req-window": 256}
    assert connector._wait_for_save_done is False

    connector._finish_save_batch = lambda _context: None
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {"req-window": 512}


@pytest.mark.parametrize("failed_group", [0, 1])
def test_decode_window_group_failure_is_atomic(failed_group: int) -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    request.save_spec.can_save_indexer = True
    request.indexer_slot_mapping = [torch.arange(4, 8, dtype=torch.long)]
    connector, _, engine = _make_connector([request])
    connector.config.dsa_two_groups = True
    connector.kv_caches = {
        "layer0.attn": torch.zeros(1),
        "layer0.indexer.k_cache": torch.zeros(1),
    }
    connector._decode_window_save_expected_start = {"req-window": 256}
    original_store_layer = engine.store_layer

    def failing_store_layer(token_ids, **kwargs):
        if kwargs.get("kv_group", 0) != failed_group:
            return original_store_layer(token_ids, **kwargs)

        def storer():
            yield None
            raise RuntimeError(f"group {failed_group} failed")

        return storer()

    engine.store_layer = failing_store_layer
    with pytest.raises(RuntimeError, match=f"group {failed_group} failed"):
        connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
        connector.save_kv_layer(
            "layer0.indexer.k_cache",
            torch.zeros(1),
            SimpleNamespace(slot_mapping=torch.arange(1, dtype=torch.long)),
        )
        connector.wait_for_save()
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}
    assert connector._layerwise_save_storers == {}
    assert connector._decode_window_save_completed_groups == set()
    assert connector._decode_window_save_expected_start == {"req-window": 256}

    engine.store_layer = original_store_layer
    connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
    connector.save_kv_layer(
        "layer0.indexer.k_cache",
        torch.zeros(1),
        SimpleNamespace(slot_mapping=torch.arange(1, dtype=torch.long)),
    )
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {"req-window": 512}


def test_decode_window_save_completion_not_reported_by_passive_rank() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, engine = _make_connector([request])
    engine.passive = True

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_completion_not_reported_without_store() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, _ = _make_connector([request])

    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_completion_not_reported_if_storer_does_not_finish() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, engine = _make_connector([request])

    def _unfinished_store_layer(token_ids, **kwargs):
        req_id = kwargs["req_id"]
        engine.store_calls.append(req_id)
        engine.store_kwargs.append(dict(kwargs))
        engine.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                engine.store_steps[req_id] += 1
                yield None

        return _storer()

    engine.store_layer = _unfinished_store_layer

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_completion_cannot_skip_unfinished_window() -> None:
    first = _make_req("req-window")
    first.is_decode_window_save = True
    first.decode_window_start = 256
    first.decode_window_end = 512
    first.decode_window_size = 256
    connector, metadata, engine = _make_connector([first])

    def _unfinished_store_layer(token_ids, **kwargs):
        req_id = kwargs["req_id"]
        engine.store_calls.append(req_id)
        engine.store_kwargs.append(dict(kwargs))
        engine.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                engine.store_steps[req_id] += 1
                yield None

        return _storer()

    engine.store_layer = _unfinished_store_layer
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()
    assert connector.get_completed_decode_window_saves() == {}

    second = _make_req("req-window")
    second.is_decode_window_save = True
    second.decode_window_start = 512
    second.decode_window_end = 768
    second.decode_window_size = 256
    metadata.requests = [second]
    engine.store_layer = _FakeEngine.store_layer.__get__(engine, _FakeEngine)

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_completion_cannot_skip_window_without_store() -> None:
    first = _make_req("req-window")
    first.is_decode_window_save = True
    first.decode_window_start = 256
    first.decode_window_end = 512
    first.decode_window_size = 256
    connector, metadata, _ = _make_connector([first])

    connector.wait_for_save()
    assert connector.get_completed_decode_window_saves() == {}

    second = _make_req("req-window")
    second.is_decode_window_save = True
    second.decode_window_start = 512
    second.decode_window_end = 768
    second.decode_window_size = 256
    metadata.requests = [second]

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_storer_is_scoped_by_kv_group() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    request.save_spec.can_save_indexer = True
    request.indexer_slot_mapping = [torch.arange(4, 8, dtype=torch.long)]
    connector, _, engine = _make_connector([request])
    connector.config.dsa_two_groups = True
    connector.kv_caches = {
        "layer0.attn": torch.zeros(1),
        "layer0.indexer.k_cache": torch.zeros(1),
    }

    connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
    connector.save_kv_layer(
        "layer0.indexer.k_cache",
        torch.zeros(1),
        SimpleNamespace(slot_mapping=torch.arange(1, dtype=torch.long)),
    )

    assert len(engine.store_calls) == 2
    assert [kwargs.get("kv_group", 0) for kwargs in engine.store_kwargs] == [0, 1]
    assert engine.store_kwargs[1]["slot_mapping"].tolist() == [4, 5, 6, 7]
    assert set(connector._layerwise_save_storers) == {
        ("req-window", "decode_window_save", 0, 256, 512)
    }

    connector.wait_for_save()
    assert connector._layerwise_save_storers == {}
    assert connector.get_completed_decode_window_saves() == {"req-window": 512}


def test_decode_window_save_completion_waits_for_required_indexer_group() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    request.save_spec.can_save_indexer = True
    connector, _, _ = _make_connector([request])
    connector.config.dsa_two_groups = True
    connector.kv_caches = {
        "layer0.attn": torch.zeros(1),
        "layer0.indexer.k_cache": torch.zeros(1),
    }

    connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_decode_window_save_completion_resumes_after_late_indexer_group() -> None:
    request = _make_req("req-window")
    request.is_decode_window_save = True
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    request.save_spec.can_save_indexer = True
    request.indexer_slot_mapping = [torch.arange(4, 8, dtype=torch.long)]
    connector, _, _ = _make_connector([request])
    connector.config.dsa_two_groups = True
    connector.kv_caches = {
        "layer0.attn": torch.zeros(1),
        "layer0.indexer.k_cache": torch.zeros(1),
    }

    connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
    connector.wait_for_save()
    assert connector.get_completed_decode_window_saves() == {}

    connector.save_kv_layer(
        "layer0.indexer.k_cache",
        torch.zeros(1),
        SimpleNamespace(slot_mapping=torch.arange(1, dtype=torch.long)),
    )
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {"req-window": 512}
    assert connector._decode_window_save_completed_groups == set()


def test_decode_window_save_tracks_multiple_windows_for_same_request() -> None:
    requests = []
    for window_start, window_end in ((0, 4), (4, 8)):
        request = _make_req("req-window")
        request.token_ids = list(range(8))
        request.slot_mapping = [torch.arange(8, dtype=torch.long)]
        request.is_decode_window_save = True
        request.decode_window_start = window_start
        request.decode_window_end = window_end
        request.decode_window_size = 4
        request.save_spec.can_save_indexer = True
        request.indexer_slot_mapping = [torch.arange(8, 16, dtype=torch.long)]
        requests.append(request)

    connector, _, _ = _make_connector(requests)
    connector.config.dsa_two_groups = True
    connector.kv_caches = {
        "layer0.attn": torch.zeros(1),
        "layer0.indexer.k_cache": torch.zeros(1),
    }

    connector.save_kv_layer("layer0.attn", torch.zeros(1), None)
    connector.save_kv_layer(
        "layer0.indexer.k_cache",
        torch.zeros(1),
        SimpleNamespace(slot_mapping=torch.arange(1, dtype=torch.long)),
    )
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {"req-window": 8}


def test_decode_window_reqmeta_saves_latent_and_indexer() -> None:
    tracker = RequestTracker(
        req_id="req-window",
        prompt_len=0,
        token_ids=list(range(8)),
        allocated_block_ids=[0, 1],
        allocated_block_ids_indexer=[2, 3],
    )

    req_meta = ReqMeta.from_decode_window_save(
        tracker,
        block_size=4,
        window_start=0,
        window_end=8,
        window_size=8,
    )

    assert req_meta is not None
    assert req_meta.save_spec is not None
    assert req_meta.save_spec.can_save_latent is True
    assert req_meta.save_spec.can_save_indexer is True
    assert len(req_meta.indexer_slot_mapping) == 1
    assert req_meta.indexer_slot_mapping[0].tolist() == list(range(8, 16))


def test_sparse_layerwise_prefill_saves_exact_tail_with_local_mappings() -> None:
    tracker = RequestTracker(
        req_id="req-chunked",
        prompt_len=20,
        token_ids=list(range(19)),
        allocated_block_ids=[10, 11, 12, 13, 14],
        allocated_block_ids_indexer=[20, 21, 22, 23, 24],
        num_saved_tokens=10,
    )
    # A one-token chunked-prefill update can set this heuristic too early.
    tracker.is_decode_phase = True

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=8,
        discard_partial_chunks=True,
        save_decode_cache=False,
        dsa_two_groups=True,
        windowed_sparse_layerwise_save=True,
    )

    assert req_meta is not None
    assert req_meta.save_spec is not None
    assert req_meta.save_spec.can_save is True
    assert tracker.num_saved_tokens == 19
    assert req_meta.token_ids == list(range(19))
    assert req_meta.windowed_sparse_save is True
    assert req_meta.save_slot_mapping_base == 8
    assert req_meta.save_slot_mapping[0].tolist() == list(range(48, 59))
    assert req_meta.save_indexer_slot_mapping[0].tolist() == list(range(88, 99))
    assert req_meta.slot_mapping[0] is req_meta.save_slot_mapping[0]
    assert (
        req_meta.indexer_slot_mapping[0]
        is req_meta.save_indexer_slot_mapping[0]
    )


def test_sparse_layerwise_producer_keeps_full_source_mapping() -> None:
    tracker = RequestTracker(
        req_id="req-producer",
        prompt_len=20,
        token_ids=list(range(19)),
        allocated_block_ids=[10, 11, 12, 13, 14],
        num_saved_tokens=10,
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=8,
        windowed_sparse_layerwise_save=True,
        save_entire_prefix=True,
    )

    assert req_meta is not None
    assert tracker.num_saved_tokens == 19
    assert req_meta.windowed_sparse_save is True
    assert req_meta.slot_mapping[0].numel() == 19
    assert req_meta.save_slot_mapping_base is None
    assert req_meta.save_slot_mapping == []


def test_decode_window_reqmeta_builds_only_windowed_save_slots() -> None:
    tracker = RequestTracker(
        req_id="req-window-local",
        prompt_len=0,
        token_ids=list(range(16)),
        allocated_block_ids=[10, 11, 12, 13],
        allocated_block_ids_indexer=[20, 21, 22, 23],
    )

    req_meta = ReqMeta.from_decode_window_save(
        tracker,
        block_size=4,
        window_start=8,
        window_end=16,
        window_size=8,
        windowed_sparse_layerwise_save=True,
    )

    assert req_meta is not None
    assert req_meta.slot_mapping[0].numel() == 8
    assert req_meta.indexer_slot_mapping[0].numel() == 8
    assert req_meta.save_slot_mapping_base == 8
    assert req_meta.save_slot_mapping[0].tolist() == list(range(48, 56))
    assert req_meta.save_indexer_slot_mapping[0].tolist() == list(range(88, 96))


def test_sparse_decode_reqmeta_extends_tokens_to_lmcache_hit() -> None:
    tracker = RequestTracker(
        req_id="req-sparse",
        prompt_len=5,
        token_ids=list(range(8)),
        allocated_block_ids=[0, 1],
    )
    tracker.is_decode_phase = True

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=8,
            can_load=True,
        ),
        is_sparse_decode=True,
        windowed_sparse_layerwise_save=True,
    )

    assert req_meta is not None
    assert req_meta.token_ids == list(range(8))
    assert tracker.sparse_token_ids == list(range(8))
    assert req_meta.slot_mapping[0].tolist() == list(range(8))
    assert req_meta.windowed_sparse_save is False
    assert req_meta.save_slot_mapping == []


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector._layerwise_save_storers == {}


def test_layerwise_prefill_does_not_publish_without_a_verified_store() -> None:
    request = _make_req("req-1", can_save=False)
    request.is_last_prefill = True
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(16)]
    connector, _, _ = _make_connector([request])
    connector.kv_role = "kv_both"

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_layerwise_prefill_publishes_verified_store_frontier() -> None:
    request = _make_req("req-1")
    request.is_last_prefill = True
    request.token_ids = list(range(18))
    request.slot_mapping = [torch.arange(18)]
    connector, _, engine = _make_connector([request])
    engine.store_results[request.req_id] = LayerwiseStoreResult(
        request_id=request.req_id,
        committed_end=18,
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {"req-1": 16}


def test_layerwise_prefill_does_not_publish_incomplete_store() -> None:
    request = _make_req("req-1")
    request.is_last_prefill = True
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(16)]
    connector, _, engine = _make_connector([request])
    engine.store_results[request.req_id] = LayerwiseStoreResult(
        request_id=request.req_id
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()

    assert connector.get_completed_decode_window_saves() == {}


def test_layerwise_prefill_waits_for_both_kv_groups() -> None:
    request = _make_req("req-1")
    request.is_last_prefill = True
    request.token_ids = list(range(16))
    request.save_spec.can_save_indexer = True
    connector, _, _ = _make_connector([request])
    connector.config.dsa_two_groups = True
    latent = LayerwiseStoreResult(
        request_id=request.req_id,
        kv_group=0,
        committed_end=16,
    )
    indexer = LayerwiseStoreResult(
        request_id=request.req_id,
        kv_group=1,
        committed_end=16,
    )

    connector._consume_completed_layerwise_store(request, 0, True, latent)
    connector._mark_prefill_committed(request)
    assert connector.get_completed_decode_window_saves() == {}

    connector._consume_completed_layerwise_store(request, 1, True, indexer)
    connector._mark_prefill_committed(request)
    assert connector.get_completed_decode_window_saves() == {"req-1": 16}


def test_layerwise_save_kv_producer_ignores_can_save_flag() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == 1
    assert set(connector._layerwise_save_storers.keys()) == {
        ("req-1", "normal_save", 0, 0, 4)
    }

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_passes_request_configs() -> None:
    request_configs = {"lmcache.tag.schema": "dsa-index-save-v2"}
    connector, _, engine = _make_connector(
        [_make_req("req-1", request_configs=request_configs)]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["request_configs"] == request_configs


def test_finished_worker_request_closes_abandoned_layerwise_storer() -> None:
    request = _make_req("req-1")
    connector, _, engine = _make_connector([request])
    closed = []

    def _abandoned_storer():
        try:
            while True:
                yield None
        finally:
            closed.append(True)

    storer = _abandoned_storer()
    next(storer)
    storer_key = connector._layerwise_save_storer_key(request, 0)
    connector._layerwise_save_storers[storer_key] = storer

    connector._release_finished_worker_requests({"req-1"})

    assert closed == [True]
    assert connector._layerwise_save_storers == {}
    assert engine.unpinned == ["req-1"]


def test_request_finished_returns_captured_final_hidden() -> None:
    connector, _, _ = _make_connector([_make_req("req-1")])
    connector.async_loading = False
    connector._request_trackers = {}
    connector.config = SimpleNamespace(
        dsa_two_groups=False,
        get_extra_config_value=lambda _key, default=None: default,
    )
    payload = {
        "version": 1,
        "dtype": "bfloat16",
        "shape": [4096],
        "encoding": "base64",
        "data": "AA==",
    }
    request = SimpleNamespace(
        request_id="req-1",
        status=None,
        kv_transfer_params={"ret_final_hidden": True},
        captured_final_hidden=payload,
    )

    should_free, return_params = connector.request_finished(request, [])

    assert should_free is False
    assert return_params == {"bootstrap_final_hidden": payload}


def test_deferred_latent_flush_drains_full_store_layer() -> None:
    request = _make_req("req-1")
    request.save_spec.can_save_latent = True
    connector, _, engine = _make_connector([request])
    connector.kv_role = "kv_both"
    connector._deferred_latent_pending.add(
        connector._layerwise_save_storer_key(request, 0)
    )
    connector._latent_kvcaches = [torch.zeros(1)]
    connector._kvcaches_for_group = lambda _kv_group: [torch.zeros(1)]
    connector._refresh_kvcaches_list = lambda: None
    engine.num_layers = 4
    engine.store_steps["req-1"] = 0

    def _finite_store_layer(_token_ids, **kwargs):
        engine.store_calls.append(kwargs["req_id"])

        def _storer():
            for _ in range(engine.num_layers + 1):
                engine.store_steps["req-1"] += 1
                yield None

        return _storer()

    engine.store_layer = _finite_store_layer

    connector._flush_deferred_latent_store(request, request.save_spec)

    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == engine.num_layers + 1
    assert not connector._deferred_latent_pending


def test_indexer_save_uses_layer_metadata_slots_not_request_slots() -> None:
    request = _make_req("req-1")
    request.save_spec = SaveSpec(
        skip_leading_tokens=0,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.indexer_slot_mapping = [torch.arange(100, 104, dtype=torch.long)]

    connector, _, engine = _make_connector([request])
    connector.config = SimpleNamespace(dsa_two_groups=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 204, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["kv_group"] == 1
    assert torch.equal(engine.store_kwargs[0]["slot_mapping"], metadata_slots)
    assert not torch.equal(
        engine.store_kwargs[0]["slot_mapping"],
        request.indexer_slot_mapping[0],
    )


def test_chunked_indexer_save_pads_layer_metadata_slots() -> None:
    request = _make_req("req-1")
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(16, dtype=torch.long)]
    request.save_spec = SaveSpec(
        skip_leading_tokens=8,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.indexer_slot_mapping = [torch.arange(100, 116, dtype=torch.long)]

    connector, _, engine = _make_connector([request])
    connector.kv_role = "kv_both"
    connector.config = SimpleNamespace(dsa_two_groups=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 208, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["kv_group"] == 1
    assert engine.store_kwargs[0]["offset"] == 8
    assert torch.equal(
        engine.store_kwargs[0]["slot_mapping"],
        torch.cat((torch.zeros(8, dtype=torch.long), metadata_slots)),
    )
    assert torch.equal(
        engine.store_kwargs[0]["mask"],
        torch.tensor(
            [False] * 8 + [True] * 8,
            dtype=torch.bool,
        ),
    )


def test_sparse_layerwise_indexer_save_uses_request_local_window() -> None:
    request = _make_req("req-batched")
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(16, dtype=torch.long)]
    request.save_spec = SaveSpec(
        skip_leading_tokens=8,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.windowed_sparse_save = True
    request.save_slot_mapping_base = 8
    request.save_slot_mapping = [torch.arange(500, 508, dtype=torch.long)]
    request.save_indexer_slot_mapping = [
        torch.arange(600, 608, dtype=torch.long)
    ]

    connector, _, engine = _make_connector([request])
    connector.kv_role = "kv_both"
    connector.enable_sparse_attention = True
    connector.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 208, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-batched"]
    kwargs = engine.store_kwargs[0]
    assert kwargs["kv_group"] == 1
    assert kwargs["offset"] == 8
    assert kwargs["slot_mapping_base"] == 8
    assert kwargs["windowed_sparse_save"] is True
    assert torch.equal(kwargs["slot_mapping"], request.save_indexer_slot_mapping[0])
    assert not torch.equal(kwargs["slot_mapping"], metadata_slots)


def test_sparse_layerwise_producer_indexer_uses_request_full_mapping() -> None:
    request = _make_req("req-producer-batched")
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(500, 516, dtype=torch.long)]
    request.indexer_slot_mapping = [
        torch.arange(600, 616, dtype=torch.long)
    ]
    request.save_spec = SaveSpec(
        skip_leading_tokens=0,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.windowed_sparse_save = True

    connector, _, engine = _make_connector([request])
    connector.enable_sparse_attention = True
    connector.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 216, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-producer-batched"]
    kwargs = engine.store_kwargs[0]
    assert kwargs["offset"] == 0
    assert kwargs["slot_mapping_base"] == 0
    assert kwargs["windowed_sparse_save"] is True
    assert torch.equal(kwargs["slot_mapping"], request.indexer_slot_mapping[0])
    assert not torch.equal(kwargs["slot_mapping"], metadata_slots)
