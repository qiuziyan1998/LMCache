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
    SaveSpec,
)


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

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                self.store_steps[req_id] += 1
                yield None

        return _storer()


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def _make_req(req_id: str, can_save: bool = True):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=[torch.arange(4, dtype=torch.long)],
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_sparse_decode=False,
        load_spec=None,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}
    connector._completed_decode_window_saves = {}
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
    assert set(connector._layerwise_save_storers.keys()) == {"req-2"}

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

    for _ in range(num_layers):
        connector.save_kv_layer("layer_x", torch.zeros(1), None)

    assert engine.store_steps["req-1"] == num_layers
    assert engine.store_steps["req-2"] == num_layers

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == num_layers + 1
    assert engine.store_steps["req-2"] == num_layers + 1
    assert connector._layerwise_save_storers == {}


def test_decode_window_save_completion_is_drained_after_wait() -> None:
    request = _make_req("req-window")
    request.decode_window_start = 256
    request.decode_window_end = 512
    request.decode_window_size = 256
    connector, _, _ = _make_connector([request])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert connector.get_completed_decode_window_saves() == {}

    connector.wait_for_save()
    assert connector.get_completed_decode_window_saves() == {"req-window": 512}
    assert connector.get_completed_decode_window_saves() == {}


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_kv_producer_ignores_can_save_flag() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == 1
    assert set(connector._layerwise_save_storers.keys()) == {"req-1"}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}
