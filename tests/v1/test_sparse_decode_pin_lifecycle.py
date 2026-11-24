# SPDX-License-Identifier: Apache-2.0
"""P0: sparse decode lookup pin lifecycle across decode steps."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import WorkerRetrieveState
from tests.v1.connector_test_utils import (
    make_non_sparse_req_meta,
    make_sparse_req_meta,
    make_stub_request,
    make_worker_connector,
    make_worker_impl,
)


class TestSparseDecodePinLifecycle:
    def test_wait_for_save_defers_unpin_across_sparse_decode_steps(self) -> None:
        sparse = make_sparse_req_meta("req-sparse")
        connector, metadata, engine = make_worker_connector([sparse])

        connector.wait_for_save()
        connector.wait_for_save()
        assert engine.unpinned == []

        metadata.requests = [sparse]
        connector.wait_for_save()
        assert engine.unpinned == []

    def test_non_sparse_wait_for_save_unpins_each_step(self) -> None:
        req = make_non_sparse_req_meta("req-prefill")
        connector, metadata, engine = make_worker_connector([req], use_layerwise=True)

        connector.wait_for_save()
        assert engine.unpinned == ["req-prefill"]

        metadata.requests = [req]
        connector.wait_for_save()
        assert engine.unpinned == ["req-prefill", "req-prefill"]

    def test_scheduler_request_finished_releases_request_owned_state(self) -> None:
        sparse = make_sparse_req_meta("req-finish")
        connector, _, engine = make_worker_connector([sparse])
        connector.wait_for_save()
        assert engine.unpinned == []
        connector._drop_layerwise_save_storers = MagicMock()
        connector._drop_worker_retrieve_state = MagicMock()
        connector._cold_perf_lookup_started = {}
        connector._dsa_kv_policy_states = {"req-finish": "sparse_managed"}

        connector.request_finished(make_stub_request("req-finish"), [])
        assert engine.unpinned == ["req-finish"]
        assert connector._dsa_kv_policy_states == {}
        connector._drop_layerwise_save_storers.assert_not_called()
        connector._drop_worker_retrieve_state.assert_called_once_with("req-finish")

    def test_drop_worker_retrieve_state_releases_pins(self) -> None:
        impl = make_worker_impl()
        engine = MagicMock()
        impl._manager = type("M", (), {"lmcache_engine": engine})()

        impl._drop_worker_retrieve_state("req-drop")
        engine.lookup_unpin.assert_called_once_with("req-drop")

    def test_sparse_decode_finished_load_spec_unpins_on_wait_for_save(self) -> None:
        finished_sparse = make_sparse_req_meta("req-done", can_load=False)
        connector, _, engine = make_worker_connector([finished_sparse])

        connector.wait_for_save()
        assert engine.unpinned == ["req-done"]

    def test_only_managed_lane_keeps_pins_and_can_release_kv(self) -> None:
        """Stale load and release fields cannot evict a full-resident prefix."""
        resident = make_sparse_req_meta("resident")
        managed = make_sparse_req_meta("managed")
        for req in (resident, managed):
            assert req.load_spec is not None
            req.load_spec.dsa_full_resident = req is resident
            req.load_spec.dsa_release_frontier = req.load_spec.lmcache_cached_tokens
        connector, _, engine = make_worker_connector([resident, managed])
        connector._completed_decode_window_saves = {}

        connector.wait_for_save()

        assert engine.unpinned == ["resident"]
        assert connector.get_completed_decode_window_saves() == {"managed": 4}

    def test_decode_window_save_metadata_does_not_unpin(self) -> None:
        window_save = make_non_sparse_req_meta("req-window")
        window_save.is_decode_window_save = True
        window_save.decode_window_start = 256
        window_save.decode_window_end = 512
        window_save.decode_window_size = 256
        connector, _, engine = make_worker_connector(
            [window_save], kv_role="kv_consumer"
        )

        connector.wait_for_save()
        assert engine.unpinned == []


class TestPruneWithoutRequestFinished:
    def test_prune_keeps_stray_warm_state_until_explicit_finish(self) -> None:
        impl = make_worker_impl()
        engine = MagicMock()
        impl._manager = type("M", (), {"lmcache_engine": engine})()
        impl._worker_retrieve_state = {
            "req-active": WorkerRetrieveState(metadata_warm=True),
            "req-stray": WorkerRetrieveState(metadata_warm=True),
        }

        impl._prune_worker_retrieve_state({"req-active"})

        assert set(impl._worker_retrieve_state) == {"req-active", "req-stray"}
        engine.lookup_unpin.assert_not_called()

    def test_prune_without_request_finished_keeps_warm_cached_state(self) -> None:
        impl = make_worker_impl()
        engine = MagicMock()
        impl._manager = type("M", (), {"lmcache_engine": engine})()
        impl._worker_retrieve_state["req-finished"] = WorkerRetrieveState(
            metadata_warm=True,
            cached_keys=[["k"]],
        )

        # Next step metadata only schedules a different request.
        impl._prune_worker_retrieve_state({"req-other"})

        assert "req-finished" in impl._worker_retrieve_state
        engine.lookup_unpin.assert_not_called()
