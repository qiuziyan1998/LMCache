# SPDX-License-Identifier: Apache-2.0
"""Public rank-zero retrieval only locates and transposes new prefill chunks."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import cache_engine as engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.prefill_locations import PrefillLocationPlan
from lmcache.v1.prefill_metadata import PrefillMetadataCache
from lmcache.v1.token_database import ChunkedTokenDatabase


class PlanningEngine(LMCacheEngine):
    """Use production token/location planning with a host-only source consumer."""

    def __init__(self) -> None:
        self.config = LMCacheEngineConfig.from_legacy(
            chunk_size=4,
            backend="cpu",
            save_unfull_chunk=True,
        )
        self.num_layers = 2
        self.metadata = LMCacheMetadata(
            "model", 8, 8, 0, 0, torch.float16, (2, 1, 4, 1, 2)
        )
        self.token_database = ChunkedTokenDatabase(self.config, self.metadata)
        self.storage_manager = object()
        self.gpu_connector = object()
        self.shared_cpu_cache_generation = 7
        self.shared_cpu_cache_strict = True
        self.stats_monitor = SimpleNamespace(on_retrieve_request=lambda _: 1)
        self._shared_cpu_request_leases = {}
        self.lookups = []
        self.rows = []

    def is_healthy(self) -> bool:
        return True

    def _is_passive(self) -> bool:
        return False

    def _should_use_shared_layerwise_retrieve(self, kv_group: int) -> bool:
        return True

    def _remote_fill_retrieve_plan(self, *args: Any) -> None:
        return None

    def _find_shared_rank0_chunk_location(self, key: Any) -> str:
        self.lookups.append(key)
        return "LocalCPUBackend"

    def _retrieve_layer_shared_rank0(self, **call: Any):
        # Device/source ownership is covered by the shared-source integration
        # suite. Here the public retrieve API still runs its real planner.
        rows = [list(row) for row in call["keys_layer_major"]]
        self.rows = rows
        kwargs = call["kwargs"]
        kwargs["cached_keys"][:] = rows
        kwargs["cached_starts"][:] = call["starts"]
        kwargs["cached_ends"][:] = call["ends"]
        kwargs["cached_memory_objs"][:] = [[object()] * len(row) for row in rows]
        lease = self._shared_cpu_request_leases.setdefault(
            call["req_id"],
            SimpleNamespace(
                generation=7,
                prefill_locations={},
                source_groups={},
            ),
        )
        lease.source_groups[call["kv_group"]] = kwargs["cached_memory_objs"]
        yield call["ret_mask"].sum()
        yield call["ret_mask"]


def test_public_rank0_planning_work_is_suffix_only(monkeypatch):
    monkeypatch.setattr(engine_module, "serving_perf_enabled", lambda: False)
    monkeypatch.setattr(engine_module, "prefill_start_timing_enabled", lambda: False)
    monkeypatch.setattr(
        engine_module, "prefill_reuse_debug_enabled", lambda rank: rank == 0
    )
    reports = []
    monkeypatch.setattr(
        engine_module,
        "prefill_reuse_debug_log",
        lambda rank, stage, **values: reports.append((rank, stage, values)),
    )
    engine, cache = PlanningEngine(), PrefillMetadataCache()
    state = {
        name: []
        for name in (
            "cached_keys",
            "cached_starts",
            "cached_ends",
            "cached_memory_objs",
        )
    }
    for total, expected_lookups in (
        (4000, 2000),
        (4004, 2),
        (4004, 0),
        (4006, 2),
        (4008, 2),
    ):
        engine.lookups.clear()
        result = list(
            engine.retrieve_layer(
                list(range(total)),
                req_id="r",
                _retain_shared_dense_cache=True,
                _reuse_shared_dense_prefix=True,
                deferred_layerwise_get=True,
                _prefill_metadata_cache=cache,
                _prefill_skip_tokens=0,
                **state,
            )
        )
        assert result[-1].tolist() == [True] * total
        assert len(engine.lookups) == expected_lookups
        expected = list(engine.token_database.process_tokens(tokens=list(range(total))))
        assert engine.rows == [
            [key.get_layer(layer) for _, _, key in expected] for layer in range(2)
        ]
        assert [entry[1] for entry in reports[-2:]] == ["plan", "src"]
        assert reports[-2][2]["k"] == expected_lookups
    # An unrelated replacement of the source rows is not trusted as ownership.
    state["cached_memory_objs"] = [list(row) for row in state["cached_memory_objs"]]
    engine.lookups.clear()
    list(
        engine.retrieve_layer(
            list(range(4008)),
            req_id="r",
            _reuse_shared_dense_prefix=True,
            deferred_layerwise_get=True,
            _prefill_metadata_cache=cache,
            **state,
        )
    )
    assert len(engine.lookups) == 2004


def test_location_plan_preserves_rows_and_replaces_only_tail():
    plan = PrefillLocationPlan(2)
    plan.append(0, 4, ["a0", "a1"], ["Local", "Local"], page=True)
    plan.append(4, 6, ["b0", "b1"], ["Remote", "Remote"], page=False)
    old_rows = tuple(plan.layer_keys)
    assert plan.location == "mixed"
    plan.truncate(1)
    plan.append(4, 8, ["c0", "c1"], ["Local", "Local"], page=True)
    assert all(
        current is old
        for current, old in zip(plan.layer_keys, old_rows, strict=True)
    )
    assert list(map(list, plan.key_rows())) == [["a0", "c0"], ["a1", "c1"]]
    assert plan.location == "Local" and plan.page_chunks == 2
    with pytest.raises(ValueError, match="prefix"):
        plan.truncate(3)
