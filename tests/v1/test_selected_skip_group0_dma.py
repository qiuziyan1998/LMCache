# SPDX-License-Identifier: Apache-2.0
"""Unit tests for selected-KV Group-0 prefill DMA skipping."""

# Standard
from types import SimpleNamespace as _NS
from typing import Any
from unittest.mock import MagicMock, patch
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.cache_engine import LMCacheEngine, _RemoteFillMaterializationError
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.shared_cpu_cache import SharedHandleBatch, SharedHandleEnvelope

_KW = {"deferred_layerwise_get": True, "_selected_skip_group0_dma": True}


def _nop(*_a: Any, **_k: Any) -> Any:
    return None


class _Obj:
    def __init__(self, lid: int) -> None:
        self.layer_id, self.ref_downs, self.is_pinned = lid, 0, True

    def is_valid(self) -> bool:
        return self.ref_downs == 0

    def unpin(self) -> None:
        self.is_pinned = False

    def ref_count_down(self) -> None:
        self.ref_downs += 1


def _batch(nl: int = 2, nc: int = 2, compact: bool = False) -> Any:
    ex: dict[str, Any] = {}
    if compact:
        ex = {"page_offsets": [0, 64], "page_physical_sizes": [64, 64]}
    offs, hashes = [i * 32 for i in range(nc)], list(range(1, nc + 1))
    return SharedHandleBatch("shm0", 0, nl, nc, [32] * nc, hashes, offs, **ex)


def _env(req: str, grp: int, b: SharedHandleBatch) -> SharedHandleEnvelope:
    return SharedHandleEnvelope(req, "dense_prefix", 0, 0, grp, "ok", 1, [], batch=b)


def _rel_once(objs: Any, *, unpin: bool = False, kwargs: Any = None) -> None:
    del kwargs
    for obj in list(objs):
        if unpin and getattr(obj, "is_pinned", False):
            obj.unpin()
        if obj.is_valid():
            obj.ref_count_down()


def _eng(
    *, nl: int = 3, r0: bool = True, abort: bool = False, rf: bool = False, **ex: Any
) -> Any:
    eng: Any = LMCacheEngine.__new__(LMCacheEngine)
    eng.config = LMCacheEngineConfig(
        chunk_size=128,
        local_cpu=True,
        max_local_cpu_size=0.1,
        use_layerwise=True,
        save_unfull_chunk=True,
        dsa_two_groups=True,
    )
    mon = _NS(on_retrieve_request=lambda *a, **k: 1, on_retrieve_finished=_nop)
    eng.stats_monitor = mon
    eng.metadata = _NS(indexer_c8_layout=None)
    eng.storage_manager = _NS(async_lookup_and_prefetch=_nop)
    eng.shared_cpu_cache_generation = 1
    eng.shared_cpu_cache_passive_allocator = object()
    eng.num_layers_for_group = lambda _g: nl
    eng.is_first_rank = lambda: r0
    eng._remote_fill_pair_lookup_enabled = lambda *a, **k: rf
    eng._shared_prefill_reuse_count = lambda *a, **k: 0
    eng._is_passive_shared_retrieve = lambda *a, **k: not r0
    for a in (
        "_shared_prefill_source_state",
        "_release_shared_retrieve_objs",
        "_release_retained_dense_retrieve_objs",
        "_remember_shared_prefill_sources",
        "_close_shared_retrieve_consumer",
    ):
        setattr(eng, a, _nop)
    eng.unsafe = []
    eng._retain_unsafe_layerwise_retrieve_objs = lambda o, **k: eng.unsafe.extend(o)
    if abort or rf:
        eng.gpu_connector = _NS(
            batched_to_gpu=lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
            prepare_layerwise_prefill_group0_skip=_nop,
        )
    if abort:
        eng._unsafe_layerwise_retrieve_sources = []
        eng._prepare_shared_prefill_sources = eng._broadcast_shared_envelope = _nop
        eng._adopt_dense_shared_retrieve_cache = lambda *_a, **_k: False
        eng._release_retained_dense_retrieve_objs = _rel_once
        eng._release_shared_retrieve_objs = _rel_once
    if rf:
        eng.is_healthy = lambda: True
        eng._num_transfer_layers_for_call = lambda _g, _k: nl
        eng._should_use_shared_layerwise_retrieve = lambda g: g == 0
        eng._is_passive = lambda *_a, **_k: False
        eng._get_req_id = lambda kw: str(kw["req_id"])
        eng.lookup_unpin = eng._broadcast_shared_envelope = _nop
        eng._dense_retrieve_token_results = lambda *a, **k: [(0, 128, "k0")]
        eng._remote_fill_local_full_hint = lambda _rc: "hint"
        eng._shared_layerwise_error_envelope = lambda **k: SharedHandleEnvelope(
            k["req_id"],
            k["phase"],
            k["request_ordinal"],
            k["layer_id"],
            k["kv_group"],
            "error",
            getattr(eng, "shared_cpu_cache_generation", 1),
            [],
            message=k.get("message"),
            error_details=k.get("details"),
        )
        err = _RemoteFillMaterializationError("rf")
        eng._remote_fill_retrieve_plan = lambda *a, **k: (_ for _ in ()).throw(err)
    eng.__dict__.update(ex)
    return eng


def _impl(
    *,
    kv_role: str = "kv_producer",
    sparse: bool = True,
    thresh: int = 0,
    sel: frozenset[str] = frozenset({"req-1"}),
    adv: bool = False,
    **ex: Any,
) -> Any:
    impl: Any = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.kv_role, impl.enable_sparse_attention = kv_role, sparse
    impl._lmcache_chunk_size = 256
    impl._dsa_scratch_capacity = 2048
    impl._dsa_kv_policy_threshold = thresh
    impl.lmcache_engine = _NS(
        enable_shared_cpu_cache=True,
        supports_dense_sparse_cache_retention=lambda: True,
        _should_use_shared_layerwise_retrieve=lambda g: g == 0,
    )
    impl._deferred_layerwise_prefill_load_active = True
    impl._layerwise_prefill_dma = impl._layerwise_prefill_p_node = True
    impl.enable_blending = False
    impl._layerwise_prefill_bank_count = 2
    impl._layerwise_group0_dma_skipped = set()
    impl._selected_skip_group0_dma_req_ids = sel
    impl._is_dsa_two_groups = lambda: True
    if adv:
        impl._initial_layerwise_prefill_load_submitted = False
        impl.num_layers, impl.current_layer = 2, 0
        impl._latent_layer_names = ["latent_0", "latent_1"]
        impl._indexer_layer_names = ["indexer_0", "indexer_1"]
        impl._layerwise_waited_groups = set()
        impl._deferred_layerwise_prefill_drained_groups = set()
        impl._layerwise_group0_dma_skipped = {"req-skip"}
        impl._layerwise_sparse_shared_ordered = [False]
        impl._layerwise_retriever_is_sparse = [False]
        impl._layerwise_wait_group = lambda n: 0 if n.startswith("latent_") else 1
        impl._layerwise_required_wait_groups = lambda: {0, 1}
        impl._layerwise_prefill_load_position = lambda n, _g: (int(n[-1]),) * 2 + (1,)
        impl._layerwise_prefill_slot_mapping = _nop
        impl._finalize_worker_retrieve_state_from_metadata = _nop
        impl._drain_layerwise_retrievers = _nop
        impl._sparse_retrieve_state_guard = lambda _r: MagicMock()
        impl._parent = _NS(_get_connector_metadata=lambda: None)
        impl._layerwise_requests = [
            _NS(
                req_id="req-skip",
                is_sparse_decode=False,
                block_allocation_mode="prefill_child",
                token_ids=[1, 2],
            )
        ]
    impl.__dict__.update(ex)
    return impl


def _r0_gen(eng: Any, nl: int, nc: int, rid: str, mid: int, hit: bool) -> Any:
    keys: Any = [[f"k{i}_{j}" for j in range(nc)] for i in range(nl)] if nc else []
    locs = [["LocalCPUBackend"] * nc for _ in range(nl)] if nc else []
    rm = (torch.ones if hit else torch.zeros)(max(nc, 1) * 128, dtype=torch.bool)
    kw = {**_KW, "kvcaches": ["k0"]} if nc else dict(_KW)
    st, en = [i * 128 for i in range(nc)], [(i + 1) * 128 for i in range(nc)]
    return rm, eng._retrieve_layer_shared_rank0(
        starts=st,
        ends=en,
        keys_layer_major=keys,
        chunk_locations_layer_major=locs,
        location="LocalCPUBackend",
        ret_mask=rm,
        monitor_req_id=mid,
        req_id=rid,
        kv_group=0,
        kwargs=kw,
    )


def _p_gen(eng: Any, rid: str, mid: int, grp: int) -> Any:
    rm, keys = torch.zeros(256, dtype=torch.bool), [["k0_0", "k0_1"], ["k1_0", "k1_1"]]
    return rm, eng._retrieve_layer_shared_passive(
        starts_all=[0, 128],
        ends_all=[128, 256],
        keys_layer_major=keys,
        ret_mask=rm,
        monitor_req_id=mid,
        req_id=rid,
        kv_group=grp,
        kwargs=dict(_KW),
    )


class TestSelectedSkipGroup0DMA(unittest.TestCase):
    def setUp(self) -> None:
        p = patch("lmcache.v1.cache_engine.assert_layerwise_gpu_connector", _nop)
        p.start()
        self.addCleanup(p.stop)

    def _abort_ok(self, eng: Any, ab_g: Any, ok_g: Any, objs: list[_Obj]) -> None:
        next(ab_g)
        next(ab_g)
        ab_g.close()
        self.assertEqual(len(eng.unsafe), 2)
        self.assertEqual((objs[0].ref_downs, objs[1].ref_downs), (0, 0))
        eng.unsafe.clear()
        list(ok_g)
        self.assertEqual(eng.unsafe, [])
        self.assertEqual((objs[0].ref_downs, objs[1].ref_downs), (1, 1))

    def test_rank0_hit_miss_and_retains_sources_during_skip(self) -> None:
        c: dict[str, list[Any]] = {k: [] for k in "btg skip srcs bcast rel".split()}
        owned, batch = [[_Obj(i), _Obj(i)] for i in range(3)], _batch(3, 2)
        eng = _eng(nl=3, r0=True)
        eng.gpu_connector = _NS(
            batched_to_gpu=lambda *a, **k: c["btg"].append(k),
            prepare_layerwise_prefill_group0_skip=lambda **k: c["skip"].append(k),
        )
        eng._prepare_shared_prefill_sources = lambda s, g, kw: c["srcs"].append(
            (g, kw.get("_shared_prefill_starts"), kw.get("_shared_prefill_ends"))
        )
        eng._adopt_dense_shared_retrieve_cache = lambda *_a, **_k: False
        eng._broadcast_shared_envelope = lambda env: c["bcast"].append(env)
        eng._resolve_shared_rank0_layer_mem_objs = lambda **k: owned[k["layer_id"]]
        eng._make_shared_handle_batch = lambda *a, **k: batch
        eng._release_shared_retrieve_objs = lambda o, **k: c["rel"].append(list(o))

        ret, gen = _r0_gen(eng, 3, 2, "r0-hit", 1, True)
        self.assertEqual(int(next(gen) or 0), 256)
        self.assertEqual((len(c["skip"]), c["btg"], c["rel"]), (1, [], []))
        self.assertEqual(c["srcs"], [(0, [0, 128], [128, 256])])
        self.assertIsNone(next(gen))
        self.assertEqual(c["rel"], [])
        self.assertIs(next(gen), ret)
        self.assertTrue(bool(ret.all()))
        self.assertEqual(len(c["rel"]), 1)
        self.assertRaises(StopIteration, next, gen)

        for k in ("skip", "srcs", "bcast"):
            c[k].clear()
        rm, gen_m = _r0_gen(eng, 3, 0, "r0-miss", 2, False)
        my = list(gen_m)
        self.assertEqual((len(my), int(my[0]), my[1]), (3, 0, None))
        self.assertIs(my[2], rm)
        self.assertEqual([e.status for e in c["bcast"]], ["skipped"] * 3)

    def test_passive_hit_and_group1_still_executes_normal_dma(self) -> None:
        btg_c, btg_s, skip_c = [], [], []

        def _btg(_s: Any, _e: Any, **kw: Any) -> Any:
            btg_c.append(kw)
            while True:
                item = yield None
                if item is not None:
                    btg_s.append(item)

        batch = _batch(2, 2, compact=True)
        views = (_NS(page_id=0), _NS(page_id=1))
        conn = _NS(
            batched_to_gpu=_btg,
            prepare_layerwise_prefill_group0_skip=lambda **k: skip_c.append(k),
        )
        eng = _eng(nl=2, r0=False, gpu_connector=conn)
        eng._prepare_shared_prefill_sources = _nop
        eng._adopt_dense_shared_retrieve_cache = lambda **_k: True
        eng.get_shared_cpu_request_lease = lambda _r: None
        eng._make_passive_layer_page_views = lambda *a, **k: views
        for grp, mid, ylen, bc, bs in ((0, 10, 3, 0, 0), (1, 11, 4, 1, 2)):
            env = _env("p-hit", grp, batch)
            eng._receive_matching_shared_envelope = lambda *a, e=env, **k: e
            rm, gp = _p_gen(eng, "p-hit", mid, grp)
            ys = list(gp)
            self.assertEqual((len(ys), int(ys[0])), (ylen, 256))
            self.assertEqual(ys[1:-1], [None] * (ylen - 2))
            self.assertEqual((len(btg_c), len(btg_s), len(skip_c)), (bc, bs, 1))
            self.assertIs(ys[-1], rm)

    def test_remote_fill_recompute_fallback(self) -> None:
        eng = _eng(nl=4, r0=True, rf=True)
        tok, m = torch.arange(128), torch.ones(128, dtype=torch.bool)
        err_mod = "lmcache.v1.cache_engine._RemoteFillMaterializationError"
        with patch(err_mod, _RemoteFillMaterializationError):
            ys = list(
                eng.retrieve_layer(
                    tokens=tok,
                    mask=m,
                    req_id="rf-skip",
                    kv_group=0,
                    deferred_layerwise_get=True,
                    _selected_skip_group0_dma=True,
                    request_configs={"remote_fill": "hint"},
                )
            )
        self.assertEqual((len(ys), ys[:2]), (3, [None, None]))
        y2: Any = ys[2]
        self.assertIsNotNone(y2)
        self.assertFalse(bool(y2.any()))
        self.assertIn("rf-skip", eng._recompute_fallback_req_ids)

        eng_off = _eng(nl=4, r0=True)
        fk: dict[str, Any] = {
            "ret_mask": m,
            "monitor_req_id": 10,
            "yielded_steps": 1,
            "kv_group": 0,
            "skip_group0_dma": False,
            "req_id": "rf-full",
        }
        yf = list(eng_off._remote_fill_recompute_result(**fk))
        self.assertEqual((len(yf), yf[:-1]), (5, [None] * 4))
        self.assertFalse(hasattr(eng_off, "_recompute_fallback_req_ids"))

    def test_disagg_gating_and_deferred_advance_skip(self) -> None:
        for sparse, has_hook, sel_ids, exp in (
            (False, True, frozenset({"req-p-only"}), True),
            (False, False, frozenset({"req-p-only"}), False),
            (True, True, frozenset(), False),
        ):
            conn = _NS()
            if has_hook:
                conn.prepare_layerwise_prefill_group0_skip = _nop
            impl = _impl(sparse=sparse, sel=sel_ids)
            skip = impl._should_skip_group0_prefill_dma("req-p-only", conn)
            if skip:
                impl._layerwise_group0_dma_skipped.add("req-p-only")
            kw = {"_selected_skip_group0_dma": True} if skip else {}
            self.assertEqual(skip, exp)
            self.assertEqual(bool(kw.get("_selected_skip_group0_dma")), exp)
            exp_s = {"req-p-only"} if exp else set()
            self.assertEqual(impl._layerwise_group0_dma_skipped, exp_s)

        t, one = torch.tensor(128), torch.ones(128, dtype=torch.bool)
        g0 = (v for v in (t, None, one))
        g1 = (v for v in (t, None, None, one))
        LMCacheConnectorV1Impl._prime_dense_prefix_retrievers(g0, g1)
        val: list[tuple[str, int, int]] = []
        impl2 = _impl(adv=True, layerwise_retrievers=[(g0, g1)])
        impl2._validate_dense_retrieve_result = lambda r, m, kv_group: val.append(
            (r.req_id, kv_group, int(m.sum().item()))
        )
        impl2._submit_initial_layerwise_prefill_load()
        adv = impl2._advance_deferred_layerwise_prefill_load
        self.assertTrue(impl2._initial_layerwise_prefill_load_submitted)
        for n in ("latent_0", "indexer_0"):
            adv(n, finish_current=False)
        self.assertEqual(val, [])
        adv("latent_1", finish_current=True)
        self.assertEqual(val, [("req-skip", 0, 128)])
        self.assertRaises(StopIteration, next, g0)
        g1_last: Any = next(g1)
        self.assertIsNotNone(g1_last)
        self.assertTrue(bool(g1_last.all()))

    def test_abort_after_two_primes_and_missing_hook_fails_closed(self) -> None:
        batch, owned = _batch(2, 1), [[_Obj(0)], [_Obj(1)]]
        eng = _eng(nl=2, r0=True, abort=True)
        eng._make_shared_handle_batch = lambda *a, **k: batch
        eng._resolve_shared_rank0_layer_mem_objs = lambda **k: owned[k["layer_id"]]
        ab_g = _r0_gen(eng, 2, 1, "r-ab", 3, False)[1]
        ok_g = _r0_gen(eng, 2, 1, "r-ok", 30, True)[1]
        self._abort_ok(eng, ab_g, ok_g, [owned[0][0], owned[1][0]])
        eng.gpu_connector = _NS()
        err = "prepare_layerwise_prefill_group0_skip"
        g_bad = _r0_gen(eng, 2, 1, "r-bad", 4, False)[1]
        self.assertRaisesRegex(RuntimeError, err, next, g_bad)

    def test_passive_abort_after_two_primes_retains_unsafe_and_completes(self) -> None:
        batch, owned = _batch(2, 2, compact=True), (_Obj(0), _Obj(1))
        eng = _eng(nl=2, r0=False, abort=True)
        eng._make_passive_layer_page_views = lambda *a, **k: owned
        recv = lambda *a, **k: _env(k.get("req_id", "p-ab"), 0, batch)
        eng._receive_matching_shared_envelope = recv
        ab_g = _p_gen(eng, "p-ab", 31, 0)[1]
        ok_g = _p_gen(eng, "p-ok", 32, 0)[1]
        self._abort_ok(eng, ab_g, ok_g, list(owned))

    def test_should_skip_group0_dma_kv_both_full_resident_guard(self) -> None:
        fn = "prepare_layerwise_prefill_group0_skip"
        conn = _NS(**{fn: _nop, "fence_debug": None})
        cases = (
            ("kv_producer", False, 1_000_000, 1024, True, True),
            ("kv_both", False, 0, 131072, True, False),
            ("kv_both", True, 1_000_000, 131072, True, False),
            ("kv_both", True, 0, 1024, True, False),
            ("kv_both", True, 0, 131072, False, False),
            ("kv_both", True, 0, 131072, True, True),
        )
        for role, sp, th, cached, last, exp in cases:
            impl = _impl(kv_role=role, sparse=sp, thresh=th)
            ls = _NS(lmcache_cached_tokens=cached)
            req = _NS(is_last_prefill=last, load_spec=ls)
            got = impl._should_skip_group0_prefill_dma("req-1", conn, req)
            self.assertEqual(got, exp)


if __name__ == "__main__":
    unittest.main()
