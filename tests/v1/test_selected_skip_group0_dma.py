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
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    LMCacheEngineBuilder,
    _RemoteFillMaterializationError,
)
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
    eng._shared_prefill_cached_prefix = lambda *a, **k: 0
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
            synchronize_selected_group0_skip=_nop,
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

    def _abort_ok(
        self,
        eng: Any,
        ab_g: Any,
        ok_fn: Any,
        ab_objs: list[_Obj],
        *,
        expect_unpin: bool,
    ) -> None:
        next(ab_g)
        next(ab_g)
        ab_g.close()
        self.assertEqual(eng.unsafe, [])
        self.assertEqual([o.ref_downs for o in ab_objs], [1, 1])
        if expect_unpin:
            self.assertTrue(all(not o.is_pinned for o in ab_objs))
        ok_objs, ok_g = ok_fn()
        list(ok_g)
        self.assertEqual(eng.unsafe, [])
        self.assertEqual([o.ref_downs for o in ok_objs], [1, 1])

    def test_rank0_hit_miss_and_retains_sources_during_skip(self) -> None:
        c: dict[str, list[Any]] = {k: [] for k in "btg skip srcs bcast rel".split()}
        owned, batch = [[_Obj(i), _Obj(i)] for i in range(3)], _batch(3, 2)
        eng = _eng(nl=3, r0=True)
        eng.gpu_connector = _NS(
            batched_to_gpu=lambda *a, **k: c["btg"].append(k),
            prepare_layerwise_prefill_group0_skip=lambda **k: c["skip"].append(k),
            synchronize_selected_group0_skip=_nop,
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
            synchronize_selected_group0_skip=_nop,
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
        self.assertIn("rf-full", eng_off._recompute_fallback_req_ids)

        eng0 = _eng(nl=2, r0=True, abort=True)
        list(_r0_gen(eng0, 2, 0, "r0-empty", 12, False)[1])
        self.assertIn("r0-empty", eng0._recompute_fallback_req_ids)

        eng_p = _eng(nl=2, r0=False, abort=True)
        eng_p._receive_matching_shared_envelope = lambda *a, **k: SharedHandleEnvelope(
            "p-empty",
            "dense_prefix",
            0,
            int(k["layer_id"]),
            0,
            "skipped",
            1,
            [],
        )
        list(_p_gen(eng_p, "p-empty", 13, 0)[1])
        self.assertIn("p-empty", eng_p._recompute_fallback_req_ids)

    def test_disagg_gating_and_deferred_advance_skip(self) -> None:
        for sparse, has_hook, sel_ids, exp in (
            (False, True, frozenset({"req-p-only"}), True),
            (False, False, frozenset({"req-p-only"}), False),
            (True, True, frozenset(), False),
        ):
            conn = _NS()
            if has_hook:
                conn.prepare_layerwise_prefill_group0_skip = _nop
                conn.synchronize_selected_group0_skip = _nop
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
        sync_calls: list[int] = []
        eng = _eng(nl=2, r0=True, abort=True)
        eng.gpu_connector.synchronize_selected_group0_skip = lambda: sync_calls.append(
            1
        )
        eng._make_shared_handle_batch = lambda *a, **k: batch
        eng._resolve_shared_rank0_layer_mem_objs = lambda **k: owned[k["layer_id"]]
        ab_g = _r0_gen(eng, 2, 1, "r-ab", 3, False)[1]
        ab_objs = [owned[0][0], owned[1][0]]

        def _ok_r0() -> tuple[list[_Obj], Any]:
            ok_owned = [[_Obj(0)], [_Obj(1)]]
            eng._resolve_shared_rank0_layer_mem_objs = lambda **k: ok_owned[
                k["layer_id"]
            ]
            return [ok_owned[0][0], ok_owned[1][0]], _r0_gen(
                eng, 2, 1, "r-ok", 30, True
            )[1]

        self._abort_ok(eng, ab_g, _ok_r0, ab_objs, expect_unpin=True)
        self.assertEqual(sync_calls, [1])
        eng.gpu_connector = _NS()
        err = "prepare_layerwise_prefill_group0_skip"
        g_bad = _r0_gen(eng, 2, 1, "r-bad", 4, False)[1]
        self.assertRaisesRegex(RuntimeError, err, next, g_bad)

    def test_passive_abort_after_two_primes_fences_and_completes(self) -> None:
        batch, owned = _batch(2, 2, compact=True), (_Obj(0), _Obj(1))
        sync_calls: list[int] = []
        eng = _eng(nl=2, r0=False, abort=True)
        eng.gpu_connector.synchronize_selected_group0_skip = lambda: sync_calls.append(
            1
        )
        eng._make_passive_layer_page_views = lambda *a, **k: owned
        recv = lambda *a, **k: _env(k.get("req_id", "p-ab"), 0, batch)
        eng._receive_matching_shared_envelope = recv
        ab_g = _p_gen(eng, "p-ab", 31, 0)[1]

        def _ok_p() -> tuple[list[_Obj], Any]:
            ok_owned = (_Obj(0), _Obj(1))
            eng._make_passive_layer_page_views = lambda *a, **k: ok_owned
            return list(ok_owned), _p_gen(eng, "p-ok", 32, 0)[1]

        self._abort_ok(eng, ab_g, _ok_p, list(owned), expect_unpin=False)
        self.assertEqual(sync_calls, [1])

        owned_no_hook = (_Obj(0), _Obj(1))
        eng_no_hook = _eng(nl=2, r0=False, abort=True)
        eng_no_hook._make_passive_layer_page_views = lambda *a, **k: owned_no_hook
        eng_no_hook._receive_matching_shared_envelope = lambda *a, **k: _env(
            k.get("req_id", "p-nohook"), 0, batch
        )
        ab_no_hook = _p_gen(eng_no_hook, "p-nohook", 35, 0)[1]
        next(ab_no_hook)
        next(ab_no_hook)
        delattr(eng_no_hook.gpu_connector, "synchronize_selected_group0_skip")
        with self.assertRaisesRegex(RuntimeError, "synchronize_selected_group0_skip"):
            ab_no_hook.close()
        self.assertEqual(len(eng_no_hook.unsafe), 2)
        self.assertEqual([o.ref_downs for o in owned_no_hook], [0, 0])

    def test_abort_with_hung_sync_hook_isolates_unsafe_objects(self) -> None:
        def _hung() -> None:
            raise RuntimeError("npu stream hung")

        batch0, owned0 = _batch(2, 1), [[_Obj(0)], [_Obj(1)]]
        eng0 = _eng(nl=2, r0=True, abort=True)
        eng0.gpu_connector.synchronize_selected_group0_skip = _hung
        eng0._make_shared_handle_batch = lambda *a, **k: batch0
        eng0._resolve_shared_rank0_layer_mem_objs = lambda **k: owned0[k["layer_id"]]
        ab_r0 = _r0_gen(eng0, 2, 1, "r-hung", 33, False)[1]
        next(ab_r0)
        next(ab_r0)
        with self.assertRaisesRegex(RuntimeError, "npu stream hung"):
            ab_r0.close()
        self.assertEqual(len(eng0.unsafe), 2)
        self.assertEqual([o.ref_downs for o in (owned0[0][0], owned0[1][0])], [0, 0])

        batch_p, owned_p = _batch(2, 2, compact=True), (_Obj(0), _Obj(1))
        eng_p = _eng(nl=2, r0=False, abort=True)
        eng_p.gpu_connector.synchronize_selected_group0_skip = _hung
        eng_p._make_passive_layer_page_views = lambda *a, **k: owned_p
        eng_p._receive_matching_shared_envelope = lambda *a, **k: _env(
            k.get("req_id", "p-hung"), 0, batch_p
        )
        ab_p = _p_gen(eng_p, "p-hung", 34, 0)[1]
        next(ab_p)
        next(ab_p)
        with self.assertRaisesRegex(RuntimeError, "npu stream hung"):
            ab_p.close()
        self.assertEqual(len(eng_p.unsafe), 2)
        self.assertEqual([o.ref_downs for o in owned_p], [0, 0])

    def test_recompute_fallback_skips_save_and_engine_close_retains_dma(self) -> None:
        stale_closed: list[bool] = []
        created: list[tuple[str, int]] = []

        def _mk_storer(req: Any, _save_spec: Any = None, kv_group: int = 0) -> Any:
            created.append((req.req_id, kv_group))
            return _NS(close=_nop)

        impl = _impl(
            use_layerwise=True,
            _kvcaches_for_group=lambda _g: True,
            _layerwise_save_storer_key=lambda req, grp: (req.req_id, 0, grp),
            _layerwise_save_storers={
                ("req-bad", 0, 0): _NS(close=lambda: stale_closed.append(True))
            },
            _layerwise_prefill_prepared_storer_keys={("req-bad", 0, 0)},
            _deferred_latent_pending={},
            _create_p_node_layerwise_save_storer=_mk_storer,
            _consume_completed_layerwise_store=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not consume storer on recompute fallback")
            ),
        )
        impl.lmcache_engine._recompute_fallback_req_ids = {"req-bad"}
        req_bad = _NS(
            req_id="req-bad",
            token_ids=[1, 2],
            save_spec=_NS(can_save=True, can_save_latent=True, can_save_indexer=True),
            block_allocation_mode="prefill_child",
        )
        req_ok = _NS(
            req_id="req-ok",
            token_ids=[1, 2],
            save_spec=_NS(can_save=True, can_save_latent=True, can_save_indexer=True),
            block_allocation_mode="prefill_child",
        )
        impl._prepare_p_node_layerwise_save_storers(_NS(requests=[req_bad, req_ok]))
        self.assertEqual(stale_closed, [True])
        self.assertNotIn(("req-bad", 0, 0), impl._layerwise_save_storers)
        self.assertNotIn(
            ("req-bad", 0, 0), impl._layerwise_prefill_prepared_storer_keys
        )
        self.assertEqual(created, [("req-ok", 0), ("req-ok", 1)])

        closed: list[str] = []
        eng_c: Any = LMCacheEngine.__new__(LMCacheEngine)
        eng_c.lmcache_worker = _NS(close=lambda: closed.append("worker"))
        eng_c.storage_manager = _NS(close=lambda: closed.append("sm"))
        eng_c.shared_cpu_cache_mapping = _NS(close=lambda: closed.append("map"))
        eng_c._release_all_shared_cpu_request_leases = lambda: closed.append("leases")
        eng_c._retain_unfenced_external_dma = True
        eng_c.close()
        self.assertEqual(closed, ["worker"])
        closed.clear()
        eng_c._retain_unfenced_external_dma = False
        eng_c._unsafe_layerwise_retrieve_sources = [object()]
        eng_c.close()
        self.assertEqual(closed, ["worker"])
        closed.clear()
        eng_c._unsafe_layerwise_retrieve_sources = []
        eng_c.close()
        self.assertEqual(closed, ["worker", "leases", "sm", "map"])

        iid = "test-unsafe-sources-retain-destroy"

        def _cleanup_builder() -> None:
            LMCacheEngineBuilder._instances.pop(iid, None)
            LMCacheEngineBuilder._cfgs.pop(iid, None)
            LMCacheEngineBuilder._metadatas.pop(iid, None)
            LMCacheEngineBuilder._stat_loggers.pop(iid, None)

        self.addCleanup(_cleanup_builder)
        eng_b: Any = LMCacheEngine.__new__(LMCacheEngine)
        eng_b.lmcache_worker = None
        eng_b.storage_manager = None
        eng_b.shared_cpu_cache_mapping = None
        eng_b._release_all_shared_cpu_request_leases = _nop
        eng_b._retain_unfenced_external_dma = False
        eng_b._unsafe_layerwise_retrieve_sources = [object()]
        meta_none: Any = None
        stat_logger_stub: Any = _NS(shutdown=_nop)
        reqs_live: list[Any] = [req_bad, req_ok]
        LMCacheEngineBuilder._instances[iid] = eng_b
        LMCacheEngineBuilder._cfgs[iid] = None
        LMCacheEngineBuilder._metadatas[iid] = meta_none
        LMCacheEngineBuilder._stat_loggers[iid] = stat_logger_stub
        LMCacheEngineBuilder.destroy(iid)
        self.assertIn(iid, LMCacheEngineBuilder._instances)

        created_live: list[tuple[str, int]] = []
        consumed_live: list[tuple[str, int]] = []
        committed_live: list[str] = []
        unpinned_live: list[str] = []

        def _mk_live_storer(req: Any, _save_spec: Any = None, kv_group: int = 0) -> Any:
            created_live.append((req.req_id, kv_group))

            def _gen() -> Any:
                while True:
                    yield None

            g = _gen()
            next(g)
            return g

        meta_live = LMCacheConnectorMetadata(requests=reqs_live)
        eng_live: Any = LMCacheEngine.__new__(LMCacheEngine)
        eng_live._record_recompute_fallback_req("req-bad")
        stale_fb_closed: list[bool] = []

        def _mk_stale_fb_gen() -> Any:
            try:
                yield None
            finally:
                stale_fb_closed.append(True)

        stale_fb_gen = _mk_stale_fb_gen()
        next(stale_fb_gen)
        impl_live = _impl(
            use_layerwise=True,
            config=_NS(dsa_two_groups=True),
            _latent_layer_names=["latent_0"],
            _indexer_layer_names=["indexer_0"],
            kv_role="kv_producer",
            kv_caches={"latent_0": torch.empty(1)},
            _kvcaches_list=[torch.empty(1)],
            _refresh_kvcaches_list=_nop,
            _kvcaches_for_group=lambda _g: [torch.empty(1)],
            _note_decode_window_save_seen=_nop,
            _should_defer_latent_save_under_tp=lambda: False,
            _layerwise_prefill_p_node=True,
            _layerwise_prefill_dma=True,
            _layerwise_group_ordinals={0: {"latent_0": 0}, 1: {"indexer_0": 0}},
            _layerwise_prefill_pending_store_finishes={("req-bad", 0, 0): "latent_0"},
            _layerwise_prefill_prepared_storer_keys=set(),
            _layerwise_save_storer_key=lambda req, grp: (req.req_id, 0, grp),
            _layerwise_save_storers={("req-bad", 0, 0): stale_fb_gen},
            _decode_window_save_completed_groups=set(),
            _decode_window_save_expected_start={},
            _prefill_save_completed_groups={("req-bad", 0): True},
            _deferred_latent_pending=set(),
            _create_p_node_layerwise_save_storer=_mk_live_storer,
            _store_result_from_yield=lambda y: y,
            _finalize_layerwise_storer=lambda _s: (True, None),
            _consume_completed_layerwise_store=lambda req, grp, *_a: (
                consumed_live.append((req.req_id, grp))
            ),
            _is_decode_window_save_request=lambda _r: False,
            _mark_prefill_committed=lambda req: committed_live.append(req.req_id),
            _mark_initial_sparse_release_ready=_nop,
            _maybe_lookup_unpin_for_request=lambda req: unpinned_live.append(
                req.req_id
            ),
            _finish_save_batch=_nop,
            _complete_worker_save_step=_nop,
            _parent=_NS(
                _connector_metadata=meta_live,
                _get_connector_metadata=lambda: meta_live,
            ),
        )
        impl_live.lmcache_engine = eng_live
        impl_live._sync_recompute_fallback_from_engine(clear_engine=True)
        self.assertEqual(eng_live._recompute_fallback_req_ids, set())
        self.assertEqual(impl_live._step_recompute_fallback_req_ids, {"req-bad"})
        impl_live.save_kv_layer("latent_0", torch.empty(1), None)
        self.assertEqual(created_live, [("req-ok", 0)])
        self.assertEqual(stale_fb_closed, [True])
        self.assertNotIn(("req-bad", 0, 0), impl_live._layerwise_save_storers)
        self.assertNotIn(
            ("req-bad", 0, 0), impl_live._layerwise_prefill_pending_store_finishes
        )
        impl_live._layerwise_prefill_pending_store_finishes[("req-bad", 0, 0)] = (
            "latent_0"
        )
        impl_live._prefill_save_completed_groups[("req-bad", 0)] = True
        impl_live.finish_layerwise_prefill_save("latent_0")
        self.assertEqual(impl_live._layerwise_prefill_pending_store_finishes, {})
        self.assertNotIn(("req-bad", 0), impl_live._prefill_save_completed_groups)
        impl_live.wait_for_save()
        self.assertEqual(unpinned_live, ["req-bad", "req-ok"])
        self.assertEqual(consumed_live, [("req-ok", 0)])
        self.assertEqual(committed_live, ["req-ok"])
        self.assertEqual(impl_live._step_recompute_fallback_req_ids, set())

    def test_should_skip_group0_dma_kv_both_full_resident_guard(self) -> None:
        fn = "prepare_layerwise_prefill_group0_skip"
        conn = _NS(
            **{fn: _nop, "synchronize_selected_group0_skip": _nop, "fence_debug": None}
        )
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

    def test_selected_kv_store_abort_fence_smoke(self) -> None:
        # Standard
        from contextlib import nullcontext

        # First Party
        import lmcache.v1.gpu_connector.gpu_connectors as gc_mod

        events: list[str] = []
        mem0, mem1 = _Obj(0), _Obj(1)
        mem0.tensor = mem1.tensor = torch.zeros((2, 4, 8), dtype=torch.float32)  # type: ignore[attr-defined]
        tmp: Any = _NS(
            tensor=torch.zeros((2, 4, 8), dtype=torch.float32),
            metadata=_NS(fmt=None, cached_positions=None),
            ref_count=1,
            ref_count_up=lambda: events.append("tmp_up"),
            ref_count_down=lambda: events.append("tmp_down"),
        )
        fake_ops = _NS(
            TransferDirection=_NS(D2H="D2H"),
            single_layer_kv_transfer=lambda *_a, **_k: None,
        )
        streams = [
            _NS(synchronize=lambda: None, wait_stream=lambda _s: None),
            _NS(
                synchronize=lambda: (_ for _ in ()).throw(RuntimeError("hung store")),
                wait_stream=lambda _s: None,
            ),
        ]
        with patch(
            "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.Stream",
            side_effect=lambda: streams.pop(0),
        ):
            conn: Any = gc_mod.VLLMPagedMemLayerwiseGPUConnector(
                hidden_dim_size=8,
                num_layers=2,
                use_gpu=True,
                chunk_size=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        conn.gpu_kv_format = 1
        conn.gpu_buffer_allocator = _NS(allocate=lambda *_a, **_k: tmp)
        cur = _NS(wait_stream=lambda _s: None)
        with (
            patch(
                "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.current_stream",
                return_value=cur,
            ),
            patch(
                "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.stream",
                side_effect=lambda _s: nullcontext(),
            ),
            patch(
                "lmcache.v1.gpu_connector.gpu_connectors.lmc_ops",
                fake_ops,
                create=True,
            ),
        ):
            gen = conn.batched_from_gpu(
                [[mem0], [mem1]],
                [0],
                [4],
                kvcaches=[torch.zeros((4, 2, 8), dtype=torch.float32)] * 2,
                slot_mapping=torch.arange(4, dtype=torch.long),
                sync=True,
            )
            next(gen)
            with self.assertRaisesRegex(RuntimeError, "hung store"):
                gen.close()
        self.assertEqual((mem0.ref_downs, mem1.ref_downs), (0, 0))
        self.assertNotIn("tmp_down", events)
        self.assertEqual(len(conn._layerwise_prefill_unfenced_stores), 1)

    def test_selected_group0_omit_skip_clamps_producer_layerwise_store(self) -> None:
        # First Party
        from lmcache.integration.vllm.vllm_v1_adapter import (
            SaveSpec,
            WorkerRetrieveState,
        )
        from lmcache.v1.prefill_metadata import PrefillMetadataCache

        def _pmc_for_req(_req: Any, st: Any) -> PrefillMetadataCache:
            if st.prefill_metadata_cache is None:
                st.prefill_metadata_cache = PrefillMetadataCache()
            return st.prefill_metadata_cache

        impl: Any = _impl(
            kv_role="kv_producer",
            device=torch.device("cpu"),
            _lmcache_chunk_size=256,
            _layerwise_store_kwargs=lambda req, kv_group: (
                LMCacheConnectorV1Impl._layerwise_store_kwargs(impl, req, kv_group)
            ),
            _is_decode_window_save_request=lambda req: bool(
                getattr(req, "is_decode_window_save", False)
            ),
            _is_recompute_fallback_request=lambda rid: bool(
                LMCacheConnectorV1Impl._is_recompute_fallback_request(impl, rid)
            ),
            _selected_group0_omit_skip_tokens=lambda req: int(
                LMCacheConnectorV1Impl._selected_group0_omit_skip_tokens(impl, req)
            ),
            _prefill_metadata_cache_enabled=lambda *_a, **_k: True,
            _prefill_metadata_scope_changed=lambda *_a, **_k: False,
            _worker_retrieve_state={},
            _set_worker_retrieve_state=lambda rid, st: (
                impl._worker_retrieve_state.__setitem__(rid, st)
            ),
            _prefill_metadata_cache_for_request=_pmc_for_req,
            _windowed_sparse_save_mapping=lambda *_a, **_k: None,
            _step_recompute_fallback_req_ids=set(),
            _selected_group0_omit_end_tokens={
                "req-p": 512,
                "req-full": 512,
                "req-fb": 512,
                "req-both": 512,
            },
        )
        impl.lmcache_engine = _NS(
            _recompute_fallback_req_ids=set(),
            _should_use_shared_layerwise_retrieve=lambda g: g == 0,
        )

        # kv_producer prefill request with omit_end=512 (chunk_size=256)
        req_p: Any = _NS(
            req_id="req-p",
            token_ids=list(range(768)),
            slot_mapping=[torch.arange(768, dtype=torch.long)],
            is_sparse_decode=False,
            is_decode_window_save=False,
            request_configs=None,
            decode_window_start=None,
            decode_window_end=None,
            decode_window_size=None,
            load_spec=None,
            can_load=False,
        )
        out_g0 = impl._prepare_layerwise_store_inputs(req_p, None, 0)
        self.assertIsNotNone(out_g0)
        assert out_g0 is not None
        _, _, mask_g0, skip_g0, kw_g0, win_g0 = out_g0
        self.assertFalse(win_g0)
        self.assertEqual(skip_g0, 512)
        self.assertFalse(bool(mask_g0[:512].any()))
        self.assertTrue(bool(mask_g0[512:].all()))
        self.assertIsInstance(kw_g0["_prefill_metadata_cache"], PrefillMetadataCache)
        self.assertEqual(kw_g0["_prefill_skip_tokens"], 512)
        self.assertEqual(kw_g0["_unverified_prefix_skip_tokens"], 512)

        # kv_group=1 is NOT clamped by Group-0 omit_skip
        out_g1 = impl._prepare_layerwise_store_inputs(req_p, None, 1)
        self.assertIsNotNone(out_g1)
        assert out_g1 is not None
        self.assertEqual(out_g1[3], 0)
        self.assertTrue(bool(out_g1[2][:512].all()))
        self.assertEqual(out_g1[4]["_prefill_skip_tokens"], 0)
        self.assertNotIn("_unverified_prefix_skip_tokens", out_g1[4])

        # omit_skip >= len(token_ids) skips saving completely
        req_full: Any = _NS(
            **{
                **req_p.__dict__,
                "req_id": "req-full",
                "token_ids": list(range(512)),
                "slot_mapping": [torch.arange(512, dtype=torch.long)],
            }
        )
        self.assertIsNone(impl._prepare_layerwise_store_inputs(req_full, None, 0))

        # Recompute fallback and decode/window requests bypass omit_skip
        impl._step_recompute_fallback_req_ids = {"req-fb"}
        req_fb: Any = _NS(**{**req_p.__dict__, "req_id": "req-fb"})
        out_fb = impl._prepare_layerwise_store_inputs(req_fb, None, 0)
        self.assertIsNotNone(out_fb)
        assert out_fb is not None
        self.assertEqual(out_fb[3], 0)
        self.assertNotIn("_unverified_prefix_skip_tokens", out_fb[4])

        req_dec: Any = _NS(**{**req_p.__dict__, "is_sparse_decode": True})
        self.assertEqual(impl._selected_group0_omit_skip_tokens(req_dec), 0)
        req_win: Any = _NS(**{**req_p.__dict__, "is_decode_window_save": True})
        self.assertEqual(impl._selected_group0_omit_skip_tokens(req_win), 0)

        # kv_both clamps skip_leading_tokens without _unverified_prefix_skip_tokens
        impl.kv_role = "kv_both"
        req_both: Any = _NS(**{**req_p.__dict__, "req_id": "req-both"})
        impl._worker_retrieve_state["req-both"] = WorkerRetrieveState(
            req_id="req-both", prefill_metadata_cache=PrefillMetadataCache()
        )
        out_both = impl._prepare_layerwise_store_inputs(
            req_both, SaveSpec(skip_leading_tokens=256, can_save=True), 0
        )
        self.assertIsNotNone(out_both)
        assert out_both is not None
        self.assertEqual(out_both[3], 512)
        self.assertFalse(bool(out_both[2][:512].any()))
        self.assertEqual(out_both[4]["_prefill_skip_tokens"], 512)
        self.assertNotIn("_unverified_prefix_skip_tokens", out_both[4])

    def test_selected_skip_group0_dma_with_prefill_metadata_and_passive_views(
        self,
    ) -> None:
        # Standard
        import importlib

        # First Party
        from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
        from lmcache.v1.memory_management import MemoryFormat, TensorMemoryObj
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache.v1.prefill_metadata import PrefillMetadataCache
        from lmcache.v1.shared_cpu_cache import SharedSlabMapping
        from lmcache.v1.token_database import ChunkedTokenDatabase

        cfg = LMCacheEngineConfig(
            chunk_size=4,
            local_cpu=True,
            max_local_cpu_size=0.1,
            use_layerwise=True,
            save_unfull_chunk=True,
            dsa_two_groups=True,
        )
        g0 = KVLayerGroupInfo(
            ["latent_0", "latent_1"],
            [0, 1],
            (1, 2, 4, 8),
            torch.bfloat16,  # type: ignore[arg-type]
        )
        g1 = KVLayerGroupInfo(
            ["indexer_0", "indexer_1"],
            [2, 3],
            (1, 2, 4, 4),
            torch.bfloat16,  # type: ignore[arg-type]
        )
        meta = LMCacheMetadata(
            model_name="test-model",
            world_size=2,
            local_world_size=2,
            worker_id=1,
            local_worker_id=1,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 1, 4, 1, 8),
            use_mla=True,
            chunk_size=4,
            engine_id="eng0",
            kv_layer_groups_manager=KVLayerGroupsManager([g0, g1]),
            runtime_kv_group_layer_names=(
                ("latent_0", "latent_1"),
                ("indexer_0", "indexer_1"),
            ),
            runtime_kv_group_layer_counts=(2, 2),
        )
        db = ChunkedTokenDatabase(cfg, meta)
        toks = list(range(100, 108))
        pre = list(db.process_tokens(tokens=toks, make_key=False))
        chunk_hashes = [int(h) for _, _, h in pre]

        buf = torch.zeros(16384, dtype=torch.uint8)
        slab_map = SharedSlabMapping(
            shm_name="shm-sel-skip-meta-passive",
            size=16384,
            ptr=int(buf.data_ptr()),
            tensor=buf,
            generation=1,
            owner=True,
        )

        def _close_slab() -> None:
            with patch.object(SharedSlabMapping, "unlink", return_value=None):
                slab_map.close()

        self.addCleanup(_close_slab)
        passive_alloc = slab_map.passive_allocator(reuse_prefill=True)
        compact_batch = SharedHandleBatch(
            shm_name="shm-sel-skip-meta-passive",
            producer_rank=0,
            num_layers=2,
            num_chunks=2,
            physical_sizes=[64, 64],
            chunk_hashes=chunk_hashes,
            offsets=[0, 64, 128, 192],
        )
        env = SharedHandleEnvelope(
            request_id="req-pm-skip",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
            status="ok",
            generation=1,
            handles=[],
            batch=compact_batch,
        )

        prepared_records: list[tuple[list[Any], list[list[int]]]] = []
        skip_calls: list[dict[str, Any]] = []

        def _prep_ptrs(
            sources: list[Any],
            host_rows: list[list[int]],
            device_rows: list[Any],
            **_k: Any,
        ) -> None:
            host_rows[:] = [
                [
                    int(obj.data_ptr() if callable(obj.data_ptr) else obj.data_ptr)
                    for obj in row
                ]
                for row in sources
            ]
            device_rows[:] = [None] * len(sources)
            prepared_records.append((sources, [list(r) for r in host_rows]))

        conn: Any = _NS(
            get_shape=lambda nt: torch.Size([1, nt, 8]),
            prepare_layerwise_prefill_source_pointers=_prep_ptrs,
            prepare_layerwise_prefill_group0_skip=lambda **kw: skip_calls.append(kw),
            synchronize_selected_group0_skip=_nop,
            batched_to_gpu=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("must not run DMA when group0 DMA is skipped")
            ),
        )
        eng: Any = LMCacheEngine.__new__(LMCacheEngine)
        eng.config = cfg
        eng.metadata = meta
        eng.token_database = db
        eng.gpu_connector = conn
        eng.stats_monitor = _NS(
            on_retrieve_request=lambda *_a, **_k: 1, on_retrieve_finished=_nop
        )
        eng.shared_cpu_cache_generation = 1
        eng.shared_cpu_cache_passive_allocator = passive_alloc
        eng._shared_cpu_request_leases = {}
        eng.get_shared_cpu_request_lease = lambda _r: None
        eng._unsafe_layerwise_retrieve_sources = []
        eng._shared_prefill_source_state = lambda *_a, **_k: None
        eng._cached_shared_prefill_source = lambda *_a, **_k: None
        eng._validate_shared_layerwise_envelope = _nop
        eng._close_shared_retrieve_consumer = _nop
        eng._remember_shared_prefill_sources = _nop
        eng.is_healthy = lambda: True
        eng._num_transfer_layers_for_call = lambda _g, _k: 2
        eng.num_layers_for_group = lambda _g: 2
        eng._should_use_shared_layerwise_retrieve = lambda g: g == 0
        eng._is_passive = lambda *_a, **_k: True
        eng._get_req_id = lambda kw: str(kw["req_id"])
        eng._remote_fill_pair_lookup_enabled = lambda: False
        eng._remote_fill_retrieve_plan = _nop
        eng._adopt_dense_shared_retrieve_cache = lambda *_a, **_k: False
        eng._receive_matching_shared_envelope = lambda *_a, **_k: env
        eng._memory_format_for_kv_group = lambda _g: MemoryFormat.KV_2LTD
        eng._shared_cpu_dtype_for_kv_group = lambda _g: torch.bfloat16

        pmc = PrefillMetadataCache()
        call_kw: dict[str, Any] = {
            "req_id": "req-pm-skip",
            "kv_group": 0,
            "deferred_layerwise_get": True,
            "_selected_skip_group0_dma": True,
            "_prefill_metadata_cache": pmc,
            "_reuse_shared_dense_prefix": True,
        }
        gen = eng.retrieve_layer(tokens=toks, mask=None, **call_kw)
        y0 = next(gen)
        self.assertEqual(int(y0 or 0), 8)
        self.assertEqual(len(skip_calls), 1)
        self.assertEqual(len(prepared_records), 1)
        srcs, host_rows = prepared_records[0]
        self.assertEqual(len(srcs), 2)
        for row, h_row in zip(srcs, host_rows, strict=True):
            self.assertEqual(len(row), 2)
            self.assertTrue(all(isinstance(obj, TensorMemoryObj) for obj in row))
            self.assertTrue(all(obj.get_ref_count() == 1 for obj in row))
            self.assertTrue(all(ptr > 0 for ptr in h_row))
        self.assertIsNone(next(gen))
        ret_mask = next(gen)
        self.assertIsInstance(ret_mask, torch.Tensor)
        self.assertTrue(bool(ret_mask.all()))
        with self.assertRaises(StopIteration):
            next(gen)
        # Normal completion releases real PassiveSharedViews cleanly.
        for row in srcs:
            self.assertTrue(
                all(obj.get_ref_count() == 0 and not obj.is_valid() for obj in row)
            )
        recycled_total = getattr(passive_alloc, "_recycled_batch_views_total", None)
        if recycled_total is not None:
            self.assertEqual(recycled_total, 4)

        # Combined coverage: seed hashes + _OmittedPrefixTokens + split cached keys
        seed_fn: Any = getattr(pmc, "seed_precomputed_hashes", None)
        omitted_cls: Any = getattr(
            importlib.import_module("lmcache.integration.vllm.vllm_v1_adapter"),
            "_OmittedPrefixTokens",
            None,
        )
        if callable(seed_fn) and omitted_cls is not None:
            prepared_records.clear()
            skip_calls.clear()
            pre_hashes = tuple(h for _, _, h in pre)
            pre_offsets = tuple(e - s for s, e, _ in pre)
            pmc_seed: Any = PrefillMetadataCache()
            pmc_seed.seed_precomputed_hashes(db, pre_hashes, pre_offsets)
            omitted_toks = omitted_cls(len(toks), ())
            seed_kw: dict[str, Any] = {
                "req_id": "req-pm-skip-seed",
                "kv_group": 0,
                "deferred_layerwise_get": True,
                "_selected_skip_group0_dma": True,
                "_prefill_metadata_cache": pmc_seed,
                "_reuse_shared_dense_prefix": True,
            }
            seed_env = SharedHandleEnvelope(
                request_id="req-pm-skip-seed",
                phase="dense_prefix",
                request_ordinal=0,
                layer_id=0,
                kv_group=0,
                status="ok",
                generation=1,
                handles=[],
                batch=compact_batch,
            )
            eng._receive_matching_shared_envelope = lambda *_a, **_k: seed_env
            ret_omitted = list(
                eng.retrieve_layer(tokens=omitted_toks, mask=None, **seed_kw)
            )
            self.assertEqual(len(ret_omitted), 3)
            self.assertTrue(bool(ret_omitted[-1].all()))
            self.assertEqual(len(prepared_records), 1)
            plan_unsplit = pmc_seed.prepare(
                db, omitted_toks, kv_group=0, num_layers=2, split_layers=False
            )
            unsplit_keys = [list(row) for row in plan_unsplit.keys_layer_major]
            self.assertTrue(
                all(isinstance(k, CacheEngineKey) for row in unsplit_keys for k in row)
            )
            ensure_split = getattr(
                LMCacheConnectorV1Impl, "_ensure_split_layer_cached_keys", None
            )
            if callable(ensure_split):
                ensure_split(unsplit_keys, 2)
                self.assertTrue(
                    all(
                        isinstance(k, LayerCacheEngineKey) and k.layer_id == lid
                        for lid, row in enumerate(unsplit_keys)
                        for k in row
                    )
                )


if __name__ == "__main__":
    unittest.main()
