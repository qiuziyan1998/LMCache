# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU unit tests for shared prefill optimizations."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch
import pickle
import sys
import threading
import types
import unittest

# Third Party
import msgspec
import torch

# First Party
from lmcache.utils import CacheEngineKey as _K
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.token_database import ChunkedTokenDatabase
import lmcache.integration.vllm.vllm_v1_adapter as va
import lmcache.v1.lookup_client.chunk_statistics_lookup_client as cslc
import lmcache.v1.lookup_client.hit_limit_lookup_client as hllc
import lmcache.v1.lookup_client.lmcache_async_lookup_client as alc
import lmcache.v1.lookup_client.lmcache_lookup_client as llc
import lmcache.v1.lookup_client.lmcache_lookup_client_bypass as blc
import lmcache.v1.lookup_client.mooncake_lookup_client as mlc_mod
import lmcache.v1.memory_management as mm
import lmcache.v1.prefill_metadata as pm
import lmcache.v1.shared_cpu_cache as scc

_SH = torch.Size([1, 2, 4, 4])
_FMT = mm.MemoryFormat.KV_MLA_FMT
_N = SimpleNamespace
_Impl = va.LMCacheConnectorV1Impl
_TMO = mm.TensorMemoryObj
_Env = scc.SharedHandleEnvelope
_Alloc = scc.PassiveSharedViewAllocator


def _mfn(_l: int, _n: int) -> tuple[torch.Size, torch.dtype, mm.MemoryFormat]:
    return (_SH, torch.uint8, _FMT)


def _cfg(cs: int = 128, *, dsa: bool = False, lw: bool = True) -> LMCacheEngineConfig:
    kw: dict[str, Any] = dict(chunk_size=cs, local_cpu=True, max_local_cpu_size=0.1)
    kw.update(use_layerwise=lw, enable_blending=False, save_unfull_chunk=True)
    return LMCacheEngineConfig(**kw, dsa_two_groups=dsa)


def _meta(
    cs: int = 128,
    nl: int = 4,
    *,
    dsa: bool = False,
    gc: tuple[int, ...] = (2, 2),
    mla: bool = True,
    ws: int = 1,
) -> LMCacheMetadata:
    mgr, tot, gn = KVLayerGroupsManager(), nl, None
    if dsa:
        n0 = [f"latent_{i}" for i in range(gc[0])]
        n1 = [f"indexer_{i}" for i in range(gc[1])]
        s0, s1 = (2, gc[0], cs, 64), (1, gc[1], cs, 32)
        g0 = KVLayerGroupInfo(n0, list(range(gc[0])), s0, torch.bfloat16)  # type: ignore[arg-type]
        i1 = list(range(gc[0], sum(gc)))
        g1 = KVLayerGroupInfo(n1, i1, s1, torch.bfloat16)  # type: ignore[arg-type]
        mgr, tot, gn = KVLayerGroupsManager([g0, g1]), sum(gc), (tuple(n0), tuple(n1))
    kw: dict[str, Any] = dict(
        model_name="test-model", world_size=ws, local_world_size=ws
    )
    kw.update(worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16, engine_id="eng0")
    kw.update(kv_shape=(tot, 2, cs, 1, 64), use_mla=mla, chunk_size=cs)
    kw.update(kv_layer_groups_manager=mgr, runtime_kv_group_layer_names=gn)
    kw.update(kv_connector_extra_config={"lmcache_rpc_port": 29111})
    kw.update(runtime_kv_group_layer_counts=gc if dsa else None)
    return LMCacheMetadata(**kw)


def _spy_hash(db: ChunkedTokenDatabase) -> dict[str, int]:
    cnt = {"_prefix_hash": 0, "hash_func": 0, "_hash_tokens": 0}
    for attr in tuple(cnt):
        orig = getattr(db, attr)

        def _w(*a: Any, _o: Any = orig, _k: str = attr, **kw: Any) -> Any:
            cnt[_k] += 1
            return _o(*a, **kw)

        setattr(db, attr, _w)
    return cnt


def _slab(
    name: str, sz: int = 16384, gen: int = 1, *, reuse: bool = True
) -> tuple[torch.Tensor, scc.SharedSlabMapping, _Alloc]:
    buf = torch.zeros(sz, dtype=torch.uint8)
    kw: dict[str, Any] = dict(shm_name=name, size=sz, ptr=int(buf.data_ptr()))
    m = scc.SharedSlabMapping(**kw, tensor=buf, generation=gen, owner=True)
    return buf, m, m.passive_allocator(reuse_prefill=reuse)


def _batch(
    name: str,
    hs: list[int],
    offs: list[int],
    lb: int = 32,
    po: list[int] | None = None,
    ps: list[int] | None = None,
) -> scc.SharedHandleBatch:
    k: dict[str, Any] = dict(shm_name=name, producer_rank=0, num_layers=2)
    k.update(num_chunks=len(hs), physical_sizes=[lb] * len(hs), chunk_hashes=hs)
    return scc.SharedHandleBatch(
        **k, offsets=offs, page_offsets=po or [], page_physical_sizes=ps or []
    )


def _gv(a: _Alloc, b: scc.SharedHandleBatch, sc: int = 0, ec: int = 3) -> Any:
    st = [i * 4 for i in range(ec)]
    k: dict[str, Any] = dict(start_chunk=sc, end_chunk=ec, starts=st, meta_fn=_mfn)
    return a.create_batch_views_for_group(b, **k, ends=[x + 4 for x in st])


def _pv(a: _Alloc, b: scc.SharedHandleBatch, prev: Any = None) -> Any:
    k: dict[str, Any] = dict(chunk_index=0, shape=_SH, dtype=torch.uint8, fmt=_FMT)
    return a.create_page_view(b, **k, cached_positions=range(0, 4), previous=prev)


def _bv(
    a: _Alloc, b: scc.SharedHandleBatch, prev: Any = None, lid: int = 0, ci: int = 1
) -> _TMO:
    k: dict[str, Any] = dict(layer_id=lid, chunk_index=ci, shape=_SH, fmt=_FMT)
    pos = range(ci * 4, (ci + 1) * 4)
    return a.create_batch_view(
        b, **k, dtype=torch.uint8, cached_positions=pos, previous=prev
    )


def _palloc(sz: int = 4096) -> mm.PagedTensorMemoryAllocator:
    return mm.PagedTensorMemoryAllocator(
        torch.zeros(sz, dtype=torch.uint8), [_SH], [torch.uint8], _FMT
    )


def _mk_eng(
    cfg: Any = None,
    meta: Any = None,
    *,
    bk: Any = None,
    sm: Any = None,
    gc: Any = None,
    alloc: Any = None,
    rel: Any = None,
) -> Any:
    e: Any = LMCacheEngine.__new__(LMCacheEngine)
    nop = lambda *_a, **_k: None  # noqa: E731
    if cfg is not None and meta is not None:
        e.config, e.metadata = cfg, meta
        e.token_database = ChunkedTokenDatabase(cfg, meta)
    else:
        e.metadata = _N(first_rank=0, indexer_c8_layout=None)
    nops = (
        "_remote_fill_retrieve_plan _cached_shared_prefill_source "
        "_validate_shared_layerwise_envelope _log_kvcache_for_check "
        "_close_shared_retrieve_consumer _release_retained_dense_retrieve_objs"
    )
    falses = (
        "_is_passive is_frozen _remote_fill_pair_lookup_enabled "
        "_adopt_dense_shared_retrieve_cache"
    )
    for k in nops.split():
        setattr(e, k, nop)
    for k in falses.split():
        setattr(e, k, lambda *_a, **_kw: False)
    for k in ("_should_use_shared_layerwise_retrieve", "_layerwise_chunk_fully_stored"):
        setattr(e, k, lambda *_a, **_kw: True)
    d = e.__dict__
    d.update(num_layers=2, _init_failed=False, _health_monitor=None)
    d.update(retrieve_locations=("LocalCPUBackend",), store_location="LocalCPUBackend")
    d.update(_layerwise_put_queue=None, _release_shared_retrieve_objs=rel or nop)
    d.update(shared_cpu_cache_passive_allocator=alloc, storage_manager=sm)
    d.update(gpu_connector=gc, _get_req_id=lambda _k: "req-split")
    rf: dict[str, Any] = dict(num_layers_for_group=2, _num_transfer_layers_for_call=2)
    rf.update(
        _make_passive_layer_page_views=(), _shared_cpu_dtype_for_kv_group=torch.bfloat16
    )
    rf.update(
        _memory_format_for_kv_group=_FMT, _dense_retrieve_token_results=[(0, 4, bk)]
    )
    rf["_expected_shared_cpu_chunk_metadata"] = (_SH, torch.uint8, _FMT)
    for k, v in rf.items():
        d[k] = lambda *_a, _v=v, **_kw: _v
    d.update(_retrieve_layer_shared_rank0=lambda *_a, **_k: iter(()))
    d.update(_lookup_key_for_kv_group=lambda k, kv_group=0, request_configs=None: k)
    d.update(_shared_page_first_location_plan=lambda gs: ["LocalCPUBackend"] * len(gs))
    d.update(_shared_local_prefix_chunk_count=lambda gs: len(list(gs)))
    sm_k: dict[str, Any] = dict(on_store_request=lambda _n: 1, on_store_finished=nop)
    d["stats_monitor"] = _N(
        on_retrieve_request=lambda _n: 2, on_retrieve_finished=nop, **sm_k
    )
    return e


class TestSharedPrefillOptimizations(unittest.TestCase):
    """Tests for shared prefill optimizations."""

    def test_sync_async_mooncake_stashing_wrappers_and_drop_on_failure(self) -> None:
        cfg, meta, toks, eq = _cfg(), _meta(), list(range(10, 310)), self.assertEqual
        fail, resps = (
            {"tr": False, "fs": False, "mc": False},
            [(256).to_bytes(4, "big")],
        )

        def _sar(_b: Any) -> list[bytes]:
            if fail["tr"]:
                raise RuntimeError("rpc failed")
            return resps

        def _chk(cli: Any, t: list[int], rid: str, pat: str | None = None) -> Any:
            if pat is not None:
                with self.assertRaisesRegex(RuntimeError, pat):
                    cli.lookup(t, lookup_id=rid)
                self.assertIsNone(cli.pop_lookup_chunk_hashes(rid))
                return None
            pop = cli.pop_lookup_chunk_hashes(rid)
            assert pop is not None
            self.assertIsNone(cli.pop_lookup_chunk_hashes(rid))
            return pop

        tr: Any = _N(world_size=1, send_and_recv_all=_sar)
        cli = llc.LMCacheLookupClient(cfg, meta, tr)
        eq(cli.lookup(toks, lookup_id="req-sync"), 256)
        sw = cslc.ChunkStatisticsLookupClient.__new__(cslc.ChunkStatisticsLookupClient)
        sw.actual_lookup_client = cli
        lw = hllc.HitLimitLookupClient.__new__(hllc.HitLimitLookupClient)
        lw.actual_lookup_client = sw
        pop = _chk(lw, toks, "req-sync")
        eq((len(pop[0]), pop[1:]), (3, ((128, 128, 44), 300, 10, 309, (137, 265, 309))))
        resps.clear()
        eq(cli.lookup(toks, lookup_id="req-empty"), 0)
        self.assertIsNone(cli.pop_lookup_chunk_hashes("req-empty"))
        fail["tr"] = True
        _chk(cli, toks, "req-err", "rpc failed")
        fail["tr"] = False
        resps.append((256).to_bytes(4, "big"))
        cli.lookup(toks, lookup_id="req-clr")
        lw.clear_lookup_status("req-clr")
        self.assertIsNone(lw.pop_lookup_chunk_hashes("req-clr"))
        msgs: list[bytes] = []
        pl = msgspec.msgpack.encode(alc.LookupCleanupResponseMsg("req-acln", 0))
        holder: list[Any] = []
        pulled = {"done": False}

        def _send(buf: bytes, copy: bool = False) -> None:
            if fail["fs"]:
                raise RuntimeError("zmq send failed")
            msgs.append(bytes(buf))

        def _recv(copy: bool = False) -> bytes:
            if not pulled["done"]:
                pulled["done"] = True
                return pl
            holder[0].running = False
            raise RuntimeError("stop")

        sk = lambda *_a: _N(send=_send) if _a[4] == "connect" else _N(recv=_recv)  # noqa: E731
        z_fn = lambda **_k: object()  # noqa: E731
        t_fn = lambda *_a, **_k: _N(start=lambda: None)  # noqa: E731
        with patch.multiple(alc, get_zmq_context=z_fn, get_zmq_socket=sk):
            with patch.object(alc.threading, "Thread", t_fn):
                ac = alc.LMCacheAsyncLookupClient(cfg, meta)
        holder.append(ac)
        ac.lookup_backoff_time = 0.0
        ac.lookup(toks[:256], lookup_id="req-a1")
        ac.lookup(toks[:256], lookup_id="req-acln")
        eq(len(msgs), 2)
        ap = _chk(ac, toks, "req-a1")
        eq(ap[1:], ((128, 128), 256, 10, 265, (137, 265)))
        ac.running = True
        ac.process_responses_from_workers()
        self.assertIsNone(ac.pop_lookup_chunk_hashes("req-acln"))
        fail["fs"] = True
        _chk(ac, toks[:256], "req-afail", "zmq send failed")

        def _mc_ex(keys: list[str]) -> list[int]:
            if fail["mc"]:
                raise RuntimeError("mooncake failure")
            return [1 for _ in keys]

        fm, fp = types.ModuleType("mooncake.store"), types.ModuleType("mooncake")
        mc_o = _N(setup=lambda *_a, **_k: 0, batch_is_exist=_mc_ex, close=lambda: None)
        fm.MooncakeDistributedStore, fp.store = lambda *_a, **_k: mc_o, fm  # type: ignore[attr-defined]
        mc_cfg, mc_m = _cfg(dsa=True), _meta(dsa=True, gc=(2, 1))
        with patch.dict(sys.modules, {"mooncake": fp, "mooncake.store": fm}):
            mc_c = mlc_mod.MooncakeLookupClient(mc_cfg, mc_m, "127.0.0.1:50051")
        eq(mc_c.lookup(toks, lookup_id="req-mc1"), 300)
        mp = _chk(mc_c, toks, "req-mc1")
        eq(mp[1:], ((128, 128, 44), 300, 10, 309, (137, 265, 309)))
        fail["mc"] = True
        _chk(mc_c, toks, "req-mcf", "mooncake failure")

    def test_omitted_prefix_contract_and_build_request_meta(self) -> None:
        seq, eq = va._OmittedPrefixTokens(256, (10, 11, 12, 13)), self.assertEqual
        eq((len(seq), seq[256], seq[-1], seq[-4]), (260, 10, 13, 10))
        eq((seq[256:], seq[256::2], seq[259:256]), ([10, 11, 12, 13], [10, 12], []))
        eq(seq[:128], va._OmittedPrefixTokens(128, ()))
        eq(seq[:258], va._OmittedPrefixTokens(256, (10, 11)))
        eq(pickle.loads(pickle.dumps(seq)), seq)
        for bad in (260, -261):
            with self.assertRaises(IndexError):
                _ = seq[bad]
        bad_ops = (
            lambda: seq[0],
            lambda: seq[-5],
            lambda: seq[64:258],
            lambda: list(seq),
        )
        for fn in bad_ops:
            with self.assertRaises(RuntimeError):
                fn()
        cs, cfg = 128, _cfg(128, dsa=True)
        cfg.enable_remote_lmcache_store = True
        cfg.extra_config = {
            "enable_shared_cpu_cache": True,
            "save_only_first_rank": True,
        }
        meta = _meta(cs, dsa=True, ws=4)
        self.assertTrue(
            _Impl.__new__(_Impl)._engine_supports_omitted_prefix_tokens_all_modes()
        )
        sub_cls: Any = type("_PlainSub", (_Impl,), {})
        sub: Any = sub_cls.__new__(sub_cls)
        self.assertFalse(sub._engine_supports_omitted_prefix_tokens_all_modes())
        eq = self.assertEqual
        d = sub.__dict__
        d.update(config=cfg, kv_role="kv_producer", force_skip_save=False)
        d.update(enable_sparse_attention=True, _discard_partial_chunks=False)
        d.update(_block_size=cs, _lmcache_chunk_size=cs, lmcache_engine=None)
        d.update(_vllm_config=None, _layerwise_prefill_p_node=True)
        d.update(_layerwise_prefill_dma=True, _resume_lookup_queries={})
        d.update(_request_trackers={}, load_specs={}, _unfinished_requests={})
        d.update(_dsa_kv_policy_states={}, _cold_perf_lookup_started={})
        sub._windowed_sparse_layerwise_save_enabled = lambda: True
        sub._scheduler_supports_shared_dense_seed = lambda: False
        toks = list(range(300))
        st_ends = (toks[127], toks[255])
        stashed = ((101, 102, 103), (128, 128, 44), 300, toks[0], toks[-1], st_ends)
        ls_off = va.LoadSpec(0, 256, can_load=True)
        self.assertFalse(hasattr(sub, "_stashed_lookup_hashes"))

        def _run_req(rid: str, mgr: Any) -> Any:
            sub._manager = mgr
            bks = (([1, 2, 3], [11, 12, 13]), ([4, 5, 6], [14, 15, 16]))
            tk: dict[str, Any] = dict(req_id=rid, prompt_len=300, token_ids=list(toks))
            tk.update(
                allocated_block_ids=[1, 2, 3], allocated_block_ids_indexer=[4, 5, 6]
            )
            tk.update(
                allocated_block_ids_by_bank=bks, block_allocation_mode="prefill_child"
            )
            tr = va.RequestTracker(**tk, num_saved_tokens=0)
            sub._unfinished_requests[rid] = tr
            with patch.object(va, "extract_mm_features", return_value=([], [])):
                sub.update_state_after_alloc(
                    _N(request_id=rid, multi_modal_features=[]), _N(), 128
                )
            rm = sub._build_request_meta(tr, ls_off)
            sc = _N(
                finished_req_ids=[rid], scheduled_new_reqs=[], scheduled_cached_reqs=[]
            )
            self.assertIsNotNone(sub.build_connector_meta(sc))
            eq(
                sub.request_finished(_N(request_id=rid, kv_transfer_params=None), []),
                (False, None),
            )
            return rm

        lu_nop = _N(clear_lookup_status=lambda _r: None)
        mgr_ns = _N(
            lmcache_engine_metadata=meta,
            lookup_client=lu_nop,
            lmcache_engine=_N(lookup_unpin=lambda _r: None),
        )
        rm_off = _run_req("req-off", mgr_ns)
        assert rm_off is not None
        self.assertNotIsInstance(rm_off.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(rm_off.token_ids, toks)
        mlc = MagicMock()
        mlc.pop_lookup_chunk_hashes.return_value = stashed
        rm_mk = _run_req("req-mk", _N(lmcache_engine_metadata=meta, lookup_client=mlc))
        assert rm_mk is not None
        self.assertNotIsInstance(rm_mk.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(
            mlc.clear_lookup_status.call_args_list, [call("req-mk"), call("req-mk")]
        )
        sub._engine_supports_omitted_prefix_tokens_all_modes = lambda: True
        sub._stashed_lookup_hashes = {}
        rm_on = _run_req("req-on", _N(lmcache_engine_metadata=meta, lookup_client=mlc))
        assert rm_on is not None
        self.assertIsInstance(rm_on.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(rm_on.token_ids.prefix_len, 256)
        self.assertEqual(list(rm_on.token_ids.suffix_tokens), toks[256:])

        # CRC32 tamper detection, skip_last_n_tokens, preemption, and exception cleanup
        # First Party
        from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

        class _Lc(LookupClientInterface):
            def lookup(self, *a: Any, **kw: Any) -> int | None:
                return 0

            def supports_producer_reuse(self) -> bool:
                return False

            def clear_lookup_status(self, lookup_id: str) -> None:
                self._drop_lookup_chunk_hashes(lookup_id)

            def close(self) -> None:
                return None

        lc_crc = _Lc()
        full_ends = (toks[127], toks[255], toks[299])
        lc_crc._stash_lookup_chunk_hashes(
            "req-tamper-ok", toks, (201, 202, 203), (128, 128, 44), full_ends
        )
        sub._manager = _N(lmcache_engine_metadata=meta, lookup_client=lc_crc)
        bks = (([1, 2, 3], [11, 12, 13]), ([4, 5, 6], [14, 15, 16]))
        tr_ok = va.RequestTracker(
            req_id="req-tamper-ok",
            prompt_len=300,
            token_ids=list(toks),
            allocated_block_ids=[1, 2, 3],
            allocated_block_ids_indexer=[4, 5, 6],
            allocated_block_ids_by_bank=bks,
            block_allocation_mode="prefill_child",
            num_saved_tokens=0,
        )
        sub._unfinished_requests["req-tamper-ok"] = tr_ok
        with patch.object(va, "extract_mm_features", return_value=([], [])):
            sub.update_state_after_alloc(
                _N(request_id="req-tamper-ok", multi_modal_features=[]), _N(), 128
            )
        rm_ok = sub._build_request_meta(tr_ok, ls_off)
        assert rm_ok is not None
        self.assertIsInstance(rm_ok.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(rm_ok.prefill_chunk_hashes, (201, 202, 203))
        self.assertEqual(rm_ok.prefill_chunk_offsets, (128, 128, 44))

        tampered = list(toks)
        tampered[42] ^= 0x5A5A
        for rid_t, p_len in (("req-tamper", 300), ("req-tamper-pfx", 256)):
            lc_crc._stash_lookup_chunk_hashes(
                rid_t, toks, (201, 202, 203), (128, 128, 44), full_ends
            )
            tr_t = va.RequestTracker(
                req_id=rid_t,
                prompt_len=p_len,
                token_ids=list(tampered[:p_len]),
                allocated_block_ids=[1, 2, 3],
                allocated_block_ids_indexer=[4, 5, 6],
                allocated_block_ids_by_bank=bks,
                block_allocation_mode="prefill_child",
                num_saved_tokens=0,
            )
            sub._unfinished_requests[rid_t] = tr_t
            with patch.object(va, "extract_mm_features", return_value=([], [])):
                sub.update_state_after_alloc(
                    _N(request_id=rid_t, multi_modal_features=[]), _N(), 128
                )
            rm_t = sub._build_request_meta(tr_t, ls_off)
            assert rm_t is not None
            self.assertNotIsInstance(rm_t.token_ids, va._OmittedPrefixTokens)
            self.assertEqual(rm_t.token_ids, tampered[:p_len])
            self.assertIsNone(rm_t.prefill_chunk_hashes)

        # Windowed sparse save must block _discard_partial_chunks omitted prefix
        cfg_ws = _cfg(cs, dsa=True, lw=True)
        cfg_ws.enable_remote_lmcache_store = True
        cfg_ws.extra_config = dict(cfg.extra_config)
        cfg_ws.save_unfull_chunk = False
        sub.config = cfg_ws
        sub._discard_partial_chunks = True
        for p_offs, p_hs, ws_enabled, ends_tup, expect_omitted in (
            ((128, 128), (301, 302), False, st_ends, True),
            ((128, 128), (301, 302), True, st_ends, False),
            ((256,), (301,), True, (toks[255],), False),
            ((128, 128, 44), (301, 302, 303), True, full_ends, True),
        ):
            rid_ws = f"req-ws-{len(p_offs)}-{ws_enabled}"
            sub._windowed_sparse_layerwise_save_enabled = lambda v=ws_enabled: v
            sub._stashed_lookup_hashes[rid_ws] = (
                (p_hs, p_offs, 300, toks[0], toks[-1], ends_tup),
                ((), ()),
            )
            tr_ws = va.RequestTracker(
                req_id=rid_ws,
                prompt_len=300,
                token_ids=list(toks),
                allocated_block_ids=[1, 2, 3],
                allocated_block_ids_indexer=[4, 5, 6],
                allocated_block_ids_by_bank=bks,
                block_allocation_mode="prefill_child",
                num_saved_tokens=0,
            )
            sub._unfinished_requests[rid_ws] = tr_ws
            rm_ws = sub._build_request_meta(tr_ws, ls_off)
            assert rm_ws is not None
            self.assertEqual(rm_ws.windowed_sparse_save, ws_enabled)
            if expect_omitted:
                self.assertIsInstance(rm_ws.token_ids, va._OmittedPrefixTokens)
            else:
                self.assertNotIsInstance(rm_ws.token_ids, va._OmittedPrefixTokens)
        sub.config = cfg
        sub._discard_partial_chunks = False
        sub._windowed_sparse_layerwise_save_enabled = lambda: True

        sub._stashed_lookup_hashes = {
            "req-sk": ("old",),
            "req-pr": ("old",),
            "req-er": ("old",),
        }
        sub.skip_last_n_tokens = 1
        mlc.pop_lookup_chunk_hashes.return_value = stashed
        sub._manager = _N(lmcache_engine_metadata=meta, lookup_client=mlc)
        sub._unfinished_requests["req-sk"] = va.RequestTracker(
            req_id="req-sk",
            prompt_len=300,
            token_ids=list(toks),
            allocated_block_ids=[],
            num_saved_tokens=0,
        )
        with patch.object(va, "extract_mm_features", return_value=([], [])):
            sub.update_state_after_alloc(
                _N(request_id="req-sk", multi_modal_features=[]), _N(), 128
            )
        self.assertNotIn("req-sk", sub._stashed_lookup_hashes)
        sub.skip_last_n_tokens = 0
        sub.update_state_after_alloc(
            _N(request_id="req-pr", multi_modal_features=[], num_preemptions=1),
            _N(),
            128,
        )
        self.assertNotIn("req-pr", sub._stashed_lookup_hashes)
        with (
            patch.object(
                va, "extract_mm_features", side_effect=RuntimeError("mm boom")
            ),
            self.assertRaisesRegex(RuntimeError, "mm boom"),
        ):
            sub.update_state_after_alloc(
                _N(request_id="req-er", multi_modal_features=[]), _N(), 128
            )
        self.assertNotIn("req-er", sub._stashed_lookup_hashes)
        self.assertNotIn("req-er", sub._unfinished_requests)

        # Unaligned vllm_cached_tokens and partial-chunk fallback + bounded __repr__
        self.assertLess(len(repr(va._OmittedPrefixTokens(131072, (9,)))), 120)
        cfg_ua = _cfg(256, dsa=True, lw=True)
        cfg_ua.enable_remote_lmcache_store = True
        cfg_ua.extra_config = {
            "enable_shared_cpu_cache": True,
            "save_only_first_rank": True,
        }
        meta_ua = _meta(256, dsa=True, ws=4)
        sub_ua: Any = _Impl.__new__(_Impl)
        sub_ua.__dict__.update(
            config=cfg_ua,
            kv_role="kv_producer",
            force_skip_save=False,
            enable_sparse_attention=True,
            _discard_partial_chunks=True,
            _block_size=256,
            _lmcache_chunk_size=256,
            lmcache_engine=None,
            _vllm_config=None,
            _layerwise_prefill_p_node=True,
            _layerwise_prefill_dma=True,
            _resume_lookup_queries={},
            _request_trackers={},
            load_specs={},
            _unfinished_requests={},
            _dsa_kv_policy_states={},
            _cold_perf_lookup_started={},
            _stashed_lookup_hashes={},
        )
        sub_ua._windowed_sparse_layerwise_save_enabled = lambda: True
        sub_ua._is_dsa_two_groups = lambda: True
        sub_ua._scheduler_supports_shared_dense_seed = lambda: True
        sub_ua._engine_supports_omitted_prefix_tokens_all_modes = lambda: True
        mlc_ua = MagicMock()
        mlc_ua.pop_lookup_chunk_hashes.return_value = (
            (301, 302),
            (256, 44),
            300,
            toks[0],
            toks[-1],
            (toks[255],),
        )
        mlc_ua.pop_lookup_chunk_digests.return_value = None
        sub_ua._manager = _N(lmcache_engine_metadata=meta_ua, lookup_client=mlc_ua)
        tr_ua = va.RequestTracker(
            req_id="req-ua",
            prompt_len=300,
            token_ids=list(toks),
            allocated_block_ids=[1, 2, 3],
            allocated_block_ids_indexer=[4, 5, 6],
            allocated_block_ids_by_bank=bks,
            block_allocation_mode="prefill_child",
            num_saved_tokens=0,
        )
        sub_ua._unfinished_requests["req-ua"] = tr_ua
        with patch.object(va, "extract_mm_features", return_value=([], [])):
            sub_ua.update_state_after_alloc(
                _N(request_id="req-ua", multi_modal_features=[]), _N(), 0
            )
        rm_ua = sub_ua._build_request_meta(tr_ua, va.LoadSpec(128, 256, can_load=True))
        assert rm_ua is not None
        self.assertNotIsInstance(rm_ua.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(rm_ua.token_ids, toks)

        sub_ua.config.save_unfull_chunk = False
        sub_ua._discard_partial_chunks = False
        sub_ua.enable_sparse_attention = True
        mlc_ua.pop_lookup_chunk_hashes.return_value = (
            (401,),
            (256,),
            300,
            toks[0],
            toks[-1],
            (toks[255],),
        )
        tr_nf = va.RequestTracker(
            req_id="req-nf",
            prompt_len=300,
            token_ids=list(toks),
            allocated_block_ids=[1, 2, 3],
            allocated_block_ids_indexer=[4, 5, 6],
            allocated_block_ids_by_bank=bks,
            block_allocation_mode="prefill_child",
            num_saved_tokens=0,
        )
        sub_ua._unfinished_requests["req-nf"] = tr_nf
        with patch.object(va, "extract_mm_features", return_value=([], [])):
            sub_ua.update_state_after_alloc(
                _N(request_id="req-nf", multi_modal_features=[]), _N(), 0
            )
        rm_nf = sub_ua._build_request_meta(tr_nf, va.LoadSpec(0, 256, can_load=True))
        assert rm_nf is not None
        self.assertNotIsInstance(rm_nf.token_ids, va._OmittedPrefixTokens)
        self.assertEqual(rm_nf.token_ids, toks)

    def test_precomputed_hashes_and_worker_retain_metadata_safety(self) -> None:
        cfg, meta, eq = _cfg(dsa=True), _meta(dsa=True), self.assertEqual
        db = ChunkedTokenDatabase(cfg, meta)
        eng = LMCacheEngine.__new__(LMCacheEngine)
        eng.token_database, eng.config, eng.metadata = db, cfg, meta
        toks = list(range(300))
        pre = list(db.process_tokens(tokens=toks, make_key=False))
        ph, po = tuple(h for _, _, h in pre), tuple(e - s for s, e, _ in pre)
        om = va._OmittedPrefixTokens(256, tuple(toks[256:]))
        exp_full = list(db.process_tokens(tokens=toks, kv_group=0))
        cnt = _spy_hash(db)
        full_h = {"prefill_chunk_hashes": ph, "prefill_chunk_offsets": po}
        rf = eng._process_tokens_with_precomputed_hashes(om, None, None, 0, full_h)
        eq(list(rf or ()), exp_full)
        eq(cnt, {"_prefix_hash": 0, "hash_func": 0, "_hash_tokens": 0})
        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:128] = False
        exp_tail = list(db.process_tokens(tokens=toks, mask=mask, kv_group=0))
        for k in cnt:
            cnt[k] = 0
        sub_h = {"prefill_chunk_hashes": ph[:2], "prefill_chunk_offsets": po[:2]}
        rp = eng._process_tokens_with_precomputed_hashes(om, mask, None, 0, sub_h)
        eq(list(rp or ()), exp_tail)
        eq((cnt["_prefix_hash"], cnt["_hash_tokens"]), (0, 1))
        impl = _Impl.__new__(_Impl)
        impl.enable_sparse_attention, impl._worker_retrieve_state = True, {}
        impl._refresh_prepared_sparse_sources = lambda state, token_count: None  # type: ignore[method-assign]
        impl._record_shared_worker_retrieve_state = lambda *a, **kw: None  # type: ignore[method-assign]
        st, req = (
            va.WorkerRetrieveState(),
            va.ReqMeta(req_id="req-w", token_ids=om, slot_mapping=[]),
        )
        eq(
            impl._warm_request_retrieve_metadata(
                st, req, om, None, kv_group=0, dsa_two_groups=True
            ),
            (None, False),
        )  # noqa: E501
        pkw: dict[str, Any] = {"location": "LocalCPUBackend", "metadata_warm": True}
        impl._publish_worker_retrieve_state(
            st, req, token_count=len(toks), reuse_prepared_sources=False, **pkw
        )  # noqa: E501
        eq(st.metadata_token_ids, [])
        eng.enable_shared_cpu_cache = eng.use_layerwise = True
        eng.save_only_first_rank = eng.save_indexer_only_first_rank = True
        eng.metadata = _N(use_mla=False, world_size=4)  # type: ignore[assignment]
        self.assertFalse(eng._should_use_shared_layerwise_retrieve(0))
        eng.metadata = _N(use_mla=True, world_size=4)  # type: ignore[assignment]
        eng.save_only_first_rank = False
        self.assertFalse(eng._should_use_shared_layerwise_retrieve(0))

    def test_prefill_sequence_view_metadata_cache_and_split_layers(self) -> None:
        base, eq = [10, 20, 30], self.assertEqual
        v = pm.PrefillSequenceView(base, 0, 3, tail=(40, 50))
        base.append(999)
        eq((len(v), v[-1], v[::2]), (5, 50, (10, 30, 50)))
        eq((list(v), list(v[1:4])), ([10, 20, 30, 40, 50], [20, 30, 40]))
        eq(v, pm.PrefillSequenceView([10, 20, 30, 40, 50]))
        for other in ([10, 20, 30, 40, 50], (10, 20, 30, 40, 50), object()):
            self.assertNotEqual(v, other)
        with self.assertRaises(ValueError):
            pm.PrefillSequenceView(base, 2, 1)
        cfg, meta = _cfg(dsa=True), _meta(dsa=True, gc=(2, 3))
        toks = list(range(1000, 1300))
        db = ChunkedTokenDatabase(cfg, meta)
        pre = list(db.process_tokens(tokens=toks, make_key=False))
        ph, po = tuple(h for _, _, h in pre), tuple(e - s for s, e, _ in pre)
        rc, sc = pm.PrefillMetadataCache(), pm.PrefillMetadataCache()
        kw_p: dict[str, Any] = {"request_configs": {"tag": "pf"}, "skip_tokens": 128}
        refs = {
            g: rc.prepare(db, toks, kv_group=g, num_layers=nl, **kw_p)
            for g, nl in ((0, 2), (1, 3))
        }
        cnt = _spy_hash(db)
        sc.seed_precomputed_hashes(db, ph, po, request_configs={"tag": "pf"})
        for g, nl in ((0, 2), (1, 3)):
            rp, sp = refs[g], sc.prepare(db, toks, kv_group=g, num_layers=nl, **kw_p)
            eq(list(sp.starts), list(rp.starts))
            eq(list(sp.ends), list(rp.ends))
            eq(list(sp.candidates), list(rp.candidates))
            skm = [list(k) for k in sp.keys_layer_major]
            eq(skm, [list(k) for k in rp.keys_layer_major])
        eq(cnt, {"_prefix_hash": 0, "hash_func": 0, "_hash_tokens": 0})
        bk = _K("m", 1, 0, ph[0], torch.bfloat16)
        eq(tuple(pm.split_layers_cached(bk, 4)), tuple(bk.split_layers(4)))
        sc._reset()
        self.assertTrue(sc.database is None and len(sc.groups) == 0)

    def test_passive_recycle_batch_release_and_clear_invalidation(self) -> None:
        _, mapping, p = _slab("shm-recycle", 16384)
        b = _batch("shm-recycle", [11, 22], [256, 288, 320, 352], po=[0], ps=[64])
        p1, bv1 = _pv(p, b), _gv(p, b, sc=1, ec=2)[0][0]
        assert isinstance(p1, mm.LayerPageMemoryObj) and isinstance(bv1, _TMO)
        self.assertTrue(p1.parent_allocator is p and bv1.parent_allocator is p)
        scc.release_passive_views_batch([p1, bv1])
        self.assertEqual((p1.get_ref_count(), bv1.get_ref_count()), (0, 0))
        extra_frees: list[list[Any]] = []
        with patch.object(
            p, "batched_free", side_effect=lambda objs: extra_frees.append(list(objs))
        ):
            scc.release_passive_views_batch([p1])
        self.assertEqual(p1.get_ref_count(), 0)
        self.assertEqual(extra_frees, [])
        p2, rows2 = _pv(p, b), _gv(p, b, sc=1, ec=2)
        self.assertTrue(p2 is p1 and rows2[0][0] is bv1)
        assert isinstance(p2, mm.LayerPageMemoryObj)
        scc.release_passive_views_batch([rows2[0][0]])
        p2._cached_npu_base_dva = (1, "npu:0", 123, 64, True, 999)
        p.clear_recycled_views()
        self.assertTrue(p._unregistered and not p2.is_valid() and not bv1.is_valid())
        got = (p2._base_data_ptr, p2._cached_npu_base_dva, bv1._cached_npu_base_dva)
        self.assertEqual(got, (0, None, None))
        with patch.object(mapping, "unlink", return_value=None):
            mapping.close()
        self.assertRaises(scc.SharedCPUCacheError, _gv, p, b, sc=1, ec=2)

    def test_pin_many_layer_page_rollback_and_uint8_tensor_slice(self) -> None:
        alloc, eq = _palloc(4096), self.assertEqual
        self.addCleanup(lambda: [b.invalidate() for b in list(alloc.free_blocks)])
        buf = alloc.buffer
        o1, o2, o3 = (alloc.allocate([_SH], [torch.uint8], _FMT) for _ in range(3))
        assert isinstance(o1, _TMO) and isinstance(o2, _TMO) and isinstance(o3, _TMO)
        o2.invalidate()
        fmon = _N(on_pin_many=lambda _o: None, on_unpin=lambda _o: None)
        with patch.object(mm.PinMonitor, "GetOrCreate", return_value=fmon):
            self.assertTrue(_TMO.pin_many([o1, o3]))
            eq(
                (
                    o1.meta.pin_count,
                    o1.meta.ref_count,
                    o3.meta.pin_count,
                    o3.meta.ref_count,
                ),
                (1, 1, 1, 1),
            )
            o1.unpin()
            o3.unpin()
            self.assertTrue(_TMO.acquire_and_pin_many([o1, o3]))
            eq((o1.meta.ref_count, o1.meta.pin_count), (2, 1))
            self.assertFalse(_TMO.acquire_and_pin_many([o1, o2]))
            eq((o1.meta.ref_count, o1.meta.pin_count), (2, 1))
            boom = lambda _o: (_ for _ in ()).throw(RuntimeError("pin boom"))  # noqa: E731
            bm_m = _N(on_pin_many=boom, on_unpin=lambda _o: None)
            with patch.object(mm.PinMonitor, "GetOrCreate", return_value=bm_m):
                with self.assertRaisesRegex(RuntimeError, "pin boom"):
                    _TMO.pin_many([o1, o3])
            eq((o1.meta.ref_count, o1.meta.pin_count), (2, 1))
            o1.unpin()
            o3.unpin()
        for obj in (o1, o1, o3, o3):
            obj.ref_count_down()
        du_m = mm.MemoryObjMetadata(
            torch.Size([2, 8]), torch.uint8, 0, 16, 1, fmt=mm.MemoryFormat.KV_T2D
        )
        tv = _TMO(buf[64:80], du_m, None).tensor
        assert tv is not None
        sh_p = (tv.shape, tv.dtype, tv.data_ptr())
        eq(sh_p, (torch.Size([2, 8]), torch.uint8, buf[64:80].data_ptr()))
        bm_k: dict[str, Any] = dict(
            pin_count=2, fmt=_FMT, shapes=[_SH, _SH], dtypes=[torch.bfloat16] * 2
        )
        bm = mm.MemoryObjMetadata(_SH, torch.bfloat16, 0, 128, 3, **bm_k)
        lp_k: dict[str, Any] = dict(
            parent_allocator=None, num_layers=2, group_prefix_sum=(0, 64, 128)
        )
        with self.assertRaises(ValueError):
            mm.LayerPageMemoryObj(
                raw_data=buf[:256], metadata=bm, **lp_k, valid_tokens=999
            )
        eq((bm.ref_count, bm.pin_count), (0, 0))

    def test_shared_slab_mapping_close_concurrent_stress_and_envelope(self) -> None:
        _, mapping, p = _slab("shm-cs", 4096, gen=7)
        b = _batch("shm-cs", [1, 2], [32, 64, 96, 128], lb=32, po=[0], ps=[64])
        lv = _gv(p, b, sc=1, ec=2)[0][0]
        lv._cached_npu_base_dva = (1, "npu:0", 1, 16, True, 99)
        p._cached_npu_slab_dva = (1, "npu:0", 1, 4096, True, 88)
        rt = _Env.from_dict(
            _Env("req-close", "prefill", 1, 0, 0, "ok", 7, [], batch=b).to_dict()
        )
        self.assertTrue(
            rt.request_id == "req-close" and rt.generation == 7 and rt.batch
        )
        unlinked, errs, bar = [], [], threading.Barrier(4)
        dlm = lambda _p, _s: None  # noqa: E731
        fops = _N(unlink_shm=lambda n: unlinked.append(n), detach_shm_pinned_ptr=dlm)

        def _w(idx: int) -> None:
            try:
                bar.wait(timeout=2.0)
                mapping.close() if idx == 0 else p.clear_recycled_views()
            except BaseException as exc:  # pragma: no cover - asserted empty
                errs.append(exc)

        with (
            patch.object(scc, "_load_lmc_ops", return_value=fops),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            list(pool.map(_w, range(4)))
        self.assertEqual((errs, unlinked), ([], ["shm-cs"]))
        self.assertIsNone(p._cached_npu_slab_dva)
        self.assertIsNone(lv._cached_npu_base_dva)
        self.assertTrue(mapping._closed and not lv.is_valid())

    def test_split_layers_cached_engine_call_sites(self) -> None:
        cfg, meta, eq = _cfg(4, lw=True), _meta(4, 2), self.assertEqual
        bk = _K("m", 1, 0, 12345, torch.bfloat16)
        pm._split_cache_engine_key_layers.cache_clear()
        va_tup, vb_tup = pm.split_layers_cached(bk, 2), pm.split_layers_cached(bk, 2)
        self.assertTrue(isinstance(va_tup, tuple) and va_tup is vb_tup)
        eq(tuple(va_tup), tuple(bk.split_layers(2)))
        inf = pm._split_cache_engine_key_layers.cache_info()
        eq((inf.misses, inf.hits), (1, 1))
        sent = [bk.get_layer(0), bk.get_layer(1)]
        self.assertIs(pm.split_layers_cached(_N(split_layers=lambda _n: sent), 2), sent)
        with patch.object(_K, "split_layers", lambda _s, _n: sent):
            self.assertIs(pm.split_layers_cached(bk, 2), sent)
        ck = _K("m", 1, 0, 54321, torch.bfloat16, {"tag": "x"})
        eq(list(pm.split_layers_cached(ck, 2)), list(ck.split_layers(2)))
        eq(pm._split_cache_engine_key_layers.cache_info().misses, 1)
        hc = {k: _N(pin=lambda: True) for k in va_tup}
        lb = _N(cpu_lock=threading.Lock(), hot_cache=hc, keys_in_request=[])
        bc_fn = lambda keys, locs=None: (len(keys), {"LocalCPUBackend": keys})  # noqa: E731
        bks = lambda search_range=None: [("LocalCPUBackend", lb)]  # noqa: E731
        sm = _N(get_active_storage_backends=bks, batched_contains=bc_fn)
        ig = lambda *_a, **_k: iter(())  # noqa: E731
        gc_m = _N(batched_to_gpu=ig, batched_from_gpu=ig)
        eng = _mk_eng(cfg, meta, bk=bk, sm=sm, gc=gc_m)
        pm._split_cache_engine_key_layers.cache_clear()
        lk: dict[str, Any] = dict(
            search_range=None, lookup_kv_groups=[0], page_lookup=False
        )
        lk.update(request_configs=None, pin=False, lookup_id=None, call_pinned={})
        w = eng._lookup_local_cpu_windowed([(0, 4, bk)], **lk, res=0)
        self.assertTrue(w[2] and w[1] == 4)
        self.assertGreaterEqual(
            pm._split_cache_engine_key_layers.cache_info().misses, 1
        )
        sc_list, osplit = [], pm.split_layers_cached

        def _ss(k: Any, n: int) -> Any:
            sc_list.append(n)
            return osplit(k, n)

        toks = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        ce = "lmcache.v1.cache_engine"
        pl_p = patch(f"{ce}.mooncake_page_layout_enabled", return_value=True)
        lp_p = patch(f"{ce}.mooncake_layer_pages_enabled", return_value=False)
        with patch(f"{ce}.split_layers_cached", side_effect=_ss):
            list(eng.store_layer(tokens=toks, req_id="req-split"))
            eq(sc_list, [2])
            sc_list.clear()
            with pl_p, lp_p:
                gn = lambda *_a, **_k: [None, None]  # noqa: E731
                eng.storage_manager = _N(
                    batched_contains=bc_fn, batched_get=gn, async_batched_get=None
                )
                list(eng.retrieve_layer(tokens=toks, req_id="req-split"))
            self.assertGreaterEqual(len(sc_list), 2)

    def test_prefill_sequence_view_flattens_slices_and_append_isolation(self) -> None:
        base, eq = [10, 20, 30, 40, 50, 60, 70], self.assertEqual
        outer = pm.PrefillSequenceView(base, 1, 6, tail=(90, 91))
        inner = pm.PrefillSequenceView(outer, 1, 6)
        sl = inner[:4]
        assert isinstance(sl, pm.PrefillSequenceView)
        self.assertNotIsInstance(inner.values, pm.PrefillSequenceView)
        self.assertNotIsInstance(sl.values, pm.PrefillSequenceView)
        self.assertTrue(inner.values is base and sl.values is base)
        to = pm.PrefillSequenceView(outer, 5, 7)
        self.assertNotIsInstance(to.values, pm.PrefillSequenceView)
        eq((list(to), sl[0], sl[3], sl[-1], sl[-4]), ([90, 91], 30, 60, 60, 30))
        eq((inner[-1], inner[::2]), (90, (30, 50, 90)))
        eq(inner[::-1], (90, 60, 50, 40, 30))
        for bad in ((2, 1), (0, 8)):
            with self.assertRaises(ValueError):
                pm.PrefillSequenceView(outer, *bad)
        for bad_i in (4, -5):
            with self.assertRaises(IndexError):
                _ = sl[bad_i]
        is_, ii = iter(sl), iter(inner)
        eq((next(is_), next(ii)), (30, 30))
        base.append(999)
        rs, ri = list(is_), list(ii)
        self.assertTrue(999 not in rs and 999 not in ri)
        eq((rs, ri), ([40, 50, 60], [40, 50, 60, 90]))

    def test_retrieve_layer_shared_passive_unlocks_batch_views_with_reuse(self) -> None:
        _, _m, alloc = _slab("shm-passive-ret", 32768)
        bg, sv, deliv, rels = [], [], [], []
        og, os_fn = _Alloc.create_batch_views_for_group, _Alloc.create_batch_view

        def _sg(sa: _Alloc, b: scc.SharedHandleBatch, **kw: Any) -> Any:
            bg.append((kw["start_chunk"], kw["end_chunk"]))
            return og(sa, b, **kw)

        def _ss(sa: _Alloc, b: scc.SharedHandleBatch, **kw: Any) -> Any:
            sv.append((kw["layer_id"], kw["chunk_index"]))
            return os_fn(sa, b, **kw)

        def _coro() -> Any:
            for _ in range(2):
                deliv.append((yield None))
            yield None
            yield None

        def _rel(objs: Any, *, unpin: bool = False) -> None:
            del unpin
            cp = list(objs)
            rels.append(cp)
            scc.release_passive_views_batch(cp)

        gc_cls = VLLMPagedMemLayerwiseGPUConnector
        gc: Any = gc_cls.__new__(gc_cls)
        gc.layerwise_batched_to_gpu, gc.batched_to_gpu = True, lambda *_a, **_k: _coro()
        eng, eq = _mk_eng(alloc=alloc, gc=gc, rel=_rel), self.assertEqual

        def _run(b: Any, rid: str, cs: list[int], ce: list[int], cm: Any) -> Any:
            for lst in (bg, sv, deliv, rels):
                lst.clear()
            nc, h0 = b.num_chunks, b.chunk_hashes[0]

            def _env(**k: Any) -> _Env:
                args = (k["req_id"], k["phase"], k["request_ordinal"])
                return _Env(*args, k["layer_id"], k["kv_group"], "ok", 1, [], batch=b)

            eng._receive_matching_shared_envelope = _env
            r_c = range(nc)
            ks = [[_K("m", 1, li, h0 + c, torch.uint8) for c in r_c] for li in range(2)]
            st, en = [c * 4 for c in r_c], [(c + 1) * 4 for c in r_c]
            kwi: dict[str, Any] = dict(
                _reuse_shared_dense_prefix=True, cached_starts=cs
            )
            kwi.update(cached_ends=ce, cached_memory_objs=cm)
            p_g = patch.object(_Alloc, "create_batch_views_for_group", _sg)
            p_s = patch.object(_Alloc, "create_batch_view", _ss)
            with p_g, p_s:
                mk = torch.zeros(nc * 4, dtype=torch.bool)
                rk: dict[str, Any] = dict(
                    starts_all=st, ends_all=en, keys_layer_major=ks, ret_mask=mk
                )
                gen = eng._retrieve_layer_shared_passive(
                    **rk, monitor_req_id=1, req_id=rid, kv_group=0, kwargs=kwi
                )
                m = next(gen)
                assert m is not None
                eq(int(m.item()), nc * 4)
                self.assertTrue(gen.send(None) is None and gen.send(None) is None)
                res = next(gen)
                with self.assertRaises(StopIteration):
                    next(gen)
                return res

        ba = _batch("shm-passive-ret", [101, 102, 103, 104], [i * 32 for i in range(8)])
        eng.get_shared_cpu_request_lease = lambda _r: _N(owns=lambda _o: False)
        eng._shared_prefill_source_state = lambda _r, _g, _k: None
        r1 = _run(ba, "req-empty", [], [], [[], []])
        assert r1 is not None
        eq((int(r1.sum().item()), bg, sv), (16, [(0, 4)], []))
        eq(([len(r) for r in deliv], len(rels[0])), ([4, 4], 8))
        self.assertTrue(all(o.get_ref_count() == 0 for o in rels[0]))
        o12 = [1024 + i * 32 for i in range(12)]
        bb = _batch("shm-passive-ret", [201, 202, 203, 204, 205, 206], o12)
        rr = [[_bv(alloc, bb, lid=lid, ci=c) for c in range(2)] for lid in range(2)]
        rset = {id(o) for row in rr for o in row}
        eng.get_shared_cpu_request_lease = lambda _r: _N(owns=lambda o: id(o) in rset)
        wp = lambda _b, sp: sp  # noqa: E731
        st_m = _N(matching_prefix=lambda _s, _e, _k, sp: sp, wire_prefix=wp)
        eng._shared_prefill_source_state = lambda _r, _g, _k: st_m
        eng._shared_prefill_cached_prefix = lambda *_a, **_k: 2
        _run(bb, "req-part", [0, 4, 8], [4, 8, 12], rr)
        eq((bg, sv, [len(r) for r in deliv]), ([(3, 6)], [(0, 2), (1, 2)], [6, 6]))
        for lid in range(2):
            self.assertTrue(deliv[lid][0] is rr[lid][0] and deliv[lid][1] is rr[lid][1])
            exp_a = [1024 + (lid * 6 + c) * 32 for c in range(6)]
            eq([o.meta.address for o in deliv[lid]], exp_a)
        self.assertTrue(len(rels[0]) == 8 and all(id(o) not in rset for o in rels[0]))
        self.assertTrue(all(o.get_ref_count() == 1 for o in rr[0] + rr[1]))
        scc.release_passive_views_batch(rr[0] + rr[1])
        alloc.clear_recycled_views()

    def test_passive_allocator_exact_recycle_exclude_live_id_and_close(self) -> None:
        eq, ok, no = self.assertEqual, self.assertTrue, self.assertFalse
        eq(_Alloc._MAX_RECYCLED_BATCH_VIEWS, 327680)
        _, mapping, alloc = _slab("shm-recycle-exact", 32768)
        no(hasattr(alloc, "_free_batch_shells"))
        lid = id(alloc._live_views)
        ba = _batch("shm-recycle-exact", [11, 12, 13], [i * 32 for i in range(6)])
        rows1 = _gv(alloc, ba)
        va_list = rows1[0] + rows1[1]
        va_ids = [id(v) for v in va_list]
        ok(all(v._live_views_id == lid for v in va_list))
        scc.release_passive_views_batch(va_list)
        eq(alloc._recycled_batch_views_total, 6)
        rs = _gv(alloc, ba)
        eq([id(v) for v in rs[0] + rs[1]], va_ids)
        eq(alloc._recycled_batch_views_total, 0)
        scc.release_passive_views_batch(rs[0] + rs[1])
        eq(alloc._recycled_batch_views_total, 6)
        bdh = _batch("shm-recycle-exact", [21, 22, 23], [i * 32 for i in range(6)])
        bdo = _batch(
            "shm-recycle-exact", [11, 12, 13], [4096 + i * 32 for i in range(6)]
        )
        rdh, rdo = _gv(alloc, bdh), _gv(alloc, bdo)
        va_set = set(va_ids)
        for v in rdh[0] + rdh[1] + rdo[0] + rdo[1]:
            ok(id(v) not in va_set and v._live_views_id == lid)
        eq(alloc._recycled_batch_views_total, 6)
        scc.release_passive_views_batch(rdh[0] + rdh[1] + rdo[0] + rdo[1])
        bex = _batch("shm-recycle-exact", [901, 902], [8192, 8224], po=[8000], ps=[64])
        pb, pp = _bv(alloc, bex), _pv(alloc, bex)
        pb.ref_count_down()
        pp.ref_count_down()
        no(pb.is_valid() or pp.is_valid())
        nb, np_ = _bv(alloc, bex, pb), _pv(alloc, bex, pp)
        ok(nb is not pb and np_ is not pp)
        eq((nb._live_views_id, np_._live_views_id), (lid, lid))
        ob, op = _bv(alloc, bex), _pv(alloc, bex)
        for v in (nb, ob, np_, op):
            v.ref_count_down()
        rb, rp = _bv(alloc, bex, ob), _pv(alloc, bex, op)
        ok(rb is nb and rp is np_)
        _, m_ord, a_ord = _slab("shm-recycle-ord", 4096, 2, reuse=False)
        b_ord = _batch("shm-recycle-ord", [701], [], po=[0], ps=[64])
        vo = _pv(a_ord, b_ord)
        assert a_ord._live_views is not None
        eq(vo._live_views_id, id(a_ord._live_views))
        a_ord._unregistered = True
        no(a_ord._matches_view(vo, (), _SH, torch.uint8, _FMT, 0, 64))
        with self.assertRaises(scc.SharedCPUCacheError):
            _pv(a_ord, b_ord)
        with patch.object(scc.SharedSlabMapping, "unlink", return_value=None):
            m_ord.close()
            mapping.close()
        ok(alloc._unregistered and alloc.buffer is None)
        eq((rb._live_views_id, rp._live_views_id), (0, 0))
        no(rb.is_valid() or rp.is_valid())
        eq(
            (alloc._recycled_batch_views_total, alloc._recycled_page_views_total),
            (0, 0),
        )
        for fn in (
            lambda: _pv(alloc, bex),
            lambda: _bv(alloc, bex),
            lambda: _gv(alloc, ba),
        ):
            with self.assertRaises(scc.SharedCPUCacheError):
                fn()

    def test_pin_monitor_unpin_and_release_many_and_lease_release(self) -> None:
        was_none = PinMonitor._instance is None
        mon, eq, ok = PinMonitor.GetOrCreate(_cfg()), self.assertEqual, self.assertTrue
        tracked: list[int] = []

        def _clean() -> None:
            with mon._objects_lock:
                for i in tracked:
                    mon._pinned_objects.pop(i, None)
                    mon._pin_generations.pop(i, None)
            if was_none:
                PinMonitor.DestroyInstance()

        self.addCleanup(_clean)
        alloc = _palloc(16384)
        self.addCleanup(lambda: [b.invalidate() for b in list(alloc.free_blocks)])

        def _at() -> _TMO:
            obj = alloc.allocate([_SH], [torch.uint8], _FMT)
            assert isinstance(obj, _TMO)
            tracked.append(id(obj))
            return obj

        _freed = lambda obj: any(x is obj for x in alloc.free_blocks)  # noqa: E731
        _in_m = lambda obj: id(obj) in mon._pinned_objects  # noqa: E731
        o1, o2, o3, o4 = _at(), _at(), _at(), _at()
        o1.pin()
        o2.pin()
        o2.pin()
        o2.ref_count_up()
        o4.invalidate()
        o4.meta.ref_count = 7
        eq((_in_m(o1), _in_m(o2), _in_m(o3)), (True, True, False))
        _boom = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # noqa: E731
        bad: Any = _N(
            is_pinned=True, is_valid=lambda: True, unpin=_boom, ref_count_down=_boom
        )
        _TMO.unpin_and_release_many([o1, o2, o3, o4, bad], catch_errors=True)
        chk = [
            (o1, True, False, 0, 0),
            (o3, True, False, 0, 0),
            (o2, False, True, 1, 1),
            (o4, False, False, 7, 0),
        ]
        for obj, ef, em, rc, pc in chk:
            eq((_freed(obj), _in_m(obj)), (ef, em))
            eq((obj.get_ref_count(), obj.meta.pin_count), (rc, pc))
        _TMO.unpin_and_release_many([o2])
        ok(_freed(o2) and not _in_m(o2))
        o5, o6, o7, toctou, zero_ref, dup = _at(), _at(), _at(), _at(), _at(), _at()
        mon.on_pin_many([o7])
        ok(_in_m(o7))
        mon.on_unpin_many([o7])
        self.assertFalse(_in_m(o7))
        o7.ref_count_down()
        toctou.meta.pin_count = 1
        mon.on_pin_many([toctou])
        mon.on_unpin_many([toctou])
        ok(_in_m(toctou))
        toctou.meta.pin_count = 0
        mon.on_unpin_many([toctou])
        self.assertFalse(_in_m(toctou))
        toctou.ref_count_down()
        zero_ref.pin()
        zero_ref.meta.ref_count = 0
        ok(_in_m(zero_ref))
        _TMO.unpin_and_release_many([zero_ref])
        eq((zero_ref.get_ref_count(), zero_ref.meta.pin_count), (0, 0))
        ok(not _in_m(zero_ref) and _freed(zero_ref))
        free_calls, ofree = [], alloc.batched_free

        def _sf(objs: list[Any], at: str | None = None) -> None:
            free_calls.append(list(objs))
            ofree(objs, at)

        with patch.object(alloc, "batched_free", side_effect=_sf):
            _TMO.unpin_and_release_many([dup, dup])
        eq((len(free_calls), len(free_calls[0])), (1, 1))
        ok(free_calls[0][0] is dup and dup.get_ref_count() == 0 and _freed(dup))
        _TMO.pin_many([o5, o6])
        ok(_in_m(o5) and _in_m(o6))
        lease = scc.SharedCPURequestLease(
            "req-lease", 1, is_rank0=True, active=True, groups={0: [[o5, o6]]}
        )
        lease.close()
        eq((_freed(o5), _freed(o6)), (True, True))
        eq(
            (o5.get_ref_count(), o6.get_ref_count(), _in_m(o5), _in_m(o6)),
            (0, 0, False, False),
        )
        claimed = _at()
        claimed.meta.pin_count = 1
        claimed.meta.ref_count = 0
        claimed._free_claimed = True
        alloc.batched_free([claimed])
        _TMO.unpin_and_release_many([claimed])
        eq(sum(1 for b in alloc.free_blocks if b is claimed), 1)
        drained: list[Any] = []
        re_obj: Any = None
        while alloc.free_blocks:
            cand = alloc.allocate([_SH], [torch.uint8], _FMT)
            assert cand is not None
            tracked.append(id(cand))
            if cand is claimed:
                re_obj = cand
                break
            drained.append(cand)
        for d_obj in drained:
            d_obj.ref_count_down()
        self.assertIs(re_obj, claimed)
        self.assertFalse(getattr(re_obj, "_free_claimed", False))
        eq(re_obj.get_ref_count(), 1)
        re_obj.ref_count_down()
        eq(sum(1 for b in alloc.free_blocks if b is re_obj), 1)

    def test_sync_lookup_client_retains_status_past_4096_and_bare_clear(self) -> None:
        rpc_calls = 0

        def _send_and_recv(_msg: Any) -> list[bytes]:
            nonlocal rpc_calls
            rpc_calls += 1
            return [(256).to_bytes(4, "big")]

        transport: Any = _N(
            world_size=1, send_and_recv_all=_send_and_recv, close=lambda: None
        )
        client = llc.LMCacheLookupClient(_cfg(128), _meta(128), transport)
        self.addCleanup(client.close)
        toks = list(range(256))
        for i in range(4100):
            self.assertEqual(client.lookup(toks, f"req-{i}"), 256)
        self.assertEqual(rpc_calls, 4100)
        self.assertEqual(client.lookup_cache("req-0"), 256)
        self.assertEqual(rpc_calls, 4100)

        bare = object.__new__(llc.LMCacheLookupClient)
        bare.reqs_status = {"x": 256}
        bare.clear_lookup_status("x")
        self.assertEqual(bare.lookup_cache("x"), -1)

    def test_release_dense_load_source_owners_batch_and_duck_fallback(self) -> None:
        _, mapping, alloc = _slab("shm-owners", 16384)
        b_ow = _batch("shm-owners", [401, 402, 403], [i * 32 for i in range(6)])
        rows, eq = _gv(alloc, b_ow), self.assertEqual
        owners = tuple(rows[0] + rows[1])
        eq(len(owners), 6)
        _Impl._release_dense_load_source_owners(owners, engine=None, synchronize=False)
        eq(alloc._recycled_batch_views_total, len(owners))
        self.assertTrue(
            all(o.get_ref_count() == 0 and not o.is_valid() for o in owners)
        )
        with patch.object(scc.SharedSlabMapping, "unlink", return_value=None):
            mapping.close()
        calls: list[str] = []

        def _mkd(ok: bool, p: str) -> Any:
            up = lambda: calls.append(f"{p}u")  # noqa: E731
            rd = lambda: calls.append(f"{p}r")  # noqa: E731
            return _N(is_pinned=True, is_valid=lambda: ok, unpin=up, ref_count_down=rd)

        ducks: tuple[Any, ...] = (_mkd(True, "v"), _mkd(False, "i"), _N())
        _Impl._release_dense_load_source_owners(ducks, engine=None, synchronize=False)
        eq(calls, ["vu", "vr", "iu"])
        calls.clear()
        _TMO.unpin_and_release_many(ducks)
        eq(calls, ["vu", "vr", "iu"])
        a, bb, c, d = (_N(tag=t) for t in ("a", "b", "c", "d"))
        st = va.WorkerRetrieveState(
            cached_memory_objs_indexer=[[a, bb], [bb, c]],
            cached_tensors_indexer=[[c, d], [a]],
        )
        dedup = _Impl._dense_load_source_owners(st)
        eq([id(x) for x in dedup], [id(a), id(bb), id(c), id(d)])

        # kv_both sparse decode first-step metadata_token_ids backfill
        impl_sd = _Impl.__new__(_Impl)
        st_sd = va.WorkerRetrieveState(
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
            cached_keys=[["k0"]],
        )
        impl_sd._worker_retrieve_state = {"req-sd": st_sd}
        impl_sd._prepared_sparse_sources_current = lambda *_a, **_k: True  # type: ignore[method-assign]
        req_sd = va.ReqMeta(
            req_id="req-sd",
            token_ids=list(range(300)),
            slot_mapping=[],
            is_sparse_decode=True,
            load_spec=va.LoadSpec(0, 256, can_load=True),
        )
        meta_sd: Any = _N(requests=[req_sd])
        impl_sd._finalize_worker_retrieve_state_from_metadata(meta_sd)
        eq(st_sd.metadata_token_ids, list(range(256)))

    def test_pin_monitor_timeout_generation_guard_and_pinned_dup_release(self) -> None:
        was_none = PinMonitor._instance is None
        mon, eq = PinMonitor.GetOrCreate(_cfg()), self.assertEqual
        tracked: list[int] = []

        def _clean() -> None:
            with mon._objects_lock:
                for i in tracked:
                    mon._pinned_objects.pop(i, None)
                    mon._pin_generations.pop(i, None)
            if was_none:
                PinMonitor.DestroyInstance()

        self.addCleanup(_clean)
        alloc = _palloc(256)
        self.addCleanup(lambda: [b.invalidate() for b in list(alloc.free_blocks)])
        obj = alloc.allocate([_SH], [torch.uint8], _FMT)
        assert isinstance(obj, _TMO)
        tracked.append(id(obj))
        obj.pin()
        mon.on_pin(obj)
        stale_gen = mon._pin_generations[id(obj)]
        obj.unpin()
        mon.on_unpin(obj)
        obj.pin()
        mon.on_pin(obj)
        self.assertFalse(mon._force_unpin_timeout_object(obj, 999.0, stale_gen))
        eq(obj.meta.pin_count, 1)
        self.assertIn(id(obj), mon._pinned_objects)
        obj.unpin()
        mon.on_unpin(obj)
        obj.ref_count_down()

        dup = alloc.allocate([_SH], [torch.uint8], _FMT)
        assert isinstance(dup, _TMO)
        tracked.append(id(dup))
        dup.pin()
        free_calls: list[list[Any]] = []
        orig_bf = alloc.batched_free

        def _cap_bf(objs: list[Any], at: str | None = None) -> None:
            free_calls.append(list(objs))
            orig_bf(objs, at)

        with patch.object(alloc, "batched_free", side_effect=_cap_bf):
            _TMO.unpin_and_release_many([dup, dup])
        eq((dup.get_ref_count(), dup.meta.pin_count, len(free_calls)), (0, 0, 1))
        self.assertTrue(any(x is dup for x in alloc.free_blocks))

    def test_bypass_lookup_client_stashes_and_clears_via_base_interface(self) -> None:
        cfg, meta = _cfg(128), _meta(128)
        db = ChunkedTokenDatabase(cfg, meta)
        lookup_calls: list[dict[str, Any]] = []
        cleaned: list[str] = []

        def _fake_lookup(**kw: Any) -> int:
            lookup_calls.append(kw)
            return 256

        fake_eng: Any = _N(
            token_database=db,
            lookup=_fake_lookup,
            cleanup_memory_objs=lambda rid: cleaned.append(rid),
        )
        client = blc.LMCacheBypassLookupClient(cfg, meta, fake_eng)
        self.addCleanup(client.close)
        toks = list(range(256))

        self.assertEqual(client.lookup(toks, "req-byp"), 256)
        self.assertEqual(len(lookup_calls), 1)
        popped_h = client.pop_lookup_chunk_hashes("req-byp")
        popped_d = client.pop_lookup_chunk_digests("req-byp")
        self.assertIsNotNone(popped_h)
        self.assertIsNotNone(popped_d)
        assert popped_h is not None and popped_d is not None
        self.assertEqual(len(popped_h[0]), 2)
        self.assertEqual(popped_h[1], (128, 128))
        self.assertEqual(len(popped_d[1]), 2)

        self.assertEqual(client.lookup(toks, "req-byp-2"), 256)
        client.cleanup_lookup("req-byp-2")
        self.assertEqual(cleaned, ["req-byp-2"])
        self.assertIsNone(client.pop_lookup_chunk_hashes("req-byp-2"))
        self.assertIsNone(client.pop_lookup_chunk_digests("req-byp-2"))

    def test_batched_from_gpu_and_shared_retrieve_abort_fences(self) -> None:
        # Standard
        from contextlib import nullcontext
        from typing import Any, cast
        from unittest.mock import patch

        # Third Party
        import torch

        # First Party
        from lmcache.v1.memory_management import MemoryFormat
        import lmcache.v1.cache_engine as ce_mod
        import lmcache.v1.gpu_connector.gpu_connectors as gc_mod

        def _make_fake_mem(events: list[str], tag: str) -> Any:
            obj: Any = _N(
                tensor=torch.zeros((2, 4, 8), dtype=torch.float32),
                metadata=_N(fmt=MemoryFormat.KV_T2D, cached_positions=None),
                ref_count=1,
                is_pinned=True,
                parent_allocator=None,
            )

            def _is_valid() -> bool:
                return bool(obj.ref_count > 0)

            def _ref_up() -> None:
                obj.ref_count += 1
                events.append(f"{tag}_up")

            def _ref_down() -> None:
                obj.ref_count -= 1
                events.append(f"{tag}_down")

            def _unpin() -> None:
                obj.is_pinned = False
                events.append(f"{tag}_unpin")

            obj.is_valid = _is_valid
            obj.ref_count_up = _ref_up
            obj.ref_count_down = _ref_down
            obj.unpin = _unpin
            return obj

        def _make_fake_key(name: str) -> Any:
            return _N(
                name=name,
                split_layers=lambda nl, n=name: [
                    _N(name=f"{n}_l{i}") for i in range(nl)
                ],
                get_layer=lambda lid, n=name: _N(name=f"{n}_l{lid}"),
            )

        def _make_layerwise_conn(store_sync_fn: Any) -> tuple[Any, Any]:
            fake_load = _N(synchronize=lambda: None, wait_stream=lambda _s: None)
            fake_store = _N(synchronize=store_sync_fn, wait_stream=lambda _s: None)
            streams = [fake_load, fake_store]
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
            return conn, fake_store

        fake_ops = _N(
            TransferDirection=_N(D2H="D2H"),
            single_layer_kv_transfer=lambda *_a, **_k: None,
        )

        # 1a) VLLMPagedMemLayerwiseGPUConnector.batched_from_gpu normal .close()
        events_ok: list[str] = []
        tmp_ok = _make_fake_mem(events_ok, "tmp")
        mem0_ok = _make_fake_mem(events_ok, "mem0")
        mem1_ok = _make_fake_mem(events_ok, "mem1")
        conn_ok, _ = _make_layerwise_conn(lambda: events_ok.append("store_sync"))
        conn_ok.gpu_kv_format = 1
        conn_ok.gpu_buffer_allocator = _N(allocate=lambda *_a, **_k: tmp_ok)
        kvcaches = [torch.zeros((4, 2, 8), dtype=torch.float32) for _ in range(2)]
        slot_map = torch.arange(4, dtype=torch.long)
        cur_stream = _N(wait_stream=lambda _s: None)

        with (
            patch(
                "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.current_stream",
                return_value=cur_stream,
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
            gen_ok = conn_ok.batched_from_gpu(
                [[mem0_ok], [mem1_ok]],
                [0],
                [4],
                kvcaches=kvcaches,
                slot_mapping=slot_map,
                sync=True,
            )
            next(gen_ok)
            gen_ok.close()
        mem0_ok.ref_count_down()
        self.assertLess(events_ok.index("store_sync"), events_ok.index("tmp_down"))
        self.assertLess(events_ok.index("tmp_down"), events_ok.index("mem0_down"))
        self.assertIsNone(getattr(conn_ok, "_layerwise_prefill_unfenced_stores", None))

        # 1b) Hung store_stream.synchronize() quarantines & blocks close/destroy
        events_hung: list[str] = []
        tmp_h = _make_fake_mem(events_hung, "tmp")
        mem0_h = _make_fake_mem(events_hung, "mem0")
        mem1_h = _make_fake_mem(events_hung, "mem1")

        def _hung_sync() -> None:
            events_hung.append("store_sync_fail")
            raise RuntimeError("hung store")

        conn_h, store_stream_h = _make_layerwise_conn(_hung_sync)
        conn_h.gpu_kv_format = 1
        conn_h.gpu_buffer_allocator = _N(allocate=lambda *_a, **_k: tmp_h)
        with (
            patch(
                "lmcache.v1.gpu_connector.gpu_connectors.torch.cuda.current_stream",
                return_value=cur_stream,
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
            gen_h = conn_h.batched_from_gpu(
                [[mem0_h], [mem1_h]],
                [0],
                [4],
                kvcaches=kvcaches,
                slot_mapping=slot_map,
                sync=True,
            )
            next(gen_h)
            with self.assertRaisesRegex(RuntimeError, "hung store"):
                gen_h.close()

        self.assertEqual(
            (mem0_h.ref_count, mem1_h.ref_count, tmp_h.ref_count), (2, 2, 2)
        )
        self.assertNotIn("tmp_down", events_hung)
        unfenced = conn_h._layerwise_prefill_unfenced_stores
        self.assertEqual(len(unfenced), 1)
        self.assertIs(unfenced[0][0], store_stream_h)
        self.assertEqual(unfenced[0][1], (mem0_h, mem1_h, tmp_h))

        sm_closed = [0]
        eng_store: Any = object.__new__(ce_mod.LMCacheEngine)
        eng_store.gpu_connector = conn_h
        eng_store.storage_manager = _N(
            close=lambda: sm_closed.__setitem__(0, sm_closed[0] + 1)
        )
        eng_store.lmcache_worker = None
        eng_store.shared_cpu_cache_mapping = None
        eng_store._shared_cpu_request_leases = {}
        inst_id = "inst-shared-store-abort-fence"
        builder = ce_mod.LMCacheEngineBuilder
        self.addCleanup(lambda: builder._instances.pop(inst_id, None))
        self.addCleanup(lambda: builder._cfgs.pop(inst_id, None))
        self.addCleanup(lambda: builder._metadatas.pop(inst_id, None))
        self.addCleanup(lambda: builder._stat_loggers.pop(inst_id, None))
        builder._instances[inst_id] = eng_store
        builder._stat_loggers[inst_id] = cast(Any, None)
        builder.destroy(inst_id)
        self.assertEqual(sm_closed[0], 0)
        self.assertIn(inst_id, builder._instances)

        # 2) Shared-CPU Retrieve real _release_retained_dense_retrieve_objs & close()
        def _make_retrieve_engine(
            role: str,
            sync_calls: list[str],
            sync_exc: Exception | None,
            objs_ev: list[str],
        ) -> tuple[Any, Any]:
            m_obj = _make_fake_mem(objs_ev, role)
            eng: Any = object.__new__(ce_mod.LMCacheEngine)
            eng.config = _N(
                remote_url=None,
                enable_controller=False,
                enable_async_loading=False,
            )
            eng.metadata = _N(first_rank=0, worker_id=0)
            eng.storage_manager = _N()
            eng.stats_monitor = _N(on_retrieve_finished=lambda *_a: None)
            eng.shared_cpu_cache_generation = 1
            eng._unsafe_layerwise_retrieve_sources = []
            eng._shared_cpu_request_leases = {}
            eng.num_layers_for_group = lambda _g: 2
            eng._remote_fill_pair_lookup_enabled = lambda: False
            eng._shared_prefill_source_state = lambda *_a, **_k: None
            eng._make_shared_handles_for_layer = lambda **_k: ["h0"]
            eng._broadcast_shared_envelope = lambda _env: None
            eng._resolve_shared_rank0_layer_mem_objs = lambda **_k: [m_obj]
            eng._receive_matching_shared_envelope = lambda **kw: (
                ce_mod.SharedHandleEnvelope(
                    request_id=kw["req_id"],
                    phase=kw["phase"],
                    request_ordinal=0,
                    layer_id=kw["layer_id"],
                    kv_group=kw["kv_group"],
                    status="ok",
                    generation=1,
                    handles=cast(Any, ["h0"]),
                )
            )
            eng._validate_shared_layerwise_envelope = lambda *_a, **_k: None
            eng._expected_shared_cpu_chunk_metadata = lambda **_k: (
                torch.Size([2, 4, 8]),
                torch.float32,
                MemoryFormat.KV_T2D,
            )
            eng.shared_cpu_cache_passive_allocator = _N(
                create_view=lambda *_a, **_k: m_obj
            )

            def _load_sync() -> None:
                sync_calls.append("load_sync")
                objs_ev.append("load_sync")
                if sync_exc is not None:
                    raise sync_exc

            conn_ret, _ = _make_layerwise_conn(lambda: None)
            conn_ret.load_stream = _N(synchronize=_load_sync)

            def _fake_batched_to_gpu(*_a: Any, **_k: Any) -> Any:
                yield
                while True:
                    yield

            conn_ret.batched_to_gpu = _fake_batched_to_gpu
            eng.gpu_connector = conn_ret
            return eng, m_obj

        for role in ("rank0", "passive"):
            sc_ok: list[str] = []
            ev_ok_r: list[str] = []
            eng_r, obj_r = _make_retrieve_engine(role, sc_ok, None, ev_ok_r)
            keys = [[_make_fake_key("k0")], [_make_fake_key("k1")]]
            if role == "rank0":
                gen_r = eng_r._retrieve_layer_shared_rank0(
                    starts=[0],
                    ends=[4],
                    keys_layer_major=keys,
                    chunk_locations_layer_major=[
                        ["LocalCPUBackend"],
                        ["LocalCPUBackend"],
                    ],
                    location="LocalCPUBackend",
                    ret_mask=torch.zeros(4, dtype=torch.bool),
                    monitor_req_id=1,
                    req_id="req-shared-abort-ok",
                    kv_group=0,
                    kwargs={"deferred_layerwise_get": False},
                )
            else:
                gen_r = eng_r._retrieve_layer_shared_passive(
                    starts_all=[0],
                    ends_all=[4],
                    keys_layer_major=keys,
                    ret_mask=torch.zeros(4, dtype=torch.bool),
                    monitor_req_id=1,
                    req_id="req-shared-abort-ok",
                    kv_group=0,
                    kwargs={"deferred_layerwise_get": False},
                )
            next(gen_r)
            gen_r.close()
            self.assertEqual(len(sc_ok), 1)
            self.assertEqual(obj_r.ref_count, 0)
            self.assertEqual(obj_r.is_pinned, role != "rank0")
            self.assertLess(ev_ok_r.index("load_sync"), ev_ok_r.index(f"{role}_down"))
            self.assertEqual(eng_r._unsafe_layerwise_retrieve_sources, [])

            sc_fail: list[str] = []
            ev_fail_r: list[str] = []
            eng_f, obj_f = _make_retrieve_engine(
                role, sc_fail, RuntimeError("hung load"), ev_fail_r
            )
            if role == "rank0":
                gen_f = eng_f._retrieve_layer_shared_rank0(
                    starts=[0],
                    ends=[4],
                    keys_layer_major=keys,
                    chunk_locations_layer_major=[
                        ["LocalCPUBackend"],
                        ["LocalCPUBackend"],
                    ],
                    location="LocalCPUBackend",
                    ret_mask=torch.zeros(4, dtype=torch.bool),
                    monitor_req_id=2,
                    req_id="req-shared-abort-fail",
                    kv_group=0,
                    kwargs={"deferred_layerwise_get": False},
                )
            else:
                gen_f = eng_f._retrieve_layer_shared_passive(
                    starts_all=[0],
                    ends_all=[4],
                    keys_layer_major=keys,
                    ret_mask=torch.zeros(4, dtype=torch.bool),
                    monitor_req_id=2,
                    req_id="req-shared-abort-fail",
                    kv_group=0,
                    kwargs={"deferred_layerwise_get": False},
                )
            next(gen_f)
            with self.assertRaisesRegex(RuntimeError, "hung load"):
                gen_f.close()
            self.assertEqual(len(sc_fail), 1)
            self.assertEqual(obj_f.ref_count, 1)
            self.assertTrue(obj_f.is_pinned)
            self.assertNotIn(f"{role}_down", ev_fail_r)
            self.assertEqual(eng_f._unsafe_layerwise_retrieve_sources, [obj_f])

        # 2b) Real TensorMemoryObj (rank0) + PassiveSharedViewAllocator on .close()
        was_none = PinMonitor._instance is None
        mon = PinMonitor.GetOrCreate(_cfg())
        tracked_ids: list[int] = []

        def _cleanup_mon() -> None:
            with mon._objects_lock:
                for oid in tracked_ids:
                    mon._pinned_objects.pop(oid, None)
                    mon._pin_generations.pop(oid, None)
            if was_none:
                PinMonitor.DestroyInstance()

        self.addCleanup(_cleanup_mon)
        real_alloc = _palloc(16384)
        self.addCleanup(
            lambda: [blk.invalidate() for blk in list(real_alloc.free_blocks)]
        )
        tmo0 = real_alloc.allocate([_SH], [torch.uint8], _FMT)
        tmo1 = real_alloc.allocate([_SH], [torch.uint8], _FMT)
        assert isinstance(tmo0, _TMO) and isinstance(tmo1, _TMO)
        tracked_ids.extend((id(tmo0), id(tmo1)))
        tmo0.pin()
        tmo1.pin()

        sc_real_r0: list[str] = []
        eng_real_r0, _ = _make_retrieve_engine("rank0", sc_real_r0, None, [])
        keys_2c = [
            [_make_fake_key("k0"), _make_fake_key("k1")],
            [_make_fake_key("k2"), _make_fake_key("k3")],
        ]
        eng_real_r0._resolve_shared_rank0_layer_mem_objs = lambda **_k: [tmo0, tmo1]
        gen_real_r0 = eng_real_r0._retrieve_layer_shared_rank0(
            starts=[0, 4],
            ends=[4, 8],
            keys_layer_major=keys_2c,
            chunk_locations_layer_major=[
                ["LocalCPUBackend", "LocalCPUBackend"],
                ["LocalCPUBackend", "LocalCPUBackend"],
            ],
            location="LocalCPUBackend",
            ret_mask=torch.zeros(8, dtype=torch.bool),
            monitor_req_id=3,
            req_id="req-shared-abort-real-r0",
            kv_group=0,
            kwargs={"deferred_layerwise_get": False},
        )
        next(gen_real_r0)
        gen_real_r0.close()
        self.assertEqual(len(sc_real_r0), 1)
        for tmo in (tmo0, tmo1):
            self.assertFalse(tmo.is_pinned)
            self.assertEqual(tmo.get_ref_count(), 0)
            self.assertIn(tmo, real_alloc.free_blocks)
            self.assertNotIn(id(tmo), mon._pinned_objects)

        _, passive_map, passive_alloc = _slab(
            "shm-shared-abort-real-passive", 16384, gen=1
        )

        def _close_passive_map() -> None:
            with patch.object(scc.SharedSlabMapping, "unlink", return_value=None):
                passive_map.close()

        self.addCleanup(_close_passive_map)
        p_batch = _batch("shm-shared-abort-real-passive", [11, 12], [0, 32], lb=32)
        pv0 = _bv(passive_alloc, p_batch, lid=0, ci=0)
        pv1 = _bv(passive_alloc, p_batch, lid=0, ci=1)
        passive_views = [pv0, pv1]

        sc_real_p: list[str] = []
        eng_real_p, _ = _make_retrieve_engine("passive", sc_real_p, None, [])
        eng_real_p._make_shared_handles_for_layer = lambda **_k: ["h0", "h1"]
        eng_real_p._receive_matching_shared_envelope = lambda **kw: (
            ce_mod.SharedHandleEnvelope(
                request_id=kw["req_id"],
                phase=kw["phase"],
                request_ordinal=0,
                layer_id=kw["layer_id"],
                kv_group=kw["kv_group"],
                status="ok",
                generation=1,
                handles=cast(Any, ["h0", "h1"]),
            )
        )
        eng_real_p.shared_cpu_cache_passive_allocator = _N(
            create_view=lambda *_a, **_k: passive_views.pop(0)
        )
        gen_real_p = eng_real_p._retrieve_layer_shared_passive(
            starts_all=[0, 4],
            ends_all=[4, 8],
            keys_layer_major=keys_2c,
            ret_mask=torch.zeros(8, dtype=torch.bool),
            monitor_req_id=4,
            req_id="req-shared-abort-real-passive",
            kv_group=0,
            kwargs={"deferred_layerwise_get": False},
        )
        next(gen_real_p)
        gen_real_p.close()
        self.assertEqual(len(sc_real_p), 1)
        for pv in (pv0, pv1):
            self.assertEqual(pv.get_ref_count(), 0)
            self.assertFalse(pv.is_valid())
        self.assertEqual(passive_alloc._recycled_batch_views_total, 2)

        ev_b: list[str] = []
        eng_b, obj_b1 = _make_retrieve_engine("dense", [], None, ev_b)
        eng_b.gpu_connector.synchronize_dense_load_stream = lambda: (
            _ for _ in ()
        ).throw(RuntimeError("hung dense sync"))
        retained_list_1 = [obj_b1]
        with self.assertRaisesRegex(RuntimeError, "hung dense sync"):
            eng_b._release_retained_dense_retrieve_objs(
                retained_list_1,
                unpin=True,
                kwargs={"_retain_shared_dense_cache": True},
            )
        self.assertEqual(
            (obj_b1.ref_count, obj_b1.is_pinned, retained_list_1), (1, True, [])
        )
        self.assertEqual(eng_b._unsafe_layerwise_retrieve_sources, [obj_b1])

        obj_b2 = _make_fake_mem(ev_b, "dense_nosync")
        eng_b.gpu_connector = _N()
        retained_list_2 = [obj_b2]
        with self.assertRaisesRegex(RuntimeError, "load-stream sync"):
            eng_b._release_retained_dense_retrieve_objs(
                retained_list_2,
                unpin=True,
                kwargs={"_retain_shared_dense_cache": True},
            )
        self.assertEqual(
            (obj_b2.ref_count, obj_b2.is_pinned, retained_list_2), (1, True, [])
        )
        self.assertEqual(eng_b._unsafe_layerwise_retrieve_sources, [obj_b1, obj_b2])

    def test_all_lookup_clients_close_clear_stashes_with_lock(self) -> None:
        # Standard
        import threading

        # First Party
        import lmcache.v1.lookup_client.lmcache_async_lookup_client as alc
        import lmcache.v1.lookup_client.lmcache_lookup_client as slc
        import lmcache.v1.lookup_client.lmcache_lookup_client_bypass as blc
        import lmcache.v1.lookup_client.mooncake_lookup_client as mlc

        class _TrackingLock:
            def __init__(self) -> None:
                self._rlock = threading.RLock()
                self.acquires = 0

            def __enter__(self) -> "_TrackingLock":
                self._rlock.acquire()
                self.acquires += 1
                return self

            def __exit__(self, *args: Any) -> None:
                self._rlock.release()

        mc: Any = object.__new__(mlc.MooncakeLookupClient)
        mc.store = _N(close=lambda: None)

        ac: Any = object.__new__(alc.LMCacheAsyncLookupClient)
        ac.running = True
        ac.thread = _N(is_alive=lambda: False, join=lambda **_k: None)
        ac.push_sockets = [_N(close=lambda **_k: None)]
        ac.pull_socket = _N(close=lambda **_k: None)
        ac.ctx = _N(term=lambda: None)

        bc: Any = object.__new__(blc.LMCacheBypassLookupClient)

        lc: Any = object.__new__(slc.LMCacheLookupClient)
        lc.transport = _N(close=lambda: None)
        lc.reqs_status = {"req-sync": 128}

        for name, raw_client in (
            ("mooncake", mc),
            ("async", ac),
            ("bypass", bc),
            ("sync", lc),
        ):
            client: Any = raw_client
            with self.subTest(client=name):
                tlock = _TrackingLock()
                client._stash_lock = tlock
                client._stash_lookup_chunk_hashes(
                    f"req-{name}",
                    [10, 20, 30, 40],
                    ["h0", "h1"],
                    [2, 2],
                    [20, 40],
                )
                before_acquires = tlock.acquires
                self.assertEqual(len(client._lookup_chunk_hashes or {}), 1)
                self.assertEqual(len(client._lookup_chunk_digests or {}), 1)
                client.close()
                self.assertGreater(tlock.acquires, before_acquires)
                self.assertEqual(client._lookup_chunk_hashes, {})
                self.assertEqual(client._lookup_chunk_digests, {})
                self.assertIsNone(client.pop_lookup_chunk_hashes(f"req-{name}"))
                self.assertIsNone(client.pop_lookup_chunk_digests(f"req-{name}"))
                if name == "sync":
                    self.assertEqual(lc.reqs_status, {})


if __name__ == "__main__":
    unittest.main()
