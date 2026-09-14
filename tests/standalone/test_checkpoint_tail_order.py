# SPDX-License-Identifier: Apache-2.0
"""Late boundary copies share cold-load readiness and terminal cleanup."""

from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest

from test_checkpoint_adapter import method


@pytest.mark.parametrize("failure", [None, "copy", "record", "fence"])
def test_tail_copy_precedes_readiness_and_failure_fences_latest_work(failure):
    calls, owner, readiness = [], object(), object()
    state = NS(
        req_id="r",
        dense_load_readiness=None,
        dense_load_source_owners=(),
        has_cache=lambda: True,
        prepared_sparse_sources={0: object()},
    )
    request = NS(
        req_id="r", load_spec=NS(checkpoint_tail_slots=[7], dsa_group1_direct_hbm=False)
    )
    dependency = Future()
    dependency.set_result((None, "older-index-event", 0, 0))
    plan = dict(
        request=request,
        token_count=7,
        tokens=list(range(7)),
        token_mask=object(),
        latent_shared_ready=Future(),
        latent_kvcaches=[object()],
        indexer_source_owners=(owner,),
    )

    def copy(*args):
        assert plan["latent_shared_ready"].done()
        calls.append("copy")
        if failure in ("copy", "fence"):
            raise RuntimeError("tail copy failed")
        return readiness

    def record(state, *, readiness, additional_owners):
        calls.append("record")
        assert readiness is not None
        if failure == "record":
            raise RuntimeError("record failed")
        state.dense_load_readiness = readiness
        state.dense_load_source_owners = additional_owners

    def fence():
        calls.append("fence")
        if failure == "fence":
            raise RuntimeError("fence failed")

    obj = NS(
        num_layers=1,
        _num_layers_for_group=lambda group: 1,
        lmcache_engine=NS(load_checkpoint_resident_tail=copy),
        _record_dsa_cold_dense_load_readiness=record,
        _synchronize_dsa_cold_dense_load=fence,
        _synchronize_dsa_cold_dense_readiness=lambda *a: pytest.fail(
            "only fenced old work"
        ),
        _release_dense_load_source_owners=lambda *a, **kw: calls.append("release"),
        _release_unadopted_shared_request_objects=lambda *a: None,
        _release_shared_worker_retrieve_state=lambda *a: None,
        _refresh_prepared_sparse_sources=lambda *a: None,
    )
    run = method("_run_dsa_cold_compact_load", logger=NS(exception=lambda *a: None))
    if failure is None:
        assert run(obj, plan, None, dependency, live_state=state) is state
        assert calls == ["copy", "record"]
        assert state.dense_load_readiness is readiness
    else:
        with pytest.raises(RuntimeError) as caught:
            run(obj, plan, None, dependency, live_state=state)
        if failure == "fence":
            assert calls == ["copy", "fence"]
            assert caught.value._lmcache_dsa_cold_state is state
            assert state.dense_load_source_owners == (owner,)
        else:
            assert calls[-2:] == ["fence", "release"]


@pytest.mark.parametrize("tail", [False, True])
def test_boundary_copy_participates_in_existing_capture_guard(tail):
    calls = []

    class Pending:
        def done(self):
            return False

        def result(self):
            calls.append("joined")

    spec = NS(dsa_group1_direct_hbm=True)
    if tail:
        spec.checkpoint_tail_slots = [1]
    entry = (1, Pending(), NS(load_spec=spec), set(), 0, Pending())
    obj = NS(_cold_load_coordinator=NS(futures={"r": entry}))
    method(
        "synchronize_staged_sfa_capture_unsafe_loads", logger=NS(info=lambda *a: None)
    )(obj)
    assert calls == (["joined", "joined"] if tail else [])
