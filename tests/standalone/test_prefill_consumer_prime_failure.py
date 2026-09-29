# SPDX-License-Identifier: Apache-2.0
"""Rank0 consumer priming failures must unblock passive shared receivers."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class MaterializationError(RuntimeError):
    pass


def rank0_method():
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_retrieve_layer_shared_rank0"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = dict(
        assert_layerwise_gpu_connector=lambda _: None,
        _RemoteFillMaterializationError=MaterializationError,
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("stage", ["constructor", "prime"])
@pytest.mark.parametrize("remote", [False, True])
def test_consumer_failure_broadcasts_layer_zero_error_before_raising(stage, remote):
    actions = []
    envelopes = []
    failure = RuntimeError("producer wait enqueue failed")

    def consume(*args, **kwargs):
        def prime():
            actions.append("prime")
            raise failure
            yield  # pragma: no cover

        actions.append("constructor")
        if stage == "constructor":
            raise failure
        return prime()

    def broadcast(envelope):
        actions.append("peer_error")
        envelopes.append(envelope)

    engine = NS(
        storage_manager=object(),
        gpu_connector=NS(batched_to_gpu=consume),
        metadata=NS(indexer_c8_layout=None),
        _remote_fill_pair_lookup_enabled=lambda: False,
        _broadcast_shared_envelope=broadcast,
        _shared_layerwise_error_envelope=lambda **kw: NS(status="error", **kw),
    )
    stream = rank0_method()(
        engine,
        starts=[0],
        ends=[4],
        keys_layer_major=[["key"]],
        chunk_locations_layer_major=[["RemoteBackend"]],
        location="RemoteBackend",
        ret_mask=None,
        monitor_req_id=0,
        req_id="r",
        kv_group=0,
        kwargs={"deferred_layerwise_get": True, "shared_cpu_request_ordinal": 9},
        planned_page_chunks=int(remote),
        remote_fill_plan=[("RemoteBackend", True)] if remote else None,
    )
    with pytest.raises(MaterializationError if remote else RuntimeError) as caught:
        next(stream)
    assert actions[-1] == "peer_error"
    assert len(envelopes) == 1 and envelopes[0].status == "error"
    assert envelopes[0].layer_id == 0 and envelopes[0].request_ordinal == 9
    assert envelopes[0].req_id == "r"
    assert envelopes[0].details["error"] == str(failure)
    assert caught.value.__cause__ is failure if remote else caught.value is failure
