# SPDX-License-Identifier: Apache-2.0
"""Exercise the public debug routing contract without vLLM or NPU imports."""

# Standard
from pathlib import Path
from types import MethodType, SimpleNamespace
import ast

# Third Party
import pytest


def connector(enabled: bool):
    source = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1Impl"
    )
    method = next(
        node for node in cls.body if getattr(node, "name", None) == "trace_attention_kv"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    scope = {"kv_hash_enabled": lambda: enabled}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    events = []
    gpu = SimpleNamespace(
        configure_kv_trace=lambda requests: events.append(("configure", requests)),
        trace_attention_kv=lambda *args, **kwargs: events.append(
            ("trace", args, kwargs)
        ),
        kv_trace=SimpleNamespace(
            error=lambda *args, **kwargs: events.append(("error", args, kwargs))
        ),
    )
    requests = [SimpleNamespace(req_id="request")]
    parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=requests)
    )
    obj = SimpleNamespace(
        lmcache_engine=SimpleNamespace(gpu_connector=gpu),
        _parent=parent,
        _layerwise_prefill_transfer_layer_id=lambda name, group: 7,
    )
    obj.trace_attention_kv = MethodType(scope[method.name], obj)
    return obj, gpu, events


def test_disabled_debug_never_reads_request_or_device_metadata():
    obj, _, events = connector(False)
    del obj._parent
    obj.trace_attention_kv("layer", group=1)
    assert events == []


def test_enabled_debug_routes_actual_group_owner_and_request():
    obj, _, events = connector(True)
    obj.trace_attention_kv("indexer", group=1, stage="P_LOAD", table="actual")
    assert events[0][0] == "configure"
    assert events[1] == ("trace", (7, 1), {"stage": "P_LOAD", "table": "actual"})


@pytest.mark.parametrize("capture", [False, True])
def test_capture_or_probe_failure_never_crashes_serving(capture):
    obj, gpu, events = connector(True)

    def fail(*args, **kwargs):
        raise (
            AssertionError("dummy capture") if capture else ValueError("probe failure")
        )

    if capture:
        obj._parent._get_connector_metadata = fail
    else:
        gpu.trace_attention_kv = fail
    obj.trace_attention_kv("layer", group=0, stage="P_LOAD")
    assert bool(events) is (not capture)
    if not capture:
        assert events[-1][0] == "error" and events[-1][1][0] == "request"
