# SPDX-License-Identifier: Apache-2.0
"""Exercise the deployed dynamic wrapper's preemption delegation contract."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest


def wrapper_class(*, include_init=False, factory=None):
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/lmcache_connector_v1.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Dynamic"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            *(["__init__"] if include_init else []),
            "handle_preemptions",
            "handle_preemptions_with_metadata",
            "supports_preemption_checkpoint",
        }
    ]
    if not cls.body:
        cls.body = [ast.Pass()]
    base = type(
        "KVBase",
        (),
        {
            "handle_preemptions": lambda self, reqs: None,
            "__init__": lambda self, **kwargs: None,
        },
    )
    ns = {"KVConnectorBase_V1": base, "SupportsHMA": type("HMA", (), {})}
    ns["LMCacheConnectorV1Impl"] = factory
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[cls.name]


def test_dynamic_connector_reaches_the_ascend_preemption_implementation():
    cls = wrapper_class()
    wrapper = cls()
    calls = []
    wrapper._lmcache_engine = NS(handle_preemptions=lambda ids: calls.append(ids))
    wrapper.handle_preemptions({"r"})
    assert calls == [{"r"}]


def test_checkpoint_binding_is_cleared_when_capture_fails():
    wrapper = wrapper_class()()
    calls = []
    wrapper.bind_connector_metadata = lambda metadata: calls.append(("bind", metadata))
    wrapper.clear_connector_metadata = lambda: calls.append("clear")

    def fail(ids):
        calls.append(("capture", ids))
        raise RuntimeError("capture failure")

    wrapper._lmcache_engine = NS(handle_preemptions=fail)
    with pytest.raises(RuntimeError, match="capture failure"):
        wrapper.handle_preemptions_with_metadata({"r"}, "checkpoint-control")
    assert calls == [("bind", "checkpoint-control"), ("capture", {"r"}), "clear"]


def test_late_ascend_patch_is_resolved_at_connector_construction(monkeypatch):
    import sys
    from types import ModuleType

    class OriginalImpl:
        def __init__(self, *args):
            pass

    class AscendImpl(OriginalImpl):
        pass

    # The dynamic wrapper may already be imported when Ascend installs its
    # implementation factory. Its previous global alias then remains stale.
    wrapper = wrapper_class(include_init=True, factory=OriginalImpl)
    module = ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
    module.LMCacheConnectorV1Impl = AscendImpl
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert isinstance(wrapper(None, None)._lmcache_engine, AscendImpl)
