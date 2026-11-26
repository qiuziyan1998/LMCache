# SPDX-License-Identifier: Apache-2.0
"""Actual diagnostic module on CPU; no device stream behavior is simulated."""

import gc
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from lmcache.v1.shared_cpu_cache import LocalLeaseHandle


@pytest.fixture
def diagnostics():
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/content_diagnostics.py"
    )
    name = "_process_source_diagnostics_cpu"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    module.configure_npu_content_diagnostics(False)
    sys.modules.pop(name, None)


def register(module, owner, request="req"):
    module.register_group0_source_probe(
        req_id=request,
        layer_id=0,
        num_layers=2,
        source_chunks=[torch.arange(8, dtype=torch.bfloat16)],
        selected_tokens=torch.tensor([0, 1]),
        selected_count=None,
        chunk_size=4,
        total_tokens=4,
        token_major=False,
        layer_cache=torch.zeros((1, 4, 1, 2)),
        source_owner=owner,
    )


@pytest.mark.parametrize(
    "cleanup", ["disabled", "interrupted", "replaced", "partial", "dropped"]
)
def test_diagnostic_cpu_owner_is_retained_and_explicitly_released_with_gc_off(
    diagnostics, cleanup
):
    retired = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        diagnostics.configure_npu_content_diagnostics(True)
        owner = LocalLeaseHandle(
            lease_id="lease",
            mapping=object(),
            retire=lambda *args: retired.append(args),
        )
        register(diagnostics, owner)
        owner.release("fenced")
        assert not retired
        if cleanup == "disabled":
            diagnostics.configure_npu_content_diagnostics(False)
        elif cleanup == "interrupted":
            diagnostics.begin_deferred_diagnostic_step()
        elif cleanup == "replaced":
            register(diagnostics, None)
        elif cleanup == "partial":
            assert diagnostics._take_group0_source_probes(["req", "missing"], 0) is None
        else:
            probes = diagnostics._take_group0_source_probes(["req"], 0)
            diagnostics.MAX_PENDING_SNAPSHOTS = 1
            diagnostics._queue_snapshot(
                diagnostics._DeferredSnapshot("first", {}, group0_source_probes=probes)
            )
            diagnostics._queue_snapshot(diagnostics._DeferredSnapshot("second", {}))
        assert retired == [("lease", "never_imported")]
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("failure", [False, True])
def test_deferred_probe_flush_releases_after_read_or_failure(
    diagnostics, monkeypatch, failure
):
    diagnostics.configure_npu_content_diagnostics(True)
    retired = []
    owner = LocalLeaseHandle(
        lease_id="lease", mapping=object(), retire=lambda *args: retired.append(args)
    )
    register(diagnostics, owner)
    probes = diagnostics._take_group0_source_probes(["req"], 0)
    diagnostics._queue_snapshot(
        diagnostics._DeferredSnapshot(
            "staged_sfa_graph_fingerprint",
            {},
            components={"plane0": torch.zeros(2, 2)},
            group0_source_probes=probes,
        )
    )
    owner.release("fenced")
    original = diagnostics._materialize_group0_source_samples

    def read(source):
        assert not retired
        if failure:
            raise RuntimeError("diagnostic failure")
        return original(source)

    monkeypatch.setattr(diagnostics, "_materialize_group0_source_samples", read)
    diagnostics.flush_deferred_diagnostics()
    assert retired == [("lease", "never_imported")]


def test_disabled_probe_does_not_retain_or_inspect_source(diagnostics):
    class Source:
        def retain(self):
            pytest.fail("disabled diagnostics must not acquire a source")

    diagnostics.configure_npu_content_diagnostics(False)
    register(diagnostics, Source())
