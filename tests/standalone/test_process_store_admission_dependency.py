# SPDX-License-Identifier: Apache-2.0
"""Record the remaining save FIFO dependency; this is not an isolation pass."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS
import ast

# First Party
from lmcache.integration.vllm.storage_process import reserve_store_step


def test_prefill_store_publication_still_waits_for_pending_admission():
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend"
        / "lmcache_ascend/v1/process_store.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_publish_completion"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(selected)
    namespace = {"replace": replace}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    @dataclass
    class Envelope:
        status: str
        error_details: dict

    admission, published, entered = Future(), Event(), Event()
    coordinator = NS(last_latent_future=admission)
    adapter = NS(_get_cold_load_coordinator=lambda: coordinator)
    kwargs = {"process_collective_predecessor": reserve_store_step(adapter)}
    engine = NS(
        _is_passive=lambda: False,
        _shared_layerwise_error_envelope=lambda **kw: Envelope("error", kw["details"]),
        _broadcast_shared_envelope=lambda envelope: published.set(),
        _validate_shared_layerwise_envelope=lambda *args, **kw: None,
    )
    result = NS(
        request_id="prefill-B", kv_group=0, process_store_plan={}, committed_end=0
    )

    def finish():
        entered.set()
        namespace["_publish_completion"](engine, result, [1, 2, 3], kwargs)

    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(finish)
        try:
            assert entered.wait(2)
            assert not published.wait(0.05)
            assert not future.done()
        finally:
            admission.set_result(None)
        future.result(2)
    assert published.is_set()
