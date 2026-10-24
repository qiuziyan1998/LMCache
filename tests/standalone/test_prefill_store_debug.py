# SPDX-License-Identifier: Apache-2.0
"""Exercise scalar diagnostics without importing the device runtime."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

# Third Party
import pytest


@pytest.fixture
def debug(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/prefill_store_debug.py"
    spec = spec_from_file_location("test_prefill_debug_runtime", path)
    module = module_from_spec(spec)
    monkeypatch.setenv("LMCACHE_PREFILL_STORE_DEBUG", "1")
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    records = []
    monkeypatch.setattr(module, "_emit", lambda tag, text: records.append((tag, text)))
    return module, records


def test_counts_every_submission_but_does_not_flood_repeated_chunks(debug):
    module, records = debug
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(
            workers.map(
                lambda _: module.prefill_store_debug_log("flat_put", "request", 0),
                range(124),
            )
        )
    assert [text for _, text in records] == ["req=request", "g0 flat_put"]
    module.prefill_store_debug_finish("request")
    assert records[-2:] == [(1, "g0 total flat_put=124"), (1, "persist_done")]


def test_late_errors_are_not_suppressed_by_earlier_successes(debug):
    module, records = debug
    for _ in range(3):
        module.prefill_store_debug_log("flat_done", "req", 1, status="ok")
    for _ in range(2):
        module.prefill_store_debug_log(
            "flat_done", "req", 1, error=True, status="TimeoutError"
        )
    assert sum("ERROR" in text for _, text in records) == 2
    module.prefill_store_debug_finish("req")
    assert (1, "g1 total flat_done=5") in records


def test_later_requests_and_groups_get_their_own_records(debug, monkeypatch):
    module, records = debug
    monkeypatch.setattr(module, "_MAX_REQUESTS", 2)
    for req in ("early", "middle", "late", "early"):
        module.prefill_store_debug_log("page_single_none", req, 0)
    assert sum(text == "req=early" for _, text in records) == 2
    module.prefill_store_debug_log("page_single_none", "early", 1)
    assert records[-1][1] == "g1 page_single_none"


def test_disabled_logging_never_formats_values_or_retains_requests(debug, monkeypatch):
    module, records = debug

    class DeviceValue:
        def __str__(self):
            pytest.fail("a device value must not be formatted")

    module.prefill_store_debug_log("test", "enabled", value=DeviceValue())
    monkeypatch.setattr(module, "_ENABLED", False)
    module.prefill_store_debug_log("test", "disabled", value=DeviceValue())
    module.prefill_store_debug_finish("disabled")
    assert [text for _, text in records] == ["req=enabled", "g-1 test"]


def test_allocator_snapshot_uses_public_capacity_only_and_tolerates_unsupported(debug):
    module, _ = debug
    allocator = SimpleNamespace(get_allocator_capacity_bytes=lambda: (2**20, 4 * 2**20))
    assert module.prefill_store_allocator_snapshot(allocator) == {
        "free_mb": 1,
        "pool_mb": 4,
    }
    assert module.prefill_store_allocator_snapshot(object()) == {
        "free_mb": -1,
        "pool_mb": -1,
    }


def test_source_tier_changes_are_not_hidden_by_first_observation(debug):
    module, records = debug
    for local, remote in ((4, 0), (4, 0), (0, 4), (1, 3)):
        module.prefill_store_debug_sources(
            "request",
            0,
            local_pages=local,
            remote_pages=remote,
            legacy_chunks=0,
            num_layers=2,
            first_page_bytes=16,
        )
    lines = [text for _, text in records]
    assert sum("source_local " in line for line in lines) == 1
    assert any("source_remote local=0 remote=4" in line for line in lines)
    assert any("source_mixed local=1 remote=3" in line for line in lines)
