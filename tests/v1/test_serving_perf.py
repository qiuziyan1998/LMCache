# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import json
import runpy

# Third Party
import pytest

# First Party
from lmcache.v1.serving_perf import (
    SERVING_PERF_ENV,
    serving_perf_log,
    serving_perf_now,
    serving_perf_scope,
)


@pytest.mark.parametrize(
    "mode,enabled,detailed",
    [
        ("", False, False),
        ("0", False, False),
        (" FALSE ", False, False),
        ("no", False, False),
        ("off", False, False),
        ("1", True, False),
        (" Detail ", True, True),
        ("DEVICE", True, True),
    ],
)
def test_mode_is_resolved_once_before_serving(monkeypatch, mode, enabled, detailed):
    monkeypatch.setenv(SERVING_PERF_ENV, mode)
    path = Path(__file__).parents[2] / "lmcache/v1/serving_perf.py"
    namespace = runpy.run_path(str(path))
    # Changing the environment after initialization must not parse it per call.
    monkeypatch.setenv(SERVING_PERF_ENV, "0" if enabled else "1")
    assert namespace["serving_perf_enabled"]() is enabled
    assert namespace["serving_perf_detailed_enabled"]() is detailed


def test_perf_serialization_does_not_stringify_caller_objects(monkeypatch):
    class NoReadback:
        def __str__(self):
            raise AssertionError("diagnostics must not stringify device data")

        __repr__ = __str__

    monkeypatch.setenv(SERVING_PERF_ENV, "1")
    monkeypatch.setattr("lmcache.v1.serving_perf._MODE", "1")
    logger = _Logger()
    serving_perf_log(logger, "safe", values=[NoReadback()])
    payload = json.loads(logger.records[0][1])
    assert payload["values"] == ["<non-JSON value>"]


class _Logger:
    def __init__(self):
        self.records = []

    def info(self, message, payload):
        self.records.append((message, payload))


def test_serving_perf_disabled(monkeypatch):
    monkeypatch.delenv(SERVING_PERF_ENV, raising=False)
    monkeypatch.setattr("lmcache.v1.serving_perf._MODE", "0")
    logger = _Logger()

    serving_perf_log(logger, "ignored")

    assert logger.records == []


def test_serving_perf_structured_log(monkeypatch):
    monkeypatch.setenv(SERVING_PERF_ENV, "1")
    monkeypatch.setattr("lmcache.v1.serving_perf._MODE", "1")
    logger = _Logger()

    serving_perf_log(
        logger,
        "stage",
        started=serving_perf_now(),
        req_id="request-1",
    )

    message, raw = logger.records[0]
    payload = json.loads(raw)
    assert message == "[LMCACHE_COLD_PERF] %s"
    assert payload["event"] == "stage"
    assert payload["req_id"] == "request-1"
    assert payload["schema"] == 1
    assert payload["elapsed_ms"] >= 0
    assert payload["wall_time_ns"] > 0
    assert payload["host"]
    assert payload["clock_domain"].startswith(f"{payload['host']}:")


def test_serving_perf_scope_correlates_and_restores(monkeypatch):
    monkeypatch.setenv(SERVING_PERF_ENV, "1")
    monkeypatch.setattr("lmcache.v1.serving_perf._MODE", "1")
    logger = _Logger()

    with serving_perf_scope(req_id="request-1", rank=3, resolver_call_id="c1"):
        serving_perf_log(logger, "nested", rank=0)
    serving_perf_log(logger, "outside")

    nested, outside = (json.loads(record[1]) for record in logger.records)
    assert (nested["req_id"], nested["rank"], nested["resolver_call_id"]) == (
        "request-1",
        0,
        "c1",
    )
    assert "req_id" not in outside


def test_serving_perf_preserves_aggregate_page_diagnostics(monkeypatch):
    monkeypatch.setenv(SERVING_PERF_ENV, "1")
    monkeypatch.setattr("lmcache.v1.serving_perf._MODE", "1")
    logger = _Logger()

    with serving_perf_scope(req_id="request-1", rank=2):
        serving_perf_log(
            logger,
            "passive_layer_prepare",
            physical_pages=515,
            logical_entries=18540,
            legacy_tail_objects=36,
            pointer_seal_ms=12.5,
            passive_wait_ms=None,
            passive_wait_status="unavailable",
        )

    payload = json.loads(logger.records[0][1])
    assert (payload["req_id"], payload["rank"]) == ("request-1", 2)
    assert payload["physical_pages"] == 515
    assert payload["logical_entries"] == 18540
    assert payload["legacy_tail_objects"] == 36
    assert payload["pointer_seal_ms"] == 12.5
    assert payload["passive_wait_ms"] is None
    assert payload["passive_wait_status"] == "unavailable"
