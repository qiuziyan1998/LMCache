# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest

from lmcache.v1 import storage_service as service


@pytest.mark.parametrize(
    "method, duration, emitted",
    [
        ("prepare", 0.01, True),
        ("native_get", 0.01, False),
        ("native_get", 0.2, True),
    ],
)
def test_service_timing_is_bounded_and_preserves_result(
    monkeypatch, method, duration, emitted
):
    times = iter((1.0, 1.0 + duration))
    monkeypatch.setattr(service.time, "perf_counter", lambda: next(times))
    logs = []
    monkeypatch.setattr(
        service, "serving_perf_log", lambda *args, **kw: logs.append(kw)
    )
    result = NS(status="COMPLETED")
    owner = NS(_handle=lambda request: result)
    request = NS(
        method=method,
        payload={},
        identity=NS(
            request_id="r", operation_id=3, service_epoch="e", client_session="w"
        ),
    )
    assert service.StorageService._handle_timed(owner, request, 0.99) is result
    assert bool(logs) == emitted
    if emitted:
        assert logs[0]["queue_ms"] == 10
        assert logs[0]["execute_ms"] == round(duration * 1000, 3)


def test_disabled_service_timing_does_not_read_clock(monkeypatch):
    def unexpected():
        raise AssertionError("disabled instrumentation read a clock")

    monkeypatch.setattr(service.time, "perf_counter", unexpected)
    result = object()
    owner = NS(_handle=lambda request: result)
    assert service.StorageService._handle_timed(owner, None, None) is result


def test_phase_collection_rolls_up_and_resets_after_error(monkeypatch):
    from lmcache.v1 import serving_perf as perf

    times = iter((1.0, 1.025))
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(times))
    with perf.serving_perf_measure() as outer:
        with pytest.raises(ValueError):
            with perf.serving_perf_measure(rollup_prefix="remote_get_") as inner:
                with perf.serving_perf_phase("native_call_ms"):
                    raise ValueError("native failure")
        assert inner == {"native_call_ms": 25.0}
        assert outer == {"remote_get_native_call_ms": 25.0}
    perf.serving_perf_record("after", 1)
    assert outer == {"remote_get_native_call_ms": 25.0}


def test_fresh_service_process_emits_perf_without_root_logging_setup():
    import os
    import subprocess
    import sys

    code = """
from types import SimpleNamespace as NS
from lmcache.v1.storage_service import StorageService
import time
request = NS(method="prepare", payload={}, identity=NS(
    request_id="r", operation_id=1, service_epoch="e", client_session="w"))
StorageService._handle_timed(NS(_handle=lambda _: NS(status="COMPLETED")),
                            request, time.perf_counter())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "PD_SERVING_PERF": "detail",
            "LMCACHE_LOG_LEVEL": "INFO",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"event":"storage_service_timing"' in result.stderr
