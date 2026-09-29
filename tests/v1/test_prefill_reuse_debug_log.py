# SPDX-License-Identifier: Apache-2.0
"""Compact diagnostics remain bounded and never inspect device objects."""

# Standard
from itertools import count
from pathlib import Path
import logging

# Third Party
import pytest

# First Party
from lmcache.v1 import serving_perf as perf


@pytest.fixture
def records(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    output: list[str] = []

    class Recorder:
        def info(self, fmt: str, *args: object) -> None:
            output.append(fmt % args)

    monkeypatch.setattr(perf, "_PREFILL_REUSE_RANK", 1)
    monkeypatch.setattr(perf, "_PREFILL_REUSE_LINES", count())
    monkeypatch.setattr(perf, "_prefill_reuse_logger", Recorder)
    return output


def test_only_selected_worker_emits_short_correlated_records(
    records: list[str],
) -> None:
    for rank in range(8):
        for group in (0, 1):
            perf.prefill_reuse_debug_log(
                rank,
                "src",
                req_id="request-with-a-long-id",
                p=81920,
                g=group,
                v="76/4",
                t="0/0",
                x="76/4",
                c=True,
                ms=12.345,
            )
    assert len(records) == 2
    assert all(len(line) <= 96 and line.startswith("[PFR] r1 ") for line in records)
    assert records[0].split()[2:4] == records[1].split()[2:4]
    assert "g0 src" in records[0] and "g1 src" in records[1]
    assert records[0].endswith("c=1 ms=12.3")
    assert "request-with-a-long-id" not in records[0]


def test_line_budget_reports_truncation_once(records: list[str]) -> None:
    for _ in range(500):
        perf.prefill_reuse_debug_log(1, "state", p=81920, a="keep")
    assert len(records) == 361
    assert records[-1] == "[PFR] r1 limit=360"


def test_no_device_object_stringification_and_one_physical_line(
    records: list[str],
) -> None:
    class DeviceObject:
        def __str__(self) -> str:
            raise AssertionError("device read")

    perf.prefill_reuse_debug_log(1, "test", value=DeviceObject(), label="a\nb\rc")
    assert "value=? label=a_b_c" in records[0]
    perf.prefill_reuse_debug_log(1, "test", label="a" * 200)
    assert len(records[1]) == 96 and records[1].endswith("...")


def test_disabled_diagnostics_do_not_construct_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perf, "_PREFILL_REUSE_RANK", -1)
    monkeypatch.setattr(perf, "_prefill_reuse_logger", lambda: pytest.fail("logger"))
    perf.prefill_reuse_debug_log(1, "ignored")


def test_info_output_has_no_framework_timestamp(capsys: pytest.CaptureFixture) -> None:
    perf._prefill_reuse_logger.cache_clear()
    logger = perf._prefill_reuse_logger()
    try:
        assert logger.level == logging.INFO and not logger.propagate
        logger.info("[PFR] r1 short")
        assert capsys.readouterr().out == "[PFR] r1 short\n"
    finally:
        logger.handlers.clear()
        perf._prefill_reuse_logger.cache_clear()


def test_selected_worker_writes_short_file_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reuse.log"
    monkeypatch.setattr(perf, "_PREFILL_REUSE_RANK", 1)
    monkeypatch.setattr(perf, "_PREFILL_REUSE_LINES", count())
    monkeypatch.setattr(perf, "_PREFILL_REUSE_FILE", str(destination))
    perf._prefill_reuse_logger.cache_clear()
    try:
        perf.prefill_reuse_debug_log(0, "ignored")
        assert not destination.exists()
        perf.prefill_reuse_debug_log(1, "src", req_id="r", p=4096, x="0/4")
        first = destination.read_text(encoding="utf-8").splitlines()
        assert len(first) == 1 and first[0].endswith("src x=0/4")
        perf.prefill_reuse_debug_log(1, "src", req_id="r", p=8192, x="4/4")
        assert len(destination.read_text(encoding="utf-8").splitlines()) == 2
    finally:
        logger = perf._prefill_reuse_logger()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        perf._prefill_reuse_logger.cache_clear()
