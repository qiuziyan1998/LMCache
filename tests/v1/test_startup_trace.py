# SPDX-License-Identifier: Apache-2.0
"""Startup markers are flushed, rank-local and diagnostically best effort."""

# Standard
from types import SimpleNamespace
import io

# Third Party
import pytest

# First Party
from lmcache.v1 import startup_trace


class FlushedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def test_startup_phase_flushes_each_rank_before_and_after_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = FlushedOutput()
    ticks = iter((1.0, 1.025, 2.0, 2.050))
    monkeypatch.setattr(startup_trace, "sys", SimpleNamespace(stderr=output))
    monkeypatch.setattr(startup_trace.socket, "gethostname", lambda: "node-a.domain")
    monkeypatch.setattr(startup_trace.os, "getpid", lambda: 42)
    monkeypatch.setattr(startup_trace.time, "perf_counter", lambda: next(ticks))
    for rank in (0, 1):
        with startup_trace.startup_phase("mooncake_setup", rank=rank):
            assert output.flushes == rank * 2 + 1
            assert output.getvalue().splitlines()[-1].endswith("state=begin")
    lines = output.getvalue().splitlines()
    assert len(lines) == output.flushes == 4
    assert all("host=node-a.domain pid=42" in line for line in lines)
    assert all(len(line) <= 110 for line in lines)
    assert "rank=0" in lines[0] and "rank=1" in lines[2]
    assert lines[1].endswith("state=end ms=25.0")
    assert lines[3].endswith("state=end ms=50.0")


def test_startup_phase_preserves_error_and_omits_sensitive_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = FlushedOutput()
    monkeypatch.setattr(startup_trace, "sys", SimpleNamespace(stderr=output))
    original = RuntimeError("mooncakestore://user:secret@host")
    with pytest.raises(RuntimeError) as caught:
        with startup_trace.startup_phase("mooncake_setup", rank=3):
            raise original
    assert caught.value is original
    lines = output.getvalue().splitlines()
    assert len(lines) == output.flushes == 2
    assert "state=error" in lines[1] and "error=RuntimeError" in lines[1]
    assert "secret" not in output.getvalue()


@pytest.mark.parametrize("operation_fails", [False, True])
@pytest.mark.parametrize("failure", ["stderr", "format", "hostname"])
def test_startup_logging_failure_does_not_change_operation(
    monkeypatch: pytest.MonkeyPatch, operation_fails: bool, failure: str
) -> None:
    class BrokenOutput:
        def write(self, text: str) -> None:
            raise BrokenPipeError("closed logging pipe")

    class BrokenDetail:
        def __str__(self) -> str:
            raise ValueError("cannot format detail")

    def broken_hostname() -> str:
        raise OSError("no hostname")

    details: dict[str, object] = {"rank": 2}
    if failure == "stderr":
        monkeypatch.setattr(
            startup_trace, "sys", SimpleNamespace(stderr=BrokenOutput())
        )
    elif failure == "hostname":
        monkeypatch.setattr(startup_trace.socket, "gethostname", broken_hostname)
    else:
        details["bytes"] = BrokenDetail()
    original = RuntimeError("native startup failed")
    calls = []

    def operation() -> None:
        with startup_trace.startup_phase("mooncake_setup", **details):
            calls.append("called")
            if operation_fails:
                raise original

    if operation_fails:
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is original
    else:
        operation()
    assert calls == ["called"]
