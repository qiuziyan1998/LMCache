# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json

# Third Party
import pytest

# First Party
from lmcache.v1.mooncake_key_trace import (
    MooncakeKeyTracingStore,
    maybe_trace_mooncake_store,
    mooncake_trace_context,
    run_mooncake_zero_lookup_retries,
)


class _Store:
    def batch_put_from(self, keys: list[str], *args: object) -> list[int]:
        return [0 for _ in keys]

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1, 0, -1][: len(keys)]

    def close(self) -> None:
        return None


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_is_disabled_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LMCACHE_MOONCAKE_KEY_TRACE_DIR", raising=False)
    store = _Store()

    assert maybe_trace_mooncake_store(store, "test") is store


def test_trace_records_exact_store_and_lookup_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_MOONCAKE_KEY_TRACE_DIR", str(tmp_path))
    traced = maybe_trace_mooncake_store(
        _Store(),
        "test",
        SimpleNamespace(role="worker", worker_id=3),
    )

    assert isinstance(traced, MooncakeKeyTracingStore)
    with mooncake_trace_context(
        "store-request",
        "store",
        kv_group=1,
        layer_id=7,
    ):
        assert traced.batch_put_from(["stored-a", "stored-b"]) == [0, 0]
    assert traced.batch_is_exist(["found", "missing", "error"]) == [1, 0, -1]
    traced.close()

    store_records = _records(next(tmp_path.glob("mooncake-store-*.jsonl")))
    lookup_records = _records(next(tmp_path.glob("mooncake-lookup-*.jsonl")))

    assert len(store_records) == 1
    assert store_records[0]["keys"] == [
        "stored-a",
        "stored-b",
    ]
    assert store_records[0]["outcomes"] == [
        "stored",
        "stored",
    ]
    assert store_records[0]["request_id"] == "store-request"
    assert store_records[0]["kv_group"] == 1
    assert store_records[0]["layer_id"] == 7
    assert len(lookup_records) == 1
    assert lookup_records[0]["keys"] == [
        "found",
        "missing",
        "error",
    ]
    assert lookup_records[0]["statuses"] == [1, 0, -1]
    assert lookup_records[0]["outcomes"] == [
        "found",
        "missing",
        "error",
    ]
    assert all(
        record["completed_at_ns"] >= record["started_at_ns"]
        for record in lookup_records
    )
    assert all(record["worker_id"] == 3 for record in lookup_records)


def test_trace_preserves_and_records_native_exception(tmp_path: Path) -> None:
    class _FailingStore(_Store):
        def batch_is_exist(self, keys: list[str]) -> list[int]:
            raise RuntimeError("lookup failed")

    traced = MooncakeKeyTracingStore(_FailingStore(), tmp_path, "test")

    with pytest.raises(RuntimeError, match="lookup failed"):
        traced.batch_is_exist(["key"])
    traced.close()

    record = _records(next(tmp_path.glob("mooncake-lookup-*.jsonl")))[0]
    assert record["keys"] == ["key"]
    assert record["outcomes"] == ["error"]
    assert record["error"] == "RuntimeError: lookup failed"


def test_zero_lookup_retry_traces_request_and_each_attempt(tmp_path: Path) -> None:
    class _DelayedStore(_Store):
        def __init__(self) -> None:
            self.calls = 0

        def batch_is_exist(self, keys: list[str]) -> list[int]:
            self.calls += 1
            return [1] if self.calls == 3 else [0]

    store = _DelayedStore()
    traced = MooncakeKeyTracingStore(store, tmp_path, "test")

    def lookup_once() -> int:
        return 256 if traced.batch_is_exist(["key"]) == [1] else 0

    result = run_mooncake_zero_lookup_retries(
        SimpleNamespace(mooncake_lookup_retry_delays_ms=[0, 0]),
        "lookup-request",
        lookup_once,
    )
    traced.close()

    assert result == 256
    assert store.calls == 3
    records = _records(next(tmp_path.glob("mooncake-lookup-*.jsonl")))
    assert [record["request_id"] for record in records] == [
        "lookup-request",
        "lookup-request",
        "lookup-request",
    ]
    assert [record["lookup_attempt"] for record in records] == [0, 1, 2]
    assert [record["retry_delay_ms"] for record in records] == [0, 0, 0]
    assert [record["outcomes"] for record in records] == [
        ["missing"],
        ["missing"],
        ["found"],
    ]


def test_empty_retry_delays_disable_zero_lookup_retry() -> None:
    calls = 0

    def lookup_once() -> int:
        nonlocal calls
        calls += 1
        return 0

    assert (
        run_mooncake_zero_lookup_retries(
            SimpleNamespace(mooncake_lookup_retry_delays_ms=[]),
            "lookup-request",
            lookup_once,
        )
        == 0
    )
    assert calls == 1
