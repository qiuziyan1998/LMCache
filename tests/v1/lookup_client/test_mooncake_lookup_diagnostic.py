# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Third Party
import pytest


def _load_diagnostic_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "diagnose_mooncake_lookup.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnose_mooncake_lookup",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StatusStore:
    def __init__(self, statuses: list[int]):
        self.statuses = statuses

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return self.statuses


class _FailingStore:
    def batch_is_exist(self, keys: list[str]) -> list[int]:
        raise RuntimeError("native lookup failed")


def test_diagnostic_preserves_miss_and_error_statuses() -> None:
    diagnostic = _load_diagnostic_module()
    store = diagnostic.StatusTracingStore(_StatusStore([1, 0, -1]))

    assert store.batch_is_exist(["found", "missing", "error"]) == [1, 0, -1]
    summary = diagnostic.summarize_trace(store.calls)

    assert summary["status_counts"] == {
        "found": 1,
        "missing": 1,
        "error": 1,
        "other": 0,
    }
    assert summary["classification"] == "lookup_error"
    assert summary["first_failure"] == {
        "call": 0,
        "key_index": 1,
        "key": "missing",
        "status": 0,
    }


def test_diagnostic_preserves_native_lookup_exception() -> None:
    diagnostic = _load_diagnostic_module()
    store = diagnostic.StatusTracingStore(_FailingStore())

    with pytest.raises(RuntimeError, match="native lookup failed"):
        store.batch_is_exist(["key"])

    summary = diagnostic.summarize_trace(store.calls)
    assert summary["exceptions"] == 1
    assert summary["classification"] == "lookup_error"
    assert summary["first_failure"] == {
        "call": 0,
        "error": "RuntimeError: native lookup failed",
        "key": "key",
    }


def test_diagnostic_classifies_zero_as_cache_miss() -> None:
    diagnostic = _load_diagnostic_module()
    store = diagnostic.StatusTracingStore(_StatusStore([1, 0]))

    store.batch_is_exist(["found", "missing"])
    summary = diagnostic.summarize_trace(store.calls)

    assert summary["classification"] == "cache_miss"
    assert summary["status_counts"]["missing"] == 1
    assert summary["status_counts"]["error"] == 0


def test_diagnostic_marks_short_status_response() -> None:
    diagnostic = _load_diagnostic_module()
    store = diagnostic.StatusTracingStore(_StatusStore([1]))

    store.batch_is_exist(["found", "omitted"])
    summary = diagnostic.summarize_trace(store.calls)

    assert summary["short_responses"] == 1
    assert summary["classification"] == "lookup_error"
    assert summary["status_counts"]["other"] == 1
    assert summary["first_failure"] == {
        "call": 0,
        "key_index": 1,
        "key": "omitted",
        "status": None,
    }


@pytest.mark.parametrize(
    ("kv_group", "expected_format"),
    (
        (0, "KV_MLA_LATENT_FMT"),
        (1, "KV_DSA_INDEX_FMT"),
    ),
)
def test_roundtrip_payload_uses_supported_layerwise_format(
    kv_group: int,
    expected_format: str,
) -> None:
    diagnostic = _load_diagnostic_module()
    key = type("_Key", (), {"kv_group": kv_group})()

    shape, dtype, fmt = diagnostic._payload_layout(key, 7)

    assert shape == diagnostic.torch.Size([1, 1, 7])
    assert dtype == diagnostic.torch.uint8
    assert fmt is getattr(diagnostic.MemoryFormat, expected_format)
    assert fmt is not diagnostic.MemoryFormat.BINARY
