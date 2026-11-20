# SPDX-License-Identifier: Apache-2.0
"""Verify storage construction and removal of retired retrieve statistics."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_store_has_no_discarded_diagnostic_aggregation() -> None:
    path = ROOT / "lmcache/v1/cache_engine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert not (
                isinstance(node.value.func, ast.Name) and node.value.func.id == "sum"
            )


def test_retrieve_statistics_have_no_remaining_state_or_calls() -> None:
    path = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert "retrieve_stats" not in node.name
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith("_retrieve_stats_")


@pytest.mark.parametrize("role", ["scheduler", "worker"])
@pytest.mark.parametrize("enable_pd", [False, True])
@pytest.mark.parametrize("local_cpu", [False, True])
@pytest.mark.parametrize("already_created", [False, True])
@pytest.mark.parametrize("capacity", [0, 1])
def test_local_cpu_backend_construction(
    role: str,
    enable_pd: bool,
    local_cpu: bool,
    already_created: bool,
    capacity: int,
) -> None:
    path = ROOT / "lmcache/v1/storage_backend/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allocation = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "metadata.role" in ast.unparse(node.test)
        and "LocalCPUBackend(" in ast.unparse(node)
    )
    created = []
    existing = object()

    def allocate(*args: object) -> str:
        created.append(args)
        return "LocalCPUBackend"

    namespace = dict(
        metadata=SimpleNamespace(role=role),
        config=SimpleNamespace(
            enable_pd=enable_pd, local_cpu=local_cpu, max_local_cpu_size=capacity
        ),
        _skip={"LocalCPUBackend"} if already_created else set(),
        local_cpu_backend=existing,
        LocalCPUBackend=allocate,
        storage_backends={},
        dst_device=None,
        lmcache_worker=None,
        logger=SimpleNamespace(info=lambda *args: None),
    )
    exec(
        compile(ast.Module(body=[allocation], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    expected = (
        role != "scheduler"
        and (not enable_pd or local_cpu)
        and not already_created
        and capacity > 0
    )
    assert bool(created) == expected
    if expected:
        assert namespace["storage_backends"] == {"LocalCPUBackend": "LocalCPUBackend"}
    else:
        assert namespace["local_cpu_backend"] is existing
