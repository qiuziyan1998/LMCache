# SPDX-License-Identifier: Apache-2.0
"""CPU configuration and actual external-page presence contract tests."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from lmcache.v1.config import LMCacheEngineConfig

ROOT = Path(__file__).resolve().parents[2]


def test_writeback_is_disabled_by_default():
    config = LMCacheEngineConfig.from_defaults()
    assert config.decode_checkpoint_writeback is False
    assert config.decode_checkpoint_retention_max_bytes == 0
    config.validate()


@pytest.mark.parametrize(
    "change",
    [
        {"decode_preemption_checkpoint": False},
        {"pd_role": "sender"},
        {"decode_checkpoint_retention_max_bytes": 0},
        {"decode_checkpoint_free_reserve_bytes": -1},
    ],
)
def test_writeback_rejects_missing_prerequisites(change):
    values = dict(
        decode_checkpoint_writeback=True,
        decode_preemption_checkpoint=True,
        pd_role="receiver",
        decode_checkpoint_retention_max_bytes=100,
        decode_checkpoint_free_reserve_bytes=10,
    )
    values.update(change)
    with pytest.raises(ValueError, match="decode_checkpoint_writeback"):
        LMCacheEngineConfig.from_defaults(**values).validate()


def test_positive_writeback_budgets_validate():
    LMCacheEngineConfig.from_defaults(
        decode_checkpoint_writeback=True,
        decode_preemption_checkpoint=True,
        pd_role="receiver",
        decode_checkpoint_retention_max_bytes=100,
        decode_checkpoint_free_reserve_bytes=10,
    ).validate()


def method(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, fn], type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("statuses", [[1, 0], [-1, 1], None, [1]])
def test_strict_mooncake_probe_rejects_errors_not_just_missing(statuses):
    check = method(
        "lmcache/v1/storage_backend/connector/mooncakestore_connector.py",
        "batched_external_pages_exist",
        dict(
            mooncake_page_key=lambda key, layers: key,
            trace_mooncake_keys=lambda *args, **kwargs: None,
        ),
    )
    connector = NS(
        _page_num_layers_for=lambda key: 2,
        store=NS(batch_is_exist=lambda keys: statuses),
    )
    if statuses == [1, 0]:
        assert check(connector, ["a", "b"], strict=True) == [True, False]
    else:
        with pytest.raises(RuntimeError):
            check(connector, ["a", "b"], strict=True)
    if statuses == [-1, 1]:
        assert check(connector, ["a", "b"]) == [False, True]


def test_strict_remote_probe_rejects_disconnected_store():
    check = method(
        "lmcache/v1/storage_backend/remote_backend.py",
        "batched_external_pages_exist",
        {},
    )
    backend = NS(connection=None)
    assert check(backend, ["key"]) == [False]
    with pytest.raises(RuntimeError, match="unavailable"):
        check(backend, ["key"], strict=True)


def test_strict_probe_reaches_wrapped_mooncake_connector():
    remote = method(
        "lmcache/v1/storage_backend/remote_backend.py",
        "batched_external_pages_exist",
        {},
    )
    wrapper = method(
        "lmcache/v1/storage_backend/connector/instrumented_connector.py",
        "batched_external_pages_exist",
        {},
    )
    calls = []
    connector = NS(
        batched_external_pages_exist=lambda keys, **kw: (
            calls.append((keys, kw)),
            [True],
        )[1]
    )
    instrumented = NS(_connector=connector)
    instrumented.batched_external_pages_exist = lambda *a, **kw: wrapper(
        instrumented, *a, **kw
    )
    backend = NS(connection=instrumented, _mla_worker_id_as0_mode=False)
    assert remote(backend, ["key"], strict=True) == [True]
    assert calls == [(["key"], {"strict": True})]
