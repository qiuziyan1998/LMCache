# SPDX-License-Identifier: Apache-2.0
"""Preparation reuses each chunk's complete layer-key expansion."""

# Standard
from typing import Any

# First Party
from lmcache.utils import CacheEngineKey

# Local
import test_cpu_cache_runtime as cpu_runtime
from test_cpu_cache_runtime import control, lookup, prepare

setup = cpu_runtime.setup


def test_prepare_expands_each_chunk_once_per_group(
    setup: Any, monkeypatch: Any
) -> None:
    owner, runtime, work, bootstrap, populate = setup
    bootstrap()
    populate()
    hit, _ = lookup(work)
    original = CacheEngineKey.split_layers
    calls = []

    def split(key: CacheEngineKey, count: int) -> list:
        calls.append((key.kv_group, key.chunk_hash, count))
        return original(key, count)

    monkeypatch.setattr(CacheEngineKey, "split_layers", split)
    prepared, request = prepare(work, runtime, hit.value)
    assert prepared.status == "COMPLETED"
    for group in prepared.value.groups:
        for participant in ("worker-0", "worker-1"):
            control(
                runtime,
                request,
                "retire",
                {"lease_id": group.lease_id, "proof": "never_imported"},
                participant=participant,
            )
    assert calls == [(0, 7, 2), (0, 8, 2), (1, 7, 3), (1, 8, 3)]
