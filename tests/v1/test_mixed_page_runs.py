# SPDX-License-Identifier: Apache-2.0
"""Ordinary and exact page retrieval across repeated LocalCPU/Mooncake runs."""

# Standard
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Local
from .test_prefill_source_work import SharedEngine, gc_disabled, shared_engines  # noqa: F401


@pytest.fixture(params=["base", "ascend", "ascend_perf"])
def resolver(request):
    if request.param == "base":
        return LMCacheEngine._resolve_shared_rank0_layer_pages
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend"
        / "lmcache_ascend/v1/cache_engine.py"
    )
    if not path.exists():
        pytest.skip("Ascend-derived resolver requires sibling LMCache-Ascend checkout")
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendLMCacheEngine"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_resolve_shared_rank0_layer_pages"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    namespace = dict(LMCacheEngine._resolve_shared_rank0_layer_pages.__globals__)
    namespace["serving_perf_enabled"] = lambda: request.param == "ascend_perf"
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize(
    "locations",
    [
        "LLLL",
        "RRRR",
        "LRL",
        "RLRL",
        "LRLRLR",
        "RLRLR",
        "RLRM",
        "RLRG",
        "RLGR",
        "RLFL",
        "RLEL",
    ],
)
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.usefixtures("shared_engines", "gc_disabled")
def test_page_runs_and_failure_cleanup(resolver, locations, group, exact):
    engine = SharedEngine(passive=False, layers=3 if group == 0 else 2)
    engine._resolve_shared_rank0_layer_pages = MethodType(resolver, engine)
    engine._num_layers_for_kv_group = engine.num_layers_for_group
    # Include a partial physical tail.
    engine.populate(len(locations) * 4 - 1, group)
    all_pages = dict(engine.backend.hot_cache)
    keys = list(all_pages)
    # M: missing; G: legacy boundary; F: read failure; E: allocation failure.
    remote = {
        key: engine.backend.hot_cache.pop(key)
        for key, location in zip(keys, locations, strict=True)
        if location != "L"
    }
    for key, location in zip(keys, locations, strict=True):
        if location in "MG":
            remote.pop(key)
    calls = []

    for name in (
        "batched_get_layer_page_prefix",
        "batched_get_prefixes_with_misses",
        "contains_all_exact",
    ):
        setattr(
            engine.backend,
            name,
            MethodType(getattr(LocalCPUBackend, name), engine.backend),
        )

    def contains(wanted):
        return next(
            (i for i, key in enumerate(wanted) if key not in remote), len(wanted)
        )

    def retrieve(wanted):
        if any(locations[keys.index(key)] == "E" for key in wanted):
            return []
        if any(locations[keys.index(key)] == "F" for key in wanted):
            raise LookupError("remote fetch failed after earlier runs")
        if any(key not in remote for key in wanted):
            raise LookupError("exact remote page unavailable")
        calls.extend(wanted)
        result = [remote[key] for key in wanted]
        for obj in result:
            obj.ref_count_up()
        return result

    def legacy_fetch(*args, **kwargs):
        raise AssertionError("Available merged pages were sent through legacy fetch")

    engine.storage_manager.storage_backends["RemoteBackend"] = SimpleNamespace(
        batched_contains_layer_pages=contains,
        batched_get_layer_pages=retrieve,
    )
    engine.storage_manager.batched_get = legacy_fetch
    if "G" in locations:
        legacy_keys = keys[locations.index("G")].split_layers(engine.num_layers)
        engine.backend.contains_all_exact = lambda wanted: wanted == legacy_keys

    if any(loc in locations for loc in "MG") and not exact:

        def legacy_tail(**kwargs):
            boundary = next(i for i, loc in enumerate(locations) if loc in "MG")
            assert kwargs["keys_layer_major"] == [
                [key.get_layer(layer) for key in keys[boundary:]]
                for layer in range(engine.num_layers)
            ]
            assert calls == [
                key
                for key, loc in zip(keys[:boundary], locations[:boundary], strict=True)
                if loc == "R"
            ]
            raise LookupError("legacy remainder reached without re-fetching pages")

        engine._resolve_shared_rank0_page_first_layers = legacy_tail

    rows = None
    error = any(loc in locations for loc in "MGFE")
    try:
        kwargs = dict(
            req_id="r",
            phase="dense_prefix",
            kv_group=group,
            keys_layer_major=[
                [key.get_layer(layer) for key in keys]
                for layer in range(engine.num_layers)
            ],
            page_chunks=len(keys),
            base_page_keys=keys,
            exact_chunk_locations=[
                "LocalCPUBackend" if loc == "L" else "RemoteBackend"
                for loc in locations
            ]
            if exact
            else None,
        )
        if error:
            if "E" in locations:
                with pytest.raises(ValueError, match="result count is inconsistent"):
                    engine._resolve_shared_rank0_layer_pages(**kwargs)
                return
            message = (
                "remote fetch failed after earlier runs"
                if "F" in locations
                else "exact remote page unavailable"
                if exact
                else "legacy remainder reached without re-fetching pages"
            )
            with pytest.raises(LookupError, match=message):
                engine._resolve_shared_rank0_layer_pages(**kwargs)
            return
        rows, count = engine._resolve_shared_rank0_layer_pages(**kwargs)
        assert count == len(keys)
        assert all(row == list(all_pages.values()) for row in rows)
        assert calls == list(remote)
    finally:
        if rows is not None:
            for obj in {id(obj): obj for row in rows for obj in row}.values():
                obj.unpin()
                obj.ref_count_down()
        for obj in all_pages.values():
            assert obj.get_ref_count() == 1 and obj.metadata.pin_count == 0
            obj.ref_count_down()
