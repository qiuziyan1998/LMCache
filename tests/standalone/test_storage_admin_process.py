# SPDX-License-Identifier: Apache-2.0
"""Actual M management methods and deployed API handlers with CPU-only transport."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from prometheus_client.parser import text_string_to_metric_families

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.internal_api_server.vllm import backend_api, bypass_api, conf_api
from lmcache.v1.internal_api_server.vllm import (
    freeze_api,
    hot_cache_api,
    load_fs_chunks_api,
)
from lmcache.v1.internal_api_server.common import metrics_api
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.rpc.storage_admin import (
    StorageAdminClient,
    get_engine_storage_admin,
    merge_storage_metrics,
)

from test_process_checkpoint_worker import worker as checkpoint_worker
from test_ascend_cpu_storage_runtime import setup as cpu_setup

setup = cpu_setup
worker = checkpoint_worker


def test_remote_admin_preserves_single_cpu_owner_and_error_types(worker):
    checkpoint, runtime, client, _ = worker
    engine = checkpoint.engine
    engine.storage_manager = None
    admin = get_engine_storage_admin(engine)
    assert admin is get_engine_storage_admin(engine)
    assert isinstance(admin, StorageAdminClient)
    owner = runtime.storage_manager.local_cpu_backend
    assert "LocalCPUBackend" in admin.list_backends()
    with pytest.raises(RuntimeError, match="restart"):
        admin.close_backend("LocalCPUBackend")
    with pytest.raises(RuntimeError, match="restart"):
        admin.recreate_backend("LocalCPUBackend")
    with pytest.raises(KeyError):
        admin.recreate_backend("NoSuchBackend")
    assert not admin.close_backend("NoSuchBackend")
    assert admin.create_backends() == {}
    assert runtime.storage_manager.local_cpu_backend is owner
    assert engine.storage_manager is None
    assert not admin._pending
    assert [m for m, p in client.controls].count("admin") == 6


def test_explicit_freeze_hot_and_bypass_have_no_cached_poll_rpc(worker):
    checkpoint, runtime, client, _ = worker
    admin = get_engine_storage_admin(checkpoint.engine)
    admin.set_freeze(True)
    admin.set_hot_cache(False)
    assert runtime.storage_manager.is_frozen()
    assert not runtime.storage_manager.is_hot_cache_enabled()
    count = len(client.controls)
    for _ in range(1000):
        assert admin.cached_frozen
        assert not admin.cached_hot_cache
    assert len(client.controls) == count
    assert admin.is_frozen()
    assert not admin.is_hot_cache_enabled()
    admin.set_backend_bypass("LocalCPUBackend", True)
    assert admin.is_backend_bypassed("LocalCPUBackend")
    assert admin.get_bypassed_backends() == ["LocalCPUBackend"]
    assert "LocalCPUBackend" in admin.get_all_backend_names()
    admin.set_backend_bypass("LocalCPUBackend", False)
    assert not admin.is_backend_bypassed("LocalCPUBackend")


def test_actual_engine_freeze_hot_methods_use_m_and_cached_step_reads(worker):
    checkpoint, runtime, client, _ = worker
    engine = checkpoint.engine
    engine.storage_manager = None
    LMCacheEngine.freeze(engine, True)
    LMCacheEngine.set_hot_cache(engine, False)
    assert runtime.storage_manager.is_frozen()
    assert not runtime.storage_manager.is_hot_cache_enabled()
    count = len(client.controls)
    for _ in range(1000):
        assert LMCacheEngine.is_frozen(engine)
        assert not LMCacheEngine.is_hot_cache_enabled(engine)
    assert len(client.controls) == count
    local = NS(storage_process=None, storage_manager=runtime.storage_manager)
    LMCacheEngine.freeze(local, False)
    assert not LMCacheEngine.is_frozen(local)
    assert len(client.controls) == count


def test_actual_backend_and_bypass_handler_use_manager_facade(worker):
    checkpoint, runtime, _, _ = worker
    manager = object.__new__(LMCacheManager)
    manager._lmcache_engine = checkpoint.engine
    checkpoint.engine.storage_manager = None
    request = NS(
        app=NS(state=NS(lmcache_adapter=manager)),
        query_params={"backend_name": "LocalCPUBackend"},
    )
    listed = asyncio.run(backend_api.list_backends(request))
    assert listed.status_code == 200
    assert "LocalCPUBackend" in json.loads(listed.body)
    bypassed = asyncio.run(bypass_api.add_bypass_backend(request))
    assert bypassed.status_code == 200
    assert json.loads(bypassed.body)["bypassed"]
    assert runtime.storage_manager.is_backend_bypassed("LocalCPUBackend")
    refused = asyncio.run(backend_api.close_backend(request, "LocalCPUBackend"))
    assert refused.status_code == 500
    assert "restart" in json.loads(refused.body)["message"]
    assert checkpoint.engine.storage_manager is None


def test_default_admin_reuses_existing_local_storage_manager():
    local = object()
    assert get_engine_storage_admin(NS(storage_manager=local)) is local


def config_request(worker, body):
    checkpoint, runtime, _, _ = worker
    runtime.config.decoder_storage_mode = "process"
    config = deepcopy(runtime.config)
    checkpoint.engine.config = config
    manager = object.__new__(LMCacheManager)
    manager._lmcache_engine = checkpoint.engine
    manager._config = config

    async def json_body():
        return body

    request = NS(app=NS(state=NS(lmcache_adapter=manager)), json=json_body)
    return config, runtime, request


def test_storage_owner_mode_is_restart_only_in_actual_config_handler(worker):
    config, _, request = config_request(worker, {"decoder_storage_mode": "local"})

    response = asyncio.run(conf_api.set_config(request))
    assert response.status_code == 400
    assert json.loads(response.body)["errors"] == {
        "decoder_storage_mode": "Config is not mutable at runtime"
    }
    assert config.decoder_storage_mode == "process"


def test_conf_forwards_valid_options_before_worker_mirror_and_rejects_layout(worker):
    config, runtime, request = config_request(
        worker,
        {
            "blocking_timeout_secs": "19",
            "chunk_size": 8,
            "no_such_config": 1,
        },
    )
    response = asyncio.run(conf_api.set_config(request))
    value = json.loads(response.body)
    assert response.status_code == 200
    assert value["updated"] == {"blocking_timeout_secs": 19}
    assert set(value["errors"]) == {"chunk_size", "no_such_config"}
    assert config.blocking_timeout_secs == runtime.config.blocking_timeout_secs == 19
    assert config.chunk_size == runtime.config.chunk_size == 4
    assert runtime.storage_manager.config is runtime.config


def test_conf_extra_backend_options_preserve_internal_slab_and_key_identity(worker):
    checkpoint, runtime, _, _ = worker
    runtime.config.extra_config["shm_name"] = "registered-slab"
    runtime.config.extra_config["shared_cpu_cache_resolved_name"] = "registered-slab"
    extra = dict(runtime.config.extra_config, transfer_timeout=17)
    del extra["shm_name"]
    del extra["shared_cpu_cache_resolved_name"]
    config, runtime, request = config_request(worker, {"extra_config": extra})
    assert asyncio.run(conf_api.set_config(request)).status_code == 200
    assert runtime.config.extra_config["transfer_timeout"] == 17
    assert (
        config.extra_config["shm_name"]
        == runtime.config.extra_config["shm_name"]
        == "registered-slab"
    )
    changed = dict(config.extra_config, mooncake_dsa_raw_token_dims={0: 577, 1: 128})
    config, runtime, request = config_request(worker, {"extra_config": changed})
    response = asyncio.run(conf_api.set_config(request))
    assert response.status_code == 400
    assert "extra_config" in json.loads(response.body)["errors"]
    assert runtime.config.extra_config["mooncake_dsa_raw_token_dims"] == {
        0: 576,
        1: 128,
    }


def test_local_conf_response_and_mutability_remain_unchanged():
    config = LMCacheEngineConfig.from_defaults()

    async def body():
        return {"chunk_size": 8}

    request = NS(app=NS(state=NS(lmcache_adapter=NS(config=config))), json=body)
    response = asyncio.run(conf_api.set_config(request))
    assert response.status_code == 200
    assert json.loads(response.body) == {"updated": {"chunk_size": 8}}
    assert config.chunk_size == 8


def test_explicit_status_queries_refresh_m_state_after_another_worker_control(worker):
    checkpoint, runtime, _, _ = worker
    manager = object.__new__(LMCacheManager)
    manager._lmcache_engine = checkpoint.engine
    request = NS(app=NS(state=NS(lmcache_adapter=manager)))
    # Simulate another rank's explicit administration: local hot-path bits are
    # stale until this explicit HTTP query refreshes them from M.
    runtime.storage_manager.set_freeze(True)
    runtime.storage_manager.set_hot_cache(False)
    admin = get_engine_storage_admin(checkpoint.engine)
    assert not admin.cached_frozen
    assert admin.cached_hot_cache
    assert json.loads(asyncio.run(freeze_api.get_freeze_status(request)).body)["freeze"]
    assert not json.loads(
        asyncio.run(hot_cache_api.get_hot_cache_status(request)).body
    )["hot_cache"]
    assert admin.cached_frozen and not admin.cached_hot_cache


def test_actual_base_clear_and_health_delegate_without_worker_owner(worker):
    checkpoint, runtime, _, _ = worker
    engine = checkpoint.engine
    engine.storage_manager = None
    engine.token_database = runtime.token_database
    page, _ = runtime.allocate_checkpoint_fragment(0, 4)
    key = next(iter(runtime.token_database.process_tokens(tokens=[0, 1, 2, 3])))[2]
    runtime.storage_manager.local_cpu_backend.batched_submit_layer_pages([key], [page])
    page.ref_count_down()
    assert (
        LMCacheEngine._clear(engine, tokens=[0, 1, 2, 3], locations=["LocalCPUBackend"])
        == 1
    )
    assert LMCacheEngine.health(engine) == 0
    assert engine.storage_manager is None


def test_fs_http_forwards_config_failures_as_original_http_status(worker, tmp_path):
    checkpoint, _, _, _ = worker
    request = NS(app=NS(state=NS(lmcache_adapter=NS(lmcache_engine=checkpoint.engine))))
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            load_fs_chunks_api.load_fs_chunks(
                request,
                load_fs_chunks_api.LoadFSChunksRequest(
                    config_path=str(tmp_path / "missing.yaml")
                ),
            )
        )
    assert error.value.status_code == 400
    assert "Invalid configuration file" in error.value.detail
    assert (
        checkpoint.engine.storage_manager is None
        if hasattr(checkpoint.engine, "storage_manager")
        else True
    )


def test_cpu_metric_merge_preserves_device_counters_and_rank_labels():
    worker = b"""# HELP lmcache:local_cache_usage allocated
# TYPE lmcache:local_cache_usage gauge
lmcache:local_cache_usage{worker_id="0"} 0
lmcache:local_cache_usage{worker_id="1"} 3
# HELP lmcache:num_store_requests_total stores
# TYPE lmcache:num_store_requests_total counter
lmcache:num_store_requests_total{worker_id="0"} 7
"""
    storage = b"""# HELP lmcache:local_cache_usage allocated
# TYPE lmcache:local_cache_usage gauge
lmcache:local_cache_usage{worker_id="0"} 123
# HELP lmcache:num_store_requests_total stores
# TYPE lmcache:num_store_requests_total counter
lmcache:num_store_requests_total{worker_id="0"} 999
"""
    merged = merge_storage_metrics(worker, storage)
    samples = [
        sample
        for family in text_string_to_metric_families(merged.decode())
        for sample in family.samples
    ]
    values = {
        (sample.name, sample.labels["worker_id"]): sample.value for sample in samples
    }
    assert values["lmcache:local_cache_usage", "0"] == 123
    assert values["lmcache:local_cache_usage", "1"] == 3
    assert values["lmcache:num_store_requests_total", "0"] == 7
    assert len(values) == len(samples) == 3


def test_metric_reset_forwards_to_m_and_preserves_worker_reset(monkeypatch):
    reset = []
    manager = NS(
        config=NS(decoder_storage_mode="process"),
        get_storage_admin=lambda: NS(reset_metrics=lambda: reset.append("M")),
    )
    request = NS(app=NS(state=NS(lmcache_adapter=manager)))
    monkeypatch.setattr(
        metrics_api, "reset_observability_metrics", lambda: reset.append("W")
    )
    assert asyncio.run(metrics_api.reset_metrics(request)).body == b"ok"
    assert reset == ["M", "W"]
