# SPDX-License-Identifier: Apache-2.0
"""Real CPU service bootstrap/liveness tests with explicit fake native registrars."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import ModuleType, SimpleNamespace
from weakref import ref
import ast
import gc
import json
import os
import socket
import subprocess
import sys
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.rpc.storage_bootstrap import (
    discover_storage_service,
    get_worker_storage_bootstrap,
    start_worker_storage,
    storage_discovery_path,
)
from lmcache.v1.rpc import storage_bootstrap
from lmcache.v1.cpu_cache import StorageOperationRegistry

_PROCESSES = []


@pytest.fixture(autouse=True)
def cleanup_probe_processes():
    yield
    while _PROCESSES:
        process = _PROCESSES.pop()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


class BootstrapProbeRuntime:
    """Exercise actual process/lifecycle transport, not native NPU registration."""

    def __init__(self, runtime_config):
        self.config = runtime_config
        self.slab = SharedMemory(create=True, size=64)
        self.registered = set()
        self.layouts = set()
        self.failure = None
        self.registry = StorageOperationRegistry(
            runtime_config["cache_instance_id"],
            runtime_config["service_epoch"],
            {
                name: bytes.fromhex(nonce)
                for name, nonce in runtime_config["participants"].items()
            },
        )

    def bootstrap_descriptor(self):
        return {
            "slab_name": self.slab.name,
            "slab_size": self.slab.size,
            "slab_generation": 1,
        }

    def is_healthy(self):
        return True

    def quiesce(self):
        pass

    def handle(self, request):
        return self.registry.handle(request, self.dispatch)

    def dispatch(self, request):
        method = request.method
        value = None
        if method == "register_participant":
            if request.payload.get("error"):
                self.failure = request.payload["error"]
            else:
                resource = request.payload["resource"]
                assert resource["native_registered"] and resource["reader_ready"]
                self.registered.add(request.payload["session"])
        elif method == "registration_status":
            value = {
                "status": "FAILED"
                if self.failure
                else (
                    "READY"
                    if self.registered == set(self.config["expected_workers"])
                    else "WAITING"
                ),
                "error": self.failure,
                "layout_status": "READY"
                if self.layouts == set(self.config["expected_workers"])
                else "WAITING",
            }
        elif method == "bind_layout":
            self.layouts.add(request.payload["session"])
        elif method == "unregister_participant":
            self.registered.discard(request.payload["session"])
        elif method == "stop":
            value = {"status": "BUSY" if self.registered else "STOPPED"}
        return value

    def close(self):
        self.slab.close()
        self.slab.unlink()


def endpoint(_kind, _epoch):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{reservation.getsockname()[1]}"


def probe_spawn(argv, **kwargs):
    path = Path(argv[argv.index("--bootstrap") + 1])
    config = json.loads(path.read_text(encoding="utf-8"))
    config["runtime_factory"] = "tests.v1.test_storage_bootstrap:BootstrapProbeRuntime"
    path.write_text(json.dumps(config), encoding="utf-8")
    process = subprocess.Popen(argv, **kwargs)
    _PROCESSES.append(process)
    return process


def start(
    tmp_path,
    engine="instance",
    rank=0,
    size=1,
    trace=None,
    failed=None,
    reader_enabled=True,
):
    trace = [] if trace is None else trace

    def mapping(**kwargs):
        trace.append(("mapping", rank))
        assert kwargs["size"] == 64 and kwargs["generation"] == 1
        return SimpleNamespace(
            preflight_device_ptr=lambda: 4096,
            close=lambda: trace.append(("detach", rank)),
        )

    def reader():
        trace.append(("reader", rank))
        return SimpleNamespace(close=lambda: trace.append(("reader-close", rank)))

    return start_worker_storage(
        engine_id=engine,
        rpc_port=0,
        tp_rank=rank,
        tp_size=size,
        runtime_config={},
        mapping_factory=mapping,
        reader_factory=reader if reader_enabled else None,
        timeout_s=10,
        base_url=str(tmp_path),
        launcher_pid=os.getpid(),
        spawn=probe_spawn,
        endpoint_factory=endpoint,
        failure_callback=(lambda reason: failed.set()) if failed else trace.append,
    )


def test_owner_is_also_registered_client_and_reuses_namespace_after_drain(tmp_path):
    trace = []
    context = start(tmp_path, trace=trace)
    descriptor = context.descriptor
    assert get_worker_storage_bootstrap("instance") is context
    assert trace == [("mapping", 0), ("reader", 0)]
    assert descriptor["owner_pid"] == context.process.pid != os.getpid()
    assert "scheduler" in descriptor["sessions"]
    context.close()
    assert trace[-2:] == [("reader-close", 0), ("detach", 0)]
    assert not storage_discovery_path("instance", base_url=str(tmp_path)).exists()
    replacement = start(tmp_path)
    try:
        assert replacement.descriptor["service_epoch"] != descriptor["service_epoch"]
    finally:
        replacement.close()


@pytest.mark.parametrize("reader_enabled", [False, True])
def test_noncollective_registration_gate_waits_for_every_rank(tmp_path, reader_enabled):
    trace = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        owner = workers.submit(
            start,
            tmp_path,
            "tp-instance",
            0,
            2,
            trace,
            reader_enabled=reader_enabled,
        )
        discover_storage_service("tp-instance", base_url=str(tmp_path), timeout_s=10)
        assert not owner.done()
        passive = workers.submit(
            start,
            tmp_path,
            "tp-instance",
            1,
            2,
            trace,
            reader_enabled=reader_enabled,
        )
        first, second = owner.result(10), passive.result(10)
    try:
        assert first.process is not None and second.process is None
        assert first.device_base == second.device_base == 4096
        assert first.descriptor["service_epoch"] == second.descriptor["service_epoch"]
        expected = [("mapping", 0), ("mapping", 1)]
        if reader_enabled:
            expected += [("reader", 0), ("reader", 1)]
        assert sorted(trace) == expected
    finally:
        first.close()
        second.close()


def test_two_instances_have_independent_epochs_and_sessions(tmp_path):
    first = start(tmp_path, "dp0")
    second = start(tmp_path, "dp1")
    try:
        assert first.descriptor["slab_name"] != second.descriptor["slab_name"]
        assert first.descriptor["work_endpoint"] != second.descriptor["work_endpoint"]
        assert first.descriptor["sessions"] != second.descriptor["sessions"]
    finally:
        first.close()
        second.close()


def test_stale_launch_and_existing_owner_are_not_adopted(tmp_path):
    context = start(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another executor launch"):
            discover_storage_service(
                "instance", base_url=str(tmp_path), launcher_pid=-1
            )
        # A live namespace can only have one creator, including duplicate startup.
        with pytest.raises(RuntimeError, match="already exists"):
            start(tmp_path)
    finally:
        context.close()


def test_service_death_fails_pending_health_even_when_worker_idle(tmp_path):
    failed = threading.Event()
    context = start(tmp_path, failed=failed)
    context.process.terminate()
    context.process.wait(timeout=5)
    assert failed.wait(3)
    assert not context.client.is_healthy()
    with pytest.raises(RuntimeError):
        context.close()


def test_registration_failure_is_published_before_worker_failure(tmp_path):
    processes, failures = [], []

    def spawn(*args, **kwargs):
        process = probe_spawn(*args, **kwargs)
        processes.append(process)
        return process

    def mapping(**kwargs):
        raise ValueError("native attach failure")

    try:
        with pytest.raises(ValueError, match="native attach failure"):
            start_worker_storage(
                engine_id="failed",
                rpc_port=0,
                tp_rank=0,
                tp_size=2,
                runtime_config={},
                mapping_factory=mapping,
                reader_factory=object,
                timeout_s=5,
                base_url=str(tmp_path),
                launcher_pid=os.getpid(),
                spawn=spawn,
                endpoint_factory=endpoint,
                failure_callback=failures.append,
            )
        assert failures == ["native attach failure"]
        descriptor = discover_storage_service("failed", base_url=str(tmp_path))
        client = storage_bootstrap.StorageClient(
            descriptor["work_endpoint"],
            descriptor["lifecycle_endpoint"],
            cache_instance_id="failed",
            service_epoch=descriptor["service_epoch"],
            client_session="scheduler",
            session_nonce=bytes.fromhex(descriptor["sessions"]["scheduler"]),
        )
        try:
            result = client.request(
                "registration_status",
                {},
                request_id="bootstrap",
                request_attempt="scheduler",
                load_generation=0,
                purpose="bootstrap",
            ).result(2)
            assert result.value["status"] == "FAILED"
            assert processes[0].poll() is None
        finally:
            client.close()
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=5)


def test_early_bootstrap_uses_effective_chunk_size_and_launcher_rank(monkeypatch):
    from lmcache.integration.vllm import utils

    from lmcache.v1.config import LMCacheEngineConfig

    config = LMCacheEngineConfig.from_defaults(
        decoder_storage_mode="process",
        pd_role="receiver",
        local_cpu=True,
        enable_shared_cpu_cache=True,
        use_layerwise=True,
        enable_sparse_attention=True,
        dsa_two_groups=True,
        save_unfull_chunk=True,
        remote_url=None,
        enable_remote_lmcache_store=False,
        dsa_group1_load_mode="p2p_preferred",
        chunk_size=1024,
    )
    config.validate()
    metadata = SimpleNamespace(
        model_name="GLM",
        world_size=4,
        local_world_size=4,
        worker_id=2,
        local_worker_id=2,
        use_mla=True,
        role="worker",
        served_model_name="GLM-5.3-falcon",
        engine_id="dp0",
        chunk_size=256,
        max_model_len=1048576,
        kv_dtype="bfloat16",
        kv_shape=(79, 1, 1024, 1, 576),
    )
    monkeypatch.setattr(utils, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(
        utils, "create_lmcache_metadata", lambda *a, **k: (metadata, config)
    )
    monkeypatch.setitem(sys.modules, "lmcache_ascend", ModuleType("lmcache_ascend"))
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm.envs = ModuleType("vllm.envs")
    vllm.envs.VLLM_ENGINE_READY_TIMEOUT_S = 600
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", vllm.envs)
    captured = {}
    monkeypatch.setattr(
        storage_bootstrap, "start_worker_storage", lambda **kw: captured.update(kw)
    )
    worker = SimpleNamespace(
        rank=2,
        local_rank=2,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            world_size=4,
            local_world_size=4,
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_num_kv_heads=lambda _: 1),
            kv_transfer_config=SimpleNamespace(
                kv_connector="LMCacheAscendConnectorV1Dynamic",
                engine_id="dp0",
                kv_connector_extra_config={},
            ),
        ),
    )
    storage_bootstrap.early_bootstrap_decoder_storage(worker)
    assert captured["tp_rank"] == 2 and captured["tp_size"] == 4
    assert captured["runtime_config"]["owner_device"] == 0
    assert captured["runtime_config"]["metadata"]["chunk_size"] == 1024
    assert captured["reader_factory"] is None


def test_local_mode_does_not_create_native_bootstrap(monkeypatch):
    from lmcache.integration.vllm import utils

    # The Ascend connector still needs its config patch in local mode.
    monkeypatch.setitem(sys.modules, "lmcache_ascend", ModuleType("lmcache_ascend"))
    monkeypatch.setattr(
        utils,
        "lmcache_get_or_create_config",
        lambda: SimpleNamespace(decoder_storage_mode="local", extra_config=None),
    )
    monkeypatch.setattr(
        storage_bootstrap,
        "start_worker_storage",
        lambda **kw: pytest.fail("local native bootstrap"),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector="LMCacheAscendConnectorV1Dynamic",
            )
        )
    )
    assert storage_bootstrap.early_bootstrap_decoder_storage(worker) is None


def test_tp4_dp4_bootstrap_preserves_native_routing_artifact_and_overrides(
    monkeypatch, tmp_path
):
    import torch
    from lmcache.integration.vllm import utils
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.mooncake_layout import resolve_remote_fill_identity
    from lmcache.v1.storage_backend import remote_backend

    model = tmp_path / "GLM-5.3"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"deepseek_v3"}')
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm.envs = ModuleType("vllm.envs")
    vllm.envs.VLLM_ENGINE_READY_TIMEOUT_S = 600
    native_utils = ModuleType("vllm.utils")
    native_utils.__path__ = []
    native_torch = ModuleType("vllm.utils.torch_utils")
    native_torch.get_kv_cache_torch_dtype = lambda *args: torch.bfloat16
    for name, module in (
        ("vllm", vllm),
        ("vllm.envs", vllm.envs),
        ("vllm.utils", native_utils),
        ("vllm.utils.torch_utils", native_torch),
        ("lmcache_ascend", ModuleType("lmcache_ascend")),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(utils, "mla_enabled", lambda model: True)
    monkeypatch.setattr(utils, "calculate_draft_layers", lambda config: 1)
    monkeypatch.setattr(
        remote_backend,
        "RemoteExternalPageReader",
        lambda config, metadata, **kw: SimpleNamespace(
            config=config, metadata=metadata
        ),
    )
    captures = []
    monkeypatch.setattr(
        storage_bootstrap, "start_worker_storage", lambda **kw: captures.append(kw)
    )
    # The production port helper is CPU-only; load it without native package init.
    helper = Path(__file__).resolve().parents[3] / (
        "LMCache-Ascend/lmcache_ascend/v1/cpu_cache_operations.py"
    )
    node = next(
        node
        for node in ast.parse(helper.read_text()).body
        if getattr(node, "name", None) == "_validate_remote_fill_control_port"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(helper), "exec"),
        namespace,
    )
    ports = []
    for dp_rank in (0, 1):
        config = LMCacheEngineConfig.from_defaults(
            decoder_storage_mode="process",
            pd_role="receiver",
            enable_shared_cpu_cache=True,
            enable_remote_lmcache_store=True,
            remote_url="mooncakestore://127.0.0.1:58888/",
            chunk_size=1024,
        )
        monkeypatch.setattr(
            utils, "lmcache_get_or_create_config", lambda cfg=config: cfg
        )
        parallel = SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            world_size=4,
            local_world_size=4,
            rank=0,
            data_parallel_index=dp_rank,
            data_parallel_rank_local=0,
            data_parallel_size=4,
        )
        transfer = SimpleNamespace(
            kv_connector="LMCacheAscendConnectorV1Dynamic",
            engine_id=f"d-dp{dp_rank}",
            kv_connector_extra_config={"lmcache.max_local_cpu_size": 3.25},
        )
        model_config = SimpleNamespace(
            model=str(model),
            served_model_name="GLM-5.3-falcon",
            dtype=torch.bfloat16,
            get_num_layers=lambda _: 78,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 576,
            max_model_len=1048576,
        )
        worker = SimpleNamespace(
            rank=0,
            local_rank=0,
            parallel_config=parallel,
            vllm_config=SimpleNamespace(
                parallel_config=parallel,
                kv_transfer_config=transfer,
                model_config=model_config,
                cache_config=SimpleNamespace(cache_dtype="auto"),
                scheduler_config=SimpleNamespace(
                    max_num_seqs=16, max_num_batched_tokens=4096
                ),
            ),
        )
        storage_bootstrap.early_bootstrap_decoder_storage(worker)
        captured = captures[-1]
        values = dict(captured["runtime_config"]["metadata"])
        values["kv_dtype"] = getattr(torch, values["kv_dtype"].removeprefix("torch."))
        values["kv_shape"] = tuple(values["kv_shape"])
        cpu_metadata = LMCacheMetadata(**values)
        reader = captured["reader_factory"]()
        extra = cpu_metadata.kv_connector_extra_config
        assert extra["lmcache_remote_fill_destination_dp_rank"] == dp_rank
        assert extra["lmcache_remote_fill_destination_dp_size"] == 4
        ports.append(
            namespace["_validate_remote_fill_control_port"](
                19000,
                extra["lmcache_remote_fill_destination_dp_rank"],
                extra["lmcache_remote_fill_destination_dp_size"],
                internal_api_enabled=True,
                internal_api_port_start=6999 + dp_rank * 100,
            )
        )
        assert cpu_metadata.served_model_name == "GLM-5.3-falcon"
        assert cpu_metadata.mla_cache_tp_replicated
        assert resolve_remote_fill_identity(config, cpu_metadata) == (
            resolve_remote_fill_identity(config, reader.metadata)
        )
        assert captured["runtime_config"]["config"]["max_local_cpu_size"] == 3.25
        assert config.extra_config["vllm_max_model_len"] == 1048576
        assert config.extra_config["vllm_max_num_seqs"] == 16
        assert config.extra_config["vllm_max_num_batched_tokens"] == 4096
    assert captures[0]["engine_id"] != captures[1]["engine_id"]
    assert ports == [19000, 19001]


def test_close_retains_native_resources_until_all_lease_guards_pass(tmp_path):
    trace = []
    context = start(tmp_path, trace=trace)
    drained = threading.Event()
    context.drain_guards.append(drained.is_set)
    with pytest.raises(RuntimeError, match="mapping retained"):
        context.close()
    assert get_worker_storage_bootstrap("instance") is context
    assert trace == [("mapping", 0), ("reader", 0)]
    assert context.process.poll() is None and context.client.is_healthy()
    drained.set()
    context.close()


def test_layout_gate_waits_for_every_worker_without_more_native_setup(tmp_path):
    trace = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        owner = workers.submit(start, tmp_path, "layout-instance", 0, 2, trace)
        discover_storage_service(
            "layout-instance", base_url=str(tmp_path), timeout_s=10
        )
        passive = workers.submit(start, tmp_path, "layout-instance", 1, 2, trace)
        first, second = owner.result(10), passive.result(10)
        before = list(trace)
        first_bound = workers.submit(first.bind_layout, {"digest": b"layout"})
        assert not first_bound.done()
        second_bound = workers.submit(second.bind_layout, {"digest": b"layout"})
        assert first_bound.result(10)["layout_status"] == "READY"
        assert second_bound.result(10)["layout_status"] == "READY"
        assert trace == before
    first.close()
    second.close()


def test_closed_context_has_no_client_callback_cycle_with_gc_disabled(tmp_path):
    enabled = gc.isenabled()
    gc.disable()
    try:
        context = start(tmp_path)
        client = context.client
        tracked = ref(context)
        context.close()
        del context
        deadline = time.monotonic() + 2
        while tracked() is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tracked() is None
        assert not client.is_healthy()
    finally:
        if enabled:
            gc.enable()
