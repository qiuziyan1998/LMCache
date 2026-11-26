# SPDX-License-Identifier: Apache-2.0
"""CPU construction tests; run with --confcutdir=tests/standalone."""

# Standard
from collections import OrderedDict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import storage_manager
from lmcache.v1.storage_backend import remote_backend


def cpu_factory_class():
    """Load the actual factory without executing Ascend package registration."""
    source = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/cpu_cache.py"
    )
    spec = spec_from_file_location("decoder_cpu_factory", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CPUBackendFactory


def config():
    """Return an ordinary LocalCPU/Mooncake decoder storage configuration."""
    result = LMCacheEngineConfig.from_defaults()
    result.local_cpu = True
    result.max_local_cpu_size = 1
    result.remote_url = "mooncakestore://localhost:58888/"
    return result


def metadata():
    """Return CPU-only metadata without device addresses or a connector."""
    return LMCacheMetadata(
        model_name="test",
        world_size=4,
        local_world_size=4,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 1, 8, 1, 16),
        use_mla=True,
        role="worker",
    )


@pytest.mark.parametrize("device_worker", [False, True])
def test_default_construction_preserves_worker_stream(monkeypatch, device_worker):
    backend_factory = Mock(return_value=OrderedDict())
    stream_factory = Mock(return_value=object())
    monkeypatch.setattr(storage_manager, "CreateStorageBackends", backend_factory)
    monkeypatch.setattr(storage_manager, "is_cuda_worker", lambda _: device_worker)
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)
    meta = SimpleNamespace(role="scheduler", worker_id=0)
    manager = storage_manager.StorageManager(config(), meta, EventManager())
    try:
        assert manager.internal_copy_stream is (
            stream_factory.return_value if device_worker else None
        )
        assert stream_factory.call_count == int(device_worker)
        assert backend_factory.call_args.kwargs["dst_device"] == (
            "cuda" if device_worker else "cpu"
        )
    finally:
        manager.close()


@pytest.mark.parametrize("use_factory", [False, True])
def test_explicit_stream_policy_skips_default_device_stream(monkeypatch, use_factory):
    forbidden = Mock(side_effect=AssertionError("unexpected device stream"))
    monkeypatch.setattr(storage_manager, "is_cuda_worker", lambda _: True)
    monkeypatch.setattr(torch.cuda, "Stream", forbidden)
    custom = Mock(return_value=object()) if use_factory else None
    manager = storage_manager.StorageManager(
        config(),
        SimpleNamespace(role="scheduler", worker_id=0),
        EventManager(),
        backend_factory=Mock(return_value=OrderedDict()),
        copy_stream_factory=custom,
    )
    try:
        forbidden.assert_not_called()
        assert manager.internal_copy_stream is (custom.return_value if custom else None)
        if custom:
            custom.assert_called_once_with()
    finally:
        manager.close()


@pytest.mark.parametrize("audit", [False, True])
def test_cpu_factory_reuses_owner_and_same_policy_on_remote_recreation(
    monkeypatch, audit
):
    remotes = []

    def remote(config, metadata, loop, local, dst_device):
        assert dst_device == "cpu"
        backend = SimpleNamespace(
            local_cpu_backend=local, close=Mock(), dst_device="cpu"
        )
        remotes.append(backend)
        return backend

    monkeypatch.setattr(remote_backend, "RemoteBackend", remote)
    monkeypatch.setattr(storage_manager, "is_cuda_worker", lambda _: True)
    monkeypatch.setattr(
        torch.cuda, "Stream", Mock(side_effect=AssertionError("device stream"))
    )
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    factory = cpu_factory_class()(allocator)
    cfg = config()
    cfg.extra_config = {"audit_backend_enabled": audit}
    manager = storage_manager.StorageManager(
        cfg,
        metadata(),
        EventManager(),
        backend_factory=factory,
        copy_stream_factory=None,
    )
    try:
        local = manager.local_cpu_backend
        assert local.memory_allocator is allocator
        assert local.dst_device == "cpu"
        assert manager.create_backends() == {}
        assert len(remotes) == 1
        assert manager.recreate_backend("RemoteBackend") == {
            "RemoteBackend": "AuditBackend" if audit else "SimpleNamespace"
        }
        assert len(remotes) == 2
        assert all(backend.local_cpu_backend is local for backend in remotes)
        assert manager.local_cpu_backend is local
        remotes[0].close.assert_called_once_with()
        close_owner = Mock(side_effect=AssertionError("owner closed"))
        with monkeypatch.context() as owner_patch:
            owner_patch.setattr(local, "close", close_owner)
            with pytest.raises(ValueError, match="requires a restart"):
                manager.recreate_backend("LocalCPUBackend")
            with pytest.raises(ValueError, match="drained service stop"):
                manager.close_backend("LocalCPUBackend")
        close_owner.assert_not_called()
    finally:
        manager.close()


def test_cpu_factory_refuses_a_second_owner_without_allocating(monkeypatch):
    cfg = config()
    cfg.remote_url = None
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    factory = cpu_factory_class()(allocator)
    loop = asyncio.new_event_loop()
    initial = factory(cfg, metadata(), loop)
    try:
        with pytest.raises(ValueError, match="cannot be replaced"):
            factory(cfg, metadata(), loop)
        assert factory(cfg, metadata(), loop, existing_backends=initial) == {}
        different = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
        with pytest.raises(ValueError, match="registered allocator"):
            cpu_factory_class()(different)(
                cfg,
                metadata(),
                loop,
                existing_backends=initial,
            )
    finally:
        initial["LocalCPUBackend"].close()
        loop.close()


@pytest.mark.parametrize("audit", [False, True])
def test_deferred_factory_binds_once_then_recreates_an_immediately_usable_remote(
    monkeypatch, audit
):
    remotes = []

    class Remote:
        def __init__(
            self, config, metadata, loop, local, dst_device, *, defer_layout=False
        ):
            self.metadata, self.local_cpu_backend = metadata, local
            self.defer_layout, self.dst_device = defer_layout, dst_device
            self.close = Mock()
            self.bound = []
            remotes.append(self)

        def bind_layout(self, metadata):
            self.bound.append(metadata.runtime_kv_group_layer_counts)
            self.metadata, self.defer_layout = metadata, False

    monkeypatch.setattr(remote_backend, "RemoteBackend", Remote)
    allocator = TensorMemoryAllocator(torch.zeros(4096, dtype=torch.uint8))
    factory = cpu_factory_class()(allocator, defer_layout=True)
    cfg, meta = config(), metadata()
    cfg.extra_config = {"audit_backend_enabled": audit}
    manager = storage_manager.StorageManager(
        cfg, meta, EventManager(), backend_factory=factory, copy_stream_factory=None
    )
    try:
        assert remotes[0].defer_layout
        meta.runtime_kv_group_layer_counts = (2, 3)
        factory.bind_layout(meta)
        factory.bind_layout(meta)
        assert remotes[0].bound == [(2, 3)]
        manager.recreate_backend("RemoteBackend")
        assert not remotes[1].defer_layout
        assert remotes[1].metadata.runtime_kv_group_layer_counts == (2, 3)
        assert remotes[1].local_cpu_backend is manager.local_cpu_backend
        factory.protect_remote_registration()
        with pytest.raises(ValueError, match="RemoteFill CPU registration"):
            manager.recreate_backend("RemoteBackend")
        with pytest.raises(ValueError, match="RemoteFill CPU registration"):
            manager.close_backend("RemoteBackend")
        remotes[1].close.assert_not_called()
    finally:
        manager.close()


@pytest.mark.parametrize(
    "setting",
    [
        "enable_pd",
        "enable_p2p",
        "gds_path",
        "maru_path",
        "storage_plugins",
        "remote_storage_plugins",
        "local_disk",
        "enable_nixl_storage",
        "remote_url",
    ],
)
def test_cpu_factory_rejects_unsupported_storage_before_allocation(
    monkeypatch, setting
):
    cfg = config()
    if setting == "enable_nixl_storage":
        cfg.extra_config = {setting: True}
    elif setting == "remote_url":
        cfg.remote_url = "redis://localhost/"
    else:
        setattr(cfg, setting, True)
        if setting == "local_disk":
            cfg.max_local_disk_size = 1
    allocator = object()
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError):
            cpu_factory_class()(allocator)(cfg, metadata(), loop)
    finally:
        loop.close()
