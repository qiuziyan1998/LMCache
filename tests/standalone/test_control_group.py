# SPDX-License-Identifier: Apache-2.0
"""Control transport routing and ownership without a vLLM/NPU runtime."""

# Standard
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock
import sys

# Third Party
import pytest
import torch.distributed as dist

# First Party
from lmcache.integration.vllm import utils, vllm_service_factory
from lmcache.v1 import gpu_connector
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.control_group import LMCacheControlGroup


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> NS:
    """Replace distributed I/O, retaining the production group and callbacks."""
    group = object()
    create = Mock(return_value=group)
    destroy = Mock()
    reduce = Mock()
    broadcast = Mock()
    monkeypatch.setattr(dist, "new_group", create)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    return NS(
        group=group, create=create, destroy=destroy, reduce=reduce, broadcast=broadcast
    )


@pytest.fixture
def factory_setup(monkeypatch: pytest.MonkeyPatch, transport: NS) -> NS:
    """Supply only the model/GPU dependencies used during factory startup."""
    tp = NS(
        ranks=[4, 5, 6, 7],
        broadcast=Mock(),
        broadcast_object=Mock(side_effect=AssertionError("shared model transport")),
    )
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: tp
    for name in ("vllm", "vllm.distributed"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    connector = Mock(return_value=object())
    monkeypatch.setattr(gpu_connector, "CreateGPUConnector", connector)
    monkeypatch.setattr(utils, "vllm_layout_hints", lambda: {})
    engine = NS(save_only_first_rank=True)
    builder = NS(get=Mock(return_value=None), get_or_create=Mock(return_value=engine))
    monkeypatch.setattr(vllm_service_factory, "LMCacheEngineBuilder", builder)
    factory = vllm_service_factory.VllmServiceFactory(
        LMCacheEngineConfig.from_defaults(), NS(), "worker"
    )
    factory.metadata = NS(use_mla=True)
    return NS(
        factory=factory, tp=tp, connector=connector, engine=engine, builder=builder
    )


def test_singleton_needs_no_distributed_initialization(transport: NS) -> None:
    """TP1 preserves local behavior and releases no model process groups."""
    group = LMCacheControlGroup([3])
    payload = {"request": "one"}
    assert group.broadcast_object(payload) is payload
    assert group.collective_all_true(True)
    assert not group.collective_all_true(False)
    group.close()
    group.close()
    transport.create.assert_not_called()
    transport.destroy.assert_not_called()
    transport.reduce.assert_not_called()
    transport.broadcast.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        group.collective_all_true(True)
    with pytest.raises(RuntimeError, match="closed"):
        group.broadcast_object(payload)


def test_worker_callbacks_and_cached_engine_share_one_owned_group(
    factory_setup: NS, transport: NS
) -> None:
    """Metadata/readiness use the new group; device payload transport is retained."""
    setup = factory_setup
    assert setup.factory.get_or_create_lmcache_engine() is setup.engine
    transport.create.assert_called_once_with(
        ranks=[4, 5, 6, 7],
        backend="gloo",
        use_local_synchronization=True,
        group_desc="lmcache_control",
    )
    args = setup.builder.get_or_create.call_args.args
    assert args[4] is setup.tp.broadcast
    broadcast, ready = args[5:7]
    payload = {"phase": "storage_source_import"}
    assert broadcast(payload, 2) is payload
    transport.broadcast.assert_called_once_with([payload], src=6, group=transport.group)
    assert ready(True)
    assert not ready(False)
    assert all(
        call.kwargs == {"op": dist.ReduceOp.MIN, "group": transport.group}
        for call in transport.reduce.call_args_list
    )
    setup.tp.broadcast_object.assert_not_called()
    owned = setup.engine.control_group
    setup.builder.get.return_value = setup.engine
    assert setup.factory.get_or_create_lmcache_engine() is setup.engine
    assert setup.engine.control_group is owned
    assert transport.create.call_count == 1
    owned.close()
    owned.close()
    transport.destroy.assert_called_once_with(transport.group)


@pytest.mark.parametrize("failure", [RuntimeError("engine init"), KeyboardInterrupt()])
def test_failed_engine_creation_releases_group(
    factory_setup: NS, transport: NS, failure: BaseException
) -> None:
    """An engine which never takes ownership cannot leave its new group behind."""
    factory_setup.builder.get_or_create.side_effect = failure
    with pytest.raises(type(failure)):
        factory_setup.factory.get_or_create_lmcache_engine()
    transport.destroy.assert_called_once_with(transport.group)
    assert factory_setup.factory.lmcache_engine is None


def test_failed_connector_does_not_create_control_group(
    factory_setup: NS, transport: NS
) -> None:
    """GPU setup failure precedes collective ownership."""
    factory_setup.connector.side_effect = RuntimeError("connector init")
    with pytest.raises(RuntimeError, match="connector init"):
        factory_setup.factory.get_or_create_lmcache_engine()
    transport.create.assert_not_called()


@pytest.mark.parametrize(
    "mode,bypass", [("local", False), ("local", True), ("process", True)]
)
def test_scheduler_never_creates_a_control_group(
    factory_setup: NS,
    transport: NS,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    bypass: bool,
) -> None:
    """Both engine-free and bypass schedulers remain independent of distributed."""
    # First Party
    from lmcache.observability import PrometheusLogger

    monkeypatch.setattr(PrometheusLogger, "GetOrCreate", Mock())
    factory = factory_setup.factory
    factory.role = "scheduler"
    factory.lmcache_config.decoder_storage_mode = mode
    factory.lmcache_config.enable_scheduler_bypass_lookup = bypass
    engine = factory.get_or_create_lmcache_engine()
    if bypass and mode == "local":
        assert engine is factory_setup.engine
        assert engine.control_group is None
        args = factory_setup.builder.get_or_create.call_args.args
        assert args[5]("marker", 0) == "marker"
        assert args[6](True) and not args[6](False)
    else:
        assert engine is None
    transport.create.assert_not_called()


@pytest.mark.parametrize("mode", ["local", "process"])
@pytest.mark.parametrize("fail_close", [False, True])
def test_engine_releases_control_only_after_storage_close(
    mode: str,
    fail_close: bool,
) -> None:
    """A failed strict drain retains the group with the engine for retry."""
    events: list[str] = []

    def close_storage() -> None:
        events.append("storage")
        if fail_close:
            raise RuntimeError("drain incomplete")

    storage = NS(close=close_storage, requires_strict_external_close=lambda: True)
    control = NS(close=lambda: events.append("control"))
    engine = NS(
        storage_process=storage if mode == "process" else None,
        storage_manager=storage if mode == "local" else None,
        lmcache_worker=None,
        shared_cpu_cache_mapping=None,
        control_group=control,
        _release_all_shared_cpu_request_leases=lambda: None,
    )
    if fail_close:
        with pytest.raises(RuntimeError, match="drain incomplete"):
            LMCacheEngine.close(engine)
        assert events == ["storage"]
        assert engine.control_group is control
    else:
        LMCacheEngine.close(engine)
        assert events == ["storage", "control"]
        assert engine.control_group is None
        LMCacheEngine.close(engine)
        assert events.count("control") == 1


def test_control_close_after_distributed_shutdown(
    transport: NS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global distributed teardown may already have destroyed the owned group."""
    group = LMCacheControlGroup([0, 1])
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    group.close()
    group.close()
    transport.destroy.assert_not_called()


def test_invalid_membership_and_source_fail_before_collective(transport: NS) -> None:
    """Invalid sources must not accidentally become global broadcast ranks."""
    with pytest.raises(ValueError, match="at least one rank"):
        LMCacheControlGroup([])
    group = LMCacheControlGroup([4, 5])
    for src in (-1, 2, 4):
        with pytest.raises(ValueError, match="source rank"):
            group.broadcast_object(None, src)
    transport.broadcast.assert_not_called()
    group.close()
