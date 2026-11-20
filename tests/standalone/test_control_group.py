# SPDX-License-Identifier: Apache-2.0
"""Transport selection, collective counts and lifetime of LMCache control."""

# Standard
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock
import sys

# Third Party
import pytest
import torch
import torch.distributed as dist

# First Party
from lmcache.integration.vllm import utils, vllm_service_factory
from lmcache.v1 import gpu_connector
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.control_group import LMCacheControlGroup


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> NS:
    """Retain real transport methods while recording distributed operations."""
    state = NS(
        create=Mock(return_value=object()),
        destroy=Mock(),
        reduce=Mock(),
        broadcast=Mock(),
    )
    monkeypatch.setattr(dist, "new_group", state.create)
    monkeypatch.setattr(dist, "destroy_process_group", state.destroy)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", state.reduce)
    monkeypatch.setattr(dist, "broadcast_object_list", state.broadcast)
    return state


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch, transport: NS) -> NS:
    """Supply vLLM/GPU collaborators without constructing a model or NPU."""
    tp = NS(
        ranks=[4, 5, 6, 7],
        world_size=4,
        cpu_group=object(),
        mq_broadcaster=object(),
        broadcast=Mock(),
        broadcast_object=Mock(),
    )
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tp_group = lambda: tp
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    connector = Mock(return_value=object())
    monkeypatch.setattr(gpu_connector, "CreateGPUConnector", connector)
    monkeypatch.setattr(utils, "vllm_layout_hints", lambda: {})
    engine = NS(save_only_first_rank=True)
    builder = NS(get=Mock(return_value=None), get_or_create=Mock(return_value=engine))
    monkeypatch.setattr(vllm_service_factory, "LMCacheEngineBuilder", builder)
    service = vllm_service_factory.VllmServiceFactory(
        LMCacheEngineConfig.from_defaults(), NS(), "worker"
    )
    service.metadata = NS(use_mla=True)
    return NS(
        service=service, tp=tp, engine=engine, builder=builder, connector=connector
    )


@pytest.mark.parametrize("queued", [False, True])
def test_queue_callback_is_unchanged_and_only_existing_vote_uses_new_group(
    factory: NS,
    transport: NS,
    queued: bool,
) -> None:
    """Repeated transfers add no group creation, wrapper or extra collective."""
    if not queued:
        factory.tp.mq_broadcaster = None
    assert factory.service.get_or_create_lmcache_engine() is factory.engine
    transport.create.assert_called_once_with(
        ranks=[4, 5, 6, 7],
        backend="gloo",
        use_local_synchronization=True,
        group_desc="lmcache_control",
    )
    args = factory.builder.get_or_create.call_args.args
    assert args[4] is factory.tp.broadcast
    broadcast, ready = args[5:7]
    if queued:
        assert broadcast is factory.tp.broadcast_object
    for step in range(20):
        payload = {"request": step}
        broadcast(payload, 0)
        assert ready(step % 2 == 0) == (step % 2 == 0)
    assert transport.reduce.call_count == 20
    for call in transport.reduce.call_args_list:
        flag = call.args[0]
        assert (
            flag.shape == (1,)
            and flag.dtype == torch.int32
            and flag.device.type == "cpu"
        )
        assert call.kwargs == {
            "op": dist.ReduceOp.MIN,
            "group": transport.create.return_value,
        }
    assert transport.broadcast.call_count == (0 if queued else 20)
    assert factory.tp.broadcast_object.call_count == (20 if queued else 0)
    if not queued:
        assert all(
            call.kwargs == {"src": 4, "group": transport.create.return_value}
            for call in transport.broadcast.call_args_list
        )
    factory.builder.get.return_value = factory.engine
    assert factory.service.get_or_create_lmcache_engine() is factory.engine
    assert transport.create.call_count == 1


def test_tp1_keeps_local_callbacks_without_a_group(factory: NS, transport: NS) -> None:
    """A single worker has neither new collectives nor control ownership."""
    factory.tp.world_size, factory.tp.ranks = 1, [4]
    factory.service.get_or_create_lmcache_engine()
    assert factory.engine.control_group is None
    args = factory.builder.get_or_create.call_args.args
    assert args[5] is factory.tp.broadcast_object and args[6] is bool
    transport.create.assert_not_called()


@pytest.mark.parametrize(
    "process,bypass", [(False, False), (False, True), (True, True)]
)
def test_scheduler_never_owns_a_group(
    factory: NS,
    transport: NS,
    monkeypatch: pytest.MonkeyPatch,
    process: bool,
    bypass: bool,
) -> None:
    """Scheduler bypass lookup stays local; process-mode schedulers have no engine."""
    from lmcache.observability import PrometheusLogger

    monkeypatch.setattr(PrometheusLogger, "GetOrCreate", Mock())
    service = factory.service
    service.role = "scheduler"
    service.lmcache_config.enable_scheduler_bypass_lookup = bypass
    service.lmcache_config.decoder_storage_mode = "process" if process else "local"
    result = service.get_or_create_lmcache_engine()
    if bypass and not process:
        assert result.control_group is None
        args = factory.builder.get_or_create.call_args.args
        assert args[5]("object", 0) == "object" and args[6] is bool
    else:
        assert result is None
    transport.create.assert_not_called()


@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_construction_releases_group_without_masking_original_error(
    factory: NS,
    transport: NS,
    close_fails: bool,
) -> None:
    """Constructor failure never installs a partly built engine."""
    factory.builder.get_or_create.side_effect = RuntimeError("engine construction")
    if close_fails:
        transport.destroy.side_effect = RuntimeError("group close")
    with pytest.raises(RuntimeError, match="engine construction"):
        factory.service.get_or_create_lmcache_engine()
    transport.destroy.assert_called_once_with(transport.create.return_value)
    assert factory.service.lmcache_engine is None


def test_connector_failure_precedes_group_creation(factory: NS, transport: NS) -> None:
    """A local GPU-connector failure does not allocate control resources."""
    factory.connector.side_effect = RuntimeError("GPU connector")
    with pytest.raises(RuntimeError, match="GPU connector"):
        factory.service.get_or_create_lmcache_engine()
    transport.create.assert_not_called()


@pytest.mark.parametrize("process", [False, True])
@pytest.mark.parametrize("fail_close", [False, True])
def test_group_is_released_only_after_successful_storage_drain(
    process: bool, fail_close: bool
) -> None:
    """An unsuccessful strict drain retains ownership for a later retry."""
    events = []

    def close_storage() -> None:
        events.append("storage")
        if fail_close:
            raise RuntimeError("storage drain")

    storage = NS(close=close_storage, requires_strict_external_close=lambda: True)
    control = NS(close=lambda: events.append("control"))
    engine = NS(
        storage_process=storage if process else None,
        storage_manager=None if process else storage,
        lmcache_worker=None,
        shared_cpu_cache_mapping=None,
        control_group=control,
        _release_all_shared_cpu_request_leases=lambda: None,
    )
    if fail_close:
        with pytest.raises(RuntimeError, match="storage drain"):
            LMCacheEngine.close(engine)
        assert engine.control_group is control and events == ["storage"]
    else:
        LMCacheEngine.close(engine)
        assert engine.control_group is None and events == ["storage", "control"]
        LMCacheEngine.close(engine)
        assert events.count("control") == 1


def test_fallback_maps_tp_local_source_and_close_never_destroys_model_group(
    transport: NS,
) -> None:
    """Nonzero TP-local sources must map to their global process rank."""
    control = LMCacheControlGroup([4, 5, 6, 7])
    payload = {"value": 3}
    assert control.broadcast_object(payload, 2) is payload
    transport.broadcast.assert_called_once_with(
        [payload], src=6, group=transport.create.return_value
    )
    with pytest.raises(ValueError, match="source rank"):
        control.broadcast_object(None, -1)
    control.close()
    control.close()
    transport.destroy.assert_called_once_with(transport.create.return_value)
    with pytest.raises(RuntimeError, match="closed"):
        control.collective_all_true(True)
    with pytest.raises(RuntimeError, match="closed"):
        control.broadcast_object(None)


def test_close_after_global_teardown_and_singleton(
    transport: NS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Global teardown and the local-only case require no further destruction."""
    control = LMCacheControlGroup([4, 5])
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    control.close()
    transport.destroy.assert_not_called()
    control = LMCacheControlGroup([4])
    assert control.collective_all_true(True) and not control.collective_all_true(False)
    assert control.broadcast_object("local") == "local"
    control.close()
    transport.reduce.assert_not_called()


def test_failed_group_close_retains_ownership_for_retry(transport: NS) -> None:
    """A failed destroy must not mark the group successfully closed."""
    control = LMCacheControlGroup([4, 5])
    transport.destroy.side_effect = [RuntimeError("group destroy"), None]
    with pytest.raises(RuntimeError, match="group destroy"):
        control.close()
    control.close()
    control.close()
    assert transport.destroy.call_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        control.collective_all_true(True)


@pytest.mark.parametrize("ranks", [[], [0, 0]])
def test_invalid_group_rejected_before_collective(
    transport: NS, ranks: list[int]
) -> None:
    """Malformed membership never reaches distributed initialization."""
    with pytest.raises(ValueError, match="ranks"):
        LMCacheControlGroup(ranks)
    transport.create.assert_not_called()
