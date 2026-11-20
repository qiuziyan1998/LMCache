# SPDX-License-Identifier: Apache-2.0
"""Configuration, addressing and lookup compatibility for decoder storage."""

# Standard
from typing import Any, Optional
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache


def test_decoder_storage_mode_defaults_local_and_reads_environment(monkeypatch) -> None:
    assert LMCacheEngineConfig.from_defaults().decoder_storage_mode == "local"
    monkeypatch.setenv("LMCACHE_DECODER_STORAGE_MODE", "process")
    assert LMCacheEngineConfig.from_env().decoder_storage_mode == "process"


@pytest.mark.parametrize("mode", ["local", "process"])
def test_decoder_storage_mode_accepts_explicit_values(mode: str) -> None:
    config = LMCacheEngineConfig.from_defaults(decoder_storage_mode=mode)
    config.validate()
    assert config.decoder_storage_mode == mode


@pytest.mark.parametrize("mode", ["", "PROCESS", "remote", None])
def test_decoder_storage_mode_rejects_other_values(mode: Optional[str]) -> None:
    config = LMCacheEngineConfig.from_defaults(decoder_storage_mode=mode)
    with pytest.raises(ValueError, match="decoder_storage_mode"):
        config.validate()


@pytest.mark.parametrize("rpc_port", [0, 100, "instance"])
def test_decoder_storage_endpoints_are_distinct_and_instance_scoped(
    rpc_port: int | str,
) -> None:
    services = (
        "lookup",
        "offload",
        "lookup_worker",
        "lookup_scheduler",
        "storage_work",
        "storage_lifecycle",
        "storage_bootstrap",
        "native_work",
        "native_lifecycle",
    )
    paths = {
        get_zmq_rpc_path_lmcache("dp0", service, rpc_port, rank, "/tmp/lmcache")
        for service in services
        for rank in (0, 1)
    }
    assert len(paths) == len(services) * 2
    assert not paths.intersection(
        get_zmq_rpc_path_lmcache("dp1", service, rpc_port, rank, "/tmp/lmcache")
        for service in services
        for rank in (0, 1)
    )


def test_decoder_storage_endpoints_reject_unknown_service() -> None:
    with pytest.raises(ValueError, match="service_name"):
        get_zmq_rpc_path_lmcache("dp0", "storage_typo", base_url="/tmp/lmcache")


class LegacyLookupClient(LookupClientInterface):
    """Existing clients need not implement ticket support."""

    def lookup(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        return len(token_ids)

    def close(self) -> None:
        pass


def test_legacy_lookup_client_has_no_ticket() -> None:
    client = LegacyLookupClient()
    assert client.peek_ticket("req") is None
    assert client.lookup([1, 2], "req") == 2


@pytest.mark.parametrize("with_ticket", [False, True])
def test_lookup_wrappers_preserve_ticket_identity_and_cleanup(
    with_ticket: bool,
) -> None:
    actual = Mock(spec=LookupClientInterface)
    ticket: Any = object() if with_ticket else None
    actual.peek_ticket.return_value = ticket
    config = LMCacheEngineConfig.from_defaults(
        hit_miss_ratio=0.0,
        chunk_statistics_auto_start_statistics=False,
    )
    limited = HitLimitLookupClient(actual, config)
    wrapped = ChunkStatisticsLookupClient(limited, config)
    try:
        assert limited.peek_ticket("req") is ticket
        assert wrapped.peek_ticket("req") is ticket
        assert actual.peek_ticket.call_count == 2
        actual.clear_lookup_status.assert_not_called()
        actual.cleanup_lookup.assert_not_called()
        wrapped.clear_lookup_status("req")
        actual.clear_lookup_status.assert_called_once_with("req")
        wrapped.cleanup_lookup("req")
        actual.cleanup_lookup.assert_called_once_with("req")
    finally:
        wrapped.close()
