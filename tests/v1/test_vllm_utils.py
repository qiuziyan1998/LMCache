# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
import lmcache.integration.vllm.utils as vllm_utils
from lmcache.integration.vllm.utils import is_false


class _ParsedConfig:
    remote_config_url = None

    def __init__(self) -> None:
        self.validations = 0

    def validate(self) -> None:
        self.validations += 1


class _ConfigFactory:
    parsed = _ParsedConfig()

    @classmethod
    def from_env(cls) -> _ParsedConfig:
        return cls.parsed


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        ("false", False),
        (" FALSE ", False),
        ("0", False),
        ("off", False),
        ("true", True),
        ("1", True),
    ],
)
def test_force_skip_save_environment_parsing(value: str, enabled: bool) -> None:
    assert (not is_false(value)) is enabled


def test_lmcache_config_is_validated_after_all_sources(monkeypatch) -> None:
    _ConfigFactory.parsed = _ParsedConfig()
    monkeypatch.delenv("LMCACHE_CONFIG_FILE", raising=False)
    monkeypatch.setattr(vllm_utils, "LMCacheEngineConfig", _ConfigFactory)
    monkeypatch.setattr(vllm_utils, "_config_instance", None)

    config = vllm_utils.lmcache_get_or_create_config()

    assert config is _ConfigFactory.parsed
    assert config.validations == 1
