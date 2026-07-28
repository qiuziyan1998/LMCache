# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.mooncake_layout import mooncake_page_key


class _FakeStore:
    def __init__(self, rets=None):
        self.keys = None
        self.calls = []
        self.rets = rets

    def batch_is_exist(self, keys):
        self.keys = keys
        self.calls.append(list(keys))
        return self.rets if self.rets is not None else [1 for _ in keys]


class _FakeTokenDatabase:
    def __init__(self, kv_group=1):
        self.kv_group = kv_group

    def process_tokens(self, token_ids, request_configs=None):
        yield (
            0,
            len(token_ids),
            CacheEngineKey(
                "model",
                1,
                0,
                0xABC,
                torch.bfloat16,
                request_configs=request_configs,
                kv_group=self.kv_group,
            ),
        )

    def _make_key_by_hash(self, chunk_hash, request_configs=None, kv_group=0):
        return CacheEngineKey(
            "model",
            1,
            0,
            chunk_hash,
            torch.bfloat16,
            request_configs=request_configs,
            kv_group=kv_group,
        )


class _FakeMultiChunkTokenDatabase(_FakeTokenDatabase):
    chunk_ends = (4, 8, 12, 14)

    def process_tokens(self, token_ids, request_configs=None):
        del token_ids
        start = 0
        for chunk_index, end in enumerate(self.chunk_ends):
            yield (
                start,
                end,
                self._make_key_by_hash(
                    0x100 + chunk_index,
                    request_configs,
                    kv_group=0,
                ),
            )
            start = end


class _PresentStore(_FakeStore):
    def __init__(self, present):
        super().__init__()
        self.present = set(present)

    def batch_is_exist(self, keys):
        self.keys = keys
        self.calls.append(list(keys))
        return [1 if key in self.present else 0 for key in keys]


def _sampled_string_keys(token_db, chunk_index, num_layers=4):
    sampled = []
    for kv_group in (0, 1):
        group_key = token_db._make_key_by_hash(
            0x100 + chunk_index,
            kv_group=kv_group,
        )
        layer_keys = group_key.split_layers(num_layers)
        sampled.extend((layer_keys[0].to_string(), layer_keys[-1].to_string()))
    return sampled


def test_mooncake_lookup_passes_request_configs_to_cache_keys():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.store = _FakeStore()
    client.token_database = _FakeTokenDatabase()

    hit_tokens = client.lookup(
        [1, 2, 3],
        request_configs={"lmcache.tag.schema": "dsa-index-save-v2"},
    )

    assert hit_tokens == 3
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@1@schema%dsa-index-save-v2"
    ]


def test_mooncake_lookup_requires_dsa_index_group_before_hit():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=False)
    client.metadata = SimpleNamespace(kv_shape=(2, 1, 256, 1, 1))
    client.store = _FakeStore(rets=[1, 0])
    client.token_database = _FakeTokenDatabase(kv_group=0)

    hit_tokens = client.lookup([1, 2, 3])

    assert hit_tokens == 0
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@0",
        "model@1@0@abc@bfloat16@1",
    ]

    client.store = _FakeStore(rets=[1, 1])
    assert client.lookup([1, 2, 3]) == 3


def test_mooncake_lookup_layerwise_checks_all_layers_and_groups():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=True)
    client.metadata = SimpleNamespace(kv_shape=(2, 1, 256, 1, 1))
    client.store = _FakeStore(rets=[1, 1, 1, 0])
    client.token_database = _FakeTokenDatabase(kv_group=0)

    assert client.lookup([1, 2, 3]) == 0
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@0@0",
        "model@1@0@abc@bfloat16@0@1",
        "model@1@0@abc@bfloat16@1@0",
        "model@1@0@abc@bfloat16@1@1",
    ]

    client.store = _FakeStore(rets=[1, 1, 1, 1])
    assert client.lookup([1, 2, 3]) == 3


def test_mooncake_sampled_lookup_reverse_scans_first_and_last_layers():
    token_db = _FakeMultiChunkTokenDatabase(kv_group=0)
    first_keys = _sampled_string_keys(token_db, 0)
    winner_keys = _sampled_string_keys(token_db, 2)
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(
        dsa_two_groups=True,
        use_layerwise=True,
        experimental_sampled_layerwise_lookup=True,
    )
    client.metadata = SimpleNamespace(kv_shape=(4, 1, 256, 1, 1))
    client.store = _PresentStore([*first_keys, *winner_keys])
    client.token_database = token_db

    assert client.lookup(list(range(14))) == 12
    assert client.store.calls == [
        first_keys,
        _sampled_string_keys(token_db, 3),
        winner_keys,
    ]


def test_mooncake_sampled_lookup_retries_complete_zero_result():
    class _DelayedStore(_FakeStore):
        def batch_is_exist(self, keys):
            self.keys = keys
            self.calls.append(list(keys))
            return [1 for _ in keys] if len(self.calls) == 3 else [0 for _ in keys]

    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(
        dsa_two_groups=True,
        use_layerwise=True,
        experimental_sampled_layerwise_lookup=True,
        mooncake_lookup_retry_delays_ms=[0, 0],
    )
    client.metadata = SimpleNamespace(kv_shape=(2, 1, 256, 1, 1))
    client.store = _DelayedStore()
    client.token_database = _FakeTokenDatabase(kv_group=0)

    assert client.lookup([1, 2, 3], lookup_id="req") == 3
    assert len(client.store.calls) == 3


def test_mooncake_page_lookup_uses_pages_and_keeps_partial_tail_legacy():
    token_db = _FakeMultiChunkTokenDatabase(kv_group=0)
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(
        dsa_two_groups=True,
        use_layerwise=True,
        experimental_sampled_layerwise_lookup=False,
        chunk_size=4,
        extra_config={"mooncake_page_first_multi_buffer": True},
    )
    client.metadata = SimpleNamespace(kv_shape=(4, 1, 4, 1, 1))
    client.store = _FakeStore()
    client.token_database = token_db

    assert client.lookup(list(range(14))) == 14

    expected = []
    for chunk_index in range(3):
        for kv_group in (0, 1):
            key = token_db._make_key_by_hash(
                0x100 + chunk_index, kv_group=kv_group
            )
            expected.append(mooncake_page_key(key, 4))
    for kv_group in (0, 1):
        key = token_db._make_key_by_hash(0x103, kv_group=kv_group)
        expected.extend(layer.to_string() for layer in key.split_layers(4))
    assert client.store.keys == expected
