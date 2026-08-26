# SPDX-License-Identifier: Apache-2.0

# Standard
from types import ModuleType, SimpleNamespace
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.mooncake_layout import (
    MOONCAKE_VALID_TOKENS_TAG,
    mooncake_legacy_key,
    mooncake_page_key,
)


class _FakeStore:
    def __init__(self, rets=None):
        self.keys = None
        self.calls = []
        self.rets = rets

    def batch_is_exist(self, keys):
        self.keys = keys
        self.calls.append(list(keys))
        return self.rets if self.rets is not None else [1 for _ in keys]


def test_mooncake_lookup_closes_store_after_setup_failure(monkeypatch):
    observed = {}

    class FailingStore:
        def setup(self, *args):
            observed["protocol"] = args[4]
            return -1

        def close(self):
            observed["closed"] = True

    package = ModuleType("mooncake")
    package.__path__ = []  # type: ignore[attr-defined]
    store_module = ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = FailingStore
    package.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    with pytest.raises(RuntimeError, match="status=-1"):
        MooncakeLookupClient(SimpleNamespace(), SimpleNamespace(), "master")

    assert observed == {"protocol": "tcp", "closed": True}


class _FakeTokenDatabase:
    def __init__(self, kv_group=1):
        self.kv_group = kv_group

    def process_tokens(
        self,
        token_ids=None,
        hashes=None,
        offsets=None,
        request_configs=None,
        kv_group=None,
    ):
        if hashes is not None:
            start = 0
            for chunk_hash, offset in zip(hashes, offsets, strict=True):
                end = start + offset
                yield (
                    start,
                    end,
                    self._make_key_by_hash(
                        chunk_hash,
                        request_configs,
                        kv_group=self.kv_group if kv_group is None else kv_group,
                    ),
                )
                start = end
            return
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

    def __init__(self, kv_group=1, tag_partial=False):
        super().__init__(kv_group)
        self.tag_partial = tag_partial

    def _configs(self, request_configs, length):
        if not self.tag_partial or length == 4:
            return request_configs
        return {
            **(request_configs or {}),
            MOONCAKE_VALID_TOKENS_TAG: length,
        }

    def process_tokens(
        self,
        token_ids=None,
        hashes=None,
        offsets=None,
        request_configs=None,
        kv_group=None,
    ):
        if hashes is not None:
            start = 0
            for chunk_hash, offset in zip(hashes, offsets, strict=True):
                end = start + offset
                yield (
                    start,
                    end,
                    self._make_key_by_hash(
                        chunk_hash,
                        self._configs(request_configs, offset),
                        kv_group=self.kv_group if kv_group is None else kv_group,
                    ),
                )
                start = end
            return
        del token_ids
        start = 0
        for chunk_index, end in enumerate(self.chunk_ends):
            yield (
                start,
                end,
                self._make_key_by_hash(
                    0x100 + chunk_index,
                    self._configs(request_configs, end - start),
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


def test_mooncake_page_lookup_uses_pages_for_partial_tail():
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
    for chunk_index in range(4):
        for kv_group in (0, 1):
            key = token_db._make_key_by_hash(
                0x100 + chunk_index, kv_group=kv_group
            )
            expected.append(mooncake_page_key(key, 4))
    assert client.store.calls == [expected]


def test_mooncake_page_lookup_expands_only_missing_legacy_page():
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
    pages = [
        mooncake_page_key(
            token_db._make_key_by_hash(0x100 + chunk, kv_group=group), 4
        )
        for chunk in range(4)
        for group in (0, 1)
    ]
    legacy_tail = [
        layer.to_string()
        for group in (0, 1)
        for layer in token_db._make_key_by_hash(
            0x103, kv_group=group
        ).split_layers(4)
    ]
    client.store = _PresentStore([*pages[:-2], *legacy_tail])
    client.token_database = token_db

    assert client.lookup(list(range(14))) == 14
    expanded = [
        key
        for call in client.store.calls
        for key in call
        if not key.startswith("__lmcache_page_v1__")
    ]
    assert expanded == [
        layer.to_string()
        for group in (0, 1)
        for layer in token_db._make_key_by_hash(
            0x103, kv_group=group
        ).split_layers(4)
    ]


def test_mooncake_partial_page_lookup_strips_tag_for_legacy_tail():
    token_db = _FakeMultiChunkTokenDatabase(kv_group=0, tag_partial=True)
    chunks = list(token_db.process_tokens(token_ids=list(range(14))))
    group1 = list(
        token_db.process_tokens(
            hashes=[key.chunk_hash for _, _, key in chunks],
            offsets=[end - start for start, end, _ in chunks],
            kv_group=1,
        )
    )
    tail_keys = [chunks[-1][2], group1[-1][2]]

    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(
        dsa_two_groups=True,
        use_layerwise=True,
        experimental_sampled_layerwise_lookup=False,
        chunk_size=4,
        extra_config={"mooncake_page_first_multi_buffer": True},
    )
    client.metadata = SimpleNamespace(kv_shape=(4, 1, 4, 1, 1))
    client.token_database = token_db
    full_pages = [
        mooncake_page_key(
            token_db._make_key_by_hash(0x100 + chunk, kv_group=group), 4
        )
        for chunk in range(3)
        for group in (0, 1)
    ]
    legacy_tail = [
        mooncake_legacy_key(layer)
        for key in tail_keys
        for layer in key.split_layers(4)
    ]
    client.store = _PresentStore([*full_pages, *legacy_tail])

    assert client.lookup(list(range(14))) == 14
    assert client.store.calls[0][-2:] == [
        mooncake_page_key(key, 4) for key in tail_keys
    ]
    assert client.store.calls[-1] == legacy_tail
