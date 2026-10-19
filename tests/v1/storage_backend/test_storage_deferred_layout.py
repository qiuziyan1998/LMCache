# SPDX-License-Identifier: Apache-2.0
"""Real transport constructors with a controlled native Mooncake store."""

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteExternalPageReader
from tests.v1.storage_backend.test_mooncake_store_completion import (
    _make_mooncake_connector,
)


def deferred_connector(monkeypatch):
    calls = []
    previous, loop = _make_mooncake_connector(
        monkeypatch,
        {
            "save_chunk_meta": False,
            "mooncake_page_first_multi_buffer": True,
            "mooncake_dsa_raw_token_dims": {0: 576, 1: 128},
        },
        calls,
        external_page_only=True,
    )
    config, metadata = previous._engine_config, previous._lmcache_metadata
    loop.run_until_complete(previous.close())
    calls.clear()
    metadata = replace(metadata, runtime_kv_group_layer_counts=None)
    connector = MooncakestoreConnector(
        "",
        0,
        "",
        loop,
        None,
        config,
        metadata,
        external_page_only=True,
        defer_layout=True,
    )
    return connector, loop, metadata, calls


def test_native_session_initializes_before_unknown_layout(monkeypatch):
    connector, loop, metadata, calls = deferred_connector(monkeypatch)
    try:
        assert calls[0][0] == "setup" and calls[0][1][2:4] == (0, 0)
        assert connector.meta_shapes == []
        with pytest.raises(RuntimeError, match="not bound"):
            connector.exists_sync(None)
        with pytest.raises(RuntimeError, match="not bound"):
            connector.batched_contains_layer_pages([])
        with pytest.raises(RuntimeError, match="not bound"):
            loop.run_until_complete(
                connector.batched_get_external_pages([], [], [], (), "request")
            )
        with pytest.raises(ValueError):
            connector.bind_layout(metadata)
        assert connector.meta_shapes == []
    finally:
        loop.run_until_complete(connector.close())
        loop.close()


def test_final_geometry_binds_without_second_native_setup(monkeypatch):
    connector, loop, metadata, calls = deferred_connector(monkeypatch)
    final = replace(
        metadata,
        runtime_kv_group_layer_counts=(3, 2),
        runtime_kv_group_layer_names=(("a", "b", "c"), ("a.indexer", "b.indexer")),
        indexer_c8_layout=IndexerC8Layout(128, (True, False)),
    )
    try:
        before = list(calls)
        connector.bind_layout(final)
        assert connector._page_group_layer_counts == (3, 2)
        assert connector._mixed_indexer_layout == final.indexer_c8_layout
        assert connector.meta_dtypes == final.get_dtypes()
        assert connector.full_chunk_size_bytes > 0
        assert calls == before
        connector.bind_layout(final)
        assert calls == before
        with pytest.raises(ValueError, match="already-bound"):
            connector.bind_layout(replace(final, runtime_kv_group_layer_counts=(4, 2)))
        with pytest.raises(ValueError, match="identity"):
            connector.bind_layout(replace(final, model_name="different-model"))
    finally:
        loop.run_until_complete(connector.close())
        loop.close()


def test_actual_reader_and_connector_factory_propagate_deferred_layout(monkeypatch):
    connector, loop, metadata, calls = deferred_connector(monkeypatch)
    loop.run_until_complete(connector.close())
    loop.close()
    reader = RemoteExternalPageReader(
        connector._engine_config, metadata, defer_layout=True,
    )
    try:
        before = list(calls)
        final = replace(metadata, runtime_kv_group_layer_counts=(3, 2))
        reader.bind_layout(final)
        assert calls == before
        reader.bind_layout(final)
        assert calls == before
    finally:
        reader.close()
