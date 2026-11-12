# SPDX-License-Identifier: Apache-2.0
"""CPU tests of file reads at the native store boundary, using real cache keys."""

import base64
import asyncio
import json
import os
import socket
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_page_key
from lmcache.v1.storage_backend.connector.file_replay import (
    FileReplayStore,
    TAG,
    read_timeout,
)
from lmcache.v1.token_database import ChunkedTokenDatabase


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4,
        use_layerwise=True,
        dsa_two_groups=True,
        file_replay_enabled=True,
        pd_role="sender",
        save_unfull_chunk=True,
        extra_config={
            "mooncake_page_first_multi_buffer": True,
            "save_chunk_meta": False,
            "mooncake_dsa_raw_token_dims": {0: 2, 1: 2},
        },
    )
    metadata = LMCacheMetadata(
        model_name="/model",
        world_size=2,
        local_world_size=2,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 1, 4, 1, 2),
        use_mla=True,
        runtime_kv_group_layer_counts=(2, 1),
        runtime_kv_group_layer_names=(("l0", "l1"), ("i0",)),
    )
    database = ChunkedTokenDatabase(config, metadata)
    state = NS(role="PD", truncate=False, requests=[], ack=[], readers=[], native=[])
    manifest = dict(
        role="PD",
        model="/model",
        signature=database.mooncake_payload_layout,
        groups=[2, 1],
        chunk_size=4,
        tokens=list(range(11)),
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            state.requests.append(self.path)
            if self.path.endswith("manifest"):
                data = json.dumps(dict(manifest, role=state.role)).encode()
            elif "/layer/" in self.path:
                group, layer, start, end = map(int, self.path.split("/")[-4:])
                data = bytes(
                    (i + group * 50 + start + layer * (end - start) * 4) % 256
                    for i in range((end - start) * 4)
                )
            else:
                group, start, end = map(int, self.path.split("/")[-3:])
                data = bytes(
                    (i + group * 50 + start) % 256
                    for i in range((end - start) * 4 * (2 - group))
                )
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:-1] if state.truncate else data)

        def do_POST(self):
            state.ack.append(self.path)
            state.readers.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(204)
            self.end_headers()

    class Native:
        def batch_is_exist(self, keys):
            state.native.extend(keys)
            return [0] * len(keys)

        def batch_get_into_multi_buffers(self, keys, *_):
            state.native.extend(keys)
            return [-1] * len(keys)

        def put(self, key, value):
            state.native.append((key, value))

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def keys(role="PD", tokens=11):
            tag = (
                base64.urlsafe_b64encode(
                    json.dumps(
                        dict(
                            role=role,
                            url=f"http://127.0.0.1:{server.server_port}/secret",
                        )
                    ).encode()
                )
                .decode()
                .rstrip("=")
            )
            return [
                (g, a, b, k, mooncake_page_key(k, 2 - g))
                for g in (0, 1)
                for a, b, k in database.process_tokens(
                    list(range(tokens)), request_configs={TAG: tag}, kv_group=g
                )
            ]

        try:
            yield NS(
                store=FileReplayStore(Native(), config, metadata),
                config=config,
                metadata=metadata,
                state=state,
                keys=keys,
                database=database,
            )
        finally:
            server.shutdown()
            thread.join()


def test_full_partial_and_multi_buffer_bytes_are_complete_before_ack(fixture):
    f = fixture
    records = f.keys()
    # Live config may exclude partial saves; the captured D tail must still be readable.
    f.config.save_unfull_chunk = False
    assert f.store.batch_is_exist([k for *_, k in records]) == [1] * 6
    assert f.config.save_unfull_chunk is False
    outputs = []
    for g, a, b, _, key in records:
        size = (b - a) * 4 * (2 - g)
        memory = torch.full((size + 16,), 255, dtype=torch.uint8)
        f.store.memory.bind_cpu(memory)
        half = size // 2
        result = f.store.batch_get_into_multi_buffers(
            [key],
            [[memory.data_ptr() + 8, memory.data_ptr() + 8 + half]],
            [[half, size - half]],
        )
        assert result == [size]
        assert memory[8 : 8 + size].tolist() == [
            (i + g * 50 + a) % 256 for i in range(size)
        ]
        assert memory[:8].tolist() == memory[-8:].tolist() == [255] * 8
        outputs.append(result)
    assert len(f.state.ack) == 6 and not f.state.native
    assert (
        f.state.readers
        == [dict(host=socket.gethostname(), pid=os.getpid(), role="P", rank=0)] * 6
    )
    # A P replay's history is also reachable by D; its newly computed tail is native.
    f.store.role = "D"
    assert f.store.batch_is_exist([r[-1] for r in records]) == [1] * 6
    later = next(
        k for g, a, b, _, k in f.keys(tokens=15) if g == 0 and a == 8 and b == 12
    )
    assert f.store.batch_is_exist([later]) == [0]
    f.store.put("ordinary", b"unchanged")
    assert f.state.native[-1] == ("ordinary", b"unchanged")


def test_failure_never_acknowledges_a_destination(fixture):
    f = fixture
    key = f.keys()[0][-1]
    memory = torch.full((32,), 255, dtype=torch.uint8)
    f.store.memory.bind_cpu(memory)
    with pytest.raises(ValueError, match="registered tensor owner"):
        f.store.batch_get_into([key], [memory.data_ptr() + 31], [32])
    assert not f.state.ack
    f.state.truncate = True
    with pytest.raises(OSError, match="before the page was complete"):
        f.store.batch_get_into([key], [memory.data_ptr()], [32])
    assert not f.state.ack


def test_role_and_deadline_isolation(fixture):
    f = fixture
    key = f.keys(role="D")[0]
    assert f.store.batch_is_exist([key[-1]]) == [0]
    assert not f.state.requests
    assert read_timeout(f.config, [key[-2]], 10) == 10
    f.config.pd_role = "receiver"
    assert read_timeout(f.config, [key[-2]], 10) == 7200
    f.config.file_replay_enabled = False
    assert read_timeout(f.config, [key[-2]], 10) == 10
    assert LMCacheEngineConfig.from_defaults().file_replay_enabled is False


def test_replay_database_does_not_reinitialize_process_hash(fixture, monkeypatch):
    import lmcache.v1.token_database as token_database

    f = fixture
    original = token_database.NONE_HASH
    # If replay invokes initialization it would overwrite the sentinel root.
    monkeypatch.setattr(token_database, "NONE_HASH", b"existing-live-root")
    reader = ChunkedTokenDatabase(f.config, f.metadata, initialize_hash=False)
    assert token_database.NONE_HASH == b"existing-live-root"
    assert reader.chunk_size == 4
    monkeypatch.setattr(token_database, "NONE_HASH", original)


def test_device_copy_finishes_before_acknowledgement(fixture, monkeypatch):
    f = fixture
    completed = []

    class Destination:
        device = NS(type="npu")

        def __getitem__(self, _):
            return self

        def copy_(self, source, non_blocking=False):
            assert not f.state.ack
            completed.append("copy submitted")

    def synchronize():
        assert completed == ["copy submitted"]
        assert not f.state.ack
        completed.append("device finished")

    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            device=lambda _: nullcontext(),
            current_stream=lambda: NS(synchronize=synchronize),
        ),
        raising=False,
    )
    monkeypatch.setattr(f.store.memory, "view", lambda *_: Destination())
    assert f.store.batch_get_into([f.keys()[0][-1]], [1], [32]) == [32]
    assert completed == ["copy submitted", "device finished"]
    assert len(f.state.ack) == 1


def test_file_switch_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("LMCACHE_FILE_REPLAY_ENABLED", "true")
    assert LMCacheEngineConfig.from_env().file_replay_enabled is True


def test_individual_layer_fallback_reads_same_snapshot(fixture):
    f = fixture
    key = f.keys()[0][-2].get_layer(1).to_string()
    owner = torch.zeros(16, dtype=torch.uint8)
    f.store.memory.bind_cpu(owner)
    assert f.store.batch_is_exist([key]) == [1]
    assert f.store.batch_get_into([key], [owner.data_ptr()], [16]) == [16]
    assert owner.tolist() == list(range(16, 32))
    assert f.state.ack[-1].endswith("/ack/0/0/4/1")
    assert not f.state.native


def test_external_connector_file_read_skips_native_registration(fixture, monkeypatch):
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakestoreConnector,
    )

    f = fixture
    connector = object.__new__(MooncakestoreConnector)
    connector.save_chunk_meta = False
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = NS(transfer_timeout=0.001)
    connector.local_cpu_backend = NS(metadata=NS(chunk_size=4))
    connector._metadata_for_raw_key = lambda _: (None, None, None, 4)
    connector._external_buffers = {}
    connector._external_put_lock = asyncio.Lock()
    connector._inflight_put_tasks = set()
    connector.store = connector._file_replay_store = f.store
    monkeypatch.setattr(
        connector,
        "_register_external_owners",
        lambda *_: pytest.fail("file read registered native memory"),
    )
    owner = torch.full((32,), 255, dtype=torch.uint8)
    asyncio.run(
        connector.batched_get_external_pages(
            [f.keys()[0][-2]], [[owner.data_ptr()]], [[32]], (owner,), "file-test"
        )
    )
    assert owner.tolist() == list(range(32))
    assert len(f.state.ack) == 1
