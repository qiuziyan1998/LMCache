# SPDX-License-Identifier: Apache-2.0
"""Real Gloo processes and storage sockets sharing actual service destinations."""

from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace as NS
import os
import socket
import subprocess
import sys

import msgspec
import pytest
import torch

from lmcache.v1.memory_management import TensorMemoryAllocator
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import unpack_uints
from lmcache.v1.storage_service import StorageService
from test_ascend_cpu_storage_runtime import setup as runtime_setup
from test_cpu_cache_operations import CPUAllocator


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


@pytest.mark.parametrize("gloo", [False, True])
def test_real_cooperative_descriptors_participants_and_canonical_shared_bytes(
    tmp_path, monkeypatch, gloo
):
    if gloo and os.name == "nt":
        pytest.skip("Windows host Torch raises unsupported gloo device; run on Linux")
    slab = SharedMemory(create=True, size=65536)

    def initialize(allocator):
        allocator.buffer = torch.frombuffer(slab.buf, dtype=torch.uint8)
        allocator.buffer.zero_()
        allocator.pin_allocator = TensorMemoryAllocator(allocator.buffer)
        allocator.shm_name = slab.name

    monkeypatch.setattr(CPUAllocator, "__init__", initialize)
    fixture = runtime_setup.__wrapped__()
    owner, runtime, _ = next(fixture)
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        NS(
            handle=runtime.handle,
            handle_validated=runtime.handle_validated,
            set_transport_notifier=runtime.set_transport_notifier,
            drain_terminal_notifications=runtime.drain_terminal_notifications,
            is_healthy=runtime.is_healthy,
            close=runtime.check_close_ready,
        ),
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=runtime.registry.sessions,
    )
    service.start()
    client = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=b"a",
    )
    children = []
    try:
        original = client.request(
            "reserve_write",
            {
                "purpose": "store",
                "participants": ("worker-0", "worker-1"),
                "groups": (
                    {
                        "group_id": 0,
                        "representation": "page",
                        "starts": (0, 4, 8),
                        "ends": (4, 8, 9),
                        "hashes": (1, 2, 3),
                    },
                ),
            },
            request_id="request",
            request_attempt="attempt",
            load_generation=1,
            purpose="store",
        )
        reserved = original.result(10)
        assert reserved.status == "COMPLETED"
        config = tmp_path / "peers.msgpack"
        config.write_bytes(
            msgspec.msgpack.encode(
                {
                    "reply": reserved.value,
                    "identity": original.identity,
                    "digest": original.payload_digest,
                    "work": work,
                    "lifecycle": lifecycle,
                    "gloo": endpoint() if gloo else None,
                }
            )
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            PYTHONUTF8="1",
            USE_LIBUV="0",
        )
        for rank in range(2):
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("dcp_write_ipc_peer.py")),
                        str(config),
                        str(rank),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            )
        for child in children:
            stdout, stderr = child.communicate(timeout=40)
            assert child.returncode == 0, stdout + stderr
        # Participant retirement alone cannot release the service producer hold.
        status = client.operation_status(original.identity).result(10)
        lease = next(iter(status.value["leases"].values()))
        assert set(lease["retired"]) == {"worker-0", "worker-1"}
        assert lease["producer_holds"] > 0
        row = reserved.value["groups"][0]
        offsets = unpack_uints(row["page_offsets"], 3)
        geometry_ids = unpack_uints(row["geometry_ids"], 3, 32)
        for start, end, offset, index in zip(
            row["starts"], row["ends"], offsets, geometry_ids, strict=True
        ):
            geometry = row["geometries"][index]
            for layer, position in enumerate(geometry["layer_starts"]):
                for width, prefix in ((1024, 0), (128, 1024)):
                    begin = offset + position + geometry["token_capacity"] * prefix
                    assert bytes(
                        slab.buf[begin : begin + (end - start) * width]
                    ) == bytes([(start // 4) % 2 + layer * 10 + 1]) * (
                        (end - start) * width
                    )
        cancelled = client.request(
            "cancel_write",
            {"reservation_id": reserved.value["reservation_id"], "proof": "fenced"},
            request_id="request",
            request_attempt="attempt",
            load_generation=1,
            purpose="store",
        )
        assert cancelled.result(10).status == "COMPLETED"
        assert (
            client.acknowledge_terminal(cancelled.identity).result(10).status
            == "ACKNOWLEDGED"
        )
        assert (
            client.acknowledge_terminal(original.identity).result(10).status
            == "ACKNOWLEDGED"
        )
        assert runtime.registry.snapshot()["leases"] == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=10)
        # Test-only cleanup: all CPU writer processes have exited; there are no
        # native/device operations in this harness. Preserve production's refusal
        # to close a service while participant/producer holds are unresolved.
        if runtime.registry.snapshot()["leases"]:
            for rank in range(2):
                peer = (
                    client
                    if rank == 0
                    else StorageClient(
                        work,
                        lifecycle,
                        cache_instance_id="cache",
                        service_epoch="epoch",
                        client_session="worker-1",
                        session_nonce=b"b",
                    )
                )
                try:
                    for row in reserved.value["groups"]:
                        peer.retire(
                            original.identity,
                            original.payload_digest,
                            row["lease_id"],
                            "fenced",
                        ).result(10)
                finally:
                    if rank:
                        peer.close()
            pending = client.request(
                "cancel_write",
                {"reservation_id": reserved.value["reservation_id"], "proof": "fenced"},
                request_id="request",
                request_attempt="attempt",
                load_generation=1,
                purpose="store",
            )
            pending.result(10)
            client.acknowledge_terminal(pending.identity).result(10)
            client.acknowledge_terminal(original.identity).result(10)
        service.close()
        client.close()
        # Advance through the existing fixture's post-yield allocator/service
        # cleanup. generator.close() would skip that ordinary teardown body.
        with pytest.raises(StopIteration):
            next(fixture)
        slab.close()
        slab.unlink()
