# SPDX-License-Identifier: Apache-2.0
"""Spawned CPU participant for the cooperative write transport contract."""

from datetime import timedelta
import importlib.util
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sys
from types import SimpleNamespace

import msgspec
import torch.distributed as dist

from lmcache.integration.vllm.dcp import DcpWriteLayout
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import OperationIdentity, unpack_uints


def main():
    config = msgspec.msgpack.decode(Path(sys.argv[1]).read_bytes())
    rank = int(sys.argv[2])
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/dcp_write_group.py"
    )
    spec = importlib.util.spec_from_file_location("dcp_write_group_ipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    group = None
    if config["gloo"]:
        dist.init_process_group(
            "gloo",
            init_method=config["gloo"],
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=15),
        )
        group = module.DcpWriteGroup(
            SimpleNamespace(rank_in_group=rank, world_size=2, ranks=[0, 1]), 15
        )
    client = StorageClient(
        config["work"],
        config["lifecycle"],
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session=f"worker-{rank}",
        session_nonce=b"b" if rank else b"a",
    )
    slab = SharedMemory(name=config["reply"]["slab_name"])
    try:
        envelope = (
            config["reply"]
            if group is None
            else group.leader("reserve", ("request", 1), lambda: config["reply"])
        )
        identity = msgspec.convert(config["identity"], type=OperationIdentity)
        row = envelope["groups"][0]
        offsets = unpack_uints(row["page_offsets"], len(row["starts"]))
        geometry_ids = unpack_uints(row["geometry_ids"], len(row["starts"]), 32)
        layout = DcpWriteLayout(2, 4, (1, 2, 3))
        if group is not None:
            group.exchange("prepared", ("request", 1), None)
        for start, end, offset, geometry_id in zip(
            row["starts"], row["ends"], offsets, geometry_ids, strict=True
        ):
            geometry = row["geometries"][geometry_id]
            for layer, layer_start in enumerate(geometry["layer_starts"]):
                capacity = geometry["token_capacity"]
                for token, _, count in layout.runs(start, end, rank):
                    for width, prefix in ((1024, 0), (128, 1024)):
                        begin = (
                            offset
                            + layer_start
                            + capacity * prefix
                            + (token - start) * width
                        )
                        slab.buf[begin : begin + count * width] = bytes(
                            [rank + layer * 10 + 1]
                        ) * (count * width)
        if group is not None:
            group.exchange("fenced", ("request", 1), True)
        reply = client.retire(
            identity, config["digest"], row["lease_id"], "fenced"
        ).result(10)
        assert reply.status == "RETIRED"
        if group is not None:
            group.exchange("retired", ("request", 1), True)
    finally:
        slab.close()
        client.close()
        if group is not None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
