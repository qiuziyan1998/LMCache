# SPDX-License-Identifier: Apache-2.0
"""Opt-in snapshot file reads beneath the ordinary merged-page connector.

Only keys carrying this run's file_replay tag on the selected P/D role are
redirected. Other reads and all writes retain the native store implementation.
No snapshot is inserted into Mooncake by this adapter.
"""

# Standard
import base64
import copy
import json
import os
import socket
import threading
import urllib.request
from typing import Any

TAG = "lmcache.tag.file_replay"
COPY_BYTES = 4 * 1024 * 1024
REPLAY_TIMEOUT = 7200.0


def read_timeout(config: Any, keys: Any, default: float) -> float:
    """Allow file-backed replay reads two hours; leave other deadlines intact."""
    if not getattr(config, "file_replay_enabled", False):
        return default
    role = "D" if config.pd_role == "receiver" else "P"
    for key in keys:
        for name, tag in key.tags or ():
            if name == "file_replay" and role in decode_tag(tag)["role"]:
                return max(default, REPLAY_TIMEOUT)
    return default


def decode_tag(tag: str) -> dict:
    """Decode the bounded, explicit per-request source capability."""
    if not isinstance(tag, str) or len(tag) > 2048:
        raise ValueError("Invalid file replay tag")
    options = json.loads(base64.urlsafe_b64decode(tag + "=" * (-len(tag) % 4)))
    if options.get("role") not in ("PD", "D") or not isinstance(
        options.get("url"), str
    ):
        raise ValueError("Invalid file replay source")
    return options


class TensorDestinations:
    """Resolve registered tensor owners without casting arbitrary device pointers."""

    def __init__(self) -> None:
        self.cpu: tuple = ()
        self.external: tuple = ()

    def bind_cpu(self, owner: Any) -> None:
        self.cpu = (owner.untyped_storage(),)

    def bind_external(self, owners: tuple) -> None:
        self.external = tuple(owner.untyped_storage() for owner in owners)

    def view(self, pointer: int, size: int) -> Any:
        # Third Party
        import torch

        if (
            type(pointer) is not int
            or type(size) is not int
            or pointer <= 0
            or size <= 0
        ):
            raise ValueError("Invalid file replay destination")
        matches = {
            (str(s.device), s.data_ptr()): s
            for s in self.cpu + self.external
            if s.data_ptr() <= pointer and pointer + size <= s.data_ptr() + s.nbytes()
        }
        if len(matches) != 1:
            raise ValueError(
                "File replay destination has no unique registered tensor owner"
            )
        storage = next(iter(matches.values()))
        return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
            storage, pointer - storage.data_ptr(), (size,), (1,)
        )


class FileReplayStore:
    """Route an explicitly tagged replay to a bounded, read-only file service."""

    def __init__(self, native: Any, config: Any, metadata: Any) -> None:
        self.native = native
        self.config = config
        self.metadata = metadata
        self.role = "D" if config.pd_role == "receiver" else "P"
        self.memory = TensorDestinations()
        self.sources: dict = {}
        self.lock = threading.RLock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.reader = json.dumps(
            dict(
                host=socket.gethostname(),
                pid=os.getpid(),
                role=self.role,
                rank=int(metadata.worker_id),
            ),
            separators=(",", ":"),
        ).encode()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def source_entry(self, key: str) -> Any:
        tags = [
            p[len("file_replay%") :]
            for p in key.split("@")
            if p.startswith("file_replay%")
        ]
        if not tags:
            return None
        if len(tags) != 1 or len(tags[0]) > 2048:
            raise ValueError("Invalid file replay tag")
        tag = tags[0]
        options = decode_tag(tag)
        if self.role not in options["role"]:
            return None
        with self.lock:
            if tag not in self.sources:
                # First Party
                from lmcache.v1.mooncake_layout import mooncake_page_key
                from lmcache.v1.token_database import ChunkedTokenDatabase

                url = options["url"].rstrip("/")
                if not url.startswith("http://"):
                    raise ValueError(
                        "File replay requires an explicit HTTP file source"
                    )
                with self.opener.open(url + "/manifest", timeout=120) as response:
                    manifest = json.load(response)
                config = copy.copy(self.config)
                config.save_unfull_chunk = True
                database = ChunkedTokenDatabase(
                    config, self.metadata, initialize_hash=False
                )
                if (
                    manifest["role"] != options["role"]
                    or manifest["model"] != self.metadata.model_name
                    or manifest["signature"] != database.mooncake_payload_layout
                    or manifest["groups"]
                    != list(self.metadata.runtime_kv_group_layer_counts)
                    or manifest["chunk_size"] != self.config.chunk_size
                ):
                    raise ValueError(
                        "File replay manifest disagrees with the live cache layout"
                    )
                entries = {}
                for group, count in enumerate(manifest["groups"]):
                    for start, end, chunk_key in database.process_tokens(
                        manifest["tokens"],
                        request_configs={TAG: tag},
                        kv_group=group,
                    ):
                        for candidate in (chunk_key, chunk_key.with_new_worker_id(0)):
                            entries[mooncake_page_key(candidate, count)] = (
                                group,
                                start,
                                end,
                            )
                self.sources[tag] = (url, entries)
            url, entries = self.sources[tag]
            entry = entries.get(key)
            if entry is not None:
                return url, (*entry, None)
            if not key.startswith("__lmcache_page_v1__@"):
                # Allocation fallback can request individual layers instead of
                # a merged page. Resolve those against the same file frontier.
                from lmcache.utils import LayerCacheEngineKey, parse_cache_key
                from lmcache.v1.mooncake_layout import mooncake_page_key

                parsed = parse_cache_key(key)
                if isinstance(parsed, LayerCacheEngineKey):
                    counts = self.metadata.runtime_kv_group_layer_counts
                    if (
                        0 <= parsed.kv_group < len(counts)
                        and 0 <= parsed.layer_id < counts[parsed.kv_group]
                    ):
                        entry = entries.get(
                            mooncake_page_key(parsed, counts[parsed.kv_group])
                        )
                        if entry is not None:
                            return url, (*entry, parsed.layer_id)
            return None

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        values = [0] * len(keys)
        missing = []
        for i, key in enumerate(keys):
            if self.source_entry(key) is None:
                missing.append(i)
            else:
                values[i] = 1
        if missing:
            results = self.native.batch_is_exist([keys[i] for i in missing])
            if len(results) != len(missing):
                raise RuntimeError("Invalid native existence result count")
            for i, result in zip(missing, results, strict=True):
                values[i] = result
        return values

    def is_exist(self, key: str) -> int:
        return self.batch_is_exist([key])[0]

    def batch_get_into(self, keys: list, pointers: list, sizes: list) -> list:
        return self.batch_get_into_multi_buffers(
            keys, [[p] for p in pointers], [[s] for s in sizes]
        )

    def batch_get_into_multi_buffers(
        self, keys: list, pointers: list, sizes: list
    ) -> list:
        if not len(keys) == len(pointers) == len(sizes):
            raise ValueError("File replay buffer list lengths differ")
        results = [0] * len(keys)
        native_indices = []
        for i, key in enumerate(keys):
            entry = self.source_entry(key)
            if entry is None:
                native_indices.append(i)
                continue
            url, (group, start, end, layer) = entry
            if not pointers[i] or len(pointers[i]) != len(sizes[i]):
                raise ValueError("Invalid file replay multi-buffer destination")
            # Validate every owner before writing any bytes. The ordinary
            # connector retains these owners until this synchronous call ends.
            views = [
                self.memory.view(p, n)
                for p, n in zip(pointers[i], sizes[i], strict=True)
            ]
            expected = sum(sizes[i])
            path = (
                f"page/{group}/{start}/{end}"
                if layer is None
                else f"layer/{group}/{layer}/{start}/{end}"
            )
            with self.opener.open(f"{url}/{path}", timeout=120) as response:
                if int(response.headers.get("Content-Length", -1)) != expected:
                    raise ValueError(
                        "File replay payload length differs from destination layout"
                    )
                for view, size in zip(views, sizes[i], strict=True):
                    for offset in range(0, size, COPY_BYTES):
                        count = min(COPY_BYTES, size - offset)
                        data = response.read(count)
                        if len(data) != count:
                            raise OSError(
                                "Snapshot file transfer ended "
                                "before the page was complete"
                            )
                        # Third Party
                        import torch

                        source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                        destination = view[offset : offset + count]
                        if destination.device.type == "cpu":
                            destination.copy_(source)
                        elif destination.device.type == "npu":
                            # A completed file read must mean the destination
                            # is usable, including direct indexer H2D. Do not
                            # return after merely enqueueing a copy on this
                            # background thread's stream.
                            with torch.npu.device(destination.device):
                                destination.copy_(source, non_blocking=False)
                                torch.npu.current_stream().synchronize()
                        else:
                            raise ValueError(
                                "Unsupported file replay destination device"
                            )
                if response.read(1):
                    raise ValueError("File replay page contains trailing bytes")
            # Count completed destination copies, not merely bytes sent by the
            # file server. This also covers direct indexer copies into NPU HBM.
            ack = urllib.request.Request(
                f"{url}/ack/{group}/{start}/{end}"
                + ("" if layer is None else f"/{layer}"),
                data=self.reader,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.opener.open(ack, timeout=120) as response:
                if response.status != 204:
                    raise OSError("File replay acknowledgement failed")
            results[i] = expected
        if native_indices:
            got = self.native.batch_get_into_multi_buffers(
                [keys[i] for i in native_indices],
                [pointers[i] for i in native_indices],
                [sizes[i] for i in native_indices],
            )
            if len(got) != len(native_indices):
                raise RuntimeError("Invalid native read result count")
            for i, value in zip(native_indices, got, strict=True):
                results[i] = value
        return results
