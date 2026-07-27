#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose a zero-token sampled LMCache lookup against Mooncake.

The lookup path uses LMCache's configuration, token database, cache keys,
sampled lookup algorithm, StorageManager, RemoteBackend, and Mooncake
connector. A transparent proxy records raw ``batch_is_exist`` results so a
Mooncake error (-1), a real miss (0), a short response, and an exception are
not collapsed into the same zero-token result.

Examples:
  PYTHONHASHSEED=0 python3 tools/diagnose_mooncake_lookup.py lookup \
    --config /workspace/qzy/lmcache_config.yaml \
    --model /workspace/models/GLM-5.1-w4a8 \
    --prompt-file prompt.txt --num-layers 78 --world-size 8

  PYTHONHASHSEED=0 python3 tools/diagnose_mooncake_lookup.py roundtrip \
    --config /workspace/qzy/lmcache_config.yaml \
    --model /workspace/models/GLM-5.1-w4a8 \
    --num-tokens 2048 --num-layers 78 --world-size 8
"""

from __future__ import annotations

# Standard
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# Third Party
import torch

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

# First Party
try:
    import lmcache_ascend  # noqa: F401
except ImportError:
    pass

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.sampled_lookup import (
    find_last_sampled_hit,
    first_last_layer_keys,
)
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.token_database import ChunkedTokenDatabase


@dataclass(frozen=True)
class RunSpec:
    config: str
    model: str
    token_ids: list[int]
    num_layers: int
    world_size: int
    worker_id: int
    kv_dtype: str
    request_configs: dict[str, Any] | None
    local_cpu_gb: float
    put_batch_size: int
    put_payload_bytes: int
    store_all_layers: bool
    operation_timeout: float


@dataclass
class ExistsCall:
    keys: list[str]
    elapsed_ms: float
    statuses: list[int] | None = None
    error: str | None = None


class StatusTracingStore:
    """Forward Mooncake calls while recording raw existence-query results."""

    def __init__(self, store: Any):
        self.store = store
        self.calls: list[ExistsCall] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        """Run and record a native Mooncake existence query.

        Args:
            keys: Serialized cache keys to query.

        Returns:
            Mooncake's unmodified per-key status values.

        Raises:
            Exception: Any exception raised by the wrapped Mooncake store.
        """
        started = time.perf_counter()
        try:
            statuses = list(self.store.batch_is_exist(keys))
        except Exception as exc:
            self.calls.append(
                ExistsCall(
                    list(keys),
                    (time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        self.calls.append(
            ExistsCall(
                list(keys),
                (time.perf_counter() - started) * 1000,
                statuses=statuses,
            )
        )
        return statuses


def _parse_json_dict(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    loaded = json.loads(path.read_text() if path.is_file() else value)
    if not isinstance(loaded, dict):
        raise ValueError("--request-configs must be a JSON object or file")
    return loaded


def _parse_delays(value: str) -> list[float]:
    delays = [float(item) for item in value.split(",") if item.strip()]
    if not delays or any(delay < 0 for delay in delays):
        raise ValueError("--retry-delays must contain non-negative seconds")
    return delays


def _load_token_ids(args: argparse.Namespace) -> list[int]:
    if args.token_ids_file:
        loaded = json.loads(Path(args.token_ids_file).read_text())
        if isinstance(loaded, dict):
            loaded = loaded.get("token_ids")
        if not isinstance(loaded, list) or not all(
            isinstance(token, int) for token in loaded
        ):
            raise ValueError(
                "--token-ids-file must contain a JSON integer list or "
                '{"token_ids": [...]}'
            )
        return loaded

    if args.prompt_file:
        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        return tokenizer.encode(
            Path(args.prompt_file).read_text(encoding="utf-8"),
            add_special_tokens=args.add_special_tokens,
        )

    return [index % 32000 for index in range(args.num_tokens)]


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return dtype


def _load_config(spec: RunSpec) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_file(spec.config)
    if not config.remote_url or not config.remote_url.startswith(
        "mooncakestore://"
    ):
        raise ValueError("the LMCache config must use remote_url=mooncakestore://...")
    if not config.use_layerwise:
        raise ValueError("the diagnostic requires use_layerwise=true")
    if not config.experimental_sampled_layerwise_lookup:
        raise ValueError(
            "the diagnostic requires experimental_sampled_layerwise_lookup=true"
        )

    # The diagnostic payload is tiny. Restrict only its staging slab so two
    # independent clients do not reserve the deployment's full CPU cache.
    # This does not participate in cache-key generation or lookup semantics.
    config.local_cpu = True
    config.max_local_cpu_size = spec.local_cpu_gb
    config.lmcache_instance_id = f"mooncake-lookup-diagnostic-{uuid.uuid4().hex}"
    config.internal_api_server_enabled = False
    config.extra_config = dict(config.extra_config or {})
    config.extra_config["first_rank_max_local_cpu_size"] = spec.local_cpu_gb
    return config


def _metadata(spec: RunSpec, chunk_size: int) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name=spec.model,
        world_size=spec.world_size,
        local_world_size=spec.world_size,
        worker_id=spec.worker_id,
        local_worker_id=spec.worker_id,
        kv_dtype=_torch_dtype(spec.kv_dtype),
        kv_shape=(spec.num_layers, 1, chunk_size, 1, 1),
        use_mla=True,
        role="worker",
        chunk_size=chunk_size,
    )


def _chunk_group_keys(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    token_ids: list[int],
    request_configs: dict[str, Any] | None,
) -> list[tuple[int, list[CacheEngineKey]]]:
    token_database = ChunkedTokenDatabase(config, metadata)
    scheduler_chunks = list(
        token_database.process_tokens(token_ids, make_key=False)
    )
    hashes = [chunk_hash for _, _, chunk_hash in scheduler_chunks]
    offsets = [end - start for start, end, _ in scheduler_chunks]
    groups = (0, 1) if config.dsa_two_groups else (0,)
    by_group = [
        list(
            token_database.process_tokens(
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
                kv_group=kv_group,
            )
        )
        for kv_group in groups
    ]
    if not by_group or any(len(group) != len(by_group[0]) for group in by_group):
        raise RuntimeError("LMCache token databases produced inconsistent KV groups")

    chunks: list[tuple[int, list[CacheEngineKey]]] = []
    for chunk_index in range(len(by_group[0])):
        boundaries = {
            (group[chunk_index][0], group[chunk_index][1]) for group in by_group
        }
        if len(boundaries) != 1:
            raise RuntimeError(
                f"KV-group chunk boundaries differ at chunk {chunk_index}: "
                f"{sorted(boundaries)}"
            )
        _, end = next(iter(boundaries))
        keys = [group[chunk_index][2] for group in by_group]
        if not all(isinstance(key, CacheEngineKey) for key in keys):
            raise RuntimeError("LMCache token database did not produce cache keys")
        chunks.append((end, keys))
    return chunks


def _sampled_keys(
    chunk: tuple[int, list[CacheEngineKey]],
    num_layers: int,
) -> list[CacheEngineKey]:
    return first_last_layer_keys(chunk[1], num_layers)


def _all_layer_keys(
    chunk: tuple[int, list[CacheEngineKey]],
    num_layers: int,
) -> list[CacheEngineKey]:
    return [
        layer_key
        for group_key in chunk[1]
        for layer_key in group_key.split_layers(num_layers)
    ]


def _storage_manager(
    spec: RunSpec,
) -> tuple[StorageManager, RemoteBackend, list[tuple[int, list[CacheEngineKey]]]]:
    config = _load_config(spec)
    metadata = _metadata(spec, config.chunk_size)
    manager = StorageManager(config, metadata, EventManager())
    remote = next(
        (
            backend
            for name, backend in manager.get_active_storage_backends(
                search_range=["RemoteBackend"]
            )
            if name == "RemoteBackend" and isinstance(backend, RemoteBackend)
        ),
        None,
    )
    if remote is None or remote.connection is None:
        manager.close()
        raise RuntimeError("LMCache did not initialize its Mooncake RemoteBackend")
    chunks = _chunk_group_keys(
        config,
        metadata,
        spec.token_ids,
        spec.request_configs,
    )
    return manager, remote, chunks


def summarize_trace(calls: Sequence[ExistsCall]) -> dict[str, Any]:
    """Classify raw Mooncake lookup statuses without collapsing failures.

    Args:
        calls: Recorded native Mooncake existence queries.

    Returns:
        Status counts, failure details, raw calls, and the overall
        classification.
    """
    counts: dict[str, int] = {"found": 0, "missing": 0, "error": 0, "other": 0}
    short_responses = 0
    exceptions = 0
    first_failure: dict[str, Any] | None = None

    for call_index, call in enumerate(calls):
        if call.error is not None:
            exceptions += 1
            if first_failure is None:
                first_failure = {
                    "call": call_index,
                    "error": call.error,
                    "key": call.keys[0] if call.keys else None,
                }
            continue
        statuses = call.statuses or []
        if len(statuses) != len(call.keys):
            short_responses += 1
        for key_index, key in enumerate(call.keys):
            status = statuses[key_index] if key_index < len(statuses) else None
            label = (
                "found"
                if status == 1
                else "missing"
                if status == 0
                else "error"
                if status == -1
                else "other"
            )
            counts[label] += 1
            if label != "found" and first_failure is None:
                first_failure = {
                    "call": call_index,
                    "key_index": key_index,
                    "key": key,
                    "status": status,
                }

    if exceptions or short_responses or counts["error"] or counts["other"]:
        classification = "lookup_error"
    elif counts["missing"]:
        classification = "cache_miss"
    elif counts["found"]:
        classification = "all_queried_keys_found"
    else:
        classification = "no_mooncake_query"

    return {
        "classification": classification,
        "status_counts": counts,
        "short_responses": short_responses,
        "exceptions": exceptions,
        "first_failure": first_failure,
        "calls": [asdict(call) for call in calls],
    }


def _lookup_once(
    manager: StorageManager,
    chunks: list[tuple[int, list[CacheEngineKey]]],
    num_layers: int,
) -> int:
    def exists_at(index: int) -> bool:
        keys = _sampled_keys(chunks[index], num_layers)
        hits, _ = manager.batched_contains(keys, ["RemoteBackend"], False)
        return hits == len(keys)

    winner = find_last_sampled_hit(len(chunks), exists_at)
    return 0 if winner is None else chunks[winner][0]


def _run_lookup(spec: RunSpec, delays: Sequence[float]) -> dict[str, Any]:
    manager, remote, chunks = _storage_manager(spec)
    connection = remote.connection
    assert connection is not None
    connector = (
        connection.getWrappedConnector()
        if isinstance(connection, InstrumentedRemoteConnector)
        else connection
    )
    mooncake_store = getattr(connector, "store", None)
    if mooncake_store is None:
        manager.close()
        raise RuntimeError("active RemoteBackend is not a Mooncake connector")
    trace = StatusTracingStore(mooncake_store)
    connector.store = trace
    attempts: list[dict[str, Any]] = []
    try:
        for delay in delays:
            if delay:
                time.sleep(delay)
            trace.calls.clear()
            started = time.perf_counter()
            hit_tokens = _lookup_once(manager, chunks, spec.num_layers)
            attempts.append(
                {
                    "delay_s": delay,
                    "hit_tokens": hit_tokens,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    **summarize_trace(trace.calls),
                }
            )
            if hit_tokens > 0:
                break
    finally:
        manager.close()

    return {
        "token_count": len(spec.token_ids),
        "chunk_count": len(chunks),
        "expected_tokens": chunks[-1][0] if chunks else 0,
        "first_sampled_keys": (
            [key.to_string() for key in _sampled_keys(chunks[0], spec.num_layers)]
            if chunks
            else []
        ),
        "last_sampled_keys": (
            [key.to_string() for key in _sampled_keys(chunks[-1], spec.num_layers)]
            if chunks
            else []
        ),
        "attempts": attempts,
    }


def _producer(
    spec: RunSpec,
    ready_queue: mp.Queue,
    release_queue: mp.Queue,
) -> None:
    manager: StorageManager | None = None
    try:
        manager, _, chunks = _storage_manager(spec)
        keys = [
            key
            for chunk in chunks
            for key in (
                _all_layer_keys(chunk, spec.num_layers)
                if spec.store_all_layers
                else _sampled_keys(chunk, spec.num_layers)
            )
        ]
        started = time.perf_counter()
        stored = 0
        for offset in range(0, len(keys), spec.put_batch_size):
            key_batch = keys[offset : offset + spec.put_batch_size]
            objects = []
            for index, _ in enumerate(key_batch):
                memory_obj = manager.allocate(
                    torch.Size([spec.put_payload_bytes]),
                    torch.uint8,
                    fmt=MemoryFormat.BINARY,
                    eviction=False,
                    busy_loop=False,
                )
                if memory_obj is None or memory_obj.raw_tensor is None:
                    raise RuntimeError("LMCache failed to allocate diagnostic payload")
                memory_obj.raw_tensor.fill_((offset + index) % 251)
                objects.append(memory_obj)
            futures = manager.batched_put(
                key_batch,
                objects,
                location="RemoteBackend",
            )
            for future in futures:
                future.result(timeout=spec.operation_timeout)
            stored += len(key_batch)
        ready_queue.put(
            {
                "ok": True,
                "stored_keys": stored,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "scope": (
                    "all_layers" if spec.store_all_layers else "sampled_layers"
                ),
            }
        )
        release_queue.get(timeout=spec.operation_timeout)
    except Exception as exc:
        ready_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if manager is not None:
            manager.close()


def _roundtrip(
    spec: RunSpec,
    delays: Sequence[float],
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    release_queue = context.Queue()
    producer = context.Process(
        target=_producer,
        args=(spec, ready_queue, release_queue),
        name="lmcache-mooncake-producer",
    )
    producer.start()
    try:
        try:
            producer_result = ready_queue.get(timeout=spec.operation_timeout)
        except queue.Empty as exc:
            raise TimeoutError("producer did not finish its LMCache store") from exc
        if not producer_result.get("ok"):
            raise RuntimeError(
                "producer failed: "
                f"{producer_result.get('error')}\n"
                f"{producer_result.get('traceback', '')}"
            )
        lookup_result = _run_lookup(spec, delays)
        return {"producer": producer_result, "lookup": lookup_result}
    finally:
        release_queue.put(True)
        producer.join(timeout=10)
        if producer.is_alive():
            producer.terminate()
            producer.join(timeout=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mode", choices=("lookup", "roundtrip"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument("--token-ids-file")
    token_group.add_argument("--prompt-file")
    token_group.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--num-layers", type=int, default=78)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--kv-dtype", default="bfloat16")
    parser.add_argument(
        "--request-configs",
        help="JSON object or path to JSON containing lmcache.tag.* values",
    )
    parser.add_argument(
        "--local-cpu-gb",
        type=float,
        default=0.25,
        help="diagnostic staging slab; does not affect cache keys",
    )
    parser.add_argument(
        "--retry-delays",
        default="0,0.01,0.05,0.2,1",
        help="per-attempt sleep durations in seconds",
    )
    parser.add_argument("--put-batch-size", type=int, default=64)
    parser.add_argument("--put-payload-bytes", type=int, default=1)
    parser.add_argument(
        "--sampled-only-store",
        action="store_false",
        dest="store_all_layers",
        help="roundtrip stores only lookup probes instead of all layer keys",
    )
    parser.set_defaults(store_all_layers=True)
    parser.add_argument("--operation-timeout", type=float, default=120)
    parser.add_argument("--min-hit-tokens", type=int)
    parser.add_argument("--output-json")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "num_layers",
        "world_size",
        "num_tokens",
        "put_batch_size",
        "put_payload_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.worker_id < args.world_size:
        parser.error("--worker-id must be in [0, --world-size)")
    if args.local_cpu_gb <= 0 or args.operation_timeout <= 0:
        parser.error("--local-cpu-gb and --operation-timeout must be positive")
    if os.environ.get("PYTHONHASHSEED") is None:
        parser.error(
            "PYTHONHASHSEED must be set before Python starts; use PYTHONHASHSEED=0"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Mooncake lookup diagnostic.

    Args:
        argv: Optional command-line arguments. The process arguments are used
            when omitted.

    Returns:
        Zero for a sufficient hit, two for an insufficient hit, or one for a
        fatal diagnostic failure.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    token_ids = _load_token_ids(args)
    if not token_ids:
        parser.error("the token input is empty")
    spec = RunSpec(
        config=str(Path(args.config).resolve()),
        model=args.model,
        token_ids=token_ids,
        num_layers=args.num_layers,
        world_size=args.world_size,
        worker_id=args.worker_id,
        kv_dtype=args.kv_dtype,
        request_configs=_parse_json_dict(args.request_configs),
        local_cpu_gb=args.local_cpu_gb,
        put_batch_size=args.put_batch_size,
        put_payload_bytes=args.put_payload_bytes,
        store_all_layers=args.store_all_layers,
        operation_timeout=args.operation_timeout,
    )
    delays = _parse_delays(args.retry_delays)

    try:
        result = (
            _run_lookup(spec, delays)
            if args.mode == "lookup"
            else _roundtrip(spec, delays)
        )
    except Exception as exc:
        result = {
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    else:
        lookup = result if args.mode == "lookup" else result["lookup"]
        final_hits = (
            lookup["attempts"][-1]["hit_tokens"] if lookup["attempts"] else 0
        )
        minimum = (
            args.min_hit_tokens
            if args.min_hit_tokens is not None
            else lookup["expected_tokens"]
        )
        result["verdict"] = {
            "minimum_hit_tokens": minimum,
            "actual_hit_tokens": final_hits,
            "passed": final_hits >= minimum,
        }
        exit_code = 0 if final_hits >= minimum else 2

    config_bytes = Path(spec.config).read_bytes()
    result["context"] = {
        "mode": args.mode,
        "config": spec.config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model": spec.model,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "world_size": spec.world_size,
        "worker_id": spec.worker_id,
        "num_layers": spec.num_layers,
        "kv_dtype": spec.kv_dtype,
        "local_cpu_gb": spec.local_cpu_gb,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
