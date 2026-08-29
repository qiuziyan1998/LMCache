# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import json
import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generator, Optional, Union

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    calculate_draft_layers,
    extract_mm_features,
    is_false,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.decode_window_commit import (
    publish_delayed_decode_window_commit,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import (
    _SHARED_SPARSE_DEFER_COMMIT,
    _SHARED_SPARSE_PREPARE_ONLY,
    LayerwiseStoreResult,
    LMCacheEngine,
)
from lmcache.v1.cold_start_perf import (
    cold_start_perf_enabled,
    cold_start_perf_log,
    cold_start_perf_now,
)
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.gpu_connector.sparse import (
    PreparedSparseSource,
    build_prepared_sparse_source,
)
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)

SPARSE_DECODE_RETRIEVE_TOKENS = int(
    os.environ.get("LMCACHE_SPARSE_DECODE_RETRIEVE_TOKENS", "2048")
)
SPARSE_DECODE_SHARED_CPU_PHASE = "sparse_decode_bootstrap"
INDEXER_RETRIEVE_FULL = "full_materialize"
INDEXER_RETRIEVE_METADATA_ONLY = "metadata_only"
INDEXER_RETRIEVE_RESIDENT_SKIP = "resident_skip"
DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS_ENV = (
    "VLLM_ASCEND_LMCACHE_DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS"
)
RETRIEVE_STATS_INTERVAL_SECONDS_ENV = (
    "VLLM_ASCEND_LMCACHE_RETRIEVE_STATS_INTERVAL_SECONDS"
)
LayerwiseSaveKey = tuple[str, str, int, int, int]


def _has_live_group1_source_for_dp(
    params: Any,
    dp_rank: int,
    token_count: int,
    tp_size: int,
) -> bool:
    """Require a complete, rank-addressable Group-1 source offer.

    Capabilities describe what the peer can do; they do not prove that the
    current request actually published source memory.  In particular, a
    prefix-hit producer may legitimately finish without a live descriptor and
    rely on persistent storage.  Treating that request as live allocates a
    destination which Mooncake must later cancel and mixes the live and
    persistent load lifecycles.
    """
    if (
        not isinstance(params, dict)
        or isinstance(dp_rank, bool)
        or not isinstance(dp_rank, int)
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
        or isinstance(tp_size, bool)
        or not isinstance(tp_size, int)
        or tp_size <= 0
    ):
        return False
    envelope = params.get("ascend_live_split_source_v1")
    if not isinstance(envelope, dict):
        return False
    descriptors = envelope.get("descriptors", ())
    if not isinstance(descriptors, (tuple, list)):
        return False
    source_tp_ranks: set[int] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        tp_value = descriptor.get("tp_rank", -1)
        dp_value = descriptor.get("dp_rank", -1)
        if (
            isinstance(tp_value, bool)
            or not isinstance(tp_value, int)
            or isinstance(dp_value, bool)
            or not isinstance(dp_value, int)
            or not 0 <= tp_value < tp_size
            or dp_value != dp_rank
        ):
            continue
        compact = descriptor.get("compact_layout")
        if not isinstance(compact, dict):
            continue
        group_value = compact.get("group_id", -1)
        compact_tokens = compact.get("token_count", 0)
        totals = descriptor.get("group_byte_totals")
        if (
            isinstance(group_value, bool)
            or not isinstance(group_value, int)
            or group_value != 1
            or isinstance(compact_tokens, bool)
            or not isinstance(compact_tokens, int)
            or compact_tokens != token_count
            or not bool(compact.get("layers"))
            or not bool(compact.get("runs"))
            or not isinstance(totals, (tuple, list))
            or len(totals) != 2
            or isinstance(totals[1], bool)
            or not isinstance(totals[1], int)
            or totals[1] <= 0
        ):
            continue
        source_tp_ranks.add(tp_value)
    return source_tp_ranks == set(range(tp_size))


def _has_live_latent_source_for_dp(
    params: Any,
    dp_rank: int,
    token_count: int,
) -> bool:
    """Require one complete TP0 hybrid source before allocating group 0."""
    if not isinstance(params, dict):
        return False
    envelope = params.get("ascend_live_split_source_v1")
    if not isinstance(envelope, dict):
        return False
    descriptors = envelope.get("descriptors", ())
    if not isinstance(descriptors, (tuple, list)):
        return False
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        try:
            tp_value = descriptor.get("tp_rank", -1)
            dp_value = descriptor.get("dp_rank", -1)
            if (
                isinstance(tp_value, bool)
                or not isinstance(tp_value, int)
                or isinstance(dp_value, bool)
                or not isinstance(dp_value, int)
            ):
                continue
            if (
                tp_value != 0
                or dp_value != dp_rank
            ):
                continue
            latent = descriptor.get("latent_layout")
            compact = descriptor.get("compact_layout")
            group_value = (
                latent.get("group_id", -1)
                if isinstance(latent, dict)
                else None
            )
            token_value = (
                latent.get("token_count", 0)
                if isinstance(latent, dict)
                else None
            )
            compact_group = (
                compact.get("group_id", -1)
                if isinstance(compact, dict)
                else None
            )
            compact_tokens = (
                compact.get("token_count", 0)
                if isinstance(compact, dict)
                else None
            )
            totals = descriptor.get("group_byte_totals")
            latent_total = descriptor.get("latent_group_byte_total")
            legacy_hybrid = (
                descriptor.get("format") == "hybrid_compact_v1"
                and isinstance(totals, (tuple, list))
                and len(totals) == 2
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and value > 0
                    for value in totals
                )
                and latent_total is None
            )
            extension_hybrid = (
                isinstance(totals, (tuple, list))
                and len(totals) == 2
                and not isinstance(totals[0], bool)
                and isinstance(totals[0], int)
                and totals[0] == 0
                and not isinstance(totals[1], bool)
                and isinstance(totals[1], int)
                and totals[1] > 0
                and not isinstance(latent_total, bool)
                and isinstance(latent_total, int)
                and latent_total > 0
            )
            if (
                isinstance(latent, dict)
                and not isinstance(group_value, bool)
                and isinstance(group_value, int)
                and group_value == 0
                and not isinstance(token_value, bool)
                and isinstance(token_value, int)
                and token_value == token_count
                and bool(latent.get("layers"))
                and bool(latent.get("pages"))
                and isinstance(compact, dict)
                and not isinstance(compact_group, bool)
                and isinstance(compact_group, int)
                and compact_group == 1
                and not isinstance(compact_tokens, bool)
                and isinstance(compact_tokens, int)
                and compact_tokens == token_count
                and bool(compact.get("layers"))
                and bool(compact.get("runs"))
                and (extension_hybrid or legacy_hybrid)
            ):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _live_split_source_dp_rank(
    params: Any,
    parallel_config: Any,
) -> int | None:
    """Validate the source DP identity used by the live-split protocol.

    DP1 accepts an older peer that omits the identity. DP>1 requires the
    explicit routing capability and an in-range, non-boolean global DP rank.
    """
    if not isinstance(params, dict):
        return None
    try:
        dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    except (TypeError, ValueError):
        return None
    if dp_size <= 0:
        return None
    raw_rank = params.get("remote_dp_rank")
    if raw_rank is None:
        return 0 if dp_size == 1 else None
    if (
        isinstance(raw_rank, bool)
        or not isinstance(raw_rank, int)
        or raw_rank < 0
        or raw_rank >= dp_size
    ):
        return None
    capabilities = params.get("live_split_capabilities", ())
    if dp_size > 1 and (
        not isinstance(capabilities, (tuple, list, set, frozenset))
        or not all(isinstance(item, str) for item in capabilities)
        or "ascend_live_split_dp_routing_v1" not in capabilities
    ):
        return None
    return raw_rank


def _mtp_dw_diag_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_MTP_DW_DIAG", "0") == "1"


def _mtp_dw_deep_diag_enabled() -> bool:
    return _mtp_dw_diag_enabled() and (
        os.environ.get("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "0") == "1"
    )


def _mtp_dw_event(stage: str, **fields: Any) -> None:
    if not _mtp_dw_diag_enabled():
        return
    payload = {"schema": 1, "stage": stage, "owner": "lmcache"}
    payload.update(fields)
    logger.info("[MTP_DW] %s", json.dumps(payload, separators=(",", ":")))


def _retrieve_stats_interval_seconds() -> int:
    raw_value = os.environ.get(RETRIEVE_STATS_INTERVAL_SECONDS_ENV, "0")
    try:
        interval = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected a non-negative integer.",
            RETRIEVE_STATS_INTERVAL_SECONDS_ENV,
            raw_value,
        )
        return 0
    if interval < 0:
        logger.warning(
            "Ignoring invalid %s=%d; expected a non-negative integer.",
            RETRIEVE_STATS_INTERVAL_SECONDS_ENV,
            interval,
        )
        return 0
    return interval


def _dsa_debug_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dsa_debug_summary_enabled() -> bool:
    return os.environ.get(
        "VLLM_ASCEND_DSA_SHRINK_DEBUG_MODE", "fail_only"
    ).lower() in ("summary", "trace", "verbose", "all")


def _dsa_debug_limit() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG_LIMIT", "8")))
    except ValueError:
        return 8


def _dsa_debug_should_log(owner: Any, site: str) -> bool:
    if not _dsa_debug_enabled():
        return False
    if not _dsa_debug_summary_enabled():
        return False
    counts = getattr(owner, "_dsa_shrink_debug_counts", None)
    if counts is None:
        counts = {}
        owner._dsa_shrink_debug_counts = counts
    count = counts.get(site, 0)
    if count >= _dsa_debug_limit():
        return False
    counts[site] = count + 1
    return True


def _dsa_debug_shape(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    try:
        return len(value)
    except TypeError:
        return type(value).__name__


def _dsa_debug_sample(value: Any, limit: Optional[int] = None) -> Any:
    if value is None:
        return None
    limit = _dsa_debug_limit() if limit is None else limit
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            return value.detach().reshape(-1)[:limit].to(device="cpu").tolist()
        return list(value[:limit])
    except Exception as exc:
        return f"{type(value).__name__}:sample_failed:{exc}"


def _dsa_debug_minmax_count(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            flat = value.detach().reshape(-1)
            return (
                flat.min().to(device="cpu").item(),
                flat.max().to(device="cpu").item(),
                int(flat.numel()),
            )
        seq = list(value)
        if not seq:
            return None
        return (min(seq), max(seq), len(seq))
    except Exception as exc:
        return f"{type(value).__name__}:minmax_failed:{exc}"

def _sparse_slot_mapping_len(prompt_tokens: int) -> int:
    return min(SPARSE_DECODE_RETRIEVE_TOKENS, prompt_tokens)


def _build_slot_mapping(
    block_ids: list[int], block_size: int, num_tokens: int
) -> torch.Tensor:
    if num_tokens <= 0:
        return torch.empty(0, dtype=torch.long)
    num_blocks = utils.cdiv(num_tokens, block_size)
    block_ids_t = torch.tensor(block_ids[:num_blocks], dtype=torch.long)
    block_offsets = torch.arange(0, block_size, dtype=torch.long)
    slots = (
        block_offsets.reshape((1, block_size))
        + block_ids_t.reshape((num_blocks, 1)) * block_size
    ).flatten()
    return slots[:num_tokens]


def _build_slot_mapping_window(
    block_ids: list[int],
    block_size: int,
    token_start: int,
    token_end: int,
) -> torch.Tensor:
    """Build slots for one absolute token window without expanding the prefix."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if token_start < 0 or token_end < token_start:
        raise ValueError(
            "Invalid slot-mapping token window: "
            f"start={token_start}, end={token_end}"
        )
    if token_end == token_start:
        return torch.empty(0, dtype=torch.long)

    first_block = token_start // block_size
    end_block = utils.cdiv(token_end, block_size)
    if end_block > len(block_ids):
        raise ValueError(
            "Slot-mapping token window exceeds allocated blocks: "
            f"start={token_start}, end={token_end}, block_size={block_size}, "
            f"required_blocks={end_block}, allocated_blocks={len(block_ids)}"
        )

    selected_block_ids = torch.tensor(
        block_ids[first_block:end_block], dtype=torch.long
    )
    block_offsets = torch.arange(block_size, dtype=torch.long)
    slots = (
        block_offsets.reshape((1, block_size))
        + selected_block_ids.reshape((-1, 1)) * block_size
    ).flatten()
    local_start = token_start - first_block * block_size
    return slots[local_start : local_start + token_end - token_start]


def _dsa_payload_event_list(payload_event: Any) -> list[Any]:
    if payload_event is None:
        return []
    if isinstance(payload_event, (list, tuple)):
        return [event for event in payload_event if event is not None]
    return [payload_event]


def _dsa_wait_payload_event(payload_event: Any) -> None:
    payload_events = _dsa_payload_event_list(payload_event)
    if not payload_events:
        return
    if not (hasattr(torch, "npu") and hasattr(torch.npu, "current_stream")):
        raise RuntimeError(
            "DSA selected-token payload event was provided, but torch.npu "
            "stream support is unavailable."
        )
    try:
        current_stream = torch.npu.current_stream()
        for event in payload_events:
            current_stream.wait_event(event)
    except Exception as exc:
        raise RuntimeError(
            "Failed to wait on DSA selected-token producer event before "
            "LMCache row selection."
        ) from exc


def _dsa_device_tensor_types(value: Any) -> set[str]:
    if isinstance(value, torch.Tensor):
        return set() if value.device.type == "cpu" else {value.device.type}
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_dsa_device_tensor_types(item))
        return out
    return set()


def _dsa_record_payload_event_if_needed(*values: Any) -> Optional[Any]:
    device_types: set[str] = set()
    for value in values:
        device_types.update(_dsa_device_tensor_types(value))
    if not device_types:
        return None
    needs_npu_event = bool(device_types & {"npu", "privateuseone"})
    needs_cuda_event = "cuda" in device_types
    if needs_npu_event and needs_cuda_event:
        raise RuntimeError(
            "DSA payload contains both NPU and CUDA tensors; refusing to "
            "record a single ordering event for mixed device backends."
        )
    if needs_npu_event:
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "Event")):
            raise RuntimeError(
                "DSA reordered payload contains NPU tensors but torch.npu.Event "
                "is unavailable."
            )
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        return event
    if needs_cuda_event:
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            raise RuntimeError(
                "DSA reordered payload contains CUDA tensors but CUDA stream "
                "support is unavailable."
            )
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return event
    raise RuntimeError(
        "DSA reordered payload contains device tensors with unsupported "
        f"device types {sorted(device_types)}."
    )


def _contiguous_row_slice(rows: list[int]) -> Optional[slice]:
    if not rows:
        return None
    start = rows[0]
    if all(row == start + offset for offset, row in enumerate(rows)):
        return slice(start, start + len(rows))
    return None


def _row_select(value: Any, rows: list[int]):
    if hasattr(value, "__getitem__"):
        if isinstance(value, torch.Tensor):
            if len(rows) == 1:
                row = rows[0]
                return value[row]
            row_slice = _contiguous_row_slice(rows)
            if row_slice is not None:
                return value[row_slice]
            return value[rows]
        row_slice = _contiguous_row_slice(rows)
        if row_slice is not None:
            return value[row_slice]
        return [value[row] for row in rows]
    raise TypeError(f"Unsupported row-indexed value type: {type(value)!r}")


def _single_row_select(value: Any, row: int):
    if hasattr(value, "__getitem__"):
        if isinstance(value, torch.Tensor):
            return value[row]
        return value[row]
    raise TypeError(f"Unsupported row-indexed value type: {type(value)!r}")


def _sparse_payload_value(value: Any):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, torch.Tensor):
                out.extend(item.reshape(-1).tolist())
            elif isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    return value


def _flatten_block_ids(block_ids) -> list[int]:
    if block_ids is None:
        return []
    flattened: list[int] = []
    for elem in block_ids:
        if isinstance(elem, (list, tuple)):
            flattened.extend(elem)
        else:
            flattened.append(elem)
    return flattened


def _split_kv_group_block_ids(block_ids) -> tuple[list[int], Optional[list[int]]]:
    """Return latent and optional indexer block ids from vLLM block metadata."""
    if block_ids is None or len(block_ids) == 0:
        return [], None
    first = block_ids[0]
    if isinstance(first, (list, tuple)):
        latent_block_ids = _flatten_block_ids(block_ids[0])
        indexer_block_ids = (
            _flatten_block_ids(block_ids[1]) if len(block_ids) > 1 else None
        )
        return latent_block_ids, indexer_block_ids
    return _flatten_block_ids(block_ids), None


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool
    # Persisted LMCache prefix. Cold prepared partial tails may make this
    # unaligned; scheduler release remains independently aligned below.
    dsa_committed_end: Optional[int] = None
    # Exact frontier below which sparse decode may remap from LMCache. This
    # differs from committed_end when a full prompt hit recomputes its final
    # token.
    dsa_remap_frontier: Optional[int] = None
    # Fixed resident prefix capacity used by DSA MTP union scratch.
    dsa_scratch_capacity: Optional[int] = None
    # Frontier authorized for scheduler-side latent block release this step.
    dsa_release_frontier: Optional[int] = None
    # Frontier released from the current vLLM block table.
    dsa_current_released_frontier: int = 0


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool
    # Whether to save the latent (MLA) group (kv_group=0).
    # Defaults to True for backward compat. When dsa_two_groups is enabled,
    # the indexer group (kv_group=1) can be independently gated.
    can_save_latent: bool = True
    # Whether to save the indexer (DSA) group (kv_group=1).
    can_save_indexer: bool = False


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


def _disagg_spec_from_request(request: Optional["Request"]) -> Optional[DisaggSpec]:
    params = getattr(request, "kv_transfer_params", None)
    raw_spec = params.get("disagg_spec") if params else None
    if raw_spec is None:
        return None

    receiver_host = raw_spec["receiver_host"]
    receiver_init_port = raw_spec["receiver_init_port"]
    return DisaggSpec(
        req_id=raw_spec["req_id"],
        receiver_id=receiver_host + str(receiver_init_port),
        receiver_host=receiver_host,
        receiver_init_port=receiver_init_port,
        receiver_alloc_port=raw_spec["receiver_alloc_port"],
    )


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


def _apply_mm_hashes(
    token_ids: list[int],
    mm_hashes: Optional[list[str]],
    mm_positions: Optional[list["PlaceholderRange"]],
) -> list[int]:
    if not mm_hashes:
        return token_ids
    assert mm_positions is not None, "tracker got mm_hashes but no mm_positions"
    token_ids_tensor = torch.tensor(token_ids)
    apply_mm_hashes_to_token_ids(token_ids_tensor, mm_hashes, mm_positions)
    return token_ids_tensor.tolist()


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]
    allocated_block_ids_indexer: Optional[list[int]] = None

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase: bool = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    # Sparse decode only: prompt token ids for retrieve keys, built once.
    sparse_token_ids: list[int] = field(default_factory=list, repr=False)
    # Sparse decode only: single-element list holding CPU then NPU slot_mapping.
    sparse_slot_mapping: list[torch.Tensor] = field(default_factory=list, repr=False)
    sparse_indexer_slot_mapping: list[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    # Sparse decode only: reused across decode steps to avoid per-step allocation.
    sparse_decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    sparse_decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    # Sparse decode only: frontier whose full metadata has been emitted.
    sparse_meta_frontier: Optional[int] = field(default=None, repr=False)
    # Decode window save only: independent progress for decode-window chunks.
    decode_window_save_next_start: Optional[int] = field(default=None, repr=False)
    decode_window_save_anchor: Optional[int] = field(default=None, repr=False)
    decode_window_save_inflight_end: Optional[int] = field(default=None, repr=False)
    # Decode window save only: highest token boundary confirmed readable from
    # LMCache by worker-side completion output.
    decode_window_save_committed_end: int = field(default=0, repr=False)
    # Highest prefix frontier whose main latent KV cannot be assumed resident.
    # This includes both scheduler-side block release and cold-compact loads
    # that intentionally materialize only the indexer group. It is sticky
    # across rollback/preemption and monotonically increases for the request.
    dsa_nonresident_frontier: int = field(default=0, repr=False)
    # Released frontier for the CURRENT vLLM block table. Preemption allocates
    # a fresh table, so this resets to zero while the sticky high-water mark
    # above remains available to recover sparse state from LMCache.
    dsa_current_released_frontier: int = field(default=0, repr=False)
    # Scheduler-side completions acknowledged to unblock subsequent saves but
    # intentionally withheld from committed_end and local-block release.
    decode_window_save_pending_commits: deque[int] = field(
        default_factory=deque, repr=False
    )

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        disagg_spec: Optional[DisaggSpec] = None,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            skip_save (bool): whether the request cache should be saved
            disagg_spec: optional request-scoped transfer metadata.
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids, indexer_block_ids = _split_kv_group_block_ids(
            new_request.block_ids
        )

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        num_tokens_to_track = min(
            len(new_request.prompt_token_ids),
            max(num_tokens_to_compute, lmcache_cached_tokens),
        )

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_track].copy(),
            allocated_block_ids=unfolded_block_ids,
            allocated_block_ids_indexer=indexer_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
            decode_window_save_committed_end=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is not None and not isinstance(new_block_ids, (list, tuple)):
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        if new_block_ids is None:
            new_block_ids = []
        new_block_ids, new_indexer_block_ids = _split_kv_group_block_ids(
            new_block_ids
        )

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            self.sparse_token_ids.clear()
            self.sparse_slot_mapping.clear()
            self.sparse_indexer_slot_mapping.clear()
            self.sparse_decode_token_mask = None
            self.sparse_decode_ret_mask = None
            self.sparse_meta_frontier = None
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            self.allocated_block_ids_indexer = new_indexer_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            self.num_lmcache_cached_tokens = lmcache_cached_tokens
            self.decode_window_save_committed_end = max(
                lmcache_cached_tokens,
                self.dsa_nonresident_frontier,
            )
            self.decode_window_save_next_start = None
            self.decode_window_save_anchor = None
            self.decode_window_save_inflight_end = None
            self.decode_window_save_pending_commits.clear()
            self.dsa_current_released_frontier = 0
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            if new_indexer_block_ids is not None:
                if self.allocated_block_ids_indexer is None:
                    self.allocated_block_ids_indexer = []
                self.allocated_block_ids_indexer.extend(new_indexer_block_ids)
            self.token_ids.extend(new_token_ids)

        if len(self.token_ids) > self.prompt_len:
            self.is_decode_phase = True

    def seed_sparse_decode_tokens(
        self,
        token_ids: list[int],
        token_count: Optional[int] = None,
    ) -> None:
        """Seed token ids used to build sparse decode chunk keys."""
        target_count = self.prompt_len if token_count is None else token_count
        sparse_tokens = token_ids[:target_count]
        if len(sparse_tokens) < target_count:
            logger.warning(
                "Request %s sparse decode token source is shorter than target: "
                "source_tokens=%d target_tokens=%d prompt_len=%d",
                self.req_id,
                len(sparse_tokens),
                target_count,
                self.prompt_len,
            )
        self.sparse_token_ids = _apply_mm_hashes(
            sparse_tokens,
            self.mm_hashes,
            self.mm_positions,
        )


@dataclass
class WorkerRetrieveState:
    """Request-owned worker cache payload used by retrieve operations."""

    req_id: Optional[str] = None
    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    cached_shared_handles: list[list[Any]] = field(default_factory=list)
    cached_keys_indexer: list[list] = field(default_factory=list)
    cached_starts_indexer: list[int] = field(default_factory=list)
    cached_ends_indexer: list[int] = field(default_factory=list)
    cached_memory_objs_indexer: list[list] = field(default_factory=list)
    cached_tensors_indexer: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs_indexer: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu_indexer: list[Optional[torch.Tensor]] = field(
        default_factory=list
    )
    cached_shared_handles_indexer: list[list[Any]] = field(default_factory=list)
    shared_latent_status: str = "missing"
    shared_index_status: str = "missing"
    indexer_npu_resident: bool = False
    indexer_npu_materialization_pending: bool = field(default=False, repr=False)
    shared_generation: int = 0
    pointer_cache_generation: int = 0
    shared_request_active: bool = False
    request_scope_token: Optional[str] = None
    shared_validation_signature: Optional[tuple[Any, ...]] = None
    dense_prefix_seed: bool = False
    location: Optional[str] = None
    metadata_warm: bool = False
    token_count: int = 0
    metadata_token_ids: list[int] = field(default_factory=list, repr=False)
    slot_mapping: Optional[torch.Tensor] = field(default=None, repr=False)
    indexer_slot_mapping: Optional[torch.Tensor] = field(default=None, repr=False)
    decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    prepared_sparse_sources: dict[int, PreparedSparseSource] = field(
        default_factory=dict,
        repr=False,
    )
    dense_load_readiness: Optional[Any] = field(default=None, repr=False)
    dense_load_readiness_consumed: bool = field(default=False, repr=False)
    dense_load_source_owners: tuple[Any, ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def cache_kwargs(self, kv_group: int, dsa_two_groups: bool) -> dict[str, Any]:
        """Return mutable engine cache arguments for one KV group."""
        suffix = "_indexer" if dsa_two_groups and kv_group == 1 else ""
        return {
            name: getattr(self, f"{name}{suffix}")
            for name in (
                "cached_keys",
                "cached_starts",
                "cached_ends",
                "cached_memory_objs",
                "cached_tensors",
                "cached_chunk_dev_ptrs",
                "cached_chunk_ptrs_npu",
                "cached_shared_handles",
            )
        }

    def clear_group(self, kv_group: int) -> None:
        for values in self.cache_kwargs(kv_group, dsa_two_groups=True).values():
            values.clear()
        self.prepared_sparse_sources.pop(kv_group, None)

    def has_cache(self) -> bool:
        return bool(self.cached_keys)

    def group_has_data(self, kv_group: int, dsa_two_groups: bool) -> bool:
        cache = self.cache_kwargs(kv_group, dsa_two_groups)
        return bool(
            cache["cached_starts"]
            or cache["cached_ends"]
            or any(cache["cached_keys"])
            or any(cache["cached_memory_objs"])
            or any(cache["cached_tensors"])
            or any(cache["cached_chunk_dev_ptrs"])
            or any(ptr is not None for ptr in cache["cached_chunk_ptrs_npu"])
            or any(cache["cached_shared_handles"])
            or kv_group in self.prepared_sparse_sources
        )


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Single-element list; sparse decode reuses tracker.sparse_slot_mapping by
    # reference.
    slot_mapping: list[torch.Tensor] = field(default_factory=list)
    indexer_slot_mapping: list[torch.Tensor] = field(default_factory=list)
    # Save-only mappings for the exact sparse-layerwise write window. Chunk
    # ranges remain absolute; the worker/connector subtracts this base before
    # indexing these tensors.
    save_slot_mapping: list[torch.Tensor] = field(default_factory=list)
    save_indexer_slot_mapping: list[torch.Tensor] = field(default_factory=list)
    save_slot_mapping_base: Optional[int] = None
    windowed_sparse_save: bool = False

    # Sparse shared CPU decode only: kv_group=1 was intentionally skipped by
    # config, so hot-path validation may accept absent DSA index state.
    shared_index_skipped: bool = False
    # Sparse decode only: shared with RequestTracker, reused across decode steps.
    decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    # Decode window save metadata, separate from the regular request save progress.
    is_decode_window_save: bool = False
    decode_window_start: Optional[int] = None
    decode_window_end: Optional[int] = None
    decode_window_size: Optional[int] = None

    # Set by scheduler when a cached request resumes after preemption.
    resumed_from_preemption: bool = False

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Whether is sparse attention and decode or not
    is_sparse_decode: bool = False
    # Frontier released from the current vLLM block table. This resets after
    # preemption because vLLM assigns a fresh table.
    dsa_current_released_frontier: int = 0
    # Sticky request-lifetime proof that main latent KV cannot be assumed
    # resident. Unlike release history, this also represents cold-compact
    # loads that never populated the main latent blocks.
    dsa_nonresident_frontier: int = 0
    # Warm sparse decode reuses request-owned worker state and omits
    # sequence-length metadata from SchedulerOutput.
    sparse_warm_ref: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Producer-side live P/D handoff requested by the routing connector.
    live_source_requested: bool = False
    live_source_token_ids: list[int] = field(default_factory=list)
    live_source_slot_mapping: list[torch.Tensor] = field(default_factory=list)
    live_source_indexer_slot_mapping: list[torch.Tensor] = field(
        default_factory=list
    )
    live_split_compact: bool = False
    live_split_latent_cpu: bool = False

    def retrieve_token_count(self) -> int:
        """Return the logical retrieve prefix represented by this metadata."""
        if self.sparse_warm_ref and self.load_spec is not None:
            return int(self.load_spec.lmcache_cached_tokens)
        return len(self.token_ids)

    @staticmethod
    def from_decode_window_save(
        tracker: RequestTracker,
        block_size: int,
        window_start: int,
        window_end: int,
        window_size: int,
        windowed_sparse_layerwise_save: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create isolated metadata for a decode-window save."""
        if window_start < 0 or window_end <= window_start:
            return None
        if window_end > len(tracker.token_ids):
            return None

        token_ids = tracker.token_ids[:window_end].copy()
        token_ids = _apply_mm_hashes(
            token_ids,
            tracker.mm_hashes,
            tracker.mm_positions,
        )

        num_blocks = len(tracker.allocated_block_ids)
        if window_end > num_blocks * block_size:
            logger.warning(
                "Skipping decode window save for request %s: "
                "window_end=%d exceeds slot capacity %d",
                tracker.req_id,
                window_end,
                num_blocks * block_size,
            )
            return None

        indexer_slot_mapping: list[torch.Tensor] = []
        if tracker.allocated_block_ids_indexer:
            indexer_num_blocks = len(tracker.allocated_block_ids_indexer)
            if window_end > indexer_num_blocks * block_size:
                logger.warning(
                    "Skipping decode window indexer save for request %s: "
                    "window_end=%d exceeds indexer slot capacity %d",
                    tracker.req_id,
                    window_end,
                    indexer_num_blocks * block_size,
                )
            else:
                indexer_slot_mapping = [
                    (
                        _build_slot_mapping_window(
                            tracker.allocated_block_ids_indexer,
                            block_size,
                            window_start,
                            window_end,
                        )
                        if windowed_sparse_layerwise_save
                        else _build_slot_mapping(
                            tracker.allocated_block_ids_indexer,
                            block_size,
                            len(token_ids),
                        )
                    )
                ]

        save_slot_mapping: list[torch.Tensor] = []
        save_indexer_slot_mapping: list[torch.Tensor] = []
        if windowed_sparse_layerwise_save:
            save_slot_mapping = [
                _build_slot_mapping_window(
                    tracker.allocated_block_ids,
                    block_size,
                    window_start,
                    window_end,
                )
            ]
            if indexer_slot_mapping:
                save_indexer_slot_mapping = indexer_slot_mapping

        slot_mapping = (
            save_slot_mapping
            if windowed_sparse_layerwise_save
            else [
                _build_slot_mapping(
                    tracker.allocated_block_ids, block_size, len(token_ids)
                )
            ]
        )

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_slot_mapping=save_slot_mapping,
            save_indexer_slot_mapping=save_indexer_slot_mapping,
            save_slot_mapping_base=(
                window_start if windowed_sparse_layerwise_save else None
            ),
            windowed_sparse_save=windowed_sparse_layerwise_save,
            is_last_prefill=True,
            save_spec=SaveSpec(
                window_start,
                True,
                can_save_latent=True,
                can_save_indexer=bool(indexer_slot_mapping),
            ),
            indexer_slot_mapping=indexer_slot_mapping,
            load_spec=None,
            disagg_spec=None,
            request_configs=tracker.request_configs,
            is_decode_window_save=True,
            decode_window_start=window_start,
            decode_window_end=window_end,
            decode_window_size=window_size,
        )

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        is_sparse_decode: bool = False,
        save_full_chunk_in_decode: bool = False,
        dsa_two_groups: bool = False,
        windowed_sparse_layerwise_save: bool = False,
        save_entire_prefix: bool = False,
        live_source_requested: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.
            windowed_sparse_layerwise_save (bool): build exact save-only slot
                mappings and preserve chunked-prefill tails.
            save_entire_prefix (bool): retain a full source mapping for producer
                roles that intentionally resend the complete prefix.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        if (
            is_sparse_decode
            and load_spec is not None
            and tracker.sparse_token_ids
        ):
            sparse_token_count = (
                int(load_spec.lmcache_cached_tokens)
                if load_spec.can_load
                else len(tracker.sparse_token_ids)
            )
            if len(tracker.sparse_token_ids) == sparse_token_count:
                input_token_ids = tracker.sparse_token_ids
            elif len(tracker.sparse_token_ids) > sparse_token_count:
                input_token_ids = tracker.sparse_token_ids[:sparse_token_count]
        input_token_len = len(input_token_ids)
        metadata_current_released_frontier = tracker.dsa_current_released_frontier
        if is_sparse_decode and load_spec is not None:
            metadata_current_released_frontier = min(
                metadata_current_released_frontier,
                int(load_spec.dsa_current_released_frontier),
            )

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        allow_final_prefill_partial_save = (
            is_last_prefill
            and not tracker.is_decode_phase
            and not discard_partial_chunks
            and tracker.num_saved_tokens > 0
            and input_token_len > tracker.num_saved_tokens
            and input_token_len < chunk_boundary
        )
        # Sparse layerwise chunked prefill must preserve every scheduled tail.
        # vLLM is free to release that tail before a later crossing chunk is
        # formed. A later longer partial/full chunk has a different hash and is
        # stored by rewinding to the LMCache chunk boundary.
        allow_sparse_layerwise_prefill_progress = (
            windowed_sparse_layerwise_save
            and not is_sparse_decode
            and input_token_len <= tracker.prompt_len
            and input_token_len > tracker.num_saved_tokens
        )
        skip_by_tracker = bool(tracker.skip_save)
        skip_by_chunk_boundary = (
            tracker.num_saved_tokens > 0
            and input_token_len < chunk_boundary
            and not allow_final_prefill_partial_save
            and not allow_sparse_layerwise_prefill_progress
        )
        skip_by_decode_phase = bool(
            tracker.is_decode_phase
            and not save_decode_cache
            and not allow_sparse_layerwise_prefill_progress
        )
        skip_by_request_config = bool(request_skip)

        skip_save = tracker.disagg_spec is None and (
            skip_by_tracker
            or skip_by_chunk_boundary
            or skip_by_decode_phase
            or skip_by_request_config
        )

        # Decode-full-chunk rule: when save_full_chunk_in_decode is enabled,
        # only save during decode if a full chunk boundary is crossed.
        # This sidesteps the plane-major append cost (each store is a complete
        # tight buffer for exactly chunk_size tokens, no in-place growth).
        # Applied to BOTH latent and indexer groups.
        if (
            tracker.is_decode_phase
            and save_full_chunk_in_decode
            and not skip_save
            and not allow_sparse_layerwise_prefill_progress
        ):
            # Only save if we've crossed a full chunk boundary since last save
            new_boundary = (
                (tracker.num_saved_tokens + input_token_len)
                // lmcache_chunk_size * lmcache_chunk_size
            )
            if new_boundary <= tracker.num_saved_tokens:
                skip_save = True

        requires_indexer_slots = dsa_two_groups and (
            (load_spec is not None and load_spec.can_load)
            or not skip_save
            or live_source_requested
        )
        if requires_indexer_slots and not tracker.allocated_block_ids_indexer:
            raise RuntimeError(
                "dsa_two_groups requires Group-1 indexer block ids for every "
                "load, save, and live-transfer request. Refusing to construct "
                "one-group-only request metadata: "
                f"req_id={tracker.req_id}, can_load="
                f"{bool(load_spec is not None and load_spec.can_load)}, "
                f"skip_save={skip_save}, live_source={live_source_requested}"
            )

        if skip_save and load_spec is None and not live_source_requested:
            if is_sparse_decode or len(tracker.token_ids) < tracker.prompt_len:
                return None
            # Full-resident KV policy (方案 A): the prefix is already resident
            # and there is nothing left to load or save, but the request must
            # stay visible in the connector metadata so the staged-SFA route
            # classifies it as DENSE_PREFIX_HIT and replays the captured
            # graph. This covers both the steady state (token_ids > prompt_len)
            # and the FIRST compute step (token_ids == prompt_len, e.g. right
            # after a cold-compact load): without the entry the step looks
            # like MISSING_CONNECTOR_METADATA, falls back to the eager path,
            # and the torch_npu fx compiler captures ACL graphs at serving
            # time — which is illegal to overlap with the async cold-compact
            # load thread's synchronized device copies (device error 507057).
            return ReqMeta(
                req_id=tracker.req_id,
                token_ids=[],
                is_last_prefill=True,
                is_sparse_decode=False,
                dsa_current_released_frontier=metadata_current_released_frontier,
                dsa_nonresident_frontier=tracker.dsa_nonresident_frontier,
                save_spec=SaveSpec(
                    skip_leading_tokens,
                    False,
                    can_save_latent=False,
                    can_save_indexer=False,
                ),
                load_spec=None,
                disagg_spec=tracker.disagg_spec,
                request_configs=tracker.request_configs,
            )

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if allow_sparse_layerwise_prefill_progress:
            num_tokens_to_save = input_token_len
        elif not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        # NOTE: num_saved_tokens is advanced optimistically before the store
        # completes. If the store fails (CPU memory pressure), the scheduler
        # will skip re-storing on later steps. This is partially mitigated by
        # the lookup-miss re-store path in the async lookup client (min(hit)
        # aggregation detects missing chunks). A full fix would defer the
        # advance until wait_for_save confirms success (requires worker-to-scheduler
        # feedback channel - future work).
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # Determine per-group save flags for two-group DSA mode.
        can_save_latent = not skip_save
        can_save_indexer = not skip_save and dsa_two_groups
        save_spec = SaveSpec(
            skip_leading_tokens,
            not skip_save,
            can_save_latent=can_save_latent,
            can_save_indexer=can_save_indexer,
        )

        # Calculate the token ids and slot mappings for load and save
        if is_sparse_decode and load_spec is not None and skip_save:
            sparse_token_count = int(load_spec.lmcache_cached_tokens)
            if (
                load_spec.can_load
                and sparse_token_count == tracker.sparse_meta_frontier
                and not save_entire_prefix
            ):
                return ReqMeta(
                    req_id=tracker.req_id,
                    token_ids=[],
                    is_last_prefill=True,
                    is_sparse_decode=True,
                    dsa_current_released_frontier=metadata_current_released_frontier,
                    dsa_nonresident_frontier=tracker.dsa_nonresident_frontier,
                    sparse_warm_ref=True,
                    save_spec=save_spec,
                    load_spec=load_spec,
                    disagg_spec=tracker.disagg_spec,
                    request_configs=tracker.request_configs,
                )
            if (
                not tracker.sparse_token_ids
                or len(tracker.sparse_token_ids) != sparse_token_count
            ):
                if len(tracker.sparse_token_ids) >= sparse_token_count:
                    tracker.sparse_token_ids = tracker.sparse_token_ids[
                        :sparse_token_count
                    ]
                else:
                    tracker.seed_sparse_decode_tokens(
                        input_token_ids,
                        token_count=sparse_token_count,
                    )
            token_ids = tracker.sparse_token_ids
            if len(token_ids) < load_spec.lmcache_cached_tokens:
                logger.warning(
                    "Request %s sparse decode token metadata is shorter than "
                    "LMCache hit: sparse_tokens=%d lmcache_cached_tokens=%d "
                    "prompt_len=%d",
                    tracker.req_id,
                    len(token_ids),
                    load_spec.lmcache_cached_tokens,
                    tracker.prompt_len,
                )
        else:
            retrieve_token_len = 0
            if load_spec is not None and load_spec.can_load:
                retrieve_token_len = load_spec.lmcache_cached_tokens
            token_len = max(num_tokens_to_save, retrieve_token_len)
            token_ids = input_token_ids[:token_len]
            if retrieve_token_len > 0 and len(token_ids) < retrieve_token_len:
                logger.warning(
                    "Request %s prefix-hit token metadata is shorter than "
                    "LMCache hit: tokens=%d lmcache_cached_tokens=%d "
                    "prompt_len=%d",
                    tracker.req_id,
                    len(token_ids),
                    retrieve_token_len,
                    tracker.prompt_len,
                )

            token_ids = _apply_mm_hashes(
                token_ids,
                tracker.mm_hashes,
                tracker.mm_positions,
            )

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        windowed_sparse_save = (
            windowed_sparse_layerwise_save
            and not is_sparse_decode
            and not skip_save
        )
        use_windowed_save_mapping = (
            windowed_sparse_save and not save_entire_prefix
        )
        save_slot_mapping: list[torch.Tensor] = []
        save_indexer_slot_mapping: list[torch.Tensor] = []
        save_slot_mapping_base: Optional[int] = None
        if use_windowed_save_mapping:
            save_slot_mapping_base = (
                skip_leading_tokens // lmcache_chunk_size * lmcache_chunk_size
            )
            save_end = len(token_ids)
            save_slot_mapping = [
                _build_slot_mapping_window(
                    tracker.allocated_block_ids,
                    block_size,
                    save_slot_mapping_base,
                    save_end,
                )
            ]
            if dsa_two_groups and tracker.allocated_block_ids_indexer:
                save_indexer_slot_mapping = [
                    _build_slot_mapping_window(
                        tracker.allocated_block_ids_indexer,
                        block_size,
                        save_slot_mapping_base,
                        save_end,
                    )
                ]

        needs_full_slots = bool(
            (load_spec is not None and load_spec.can_load) or save_entire_prefix
        )
        if is_sparse_decode and load_spec is not None:
            num_slots = _sparse_slot_mapping_len(load_spec.lmcache_cached_tokens)
            current_slots = (
                int(tracker.sparse_slot_mapping[0].numel())
                if tracker.sparse_slot_mapping
                else -1
            )
            if current_slots != num_slots:
                tracker.sparse_slot_mapping.clear()
                tracker.sparse_slot_mapping.append(
                    _build_slot_mapping(
                        tracker.allocated_block_ids, block_size, num_slots
                    )
                )
            slot_mapping = tracker.sparse_slot_mapping
        elif use_windowed_save_mapping and not needs_full_slots:
            slot_mapping = save_slot_mapping
        else:
            slot_mapping = [
                _build_slot_mapping(
                    tracker.allocated_block_ids, block_size, len(token_ids)
                )
            ]

        indexer_slot_mapping: list[torch.Tensor] = []
        if dsa_two_groups and tracker.allocated_block_ids_indexer:
            indexer_num_blocks = len(tracker.allocated_block_ids_indexer)
            if len(token_ids) > indexer_num_blocks * block_size:
                logger.error(
                    "The number of tokens is more than the number of indexer "
                    "blocks for request %s. Something might be wrong in "
                    "scheduling logic!",
                    tracker.req_id,
                )
                logger.error(
                    "Num tokens: %d, num indexer blocks: %d, block size: %d",
                    len(token_ids),
                    indexer_num_blocks,
                    block_size,
                )
            if is_sparse_decode and load_spec is not None and load_spec.can_load:
                if (
                    not tracker.sparse_indexer_slot_mapping
                    or tracker.sparse_indexer_slot_mapping[0].numel()
                    < load_spec.lmcache_cached_tokens
                ):
                    tracker.sparse_indexer_slot_mapping.clear()
                    tracker.sparse_indexer_slot_mapping.append(
                        _build_slot_mapping(
                            tracker.allocated_block_ids_indexer,
                            block_size,
                            load_spec.lmcache_cached_tokens,
                        )
                    )
                indexer_slot_mapping = tracker.sparse_indexer_slot_mapping
            elif not is_sparse_decode:
                if use_windowed_save_mapping and not needs_full_slots:
                    indexer_slot_mapping = save_indexer_slot_mapping
                else:
                    indexer_slot_mapping = [
                        _build_slot_mapping(
                            tracker.allocated_block_ids_indexer,
                            block_size,
                            len(token_ids),
                        )
                    ]
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        decode_token_mask: Optional[torch.Tensor] = None
        decode_ret_mask: Optional[torch.Tensor] = None
        if is_sparse_decode and load_spec is not None:
            num_retrieve_tokens = len(token_ids)
            if load_spec.vllm_cached_tokens > 0:
                if (
                    tracker.sparse_decode_token_mask is None
                    or tracker.sparse_decode_token_mask.numel()
                    != num_retrieve_tokens
                ):
                    tracker.sparse_decode_token_mask = torch.ones(
                        num_retrieve_tokens, dtype=torch.bool
                    )
                decode_token_mask = tracker.sparse_decode_token_mask
            else:
                tracker.sparse_decode_token_mask = None
            if (
                tracker.sparse_decode_ret_mask is None
                or tracker.sparse_decode_ret_mask.numel() != num_retrieve_tokens
            ):
                tracker.sparse_decode_ret_mask = torch.zeros(
                    num_retrieve_tokens, dtype=torch.bool, device="cpu"
                )
            decode_ret_mask = tracker.sparse_decode_ret_mask

        live_source_token_ids: list[int] = []
        live_source_slot_mapping: list[torch.Tensor] = []
        live_source_indexer_slot_mapping: list[torch.Tensor] = []
        if live_source_requested and not is_sparse_decode:
            live_source_token_ids = _apply_mm_hashes(
                input_token_ids,
                tracker.mm_hashes,
                tracker.mm_positions,
            )
            live_source_slot_mapping = [
                _build_slot_mapping(
                    tracker.allocated_block_ids,
                    block_size,
                    len(live_source_token_ids),
                )
            ]
            if dsa_two_groups and tracker.allocated_block_ids_indexer:
                live_source_indexer_slot_mapping = [
                    _build_slot_mapping(
                        tracker.allocated_block_ids_indexer,
                        block_size,
                        len(live_source_token_ids),
                    )
                ]

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        req_meta = ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            indexer_slot_mapping=indexer_slot_mapping,
            save_slot_mapping=save_slot_mapping,
            save_indexer_slot_mapping=save_indexer_slot_mapping,
            save_slot_mapping_base=save_slot_mapping_base,
            windowed_sparse_save=windowed_sparse_save,
            is_last_prefill=is_last_prefill,
            is_sparse_decode=is_sparse_decode,
            dsa_current_released_frontier=metadata_current_released_frontier,
            dsa_nonresident_frontier=tracker.dsa_nonresident_frontier,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            decode_token_mask=decode_token_mask,
            decode_ret_mask=decode_ret_mask,
            live_source_requested=live_source_requested,
            live_source_token_ids=live_source_token_ids,
            live_source_slot_mapping=live_source_slot_mapping,
            live_source_indexer_slot_mapping=live_source_indexer_slot_mapping,
        )
        if (
            is_sparse_decode
            and load_spec is not None
            and load_spec.can_load
            and skip_save
            and not save_entire_prefix
            and len(token_ids) == load_spec.lmcache_cached_tokens
            and slot_mapping
            and slot_mapping[0].numel()
            == _sparse_slot_mapping_len(load_spec.lmcache_cached_tokens)
        ):
            tracker.sparse_meta_frontier = int(load_spec.lmcache_cached_tokens)
        return req_meta


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # Hybrid source capture is a two-sided in-process protocol.  Keep it
        # off until AscendMulti has verified that both the LMCache provider
        # and Mooncake borrower support the same transport.
        self._live_latent_split_requested = False

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

        if config.extra_config is None:
            config.extra_config = {}

        model_config = getattr(vllm_config, "model_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if model_config is not None:
            max_model_len = getattr(model_config, "max_model_len", None)
            if max_model_len is not None:
                config.extra_config["vllm_max_model_len"] = max_model_len
        if scheduler_config is not None:
            max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
            if max_num_seqs is not None:
                config.extra_config["vllm_max_num_seqs"] = max_num_seqs
            max_num_batched_tokens = getattr(
                scheduler_config,
                "max_num_batched_tokens",
                None,
            )
            if max_num_batched_tokens is not None:
                config.extra_config["vllm_max_num_batched_tokens"] = (
                    max_num_batched_tokens
                )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        # Each entry is a (primary, secondary) retriever pair. primary is the
        # latent (kv_group=0) retriever; secondary is the indexer (kv_group=1)
        # retriever for two-group prefix/sparse retrieve, or None for
        # single-group. wait_for_layer_load routes latent and indexer layer
        # waits to the matching group and advances current_layer after all
        # required groups for that layer have completed.
        self.layerwise_retrievers: list[
            tuple[Optional[Generator[Optional[torch.Tensor], None, None]],
                  Optional[Generator[Optional[torch.Tensor], None, None]]]
        ] = []
        self._layerwise_requests: list[ReqMeta] = []
        self._layerwise_retriever_is_sparse: list[bool] = []
        self._layerwise_sparse_req_ids: list[str] = []
        self._layerwise_sparse_row_groups_key: Optional[
            tuple[tuple[str, ...], tuple[str, ...]]
        ] = None
        self._layerwise_sparse_row_groups: Optional[dict[str, list[int]]] = None
        self._layerwise_waited_groups: set[int] = set()
        self._layerwise_sparse_indexer_sent_layers: set[tuple[str, int]] = set()
        self._layerwise_sparse_shared_ordered: list[bool] = []
        self._layerwise_required_wait_groups_cache: Optional[set[int]] = None
        self._layerwise_save_storers: dict[
            LayerwiseSaveKey, Generator[Optional[LayerwiseStoreResult], None, None]
        ] = {}
        # Under dsa_two_groups + TP>1, latent store_layer is deferred until
        # after all indexer layers in a forward to avoid interleaved latent/
        # indexer GPU transfers on store_stream (MTE OOB on chunk 2+).
        self._deferred_latent_pending: set[LayerwiseSaveKey] = set()
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.enable_sparse_attention = config.enable_sparse_attention
        self._retrieve_stats_interval_seconds = (
            _retrieve_stats_interval_seconds()
        )
        self._retrieve_stats_window_started_at: Optional[float] = None
        self._retrieve_stats_request_count = 0
        self._retrieve_stats_row_count = 0
        self._retrieve_stats_token_count = 0
        self._cold_perf_lookup_started: dict[str, float] = {}
        self._cold_perf_load_started: dict[str, tuple[float, int]] = {}
        self._cold_perf_dense_load_started: dict[str, tuple[float, int]] = {}
        self._cold_perf_dense_load_completed: dict[str, float] = {}
        if (
            role != KVConnectorRole.SCHEDULER
            and self._retrieve_stats_interval_seconds > 0
        ):
            logger.info(
                "LMCache sparse retrieve statistics enabled; reporting every "
                "%d seconds.",
                self._retrieve_stats_interval_seconds,
            )

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._kvcaches_list: list[torch.Tensor] = []
        # Two-group MLA+DSA: latent (kv_group=0) and indexer (kv_group=1)
        # caches are partitioned by layer name ("indexer" in name) so that
        # per-group store/retrieve can pass the correct, group-filtered
        # kvcaches list to the connector. _kvcaches_list stays equal to the
        # latent list for backward-compatible latent-only callers.
        self._latent_layer_names: list[str] = []
        self._indexer_layer_names: list[str] = []
        self._latent_kvcaches: list[torch.Tensor] = []
        self._indexer_kvcaches: list[torch.Tensor] = []
        self._block_size = vllm_config.cache_config.block_size
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = getattr(vllm_config.model_config, "hf_config", None)
        dsa_topk = int(getattr(hf_config, "index_topk", 0) or 0)
        self._dsa_scratch_capacity = (
            (1 + max(int(getattr(vllm_config, "num_speculative_tokens", 0)), 0))
            * dsa_topk
        )
        # This threshold only selects the connector's KV loading/residency
        # policy. It does not select the attention kernel: SFA attention remains
        # sparse for both policies. 0 (default, unset or "0") disables the
        # short-context full-resident policy; a positive value is used as-is.
        raw_policy_threshold = os.environ.get(
            "LMCACHE_DSA_KV_POLICY_THRESHOLD"
        )
        if raw_policy_threshold is None:
            self._dsa_kv_policy_threshold = 0
        else:
            try:
                parsed_threshold = int(raw_policy_threshold)
            except (TypeError, ValueError):
                parsed_threshold = 0
            self._dsa_kv_policy_threshold = (
                parsed_threshold if parsed_threshold > 0 else 0
            )
        self.load_specs: dict[str, LoadSpec] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        # Per-request KV policy state for diagnostics. This is intentionally
        # named independently of the attention path: both policies use sparse
        # attention. Log policy (re)entries/switches once per transition.
        self._dsa_kv_policy_log = os.environ.get(
            "LMCACHE_DSA_KV_POLICY_LOG", "0"
        ).lower() in ("1", "true", "yes", "on")
        self._dsa_kv_policy_states: dict[str, str] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
            and not self.enable_sparse_attention
        )

        self._lmcache_chunk_size = config.chunk_size
        self._decode_window_save_window_size = (
            self._get_decode_window_save_window_size(config)
        )
        self._decode_window_save_commit_delay_windows = max(
            int(
                os.environ.get(
                    DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS_ENV,
                    "0",
                )
            ),
            0,
        )
        if (
            self._decode_window_save_commit_delay_windows > 0
            and self._decode_window_save_window_size <= 0
        ):
            logger.warning(
                "%s=%d is ignored because decode-window save is disabled.",
                DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS_ENV,
                self._decode_window_save_commit_delay_windows,
            )
        elif self._decode_window_save_commit_delay_windows > 0:
            logger.info(
                "Decode-window committed_end and local release will lag by "
                "%d completed saves.",
                self._decode_window_save_commit_delay_windows,
            )

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        base_num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        metadata_kv_shape = getattr(self.lmcache_engine_metadata, "kv_shape", None)
        metadata_num_layers = (
            metadata_kv_shape[0]
            if metadata_kv_shape is not None and len(metadata_kv_shape) > 0
            else None
        )
        self.num_layers = metadata_num_layers or (
            base_num_layers + calculate_draft_layers(vllm_config)
        )
        self.current_layer = 0

        self.force_skip_save = not is_false(
            os.environ.get("LMCACHE_FORCE_SKIP_SAVE", "false")
        )
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        if role != KVConnectorRole.SCHEDULER:
            self._worker_retrieve_state: dict[str, WorkerRetrieveState] = {}
            self._worker_retrieve_registry_version = 0
            self._worker_retrieve_last_prune_key: Optional[
                tuple[frozenset[str], int]
            ] = None
            self._wait_for_save_done = True
            self._finished_req_ids_waiting_for_save: set[str] = set()
            self._late_finished_sending: set[str] = set()
            self._completed_decode_window_saves: dict[str, int] = {}
            self._decode_window_save_completed_groups: set[LayerwiseSaveKey] = set()
            self._prefill_save_completed_groups: dict[LayerwiseSaveKey, int] = {}
            self._decode_window_save_expected_start: dict[str, int] = {}
            self._warn_mla_per_rank_lookup_config(config)

    def _warn_mla_per_rank_lookup_config(self, config: LMCacheEngineConfig) -> None:
        metadata = self.lmcache_engine_metadata
        if metadata is None or not metadata.use_mla:
            return
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank:
            return
        lookup_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        if len(lookup_ids) < metadata.world_size:
            logger.warning(
                "MLA per-rank store (save_only_first_rank=false) but lookup "
                "server runs on ranks %s only (world_size=%d). The scheduler "
                "may trust rank0 hit count while other TP ranks miss KV on "
                "retrieve_layer -> garbled generation. Remove "
                "lookup_server_worker_ids override or list all TP ranks.",
                lookup_ids,
                metadata.world_size,
            )

    def _get_decode_window_save_window_size(
        self, config: LMCacheEngineConfig
    ) -> int:
        assert self._lmcache_chunk_size % self._block_size == 0, (
            "LMCache chunk_size must be an integer multiple of vLLM "
            f"block_size: chunk_size={self._lmcache_chunk_size}, "
            f"block_size={self._block_size}. Configure LMCache chunk_size "
            "to N * block_size."
        )
        raw_window_size = os.environ.get("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE")
        if raw_window_size is None:
            raw_window_size = config.get_extra_config_value(
                "decode_window_save_window_size", 0
            )

        try:
            window_size = int(raw_window_size or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "decode_window_save_window_size must be an integer, "
                f"got {raw_window_size!r}"
            ) from exc

        if window_size == 0:
            return 0
        if window_size < self._lmcache_chunk_size:
            raise ValueError(
                "decode_window_save_window_size must be >= lmcache chunk_size "
                f"({self._lmcache_chunk_size}), got {window_size}"
            )
        if window_size % self._lmcache_chunk_size != 0:
            raise ValueError(
                "decode_window_save_window_size must be a multiple of "
                f"lmcache chunk_size ({self._lmcache_chunk_size}), got {window_size}"
            )
        if not config.save_unfull_chunk:
            raise ValueError(
                "decode window save requires save_unfull_chunk=true so the "
                "prefill partial LMCache chunk is available to a PD decoder."
            )

        logger.info(
            "Decode window save enabled: window_size=%d, lmcache_chunk_size=%d",
            window_size,
            self._lmcache_chunk_size,
        )
        return window_size

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        manager = getattr(self, "_manager", None)
        if manager is None:
            return None
        return manager.lmcache_engine

    @lmcache_engine.setter
    def lmcache_engine(self, value: Optional[LMCacheEngine]) -> None:
        """Set the LMCache engine instance on manager-backed adapters."""
        manager = getattr(self, "_manager", None)
        if manager is None:
            self._manager = SimpleNamespace(lmcache_engine=value)
            return
        manager.lmcache_engine = value

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    def _build_kv_layer_groups(self):
        # Build KV layer groups structure if not already built
        if self.lmcache_engine is not None:
            assert len(self.kv_caches) > 0
            kv_layer_groups_manager = (
                self.lmcache_engine.metadata.kv_layer_groups_manager
            )
            kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)
            self._normalize_dsa_kv_layer_groups()

    def _normalize_dsa_kv_layer_groups(self) -> None:
        """Keep metadata group index aligned with the DSA kv_group contract."""
        if not self._is_dsa_two_groups() or self.lmcache_engine is None:
            return
        manager = self.lmcache_engine.metadata.kv_layer_groups_manager
        groups = list(manager.kv_layer_groups)
        if not groups:
            return

        latent_names = set(getattr(self, "_latent_layer_names", []))
        indexer_names = set(getattr(self, "_indexer_layer_names", []))
        if not latent_names and not indexer_names:
            self._refresh_kvcaches_list()
            latent_names = set(getattr(self, "_latent_layer_names", []))
            indexer_names = set(getattr(self, "_indexer_layer_names", []))
        if not indexer_names:
            return

        latent_groups = []
        indexer_groups = []
        for group in groups:
            names = set(group.layer_names)
            has_indexer = bool(names & indexer_names) or any(
                "indexer" in name for name in names
            )
            has_latent = bool(names & latent_names) or not has_indexer
            if has_indexer and has_latent:
                raise RuntimeError(
                    "DSA two-group KV metadata is ambiguous: one metadata "
                    "group contains both latent and indexer layers. "
                    f"layer_names={group.layer_names}"
                )
            if has_indexer:
                indexer_groups.append(group)
            else:
                latent_groups.append(group)

        if len(latent_groups) != 1 or len(indexer_groups) != 1:
            raise RuntimeError(
                "DSA two-group KV metadata requires exactly one latent "
                "metadata group and one indexer metadata group so kv_group=0 "
                "maps to latent and kv_group=1 maps to indexer. "
                f"latent_groups={len(latent_groups)}, "
                f"indexer_groups={len(indexer_groups)}, "
                f"groups={groups}"
            )

        normalized_groups = latent_groups + indexer_groups
        if manager.kv_layer_groups != normalized_groups:
            logger.info(
                "Reordered DSA KV metadata groups to latent/indexer order: "
                "latent_dtype=%s, indexer_dtype=%s",
                normalized_groups[0].dtype,
                normalized_groups[1].dtype,
            )
            manager.kv_layer_groups = normalized_groups

    def _refresh_kvcaches_list(self) -> None:
        self._latent_layer_names = []
        self._indexer_layer_names = []
        self._latent_kvcaches = []
        self._indexer_kvcaches = []
        dsa_two_groups = getattr(self.config, "dsa_two_groups", False)
        for layer_name, kv_cache in self.kv_caches.items():
            if dsa_two_groups and "indexer" in layer_name:
                self._indexer_layer_names.append(layer_name)
                self._indexer_kvcaches.append(kv_cache)
            else:
                self._latent_layer_names.append(layer_name)
                self._latent_kvcaches.append(kv_cache)
        # Backward-compatible flat list = latent group (the default group).
        self._kvcaches_list = self._latent_kvcaches
        worker_has_caches = (
            len(self.kv_caches) > 0
            and getattr(self, "_role", None) != KVConnectorRole.SCHEDULER
        )
        if dsa_two_groups and worker_has_caches and not self._latent_kvcaches:
            raise RuntimeError(
                "dsa_two_groups is enabled but no latent KV caches were "
                "registered with the connector. Refusing to run because "
                "Group 0 and Group 1 must be loaded and stored atomically."
            )
        if dsa_two_groups and worker_has_caches and not self._indexer_kvcaches:
            raise RuntimeError(
                "dsa_two_groups is enabled but no indexer KV caches were "
                "registered with the connector (no layer name contains "
                "'indexer'). Refusing to run because skipping Group 1 would "
                "mix valid latent KV with stale or uninitialized index rows. "
                "Ensure vLLM registers the indexer KV cache group with this "
                "connector."
            )

    def _kvcaches_for_group(self, kv_group: int) -> list[torch.Tensor]:
        """Return the per-group kv_caches list for the connector."""
        if not hasattr(self, "_latent_kvcaches"):
            if hasattr(self, "kv_caches"):
                self._refresh_kvcaches_list()
            else:
                self._latent_kvcaches = list(getattr(self, "_kvcaches_list", []))
        if not hasattr(self, "_indexer_kvcaches"):
            self._indexer_kvcaches = []
        if kv_group == 1 and getattr(
            getattr(self, "config", None), "dsa_two_groups", False
        ):
            return self._indexer_kvcaches
        return self._latent_kvcaches

    def _num_layers_for_group(self, kv_group: int) -> int:
        return len(self._kvcaches_for_group(kv_group))

    def _is_dsa_two_groups(self) -> bool:
        return bool(getattr(getattr(self, "config", None), "dsa_two_groups", False))

    def _ensure_retrieve_stats_state(self) -> int:
        """Lazily initialize stats for lightweight/test connector instances."""
        interval = getattr(self, "_retrieve_stats_interval_seconds", None)
        if interval is None:
            interval = _retrieve_stats_interval_seconds()
            self._retrieve_stats_interval_seconds = interval
            self._retrieve_stats_window_started_at = None
            self._retrieve_stats_request_count = 0
            self._retrieve_stats_row_count = 0
            self._retrieve_stats_token_count = 0
        return int(interval)

    def _record_sparse_retrieve_stats(
        self,
        selected_tokens: Any,
        selected_token_counts: Any,
        row_count: int,
    ) -> None:
        """Record one request, combining all of its MTP rows."""
        interval = self._ensure_retrieve_stats_state()
        if interval <= 0:
            return

        if selected_token_counts is not None:
            if isinstance(selected_token_counts, torch.Tensor):
                retrieved_tokens = int(
                    selected_token_counts.detach().sum().to(device="cpu").item()
                )
            elif isinstance(selected_token_counts, (list, tuple)):
                retrieved_tokens = sum(
                    int(count) for count in selected_token_counts
                )
            else:
                retrieved_tokens = int(selected_token_counts)
        elif selected_tokens is not None:
            if isinstance(selected_tokens, torch.Tensor):
                retrieved_tokens = int(selected_tokens.numel())
            elif hasattr(selected_tokens, "__len__"):
                retrieved_tokens = len(selected_tokens)
            else:
                retrieved_tokens = 1
        else:
            return

        now = time.monotonic()
        if self._retrieve_stats_window_started_at is None:
            self._retrieve_stats_window_started_at = now
        self._retrieve_stats_request_count += 1
        self._retrieve_stats_row_count += max(int(row_count), 1)
        self._retrieve_stats_token_count += retrieved_tokens

        elapsed = now - self._retrieve_stats_window_started_at
        if elapsed < interval:
            return

        request_count = self._retrieve_stats_request_count
        average = (
            self._retrieve_stats_token_count / request_count
            if request_count
            else 0.0
        )
        logger.info(
            "[LMCacheRetrieveStats] elapsed=%.3fs requests=%d rows=%d "
            "retrieved_tokens=%d avg_retrieved_tokens_per_request=%.3f",
            elapsed,
            request_count,
            self._retrieve_stats_row_count,
            self._retrieve_stats_token_count,
            average,
        )
        self._retrieve_stats_window_started_at = now
        self._retrieve_stats_request_count = 0
        self._retrieve_stats_row_count = 0
        self._retrieve_stats_token_count = 0

    def _is_indexer_layer_wait(self, layer_name: str) -> bool:
        if not self._is_dsa_two_groups():
            return False
        indexer_names = getattr(self, "_indexer_layer_names", [])
        return layer_name in indexer_names or "indexer" in layer_name

    def _layerwise_wait_group(self, layer_name: str) -> int:
        return 1 if self._is_indexer_layer_wait(layer_name) else 0

    @staticmethod
    def _layerwise_layer_id_from_name(layer_name: str) -> Optional[int]:
        marker = "layers."
        marker_idx = layer_name.find(marker)
        if marker_idx < 0:
            return None
        start = marker_idx + len(marker)
        end = start
        while end < len(layer_name) and layer_name[end].isdigit():
            end += 1
        if end == start:
            return None
        return int(layer_name[start:end])

    def _layerwise_required_wait_groups(self) -> set[int]:
        cached = getattr(self, "_layerwise_required_wait_groups_cache", None)
        if cached is not None:
            return cached

        required = {0}
        if self._is_dsa_two_groups():
            # Dense two-group loads are complete only after both groups have
            # crossed the same layer boundary.  Do not infer this requirement
            # from indexer_retriever being non-None: that makes a missing
            # Group-1 retriever silently turn into a Group-0-only fast path.
            sparse_flags = getattr(self, "_layerwise_retriever_is_sparse", [])
            if any(not is_sparse for is_sparse in sparse_flags):
                required.add(1)
        self._layerwise_required_wait_groups_cache = required
        return required

    def _layerwise_wait_should_advance(self, wait_group: int) -> bool:
        waited_groups = getattr(self, "_layerwise_waited_groups", None)
        if waited_groups is None:
            waited_groups = set()
            self._layerwise_waited_groups = waited_groups
        waited_groups.add(wait_group)
        if self._layerwise_required_wait_groups().issubset(waited_groups):
            waited_groups.clear()
            return True
        return False

    def _shared_cpu_config_value(self, key: str, default: Any = None) -> Any:
        missing = object()

        def config_value(config: Any) -> Any:
            extra_config = getattr(config, "extra_config", None)
            if isinstance(extra_config, dict) and key in extra_config:
                return extra_config[key]
            config_dict = getattr(config, "__dict__", None)
            if isinstance(config_dict, dict) and key in config_dict:
                return config_dict[key]
            getter = getattr(config, "get_extra_config_value", None)
            if (
                callable(getter)
                and not type(config).__module__.startswith("unittest.mock")
            ):
                return getter(key, default)
            return missing

        engine = getattr(self, "lmcache_engine", None)
        if engine is not None:
            getter = getattr(engine, "_get_shared_config_value", None)
            if callable(getter):
                return getter(key, default)
            config = getattr(engine, "config", None)
            if config is not None:
                value = config_value(config)
                if value is not missing:
                    return value
            engine_dict = getattr(engine, "__dict__", None)
            if isinstance(engine_dict, dict) and key in engine_dict:
                return engine_dict[key]

        config = getattr(self, "config", None)
        if config is not None:
            value = config_value(config)
            if value is not missing:
                return value
        return default

    def _shared_cpu_materialize_index_on_decode_cold(self) -> bool:
        return bool(
            self._shared_cpu_config_value(
                "shared_cpu_materialize_index_on_decode_cold",
                True,
            )
        )

    def _sparse_decode_requires_index_materialization(
        self,
        request: "ReqMeta",
        shared_cpu_enabled: bool,
    ) -> bool:
        """True when sparse decode must materialize DSA index from LMCache.

        In the non-shared kv_both path, prefill may populate the resident DSA
        index cache in vLLM. A shared-CPU sparse decode hit, however, can skip
        prompt prefill entirely, so it must materialize the index group from
        LMCache instead of assuming resident index state is valid.
        """
        if not self._is_dsa_two_groups():
            return False
        if not self._shared_cpu_materialize_index_on_decode_cold():
            return False
        if shared_cpu_enabled:
            return True
        kv_role = getattr(self, "kv_role", "kv_both")
        return kv_role == "kv_consumer"

    @staticmethod
    def _shared_sparse_decode_indexer_retrieve_mode(
        request: "ReqMeta",
        bound_state: Optional[WorkerRetrieveState],
        token_count: int,
    ) -> str:
        """Select how group-1 state reaches the sparse decode frontier.

        Shared CPU decode must cold-materialize the DSA index once because a
        prefix hit may skip prefill. A live request then keeps that NPU state
        while frontier growth refreshes only its shared CPU metadata.
        """
        if (
            bound_state is None
            or not bound_state.shared_request_active
            or bound_state.shared_index_status != "present"
            or not bound_state.indexer_npu_resident
            or request.load_spec is None
            or bool(getattr(request, "resumed_from_preemption", False))
        ):
            return INDEXER_RETRIEVE_FULL
        metadata_current = (
            int(request.load_spec.lmcache_cached_tokens) <= int(bound_state.token_count)
            and int(token_count) <= int(bound_state.token_count)
        )
        if metadata_current:
            return INDEXER_RETRIEVE_RESIDENT_SKIP
        return INDEXER_RETRIEVE_METADATA_ONLY

    @staticmethod
    def _shared_request_scope_token(
        req_id: str,
        generation: int,
        token_count: int,
    ) -> str:
        return f"{req_id}:{generation}:{token_count}"

    @staticmethod
    def _shared_retrieve_token_count_for_request(
        request: ReqMeta,
    ) -> int:
        token_count = len(request.token_ids)
        if request.is_sparse_decode and request.load_spec is not None:
            token_count = int(request.load_spec.lmcache_cached_tokens)
        return token_count

    @classmethod
    def _shared_request_scope_token_for_request(
        cls,
        request: ReqMeta,
        generation: int,
    ) -> str:
        return cls._shared_request_scope_token(
            request.req_id,
            generation,
            cls._shared_retrieve_token_count_for_request(request),
        )

    def _shared_worker_validation_signature(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        *,
        current_generation: int,
        pointer_generation: int,
        materialize_index: bool,
    ) -> tuple[Any, ...]:
        return (
            state.request_scope_token,
            current_generation,
            pointer_generation,
            state.shared_latent_status,
            state.shared_index_status,
            state.indexer_npu_resident,
            bool(self._is_dsa_two_groups()),
            bool(materialize_index),
            int(getattr(self, "num_layers", 0) or 0),
            request.req_id,
            len(state.cached_starts or []),
            len(state.cached_ends or []),
            id(state.cached_memory_objs),
            id(state.cached_chunk_ptrs_npu),
        )

    def _validate_shared_worker_retrieve_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        engine = getattr(self, "lmcache_engine", None)
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not getattr(request, "is_sparse_decode", False)
            or not state.shared_request_active
        ):
            return

        current_generation = int(
            getattr(engine, "shared_cpu_cache_generation", 0) or 0
        )
        state_generation = int(state.shared_generation or 0)
        pointer_generation = int(
            getattr(state, "pointer_cache_generation", 0) or state_generation
        )
        if state_generation != current_generation:
            raise RuntimeError(
                "Shared CPU sparse decode state generation mismatch before "
                "hot-path reuse: "
                f"req_id={request.req_id}, state_generation="
                f"{state.shared_generation}, current_generation="
                f"{current_generation}"
            )
        if pointer_generation != current_generation:
            raise RuntimeError(
                "Shared CPU sparse decode pointer-cache generation mismatch "
                "before hot-path reuse: "
                f"req_id={request.req_id}, pointer_cache_generation="
                f"{pointer_generation}, current_generation={current_generation}"
            )
        retrieve_token_count = self._shared_retrieve_token_count_for_request(
            request
        )
        state_token_count = int(state.token_count or retrieve_token_count)
        validated_token_count = (
            state_token_count
            if request.sparse_warm_ref and state_token_count >= retrieve_token_count
            else min(retrieve_token_count, state_token_count)
        )
        prepared_latent = state.prepared_sparse_sources.get(0)
        if (
            prepared_latent is not None
            and prepared_latent.total_tokens == validated_token_count
        ):
            if state.req_id != request.req_id:
                raise RuntimeError(
                    "Shared CPU sparse decode request identity changed before "
                    f"prepared reuse: state={state.req_id!r}, "
                    f"request={request.req_id!r}"
                )
            # Shared status, prefix coverage, layer counts, and pointer tables
            # were checked before this immutable source was published. Only
            # process generation and request identity can invalidate it here.
            return

        expected_scope_token = self._shared_request_scope_token(
            request.req_id,
            current_generation,
            validated_token_count,
        )
        if state.request_scope_token != expected_scope_token:
            raise RuntimeError(
                "Shared CPU sparse decode request scope mismatch before "
                "hot-path reuse: "
                f"req_id={request.req_id}, request_scope_token="
                f"{state.request_scope_token!r}, expected="
                f"{expected_scope_token!r}"
            )
        if state.shared_latent_status != "present":
            raise RuntimeError(
                "Shared CPU sparse decode hot path requires MLA latent "
                "state before transfer: "
                f"req_id={request.req_id}, status={state.shared_latent_status!r}"
            )
        materialize_index = False
        if self._is_dsa_two_groups():
            materialize_index = self._sparse_decode_requires_index_materialization(
                request,
                True,
            )
            allowed = ("present",) if materialize_index else ("present", "skipped")
            if state.shared_index_status not in allowed:
                raise RuntimeError(
                    "Shared CPU sparse decode hot path has invalid DSA index "
                    "state before transfer: "
                    f"req_id={request.req_id}, status="
                    f"{state.shared_index_status!r}, materialize_index="
                    f"{materialize_index}"
                )

        validation_signature = self._shared_worker_validation_signature(
            state,
            request,
            current_generation=current_generation,
            pointer_generation=pointer_generation,
            materialize_index=materialize_index,
        )
        if state.shared_validation_signature == validation_signature:
            return

        if not self._cached_ranges_cover_prefix(
            state.cached_starts,
            state.cached_ends,
            validated_token_count,
        ):
            cached_ranges = list(
                zip(state.cached_starts, state.cached_ends, strict=False)
            )
            raise RuntimeError(
                "Shared CPU sparse decode hot path has non-contiguous MLA "
                "latent prefix coverage before transfer: "
                f"req_id={request.req_id}, kv_group=0, "
                f"token_count={validated_token_count}, "
                f"cached_ranges={cached_ranges}"
            )
        expected_layers = int(getattr(self, "num_layers", 0) or 0)
        required_latent_chunks = self._shared_required_chunk_count(
            state.cached_starts,
            state.cached_ends,
            state.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_layer_cache_coverage(
            state.cached_memory_objs,
            expected_layers,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Shared CPU sparse decode hot path has incomplete MLA "
                "latent state before transfer: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )
        missing_latent_pointer_layers = self._missing_shared_pointer_cache_layers(
            state.cached_memory_objs,
            state.cached_chunk_ptrs_npu,
            required_latent_chunks,
        )
        if missing_latent_pointer_layers:
            raise RuntimeError(
                "Shared CPU sparse decode hot path is missing MLA latent "
                "NPU pointer-cache tensors before transfer: "
                f"req_id={request.req_id}, "
                f"missing_layers={missing_latent_pointer_layers}"
            )
        # DSA index is cold-materialized and admitted into the live request
        # state before warm sparse decode reuse. Warm decode retrieves only MLA
        # latent rows, so avoid walking index metadata or pointer caches here.
        state.shared_validation_signature = validation_signature

    def _shared_worker_retrieve_state_is_current(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        token_count: int,
    ) -> bool:
        if not state.shared_request_active or state.shared_validation_signature is None:
            return False
        if request.load_spec is not None and (
            int(request.load_spec.lmcache_cached_tokens) > int(state.token_count)
        ):
            return False
        if int(token_count) > int(state.token_count):
            return False

        engine = getattr(self, "lmcache_engine", None)
        current_generation = int(
            getattr(engine, "shared_cpu_cache_generation", 0) or 0
        )
        pointer_generation = int(
            getattr(state, "pointer_cache_generation", 0)
            or int(state.shared_generation or 0)
        )
        materialize_index = (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(request, True)
        )
        validation_signature = self._shared_worker_validation_signature(
            state,
            request,
            current_generation=current_generation,
            pointer_generation=pointer_generation,
            materialize_index=materialize_index,
        )
        return state.shared_validation_signature == validation_signature

    @staticmethod
    def _indexer_slot_mapping_from_attn_metadata(
        attn_metadata, layer_name: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return the DSA indexer slot mapping from vLLM attention metadata.

        vLLM may pass either a single metadata object or a per-layer metadata
        dict. In the dict form, indexer layers have their own
        DeepseekV32IndexerMetadata whose slot mapping is stored as
        ``slot_mapping``. Latent/SFA metadata instead carries the group-1 slot
        mapping as ``indexer_slot_mapping``.
        """
        if isinstance(attn_metadata, dict):
            if layer_name is not None:
                meta = attn_metadata.get(layer_name)
                if meta is not None:
                    slot_mapping = getattr(
                        meta, "indexer_slot_mapping", None
                    )
                    if slot_mapping is not None:
                        return slot_mapping
                    slot_mapping = getattr(meta, "slot_mapping", None)
                    if slot_mapping is not None:
                        return slot_mapping

                latent_layer_name = layer_name.replace(
                    ".indexer.k_cache", ".attn"
                )
                latent_meta = attn_metadata.get(latent_layer_name)
                if latent_meta is not None:
                    slot_mapping = getattr(
                        latent_meta, "indexer_slot_mapping", None
                    )
                    if slot_mapping is not None:
                        return slot_mapping

            for name, meta in attn_metadata.items():
                if "indexer" not in name:
                    continue
                slot_mapping = getattr(meta, "slot_mapping", None)
                if slot_mapping is not None:
                    return slot_mapping

            for meta in attn_metadata.values():
                slot_mapping = getattr(meta, "indexer_slot_mapping", None)
                if slot_mapping is not None:
                    return slot_mapping
            return None

        slot_mapping = getattr(attn_metadata, "indexer_slot_mapping", None)
        if slot_mapping is not None:
            return slot_mapping
        # This helper is used only for the independent Group-1 address space.
        # A generic slot_mapping belongs to Group 0 on single-object SFA
        # metadata; accepting it here silently stores index rows into latent
        # blocks. Legacy shared-block mode does not use this two-group helper.
        return None

    @staticmethod
    def _pad_chunk_local_slot_mapping(
        slot_mapping: torch.Tensor,
        total_tokens: int,
        token_offset: int,
    ) -> torch.Tensor:
        """Convert a chunk-local slot mapping to token-sequence coordinates.

        LMCache store_layer returns absolute token ranges, e.g. [4096, 8192)
        for the second chunked-prefill step. vLLM's per-layer indexer metadata
        may only carry the current chunk's slot mapping of length 4096. Pad the
        leading range with dummy values so later slot_mapping[start:end] slicing
        returns the chunk-local mapping.
        """
        if token_offset <= 0 or len(slot_mapping) >= total_tokens:
            return slot_mapping

        expected_local_tokens = total_tokens - token_offset
        if len(slot_mapping) != expected_local_tokens:
            return slot_mapping

        padded = torch.empty(
            total_tokens, device=slot_mapping.device, dtype=slot_mapping.dtype
        )
        padded[:token_offset] = 0
        padded[token_offset:] = slot_mapping
        return padded

    def _indexer_retrieve_slot_mapping(
        self,
        attn_metadata,
        lmcache_cached_tokens: int,
        layer_name: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Return the indexer group's slot mapping for prefix retrieve.

        Mirrors the save path's indexer slot logic and handles both vLLM's
        per-layer metadata dict and single-object metadata forms.
        """
        candidates: list[tuple[str, torch.Tensor]] = []

        def add_candidate(source: str, slot_mapping) -> None:
            if isinstance(slot_mapping, torch.Tensor):
                candidates.append((source, slot_mapping))

        if isinstance(attn_metadata, dict):
            if layer_name is not None:
                latent_layer_name = layer_name.replace(
                    ".indexer.k_cache", ".attn"
                )
                latent_meta = attn_metadata.get(latent_layer_name)
                if latent_meta is not None:
                    add_candidate(
                        "latent_meta.indexer_slot_mapping",
                        getattr(latent_meta, "indexer_slot_mapping", None),
                    )

                meta = attn_metadata.get(layer_name)
                if meta is not None:
                    add_candidate(
                        "indexer_meta.indexer_slot_mapping",
                        getattr(meta, "indexer_slot_mapping", None),
                    )
                    add_candidate(
                        "indexer_meta.slot_mapping",
                        getattr(meta, "slot_mapping", None),
                    )

            for name, meta in attn_metadata.items():
                if "indexer" in name:
                    continue
                add_candidate(
                    "any_latent_meta.indexer_slot_mapping",
                    getattr(meta, "indexer_slot_mapping", None),
                )

            for name, meta in attn_metadata.items():
                if "indexer" not in name:
                    continue
                add_candidate(
                    "any_indexer_meta.indexer_slot_mapping",
                    getattr(meta, "indexer_slot_mapping", None),
                )
                add_candidate(
                    "any_indexer_meta.slot_mapping",
                    getattr(meta, "slot_mapping", None),
                )
        else:
            add_candidate(
                "attn_metadata.indexer_slot_mapping",
                getattr(attn_metadata, "indexer_slot_mapping", None),
            )

        idx_slot = None
        for _, candidate in candidates:
            if len(candidate) >= lmcache_cached_tokens:
                idx_slot = candidate
                break

        if idx_slot is None:
            return None
        idx_slot = idx_slot.to(device=self.device, dtype=torch.long)
        if lmcache_cached_tokens < len(idx_slot):
            idx_slot = idx_slot[:lmcache_cached_tokens]
        return idx_slot

    def _indexer_save_slot_mapping(
        self,
        request: "ReqMeta",
        attn_metadata,
        layer_name: Optional[str],
        token_count: int,
    ) -> Optional[torch.Tensor]:
        """Return generic indexer save slots from active layer metadata.

        Sparse layerwise saves bypass this helper and use the scheduler's exact
        per-request save window. Other paths retain the existing metadata-based
        behavior.
        """
        # ReqMeta builds this mapping directly from Group-1 block ids for the
        # active scheduler step. It is the authoritative source for every
        # two-group store, not only decode-window stores. Attention metadata
        # is a compatibility source because single-object metadata also has a
        # generic Group-0 slot_mapping that must never address Group 1.
        if request.indexer_slot_mapping:
            return request.indexer_slot_mapping[0]
        _ = token_count
        return self._indexer_slot_mapping_from_attn_metadata(
            attn_metadata, layer_name
        )

    def _sparse_indexer_slot_mapping(
        self,
        attn_metadata,
        latent_sparse_slots: torch.Tensor,
        lmcache_cached_tokens: int,
        request_indexer_slots: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Indexer slots for sparse decode, covering the full LMCache-hit prefix.

        The latent group loads only selected top-k rows into a compact scratch
        window, but the DSA index group must be fully materialized before top-k
        selection. Capping index slots to the latent scratch window leaves most
        prompt index rows stale and degrades sparse decode quality.
        """
        sparse_len = len(latent_sparse_slots)
        indexer_len = int(lmcache_cached_tokens)
        if request_indexer_slots is not None and request_indexer_slots.numel() > 0:
            request_indexer_slots = request_indexer_slots.to(
                device=self.device, dtype=torch.long
            )
            if request_indexer_slots.numel() >= indexer_len:
                return request_indexer_slots[:indexer_len]

        idx_slot = self._indexer_retrieve_slot_mapping(
            attn_metadata, lmcache_cached_tokens
        )
        if idx_slot is not None and idx_slot.numel() >= indexer_len:
            return idx_slot[:indexer_len]
        request_len = (
            int(request_indexer_slots.numel())
            if request_indexer_slots is not None
            else 0
        )
        metadata_len = int(idx_slot.numel()) if idx_slot is not None else 0
        raise RuntimeError(
            "Sparse decode with dsa_two_groups=true could not resolve "
            "the full Group-1 index slot mapping. Refusing to fall back "
            "to Group-0 compact scratch slots because that would load "
            "indexer KV into the wrong cache address space: "
            f"indexer_len={indexer_len}, sparse_len={sparse_len}, "
            f"request_indexer_slots={request_len}, "
            f"metadata_indexer_slots={metadata_len}, "
            f"lmcache_cached_tokens={lmcache_cached_tokens}"
        )

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

        self._refresh_kvcaches_list()
        self._build_kv_layer_groups()

    ####################
    # Worker side APIs
    ####################
    @staticmethod
    def _load_tokens_for_retrieve(
        tokens: list[int], lmcache_cached_tokens: int, *, is_sparse_decode: bool
    ) -> list[int]:
        """Return token ids for retrieve without redundant list copy on decode."""
        if is_sparse_decode:
            # Sparse decode scatters into a compact scratch slot window, but
            # selected_tokens can point anywhere in the cached prefix. Retrieve
            # metadata and cached chunk pointers must therefore cover the full
            # LMCache-hit prefix, not only the scratch window length.
            if 0 < lmcache_cached_tokens < len(tokens):
                return tokens[:lmcache_cached_tokens]
            return tokens
        if lmcache_cached_tokens >= len(tokens):
            return tokens
        return tokens[:lmcache_cached_tokens]

    @staticmethod
    def _load_token_mask_for_retrieve(
        request: "ReqMeta",
        token_count: int,
        lmcache_chunk_size: int,
    ) -> Optional[torch.Tensor]:
        """Build or reuse the token mask for a retrieve call."""
        if (
            request.is_sparse_decode
            and request.load_spec is not None
            and request.load_spec.vllm_cached_tokens <= 0
        ):
            request.decode_token_mask = None
            return None

        if request.is_sparse_decode and request.decode_token_mask is not None:
            mask = request.decode_token_mask
            if mask.numel() == token_count:
                token_mask = mask.clone()
            else:
                token_mask = torch.ones(token_count, dtype=torch.bool)
        else:
            token_mask = torch.ones(token_count, dtype=torch.bool)

        if request.load_spec is not None:
            prefix_tokens = request.load_spec.vllm_cached_tokens
            # Sparse decode still needs LMCache chunks for the selected prefix
            # tokens. lmcache_cached_tokens means "available in LMCache", not
            # "already resident in vLLM", so do not mask it out here.
            prefix_tokens = min(prefix_tokens, token_count)
            masked_token_count = (
                prefix_tokens
                // lmcache_chunk_size
                * lmcache_chunk_size
            )
            if masked_token_count:
                token_mask[:masked_token_count] = False

        if request.is_sparse_decode:
            request.decode_token_mask = token_mask
        return token_mask

    @staticmethod
    def _full_hit_recalc_last_token(
        load_spec: Optional[LoadSpec],
        prompt_len: int,
        *,
        is_sparse_decode: bool,
    ) -> bool:
        """True when vLLM expects the last prompt token to be recomputed, not loaded."""
        if is_sparse_decode or load_spec is None:
            return False
        return (
            load_spec.lmcache_cached_tokens >= prompt_len
            and load_spec.lmcache_cached_tokens > load_spec.vllm_cached_tokens
        )

    @staticmethod
    def _trim_prefill_for_recalc_last(
        request: "ReqMeta",
        retrieve_tokens: list[int],
        slot_mapping: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        """Handle vLLM recalc_last=1 on a full-cache-hit prefill retrieve.

        We intentionally do NOT trim retrieve_tokens or slot_mapping. Rationale:

        Chunk keys hash the chunk's tokens (token_database._prefix_hash yields
        the hash AFTER each chunk), so the last partial chunk's key depends on
        its token count. The store saved the full prompt (partial =
        prompt_len % chunk_size tokens, e.g. 191 for a 18879-prompt with
        chunk_size=256). If we trimmed retrieve_tokens to prompt_len-1 here,
        the queried partial chunk would be 190 tokens and its key
        H(tokens[0:prompt_len-1]) would NOT match the stored key
        H(tokens[0:prompt_len]) -- the last partial chunk silently misses
        (the "missing 190" / "loaded 18688/18878" shortfall).

        By keeping retrieve_tokens and slot_mapping at prompt_len, the retrieve
        queries the same partial chunk the store saved (191 tokens, matching
        key) and scatters KV to all prompt_len slots. vLLM, on a full-hit with
        recalc_last=1, still recomputes the last prompt token's logits (and KV)
        -- overwriting whatever we scattered into that slot -- so loading it is
        harmless. This also keeps token_count == len(slot_mapping) so the dense
        prefill retrieve copies exactly match chunk sizes (no OOB).

        Note: this means num_retrieved_tokens (18879) will be 1 more than
        num_expected_load (18878 = lmcache_cached - recalc_last); the shortfall
        guard uses a strict `<`, so no false warning is emitted.
        """
        return retrieve_tokens, slot_mapping

    def _drain_layerwise_retrievers(self, *, finish_dense: bool = True) -> None:
        """Finish suspended layerwise generators to avoid GC cost on reset."""
        try:
            for idx, retriever_pair in enumerate(self.layerwise_retrievers):
                is_sparse = (
                    idx < len(self._layerwise_retriever_is_sparse)
                    and self._layerwise_retriever_is_sparse[idx]
                )
                for retriever in retriever_pair:
                    if retriever is None:
                        continue
                    if is_sparse or not finish_dense:
                        self._close_layerwise_retriever(retriever)
                        continue
                    for _ in retriever:
                        pass
        finally:
            for retriever_pair in self.layerwise_retrievers:
                for retriever in retriever_pair:
                    if retriever is not None:
                        self._close_layerwise_retriever(retriever)
            self.layerwise_retrievers.clear()
            if hasattr(self, "_layerwise_requests"):
                self._layerwise_requests.clear()
            self._layerwise_retriever_is_sparse.clear()
            if hasattr(self, "_layerwise_sparse_shared_ordered"):
                self._layerwise_sparse_shared_ordered.clear()
            if hasattr(self, "_layerwise_sparse_req_ids"):
                self._layerwise_sparse_req_ids.clear()
            self._layerwise_sparse_row_groups_key = None
            self._layerwise_sparse_row_groups = None
            if hasattr(self, "_layerwise_waited_groups"):
                self._layerwise_waited_groups.clear()
            if hasattr(self, "_layerwise_sparse_indexer_sent_layers"):
                self._layerwise_sparse_indexer_sent_layers.clear()
            self._layerwise_required_wait_groups_cache = None

    @staticmethod
    def _close_layerwise_retriever(
        retriever: Generator[Any, Any, Any],
    ) -> None:
        """Close a suspended layerwise retriever during step cleanup."""
        try:
            retriever.close()
        except (GeneratorExit, RuntimeError, ValueError):
            pass

    def _abort_layerwise_retrieve_step(
        self,
        requests: Iterable[ReqMeta],
    ) -> None:
        """Release a partially constructed layerwise retrieve step."""
        for request in requests:
            self._cold_perf_dense_load_started.pop(request.req_id, None)
            self._cold_perf_dense_load_completed.pop(request.req_id, None)
            perf_state = self._cold_perf_load_started.pop(
                request.req_id,
                None,
            )
            if perf_state is not None:
                request_started, token_count = perf_state
                cold_start_perf_log(
                    logger,
                    "worker_load_abort",
                    started=request_started,
                    req_id=request.req_id,
                    tokens=token_count,
                )
            state = self._worker_retrieve_state.get(request.req_id)
            if state is not None:
                self._release_unadopted_shared_request_objects(state, request)
            self._drop_worker_retrieve_state(request.req_id)
        self._drain_layerwise_retrievers(finish_dense=False)

    def _should_defer_lookup_unpin_for_sparse_decode(self, request: ReqMeta) -> bool:
        """Keep lookup pins across decode steps while sparse retrieve is active."""
        return (
            getattr(request, "is_sparse_decode", False)
            and request.load_spec is not None
            and request.load_spec.can_load
        )

    @staticmethod
    def _is_decode_window_save_request(request: ReqMeta) -> bool:
        return bool(getattr(request, "is_decode_window_save", False))

    def _windowed_sparse_layerwise_save_enabled(self) -> bool:
        config = getattr(self, "config", None)
        use_layerwise = getattr(
            self,
            "use_layerwise",
            getattr(config, "use_layerwise", False),
        )
        return bool(
            use_layerwise
            and getattr(self, "enable_sparse_attention", False)
        )

    def _windowed_sparse_save_mapping(
        self,
        request: ReqMeta,
        kv_group: int,
        expected_base: int,
    ) -> Optional[torch.Tensor]:
        """Return the request-local save mapping for the selected KV group."""
        if not self._windowed_sparse_layerwise_save_enabled():
            return None
        if not bool(getattr(request, "windowed_sparse_save", False)):
            return None
        # Disaggregated producers intentionally resend the whole prefix. Use
        # their request-owned full mapping rather than batched attention
        # metadata, which can belong to another request in the same forward.
        if (
            self.kv_role == "kv_producer"
            and not self._is_decode_window_save_request(request)
        ):
            mapping_attr = (
                "indexer_slot_mapping" if kv_group == 1 else "slot_mapping"
            )
            mappings = getattr(request, mapping_attr, None)
            base = 0
        else:
            mapping_attr = (
                "save_indexer_slot_mapping"
                if kv_group == 1
                else "save_slot_mapping"
            )
            mappings = getattr(request, mapping_attr, None)
            base = getattr(request, "save_slot_mapping_base", None)
        if not mappings or base is None:
            raise RuntimeError(
                "Sparse layerwise save is missing its request-local slot mapping: "
                f"req_id={request.req_id}, kv_group={kv_group}, "
                f"mapping_attr={mapping_attr}, base={base}"
            )
        if int(base) != int(expected_base):
            raise RuntimeError(
                "Sparse layerwise save slot-mapping base does not match the "
                "store range: "
                f"req_id={request.req_id}, kv_group={kv_group}, "
                f"mapping_base={base}, store_base={expected_base}"
            )
        expected_tokens = len(request.token_ids) - int(base)
        mapping = mappings[0]
        if mapping.numel() != expected_tokens:
            raise RuntimeError(
                "Sparse layerwise save slot mapping has the wrong length: "
                f"req_id={request.req_id}, kv_group={kv_group}, "
                f"mapping_tokens={mapping.numel()}, expected_tokens={expected_tokens}, "
                f"range=[{base}, {len(request.token_ids)})"
            )
        return mapping

    def _layerwise_save_range(self, request: ReqMeta) -> tuple[int, int]:
        if self._is_decode_window_save_request(request):
            return (
                int(getattr(request, "decode_window_start", 0) or 0),
                int(getattr(request, "decode_window_end", len(request.token_ids))),
            )

        start = 0
        save_spec = request.save_spec
        if self.kv_role != "kv_producer" and save_spec is not None:
            start = save_spec.skip_leading_tokens
            start = start // self._lmcache_chunk_size * self._lmcache_chunk_size
        return start, len(request.token_ids)

    def _layerwise_save_storer_key(
        self, request: ReqMeta, kv_group: int = 0
    ) -> LayerwiseSaveKey:
        start, end = self._layerwise_save_range(request)
        kind = (
            "decode_window_save"
            if self._is_decode_window_save_request(request)
            else "normal_save"
        )
        return request.req_id, kind, kv_group, start, end

    def _clear_decode_window_save_groups_for_req(self, req_id: str) -> None:
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is not None:
            for group_key in list(groups):
                if group_key[0] == req_id:
                    groups.discard(group_key)
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is not None:
            expected.pop(req_id, None)

    def _clear_prefill_save_groups_for_req(self, req_id: str) -> None:
        completed = getattr(self, "_prefill_save_completed_groups", None)
        if completed is not None:
            for group_key in list(completed):
                if group_key[0] == req_id:
                    completed.pop(group_key, None)

    def _clear_decode_window_save_groups_for_window(
        self,
        request: ReqMeta,
    ) -> None:
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return
        for kv_group in self._decode_window_save_required_groups(request):
            groups.discard(self._layerwise_save_storer_key(request, kv_group))

    def _record_decode_window_save_group_completed(
        self,
        request: ReqMeta,
        kv_group: int,
        result: Optional[LayerwiseStoreResult] = None,
    ) -> None:
        if not self._is_decode_window_save_request(request):
            return
        if self._decode_window_save_uses_shared_cpu() and (
            result is None
            or not self._decode_window_save_group_pointer_ready(
                request,
                kv_group,
                result,
            )
        ):
            logger.debug(
                "Decode-window save group lacks complete shared CPU pointer "
                "coverage: req_id=%s kv_group=%s",
                request.req_id,
                kv_group,
            )
            return
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return
        groups.add(self._layerwise_save_storer_key(request, kv_group))
        _mtp_dw_event(
            "store",
            req=request.req_id,
            event="group_complete",
            frontier=len(request.token_ids),
            window_start=request.decode_window_start,
            window_end=request.decode_window_end,
            kv_group=kv_group,
            required_groups=sorted(
                self._decode_window_save_required_groups(request)
            ),
        )

    def _note_decode_window_save_seen(self, request: ReqMeta) -> None:
        if not self._is_decode_window_save_request(request):
            return
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is None:
            return
        window_start = getattr(request, "decode_window_start", None)
        if window_start is None:
            return
        window_start = int(window_start)
        expected.setdefault(request.req_id, window_start)

    def _decode_window_save_is_next_expected(self, request: ReqMeta) -> bool:
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is None:
            return True
        window_start = getattr(request, "decode_window_start", None)
        if window_start is None:
            return True
        self._note_decode_window_save_seen(request)
        return int(window_start) == int(expected.get(request.req_id, window_start))

    def _decode_window_save_required_groups(self, request: ReqMeta) -> set[int]:
        if not self._is_decode_window_save_request(request):
            return set()
        save_spec = request.save_spec
        if save_spec is None:
            return set()
        can_save_latent = getattr(
            save_spec, "can_save_latent", getattr(save_spec, "can_save", False)
        )
        can_save_indexer = getattr(save_spec, "can_save_indexer", False)
        if getattr(self.config, "dsa_two_groups", False):
            return {0, 1} if can_save_latent or can_save_indexer else set()
        required: set[int] = set()
        if can_save_latent:
            required.add(0)
        return required

    def _decode_window_save_has_required_groups(self, request: ReqMeta) -> bool:
        required = self._decode_window_save_required_groups(request)
        if not required:
            return False
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return False
        return all(
            self._layerwise_save_storer_key(request, kv_group) in groups
            for kv_group in required
        )

    def _decode_window_save_uses_shared_cpu(self) -> bool:
        engine = getattr(self, "lmcache_engine", None)
        return bool(getattr(engine, "enable_shared_cpu_cache", False))

    def _decode_window_save_group_pointer_ready(
        self,
        request: ReqMeta,
        kv_group: int,
        result: LayerwiseStoreResult,
    ) -> bool:
        if result.request_id != request.req_id or result.kv_group != kv_group:
            return False
        starts = result.starts
        ends = result.ends
        memory_objs = result.memory_objs
        chunk_ptrs = result.chunk_ptrs
        window_start = getattr(request, "decode_window_start", None)
        window_end = getattr(request, "decode_window_end", None)
        if window_start is not None and window_end is not None:
            if not self._cached_ranges_cover_interval(
                starts,
                ends,
                int(window_start),
                int(window_end),
            ):
                return False
        required_chunks = self._shared_required_chunk_count(
            starts,
            ends,
            memory_objs,
        )
        if required_chunks <= 0:
            return False
        expected_layers = int(getattr(self, "num_layers", 0) or 0)
        if expected_layers <= 0:
            expected_layers = len(memory_objs or [])
        if expected_layers <= 0:
            return False
        if self._missing_shared_layer_cache_coverage(
            memory_objs,
            expected_layers,
            required_chunks,
        ):
            return False
        return not self._missing_shared_pointer_cache_layers(
            memory_objs,
            chunk_ptrs,
            required_chunks,
        )

    def _drop_layerwise_save_storers(self, req_id: str) -> None:
        for storer_key in list(self._layerwise_save_storers):
            if storer_key[0] != req_id:
                continue
            self._close_layerwise_storer(
                self._layerwise_save_storers.pop(storer_key, None)
            )
        self._clear_decode_window_save_groups_for_req(req_id)
        self._clear_prefill_save_groups_for_req(req_id)

        for pending_key in list(self._deferred_latent_pending):
            if pending_key[0] == req_id:
                self._deferred_latent_pending.discard(pending_key)

    def _abort_save_step(self, requests: Iterable[ReqMeta]) -> None:
        """Discard partial stores without advancing decode-window progress."""
        expected = getattr(self, "_decode_window_save_expected_start", None)
        saved_expected = dict(expected or {})
        for request in requests:
            self._drop_layerwise_save_storers(request.req_id)
        if expected is not None:
            expected.update(saved_expected)

    def _release_request_lookup_pins(self, req_id: str) -> None:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        engine = manager.lmcache_engine
        if engine is not None:
            engine.lookup_unpin(req_id)

    def _maybe_lookup_unpin_for_request(self, request: ReqMeta) -> None:
        if self._is_decode_window_save_request(request):
            return
        if self._should_defer_lookup_unpin_for_sparse_decode(request):
            return
        self._release_request_lookup_pins(request.req_id)

    def _mark_decode_window_save_completed(self, request: ReqMeta) -> None:
        if not self._is_decode_window_save_request(request):
            return
        self._note_decode_window_save_seen(request)
        engine = self.lmcache_engine
        is_passive = getattr(engine, "_is_passive", None)
        if callable(is_passive) and is_passive():
            return
        if not self._decode_window_save_has_required_groups(request):
            if _mtp_dw_deep_diag_enabled():
                required_groups = self._decode_window_save_required_groups(request)
                completed_state = getattr(
                    self, "_decode_window_save_completed_groups", set()
                )
                completed_groups = sorted(
                    kv_group
                    for kv_group in required_groups
                    if self._layerwise_save_storer_key(request, kv_group)
                    in completed_state
                )
                metadata = getattr(engine, "metadata", None)
                worker_rank = getattr(metadata, "worker_id", None)
                if completed_groups:
                    seen = getattr(self, "_mtp_dw_deep_window_group_wait_seen", None)
                    if seen is None:
                        seen = set()
                        self._mtp_dw_deep_window_group_wait_seen = seen
                    key = (
                        request.req_id,
                        request.decode_window_start,
                        request.decode_window_end,
                    )
                    if key not in seen:
                        if len(seen) >= 256:
                            seen.pop()
                        seen.add(key)
                        _mtp_dw_event(
                            "deep",
                            event="window_group_wait",
                            req=request.req_id,
                            worker_rank=worker_rank,
                            tp_rank=worker_rank,
                            tp_world=getattr(metadata, "world_size", None),
                            frontier=len(request.token_ids),
                            window_start=request.decode_window_start,
                            window_end=request.decode_window_end,
                            kv_group=None,
                            required_groups=sorted(required_groups),
                            completed_groups=completed_groups,
                            missing_groups=sorted(
                                required_groups - set(completed_groups)
                            ),
                        )
            logger.debug(
                "Decode-window save not marked complete before required "
                "store groups are seen: req_id=%s required_groups=%s",
                request.req_id,
                sorted(self._decode_window_save_required_groups(request)),
            )
            return
        if not self._decode_window_save_is_next_expected(request):
            expected = getattr(self, "_decode_window_save_expected_start", {})
            logger.debug(
                "Decode-window save not marked complete out of order: "
                "req_id=%s window_start=%s expected_start=%s",
                request.req_id,
                getattr(request, "decode_window_start", None),
                expected.get(request.req_id),
            )
            return
        window_end = request.decode_window_end
        if window_end is None:
            return
        completed = getattr(self, "_completed_decode_window_saves", None)
        if completed is None:
            return
        completed[request.req_id] = max(completed.get(request.req_id, 0), window_end)
        _mtp_dw_event(
            "commit",
            req=request.req_id,
            event="publish_completed",
            frontier=len(request.token_ids),
            window_start=request.decode_window_start,
            window_end=window_end,
            required_groups=sorted(
                self._decode_window_save_required_groups(request)
            ),
            completed_end=completed[request.req_id],
        )
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is not None:
            expected[request.req_id] = max(
                int(expected.get(request.req_id, 0)),
                int(window_end),
            )
        self._clear_decode_window_save_groups_for_window(request)

    def _prefill_save_required_groups(self, request: ReqMeta) -> set[int]:
        save_spec = request.save_spec
        if save_spec is None:
            return {0} if self.kv_role == "kv_producer" else set()
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return set()
        if not getattr(self.config, "dsa_two_groups", False):
            return {0}
        if save_spec.can_save_latent or save_spec.can_save_indexer:
            # A two-group frontier is publishable only as a coherent pair.
            # Per-group capability flags control submission, not commit
            # semantics; accepting one completed group can expose mixed data
            # from different executions under the same prefix frontier.
            return {0, 1}
        return set()

    def _record_prefill_save_group_completed(
        self,
        request: ReqMeta,
        kv_group: int,
        result: Optional[LayerwiseStoreResult],
    ) -> None:
        if (
            result is None
            or result.committed_end <= 0
            or not request.is_last_prefill
            or request.is_sparse_decode
            or self._is_decode_window_save_request(request)
        ):
            return
        completed = getattr(self, "_prefill_save_completed_groups", None)
        if completed is not None:
            completed[self._layerwise_save_storer_key(request, kv_group)] = (
                result.committed_end
            )

    def _mark_prefill_committed(
        self,
        request: ReqMeta,
        committed_end: Optional[int] = None,
    ) -> None:
        """Publish the full-chunk prefill frontier after its save completes."""
        if (
            not request.is_last_prefill
            or request.is_sparse_decode
            or self._is_decode_window_save_request(request)
        ):
            return
        engine = getattr(self, "lmcache_engine", None)
        is_passive = getattr(engine, "_is_passive", None)
        if callable(is_passive) and is_passive():
            return
        if committed_end is None:
            required = self._prefill_save_required_groups(request)
            groups = getattr(self, "_prefill_save_completed_groups", {})
            keys = [
                self._layerwise_save_storer_key(request, kv_group)
                for kv_group in required
            ]
            if not keys or not all(key in groups for key in keys):
                return
            committed_end = min(groups.pop(key) for key in keys)
        committed_end = (
            min(int(committed_end), len(request.token_ids))
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        completed = getattr(self, "_completed_decode_window_saves", None)
        if completed is not None and committed_end > 0:
            completed[request.req_id] = max(
                completed.get(request.req_id, 0), committed_end
            )

    def _mark_initial_sparse_release_ready(self, request: ReqMeta) -> None:
        """Publish the cache-hit frontier after the first sparse step completes."""
        if not request.is_sparse_decode or request.load_spec is None:
            return
        committed_end = request.load_spec.dsa_release_frontier
        if committed_end is None or committed_end <= 0:
            return
        published = getattr(self, "_initial_sparse_release_published", None)
        if published is None:
            published = set()
            self._initial_sparse_release_published = published
        if request.resumed_from_preemption and not getattr(
            request,
            "_initial_sparse_release_rearmed",
            False,
        ):
            # The request ID is unchanged, but vLLM allocated a new block
            # table. Re-arm the one-shot release after its sparse recovery
            # load completes.
            published.discard(request.req_id)
            request._initial_sparse_release_rearmed = True
        if request.req_id in published:
            return
        completed = getattr(self, "_completed_decode_window_saves", None)
        if (
            completed is None
            or committed_end <= request.dsa_current_released_frontier
            or committed_end <= request.load_spec.dsa_current_released_frontier
        ):
            return
        completed[request.req_id] = max(
            completed.get(request.req_id, 0), int(committed_end)
        )
        published.add(request.req_id)

    def get_completed_decode_window_saves(self) -> dict[str, int]:
        completed = getattr(self, "_completed_decode_window_saves", None)
        if not completed:
            return {}
        drained = dict(completed)
        completed.clear()
        return drained

    def update_connector_output(self, connector_output: Any) -> None:
        validation_blocks = getattr(
            self, "_dsa_cold_indexer_block_ids", None
        )
        if validation_blocks is not None:
            cold_loaded = getattr(self, "_dsa_cold_loaded_req_ids", None)
            if cold_loaded is None:
                cold_loaded = set()
                self._dsa_cold_loaded_req_ids = cold_loaded
            invalid_blocks = set(
                getattr(connector_output, "invalid_block_ids", None) or ()
            )
            for req_id in (
                getattr(connector_output, "finished_recving", None) or ()
            ):
                load_spec = self.load_specs.get(req_id)
                request_blocks = validation_blocks.pop(req_id, None)
                failed = request_blocks is None or bool(
                    request_blocks.intersection(invalid_blocks)
                )
                if (
                    not failed
                    and load_spec is not None
                    and getattr(load_spec, "dsa_cold_compact_load", False)
                ):
                    cold_loaded.add(req_id)
            if not validation_blocks:
                del self._dsa_cold_indexer_block_ids
        completed = getattr(connector_output, "completed_decode_window_saves", None)
        if not completed:
            return
        published: dict[str, int] = {}
        for req_id, window_end in completed.items():
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                continue
            committed_end = int(window_end)
            window_size = int(getattr(self, "_decode_window_save_window_size", 0) or 0)
            prefill_end = (
                tracker.prompt_len
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            is_initial_frontier = (
                window_size > 0 and committed_end == prefill_end
            )
            if window_size > 0 and tracker.decode_window_save_next_start is None:
                if committed_end != prefill_end:
                    logger.debug(
                        "Ignoring completion before the initial prefill "
                        "frontier: req_id=%s completed_end=%s expected=%s",
                        req_id,
                        window_end,
                        prefill_end,
                    )
                    continue
                tracker.decode_window_save_anchor = prefill_end
                tracker.decode_window_save_next_start = prefill_end
            if committed_end > len(tracker.token_ids):
                raise RuntimeError(
                    f"LMCache committed_end={committed_end} exceeds request "
                    f"frontier={len(tracker.token_ids)} for request {req_id}."
                )
            initial_cached_end = (
                tracker.num_lmcache_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            if (
                window_size > 0
                and initial_cached_end > 0
                and committed_end == initial_cached_end
                and tracker.decode_window_save_next_start is not None
                and committed_end < int(tracker.decode_window_save_next_start)
            ):
                # The first sparse step confirms that the externally loaded
                # prefix can be released. It may arrive after decode-window
                # tracking has already advanced to the full appended prompt.
                tracker.decode_window_save_committed_end = max(
                    tracker.decode_window_save_committed_end,
                    committed_end,
                )
                release_end = self._eligible_dsa_release_frontier(
                    tracker,
                    committed_end,
                )
                if release_end > tracker.dsa_current_released_frontier:
                    tracker.dsa_current_released_frontier = release_end
                    tracker.dsa_nonresident_frontier = max(
                        tracker.dsa_nonresident_frontier,
                        release_end,
                    )
                    published[req_id] = release_end
                continue
            if (
                tracker.decode_window_save_next_start is not None
                and committed_end
                != int(tracker.decode_window_save_next_start)
            ):
                raise RuntimeError(
                    f"LMCache completed unexpected frontier {committed_end} "
                    f"for request {req_id}; expected emitted frontier "
                    f"{tracker.decode_window_save_next_start}."
                )
            if committed_end % self._lmcache_chunk_size:
                raise RuntimeError(
                    f"LMCache reported unaligned committed_end={committed_end} "
                    f"for request {req_id}; "
                    f"lmcache_chunk_size={self._lmcache_chunk_size}."
                )
            anchor = tracker.decode_window_save_anchor
            if anchor is None:
                anchor = (
                    tracker.prompt_len
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                tracker.decode_window_save_anchor = anchor
            if (
                window_size > 0
                and anchor is not None
                and (committed_end - anchor) % window_size
            ):
                raise RuntimeError(
                    f"LMCache committed_end={committed_end} is not on the "
                    f"decode-window lattice anchor={anchor}, window_size={window_size} "
                    f"for request {req_id}."
                )
            # Acknowledge the physical save immediately so another window can
            # be emitted even when committed_end/release are intentionally
            # delayed.
            if (
                tracker.decode_window_save_inflight_end is not None
                and committed_end >= tracker.decode_window_save_inflight_end
            ):
                tracker.decode_window_save_inflight_end = None

            delay_windows = int(
                getattr(
                    self,
                    "_decode_window_save_commit_delay_windows",
                    0,
                )
                or 0
            )
            publish_end = publish_delayed_decode_window_commit(
                tracker.decode_window_save_pending_commits,
                committed_end,
                delay_windows if window_size > 0 else 0,
                is_initial_frontier=is_initial_frontier,
            )

            if publish_end is None:
                _mtp_dw_event(
                    "commit",
                    req=req_id,
                    event="frontier_delayed",
                    frontier=len(tracker.token_ids),
                    completed_end=committed_end,
                    committed_end=tracker.decode_window_save_committed_end,
                    pending_windows=len(
                        tracker.decode_window_save_pending_commits
                    ),
                    delay_windows=delay_windows,
                )
                continue

            committed_before = tracker.decode_window_save_committed_end
            tracker.decode_window_save_committed_end = max(
                committed_before, publish_end
            )
            release_end = self._eligible_dsa_release_frontier(
                tracker,
                publish_end,
            )
            if release_end > tracker.dsa_current_released_frontier:
                tracker.dsa_current_released_frontier = release_end
                tracker.dsa_nonresident_frontier = max(
                    tracker.dsa_nonresident_frontier,
                    release_end,
                )
                published[req_id] = release_end
            _mtp_dw_event(
                "commit",
                req=req_id,
                event="frontier_update",
                frontier=len(tracker.token_ids),
                window_start=max(
                    0,
                    tracker.decode_window_save_committed_end - window_size,
                ),
                window_end=tracker.decode_window_save_committed_end,
                committed_before=committed_before,
                committed_after=tracker.decode_window_save_committed_end,
                completed_end=int(window_end),
                published_end=publish_end,
            )
            if tracker.decode_window_save_committed_end < committed_before:
                _mtp_dw_event(
                    "fail",
                    req=req_id,
                    frontier=len(tracker.token_ids),
                    invariant="committed_frontier_monotonic",
                    committed_before=committed_before,
                    committed_after=tracker.decode_window_save_committed_end,
                )
        # The vLLM scheduler consumes this same mapping after this callback to
        # release local blocks. Replacing raw completions with delayed
        # frontiers keeps release and split_boundary on the same commit point.
        completed.clear()
        completed.update(published)

    def _eligible_dsa_release_frontier(
        self,
        tracker: RequestTracker,
        committed_end: int,
    ) -> int:
        """Return a safe release authorization for a persisted frontier."""
        policy_threshold = int(
            getattr(self, "_dsa_kv_policy_threshold", 0) or 0
        )
        release_gate = max(self._dsa_scratch_capacity, policy_threshold)
        sparse_active = bool(
            tracker.dsa_nonresident_frontier > 0
            or len(tracker.token_ids) > policy_threshold
        )
        if not sparse_active or committed_end <= release_gate:
            return 0
        return int(committed_end)

    def _mark_worker_retrieve_registry_changed(self) -> None:
        self._worker_retrieve_registry_version = (
            getattr(self, "_worker_retrieve_registry_version", 0) + 1
        )

    def _set_worker_retrieve_state(
        self, req_id: str, state: WorkerRetrieveState
    ) -> None:
        self._worker_retrieve_state[req_id] = state
        self._mark_worker_retrieve_registry_changed()

    def _prune_worker_retrieve_state(
        self,
        active_req_ids: set[str],
        resumed_req_ids: set[str] | None = None,
    ) -> None:
        if hasattr(self, "_pd_partial_restored_req_ids"):
            self._pd_partial_restored_req_ids.intersection_update(active_req_ids)
        if hasattr(self, "_initial_sparse_release_published"):
            self._initial_sparse_release_published.intersection_update(active_req_ids)
            if resumed_req_ids:
                self._initial_sparse_release_published.difference_update(
                    resumed_req_ids
                )
        for req_id in resumed_req_ids or ():
            self._clear_prefill_save_groups_for_req(req_id)
        if not hasattr(self, "_worker_retrieve_state"):
            return
        active_key = frozenset(active_req_ids)
        prune_key = (
            active_key,
            getattr(self, "_worker_retrieve_registry_version", 0),
        )
        if prune_key == getattr(self, "_worker_retrieve_last_prune_key", None):
            return
        dropped_req_ids = set(self._worker_retrieve_state) - active_req_ids
        for req_id in dropped_req_ids:
            state = self._worker_retrieve_state.get(req_id)
            shared_request_active = bool(
                state is not None and state.shared_request_active
            )
            if shared_request_active and getattr(
                state, "_dsa_cold_prune_protected", False
            ):
                # TP workers can report cold-load completion on different
                # forwards. Keep an early worker's resident indexer and shared
                # latent lease until the scheduler's authoritative request
                # finish/abort cleanup; otherwise that worker would reload the
                # indexer when the final TP lets the request resume.
                continue
            if shared_request_active:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
            if state is not None and (state.metadata_warm or state.has_cache()):
                continue
            if state is not None and not shared_request_active:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
            self._release_request_lookup_pins(req_id)
        self._worker_retrieve_state = {
            req_id: state
            for req_id, state in self._worker_retrieve_state.items()
            if req_id in active_req_ids or (state.metadata_warm or state.has_cache())
        }
        if dropped_req_ids:
            self._mark_worker_retrieve_registry_changed()
        self._worker_retrieve_last_prune_key = (
            active_key,
            getattr(self, "_worker_retrieve_registry_version", 0),
        )

    def _drop_worker_retrieve_state(self, req_id: str) -> None:
        engine = getattr(self, "lmcache_engine", None)
        state = None
        if hasattr(self, "_worker_retrieve_state"):
            state = self._worker_retrieve_state.get(req_id)
        if state is not None:
            self._release_shared_worker_retrieve_state(state, engine)
            self._worker_retrieve_state.pop(req_id, None)
            self._mark_worker_retrieve_registry_changed()
        elif engine is not None:
            release_fn = getattr(engine, "release_shared_cpu_sparse_request", None)
            if callable(release_fn):
                release_fn(req_id)
        self._release_request_lookup_pins(req_id)

    def _release_finished_worker_requests(self, req_ids: Iterable[str]) -> None:
        """Release request-owned cache state in the worker process."""
        for req_id in req_ids:
            self._cold_perf_dense_load_started.pop(req_id, None)
            self._cold_perf_dense_load_completed.pop(req_id, None)
            self._drop_layerwise_save_storers(req_id)
            self._drop_worker_retrieve_state(req_id)

    def _request_may_store_in_wait_for_save(self, request: ReqMeta) -> bool:
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return self.kv_role == "kv_producer"
        can_save = save_spec.can_save or self.kv_role == "kv_producer"
        return can_save and save_spec.skip_leading_tokens != len(request.token_ids)

    def _finalize_worker_requests_after_store(
        self,
        req_ids: set[str],
    ) -> set[str]:
        states = getattr(self, "_worker_retrieve_state", {})
        finished_sending = {
            req_id
            for req_id in req_ids
            if getattr(states.get(req_id), "_dsa_cold_prune_protected", False)
        }
        self._release_finished_worker_requests(req_ids)
        return finished_sending

    def _complete_worker_save_step(self) -> None:
        self._wait_for_save_done = True
        pending_req_ids = getattr(self, "_finished_req_ids_waiting_for_save", None)
        if not pending_req_ids:
            return
        waiting_req_ids = set(pending_req_ids)
        self._late_finished_sending.update(
            self._finalize_worker_requests_after_store(waiting_req_ids)
        )
        pending_req_ids.difference_update(waiting_req_ids)

    @staticmethod
    def _release_shared_worker_retrieve_state(
        state: WorkerRetrieveState,
        engine: Optional[Any] = None,
        *,
        release_request: bool = True,
    ) -> None:
        """Drop worker bindings and release the engine-owned request lease."""

        req_id = state.req_id
        readiness = state.dense_load_readiness
        if readiness is not None:
            synchronize = getattr(
                getattr(engine, "gpu_connector", None),
                "synchronize_dense_load_readiness",
                None,
            )
            if not callable(synchronize):
                raise RuntimeError(
                    "NPU connector has no dense load readiness sync API"
                )
            synchronize(readiness)
            state.dense_load_readiness = None
            state.dense_load_readiness_consumed = False
        LMCacheConnectorV1Impl._release_dense_load_source_owners(
            state.dense_load_source_owners,
            engine,
            synchronize=readiness is None,
        )
        state.dense_load_readiness_consumed = False
        state.dense_load_source_owners = ()
        state.prepared_sparse_sources.clear()
        for kv_group in (0, 1):
            cache = state.cache_kwargs(kv_group, dsa_two_groups=True)
            for name in (
                "cached_memory_objs",
                "cached_tensors",
                "cached_chunk_dev_ptrs",
                "cached_chunk_ptrs_npu",
                "cached_shared_handles",
            ):
                cache[name].clear()

        if release_request and engine is not None and req_id:
            release_fn = getattr(engine, "release_shared_cpu_sparse_request", None)
            if callable(release_fn):
                release_fn(req_id)

        state.shared_latent_status = "missing"
        state.shared_index_status = "missing"
        state.indexer_npu_resident = False
        state.indexer_npu_materialization_pending = False
        state.shared_generation = 0
        state.pointer_cache_generation = 0
        state.shared_request_active = False
        state.request_scope_token = None
        state.shared_validation_signature = None
        state.dense_prefix_seed = False
        state.metadata_token_ids.clear()
        state.slot_mapping = None
        state.indexer_slot_mapping = None
        state.decode_ret_mask = None
        state.req_id = None
        if hasattr(state, "_dsa_cold_prune_protected"):
            delattr(state, "_dsa_cold_prune_protected")

    @staticmethod
    def _release_dense_load_source_owners(
        owners: tuple[Any, ...],
        engine: Optional[Any],
        *,
        synchronize: bool = True,
    ) -> None:
        if not owners:
            return
        if synchronize:
            synchronize_fn = getattr(
                getattr(engine, "gpu_connector", None),
                "synchronize_dense_load_stream",
                None,
            )
            if synchronize_fn is None:
                raise RuntimeError(
                    "NPU connector has no dense load-stream sync API"
                )
            synchronize_fn()
        for owner in owners:
            unpin = getattr(owner, "unpin", None)
            if callable(unpin) and getattr(owner, "is_pinned", False):
                unpin()
            release = getattr(owner, "ref_count_down", None)
            valid = getattr(owner, "is_valid", None)
            if callable(release) and (not callable(valid) or valid()):
                release()

    @staticmethod
    def _dense_load_source_owners(state: WorkerRetrieveState) -> tuple[Any, ...]:
        owners: dict[int, Any] = {}
        for layers in (
            state.cached_memory_objs_indexer,
            state.cached_tensors_indexer,
        ):
            for layer in layers:
                for owner in layer:
                    owners.setdefault(id(owner), owner)
        return tuple(owners.values())

    @staticmethod
    def _shared_required_chunk_count(
        starts: list[int],
        ends: list[int],
        layers: list[list[Any]],
    ) -> int:
        count = max(len(starts or []), len(ends or []))
        if count > 0:
            return count
        if any(layer for layer in (layers or [])):
            return 1
        return 0

    @staticmethod
    def _cached_prefix_covered_token_count(
        starts: list[int],
        ends: list[int],
    ) -> int:
        covered = 0
        for start, end in zip(starts or [], ends or [], strict=False):
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            if start > covered:
                break
            covered = max(covered, end)
        return covered

    @staticmethod
    def _cached_ranges_cover_interval(
        starts: list[int],
        ends: list[int],
        interval_start: int,
        interval_end: int,
    ) -> bool:
        interval_start = max(0, int(interval_start))
        interval_end = max(interval_start, int(interval_end))
        if interval_end == interval_start:
            return True
        covered = interval_start
        for start, end in zip(starts or [], ends or [], strict=False):
            start = int(start)
            end = int(end)
            if end <= start or end <= covered:
                continue
            if start > covered:
                break
            covered = max(covered, end)
            if covered >= interval_end:
                return True
        return False

    @classmethod
    def _cached_ranges_cover_prefix(
        cls,
        starts: list[int],
        ends: list[int],
        token_count: int,
    ) -> bool:
        token_count = max(0, int(token_count))
        if token_count == 0:
            return True
        return cls._cached_prefix_covered_token_count(starts, ends) >= token_count

    @staticmethod
    def _shared_pointer_cache_entry_covers(
        entry: Any,
        required_chunks: int,
    ) -> bool:
        if entry is None:
            return False
        required_chunks = max(1, int(required_chunks))
        if isinstance(entry, torch.Tensor):
            return int(entry.numel()) >= required_chunks
        try:
            return len(entry) >= required_chunks
        except TypeError:
            return required_chunks == 1

    @staticmethod
    def _missing_shared_layer_cache_coverage(
        layers: list[list[Any]],
        expected_layers: int,
        required_chunks: int,
    ) -> list[int]:
        if expected_layers <= 0:
            return []
        required_chunks = max(1, int(required_chunks))
        missing = []
        layers = layers or []
        for layer_id in range(expected_layers):
            if layer_id >= len(layers):
                missing.append(layer_id)
                continue
            layer_cache = layers[layer_id]
            if layer_cache is None or len(layer_cache) < required_chunks:
                missing.append(layer_id)
        return missing

    @staticmethod
    def _missing_shared_pointer_cache_layers(
        layers: list[list[Any]],
        chunk_ptrs: list[Optional[torch.Tensor]],
        required_chunks: int = 1,
    ) -> list[int]:
        missing: list[int] = []
        for layer_id, layer_entries in enumerate(layers or []):
            if not layer_entries:
                continue
            if layer_id >= len(chunk_ptrs):
                missing.append(layer_id)
                continue
            if not LMCacheConnectorV1Impl._shared_pointer_cache_entry_covers(
                chunk_ptrs[layer_id],
                required_chunks,
            ):
                missing.append(layer_id)
        return missing

    def _validate_decode_save_shared_pointer_cache(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        required_latent_chunks = self._shared_required_chunk_count(
            state.cached_starts,
            state.cached_ends,
            state.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_pointer_cache_layers(
            state.cached_memory_objs,
            state.cached_chunk_ptrs_npu,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Decode-window save merge produced incomplete shared CPU MLA "
                "latent pointer cache before refreshing the hot request "
                "scope: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )

        if (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(request, True)
        ):
            expected_index_layers = int(getattr(self, "num_layers", 0) or 0)
            if expected_index_layers <= 0:
                expected_index_layers = len(state.cached_memory_objs_indexer or [])
            required_index_chunks = max(
                required_latent_chunks,
                self._shared_required_chunk_count(
                    state.cached_starts_indexer,
                    state.cached_ends_indexer,
                    state.cached_memory_objs_indexer,
                ),
            )
            missing_index_layers = self._missing_shared_layer_cache_coverage(
                state.cached_memory_objs_indexer,
                expected_index_layers,
                required_index_chunks,
            )
            if missing_index_layers:
                raise RuntimeError(
                    "Decode-window save merge produced incomplete shared CPU "
                    "DSA index state before refreshing the hot request scope: "
                    f"req_id={request.req_id}, kv_group=1, "
                    f"missing_layers={missing_index_layers}"
                )
            missing_index_layers = self._missing_shared_pointer_cache_layers(
                state.cached_memory_objs_indexer,
                state.cached_chunk_ptrs_npu_indexer,
                required_index_chunks,
            )
            if missing_index_layers:
                raise RuntimeError(
                    "Decode-window save merge produced incomplete shared CPU "
                    "DSA index pointer cache before refreshing the hot "
                    "request scope: "
                    f"req_id={request.req_id}, kv_group=1, "
                    f"missing_layers={missing_index_layers}"
                )

    @staticmethod
    def _copy_layer_cache(cache: list[list[Any]]) -> list[list[Any]]:
        return [
            list(layer_cache) if isinstance(layer_cache, list) else layer_cache
            for layer_cache in (cache or [])
        ]

    def _snapshot_worker_retrieve_cache_state(
        self,
        state: WorkerRetrieveState,
    ) -> dict[str, Any]:
        return {
            "cached_starts": list(state.cached_starts),
            "cached_ends": list(state.cached_ends),
            "cached_keys": self._copy_layer_cache(state.cached_keys),
            "cached_memory_objs": self._copy_layer_cache(state.cached_memory_objs),
            "cached_tensors": self._copy_layer_cache(state.cached_tensors),
            "cached_chunk_dev_ptrs": self._copy_layer_cache(
                state.cached_chunk_dev_ptrs
            ),
            "cached_chunk_ptrs_npu": list(state.cached_chunk_ptrs_npu),
            "cached_shared_handles": self._copy_layer_cache(
                state.cached_shared_handles
            ),
            "cached_starts_indexer": list(state.cached_starts_indexer),
            "cached_ends_indexer": list(state.cached_ends_indexer),
            "cached_keys_indexer": self._copy_layer_cache(
                state.cached_keys_indexer
            ),
            "cached_memory_objs_indexer": self._copy_layer_cache(
                state.cached_memory_objs_indexer
            ),
            "cached_tensors_indexer": self._copy_layer_cache(
                state.cached_tensors_indexer
            ),
            "cached_chunk_dev_ptrs_indexer": self._copy_layer_cache(
                state.cached_chunk_dev_ptrs_indexer
            ),
            "cached_chunk_ptrs_npu_indexer": list(
                state.cached_chunk_ptrs_npu_indexer
            ),
            "cached_shared_handles_indexer": self._copy_layer_cache(
                state.cached_shared_handles_indexer
            ),
            "location": state.location,
            "metadata_warm": state.metadata_warm,
            "token_count": state.token_count,
            "metadata_token_ids": list(state.metadata_token_ids),
            "slot_mapping": state.slot_mapping,
            "indexer_slot_mapping": state.indexer_slot_mapping,
            "decode_ret_mask": state.decode_ret_mask,
            "shared_generation": state.shared_generation,
            "pointer_cache_generation": state.pointer_cache_generation,
            "indexer_npu_resident": state.indexer_npu_resident,
            "indexer_npu_materialization_pending": (
                state.indexer_npu_materialization_pending
            ),
            "request_scope_token": state.request_scope_token,
            "shared_validation_signature": state.shared_validation_signature,
            "dense_prefix_seed": state.dense_prefix_seed,
            "prepared_sparse_sources": dict(state.prepared_sparse_sources),
            "dense_load_readiness": state.dense_load_readiness,
            "dense_load_readiness_consumed": (
                state.dense_load_readiness_consumed
            ),
            "dense_load_source_owners": state.dense_load_source_owners,
        }

    @staticmethod
    def _restore_worker_retrieve_cache_state(
        state: WorkerRetrieveState,
        snapshot: dict[str, Any],
    ) -> None:
        for attr, value in snapshot.items():
            setattr(state, attr, value)

    def _release_unadopted_shared_request_objects(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        engine = self.lmcache_engine
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not request.is_sparse_decode
        ):
            return

        groups = {0: state.cached_memory_objs}
        if state.cached_memory_objs_indexer:
            groups[1] = state.cached_memory_objs_indexer
        engine.release_shared_cpu_unowned_objects(request.req_id, groups)

    def _record_shared_worker_retrieve_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        previous_token_count: int = 0,
    ) -> None:
        engine = self.lmcache_engine
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not request.is_sparse_decode
        ):
            return

        def has_entries(layers: list[list[Any]]) -> bool:
            return bool(layers and any(layers))

        generation = int(getattr(engine, "shared_cpu_cache_generation", 0) or 0)
        expected_layers = int(getattr(self, "num_layers", 0) or 0)
        cold_compact = bool(
            request.load_spec is not None
            and getattr(request.load_spec, "dsa_cold_compact_load", False)
        )
        materialize_index = not cold_compact and (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(request, True)
        )
        skip_index_hot_state = self._is_dsa_two_groups() and not materialize_index

        required_latent_chunks = self._shared_required_chunk_count(
            state.cached_starts,
            state.cached_ends,
            state.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_layer_cache_coverage(
            state.cached_memory_objs,
            expected_layers,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Shared CPU sparse decode cannot mark request state "
                "hot-reusable with incomplete MLA latent state: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )

        required_index_chunks = max(
            required_latent_chunks,
            self._shared_required_chunk_count(
                state.cached_starts_indexer,
                state.cached_ends_indexer,
                state.cached_memory_objs_indexer,
            ),
        )
        missing_index_layers = self._missing_shared_layer_cache_coverage(
            state.cached_memory_objs_indexer,
            expected_layers,
            required_index_chunks,
        )
        if materialize_index and missing_index_layers:
            raise RuntimeError(
                "Shared CPU sparse decode cannot mark request state "
                "hot-reusable without complete materialized DSA index state: "
                f"req_id={request.req_id}, kv_group=1, "
                f"missing_layers={missing_index_layers}"
            )

        groups = [(0, state.cached_memory_objs, required_latent_chunks)]
        if materialize_index and state.cached_memory_objs_indexer:
            groups.append(
                (1, state.cached_memory_objs_indexer, required_index_chunks)
            )

        owned_groups: dict[int, list[list[Any]]] = {}
        for kv_group, layers, required_chunks in groups:
            if not has_entries(layers):
                continue
            required_token_count = (
                state.token_count
                if state.token_count > 0
                else self._shared_retrieve_token_count_for_request(request)
            )
            cache = state.cache_kwargs(
                kv_group,
                dsa_two_groups=self._is_dsa_two_groups(),
            )
            range_starts = cache["cached_starts"]
            range_ends = cache["cached_ends"]
            if (range_starts or range_ends) and not self._cached_ranges_cover_prefix(
                range_starts,
                range_ends,
                required_token_count,
            ):
                cached_ranges = list(zip(range_starts, range_ends, strict=False))
                raise RuntimeError(
                    "Shared CPU sparse decode cannot mark request state "
                    "hot-reusable with non-contiguous prefix coverage: "
                    f"req_id={request.req_id}, kv_group={kv_group}, "
                    f"token_count={required_token_count}, "
                    f"cached_ranges={cached_ranges}"
                )
            missing_pointer_layers = self._missing_shared_pointer_cache_layers(
                layers,
                cache["cached_chunk_ptrs_npu"],
                required_chunks,
            )
            if missing_pointer_layers:
                raise RuntimeError(
                    "Shared CPU sparse decode cannot mark request state "
                    "hot-reusable before NPU pointer-cache install: "
                    f"req_id={request.req_id}, kv_group={kv_group}, "
                    f"missing_layers={missing_pointer_layers}"
                )
            owned_groups[kv_group] = layers

        if not owned_groups:
            return
        if state.token_count <= 0:
            state.token_count = self._shared_retrieve_token_count_for_request(request)

        state.req_id = request.req_id
        state.shared_generation = generation
        state.pointer_cache_generation = generation
        state.request_scope_token = self._shared_request_scope_token(
            request.req_id,
            generation,
            state.token_count,
        )
        state.shared_latent_status = (
            "present" if has_entries(state.cached_memory_objs) else "missing"
        )
        if cold_compact or (
            materialize_index and has_entries(state.cached_memory_objs_indexer)
        ):
            state.shared_index_status = "present"
        elif (
            getattr(request, "shared_index_skipped", False)
            or state.shared_index_status == "skipped"
            or skip_index_hot_state
        ):
            state.shared_index_status = "skipped"
        else:
            state.shared_index_status = "missing"
        validation_signature = self._shared_worker_validation_signature(
            state,
            request,
            current_generation=generation,
            pointer_generation=generation,
            materialize_index=materialize_index,
        )
        append_from = None
        if 0 < previous_token_count < state.token_count:
            cache = state.cache_kwargs(0, dsa_two_groups=self._is_dsa_two_groups())
            previous_chunks = sum(
                int(end) <= previous_token_count for end in cache["cached_ends"]
            )
            if (
                previous_chunks
                and int(cache["cached_ends"][previous_chunks - 1])
                == previous_token_count
            ):
                append_from = dict.fromkeys(owned_groups, previous_chunks)
        if append_from is None:
            engine.register_shared_cpu_sparse_request(
                request.req_id,
                owned_groups=owned_groups,
            )
        else:
            engine.register_shared_cpu_sparse_request(
                request.req_id,
                owned_groups=owned_groups,
                append_from=append_from,
            )
        state.shared_request_active = True
        state.dense_prefix_seed = False
        state.shared_validation_signature = validation_signature

    def _worker_retrieve_state_can_extend(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        token_count: int,
    ) -> bool:
        committed = int(state.token_count or 0)
        chunk_size = int(getattr(self, "_lmcache_chunk_size", 1) or 1)
        load_spec = request.load_spec
        return bool(
            committed > 0
            and token_count > committed
            and token_count <= request.retrieve_token_count()
            and load_spec is not None
            and int(load_spec.lmcache_cached_tokens) == token_count
            and committed % chunk_size == 0
            and token_count % chunk_size == 0
            and state.cached_ends
            and int(state.cached_ends[-1]) == committed
            and self._cached_ranges_cover_prefix(
                state.cached_starts,
                state.cached_ends,
                committed,
            )
        )

    def _trim_dense_prefix_seed_for_sparse(
        self,
        state: WorkerRetrieveState,
        token_count: int,
    ) -> bool:
        """Drop only the dense partial tail before sparse full-chunk reuse."""
        if not state.dense_prefix_seed or not state.req_id:
            return False
        token_count = int(token_count)
        if token_count == state.token_count:
            return True
        if token_count <= 0 or token_count > state.token_count:
            return False

        plans: list[tuple[dict[str, Any], int]] = []
        owned_groups: dict[int, list[list[Any]]] = {}
        groups = (0, 1) if self._is_dsa_two_groups() else (0,)
        for kv_group in groups:
            cache = state.cache_kwargs(
                kv_group,
                dsa_two_groups=self._is_dsa_two_groups(),
            )
            memory_objs = cache["cached_memory_objs"]
            if not cache["cached_ends"]:
                continue
            keep = sum(int(end) <= token_count for end in cache["cached_ends"])
            if (
                keep <= 0
                or int(cache["cached_ends"][keep - 1]) != token_count
            ):
                return False
            plans.append((cache, keep))
            if memory_objs and any(memory_objs):
                if any(len(layer) < keep for layer in memory_objs):
                    return False
                owned_groups[kv_group] = [
                    list(layer[:keep]) for layer in memory_objs
                ]
        if not owned_groups:
            return False

        engine = getattr(self, "lmcache_engine", None)
        register = getattr(engine, "register_shared_cpu_sparse_request", None)
        if not callable(register):
            return False
        register(state.req_id, owned_groups=owned_groups)

        for cache, keep in plans:
            del cache["cached_starts"][keep:]
            del cache["cached_ends"][keep:]
            for name in (
                "cached_keys",
                "cached_memory_objs",
                "cached_tensors",
                "cached_chunk_dev_ptrs",
                "cached_shared_handles",
            ):
                for layer in cache[name]:
                    del layer[keep:]
            for layer_id, pointers in enumerate(cache["cached_chunk_ptrs_npu"]):
                if isinstance(pointers, torch.Tensor):
                    cache["cached_chunk_ptrs_npu"][layer_id] = pointers[:keep]
        state.prepared_sparse_sources.clear()
        state.token_count = token_count
        if state.metadata_token_ids:
            state.metadata_token_ids = state.metadata_token_ids[:token_count]
        return True

    def _should_invalidate_worker_retrieve_state(
        self, request: ReqMeta, token_count: int
    ) -> bool:
        if request.resumed_from_preemption:
            return True
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None:
            return False
        extending = self._worker_retrieve_state_can_extend(
            state,
            request,
            token_count,
        )
        if request.is_sparse_decode:
            if state.shared_request_active:
                engine = getattr(self, "lmcache_engine", None)
                generation = int(
                    getattr(engine, "shared_cpu_cache_generation", 0) or 0
                )
                prepared_latent = state.prepared_sparse_sources.get(0)
                if prepared_latent is not None:
                    if (
                        state.req_id != request.req_id
                        or state.shared_generation != generation
                        or prepared_latent.total_tokens != state.token_count
                    ):
                        return True
                else:
                    expected_scope_token = self._shared_request_scope_token(
                        request.req_id,
                        generation,
                        state.token_count,
                    )
                    if state.request_scope_token != expected_scope_token:
                        return True
            if state.cached_starts and state.cached_starts[0] != 0:
                return True
            # Sparse decode metadata is keyed by the full LMCache-hit prefix.
            # A shorter current prefix means the cached request state is stale.
            if state.token_count and (
                token_count < state.token_count
                or request.retrieve_token_count() < state.token_count
            ):
                return True
            if token_count > state.token_count:
                return not extending
            return False
        if state.cached_ends and token_count < state.cached_ends[-1]:
            return True
        if (
            request.load_spec is not None
            and request.load_spec.lmcache_cached_tokens > state.token_count
        ):
            return True
        return False

    def _worker_retrieve_state_for_request(
        self, request: ReqMeta
    ) -> Optional[WorkerRetrieveState]:
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None or not (state.metadata_warm or state.has_cache()):
            return None
        self._validate_shared_worker_retrieve_state(state, request)
        return state

    def _worker_retrieve_state_for_warm_ref(
        self, request: ReqMeta
    ) -> WorkerRetrieveState:
        load_spec = request.load_spec
        if (
            not request.is_sparse_decode
            or load_spec is None
            or not load_spec.can_load
        ):
            raise RuntimeError(
                f"Invalid sparse warm metadata for request {request.req_id}"
            )
        token_count = int(load_spec.lmcache_cached_tokens)
        state = self._worker_retrieve_state_for_request(request)
        # Async scheduling may queue one or more references before a
        # decode-window completion reaches the scheduler. The worker can
        # already own a larger prefix; that superset safely covers the older
        # scheduler frontier until a full refresh arrives.
        if (
            state is None
            or state.token_count < token_count
            or state.slot_mapping is None
            or self._prepared_sparse_source(state, 0, state.token_count) is None
        ):
            raise RuntimeError(
                "Sparse warm metadata has no matching prepared worker state: "
                f"req_id={request.req_id}, frontier={token_count}, "
                f"state_frontier={getattr(state, 'token_count', None)}"
            )
        return state

    @staticmethod
    def _bind_worker_retrieve_cache_to_request(
        request: ReqMeta,
        state: WorkerRetrieveState,
    ) -> None:
        """Expose legacy cache metadata only when bootstrap still needs it."""
        request.cached_keys = state.cached_keys
        request.cached_starts = state.cached_starts
        request.cached_ends = state.cached_ends
        request.cached_memory_objs = state.cached_memory_objs
        request.cached_tensors = state.cached_tensors
        request.cached_chunk_dev_ptrs = state.cached_chunk_dev_ptrs
        request.cached_chunk_ptrs_npu = state.cached_chunk_ptrs_npu
        request.cached_shared_handles = state.cached_shared_handles
        request.cached_keys_indexer = state.cached_keys_indexer
        request.cached_starts_indexer = state.cached_starts_indexer
        request.cached_ends_indexer = state.cached_ends_indexer
        request.cached_memory_objs_indexer = state.cached_memory_objs_indexer
        request.cached_tensors_indexer = state.cached_tensors_indexer
        request.cached_chunk_dev_ptrs_indexer = state.cached_chunk_dev_ptrs_indexer
        request.cached_chunk_ptrs_npu_indexer = state.cached_chunk_ptrs_npu_indexer
        request.cached_shared_handles_indexer = state.cached_shared_handles_indexer

    def _worker_retrieve_state_invalidation_reason(
        self,
        request: ReqMeta,
        token_count: int,
        state: WorkerRetrieveState,
    ) -> Optional[str]:
        if request.resumed_from_preemption:
            return "resumed_from_preemption"
        if request.is_sparse_decode:
            if state.shared_request_active:
                engine = getattr(self, "lmcache_engine", None)
                generation = int(
                    getattr(engine, "shared_cpu_cache_generation", 0) or 0
                )
                expected_scope_token = self._shared_request_scope_token(
                    request.req_id,
                    generation,
                    token_count,
                )
                if state.request_scope_token != expected_scope_token:
                    return "request_scope_changed"
            if state.cached_starts and state.cached_starts[0] != 0:
                return "nonzero_cached_start"
            if (
                request.load_spec is not None
                and request.load_spec.lmcache_cached_tokens > state.token_count
            ):
                return "load_frontier_advanced"
            # Sparse decode metadata is keyed by the full LMCache-hit prefix.
            # A shorter current prefix means the cached request state is stale.
            if state.token_count and (
                token_count < state.token_count
                or request.retrieve_token_count() < state.token_count
            ):
                return "retrieve_prefix_shrunk"
            return None
        if state.cached_ends and token_count < state.cached_ends[-1]:
            return "cached_end_past_retrieve"
        if (
            request.load_spec is not None
            and request.load_spec.lmcache_cached_tokens > state.token_count
        ):
            return "load_frontier_advanced"
        return None

    @staticmethod
    def _deep_retrieve_range_summary(
        starts: list[int], ends: list[int]
    ) -> dict[str, Optional[int]]:
        ranges = list(zip(starts, ends, strict=False))
        return {
            "count": len(ranges),
            "first_start": int(ranges[0][0]) if ranges else None,
            "last_end": int(ranges[-1][1]) if ranges else None,
        }

    @staticmethod
    def _deep_cache_values_present(values: Any) -> bool:
        if values is None:
            return False
        if isinstance(values, torch.Tensor):
            return values.numel() > 0
        if isinstance(values, dict):
            return any(
                LMCacheConnectorV1Impl._deep_cache_values_present(value)
                for value in values.values()
            )
        if isinstance(values, (list, tuple, set)):
            return any(
                LMCacheConnectorV1Impl._deep_cache_values_present(value)
                for value in values
            )
        return True

    @staticmethod
    def _deep_retrieve_group_cache_present(
        state: WorkerRetrieveState, kv_group: int
    ) -> bool:
        suffix = "_indexer" if kv_group == 1 else ""
        fields = (
            "cached_keys",
            "cached_memory_objs",
            "cached_tensors",
            "cached_chunk_dev_ptrs",
            "cached_chunk_ptrs_npu",
            "cached_shared_handles",
        )
        if any(
            LMCacheConnectorV1Impl._deep_cache_values_present(
                getattr(state, field + suffix)
            )
            for field in fields
        ):
            return True
        return False

    def _trace_deep_retrieve_state(
        self,
        request: ReqMeta,
        prior_state: Optional[WorkerRetrieveState],
        prior_snapshot: Optional[dict[str, Any]],
        *,
        invalidated: bool,
        invalidation_reason: Optional[str],
        post_state: Optional[WorkerRetrieveState],
        prior_state_rebound: bool,
        kv_group0_retriever_present: bool,
        kv_group1_retriever_present: bool,
    ) -> None:
        if not _mtp_dw_deep_diag_enabled() or request.load_spec is None:
            return
        frontier = int(request.load_spec.lmcache_cached_tokens)
        prior_frontier = (
            int(prior_snapshot["frontier"])
            if prior_snapshot is not None
            else 0
        )
        if prior_state is not None and prior_snapshot is None:
            return
        if frontier <= prior_frontier:
            return

        transitions = getattr(self, "_mtp_dw_deep_retrieve_transitions", None)
        if transitions is None:
            transitions = {}
            self._mtp_dw_deep_retrieve_transitions = transitions
        if request.req_id in transitions:
            return
        if len(transitions) >= 256:
            transitions.pop(next(iter(transitions)))
        transitions[request.req_id] = frontier

        stale_retained = (
            prior_state is not None
            and prior_state_rebound
            and post_state is prior_state
            and prior_frontier < frontier
        )
        metadata = getattr(getattr(self, "lmcache_engine", None), "metadata", None)
        worker_rank = getattr(metadata, "worker_id", None)
        if stale_retained:
            _mtp_dw_event(
                "fail",
                req=request.req_id,
                invariant="stale_retrieve_state",
                frontier=frontier,
                prior_frontier=prior_frontier,
                window_start=prior_frontier,
                window_end=frontier,
            )
        _mtp_dw_event(
            "deep",
            event="retrieve_state",
            req=request.req_id,
            worker_rank=worker_rank,
            tp_rank=worker_rank,
            tp_world=getattr(metadata, "world_size", None),
            frontier=frontier,
            prior_frontier=prior_frontier,
            window_start=prior_frontier,
            window_end=frontier,
            kv_group=None,
            prior_state_present=prior_state is not None,
            post_state_present=post_state is not None,
            invalidated=invalidated,
            invalidation_reason=invalidation_reason,
            prior_state_rebound=prior_state_rebound,
            rebound_range_count=(
                min(len(post_state.cached_starts), len(post_state.cached_ends))
                if post_state is not None
                else 0
            ),
            prior_kv_group0_cached_ranges=(
                prior_snapshot["kv_group0_cached_ranges"]
                if prior_snapshot is not None
                else self._deep_retrieve_range_summary([], [])
            ),
            prior_kv_group1_cached_ranges=(
                prior_snapshot["kv_group1_cached_ranges"]
                if prior_snapshot is not None
                else self._deep_retrieve_range_summary([], [])
            ),
            kv_group0_retriever_present=kv_group0_retriever_present,
            kv_group1_retriever_present=kv_group1_retriever_present,
            prior_kv_group0_cache_present=(
                prior_snapshot["kv_group0_cache_present"]
                if prior_snapshot is not None
                else False
            ),
            prior_kv_group1_cache_present=(
                prior_snapshot["kv_group1_cache_present"]
                if prior_snapshot is not None
                else False
            ),
            post_kv_group0_cache_present=(
                self._deep_retrieve_group_cache_present(post_state, 0)
                if post_state is not None
                else False
            ),
            post_kv_group1_cache_present=(
                self._deep_retrieve_group_cache_present(post_state, 1)
                if post_state is not None
                else False
            ),
            prior_scope_token=(
                prior_snapshot["scope_token"] if prior_snapshot is not None else None
            ),
            prior_scope_token_present=(
                prior_snapshot["scope_token_present"]
                if prior_snapshot is not None
                else False
            ),
            shared_request_active=(
                prior_snapshot["shared_request_active"]
                if prior_snapshot is not None
                else False
            ),
            shared_generation=(
                prior_snapshot["shared_generation"]
                if prior_snapshot is not None
                else 0
            ),
            pointer_cache_generation=(
                prior_snapshot["pointer_cache_generation"]
                if prior_snapshot is not None
                else 0
            ),
            resumed_from_preemption=bool(request.resumed_from_preemption),
            stale_state_retained=stale_retained,
        )

    def _state_has_retrieve_tensor_cache(self, state: WorkerRetrieveState) -> bool:
        num_layers = self._num_layers_for_group(0)
        tensors = state.cached_tensors
        if num_layers <= 0:
            if tensors and any(tensors):
                return True
            mem = state.cached_memory_objs
            return bool(mem and any(mem))
        if tensors and len(tensors) == num_layers and any(tensors):
            return True
        mem = state.cached_memory_objs
        return bool(mem and len(mem) == num_layers and any(mem))

    def _resolve_store_retrieve_location(
        self, state: WorkerRetrieveState
    ) -> Optional[str]:
        engine = self.lmcache_engine
        if engine is None or not state.cached_keys or not state.cached_keys[0]:
            return None
        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is None:
            return getattr(engine, "store_location", None)
        return storage_manager.contains(
            state.cached_keys[0][0],
            getattr(engine, "retrieve_locations", None),
        )

    @staticmethod
    def _ensure_layer_cache_shape(dst: list, src: list) -> None:
        if not src:
            return
        if not dst:
            dst.extend([] for _ in range(len(src)))
        while len(dst) < len(src):
            dst.append([])

    @classmethod
    def _merge_cache_group_by_ranges(
        cls,
        *,
        dst_starts: list[int],
        dst_ends: list[int],
        dst_keys: list[list],
        dst_memory_objs: list[list],
        dst_tensors: list[list],
        dst_chunk_dev_ptrs: list[list[int]],
        dst_chunk_ptrs_npu: list[Optional[torch.Tensor]],
        dst_shared_handles: list[list[Any]],
        src_starts: list[int],
        src_ends: list[int],
        src_keys: list[list],
        src_memory_objs: list[list],
        src_tensors: list[list],
        src_chunk_dev_ptrs: list[list[int]],
        src_chunk_ptrs_npu: list[Optional[torch.Tensor]],
        src_shared_handles: list[list[Any]],
        require_pointer_cache: bool = False,
    ) -> int:
        if not src_starts or not src_ends:
            return 0

        replace_at: Optional[int] = None
        for src_start, src_end in zip(src_starts, src_ends, strict=False):
            for dst_index, (dst_start, dst_end) in enumerate(
                zip(dst_starts, dst_ends, strict=False)
            ):
                if dst_start == src_start and src_end > dst_end:
                    replace_at = dst_index
                    break
            if replace_at is not None:
                break
        existing_chunks = len(dst_starts)
        cached_prefix_chunks = existing_chunks if replace_at is None else replace_at

        existing_ranges = set(
            zip(
                dst_starts if replace_at is None else dst_starts[:replace_at],
                dst_ends if replace_at is None else dst_ends[:replace_at],
                strict=False,
            )
        )
        existing_ends_by_start: dict[int, int] = {}
        for start, end in existing_ranges:
            existing_ends_by_start[start] = max(
                existing_ends_by_start.get(start, -1), end
            )
        append_indices: list[int] = []
        for chunk_idx, chunk_range in enumerate(
            zip(src_starts, src_ends, strict=False)
        ):
            if chunk_range in existing_ranges:
                continue
            chunk_start, chunk_end = chunk_range
            if existing_ends_by_start.get(chunk_start, -1) >= chunk_end:
                continue
            existing_ranges.add(chunk_range)
            existing_ends_by_start[chunk_start] = chunk_end
            append_indices.append(chunk_idx)

        if not append_indices:
            return 0

        def selected_ptrs_from_source(src_ptrs: torch.Tensor) -> Optional[torch.Tensor]:
            if not isinstance(src_ptrs, torch.Tensor):
                return None
            if any(chunk_idx >= int(src_ptrs.numel()) for chunk_idx in append_indices):
                return None
            if (
                append_indices[0] == 0
                and append_indices[-1] == len(append_indices) - 1
            ):
                return src_ptrs[: len(append_indices)]
            if append_indices[-1] - append_indices[0] + 1 == len(append_indices):
                return src_ptrs[append_indices[0] : append_indices[-1] + 1]
            return None

        def select_layer_ptr_tensors() -> Optional[list[torch.Tensor]]:
            if not src_chunk_ptrs_npu:
                return None
            selected_by_layer: list[torch.Tensor] = []
            for src_ptrs in src_chunk_ptrs_npu:
                selected = selected_ptrs_from_source(src_ptrs)
                if selected is None:
                    return None
                selected_by_layer.append(selected)
            return selected_by_layer

        selected_ptrs_by_layer = select_layer_ptr_tensors()
        source_layer_count = max(
            len(src_memory_objs or []),
            len(src_tensors or []),
            len(src_keys or []),
        )
        if cached_prefix_chunks and (
            len(dst_memory_objs) != source_layer_count
            or any(len(layer) != existing_chunks for layer in dst_memory_objs)
        ):
            logger.warning(
                "Skipping suffix cache promotion without complete prefix "
                "owners: prefix_chunks=%d, layers=%d",
                cached_prefix_chunks,
                source_layer_count,
            )
            return 0

        def can_append_layer_ptr_tensors() -> bool:
            if (
                selected_ptrs_by_layer is None
                or len(selected_ptrs_by_layer) != source_layer_count
            ):
                return False
            for layer_id in range(len(selected_ptrs_by_layer)):
                existing = (
                    dst_chunk_ptrs_npu[layer_id]
                    if layer_id < len(dst_chunk_ptrs_npu)
                    else None
                )
                if existing_chunks and (
                    not isinstance(existing, torch.Tensor)
                    or int(existing.numel()) != existing_chunks
                ):
                    return False
                if not existing_chunks and existing is not None and (
                    not isinstance(existing, torch.Tensor)
                    or int(existing.numel()) != 0
                ):
                    return False
            return True

        can_append_ptrs = can_append_layer_ptr_tensors()
        if require_pointer_cache and not can_append_ptrs:
            return 0

        if replace_at is not None:
            del dst_starts[replace_at:]
            del dst_ends[replace_at:]

            def truncate_layer_values(dst: list) -> None:
                for layer_values in dst or []:
                    if isinstance(layer_values, list):
                        del layer_values[replace_at:]

            for dst_layers in (
                dst_keys,
                dst_memory_objs,
                dst_tensors,
                dst_chunk_dev_ptrs,
                dst_shared_handles,
            ):
                truncate_layer_values(dst_layers)
            for layer_id, ptrs in enumerate(dst_chunk_ptrs_npu or []):
                if isinstance(ptrs, torch.Tensor):
                    dst_chunk_ptrs_npu[layer_id] = ptrs[:replace_at]

        assert len(dst_starts) == cached_prefix_chunks
        for chunk_idx in append_indices:
            dst_starts.append(src_starts[chunk_idx])
            dst_ends.append(src_ends[chunk_idx])

        def append_layer_values(dst: list, src: list) -> None:
            if not src:
                return
            cls._ensure_layer_cache_shape(dst, src)
            for layer_id, src_layer in enumerate(src):
                for chunk_idx in append_indices:
                    if chunk_idx < len(src_layer):
                        dst[layer_id].append(src_layer[chunk_idx])

        complete_tensors = (
            len(src_tensors) == source_layer_count
            and all(
                all(chunk_idx < len(layer) for chunk_idx in append_indices)
                for layer in src_tensors
            )
            and (
                (cached_prefix_chunks == 0 and not any(dst_tensors))
                or (
                    len(dst_tensors) == source_layer_count
                    and all(
                        len(layer) == cached_prefix_chunks for layer in dst_tensors
                    )
                )
            )
        )
        append_layer_values(dst_keys, src_keys)
        append_layer_values(dst_memory_objs, src_memory_objs)
        if complete_tensors:
            append_layer_values(dst_tensors, src_tensors)
        else:
            # Never expose a suffix-only tensor cache; the complete MemoryObj
            # and pointer caches remain authoritative.
            dst_tensors.clear()
        append_layer_values(dst_chunk_dev_ptrs, src_chunk_dev_ptrs)
        append_layer_values(dst_shared_handles, src_shared_handles)

        def append_layer_ptr_tensors() -> bool:
            if not can_append_ptrs:
                return False
            if not dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.extend(
                    None for _ in range(len(src_chunk_ptrs_npu))
                )
            while len(dst_chunk_ptrs_npu) < len(src_chunk_ptrs_npu):
                dst_chunk_ptrs_npu.append(None)

            for layer_id, selected in enumerate(selected_ptrs_by_layer):
                existing = dst_chunk_ptrs_npu[layer_id]
                dst_chunk_ptrs_npu[layer_id] = (
                    selected
                    if existing is None
                    else torch.cat((existing, selected))
                )
            return True

        if not append_layer_ptr_tensors():
            if dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.clear()
            if src_chunk_ptrs_npu and not dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.extend(None for _ in range(len(src_chunk_ptrs_npu)))
        return len(append_indices)

    def _merge_store_result_into_worker_state(
        self,
        destination: WorkerRetrieveState,
        result: LayerwiseStoreResult,
        request: ReqMeta,
    ) -> int:
        cache = destination.cache_kwargs(
            result.kv_group,
            dsa_two_groups=self._is_dsa_two_groups(),
        )
        require_pointer_cache = (
            self._is_decode_window_save_request(request)
            and bool(
                getattr(
                    getattr(self, "lmcache_engine", None),
                    "enable_shared_cpu_cache",
                    False,
                )
            )
        )
        return self._merge_cache_group_by_ranges(
            dst_starts=cache["cached_starts"],
            dst_ends=cache["cached_ends"],
            dst_keys=cache["cached_keys"],
            dst_memory_objs=cache["cached_memory_objs"],
            dst_tensors=cache["cached_tensors"],
            dst_chunk_dev_ptrs=cache["cached_chunk_dev_ptrs"],
            dst_chunk_ptrs_npu=cache["cached_chunk_ptrs_npu"],
            dst_shared_handles=cache["cached_shared_handles"],
            src_starts=result.starts,
            src_ends=result.ends,
            src_keys=result.keys,
            src_memory_objs=result.memory_objs,
            src_tensors=result.tensors,
            src_chunk_dev_ptrs=result.chunk_dev_ptrs,
            src_chunk_ptrs_npu=result.chunk_ptrs,
            src_shared_handles=[],
            require_pointer_cache=require_pointer_cache,
        )

    def _warm_request_retrieve_metadata(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        kv_group: int,
        dsa_two_groups: bool,
    ) -> tuple[Optional[str], bool]:
        engine = self.lmcache_engine
        if (
            engine is not None
            and getattr(engine, "enable_shared_cpu_cache", False)
            and getattr(engine, "storage_manager", None) is None
        ):
            is_passive_fn = getattr(engine, "_is_passive", None)
            if callable(is_passive_fn) and is_passive_fn():
                return None, False

        ensure_metadata = getattr(
            engine, "_ensure_retrieve_chunk_metadata", None
        )
        if ensure_metadata is None:
            return None, False

        cache_kwargs = state.cache_kwargs(kv_group, dsa_two_groups)
        cached_keys = cache_kwargs["cached_keys"]
        cached_starts = cache_kwargs["cached_starts"]
        cached_ends = cache_kwargs["cached_ends"]
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        retrieve_kwargs: dict[str, Any] = {"kv_group": kv_group}

        location, _, _, _ = ensure_metadata(
            tokens=tokens,
            mask=mask,
            request_configs=request.request_configs,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=retrieve_kwargs,
        )
        if location is None:
            location = retrieve_kwargs.get("cached_retrieve_location")
        metadata_warm = bool(
            retrieve_kwargs.get("_retrieve_metadata_warm")
            and cached_keys
            and cached_ends
        )
        return location, metadata_warm

    def _retain_shared_store_seed_state(
        self,
        state: WorkerRetrieveState,
    ) -> None:
        """Retain store output through the engine-owned request lease."""

        engine = self.lmcache_engine
        if engine is None or not getattr(engine, "enable_shared_cpu_cache", False):
            return

        groups = {0: state.cached_memory_objs}
        if state.cached_memory_objs_indexer:
            groups[1] = state.cached_memory_objs_indexer
        groups = {
            kv_group: layers
            for kv_group, layers in groups.items()
            if any(layers)
        }
        if not groups or not state.req_id:
            return

        engine.retain_shared_cpu_store_seed(state.req_id, groups)

    def _store_result_has_retrieve_data(
        self,
        result: LayerwiseStoreResult,
    ) -> bool:
        if not result.has_cache():
            return False
        num_layers = self._num_layers_for_group(result.kv_group)
        if num_layers <= 0:
            return bool(result.memory_objs and any(result.memory_objs))
        return (
            len(result.memory_objs) == num_layers
            and all(result.memory_objs)
        )

    def _promote_layerwise_store_result(
        self,
        request: ReqMeta,
        result: Optional[LayerwiseStoreResult],
    ) -> None:
        """Promote completed store output into request-owned retrieve state."""
        if result is None:
            return
        if result.request_id != request.req_id:
            raise RuntimeError(
                "Layerwise store result request mismatch: "
                f"expected={request.req_id}, got={result.request_id}"
            )
        if result.kv_group not in (0, 1):
            raise RuntimeError(
                f"Unsupported layerwise store result group {result.kv_group}"
            )
        if (
            request.is_sparse_decode
            or not hasattr(self, "_worker_retrieve_state")
            or not self._store_result_has_retrieve_data(result)
        ):
            return

        if (
            self._is_decode_window_save_request(request)
            and self._decode_window_save_uses_shared_cpu()
        ):
            # All TP ranks must refresh this window through the next collective
            # sparse load; rank0 cannot privately extend its hot request state.
            return

        state = self._worker_retrieve_state.get(request.req_id)
        state_is_warm = state is not None and (
            state.metadata_warm or state.has_cache()
        )
        if (
            state_is_warm
            and self._is_decode_window_save_request(request)
            and not self._decode_window_save_is_next_expected(request)
        ):
            logger.debug(
                "Skipping out-of-order decode-window store result: "
                "req_id=%s window_start=%s",
                request.req_id,
                getattr(request, "decode_window_start", None),
            )
            return
        if (
            not state_is_warm
            and self._is_decode_window_save_request(request)
            and not self._cached_ranges_cover_prefix(
                result.starts,
                result.ends,
                len(request.token_ids),
            )
        ):
            logger.debug(
                "Skipping decode-window store result without full-prefix "
                "coverage: req_id=%s ranges=%s token_count=%d",
                request.req_id,
                list(zip(result.starts, result.ends, strict=False)),
                len(request.token_ids),
            )
            return

        is_new_state = state is None
        if state is None:
            state = WorkerRetrieveState(req_id=request.req_id)
        rollback_snapshot = (
            None
            if is_new_state
            else self._snapshot_worker_retrieve_cache_state(state)
        )
        try:
            if self._merge_store_result_into_worker_state(
                state,
                result,
                request,
            ) == 0:
                return

            # An index result may arrive before latent under two-group DSA.
            # Keep it request-owned but invisible to retrieve until latent is
            # present and the request state can be finalized.
            if not (
                state.cached_keys
                and state.cached_starts
                and state.cached_ends
                and self._state_has_retrieve_tensor_cache(state)
            ):
                self._retain_shared_store_seed_state(state)
                self._set_worker_retrieve_state(request.req_id, state)
                return

            state.location = (
                self._resolve_store_retrieve_location(state) or state.location
            )
            state.metadata_warm = True
            state.token_count = max(
                state.token_count,
                self._cached_prefix_covered_token_count(
                    state.cached_starts,
                    state.cached_ends,
                ),
            )
            if state.shared_request_active:
                self._validate_decode_save_shared_pointer_cache(state, request)
                engine = getattr(self, "lmcache_engine", None)
                generation = int(
                    getattr(engine, "shared_cpu_cache_generation", 0) or 0
                )
                state.shared_generation = generation
                state.pointer_cache_generation = generation
                state.request_scope_token = self._shared_request_scope_token(
                    request.req_id,
                    generation,
                    state.token_count,
                )
                state.shared_validation_signature = None

            self._refresh_prepared_sparse_sources(state, state.token_count)
            self._retain_shared_store_seed_state(state)
            self._set_worker_retrieve_state(request.req_id, state)
        except Exception:
            if is_new_state:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
            elif rollback_snapshot is not None:
                self._restore_worker_retrieve_cache_state(
                    state,
                    rollback_snapshot,
                )
            raise

    def _refresh_prepared_sparse_sources(
        self,
        state: WorkerRetrieveState,
        token_count: int,
    ) -> None:
        """Seal complete per-group source caches after bootstrap or store."""
        prepared: dict[int, PreparedSparseSource] = {}
        dsa_two_groups = self._is_dsa_two_groups()
        group_ids = (0, 1) if dsa_two_groups else (0,)
        for kv_group in group_ids:
            cache = state.cache_kwargs(kv_group, dsa_two_groups)
            cached_starts = cache["cached_starts"]
            cached_ends = cache["cached_ends"]
            chunk_token_counts = None
            if cached_starts and len(cached_starts) == len(cached_ends):
                if not self._cached_ranges_cover_prefix(
                    cached_starts,
                    cached_ends,
                    token_count,
                ):
                    continue
                chunk_token_counts = tuple(
                    int(end) - int(start)
                    for start, end in zip(
                        cached_starts,
                        cached_ends,
                        strict=False,
                    )
                )
            expected_pointer_device = getattr(self, "device", None)
            if expected_pointer_device is not None:
                expected_pointer_device = torch.device(expected_pointer_device)
            source = build_prepared_sparse_source(
                cache["cached_tensors"],
                cache["cached_chunk_ptrs_npu"],
                num_layers=self._num_layers_for_group(kv_group),
                total_tokens=token_count,
                chunk_token_counts=chunk_token_counts,
                expected_pointer_device=expected_pointer_device,
                cached_memory_objs=cache["cached_memory_objs"],
            )
            if source is not None:
                prepared[kv_group] = source
        state.prepared_sparse_sources = prepared

    def _prepared_sparse_source(
        self,
        state: Optional[WorkerRetrieveState],
        kv_group: int,
        token_count: int,
    ) -> Optional[PreparedSparseSource]:
        if state is None:
            return None
        engine = getattr(self, "lmcache_engine", None)
        if (
            engine is not None
            and getattr(engine, "enable_shared_cpu_cache", False)
            and not state.shared_request_active
        ):
            return None
        source = state.prepared_sparse_sources.get(kv_group)
        if source is None or source.total_tokens != token_count:
            return None
        return source

    def _prepared_sparse_sources_current(
        self,
        state: Optional[WorkerRetrieveState],
        request: ReqMeta,
        token_count: int,
    ) -> bool:
        if self._prepared_sparse_source(state, 0, token_count) is None:
            return False
        if (
            state is not None
            and (state.cached_tensors_indexer or state.cached_memory_objs_indexer)
            and not request.shared_index_skipped
            and self._prepared_sparse_source(state, 1, token_count) is None
        ):
            return False
        return True

    def _publish_worker_retrieve_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        *,
        location: Optional[str],
        metadata_warm: bool,
        token_count: int,
    ) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        if not metadata_warm and not state.has_cache():
            return

        states = self._worker_retrieve_state
        previous_state = states.get(request.req_id)
        previous_token_count = (
            int(state.token_count) if previous_state is state else 0
        )
        state.req_id = request.req_id
        state.location = location or state.location
        state.metadata_warm = metadata_warm or state.metadata_warm
        state.token_count = token_count
        try:
            self._refresh_prepared_sparse_sources(state, token_count)
            self._record_shared_worker_retrieve_state(
                state,
                request,
                previous_token_count,
            )
            if not request.sparse_warm_ref and len(request.token_ids) >= token_count:
                state.metadata_token_ids = list(request.token_ids[:token_count])
        except Exception:
            self._release_unadopted_shared_request_objects(state, request)
            preserve_previous_lease = (
                previous_state is not None
                and previous_state is not state
            )
            self._release_shared_worker_retrieve_state(
                state,
                self.lmcache_engine,
                release_request=not preserve_previous_lease,
            )
            if states.get(request.req_id) is state:
                states.pop(request.req_id, None)
                self._mark_worker_retrieve_registry_changed()
            raise
        self._set_worker_retrieve_state(request.req_id, state)

    def _finalize_worker_retrieve_state_from_metadata(
        self, metadata: LMCacheConnectorMetadata
    ) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        for request in metadata.requests:
            if not request.is_sparse_decode:
                continue
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            state = self._worker_retrieve_state.get(request.req_id)
            if state is None or not state.has_cache():
                continue

            if state.indexer_npu_materialization_pending:
                state.indexer_npu_resident = True
                state.indexer_npu_materialization_pending = False

            token_count = int(request.load_spec.lmcache_cached_tokens)
            if request.sparse_warm_ref:
                # Do not shrink state while consuming an async reference that
                # predates a worker-side decode-save promotion.
                token_count = max(token_count, int(state.token_count))
            if self._prepared_sparse_sources_current(
                state,
                request,
                token_count,
            ):
                continue
            if self._shared_worker_retrieve_state_is_current(
                state,
                request,
                token_count,
            ):
                self._refresh_prepared_sparse_sources(state, token_count)
                continue
            self._publish_worker_retrieve_state(
                state,
                request,
                location=state.location,
                metadata_warm=state.metadata_warm or state.has_cache(),
                token_count=token_count,
            )

    def _sparse_decode_bootstrap_reuse_kwargs(
        self,
        token_count: int,
        bound_state: Optional[WorkerRetrieveState],
    ) -> dict[str, Any]:
        warm_kwargs: dict[str, Any] = {}
        if bound_state is None:
            return warm_kwargs
        if bound_state.location is not None:
            warm_kwargs["cached_retrieve_location"] = bound_state.location
        if (
            bound_state.metadata_warm
            and bound_state.cached_keys
            and bound_state.cached_ends
            and token_count <= bound_state.token_count
            and (not bound_state.cached_starts or bound_state.cached_starts[0] == 0)
        ):
            warm_kwargs["_retrieve_metadata_warm"] = True
        return warm_kwargs

    def _sparse_retrieve_kwargs(
        self,
        request: ReqMeta,
        retrieve_state: WorkerRetrieveState,
        bound_state: Optional[WorkerRetrieveState],
        *,
        kvcaches: list[torch.Tensor],
        slot_mapping: Any,
        sync: bool,
        kv_group: int,
        request_ordinal: int,
        dsa_two_groups: bool,
        token_count: int,
        shared_cpu_enabled: bool,
        shared_cpu_preflight_state: Optional[dict[str, Any]],
        metadata_only: bool = False,
    ) -> tuple[
        dict[str, Any],
        Optional[dict[str, Any]],
        Optional[PreparedSparseSource],
    ]:
        assert request.load_spec is not None
        prepared_token_count = (
            retrieve_state.token_count
            if request.sparse_warm_ref
            else token_count
        )
        prepared_source = None
        if not metadata_only:
            prepared_source = self._prepared_sparse_source(
                retrieve_state,
                kv_group,
                prepared_token_count,
            )
        if request.sparse_warm_ref and prepared_source is None:
            raise RuntimeError(
                "Sparse warm metadata cannot enter bootstrap without token "
                f"metadata: req_id={request.req_id}, kv_group={kv_group}, "
                f"frontier={token_count}, "
                f"state_frontier={retrieve_state.token_count}"
            )
        if prepared_source is not None:
            retrieve_kwargs: dict[str, Any] = {
                "kvcaches": kvcaches,
                "slot_mapping": slot_mapping,
                "sync": sync,
                "kv_group": kv_group,
                "req_id": request.req_id,
                "lmcache_cached_tokens": prepared_token_count,
                "prepared_sparse_source": prepared_source,
            }
        else:
            if (
                shared_cpu_enabled
                and dsa_two_groups
                and shared_cpu_preflight_state is None
            ):
                shared_cpu_preflight_state = {}
            retrieve_kwargs = {
                "kvcaches": kvcaches,
                "slot_mapping": slot_mapping,
                "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                "lmcache_cached_tokens": request.load_spec.lmcache_cached_tokens,
                "sync": sync,
                "kv_group": kv_group,
                "req_id": request.req_id,
                "request_configs": request.request_configs,
                "shared_cpu_phase": SPARSE_DECODE_SHARED_CPU_PHASE,
                "shared_cpu_request_ordinal": request_ordinal,
                "cached_metadata_token_ids": retrieve_state.metadata_token_ids,
                **retrieve_state.cache_kwargs(kv_group, dsa_two_groups),
            }
            if shared_cpu_preflight_state is not None:
                retrieve_kwargs["shared_cpu_request_preflight_state"] = (
                    shared_cpu_preflight_state
                )
            retrieve_kwargs.update(
                self._sparse_decode_bootstrap_reuse_kwargs(token_count, bound_state)
            )
            if metadata_only:
                retrieve_kwargs["materialize_only"] = True
        ret_mask = (
            request.decode_ret_mask
            if request.decode_ret_mask is not None
            else retrieve_state.decode_ret_mask
        )
        if ret_mask is not None:
            retrieve_kwargs["ret_mask"] = ret_mask
        return retrieve_kwargs, shared_cpu_preflight_state, prepared_source

    @staticmethod
    def _prime_dense_prefix_retrievers(
        layerwise_retriever: Generator[Optional[torch.Tensor], None, None],
        indexer_retriever: Optional[Generator[Optional[torch.Tensor], None, None]],
    ) -> None:
        """Prime dense prefix retrievers without breaking two-group ordering."""
        next(layerwise_retriever)
        if indexer_retriever is not None:
            next(indexer_retriever)
        next(layerwise_retriever)
        if indexer_retriever is not None:
            next(indexer_retriever)

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._refresh_kvcaches_list()
        self._build_kv_layer_groups()
        self._manager.post_init()

    def _get_dsa_cold_load_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_dsa_cold_load_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="lmcache-dsa-cold",
            )
            self._dsa_cold_load_executor = executor
        return executor

    def _synchronize_dsa_cold_dense_load(self, stream: Any = None) -> None:
        assert self.lmcache_engine is not None
        synchronize = getattr(
            self.lmcache_engine.gpu_connector,
            "synchronize_dense_load_stream",
            None,
        )
        if synchronize is None:
            raise RuntimeError("NPU connector has no dense load-stream sync API")
        if stream is None:
            synchronize()
        else:
            synchronize(stream)

    def _synchronize_dsa_cold_dense_readiness(self, readiness: Any) -> None:
        assert self.lmcache_engine is not None
        synchronize = getattr(
            self.lmcache_engine.gpu_connector,
            "synchronize_dense_load_readiness",
            None,
        )
        if synchronize is None:
            raise RuntimeError("NPU connector has no dense load readiness sync API")
        synchronize(readiness)

    def _record_dsa_cold_dense_load_readiness(
        self,
        state: WorkerRetrieveState,
        readiness: Any = None,
        additional_owners: tuple[Any, ...] = (),
    ) -> None:
        assert self.lmcache_engine is not None
        connector = self.lmcache_engine.gpu_connector
        record = getattr(connector, "record_dense_load_readiness", None)
        if record is None:
            raise RuntimeError("NPU connector has no dense load readiness API")
        state.dense_load_readiness = record() if readiness is None else readiness
        state.dense_load_readiness_consumed = False
        owners: list[Any] = []
        seen: set[int] = set()
        for owner in additional_owners:
            if id(owner) not in seen:
                seen.add(id(owner))
                owners.append(owner)
        state.dense_load_source_owners = tuple(owners)

    def _consume_dsa_cold_dense_load_readiness(
        self, state: WorkerRetrieveState
    ) -> None:
        readiness = state.dense_load_readiness
        if readiness is None or state.dense_load_readiness_consumed:
            return
        assert self.lmcache_engine is not None
        consume = getattr(
            self.lmcache_engine.gpu_connector,
            "consume_dense_load_readiness",
            None,
        )
        if consume is None:
            raise RuntimeError("NPU connector has no dense load readiness API")
        consume(readiness)
        state.dense_load_readiness_consumed = True

    def _submit_dsa_cold_compact_load(self, request: ReqMeta) -> None:
        futures = getattr(self, "_dsa_cold_load_futures", None)
        if futures is None:
            futures = {}
            self._dsa_cold_load_futures = futures
        assert request.load_spec is not None
        generation = request.load_spec.dsa_cold_load_generation
        existing = futures.get(request.req_id)
        if existing is not None:
            if existing[0] == generation:
                return
            raise RuntimeError(
                "Cold compact request ID was reused while an earlier load "
                f"is still active: req_id={request.req_id}, "
                f"active_generation={existing[0]}, generation={generation}"
            )
        indexer_slots = request.indexer_slot_mapping[0]
        indexer_block_ids = set(
            (indexer_slots // self._block_size).tolist()
        )
        npu_device_id = (
            int(torch.npu.current_device()) if hasattr(torch, "npu") else None
        )
        plan_started = cold_start_perf_now()
        token_count = getattr(request.load_spec, "lmcache_cached_tokens", 0)
        plan = {
            "request": request,
            "tokens": request.token_ids[:token_count],
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "token_count": token_count,
            "indexer_slots_cpu": request.indexer_slot_mapping[0],
            "latent_kvcaches": self._kvcaches_for_group(0),
            "indexer_kvcaches": self._kvcaches_for_group(1),
            "planned_at": cold_start_perf_now(),
            "plan_started": plan_started,
            "latent_shared_ready": Future(),
        }
        submitted_at = cold_start_perf_now()
        cold_start_perf_log(
            logger,
            "worker_load_start",
            req_id=request.req_id,
            tokens=token_count,
            mode="dsa_cold_compact",
        )
        executor = self._get_dsa_cold_load_executor()
        indexer_future = executor.submit(
            self._run_dsa_cold_indexer_load,
            plan,
            npu_device_id,
        )
        previous_latent_future = getattr(
            self, "_dsa_cold_last_latent_future", None
        )
        latent_future = executor.submit(
            self._run_dsa_cold_compact_load,
            plan,
            npu_device_id,
            indexer_future,
            previous_latent_future,
        )
        self._dsa_cold_last_latent_future = latent_future
        futures[request.req_id] = (
            generation,
            latent_future,
            request,
            indexer_block_ids,
            submitted_at,
            indexer_future,
        )
        cold_start_perf_log(
            logger,
            "cold_compact_plan_submit",
            started=plan_started,
            req_id=request.req_id,
            tokens=token_count,
            jobs=2,
        )

    def _try_prepare_dsa_live_split(self, request: ReqMeta) -> bool:
        """Reserve cold groups until live import succeeds or falls back."""
        pending = getattr(self, "_dsa_live_split_pending", None)
        futures = getattr(self, "_dsa_cold_load_futures", None)
        if (
            pending is not None and request.req_id in pending
        ) or (
            futures is not None and request.req_id in futures
        ):
            raise RuntimeError(
                "Cold compact live-split request generation is still active: "
                f"req_id={request.req_id}"
            )
        if not getattr(request, "live_split_requested", False):
            return False
        if not getattr(request, "live_split_compact", False):
            return False
        remote_block_ids = getattr(request, "live_split_remote_block_ids", None)
        prepare = getattr(self.lmcache_engine, "_prepare_live_split_import", None)
        if not remote_block_ids or not callable(prepare):
            return False
        assert request.load_spec is not None
        token_count = request.load_spec.lmcache_cached_tokens
        plan_started = cold_start_perf_now()
        plan = {
            "request": request,
            "tokens": request.token_ids[:token_count],
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "token_count": token_count,
            "indexer_slots_cpu": request.indexer_slot_mapping[0],
            "latent_kvcaches": self._kvcaches_for_group(0),
            "indexer_kvcaches": self._kvcaches_for_group(1),
            "planned_at": cold_start_perf_now(),
            "plan_started": plan_started,
            "latent_shared_ready": Future(),
        }
        indexer_completion: Future = Future()
        generation = request.load_spec.dsa_cold_load_generation
        indexer_blocks = set(
            (request.indexer_slot_mapping[0] // self._block_size).tolist()
        )
        if futures is None:
            futures = self._dsa_cold_load_futures = {}
        npu_device_id = (
            int(torch.npu.current_device()) if hasattr(torch, "npu") else None
        )
        previous = getattr(self, "_dsa_cold_last_latent_future", None)
        latent_gate: Future = Future()
        completion: Future = Future()

        def start_latent(gate: Future) -> None:
            if completion.done():
                return
            try:
                live_state = gate.result()
            except BaseException as exc:
                completion.set_exception(exc)
                return
            task = self._get_dsa_cold_load_executor().submit(
                self._run_dsa_cold_compact_load,
                plan,
                npu_device_id,
                indexer_completion,
                previous,
                live_state,
            )

            def finish_latent(done: Future) -> None:
                if completion.done():
                    return
                try:
                    completion.set_result(done.result())
                except BaseException as exc:
                    completion.set_exception(exc)

            task.add_done_callback(finish_latent)

        latent_gate.add_done_callback(start_latent)
        self._dsa_cold_last_latent_future = completion
        futures[request.req_id] = (
            generation,
            completion,
            request,
            indexer_blocks,
            cold_start_perf_now(),
            indexer_completion,
        )
        if pending is None:
            pending = self._dsa_live_split_pending = {}
        pending[request.req_id] = {
            "request": request,
            "plan": plan,
            "npu_device_id": npu_device_id,
            "indexer_completion": indexer_completion,
            "latent_gate": latent_gate,
            "context": None,
            "offered": False,
        }
        # Passive ranks never own the shared slab allocation.  They can enter
        # the existing receive side immediately while rank 0 keeps its gate
        # closed until live group-0 DMA is admitted or falls back.
        if (
            not getattr(request, "live_split_latent_cpu", False)
            or get_tensor_model_parallel_rank() != 0
        ):
            latent_gate.set_result(None)
        return True

    def _cancel_live_split_entry(
        self,
        entry: dict[str, Any],
        reason: str,
        *,
        release_context: bool = True,
    ) -> None:
        context = entry.get("context")
        if context is not None and release_context:
            self.lmcache_engine._release_live_split_import(context)
            entry["context"] = None
        entry["cancelled"] = True
        error = RuntimeError(reason)
        entry["cancel_error"] = error
        gate = entry["latent_gate"]
        request = entry.get("request")
        rank0_latent = bool(
            getattr(request, "live_split_latent_cpu", False)
            and not gate.done()
        )
        if rank0_latent:
            # Passive ranks have already entered the ordinary group-0
            # collective. Preserve that collective slot. Before an offer the
            # fallback may start now; after an offer the result callback first
            # fences native DMA and then opens this gate.
            if release_context:
                gate.set_result(None)
        elif not gate.done():
            gate.set_exception(error)
        completion = entry["indexer_completion"]
        # An offered destination can still be owned by Mooncake's synchronous
        # native DMA.  Do not make the group-1 dependency terminal until the
        # worker result callback proves that DMA has returned.  Pre-offer
        # cancellation (and shutdown after Mooncake has been fenced) may finish
        # immediately because no writer can still target these blocks.
        if release_context and not completion.done():
            completion.set_exception(error)

    def _fallback_live_split_indexer(self, entry: dict[str, Any]) -> None:
        completion = entry["indexer_completion"]
        if completion.done():
            return
        # Run on the worker callback thread: two concurrent latent jobs may both
        # be waiting on their dependency, so queueing fallback onto the same
        # two-thread executor could deadlock.
        try:
            result = self._run_dsa_cold_indexer_load(
                entry["plan"], entry["npu_device_id"]
            )
            completion.set_result(result)
        except BaseException as exc:
            completion.set_exception(exc)

    def take_live_split_destination_plans(
        self, handled_groups: tuple[int, ...]
    ) -> dict[str, dict[str, Any]]:
        pending = getattr(self, "_dsa_live_split_pending", None)
        cold_start_perf_log(
            logger,
            "live_source_destination_provider_entry",
            pending_count=len(pending or {}),
            handled_groups=handled_groups,
        )
        if not pending:
            return {}
        handled_groups = tuple(handled_groups)
        if handled_groups not in ((1,), (0, 1)):
            for req_id, entry in list(pending.items()):
                gate = entry["latent_gate"]
                if not gate.done():
                    gate.set_result(None)
                self._fallback_live_split_indexer(entry)
                pending.pop(req_id, None)
            return {}
        parallel = getattr(self._vllm_config, "parallel_config", None)
        result: dict[str, dict[str, Any]] = {}
        for req_id, entry in list(pending.items()):
            if entry["offered"]:
                continue
            request = entry["request"]
            plan = entry["plan"]
            request_groups = handled_groups
            if not getattr(request, "live_split_latent_cpu", False):
                request_groups = tuple(
                    group for group in request_groups if group == 1
                )
            try:
                destination, context = (
                    self.lmcache_engine._prepare_live_split_import(
                        tokens=plan["tokens"],
                        latent_kvcaches=plan["latent_kvcaches"],
                        indexer_slots=plan["indexer_slots_cpu"],
                        indexer_kvcaches=plan["indexer_kvcaches"],
                        request_configs=request.request_configs,
                        tp_rank=get_tensor_model_parallel_rank(),
                        dp_rank=int(
                            getattr(parallel, "data_parallel_index", 0)
                            or 0
                        ),
                        handled_groups=request_groups,
                    )
                )
                destination["requested_groups"] = request_groups
                entry["context"] = context
                entry["handled_groups"] = request_groups
                entry["offered"] = True
                result[req_id] = destination
                if 0 not in request_groups:
                    gate = entry["latent_gate"]
                    if not gate.done():
                        gate.set_result(None)
            except Exception:
                logger.warning(
                    "Live split destination preparation failed; falling back",
                    exc_info=True,
                )
                entry["offered"] = True
                gate = entry["latent_gate"]
                if not gate.done():
                    gate.set_result(None)
                self._fallback_live_split_indexer(entry)
                pending.pop(req_id, None)
        return result

    def accept_live_split_results(self, results: dict[str, str]) -> None:
        pending = getattr(self, "_dsa_live_split_pending", None)
        cold_start_perf_log(
            logger,
            "live_source_result_accept_entry",
            pending_count=len(pending or {}),
            result_ids=sorted(results),
            statuses=results,
        )
        if not pending:
            return
        assert self.lmcache_engine is not None
        for req_id, status in results.items():
            entry = pending.get(req_id)
            if entry is None:
                continue
            context = entry["context"]
            completion = entry["indexer_completion"]
            if entry.get("cancelled"):
                if context is not None:
                    self.lmcache_engine._release_live_split_import(context)
                    entry["context"] = None
                if not completion.done():
                    completion.set_exception(
                        entry.get(
                            "cancel_error",
                            RuntimeError(
                                f"Live split request was cancelled: {req_id}"
                            ),
                        )
                    )
                gate = entry["latent_gate"]
                if not gate.done():
                    # Native transfer has returned before Mooncake publishes
                    # this result, so persistent fallback can safely occupy
                    # TP0's existing collective slot now.
                    gate.set_result(None)
                pending.pop(req_id, None)
                continue
            if status == "success":
                try:
                    if context is None:
                        raise RuntimeError("live split ACK has no destination")
                    if 0 in entry.get("handled_groups", (1,)):
                        admit = getattr(
                            self.lmcache_engine, "admit_live_split_pages", None
                        )
                        if not callable(admit):
                            raise RuntimeError(
                                "live split engine cannot admit latent pages"
                            )
                        admit(context)
                    record = getattr(
                        self.lmcache_engine.gpu_connector,
                        "record_dense_load_readiness",
                        None,
                    )
                    readiness = record() if callable(record) else None
                    entry["plan"]["indexer_source_owners"] = tuple(
                        context.get("destination_owners", ())
                    )
                    completion.set_result((None, readiness, 0.0, 0.0))
                    gate = entry["latent_gate"]
                    if not gate.done():
                        gate.set_result(None)
                    pending.pop(req_id, None)
                    continue
                except BaseException:
                    logger.exception("Live split commit failed for %s", req_id)
            if context is not None:
                self.lmcache_engine._release_live_split_import(context)
                entry["context"] = None
            gate = entry["latent_gate"]
            if not gate.done():
                gate.set_result(None)
            self._fallback_live_split_indexer(entry)
            pending.pop(req_id, None)

    def _run_dsa_cold_indexer_load(
        self, plan: dict[str, Any], npu_device_id: Optional[int]
    ) -> tuple[torch.Tensor, Any, float, float]:
        """Load Group 1 densely after Group 0 shared-CPU publication."""
        perf_breakdown = {} if cold_start_perf_enabled() else None
        if perf_breakdown is not None:
            plan["indexer_perf"] = perf_breakdown

        def stage_start() -> float:
            return cold_start_perf_now() if perf_breakdown is not None else 0.0

        def finish_stage(name: str, stage_started: float) -> None:
            if perf_breakdown is not None:
                perf_breakdown[name] = round(
                    (cold_start_perf_now() - stage_started) * 1000, 3
                )

        producer_setup_started = stage_start()
        if npu_device_id is not None:
            torch.npu.set_device(npu_device_id)
        finish_stage("producer_setup_ms", producer_setup_started)
        assert self.lmcache_engine is not None
        started = cold_start_perf_now()
        queue_ms = (started - plan["planned_at"]) * 1000
        request = plan["request"]
        indexer_slots_cpu = plan["indexer_slots_cpu"]
        indexer_kvcaches = plan["indexer_kvcaches"]
        validate_slots = getattr(
            self.lmcache_engine.gpu_connector,
            "validate_layerwise_slot_mapping",
            None,
        )
        slot_validate_started = stage_start()
        if callable(validate_slots):
            validate_slots(indexer_slots_cpu, indexer_kvcaches, kv_group=1)
        finish_stage("slot_validate_ms", slot_validate_started)
        slot_submit_started = stage_start()
        indexer_slots = indexer_slots_cpu.to(
            device=self.device,
            dtype=torch.long,
        )
        finish_stage("slot_mapping_submit_ms", slot_submit_started)
        retrieve_state = WorkerRetrieveState(req_id=request.req_id)
        supports_dense_retention = getattr(
            self.lmcache_engine, "supports_dense_sparse_cache_retention", None
        )
        if not callable(supports_dense_retention) or not supports_dense_retention():
            raise RuntimeError(
                "Cold compact indexer load requires dense source retention"
            )
        retrieve_kwargs_started = stage_start()
        readiness_out: list[Any] = []
        retrieve_kwargs = {
            "kvcaches": indexer_kvcaches,
            "slot_mapping": indexer_slots,
            "sync": True,
            "kv_group": 1,
            "req_id": request.req_id,
            "request_configs": request.request_configs,
            "shared_cpu_phase": "dsa_cold_compact_indexer",
            "shared_cpu_request_ordinal": 0,
            "_retain_shared_dense_cache": True,
            "_dense_load_readiness_out": readiness_out,
            **retrieve_state.cache_kwargs(1, dsa_two_groups=True),
        }
        finish_stage("retrieve_kwargs_ms", retrieve_kwargs_started)
        prefetch_owners: tuple[Any, ...] = ()
        if (
            getattr(
                getattr(self, "config", None),
                "dsa_group1_load_mode",
                "p2p_preferred",
            )
            == "persistent_parallel_prefetch"
        ):
            prefetch = getattr(
                self.lmcache_engine, "prefetch_shared_layer_pages", None
            )
            if callable(prefetch):
                prefetch_kwargs = dict(retrieve_kwargs)
                prefetch_kwargs.pop("_retain_shared_dense_cache")
                prefetch_kwargs.pop("_dense_load_readiness_out")
                try:
                    prefetch(
                        plan["tokens"],
                        plan["token_mask"],
                        retrieve_kwargs=prefetch_kwargs,
                    )
                    prefetch_owners = self._dense_load_source_owners(retrieve_state)
                    retrieve_state.clear_group(1)
                except Exception as error:
                    self._release_dense_load_source_owners(
                        self._dense_load_source_owners(retrieve_state),
                        self.lmcache_engine,
                        synchronize=False,
                    )
                    retrieve_state.clear_group(1)
                    logger.warning(
                        "Group-1 persistent prefetch failed for %s; using "
                        "the serial persistent path: %s",
                        request.req_id,
                        error,
                    )
        latent_gate_started = stage_start()
        try:
            plan["latent_shared_ready"].result()
        except BaseException:
            self._release_dense_load_source_owners(
                prefetch_owners,
                self.lmcache_engine,
                synchronize=False,
            )
            raise
        finish_stage("latent_gate_wait_ms", latent_gate_started)
        generator_started = stage_start()
        try:
            retriever = self.lmcache_engine.retrieve_layer(
                plan["tokens"],
                plan["token_mask"],
                **retrieve_kwargs,
            )
        except BaseException:
            self._release_dense_load_source_owners(
                prefetch_owners,
                self.lmcache_engine,
                synchronize=False,
            )
            raise
        finish_stage("generator_create_ms", generator_started)
        result = None
        load_stream_fenced = False
        try:
            layer_submit_started = stage_start()
            try:
                try:
                    result = next(retriever)
                    for _ in range(self.num_layers + 1):
                        result = next(retriever)
                except BaseException:
                    # The engine fences retained-source cleanup internally;
                    # fence any remaining load-stream work before outer-state
                    # cleanup as well.
                    self._synchronize_dsa_cold_dense_load()
                    load_stream_fenced = True
                    raise
            finally:
                retriever.close()
            finish_stage("layer_submit_host_ms", layer_submit_started)
            result_check_started = stage_start()
            if result is None or int(result.sum().item()) != plan["token_count"]:
                raise RuntimeError("Cold compact indexer retrieve was incomplete")
            if plan["token_count"] and not (
                any(retrieve_state.cached_memory_objs_indexer)
                or any(retrieve_state.cached_tensors_indexer)
            ):
                raise RuntimeError(
                    "Cold compact indexer retrieve did not retain its sources"
                )
            finish_stage("result_check_ms", result_check_started)
            readiness_started = stage_start()
            if len(readiness_out) != 1:
                raise RuntimeError("Cold compact indexer readiness was not recorded")
            readiness = readiness_out[0]
            finish_stage("readiness_record_ms", readiness_started)
        except BaseException:
            if not load_stream_fenced:
                self._synchronize_dsa_cold_dense_load()
            self._release_dense_load_source_owners(
                prefetch_owners,
                self.lmcache_engine,
                synchronize=False,
            )
            raise
        self._release_dense_load_source_owners(
            prefetch_owners,
            self.lmcache_engine,
            synchronize=False,
        )
        return result, readiness, cold_start_perf_now() - started, queue_ms

    def synchronize_staged_sfa_capture_unsafe_loads(self) -> None:
        """Finish background NPU loads before serving-time graph capture.

        Cold compact workers issue synchronous NPU copies and stream waits.
        Those operations cannot overlap an ACL graph capture in another
        thread. Completed futures remain queued for ``get_finished`` so their
        normal publication and scheduler notification semantics are preserved.
        """
        futures = getattr(self, "_dsa_cold_load_futures", None)
        if not futures:
            return
        pending_req_ids = [
            req_id
            for req_id, entry in futures.items()
            if any(
                not future.done()
                for future in (entry[1], *entry[5:6])
            )
        ]
        if pending_req_ids:
            logger.info(
                "Waiting for capture-unsafe cold compact loads before native "
                "model execution: requests=%s",
                pending_req_ids,
            )
        failed_req_ids = []
        for req_id, entry in list(futures.items()):
            request_failed = False
            for future in (entry[1], *entry[5:6]):
                try:
                    future.result()
                except BaseException:
                    request_failed = True
            if request_failed:
                failed_req_ids.append(req_id)
        if failed_req_ids:
            # A failed future is reported through get_finished(), which also
            # publishes invalid blocks to the scheduler. Only prove here that
            # no capture-unsafe stream work remains before model collectives.
            self._synchronize_dsa_cold_dense_load()
            logger.warning(
                "Cold compact loads failed before native model execution; "
                "deferring scheduler error publication to get_finished: "
                "requests=%s",
                failed_req_ids,
            )

    def _run_dsa_cold_compact_load(
        self,
        plan: dict[str, Any],
        npu_device_id: Optional[int],
        indexer_future: Future,
        previous_latent_future: Optional[Future] = None,
        live_state: Optional[WorkerRetrieveState] = None,
    ) -> WorkerRetrieveState:
        if npu_device_id is not None:
            torch.npu.set_device(npu_device_id)
        request = plan["request"]
        assert request.load_spec is not None
        assert self.lmcache_engine is not None
        latent_layers = self._num_layers_for_group(0)
        indexer_layers = self._num_layers_for_group(1)
        if latent_layers != self.num_layers or indexer_layers != self.num_layers:
            raise RuntimeError(
                "Cold compact load requires matching latent/indexer layer counts: "
                f"expected={self.num_layers}, latent={latent_layers}, "
                f"indexer={indexer_layers}"
            )
        token_count = plan["token_count"]
        tokens = plan["tokens"]
        token_mask = plan["token_mask"]
        state = live_state or WorkerRetrieveState(req_id=request.req_id)
        started = cold_start_perf_now()
        indexer_readiness = None
        try:
            if previous_latent_future is not None:
                try:
                    previous_latent_future.result()
                except BaseException:
                    # The predecessor's failure must not reorder or poison this
                    # request's fixed group-0 collective publication slot.
                    pass
            retrieve_location = "LocalCPU"
            if live_state is None:
                empty_slots = torch.empty(0, dtype=torch.long)
                retrieve_kwargs, _, _ = self._sparse_retrieve_kwargs(
                    request,
                    state,
                    None,
                    kvcaches=plan["latent_kvcaches"],
                    slot_mapping=empty_slots,
                    sync=True,
                    kv_group=0,
                    request_ordinal=0,
                    dsa_two_groups=True,
                    token_count=token_count,
                    shared_cpu_enabled=True,
                    shared_cpu_preflight_state=None,
                )
                retrieve_kwargs["materialize_only"] = True
                retrieve_kwargs["shared_cpu_phase"] = "dsa_cold_compact_latent"
                latent_retriever = (
                    self.lmcache_engine.retrieve_layer_head_token_wise(
                        tokens,
                        token_mask,
                        **retrieve_kwargs,
                    )
                )
                try:
                    latent_result = next(latent_retriever)
                    for _ in range(self.num_layers):
                        latent_result = latent_retriever.send(None)
                finally:
                    latent_retriever.close()

                if (
                    latent_result is None
                    or int(latent_result.sum().item()) != token_count
                ):
                    raise RuntimeError("Cold compact latent retrieve was incomplete")
                retrieve_location = retrieve_kwargs.get(
                    "cached_retrieve_location"
                )
            latent_shared_ready = plan["latent_shared_ready"]
            if not latent_shared_ready.done():
                latent_shared_ready.set_result(None)
            dependency_wait_started = cold_start_perf_now()
            (
                _,
                indexer_readiness,
                indexer_remote_s,
                indexer_queue_ms,
            ) = indexer_future.result()
            dependency_wait_ms = (
                cold_start_perf_now() - dependency_wait_started
            ) * 1000
            self._record_dsa_cold_dense_load_readiness(
                state,
                indexer_readiness,
                tuple(plan.get("indexer_source_owners", ())),
            )

            seal_started = cold_start_perf_now()
            state.indexer_npu_resident = True
            state.location = retrieve_location
            state.metadata_warm = state.has_cache()
            state.token_count = token_count
            self._refresh_prepared_sparse_sources(state, token_count)
            if state.prepared_sparse_sources.get(0) is None:
                raise RuntimeError("Cold compact latent source was not sealed")
            completed_at = cold_start_perf_now()
            state._dsa_cold_load_completed_at = completed_at
            cold_start_perf_log(
                logger,
                "cold_compact_retrieve_complete",
                started=started,
                req_id=request.req_id,
                tokens=token_count,
                plan_ms=round(
                    (plan["planned_at"] - plan["plan_started"]) * 1000, 3
                ),
                remote_ms=round(indexer_remote_s * 1000, 3),
                queue_ms=round(indexer_queue_ms, 3),
                event_wait_ms=round(dependency_wait_ms, 3),
                seal_ms=round((completed_at - seal_started) * 1000, 3),
                **plan.get("indexer_perf", {}),
            )
            return state
        except BaseException as exc:
            latent_shared_ready = plan["latent_shared_ready"]
            if not latent_shared_ready.done():
                latent_shared_ready.set_exception(exc)
            # Resolve the sibling result even when Group 0 failed before its
            # normal dependency wait. A successful result carries the exact
            # producer-stream fence; a failed result fenced its stream locally.
            if indexer_readiness is None:
                try:
                    _, indexer_readiness, _, _ = indexer_future.result()
                except BaseException:
                    pass
            owners_by_id = {
                id(owner): owner
                for owner in (
                    *state.dense_load_source_owners,
                    *tuple(plan.get("indexer_source_owners", ())),
                )
            }
            combined_owners = tuple(owners_by_id.values())
            try:
                if indexer_readiness is not None:
                    self._synchronize_dsa_cold_dense_readiness(indexer_readiness)
            except BaseException:
                # The transfer may still be writing. Preserve every owner on
                # the surfaced state so the scheduler can retain the blocks
                # and a later safe cleanup cannot lose the indexer future's
                # source allocation.
                state.dense_load_source_owners = combined_owners
                plan["indexer_source_owners"] = ()
                exc._lmcache_dsa_cold_state = state
                logger.exception(
                    "Cold compact cleanup could not synchronize the dense "
                    "load stream; scheduler blocks must remain retained"
                )
            else:
                # The stream is already fenced. Release both adopted and
                # not-yet-adopted indexer owners exactly once, without a
                # redundant second device synchronization.
                state.dense_load_readiness = None
                state.dense_load_readiness_consumed = False
                self._release_dense_load_source_owners(
                    combined_owners,
                    self.lmcache_engine,
                    synchronize=False,
                )
                state.dense_load_source_owners = ()
                plan["indexer_source_owners"] = ()
                self._release_unadopted_shared_request_objects(state, request)
                self._release_shared_worker_retrieve_state(
                    state, self.lmcache_engine
                )
            raise

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start this step's KV loads and atomically discard partial setup."""
        try:
            self._start_load_kv(forward_context, **kwargs)
        except BaseException:
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            self._abort_layerwise_retrieve_step(
                request
                for request in metadata.requests
                if request.load_spec is not None and request.load_spec.can_load
                and not getattr(
                    request.load_spec, "dsa_cold_compact_load", False
                )
            )
            raise

    def _start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0
        self._wait_for_save_done = False

        attn_metadata = forward_context.attn_metadata
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)
        if getattr(metadata, "dsa_cold_compact_load_pending", False):
            cold_requests = [
                request
                for request in metadata.requests
                if request.load_spec is not None
                and request.load_spec.can_load
                and getattr(
                    request.load_spec, "dsa_cold_compact_load", False
                )
            ]
            if cold_requests and not self.kv_caches:
                if attn_metadata is None:
                    raise RuntimeError(
                        "Cold compact load requires register_kv_caches "
                        "before no-forward"
                    )
                self._init_kv_caches_from_forward_context(forward_context)
            for request in cold_requests:
                if not self._try_prepare_dsa_live_split(request):
                    self._submit_dsa_cold_compact_load(request)
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        active_req_ids: set[str] = set()
        resumed_req_ids: set[str] = set()
        loadable_requests: list[tuple[int, ReqMeta]] = []
        vllm_hit_tokens = 0
        prompt_tokens = 0
        staged_load_count = 0
        has_load_spec = False
        for idx, request in enumerate(metadata.requests):
            if not self._is_decode_window_save_request(request):
                active_req_ids.add(request.req_id)
            if request.resumed_from_preemption:
                resumed_req_ids.add(request.req_id)
            load_spec = request.load_spec
            if load_spec is None:
                continue
            if getattr(load_spec, "dsa_cold_compact_load", False):
                continue
            if not load_spec.can_load and getattr(
                load_spec, "dsa_cold_compact_resume", False
            ):
                raise RuntimeError(
                    "Cold compact resume requires a prepared worker load: "
                    f"req_id={request.req_id}"
                )
            has_load_spec = True
            vllm_hit_tokens += load_spec.vllm_cached_tokens
            prompt_tokens += request.retrieve_token_count()
            if not load_spec.can_load:
                continue
            loadable_requests.append((idx, request))
            if self.use_layerwise and not request.is_sparse_decode:
                staged_load_count += 1

        self._prune_worker_retrieve_state(active_req_ids, resumed_req_ids)

        assert len(self.kv_caches) > 0
        if not self._kvcaches_list:
            self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_list

        assert self.lmcache_engine is not None

        self._drain_layerwise_retrievers()
        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        if staged_load_count and gpu_connector is not None and hasattr(
            gpu_connector, "set_layerwise_staging_concurrency"
        ):
            # Each staged load holds a buffer for the full layer loop; add one
            # slot for an overlapping layerwise store.
            gpu_connector.set_layerwise_staging_concurrency(
                max(2, staged_load_count + 1)
            )
        if has_load_spec:
            self._stats_monitor.update_interval_vllm_hit_tokens(vllm_hit_tokens)
            self._stats_monitor.update_interval_prompt_tokens(prompt_tokens)

        for load_idx, (idx, request) in enumerate(loadable_requests):
            request_perf_started = (
                cold_start_perf_now() if cold_start_perf_enabled() else 0.0
            )
            tokens = request.token_ids
            assert request.load_spec is not None
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            sparse_bound_state = (
                self._worker_retrieve_state_for_warm_ref(request)
                if request.sparse_warm_ref
                else None
            )

            if sparse_bound_state is not None:
                assert sparse_bound_state.slot_mapping is not None
                slot_mapping = sparse_bound_state.slot_mapping
            elif request.is_sparse_decode:
                assert request.slot_mapping
                if (
                    request.slot_mapping[0].device.type
                    != torch.device(self.device).type
                    or request.slot_mapping[0].dtype != torch.long
                ):
                    request.slot_mapping[0] = request.slot_mapping[0].to(
                        device=self.device, dtype=torch.long
                    )
                slot_mapping = request.slot_mapping[0]
            else:
                assert request.slot_mapping
                slot_mapping = request.slot_mapping[0].to(
                    device=self.device, dtype=torch.long
                )

            if not request.is_sparse_decode:
                assert len(tokens) == len(slot_mapping)

            retrieve_tokens = (
                []
                if request.sparse_warm_ref
                else self._load_tokens_for_retrieve(
                    tokens,
                    lmcache_cached_tokens,
                    is_sparse_decode=request.is_sparse_decode,
                )
            )
            recalc_last_applied = self._full_hit_recalc_last_token(
                request.load_spec,
                request.retrieve_token_count(),
                is_sparse_decode=request.is_sparse_decode,
            )
            if recalc_last_applied:
                retrieve_tokens, slot_mapping = self._trim_prefill_for_recalc_last(
                    request, retrieve_tokens, slot_mapping
                )
            token_count = (
                lmcache_cached_tokens
                if request.sparse_warm_ref
                else len(retrieve_tokens)
            )
            token_mask = (
                None
                if request.sparse_warm_ref
                else self._load_token_mask_for_retrieve(
                    request, token_count, self._lmcache_chunk_size
                )
            )
            indexer_token_mask = token_mask
            if (
                not request.is_sparse_decode
                and token_count > len(slot_mapping)
            ):
                logger.warning(
                    "Request %s: retrieve_len=%d exceeds slot_mapping len=%d "
                    "(KV scatter will be incomplete -> garbage). "
                    "Often chunked-prefill metadata out of sync with lookup_hit.",
                    request.req_id,
                    token_count,
                    len(slot_mapping),
                )

            if self.use_layerwise or request.is_sparse_decode:
                sync = load_idx == len(loadable_requests) - 1
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    if token_mask is None:
                        token_mask = torch.ones(token_count, dtype=torch.bool)
                    self.blender.blend(
                        retrieve_tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                elif request.is_sparse_decode:
                    prior_retrieve_state = sparse_bound_state
                    prior_retrieve_snapshot = None
                    invalidation_reason = None
                    retrieve_state_invalidated = False
                    if hasattr(self, "_worker_retrieve_state"):
                        if prior_retrieve_state is None:
                            prior_retrieve_state = self._worker_retrieve_state.get(
                                request.req_id
                            )
                        if (
                            prior_retrieve_state is not None
                            and not request.sparse_warm_ref
                        ):
                            self._trim_dense_prefix_seed_for_sparse(
                                prior_retrieve_state,
                                token_count,
                            )
                        if (
                            _mtp_dw_deep_diag_enabled()
                            and prior_retrieve_state is not None
                        ):
                            prior_retrieve_snapshot = {
                                "frontier": prior_retrieve_state.token_count,
                                "kv_group0_cached_ranges": (
                                    self._deep_retrieve_range_summary(
                                        prior_retrieve_state.cached_starts,
                                        prior_retrieve_state.cached_ends,
                                    )
                                ),
                                "kv_group1_cached_ranges": (
                                    self._deep_retrieve_range_summary(
                                        prior_retrieve_state.cached_starts_indexer,
                                        prior_retrieve_state.cached_ends_indexer,
                                    )
                                ),
                                "kv_group0_cache_present": (
                                    self._deep_retrieve_group_cache_present(
                                        prior_retrieve_state, 0
                                    )
                                ),
                                "kv_group1_cache_present": (
                                    self._deep_retrieve_group_cache_present(
                                        prior_retrieve_state, 1
                                    )
                                ),
                                "scope_token_present": (
                                    prior_retrieve_state.request_scope_token is not None
                                ),
                                "scope_token": (
                                    prior_retrieve_state.request_scope_token
                                ),
                                "shared_request_active": (
                                    prior_retrieve_state.shared_request_active
                                ),
                                "shared_generation": (
                                    prior_retrieve_state.shared_generation
                                ),
                                "pointer_cache_generation": (
                                    prior_retrieve_state.pointer_cache_generation
                                ),
                            }
                        retrieve_state_invalidated = (
                            not request.sparse_warm_ref
                            and self._should_invalidate_worker_retrieve_state(
                                request, token_count
                            )
                        )
                        if retrieve_state_invalidated:
                            if _mtp_dw_deep_diag_enabled():
                                invalidation_reason = "resumed_from_preemption"
                                if prior_retrieve_state is not None:
                                    invalidation_reason = (
                                        self._worker_retrieve_state_invalidation_reason(
                                            request,
                                            token_count,
                                            prior_retrieve_state,
                                        )
                                    )
                            self._drop_worker_retrieve_state(request.req_id)
                        bound_state = (
                            sparse_bound_state
                            if sparse_bound_state is not None
                            else self._worker_retrieve_state_for_request(request)
                        )
                    else:
                        bound_state = None

                    retrieve_state = bound_state
                    if retrieve_state is None:
                        retrieve_state = WorkerRetrieveState(req_id=request.req_id)
                    if not request.sparse_warm_ref:
                        retrieve_state.slot_mapping = slot_mapping
                        if request.decode_ret_mask is not None:
                            retrieve_state.decode_ret_mask = request.decode_ret_mask

                    dsa_two_groups = self._is_dsa_two_groups()
                    shared_cpu_enabled = bool(
                        getattr(
                            self.lmcache_engine,
                            "enable_shared_cpu_cache",
                            False,
                        )
                    )
                    shared_cpu_preflight_state: Optional[dict[str, Any]] = None
                    (
                        retrieve_kwargs,
                        shared_cpu_preflight_state,
                        latent_prepared,
                    ) = self._sparse_retrieve_kwargs(
                        request,
                        retrieve_state,
                        bound_state,
                        kvcaches=kvcaches,
                        token_count=token_count,
                        slot_mapping=slot_mapping,
                        sync=sync,
                        kv_group=0,
                        request_ordinal=idx,
                        dsa_two_groups=dsa_two_groups,
                        shared_cpu_enabled=shared_cpu_enabled,
                        shared_cpu_preflight_state=shared_cpu_preflight_state,
                    )
                    if latent_prepared is None and getattr(
                        request.load_spec,
                        "dsa_cold_compact_resume",
                        False,
                    ):
                        raise RuntimeError(
                            "Cold compact resume lost its prepared Group-0 "
                            f"source: req_id={request.req_id}, "
                            f"tokens={token_count}"
                        )
                    cold_perf_active = bool(
                        request_perf_started and latent_prepared is None
                    )
                    dense_completed = self._cold_perf_dense_load_completed.pop(
                        request.req_id,
                        None,
                    )
                    if cold_perf_active:
                        load_started = cold_start_perf_now()
                        self._cold_perf_load_started[request.req_id] = (
                            request_perf_started,
                            token_count,
                        )
                        cold_start_perf_log(
                            logger,
                            "worker_load_start",
                            req_id=request.req_id,
                            tokens=token_count,
                            layers=getattr(self.lmcache_engine, "num_layers", 0),
                            dsa_two_groups=dsa_two_groups,
                            setup_ms=round(
                                (load_started - request_perf_started) * 1000,
                                3,
                            ),
                            post_dense_gap_ms=(
                                round((load_started - dense_completed) * 1000, 3)
                                if dense_completed is not None
                                else None
                            ),
                            rank=getattr(
                                getattr(self.lmcache_engine, "metadata", None),
                                "worker_id",
                                None,
                            ),
                        )

                    layerwise_retriever = (
                        self.lmcache_engine.retrieve_layer_head_token_wise(
                            retrieve_tokens,
                            token_mask,
                            **retrieve_kwargs,
                        )
                    )
                    self.layerwise_retrievers.append(
                        (layerwise_retriever, None)
                    )
                    self._layerwise_requests.append(request)
                    self._layerwise_retriever_is_sparse.append(True)
                    self._layerwise_sparse_req_ids.append(request.req_id)
                    self._layerwise_sparse_shared_ordered.append(False)
                    # NOTE: retrieve layers one by one with cpu prefetch
                    prime_started = (
                        cold_start_perf_now() if cold_perf_active else 0.0
                    )
                    next(layerwise_retriever)
                    if prime_started:
                        cold_start_perf_log(
                            logger,
                            "prime_group",
                            started=prime_started,
                            req_id=request.req_id,
                            kv_group=0,
                            tokens=token_count,
                            prepared=False,
                            scope="blocking_wall",
                        )

                    indexer_retriever = None
                    indexer_skipped = False
                    if dsa_two_groups:
                        indexer_kvcaches = self._kvcaches_for_group(1)
                        materialize_index = (
                            self._sparse_decode_requires_index_materialization(
                                request,
                                shared_cpu_enabled,
                            )
                        )
                        indexer_mode = INDEXER_RETRIEVE_FULL
                        if shared_cpu_enabled and materialize_index:
                            indexer_mode = (
                                self._shared_sparse_decode_indexer_retrieve_mode(
                                    request,
                                    bound_state,
                                    token_count,
                                )
                            )
                        if (
                            shared_cpu_enabled
                            and not materialize_index
                        ):
                            indexer_skipped = True
                        elif (
                            shared_cpu_enabled
                            and materialize_index
                            and indexer_mode == INDEXER_RETRIEVE_RESIDENT_SKIP
                        ):
                            logger.debug(
                                "Skipping shared CPU DSA index retrieve for "
                                "resident live sparse decode state: req_id=%s "
                                "token_count=%d",
                                request.req_id,
                                token_count,
                            )
                        elif not materialize_index:
                            indexer_skipped = True
                        elif not indexer_kvcaches:
                            raise RuntimeError(
                                "Sparse decode with dsa_two_groups=true "
                                "requires DSA index kvcaches for kv_group=1."
                            )
                        else:
                            indexer_setup_started = (
                                cold_start_perf_now()
                                if cold_perf_active
                                else 0.0
                            )
                            latent_sparse_slots = (
                                slot_mapping[0]
                                if isinstance(slot_mapping, list)
                                else slot_mapping
                            )
                            request_indexer_slots = (
                                retrieve_state.indexer_slot_mapping
                                if request.sparse_warm_ref
                                else (
                                    request.indexer_slot_mapping[0]
                                    if request.indexer_slot_mapping
                                    else None
                                )
                            )
                            if (
                                request_indexer_slots is not None
                                and (
                                    request_indexer_slots.device.type
                                    != torch.device(self.device).type
                                    or request_indexer_slots.dtype != torch.long
                                )
                            ):
                                request_indexer_slots = request_indexer_slots.to(
                                    device=self.device, dtype=torch.long
                                )
                                if request.sparse_warm_ref:
                                    retrieve_state.indexer_slot_mapping = (
                                        request_indexer_slots
                                    )
                                else:
                                    request.indexer_slot_mapping[0] = (
                                        request_indexer_slots
                                    )
                            idx_slot = self._sparse_indexer_slot_mapping(
                                attn_metadata,
                                latent_sparse_slots,
                                request.load_spec.lmcache_cached_tokens,
                                request_indexer_slots=request_indexer_slots,
                            )
                            assert idx_slot is not None
                            if not request.sparse_warm_ref:
                                retrieve_state.indexer_slot_mapping = idx_slot
                            (
                                indexer_kwargs,
                                shared_cpu_preflight_state,
                                indexer_prepared,
                            ) = self._sparse_retrieve_kwargs(
                                request,
                                retrieve_state,
                                bound_state,
                                kvcaches=indexer_kvcaches,
                                token_count=token_count,
                                slot_mapping=idx_slot,
                                sync=sync,
                                kv_group=1,
                                request_ordinal=idx,
                                dsa_two_groups=dsa_two_groups,
                                shared_cpu_enabled=shared_cpu_enabled,
                                shared_cpu_preflight_state=shared_cpu_preflight_state,
                                metadata_only=(
                                    indexer_mode
                                    == INDEXER_RETRIEVE_METADATA_ONLY
                                ),
                            )
                            indexer_retriever = (
                                self.lmcache_engine.retrieve_layer_head_token_wise(
                                    retrieve_tokens,
                                    token_mask,
                                    **indexer_kwargs,
                                )
                            )
                            self.layerwise_retrievers[-1] = (
                                layerwise_retriever,
                                indexer_retriever,
                            )
                            shared_group_retrieve = getattr(
                                self.lmcache_engine,
                                "_should_use_shared_layerwise_retrieve",
                                None,
                            )
                            self._layerwise_sparse_shared_ordered[-1] = bool(
                                shared_cpu_enabled
                                and latent_prepared is None
                                and indexer_prepared is None
                                and callable(shared_group_retrieve)
                                and all(
                                    shared_group_retrieve(group)
                                    for group in (0, 1)
                                )
                            )
                            if indexer_mode == INDEXER_RETRIEVE_FULL:
                                retrieve_state.indexer_npu_materialization_pending = (
                                    True
                                )
                            prime_started = (
                                cold_start_perf_now()
                                if cold_perf_active
                                else 0.0
                            )
                            next(indexer_retriever)
                            if prime_started:
                                cold_start_perf_log(
                                    logger,
                                    "prime_group",
                                    started=prime_started,
                                    req_id=request.req_id,
                                    kv_group=1,
                                    tokens=token_count,
                                    prepared=False,
                                    scope="blocking_wall",
                                    setup_ms=round(
                                        (prime_started - indexer_setup_started)
                                        * 1000,
                                        3,
                                    ),
                                )

                    if indexer_skipped:
                        request.shared_index_skipped = True
                        retrieve_state.clear_group(1)
                        retrieve_state.indexer_npu_resident = False
                        retrieve_state.indexer_npu_materialization_pending = False
                        if cold_perf_active:
                            cold_start_perf_log(
                                logger,
                                "prime_group",
                                req_id=request.req_id,
                                kv_group=1,
                                tokens=token_count,
                                status="skipped",
                            )
                    retrieve_location = retrieve_kwargs.get(
                        "cached_retrieve_location"
                    )
                    metadata_warm = bool(
                        retrieve_kwargs.get("_retrieve_metadata_warm")
                        or retrieve_state.has_cache()
                    )
                    if latent_prepared is None:
                        self._set_worker_retrieve_state(
                            request.req_id, retrieve_state
                        )
                        retrieve_state.location = (
                            retrieve_location or retrieve_state.location
                        )
                        retrieve_state.metadata_warm = (
                            metadata_warm or retrieve_state.metadata_warm
                        )
                        logger.debug(
                            "Deferring sparse retrieve state publication until "
                            "layer loading completes: req_id=%s",
                            request.req_id,
                        )
                    post_retrieve_state = (
                        self._worker_retrieve_state.get(request.req_id)
                        if hasattr(self, "_worker_retrieve_state")
                        else None
                    )
                    self._trace_deep_retrieve_state(
                        request,
                        prior_retrieve_state,
                        prior_retrieve_snapshot,
                        invalidated=retrieve_state_invalidated,
                        invalidation_reason=invalidation_reason,
                        post_state=post_retrieve_state,
                        prior_state_rebound=bound_state is prior_retrieve_state,
                        kv_group0_retriever_present=layerwise_retriever is not None,
                        kv_group1_retriever_present=indexer_retriever is not None,
                    )
                else:
                    if request_perf_started:
                        self._cold_perf_dense_load_completed.pop(
                            request.req_id,
                            None,
                        )
                        self._cold_perf_dense_load_started[request.req_id] = (
                            request_perf_started,
                            token_count,
                        )
                    retrieve_slot_mapping = slot_mapping
                    if lmcache_cached_tokens < len(slot_mapping):
                        retrieve_slot_mapping = slot_mapping[:lmcache_cached_tokens]
                    retrieve_state = self._worker_retrieve_state.get(
                        request.req_id
                    )
                    if retrieve_state is None:
                        retrieve_state = WorkerRetrieveState(req_id=request.req_id)
                    dsa_two_groups = self._is_dsa_two_groups()
                    shared_cpu_enabled = bool(
                        getattr(
                            self.lmcache_engine,
                            "enable_shared_cpu_cache",
                            False,
                        )
                    )
                    supports_dense_retention = getattr(
                        self.lmcache_engine,
                        "supports_dense_sparse_cache_retention",
                        None,
                    )
                    retain_dense_seed = (
                        shared_cpu_enabled
                        and getattr(self, "enable_sparse_attention", False)
                        and callable(supports_dense_retention)
                        and supports_dense_retention()
                    )
                    if retain_dense_seed and (
                        retrieve_state.shared_request_active
                        or retrieve_state.dense_prefix_seed
                        or retrieve_state.group_has_data(0, dsa_two_groups)
                        or (
                            dsa_two_groups
                            and retrieve_state.group_has_data(
                                1,
                                dsa_two_groups=True,
                            )
                        )
                    ):
                        self._release_shared_worker_retrieve_state(
                            retrieve_state,
                            self.lmcache_engine,
                        )
                        retrieve_state = WorkerRetrieveState(req_id=request.req_id)
                    dense_preflight_state: dict[str, Any] = {}
                    latent_cache = retrieve_state.cache_kwargs(
                        0,
                        dsa_two_groups,
                    )
                    indexer_cache = (
                        retrieve_state.cache_kwargs(1, dsa_two_groups=True)
                        if dsa_two_groups
                        else None
                    )
                    if retain_dense_seed:
                        retrieve_state.dense_prefix_seed = True
                        retrieve_state.metadata_warm = True
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        retrieve_tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=retrieve_slot_mapping,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                        kv_group=0,
                        req_id=request.req_id,
                        request_configs=request.request_configs,
                        shared_cpu_request_ordinal=idx,
                        shared_cpu_request_preflight_state=dense_preflight_state,
                        _retain_shared_dense_cache=retain_dense_seed,
                        **(latent_cache if retain_dense_seed else {}),
                    )
                    self.layerwise_retrievers.append(
                        (layerwise_retriever, None)
                    )
                    self._layerwise_requests.append(request)
                    self._layerwise_retriever_is_sparse.append(False)
                    self._layerwise_sparse_shared_ordered.append(False)

                    # Two-group DSA: also retrieve the indexer group (kv_group=1)
                    # for the same latent hit token count, scattering into vLLM's
                    # indexer KV via the indexer slot mapping. Decode stays
                    # latent-only (this branch is prefill/prefix, not sparse).
                    indexer_retriever = None
                    indexer_kvcaches = []
                    idx_slot = None
                    if dsa_two_groups:
                        indexer_kvcaches = self._kvcaches_for_group(1)
                        if not indexer_kvcaches:
                            raise RuntimeError(
                                "Dense prefix retrieval with dsa_two_groups=true "
                                "requires DSA index kvcaches for kv_group=1."
                            )
                    if dsa_two_groups and indexer_kvcaches:
                        indexer_layer_name = (
                            self._indexer_layer_names[0]
                            if self._indexer_layer_names
                            else None
                        )
                        if request.indexer_slot_mapping:
                            idx_slot = request.indexer_slot_mapping[0].to(
                                device=self.device, dtype=torch.long
                            )
                            if lmcache_cached_tokens < len(idx_slot):
                                idx_slot = idx_slot[:lmcache_cached_tokens]
                            if len(idx_slot) < lmcache_cached_tokens:
                                idx_slot = None
                        if idx_slot is None:
                            idx_slot = self._indexer_retrieve_slot_mapping(
                                attn_metadata,
                                lmcache_cached_tokens,
                                indexer_layer_name,
                            )
                        if idx_slot is None:
                            raise RuntimeError(
                                "Dense prefix retrieval with "
                                "dsa_two_groups=true could not resolve the "
                                "Group-1 index slot mapping. Refusing to mix "
                                "loaded Group-0 latent KV with stale or "
                                "uninitialized index rows."
                            )
                        assert indexer_cache is not None
                        indexer_retriever = self.lmcache_engine.retrieve_layer(
                            retrieve_tokens,
                            indexer_token_mask,
                            kvcaches=indexer_kvcaches,
                            slot_mapping=idx_slot,
                            vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                            sync=sync,
                            kv_group=1,
                            req_id=request.req_id,
                            request_configs=request.request_configs,
                            shared_cpu_request_ordinal=idx,
                            shared_cpu_request_preflight_state=(
                                dense_preflight_state
                            ),
                            _retain_shared_dense_cache=retain_dense_seed,
                            **(indexer_cache if retain_dense_seed else {}),
                        )
                        self.layerwise_retrievers[-1] = (
                            layerwise_retriever,
                            indexer_retriever,
                        )

                    # Prime the same two-step window as the legacy dense path,
                    # but interleave groups so shared-cache collectives remain
                    # layer-major: latent L0, index L0, latent L1, index L1.
                    self._prime_dense_prefix_retrievers(
                        layerwise_retriever,
                        indexer_retriever,
                    )

                    if retain_dense_seed:
                        retrieve_state.location = "LocalCPUBackend"
                        retrieve_state.token_count = len(retrieve_tokens)
                        self._set_worker_retrieve_state(
                            request.req_id,
                            retrieve_state,
                        )
                        continue
                    prefix_location, metadata_warm = (
                        self._warm_request_retrieve_metadata(
                            retrieve_state,
                            request,
                            retrieve_tokens,
                            token_mask,
                            kv_group=0,
                            dsa_two_groups=dsa_two_groups,
                        )
                    )
                    indexer_metadata_warm = False
                    if indexer_retriever is not None:
                        indexer_location, indexer_metadata_warm = (
                            self._warm_request_retrieve_metadata(
                                retrieve_state,
                                request,
                                retrieve_tokens,
                                token_mask,
                                kv_group=1,
                                dsa_two_groups=dsa_two_groups,
                            )
                        )
                        if prefix_location is None:
                            prefix_location = indexer_location
                    if prefix_location is None:
                        prefix_location = self._resolve_store_retrieve_location(
                            retrieve_state
                        )
                    metadata_warm = bool(metadata_warm or indexer_metadata_warm)
                    self._publish_worker_retrieve_state(
                        retrieve_state,
                        request,
                        location=prefix_location,
                        metadata_warm=metadata_warm,
                        token_count=lmcache_cached_tokens,
                    )
            else:
                retrieve_slot_mapping = slot_mapping
                if lmcache_cached_tokens < len(slot_mapping):
                    retrieve_slot_mapping = slot_mapping[:lmcache_cached_tokens]
                ret_token_mask = self.lmcache_engine.retrieve(
                    retrieve_tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=retrieve_slot_mapping,
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                self._validate_dense_retrieve_result(
                    request,
                    ret_token_mask,
                    kv_group=0,
                    slot_mapping=retrieve_slot_mapping,
                    expected_mask=token_mask,
                    recalc_last_applied=recalc_last_applied,
                )

    def _validate_dense_retrieve_result(
        self,
        request: ReqMeta,
        ret_mask: torch.Tensor,
        *,
        kv_group: int,
        slot_mapping: Optional[torch.Tensor] = None,
        expected_mask: Optional[torch.Tensor] = None,
        recalc_last_applied: Optional[bool] = None,
    ) -> None:
        """Fail a cache hit when any required KV group is incomplete.

        Layerwise retrieval returns one completion mask per group.  A request
        is usable only when every enabled group reaches the scheduler's common
        hit frontier; accepting a short Group-1 mask mixes valid latent KV with
        missing/stale index rows and produces silent quality corruption.
        """
        assert request.load_spec is not None
        load_spec = request.load_spec
        if recalc_last_applied is None:
            recalc_last_applied = self._full_hit_recalc_last_token(
                load_spec,
                request.retrieve_token_count(),
                is_sparse_decode=request.is_sparse_decode,
            )
        token_count = min(
            int(load_spec.lmcache_cached_tokens),
            request.retrieve_token_count(),
        )
        if expected_mask is None:
            expected_mask = self._load_token_mask_for_retrieve(
                request,
                token_count,
                self._lmcache_chunk_size,
            )
        if expected_mask is None:
            raise RuntimeError(
                "Dense retrieve validation has no expected token mask: "
                f"req_id={request.req_id}, kv_group={kv_group}"
            )
        expected_mask = expected_mask[:token_count].clone()
        if recalc_last_applied and expected_mask.numel():
            # vLLM only recomputes the final prompt token.  A count-only
            # allowance would incorrectly accept a missing token anywhere in
            # the prefix when the final token happened to load successfully.
            expected_mask[-1] = False
        num_expected_tokens = int(expected_mask.sum().item())
        ret_mask_for_validation = ret_mask[:token_count]
        masks_align = ret_mask_for_validation.shape == expected_mask.shape
        num_retrieved_tokens = (
            int((ret_mask_for_validation & expected_mask).sum().item())
            if masks_align
            else int(ret_mask.sum().item())
        )
        if masks_align and bool(
            torch.all(ret_mask_for_validation | ~expected_mask).item()
        ):
            return

        logger.error(
            "Request %s KV group %d did not retrieve every required token: "
            "retrieved=%d expected=%d mask_shape=%s expected_shape=%s",
            request.req_id,
            kv_group,
            num_retrieved_tokens,
            num_expected_tokens,
            tuple(ret_mask.shape),
            tuple(expected_mask.shape),
        )
        # vLLM's compact-external validation frontier is owned by the
        # non-latent (Group-1) manager.  Therefore a short read in *either*
        # cache group must be reported using Group-1 block ids when two-group
        # DSA is active.  Reporting Group-0 ids here can miss the request
        # entirely (or collide with an unrelated Group-1 allocation), letting
        # execution continue with a mixed/incomplete cache pair.
        validation_uses_indexer = self._is_dsa_two_groups()
        if validation_uses_indexer:
            mappings = request.indexer_slot_mapping
            slot_mapping = mappings[0] if mappings else None
        elif slot_mapping is None:
            mappings = (
                request.indexer_slot_mapping
                if kv_group == 1
                else request.slot_mapping
            )
            if not mappings:
                raise RuntimeError(
                    "Incomplete dense retrieve has no destination mapping: "
                    f"req_id={request.req_id}, kv_group={kv_group}"
                )
            slot_mapping = mappings[0]
        if slot_mapping is None:
            raise RuntimeError(
                "Incomplete two-group dense retrieve has no Group-1 "
                "validation mapping: "
                f"req_id={request.req_id}, failed_kv_group={kv_group}"
            )
        retrieve_slot_mapping = slot_mapping[:token_count]
        missing_blocks = self.record_failed_blocks(
            request.req_id,
            expected_mask,
            ret_mask,
            retrieve_slot_mapping,
        )
        if not missing_blocks:
            # A malformed backend mask must not bypass invalidation merely
            # because its shape cannot be paired with the expected mask.
            mapping_cpu = retrieve_slot_mapping.to(device="cpu", dtype=torch.long)
            expected_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
            usable = min(int(mapping_cpu.numel()), int(expected_cpu.numel()))
            mapped = mapping_cpu[:usable][expected_cpu[:usable]]
            missing_blocks = {
                int(block)
                for block in torch.unique(mapped // self._block_size).tolist()
                if int(block) >= 0
            }
        if not missing_blocks:
            raise RuntimeError(
                "Incomplete dense retrieve could not identify invalid blocks: "
                f"req_id={request.req_id}, kv_group={kv_group}, "
                f"retrieved={num_retrieved_tokens}, "
                f"expected={num_expected_tokens}"
            )
        self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        mapped_missing_mask = missing_mask[: slot_mapping_cpu.shape[0]]
        missing_indices = torch.nonzero(mapped_missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {
            int(block.item())
            for block in missing_blocks_tensor
            if int(block.item()) >= 0
        }

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @contextmanager
    def _sparse_retrieve_state_guard(
        self,
        requests: Iterable[ReqMeta],
    ) -> Generator[None, None, None]:
        try:
            yield
        except BaseException:
            self._abort_layerwise_retrieve_step(requests)
            raise

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list = None,
        token_start_index: list = None,
        request_ids: list = None,
        target_slot_mapping=None,
        payload_event=None,
        selected_token_counts=None,
    ) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
            selected_tokens: sparse token indices per decode row.
            token_start_index: legacy per-row start offset into slot_mapping.
            request_ids: req_id for each selected_tokens row (duplicates allowed).
            target_slot_mapping: optional batched physical destination slots,
                row-aligned with selected_tokens.
            selected_token_counts: number of real selected tokens per row in
                an explicit sparse payload. Padding beyond each count is not
                transferred to the NPU KV cache.
            payload_event: optional producer event recorded by vLLM after
                selected_tokens/target_slot_mapping/selected_token_counts were
                built. LMCache waits on this before row-selecting those tensors.
        """
        if self.layerwise_retrievers and logger.isEnabledFor(10):
            logger.debug("Waiting for layer %d to be loaded", self.current_layer)

        if not self.layerwise_retrievers:
            return

        metadata: Optional[LMCacheConnectorMetadata] = None

        layerwise_requests = getattr(self, "_layerwise_requests", None)
        if not layerwise_requests:
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            layerwise_requests = [
                request
                for request in metadata.requests
                if request.load_spec is not None and request.load_spec.can_load
            ]

        rows_of_req = None
        if request_ids is not None:
            sparse_req_ids = getattr(self, "_layerwise_sparse_req_ids", None)
            if sparse_req_ids is None:
                if metadata is None:
                    metadata = self._parent._get_connector_metadata()
                    assert isinstance(metadata, LMCacheConnectorMetadata)
                sparse_req_ids = [
                    request.req_id
                    for request in metadata.requests
                    if request.load_spec is not None
                    and request.load_spec.can_load
                    and request.is_sparse_decode
                ]
            row_groups_key = (tuple(request_ids), tuple(sparse_req_ids))
            if (
                getattr(self, "_layerwise_sparse_row_groups_key", None)
                == row_groups_key
            ):
                rows_of_req = getattr(
                    self, "_layerwise_sparse_row_groups", None
                )
            else:
                ordered_sparse_rows = (
                    len(request_ids) == len(sparse_req_ids)
                    and request_ids == sparse_req_ids
                )
                if not ordered_sparse_rows:
                    rows_of_req = {}
                    for row, rid in enumerate(request_ids):
                        rows_of_req.setdefault(rid, []).append(row)
                self._layerwise_sparse_row_groups_key = row_groups_key
                self._layerwise_sparse_row_groups = rows_of_req

        selected_rows = None
        if selected_tokens is not None:
            # After this wait, row selection and connector-side packing are
            # ordered by the current stream; the load stream later waits on it.
            _dsa_wait_payload_event(payload_event)
            selected_rows = (
                int(selected_tokens.shape[0])
                if hasattr(selected_tokens, "shape")
                and len(selected_tokens.shape) > 0
                else len(selected_tokens)
            )

        wait_group = self._layerwise_wait_group(layer_name)
        parsed_layer_id = None
        parsed_layer_id_loaded = False
        sparse_indexer_sent_layers = None

        join_context = nullcontext()
        sparse_retrievers = getattr(self, "_layerwise_retriever_is_sparse", ())
        if (
            len(self.layerwise_retrievers) > 1
            and len(sparse_retrievers) == len(self.layerwise_retrievers)
            and all(sparse_retrievers)
        ):
            gpu_connector = getattr(
                getattr(self, "lmcache_engine", None),
                "gpu_connector",
                None,
            )
            defer_consumer_wait = getattr(
                gpu_connector,
                "defer_sparse_load_consumer_wait",
                None,
            )
            if callable(defer_consumer_wait):
                join_context = defer_consumer_wait()

        with self._sparse_retrieve_state_guard(layerwise_requests), join_context:
            if len(self.layerwise_retrievers) != len(layerwise_requests):
                raise RuntimeError(
                    "Layerwise retrieve request/retriever count mismatch; "
                    "refusing to continue with a partially initialized cache "
                    f"step: requests={len(layerwise_requests)}, "
                    f"retrievers={len(self.layerwise_retrievers)}, "
                    f"layer={layer_name}"
                )
            idx = 0
            decode_row = 0
            for request in layerwise_requests:
                if idx >= len(self.layerwise_retrievers):
                    raise RuntimeError(
                        "Layerwise retrieve lost the retriever for request "
                        f"{request.req_id}: idx={idx}, "
                        f"retrievers={len(self.layerwise_retrievers)}, "
                        f"layer={layer_name}"
                    )
                layerwise_retriever, indexer_retriever = self.layerwise_retrievers[idx]
                if (
                    wait_group == 1
                    and not request.is_sparse_decode
                    and indexer_retriever is None
                ):
                    raise RuntimeError(
                        "Dense two-group prefix load reached a Group-1 layer "
                        "without a Group-1 retriever; refusing to decode with "
                        f"partial cache state: req_id={request.req_id}, "
                        f"layer={layer_name}"
                    )
                if wait_group == 1:
                    state = self._worker_retrieve_state.get(request.req_id)
                    if state is not None:
                        self._consume_dsa_cold_dense_load_readiness(state)
                shared_ordered_retrievers = getattr(
                    self, "_layerwise_sparse_shared_ordered", ()
                )
                shared_ordered = bool(
                    idx < len(shared_ordered_retrievers)
                    and shared_ordered_retrievers[idx]
                )
                if request.is_sparse_decode:
                    payload = None
                    rows = None
                    row_count = 1
                    row_selection_requires_event = False
                    selected_token_counts_per_req = None
                    if selected_tokens is None:
                        selected_tokens_per_req = None
                        token_start_index_per_req = 0
                        target_slot_mapping_per_req = None
                    else:
                        assert selected_rows is not None
                        if rows_of_req is None:
                            row = decode_row
                            if row >= selected_rows:
                                raise RuntimeError(
                                    "Sparse decode row out of bounds for "
                                    f"layer={layer_name} req={request.req_id} "
                                    f"rows={[row]} selected_rows={selected_rows}"
                                )
                            selected_tokens_per_req = _single_row_select(
                                selected_tokens, row
                            )
                        else:
                            if request.req_id not in rows_of_req:
                                raise RuntimeError(
                                    "Missing sparse decode row for "
                                    f"layer={layer_name} req={request.req_id} "
                                    f"sparse_decode_row={decode_row}"
                                )
                            rows = rows_of_req[request.req_id]
                            row_count = len(rows)
                            row_selection_requires_event = (
                                _contiguous_row_slice(rows) is None
                            )
                            if max(rows) >= selected_rows:
                                raise RuntimeError(
                                    "Sparse decode row out of bounds for "
                                    f"layer={layer_name} req={request.req_id} "
                                    f"rows={rows} selected_rows={selected_rows}"
                                )
                            selected_tokens_per_req = _row_select(selected_tokens, rows)
                        if target_slot_mapping is not None:
                            if rows_of_req is None:
                                target_slot_mapping_per_req = _single_row_select(
                                    target_slot_mapping, row
                                )
                            else:
                                target_slot_mapping_per_req = _row_select(
                                    target_slot_mapping, rows
                                )
                            if selected_token_counts is not None:
                                selected_token_counts_per_req = (
                                    _single_row_select(selected_token_counts, row)
                                    if rows_of_req is None
                                    else _row_select(selected_token_counts, rows)
                                )
                            selected_tokens_payload = _sparse_payload_value(
                                selected_tokens_per_req
                            )
                            target_slot_mapping_payload = _sparse_payload_value(
                                target_slot_mapping_per_req
                            )
                            selected_token_counts_payload = _sparse_payload_value(
                                selected_token_counts_per_req
                            )
                            local_payload_event = (
                                _dsa_record_payload_event_if_needed(
                                    selected_tokens_payload,
                                    target_slot_mapping_payload,
                                    selected_token_counts_payload,
                                )
                                if row_selection_requires_event
                                else None
                            )
                            if (
                                local_payload_event is not None
                                or selected_token_counts_payload is not None
                            ):
                                payload = {
                                    "selected_token_ids": selected_tokens_payload,
                                    "target_slot_mapping": target_slot_mapping_payload,
                                    "selected_token_counts": (
                                        selected_token_counts_payload
                                    ),
                                }
                                if local_payload_event is not None:
                                    payload["payload_event"] = local_payload_event
                            else:
                                payload = (
                                    selected_tokens_payload,
                                    None,
                                    target_slot_mapping_payload,
                                )
                            token_start_index_per_req = None
                        else:
                            token_start_index_per_req = (
                                0
                                if token_start_index is None
                                else (
                                    _single_row_select(token_start_index, row)
                                    if rows_of_req is None
                                    else _row_select(token_start_index, rows)
                                )
                            )
                            selected_tokens_payload = _sparse_payload_value(
                                selected_tokens_per_req
                            )
                            token_start_payload = _sparse_payload_value(
                                token_start_index_per_req
                            )
                            local_payload_event = (
                                _dsa_record_payload_event_if_needed(
                                    selected_tokens_payload,
                                    token_start_payload,
                                )
                                if row_selection_requires_event
                                else None
                            )
                            if local_payload_event is not None:
                                payload = {
                                    "selected_token_ids": selected_tokens_payload,
                                    "token_start_index": token_start_payload,
                                    "payload_event": local_payload_event,
                                }
                            else:
                                selected_tokens_per_req = selected_tokens_payload
                                token_start_index_per_req = token_start_payload
                    if self.current_layer == 0 and wait_group == 0:
                        self._record_sparse_retrieve_stats(
                            selected_tokens_per_req,
                            selected_token_counts_per_req,
                            row_count,
                        )
                    sparse_payload = (
                        payload
                        if payload is not None
                        else (selected_tokens_per_req, token_start_index_per_req)
                    )
                    indexer_sent_key = (
                        (request.req_id, self.current_layer)
                        if indexer_retriever is not None
                        else None
                    )
                    if indexer_retriever is not None:
                        if not parsed_layer_id_loaded:
                            parsed_layer_id = self._layerwise_layer_id_from_name(
                                layer_name
                            )
                            parsed_layer_id_loaded = True
                        if sparse_indexer_sent_layers is None:
                            sparse_indexer_sent_layers = getattr(
                                self,
                                "_layerwise_sparse_indexer_sent_layers",
                                None,
                            )
                            if sparse_indexer_sent_layers is None:
                                sparse_indexer_sent_layers = set()
                                self._layerwise_sparse_indexer_sent_layers = (
                                    sparse_indexer_sent_layers
                                )
                    if wait_group == 1:
                        ret_token_mask = None
                        if (
                            indexer_retriever is not None
                            and sparse_indexer_sent_layers is not None
                            and (
                                parsed_layer_id is None
                                or parsed_layer_id == self.current_layer
                            )
                            and indexer_sent_key not in sparse_indexer_sent_layers
                        ):
                            if shared_ordered:
                                layerwise_retriever.send(
                                    {_SHARED_SPARSE_PREPARE_ONLY: True}
                                )
                            indexer_retriever.send(
                                {
                                    "selected_token_ids": None,
                                    "token_start_index": 0,
                                    _SHARED_SPARSE_DEFER_COMMIT: (
                                        shared_ordered
                                        and self.current_layer
                                        == self.num_layers - 1
                                    ),
                                }
                                if shared_ordered
                                else (None, 0)
                            )
                            sparse_indexer_sent_layers.add(indexer_sent_key)
                    else:
                        ret_token_mask = layerwise_retriever.send(sparse_payload)
                        if (
                            indexer_retriever is not None
                            and sparse_indexer_sent_layers is not None
                            and indexer_sent_key not in sparse_indexer_sent_layers
                        ):
                            indexer_ret_mask = indexer_retriever.send((None, 0))
                            sparse_indexer_sent_layers.add(indexer_sent_key)
                            if ret_token_mask is None:
                                ret_token_mask = indexer_ret_mask
                        elif (
                            indexer_retriever is not None
                            and shared_ordered
                            and self.current_layer == self.num_layers - 1
                            and indexer_sent_key in sparse_indexer_sent_layers
                        ):
                            indexer_ret_mask = next(indexer_retriever)
                            if ret_token_mask is None:
                                ret_token_mask = indexer_ret_mask
                    decode_row += row_count
                else:
                    if wait_group == 1:
                        ret_token_mask = (
                            next(indexer_retriever)
                            if indexer_retriever is not None
                            else None
                        )
                    else:
                        ret_token_mask = next(layerwise_retriever)

                if (
                    self.current_layer == self.num_layers - 1
                    and not request.is_sparse_decode
                    and (wait_group == 0 or indexer_retriever is not None)
                ):
                    assert ret_token_mask is not None
                    num_retrieved_tokens = ret_token_mask.sum().item()
                    logger.info(
                        "Retrieved %d tokens for KV group %d",
                        num_retrieved_tokens,
                        wait_group,
                    )
                    self._validate_dense_retrieve_result(
                        request,
                        ret_token_mask,
                        kv_group=wait_group,
                    )
                idx += 1

        if self.layerwise_retrievers and self._layerwise_wait_should_advance(
            wait_group
        ):
            self.current_layer += 1
            if self.current_layer >= self.num_layers:
                completed_requests = tuple(layerwise_requests)
                dense_perf_states = [
                    (
                        request,
                        self._cold_perf_dense_load_started.get(
                            request.req_id,
                            None,
                        ),
                    )
                    for request in completed_requests
                ]
                finalize_started = (
                    cold_start_perf_now()
                    if any(
                        request.req_id in self._cold_perf_load_started
                        for request in completed_requests
                    )
                    or any(state is not None for _, state in dense_perf_states)
                    else 0.0
                )
                with self._sparse_retrieve_state_guard(
                    completed_requests
                ):
                    if metadata is None:
                        metadata = self._parent._get_connector_metadata()
                        assert isinstance(metadata, LMCacheConnectorMetadata)
                    self._finalize_worker_retrieve_state_from_metadata(metadata)
                    self._drain_layerwise_retrievers()
                finalize_ms = (
                    (cold_start_perf_now() - finalize_started) * 1000
                    if finalize_started
                    else 0.0
                )
                dense_completed = cold_start_perf_now()
                for request, perf_state in dense_perf_states:
                    if perf_state is None:
                        continue
                    self._cold_perf_dense_load_started.pop(request.req_id, None)
                    request_started, token_count = perf_state
                    self._cold_perf_dense_load_completed[request.req_id] = (
                        dense_completed
                    )
                    cold_start_perf_log(
                        logger,
                        "dense_worker_load_complete",
                        started=request_started,
                        req_id=request.req_id,
                        tokens=token_count,
                        layers=self.num_layers,
                        finalize_ms=round(finalize_ms, 3),
                        scope="layerwise_wall",
                        includes_model_compute=True,
                    )
                for request in completed_requests:
                    perf_state = self._cold_perf_load_started.pop(
                        request.req_id,
                        None,
                    )
                    if perf_state is None:
                        continue
                    request_started, token_count = perf_state
                    cold_start_perf_log(
                        logger,
                        "worker_load_complete",
                        started=request_started,
                        req_id=request.req_id,
                        tokens=token_count,
                        layers=self.num_layers,
                        finalize_ms=round(finalize_ms, 3),
                        scope="layerwise_wall",
                        includes_model_compute=True,
                    )

        return

    def _should_defer_latent_save_under_tp(self) -> bool:
        if not getattr(self.config, "dsa_two_groups", False):
            return False
        meta = getattr(self.lmcache_engine, "metadata", None)
        world_size = getattr(meta, "world_size", 1) if meta else 1
        return world_size > 1

    @staticmethod
    def _store_result_from_yield(
        value: Optional[LayerwiseStoreResult],
    ) -> Optional[LayerwiseStoreResult]:
        if value is not None and not isinstance(value, LayerwiseStoreResult):
            raise TypeError(
                "Layerwise store generator yielded unsupported value "
                f"{type(value)}"
            )
        return value

    def _layerwise_storer_drain_limit(self) -> int:
        engine = getattr(self, "lmcache_engine", None)
        num_layers = int(getattr(engine, "num_layers", 0) or 0)
        if num_layers <= 0:
            num_layers = len(getattr(self, "_latent_layer_names", []) or [])
        if num_layers <= 0:
            num_layers = len(getattr(self, "kv_caches", {}) or {})
        return max(num_layers + 2, 2)

    def _drain_layerwise_storer_fully(
        self,
        storer,
    ) -> tuple[bool, Optional[LayerwiseStoreResult]]:
        if storer is None:
            return True, None
        result: Optional[LayerwiseStoreResult] = None
        for _ in range(self._layerwise_storer_drain_limit()):
            try:
                yielded = self._store_result_from_yield(next(storer))
                if yielded is not None:
                    if result is not None:
                        raise RuntimeError(
                            "Layerwise store generator yielded multiple results"
                        )
                    result = yielded
            except StopIteration:
                return True, result
        logger.warning(
            "Layerwise storer did not finish after bounded drain; closing it"
        )
        return False, result

    @staticmethod
    def _close_layerwise_storer(storer) -> None:
        if storer is None:
            return
        try:
            storer.close()
        except (GeneratorExit, RuntimeError, ValueError):
            pass

    def _finalize_layerwise_storer(
        self,
        storer,
    ) -> tuple[bool, Optional[LayerwiseStoreResult]]:
        try:
            return self._drain_layerwise_storer_fully(storer)
        finally:
            self._close_layerwise_storer(storer)

    def _consume_completed_layerwise_store(
        self,
        request: ReqMeta,
        kv_group: int,
        completed: bool,
        result: Optional[LayerwiseStoreResult],
    ) -> None:
        if not completed:
            return
        if result is not None and result.kv_group != kv_group:
            raise RuntimeError(
                "Layerwise store result group mismatch: "
                f"expected={kv_group}, got={result.kv_group}"
            )
        if result is not None and result.request_id != request.req_id:
            raise RuntimeError(
                "Layerwise store result request mismatch: "
                f"expected={request.req_id}, got={result.request_id}"
            )
        self._promote_layerwise_store_result(request, result)
        self._record_prefill_save_group_completed(request, kv_group, result)
        self._record_decode_window_save_group_completed(
            request,
            kv_group,
            result,
        )

    def _layerwise_store_kwargs(
        self,
        request: ReqMeta,
        kv_group: int,
    ) -> dict[str, Any]:
        """Build engine arguments for one layerwise store generator."""
        decode_window_save = self._is_decode_window_save_request(request)
        store_kwargs: dict[str, Any] = {
            "decode_window_save": decode_window_save,
            "decode_window_start": getattr(request, "decode_window_start", None),
            "decode_window_end": getattr(request, "decode_window_end", None),
            "decode_window_size": getattr(request, "decode_window_size", None),
            "request_configs": request.request_configs,
        }
        if self._is_dsa_two_groups() and kv_group == 1:
            store_kwargs["kv_group"] = 1
        return store_kwargs

    def _prepare_layerwise_store_inputs(
        self,
        request: ReqMeta,
        save_spec: Optional[SaveSpec],
        kv_group: int,
    ) -> Optional[
        tuple[
            list[int],
            torch.Tensor,
            torch.Tensor,
            int,
            dict[str, Any],
            bool,
        ]
    ]:
        token_ids = request.token_ids
        assert isinstance(token_ids, list)
        assert request.slot_mapping is not None and len(request.slot_mapping) > 0

        slot_mapping = request.slot_mapping[0]
        if request.is_sparse_decode:
            if (
                slot_mapping.device.type != torch.device(self.device).type
                or slot_mapping.dtype != torch.long
            ):
                slot_mapping = slot_mapping.to(device=self.device, dtype=torch.long)
                request.slot_mapping[0] = slot_mapping
        else:
            slot_mapping = slot_mapping.to(device=self.device, dtype=torch.long)

        if (
            self.kv_role == "kv_producer"
            and not self._is_decode_window_save_request(request)
        ):
            skip_leading_tokens = 0
        else:
            assert save_spec is not None
            skip_leading_tokens = save_spec.skip_leading_tokens
            if skip_leading_tokens == len(token_ids):
                return None
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

        windowed_slot_mapping = self._windowed_sparse_save_mapping(
            request,
            kv_group=kv_group,
            expected_base=skip_leading_tokens,
        )
        windowed_sparse_save = windowed_slot_mapping is not None
        if windowed_slot_mapping is not None:
            slot_mapping = windowed_slot_mapping.to(
                device=self.device,
                dtype=torch.long,
            )

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False
        store_kwargs = self._layerwise_store_kwargs(request, kv_group)
        if windowed_sparse_save:
            store_kwargs["slot_mapping_base"] = skip_leading_tokens
            store_kwargs["windowed_sparse_save"] = True

        return (
            token_ids,
            slot_mapping,
            store_mask,
            skip_leading_tokens,
            store_kwargs,
            windowed_sparse_save,
        )

    def _flush_deferred_latent_store(
        self,
        request: "ReqMeta",
        save_spec: Optional["SaveSpec"],
    ) -> None:
        """Run a full latent store_layer after indexer layers finish (TP>1)."""
        pending_key = self._layerwise_save_storer_key(request, 0)
        if pending_key not in self._deferred_latent_pending:
            return
        self._note_decode_window_save_seen(request)
        if save_spec is None or not save_spec.can_save_latent:
            self._deferred_latent_pending.discard(pending_key)
            return

        self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_for_group(0)
        if not kvcaches:
            self._deferred_latent_pending.discard(pending_key)
            self._abort_save_step((request,))
            raise RuntimeError(
                "Deferred two-group save has no Group-0 latent KV caches: "
                f"req_id={request.req_id}"
            )

        store_inputs = self._prepare_layerwise_store_inputs(request, save_spec, 0)
        if store_inputs is None:
            self._deferred_latent_pending.discard(pending_key)
            return
        (
            token_ids,
            slot_mapping,
            store_mask,
            skip_leading_tokens,
            store_kwargs,
            _,
        ) = store_inputs
        # All latent layers are complete once the final indexer callback
        # reaches this flush. Ascend can therefore use its existing all-layer
        # transfer without reintroducing latent/indexer stream interleaving.
        store_kwargs["all_layers_ready"] = True

        storer = self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            sync=True,
            req_id=request.req_id,
            **store_kwargs,
        )
        latent_completed, store_result = self._finalize_layerwise_storer(
            storer,
        )
        try:
            self._consume_completed_layerwise_store(
                request,
                0,
                latent_completed,
                store_result,
            )
        finally:
            self._deferred_latent_pending.discard(pending_key)
        indexer_required = (
            self._is_decode_window_save_request(request)
            and getattr(save_spec, "can_save_indexer", False)
            and getattr(self.config, "dsa_two_groups", False)
        )
        if not indexer_required:
            self._mark_decode_window_save_completed(request)

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
            layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        if not self._kvcaches_list:
            self._refresh_kvcaches_list()

        dsa_two_groups = getattr(self.config, "dsa_two_groups", False)
        is_indexer_layer = dsa_two_groups and "indexer" in layer_name
        kv_group = 1 if is_indexer_layer else 0
        # Latent path uses the same kv list as dev-qzy (_kvcaches_list); indexer
        # uses the partitioned indexer caches only.
        kvcaches = self._kvcaches_for_group(kv_group)
        if not kvcaches:
            if dsa_two_groups:
                self._abort_save_step(connector_metadata.requests)
                raise RuntimeError(
                    "Two-group save has no registered KV caches for "
                    f"kv_group={kv_group}, layer={layer_name}. Refusing to "
                    "publish or persist a one-group-only prefix."
                )
            return

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue
            self._note_decode_window_save_seen(request)

            # Per-group gating: in two-group mode, skip indexer save if
            # can_save_indexer is False, and skip latent save if
            # can_save_latent is False.
            if dsa_two_groups and save_spec is not None:
                if is_indexer_layer and not save_spec.can_save_indexer:
                    continue
                if not is_indexer_layer and not save_spec.can_save_latent:
                    continue

            # TP>1 + dsa_two_groups: skip interleaved latent saves during
            # forward; flush the full latent store after the last indexer
            # layer (or in wait_for_save as fallback).
            if self._should_defer_latent_save_under_tp() and not is_indexer_layer:
                if save_spec is not None and save_spec.can_save_latent:
                    self._deferred_latent_pending.add(
                        self._layerwise_save_storer_key(request, 0)
                    )
                continue

            storer_key = self._layerwise_save_storer_key(request, kv_group)
            layerwise_storer = self._layerwise_save_storers.get(storer_key)
            # Forward-boundary recovery: the store_layer generator is sized for
            # exactly one forward (num_layers layer yields + 1 drain yield). It
            # is normally drained and popped by wait_for_save between forwards.
            # Some vLLM-Ascend forward paths do not call wait_for_save between
            # consecutive forwards (e.g. chunked prefill), which would leave the
            # previous forward's storer in place and cause the next forward's
            # save_kv_layer calls to exhaust it (StopIteration). When we see the
            # group's first layer again while a storer still exists, drain the
            # old storer fully and create a fresh one for the new forward.
            _first_layer = (
                self._indexer_layer_names[0]
                if kv_group == 1 and self._indexer_layer_names
                else (
                    self._latent_layer_names[0]
                    if self._latent_layer_names
                    else None
                )
            )
            if _first_layer is not None and layer_name == _first_layer:
                active_keys = {
                    self._layerwise_save_storer_key(req, kv_group)
                    for req in connector_metadata.requests
                }
                stale_keys = [
                    key
                    for key in list(self._layerwise_save_storers)
                    if key[0] == request.req_id
                    and key[2] == kv_group
                    and (key == storer_key or key not in active_keys)
                ]
                for stale_key in stale_keys:
                    stale_storer = self._layerwise_save_storers.pop(stale_key)
                    completed, store_result = self._finalize_layerwise_storer(
                        stale_storer,
                    )
                    if stale_key == storer_key:
                        self._consume_completed_layerwise_store(
                            request,
                            kv_group,
                            completed,
                            store_result,
                        )
                if stale_keys:
                    layerwise_storer = None
            if layerwise_storer is None:
                # Refresh from the live kv_caches dict before creating a new
                # storer. Chunked prefill may update registered buffers between
                # forwards; stale _latent_kvcaches pointers cause MTE OOB.
                self._refresh_kvcaches_list()
                kvcaches = self._kvcaches_for_group(kv_group)
                store_inputs = self._prepare_layerwise_store_inputs(
                    request, save_spec, kv_group
                )
                if store_inputs is None:
                    continue
                (
                    token_ids,
                    slot_mapping,
                    store_mask,
                    skip_leading_tokens,
                    store_kwargs,
                    windowed_sparse_save,
                ) = store_inputs

                if not windowed_sparse_save and is_indexer_layer:
                    # Generic layerwise indexer save still follows the active
                    # layer metadata. Sparse layerwise mode uses the exact
                    # per-request scheduler window above so batched requests
                    # cannot borrow one another's slots.
                    idx_slot = self._indexer_save_slot_mapping(
                        request,
                        attn_metadata,
                        layer_name,
                        len(token_ids),
                    )
                    if idx_slot is None:
                        self._abort_save_step((request,))
                        raise RuntimeError(
                            "DSA two-group save could not resolve the Group-1 "
                            "index slot mapping; the request was aborted to "
                            "prevent a partial Group-0-only cache update: "
                            f"req_id={request.req_id}, layer={layer_name}"
                        )
                    slot_mapping = idx_slot.to(
                        device=self.device, dtype=torch.long
                    )

                if is_indexer_layer and not windowed_sparse_save:
                    slot_mapping = self._pad_chunk_local_slot_mapping(
                        slot_mapping,
                        total_tokens=len(token_ids),
                        token_offset=skip_leading_tokens,
                    )
                    if len(slot_mapping) < len(token_ids):
                        self._abort_save_step((request,))
                        raise RuntimeError(
                            "DSA two-group save has an incomplete Group-1 "
                            "slot mapping; the request was aborted to prevent "
                            "a partial Group-0-only cache update: "
                            f"req_id={request.req_id}, layer={layer_name}, "
                            f"mapping_tokens={len(slot_mapping)}, "
                            f"required_tokens={len(token_ids)}, "
                            f"store_range=[{skip_leading_tokens}, "
                            f"{len(token_ids)})"
                        )

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )
                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                # Match dev-qzy: sync=True when creating the latent storer.
                # Under TP>1 + dsa_two_groups, also sync indexer storers so
                # latent/indexer transfers do not overlap on store_stream.
                _meta = getattr(self.lmcache_engine, "metadata", None)
                _world_size = getattr(_meta, "world_size", 1) if _meta else 1
                sync = layerwise_storer is None and (
                    kv_group == 0
                    or (dsa_two_groups and _world_size > 1)
                )
                logger.debug(
                    "Creating layerwise save storer: req_id=%s key=%s "
                    "layer=%s kv_group=%s decode_window=%s "
                    "range=[%s,%s) tokens=%d skip=%d slot_mapping_len=%d "
                    "kvcaches=%d can_save_latent=%s can_save_indexer=%s",
                    request.req_id,
                    storer_key,
                    layer_name,
                    kv_group,
                    self._is_decode_window_save_request(request),
                    getattr(request, "decode_window_start", None),
                    getattr(request, "decode_window_end", None),
                    len(token_ids),
                    skip_leading_tokens,
                    len(slot_mapping),
                    len(kvcaches),
                    getattr(save_spec, "can_save_latent", None),
                    getattr(save_spec, "can_save_indexer", None),
                )
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=sync,
                    req_id=request.req_id,
                    **store_kwargs,
                )
                self._layerwise_save_storers[storer_key] = layerwise_storer

            indexer_group_last = (
                is_indexer_layer
                and self._indexer_layer_names
                and layer_name == self._indexer_layer_names[-1]
            )

            try:
                next(layerwise_storer)
                if indexer_group_last:
                    indexer_completed, store_result = (
                        self._finalize_layerwise_storer(
                            layerwise_storer,
                        )
                    )
                    self._layerwise_save_storers.pop(storer_key, None)
                    layerwise_storer = None
                    self._consume_completed_layerwise_store(
                        request,
                        kv_group,
                        indexer_completed,
                        store_result,
                    )
                    if self._should_defer_latent_save_under_tp():
                        self._flush_deferred_latent_store(request, save_spec)
            except StopIteration:
                logger.error(
                    "Layerwise save storer exhausted early: req_id=%s key=%s "
                    "layer=%s kv_group=%s decode_window=%s range=[%s,%s) "
                    "active_storer_keys=%s deferred_latent_pending=%s",
                    request.req_id,
                    storer_key,
                    layer_name,
                    kv_group,
                    self._is_decode_window_save_request(request),
                    getattr(request, "decode_window_start", None),
                    getattr(request, "decode_window_end", None),
                    list(self._layerwise_save_storers.keys()),
                    list(getattr(self, "_deferred_latent_pending", set())),
                )
                self._abort_save_step((request,))
                raise
            except BaseException:
                self._abort_save_step((request,))
                raise

    def _effective_skip_leading_tokens(
        self,
        request: ReqMeta,
        save_spec: Any,
    ) -> int:
        skip_leading_tokens = save_spec.skip_leading_tokens
        if self.kv_role == "kv_producer" and request.disagg_spec:
            skip_leading_tokens = min(
                skip_leading_tokens,
                request.disagg_spec.num_transferred_tokens,
            )
        return skip_leading_tokens

    def _prepare_direct_store_inputs(
        self,
        request: ReqMeta,
        slot_mapping: torch.Tensor,
        _save_context: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if request.is_sparse_decode:
            request_slot_mapping = request.slot_mapping[0]
            if (
                request_slot_mapping.device.type
                != torch.device(self.device).type
                or request_slot_mapping.dtype != torch.long
            ):
                request_slot_mapping = request_slot_mapping.to(
                    device=self.device,
                    dtype=torch.long,
                )
                request.slot_mapping[0] = request_slot_mapping
            return request_slot_mapping[: len(slot_mapping)], {}
        return slot_mapping.to(device=self.device, dtype=torch.long), {}

    def _finish_save_batch(self, _save_context: dict[str, Any]) -> None:
        pass

    def _handle_save_request_error(
        self,
        _request: ReqMeta,
        _error: Exception,
    ) -> bool:
        return False

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Block until this step's KV saves reach the connector boundary."""

        save_context: dict[str, Any] = {}
        save_fence_complete = False
        try:
            try:
                self._wait_for_save_impl(save_context)
            finally:
                self._finish_save_batch(save_context)
                save_fence_complete = True
        except BaseException:
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            self._abort_save_step(metadata.requests)
            if save_fence_complete:
                self._complete_worker_save_step()
            raise
        else:
            for request in save_context.get("decode_window_saves", ()):
                self._mark_decode_window_save_completed(request)
            self._complete_worker_save_step()

    def _wait_for_save_impl(self, save_context: dict[str, Any]) -> None:
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self._maybe_lookup_unpin_for_request(request)
            return

        if self.use_layerwise:
            if self._should_defer_latent_save_under_tp():
                for request in connector_metadata.requests:
                    if (
                        self._layerwise_save_storer_key(request, 0)
                        in self._deferred_latent_pending
                    ):
                        self._flush_deferred_latent_store(
                            request,
                            request.save_spec,
                        )
            for request in connector_metadata.requests:
                for kv_group in (0, 1):
                    storer_key = self._layerwise_save_storer_key(
                        request,
                        kv_group,
                    )
                    layerwise_storer = self._layerwise_save_storers.pop(
                        storer_key,
                        None,
                    )
                    if layerwise_storer is not None:
                        save_completed, store_result = (
                            self._finalize_layerwise_storer(
                                layerwise_storer,
                            )
                        )
                        self._consume_completed_layerwise_store(
                            request,
                            kv_group,
                            save_completed,
                            store_result,
                        )
                if self._is_decode_window_save_request(request):
                    save_context.setdefault("decode_window_saves", []).append(
                        request
                    )
                self._mark_prefill_committed(request)
                self._mark_initial_sparse_release_ready(request)
                self._maybe_lookup_unpin_for_request(request)
            return

        assert self.kv_caches
        assert self.lmcache_engine is not None
        kvcaches = list(self.kv_caches.values())

        for request in connector_metadata.requests:
            self._maybe_lookup_unpin_for_request(request)
            self._mark_initial_sparse_release_ready(request)

            try:
                save_spec = request.save_spec
                if (
                    save_spec is None or not save_spec.can_save
                ) and self.kv_role != "kv_producer":
                    continue
                assert save_spec is not None

                token_ids = request.token_ids
                assert request.slot_mapping
                slot_mapping = request.slot_mapping[0]
                if not request.is_sparse_decode:
                    assert len(slot_mapping) == len(token_ids)

                skip_leading_tokens = self._effective_skip_leading_tokens(
                    request,
                    save_spec,
                )
                if skip_leading_tokens == len(token_ids):
                    continue
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False
                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                if request.is_last_prefill:
                    if request.disagg_spec:
                        request.disagg_spec.is_last_prefill = True
                elif not self.enable_blending:
                    aligned_token_len = (
                        len(token_ids)
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

                slot_mapping, store_kwargs = self._prepare_direct_store_inputs(
                    request,
                    slot_mapping,
                    save_context,
                )
                self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    transfer_spec=request.disagg_spec,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    **store_kwargs,
                )
                self._record_decode_window_save_group_completed(request, 0)
                if self._is_decode_window_save_request(request):
                    save_context.setdefault("decode_window_saves", []).append(
                        request
                    )
                self._mark_prefill_committed(request, len(token_ids))

                if get_pp_group().is_last_rank:
                    save_spec.skip_leading_tokens = len(token_ids)
                    if request.disagg_spec:
                        request.disagg_spec.num_transferred_tokens = len(
                            token_ids
                        )
            except Exception as error:
                if self._handle_save_request_error(request, error):
                    continue
                raise

    def _drain_dsa_cold_load_futures(self) -> Optional[set[str]]:
        futures = getattr(self, "_dsa_cold_load_futures", None)
        if not futures:
            return None
        finished: set[str] = set()
        for req_id, entry in list(futures.items()):
            (
                generation,
                future,
                request,
                indexer_block_ids,
                submitted_at,
                indexer_future,
            ) = entry
            if not future.done() or not indexer_future.done():
                continue
            state = None
            aborted_ids = getattr(self, "_dsa_cold_aborted_req_ids", None)
            was_aborted = bool(aborted_ids is not None and req_id in aborted_ids)
            try:
                assert request.load_spec is not None
                state = future.result()
                actual_generation = getattr(
                    request.load_spec, "dsa_cold_load_generation", None
                )
                if generation != actual_generation:
                    raise RuntimeError(
                        "Cold compact completion generation mismatch: "
                        f"req_id={req_id}, expected={generation}, "
                        f"actual={actual_generation}"
                    )
                completed_at = getattr(
                    state, "_dsa_cold_load_completed_at", cold_start_perf_now()
                )
                publish_started = cold_start_perf_now()
                if was_aborted:
                    self._release_unadopted_shared_request_objects(state, request)
                    self._release_shared_worker_retrieve_state(
                        state, self.lmcache_engine
                    )
                    self._release_request_lookup_pins(req_id)
                else:
                    self._publish_worker_retrieve_state(
                        state,
                        request,
                        location=state.location,
                        metadata_warm=True,
                        token_count=request.load_spec.lmcache_cached_tokens,
                    )
                    # Completion reports from different TP workers may be
                    # accumulated across multiple forwards. Preserve this
                    # worker's state while the scheduler is still waiting for
                    # the other workers; explicit finish/abort remains the
                    # authoritative cleanup path.
                    state._dsa_cold_prune_protected = True
                published_at = cold_start_perf_now()
                cold_start_perf_log(
                    logger,
                    "worker_load_complete",
                    started=submitted_at,
                    req_id=req_id,
                    tokens=request.load_spec.lmcache_cached_tokens,
                    mode="dsa_cold_compact",
                    background_ms=round((completed_at - submitted_at) * 1000, 3),
                    scheduler_poll_ms=round(
                        (publish_started - completed_at) * 1000, 3
                    ),
                    publish_ms=round((published_at - publish_started) * 1000, 3),
                    final_unhidden_ms=round(
                        (published_at - completed_at) * 1000, 3
                    ),
                )
                logger.info(
                    "[DSA_COLD_COMPACT] request=%s generation=%d "
                    "status=%s tokens=%d indexer_blocks=%d elapsed_ms=%.3f",
                    req_id,
                    generation,
                    "aborted" if was_aborted else "ready",
                    request.load_spec.lmcache_cached_tokens,
                    len(indexer_block_ids),
                    (cold_start_perf_now() - submitted_at) * 1000,
                )
            except BaseException as exc:
                try:
                    self._synchronize_dsa_cold_dense_load()
                except BaseException:
                    logger.critical(
                        "Cold compact load failed and its dense load stream "
                        "could not be synchronized; retaining request blocks: %s",
                        req_id,
                        exc_info=True,
                    )
                    continue
                failed_state = getattr(exc, "_lmcache_dsa_cold_state", None)
                states = getattr(self, "_worker_retrieve_state", {})
                if (
                    failed_state is None
                    and state is not None
                    and state.req_id is not None
                    and states.get(req_id) is not state
                ):
                    failed_state = state
                if failed_state is not None:
                    self._release_unadopted_shared_request_objects(
                        failed_state, request
                    )
                    self._release_shared_worker_retrieve_state(
                        failed_state, self.lmcache_engine
                    )
                self._invalid_block_ids.update(indexer_block_ids)
                if was_aborted:
                    self._release_request_lookup_pins(req_id)
                logger.exception(
                    "[DSA_COLD_COMPACT] request=%s generation=%d "
                    "status=failed indexer_blocks=%d elapsed_ms=%.3f",
                    req_id,
                    generation,
                    len(indexer_block_ids),
                    (cold_start_perf_now() - submitted_at) * 1000,
                )
            futures.pop(req_id, None)
            if was_aborted:
                assert aborted_ids is not None
                aborted_ids.discard(req_id)
                if not aborted_ids:
                    del self._dsa_cold_aborted_req_ids
            finished.add(req_id)
        if not futures:
            del self._dsa_cold_load_futures
            if hasattr(self, "_dsa_cold_last_latent_future"):
                del self._dsa_cold_last_latent_future
            executor = getattr(self, "_dsa_cold_load_executor", None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=False)
                del self._dsa_cold_load_executor
        return finished or None

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        aborted_cold: set[str] = set()
        live_pending = getattr(self, "_dsa_live_split_pending", None)
        if live_pending and finished_req_ids:
            for req_id in finished_req_ids.intersection(live_pending):
                entry = live_pending[req_id]
                offered = bool(entry.get("offered"))
                self._cancel_live_split_entry(
                    entry,
                    f"Live split request was cancelled: {req_id}",
                    release_context=not offered,
                )
                if not offered:
                    live_pending.pop(req_id, None)
        cold_futures = getattr(self, "_dsa_cold_load_futures", None)
        if cold_futures and finished_req_ids:
            aborted_cold = finished_req_ids.intersection(cold_futures)
            if aborted_cold:
                aborted_ids = getattr(
                    self, "_dsa_cold_aborted_req_ids", None
                )
                if aborted_ids is None:
                    aborted_ids = set()
                    self._dsa_cold_aborted_req_ids = aborted_ids
                aborted_ids.update(aborted_cold)
        releasable_req_ids = set(finished_req_ids)
        releasable_req_ids -= aborted_cold
        if releasable_req_ids and not getattr(self, "_wait_for_save_done", True):
            connector_metadata = self._parent._get_connector_metadata()
            assert isinstance(connector_metadata, LMCacheConnectorMetadata)
            waiting_req_ids = {
                request.req_id
                for request in connector_metadata.requests
                if request.req_id in releasable_req_ids
                and self._request_may_store_in_wait_for_save(request)
            }
            if waiting_req_ids:
                self._finished_req_ids_waiting_for_save.update(waiting_req_ids)
                releasable_req_ids -= waiting_req_ids

        finished_sending = self._finalize_worker_requests_after_store(
            releasable_req_ids
        )
        finished_sending.update(self._late_finished_sending)
        self._late_finished_sending.clear()
        finished_recving = self._drain_dsa_cold_load_futures()
        return finished_sending or None, finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        self._dsa_kv_policy_states.clear()
        live_pending = getattr(self, "_dsa_live_split_pending", None)
        if live_pending:
            for req_id, entry in list(live_pending.items()):
                self._cancel_live_split_entry(
                    entry, f"Live split connector is shutting down: {req_id}"
                )
            live_pending.clear()
        executor = getattr(self, "_dsa_cold_load_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._synchronize_dsa_cold_dense_load()
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    def _supports_dsa_split_layout(self) -> bool:
        vllm_config = getattr(self, "_vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        prefix_caching = bool(
            getattr(cache_config, "enable_prefix_caching", False)
        )
        extra_config = getattr(self.config, "extra_config", None)
        shared_cpu_enabled = bool(
            extra_config.get(
                "enable_shared_cpu_cache",
                self.config.enable_shared_cpu_cache,
            )
            if isinstance(extra_config, dict)
            else self.config.enable_shared_cpu_cache
        )
        return bool(
            not prefix_caching
            and self.config.enable_sparse_attention
            and self.config.dsa_two_groups
            and shared_cpu_enabled
            and self.config.use_layerwise
        )

    def _group1_p2p_preferred(self) -> bool:
        return (
            getattr(self.config, "dsa_group1_load_mode", "p2p_preferred")
            == "p2p_preferred"
        )

    def supports_dsa_live_split(self) -> bool:
        return bool(
            self._group1_p2p_preferred()
            and self.config.get_extra_config_value(
                "mooncake_direct_npu_prefill_store", False
            )
            and self._supports_dsa_split_layout()
        )

    def _supports_dsa_live_latent_common(self) -> bool:
        return bool(
            self.config.get_extra_config_value(
                "enable_dsa_live_latent_split", False
            )
            and self.config.get_extra_config_value(
                "mooncake_reuse_vllm_transfer_engine", False
            )
        )

    def supports_dsa_live_latent_source(self) -> bool:
        return bool(
            self._supports_dsa_live_latent_common()
            and self.supports_dsa_live_split()
        )

    def supports_dsa_live_latent_destination(self) -> bool:
        return bool(
            self._supports_dsa_live_latent_common()
            and self.supports_dsa_cold_compact_load()
        )

    def supports_dsa_live_latent_split(self) -> bool:
        """Whether this deployment opts into rank-0 latent live transfer."""
        # Source capture needs the producer's direct-NPU store layout, whereas
        # a pure decoder only needs the cold-compact destination/import path.
        # Requiring the producer-only flag on kv_consumer silently prevented
        # group-0 negotiation in the normal role-specific deployment.
        if self.kv_role == "kv_consumer":
            role_capable = self.supports_dsa_live_latent_destination()
        elif self.kv_role == "kv_producer":
            role_capable = self.supports_dsa_live_latent_source()
        else:
            # kv_both is used by the disaggregated deployment on each side.
            # Activate when this local process owns either half of the
            # protocol; request metadata still admits group 0 only on an
            # actual cold-compact destination with an exact TP0 source.
            role_capable = (
                self.supports_dsa_live_latent_source()
                or self.supports_dsa_live_latent_destination()
            )
        return bool(self._group1_p2p_preferred() and role_capable)

    def configure_live_latent_source(self, enabled: bool) -> None:
        """Apply the two-sided hybrid-transport capability decision."""
        self._live_latent_split_requested = bool(
            enabled and self.supports_dsa_live_latent_split()
        )

    def supports_dsa_cold_compact_load(self) -> bool:
        return bool(
            self.config.enable_dsa_cold_compact_load
            and self._supports_dsa_split_layout()
        )

    def should_load_kv_async(self, req_id: str) -> bool:
        load_spec = self.load_specs.get(req_id)
        return bool(
            load_spec is not None
            and getattr(load_spec, "dsa_cold_compact_load", False)
        )

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id
        resumed = getattr(request, "status", None) == RequestStatus.PREEMPTED
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        resumed_prompt_lookup = (
            resumed
            and not self.config.save_decode_cache
            and prompt_token_ids is not None
        )
        request_prompt_tokens = len(prompt_token_ids or ())
        lookup_query_tokens = (
            request_prompt_tokens
            if resumed_prompt_lookup
            else request.num_tokens
        )
        lookup_query_tokens = max(
            lookup_query_tokens - getattr(self, "skip_last_n_tokens", 0),
            0,
        )
        lookup_call_started = (
            cold_start_perf_now() if cold_start_perf_enabled() else 0.0
        )
        if lookup_call_started:
            self._cold_perf_lookup_started.setdefault(req_id, lookup_call_started)

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(
                "Looking up cache for the first time for request %s!",
                req_id,
            )
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # Decode output is not persisted when save_decode_cache is false.
            # Including it after preemption changes the prompt-tail chunk key.
            if resumed_prompt_lookup:
                assert prompt_token_ids is not None
                token_ids = prompt_token_ids
            else:
                token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                token_ids = _apply_mm_hashes(
                    request.prompt_token_ids,
                    mm_hashes,
                    mm_positions,
                )

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return None

        lookup_started = self._cold_perf_lookup_started.pop(
            req_id,
            lookup_call_started,
        )
        cold_start_perf_log(
            logger,
            "scheduler_lookup",
            started=lookup_started or None,
            req_id=req_id,
            prompt_tokens=request.num_tokens,
            request_prompt_tokens=request_prompt_tokens,
            lookup_query_tokens=lookup_query_tokens,
            query_scope=(
                "prompt" if resumed_prompt_lookup else "all_tokens"
            ),
            resumed=resumed,
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            lookup_call_ms=round(
                (cold_start_perf_now() - lookup_call_started) * 1000,
                3,
            )
            if lookup_call_started
            else 0.0,
        )

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        full_request_hit = num_external_hit_tokens == request.num_tokens
        full_resumed_prompt_hit = (
            resumed_prompt_lookup
            and lookup_query_tokens == request_prompt_tokens
            and num_external_hit_tokens == lookup_query_tokens
        )
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # A hit covering every logical token still recomputes the final token.
        # A prompt-scoped resume has uncached output tokens after the hit and
        # therefore keeps the complete prompt frontier.
        if full_request_hit:
            need_to_allocate -= 1
        compact_remap_frontier = num_external_hit_tokens - int(
            full_request_hit
        )

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        dsa_prefix_hit = (
            self.enable_sparse_attention
            and self._dsa_scratch_capacity > 0
            and num_external_hit_tokens > 0
        )
        dsa_cold_compact_load = (
            self.supports_dsa_cold_compact_load()
            and num_computed_tokens == 0
            and (full_request_hit or full_resumed_prompt_hit)
            and need_to_allocate > 0
            and cdiv(num_external_hit_tokens, self._block_size)
            == cdiv(need_to_allocate, self._block_size)
            # The compact load materializes the prefix in indexer blocks that
            # only the SFA compact-scratch remap can read. That machinery
            # requires a frontier of zero or >= scratch_capacity; a smaller
            # frontier (e.g. a short prompt) is rejected by the staged-SFA
            # route (frontier_too_short FATAL). Such prompts take the normal
            # dense-prefix load path instead (short-context full-resident
            # policy).
            and compact_remap_frontier >= self._dsa_scratch_capacity
            # Short-context full-resident policy (方案 A): a prompt within the
            # threshold is served from resident main blocks, so compact KV load
            # is pure overhead. Skip it in favor of normal dense-prefix load.
            and num_external_hit_tokens > getattr(
                self, "_dsa_kv_policy_threshold", 0
            )
        )
        below_min_retrieve = (
            not dsa_prefix_hit
            and min_retrieve > 0
            and need_to_allocate < min_retrieve
        )

        if below_min_retrieve:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
            dsa_current_released_frontier=0,
        )
        if dsa_cold_compact_load:
            self.load_specs[req_id].dsa_cold_compact_load = True

        if dsa_prefix_hit:
            remap_frontier = (
                compact_remap_frontier
                if dsa_cold_compact_load
                else num_external_hit_tokens
            )
            release_frontier = (
                remap_frontier
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            if dsa_cold_compact_load:
                self.load_specs[req_id].dsa_committed_end = (
                    num_external_hit_tokens
                )
                self.load_specs[req_id].dsa_remap_frontier = remap_frontier
                self.load_specs[req_id].dsa_release_frontier = release_frontier
            else:
                self.load_specs[req_id].dsa_committed_end = release_frontier
            self.load_specs[req_id].dsa_scratch_capacity = (
                self._dsa_scratch_capacity
            )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        return need_to_allocate

    def _build_dsa_cold_compact_meta(
        self, request: Any, blocks: Any, load_spec: LoadSpec
    ) -> ReqMeta:
        if hasattr(blocks, "get_unhashed_block_ids_all_groups"):
            block_ids = blocks.get_unhashed_block_ids_all_groups()
        elif hasattr(blocks, "get_block_ids"):
            block_ids = blocks.get_block_ids()
        else:
            block_ids = blocks
        _, indexer_block_ids = _split_kv_group_block_ids(block_ids)
        if not indexer_block_ids:
            raise ValueError("Cold compact load requires allocated indexer blocks")
        required_indexer_blocks = cdiv(
            load_spec.lmcache_cached_tokens, self._block_size
        )
        if len(indexer_block_ids) < required_indexer_blocks:
            raise ValueError(
                "Cold compact indexer allocation is smaller than the cache hit: "
                f"req_id={request.request_id}, required_blocks="
                f"{required_indexer_blocks}, allocated_blocks="
                f"{len(indexer_block_ids)}"
            )
        token_ids = list(
            request.all_token_ids[: load_spec.lmcache_cached_tokens]
        )
        if len(token_ids) != load_spec.lmcache_cached_tokens:
            raise ValueError(
                "Cold compact metadata does not cover the complete cache hit: "
                f"req_id={request.request_id}, tokens={len(token_ids)}, "
                f"cached={load_spec.lmcache_cached_tokens}"
            )
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            token_ids = _apply_mm_hashes(
                request.prompt_token_ids[: len(token_ids)],
                mm_hashes,
                mm_positions,
            )
        indexer_slot_mapping = [
            _build_slot_mapping(
                indexer_block_ids,
                self._block_size,
                len(token_ids),
            )
        ]
        validation_blocks = getattr(
            self, "_dsa_cold_indexer_block_ids", None
        )
        if validation_blocks is None:
            validation_blocks = {}
            self._dsa_cold_indexer_block_ids = validation_blocks
        validation_blocks[request.request_id] = set(indexer_block_ids)
        remap_frontier = (
            load_spec.dsa_remap_frontier
            if load_spec.dsa_remap_frontier is not None
            else load_spec.dsa_committed_end
        )
        req_meta = ReqMeta(
            req_id=request.request_id,
            token_ids=token_ids,
            indexer_slot_mapping=indexer_slot_mapping,
            is_last_prefill=True,
            is_sparse_decode=True,
            dsa_nonresident_frontier=int(remap_frontier or 0),
            load_spec=load_spec,
            request_configs=extract_request_configs(request.sampling_params),
        )
        params = getattr(request, "kv_transfer_params", None)
        raw_capabilities = (
            params.get("live_split_capabilities", ())
            if isinstance(params, dict)
            else ()
        )
        capabilities = (
            raw_capabilities
            if isinstance(
                raw_capabilities, (tuple, list, set, frozenset)
            )
            and all(isinstance(item, str) for item in raw_capabilities)
            else ()
        )
        source_dp_rank = _live_split_source_dp_rank(
            params,
            self._vllm_config.parallel_config,
        )
        try:
            source_tp_size = int(
                getattr(
                    self._vllm_config.parallel_config,
                    "tensor_parallel_size",
                    1,
                )
                or 1
            )
        except (TypeError, ValueError):
            source_tp_size = 0
        if (
            self._group1_p2p_preferred()
            and "ascend_live_split_v2" in capabilities
            and source_dp_rank is not None
            and _has_live_group1_source_for_dp(
                params,
                source_dp_rank,
                load_spec.lmcache_cached_tokens,
                source_tp_size,
            )
        ):
            remote_block_ids = params.get("remote_block_ids")
            if remote_block_ids:
                req_meta.live_split_requested = True
                req_meta.live_split_remote_block_ids = list(remote_block_ids)
                req_meta.live_split_compact = (
                    "ascend_live_split_compact_v1" in capabilities
                )
                req_meta.live_split_latent_cpu = bool(
                    req_meta.live_split_compact
                    and "ascend_live_split_latent_cpu_v1" in capabilities
                    and getattr(
                        self, "_live_latent_split_requested", False
                    )
                    and _has_live_latent_source_for_dp(
                        params,
                        source_dp_rank,
                        load_spec.lmcache_cached_tokens,
                    )
                )
        return req_meta

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self, request: "Request", num_external_tokens: int, blocks: Any = None
    ):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        self.lookup_client.clear_lookup_status(request.request_id)

        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            cold_loaded_ids = getattr(self, "_dsa_cold_loaded_req_ids", None)
            if (
                cold_loaded_ids is not None
                and request.request_id in cold_loaded_ids
                and getattr(
                    self.load_specs[request.request_id],
                    "dsa_cold_compact_load",
                    False,
                )
            ):
                # Async resume has no new lookup result, but the completed cold
                # load remains the authoritative sparse source for this step.
                self.load_specs[request.request_id].can_load = True
                return
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        load_spec = self.load_specs[request.request_id]
        if getattr(load_spec, "dsa_cold_compact_load", False):
            if blocks is None:
                raise ValueError("Cold compact load requires KVCacheBlocks metadata")
            generation = getattr(self, "_dsa_cold_load_generation", 0) + 1
            self._dsa_cold_load_generation = generation
            load_spec.dsa_cold_load_generation = generation
            pending = getattr(self, "_pending_dsa_cold_load_metas", None)
            if pending is None:
                pending = {}
                self._pending_dsa_cold_load_metas = pending
            pending[request.request_id] = (
                self._build_dsa_cold_compact_meta(request, blocks, load_spec)
            )

    def _should_decode_window_save(self, tracker: RequestTracker) -> bool:
        window_size = getattr(self, "_decode_window_save_window_size", 0)
        if window_size <= 0:
            return False
        if self.kv_role == "kv_consumer":
            return False
        if tracker.disagg_spec is not None:
            return False
        if tracker.skip_save:
            return False
        if (tracker.request_configs or {}).get("lmcache.skip_save", False):
            return False
        if not tracker.is_decode_phase:
            return False
        return len(tracker.token_ids) > tracker.prompt_len

    def _decode_window_save_skip_reason(self, tracker: RequestTracker) -> str:
        """Describe why the authoritative eligibility check rejected a save."""
        window_size = getattr(self, "_decode_window_save_window_size", 0)
        if window_size <= 0:
            return "window_disabled"
        if self.kv_role == "kv_consumer":
            return "kv_consumer"
        if tracker.disagg_spec is not None:
            return "disaggregated_request"
        if tracker.skip_save:
            return "tracker_skip_save"
        if (tracker.request_configs or {}).get("lmcache.skip_save", False):
            return "request_skip_save"
        if not tracker.is_decode_phase:
            return "not_decode_phase"
        if len(tracker.token_ids) <= tracker.prompt_len:
            return "no_decode_tokens"
        return "unknown"

    def _trace_decode_window_decision(
        self,
        tracker: RequestTracker,
        decision: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        if not _mtp_dw_diag_enabled():
            return
        should_save = self._should_decode_window_save(tracker)
        eligibility_reason = (
            "eligible"
            if should_save
            else self._decode_window_save_skip_reason(tracker)
        )
        tracker_len = len(tracker.token_ids)
        window_size = int(getattr(self, "_decode_window_save_window_size", 0) or 0)
        next_start = tracker.decode_window_save_next_start
        next_end = (
            int(next_start) + window_size
            if next_start is not None and window_size > 0
            else None
        )
        states = getattr(self, "_mtp_dw_window_decision_states", None)
        if states is None:
            states = {}
            self._mtp_dw_window_decision_states = states
        state = states.setdefault(
            tracker.req_id,
            {
                "signature": None,
                "near": set(),
                "reached": set(),
                "forced": set(),
            },
        )
        signature = (should_save, eligibility_reason, next_start)
        if decision is None:
            if state["signature"] is None:
                decision = "initial"
            elif should_save and next_end is not None:
                if tracker_len >= next_end and next_end not in state["reached"]:
                    decision = "boundary_reached"
                    state["reached"].add(next_end)
                elif (
                    tracker_len >= next_end - 4
                    and next_end not in state["near"]
                ):
                    decision = "near_boundary"
                    state["near"].add(next_end)
                elif state["signature"] != signature:
                    decision = "eligible"
            elif state["signature"] != signature:
                decision = "blocked"
        state["signature"] = signature
        if decision is None:
            return
        if reason is None:
            if not should_save:
                reason = eligibility_reason
            elif next_end is not None and tracker_len < next_end:
                reason = "awaiting_frontier"
            else:
                reason = "ready"
        if decision == "blocked":
            forced_key = (decision, reason, next_start)
            if forced_key in state["forced"]:
                return
            state["forced"].add(forced_key)
        _mtp_dw_event(
            "meta",
            req=str(tracker.req_id),
            event="window_decision",
            decision=decision,
            reason=reason,
            skip_reason=(
                reason if not should_save or decision == "blocked" else None
            ),
            frontier=tracker_len,
            tracker_len=tracker_len,
            prompt_len=tracker.prompt_len,
            next_start=(int(next_start) if next_start is not None else None),
            next_end=next_end,
            committed_end=int(tracker.decode_window_save_committed_end),
            num_saved_tokens=int(tracker.num_saved_tokens),
            is_decode_phase=bool(tracker.is_decode_phase),
            should_save=should_save,
        )

    def _init_decode_window_save_start(self, tracker: RequestTracker) -> int:
        if tracker.decode_window_save_next_start is not None:
            return tracker.decode_window_save_next_start
        prompt_start = (
            tracker.prompt_len
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        nonresident_chunk_frontier = (
            tracker.dsa_nonresident_frontier
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        start = max(prompt_start, nonresident_chunk_frontier)
        tracker.decode_window_save_anchor = start
        tracker.decode_window_save_next_start = start
        tracker.decode_window_save_committed_end = max(
            nonresident_chunk_frontier,
            min(tracker.decode_window_save_committed_end, start)
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size,
        )
        return start

    def _log_dsa_kv_policy(
        self,
        req_id: str,
        is_sparse_decode: bool,
        prompt_len: int,
        num_computed_tokens: int,
    ) -> None:
        """Log the connector KV policy once per request transition.

        ``full_resident`` uses full-prefix loading and keeps the main latent
        resident. ``sparse_managed`` enables sparse loading/offload lifecycle
        management. Both policies continue to use sparse attention. Controlled
        by ``LMCACHE_DSA_KV_POLICY_LOG=1``.
        """
        policy = "sparse_managed" if is_sparse_decode else "full_resident"
        prev = self._dsa_kv_policy_states.get(req_id)
        if prev == policy:
            return
        self._dsa_kv_policy_states[req_id] = policy
        logger.info(
            "[DSA_KV_POLICY] req=%s kv_policy=%s%s prompt_len=%d computed=%d "
            "threshold=%d",
            req_id,
            policy,
            f" (switched from {prev})" if prev is not None else " (first)",
            prompt_len,
            num_computed_tokens,
            getattr(self, "_dsa_kv_policy_threshold", 0),
        )

    def _build_request_meta(
        self,
        tracker: RequestTracker,
        load_spec: Optional[LoadSpec],
        *,
        is_sparse_decode: bool = False,
    ) -> Optional[ReqMeta]:
        request = self._unfinished_requests.get(tracker.req_id)
        params = getattr(request, "kv_transfer_params", None)
        live_source_requested = bool(
            params and params.get("do_remote_decode")
        )
        metadata = ReqMeta.from_request_tracker(
            tracker,
            self._block_size,
            self._lmcache_chunk_size,
            load_spec=load_spec,
            discard_partial_chunks=self._discard_partial_chunks,
            save_decode_cache=self.config.save_decode_cache,
            is_sparse_decode=is_sparse_decode,
            save_full_chunk_in_decode=getattr(
                self.config,
                "save_full_chunk_in_decode",
                False,
            ),
            dsa_two_groups=getattr(self.config, "dsa_two_groups", False),
            windowed_sparse_layerwise_save=(
                self._windowed_sparse_layerwise_save_enabled()
            ),
            save_entire_prefix=self.kv_role == "kv_producer",
            live_source_requested=live_source_requested,
        )
        return metadata

    def _add_decode_window_save_metas(
        self,
        meta: LMCacheConnectorMetadata,
        tracker: RequestTracker,
    ) -> None:
        if not self._should_decode_window_save(tracker):
            self._trace_decode_window_decision(tracker)
            return

        window_size = self._decode_window_save_window_size
        next_start = self._init_decode_window_save_start(tracker)
        self._trace_decode_window_decision(tracker)

        if tracker.decode_window_save_inflight_end is not None:
            return

        available_end = next_start + (
            (len(tracker.token_ids) - next_start) // window_size * window_size
        )
        while available_end > next_start:
            window_start = next_start
            # Catch up in one store when decode has crossed multiple complete
            # windows. committed_end advances only after this whole range is
            # reported complete by the worker.
            window_end = available_end
            req_meta = ReqMeta.from_decode_window_save(
                tracker,
                self._block_size,
                window_start,
                window_end,
                window_size,
                windowed_sparse_layerwise_save=(
                    self._windowed_sparse_layerwise_save_enabled()
                ),
            )
            if req_meta is None:
                self._trace_decode_window_decision(
                    tracker, decision="blocked", reason="metadata_unavailable"
                )
                return
            if _mtp_dw_diag_enabled():
                slot_mapping = (
                    req_meta.slot_mapping[0].detach().cpu().reshape(-1)
                    if req_meta.slot_mapping
                    else torch.empty(0, dtype=torch.long)
                )
                _mtp_dw_event(
                    "config",
                    req=tracker.req_id,
                    event="decode_window",
                    frontier=len(tracker.token_ids),
                    window_start=window_start,
                    window_end=window_end,
                    window_size=window_size,
                    chunk_size=self._lmcache_chunk_size,
                )
                _mtp_dw_event(
                    "meta",
                    req=tracker.req_id,
                    frontier=len(tracker.token_ids),
                    prompt_frontier=tracker.prompt_len,
                    committed_end=tracker.decode_window_save_committed_end,
                    window_start=window_start,
                    window_end=window_end,
                    window_size=window_size,
                    slot_count=int(slot_mapping.numel()),
                    slot_min=(
                        int(slot_mapping.min()) if slot_mapping.numel() else None
                    ),
                    slot_max=(
                        int(slot_mapping.max()) if slot_mapping.numel() else None
                    ),
                    slot_sample=slot_mapping[:8].tolist(),
                )
                if (
                    window_end > len(tracker.token_ids)
                    or (window_end - tracker.decode_window_save_anchor) % window_size
                ):
                    _mtp_dw_event(
                        "fail",
                        req=tracker.req_id,
                        frontier=len(tracker.token_ids),
                        window_start=window_start,
                        window_end=window_end,
                        invariant="synthetic_window_frontier",
                        window_size=window_size,
                    )
            if (
                self._is_dsa_two_groups()
                and getattr(req_meta.save_spec, "can_save_indexer", False) is False
            ):
                logger.warning(
                    "Skipping decode-window save for request %s because "
                    "dsa_two_groups requires matching DSA index slots on "
                    "every storage backend: window=[%d,%d)",
                    tracker.req_id,
                    window_start,
                    window_end,
                )
                self._trace_decode_window_decision(
                    tracker,
                    decision="blocked",
                    reason="missing_indexer_slots",
                )
                return

            if _mtp_dw_deep_diag_enabled():
                planned_reqs = getattr(
                    self, "_mtp_dw_deep_window_group_planned_reqs", None
                )
                if planned_reqs is None:
                    planned_reqs = set()
                    self._mtp_dw_deep_window_group_planned_reqs = planned_reqs
                if tracker.req_id not in planned_reqs:
                    if len(planned_reqs) >= 256:
                        planned_reqs.pop()
                    planned_reqs.add(tracker.req_id)
                    save_spec = req_meta.save_spec
                    kv_group0_save = bool(
                        save_spec is not None and save_spec.can_save_latent
                    )
                    kv_group1_save = bool(
                        save_spec is not None and save_spec.can_save_indexer
                    )
                    dsa_two_groups = self._is_dsa_two_groups()
                    _mtp_dw_event(
                        "deep",
                        event="window_group_plan",
                        req=tracker.req_id,
                        worker_rank=None,
                        tp_rank=None,
                        tp_world=None,
                        frontier=len(tracker.token_ids),
                        window_start=window_start,
                        window_end=window_end,
                        kv_group=None,
                        dsa_two_groups=dsa_two_groups,
                        shared_cpu_enabled=bool(
                            self._shared_cpu_config_value(
                                "enable_shared_cpu_cache", False
                            )
                        ),
                        latent_only=(
                            dsa_two_groups and kv_group0_save and not kv_group1_save
                        ),
                        indexer_disabled=dsa_two_groups and not kv_group1_save,
                        kv_group0_save=kv_group0_save,
                        kv_group1_save=kv_group1_save,
                        required_groups=sorted(
                            self._decode_window_save_required_groups(req_meta)
                        ),
                    )

            self._trace_decode_window_decision(
                tracker, decision="emitted", reason="window_ready"
            )
            meta.add_request(req_meta)
            tracker.decode_window_save_next_start = window_end
            tracker.decode_window_save_inflight_end = window_end
            break

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()
        pending_cold = getattr(self, "_pending_dsa_cold_load_metas", None)

        for finished_req_id in scheduler_output.finished_req_ids:
            tracker = self._request_trackers.pop(finished_req_id, None)
            self._dsa_kv_policy_states.pop(finished_req_id, None)
            if tracker is not None:
                self._trace_decode_window_decision(
                    tracker, decision="request_finish", reason="request_finished"
                )
            if _mtp_dw_diag_enabled():
                states = getattr(self, "_mtp_dw_window_decision_states", None)
                if states is not None:
                    states.pop(finished_req_id, None)
            planned_reqs = getattr(
                self, "_mtp_dw_deep_window_group_planned_reqs", None
            )
            if planned_reqs is not None:
                planned_reqs.discard(finished_req_id)
            waits = getattr(self, "_mtp_dw_deep_window_group_wait_seen", None)
            if waits is not None:
                waits_copy = {
                    key for key in waits if key[0] != finished_req_id
                }
                self._mtp_dw_deep_window_group_wait_seen = waits_copy
            self._unfinished_requests.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)
            if pending_cold is not None:
                pending_cold.pop(finished_req_id, None)
            cold_loaded = getattr(self, "_dsa_cold_loaded_req_ids", None)
            if cold_loaded is not None:
                cold_loaded.discard(finished_req_id)
                if not cold_loaded:
                    del self._dsa_cold_loaded_req_ids
            validation_blocks = getattr(
                self, "_dsa_cold_indexer_block_ids", None
            )
            if validation_blocks is not None:
                validation_blocks.pop(finished_req_id, None)
                if not validation_blocks:
                    del self._dsa_cold_indexer_block_ids

        if pending_cold:
            meta.dsa_cold_compact_load_pending = True
            for req_meta in pending_cold.values():
                meta.add_request(req_meta)
            pending_cold.clear()
            del self._pending_dsa_cold_load_metas
        elif pending_cold is not None:
            del self._pending_dsa_cold_load_metas

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            cold_loaded_ids = getattr(self, "_dsa_cold_loaded_req_ids", None)
            cold_compact_resume = bool(
                cold_loaded_ids is not None and request.req_id in cold_loaded_ids
            )
            if cold_compact_resume:
                assert cold_loaded_ids is not None
                cold_loaded_ids.remove(request.req_id)
                if not cold_loaded_ids:
                    del self._dsa_cold_loaded_req_ids
                if load_spec is None:
                    raise RuntimeError("Cold compact resume lost its LoadSpec")
                delattr(load_spec, "dsa_cold_compact_load")
                # The scheduler calls update_state_after_alloc(..., 0) when it
                # promotes a completed asynchronous load. That callback clears
                # can_load for ordinary requests, but a cold resume still needs
                # the prepared per-layer Group-0 transfer on its first forward.
                load_spec.can_load = True
                load_spec.dsa_cold_compact_resume = True
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
                disagg_spec=_disagg_spec_from_request(
                    self._unfinished_requests.get(request.req_id)
                ),
            )
            if load_spec is not None and load_spec.dsa_committed_end is not None:
                release_frontier = load_spec.dsa_release_frontier
                request_tracker.decode_window_save_committed_end = int(
                    release_frontier
                    if release_frontier is not None
                    else load_spec.dsa_committed_end
                )
                request_tracker.dsa_current_released_frontier = int(
                    load_spec.dsa_current_released_frontier
                )
            if cold_compact_resume:
                remap_frontier = int(
                    load_spec.dsa_remap_frontier
                    if load_spec.dsa_remap_frontier is not None
                    else load_spec.dsa_committed_end
                )
                request_tracker.dsa_nonresident_frontier = max(
                    request_tracker.dsa_nonresident_frontier,
                    remap_frontier,
                )
                request_tracker.sparse_remap_frontier = remap_frontier
            self._request_trackers[request.req_id] = request_tracker

            if cold_compact_resume:
                request_tracker.seed_sparse_decode_tokens(
                    list(request.prompt_token_ids),
                    token_count=load_spec.lmcache_cached_tokens,
                )

            req_meta = self._build_request_meta(
                request_tracker,
                load_spec,
                is_sparse_decode=cold_compact_resume,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
            self._add_decode_window_save_metas(meta, request_tracker)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for req in cached_reqs:
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                self._add_decode_window_save_metas(meta, request_tracker)
                req_meta = self._build_request_meta(request_tracker, load_spec)
                if req_meta is not None:
                    req_meta.resumed_from_preemption = req.resumed_from_preemption
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )
                request_tracker.decode_window_save_committed_end = max(
                    min(
                        request_tracker.decode_window_save_committed_end,
                        tokens_to_keep,
                    ),
                    request_tracker.dsa_nonresident_frontier
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size,
                )
                request_tracker.decode_window_save_next_start = None
                request_tracker.decode_window_save_anchor = None
                request_tracker.decode_window_save_inflight_end = None
                request_tracker.decode_window_save_pending_commits.clear()
                request_tracker.sparse_meta_frontier = None
                # Main latent non-residency is not rolled back with logical
                # token progress. This includes cold-compact loads even when
                # scheduler-side release was suppressed at the threshold.
                if request_tracker.dsa_nonresident_frontier > 0:
                    request_tracker.sparse_remap_frontier = (
                        request_tracker.dsa_nonresident_frontier
                    )
                elif hasattr(request_tracker, "sparse_remap_frontier"):
                    delattr(request_tracker, "sparse_remap_frontier")

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )
            self._add_decode_window_save_metas(meta, request_tracker)
            # KV policy (方案 A) follows the CURRENT context length, not the
            # initial prompt_len. A short prompt that grows past the threshold
            # switches from full-resident to sparse-managed loading/release;
            # attention remains sparse under both policies.
            is_sparse_decode = (
                self.enable_sparse_attention
                and (
                    (
                        request.num_computed_tokens
                        >= request_tracker.prompt_len
                        and len(request_tracker.token_ids)
                        > getattr(self, "_dsa_kv_policy_threshold", 0)
                    )
                    or request_tracker.dsa_nonresident_frontier > 0
                )
            )
            if self._dsa_kv_policy_log:
                self._log_dsa_kv_policy(
                    req_id,
                    is_sparse_decode,
                    request_tracker.prompt_len,
                    request.num_computed_tokens,
                )
            if is_sparse_decode:
                # Sparse direct decode should only retrieve the prefix whose
                # boundary is also used by SFA scratch_remap. Keep the final
                # partial prompt chunk in the live vLLM tail by default; the
                # sparse direct path does not carry per-chunk sizes.
                lmcache_cached_for_sparse = (
                    request_tracker.prompt_len
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                if self._decode_window_save_window_size > 0:
                    token_len = len(request_tracker.token_ids)
                    save_frontier = request_tracker.decode_window_save_next_start
                    if save_frontier is None:
                        save_frontier = self._init_decode_window_save_start(
                            request_tracker
                        )
                    save_frontier = min(int(save_frontier), token_len)
                    lmcache_cached_for_sparse = min(
                        request_tracker.decode_window_save_committed_end,
                        save_frontier,
                        token_len,
                    )
                lmcache_cached_for_sparse = max(
                    lmcache_cached_for_sparse,
                    request_tracker.dsa_nonresident_frontier,
                )
                committed_end = (
                    lmcache_cached_for_sparse
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                dsa_release_frontier = (
                    committed_end
                    if committed_end
                    > max(
                        self._dsa_scratch_capacity,
                        getattr(self, "_dsa_kv_policy_threshold", 0),
                    )
                    else 0
                )
                dsa_remap_frontier = getattr(
                    request_tracker, "sparse_remap_frontier", None
                )
                if dsa_remap_frontier is not None:
                    dsa_remap_frontier = max(
                        int(dsa_remap_frontier),
                        int(dsa_release_frontier),
                        request_tracker.dsa_nonresident_frontier,
                    )
                    request_tracker.sparse_remap_frontier = dsa_remap_frontier
                else:
                    dsa_remap_frontier = max(
                        dsa_release_frontier,
                        request_tracker.dsa_nonresident_frontier,
                    )
                committed_end = max(committed_end, int(dsa_remap_frontier))
                cold_compact_live = hasattr(
                    request_tracker, "sparse_remap_frontier"
                )
                if self.kv_role == "kv_consumer" or cold_compact_live:
                    # Include the final partial prompt chunk in worker metadata;
                    # only the release frontier must remain chunk-aligned. A
                    # cold compact kv_both request has already materialized its
                    # complete prompt source, so shrinking this frontier would
                    # discard the prepared worker state and reload both groups.
                    lmcache_cached_for_sparse = max(
                        lmcache_cached_for_sparse,
                        request_tracker.prompt_len,
                        int(dsa_remap_frontier),
                    )
                if len(request.all_token_ids) < lmcache_cached_for_sparse:
                    raise RuntimeError(
                        "Sparse frontier exceeds available request tokens: "
                        f"req_id={req_id} frontier={lmcache_cached_for_sparse} "
                        f"tokens={len(request.all_token_ids)}"
                    )
                if (
                    len(request_tracker.sparse_token_ids)
                    < lmcache_cached_for_sparse
                ):
                    request_tracker.seed_sparse_decode_tokens(
                        list(request.all_token_ids),
                        token_count=lmcache_cached_for_sparse,
                    )
                load_spec = LoadSpec(
                    vllm_cached_tokens=0,
                    lmcache_cached_tokens=lmcache_cached_for_sparse,
                    can_load=lmcache_cached_for_sparse > 0,
                    dsa_committed_end=committed_end,
                    dsa_remap_frontier=dsa_remap_frontier,
                    dsa_scratch_capacity=self._dsa_scratch_capacity,
                    dsa_release_frontier=(
                        dsa_release_frontier
                        if dsa_release_frontier > 0
                        else None
                    ),
                    dsa_current_released_frontier=request_tracker.dsa_current_released_frontier,
                )

            req_meta = self._build_request_meta(
                request_tracker,
                load_spec,
                is_sparse_decode=is_sparse_decode,
            )
            if req_meta is not None:
                req_meta.resumed_from_preemption = preempted
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # This callback runs in the scheduler process. Worker-owned state is
        # released when the same request ID reaches worker-side get_finished().
        self._cold_perf_lookup_started.pop(request.request_id, None)
        self._dsa_kv_policy_states.pop(request.request_id, None)
        self._release_request_lookup_pins(request.request_id)
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._drop_layerwise_save_storers(request.request_id)

        self._drop_worker_retrieve_state(request.request_id)
        transitions = getattr(self, "_mtp_dw_deep_retrieve_transitions", None)
        if transitions is not None:
            transitions.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED and self.async_loading:
            # Cancel any ongoing async lookup and prefetch tasks on workers
            lookup_id = request.request_id
            assert self.lookup_client is not None
            self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        if (
            params is not None
            and params.get("do_remote_decode")
            and self.supports_dsa_live_split()
        ):
            params["request_live_split"] = True

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return bool(getattr(request, "dsa_compact_allocated", False)), return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
