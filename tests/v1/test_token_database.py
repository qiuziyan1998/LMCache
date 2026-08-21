# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.mooncake_layout import (
    mooncake_legacy_key,
    mooncake_page_key,
    mooncake_payload_layout,
    mooncake_valid_tokens,
)
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


def test_chunked_token_database_rejects_mismatched_key_inputs():
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu")
    db = ChunkedTokenDatabase(cfg, dumb_metadata())
    tokens = generate_tokens(16, "cpu")

    with pytest.raises(ValueError, match="mask length"):
        list(db.process_tokens(tokens=tokens, mask=torch.ones(15, dtype=torch.bool)))
    with pytest.raises(ValueError, match="counts must match"):
        list(db.process_tokens(hashes=[1, 2], offsets=[16]))


def test_mooncake_page_keys_include_payload_layout_signature(monkeypatch):
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend="cpu")
    cfg.extra_config = {"mooncake_page_first_multi_buffer": True}
    db = ChunkedTokenDatabase(cfg, dumb_metadata())

    key = next(iter(db.process_tokens(tokens=generate_tokens(256, "cpu"))))[2]

    assert dict(key.tags or ())["payload_v3"] == db.mooncake_payload_layout

    monkeypatch.setenv("LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK", "2048")
    monkeypatch.setenv("VLLM_ASCEND_DSA_SHRINK_LATENT", "2")
    assert (
        ChunkedTokenDatabase(cfg, dumb_metadata()).mooncake_payload_layout
        == db.mooncake_payload_layout
    )

    other = LMCacheEngineConfig.from_legacy(chunk_size=512, backend="cpu")
    other.extra_config = {"mooncake_page_first_multi_buffer": True}
    other_db = ChunkedTokenDatabase(other, dumb_metadata())
    assert other_db.mooncake_payload_layout != db.mooncake_payload_layout

    cfg.extra_config["save_chunk_meta"] = False
    assert (
        ChunkedTokenDatabase(cfg, dumb_metadata()).mooncake_payload_layout
        != db.mooncake_payload_layout
    )


def test_payload_v3_identity_includes_remote_fill_abi() -> None:
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")
    cfg.remote_fill_cache_namespace = "deployment-a"
    cfg.remote_fill_model_artifact_id = "weights-build-a"
    cfg.dsa_two_groups = True
    cfg.extra_config = {
        "mooncake_page_first_multi_buffer": True,
        "mooncake_layer_merged_page_objects": True,
        "mooncake_dsa_raw_token_dims": {0: 576, 1: 128},
        "mooncake_cache_bearing_layers": 79,
        "mooncake_group1_schema_version": "dsa-index-v2",
        "mooncake_mtp_layout_version": "mtp-1",
    }

    metadata = dumb_metadata(kv_shape=(32, 2, 1024, 8, 128))
    signature, descriptor = mooncake_payload_layout(cfg, metadata)

    assert descriptor["version"] == 3
    assert descriptor["deployment_namespace"] == "deployment-a"
    assert descriptor["model_artifact_id"] == "weights-build-a"
    assert descriptor["model_artifact_scope"] == "serving-bundle-v1"
    assert descriptor["page_abi_version"] == "lmcache-layer-page-v3"
    assert descriptor["group0_raw_token_dim"] == 576
    assert descriptor["group1_raw_token_dim"] == 128
    assert descriptor["cache_bearing_layers"] == 79
    assert descriptor["group1_schema_version"] == "dsa-index-v2"
    assert descriptor["mtp_layout_version"] == "mtp-1"

    cfg.remote_fill_model_artifact_id = "weights-build-b"
    other_signature, _ = mooncake_payload_layout(cfg, metadata)
    assert other_signature != signature


def test_payload_v3_automatically_fingerprints_serving_bundle(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "LMCACHE_REMOTE_FILL_H0_QUALIFICATION",
        "mooncake-sync-write-visible-v1",
    )
    first_model = tmp_path / "model-a"
    second_model = tmp_path / "model-b"
    first_model.mkdir()
    second_model.mkdir()
    (first_model / "config.json").write_text('{"model_type":"test"}')
    (second_model / "config.json").write_text('{"model_type":"test"}')
    (first_model / "model.safetensors").write_bytes(b"first-weights")
    (second_model / "model.safetensors").write_bytes(b"second-weights")
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")
    cfg.enable_remote_lmcache_store = True

    first_signature, first_descriptor = mooncake_payload_layout(
        cfg, dumb_metadata_with_model_name(str(first_model))
    )
    repeated_signature, repeated_descriptor = mooncake_payload_layout(
        cfg, dumb_metadata_with_model_name(str(first_model))
    )
    second_signature, second_descriptor = mooncake_payload_layout(
        cfg, dumb_metadata_with_model_name(str(second_model))
    )

    assert first_signature == repeated_signature
    assert first_descriptor["model_artifact_id"] == repeated_descriptor[
        "model_artifact_id"
    ]
    assert first_descriptor["deployment_namespace"].startswith("remote-fill-")
    assert first_descriptor["model_artifact_id"].startswith(
        "auto-sampled-serving-bundle-v1-"
    )
    assert first_signature != second_signature
    assert first_descriptor["model_artifact_id"] != second_descriptor[
        "model_artifact_id"
    ]


def test_payload_v3_disabled_path_does_not_walk_serving_bundle(
    monkeypatch,
) -> None:
    # Local import lets the test replace the startup-only walker itself.
    from lmcache.v1 import mooncake_layout

    monkeypatch.delenv("LMCACHE_REMOTE_FILL_H0_QUALIFICATION", raising=False)
    monkeypatch.setattr(
        mooncake_layout,
        "_derive_remote_fill_artifact_id",
        lambda *_args: pytest.fail("disabled path walked the serving bundle"),
    )
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=1024, backend="cpu")

    _, descriptor = mooncake_payload_layout(cfg, dumb_metadata())

    assert descriptor["deployment_namespace"] == ""
    assert descriptor["model_artifact_id"] == ""


@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
@pytest.mark.parametrize("length", [1, 255, 256, 257, 511, 512, 513])
def test_partial_page_keys_encode_only_tail_length(
    chunk_size: int, length: int
) -> None:
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=chunk_size, backend="cpu")
    cfg.extra_config = {"mooncake_page_first_multi_buffer": True}
    db = ChunkedTokenDatabase(cfg, dumb_metadata())
    results = list(db.process_tokens(tokens=generate_tokens(length, "cpu")))

    for start, end, key in results:
        valid_tokens = end - start
        assert mooncake_valid_tokens(key, chunk_size) == valid_tokens
        assert ("internal.valid_tokens" in dict(key.tags or ())) == (
            valid_tokens < chunk_size
        )
    page_keys = [mooncake_page_key(key, 2) for _, _, key in results]
    assert len(page_keys) == len(set(page_keys))
    if results and results[-1][1] - results[-1][0] < chunk_size:
        tail = results[-1][2]
        assert "internal.valid_tokens" in tail.to_string()
        assert "internal.valid_tokens" not in mooncake_legacy_key(tail)


@pytest.mark.parametrize("kv_group", [0, 1])
def test_chunked_token_database_processes_only_incremental_suffix(kv_group):
    chunk_length = 256
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length,
        backend="cpu",
        save_unfull_chunk=True,
    )
    db = ChunkedTokenDatabase(cfg, dumb_metadata())
    tokens = generate_tokens(6144, "cpu")
    full_results = list(db.process_tokens(tokens=tokens, kv_group=kv_group))
    prefix_chunks = 23
    prefix_token_count = prefix_chunks * chunk_length
    prefix_hash = full_results[prefix_chunks - 1][2].chunk_hash
    original_hash_func = db.hash_func
    hash_calls = 0

    def counting_hash(value):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash_func(value)

    db.hash_func = counting_hash
    incremental_results = list(
        db.process_tokens_from_prefix(
            tokens,
            prefix_token_count=prefix_token_count,
            prefix_hash=prefix_hash,
            kv_group=kv_group,
        )
    )

    assert incremental_results == full_results[prefix_chunks:]
    assert hash_calls == 1


@pytest.mark.parametrize("prefix_token_count", [-1, 15, 65])
def test_chunked_token_database_rejects_invalid_incremental_prefix(
    prefix_token_count,
):
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=16, backend="cpu")
    db = ChunkedTokenDatabase(cfg, dumb_metadata())
    tokens = generate_tokens(64, "cpu")

    with pytest.raises(ValueError, match="Incremental token prefix"):
        list(
            db.process_tokens_from_prefix(
                tokens,
                prefix_token_count=prefix_token_count,
                prefix_hash=0,
            )
        )


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_tuple = tuple(sys_tokens.cpu().tolist())
    sys_hash = hash((None, sys_tuple, None))
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_tuple = tuple(token_chunk.cpu().tolist())
        token_hash = hash((None, token_tuple, None))
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_tuple = tuple(query_tokens.cpu().tolist())
    query_hash = hash((None, query_tuple, None))
    hashes.append(query_hash)
    starts.append(start)
    ends.append(start + query_length)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        # print(st, starts[i])
        # print(ed, ends[i])
