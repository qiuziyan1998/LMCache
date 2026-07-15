# SPDX-License-Identifier: Apache-2.0
"""Final-hidden handoff request rewriting for the disaggregated proxy."""

# Standard
import base64
import hashlib

# First Party
from examples.disagg_prefill.disagg_proxy_server import (
    is_valid_final_hidden_payload,
    prepare_decode_request,
)


def _payload() -> dict:
    raw = b"\x00\x01" * 4
    return {
        "version": 1,
        "dtype": "bfloat16",
        "shape": [4],
        "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii"),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "prompt_length": 3,
        "prompt_sha256": "a" * 64,
        "model_fingerprint": "b" * 64,
    }


def test_hidden_handoff_preserves_prompt_and_generation_budget() -> None:
    artifact = _payload()
    req_data = {
        "prompt": [1, 2, 3],
        "max_tokens": 1,
        "max_completion_tokens": 1,
        "kv_transfer_params": {"ret_final_hidden": True},
    }
    prefill_output = {
        "kv_transfer_params": {"bootstrap_final_hidden": artifact}
    }

    emit_prefill_token = prepare_decode_request(
        req_data, prefill_output, 16, 16, True
    )

    assert not emit_prefill_token
    assert req_data["prompt"] == [1, 2, 3]
    assert req_data["max_tokens"] == 16
    assert req_data["max_completion_tokens"] == 16
    assert req_data["kv_transfer_params"] == {
        "bootstrap_final_hidden": artifact
    }


def test_invalid_hidden_handoff_falls_back_to_normal_prefill() -> None:
    artifact = _payload()
    artifact["data_sha256"] = "0" * 64
    req_data = {
        "prompt": [1, 2, 3],
        "max_tokens": 1,
        "kv_transfer_params": {"ret_final_hidden": True},
    }

    emit_prefill_token = prepare_decode_request(
        req_data,
        {"kv_transfer_params": {"bootstrap_final_hidden": artifact}},
        16,
        None,
        True,
    )

    assert not emit_prefill_token
    assert req_data["prompt"] == [1, 2, 3]
    assert req_data["max_tokens"] == 16
    assert "kv_transfer_params" not in req_data
    assert not is_valid_final_hidden_payload(artifact)


def test_legacy_first_token_handoff_is_unchanged() -> None:
    req_data = {
        "prompt": [1, 2, 3],
        "max_tokens": 1,
        "kv_transfer_params": {"ret_first_tok": True},
    }

    emit_prefill_token = prepare_decode_request(
        req_data,
        {"kv_transfer_params": {"first_tok": 4}},
        16,
        None,
        False,
    )

    assert emit_prefill_token
    assert req_data["prompt"] == [1, 2, 3, 4]
    assert req_data["max_tokens"] == 15
    assert "kv_transfer_params" not in req_data
