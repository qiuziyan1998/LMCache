# SPDX-License-Identifier: Apache-2.0
"""Audit contracts without torch, NPU, or serving dependencies."""

import ctypes
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit(tmp_path):
    path = Path(__file__).parents[2] / "lmcache/v1/prefill_audit.py"
    spec = importlib.util.spec_from_file_location("isolated_prefill_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ENABLED = True
    module._PATH = tmp_path
    return module


def records(audit):
    return [
        json.loads(line)
        for path in audit._PATH.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_sample_does_not_depend_on_transport_partition(audit):
    memory = (ctypes.c_ubyte * 1024)(*[i % 251 for i in range(1024)])
    pointer = ctypes.addressof(memory)
    assert audit.cpu_payload_samples(
        [[pointer]], [[1024]]
    ) == audit.cpu_payload_samples(
        [[pointer, pointer + 5, pointer + 700]],
        [[5, 695, 324]],
    )


def test_source_mutation_is_reported(audit):
    memory = (ctypes.c_ubyte * 64)(*range(64))
    pointer = ctypes.addressof(memory)
    audit.audit_cpu_put(
        lambda: memory.__setitem__(0, 255), (), ["key"], [[pointer]], [[64]]
    )
    assert records(audit)[-1]["error"] == "source_changed_during_put"


def test_legal_bank_rotation_has_no_error(audit):
    flow = audit.BankAudit(1, "P")
    flow.begin(["req"])
    flow.load(0, 0, 1, 3, 101)
    flow.load(0, 0, 1, 3, 102)  # two requests; final FIFO event covers both
    flow.wait(0, 0, 1, 3, 102)
    flow.use(0, 0, 1, 3, 201)
    flow.save(0, 0, 1, 3, 301)
    flow.publish(0, 3, [301])
    flow.finish("req")
    assert not any(row.get("error") for row in records(audit))


@pytest.mark.parametrize(
    "fault, expected",
    [
        ("wait", "wrong_load_wait"),
        ("use", "bank_used_without_wait_hook"),
        ("save", "save_without_current_bank_handoff"),
        ("publish", "publication_missing_bank_tail"),
    ],
)
def test_protocol_errors_do_not_abort_serving(audit, fault, expected):
    flow = audit.BankAudit(0, "P")
    flow.begin(["req"])
    flow.load(0, 1, 1, 1, 10)
    if fault != "use":
        flow.wait(0, 1, 1, 1, 9 if fault == "wait" else 10)
    if fault != "save":
        flow.use(0, 1, 1, 1, 11)
    flow.save(0, 1, 1, 1, 12)
    flow.publish(0, 1, [] if fault == "publish" else [12])
    assert next(row for row in records(audit) if row.get("error"))["error"] == expected


def test_disabled_payload_never_dereferences_an_address(audit):
    audit._ENABLED = False
    assert audit.audit_payload("get", ["key"], [[1]], [[64]]) is None
    assert not records(audit)


def test_only_known_cpu_storage_can_be_sampled(audit):
    obj = SimpleNamespace(
        is_valid=lambda: True,
        parent_allocator=SimpleNamespace(
            buffer=SimpleNamespace(device=SimpleNamespace(type="npu"))
        ),
    )
    assert not audit.cpu_backed([obj])
    obj.parent_allocator.buffer.device.type = "cpu"
    assert audit.cpu_backed([obj])


def test_stale_generation_is_reported(audit):
    flow = audit.BankAudit(0, "P")
    flow.begin(["req"])
    flow.load(0, 0, 0, 1, 11)
    flow.wait(0, 0, 0, 2, 11)
    assert records(audit)[-1]["error"] == "load_generation_mismatch"


def test_audit_io_error_is_nonfatal(audit, tmp_path):
    path = tmp_path / "file"
    path.write_text("x")
    audit._PATH = path
    audit.audit_record("flow", error="test")
