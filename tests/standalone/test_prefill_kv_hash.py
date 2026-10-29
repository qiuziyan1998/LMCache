# SPDX-License-Identifier: Apache-2.0
"""Exercise real background hashes, fences, ownership and output bounds."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess
import sys
import threading

# Third Party
import pytest
import torch


@pytest.fixture
def hashing():
    path = Path(__file__).resolve().parents[2] / "lmcache/v1/prefill_kv_hash.py"
    spec = spec_from_file_location("kv_hash_test_runtime", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Object:
    def __init__(self, data):
        self.raw_tensor = torch.tensor(data, dtype=torch.uint8)
        self.refs = 1

    @property
    def data_ptr(self):
        return self.raw_tensor.data_ptr()

    def get_size(self):
        return self.raw_tensor.numel()

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1
        assert self.refs >= 1


def digest_rows(lines):
    result = {}
    for _, line in lines:
        stage = line.split()[0]
        for start, end, digest in re.findall(r"(\d+)-(\d+)=([0-9a-f]{16})", line):
            result[(stage, int(start), int(end))] = digest
    return result


def test_store_get_and_decode_hashes_match_despite_batch_boundaries(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    rows = [[Object([layer, i, 42, 0]) for i in range(8)] for layer in range(4)]
    recorder.submit("S", "warmup", 0, [0, 4], [4, 8], [r[:2] for r in rows], 4)
    recorder.submit(
        "S",
        "warmup",
        0,
        [8, 12, 16, 20, 24, 28],
        [12, 16, 20, 24, 28, 32],
        [r[2:] for r in rows],
        4,
    )
    for stage in ("P", "D"):
        recorder.submit(
            stage, "followup", 0, list(range(0, 32, 4)), list(range(4, 33, 4)), rows, 4
        )
        # Repeated prefix visits are suppressed rather than rehashed.
        recorder.submit(
            stage, "followup", 0, list(range(0, 32, 4)), list(range(4, 33, 4)), rows, 4
        )
    assert recorder.drain(5)
    digests = digest_rows(lines)
    assert digests[("S", 0, 32)] == digests[("P", 0, 32)] == digests[("D", 0, 32)]
    assert all(obj.refs == 1 for row in rows for obj in row)


def test_background_waits_for_actual_d2h_before_hashing(hashing):
    lines, waiting, release = [], threading.Event(), threading.Event()
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    source = Object([0, 0, 0, 0])
    host_thread, wait_threads = threading.get_ident(), []

    class Producer:
        def synchronize(self):
            wait_threads.append(threading.get_ident())
            waiting.set()
            assert release.wait(5)
            source.raw_tensor.fill_(37)

    recorder.submit(
        "S", "req", 0, [0], [4], [[source]], 4, producer_events=(Producer(),)
    )
    try:
        assert waiting.wait(5)
        assert source.refs == 2
        assert not digest_rows(lines)
    finally:
        release.set()
    recorder.submit("P", "req", 0, [0], [4], [[source]], 4)
    assert recorder.drain(5)
    assert digest_rows(lines)[("S", 0, 4)] == digest_rows(lines)[("P", 0, 4)]
    assert wait_threads == [recorder.worker.ident] and wait_threads[0] != host_thread
    assert source.refs == 1


def test_bit_corruption_is_visible_in_selected_layer(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    recorder.submit("S", "req", 1, [0], [4], [[Object([1, 2, 3, 4])]], 4)
    recorder.submit("D", "req", 1, [0], [4], [[Object([1, 2, 3, 5])]], 4)
    assert recorder.drain(5)
    assert digest_rows(lines)[("S", 0, 4)] != digest_rows(lines)[("D", 0, 4)]


def test_missing_or_failed_fence_is_incomplete_not_success(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))

    class Failed:
        def synchronize(self):
            raise RuntimeError("device failure")

    source = Object([1, 2, 3, 4])
    recorder.submit("S", "req", 0, [0], [4], [[source]], 4, producer_events=(Failed(),))
    assert recorder.drain(5)
    assert source.refs == 1
    assert any("ok=0" in text and "RuntimeError" in text for _, text in lines)
    assert not digest_rows(lines)


def test_output_keeps_first_and_last_four_intervals_and_fits_copy_budget(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    starts = list(range(0, 12 * hashing.SEGMENT_TOKENS, hashing.SEGMENT_TOKENS))
    ends = [start + hashing.SEGMENT_TOKENS for start in starts]
    # Tiny payloads are enough to verify interval framing and output policy.
    for stage in ("S", "P", "D"):
        for group in (0, 1):
            rows = [[Object([i, layer]) for i in range(12)] for layer in (0, 1)]
            recorder.submit(
                stage, stage, group, starts, ends, rows, hashing.SEGMENT_TOKENS
            )
    assert recorder.drain(5)
    for stage in ("S", "P", "D"):
        selected = {
            start // hashing.SEGMENT_TOKENS
            for (s, start, _) in digest_rows(lines)
            if s == stage
        }
        assert selected == {0, 1, 2, 3, 8, 9, 10, 11}
    output = "\n".join(f"[KVH] p123456 q{tag} {text}" for tag, text in lines)
    assert len(output.encode()) < 4096
    assert max(map(len, output.splitlines())) < 110


def test_full_chunks_only_and_actual_tail_range(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    recorder.submit("S", "req", 0, [0, 4], [4, 6], [[Object([1]), Object([2])]], 4)
    assert recorder.drain(5)
    assert set(digest_rows(lines)) == {("S", 0, 4)}


def test_queue_full_never_blocks_or_acquires_extra_source_references(hashing):
    lines, waiting, release = [], threading.Event(), threading.Event()
    recorder = hashing.KVHashRecorder(
        max_jobs=1, emit=lambda tag, text: lines.append((tag, text))
    )

    class Producer:
        def synchronize(self):
            waiting.set()
            assert release.wait(5)

    recorder.submit(
        "S", "req", 0, [0], [4], [[Object([1])]], 4, producer_events=(Producer(),)
    )
    try:
        assert waiting.wait(5)
        rejected = Object([2])
        assert recorder.submit("D", "direct", 1, [0], [4], [[rejected]], 4) is None
        assert rejected.refs == 1
    finally:
        release.set()
    assert recorder.drain(5)
    assert any("queue_full" in text and "ok=0" in text for _, text in lines)


def test_store_hash_fence_releases_writer_after_hash_not_before(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    source = Object([37] * 4)
    fence = recorder.submit("S", "req", 0, [0], [4], [[source]], 4)

    def writer():
        fence.synchronize()
        source.raw_tensor.fill_(99)

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    recorder.submit("P", "req", 0, [0], [4], [[Object([37] * 4)]], 4)
    recorder.submit("D", "req", 0, [0], [4], [[source]], 4)
    assert recorder.drain(5)
    values = digest_rows(lines)
    assert values[("S", 0, 4)] == values[("P", 0, 4)]
    assert values[("S", 0, 4)] != values[("D", 0, 4)]


def test_extractor_keeps_three_checkpoints_without_worker_prefix(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(
        "Worker_PID [KVH] p1 q1 req=warmup-uuid\n"
        "Worker_PID [KVH] p1 q1 S g0 0-4=0123456789abcdef\n"
        "Worker_PID [KVH] p1 q1 P g0 warmup_self_load\n"
        "Worker_PID [KVH] p1 q2 req=followup-uuid\n"
        "Worker_PID [KVH] p1 q2 P g0 0-4=0123456789abcdef\n"
        "Worker_PID [KVH] p1 q2 D g0 0-4=0123456789abcdef\n"
        "Worker_PID [KVH] p1 q3 req=other-uuid\n"
        "Worker_PID [KVH] p1 q3 S g0 irrelevant\n"
    )
    script = Path(__file__).resolve().parents[2] / "tools/extract_prefill_kv_hash.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--save-request-id",
            "warmup-uuid",
            "--request-id",
            "followup-uuid",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("0-4=0123456789abcdef") == 3
    assert "Worker_PID" not in result.stdout
    assert "warmup_self_load" not in result.stdout
    assert "other-uuid" not in result.stdout


def test_disabled_switch_does_not_start_worker_or_touch_sources(hashing, monkeypatch):
    monkeypatch.setattr(hashing, "_ENABLED", False)
    monkeypatch.setattr(hashing, "_recorder", lambda: pytest.fail("disabled worker"))
    assert hashing.submit_kv_hash("S", "req", 0, [], [], [object()], 4) is None
    hashing.finish_kv_hash("req")


def test_feature_mode_never_starts_legacy_hash_even_with_debug_enabled(hashing, monkeypatch):
    monkeypatch.setattr(hashing, "_ENABLED", True)
    monkeypatch.setattr(hashing, "_recorder", lambda: pytest.fail("legacy hash must not run"))
    assert hashing.submit_kv_hash("S", "req", 0, [], [], [object()], 4, feature_mode=True) is None


@pytest.mark.parametrize("value, enabled", [(None, False), ("0", False), ("1", True)])
def test_store_debug_switch_controls_metadata_and_hashes(
    value: str | None, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv("LMCACHE_PREFILL_STORE_DEBUG", raising=False)
    else:
        monkeypatch.setenv("LMCACHE_PREFILL_STORE_DEBUG", value)
    root = Path(__file__).resolve().parents[2] / "lmcache/v1"
    for name, gate in (
        ("prefill_store_debug", "prefill_store_debug_enabled"),
        ("prefill_kv_hash", "kv_hash_enabled"),
    ):
        spec = spec_from_file_location(f"test_shared_debug_{name}", root / f"{name}.py")
        module = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        assert getattr(module, gate)() is enabled


def test_decode_hash_does_not_expand_into_generated_history(hashing):
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    rows = [[Object([1]), Object([2])]]
    recorder.submit("D", "req", 0, [0], [4], [rows[0][:1]], 4)
    recorder.submit("D", "req", 0, [0, 4], [4, 8], rows, 4)
    assert recorder.drain(5)
    assert set(digest_rows(lines)) == {("D", 0, 4)}


def test_merged_page_and_flat_objects_match_without_allocator_padding(hashing):
    class Page(Object):
        def layer_data_ptr(self, layer):
            return self.data_ptr + layer * 4

        def layer_size_bytes(self, layer):
            return 4

    pages = [
        Page(
            [value for layer in range(4) for value in (layer, chunk, 42, 0)]
            + [255] * 32
        )
        for chunk in range(2)
    ]
    flat = [[Object([layer, chunk, 42, 0]) for chunk in range(2)] for layer in range(4)]
    lines = []
    recorder = hashing.KVHashRecorder(emit=lambda tag, text: lines.append((tag, text)))
    recorder.submit("S", "source", 0, [0, 4], [4, 8], [pages] * 4, 4)
    recorder.submit("P", "target", 0, [0, 4], [4, 8], flat, 4)
    assert recorder.drain(5)
    values = digest_rows(lines)
    assert values[("S", 0, 8)] == values[("P", 0, 8)]
    assert all(page.refs == 1 for page in pages)
