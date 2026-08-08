# SPDX-License-Identifier: Apache-2.0
# Standard
import pytest

# First Party
from lmcache.integration.vllm.async_decode_save import AsyncDecodeSaveState


def test_out_of_order_completion_advances_only_contiguous_prefix() -> None:
    state = AsyncDecodeSaveState(generation=7, initial_end=256)
    first = state.issue(256, 512)
    second = state.issue(512, 768)

    advance = state.complete(
        generation=7,
        job_id=second.job_id,
        start=second.start,
        end=second.end,
    )
    assert advance.committed_jobs == ()
    assert advance.committed_end == 256
    assert state.pending_count == 2

    advance = state.complete(
        generation=7,
        job_id=first.job_id,
        start=first.start,
        end=first.end,
    )
    assert [job.job_id for job in advance.committed_jobs] == [
        first.job_id,
        second.job_id,
    ]
    assert advance.committed_end == 768
    assert state.pending_count == 0


def test_done_job_behind_gap_still_consumes_capacity() -> None:
    state = AsyncDecodeSaveState(generation=1, initial_end=0)
    first = state.issue(0, 256)
    second = state.issue(256, 512)

    state.complete(
        generation=1,
        job_id=second.job_id,
        start=second.start,
        end=second.end,
    )

    assert state.pending_count == 2
    assert state.pending_jobs[1].job_id == second.job_id
    assert state.pending_jobs[0].job_id == first.job_id


def test_duplicate_completion_is_idempotent() -> None:
    state = AsyncDecodeSaveState(generation=2, initial_end=128)
    job = state.issue(128, 256)
    state.complete(
        generation=2,
        job_id=job.job_id,
        start=job.start,
        end=job.end,
    )

    duplicate = state.complete(
        generation=2,
        job_id=job.job_id,
        start=job.start,
        end=job.end,
    )
    assert duplicate.duplicate
    assert duplicate.committed_end == 256
    assert duplicate.committed_jobs == ()


@pytest.mark.parametrize(
    ("generation", "job_id", "start", "end", "match"),
    [
        (4, 1, 0, 256, "stale decode-save generation"),
        (5, 99, 0, 256, "unknown decode-save job_id"),
        (5, 1, 1, 256, "does not match issued job"),
        (5, 1, 0, 255, "does not match issued job"),
    ],
)
def test_invalid_completion_is_rejected(
    generation: int,
    job_id: int,
    start: int,
    end: int,
    match: str,
) -> None:
    state = AsyncDecodeSaveState(generation=5, initial_end=0)
    state.issue(0, 256)

    with pytest.raises(ValueError, match=match):
        state.complete(
            generation=generation,
            job_id=job_id,
            start=start,
            end=end,
        )


def test_final_job_closes_issue_stream() -> None:
    state = AsyncDecodeSaveState(generation=3, initial_end=256)
    final = state.issue(256, 300, is_final=True)

    with pytest.raises(ValueError, match="after the final job"):
        state.issue(300, 512)

    advance = state.complete(
        generation=3,
        job_id=final.job_id,
        start=final.start,
        end=final.end,
        is_final=True,
    )
    assert advance.committed_end == 300
    assert advance.committed_jobs == (final,)
