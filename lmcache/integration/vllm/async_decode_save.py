# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from dataclasses import dataclass
from enum import Enum


class DecodeSaveJobState(str, Enum):
    """Lifecycle states visible to the decode-save ordering coordinator."""

    ISSUED = "issued"
    DONE = "done"


@dataclass(slots=True)
class DecodeSaveJob:
    """One immutable decode-cache range plus its mutable completion state."""

    generation: int
    job_id: int
    start: int
    end: int
    is_final: bool = False
    state: DecodeSaveJobState = DecodeSaveJobState.ISSUED


@dataclass(frozen=True, slots=True)
class DecodeSaveAdvance:
    """Result of acknowledging one physical decode-save completion."""

    duplicate: bool
    committed_end: int
    committed_jobs: tuple[DecodeSaveJob, ...]


class AsyncDecodeSaveState:
    """Track issued decode saves and advance only across a contiguous prefix.

    Physical stores may complete out of order. A completion marks its exact job
    as done, but ``committed_end`` advances only while the head of the issue
    queue is done. Jobs remain capacity-accounted until they are committed so a
    stalled head cannot create an unbounded tail of completed jobs.
    """

    def __init__(self, generation: int, initial_end: int) -> None:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if initial_end < 0:
            raise ValueError("initial_end must be non-negative")
        self.generation = generation
        self.committed_end = initial_end
        self.issued_end = initial_end
        self._next_job_id = 1
        self._jobs: deque[DecodeSaveJob] = deque()
        self._jobs_by_id: dict[int, DecodeSaveJob] = {}
        self._committed_job_ids: set[int] = set()

    @property
    def pending_count(self) -> int:
        """Return jobs that have not crossed the ordered commit frontier."""
        return len(self._jobs)

    @property
    def pending_jobs(self) -> tuple[DecodeSaveJob, ...]:
        """Return pending jobs in issue order."""
        return tuple(self._jobs)

    def issue(self, start: int, end: int, *, is_final: bool = False) -> DecodeSaveJob:
        """Append one irrevocable range beginning at the issued frontier."""
        if start != self.issued_end:
            raise ValueError(
                "decode-save jobs must be contiguous: "
                f"start={start}, issued_end={self.issued_end}"
            )
        if end <= start:
            raise ValueError(
                f"decode-save range must be non-empty: start={start}, end={end}"
            )
        if self._jobs and self._jobs[-1].is_final:
            raise ValueError("cannot issue a decode-save job after the final job")

        job = DecodeSaveJob(
            generation=self.generation,
            job_id=self._next_job_id,
            start=start,
            end=end,
            is_final=is_final,
        )
        self._next_job_id += 1
        self._jobs.append(job)
        self._jobs_by_id[job.job_id] = job
        self.issued_end = end
        return job

    def complete(
        self,
        *,
        generation: int,
        job_id: int,
        start: int,
        end: int,
        is_final: bool = False,
    ) -> DecodeSaveAdvance:
        """Acknowledge an exact job and drain all newly contiguous done jobs."""
        if generation != self.generation:
            raise ValueError(
                "stale decode-save generation: "
                f"got={generation}, expected={self.generation}"
            )
        if job_id in self._committed_job_ids:
            return DecodeSaveAdvance(True, self.committed_end, ())

        job = self._jobs_by_id.get(job_id)
        if job is None:
            raise ValueError(f"unknown decode-save job_id={job_id}")
        if (job.start, job.end, job.is_final) != (start, end, is_final):
            raise ValueError(
                "decode-save completion does not match issued job: "
                f"job_id={job_id}, issued=({job.start}, {job.end}, "
                f"final={job.is_final}), completed=({start}, {end}, "
                f"final={is_final})"
            )
        if job.state is DecodeSaveJobState.DONE:
            return DecodeSaveAdvance(True, self.committed_end, ())

        job.state = DecodeSaveJobState.DONE
        committed: list[DecodeSaveJob] = []
        while self._jobs and self._jobs[0].state is DecodeSaveJobState.DONE:
            head = self._jobs.popleft()
            if head.start != self.committed_end:
                raise RuntimeError(
                    "decode-save queue lost contiguity: "
                    f"head_start={head.start}, committed_end={self.committed_end}"
                )
            self._jobs_by_id.pop(head.job_id)
            self._committed_job_ids.add(head.job_id)
            self.committed_end = head.end
            committed.append(head)

        return DecodeSaveAdvance(False, self.committed_end, tuple(committed))

