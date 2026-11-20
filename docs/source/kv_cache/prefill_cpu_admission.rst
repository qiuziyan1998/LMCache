Layerwise prefill CPU admission
==============================

The Ascend layerwise prefill integration automatically budgets LocalCPUBackend
memory before a new request performs a cache lookup. It uses the actual worker
allocator capacity, including first-rank overrides and startup memory-size
adjustments, rather than assuming that ``max_local_cpu_size`` is fully available.
No additional environment variable is required.

Each request reserves the complete prompt's latent KV and indexer bytes, rounded
to LMCache chunks, plus one maximum prefill chunk and two tail pages for temporary
save/replacement storage. Per-layer allocator alignment is included even when
the ordinary allocation fallback is used. A 500k prompt reserves its full size
before its first 16k computation chunk; later chunks do not reserve it again.
NPU bank count does not multiply the CPU history size.

Workers report free plus evictable bytes. Pinned objects, outstanding transfers,
and request-owned objects are unavailable. Admission subtracts all outstanding
request reservations from that report. This is intentionally conservative:
already-materialized active data may be charged both as unavailable memory and
as a reservation. It can delay another request beyond the minimum necessary,
but does not rely on guessing which asynchronous owners have been released.

A request which does not fit stays in the scheduler's waiting queue without
performing a lookup or acquiring lookup pins. Existing admitted requests retain
their reservations across computation chunks and preemption. A reservation is
released only after request termination and worker cleanup acknowledgements
from every TP rank. An individual request larger than the entire pool fails
with a capacity error rather than waiting indefinitely.

``[P_CPU_ADMISSION]`` logs show ``action=admit``, ``action=wait``, or
``action=reject`` and the required, reserved, available and total byte counts.
Temporary refusals are logged once per request rather than on each scheduler
poll. Reports use host metadata; there is no tensor copy, hash, or blocking NPU
synchronization. Allocator scans on otherwise idle polls are limited to four
per second. Cleanup uses the existing nonblocking readiness query.

This controls P-node CPU admission. It does not change NPU bank event ordering,
replace the NPU KV block scheduler, compact fragmented CPU memory, or diagnose
incorrect model output. Decoder-only instances retain their existing path.
