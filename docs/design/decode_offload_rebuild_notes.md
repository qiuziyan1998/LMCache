# Decode Offload 设计与重建说明

完整的 window、异步保存和 `committed_end` 状态机需求见
[`decode_offload_window_requirements.md`](../requirements/decode_offload_window_requirements.md)。

## Non-Negotiable Constraints

- The reference patches are known to contain bugs.
- Rebuild the feature in small, separable steps so the bug can be isolated.
- Use the patches only as behavioral reference.
- Do not copy implementation lines from the patches; any copied line may be the bug.
- Ignore the old debug/logging additions unless new debugging is explicitly needed.

## Current First Step

- Add a decode backup window.
- The original decode path is untouched: full decode KV stays resident on NPU.
- Top-k KV selection still reads from the original full NPU KV; it must not read
  from the LMCache backup in this step.
- The backup window size must be an integer multiple of the latent block size.
- The backup window is separate continuous NPU memory. It is not built from
  shrink-latent block tables and does not affect the normal decode path.
- At prefill completion, copy the prompt tail that does not make a full window
  into the start of the current backup window. Do the same if that prompt tail
  arrived through an LMCache prefill hit.
- Save only on absolute window boundaries. Example: prompt_len=10 and
  window_size=4 saves 8..11 when token_count reaches 12, then 12..15 when
  token_count reaches 16.
- Keep full-prefix token_ids for LMCache chunk hashing, but use window-local
  slot mapping and window kvcaches only for `ReqMeta.is_decode_backup`.
- The LMCache copy is a side-channel backup only.
- This step only adds a save action and must not alter allocation, remapping,
  retrieval, or attention behavior.
