# Decode Offload Window 与 `committed_end` 需求文档

## 1. 范围

本文定义 decode offload 的 window 划分、异步保存、完成聚合、
`committed_end` 推进、本地 KV block 释放、最终尾部保存和异常处理需求。

该能力是 full-resident decode KV 的旁路持久化与释放机制：保存完成前，decode
attention 仍读取原有 NPU KV；只有 scheduler 收到可发布的 `committed_end` 后，
才允许释放对应的本地 latent KV block。该能力不得改变 Lightning Indexer、Top-K、
attention 输入或 logits 语义。

## 2. 术语

设：

- `B`：vLLM KV block size。
- `C`：LMCache chunk size。
- `W`：decode save window size。
- `prompt_len`：请求原始 prompt token 数。
- `anchor`：decode window 的绝对起点。
- `[start, end)`：左闭右开的绝对 token 区间，`end` 也是 frontier。

### 2.1 四类进度

实现和日志必须区分以下状态，不能都称为“保存完成”：

1. `issued_end`
   - 已经生成 save job 的最大 end。
   - 表示任务已经发出，不表示 NPU copy、host copy 或后端持久化完成。
2. physical completion
   - 某个确定 job 的所有必需 KV group 已完成后端持久化。
   - 使用 `(request_id, generation, job_id, start, end, is_final)` 标识。
3. `ordered_committed_end`
   - 从初始 frontier 开始，所有 job 均已物理完成的最大连续前缀。
   - 后面的 job 即使先完成，也不能跨过前面的空洞推进该值。
4. `published_committed_end`
   - 已发布给 scheduler、允许推进 sparse remap 和释放本地 latent block 的 frontier。
   - 默认等于最新可发布的 `ordered_committed_end`；配置 commit delay 时可以故意落后。

本文未加限定的 `committed_end` 指 scheduler 可消费的
`published_committed_end`。设备任务入队、D2H 完成或 MemoryObj 创建均不能单独推进
`committed_end`。

## 3. 配置约束

### DW-1：尺寸关系

- `C` 必须是 `B` 的整数倍。
- `W == 0` 表示关闭 decode-window save。
- 启用时必须满足 `W >= C` 且 `W % C == 0`。
- 必须启用 `save_unfull_chunk=true`，确保 prompt 的非整 chunk 尾部可以进入首个 window。
- 非 final window 的 start/end 必须同时满足 LMCache chunk 对齐。
- final tail 可以不是 `W` 的整数倍，其 end 也可以不是 chunk 边界。

### DW-2：开关

- `LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE` 设置 `W`。
- `LMCACHE_ASYNC_DECODE_SAVE` 显式控制异步模式。
- 未显式设置异步开关时，仅在 `W > 0` 且 `use_layerwise=true` 时默认启用。
- 异步模式要求 `W > 0` 和 `use_layerwise=true`，否则启动失败。
- `VLLM_ASCEND_LMCACHE_DECODE_WINDOW_SAVE_COMMIT_DELAY_WINDOWS` 设置发布延迟窗口数，默认 `0`。

### DW-3：并发上限

- 每请求 pending job 默认最多 `2` 个。
- 每 worker pending job 默认最多 `32` 个。
- 分别由以下变量配置：
  - `LMCACHE_ASYNC_DECODE_SAVE_MAX_PENDING_PER_REQUEST`
  - `LMCACHE_ASYNC_DECODE_SAVE_MAX_PENDING_PER_WORKER`
- 达到上限时暂停发出新 job，不得丢弃 window，也不得越过未发出的范围推进 frontier。

## 4. Window 划分

### DW-4：Anchor

初始 anchor 必须定义为：

```text
anchor = floor(prompt_len / C) * C
```

这样 prompt 的最后一个非整 chunk 尾部会与最早的 decode token 一起进入首个
decode window。anchor 是请求级固定值，不能随 batch、decode step 或重调度改变。

示例：

```text
prompt_len = 10, C = 4, W = 4
anchor = 8

首个 window: [8, 12)
第二个 window: [12, 16)
```

### DW-5：普通 Window Lattice

- 普通 window frontier 必须位于 `anchor + n * W`。
- 下一个待发范围从 `issued_end` 开始，不能从当前 batch 长度重新计算。
- 只有已存在且安全的 token 达到完整 window 时才能发出普通 job。
- `available_end` 按以下规则计算：

```text
available_frontier = min(tracked_token_end, safe_end)
available_end = issued_end
              + floor((available_frontier - issued_end) / W) * W
```

- `safe_end` 是本次 speculative forward 之前已经确定存在 target KV 的最大位置。
- save job 不得跨过 `safe_end` 保存尚未验证或可能被丢弃的 speculative token。

### DW-6：Catch-up

如果调度器一次更新时已经跨过多个完整 window，可以将连续范围合并为一个 job：

```text
issued_end = 256
available_end = 1024
job = [256, 1024)
```

该 job 在容量核算中算一个 save completion，但只有整个 `[256, 1024)` 的必需数据
均持久化完成后，才允许把 ordered frontier 推进到 `1024`。

### DW-7：Final Tail

- 请求正常结束时，计算 `final_end = min(num_computed_tokens, tracked_token_end)`。
- 如果 `final_end > issued_end`，必须发出 `[issued_end, final_end)` 的 final job。
- final job 可以短于 `W`，也可以不在普通 window lattice 上。
- final job 发出后禁止继续发出任何 job。
- `final_end < issued_end` 表示请求回滚穿过已发出的范围，必须报错，不能静默截断。
- finished request 的 tracker、source block lease 和 worker 状态必须保留到 final job 完成。

## 5. Job 身份与发出要求

### DW-8：Generation 与 Job ID

- 每个新的 request tracker 分配新的 `generation`。
- 同一 generation 内 `job_id` 单调增加。
- job 必须记录不可变的 `start`、`end` 和 `is_final`。
- completion 的 generation/job_id/range/final 标志必须与已发 job 完全一致。
- 旧 generation、未知 job、范围不匹配的 completion 必须拒绝。
- 重复 completion 必须幂等，不重复释放 lease、不重复推进 frontier。

### DW-9：连续发出

- 新 job 必须满足 `start == issued_end`。
- `end` 必须严格大于 `start`。
- job queue 必须保持无空洞、无重叠的发出顺序。
- 发出 job 时同时申请精确 source block lease，lease key 为
  `(source, request_id, generation, job_id)`。
- request 完成或被 scheduler 移除后，只要 job 未物理完成，lease 仍需保持源 block 有效。

## 6. 物理完成要求

### DW-10：完成定义

一个 job 只有满足以下全部条件才算 physical completion：

- NPU store kernel 已完成读取 source KV。
- 所有后端 persistence future 已成功完成。
- latent group 已覆盖 `[start, end)`。
- 启用 DSA two-group 时，indexer group 也已覆盖相同区间。
- TP 配置要求保存多个 rank 时，完成聚合已达到 expected worker count。
- shared CPU 路径已经验证完整 layer/chunk pointer coverage。

仅完成某一 layer、某一 group、某一 rank、D2H copy 或 `batched_put` 提交均不能上报
physical completion。

### DW-11：精确 Lease 释放

- 某个 job 物理完成后，可以立即释放该 job 的精确 connector block lease。
- 精确 lease 的释放不代表 ordered/published frontier 可以跨过前序空洞。
- request 自身持有的普通 KV block 只能按 published `committed_end` 统一释放。

这样可以避免后发 job 已经完成但前序 job 卡住时，无限持有该 job 的额外 lease，同时
仍保证 `committed_end` 不发生跳跃。

## 7. `committed_end` 推进

### DW-12：按序提交

收到 physical completion 后：

1. 将对应 job 标记为 `DONE`。
2. 从 issue queue 的队首开始检查。
3. 只要队首为 `DONE`，就移除队首并将 `ordered_committed_end` 更新为其 end。
4. 遇到第一个未完成 job 立即停止。

必须保持：

```text
initial_end <= ordered_committed_end <= issued_end <= tracked_token_end
```

普通 job 的 `ordered_committed_end` 必须 chunk 对齐；final job 例外。

### DW-13：乱序完成示例

```text
initial_end = 256
A = [256, 512)
B = [512, 768)

B 先完成:
  physical_done(B) = true
  ordered_committed_end = 256

A 后完成:
  queue 连续完成 A、B
  ordered_committed_end = 768
```

禁止在 B 单独完成时发布 `768`，因为 `[256, 512)` 仍不可证明可从 LMCache 读取。

### DW-14：发布延迟

当 commit delay 为 `D` 时，保留最新 `D` 次普通 save completion，不立即发布给
scheduler：

- `D = 0`：ordered frontier 立即发布。
- `D = 1`：完成 512 时暂不发布；完成 768 时发布 512，并保留 768。
- `D = 2`：完成 512、768 时均不发布；完成 1024 时发布 512。
- initial prefill frontier 不受 delay 影响，必须立即发布。
- catch-up job 的 completion 计为一次，不按其中包含的 window 数拆分 delay 次数。
- final job 用于结束清理，不得因 delay 永久阻塞 request 完成。

commit delay 只控制 `published_committed_end` 和本地释放，不得阻塞后续 window 的
发出；物理完成仍应及时清除对应 inflight 状态。

### DW-15：单调性和边界

- published `committed_end` 只能单调增加。
- 不得超过 tracker 当前持有的 token frontier。
- 非 final published frontier 必须满足 `committed_end % C == 0`。
- window 模式下必须满足 `(committed_end - anchor) % W == 0`，initial frontier 除外。
- 延迟到达的 initial prefill completion 可以补充发布，但不能回退已发出的
  `issued_end` 或覆盖 async queue。

## 8. Scheduler 与本地 KV 释放

### DW-16：释放触发条件

- scheduler 只能消费 `completed_decode_window_saves[request_id] = committed_end`
  触发本地释放。
- `num_computed_tokens`、decode step 数或 issued window 均不能证明 LMCache 已持久化。
- request 已不存在时可以忽略 frontier 通知，但仍必须完成精确 lease 清理。

### DW-17：释放范围

- `committed_end` 必须是 vLLM block size 的整数倍，才能执行普通本地 block 释放。
- 只释放 DSA latent manager 中 `[scratch_blocks, committed_end / B)` 的 block。
- `scratch_blocks` 对应的固定 resident scratch 永远保留。
- 已释放位置写回 null block，重复 frontier 不得重复 free。
- Lightning Indexer 的全历史 resident cache 不通过此接口释放。

### DW-18：Sparse Remap

- sparse decode 的 LMCache remap frontier 与本地 release frontier 必须使用同一个
  published `committed_end`。
- 如果 committed frontier 未超过固定 scratch capacity，不得启用外部 latent remap。
- final partial prompt/decode chunk 可以存在于 LMCache，但普通 release frontier 仍需保持
  block/chunk 对齐。

## 9. 异步执行与重试

### DW-19：Worker Queue

- 同一请求的 job 按 issue 顺序进入 worker queue。
- 不同请求可以并发处理，但 completion 必须携带 request/job 身份。
- 后台 worker 必须在正确 NPU device 上创建和使用 stream/event。
- ordering event 必须保证 KV producer 写完成后 store kernel 才开始读取。

### DW-20：Persistence 重试

- 后端 future 失败或超时不得推进任何 committed frontier。
- 默认最多重试 `3` 次，默认退避为 `0.1s、0.5s、2.0s`。
- `LMCACHE_ASYNC_DECODE_SAVE_MAX_RETRIES` 可以覆盖重试次数。
- 重试必须复用同一个 generation/job/range，不得创建重叠的新 job。
- 达到重试上限后设置全局可见错误，后续 save、finish 和 shutdown 必须暴露该错误。

### DW-21：请求结束与 Shutdown

- 正常结束必须补发 final tail，并等待 pending job 完成后再报告 finished sending。
- scheduler 的 finished ID 是边沿事件；如果 final job 因队列满暂时未发出，connector
  必须从自身 tracker 状态继续重试发出。
- shutdown 必须等待 job queue 清空，或明确报告后台失败。
- 不得在 NPU store 或后端 persistence 尚未完成时释放 request tracker、MemoryObj、
  pinned host memory 或 source block。

## 10. Preemption、Rollback 与异常

### DW-22：Rollback

- 没有 pending async state 时，rollback 可以将 tracker frontier 截断到有效 token 数。
- 存在 async state 时，如果 rollback 位置小于 `issued_end`，必须报错。
- 不允许通过回退 `committed_end` 掩盖已经发出的 save job。

### DW-23：Stale Completion

- request ID 复用不能使旧 completion 作用于新 tracker。
- generation 不匹配必须拒绝。
- completion 已聚合但 tracker 已结束时，只做安全清理，不重新创建请求状态。

### DW-24：部分完成

- two-group 模式下，只有 latent 或只有 indexer 完成不能推进 frontier。
- TP 多 rank 模式下，未达到 expected worker count 不能推进 frontier。
- shared CPU pointer 或 layer coverage 不完整时不能标记 group complete。
- 所有部分完成状态必须保留足够身份信息以便重试或诊断。

## 11. 可观测性

日志和 profile 至少应记录：

- request ID、generation、job ID。
- window start/end、window size、anchor、safe end。
- issued end、physical completed end、ordered committed end、published committed end。
- pending job 数、pending delayed commit 数。
- KV group、worker/rank、expected completion count。
- retry attempt、backend future 数和错误原因。
- scheduler 实际释放的 latent block 数和保留的 scratch block 数。

结构化 completion 日志默认关闭，通过
`LMCACHE_ASYNC_DECODE_SAVE_LOG_COMPLETIONS=1` 开启。诊断日志不得通过额外 NPU
synchronize 改变异步关键路径。

## 12. 验收测试

### 12.1 Window 算术

- prompt 恰好 chunk 对齐和不对齐。
- `W == C`、`W > C` 和一次跨越多个 W 的 catch-up。
- final tail 小于 W、非 chunk 对齐和无 final tail。
- speculative safe_end 小于当前 token 数。

### 12.2 Frontier

- 顺序完成立即推进。
- B 先于 A 完成时不跨空洞推进。
- A 完成后一次 drain A+B。
- duplicate completion 幂等。
- stale generation、未知 job、range mismatch 拒绝。
- commit delay 为 0、1、2。
- late initial frontier 不回退 issued state。

### 12.3 生命周期

- request 完成时 final tail 必须落盘。
- completion 前 source block lease 不释放。
- physical completion 后精确 lease 释放。
- published committed_end 后本地 latent blocks 才释放。
- scratch 与 indexer resident cache 不被误释放。
- retry、超时、shutdown、rollback 和 request ID 复用。

### 12.4 多组与多卡

- latent/indexer two-group 同区间完成。
- 单 group 失败不推进 frontier。
- TP rank completion 聚合。
- save-only-first-rank 配置。
- shared CPU 和 direct NPU store 路径。

## 13. 验收标准

- 相同输入下，启用/关闭 decode offload 的 token 输出满足既定精度阈值。
- published `committed_end` 始终是 LMCache 可完整读取的连续前缀。
- 任意乱序、重试、重复消息或延迟发布都不会造成 frontier 跳跃或回退。
- 本地 latent block 只在 scheduler 消费 published frontier 后释放。
- 请求结束、异常和 shutdown 后不存在 block lease、MemoryObj 或 pinned memory 泄漏。
- async save 不阻塞 decode 主计算 stream，后台失败可以被稳定上报。

