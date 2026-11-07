# Continuous batching for the CPU engine

Status: design only, no code. Everything below is checked against the tree as it stands today;
line references are to files, not to a commit, because the file is the thing that changes.

## What this is for

Latency per request is already at the memory wall, so the only large win left on one machine is to
serve several requests per weight read. Continuous batching means: instead of running one sequence
to completion, form a batch of up to `B` sequences at each decode step, read each weight tensor once,
and multiply it against `B` activations. Requests join and leave the batch at step boundaries, so a
short request does not wait for a long one.

Throughput, not latency, is the metric. A single request should not get slower when the machine is
idle; that is a constraint, not a goal.

## Where the time goes today

- One decoded token reads ~1.1 GB of weights in ~63-65 ms: about 17 GB/s effective, inside the
  15-20 GB/s the `matvec` kernels reach in isolation (`quant::tests::pooled_matvec_throughput_by_tensor_type`).
  The pass is bandwidth-bound, not arithmetic-bound.
- A token makes 197 `matvec` calls: 7 per layer over 28 layers plus the output projection.
- The engine already holds several sequences (`Engine::sequences: BTreeMap<u64, Sequence>`,
  `crates/vipera-runtime/src/engine.rs`), but each `generate` call drives its own forward pass, and
  the server serializes all requests behind one `Mutex<Engine<CpuBackend>>`
  (`crates/vipera-server/src/serve.rs`).
- `ForwardPass::logits` takes one `ForwardContext` and returns the logits of the last token of that
  one sequence (`crates/vipera-runtime/src/pipeline.rs`).

So the weight read is repeated per sequence, and that repetition is the thing worth removing.

## The one idea, stated once

For a batch of `B` decode steps:

```
bytes moved per step   = weights once          (independent of B)
dots per step          = 197 * B
tokens produced        = B
```

Time per step is `max(weights / bandwidth, B * dot_cost)`. Aggregate tokens per second therefore
rises with `B` until the dots stop hiding under the memory read. Where that cross-over sits is not
known for this machine and must be measured (see "What to measure"), not assumed; the design has to
stay useful at `B = 2` as well as `B = 8`.

Note what does *not* change: attention reads `B` separate KV histories, the output projection still
touches the vocabulary per sequence, and sampling still runs once per sequence per step. Those grow
with `B`; the weights do not. All of them are small against 1.1 GB, but they set the ceiling on the
win, so they belong in the measurement.

## Constraints this design has to respect

- No external crates: the batch is a slice, the queue is `std::sync::mpsc`, the pool is the
  existing scoped one.
- `unsafe_code = "forbid"` outside `vipera-backend-cpu/src/simd.rs`; nothing here needs `unsafe`.
- No behaviour change for one request: `--max-tokens 24 --temperature 0` must keep producing the
  same ids it produces today.
- Every public item documented, every error path a `ViperaError`, no `unwrap`/`panic` outside tests.

## What changes, in dependency order

### 1. Batched kernel (`vipera-backend-cpu/src/simd.rs`, `quant.rs`)

The current kernel dequantizes one row and dots it against one quantized activation. The batched
form loads each block once and dots it against `B` activations while the block is in registers:

```text
q4k_dot_batch(row_block, xs: [&QuantizedX; B], out: &mut [f32; B])
```

`B` as a const generic (`const B: usize`) keeps the activations in registers and avoids a dynamic
loop bound; the pool's jobs then carry a fixed-size array of owned activations instead of one.

Where the row read is amortized: everything except the final maddubs. The dequantize step
(mask, shift, multiply by scale and minimum) happens once per block, and only the dot repeats.

### 2. Pool (`crates/vipera-backend-cpu/src/pool.rs`)

`Job` carries `Arc<[OwnedActivation; B]>` instead of `Arc<OwnedActivation>`, and the reply is
`Vec<f32>` of length `B * rows_in_chunk`. Layout matters: row-major over `(token, row)` so the copy
back into `B` output slices stays a few big memcpys.

### 3. Backend (`crates/vipera-backend-cpu/src/model.rs`)

`run` becomes a batch loop. Per layer:

- `rms_norm` per entry (cheap, no change).
- The seven matvecs become one batched matmul each.
- `rope` and the KV write per entry, each with its own block table.
- Attention per entry: unchanged arithmetic, `B` calls, each over its own `k_all`/`v_all`.
- The residual adds stay per entry.

Scratch buffers (`x`, `h`, `q`, `k`, `v`, `attn`, `gate`, `up`, `scores`) become `B` rows each, or a
single `B × width` buffer reused by every entry. Prefer one buffer of `B × width`: it keeps the pool's
inputs contiguous and the memory bounded by `B`, not by `B²`.

### 4. Runtime (`crates/vipera-runtime/src/engine.rs`)

A `Scheduler` that owns the step:

```text
loop {
    admit up to (max_batch - running) sequences from the queue, prefilling at most N tokens each
    if running is empty: break
    run one batched forward step over running
    for each entry: sample one token, append, check max_new_tokens / stop ids / cancel
    release the KV of finished entries, promote prefilled entries to decode
}
```

`ForwardPass` gains one method with a default implementation, so existing implementors keep
compiling:

```text
fn logits_batch(&mut self, batch: &mut [ForwardContext]) -> Result<Vec<Vec<f32>>, ViperaError> {
    // default: one call per entry, which is exactly today's behaviour
}
```

The CPU backend overrides it with the batched path. `logits` stays as the `B = 1` case.

Prefill: the honest first version prefills one sequence at a time (chunked later), because prefill is
compute-heavy and mixing it with decode complicates the kernels for little gain on a bandwidth-bound
machine.

### 5. Sampling

One draw per sequence per step, from that sequence's own `Sampler` (seeded per sequence, so a
sequence's tokens do not depend on what else is in the batch). Measured cost today is ~0.3 ms per
token for greedy, so `B = 4` adds ~1 ms to a ~65 ms step: acceptable, and it stays inside the step
rather than on another thread, which keeps sequences' RNG streams ordered.

### 6. Server (`crates/vipera-server/src/serve.rs`)

Replace `Mutex<Engine>` with:

- a bounded `mpsc` request queue;
- one batching worker thread that owns the `Engine` and calls `step`;
- one `mpsc::Sender<TokenEvent>` per request, consumed by the connection thread that is already
  spawned per connection (`serve.rs` spawns a thread per accepted connection today).

Backpressure is the bounded queue: when it is full, the connection thread blocks before reading the
next request rather than growing memory. `cancel` sets a flag in a shared `BTreeSet<u64>` that the
worker reads at a step boundary; the engine's `cancel` already exists and only needs to be reachable
without the `Mutex`.

## KV cache and memory

`KvCache` already allocates per sequence (`reserve(sequence, tokens)`, `table(sequence)`), and the
backend already resolves a position through `block_table[p / tokens_per_block]`. Batching needs no
new allocation model, only one `storage_mut()` borrow covering all `B` block tables.

Cost of holding more sequences is one partially filled block per sequence. At `--ctx 2048` and `f16`
the whole cache is 58.7 MB for 2048 tokens, so one 16-token block is 58.7 MB × 16 / 2048 ≈ 0.46 MB;
eight sequences waste at most 8 × 0.46 MB ≈ 3.7 MB in total. That is the price of admission, not a
problem.

Batch scratch memory is the real growth: `B × hidden` and `B × ffn` floats, i.e. a few hundred KB at
`B = 8`. Negligible against the cache.

## What to measure, and what would falsify the design

1. Kernel: aggregate GB/s of `q4k_dot_batch` at `B = 1, 2, 4, 8` on the four model shapes. `B = 1`
   must match today's numbers; the ratio of aggregate tokens/s to `B = 1` is the whole hypothesis.
   If `B = 4` gives less than ~2× at the kernel level, stop and keep the serial path.
2. End to end: 1, 2, 4, 8 concurrent requests of the same prompt, report aggregate tok/s and
   per-request latency. Per-request latency with one request in flight must not regress.
3. Bytes per generated token (count matvec calls × tensor bytes) must fall roughly as `1/B`.
4. Attention and sampling share of the step at `B = 8`; if they exceed ~20%, the cross-over is
   already visible and a larger batch is not worth the complexity.

Nothing in this document is a prediction of a number. The only measured facts it leans on are the
~17 GB/s effective, the 15-20 GB/s kernel bandwidth, and the ~65 ms per token, all from the pooled
`matvec` measurement.

## Correctness plan

- Bit-exactness per sequence: each row's dot for a given sequence is the same arithmetic in the same
  order whether or not other sequences share the read, so a batched step must reproduce the
  unbatched logits exactly. `logits_batch` with `B = 1` must equal `logits`, and every entry of a
  `B = 4` batch must equal its own `B = 1` run. That is a cheap, strong test.
- Golden tokens: greedy 24 tokens `[12095, 13, 576, 6722, 315, 9625, 374, ...]` and temperature 0.8
  / top-k 40 / seed 7, run at concurrency 1 and at concurrency 4, must both match.
- Concurrency test without a model: three sequences on a synthetic `ForwardPass` that records the
  order of `logits_batch` calls, asserting one step per sequence per tick and no interleaving of a
  sequence's positions.
- Cancellation and release: a cancelled sequence's blocks return to `free_blocks` at the step
  boundary, checked against `RuntimeStats`.

## Risks, named

- **The win may be small.** If the dots do not hide under the memory read on 4 cores, `B = 4` buys
  little and the design's value collapses. Step 1 measures this before any runtime work.
- **Attention complexity.** Paged gather per entry is the fiddliest part and the least rewarding
  (small next to weights). Keep attention per entry rather than fusing it into the batch loop.
- **API churn.** `ForwardPass` is public. The default implementation is what keeps this additive.
- **Latency regression.** A long prefill in the batch delays every decode step. Chunked prefill is
  the fix; it is deliberately out of scope until the simple version is measured.
- **Non-determinism from scheduling.** Two runs with the same seeds must produce the same tokens
  regardless of arrival order; that is why RNG state is per sequence and draws happen inside the
  step in a fixed order.

## Order of work

Each step is shippable and has its own gate; stop at any point.

1. `q4k_dot_batch` / `q6k_dot_batch` + kernel bench at `B = 1, 2, 4, 8`. Gate: `B = 1` unchanged,
   aggregate bandwidth flat or better as `B` grows.
2. Pool jobs carry `B` activations; test: batched reply equals `B` separate calls, exactly.
3. `logits_batch` on the CPU backend with `B = 1` delegating to `run`; conformance test for
   equality; then `B > 1`.
4. `Engine::step` with the scheduler, `generate` reimplemented on top of it; single-sequence golden
   tokens must not move.
5. Server queue and worker thread; two concurrent requests must both complete with the tokens they
   would have produced alone.
6. Only if measurements justify it: chunked prefill, preemption, priority.

## Open questions

- `&mut [ForwardContext]` versus a `BatchContext` of parallel arrays: the former reuses the existing
  struct and its documentation; the latter avoids `B` mutable borrows of one KV slice.
- Does the batched kernel stay register-clean at `B = 4` on AVX2 (16 ymm registers, 64 activation
  floats per block), or does it spill into a `B = 2` special case?
- Should the pool size grow with the batch, or stay at the core count now that each job does `B×`
  the dots?
