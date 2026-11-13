# Continuous batching for the CPU engine

Status: steps 1-4 implemented (batched kernels, pool, `ForwardPass::logits_batch` on the CPU backend,
and `Engine::step` driving a batch), with their numbers in the measured sections; steps 5-6 are still
design. Everything below is checked against the tree as it stands today; line references are to files,
not to a commit, because the file is the thing that changes.

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
  `crates/vipera-runtime/src/engine.rs`), but before step 4 each `generate` call drove its own forward
  pass, and the server serializes all requests behind one `Mutex<Engine<CpuBackend>>`
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

`Engine::step` is the scheduler: one unit of work for every running turn.

```text
admit queued turns, oldest first, up to max_batch
forget turns whose sequence was cancelled or released since the last step
forward: prompt chunks if any turn needs one, otherwise one decode step per turn, in id order
draw: one token per turn holding logits, in id order, then finish the turn or continue it
```

`ForwardPass` gained one method with a default implementation, so existing implementors kept
compiling:

```text
fn logits_batch(&mut self, batch: &mut BatchContext) -> Result<Vec<Vec<f32>>, ViperaError> {
    // default: one call per entry, which is exactly today's behaviour
}
```

The CPU backend overrides it with the batched path. `logits` stays as the `B = 1` case.

Four things had to differ from the sketch this section started with.

**The batch seam is a `BatchContext`, not `&mut [ForwardContext]`.** A runtime has one KV allocation
with several block tables into it, so `B` `ForwardContext`s would each hold `kv: &mut [u8]` over the
same buffer: not expressible without `unsafe`, and unsound if it were. `BatchContext { kv, entries }`
carries the buffer once and one `BatchEntry` per sequence — everything `ForwardContext` carries except
`kv`, which `BatchEntry::context` reborrows for a backend that has to run the entries one at a time.

**Prefill is a chunk of at most one block, not one sequence at a time.** A turn is in one of three
phases: compute the next chunk of new tokens, compute the last retained position, or sample from the
logits it holds. A step runs one kind of forward pass only — chunks if any turn needs one, otherwise
decode steps — because the CPU backend falls back to one `run` per entry as soon as an entry carries
more than one token, so a mixed batch would batch nothing. A decode step therefore waits at most one
chunk of `tokens_per_block`.

**Sampling stays inside the step.** Each turn owns its `Sampler`, and draws happen in sequence id
order after the forward pass, so a sequence's random stream and its tokens do not depend on what else
was in the batch. A step's compute time is charged to every entry that shared the call: that is the
wall time the caller waited for, and at `B = 1` it is exactly the turn's own cost.

**`generate` is a client of the scheduler, not a parallel path.** `generate_with` enqueues and then
steps until its sequence's turn ends, so a batch of one is the old path forward call for forward call.

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

Batch scratch memory is the real growth: one `B × width` buffer per stage plus `B × vocab` logits,
i.e. about 5 MB at `B = 8` (4.9 MB of it the logits). Negligible against the cache, but not the
few hundred KB this note first guessed.

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

### Measured, once steps 1-3 were built

Kernel, `quant::tests::pooled_matvec_batch_throughput`, fastest of three rounds, 8 threads on the
i5-11300H. Throughput is aggregate tokens/s relative to `B = 1` on the same tensor:

| tensor | MB per read | `B = 1` | `B = 4` | `B = 8` |
| --- | --- | --- | --- | --- |
| `wq` Q4_K | 1.3 | 9.8 GB/s | 2.40x | 2.62x |
| `ffn_gate` Q4_K | 7.7 | 20.9 GB/s | 1.32x | 1.41x |
| `ffn_down` Q6_K | 11.3 | 15.9 GB/s | 1.87x | 2.40x |
| `output` Q6_K | 191.4 | 16.4 GB/s | 2.01x | 2.44x |

Step, `tests/batch.rs::batched_step_speedup_on_the_real_model`, one decode step per entry against the
same sequences run one after another:

| `B` | batched ms/step | serial ms/step | speedup | aggregate tok/s |
| --- | --- | --- | --- | --- |
| 1 | 61.2 | 64.1 | 1.05x | 16.3 |
| 2 | 101.9 | 129.1 | 1.27x | 19.6 |
| 4 | 164.7 | 271.9 | 1.65x | 24.3 |
| 8 | 287.9 | 537.5 | 1.87x | 27.8 |

Read against the gates above: `B = 1` is unchanged (one request is not slower), and aggregate
bandwidth is flat or better at every batch size, so step 1's own gate passes. The falsification
threshold — ~2x at `B = 4` — is met on Q6_K and on the small Q4_K tensor but not on `ffn_gate`, whose
single-entry read already runs at ~21 GB/s on this machine; weighted by bytes read, `B = 4` is about
1.7x and `B = 8` about 1.9x. The dots do not fully hide under the memory read on four cores, but the
step does get faster rather than merely cheaper per token.

### The Q4_K batch kernel, second pass

`ffn_gate` is where the batch ratio lagged, so the Q4_K kernel was the next target. Profiling by
removal (drop the minimum correction, measure, restore) put 35% of the kernel's time in that
correction: per entry and per 32-value sub-block it cost four `sums` loads, two adds, two multiplies
and two subtracts, while the `maddubs` chain it corrects is about as long. Two changes followed, both
keeping every float operation and its order intact, so single-request output is unchanged:

- **Hoisting.** The block's `(d * scale)` and `(dmin * min)` products do not depend on the entry, so
  they are computed once per sub-block instead of once per entry, and each entry's total lives in a
  register array written back once per call instead of a slice read-modify-written eight times.
- **32-value sums.** `x_sums` now emits the eight sub-block sums of each 256-value block instead of
  sixteen 16-value sums, built as the same pair of partial sums added in the same order. The
  correction's two adds per sub-block disappear from every row of every Q4_K tensor; the scalar
  reference reads the same values, one per sub-block.

Measured as an alternating A/B, fastest of eight interleaved runs of
`pooled_matvec_batch_throughput`, `ffn_gate` only:

| `ffn_gate` | before | after |
| --- | --- | --- |
| `B = 1` | 0.29 ms | 0.28 ms |
| `B = 2` | 0.49 ms | 0.42 ms |
| `B = 4` | 0.84 ms | 0.72 ms |
| `B = 8` | 1.86 ms | 1.60 ms |
| speedup at `B = 4` / `B = 8` | 1.38x / 1.25x | 1.56x / 1.40x |

The smaller Q4_K tensor moved with it (`wq` at `B = 8`: 1.78x to 2.13x, `B = 4` near 2x). Absolute
milliseconds drift between sessions on this laptop — the same binary measured `ffn_gate` at `B = 4`
between 0.70 and 0.84 ms — so the ratio, not the millisecond, is the number to trust.

One faster variant was built and thrown away: pairing the sixteen `sums` with one vector add and
scaling all eight minimum terms with one multiply, in both kernels, cut `ffn_gate` at `B = 4` to
0.68 ms. It changes the order the correction terms are added in, and that was enough to flip the
greedy fifth token from " also" to " Italy", which the correctness plan forbids. Re-wording the gate
to allow it is a separate decision, not a kernel change.

**AVX-512 was tried and lost.** This machine reports `avx512f`, `avx512bw`, `avx512vl`, and the
Tiger Lake core has 32 vector registers, so a 512-bit Q4_K kernel looked like the obvious next step:
one 64-byte activation load and one `vpmaddubsw` per 64-value sub-block instead of two of each, with
both nibble planes in one register and a coefficient vector carrying one scale per half. Isolated
throughput says the width is real — a tight `vpmaddubsw` loop measured 1.6-1.8x the values per
second of the 256-bit form, and a 64-byte load crossing a cache line cost nothing on this core. In
the actual kernel it was slower at every batch size, four alternating rounds, fastest of each:

| | `B = 1` | `B = 2` | `B = 4` | `B = 8` |
| --- | --- | --- | --- | --- |
| `ffn_gate` AVX2 | 0.28 ms | 0.40 ms | 0.76 ms | 1.41 ms |
| `ffn_gate` AVX-512 | 0.32 ms | 0.47 ms | 0.96 ms | 1.66 ms |
| `wq` AVX2 | 0.07 ms | 0.10 ms | 0.16 ms | 0.30 ms |
| `wq` AVX-512 | 0.08 ms | 0.11 ms | 0.19 ms | 0.33 ms |

14% slower at `B = 1`, where nothing spills and register pressure is a non-issue, so it is the
per-group cost of the wide form on this core rather than the batch structure. The 512-bit version
also changes the last bits — its final reduction adds the two halves separately — and that flipped
the same greedy token as the variant below. Rejected on both counts; the AVX2 kernel is what ships.

What this does not buy: `ffn_gate`'s matvec is 0.7 ms of a ~120 ms step, so a 14% kernel win is
~0.1% of the step. Re-measuring `batched_step_speedup_on_the_real_model`, three alternating runs
before and after, the step is unchanged within the noise at `B = 1, 2, 4` (-2.2%, +2.6%, -2.6%) and
about 9% faster at `B = 8` (256.8 to 233.4 ms/step). The same binary also re-measured the step table
above 15-25% lower than its original session — 53-56 ms at `B = 1` and 122-128 ms at `B = 4`, i.e.
1.8-2.0x rather than 1.65x — with no code change in the Q6_K tensors that dominate it. The machine's
absolute speed drifts by a quarter between sessions; only same-session A/B comparisons are worth
reading, and the numbers in the two tables come from different sessions.

### Step 4, as built

Gate: the single-sequence goldens must not move. They did not. `vipera generate ... --max-tokens 24
--temperature 0` still returns `[12095, 13, 576, 6722, 315, 9625, ...]` with 28 processed tokens, and
`--temperature 0.8 --top-k 40 --seed 7` still returns the sampled baseline, both byte for byte against
`target/baseline/greedy24.json` and `sampled24.json`. `tests/batch.rs` still shows every entry of a
batch reproducing its own single run bit for bit, now through the shared-buffer seam rather than
through one buffer per entry.

Two costs this bought, both visible in the code rather than measured: a block table is copied per
entry per step (one `u32` per block, so a few hundred bytes at `--ctx 2048`), and the whole batch's
compute time is charged to each entry's `compute` stage, which inflates the sum across sequences by
the batch factor. Neither is worth a data structure at this size.

Not yet measured: scheduler throughput at concurrency 4 on the real model. The server still holds
`Mutex<Engine>` and calls `generate`, so requests queue rather than share a step; the end-to-end
number needs step 5's queue and worker thread. The per-step table above is the stand-in, and the
engine tests (`two_sequences_in_one_step_produce_the_tokens_they_produce_alone`, `a_request_that_arrives_mid_turn_joins_the_batch`,
`cancelling_at_a_step_boundary_frees_the_blocks_and_ends_the_turn`, `arrival_order_does_not_change_a_sequence_tokens`)
cover the scheduling invariants on a synthetic forward pass.

### The Q6_K kernel on `i8` activations

Comparing the whole engine against llama.cpp (build `818ac6b`, same model file, same machine, CPU
backend, `-p 0 -n 128`) put the kernels back on the table:

| decode, tokens/s | 1 thread | 4 threads | 8 threads |
| --- | --- | --- | --- |
| llama.cpp | 14.60 | 38.61 | 27.9-37.8 (unstable) |
| vipera, before | 6.18 | 14.07 | 16.53 |
| vipera, after | 7.73 | 16.56 | 19.29 |

The gap was not the batching and not the thread count: at one thread, with no pool and no dispatch,
llama.cpp was still 2.4x faster. Q6_K was half the bytes of the model (507 MB of the 1.04 GiB read per
token) and the slowest kernel, because it multiplied six-bit levels against **`f32` activations**: eight
values per 256-bit vector, about 1.5 instructions per value, which caps a core near 5-7 GB/s. Q4_K had
already taken the other road — `QuantizedX` rounds the activation to `i8` once per matvec — and was
30-50% faster per byte on the same CPU.

`simd::q6k_dot_q8` now does the same: the six-bit levels are unpacked 32 at a time, `maddubs`
dotters them against the `i8` activation, and the `-32` of the levels is removed inside the integer
domain (`32 * xq` per pair) so the float side is one multiply per 16 values and one horizontal sum per
block. Throughput per tensor, 8 threads, from `pooled_matvec_throughput_by_tensor_type`:

| tensor | before | after |
| --- | --- | --- |
| `ffn_down` Q6_K | 13.6 GB/s | 22.6 GB/s |
| `output` Q6_K | 16.0 GB/s | 29.0 GB/s |
| `ffn_down` Q6_K, 1 thread | 4.6 GB/s | 9.6 GB/s |

The kernel changes what the engine generates: quantizing the activation is a loss of precision, and the
greedy fifth token moves from " also" to a different continuation at the same position. So the old
kernel is still there behind `VIPERA_EXACT_DOT=1`, which reproduces the pre-q8 tokens byte for byte
(check: `greedy24.json` matches with the switch and not without). `q6k_dot_batch_q8` shares the
unpacking across entries and is bit-identical per entry to the single call, so the batch invariant of
step 4 still holds.

What this does **not** buy yet: Q4_K is now the slow half (7.7 GB/s on one thread, 21.5 on eight), and
thread scaling is still far from linear (7.7 -> 19.3 from one thread to eight, against 14.6 -> 38.6
for llama.cpp from one to four). Beating llama.cpp needs both: a Q4_K kernel that streams at Q6_K's
rate, and a pool that scales. Single-core streaming on this machine measures ~11.6 GB/s by `dd`, which
is what a kernel should be aiming at.

### Four concurrent sequences, against llama.cpp

Both engines on the same machine (i5-11300H, 4C/8T) and model
(`qwen2.5-1.5b-instruct-q4_k_m.gguf`), greedy, fastest of two rounds. vipera's numbers are the step
bench in `crates/vipera-backend-cpu/tests/batch.rs`, which decodes one token per sequence per step and
leaves out the sampler; llama.cpp's are `llama-server --parallel 4` answering four requests at once.

| | 1 request | 4 concurrent | batching gain |
| --- | --- | --- | --- |
| vipera, 4 threads | 20.1 tok/s | 32.5 tok/s | 1.62x |
| llama.cpp, 4 threads | 38.4 tok/s | 90.1 tok/s | 2.34x |
| vipera, 8 threads | 23.9 tok/s | 35.4 tok/s | 1.48x |
| llama.cpp, 8 threads | 35.2 tok/s | 83.2 tok/s | 2.36x |

So batching helps vipera, but it widens the gap rather than closing it: 1.9x behind at one request,
2.8x behind at four. The per-step numbers say why. At four threads a vipera step costs 49.8 ms for one
sequence and 123.1 ms for four, so the step grows 2.5x to serve 4x the tokens; llama.cpp's grows about
1.7x. Streaming the 925 MB of weights once per step, vipera's batched step runs at 7.5 GB/s against
18.6 GB/s for the single-sequence step: `q6k_dot_batch_q8` does four times the integer work per byte
it loads, so at `B = 4` it is compute-bound on the AVX2 multiply-add units, not memory-bound. The next
kernel to write is a batched gemv that keeps the row register-blocked across entries, not `B` dots
sharing one load.

### The batched kernel, register-blocked

The scale path, not the dot, was the cost. Each entry built its own scale vector from scalars:
`_mm256_set_m128(set1(high_scale * dx), set1(low_scale * dx))`, which disassembles to two
`vmulss`, two broadcasts and a `vinsertf128` per 32 values per entry — 216 `vmulss` in the Q6_K
kernel against 80 `vpmaddwd`. The weight side of that product does not depend on the entry, so it is
now built once per group and an entry multiplies it by its own `dx` with one vector multiply:
`(d * scale) * dx`, the same operands in the same order, so nothing moves. Q4_K got the same
treatment.

| at 4 threads | before | after |
| --- | --- | --- |
| step, `B = 1` | 49.8 ms | 46.1 ms |
| step, `B = 4` | 123.1 ms | 102.3 ms |
| 4 concurrent, aggregate | 32.5 tok/s | 39.1 tok/s |
| 4 concurrent, per request | 8.1 tok/s | 9.8 tok/s |
| scalar multiplies in the Q6_K kernel | 216 `vmulss` | 64 |

The two batch equality tests (`avx2_q4k_batch_is_exactly_each_entry_alone`,
`avx2_q6k_q8_batch_is_each_entry_alone`) still pass, so a batched entry is still its own single call
bit for bit. The remaining distance to llama.cpp's 90 tok/s is the arithmetic of four entries per
loaded byte: at `B = 4` the step moves 925 MB in 102.3 ms, 9.0 GB/s against 20 GB/s for `B = 1`, and
the op count per 32 values per entry (two `maddubs`, `madd`, convert, two multiplies, add) is close to
what the profile shows. VNNI's `vpdpbusd` collapses two of those into one, which is worth trying now
that the kernel is arithmetic-bound rather than memory-bound — the earlier VNNI measurement was taken
against a memory-bound single-row kernel and therefore said nothing about this case.

### Where a decode step actually goes

The same weights are streamed whatever the context length, so the difference between a step at 32
keys and one at 1024 is attention and the KV it reads — the entire budget any attention kernel could
claim. From `what_a_decode_step_spends_on_attention`, one sequence, four threads:

| context | ms/step | attention and KV |
| --- | --- | --- |
| 32 | 59.4 | 0.0 (0%) |
| 256 | 68.4 | 9.1 (13%) |
| 1024 | 117.9 | 58.5 (50%) |

At 1024 keys attention costs more than the weights do. The 29 MB of KV it reads took 58.5 ms, about
0.6 GB/s, which is not a bandwidth limit but the shape of the loops: keys walked one position at a
time, and every value of the KV decoded through a scalar `f16` loop. So at server-length contexts the
next win is attention, not another matvec; at 128-token generations (`ctx` under ~100) it stays
invisible, which is why the end-to-end numbers barely move.

### The attention path, three loops

Three changes, in the order their cost was measured, all at four threads and one sequence:

| | before | after |
| --- | --- | --- |
| step at 32 keys | 59.4 ms | 54.6 ms |
| step at 1024 keys | 117.9 ms | 73.2 ms |
| attention and KV at 1024 keys | 58.5 ms | 18.6 ms |

1. **The score dot** (`simd::dot_f32`). 128 products per key position in one `addss` chain: four
   cycles of latency per product, per position, per head. Eight lanes and two accumulators cut the
   chain to sixteen links. This is a tree reduction, so it is *not* the same rounding as the
   sequential sum, and `VIPERA_EXACT_DOT=1` keeps the sequential loop for reproducing the golden
   tokens. The 24-token greedy run is unchanged by it at short context; long contexts can differ.
2. **The value accumulation** (`simd::axpy_f32`). Same eight lanes, but `target[i] += weight *
   source[i]` in that order, so it is bit-identical and needs no switch.
3. **The KV read** (`simd::half_to_float`). `load_layer` decoded `f16` to `f32` one value at a time
   for every position of every layer, and looked the block up per position. F16C converts eight at a
   time, and the test compares it against the scalar conversion over all 65536 bit patterns rather
   than trusting it; the block table is now consulted once per block. Both are exact — a test of the
   batched path against the alone path caught the one real bug here, an offset taken from the table
   index instead of the physical block.

What is left of the 18.6 ms is the softmax `exp` loop (344k libm calls a step) and the score dots
(4.8 ms). A vectorized `exp` would change the values, so it belongs behind the same switch.

**VNNI was tried and dropped.** `vpdpbusd` collapses `maddubs` + `madd` into one instruction, and the
kernel is arithmetic-bound, so the op count drops by one per entry per group. Bit-exactness held (the
batch equality tests run through it). The step did not move: four runs with VNNI measured 107.2, 112.5,
123.8, 117.9 ms against two without at 125.2 and 122.0 ms in the same session — and the batched step's
own run-to-run spread is ±10%, which is as large as the effect. The kernels were removed rather than
kept as dead code, exactly as the four-row kernels were.

Two things measured along the way, both negative, both worth not repeating:

- **Four-row matvec kernels.** `q4k_dot_four` and `q6k_dot_four_q8` were written three ways (arrays of
  accumulators, one macro expansion per row, four rows interleaved inside the `j` loop) and were
  bit-identical to four single calls in all three. None was faster: `output` Q6_K on one thread went
  9.7 -> 8.6 -> 7.3 GB/s, and at eight threads 29.0 -> 25.5. A C prototype of the same shape measured
  18 GB/s because it skipped the per-block scalar work (f16 scale decode, `q4k_scale_min`), which is
  where this kernel actually spends its time. Four streams in flight only pay when the loop is waiting
  on memory, and at ~10 GB/s per thread this one is waiting on scalar bookkeeping.
- **Pool dispatch.** `fill_batch` sends each job over an `mpsc` channel and copies the reply back, once
  per `matvec` (197 times a token). Compared against spawning threads per call on the 191 MB `output`
  tensor: 9.64 vs 9.99 GB/s at one thread, 28.96 vs 27.66 at four, 37.79 vs 37.12 at eight. The channel
  round trip is inside the noise, so the pool is not what is holding scaling back; the 3.0x the kernel
  does get at four threads (9.6 -> 29.0 GB/s) is the memory system, and the token adds its serial
  share on top.
- **Grouped projections.** `wq`, `wk` and `wv` multiply the same activation, as do `wg` and `wu`, so
  they were issued as one pool dispatch instead of one per tensor: seven dispatches a layer became
  five. Bit-identical (a pooled unit test compares every output's bits against the per-tensor path,
  and 32 generated tokens match to the last id) and inert: 24 interleaved 32-token runs, alternating
  which binary ran first, gave a paired `group / base` median of **0.999** on `compute` micros (min
  0.927, max 1.097). The ceiling was visible before the attempt — the whole non-dot remainder of a
  step, all 197 dispatches included, is 0.90-1.5 ms of a 35-45 ms step — so two dispatches a layer
  are under 0.3 ms. Reverted; the idea is recorded so it is not rebuilt from scratch.

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

- ~~`&mut [ForwardContext]` versus a `BatchContext` of parallel arrays~~ — settled by step 4: the KV
  buffer is carried once in a `BatchContext` and each entry holds everything else, because `B` mutable
  borrows of one allocation cannot be expressed without `unsafe`.
- Does the batched kernel stay register-clean at `B = 4` on AVX2 (16 ymm registers, 64 activation
  floats per block), or does it spill into a `B = 2` special case?
- Should the pool size grow with the batch, or stay at the core count now that each job does `B×`
  the dots?
