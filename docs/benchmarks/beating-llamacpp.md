# What would actually beat llama.cpp on this laptop

*Generated 2026-11-09. Sources: 11. Confidence: medium-high on the bandwidth ceiling (measured
here), medium on the kernel recommendations (sourced from other CPUs), low on the exact payoff of
any single change on Tiger Lake.*

This report answers one question: vipera.rs is 2x slower than llama.cpp on the same model, and the
previous report ([`cpu-algorithms.md`](./cpu-algorithms.md)) said the gap is kernel shape. Before
writing another kernel, measure the ceiling. That measurement changes the plan.

**Update 2026-11-14 — two of its recommendations are now measured and wrong on this machine.** The
numbers behind them are in [`../benchmarks.md`](../benchmarks.md).

- *"Layout first: repacked `block_q4_Kx8` GEMV, then 256-bit VNNI, never 512-bit alone."* Timed on
the ADR-001 shape (8960x1536, 128 activation rows), one thread: ggml's repacked
`ggml_gemm_q4_K_8x8_q8_K` does **170.0 GFLOPS** against **198.6** for the 512-bit VNNI tiled kernel
the engine has had since 2026-11-13, which reaches 776 GFLOPS on eight threads. The repack would add
a 625 MB load-time transform and an FFI dependency for a loss.
- *"Prefill is already at the bandwidth wall; only speculation can beat llama.cpp."* Prefill turned
out to be compute-bound, not bandwidth-bound: a 128-token chunk spends 59% in Q4_K at 776 GFLOPS and
17% in Q6_K, and one-pass activation quantization alone moved `pp128` from 115.5 to 159.6 tok/s
while `pp512` went from 96.2 to 123.9.

The bandwidth ceiling in section 1 — 44.7 GB/s at four threads — still holds, and the decode side of
the analysis is unchanged.

## Executive summary

1. **The machine has a hard ceiling of 44.7 GB/s at four threads and 18.7 GB/s at one** (measured
   here, best of five rounds). A decode step must read 1.117 GB of Q4_K/Q6_K weights per token.
2. **llama.cpp already runs at 87% (1 thread) and 96% (4 threads) of that ceiling.** Its recorded
   14.59 and 38.29 tok/s correspond to 16.3 and 42.8 GB/s.
3. vipera runs at 52% and 57% of the same ceiling. The entire gap is streaming efficiency inside
   the kernel path, and **the maximum any kernel work can buy is 1.05x over llama.cpp** — parity,
   not a win.
4. To *beat* llama.cpp the token count per weight read has to fall. The only lever in the earlier
   report that does that is prompt-lookup / speculative decoding, which emits more than one token
   per pass. Everything else competes for the same 44.7 GB/s.
5. If kernel work is still wanted (it is needed for batch, where we are compute-bound), the
   evidence points at **8-row interleaved `block_q4_Kx8` GEMV**, then **256-bit AVX-VNNI** — and
   explicitly not at 512-bit kernels, which we already tried and which a second team also measured
   as slower.

## 1. The ceiling, measured on this machine

A 40-line C program reads 1.174 GB with 1 and 4 threads (`malloc` + strided `double` loads, best of
5 rounds, i5-11300H, DDR4-3200 dual channel) [measured here, no URL]:

| threads | best | other rounds |
|---|---|---|
| 1 | **18.7 GB/s** | 18.0, 18.1, 18.2, 18.7 |
| 4 | **44.7 GB/s** | 42.9, 44.3, 44.7, 28.8 (one outlier) |

## 2. Re-deriving the target from that ceiling

One decode step reads ~1.117 GB (the model's quantized weights; attention and KV are 0% of a step
at context 32 and 1% at 256, measured by `what_a_decode_step_spends_on_attention`). So
`tok/s = bandwidth / 1.117 GB`:

| | tok/s | GB/s | share of ceiling |
|---|---|---|---|
| bandwidth ceiling, 1 thread | 16.7 | 18.7 | 100% |
| llama.cpp, `-t1` | 14.59 | 16.3 | **87%** |
| vipera, `VIPERA_THREADS=1` | 8.7 | 9.7 | 52% |
| bandwidth ceiling, 4 threads | 40.0 | 44.7 | 100% |
| llama.cpp, `-t4` | 38.29 | 42.8 | **96%** |
| vipera, `VIPERA_THREADS=4` | 19.5-23.9 | 21.8-26.7 | 49-60% |

Two conclusions follow directly. First, llama.cpp's numbers are not a mystery to be reverse
engineered from algorithms: they are a well-streamed 2 GB/s-wide memory bus. Second, **there is no
1.6x available from kernel work against llama.cpp** — there is 1.045x, and only on the 4-thread
number. The 1.6x is available against *ourselves*.

## 3. What llama.cpp actually executes for Q4_K

Two facts kill the obvious hypotheses.

- The AVX2 Q4_K dot kernel is `ggml_vec_dot_q4_K_q8_K` and it **asserts `nrc == 1`**: one weight
  row, one activation column per call. There is no multi-row AVX2 kernel, and the ARM `nrc == 2`
  variant needs SVE + i8mm. Its cost is ~100-104 uops per 256-weight superblock, of which 8 are the
  actual dot products; ~38% is per-superblock bookkeeping, and it loads 256 B of shuffle-table
  constants per 128 B of weight bytes ([E026, kernel anatomy](https://github.com/Isuhail29/affordable-local-llm-research/blob/main/notes/e026-vecdot-anatomy.md)).
  So ggml's per-row kernel is *not* leaner than the one in `simd.rs`; it is roughly the same class
  of code.
- The k-quant x86 kernels are AVX2-only — no AVX-512, no VNNI in the Q4_K path
  ([same source](https://github.com/Isuhail29/affordable-local-llm-research/blob/main/notes/e026-vecdot-anatomy.md),
  [llamafile issue #973 write-up](https://gist.github.com/aittalam/de0f8a494ff17d38c100f7ad08c0d7da)).

What llama.cpp does have, and we do not, is a **repacked 8-row interleaved layout**:
`block_q4_Kx8 { d[8]; dmin[8]; scales[96]; qs[1024]; }` = 1152 bytes for 8 rows × 256 values, the
same footprint as 8 separate Q4_K blocks, selected at runtime on AVX2 for any tensor whose row
count is a multiple of eight
([`repack.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.h?plain=1),
[repack source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp)).
Its GEMV, `ggml_gemv_q4_K_8x8_q8_K`, covers the thread's whole row slice in one call: activation
loads and bsums are shared across 8 rows, the 6-bit scale unpack is amortized 4x, and the 8 outputs
stay in registers so there is no per-row horizontal reduction
([E026 §"Multi-row verdict"](https://github.com/Isuhail29/affordable-local-llm-research/blob/main/notes/e026-vecdot-anatomy.md)).

How much that is worth, measured by someone else: **prefill 2-3x, decode only 1.05-1.06x** — 4.48 →
4.73 tok/s on Ministral-3-8B Q4_K_M at 4 threads, llama.cpp b11539, because "a repacked GEMV reads
the same bytes per token"
([repack baseline](https://github.com/gregordinary/rockchip-npu-notes/blob/main/perf/cpu-repack-baseline.md)).

That last sentence is the whole problem. At batch 1 the layout is a latency win, not a bandwidth
win, and llama.cpp is already at 87-96% of bandwidth. Our 52-57% says our kernel *is* losing to
latency and bookkeeping, so the layout should help us more than it helped them — but it cannot take
us past the ceiling.

## 4. Why 512-bit did not win, and why 256-bit VNNI still might

Our AVX-512 Q4_K and Q6_K/VNNI attempts measured 8.7 vs 8.7 tok/s (1 thread) and 19.8 vs 20.0
(4 threads); the work was reverted, and the note lives at the top of `simd.rs`. Two independent
results explain it rather than dismiss it:

- **Frequency was not the cause.** On Ice Lake and Rocket Lake client parts there is essentially no
  license-based downclock: heavy 512-bit code ran at 3.6 GHz against 3.7 GHz scalar, and "256-bit
  heavy instructions have the same license as 512-bit light"
  ([Travis Downs, ICL AVX-512 downclocking](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html)).
  So the 512-bit kernel lost on its own shape, not on clocks — it was the same per-row,
  per-superblock-bookkeeping structure, just twice as wide.
- **Wider VNNI can be slower than half-width maddubs.** A .NET runtime measured 512-bit `VPDPBUSD`
  at 1.43x *slower* than 256-bit `vpmaddubsw` + `vpmaddwd`, and the 256-bit AVX-VNNI path at
  **1.3-1.6x faster** on Meteor Lake — Intel client silicon with the same VNNI generation as Tiger
  Lake ([dotLLM #322](https://github.com/kkokosa/dotLLM/issues/322)).

So the order matters: layout first (it removes the per-row serial chains), then 256-bit VNNI on top
of the layout, never 512-bit alone.

## 5. Options, ranked by expected value

### A. Prompt-lookup decoding — the only lever that goes past the wall

Every option below re-reads 1.117 GB per token. Prompt-lookup instead verifies a draft of *n*
tokens copied from the prompt, so it emits up to *n*+1 tokens per weight read. It needs no model,
no crates, and it is already the section 4 recommendation of
[`cpu-algorithms.md`](./cpu-algorithms.md) (2.4x measured on grounded tasks there; that number is
from a different implementation and remains unverified here).

Why it is the answer to "beat llama.cpp": with the bus at 96% of ceiling, a 1.5-2x win can only come
from doing fewer passes. A 2x win puts vipera at 40-48 tok/s at 4 threads against llama.cpp's 38.29,
i.e. the only path that is not a rounding error.

Effort: M. Gate: ≥1.5x aggregate tokens/s on a document-grounded prompt set, measured interleaved,
with accepted-token counts reported per run so a rejection-heavy run cannot fake the win.

#### Measured outcome: 0.99x, and it cannot do better until B exists

Implemented end to end (n-gram drafts from the sequence's own context, all draft logits from one
forward pass, argmax verification per position, KV rollback, greedy-only, `VIPERA_PLD`) and measured
on `qwen2.5-1.5b-instruct-q4_k_m`, 4 threads, 96 tokens, interleaved:

| run | wall | tok/s | drafts | accepted |
| --- | --- | --- | --- | --- |
| `VIPERA_PLD=0` | 6.65 s, 6.66 s | 15.53, 15.48 | — | — |
| `VIPERA_PLD=4` | 6.72 s, 6.98 s | 15.35, 14.77 | 60 | **60 (100%)** |

Tokens were bit-identical between the two arms, so the machinery was correct and acceptance was as
good as it gets; the speed simply did not move. The counters explain it: a 5-token draft pass cost
209-251 ms, while a plain single-token step costs ~44 ms. The cost of a chunk is **linear in its
length** — `run` loops tokens on the outside and layers on the inside, so position *i* re-reads all
1.117 GB. Five positions therefore read 5.6 GB, exactly as five plain steps would. The only token
speculation ever saves is the correction token, and that saving is cancelled by the drafting and
flush overhead: 18 draft passes covered 96 tokens, i.e. the same number of weight reads as plain
decoding.

So A is not an independent lever. It pays off only once the positions of one chunk share a weight
read, which means a batched prefill GEMM over the chunk — the `ggml_gemm_q4_K_8x8_q8_K` shape that
B introduces. **B is a prerequisite for A, not a follow-up.** The code was removed after the
measurement, per the rule that an optimization which cannot be measured is not kept; this section
is the record. Re-attempt A after B, with the same gate.

### B. 8-row interleaved Q4_K GEMV — closes our own 1.6x, reaches parity

Repack at load into `block_q4_Kx8` (1152 B per 8 rows, byte permutation only), then a GEMV that
keeps 8 accumulators in registers, shares the activation block and its bsums across the 8 rows, does
one horizontal reduction per row group, and leaves a scalar tail for rows not divisible by 8. All
our matvec shapes qualify (1536, 8960, 151936 are multiples of 8).

Expected: sourced decode 1.05-1.25x, prefill 2-3x, and a larger decode gain here because our
per-row bookkeeping is the thing that is 40% off the ceiling. At batch 4 the same layout turns one
weight load into 32 row-entry dots, which is exactly the regime our 4-entry step is compute-bound
in; ik_llama.cpp reports B=4 token generation 1.88x over mainline with row-interleaved weights
(2.44 → 4.58 tok/s), against only 1.24x at B=2
([discussion #164](https://github.com/ikawrakow/ik_llama.cpp/blob/main/github-data/discussions/164%20-%20Latest%20CPU%20performance%20comparison%20with%20llama.cpp.md)).

Effort: L (layout + repack + kernel + tails + tests). Gate: q4_Kx8 GEMV ≥1.4x over the current
`q4k_dot` on the 8960×1536 shape at 4 threads, then step time at B=1 and B=4, then the HTTP bench.
Bit-exactness stays testable: the integer sums are exact and each row's float combination order is
unchanged, so `to_bits` equality per sequence must still hold.

#### Measured outcome: 0.88-0.98x — gate failed

Built the layout (1152 B per superblock: `d[8]` at 0, `dmin[8]` at 16, `32 + 12*row` for the scale
bytes, `128 + (j*8 + row)*32` for the nibble groups) and a four-row-at-a-time GEMV over it, with
each row's coefficient and accumulator order copied from `q4k_dot` exactly. Every row of every group
was **bit-identical** to its own `q4k_dot` call (`max |diff| 0`) on both real shapes, so the layout is
correct — and slower:

| shape | threads | `q4k_dot` per row | 8-row GEMV | ratio |
| --- | --- | --- | --- | --- |
| ffn_gate+up 8960×1536 | 1 | 11.3 GB/s | 10.2 GB/s | 0.90x |
| ffn_gate+up 8960×1536 | 4 | 21.5 GB/s | 18.9 GB/s | 0.88x |
| ffn_down 1536×8960 | 1 | 11.4 GB/s | 10.3 GB/s | 0.90x |
| ffn_down 1536×8960 | 4 | 34.4 GB/s | 33.6 GB/s | 0.98x |

The breakdown in §5's opening measurement says why: sharing the *activation* loads saves the part
of the time that was loads (27-52%), while the per-block bookkeeping (48-73% of the kernel) is
per-row — every row has its own `d`, `dmin` and twelve scale bytes, so eight rows still do eight
scale unpacks, eight coefficient builds and eight minimum accumulations. The repack buys sharing on
the smaller half. That matches mainline's own 1.05-1.06x for decode, and the 2-3x figure is a
**prefill GEMM** property, which needs several activations per weight read, not eight rows per one
activation.

Removed after the measurement, per the rule that an optimization which cannot be measured is not
kept. The kernels that produced these numbers were test-only, so no production path was touched.

### C. 256-bit AVX-VNNI on top of B

`vpdpbusd` on YMM, no 512-bit registers: 1.3-1.6x on Meteor Lake for the same class of kernel
([dotLLM #322](https://github.com/kkokosa/dotLLM/issues/322)). Only worth writing once B exists,
because VNNI reduces instruction count and B is what removes the serial chains; our earlier attempt
did neither.

Effort: M after B. Gate: microbench only, with the same interleaved protocol as the 512-bit attempt.

#### Measured outcome: 1.09-1.21x at the kernel, 1.05x end to end — kept

B failed, but its measurement left the reason to try C anyway, and this CPU has the instruction. It
has no `avx_vnni`; it has **`avx512_vnni` + `avx512_vl`**, which is the 256-bit `vpdpbusd` on YMM
without a 512-bit register, so the downclock that sank the earlier attempt does not apply. One
`vpdpbusd` replaces the `maddubs` + `madd`-against-ones pair with the same integer result, so the
float path is untouched and every row stays bit-exact.

| shape | threads | `maddubs` | `vpdpbusd` | ratio |
| --- | --- | --- | --- | --- |
| attn_q 1536×1536 | 1 | 11.7 GB/s | 13.2 GB/s | 1.13x |
| attn_q 1536×1536 | 4 | 20.1 GB/s | 21.9 GB/s | 1.09x |
| ffn_gate+up 8960×1536 | 1 | 10.7 GB/s | 11.8 GB/s | 1.11x |
| ffn_gate+up 8960×1536 | 4 | 32.9 GB/s | 39.9 GB/s | 1.21x |
| ffn_down 1536×8960 | 1 | 11.9 GB/s | 13.7 GB/s | 1.15x |
| ffn_down 1536×8960 | 4 | 36.4 GB/s | 41.8 GB/s | 1.15x |

Every row bit-identical (`to_bits` equality) on all three shapes. End to end, 96 tokens, greedy,
interleaved, switching kernels with `VIPERA_NO_VNNI=1`:

| threads | `maddubs` | `vpdpbusd` | speedup | tok/s |
| --- | --- | --- | --- | --- |
| 4 | 5.72 s | 5.45 s | **1.049x** | 18.3 → 19.3 |
| 1 | 12.51 s | 11.55 s | **1.083x** | 7.9 → 8.5 |

One finding worth keeping: the decode path does not call the single-row dot at all. Every step goes
through the pooled batch kernel, even at batch 1, so the first version of this change — which only
replaced the single-row kernel — measured **1.000x** end to end while the microbench showed 1.15x. The
gain only appears once `q4k_dot_batch` itself uses `vpdpbusd`, which is where it now lives, behind
the same feature check. `VIPERA_NO_VNNI` stays as the A/B switch; both arms produce identical
tokens.

### D. Rejected

- 512-bit kernels without the layout change: measured no win here, and independently measured 1.43x
  slower in another project.
- Any kernel work that does not reduce bytes read, as a way to beat llama.cpp: the ceiling is
  44.7 GB/s and they are at 42.8.
- Q4_K by itself is the wrong lever for the 1.7x: 1.117 GB/token is a property of the file, not the
  code. A smaller quant would move the ceiling, but it is a different model.

### E. Measured: where a decode step's time goes (and what that rules out)

`VIPERA_PROFILE_STEP=1` accumulates the nanoseconds of each phase and prints them per pass. Prompt
chunks are printed as `PROFILE chunk` and decode steps as `PROFILE decode`, because a chunk computes
several positions per weight read and averaging the two describes neither. **Earlier numbers in this
section were the mixed average and understated decode by about 1.7x; they are corrected here.**
Decode-only, `qwen2.5-1.5b`, prompt `Hi` (so attention has almost nothing to read), 24 tokens,
`--ctx 512`, median of 5 runs at 4 threads:

| phase | 1 thread | 2 threads | 4 threads | share at 4 | in-situ GB/s at 4 |
| --- | --- | --- | --- | --- | --- |
| Q4_K products (625 MB) | 59.9 ms | 39.5 ms | 27.1 ms | 61% | 23.1 |
| Q6_K output projection (191 MB) | 37.4 ms | 21.4 ms | 15.9 ms | 36% | 12.0 |
| attention | 0.13 ms | 0.13 ms | 0.13 ms | 0% | — |
| rest (norms, rope, adds, dispatch) | 1.5 ms | 1.5 ms | 1.5 ms | 3% | — |
| **step** | **98.9 ms** | **62.6 ms** | **44.6 ms** | | **22.4 tok/s** |

The pool is not the problem: one dispatch costs **8.4 µs** measured with a tiny product, and the
whole non-dot remainder is 1.5 ms of a 44.6 ms step. Q4_K scales only 2.2x from one thread to four
(59.9 → 27.1 ms) while the machine's own ceiling is 18.7 → 44.7 GB/s, so the in-situ kernel reaches
roughly half the ceiling at every thread count. Two attempts to close that with more memory-level
parallelism were measured and thrown away: a two-row kernel (`q4k_dot_batch_pair`, verified live by
forcing it wrong and watching the tokens change) was **0% at 1, 2 and 4 threads**, and the eight-row
layout in §5B was 0.88-0.98x. The loop is bound by instruction throughput, not load latency, which
is also why an in-kernel prefetch measured neutral earlier.

At long context the picture inverts. Same model, 4 threads, a 2281-token prompt repeated from one
sentence, `--ctx 4096`, 48 decode steps; the marginal cost per step at ~2300 tokens of context:

| phase | ms/step | share |
| --- | --- | --- |
| attention | 85.8 | 25% |
| Q4_K products | 142 | 41% |
| Q6_K output projection | 86 | 25% |
| rest | 30 | 9% |
| **step** | **344** | **2.9 tok/s** |

A 2300-token context costs **7.7x a fresh one** (344 vs 44.6 ms) and attention goes from 0.1 ms to
86 ms, so at this length the KV path, not the weights, is the second-largest consumer. That traffic
is 392 MB per step (2280 positions x 28 layers x 2 KV heads x 2 tensors x f16) against 816 MB of
weights, and every phase slows at once: Q4_K reads the same 625 MB as before yet takes 142 ms
instead of 27. The leading explanation is DRAM contention from the gather pattern of attention, but
the per-phase attribution at long context is **not yet reconciled** (at decode step 1 the same run
reports 1.2 ms of attention while its Q4_K phase is already 151 ms), so treat these shares as a
ranking and not as a budget. The next lever at long context is the attention kernel and the KV
layout — batched gather, q8_0 KV, or flash-style streaming — measured the same way.

Q6_K, with VNNI, measured **1.02x**: the six-bit kernel needs `q - 32`, so a second `vpdpbusd`
against ones plus a shift takes back what the first one saved. Not kept.

Prefill, on the other hand, is compute-bound and was leaving 8x on the table: it ran one position per
weight read. The chunk now goes through the batch path, one entry per position, ordered so each
entry's K/V is written before the position that reads it. Prompt of ~160 tokens, 4 threads, wall
including load: **4.94 / 5.02 / 5.08 s → 3.80 / 3.63 / 3.75 s (1.35x)**, tokens bit-identical
(`sha` equal), which also takes a 96-token run on the grounded prompt from 6.90 s to 5.93 s.

## 6. The prediction to hold the work to

- B alone: predicted 35-40 tok/s at 4 threads, **measured 0.88-0.98x at the kernel**, so the
  prediction is withdrawn: on this CPU the 8-row layout does not pay for the per-row bookkeeping it
  cannot share.
- A alone: 0.99x, measured, because a chunk still reads the weights once per position (see above).
- C alone: **1.05x at 4 threads, 1.08x at 1 thread**, measured and bit-exact — real, but nowhere
  near the 1.7x gap to llama.cpp.
- Chunked prefill: **1.35x on wall time to the first tokens**, bit-identical, kept — it changes
  latency, not decode throughput.
- A + B: withdrawn with B. The remaining candidate for a win over llama.cpp is a batched prefill
  GEMM (several activations per weight read), which is what makes both A and B pay; it is not any
  of the four options above and would be its own research question.

## 7. Measurement plan

1. Re-run the bandwidth probe (1 and 4 threads) before and after each change; it moves ±10% with
   machine state, and the whole analysis is a ratio against it.
2. Kernel microbench: one 8960×1536 Q4_K tensor, current `q4k_dot` vs the candidate, interleaved
   A/B/A/B, 4 rounds each, report GB/s against the probe.
3. Step bench: `batched_step_speedup_on_the_real_model` for B=1/2/4/8.
4. HTTP bench: 1 and 4 concurrent at 4 threads, 3 rounds, printing aggregate tok/s and tokens.
5. Correctness: `to_bits` per sequence, batch == alone over HTTP, greedy token list unchanged
   (`[12095, 13, 576, 6722, 315, 15344]`).

## 8. Gaps this research could not close

- No published measurement of `block_q4_Kx8` decode on Tiger Lake or on a 4-core laptop; the 1.05x
  figure is from a Cortex-A76 phone SoC at 4 threads.
- Our own 4-thread vipera numbers come from two different harnesses (CLI 19.5-23.9 tok/s, model
  bench 44.0 ms/step = 22.7 tok/s); the spread is real and the HTTP path is the slowest of them.
- No profile of where our 44 ms/step goes *inside* the weights (per tensor); the 4-row GEMV
  decision rests on the aggregate GB/s gap, not on a per-tensor flame graph.

## Sources

1. [E026: Anatomy of `ggml_vec_dot_q4_K_q8_K` on AVX2](https://github.com/Isuhail29/affordable-local-llm-research/blob/main/notes/e026-vecdot-anatomy.md) — per-superblock uop counts, 38% bookkeeping, repack verdict.
2. [llama.cpp `arch/x86/quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c) — the AVX2 kernel itself.
3. [llama.cpp `repack.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.h?plain=1) — `block_q4_Kx8` definition and size.
4. [llama.cpp `repack.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/repack.cpp) — interleaving code, `ggml_gemv_q4_K_8x8_q8_K`.
5. [Repack baseline notes](https://github.com/gregordinary/rockchip-npu-notes/blob/main/perf/cpu-repack-baseline.md) — decode 1.05-1.06x with repack, prefill 2.5-3x.
6. [ik_llama.cpp discussion #164](https://github.com/ikawrakow/ik_llama.cpp/blob/main/github-data/discussions/164%20-%20Latest%20CPU%20performance%20comparison%20with%20llama.cpp.md) — B=1/2/4/8 token generation with row-interleaved weights.
7. [ik_llama.cpp July-2024 TG comparison](https://github.com/ikawrakow/ik_llama.cpp/wiki/July-2024:-Token-generation-performance-comparison) — k-quants are memory bound at 4 threads; Q4_K_S AVX2 9.61 → 10.72 tok/s (1.116x).
8. [Travis Downs, Ice Lake AVX-512 downclocking](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) — no license downclock on client ICL/RKL.
9. [dotLLM issue #322](https://github.com/kkokosa/dotLLM/issues/322) — 256-bit VNNI 1.3-1.6x on Meteor Lake, 512-bit VNNI 1.43x slower.
10. [llama.cpp PR #12829](https://github.com/ggml-org/llama.cpp/pull/12829) — AVX-512 Q4_Kx8 GEMM gains prompt processing only; tg128 unchanged (14.78 → 14.77).
11. [llamafile issue #973 write-up](https://gist.github.com/aittalam/de0f8a494ff17d38c100f7ad08c0d7da) — mainline k-quant matmul runs the single-row AVX2 kernel per (expert × token × row).

## Methodology

The `deep-research` skill's firecrawl/exa MCP servers are not configured in this session, so the
workflow was run with the available web search and page reader: 6 queries, 11 sources, 2 read in
full (E026, dotLLM #322), plus two local measurements (bandwidth probe, attention share) and the
numbers already recorded in this repository. Sub-questions: (1) what does llama.cpp actually execute
for Q4_K on x86, (2) how much do row-interleaved layouts buy at decode vs prefill, (3) does AVX-512
or VNNI width explain our failed attempt, (4) what is the ceiling on this hardware.
