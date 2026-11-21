# Benchmarks

The measurement ledger. Every performance claim in this repository points at a dated section below,
and the experiments that lost stay in it: a dead end that is written down is a dead end nobody builds
twice.

## The machine these numbers come from

Unless a section says otherwise: Intel i5-11300H (4 cores, 8 threads, AVX2, F16C, AVX-512 VNNI but
double-pumped), 2×DDR4-3200, Linux, the toolchain pinned by `rust-toolchain.toml`, Qwen2.5-1.5B-Instruct
Q4_K_M, `VIPERA_THREADS=8`, f16 KV cache. The measured DRAM ceiling on this box is 43 GB/s, and that
number is what most of the percentages below are relative to.

Two habits the numbers depend on:

- **Quote ratios, not milliseconds.** Absolute times here move by ~20% between runs because the
  session's own tooling holds a core; two arms measured back to back do not move by more than a point
  or two.
- **Compare the arms inside one process** when a change only moves scheduling, a cache hint or an
  address. The recipe is "How to measure a switch on this machine" below, and
  [`benchmarks/README.md`](benchmarks/README.md) indexes the rest of this directory.

## Template

Run on each hardware configuration, and record the model and the thread count with the result:

```bash
vipera profile --json
vipera plan model.gguf --ctx 4096 --json
vipera verify model.gguf --ctx 4096 --json
```

Record: CPU model, physical cores and `VIPERA_THREADS`; OS and `rustc --version`; the GGUF (arch,
quant, size); the llama.cpp build when comparing; and the exact command that produced the timing. A
number without those is not reproducible and does not settle an argument.

## DRAM ceiling — i5-11300H, 4 cores / 8 threads, 2×DDR4

Measured 2026-11-13, because "the Q4_K/Q6_K loops are memory-bound" was being argued instead of
checked. Method: official STREAM 5.10 (McCalpin) compiled with `gcc -O3 -march=native -fopenmp`,
`STREAM_ARRAY_SIZE=32000000` (256 MB per array, 768 MB working set, against 8 MiB L3), `NTIMES=10`;
plus a read-only kernel over 400 MB of `uint64` with four independent accumulators (a single
accumulator measures add latency, not bandwidth — the first attempt read 30 GB/s for that reason).
Best of 10 passes, `OMP_PROC_BIND=close`, two runs per thread count.

| Kernel | Streams | 4 threads | 8 threads |
|---|---|---|---|
| Copy | 1 read, 1 write | 39.7 GB/s | 38.8 GB/s |
| Scale | 1 read, 1 write | 27.2 GB/s | 29.6 GB/s |
| Add | 2 reads, 1 write | 30.3 GB/s | 31.2 GB/s |
| Triad | 2 reads, 1 write | 29.6 GB/s | 30.5 GB/s |
| read-only | 1 read | 42.8 GB/s | 41.7 GB/s |

Reading:

- **SMT buys nothing on a bandwidth stream.** 8 threads is within noise of 4, and on Copy it is
  slightly worse. Only the core count matters here.
- **Practical read ceiling ≈ 43 GB/s** on this machine, against 51.2 GB/s theoretical for DDR4-3200
  dual channel (≈84%). Triad's 30 GB/s is lower because it mixes writes and pays read-for-ownership
  — the wrong ceiling to quote for a weight stream.
- **A decode token reads 0.980 GB**, not the 1.117 GB file size: the tensor table is 1.111 GB and
  `token_embd.weight` is 125 MB (Q4_K), which decode never touches. At the 43 GB/s read ceiling the
  floor is **22.8 ms/token ≈ 44 tok/s**.
- Against that floor: vipera at 4 threads 34.0 ms/step = **28.8 GB/s (67%)**, at 8 threads 31.3 ms =
  **31.3 GB/s (73%)**; llama.cpp 39.04 tok/s (4 threads) = **38.3 GB/s (89%)**. So the loops are
  memory-bound, but vipera is not *at* the wall. The gap to llama.cpp is 1.34x at equal thread
  count, and the levers that were tried and measured negative are listed under "Decode headroom"
  below — it is not prefetch.

## llama.cpp CPU baseline — same machine, same model

Measured 2026-11-13 to replace the second-hand `pp128 256 / tg64 34.7` in `HANDOFF.md`, which does
not reproduce here. Upstream `ggml-org/llama.cpp` build **5676dfb** (0.9.0-dev), cloned fresh; the
local `vipera.cpp` fork does not compile at 49e1485 (`k_scale_lane_idx undeclared` in
`ggml/src/ggml-cpu/arch/x86/quants.c`).

Build: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_AVX512=ON -DGGML_NATIVE=ON
-DGGML_OPENMP=ON -DGGML_VULKAN=OFF`, then `--target llama-bench`. Model:
`qwen2.5-1.5b-q4km.gguf` (1.04 GiB, Q4_K_M). Command: `llama-bench -m <model> -p 128 -n 64 -t <T>
-r 3 -ngl 0`.

| threads | pp128 (t/s) | tg64 (t/s) |
|---|---|---|
| 8 | 209.06 ± 0.62 | 37.13 ± 0.46 |
| 4 | 211.26 ± 1.05 | **39.04 ± 0.03** |

vipera on the same run of measurements: `pp128` 44.4–45.0 t/s, `tg` at ctx 32 28.1–32.0 t/s
(4 threads 34.0 ms/step, 8 threads 31.3 ms/step).

Verdict: **prefill gap 4.6x** (209.06 / 45.0), **decode gap 1.16–1.34x** depending on thread count.
At equal 4 threads llama.cpp streams 38.3 GB/s (89% of the read ceiling) against vipera's 28.8 GB/s
(67%).

**Measurement caveat:** the governor is `powersave` and cannot be changed without root, so core
frequency swings between 0.7 and 3.8 GHz; repeated runs of the same binary differ by up to 15%
(31.3–36.3 ms/step for identical settings). Single samples are not comparable — use best-of-N or
median, and compare within one session.

## Decode headroom — what does not work

Both ideas in the obvious list were already implemented and measured on this machine, and both lost:

- **Software prefetch of weight rows** (`_MM_HINT_T0`, four rows ahead): 31.5 GB/s against **40.3
  GB/s** without the hint, on a 200 MB Q4_K tensor at eight threads. The hardware streamer already
  runs ahead of the loop, and the hint evicts the activation block the kernel rereads every row.
  Removed; the comment in `quant.rs` keeps the number.
- **Two rows in flight (row blocking)**: 0% at every thread count — the loop is bound by instruction
  throughput, not load latency, so overlapping rows buys nothing. Recorded in `fill_rows_batch`.


**Grouping the projections did not help (2026-11-13).** `wq`/`wk`/`wv` share one activation, as do
`wg`/`wu`; grouping them turned seven dispatches a layer into five. Bit-identical, and 24 interleaved
32-token runs gave a paired `group / base` median of **0.999** on `compute` micros (min 0.927, max
1.097) — no effect. One dispatch is 8.4 us and the whole non-dot remainder of a step is 0.90 ms of 40,
so two dispatches a layer were never going to show. Reverted; full record in
`docs/design/continuous-batching.md`.

## Head-to-head after the activation-quantization change — 2026-11-14

llama.cpp build **9183a55**, cloned fresh and built with `cmake -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`
(the `/tmp/llamacpp` checkout used for 2026-11-13 was gone), model `qwen2.5-1.5b-q4km.gguf`, all at 8
threads. `llama-bench -m <model> -p 128 -n 64 -r 3 -t 8` and `-p 512 -n 0 -r 3`.

| | vipera | llama.cpp | gap |
|---|---|---|---|
| pp128 (t/s) | 159.6 | 199.67 ± 1.48 | 1.25x |
| pp512 (t/s) | 123.9 | 200.39 ± 2.38 | 1.62x |
| decode (ms/step) | 32.6 at ctx 32 | 28.4 at ctx 128+ (`tg64` 35.26 ± 0.08 t/s) | 1.15x |

For reference, the pre-change column of the same session: pp128 115.5, pp512 96.2, decode 34.5. The
one-pass activation quantizer moved all three; the prefill gap against llama.cpp shrank from 1.72x to
1.25x at `pp128` but stays 1.62x at `pp512`, where four 128-token chunks pay four passes over the
weights.

### The repacked 8x8 Q4_K GEMM would not help — measured

Before porting ggml's `block_q4_Kx8` GEMM into the prefill path, both kernels were timed on the shape
the ADR-001 prototype uses (8960 weight rows x 1536 cols x 128 activation rows), one thread, same
session:

| kernel | time | GFLOPS |
|---|---|---|
| ggml `ggml_gemm_q4_K_8x8_q8_K` on repacked weights (`vipera-ggml` shim) | 20.7 ms | **170.0** |
| vipera `q4k_matvec_tiled_range` (AVX-512 VNNI tiled, no repack) | 17.74 ms | **198.6** |
| vipera, same kernel, 8 threads | 4.54 ms | 776 |

The existing kernel is 17% ahead of the repacked one at equal thread count and scales to 776 GFLOPS
over eight threads, so the integration has no headroom to win: it would add a load-time repack of
~625 MB of weights and an FFI/C toolchain for at best nothing. Not integrated. The earlier "8-row
layout first" recommendation in `docs/benchmarks/beating-llamacpp.md` was based on kernel-quality
work measured on other CPUs; on this one the layout is a loss.

Where a prefill chunk's time actually goes (`VIPERA_PROFILE_STEP=1`, 128-token chunk): q4k **59%**,
q6k **17%**, attention **11%**, norm 2%, rest 11%. With Q4_K already at 59% of the chunk and 776
GFLOPS in the kernel, the remaining prefill gap is not in the Q4_K matmul.

### Q4_K block scales via F16C — evaluated, not shipped

The patch replaces the branchy `f16_to_f32` widening of `d` and `dmin` with one `vcvtph2ps` per block
in `q4k_dot`, `q4k_dot_batch`, `q4k_dot_batch_vnni` and `q4k_dot_vnni` (and, in the version measured
here, in `q4k_matvec_tiled_range` too). With the patch applied, the widened exhaustive test — which the
patch also tightens from "both are NaN" to `to_bits` equality — passes over all 65536 patterns, so the
instruction is bit-identical to the software conversion.

Paired runs, old and new binaries alternating inside one session (`win_vs_llama`,
`prefill_vs_decode`, three pairs):

| | before (ms) | after (ms) |
|---|---|---|
| pp64 chunked | 370.1 / 386.0 / 395.1 | 387.0 / 392.2 / 396.1 |
| pp128 chunked | 766.7 / 763.5 / 784.3 | 778.3 / 789.9 / 791.1 |
| pp512 chunked | 4079 / 4130 / 4177 | 4106 / 4138 / 4149 |
| decode ctx 32, ms/step | 31.9 / 32.2 / 32.6 | 32.6 / 33.1 / 32.8 |

The direction is slightly *negative* in every column, which is this machine's frequency noise rather
than a real loss, but it is certainly not a win. The compute-bound microbench (`q4k_dot_throughput`,
8960x1536 single thread, L3-resident, best-of-rounds over 20 invocations per build) agrees: 0.985 ms
before against 0.971 ms after, identical medians at 0.970.

Why it cannot matter here: the conversion is paid twice per 144-byte block, about ten instructions
each, so removing it entirely is bounded by ~1.1M instructions out of ~400M cycles in a decode pass at
eight threads (**≤0.3%**), and by ~0.02% of a prefill chunk, where the tiled kernel reads the scales
once per 32-row window instead of once per row per half. At eight threads the Q4_K kernel is
bandwidth-bound anyway (33.0 GB/s against a 42.8 GB/s read ceiling), so ALU work in the block prologue
is free.

Cost: `available()` gains an F16C requirement, so an AVX2-without-F16C CPU would lose every vector
kernel (attention, quantize, Q6_K) to save this. Not shipped; these numbers are here if a compute-bound
target (single thread, small batch) ever makes 0.3% worth the gate.

**The Q4_K kernel is not where the 65% goes (2026-11-13).** Per-shape table from
`VIPERA_PROFILE_STEP=1` (8 threads, `--ctx 128`, 20 tokens):

| shape | bytes/pass | ms/pass | GB/s |
|---|---|---|---|
| Q4_K 8960x1536 (`wg`, `wu`) | 431 MB | 13.15 | **33.0** |
| Q4_K 1536x1536 (`wq`, `wo`) | 37 MB | 3.98 | **18.7** |
| Q4_K 1536x8960 | 36 MB | 3.91 | 27.7 |
| Q4_K 256x1536 (`wk`) | 6 MB | 1.65 | **5.6** |
| Q6_K 1536x8960 (`wd`) | 355 MB | 5.51 | 28.7 |
| Q6_K 151936x1536 (`output`) | 103 MB | 5.17 | 37.1 |
| Q6_K 256x1536 (`wv`) | 6 MB | 0.59 | 7.6 |

The kernel itself is at 38.2 GB/s on a flat 63 MB buffer at 4 threads (12.1 at one), against 42.8 for
STREAM, and the shape carrying 87% of the Q4_K bytes already runs at 33.0 in the step. The loss is the
small shapes: 9% of the bytes, 25% of the Q4_K time, and it is DRAM latency rather than bandwidth
(0.22 MB = 55 pages). Running sub-0.5 MB tensors on the calling thread made it worse — q4k median
23.10 -> 23.37 ms, q6k 11.36 -> 11.80 over five runs — because one thread exposes the latency the pool
hides. Reverted.

What the thread curve says instead (decode ctx 32, same session):

| threads | ms/step | tok/s | GB/s |
|---|---|---|---|
| 1 | 97.7 | 10.2 | 10.0 |
| 2 | 56.4 | 17.7 | 17.4 |
| 4 | 34.0 | 29.4 | 28.8 |
| 8 | 31.3 | 32.0 | 31.3 |

The first thread already reaches ~10 GB/s, which is what one core gets from this memory subsystem
(STREAM read-only at 4 threads is 42.8 GB/s, i.e. 10.7 per thread). Going 1 → 4 threads scales 2.9x
instead of 4x, so the loss is in how the four streams share the memory system while also running the
per-block integer work — not in a missing prefetch and not in row latency.

### `silu_mul` was the whole "rest" bucket, and it vectorises at 2.4x — 2026-11-14

The chunk profile had an 11% `rest` bucket nobody had opened. Instrumenting it per chunk (b128,
Qwen2.5-1.5B Q4_K_M, one thread, temporary counters since reverted) put every millisecond of it
somewhere:

| part | ms per 128-token chunk | share of rest |
|---|---|---|
| `silu_mul` (28 layers x 128 x 8960) | 67–69 | 76% |
| rope + `kv.write` | ~16 | 18% |
| residual adds | 3.4 | 4% |
| bias adds | 1.7 | 2% |
| embedding gather | 0.05 | 0% |

So the bucket was one function: SwiGLU's `up * silu(gate)`, 32.1M elements per chunk, run scalar.
It is now AVX2 with `_mm256_div_ps` feeding the same exp polynomial the scalar code uses, and the
same operation order — the point was to not move the logits, so the vector path is bit-identical,
not merely close:

- `simd::tests::silu_mul_matches_the_scalar_one_bit_for_bit` compares `f32::to_bits` over -200..200
  in 1/256 steps, 200k random bit patterns, the specials (±0, denormal, ±inf, NaN) and tails that
  are not a multiple of 8.
- Microbench (`silu_mul_cost`, 11.0M elements, single thread, three runs each): scalar 22.66 / 21.44
  / 20.33 ms, AVX2 9.41 / 9.26 / 8.38 ms. **2.4x**, not 8x — 11M x 4 bytes x 2 arrays is 88 MB
  touched, so the kernel is DRAM-bound at this size, and the floor is the bytes.

End to end, measured as the share of the chunk rather than in tok/s, because this session ran at
load average 5 on 4 cores and wall clock moved ±25% between runs. The rest bucket inside a chunk,
same machine, alternating binaries:

| run | rest, scalar | rest, AVX2 |
|---|---|---|
| a | 114.31 ms (10%) | 55.79 ms (5%) |
| b | 104.34 ms (10%) | 39.02 ms (6%) |

The share is frequency-independent, so it survives the noise the absolute times do not: rest halves,
about 55 ms off a 1.0 s chunk. The `exp` helper also saturated `whole as i32` before adding the
exponent bias, so inputs beyond 2^22 wrapped to an exponent field of 0 and returned 0 instead of inf;
the vector path hit that in testing and both paths now clamp first.

### Where a prefill chunk's time goes, and what the kernel does inside it — 2026-11-14

A 128-token chunk on this machine splits as q4k 63%, q6k 17%, attention 12%, norm 2%, rest 6%. The
question that matters is whether the Q4_K phase is slow because of the kernel or because of the
bookkeeping around it, so both were timed separately with temporary counters (since reverted):

| quantity | measured |
|---|---|
| Q4_K batched products, whole call | 607 GFLOPS |
| ... the tiled kernel call alone | 786 GFLOPS |
| ... building the activation it reads | 830 ms against 6065 ms of kernel |
| the same kernel standalone, 8 threads | 776 GFLOPS |

The kernel in the model runs at the rate it runs at on its own, so there is no hidden overhead to
recover inside the Q4_K phase: 75% of a chunk is kernel work already at its measured ceiling, and the
gap to llama.cpp at the same prompt length (641 ms per chunk against 780) is not in this phase.

### Reusing one prepared activation across projections — measured, not shipped

`q`, `k` and `v` read the same normalised vector, and `gate` and `up` another, yet each product
rebuilt the activation: a copy of 128x1536 f32, its 32-value sums, and the `i8` rounding. Built once
per group instead — `prepare_batch` plus `matvec_pooled_batch_prepared` — the prefill chunk should
have dropped by the 32 ms per chunk that four saved preparations per layer come to, on a
`prepare_batch` measured at 0.29 ms per 128x1536 batch (1.5 ns/value, of which 0.08 ms is the
rounding and 0.20 ms the copy and sums).

It did not. Six alternating pairs, best of each: `pp128` 754.3 ms before, 758.1 ms after; `pp512`
4095 ms before, 4117 ms after. The medians lean the other way (782.5 -> 762.0 ms on `pp128`), which
is the size of the noise here, so the honest reading is "no measurable change" — and the isolated
microbench overstates what the copy costs inside the model, where the same buffers are recycled by
the allocator and stay in cache. Reverted: an API split with no measured payoff is not worth its
surface. What would move this properly is one allocation for the whole batch instead of 128 per
call, which the standalone number says is worth at most 20 ms of a 780 ms chunk anyway.

### Head-to-head, llama.cpp rebuilt for the same window — 2026-11-14

llama.cpp `9183a552e` rebuilt here (`GGML_NATIVE=ON`, Release) and run minutes apart from the engine,
so the two columns see the same load and the same clock. `llama-bench -r 3` for the reference,
best of two `prefill_vs_decode` runs for the engine.

| test | llama.cpp, 4 threads | vipera, 8 threads | ratio |
|---|---|---|---|
| pp128 | 199.18 ± 0.77 tok/s | 167.7 tok/s (763.2 ms) | 0.84 |
| pp512 | 194.18 ± 1.12 tok/s | 128.2 tok/s (3993 ms) | 0.66 |
| tg64 | 38.69 ± 0.07 tok/s | 30.3 tok/s (33.0 ms/step) | 0.78 |

Two things the table does not show. `llama-bench` runs a 512-token prompt as one pass, while the
engine spreads it over four 128-token chunks, which is why its `pp512` falls from 199 to 194 tok/s
and ours from 168 to 128: per 128 tokens llama needs 641 ms and the engine 763 ms at short context,
and the gap widens only because the engine's later chunks pay for a longer attention window. On
decode, llama reads the 1.117 GB of weights in 25.9 ms (43.2 GB/s, 97% of the 44.7 GB/s ceiling
measured in section 1); the engine reads them in 33.0 ms (33.8 GB/s, 76%).

### Attention in a prompt chunk: one worker per position — shipped, 2026-11-14

A chunk ran its matvecs on the pool and its attention on the caller, one entry at a time: at `pp512`
that was 66-75 ms of a 576-615 ms chunk (11-12%), and it was the part of the chunk that grew with
context. The rows of a chunk are independent, and so are the writes each produces, so the work now
goes to `min(VIPERA_THREADS, entries)` workers, each with its own scratch, in one `thread::scope` per
layer. Nothing about the arithmetic changed: every row does the same sums in the same order as a lone
`attend` call.

Measured with `prefill_vs_decode`, three runs each side, minutes apart, best of each:

| quantity | before | after |
|---|---|---|
| attention, chunk of `pp512` | 66.1 / 70.9 / 74.9 ms | 30.8 / 32.3 / 33.8 ms |
| chunk wall time | 575.8 / 596.6 / 614.9 ms | 555.7 / 574.6 / 590.6 ms |
| `pp512` | 4140 ms, 123.8 tok/s | 3188 ms, 160.6 tok/s |
| `pp128` | 757.9 ms, 168.9 tok/s | 707.0 ms, 181.0 tok/s |
| `pp64` | 375.3 ms, 170.5 tok/s | 360.0 ms, 177.8 tok/s |
| decode, ctx 32 | 31.7 ms/step | 31.5 ms/step |

The end-to-end gain at 512 is larger than the 35 ms the chunk profile gives back, which is what the
old chunk-by-chunk lines looked like under a different split; the wall-clock column is the one to
quote. Chunk speedup over token-by-token prefill went from 4.75x to 6.31x.

Bit-identity was checked against the pre-change build on a 512-token prompt: the logits hash over the
whole vocabulary is `e9bf2869cdfc7212` on both, and a three-step decode after it hashes to
`af6c31d79d0e4777` on both. The in-tree `logits_each_matches_single_passes` still passes with the
model set.

What was measured and rejected: splitting one entry's **heads** instead, which is the only way to
spread a decode step's attention, made decode at ctx 1024 slower (45.3 -> 48.7 ms/step) because each
worker decodes the same KV rows again and eight spawns per layer cost more than twelve heads divide.
Attention is therefore split by entry only, and a lone decode step keeps the serial kernel.

### One KV decode shared by the whole chunk — measured, not shipped

Every entry of a chunk decodes its own copy of the layer to `f32` (the streaming arm twice over), so
a 128-token chunk reads each row of the same history about 128 times, and the entries' histories are
prefixes of one another: one `load_layer` of the chunk's longest history could serve all of them, at
`2 * positions * width` f32 of buffer (16 MB of ceiling was chosen for a 4096-token history).

Built, hashed against the previous build (identical: `e9bf2869cdfc7212` for `pp512`, and the decode
steps after it unchanged at `af6c31d79d0e4777`), then measured: attention per chunk 30.8 / 32.3 /
33.8 ms before, 28.5 / 30.4 / 31.8 ms after, `pp512` 3188 ms against 3214-3259 ms. The saving is
about 2 ms of a 580 ms chunk, which is the noise here, and the reason is visible in the sizes: at
512 positions a layer's K/V is 256 KB per half and stays in L2, so decoding it again costs nearly
nothing. Reverted rather than shipped as a second code path with no measured payoff.

### Why the Q6_K kernel reads 34 GB/s where a plain read gets 44 — 2026-11-14

Same 191 MB Q6_K buffer, same 8 threads: `matvec` 34.4-38.0 GB/s, a byte-sum over the same buffer
44.2 GB/s. The buffer, the thread count and the read pattern are identical, so the difference is
inside the kernel. Throughput against working-set size, which separates the kernel's own ceiling
(data in L2) from the DRAM rate, `VIPERA_THREADS=1` and 8:

| Q6_K working set | 1 thread | 8 threads |
|---|---|---|
| 0.65 MB (L2) | 11.3 GB/s | 7.5 |
| 5.2 MB (L2/L3) | 10.8 | 24.9 |
| 191 MB (DRAM) | 9.6 | 38.0 |

One core tops out at ~11 GB/s whether the data is in L2 or streaming from DRAM, and the DRAM figure
is 85% of that ceiling: this kernel is bound by its own instructions per byte, not by memory. Eight
threads on four cores buy little because both siblings share the same execution ports, which is why
the 4-core model lands well under the 44 GB/s a plain read reaches. The gap to llama.cpp is therefore
a leaner 6-bit unpack, not traffic.

What was tried: the trick that worked for Q4_K, one `hsum` per row instead of one per block, with the
block scale folded into the lane scale. Per core it does help — 0.65 MB 11.3 -> 12.6 GB/s, 191 MB 9.6
-> 10.7, +11% — but in the model nothing moves: `q6k` per chunk 109.2 / 110.1 ms before and 109.2 /
113.5 after, decode 32.3 -> 33.1 ms/step. It also reorders float adds, which would have needed the
`VIPERA_EXACT_DOT` escape hatch as a second kernel. Reverted; the measurement is the result.

### A leaner Q6_K unpack: 39% faster per core, invisible end to end — 2026-11-14

The Q6_K decode kernel rebuilt its scales from scalars for every 32-value run and computed the
6-bit bias with a second `maddubs`. Both can go: build the block's eight scales once as a vector,
spread a run's pair with one `permutevar8x32_ps` and one broadcast of the activation scale, and let
`vpdpbusd` do the four products and the bias subtraction in two instructions. The integer sums and
the float associations are unchanged, so the kernel is bit-identical to the AVX2 one (checked: the
`pp512` logits hash `e9bf2869cdfc7212` and the three-step decode `05437e1e5c745384` are the same on
both builds).

Throughput of the kernel, same buffers as the section above:

| Q6_K working set | before | after |
|---|---|---|
| 0.65 MB (L2), 1 thread | 11.3 GB/s | 15.7 (+39%) |
| 5.2 MB, 1 thread | 10.8 | 15.2 |
| 191 MB (DRAM), 1 thread | 9.6 | 12.4 |
| 191 MB (DRAM), 8 threads | 38.0 | 41.8 (+10%) |

End to end: decode `q6k` 10.0-11.2 ms per step before and 10.0-11.2 after, step 33.3 / 33.2 / 34.7 ms
against 32.3 / 33.2 / 33.5, `pp512` 3161-3295 ms against 3259-3279 (prefill uses the batched Q6_K
kernel, which this did not touch). Reverted: a second kernel with no measured payoff.

The number that matters from the exercise is the gap it exposes. In a decode step the Q6_K phase reads
253 MB and takes 10.2 ms — 27 GB/s — while the same kernel over the same shapes standalone reaches
41.8 GB/s. At 41.8 that phase would be 6.1 ms, so ~4 ms of a 33 ms token is spent outside the kernel:
the per-layer Q6_K calls are 0.32 MB and 1.9 MB, and the 197 `matvec` calls of a step are where that
time goes. That is the next thing to measure, not the kernel.

### Is SMT worth it for the q4k phase? — measured, yes

Same chunk (`pp512`, third chunk), `VIPERA_THREADS` 4 against 8, two runs each:

| phase | 4 threads | 8 threads |
|---|---|---|
| q4k | 499.7 / 527.4 ms | 415.8 / 419.9 ms |
| q6k | 128.4 / 130.8 ms | 115.1 / 115.2 ms |
| attention | 41.1 / 42.0 ms | 34.9 / 35.1 ms |
| chunk | 720.3 / 751.5 ms | 621.3 / 625.3 ms |
| `pp512` | 3880 / 3936 ms (132 / 130 tok/s) | 3259 / 3279 ms (157 / 156 tok/s) |

Prefill is compute-heavy enough that the sibling threads pay for themselves (q4k 1.21x, chunk 1.19x),
so the default of using every logical CPU is right for a prompt chunk even though the same choice buys
only 9% on a decode step, where the work is bandwidth-bound.

### A polling window between pool dispatches: measured, slower — 2026-11-14

The old `pool_dispatch_cost` was measuring the wrong thing. It timed an 8-row product through
`matvec_pooled`, and `thread_budget` gives a product that small one thread, so no worker was ever
woken: what it timed was building the activation. The replacement hands the thread count straight to
`fill_batch`, builds the activation outside the loop, and inserts a spin gap between products to stand
in for the rope, norms and attention a decode step does between two of its 197 products.

What one product costs the caller then, 64 rows, 8 participants:

| work between two products | per product |
|---|---|
| none | 8.8 us |
| 30 us | 8.9 us |
| 300 us | 37.2 us |

So a dispatch whose workers have gone idle costs 28 us more than one they are still around for, and
197 of those is 5.5 ms of a 33 ms token. That is the size of the headroom behind the Q6_K kernel gap
above, and it is a wake-up cost, not arithmetic.

The obvious fix — let workers poll their queue for `VIPERA_SPIN_US` out of the last 45 us of the gap
— moved the dispatch bench the right way (40.3 us at no polling against 9.0 us with a 200 us window)
and the model the wrong way:

| `VIPERA_SPIN_US` | decode ctx32 | `pp512` |
|---|---|---|
| 0 (blocking) | 33.0 / 33.6 ms | 3142 / 3207 ms (163.0 / 159.7 tok/s) |
| 50 | 37.5 / 37.0 | 3258 / 3273 (157.1 / 156.4) |
| 200 | 40.0 / 37.8 | 3360 / 3376 (152.4 / 151.7) |
| 500 | 37.8 / 41.7 | 3476 / 3497 (147.3 / 146.4) |

At 4 threads, 0 against 200 us is inside this machine's noise (42.2 / 33.7 against 31.3 / 33.1). The
loss is contention, not the idea: eight workers polling hold their logical CPUs against the four
physical ones, and the sibling threads that made prefill 1.21x faster lose as much as the wake-up
saves. Reverted, including the window plumbing. The corrected bench stays, because a benchmark that
measures the wrong thing is worse than none.

### Correcting the last section: that 28 us is a C-state, not a wake — 2026-11-14

The dispatch bench above sleeps between products, and a benchmark that sleeps measures the laptop's
power management as much as the pool. Repeated with the gap spent spinning on the calling thread
instead, one 1536x1536 Q4_K product at 8 participants:

| work between two products | sleeping | busy |
|---|---|---|
| none | 26.9 us | 28.6 us |
| 30 us | 27.1 | 28.4 |
| 167 us | 28.5 | 28.5 |
| 300 us | 70.0 | 61.0 |

Flat to 167 us in both cases, and only past ~200 us does the cost jump — that is the core reaching a
deeper idle state, and no amount of polling in the pool fixes it. A decode step averages 167 us per
dispatch, so the 5.5 ms of "headroom" claimed above does not exist: the 28 us at a warm dispatch is
the work (1.33 MB at ~47 GB/s while the buffer is in L3), not overhead.

What that leaves is honest but dull: decode reads ~1.0 GB per token in 33 ms, 30 GB/s against the
41.8 GB/s this machine reaches on a streaming read, and the shorter products cannot saturate DRAM
however they are split — the 0.22 MB ones are 8 KB per worker at 8 participants and are already
faster that way than on one core, and the 7.7 MB ones run at 43-61 GB/s.

### Splitting small products across fewer workers: reverted

The corollary looked obvious: if a dispatch that a single core finishes in 20 us costs 28 us, do not
wake eight workers for it. `thread_budget` got a floor of 2M multiply-adds per worker, so a 1536x1536
product (2.4M) went to one participant and the 8960-column ones to six. Decode went from 33.0 / 33.6
ms to 40.5 / 40.7 ms and the q4k phase from 19.8 to 28.1-29.6 ms — worse by a third, and by more than
the wake it saved. One core reads 1.33 MB in ~120 us; eight read it in ~28 us, of which the wake is
nothing at this gap. Reverted.

(Halfway through this round the machine picked up external load — `freebuff` at ~115% CPU, load
average 3.1 — which slowed every configuration by about 20% and made the 5/6/7/8-thread decode
comparison inconclusive. The two results above were taken back to back before that drift, and the
A/B ones are within one process.)

## Re-measure 2026-11-15 — the 2026-11-14 gap does not reproduce

Same machine, same model (`qwen2.5-1.5b-instruct-q4_k_m.gguf`), 8 threads, governor still `powersave`.
This section exists because the 1.62x `pp512` gap above is quoted as the reason to widen the prefill
chunk, and on this build it is no longer there.

Commands:

```bash
VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf VIPERA_THREADS=8 \
  cargo test -p vipera-backend-cpu --release --test win_vs_llama prefill_vs_decode -- --ignored --nocapture

# CPU-only llama.cpp binary on this disk (the fork, not upstream): backend CPU, build 49e1485.
./vipera.cpp/build-clang/bin/llama-bench -m ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p 128,512 -n 64 -t 8 -r 3 -ngl 0
```

| test | vipera, run 1 | vipera, run 2 | llama.cpp fork 49e1485 (CPU) | ratio (vipera / fork) |
|---|---|---|---|---|
| pp64 | 181.9 | 196.6 | — | — |
| pp128 | 179.0 | 194.2 | 149.88 ± 1.13 | 1.19–1.30 |
| pp512 | 169.9 | 174.7 | 153.09 ± 1.98 | 1.11–1.14 |
| decode ctx 32 | 33.9 (29.5 ms) | 35.0 (28.6 ms) | 38.65 ± 0.26 (`tg64`) | 0.88–0.91 |

What this says:

- **The prefill lever is nearly exhausted at the chunk width the engine already uses.** `pp128` is one
  128-token chunk and `pp512` is four of them, so the four-pass weight traffic costs only 5-10%
  (179 → 170, 194 → 175), not the 1.6x the 2026-11-14 table implies. Widening the chunk to 512 has a
  ceiling of roughly 10% here; it is not the missing 4.6x.
- **Against upstream 9183a55** (the numbers earlier in this file, not re-runnable on this disk) the
  picture is pp128 ≈ parity (194.2 vs 199.67), `pp512` ~11% behind (174.7 vs 200.39), decode ≈ parity
  (35.0 vs 35.26). The comparison above is against the fork, which is a different binary from the
  upstream build used earlier; do not mix the two columns.
- **Decode is the remaining gap to the fork** (about 10%), and it is the kernel, not the chunking.
  See the Q6_K section above: the phase reads 27 GB/s inside the model against 41.8 GB/s standalone.

Run-to-run spread on this box is 8-15% (governor), so the two vipera columns are the honest unit of
comparison — one sample against one llama-bench line is not.

### Two Q6_K blocks per `zmm`: 22-26% at the kernel, nothing end to end — 2026-11-15

llama.cpp PR #27590 puts two K-quant blocks in the two 256-bit halves of one `zmm` (Q6_K/`Q8_K` dot
1.44x, `PP512` +16.7%, `TG` unchanged there). Built here as a prototype: block `b` in the low half and
`b + 1` in the high half, one `maddubs`/`madd` pair for both, and each block's eight lanes still go
through the old `hsum` in block order, so the result is bit-identical to `q6k_dot_q8` (checked with
`to_bits` over 12 random rows).

Kernel, `ffn_down` shape (8960 rows x 1536 cols, one thread, best of three each, same session):

| working set | 256-bit | two blocks per `zmm` |
|---|---|---|
| 10 MB | 10.82 GB/s | 13.66 (+26%) |
| 182 MB | 10.37 GB/s | 12.68 (+22%) |

End to end, `prefill_vs_decode`, three alternating pairs with the kernel behind `VIPERA_Q6K_ZMM`:

| | pair 1 | pair 2 | pair 3 | median |
|---|---|---|---|---|
| decode ms/step, off | 28.0 | 30.0 | 28.9 | 28.9 |
| decode ms/step, on | 28.6 | 28.7 | 29.4 | 28.7 |
| `pp128` tok/s, off | 190.3 | 185.3 | 183.6 | 185.3 |
| `pp128` tok/s, on | 185.6 | 187.9 | 184.7 | 185.6 |

The 0.2 ms difference is inside this machine's noise (the same configuration moves 28.0-30.0), and
`pp128` is flat because prefill uses the tiled Q6_K kernel. Why a 22% kernel does not reach the step:
in the model the Q6_K calls are 0.32 MB and 1.9 MB spread over 197 `matvec`s at eight threads, where
the phase is shared bandwidth and per-call work, not one thread's instruction count — the same reason
the `hsum`-per-row and leaner-unpack experiments above came back neutral. **Removed** (the gate and
the prototype), numbers kept here so it is not rebuilt a third time.

What would move decode instead: fewer calls or less traffic, not a narrower instruction stream. The
`BatchScratch` logits buffer (`MAX_BATCH * vocab`, 78 MB at `MAX_BATCH = 128`) is the kind of thing to
fix before another kernel is written.

## Against a clean upstream llama.cpp, e574a17 — 2026-11-15

The section above compares against `vipera.cpp/build-clang`, and that binary understates the gap:
its prefill path is 30-35% below a clean upstream build of the same date (`pp128` 147.2 against 225.3,
`pp512` 156.6 against 224.3), so a comparison against it credits the engine with prefill it does not
have. Built here instead:

```bash
# upstream, newest at the time of measuring, CPU only, no local patches
git clone --depth 400 https://github.com/ggml-org/llama.cpp llama-latest
cd llama-latest && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j8 --target llama-bench

llama-latest/build/bin/llama-bench -m ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p 128,512 -n 64 -t 8 -r 3 -ngl 0
```

Median of three paired rounds, 8 threads, same model, `e574a17` (build 466):

| workload | vipera | llama.cpp e574a17 | ratio |
|---|---|---|---|
| pp128 | 186 (182–193) | 217 (216–222) | 0.86 |
| pp512 | 171 (169–174) | 217 (216–223) | 0.79 |
| pp512, same `ubatch` 128 | 171 | 208 | 0.82 |
| decode, depth 0 | 34.3 tok/s (29.1 ms) | 37.5 (26.6 ms) | 0.91 |
| decode, depth 256 | 32.8 (30.5 ms) | 35.6 (28.1 ms) | 0.92 |
| decode, depth 1024 | 24.9 (40.1 ms) | 33.4 (29.9 ms) | 0.75 |

What the three columns separate:

- **The prefill gap is kernel work, not batch width.** Sweeping upstream's `-ub` on `pp512` gives
  207.6 (128), 209.5 (256), 215.6 (512): widening the batch buys 4%, and at the engine's own width
  of 128 it still loses 18%. Upstream repacks Q4_K at load into an interleaved blocked layout
  (`ggml/src/ggml-cpu/repack.cpp`, `tiled/tiled-kernel.cpp`) and its inner loop reads packed nibbles
  contiguously; the engine unpacks a 256-column slab per 32-row window from the GGUF layout in
  `simd::q4k_matvec_tiled_range`. The tiling itself is load-bearing here — `VIPERA_TILED=0` drops
  `pp512` from 168 to 49.7 tok/s — so the remaining 18% is the unpack and the strided row reads, not
  a missing tile.
- **Decode loses little until context arrives, then loses a quarter.** Upstream's own depth curve
  (`-d`) is 37.58 → 35.63 → 33.42 → 30.07 tok/s at depth 0/256/1024/2048, i.e. attention costs
  1.95 ms at 256 and 4.16 ms at 1024. The engine's curve over the same depths is 34.3 → 32.8 → 24.9,
  i.e. 1.4 ms at 256 and 10.8 ms at 1024 — 2.6x upstream's attention cost. `model::attention_heads`
  walks one `(head, position)` at a time: a `dot_f32`, a scalar `exp`, and an `axpy_f32` per
  position, with `PARALLEL_ATTENTION_MIN = 4096` score lanes leaving everything below ~342 positions
  on one thread. Upstream uses a flash-attention-style kernel over F16 KV.
- **The old upstream column was low.** `9183a55` at 199.67/200.39/35.26 and the fork's 149.88/153.09
  are both under a clean build of their own date: upstream `45b6604` (2026-11-06), the fork's base
  date, measures 225.3/224.3/37.8 here. So the honest summary to date is **−14% `pp128`, −21%
  `pp512`, −9% decode, −25% at 1k context**, not "parity except a 10% decode gap".

Spread on this box is 8-15% between windows (a `powersave` governor on 4 cores/8 threads, plus other
load), so single lines are not comparable across sections; the paired tables are.

### On-demand pool chunks: no change — 2026-11-15

A patch replaced the fixed row split with chunks claimed from an atomic counter, on the theory that
a decode product waits on a worker that has not left its idle state yet. Measured at the locus it
targets, `pool_dispatch_cost`, best of 200 dispatches:

| rows | fixed, 0 µs gap | on demand x4, 0 µs gap | fixed, 100 µs gap | on demand x4, 100 µs gap |
|---|---|---|---|---|
| 256 | 9.7 µs | 11.2 µs | 12.0 µs | 12.0 µs |
| 1536 | 27.1 | 27.9 | 28.3 | 29.1 |
| 8960 | 132.1 | 131.6 | 125.4 | 126.6 |

End to end, two alternating pairs: decode 28.9/28.9 ms off against 29.1/29.2 ms on. The tail the
patch removes is not there at 256 rows — the product is bound by the rows it has, not by the last
worker to wake — and the counter costs slightly more than it saves. **Reverted**; it also needed a
`#[allow(clippy::too_many_arguments)]`, which `CONSTRAINTS.md` does not allow.

### Four positions per attention value pass: bit-exact, no faster — 2026-11-15

The value pass calls `axpy_f32` once per `(head, position)`, and each call reads and writes the
`head_dim` output buffer — 512 B in and 512 B out, 344k times per token at depth 1024 (12 heads x
1024 positions x 28 layers). Grouping four positions into one pass looks like a 2.4x cut in that
traffic: one load and one store per four positions instead of four each.

Implemented as `simd::axpy4_f32`, adding the four products to the running vector one at a time in
position order (grouping the four sums first rounds 1 ULP differently and broke the two
bit-identical attention tests). A/B at depth 1024, three alternating pairs behind `VIPERA_ATTN4`:

| | pair 1 | pair 2 | pair 3 | median |
|---|---|---|---|---|
| 1024 ms/step, four axpy per position | 43.4 | 43.9 | 42.8 | 43.4 |
| 1024 ms/step, four positions per pass | 44.0 | 44.1 | 42.6 | 44.0 |

Neutral. The output buffer is 512 B and stays in L1, so moving it was never the cost: the pass is
multiply-add throughput plus the V row loads, and the same number of those is paid either way.
**Reverted.** The attention gap to upstream at this depth stays at ~2.6x (`4.16 ms` against
`10.8 ms`), and closing it needs the flash-attention shape — online softmax over a block of
positions with the scaling folded in — not a wider accumulation of the same work.

### The ggml repacked GEMM is slower than the tiled Rust kernel — 2026-11-15

ADR-001 rested on one premise: the prefill gap is *kernel quality*, and the way to buy ggml's
repacked `block_q4_Kx8` GEMM is a C++ bind. That premise had never been measured at the kernel
level — the ADR compares whole-engine tok/s. So: same shapes (8960x1536 and 1536x8960), same
batch (128 activations), same thread count (1), alternating pairs in one window, synthetic Q4_K
weights in both. ggml is the `vipera-ggml` prototype, compiled for `avx512vnni` from
`llama-latest`, calling `ggml_gemm_q4_K_8x8_q8_K` on repacked weights and `ggml_quantize_mat_q8_K_4x8`
activations; the Rust side is `simd::q4k_matvec_tiled_range` behind the new
`q4k_tiled_batch128_throughput` bench.

| Kernel, 1 thread, B=128 | 8960x1536 | 1536x8960 |
|---|---|---|
| Rust tiled, `q4k_matvec_tiled_range` | **154.1 / 152.5 / 153.7 GFLOPS** (median 153.7) | **173.3 / 174.4 / 175.6** (median 174.4) |
| ggml repacked, `ggml_gemm_q4_K_8x8_q8_K` | 125.1 / 125.5 / 124.8 (median 125.1) | 128.8 / 128.5 / 128.9 (median 128.8) |

The Rust kernel is **23% faster** on the gate shape and **35%** faster on the 35-block projection
shape. Binding ggml's GEMM would make prefill slower, not faster, before a single line of seam
code is written: the 2-4 dev-month cost stays unpaid and so does the second toolchain. **ADR-001
implementation is closed**; the policy exception (EXC-001) stays until its expiry for the Q6_K
question only.

Two things this also says:

- The 18% prefill gap to llama.cpp (171 against 209 tok/s pp512) is **not** the Q4_K GEMM kernel.
  llama.cpp's kernel is the slower one per thread, so its win comes from elsewhere in the pass —
  the streaming/residency behaviour of the real 0.98 GB weight sweep, or its threading, not the
  microkernel. That is the next hypothesis to test, and this harness is where to test it.
- Both figures are L3-resident (7.7 MB matrix), so they price the microkernel, not the stream. The
  gap between 154 GFLOPS here and the ~42 GFLOPS/thread prefill actually sustains end to end is
  the real headroom, and it lives outside the inner loop.

### Where the "27% of kernel rate" went: SMT, not the tile and not the stream — 2026-11-15

Follow-up on the last paragraph above. Three candidates, three measurements, all single thread
unless stated, on the 6-block gate shape (8960x1536).

**(a) The activation tile.** `q4k_matvec_tiled_range` builds `act`/`dxs`/`sumt` per thread:
`blocks * halves * 64 * 16 * 4` bytes = **192 KB at B=128** (not the 786 KB first reported to the
user — that multiplied by 16 twice), 1.5 MB across eight threads. Sweep (`q4k_tiled_batch_sweep`):

| B | act tile | ms | GFLOPS |
|---|---|---|---|
| 16 | 24 KB | 6.3 / 6.5 | 70.1 / 68.0 |
| 32 | 48 KB | 6.2 / 6.3 | 143.1 / 140.7 |
| 64 | 96 KB | 13.5 / 13.1 | 130.9 / 134.9 |
| 128 | 192 KB | 27.8 / 26.7 | 126.8 / 132.0 |

No residency cliff: 32 is the best point (143) and 128 costs ~8% against it, which is the reuse
trade-off, not a spill. The tile is not the limiter.

**(b) The stream.** `q4k_tiled_streaming_throughput` runs the kernel over 127 copies of the shape —
0.98 GB, the weight bytes one prefill chunk reads — B=128, one thread:
**3000 / 3044 ms, 147.0–149.2 GFLOPS, 0.32–0.33 GB/s.** The rate *rises* against the L3-resident
figure (127–132 in the same window). Streaming from RAM is free here: 256 FLOP per weight byte
means the kernel never gets near the 43 GB/s ceiling.

**(c) The pool.** End to end, pp512 chunked, alternating 4 and 8 threads:

| | run 1 | run 2 |
|---|---|---|
| 4 threads | 105.8 tok/s | 95.3 tok/s |
| 8 threads | 99.9 tok/s | 95.8 tok/s |

Four threads equal eight: on this 4C/8T part the second hyperthread of a core buys nothing in
prefill. That is the whole explanation for the 27%: one thread alone gets a full core (126–154
GFLOPS), and a prefill thread shares its core's execution units, so per-thread rate halves. Eight
threads x ~66 GFLOPS is the four-core ceiling, not headroom.

**Consequence.** Prefill on this CPU is within a few percent of the core-limited ceiling, so the
18% gap to llama.cpp cannot be closed by making this kernel wider: at 209 tok/s and ~3.05 GFLOP
per token upstream would need ~159 GFLOPS/core, which is already at our single-thread kernel rate.
Any further prefill work has to reduce work or threads, not tighten the loop. Tile split, cache
blocking and more threads are all closed by the above.

### Prompt chunk size 128 against 64 and 32: within noise — 2026-11-15

With the tile sweep showing 143 GFLOPS at B=32 against 127-132 at B=128, the prompt chunk was made
a named constant and swept end to end (pp512, chunked, 4 threads):

| chunk | samples (tok/s) |
|---|---|
| 128 | 134.5, 146.3 |
| 64 | 138.8, 138.6 |
| 32 | 129.1 |

The kernel's B=32 advantage does not survive the pass: four times as many chunks means four times
the weight sweeps, and that costs more than the tile saves. 64 against 128 flips order between
runs, so both stay inside this machine's noise. **Constant reverted**, chunking stays at
`MAX_BATCH`.

### Pool on-demand chunking: neutral at the kernel and neutral end to end — 2026-11-15

Hypothesis: `fill_batch` hands each thread a fixed slice per product, so a product finishes only
when the slowest-woken worker finishes its slice; llama.cpp instead takes chunks from an atomic
counter on demand. The patch under test (`0001-pool-on-demand-chunks.patch`, from the external
contributor) adds `Split::Dynamic { per_participant }`, defaults off, `VIPERA_POOL_CHUNKS=N`
enables N chunks per participant for B<8 only, so the tiled prefill path is untouched. Every
product stayed bit-identical to the sequential kernel in their suite (288 tests, 2 more than
baseline).

Dispatch cost, 8 participants, same binary (`quant::tests::pool_dispatch_cost`):

| rows | between products | fixed | on demand x4 |
|---|---|---|---|
| 256 | 0 us | 9.4 us | 11.2 us |
| 256 | 100 us | 13.2 us | 13.1 us |
| 1536 | 0 us | 29.2 us | 29.3 us |
| 1536 | 100 us | 31.4 us | 31.5 us |
| 8960 | 0 us | 136.2 us | 134.5 us |
| 8960 | 100 us | 144.9 us | 134.8 us |

Decode end to end, pp32 decode, 8 threads, interleaved in one binary:

| run | fixed | on demand x4 |
|---|---|---|
| 1 (cold weights) | 45.0 ms/step | 33.6 ms/step |
| 2 | 33.8 ms/step | 33.7 ms/step |

The two columns are the same to within 0.3%, and the kernel-level comparison is flat at every
shape. The wake-up tail this was meant to remove is not visible on this part, which matches the
earlier finding that the sleep-window cost attributed to wake-ups was really a core C-state. **The
patch is not carried**: revert, as the contributor asked for a neutral result.

Worth recording for whoever picks this up: a 256-row product is 9.4 us at eight threads, so the
twelve small products of a decode step (wk/wv 256 rows, wq/wo 1536 rows, x28 layers) are ~0.3 ms
of a 33.8 ms step. Even a perfect tail fix there is worth under 1%, which is why the end-to-end
number cannot move: the 25% of Q4_K time in those products is real work, not dispatch overhead.

### Prefix cache: 2791 ms of prompt work becomes 61 ms, with the same tokens — 2026-11-16

The contributor's `0001-prefix-cache-park-and-copy.patch` keeps a finished request's prompt KV
parked (LRU, at most 32 prompts, given back to any allocation that needs the blocks) and copies
the longest shared prompt prefix into a later request instead of computing it. Off by default
(`VIPERA_PREFIX_CACHE=1` or `Engine::set_prefix_cache(true)`), because it changes what a request
computes.

Measured on Qwen2.5-1.5B Q4_K_M with `tests/prefix_cache.rs`: a 480-token shared prompt plus four
tokens that differ, one generated token, same engine handling both requests.

| | ms | processed | reused |
|---|---|---|---|
| cache off | 2790.7 | 484 | 0 |
| cache on | 61.0 | 4 | 480 |

46x on that request, and the twelve-token continuation of the warm request is identical to the
cold one (`[11, 220, 16, 15, ...]`), so the copied KV is the KV the prompt pass would have written
there — including the case where the donor's prefix was computed in 128-token chunks and the copy
covers all but four tokens.

Every workspace test passes (runtime engine tests 48 -> 64). Not a kernel change, so nothing here
touches the prefill ceiling above: it removes prompt work rather than speeding it up, which is why
it is the only change in this batch that moves a number by more than a few percent.

### Prefix cache through the HTTP server, eight at once — 2026-11-16

`vipera serve` (its `max_batch` is 4), one seed request to park a 480-token prompt, then eight
concurrent requests that share it and differ only in their four-token question. Non-streaming, so
`time_total` is the whole request; two rounds, cache off against on, with `--ctx 8192`:

| round | cache off | cache on |
|---|---|---|
| 1 | 33.83 s wall (per request 33.78) | 1.15 s wall (per request 0.95-1.11) |
| 2 | 34.81 s wall (per request 34.52-34.77) | 1.17 s wall (per request 0.88-1.12) |

29x on the wall clock of the burst, and the text of all eight answers is identical between the two
modes.

The cache does not shrink KV demand, which is the operational catch: at `--ctx 2048` the same
burst took 16.70 s against 7.34 s with the cache on (2.3x instead of 29x), because eight sequences
of ~500 tokens no longer fit the 2048-token budget, so later requests find the donor evicted and
fall back to computing. That is arithmetic, not a measured decomposition — but it is enough to say
the knob belongs with a context large enough to hold the concurrent prompts plus one parked copy.

### A copied prefix is the same on Q8_0 KV as on F16 — 2026-11-16

`prefix_cache_gives_the_same_tokens_for_every_kv_type`: 480 shared tokens plus a four-token tail,
twelve generated tokens, cache off against cache on, per KV type. F16 and Q8_0 both reuse 480 and
return identical tokens. Q4_0 is not carried by this backend at all (`KvLayout::of` returns `None`,
so planning refuses it before any of this matters), so it is reported as skipped rather than
passing.

### Attention decode at depth 1024: read f16 in place, then split the heads — 2026-11-16

Baseline in this window: 44.4 ms/step, attention+KV 12.8-14.1 ms (the quiet-machine figure is
10.8, so only ratios below are worth reading).

**In place, no row buffer.** `simd::dot_f16_f32` / `axpy_f16_f32` convert inside the loop with
F16C and keep `dot_f32_avx2`'s two-accumulator reduction, so they are bit-identical to decoding
the row and calling the f32 kernels (new test `f16_rows_give_the_same_bits_as_decoded_ones` checks
`.to_bits()` over 1..256 values). Alone it is neutral: 13.4 against 13.2 ms, alternating, same
binary. The row buffer was never the cost — 176 MFLOP per token of score and weighted sum is.

**Heads across the pool.** That arithmetic picture is what makes the head split worth retrying:
this split was measured and reverted earlier (45.3 -> 48.7 ms) because with a decoded row every
worker decoded all 1024 rows again. Reading rows in place, a worker repeats only arithmetic, which
is the thing there is too much of. One entry with `>= PARALLEL_ATTENTION_MIN` score lanes now
sends its 12 heads to `min(threads, heads)` workers, each with its own scratch and output slice:

| | attention+KV | step |
|---|---|---|
| split off | 15.0, 15.2, 15.0, 15.4, 14.5, 13.0, 11.5 (median 15.0) | median 46.9 |
| split on | 11.0, 11.2, 11.3, 12.0, 13.0, 13.1, 15.0 (median 12.0) | median 44.6 |

20% off attention and ~5% off the step in that window, with the same bits: a head's arithmetic and
its order do not depend on which worker holds it, and every attention test still compares bit for
bit against the decoded oracle. Measured again later on cores the agent was not using, it is 29%
off attention and 4% off the step — see the section below, which supersedes these samples.

**Not the 5 ms target.** Attention is arithmetic-bound at roughly 13-15 GFLOPS of one core here;
the next step is a blocked, head-major score pass (Q x K^T as a small GEMM with the positions
vectorized instead of one 128-wide dot per position), which needs FMA and therefore gives up bit
equality with the buffered oracle. That is a tolerance policy for attention, not a kernel detail,
so it is not taken here.

Knobs for a same-binary A/B: `VIPERA_INPLACE_KV=0` and `VIPERA_ATTN_HEAD_SPLIT=0`.

### Where a depth-1024 attention layer's time is not: the arithmetic — 2026-11-16

`simd::tests::attention_phase_cost` times one layer's two phases at their own shape (1024
positions x 12 heads x 128 values, f16 rows, L1-hot): **score 0.13 ms, weighted sum 0.11 ms,
0.24 ms together**. Twenty-eight layers of that is 6.7 ms, against the 12-15 ms an attention step
actually takes at ctx 1024 in the same window.

So a blocked, position-vectorised score pass (Q x K^T as a small GEMM) cannot pay for itself here:
it would shorten 0.13 ms of a layer and add a transpose of every K row, and the transpose has to be
built from a row layout where consecutive positions are `bytes_per_token` apart (28 KB on this
model). The same arithmetic says the head split was the right lever, and that the remaining cost is
the strided walk, not the loop.

### The head split and the prefetch, measured on cores the agent is not using — 2026-11-16

The earlier samples for both were taken while this session's own harness held a core (100% of one
logical CPU), so load average never fell below ~1.6 and the samples spanned 6.2-15.0 ms. Rather than
wait for an idle machine that cannot exist while the agent runs, the benchmark was pinned to the
three physical cores the agent is not on (`taskset -c 2-7`, six threads) and the two switches were
alternated in one binary, three rounds each:

| | attention+KV | median | step | median |
|---|---|---|---|---|
| head split off | 10.6, 11.6, 12.1 | 11.6 | 44.0, 45.6, 45.6 | 45.6 |
| head split on | 7.7, 8.2, 9.1 | 8.2 | 42.2, 43.7, 43.7 | 43.7 |
| prefetch off | 9.4, 9.5, 9.3 | 9.4 | 44.1, 43.6, 44.2 | 44.1 |
| prefetch on | 8.6, 9.0, 10.5 | 9.0 | 43.5, 43.8, 45.2 | 43.8 |

**Head split: confirmed, and better than the loaded window suggested** — 29% off attention (11.6
to 8.2 ms) and 4% off the step. Kept.

**Prefetch: 4% off attention, nothing at the step boundary** (44.1 against 43.8 is inside the
spread), and one of the three rounds regressed. That does not clear the bar the other changes were
held to, so it is **reverted**: `simd::prefetch` and the two calls in `model.rs` are gone, and the
`VIPERA_ATTN_PREFETCH` switch with them. `simd::prefetch_ahead` stays where it was, in the matvec
kernels, where it has a measurement behind it.

Method note for the next measurement here: `taskset -c 2-7` plus `VIPERA_THREADS=6` is what makes
this repeatable. Without it the same binaries spread by 2x.

### Prefix cache: the KV headroom a burst needs — 2026-11-16

Same burst as above (seed parks a 480-token prompt, eight concurrent requests share it), one round
per context size:

| `--ctx` | cache off | cache on | gain |
|---|---|---|---|
| 2048 | 16.70 s | 7.34 s | 2.3x |
| 4096 | 37.05 s | 4.64 s | 7.9x |
| 8192 | 33.83 s | 1.15 s | 29x |
| 16384 | 37.17 s | 1.25 s | 28x |

At 4096 the eight requests split into two groups (1.39 s and 4.5 s): the first four copy the
parked prompt, then the cache fills and the rest recompute. The threshold is arithmetic — the
budget has to hold the concurrent prompted sequences plus the parked copy, so ~8 x 500 tokens plus
one prompt needs 8192 and has nothing to spare at 4096. Past that the gain is flat, so 8192 is the
size to recommend for this burst shape.

### A transposed K layout with a block GEMM score pass: 4.6x slower — 2026-11-16

The remaining cost of a depth-1024 attention step is the strided walk over KV rows (each K row is
512 bytes, consecutive positions are `bytes_per_token` = 28 KB apart), so the proposal was to store
K as `[dim][position]` per block, which makes positions contiguous for a dimension and lets the
score pass become Q x K^T with the reduction vertical instead of one horizontal sum per position.

Before changing the cache format, the pair was prototyped in `simd::tests::transposed_k_scores`
over the real shape (1024 positions x 12 heads x 128 dims, f16 rows), on a separate branch:

| score pass | ms |
|---|---|
| per-position dots (today) | 0.14 |
| block GEMM of 64 positions, transposed K given | 0.37 (0.67 with the transpose counted) |
| transpose alone | 0.27-0.33 |

Both produce the same scores on the same data (largest gap 0.000). The block GEMM is 2.6x slower
even when the transpose is free, and 4.6x slower when it is not. The reason is loads: the GEMM
reads one 32-byte vector of `kt` per FMA, and re-reads the whole block once per head, while the
per-position dot reads a contiguous 128-value row that is already in L1 and reduces it with two
accumulators.

**Verdict: the format change is not carried.** It would cost a change to the KV format, the write
path, `load_layer`, prefix-cache copies and every KV test, to make the already-cheap phase slower.
The score pass is 0.14 ms of a 12 ms attention step, so even making it free would leave ~4 ms of
the step untouched: whatever is left is the walk and the weighted sum, not the scoring.

The prototype stays as an ignored bench, so the next person does not have to rebuild it to find
the same answer.

### Layer-major KV inside a block: 27% off the walk at depth 1024 — 2026-11-16

The previous entry established that what is left of a depth-1024 attention step is the walk, not the
arithmetic. The walk is strided because a token owns every layer: K rows are 512 bytes, and
consecutive positions are `bytes_per_token` = 28 KB apart, so one layer of a 16-token block is 16
rows scattered across 448 KB.

Before touching the format, the same bytes were re-ordered *inside* a block — layer-major, every
layer's rows for the whole block together — behind `VIPERA_KV_LAYER_MAJOR=1`. The block table, the
block size and the row encoding are unchanged; only the addresses move, so the two arms read the
same values in the same order and must compute the same bits.

Measured by `kv_row_order_at_1024` (28 layers x 1024 positions, f16, `tokens_per_block` = 16, best
of five per arm, both arms in one process, pinned to cores 2-7 with `VIPERA_THREADS=6`):

| run | token-major | layer-major | gain |
|---|---|---|---|
| 1 | 15.22 ms | 11.10 ms | -27% |
| 2 | 15.62 ms | 11.12 ms | -29% |
| 3 | 15.21 ms | 12.50 ms | -18% |

Per layer that is 0.543 ms to 0.396 ms. The bench asserts the two arms' outputs are equal bit for
bit before it prints anything, so the gain is locality and nothing else.

Exactness through the model, not just the bench: `vipera generate` with the switch off and on gave
identical token ids on (a) a 920-token prompt at f16 and (b) a 1534-token prompt at `q8_0` (34-byte
rows, six blocks), greedy, same seed.

**Verdict: worth keeping, not yet worth making default.** A 1024-deep step spends ~8-12 ms in
attention and KV, so 27% of that is 2-3 ms of a ~44 ms step (5-7%). The end-to-end decode bench
cannot resolve 3 ms on this machine — its noise floor is +-10 ms even at 32 positions, where the two
orders read exactly the same bytes — so the win is measured where it happens and nowhere else. The
switch stays, off by default, until a quieter machine can confirm the step.

### Settled in one process: head split 36%, prefetch nothing — 2026-11-16

Every earlier sample of these two switches compared two processes, and this machine is never idle:
`freebuff` holds a full logical CPU for the length of a session, and the load average has not gone
below 3.5. Comparing two runs therefore compared two different amounts of interference, which is
why the same switch came out as 20% once and 29% later.

The fix is not a quieter machine but a tighter experiment: `attention_combinations` builds the
KV once, then times both arms **in the same process on the same bytes**, four sweeps per arm with
the sweep direction alternating so no arm holds a systematically warmer position, best sample per
arm. It asserts all arms produce identical output before printing, so the flags can only be moving
time. Pinned to cores 2-7 with `VIPERA_THREADS=6`.

| run | split off | split on | gain |
|---|---|---|---|
| 1 | 14.93 ms | 9.76 ms | -35% |
| 2 | 14.48 ms | 9.52 ms | -34% |
| 3 | 14.91 ms | 9.58 ms | -36% |

28 layers x 1024 positions, f16, `tokens_per_block` = 16. Per layer 0.533 ms to 0.342 ms. The
arithmetic does not depend on which worker holds a head, and the output comparison in the bench says
so.

**Head split: 34-36%, not 20%.** Two independent methods now agree (this one and the cross-process
`taskset` run that gave 29%), so the number to quote is roughly a third of the attention walk at
depth 1024, worth ~3 ms of a ~44 ms step.

**Prefetch: nothing worth keeping.** Restored behind the switch and measured in the same harness:

| | prefetch off | prefetch on |
|---|---|---|
| split off | 14.44 ms | 13.81 ms (-4%) |
| split on | 9.47 ms | 9.59 ms (+1%) |

It helps the single-threaded walk by 4% and is a wash once the heads are split — which is the
configuration that ships. So it is not the 10% it was once credited with: that number came from
comparing runs, not from the switch. It is removed again, and the bench that measured it stays as
an ignored test so the next attempt starts from the same table.

Method note for the next switch: compare arms inside one process wherever the switch only changes
scheduling or adds a hint. Cross-process A/B on this machine cannot resolve anything smaller than
~10 ms.

### How to measure a switch on this machine

Every failure in this ledger's history of the two attention switches had the same shape: two runs
compared, one arm landing in a quieter moment than the other, and a switch credited with whatever
the machine happened to be doing. The machine is never quiet here (`freebuff` holds a core for the
length of a session, load average 3.5-5.8), so the recipe is:

1. **Build the input once**, in the test, and use the same bytes for every arm. No model load, no
   prefill: the thing being changed is the loop, not the loader.
2. **Time every arm in one process**, in sweeps, with the sweep direction alternating, and keep the
   best sample per arm. Two sweeps warm, two measured is enough.
3. **Assert the arms compute the same output before printing**, and make it fail the test if they
   do not. A switch that changes a bit is not a performance switch, and a bench that cannot tell is
   measuring the wrong thing.
4. **Pin** (`taskset -c 2-7 VIPERA_THREADS=6`) so the pool gets whole physical cores.
5. Quote the **ratio**, not the milliseconds: the same 2x2 table moves by ~20% run to run in
   absolute terms and by 1-2 points in ratio.

`attention_combinations` is that recipe for both attention switches at once, and
`kv_row_order_at_1024` is the same idea for one.

### The two attention switches combined, at 1024 and 4096 — 2026-11-16

`attention_combinations` crosses the KV row order with the head split, both arms in one process,
28 layers, f16, `tokens_per_block` = 16. Two runs, best per arm:

| depth | arm | run 1 | run 2 | vs baseline |
|---|---|---|---|---|
| 1024 | token-major, split off | 15.31 ms | 14.20 ms | — |
| 1024 | token-major, split on | 9.80 ms | 9.19 ms | -35% |
| 1024 | layer-major, split off | 10.56 ms | 10.23 ms | -28% |
| 1024 | layer-major, split on | 7.70 ms | 7.81 ms | -46% |
| 4096 | token-major, split off | 68.66 ms | 68.53 ms | — |
| 4096 | token-major, split on | 48.91 ms | 49.10 ms | -29% |
| 4096 | layer-major, split off | 45.13 ms | 44.03 ms | -34% |
| 4096 | layer-major, split on | 28.16 ms | 26.68 ms | -59% |

**The head split holds at 4096** (-28%, against -35% at 1024): splitting twelve heads over six
workers buys the same quarter-to-third of the walk at four times the depth, so it is a property of
the work distribution and not of the buffer fitting in L3.

**The two gains roughly add, and at 4096 they add better than the parts predict.** At 1024 the
combined arm is -46%, where independent gains of 35% and 30% would give -54%: slightly under, which
is what two changes competing for the same memory bandwidth looks like. At 4096 it is -59% against
-54% predicted: the layer-major walk turns 4096 scattered rows into 256 contiguous 8 KB runs, and
once the rows arrive in runs the six workers stop fighting over the same lines, so the split gets
cheaper at the same time.

For a 1024-deep step that is ~7-8 ms off the ~44 ms step; for 4096 it is ~40 ms off a step that was
spending most of its time in the walk. Two switches, neither changing a bit.

### Layer-major becomes the default — 2026-11-16

Two things stood between the prototype and making it the default: it had to not cost anything on
the write path, and the combined gain had to hold deep enough to matter.

**Write path, prefill batch 128** (`kv_write_batch_at_128`, 128 tokens x 28 layers x 10 passes per
sample, best of two samples, both orders in one process):

| run | token-major | layer-major |
|---|---|---|
| 1 | 2.71 ms | 2.70 ms |
| 2 | 2.71 ms | 2.65 ms |
| 3 | 2.66 ms | 2.78 ms |

Equal within noise: 0.75 us per token-layer either way. A 128-token chunk writes its KV in 2.7 ms
against a prefill of roughly a second, so the write path is 0.3% either way and the row order does
not move it.

**Combined gain at three depths** (`attention_combinations`, two runs, best per arm):

| depth | token/split-off | token/split-on | layer/split-off | layer/split-on |
|---|---|---|---|---|
| 1024 | 14.19 / 14.48 ms | -30% | -27% | **-45%** |
| 4096 | 69.95 / 75.00 ms | -29% | -38% | **-58%** |
| 8192 | 148.60 / 151.23 ms | -37% | -36% | **-60%** |

The two switches stay additive as depth grows (independent 37% and 36% predict -60%), so there is
no saturation point to work around: the walk is still the cost at 8192 and both changes still cut
it.

**Exactness with the switch unset**, i.e. the new default: `vipera generate` reproduces the same
token ids recorded for layer-major earlier — 920-token prompt at f16, 1534-token prompt at `q8_0`,
greedy, same seeds.

So `VIPERA_KV_LAYER_MAJOR` now defaults on, and `=0` keeps the token-major walk for an A/B. The
format is unchanged: same blocks, same block size, same row encoding, same block table, same prefix
cache copies (whole-block `copy_within`, order-agnostic).

### A Zig kernel for the Q4_K prefill matmul: 1.5%, inside noise — 2026-11-12

Prefill is the only compute-bound loop left in the backend, so the experiment replaced the Rust
tiled Q4_K matmul with a Zig kernel, compiled by `build.rs` when a Zig compiler is on `PATH`,
entered only after an AVX2/FMA check at run time, with the Rust tiled path as the fallback.

Qwen2.5-1.5B Q4_K_M, 457-token prompt, greedy, four runs alternating:

| arm | prefill |
|---|---|
| Rust tiled | 6692 / 6759 ms |
| Zig kernel | 6564 / 6689 ms |

~1.5%, which is inside run-to-run noise. Token ids identical. The same source at `x86-64-v4`
(512-bit) measured 14% slower, consistent with this machine's earlier finding that wider vectors
lose to downclocking.

The arithmetic says why there was nothing to win: 1.42 TFLOP over ~5.6 s is ~250 GFLOPS against
this i5's AVX2 peak of ~282, so the kernel was already at ~89% of the ceiling and the headroom it
was chasing was ~1%. What is left is not the kernel but the 197 `matvec` calls a step makes — the
dispatch and C-state sections above.

**Not carried, and not pushed as a PR.** Three reasons: it is neutral by its own measurement and
this repository reverts neutral work; it cannot be built or tested on this machine (no `zig`), and
`ci.yml` does not install Zig either, so the kernel would be compiled by nobody and tested
nowhere; and dead code behind a `cfg` no build sets is exactly the kind of thing that rots.

The numbers are recorded from commit `673da58` on the dropped branch `feat/zig-prefill`, so the
next person does not spend a day rebuilding it to learn the same thing.
### Per-layer attention head split: workers capped at physical cores — 2026-11-17

The split spawns and joins its workers once per layer, 28 times a token. A patch proposed capping
that count, so the spawn was measured first (`scoped_spawn_cost`, 300 scopes per sample, best of
seven, empty worker bodies):

| workers | us/scope | x28 layers |
|---|---|---|
| 1 | 16.5 | 0.46 ms |
| 2 | 25.1 | 0.70 ms |
| 3 | 31.2 | 0.87 ms |
| 4 | 37.7 | 1.06 ms |
| 6 | 52.9 | 1.48 ms |
| 8 | 68.2 | 1.91 ms |

So 8 workers pay 1.91 ms a token in spawn alone at ctx 1024, against a ~32 ms decode step. That is
real, but smaller than the arithmetic the split takes off. The arms below are `attention_combinations`
on the 4-core/8-thread i5-11300H: layer-major, f16, 28 layers, `tokens_per_block` 16, every arm in
one process on the same bytes, unpinned at the default 8 logical threads, best per arm. Both columns
are the same five sweeps, with both arms present in each.

| depth | 8 workers | 4 workers (physical cores) | change |
|---|---|---|---|
| 1024 | 7.74-9.16 ms | 7.55-7.73 ms | -2% to -16%, 4 wins all five |
| 4096 | 27.10-32.81 ms | 25.49-27.80 ms | -3% to -20%, 4 wins all five |
| 8192 | 52.68-58.68 ms | 52.40-56.28 ms | -9% to +6%, mixed (3 of 5) |

Reading:

- **The mechanism is SMT contention, not spawn.** The spawn delta between 4 and 8 workers is 0.85 ms
  a token; the difference at 4096 is 1.5-6.7 ms. Four workers put one thread on each physical core;
  eight put siblings on the same cores, and this walk is compute-bound, which is the opposite of the
  memory-bound matvecs where 8 threads measured better than 4.
- **3 workers are within noise of 4**, sometimes a point or two ahead because the session's own
  tooling holds a core. The rule is physical cores, not a constant fitted to this laptop.
- **`VIPERA_THREADS=6` is capped to 4 too, and measured equal or better than 6** in the same harness
  (1024: 6.78-6.95 vs 7.01-7.10 ms; 4096: 25.88-28.20 vs 27.13-28.29; 8192 mixed). An explicit
  `VIPERA_ATTN_WORKERS=N` still raises the cap for a measurement.

The output is unchanged by construction and the harness asserts it: the arms differ only in which
worker computes which head.

**End to end the cap does not resolve.** `decode_attention_scaling` at ctx 1024, three alternating
runs per arm across two processes (default 4 workers vs `VIPERA_ATTN_WORKERS=8`, same binary):
41.7 / 41.4 / 42.2 ms per step against 41.5 / 42.1 / 43.1 ms, medians 41.7 and 42.1, spread ±1 ms.
`decode_arm_switch` then timed both arms on the same hundred decode steps in one process, five
times: arm medians 43.55 vs 43.75, 43.69 vs 43.27, 43.59 vs 44.19, 44.19 vs 44.17 and 43.83 vs 44.14
ms; paired per-round medians +0.40, -0.24, +0.12, +0.11 and -0.30 ms, eight workers slower in 54, 46,
51, 51 and 49 of 100 rounds; best steps 39.9-40.9 ms an arm. At ctx 1024 the cap is therefore neutral end to end to within ±0.4 ms on a 43 ms
step — the 0.2-1 ms the attention walk gains is below what this machine resolves, so the cap's case
rests on the walk and on the spawn accounting above, not on a decode number. The in-process harness
is the ignored `decode_arm_switch` lib bench; it forces both arms on the same steps and needs
`VIPERA_TEST_MODEL`.

**Reusing one worker team across the 28 layers is not expressible safely here.** Such a team would
outlive a layer, so it would have to hold the KV buffer for the whole pass, while `kv.write` needs
it `&mut` between layers; `pool.rs` gets away with the same shape only because the weights it
borrows are immutable for the pass. Copying the KV rows a worker needs per layer costs far more
than the 1.91 ms of spawns, and `unsafe` is forbidden outside `simd.rs`.

**So the calling thread takes one head range.** The split used to spawn `workers` threads and leave
the caller waiting; it now spawns `workers - 1` and computes the last part itself, so the same cores
are busy with one fewer thread created and joined per layer (6.5 us a layer at four workers, 0.18 ms
a token, from the spawn table). `attention_combinations`, layer-major, f16, 28 layers, one arm per
build, three runs of each binary alternating on the same machine:

| depth | before | after | change |
|---|---|---|---|
| 1024 | 7.69 ms | 6.97 ms | -9%, ahead in 3/3 pairs |
| 4096 | 25.42 ms | 24.81 ms | -2.4%, ahead in 3/3 pairs |
| 8192 | 53.84 ms | 51.23 ms | -4.8%, ahead in 2/3 pairs |

Medians of three runs. A single sweep spreads ±5%, so only the direction is worth quoting: every
depth moved the same way, by about the size of the spawn the change removes. The output is still
bit-identical, and the harness asserts it: the split decides which thread computes a head, never
what a head computes.

**This is the only thread creation left in a decode token.** The six matvecs a layer runs and the
output projection share one `Pool` scoped to the whole pass, created once and reused across the 28
layers; `quant::matvec`'s own scoped spawn is not on this path. What remains is the attention split:
28 scopes a token, 31-38 us each from the spawn table above, so 0.87-1.06 ms of a 43 ms step.

A base-plus-offset KV layout does not remove it. Tried: the compiler refuses the shape whatever the
addressing looks like, because the obstacle is a lifetime and not an offset — a read view of the
storage held by a pass-long worker is borrowed for the whole scope, and the next layer's `kv.write`
needs the same storage `&mut`:

```text
error[E0502]: cannot borrow `storage` as mutable because it is also borrowed as immutable
   |     let read = KvRead { storage: &storage };
   |                                      -------- immutable borrow occurs here
   |     scope.spawn(move || { let _ = read.storage.len(); });
   |     storage[0] = 1;
   |     ^^^^^^^ mutable borrow occurs here
```

The safe shapes are a lock shared with the team (`RwLock` around the storage, one lock pair a layer,
and the buffer would have to stop arriving as the runtime's `&mut [u8]`) or running the layer loop
inside the team so it owns the storage. Both can buy at most the 0.87 ms of spawn, because the split
itself is worth much more and stays either way: layer-major, split off against four workers, three
runs, 10.3 -> 7.2 ms at 1024, 43.6 -> 27.3 at 4096 and 86.0 -> 51.1 at 8192. Two percent of a step is
not worth moving KV ownership out of the runtime, so the spawn stays per layer.

**At long history the cap is not a measured win.** The same sweep at 4096 and 8192, two runs, four
workers against eight: 27.17 vs 25.83 and 27.46 vs 25.68 ms at 4096 (eight ahead by 5-6%), 52.13 vs
51.07 and 50.11 vs 52.13 ms at 8192 (the sign flips between runs). Only 1024 reproduces the cap's
advantage across runs, so at longer contexts the case for four workers is the oversubscription one —
half the spawns and no sibling threads fighting for the same cores on a machine that is never idle —
not a measured gain.

### Attention split bookkeeping off the heap — 2026-11-17

The single-entry (decode) and small multi-entry head/entry splits built `Vec`s for slots, handles and
results once per layer/call. Both now keep that bookkeeping in fixed stack arrays when the split fits
(≤ 16 workers/parts), so the hot decode path allocates no per-layer split vectors. `clear_forced_
switches` also now resets the forced worker cap, which a bench could otherwise leak across
measurements in one process.

Measured on the same busy 4-core/8-thread machine, layer-major split, 4 workers,
`attention_combinations`: before 7.33 / 26.12 / 54.72 ms at 1024 / 4096 / 8192 (one run); after
7.51-8.00 / 27.03-28.15 / 57.25-61.96 ms across three runs. The spread is larger than the change and
dominated by machine load, so no speedup is claimed — this is a structural cleanup of per-layer
churn, not a measured win. Correctness is unchanged (the same tests and the attention harness still
match bit for bit).

### KV-chunk parallel decode — 2026-11-17

Single-entry (decode) attention can also be parallelized by splitting the KV history into contiguous
position chunks. Each worker walks its chunk once: the K row yields every head's score, then the V row
is folded into a running `(max, sum_exp, weighted_v)` with an online-softmax update (one `exp` per head
per position, and a rescale of the accumulator only when a score raises the running maximum). The
caller merges the partials in increasing chunk order and takes the last chunk itself, so `workers - 1`
threads are spawned — the same cost model as the head split (~sub-ms per 28-layer token). The ranged
row-walk means each worker reads only its own chunk.

Unlike the head split, the merge groups floating-point sums differently from the serial kernel, so the
result is not bit-identical (~1e-7 max-abs on the 12q/2kv/hd128 decode shape) while mathematically
equivalent. That is the same trade the tiled prefill path makes, so the KV chunk is on by default and
`VIPERA_ATTN_KV_SPLIT=0` restores the bit-exact serial/head-split path; exact-dot mode always skips it
and the bit-exact streaming/buffered contracts are unchanged.

Integrated component measurement, one layer (f16 KV, layer-major, 4 workers, i5-11300H 4C/8T, busy),
median of 30 rounds, three runs, vs the serial single-entry path:

| positions | serial | KV-chunk ×4 | speedup | max-abs diff |
|---:|---:|---:|---:|---:|
| 1024 | 0.36 ms | 0.18 ms | ~2.0× | 8.9e-8 |
| 4096 | 1.41 ms | 0.55 ms | ~2.6× | 8.4e-8 |
| 8192 | 2.96 ms | 1.02 ms | ~2.9× | 7.6e-8 |

An earlier two-pass variant — a max pass over K, then a weighted-V pass over a per-worker
`heads × chunk_positions` score buffer — measured 0.25 / 0.48 / 0.93 ms on the same shape (one run
each): clearly slower at 1024, faster at 4096/8192. The fused walk ships anyway because it removes the
score buffer (`heads × chunk_positions × 4` bytes per worker, ~98 KB at 8192 with four workers) and
because it is the variant every end-to-end number below was taken with.

End-to-end decode on Qwen2.5-1.5B Q4_K_M (greedy, `decode_kv_split_switch`, 60 paired rounds with the
arm order alternating every round, 4 workers, `first_position` growing from ctx):

| ctx | head split | KV-chunk | delta | paired median | KV-chunk slower in |
|---:|---:|---:|---:|---:|---:|
| 1024 | 43.99 ms/step | 41.33 ms/step | −6.0% | −2.74 ms | 13/60 rounds |
| 4096 | 64.88 ms/step | 54.26 ms/step | −16.4% | −11.59 ms | 0/60 rounds |

It wins end-to-end at both depths, which is why it is on by default. The split stays inside one
`pool`-free scope per call; a team that lives across layers is still blocked on the KV storage
ownership, as recorded above.

The numerics are pinned by `kv_chunk_split_matches_the_serial_kernel_within_rounding`: the chunked
arm must stay within 1e-5 of the serial kernel on the decode shape, and a single worker must fall
back to the serial kernel bit-for-bit.

### Row-walk prefetch in the KV chunk — tried and reverted — 2026-11-17

The chunk walk reads two interleaved streams per position: the K row, then the V row
`layer_bytes * tokens_per_block` away. The block table restarts both streams at every block
boundary, so the walk was given explicit `prefetch_ahead` hints for the rows three positions ahead
(two cache lines each, K and V). Measured as an in-process A/B — a `PREFETCH_ROWS` toggle read by the
walk, arm order alternating, 40 paired rounds per depth, KV-chunk arm, 4 workers:

| positions | no prefetch | prefetch | delta | prefetch faster in |
|---:|---:|---:|---:|---:|
| 4096 | 0.59–0.62 ms | 0.60–0.68 ms | +2.4 … +9.5% | 16–17 / 40 rounds |
| 8192 | 1.28 ms | 1.34–1.38 ms | +4.9 … +7.1% | 17–19 / 40 rounds |

The hints cost more than they hide: at these depths a worker's stream is a few hundred kilobytes, the
hardware prefetcher already runs ahead of the dot product, and the extra address arithmetic sits on
the critical path. Reverted — no prefetch code shipped.

### A worker team across the 28 layers — still blocked — 2026-11-17

The KV chunk spawns `workers - 1` threads per attention call, once per layer and token. Spawn and join
was measured at 31–38 µs per four-thread scope, so all 28 layers cost ~0.9–1.1 ms per token: 2.4% of
the 41.33 ms/step at ctx 1024 and 1.5% of 54.26 ms/step at 4096 (numbers above). A team that lives for
the whole pass would remove that, and it stays blocked on ownership, not on effort: `thread::scope`
cannot hand a worker a `&[u8]` that lives for the pass while the layer loop needs `&mut` on the same
buffer for the next layer's KV append (E0502), and `unsafe` is forbidden outside `simd.rs`. Unblocking
it means a read-only KV view separate from the write handle, with the buffer owned by the pass and
handed to workers through an `RwLock` (readers during a layer, the writer between layers) or copied
per layer — a storage redesign that the ~2% ceiling does not pay for today.

Keeping a team hot across layers (the "single spawn scope budget" this work asked for) was measured
and does not pay. A spin-wait handoff to three workers already in `thread::scope` is 0.2 µs, which
would drop the 28-scope cost from ~1.06 ms to ~0.01 ms, but those three cores then compete with the
matvec pool between attention calls: on the `8960 × 1536` Q4_K shape (`thread_budget` 8, 4C/8T,
release, 20 calls) a parked baseline ran 0.16 ms/call while the same call with three cores spinning
ran 0.26 ms/call, +61%. Parking the team between layers instead of spinning removes the contention but
costs roughly what a scope does (~36 µs per dispatch in `pool_dispatch_cost`), so it cannot beat the
37.9 µs scope it would replace. The ownership blocker is unchanged, and the contention/cost numbers
close the remaining gap: there is no safe, non-regressing way to keep attention workers alive across
the 28 layers on this machine.

### Prefill at 1024 and 4096 prompt tokens — 2026-11-17

Measured on the same rig (i5-11300H 4C/8T, release, `~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`,
f16 KV) with `VIPERA_PROFILE_STEP=1`. The engine prefills in 128-token chunks (`ceil(n/128)` passes);
the profiler labels the first chunk of each run `chunk` and the rest `decode`, but every pass in this
measurement is a prompt-chunk forward (attention grows with position). Best of 3 rounds at 1024, 2 at
4096:

| prefill | time | throughput | chunks |
|---|---|---|---|
| 1024 | 7276.5 ms | 140.7 tok/s | 8 × 128 |
| 4096 | 50534.4 ms | 81.1 tok/s | 32 × 128 |

Phase split per 128-token chunk (ms and share of that chunk):

| chunk (position) | q4k | q6k | attention | norm | rest | total |
|---|---|---|---|---|---|---|
| first (0–127) | 482 ms (67%) | 128 ms (18%) | 42 ms (6%) | 14 ms (2%) | 52 ms (7%) | 718 ms |
| mid 1k (~896–1023) | 507 ms (54%) | 136 ms (14%) | 230 ms (24%) | 15 ms (2%) | 54 ms (6%) | 941 ms |
| late 4k (~3968–4095) | 527 ms (36%) | 138 ms (10%) | 710 ms (49%) | 16 ms (1%) | 56 ms (4%) | 1447 ms |

**Bottleneck.** The matmul work per chunk is flat with prompt length (q4k+q6k ~610–665 ms, the weight
read done once per 128 tokens) while attention grows with the history it scans: 42 ms (6%) at the start,
~230 ms (24%) by the end of 1k, and ~710 ms (49%) by the end of 4k, overtaking q4k as the largest phase.
Prefill throughput therefore falls with prompt length (141 → 81 tok/s from 1k to 4k) because total
attention scales as O(n²) over the prompt. The next lever at 4k prefill is attention, not the matmuls:
cutting matmul cost helps every chunk equally, but the late chunks that dominate a 4k prompt are now
attention-bound (and the decode-time KV-chunk kernel is the one that would have to carry a batched
prefill form).

### Head-to-head against the llama.cpp fork, same model and threads — 2026-11-18

Asked to settle "which one wins" with a run rather than a reading. Both engines: Qwen2.5-1.5B-Instruct
Q4_K_M (`~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`), 8 threads, `powersave` governor, two
passes interleaved (llama.cpp, vipera, llama.cpp, vipera) because the frequency noise on this machine
is larger than several of the differences below.

- llama.cpp: `vipera.cpp/build-clang/bin/llama-bench -p 64,128,512 -n 64,128 -t 8 -r 3`, build
  49e1485 (24), CPU backend, AVX-512 + VNNI.
- vipera: `cargo test -p vipera-backend-cpu --release --test win_vs_llama -- --ignored --nocapture
  prefill_vs_decode` with `VIPERA_THREADS=8`; `pp` is the chunked `run_positions` path, decode is a
  step at ctx 32 (KV pre-filled), which matches the average depth of a 64-token `tg` run. Both use
  an f16 KV cache, and neither count includes tokenizer or sampler time.

| test | pass | llama.cpp tok/s | vipera tok/s | vipera / llama.cpp |
|---|---|---:|---:|---:|
| pp64 | 1 | 93.3 ± 6.3 | 153.3 | 1.64x |
| pp64 | 2 | 128.9 ± 1.4 | 187.5 | 1.45x |
| pp128 | 1 | 114.0 ± 9.4 | 185.6 | 1.63x |
| pp128 | 2 | 140.3 ± 4.0 | 183.3 | 1.31x |
| pp512 | 1 | 147.3 ± 4.6 | 167.7 | 1.14x |
| pp512 | 2 | 141.8 ± 1.5 | 166.8 | 1.18x |
| tg64/tg128 | 1 | 36.6 / 31.9 | 33.7 | 0.92x / 1.06x |
| tg64/tg128 | 2 | 38.2 / 38.7 | 34.0 | 0.89x / 0.88x |

**Result.** Prefill goes to vipera in both passes at every prompt length: the widest margin is the
shortest prompt (1.45–1.64x at 64 tokens) and it narrows to 1.14–1.18x at 512, consistent with
llama.cpp amortising its per-call overhead better but not closing the kernel gap. Decode is a loss or
a wash: vipera is 8–12% behind `tg64`, and against `tg128` it is 6% ahead in one pass and 12% behind in
the other — the pass-to-pass spread (llama's own `tg128` moved 31.9 → 38.7 tok/s) is as large as the
difference, so the honest reading is "llama.cpp decodes slightly faster, within a wide noise band".

This confirms the 2026-11-15 finding on a second day: the engine is not 4.6x behind; it leads prefill
and trails decode by roughly 10%.

### Long-prompt prefill against llama.cpp: vipera wins 1k, loses 4k — 2026-11-18

Same rig, model, 8 threads and interleaved passes as the head-to-head above; `llama-bench -p
1024,4096 -t 8 -r 3` against a new ignored test `prefill_long_prompt` in
`crates/vipera-backend-cpu/tests/win_vs_llama.rs` (batch path only; `timed_best` does one warm-up and
returns the fastest round, so the KV is already allotted).

| prefill | pass | llama.cpp tok/s | vipera tok/s | vipera / llama.cpp |
|---|---|---:|---:|---:|
| 1024 | 1 | 142.7 ± 0.6 | 153.3 | 1.07x |
| 1024 | 2 | 138.8 ± 0.2 | 147.1 | 1.06x |
| 4096 | 1 | 118.6 ± 0.3 | 87.8 | 0.74x |
| 4096 | 2 | 116.5 ± 0.2 | 85.2 | 0.73x |

llama.cpp loses only 18% from 1k to 4k (142.7 → 118.6); vipera loses 40% (153.3 → 87.8). The crossover
is between 1k and 4k, and it lines up with the phase split measured 2026-11-17: vipera's attention is
6% of the first chunk, 24% by the end of 1k, 49% by the end of 4k, while its matmul cost per chunk is
flat. llama.cpp's prompt processing does not degrade like that, which points at the batched attention
path (the decode-time KV-chunk kernel has no batched prefill form yet) rather than at the Q4_K/Q6_K
matvecs — those are the same at 4k as at 1k and are where vipera's 1k lead comes from.

Consequence for the "which wins" question: vipera leads prefill up to roughly 1k tokens and loses it
at 4k; decode is llama.cpp by ~10%. Attention at depth is the single biggest gap in the engine.

### Lean pass: 77.8 MB less scratch, 59 s → 9.5 s builds — 2026-11-18

Two findings from `REVIEW_vipera_rs_2026-11-15.md` (5 and 10) were still open; both are closed here.

**Scratch.** `BatchScratch` allocated `MAX_BATCH * vocab` logits rows at load — 77.8 MB for Qwen2.5
(128 x 151,936 x 4 B) — while chunked prefill already asks `run_batch` for the last row only
(`only_last`). Only `logits_batch` (serving) and `logits_range` (speculative verify) want every row,
so that buffer now starts empty and grows on first use. Verified: `logits_each_matches_single_passes`
and the new `a_batch_that_splits_across_kernel_widths_matches_every_sequence_run_alone` pass, and
`vipera verify` reports the same measured host bytes (2,081,792,000) because the untouched
`vec![0.0; ...]` was never faulted in on the single-sequence path — the saving is address space
(and RSS if a batch call ever grows it), not resident pages of a plain generate run.

**Build.** `run_batch` instantiated `run_batch_n` for all 128 widths. The widths are now 1..=8 (a
decode batch is never larger here) plus the multiples of 8 up to `MAX_BATCH`, and any other size runs
as `width + remainder` (remainder < 8, one extra weight read of at most seven tokens). `run_batch_n`
instantiations: 128 → 23.

| | before | after |
|---|---:|---:|
| `cargo build -p vipera-backend-cpu --release`, after touching `model.rs` | 59.37 s | 9.52 s |
| `libvipera_backend_cpu-*.rlib` | 25,284,014 B | 5,421,870 B |

Prefill and decode throughput are unmoved, as expected: every measured length splits into supported
widths (64 = 64, 512 = 4x128). `prefill_vs_decode` after the change: pp64 188.4, pp128 188.1,
pp512 167.6 tok/s, decode ctx 32 33.9 tok/s, against 187.5 / 183.3 / 166.8 / 34.0 before.
The new split test asserts a batch of 10 (8 + 2) is bit-identical to ten single-sequence steps.

**Decode, tried and rejected.** Two hypotheses for the ~10% decode gap against llama.cpp, both
measured and both dropped:

- *Small shapes are dispatch-bound.* Profile shows `Q4_K 1536x1536` at 21.5 GB/s and
  `Q4_K 256x1536` at 5.7 GB/s, 16% of a step. Making the worker count proportional to work
  (`thread_budget` returning `product / PARALLEL_MIN_PRODUCT`) sent the 256-row shape to one worker
  and left it at 39 us a call (was 36), i.e. it is small, not sync-bound. Decode: 33.8 / 34.3 tok/s
  against 33.7-34.0 unchanged. Reverted.
- *Huge pages.* Not a lever here: `/sys/kernel/mm/transparent_hugepage/enabled` is `always`, so the
  weights already sit in huge pages.

Where a decode step actually goes at ctx 32 (8 threads, 29.5 ms): q4k 64%, q6k 33%, attention 0%,
norm 0.5%, rest 2%. The per-shape table gives 35.6 GB/s for the dominant `8960x1536` Q4_K read and
36.4 GB/s for the Q6_K output projection, against the 41.7 GB/s read ceiling. llama.cpp's 38.3 GB/s
sits between the two. So the remaining gap is the streaming efficiency of the Q4_K/Q6_K row reads,
not dispatch, prefetch, blocking or scale conversion — all of which are already recorded above as
measured negatives.

### An attention crew that lives across the pass — measured, then reverted — 2026-11-18

The 2026-11-17 entry above left one shape of the idea open: keep attention workers alive for the whole
forward pass, with the KV buffer behind an `RwLock` (readers during a layer, writer between layers) so
the layer loop can still append. It was built and measured (`Crew`, `CrewJob`, `CrewReply`,
`DecodeAttention`, gated on `VIPERA_ATTN_CREW`), and then removed again: the numbers below say it does
not pay, and the storage redesign it needs (KV behind a lock, `KvGeometry`, `attend_read`) is not worth
carrying for a switch that never wins. The measurements stay here so the idea is not re-litigated.

Three tests pinned it while it existed: a crew of any size computes what one thread computes, a crew
reads what the pass wrote between two layers, and a layer that fails leaves the worker usable. The
first two failed on first landing at 450 positions, because the reference was `attend_read`, which on
this tree takes the approximate KV-chunk path once the history passes `STREAMING_MIN_POSITIONS`
(`kv_split` is on by default), while the crew runs the exact per-head kernel. Comparing against
`attend_entry` over all heads — the same arithmetic the crew splits — made them bit-identical, which
is what the test meant by "what one thread computes".

Kernel, `VIPERA_THREADS=8`, release, one layer, 12 heads, 64 head_dim:

| positions | one thread | per-layer threads | crew | crew / per-layer |
|---:|---:|---:|---:|---:|
| 512 | 0.169 ms | 0.118 ms | 0.091 ms | **1.30x** |
| 1024 | 0.368 ms | 0.182 ms | 0.191 ms | 0.96x |
| 2048 | 0.712 ms | 0.444 ms | 0.364 ms | **1.22x** |
| 4096 | 1.397 ms | 0.559 ms | 0.679 ms | 0.82x |

End to end (`what_a_decode_step_spends_on_attention`, two interleaved rounds each, ms/step):

| ctx | crew off | crew on |
|---:|---:|---:|
| 32 | 34.0, 30.1 | 31.9, 29.5 |
| 256 | 32.1, 32.3 | 31.5, 33.0 |
| 1024 | 35.4, 34.5 | 35.5, 36.9 |

Verdict: **the crew moves the kernel but not the step.** At ctx 32 attention is 0% of a step, so
nothing there can show; at 1024 the crew costs 1–2 ms instead of saving any, and the two rounds on
each side overlap. This is the same conclusion as the 2026-11-17 entry, now measured through the
storage redesign it named as the unblocker: a live team removes the ~1 ms of scopes, but attention is
1.4 ms of a 35 ms step at ctx 1024, and the workers that stay parked contend with the matvec pool —
the 2% ceiling was real. Output is byte-identical on a 12-token greedy run (`"The capital of France
is"` → ` Paris. The capital of Italy is Rome. The capital of`) with the switch on and off.

One numeric side effect of that design, noted because it would have mattered had the crew won: with
the crew on, attention no longer takes the approximate KV-chunk path even when `VIPERA_ATTN_KV_SPLIT`
is set, so the step was slightly *more* exact than the default, not merely faster — a second reason not
to make it the default path.

### Where a long context actually goes — 2026-11-18

Asked to find an algorithm that keeps tok/s up as the context grows, so the first step was measuring
the curve instead of assuming it is KV bandwidth. `decode_attention_scaling` extended to 2048 and 4096
and to both KV layouts (`VIPERA_THREADS=8`, release, Qwen2.5-1.5B Q4_K_M, f16 vs q8_0 KV, best of 3
steps per length, `attention+KV` is the step minus the ctx-32 step):

| ctx | F16 ms/step | F16 tok/s | F16 attention | Q8_0 ms/step | Q8_0 tok/s | Q8_0 attention |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 27.9 | 35.8 | 0.0 ms | 29.7 | 33.6 | 0.0 ms |
| 256 | 31.5 | 31.7 | 3.6 ms (11%) | 31.7 | 31.5 | 2.0 ms (6%) |
| 1024 | 34.1 | 29.3 | 6.2 ms (18%) | 34.3 | 29.1 | 4.6 ms (13%) |
| 2048 | 36.6 | 27.3 | 8.6 ms (24%) | 38.1 | 26.2 | 8.4 ms (22%) |
| 4096 | 44.1 | 22.7 | 16.2 ms (37%) | 48.5 | 20.6 | 18.8 ms (39%) |

Two readings that change the plan:

- **Quantising the KV cache does not buy throughput.** It halves the bytes and loses end to end
  (4096: 48.5 vs 44.1 ms), because the attention pass is not bandwidth-bound at these lengths: 117.4 MB
  of K and V are read per token at 4096, which is **2.8 ms at the 41.7 GB/s ceiling**, against the
  **16.2 ms** measured. q8_0 keeps its place as the lever that fits more context in RAM, not as a
  speed lever.
- **Attention is bound by kernel efficiency, by a large factor.** 16.2 ms per token at 4096 is ~5.8x
  its traffic floor and ~5% of the machine's f32 FMA peak (0.35 GFLOP of score+value work in 16.2 ms
  is 21.7 GFLOP/s against ~380 GFLOP/s peak). The streaming kernel scores one 64-wide dot per (head,
  position) and walks the history twice, so it runs latency, not arithmetic.

The algorithm with headroom is therefore the one that raises arithmetic intensity and reads KV once:
**tiled attention with an online softmax** (flash-attention's shape, scalar/AVX2, no `unsafe`). Score a
tile of keys against all of a layer's heads while the tile sits in L1, keep the running max and
normaliser per head, and fold the value accumulation into the same walk instead of a second pass. At
decode the tile is the key dimension: 64 K rows (16 KB, L1-resident) scored against all 12 heads into a
196 KB score tile that is consumed by the fused softmax/value step while it is still in L2, instead of
196 KB per head written and re-read. At prefill the same tile is scored against a 128-query block,
which is where the 4096-token prefill loses to llama.cpp today (attention 49% of the late chunk).

Expected ceiling, from the measured split rather than a guess: attention at 4096 would have to fall
from 16.2 ms to ~4-5 ms to reach 2-3x, which puts the step at ~32 ms (31 tok/s) instead of 44.1 ms
(22.7). Below ~2 ms the weights dominate and the step stops caring about context.

Rejected before writing code, for the record: KV quantisation as a speed lever (measured above);
attention sinks / sliding windows / eviction (they change the tokens, and they only pay once the kernel
is efficient — at 16.2 ms the kernel is the problem, not the window); more threads (the split already
caps at physical cores and the 2026-11-17 entries measured the contention).

### A single-pass online-softmax decode kernel — tried and reverted — 2026-11-18

The entry above named "read KV once with a running softmax" as the lever. It was built
(`attention_online_heads`: 64-position tiles of K/V read in place from an f16 cache, scored against
every head, folded into a running max, normaliser and weighted V sum; `VIPERA_ATTN_ONLINE`), checked
against the two-pass kernel at 1, 37, 64, 65, 300 and 1024 positions (max deviation 8.2e-8), and
measured. It does not pay:

| positions | two pass ms/layer | online ms/layer | ratio |
|---:|---:|---:|---:|
| 512 | 0.13 | 0.12 | 1.08x |
| 1024 | 0.27 | 0.25 | 1.05x |
| 2048 | 0.51 | 0.48 | 1.07x |
| 4096 | 0.96 | 1.01 | 0.95x |

End to end (`decode_attention_scaling`, two runs interleaved, ms/step): 4096 44.4 offline vs 46.0
online; 1024 42.0 vs 36.1, which is the same run-to-run spread the whole file warns about, not a win.
Reverted; the kernel and its switch are gone.

**What this rules out, and the correction.** The second walk over the history is not what costs: fusing
it away moves nothing, so the score pass is essentially the whole 16.2 ms at 4096. That pass spends
~20 ns per (head, position), about four times the 64-wide f16 dot it performs, which puts the cost in
per-score overhead (reduction and dispatch), not in traffic and not in the extra pass. A tiled kernel
only helps if the tile is blocked over *queries* as well — 128 prefill queries against one K tile, where
the K rows are loaded once for 128 reductions — which is the prefill shape and not the decode shape. So
for decode, the remaining lever is a blocked score pass over several heads at once, and the 2-3x this
section hoped for is not available from fusing the value pass.

### Query-blocked prefill attention — tried and reverted — 2026-11-18

A prefill chunk is `B` consecutive positions of one sequence, so every entry scans almost the same
keys, and `attend_entry` walks those rows once per entry. The block kernel scored a 64-row key tile
against all `B` entries (8 at a time) and every head from one visit, with a running softmax per entry,
so each key slice fed eight dots instead of one. It was checked against the per-entry kernel at every
entry of a run (max deviation 1e-4 at 300-307 positions) and measured on the 1024/4096 prompt
(`prefill_long_prompt`, 8 threads, two interleaved passes each):

| prefill | block off | block on |
|---:|---:|---:|
| 1024 | 143.9 / 133.5 tok/s | 132.2 / 130.6 tok/s |
| 4096 | 77.3 / 75.8 tok/s | 77.7 / 77.0 tok/s |

No change at 4096 and slightly worse at 1024: the reuse saves bookkeeping, but the work that decides
the time is inside the kernel, where nothing is reused. Reverted.

**Why, with the split measured.** [`attention_cost_per_position`] separates the score dots from the
rest at the same shape and count: at 1024 positions the two-pass attention is 0.55 ms and the dots
alone 0.20 ms, so **36% is dot, ~64% is exp, max and the V axpy** — the value pass and the softmax,
which every query needs for its own row and which no tiling removes. So for prefill at 4k, exact
attention is priced by its arithmetic, not by its addressing, and the remaining lever is a wider
kernel (AVX-512 f32/f16 dot and axpy) rather than a better loop shape. That lever changes the
reduction order: the tests that pin the f16 dot to the decode-then-dot oracle bit for bit would have to
become tolerance checks, so it is a separate decision, not a cleanup.

### How much context fits in RAM, and what it costs — 2026-11-18

Qwen2.5-1.5B Q4_K_M declares `context_limit: 32768`, so on this model the context ceiling is the
model's training length, not memory. Per position the cache holds `2 * layers * kv_heads * head_dim`
values (28 x 2 x 2 x 128 = 14,336), stored as f16 (2 B) or q8_0 (34 B per 32 values, and 256 is a
multiple of 32):

| ctx | f16 KV | q8_0 KV | host total, f16 | host total, q8_0 |
|---:|---:|---:|---:|---:|
| 4096 | 117.4 MB | 62.4 MB | 1.25 GB | 1.19 GB |
| 8192 | 234.9 MB | 124.8 MB | — | — |
| 16384 | 469.8 MB | 249.6 MB | — | — |
| 32768 | 939.5 MB | 499.1 MB | 2.09 GB | 1.64 GB |

Measured, not planned: `vipera verify --ctx 32768 --kv-type <t> --cpu` reports `kv_cache_bytes`
939,524,096 (f16) and 499,122,176 (q8_0), `predicted_host_bytes` 2,050,894,336 / 1,610,492,416 against
`measured_host_bytes` 2,081,525,760 / 1,641,201,664 (+1.5% / +1.9%). Weights are 1,111,370,240 B.
On this machine (15.8 GB, ~10.9 GB available) a q8_0 instance at the full 32,768 context is 1.6 GB, so
about six fit before anything swaps; f16 fits about five.

RAM does not become the limit at this model's size: 10 GB of q8_0 cache is 690k positions, twenty times
the model's maximum. It becomes one for larger models and long contexts, and q8_0's value is exactly
halving that term — not speed, as the decode numbers below show.

**The tok/s cost of depth** (`decode_attention_scaling`, 8 threads, ms/step and tok/s, best of 3;
`attention+KV` is the step minus the ctx-32 step):

| ctx | q8_0 ms/step | q8_0 tok/s | q8_0 attention | f16 ms/step | f16 tok/s |
|---:|---:|---:|---:|---:|---:|
| 32 | 38.5 | 26.0 | 0.0 | 42.2 | 23.7 |
| 1024 | 45.7 | 21.9 | 7.3 ms | 44.7 | 22.4 |
| 2048 | 52.8 | 18.9 | 14.4 ms | 50.7 | 19.7 |
| 4096 | 61.5 | 16.3 | 23.1 ms | 60.6 | 16.5 |
| 8192 | 82.6 | **12.1** | 44.1 ms | 81.7 | 12.2 |
| 16384 | *~127* | *~7.9* | *~88 ms* | — | — |
| 32768 | *~215* | *~4.7* | *~176 ms* | — | — |

Up to 8192 is measured; 16384 and 32768 are extrapolated from the measured 5.5 ms per 1024 positions
(prefilling a 32k prompt at this engine's prefill speed takes close to an hour, so it was not run).
q8_0 is 0.1 tok/s *slower* than f16 at 8192 and its attention is 4.6 ms more expensive: the cache is
half the bytes but each row is dequantised, so on this kernel the trade is memory for a little speed,
never speed.

**Correction to the entry above.** That entry quoted "58.7 MB of K and V per token at 4096, which is
1.4 ms at the 41.7 GB/s ceiling" -- it counted one 128-value K row per layer instead of K and V of a
256-value key/value width. The correct figure is 117.4 MB and 2.8 ms, so attention at 4096 is 5.8x its
traffic floor rather than 11x. The conclusion does not change (16.2 ms is still far above any traffic
floor, and the score pass is still the cost), but the ratio was wrong and is fixed here.

### A sliding window makes large-context decode flat — 2026-11-18

Every exact kernel tried above (online-softmax fusion, query blocking) left the step cost growing with
context, because the arithmetic per token is O(positions) and nothing loop-shaped removes it. So the
lever that actually keeps tok/s up at depth is to stop reading the middle of the history: attention
over the first `sinks` tokens (StreamingLLM's stabiliser) plus the last `window` positions, one softmax
over that union, one weighted sum of V. `VIPERA_ATTN_WINDOW=<positions>`, `VIPERA_ATTN_SINKS=<n>`
(default 4), off unless set.

Measured with `decode_cost_at_depth_without_prefill`: the KV cache is sized for the context and left at
zero, which is a legal f16 row, so a decode step can be timed at 32k without paying for a 32k prefill
(that prefill takes close to an hour). 8 threads, ms/step and tok/s:

| ctx | full history | window 4096 | window 2048 | window 1024 |
|---:|---:|---:|---:|---:|
| 4096 | 55.8 ms (17.9) | 63.7 ms (15.7) | 54.0 ms (18.5) | 46.5 ms (21.5) |
| 8192 | 73.7 ms (13.6) | 66.0 ms (15.2) | 51.7 ms (19.3) | 48.4 ms (20.7) |
| 16384 | 104.5 ms (9.6) | 65.2 ms (15.3) | 50.4 ms (19.8) | 46.5 ms (21.5) |
| 32768 | 183.7 ms (5.4) | 64.6 ms (15.5) | 52.0 ms (19.2) | 44.6 ms (22.4) |

At the model's full 32,768-token context that is **5.4 to 19.2 tok/s** depending on the window, and the
step stops caring about depth: window 2048 ranges 50.4-54.0 ms from 4k to 32k, against 55.8-183.7 ms
for the full history. At 4096 or less the window is not applied at all (nothing to drop), which is why
the 4096 row is flat and slightly worse for 4096: the window path does the same work with one more
control branch.

The kernel is `attention_window_heads`: it collects the kept positions' scores side by side, runs one
softmax, then one weighted sum of V — exactly full causal attention over a history whose middle was
dropped, and pinned by `the_window_kernel_matches_attention_over_the_positions_it_keeps` against a
reference that sums the same positions by hand (1e-5). One integration detail that was needed for it to
bite at all: the decode path at depth takes the KV-chunk split, which scans the whole history, so that
branch stands aside when a window is set and the head split runs instead.

**This is lossy and opt-in.** The model cannot see the middle of its history: a task that needs a fact
from the start of a long prompt needs `sinks` large enough to hold it, or the window large enough to
cover it. 2048-4096 is the range where the step is flat and most of a conversation is still visible;
the default stays the full history.

### The window kernel was splitting the work the wrong way — 2026-11-18

With `VIPERA_ATTN_WINDOW=2048` at 32k the step was 44.4 ms, and the profile put **13.4 ms of it
(28%) in attention** — for a history the window had already cut to 2048 positions, while the
full-history chunked kernel spent 63.2 ms on 16384 positions, i.e. 3.9 µs/position against the
window kernel's 6.1. Same work, same KV, 1.6x worse per position, so the kernel was the problem.

`attention_window_heads` partitioned by head: eight workers, each walking every kept row, so the four
query heads that share a KV head read the same K row four times over (2 KB of reads per 512-byte
row), and the scores went into a `heads x kept` array that was then walked twice, once for K and
once for V. The chunked full-history kernel instead partitions positions, reads each row once for
every head, and folds K and V into a running `(max, sum, weighted_v)` in one traversal.

`attention_window_chunked_heads` is that shape with the window's kept ranges: a chunk is a slice of
the kept-index space `[0, sinks + window)`, which may step over the dropped middle, and the workers
merge in increasing chunk order. Measured at 32k, 8 threads, ms/step and tok/s:

| window | two-pass window kernel | chunked window kernel | attention ms |
|---|---|---|---|
| 1024 | 42.8 (23.4) | 42.8 (23.4) | 7.45 -> 5.36 |
| 2048 | 44.4 (22.5) | **40.0 (25.0)** | 13.42 -> 9.63 |
| 4096 | 57.0 (17.5) | **47.9 (20.9)** | 26.24 -> 19.83 |

11% on the step at 2048 and 16% at 4096 — the wider the window the more KV reads the head split was
duplicating, which is why 4096 gains most. 1024's step did not move because attention there is only
5.4 ms of a 43 ms step already dominated by the weight stream (21.0 ms of q4k plus 11.9 of q6k,
against a 26 ms floor for 1.117 GB at the 43 GB/s read ceiling).

`exact_dot` mode keeps the two-pass kernel, which is the summation-order reference; the chunked
kernel groups the floating-point sums by chunk and differs by rounding (~1e-7). Both are checked
against a hand-summed reference over the kept positions at 1, 3 and 8 workers
(`the_window_kernel_matches_attention_over_the_positions_it_keeps`, 1e-5 for the two-pass kernel and
1e-4 for the chunked one), and greedy output on a short prompt is unchanged by the switch.

What this leaves: at 32k with a 2048 window the step is 40 ms, of which 32.5 ms is the weight stream
at 34 GB/s — 79% of the 43 GB/s read ceiling. Attention is 9.6 ms; the rest is under a millisecond.
So on this hardware the remaining large-context lever is bytes per token (or tokens per weight pass,
which speculative decoding already does), not the context length.

### A ring cache: a windowed 32k run holds 61 MB of KV, not 896 MB — 2026-11-18

The window made the *step* flat but not the *cache*: `KvConfig` still sized the KV by
`ctx x bytes_per_token`, so `--ctx 32768` reserved 939,524,096 bytes of f16 KV while the kernel could
only ever read the sinks and the last 2048 positions. Two thirds of that is reserve that no step can
touch, and on a 16 GB box it is most of what stands between a 32k context and a second model.

`KvConfig::ring` keeps a pinned block for the sinks and a ring of
`ceil((window + PREFILL_HEADROOM + tpb) / tpb) + 1` blocks for the window. The block table does the
rest: a slot that has been overwritten points its entry at the same physical block as the position
that overwrote it, so **no kernel or `KvRead` change was needed** — the window kernel already only
reads `[0, sinks)` and `[last - window, last)`, and those slots are never the ones a write is about
to reuse. `KvConfig::for_run` chooses the ring only when `ctx > window + sinks + PREFILL_HEADROOM`;
a small context would otherwise pay for a ring that holds everything anyway.

`vipera verify --ctx 32768 --cpu`, same model, window off and on:

| | kv blocks | kv_cache_bytes | predicted host | measured host |
|---|---|---|---|---|
| full history | 2048 | 939,524,096 | 2,050,894,336 | 2,081,591,296 (+1.50%) |
| `VIPERA_ATTN_WINDOW=2048` | 139 | **63,766,528** | 1,174,792,704 | 1,205,768,192 (+2.64%) |

14.7x less KV, and `plan` accounts for it without a second formula (`PlanRequest.window` feeds
`ModelShape::window_kv_bytes`, deliberately an upper bound: it assumes a 16-token block and adds two
blocks of slack).

What it costs at the step, 32k, `VIPERA_ATTN_WINDOW=2048`, 8 threads, best of 3, both tables in the
same run (`decode_cost_at_depth_with_ring_cache`):

| table | KV held | ms/step | tok/s |
|---|---|---|---|
| linear (2049 blocks) | 896.4 MB | 34.8 | 28.8 |
| ring (139 blocks) | **60.4 MB** | 35.3 | 28.3 |

The 1.4% is inside this box's 8-15% window-to-window spread, and the work is identical by
construction: same kept positions, same rows, one table lookup either way. Correctness is pinned in
two places — `the_window_kernel_reads_a_ring_table_like_a_linear_one` runs the window kernel over a
ring table and over a full one and requires the same output (1e-5), and greedy generation at
`VIPERA_ATTN_WINDOW=16` (a window smaller than the output, so the middle really is evicted) produces
the same 40 tokens from a ring cache and from a full table read through the same window.

A ring cannot park blocks for a prefix cache (`park` and `copy_from_parked` refuse it; the prefix
cache turns itself off when a window is set), and `release` sorts and dedups so aliased table entries
free each physical block exactly once. Both are covered in `kv.rs` along with the slot count and the
refusal to address past the context.

### Speculative decoding at 32k with the ring cache — loses, and does not reproduce — 2026-11-18

The remaining 32.5 ms of a windowed 32k step is the weight stream, so the obvious lever is more tokens
per weight pass: prompt-lookup speculation. Measured with a real prefill (32,768 tokens of a ten-token
period, 173 tok/s, into a ring cache — the prefill itself works, so a chunk reading a window it is
still writing is fine), 8 threads, `VIPERA_ATTN_WINDOW=2048`, 32 tokens out:

| draft | ms for 32 tokens | tok/s | speedup | passes | drafts accepted |
|---|---|---|---|---|---|
| plain | 1350 | 23.7 | — | 32 | — |
| n=1 k=1 | 1621.2 | 19.7 | 0.83x | 31 | 1/11 |
| n=1 k=2 | 1755.0 | 18.2 | 0.77x | 31 | 1/19 |
| n=1 k=3 | 1868.5 | 17.1 | 0.72x | 31 | 1/26 |
| n=2 k=2 | 1440.8 | 22.2 | 0.94x | 32 | 0/4 |
| n=2 k=4 | 1551.8 | 20.6 | 0.87x | 32 | 0/8 |

**It does not go past 25 tok/s; it goes backwards.** Two reasons, and the second one is the real
result:

- **The acceptance rate is ~zero on this prompt.** A ten-token period makes the last-occurrence draft
  almost always wrong after the prefill: 1 of 11 for `n=1 k=1`, 0 of 4 for `n=2`. One rejected draft
  still costs a 2-token pass (50 ms against 41.6), so every round pays more for one token. This is
  prompt-dependent by construction and says nothing about a draft model, which is not implemented.
- **The verify pass is not bit-identical to plain decoding at depth.** `a_verify_pass_matches_a_
  single_token_pass` compares `logits_each` row 0 against `logits` for the same token, same KV: at
  depth 1024 it differs by 1.9, at 4096 by 9.3, and the argmax flips in the window-off case. With
  `VIPERA_THREADS=1` it is exactly 0, and it reproduces identically run to run at 8 threads, so it is
  a summation-order difference in the batched kernels, not a race. It is not the window and not the
  ring: it reproduces with the window off, and with a ring of the window's size it is if anything
  smaller (7.8 vs 9.3 at 4096).

So speculation at a deep context has no measured win here and cannot yet be claimed as token-exact;
the blocked piece is that `run_batch_n`'s "entry `e`'s logits are bit-identical to `logits`" does not
hold with more than one thread. Fixing that (one reduction order for batched and single passes) is
the prerequisite for revisiting a draft model, which is the version that would actually pay: this
n-gram drafter is out of ideas after 32k tokens of context.

### The window kernel paid per-head call overhead, and the weight stream is not bit-limited — 2026-11-18

With the ring cache in place, a 32k step with a 2048 window profiles as: **q4k 19.2 ms (48%), q6k
11.1 ms (28%), attention 9.0 ms (23%), rest 0.7 ms (2%)** — 40 ms, 25.2 tok/s. Two things to test
from there.

**Lower-bit weights do not apply where they look like they do.** The kernel-level rates, from
`kernel_gbps_fused_q4k_q6k` on this box:

| tensor | type | MB/call | ms/call | GB/s |
|---|---|---|---|---|
| ffn_gate | Q4_K | 7.7 | 0.3 | 30.42 |
| wq | Q4_K | 1.3 | 0.1 | 13.11 |
| ffn_down | Q6_K | 11.3 | 0.4 | 29.02 |
| output | Q6_K | 191.4 | 6.4 | 30.05 |

Q4_K and Q6_K stream at the same rate (30 GB/s), so bytes map straight to time and requantizing the
Q6_K tensors (`ffn_down`, `output`, 507 MB) down to Q4_K would save ~160 MB ≈ **5 ms** — but those
are exactly the tensors the Q4_K_M mix keeps at six bits because they tolerate four worst; the layers
that tolerate fewer bits are the ones already at Q4_K, and going below needs a Q3_K/Q2_K kernel this
engine does not have. Not pursued: a naive requantization would spend accuracy on the sensitive half.
The real weight-stream finding is that both kernels sit at **30 GB/s against a 43 GB/s read ceiling**,
i.e. the q4k kernel's own unpack and reduction are the binding constraint, not the format's size.

**Attention was paying a call per head per row.** `q4k_dot` and friends are cheap, but
`attention_window_chunked_heads` called `dot_f16_f32` and `axpy_f16_f32` once per head per row — 24
calls per (layer, position), 1.4M per step — and each call re-runs an `OnceLock` load, two
`is_x86_feature_detected!` checks and a fresh loop setup for a 64-value dot. `dot_f16_f32_heads` and
`axpy_f16_f32_heads` in `simd.rs` hoist all of that to one check per KV head and run the six query
heads that share its row back to back; the per-head arithmetic is unchanged, so the scores are
bit-identical. Measured at 32k, 8 threads, `VIPERA_ATTN_WINDOW=2048`:

| | attention | ms/step | tok/s |
|---|---|---|---|
| per-head calls | 9.0–9.2 | 39.6 | 25.2 |
| one call per KV head | **7.85** | **36.9** | **27.1** |

7.4% off the step, 1.2 ms of the 9.0, and the per-position output is unchanged (greedy text at
`VIPERA_ATTN_WINDOW=16`, which evicts the middle of the history, is byte-for-byte what it was). The
remaining gap is not call overhead: 57 MB of KV per step at 7.85 ms is 7.4 GB/s, so what is left is
the strided row walk itself, which is the next thing to measure rather than guess at.

Scaling check, since a flat 32k step invites the question of whether more threads help: 2 threads 64.7
ms, 4 threads 44.8, 6 threads 40.8, 8 threads 37.4 (before the change). 8 is the ceiling on this
4-core/8-thread box, so the thread count is not leaving anything on the table.

**Inside the attention kernel, two changes, one discarded.** After batching the heads the kernel was
still 7.85 ms for 57 MB of KV, so the question was whether it waits on memory or on arithmetic. A
serial run answered it: 20.9 ms single-threaded is 17 GFLOP/s, which is a two-accumulator mul+add dot
at its dependency-chain limit, not a bandwidth figure. So:

- **Software prefetch of the next rows: no gain, reverted.** `for_each_row_range` reads one K row and
  one V row per position out of a 28,672-byte token, so the hardware prefetcher does stop at the
  first gap; asking for the next two rows' lines by hand changed attention from 7.75 to 7.74 ms. The
  out-of-order window was already covering the two loads per position, and the chain was the limit.
- **Interleave the heads of one KV head: kept.** One pass over the row, converted once, multiplied
  into one accumulator pair per head — six independent chains instead of one, and one row load per
  group instead of one per head, with a fused multiply-add in place of mul+add. The fused rounding
  moves the window kernel's scores in the last bit relative to `dot_f16_f32`, which is why only this
  batched path takes it: the two-pass window kernel and `VIPERA_EXACT_DOT` stay the reference, and
  the kernel/tolerance tests still pass.

At 32k, 8 threads, `VIPERA_ATTN_WINDOW=2048`:

| attention kernel | attention | median ms/step | tok/s |
|---|---|---|---|
| per-head calls | 9.0 | 39.6 | 25.2 |
| one call per KV head | 7.85 | 36.9 | 27.1 |
| interleaved heads + FMA | **6.3** | 36.3 | **27.6** |

30% off attention over the three versions, but only 4% off the step, and that is the honest headline:
**attention is no longer where the time is.** q4k is 19.0 ms (52%) of a 36.3 ms step and q6k 10.9
(29%), against the 30–36 GB/s that the two kernels themselves reach on a 191 MB tensor (35.9 GB/s at
8 threads, i.e. 84% of the 43 GB/s ceiling). Everything left on this hardware is in the q4k stream.
Greedy text is unchanged by all of it: `VIPERA_ATTN_WINDOW=16` (window smaller than the output, so the
history's middle is really evicted) gives the same 40 tokens as before.

### Prefill is attention-bound, and the same batching pays there — 2026-11-18

A 4096-token prefill profiled at **attention 49%, q4k 36%, q6k 10%** — the mirror image of decode, and
the reason prefill was the last big number left: 50.1 s for 4096 tokens, 81.8 tok/s, against
llama.cpp's 217 on pp512. The earlier note here said the prefill lever would be "a wider kernel, and
that changes the reduction order, so the tests that pin the f16 dot to the decode-then-dot oracle
would have to become tolerance checks". It does not have to: the interleaved kernel takes a `fused`
flag, and with it off it does the same multiply and the same add `dot_f16_f32_avx2` does, so
`streaming_attention_is_bit_identical_to_decoding_the_layer_first` still holds bit for bit —
`VIPERA_EXACT_DOT=1` and the unfused path agree exactly, as before.

`dot_f16_f32_heads_exact` / `axpy_f16_f32_heads_exact` are now what the streaming kernel (prefill's hot
path) calls when every head is wanted and the KV is f16 in place, which is what a prefill chunk always
asks for. One row load and one feature check per KV head instead of one per head, six independent
chains per row:

| | before | after |
|---|---|---|
| pp1024 | 143.9 / 133.5 tok/s | **154.6** |
| pp4096 | 81.8 (50.1 s) | **95.7** (42.8 s) |
| attention in a 4096-token chunk, last pass | 683.9–695.9 ms | **503.4–512.3 ms** |

16% on pp4096 and 26% off the attention phase, with prefill now 42% q4k, 41% attention, 12% q6k
instead of half attention. Decode at 32k with the window is unmoved (median 36.1 ms, 27.7 tok/s, three
runs), as it should be: this path is the streaming kernel, and a windowed decode does not take it.

### A 512-bit VNNI Q4_K matvec for decode — measured, not kept — 2026-11-18

The largest remaining term is the q4k stream (42–52% of both a prefill chunk and a 32k decode step),
so the obvious move is the AVX-512 VNNI form the prefill path already has. The engine has one:
`q4k_matvec_tiled_range`, which scores 32 weight rows against one activation tile per pass. It is
gated to `B >= 16` because this CPU double-pumps AVX-512 and a sixteen-entry tile wastes fifteen lanes
for one row.

Re-measured at the shape a decode step actually runs, on a synthetic Q4_K tensor of `ffn_gate`'s size
(8960x1536, one activation, single thread, best of four), now that the row kernel has changed underneath:

| kernel | GB/s |
|---|---:|
| row kernel (VNNI-256, `fill_rows`) | **11.47** |
| 512-bit VNNI tile at B=1 | 2.80 |

4.1x slower, which is the double-pumping arithmetic rather than a bug: on this core every 512-bit
integer op is two 256-bit uops, so the wider form saves no issue slots, and a 1-row tile fills 1 of
16 lanes. A row-shaped 512-bit kernel would not waste lanes, but it still pays two uops per
instruction and two extra inserts per row (the low and high nibble planes of two adjacent 32-value
runs are not contiguous in the activation vector), so the ceiling it could reach is the 256-bit one.

So the answer to "write a 512-bit q4k matvec" is the measurement, not a patch: it does not improve
decode, and prefill already takes the 512-bit path where it does pay (B=128). The measurement is kept
as `a_512_bit_decode_row_is_measured_against_the_row_kernel` so the next person gets the number in a
minute rather than in a rewrite.

### Prefill attention in three parts: the V sum was the expensive one — 2026-11-18

After the head batching, prefill's attention was 41% of a 4096-token chunk, and the profile could not
say which of its three passes to attack: the score dots, the softmax's `exp`, and the weighted sum of
V all hide behind one "attention" line. Timed separately on one 2048-position history, one thread, f16
KV read in place, at this model's shape (12 heads, 2 KV heads, 128 values per head) — `prefill_attention_parts`
in `model.rs`, which needs no model file:

| part | ms per history |
|---|---:|
| score dots, `dot_f16_f32_heads_exact` | 0.20 |
| softmax: `exp` and the sum | 0.06 |
| V sum, `axpy_f16_f32_heads_exact` once per position | 0.16 |

The same pass timed with the row walk inside the loop reads 0.28 ms, because the walk's block lookup
and stride step cost as much as the arithmetic at this size: the kernel, not the traversal, is what a
change should be measured against.

The V sum is the largest term, and most of what it moved was its own accumulator: six output rows of
128 floats loaded and stored once per position, against one 512-byte V row. Holding those rows in
registers for eight positions and touching them once per block
(`axpy_f16_f32_heads_exact_block` called from `sum_v_block`) removes 7/8 of that traffic. Per element
the arithmetic is unchanged — the same multiply, then the same add, in increasing position order — so
the result is bit-identical to the per-row kernel, and
`streaming_attention_is_bit_identical_to_decoding_the_layer_first` still passes unchanged.

| | before | after |
|---|---:|---:|
| V sum per 2048-position history | 0.16 ms | **0.13 ms** |
| pp4096 | 95.7 tok/s | **99.7, 100.1** (41.1, 40.9 s) |
| pp1024 | 154.6 tok/s | 153.9 |
| decode 32k, window 2048 | 36.1 ms median | 35.1–35.7 ms |

4% on pp4096 and nothing anywhere else, which is roughly what 0.03 ms of 0.39 ms of attention in a
4k chunk predicts. Run twice back to back at 99.7 and 100.1 tok/s against a 95.7 baseline from the same
command and environment, so the 4% is the change and not the machine. A sixteen-position block measures 0.12 ms against 0.13 and was not taken: that is
the run-to-run spread, and it costs register pressure on any core with a narrower register file.

Where this leaves the pass: 3.1M values in 0.13 ms is 24 G values/s on one core, at a load, a convert,
a multiply, an add and a store per eight values, so it is at the instruction limit and there is little
left in it. The remaining term is `exp` (0.06 ms), and every way to shorten it — a polynomial, a
vectorized approximation — changes the last bits of the softmax, which is exactly what the
bit-identity test exists to forbid. That trade is not taken without an explicit decision to drop the
exact path; the number is here so the decision can be made with the cost in front of it.

### The whole engine re-measured against upstream llama.cpp: prefill is a loss at every length — 2026-11-19

Asked to re-run the comparison now that the batched prefill attention and the blocked V sum have landed,
and to correct the README if the numbers moved. They moved, and the first thing they corrected was the
reference build.

The llama.cpp column the README carried came from `vipera.cpp/build-clang`, the fork's build `49e1485`
(24) of 2026-11-05. Re-run in this session, that binary still measures **143.6 / 150.0 / 115.6 tok/s**
on `pp128 / 512 / 4096`. Upstream `e574a17` (466), the newest on this disk, measures **205–230 /
217–227 / 159–168** on the same box. Every "1.3–1.6x ahead" line the README had, and the conclusion
"vipera leads prefill up to roughly 1k" in the two 2026-11-18 head-to-head sections above, came from the
slower binary. Against the project's own newest build the direction is the other way at every length.

Method, one session, one machine (`powersave`, other tooling down): Qwen2.5-1.5B-Instruct Q4_K_M,
`VIPERA_THREADS=8`, f16 KV both sides, two interleaved passes (llama, vipera, llama, vipera) with the
vipera tests run one at a time. Prefill on the llama side is the default `n_ubatch` (512) — the arm
that flatters llama.cpp — with a matched-width arm below.

```bash
M=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
llama-latest/build/bin/llama-bench -m $M -p 128,512,1024,4096 -n 0 -t 8 -r 3 -ngl 0
llama-latest/build/bin/llama-bench -m $M -p 0 -n 128 -t 8 -r 3 -ngl 0
llama-latest/build/bin/llama-bench -m $M -p 0 -n 128 -d 4096 -t 8 -r 3 -ngl 0
VIPERA_THREADS=8 VIPERA_TEST_MODEL=$M cargo test --release -p vipera-backend-cpu \
  --test win_vs_llama -- --ignored --nocapture prefill_vs_decode
VIPERA_THREADS=8 VIPERA_TEST_MODEL=$M cargo test --release -p vipera-backend-cpu \
  --test win_vs_llama -- --ignored --nocapture prefill_long_prompt
```

| workload | pass | llama.cpp `e574a17` | vipera | vipera / llama.cpp |
|---|---|---:|---:|---:|
| pp128 | 1 | 204.7 ± 22.7 | 188.1 | 0.92x |
| pp128 | 2 | 230.3 ± 2.0 | 188.3 | 0.82x |
| pp512 | 1 | 216.7 ± 1.9 | 172.0 | 0.79x |
| pp512 | 2 | 226.9 ± 0.9 | 172.0 | 0.76x |
| pp1024 | 1 | 203.7 ± 0.7 | 164.0 | 0.81x |
| pp1024 | 2 | 211.5 ± 5.0 | 164.0 | 0.78x |
| pp4096 | 1 | 159.3 ± 1.9 | 104.7 | 0.66x |
| pp4096 | 2 | 168.1 ± 0.9 | 104.7 | 0.62x |
| tg128, depth 0 | 1 | 33.9 ± 0.6 | 34.6 | 1.02x |
| tg128, depth 0 | 2 | 36.7 ± 0.9 | 34.6 | 0.94x |
| tg128, depth 4096, full history | 1 | 25.3 ± 0.2 | 24.2 | 0.96x |

The engine's `pp4096` reads 104.7 here against 99.7–100.1 in yesterday's section on the same binary:
that is the machine (other tooling down today), not a change. The ratio measured in one session is the
number to compare, which is why both arms are paired above.

The engine's own prefill batch width is 128, so llama.cpp was also run at `-ub 128`: **215.6 / 203.2 /
189.7 / 138.7** for 128 / 512 / 1024 / 4096, i.e. vipera is 0.87x, 0.85x, 0.86x and 0.75x of a
same-width llama.cpp. Matching the width closes about half of the 4k gap (61% → 33%) and none of the story:
llama.cpp is 15–18% ahead at short and medium prompts and 33% ahead at 4k even then.

The gap widens with length on both sides, and faster on ours: from 128 to 4096 llama.cpp loses 36%
(215.6 → 138.7 at matched width) and vipera loses 44% (188.3 → 104.7). That is the same
signature the 2026-11-18 three-part measurement found from the inside — prefill attention is 41% of a
4k chunk and only 6% of the first one — and it says the next lever is still attention, not the Q4_K
stream: the matvecs cost the same per chunk at 4k as at 1k, while attention grows with the history.

Decode is level, as it has been for four sessions: 0.94–1.02x at depth 0 and 0.96x at 4k of history.

Full-history decode on the engine, from the same state-fabricating test (`decode_cost_at_depth_without_prefill`,
no window): 24.2 tok/s at 4k, 18.3 at 8k, 11.5 at 16k, **6.8 at 32k** (147.2 ms/step). With
`VIPERA_ATTN_WINDOW=2048` that 32k step is 35.8 ms, **28.0 tok/s**.

Two numbers in the 2026-11-18 ring-cache section sit away from these: 183.7 ms full history and
52.0 ms with the 2048 window at 32k, against 147.2 and 35.8 today. Both were real runs, so this is
recorded as unresolved rather than reconciled — a 45% swing on the windowed step is not clock noise.
The ratios inside one session are what the table above uses. llama.cpp above 4k depth was not measured
here either: a `-d 32768` arm prefills 32k tokens per repetition, so three repetitions would spend ten
minutes of prefill to time 128 decode steps.

**Reading.** Prefill: llama.cpp wins, by 9–22% at 128 tokens, 26–32% at 512, 24–29% at 1024 and 52–61%
at 4096. Decode: a wash. Long context: the engine's only lead is the one it buys with the window, which
is lossy and opt-in. The README now says that, and quotes `e574a17` rather than the fork build.

## Prompt-chunk rotary, and two dead ends — 2026-11-19

Setup as above: `VIPERA_THREADS=8`, Qwen2.5-1.5B Q4_K_M, one session, arms back to back.

**The waste.** `rope` was called once per layer per tensor, and each call rebuilt the position's
angles: 64 `powf` plus 64 `sin_cos` per call, 56 calls per position, for values that depend only on
the position. It also computed the `inv_freq` table and never read it. A new measurement test
(`norm_phase_cost`, `--ignored`) prices the two parts of the profile's `Norm` phase for one 128-token
chunk on one thread:

| per 128-token chunk | before | after |
|---|---:|---:|
| rotary, one table built per call (as before) | 10.27 ms | 9.48 ms |
| rotary, one table per position reused by all 28 layers | — | 3.76 ms |
| `rms_norm` (unchanged) | 11.83 ms | 11.83 ms |

The chunk pass now builds `RopeAngles` per position before the first layer and rotates Q and K from
it. The phase split of the same run says the work moved: `rest` per 128-token chunk went from
**48.9–50.1 ms to 43.3–44.2 ms** (−5.8 ms, −12%), while `q4k`, `q6k`, `attention` and `norm` sat
where they were. `pp1024` was 6399.1 ms (160.0 tok/s) after and 6407.0 ms (159.8 tok/s) before —
inside this machine's run-to-run spread, so the end-to-end claim is only "no slower"; the 0.9% of a
chunk the phase table removes cannot be resolved by a whole-model number here. The rotary arithmetic
itself is untouched, element by element, so no logit moved: `batch.rs`'s bit-for-bit
batched-versus-alone checks pass.

`rms_norm`'s 11.8 ms is the other half of the `Norm` phase and it is latency, not throughput: the
sum is a serial `f32` chain (1536 dependent adds, ~4 cycles each) that cannot be reassociated without
changing the logits, so it is left alone and recorded here.

**Dead end 1: walk K/V once per chunk.** Attention currently decodes the same K/V rows once per
query row. Sharing the walk across a whole 128-row chunk was implemented bit-exactly
(`a_joint_chunk_walk_is_bit_identical_to_attending_each_row_alone` passed for f16 and Q8_0, contiguous
and gapped positions) and measured **153.1 tok/s at 1k and 88.3 at 4k against 165.7 / 106.1** — 8% and
17% slower. The score matrix of 16 rows per worker is 3 MB and leaves L2, which the per-entry path's
196 KB does not. Reverted in full. A sharing group of 4 rows (≈768 KB, still inside L2) is the
untried version of this idea.

**Dead end 2: fewer prefetch hints in the tiled Q4_K kernel.** The tiled kernel issues three
`prefetch_ahead` per row per block, ~19% of the inner loop's uops. Cutting them to one measured
6581.1 ms / 40119.0 ms for 1k / 4k against 6407.0 / 39517.3 — no worse inside the spread, and no
better. The hints stay.

### Four-accumulator rms_norm: 3.1x on the norm, 0.3 on the logits — reverted — 2026-11-19

`rms_norm`'s sum is one dependent chain of 1536 `f32` adds, so each add waits for the one before it.
Four independent accumulators, combined pairwise at the end, break the chain without changing the
values being summed. Measured with `norm_phase_cost` on one thread, per 128-token chunk:

| per 128-token chunk | ms |
|---|---:|
| `rms_norm`, serial chain | 12.25 |
| `rms_norm`, four accumulators | 3.95 |

In the real pass (`VIPERA_PROFILE_STEP=1`, 8 threads, `prefill_vs_decode`) the profile's `norm` line
per 128-token prompt chunk goes from **7.51–7.76 ms to 2.49–2.52 ms** (−67%), with `q4k`, `q6k`,
`attention` and `rest` unchanged. That is ~5 ms of a ~600 ms chunk, about 0.8%.

**What it costs.** Logits for a fixed 96-token prompt plus 12 forced decode steps (so both builds
stay on the same token path), 13 vectors of 151,936 values:

| step | max abs | mean abs | same argmax | same top-5 |
|---|---:|---:|---|---|
| prefill | 0.37 | 0.061 | yes | 5/5 |
| decode 1–12 | 0.27–0.60 | 0.047–0.101 | yes, 12/12 | 12/13 |

The greedy stream is the same and the argmax never moves on these 13 steps, but the numbers are not
"last bits": a 1e-6 relative change in a norm surfaces as ~0.1 in a logit. Something downstream
amplifies it — the per-block i8 activation quantization is the likely mechanism (a last-bit change in
one activation flips its quantized value by a whole 1/127 step), plus 28 layers of residual
propagation. That is a hypothesis, not a measurement.

**Verdict: reverted.** The tolerance class this repository has recorded for an approximate path is
~1e-7 (`VIPERA_ATTN_KV_SPLIT`, a merge of partial softmaxes, opt-in). 0.1 mean-abs is five orders of
magnitude past it and would be a change to the model's answers, not a rounding detail — a policy
decision of the same kind as the KV split, made deliberately and behind a switch, not an
optimization slipped into the default path. The end-to-end arms agree that nothing is on the table
anyway: `pp1024` 7584.8 ms four-lane against 7728.3 ms serial, `pp4096` 47983.2 against 46059.1 —
the machine slowed by ~20% during this window (the same test read 6407.0 / 39517.3 ms earlier in the
session), and the chunk totals moved with it between arms, so neither direction is attributable while
the `norm` line is.

### Four rows sharing one K/V walk: the L2 budget was not the only problem — 2026-11-19

The idea, stated properly this time: a causal prompt chunk's rows all read the same cache, and each
row reads a *prefix* of what the next one reads, so four adjacent rows can decode and load each K and
V row once and use it for all four. Four rows of 12 heads at a 4k history is a 786 KB score buffer,
against 196 KB for one row and 3 MB for the sixteen-row version that lost 8–17% earlier the same day,
so this was the version the earlier attempt should have been.

It is bit-identical, and that was checked before any timing: the same per-head dot, the same softmax
in increasing position order, one position at a time into each output element. A test compared four
rows of lengths 61/48/33/17 (stopping on either side of 16-token block boundaries, blocks out of
logical order) against attending each row alone, on both row orders and both layers, and the output
bits matched. It was reverted after the measurement, so the test is not in the tree either.

Prefill, 8 threads, `prefill_long_prompt`, arms interleaved in one binary (`VIPERA_ATTN_ROW_GROUPS`):

| | pp1024 | pp4096 |
|---|---:|---:|
| one row per walk | **6865.7, 6957.1 ms** (149.1, 147.2 tok/s) | **42469.2, 42759.9 ms** (96.4, 95.8 tok/s) |
| four rows per walk | 7354.2, 7392.9 ms (139.2, 138.5 tok/s) | 50968.2, 48609.9 ms (80.4, 84.3 tok/s) |

So: 7% slower at 1k and 14–19% slower at 4k, against a 786 KB buffer that was supposed to fit. The
measurement refutes the model I had for the sixteen-row loss — 3 MB of scores spilling L2 would give a
loss that shrinks with the buffer, and 786 KB loses as much. What the grouped kernel actually does
differently is two walks of the group's longest prefix (one for the dots, one for the V sum) instead of
one walk per row, so the group reads `4 x longest` rows where four separate rows read
`sum(lengths) ≈ 4 x longest - 6` — the saving is real but it is ~6 rows in 16,000. Against that it
pays four predicate tests per (position, row) in both walks, four live score streams, and a working
set that is now a whole group wide, which is the part the per-row kernel keeps narrow on purpose.

The honest summary of the whole line of attack, after two attempts: at this shape the K/V row walk is
~0.08 ms of a 0.39 ms row at a 2048-position history, one fifth of attention, and attention is under a
tenth of a prompt chunk at 1k. The best a perfectly shared walk can return is a few percent of pp4096,
and both measured attempts paid more for the sharing than the walk was worth.

### Query heads of one key head share each K and V row in streaming attention — opt-in, 2026-11-19

`VIPERA_ATTN_GQA_SHARE=1` (off by default) makes `attention_streaming_heads` treat the query heads of
one key head as a group. Pass 1 scores a K row against the whole group with `simd::dot_f16_group`: the
row is converted from f16 once per sixteen values and feeds up to four heads from the same registers,
where the per-head kernel converts and loads it once per head. Pass 2 folds V rows with
`simd::axpy_f16_group`: sixteen positions at a time, one conversion per value vector for up to eight
heads, the output accumulators held in registers across the sixteen positions instead of loaded and
stored per head and position. Every element still accumulates in increasing position order. The
KV-chunk decode kernel takes the same switch, sixteen positions per block (`fold_chunk_block`), one
rescale per block instead of one per new maximum. Only `f16` rows read in place take it; `q8_0`, the
sequential `VIPERA_EXACT_DOT=1` path, and groups larger than 16 keep the per-head kernels unchanged.

**It is not bit-identical.** Fused multiply-add and the regrouped sums round differently from the
per-head kernel, within a few ulps of the sum of absolute terms. That is why it is a switch and not the
default.

Correctness is pinned by four tests: `group_scores_stay_within_rounding_of_the_exact_sum` and
`group_fold_stays_within_rounding_of_the_exact_sum` (kernels against the exact `f64` sum),
`shared_group_attention_matches_the_per_head_kernel_within_rounding` (whole streaming kernel, group
sizes 1/3/6/7, head ranges that cut a group, positions around the 16-row blocks, shuffled block table;
`q8_0` stays bit-equal) and `shared_kv_chunk_matches_the_serial_kernel_within_rounding` (KV-chunk
kernel, 2/3/4 workers).

**Measured on the i5-11300H, not on the AVX-512 sandbox these numbers were first taken on**, so the
ratios are lower than that draft reported: this machine's per-head kernel is not the same baseline.
Single thread, arms interleaved, best of 7, f16 KV, shuffled block table, 12 query heads over 2 key
heads of 128.

| layer-level, single entry | per-head | shared | ratio |
|---|---|---|---|
| 256 positions | 0.060 ms | 0.044 ms | 1.36x |
| 1024 | 0.242 ms | 0.181 ms | 1.33x |
| 4096 | 1.005 ms | 0.782 ms | 1.29x |

| KV-chunk decode, 4 workers | per-head | shared | ratio |
|---|---|---|---|
| 1024 positions | 0.171 ms | 0.095 ms | 1.80x |
| 4096 | 0.585 ms | 0.274 ms | 2.14x |
| 8192 | 1.209 ms | 0.558 ms | 2.17x |

At kernel level (`attention_group_throughput`, head dim 128) the score pass is 2.0–2.4x and the value
pass 2.3–3.1x for groups of 3 to 7 — the kernels themselves behave as intended; the layer-level ratio
is what is left after the softmax, the row walk and memory.

**End to end, one binary, 901-token prompt, `VIPERA_PROFILE_STEP=1`:**

| | per-head | shared |
|---|---|---|
| prefill chunk 6 (attention) | 110.8 ms | 95.7 ms (1.16x) |
| prefill chunk 7 | 125.4 ms | 108.4 ms (1.16x) |
| prefill chunk 8 | 111.8 ms | 96.5 ms (1.16x) |
| three chunks, compute total | 841.8 / 856.3 / 760.2 ms | 822.5 / 837.6 / 743.2 ms (~2%) |
| decode at ~900 positions (attention) | 6.4 / 11.2 / 9.4 ms | 4.7 / 9.0 / 7.7 ms (1.2–1.4x) |
| decode step total | 42.5 / 71.5 / 60.7 ms | 41.2 / 66.6 / 59.0 ms |

Attention is 11–16% of a decode step and 13% of a prefill chunk at this prompt length on this machine,
so a 1.2–1.4x faster attention is about 3% of a decode step and about 2% of a prefill chunk — not the
20% an earlier projection implied, which assumed attention was half the chunk on a 4k prompt and a
sandbox baseline. On this hardware the switch is a small, honest win, and it stays off by default
because it is not bit-identical to the kernel the golden numbers were recorded with.

### The softmax exponent in eight lanes — opt-in, 2026-11-19

`VIPERA_FAST_SOFTMAX=1` (off by default) replaces the softmax pass's `f32::exp` calls with
`simd::exp_sum_shifted`: the subtract, the exponent and the sum run eight values at a time through the
same degree-five polynomial and exponent-byte trick that `model::exp` uses for the SwiGLU sigmoid, so
a lane writes the scalar polynomial's bits exactly (`exp_sum_shifted_matches_the_scalar_exp_bit_for_bit`,
lengths 0/1/7/8/9/15/16/17/31/33/1024 against the scalar loop, plus a random sweep and the special
values). All four softmax sites take it: the streaming kernel's per-head pass, the simple short-context
pass, and the KV-chunk kernel's sixteen-position block.

**The measurement that started this does not support the premise.** `f32::exp` costs 2.5 ns on this
machine (not the 4.5 ns the `model::exp` docstring records), and the scalar polynomial is *slower* at
4.4 ns — libm's `expf` wins scalar. Vectorising it is where the win is, and the pass it sits in is
12% of a 4096-position attention layer, not the majority: the two dot passes are ~80%.

**It is not bit-identical, and the error is bigger than the docstring claimed.** Measured against
libm over `[-40, 5]`, the polynomial's worst relative error is **1.9e-6** (about sixteen `f32` ulps),
not the "<1e-7" the comment claimed; below -40 it grows to 3.5e-6 in the denormal tail, where the
value is under 1e-17 and contributes nothing. `the_polynomial_exp_stays_within_a_quarter_ppm_of_libm`
pins both bands.

Measured on the i5-11300H, one process per arm, arms interleaved inside the run, best of 7, single
thread, f16 KV, shuffled block table, 12 heads over 2 key heads of 128:

| streaming layer, 1024 positions | fast off | fast on | |
|---|---|---|---|
| per-head kernel | 0.239–0.245 ms | 0.209–0.213 ms | 1.14x |
| shared kernel (`VIPERA_ATTN_GQA_SHARE=1`) | 0.181–0.185 ms | 0.152–0.153 ms | 1.19x |

| streaming layer, 4096 positions | fast off | fast on | |
|---|---|---|---|
| per-head kernel | 1.003–1.026 ms | 0.887–0.919 ms | 1.11–1.13x |
| shared kernel | 0.773–0.799 ms | 0.666–0.681 ms | 1.15–1.18x |

| KV-chunk decode, 4 workers, shared kernel | fast off | fast on | |
|---|---|---|---|
| 4096 positions | 0.271 ms | 0.240 ms | 1.13x |
| 8192 positions | 0.557 ms | 0.489 ms | 1.14x |

End to end, one binary, 901-token prompt, `VIPERA_PROFILE_STEP=1`, two runs: prefill attention
96.5/109.6/123.9/110.3 ms -> 90.6/102.5/115.0/102.0 ms (1.06–1.08x), chunk compute 815/833/849/753 ms
-> 810/825/838/744 ms (~1%); decode attention at ~900 positions 6.3/8.9 ms -> 6.1/8.1 ms.

So a softmax that is 12% of the layer, made ~2x cheaper per value, is worth a tenth of the layer and
about 1% of a prefill chunk. Off by default: it is not the libm bits, and no golden-token run has been
made with it on.

### Where a 4k prefill chunk goes, and what the two dot passes will and will not give — 2026-11-19

**The 4k A/B asked for.** 4001-token prompt, `--ctx 4096`, 32 chunks of 128, one run per arm,
`VIPERA_PROFILE_STEP=1`, i5-11300H:

| arm | prefill | attention | share of chunk |
|---|---|---|---|
| baseline | 31.75 s | 8753 ms | **27.6%** |
| `VIPERA_ATTN_GQA_SHARE=1` | 30.68 s | 7164 ms | 23.4% |
| `VIPERA_FAST_SOFTMAX=1` | 32.18 s | 8122 ms | 25.2% |
| both | 30.17 s | 6322 ms | 21.0% |
| both, after the score-kernel change below | 29.85 s | 6002 ms | 20.1% |

So at 4k attention is a quarter to a third of a prefill chunk, not the majority — the matmuls are.
The two switches together buy ~5% of the chunk wall clock at this length, and more of the attention
itself (1.46x).

**The split inside a layer.** With the fold pass short-circuited, the score pass plus softmax is 65%
of a 4096-position streaming layer and the value fold 35%. At kernel level (`attention_group_throughput`,
group 6, 128-wide, per position) the score pass is 28.7 ns and the fold 19.7 ns for the same 768
multiply-adds, so the score pass is the odd one out: it re-reads the six query rows from L1 for every
K row (96 unaligned 32-byte loads per position) where the fold keeps its accumulators in registers
across sixteen positions.

**What did not work.** A sixteen-lane AVX-512 version of both kernels is **1.6x slower** (4096-position
layer 0.773 -> 1.246 ms): this compiler's stdarch has no `_mm512_fmadd_ps`, so the fused multiply-add
becomes a separate multiply and add, and the wider path gives back the rest in frequency. Reverted.

**What did.** Six heads in one `score_block_single` call with one accumulator each — the group a 12/2
layer has, so its K row is converted once instead of once per four-head block — while groups of four
and fewer keep the two-accumulator kernel that fills the registers better. Score kernel per position:
28.7 -> 25.5–26.2 ns (1.10–1.13x); the 4096-position shared layer 0.80 -> 0.77 ms (1.04x, medians of
three runs); a 4k prefill with both switches 30.17 -> 29.85 s.

**Not attempted here, and the honest next lever:** the streaming pass takes the whole position walk on
one thread per layer. Splitting that walk across the existing attention workers — the KV-chunk decode
kernel already does it — is worth far more than another instruction of the dot kernels.

### The two attention switches are now the default — 2026-11-19

`VIPERA_ATTN_GQA_SHARE` and `VIPERA_FAST_SOFTMAX` both defaulted off while their numbers were taken,
because neither is bit-identical to the kernels that came before: the first regroups each head's sum,
the second evaluates the exponent with a polynomial that agrees with libm to 1.9e-6 relative. Both
have been measured, both hold their accuracy tests, so they are on unless the variable says `0`.

Default binary, 4001-token prompt, `--ctx 4096`:

| | prefill | attention | share of chunk |
|---|---|---|---|
| switches off (`=0`) | 33.3–34.1 s | 9243–9452 ms | 27.7% |
| defaults | 30.3–30.7 s | 6153–6186 ms | 20.0–20.4% |

Decode at 4k, median of three steps after the first: attention 30.2 -> 18.1 ms, step 79.0 -> 67.8 ms.
The machine was noisier than the prefill runs (the same arm read 46–79 ms across runs), so the step
figure is the weaker of the two; the attention figure is the one to trust.

The tests that pinned "the streaming kernel equals the buffered kernel bit for bit" now pin it for the
per-head kernel with sharing passed explicitly (`streaming_with(false, ...)`); through `attend`, which
takes the default, they hold the two within 1e-5 of the largest magnitude, like the shared-kernel test
next to them. The one-worker KV-chunk fallback still has to be bit-identical, and it takes its switch
from the same place the caller does.

### Where a 4k prefill chunk's q4k time goes, and llama.cpp on the same model — 2026-11-19

**The chunk.** Profile of one 128-token chunk at the end of a 4001-token prompt (`--ctx 4096`,
`VIPERA_PROFILE_STEP=1`, i5-11300H, 8 threads), 1045 ms:

| phase | ms | share |
|---|---|---|
| Q4_K products | 507 | 48% |
| attention | 328 | 31% |
| Q6_K products | 153 | 15% |
| rest (rope, adds, copies) | 43 | 4% |
| RMS norm | 15 | 1% |

**Inside the 660 ms of products**, by tensor shape (ms per chunk, `SHAPES` in the profile):

| tensor | format | calls | ms | share of products |
|---|---|---|---|---|
| FFN gate + up, 8960x1536 | Q4_K | 56 | 292 | 44% |
| FFN down, 1536x8960 | Q6_K (14 layers) + Q4_K (14) | 28 | 137 + 112 | 38% |
| attention q + o, 1536x1536 | Q4_K | 56 | 75 | 11% |
| attention k + v, 256x1536 | Q4_K + Q6_K | 56 | 23 + 8 | 5% |
| lm_head, 151936x1536 | Q6_K | 1 | 6 | 1% |

So the feed-forward block is 82% of the prefill's matrix work and the attention projections 16%,
which is why chunked prefill (one weight read for 128 activations) is worth what it is.

**It is compute-bound, not bandwidth-bound.** A chunk reads the 1.04 GiB of weights once for 128
tokens: 1.12 GB in 0.66 s is 1.7 GB/s, and the same matrices are 228 GMAC in that time, 345 GMAC/s.
Whatever is left is the dequantisation and activation quantisation inside the kernel, not DRAM.

**The same model against llama.cpp**, built from source on this machine
(`cmake -DGGML_NATIVE=ON`, build 4db8b76, `-t 8`), both with the same 4010-token prompt:

| | prefill | decode at ~4k |
|---|---|---|
| vipera (defaults) | 134–137 tok/s | 13.4–13.6 tok/s (73–75 ms/step) |
| llama.cpp `llama-cli` | 146–151 tok/s | 23.0–23.7 tok/s (42–43 ms/step) |
| llama-bench, `pp4096` / `tg @ d4096` | 138–142 tok/s | 21.4–22.7 tok/s |

Prefill is within 10% and slightly behind. Decode at depth is 1.7x behind, and the profile says where:
a decode step reads the same 1.04 GiB for one token, which at 73 ms is 15 GB/s where llama.cpp's 43 ms
is 26 GB/s. The matvec at batch 1 — not attention, which is 27 ms of the 73 — is the gap.

**A caveat on this host, and it is not small.** `intel_pstate` runs in `powersave` and is not ours to
change (no root): during a decode burst the clock reads 0.8–1.4 GHz, and one 25-token run drifted
64 → 120 ms/step while the prompt before it ran at 3.2 GHz. Decode figures here are only comparable
between arms measured back to back in the same session, which is how every number above was taken.
The prefill figures are steady because eight busy threads hold the clock up.

**Repeatable measurement:** `scripts/bench-4k.sh` runs a 4k prompt twice per attention-switch arm and
prints prefill, the attention share of it, decode, and llama-bench's own numbers when
`LLAMA_BENCH=/path/to/llama-bench` is set.

### The sliding-window kernels share K/V rows too — 2026-11-19

`VIPERA_ATTN_GQA_SHARE` only reached the full-history kernels. A run with `VIPERA_ATTN_WINDOW` set —
the one configuration where decode cost stops growing with depth, and so the one where decode
attention matters most — went through `attention_window_heads` and
`attention_window_chunked_heads`, which had no shared arm and read each K and V row once per query
head. Both take the switch now. The two-pass window kernel scores each kept K row against a key
head's whole group in one `simd::dot_f16_group` call and folds the kept V rows sixteen at a time
through `fold_value_block` (`positions` became `stride`, with the score slot passed explicitly, so
the kept positions stay numbered `[0, kept)` across the gap the dropped middle leaves). The chunked
window kernel, which is what a decode step runs, folds each chunk through `fold_chunk_block`, the
same block online-softmax the full-history KV-chunk kernel uses.

Both remain within rounding of their per-head arms rather than bit-identical, and `q8_0` rows, the
sequential `VIPERA_EXACT_DOT=1` path and groups over 16 heads keep the per-head kernels bit for bit:
`shared_window_matches_the_per_head_kernel_within_rounding` holds them to 1e-5 across group sizes
1/3/6/7, head ranges that cut a group, kept-position counts of 12/19/20/21/44 that straddle the
sixteen-row blocks, and 2/3/4 workers, and checks the `q8_0` arm for equality.

The end-to-end A/B the section above describes is runnable again — `decode_gqa_share_switch` and
`prefill_gqa_share_switch` force the switch per call (`force_share`) instead of reading the
environment, so one process times both arms. Qwen2.5-1.5B Q4_K_M, i5-11300H, 8 threads, arms
interleaved, decode with the KV chunk and four workers on in both arms:

| | per-head | shared |
|---|---|---|
| decode, ctx 1024 | 39.15 ms/step | 37.47 ms/step (-4.3%, slower in 18/60 rounds) |
| decode, ctx 4096 | 51.21 ms/step | 43.07 ms/step (-15.9%, slower in 1/60) |
| decode, ctx 4096, `VIPERA_ATTN_WINDOW=2048` | 42.40 ms/step | 39.31 ms/step (-7.3%) |
| prefill 1024, best of 3 | 164.8 tok/s | 171.4 tok/s (+4.0%) |
| prefill 4096, best of 2 | 112.4 tok/s | 125.7 tok/s (+11.8%) |

The `VIPERA_ATTN_WINDOW=2048` row is the chunked window kernel (ctx 4096 is past the window plus
sinks); the ctx 1024 row of that run is still the full-history kernel, since the window only applies
once the history outgrows it. That is the end-to-end gain for the window path specifically: 3.1 ms
of a 42.4 ms step.

The same pairs on Qwen2.5-7B Q4_K_M, where the step is bandwidth-bound and attention is a smaller
share, so the ratio shrinks with the model:

| | per-head | shared |
|---|---|---|
| decode, ctx 1024 | 136.26 ms/step | 132.20 ms/step (-3.0%) |
| decode, ctx 4096 | 166.22 ms/step | 147.76 ms/step (-11.1%, slower in 0/60 rounds) |
| prefill 1024, best of 3 | 42.7 tok/s | 43.1 tok/s (+1.0%) |
| prefill 4096, best of 2 | 32.2 tok/s | 34.5 tok/s (+6.5%) |

The 14B GGUF on this machine is Hermes 4 14B, which is `qwen3`. The CPU backend reads that layout
too now — `qwen3` blocks differ from `qwen2` by per-head q/k RMSNorm before RoPE, no attention
biases, and a head width that comes from `attention.key_length` — so the same pair was taken on it
(Hermes 4 14B Q4_K_M, 40 layers, 40 heads / 8 KV heads, head width 128, 40k context):

| | per-head | shared |
|---|---|---|
| decode, ctx 1024 | 254.29 ms/step | 246.79 ms/step (-3.0%, slower in 13/60 rounds) |
| decode, ctx 4096 | 320.07 ms/step | 274.72 ms/step (-14.2%, slower in 0/60 rounds) |
| prefill 1024, best of 3 | 22.0 tok/s | 22.0 tok/s (+0.3% time) |
| prefill 4096, best of 2 | 15.9 tok/s | 16.7 tok/s (-5.2% time) |

At 4k the 14B decode step saves 45 ms, the largest absolute saving of the three models, and it is
the same ratio as the 7B because the step is bandwidth-bound for both. The 1024 prefill is a wash
(46.4 s against 46.6 s, inside run-to-run noise at this size) and the 4096 one gains 5%.

The qwen3 path is held to llama.cpp's logits rather than to its prose: `scripts/logits-dump.sh`
runs one fixed token sequence through llama.cpp (no sampling, no chat template) and
`crates/vipera-backend-cpu/tests/logits_reference.rs` runs the same ids through this backend. On the
14B file the argmax is the same token (19846), the top eight are the same set, and the worst of the
top sixteen moves 0.25 logits against a top of 22.6 — the rank flips start where neighbouring rows
sit 0.05 apart, inside the accumulation noise of two different f32 orders. Coherent greedy text
from the first token is a weaker signal than that and is not what the check rests on.

```bash
cargo test --release -p vipera-backend-cpu --lib decode_gqa_share_switch  -- --ignored --nocapture
cargo test --release -p vipera-backend-cpu --lib prefill_gqa_share_switch -- --ignored --nocapture
VIPERA_ATTN_WINDOW=2048 cargo test --release -p vipera-backend-cpu --lib decode_gqa_share_switch -- --ignored --nocapture
# the 14B rows: point the measurement at the qwen3 file
VIPERA_TEST_MODEL=~/.vipera/models/hermes4-14b-q4km.gguf \
  cargo test --release -p vipera-backend-cpu --lib decode_gqa_share_switch -- --ignored --nocapture
```

### The GPU attention kernel shares K rows across a head group too — opt-in, 2026-11-20

The CPU sharing numbers above had no GPU counterpart: `vipera-backend-gpu` had one attention kernel,
one block per query head, and every head of a group read the same `positions * head_dim` of K and V.
At 4k positions on a 40-head/8-KV-head model that is ~168 MB of cache traffic per layer for ~33 MB
of distinct rows, and a 4 MB L2 holds neither side of that.

`attention_decode_shared` (in `layer.cu`) puts one block per *KV* head instead: the block's first
`head_dim` threads stage the K row in shared memory, all five heads of the group score it from
there, and the group's warps read each V row with the same index, so L1 serves one row where the
per-head kernel read five. Each block owns `group * head_dim` = 640 threads. The softmax weights are
the only thing that leaves the block — `group * positions` floats do not fit in shared memory at long
contexts (80 KB at 4096) — so they go to a per-layer scratch buffer of `heads * max_positions` on
the device. 80 KB of traffic against the MBs the shares save.

The arm is behind `VIPERA_GPU_ATTN_SHARE=on` (or `layer::force_shared(Some(true))` for a paired run)
and defaults off: every earlier GPU number was taken with the per-head kernel. Shapes without a
group, without whole-warps-per-head, or with `group * head_dim` over 1024 keep the per-head kernel.

Correctness: `shared_attention_matches_the_per_head_kernel` holds the two arms to 1e-5 of the output
magnitude at 1, 7, 64 and 511 positions on Qwen3-14B's attention shape, and both against a CPU
reference — worst difference 6e-8, which is rounding, not an arm.

The A/B (`attention_arms_at_long_context`, arms interleaved inside every round, 20 rounds after
three warm launches, GTX 1650, Qwen3-14B attention shape, synthetic rows so a GGUF quantizer is not
in front of the kernel):

| positions | per-head | shared | time change |
|---|---|---|---|
| 512 | 0.535 ms | 0.507 ms | -5.2% |
| 1024 | 1.061 ms | 1.006 ms | -5.2% |
| 2048 | 2.116 ms | 2.006 ms | -5.2% |
| 4096 | 3.651 ms | 3.355 ms | -8.1% |

Honest reading: this is the sharing working and it is not the win the traffic math promises. The K
reads that go away are ~64 MB at 4096, which at this card's ~200 GB/s is ~0.3 ms — exactly what was
measured. The rest of the 3.4 ms is not bandwidth: the per-position loop carries two
`__syncthreads()` and the V pass is a serial dependent add, so the kernel runs at a fraction of the
card's bandwidth and moving fewer bytes barely moves it. One block per KV head is also 8 blocks
across 14 SMs, i.e. six idle. Making this pay needs the split-positions attention the decode path
does not have yet (flash-decoding style: blocks over position chunks, then a combine pass), not a
better sharing kernel.

End to end the switch is invisible at the lengths this backend is exercised at: the 1.5B model's
23.6 ms/token with `VIPERA_GPU_ATTN_SHARE=on` lands inside run-to-run noise, because attention is
under a millisecond of that step.

```bash
cargo test -p vipera-backend-gpu --release --test attention_shared -- --ignored --nocapture
VIPERA_GPU_ATTN_SHARE=on cargo test -p vipera-backend-gpu --release --test model_gpu -- --ignored --nocapture
```

### Split the history across blocks: the GPU attention kernel cut along positions — 2026-11-20

The section above ended on a lead: the shared kernel's 4k time (3.36 ms) was not its bandwidth
(0.3 ms of saving from the K rows it stopped re-reading), it was latency — eight blocks on a
fourteen-SM card, and two `__syncthreads()` per position on a 4096-long loop. The fix is the one
flash-decoding uses: cut the history, not the heads.

`attention_split` runs one block per (KV head, chunk of 128 positions): it stages K rows in shared
memory, scores the whole group against them, and writes an unnormalised partial — the chunk's max,
its softmax denominator, and `sum(exp(score - max) * v)` per dimension. `attention_combine` folds
the chunks with the online-softmax identity (one block per head, one thread per dimension). Scores
stay in shared memory, so the shared kernel's global score scratch disappears. Chunk length was
swept at 64/128/256/512: 64 and 128 tie at 4096 (1.92 ms), 512 loses at 2048 (1.19 against 1.00),
so 128 is what ships.

The arm is on by default from 512 positions up (`VIPERA_GPU_ATTN_SPLIT=off` pins the older arms;
`force_split` does it per call). Parity holds where it did for the shared arm:
`split_attention_matches_the_per_head_kernel` covers 1/7/255/256/257/511/512/513/1023/1024 positions
— chunk edges, thread edges and the empty tail past 1020 — to 6e-8, and
`the_split_arm_preserves_a_real_layer` runs a real Qwen2.5-1.5B block for 700 positions twice, once
per arm: worst 1.43e-6 on values up to 4.7.

Three arms, same rows, arms interleaved inside every round, 20 rounds after warming all three
(GTX 1650; synthetic K/V so no quantizer is in front of the kernel):

| shape | positions | per-head | shared | split |
|---|---|---|---|---|
| qwen3-14b, 40/8 heads | 512 | 0.537 ms | 0.509 (-5.4%) | **0.380 (-29.2%)** |
| | 1024 | 1.061 ms | 1.006 (-5.2%) | **0.627 (-40.9%)** |
| | 2048 | 2.116 ms | 2.006 (-5.2%) | **1.241 (-41.3%)** |
| | 4096 | 3.549 ms | 3.275 (-7.7%) | **1.923 (-45.8%, 1.8x)** |
| qwen2.5-1.5b, 12/2 heads | 512 | 0.332 ms | 0.412 (**+24.1%**) | **0.116 (-65.1%)** |
| | 1024 | 0.884 ms | 0.885 (-0.1%) | **0.226 (-74.4%)** |
| | 2048 | 1.753 ms | 1.763 (+0.6%) | **0.339 (-80.7%)** |
| | 4096 | 3.504 ms | 3.546 (+1.2%) | **0.579 (-83.5%, 6.0x)** |

The 1.5B row is the one that matters for this card, and it also corrects the section above: with two
KV heads the shared arm is *not* worth taking at all — it is 2 blocks where the per-head kernel is
12 — so the sharing that helped a 40/8 model is a loss on the model this machine actually serves.
The split arm is not: at 4k it is 6x, because 2 KV heads × 32 chunks is 64 blocks instead of 2.

End to end on one real Qwen2.5-1.5B layer (`the_split_arm_preserves_a_real_layer`, step 680–699 of a
700-position history, per arm, not interleaved — the card throttles under a sustained loop, and this
is the honest label rather than a claim of a paired number): **1.195 ms/step per-head against
0.795 ms split, −34% of the layer step**. Attention is the part that grows with depth, so the
saving grows with it; the matvecs do not change.

```bash
cargo test -p vipera-backend-gpu --release --test attention_shared -- --ignored --nocapture
cargo test -p vipera-backend-gpu --release --test layer_gpu the_split_arm -- --ignored --nocapture
```

### Building a batched product's activations on the calling thread: 58 ms of a chunk — opt-in pool, 2026-11-20

`matvec_pooled_batch` builds the owned activations of a product (copy, group sums, Q8 quantiser) on
the calling thread, one entry after another, before it wakes any worker, so every worker is idle for
that time. A prefill chunk makes 196 such products (per layer: q/k/v at width 1536, o at 1536,
gate and up at 1536, down at 8960 — 168 products at 1536 and 28 at 8960), and the profile's Q4_K and
Q6_K phases include it, because the stamp wraps the whole call. `bench_activation_prep`, 128
entries, one thread, best of 5, two runs:

| width | `OwnedActivation::new` | of which group sums | quantiser | copy |
|---|---|---|---|---|
| 1536 | 0.163–0.165 ms | 0.045–0.047 | 0.109 | 0.014–0.015 |
| 8960 | 1.094–1.099 ms | 0.291–0.304 | 0.595–0.617 | 0.208–0.213 |

That is 168 x 0.165 + 28 x 1.1 = **~58 ms per chunk** of calling-thread time with the workers waiting:
5.5% of the 1045 ms 4k chunk and 8% of the 718 ms first chunk, at this sandbox's speed (a 1-vCPU
AVX-512 Xeon, not the i5-11300H).

`VIPERA_PARALLEL_PREP=1` (off by default) has the pool's workers build them: the calling thread copies
the entries into buffers (a worker cannot borrow its stack), hands each participant a run of entries
and keeps the last run, and each worker finishes its buffers in place (`OwnedActivation::from_buffer`).
It applies from 16 entries up, so a decode step or a short draft never pays a wake-up for it. **The
result is bit-identical**, entry for entry, to the serial build, because the same sums and the same
quantiser run on the same values (`prepared_activations_are_bit_identical_to_the_serial_ones`: Q4_K,
Q6_K and F32, batches of 1, 2, 7, 16, 32 and 33 entries, widths 32/96/1536, groups of zeros and of
negative zeros, worker counts 0/3/7). A worker that is gone or panics does not hang or half-fill the
batch: its run is built on the caller, or the call returns an error, as `fill_batch` does for rows.

**What was and was not measured.** On this 1-vCPU host the pool cannot add parallelism, so
`pool_prep_cost` only bounds the overhead: calling thread against pool 0.99x and 0.98x at one
participant, 0.98x and 1.03x at four time-sliced ones (1536 and 8960 wide), i.e. no cost for taking
the path. The gain is a parallel one and has to be measured on the rig: if eight participants split
the 0.9 ms of work at 8960, the product's preparation falls from about 1.1 ms to about 0.2 (copy)
+ 0.11 + a wake-up, and from 0.165 to about 0.065 at 1536, which over 196 products is roughly 40 ms
of a chunk, 4–6% — a projection, not a result. Prior polling and on-demand experiments in this file
found wake-ups costing what they saved on this part, so it is off until it is measured:

```bash
for v in 0 1; do VIPERA_PARALLEL_PREP=$v VIPERA_PROFILE_STEP=1 <the 4k prefill run>; done
cargo test -p vipera-backend-cpu --release --lib pool_prep_cost -- --ignored --nocapture
```

**What this entry also found, not changed.** The tiled kernels build their activation tile inside
every call on every thread (`bench_tiled_q4k_vs_q6k_shapes`, B = 128, one thread, row count 0 so the
call is the tile alone): 0.03–0.06 ms at width 1536 and 0.37–0.47 ms at 8960, for both Q4_K and Q6_K.
A down projection gives each of eight threads 192 rows, about 2.4 ms of kernel work at the single-thread
rate, so the duplicated tile is ~16% of that call and ~21 ms of CPU per thread per chunk in total.
Sharing one tile across the participants is the next lever of this kind, and unlike the preparation it
needs a barrier. And the two prefill kernels, single thread, 128 entries: gate/up 8960x1536 Q4_K
77.3 GMAC/s against Q6_K 54.6 (1.42x), down 1536x8960 Q4_K 94.8 against Q6_K 83.8 (1.13x) — the
Q6_K layers are only the FFN down and attention v, so the gap is worth about 16 ms of a chunk, not the
whole of the Q6_K phase's 15%.

### A chunk's attention with one walk per four queries: 7.7% of the phase, off by default — 2026-11-20

A prompt chunk calls `attend_entries` once per layer with one query row per position, and every row
walks the whole history. At 4093 positions that is 128 passes over the same K and V rows per layer,
~11 GB of reads for one 128-token chunk, against 37 GFLOP of dots and value multiply-adds that no
sharing can remove. The rows are identical and the positions are consecutive, so
`attend_rows_shared` walks the history once per block of four queries, and each query folds the same
sixteen gathered rows with `fold_chunk_block`, the kernel the position-chunked path already uses.
An entry sees positions below its own, so the rows past that are a prefix of the block it skips.

i5-11300H, `VIPERA_THREADS=8`, Qwen2.5-1.5B Q4_K_M, f16 KV, 4093-token prompt, 32 chunks, one run
per arm back to back, attention ms per chunk:

| chunk | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|
| per entry (today) | 362.6 | 375.8 | 390.4 | 405.3 | 420.2 | 434.0 | 445.1 |
| one walk per four | 336.3 | 349.6 | 360.8 | 373.5 | 386.5 | 398.2 | 410.6 |
| | 7.3% | 7.0% | 7.6% | 7.8% | 8.0% | 8.2% | 7.7% |

**The end-to-end number is not quoted as a win.** Chunk totals moved 1289.9 -> 1272.6 ms at the
last chunk, ~1.3%, but the second arm's Q4_K rose 602 -> 615 ms with nothing about it changed: the
box ran hotter, and that is larger than the delta. Attention at 34% of a 4k chunk bought back about
2.7% of the chunk, inside this ledger's 8-15% window spread.

**What the sharing can and cannot reach.** It removes the row traffic and leaves the arithmetic:
every (query, position) pair still pays 12 dots and 12 value multiply-adds of 128 values each. That
is why eight queries per block measures the same as four (304.9/317.3/331.1/326.3 ms at chunks
24-27) and why the gain stops at 7-8% rather than approaching the 4x the traffic falls by. Attention
here is bound by its own multiply-adds, not by memory.

**What the first attempt got wrong, since it is the obvious shape.** Folding one position at a time
kept four `heads x head_dim` accumulators hot and touched each of them once per position: 677.5 ms
of chunk 24 against 324.6 for the per-entry kernel, twice as slow with three quarters of the rows
gone. The V fold (`fold_chunk_block`, sixteen positions per call) is what makes the accumulator
traffic stop mattering, so the row blocking is load-bearing and the query blocking is the 7%.

Off by default (`VIPERA_ATTN_QBLOCK=1` turns it on) because it regroups the softmax sums - within
rounding, ~1e-7, of the per-entry kernel, the same policy as the position-chunked kernel - and 2.7%
of a chunk at 4k does not buy a change in a chunk's bits. What would keep it: a workload where
attention is a much larger share (a longer context than this one, or the sliding window on, where
attention is already split by position), or a machine whose f16 dot runs closer to peak than this
one's, where the row traffic is a larger fraction of the phase.

```bash
VIPERA_ATTN_QBLOCK=0 VIPERA_PROFILE_STEP=1 <the 4k prefill run>   # per entry
VIPERA_ATTN_QBLOCK=1 VIPERA_PROFILE_STEP=1 <the 4k prefill run>   # one walk per four
cargo test -p vipera-backend-cpu --release --lib qblock_attention_matches -- --nocapture
```

The kernel is held to the per-entry one by `qblock_attention_matches_the_per_entry_kernel_within_rounding`
(9 entries, 260+ positions, a causal tail where the queries of one block see different histories).

**Where the prefill gap actually is, measured in the same run.** A 4k chunk is 1288 ms: Q4_K 47-50%
(602 ms), Q6_K 14-15% (181 ms), attention 34%, the rest under 5%. 128 tokens in 1288 ms is 99 tok/s,
against the 159-168 tok/s this ledger's llama.cpp rows record for `pp4096` on this rig: 0.59-0.62x.
Even a free attention pass would leave 0.8x, so this kernel is not the prefill's problem.

**What the remaining 0.62x is not.** "The ggml repacked GEMM is slower than the tiled Rust kernel"
(2026-11-15) already answered the obvious reading: llama.cpp's Q4_K microkernel is *slower* per thread
than this engine's, four cores are within a few percent of what they can deliver for it, and "any
further prefill work has to reduce work or threads, not tighten the loop". So the gap is not a kernel
to port and not a loop to widen. What is left, untested, is work: this chunk multiplies every weight
once per token for 128 tokens, and llama.cpp's `pp4096` advantage has to come from doing less of it
per token - chunking, residency, or a batched path this engine does not have. That is a design
question, not a constant.

**Axis, because the names collide.** K and V rows are already shared across a head group by default
(`9abfa5e`, `a31d5ee`). This kernel shares them across *queries* instead - a different axis, which is
why it is a second kernel and why its gain (7.7%) is smaller than the traffic it removes.
## Four-row decode restructure — 2026-11-21 (rejected: 1.17x slower on re-measure)

Hypothesis: four consecutive Q4_K/Q6_K rows in one loop body give four independent DRAM streams and
load the activation bytes once per group, so decode should rise from 29 GB/s toward 38.

Method: one binary, `VIPERA_ROW_KERNEL=0` (one-row) against `=1` (four-row, switch on the new
`fill_four_rows`), arms alternating direction across rounds so neither arm always runs on the warmer
cache. Per run: median of the first ten decode steps, `VIPERA_THREADS=8`, fixed prompt,
`--temperature 0`, `VIPERA_PROFILE_STEP=1`. Absolute step time moved 33.8 → 38 ms between sessions
(the machine is never quiet here), so only the ratio is meaningful.

The first pass over this patch reported the arms as level (1.009x). Re-measured after rebuilding the
same source, with the order reversed in half the rounds, it does not reproduce:

| arm | median step (4 rounds) | q4k | q6k | vs one-row |
|---|---|---|---|---|
| one-row | 36.93 / 38.27 / 39.35 / 39.46 ms | 24.0–25.2 ms | 12.4–13.9 ms | 1.000 |
| four-row | 43.55 / 44.37 / 44.52 / 44.68 ms | 25.5–26.4 ms | 17.3–18.1 ms | **1.17x slower** |

The regression is concentrated in Q6_K (+34%), which is the kernel the codegen gate already showed
spilling: 27 stores and 17 loads of `ymm` through `%rsp` per call. Q4_K pays about 6%. The token
stream is still identical — 64 greedy tokens, `sha256` equal on both arms — so this is a speed
result, not a correctness one.

Codegen, `tools`-free check on the monomorphized sections:

- `q4k_dot_4rows_vnni`: 848 instructions, 3756 B, **zmm 0, zero vector traffic through `%rsp`**,
  32 `vpdpbusd` (8 × 4 rows), 36 `vfmadd`, 43 branches (8 inlined `f16_to_f32` × 5 + 3 loop).
- `q6k_dot_4rows_q8`: 765 instructions, 3303 B, zmm 0, `vpmaddubsw` 20, but **27 stores / 17 loads of
  `ymm` through `%rsp`**. `q6k_dot_4rows_q8` is `target_feature("avx2")`, i.e. 16 vector registers;
  four rows of (accumulator + `qh` + temps) do not fit. Zeroing that needs an `avx512vl` variant
  (32 registers) or two rows at a time, both of which change what is being measured.
- Replacing the twelve per-block scalar `f16_to_f32` calls with `vcvtph2ps` (bit-identical: the
  exhaustive 65536-pattern test already proves the two agree) cut branches to 2 and 3, but introduced
  17/6 and 23/13 stack traffic and measured 2.4% slower. Reverted.

Verdict: **row blocking loses on this loop**, and the earlier neutral label was measurement noise on
this machine. The four-row kernels and the `VIPERA_ROW_KERNEL` switch are removed from the tree; the
one-row loop stays the only decode kernel. The stack traffic above is the mechanism, not a mystery:
four rows do not fit in the 16 vector registers an `avx2` function gets, so the Q6_K arm loads and
stores its accumulators every run.

Lesson for the next hypothesis: `q4_k` is 61% of decode and streams at ~30 GB/s against a 43 GB/s
read ceiling, so the gap is not per-row scheduling. It is either the byte stream itself (layout and
prefetch distance, both already measured negative) or the balance of `q6_k`'s 36% — measure before
writing another loop.

## Parallel activation prep on the rig — 2026-11-21 (neutral, stays opt-in)

Hypothesis from the 2026-11-20 entry above: building a product's activations on the pool's workers
should recover part of the ~58 ms per 4k chunk that `matvec_pooled_batch` spends preparing them on the
calling thread, worth 4–6% of a chunk once the workers are real.

Method: the switch already existed (`VIPERA_PARALLEL_PREP`, default off). Same release binary, same
model and prompt set, `VIPERA_THREADS=8`, `prefill_vs_decode` from `win_vs_llama`; arms run back to
back in alternating order because the machine's absolute rate moved 157 → 180 tok/s between windows.
Threshold fixed before measuring: **pp512 ≥3% faster, with pp128 not regressing**.

| arm | pp512 (tok/s) | pp128 (tok/s) | decode ctx 32 (ms/step) |
|---|---|---|---|
| off | 172.1 / 174.9 / 170.1 (median 172.1) | 180.2 / 183.8 / 175.3 | 29.5–29.9 |
| on | 180.3 / 176.4 / 176.6 / 173.8 (median 176.5) | 188.8 / 184.1 / 182.3 / 179.2 | 29.8–30.4 |

Paired in measurement order: +4.8%, +2.5%, +1.0%, +2.2% → median **+2.4%** on pp512. pp128 is level
(+4.8%, +2.2%, −0.8%, +2.2%, median +2.2%) and decode does not move. That is below the 3% gate and the
projection was optimistic; the switch stays off by default and no default changes.

Reading: the work is real but small on this part. 196 products x 0.165 ms at 1536 is 32 ms of a
1.0 s chunk, and only the preparation moves — the copy still runs on the caller for 16+ entries, and
the workers' wake-up is paid per product. The 8960-wide `down` products (1.1 ms each) are where the
remaining gain would be, but they are 28 of 196 calls and a fifth of the prep time, so the ceiling is
of the same order as the noise.

## Sharing the prefill activation tile across threads — 2026-11-21 (closed by measurement: 1.3% ceiling)

Hypothesis from the entry above: `fill_rows_batch` hands every worker its own row range, and each one
enters `q4k_matvec_tiled_range`, which builds the batch-wide activation tile (`act`/`dxs`/`sumt`, 192 KB
at B=128 and 1536 wide, 1.1 MB at 8960) before it touches a weight byte. Eight threads therefore build
the same tile eight times per product. Building it once on the caller and lending it to the workers
would remove seven of those eight copies without needing a barrier, since the tile is read-only.

Measured before writing that plumbing, because the ceiling is what decides whether it is worth it.
`bench_tiled_q4k_vs_q6k_shapes`, B=128, release, one thread, `rows 0` so the call is the tile alone
(best of three, 2026-11-21):

| shape | q4k tile | q6k tile |
|---|---|---|
| 8960x1536 (gate/up) | 0.03 ms | 0.03 ms |
| 1536x8960 (down) | 0.17 ms | 0.21 ms |

A Qwen2.5-1.5B layer is q, k, o, gate/up at 1536 wide on Q4_K (4 x 0.03), v at 1536 wide on Q6_K
(0.03) and down at 8960 wide on Q6_K (0.21): **0.33 ms per thread per layer, ~9 ms per 128-token chunk
per thread**. That chunk takes ~0.73 s at the 176 tok/s pp512 measured in the entry above, so removing
seven copies of the tile is worth **~1.3% of a chunk at most**, and less in practice: the remaining copy
is built on the calling thread while the workers wait for their jobs, and the workers' own share is
only 9 ms of a ~90 ms slice each.

That is below the 3% gate this session pre-registered for prefill work, and below the 2-5% spread this
box shows between identical arms (see the parallel-prep table), so no implementation is recorded: an
unmeasurable change cannot be reported as a win, and the plumbing (a borrowed tile through `Job`'s
lifetimes in `pool.rs`) is not free. The 2026-11-15 sweep already said the same thing from the other
side — B=32 beats B=128 by ~8% because the tile is a reuse/latency trade, not a residency cliff.

What would change the verdict: a batch wider than 128 (the tile grows linearly with B, the chunk does
not), a machine with more cores than this 4C/8T part (the copies scale with threads), or a shape whose
`cols` is large enough for the 0.17-0.21 ms tile to matter against its row count. Until one of those
holds, the tile stays rebuilt per thread.

## Where a prefill chunk's last 8% goes — 2026-11-21

`VIPERA_PROFILE_STEP` named five phases and lumped everything else into `rest`. On a 128-token chunk
that remainder was 6% (41 ms) with no way to say what it was, so the profiler now splits it into
`proj` (the bias adds and per-head Q/K norms after the projections), `rope`, `kv_write` and `elem`
(the residual adds and the gated FFN product). Same run, same rig as the entry above
(i5-11300H, 8 threads, `qwen2.5-1.5b-instruct-q4_k_m.gguf`, ~128-token chunk):

| phase | ms/chunk | share |
|---|---|---|
| q4k | 442-508 | 65-66% |
| q6k | 137-160 | 20-21% |
| attention | 39-61 | 5-8% |
| **elem** | **28-33** | **4%** |
| norm | 12-14 | 2% |
| rope | 3.9-4.5 | 0.6% |
| kv_write | 3.1-3.6 | 0.5% |
| proj | 1.7-2.0 | 0.3% |
| rest | 0.9-1.1 | 0.1% |

The bucket is now empty, which is the point: the phases that were hiding in it are all on the calling
thread, one element at a time, and together they are **56 ms, 7.4% of the chunk**. Decode, same build:
`elem` 1.25 ms, `norm` 0.59, `rope` 0.18, `kv_write` 0.16 of a 36 ms step — the same work, 32x smaller,
and not worth touching there.

**`silu_mul` alone does not clear the gate.** It is already the AVX2 eight-lane kernel (bit-identical to
the scalar, `silu_mul_matches_the_scalar_one_bit_for_bit`), so what is left is that it runs on one
thread: 32M `exp`+divide per chunk of a 762 ms chunk. Spread over eight scoped threads at the call site
it goes 32.4 -> 19.1 ms, i.e. **2.0% of the chunk including the ~3 ms of thread startup** the crude
version pays 56 times a chunk. Under the gate, so nothing was kept; the crude arm was reverted.

What the numbers do support is the mechanism, not the one kernel: 7.4% of a chunk is serial
elementwise work that the pool spends idle for. Moving `elem`, `norm` and `rope` onto the crew that
already exists costs one job per worker per phase (280 messages a chunk, ~0.3 ms) instead of 56 thread
starts, and 3x on 56 ms is **~4.5% of a chunk** with the chunk unchanged otherwise. That is the next
prefill lever; the tile-sharing idea closed in the entry above stays closed.

Two things this measurement does not change: q4k at 720 GFLOPS and q6k at 755 against the 776 the
kernel scored on its own are still the wall, and `rest` was never the problem — it was the label.

End to end, this costs nothing: the same build, 510 tokens in four full chunks, the chunk sum is
2.85 / 2.90 / 2.98 s, i.e. **171.8 / 176.6 / 179.8 tok/s pp512 (median 176.6)** against the 172.1 /
176.5 of the 2026-11-16 entry, and profiler on and off are the same wall clock (3.63 s for the whole
process, both arms). `scripts/bench-4k.sh` at `PROMPT=512` prints 134.8 for this prompt, which is the
script's arithmetic and not the engine: 52 sentences tokenise to 520, so there is a fifth chunk of 8
tokens that still costs 606 ms, and the script divides the sum of all five by 512.

### The crew the numbers pointed at does not survive the borrow checker — 2026-11-21

The obvious reading of 7.4% serial is "hand those phases to the pool, it is idle anyway". Written that
way it does not compile, and the reason is structural rather than a mistake in the write-up: a thread
that is started once per pass is reached through a channel, and a channel has one message type, so the
borrows inside that type get one lifetime — the pass's. `&mut batch.h` handed to such a channel has to
stay borrowed for the whole pass, and then the `matvec_batch` calls that follow cannot touch `batch.h`
at all (E0502/E0499 on the first phase that is followed by a product, `model.rs:981-997` for the norm
and `:1160` for the residual add). Starting the crew per call instead is what the 2.0% arm already
measured: it borrows fine and does not pay.

`pool.rs` says the same thing in its own doc comment, from the other direction — "handing a worker
`&mut [f32]` would make that borrow last as long as the pool, and the caller could not touch its own
output buffer afterwards" — and that is why the pool's matvec results come back as owned vectors over
a channel. Elementwise output is too big to copy: silu's 4.6 MB per layer, 28 layers, is ~14 ms a chunk
at the measured copy bandwidth, which is more than half of the 32 ms it would save.

So the crew arm is recorded as **not implementable in this workspace's safe Rust**, not as a win that
was skipped, and `elem`, `norm` and `rope` stay on the calling thread. The one route that needs no copy
is folding the transform into the preparation job that already copies each activation into a worker
buffer — but that path only runs behind `VIPERA_PARALLEL_PREP` (off by default, and the norm would be
recomputed once per product that shares it, three times for the attention projections), so it is a
different hypothesis with its own measurement before it, not this one carrying the same 4.5% ceiling.

## Ledger sweep on 2026-11-21: what still matches, what does not — 2026-11-21

Every phase this file records for a 128-token chunk, re-measured in one sitting on the same box, same
model, 8 threads. The box was shared (load average 3.5, a browser and the agent itself on 8 threads),
which is worth ~3% here — that is the size of the drift below, not a regression:

| phase | recorded | today | verdict |
|---|---|---|---|
| chunk total | 762-776 ms | 773-792 ms | +2% box load |
| q4k | 442-508 | 515-533 | +5%, top of the band |
| q6k | 137-160 | 154-159 | inside |
| attention | 39-61 | 38.6-41.4 (chunk 1), 55-73 (deeper) | inside |
| elem | 28-33 | 33.9-36.2 | +8%, still 4-5% |
| norm | 12-14 | 14.0-14.5 | top of the band |
| proj | 1.7-2.0 | 1.84-2.44 | top of the band |
| rope | 3.9-4.5 | 4.29-4.65 | inside |
| kv_write | 3.1-3.6 | 3.08-3.83 | inside |
| rest | 0.9-1.1 | 0.96-1.30 | inside |

Decode is the one entry that does not reproduce: the profile prints 82-92 ms for the first two steps,
while the wall clock says 18 steps on top of a 470-token prefill average **42 ms/step**, and the first
steps carry the page faults of a 1.1 GB mapping. The 36.4-38.4 ms recorded earlier came from a quieter
box and a second or third step; the number to compare against is the wall clock, and today it is 42.

### `VIPERA_PARALLEL_PREP=1` is worth 3.8% and is still off by default

Interleaved pairs (off, on), chunk 1, medians of three: **781.4 -> 751.8 ms, -3.8%**, and the same
direction in 3 of 3 pairs and in chunks 2 and 3 (813.6 -> 782.8, 709.4 -> 685.0). Prep runs 196 times a
chunk, and the pool is idle for each of them: what the switch buys is exactly the 0.16 ms (width 1536)
and 1.1 ms (width 8960) the calling thread spends turning activations into owned quantised ones.

Decode is untouched by it (`PARALLEL_PREP_MIN_BATCH` is 16, a decode step has 1 to 8 entries), and it
is bit-identical — `prepared_activations_are_bit_identical_to_the_serial_ones` checks the split, not
just the total. So the switch is a default waiting to be flipped on evidence from more than one box,
not a new kernel; it is also the only place the norm/silu fusion of the entry above could live.

### The other two levers, bounded

**Attention is the one with room left.** Its share of a chunk is 5% at depth 128 and 8% at depth ~380,
and the two switches already in the tree account for 8-17% of that (both off: 42.3 / 72.2 / 66.2 ms;
defaults: 40.1 / 64.9 / 55.1 ms). What is left is not bandwidth: the K/V a chunk of 128 queries reads
at depth 380 is 28 layers x 380 tokens x 1 KB = 10.6 MB, i.e. 0.27 ms of the 55. The arithmetic is
~8.4 GFLOP (QK^T and AV, 12 heads x 128 queries x ~380 keys x 128 dims x 2), so the kernel runs at
**~150 GFLOPS, a fifth of the 720 the Q4_K matvec reaches**. A tile-resident micro-GEMM with the
softmax fused would have to beat 150 by 2.5x to be worth the gate; that is ~4% of a chunk at the top
end and needs a new kernel, not a switch.

**`proj` has none.** 1.84-2.44 ms a chunk is **0.3%**: the bias adds and the per-head Q/K norms after
the projections are already vectorised, and even deleting them entirely would not clear the 3% gate.
NO-GO without writing anything, and the same holds for `rope` (0.6%) and `kv_write` (0.5%).

## `VIPERA_PARALLEL_PREP` on by default — 2026-11-21

Same box (i5-11300H, 8 threads, `Qwen2.5-1.5B-Instruct-Q4_K_M`), same prompt as the phase-split
ledger (24× "the quick brown fox...", 128-token chunks, ctx 1024, seed 1, `--max-tokens 20`,
`VIPERA_PROFILE_STEP=1`). Interleaved (on, off) × 3, median of chunk totals:

| chunk | on (default) | off (`VIPERA_PARALLEL_PREP=0`) | delta |
|---|---:|---:|---:|
| 1 | 721.77 ms | 774.13 ms | -6.76% |
| 2 | 757.23 ms | 797.40 ms | -5.04% |
| 3 | 664.77 ms | 695.73 ms | -4.45% |

Direction held in 3/3 pairs for every chunk. Decode (batch 1-8, below `PARALLEL_PREP_MIN_BATCH=16`)
is unchanged by construction; decode profile steps 2-4 stayed in the same noise band (on median
~97.6/105.3/89.6 ms vs off ~94.7/107.2/88.8 ms).

Change: `parallel_prep_enabled()` now defaults to on; set `VIPERA_PARALLEL_PREP=0` (or
`false`/`off`/`no`) to restore the serial activation build. Result is bit-identical either way;
`prepared_activations_are_bit_identical_to_the_serial_ones` covers the split.

## Folding the norms and the SwiGLU multiply into the preparation job — 2026-11-21

On by default (set `VIPERA_PREP_FUSION=0`/`false`/`off`/`no` to keep the transforms on the calling
thread). The chunk path used to do three passes on the calling thread between
products: `attn_norm` and `ffn_norm` (14.0-14.8 ms a chunk) and `silu_mul` (30-37 ms). Both are
elementwise and read nothing but their own inputs, and a batched product already copies each
activation into a worker's buffer — so the transform, the group sums and the quantiser now run in
that copy, on the pool, while the caller would otherwise sit on a serial pass. The norm weight
travels as an `Arc<[f32]>`, the gate rows as a shared `Arc<[f32]>` (a job travels with the pool's
lifetime, and the caller's scratch is shorter-lived than that).

Ceiling measured first, by making both passes no-ops in a throwaway build: chunk 654.5 / 674.5 /
670.6 ms against 693.1 / 733.8 / 696.8 ms, i.e. `elem` 34 -> 3.8 ms and `norm` 15.2 -> 3.0 ms. That
is ~6% of a chunk, which is what made this worth writing rather than a 1.9% norm-only fusion.

Interleaved (on, off) x3, chunk totals:

| pair | chunk 1 | chunk 2 | chunk 3 |
|---|---:|---:|---:|
| 1 | 727.48 vs 770.83 (-5.6%) | 761.74 vs 799.06 (-4.7%) | 673.56 vs 706.59 (-4.7%) |
| 2 | 745.03 vs 752.31 (-1.0%) | 780.94 vs 783.62 (-0.3%) | 689.79 vs 695.78 (-0.9%) |
| 3 | 738.62 vs 776.93 (-4.9%) | 765.11 vs 839.45 (-8.9%) | 676.65 vs 740.17 (-8.6%) |

Median pair delta **-4.7%**; chunk 1 medians 738.62 vs 770.83 ms, **-4.2%**. The spread is the box,
not the arm: pair 2 ran while the machine was busiest. What the profile says moved is exact — `norm`
14.7 -> 0.01 ms and `elem` 37.0 -> 4.9 ms, with the same 40 ms reappearing inside `q4k`/`q6k`, because
the prep the workers now do is charged to the product that owns it. The win is the caller's idle
window, not less total work, so it is bounded by how much of those 45 ms was serial on a saturated
pool.

Bit-identity: `cmp` on a 32-token `--temperature 0 --seed 1` run is clean between the arms, and
`rms_norm_in_place_is_rms_norm` (bits, 1536 values) plus
`a_fused_activation_matches_the_transform_built_serially` (values, sums and quantised activations,
both transforms, two entries) pin the two pieces the fusion moved.

Default flipped after the same-box measurement above; pair 2 shows how much a shared machine can eat
of it, so keep `VIPERA_PREP_FUSION=0` handy if a box regresses.

## The 4-query attention walk is worth 4-6% of a chunk — 2026-11-21

Behind `VIPERA_ATTN_GEMM=1` (alias of `VIPERA_ATTN_QBLOCK`). **This is not a new kernel**: the
4-query shared walk of `attend_rows_shared` with `QBLOCK=4` is already in the tree, and has been
since the position-chunked work. What it never had was a measurement at chunk depth, or a default:
the 2026-11-20 entry measured it at 4k, where it took attention from 435-445 ms to 399-411 ms of a
chunk and a chunk from 1280 to 1261-1273 ms — inside the window spread, so it stayed off.

Measured here at the depth a chat prompt actually runs (128-token chunks, ctx 1024, history 128 /
256 / 384) it is not marginal. Interleaved (on, off) x3, on `main` as it stands after #55-#57
(`VIPERA_PARALLEL_PREP` and `VIPERA_PREP_FUSION` both on):

| pair | chunk 1 | chunk 2 | chunk 3 |
|---|---:|---:|---:|
| 1 | 649.03 vs 681.21 (-4.7%) | 672.79 vs 713.98 (-5.8%) | 599.35 vs 626.94 (-4.4%) |
| 2 | 649.09 vs 672.05 (-3.4%) | 668.80 vs 707.59 (-5.5%) | 596.88 vs 622.45 (-4.1%) |
| 3 | 653.01 vs 668.51 (-2.3%) | 672.92 vs 706.39 (-4.7%) | 597.46 vs 621.25 (-3.8%) |

Median: chunk 1 **-3.4%**, chunk 2 **-5.5%**, chunk 3 **-4.1%**; 9 of 9 pairs negative. Attention
itself, medians: 22.8 / 35.6 / 38.6 ms against 42.4 / 69.1 / 59.8 ms — **-46% / -48% / -36%**.

Rate, counting the full 128-query x history matrix (the causal mask makes the kernel do about half
of it): 66.5 -> **124 GFLOPS** at depth 128, 81.7 -> **158** at 256, 141.4 -> **219** at 384. The
384-deep chunk is the one the earlier entry timed at ~150, so the kernel is now ~1.55x that, and
attention's share of a chunk falls from 6.2-10.4% to 3.5-6.4% — what remains is 23-39 ms that a
perfect kernel would have to keep finding, which is why there is no second kernel in this phase.

An earlier run of the same A/B, before #57 flipped the prep fusion on, read -4.6 / -5.7 / -4.0%: the
arms are within a point of each other, which is what the ledger's noise window predicts.

Combined queue pass (same machine/prompt, ×3 interleaved, on current main defaults): all three
levers on (`PARALLEL_PREP=1 PREP_FUSION=1 ATTN_GEMM=1`) vs all off. Medians: chunk 1 605.46 vs
674.73 ms (**-10.3%**), chunk 2 620.96 vs 704.86 ms (**-11.9%**), chunk 3 554.97 vs 614.32 ms
(**-9.7%**), 9/9 negative. This is a stack sanity check, not a replacement for the per-lever A/Bs.

Parity: `cmp` is clean between the arms on a 32-token `--temperature 0 --seed 1` run at these
depths. That is evidence, not a guarantee: the walk regroups the softmax sums and is documented as
within rounding (~1e-7), not bit-identical, which is why it ships behind a switch rather than on by
default, same policy as the position-chunked kernel and `VIPERA_ATTN_GQA_SHARE`.

Still off by default: it changes the summation order, and that is a numeric-policy decision, not a
speed one.
