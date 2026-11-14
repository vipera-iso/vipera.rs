# Benchmarks

## Template

Chạy các lệnh sau trên từng cấu hình hardware:

```bash
vipera profile > profile.json
vipera plan -m model.gguf --json > plan.json
vipera verify -m model.gguf --json > verify.json
```

Ghi rõ:
- GPU model, VRAM, driver version
- llama.cpp version
- OS + kernel version
- Model GGUF (arch, quant, size)

## Kết quả (điền sau khi chạy)

| GPU | Model | n_gpu_layers | Predicted VRAM | Actual VRAM | Predicted RAM | Actual RAM | Throughput | Error VRAM | Error RAM | Pass? |
|-----|-------|--------------|----------------|-------------|---------------|------------|------------|------------|-----------|-------|
| GTX 1650 4GB | Qwen2.5-1.5B-Q4_K | 14 | 3200 MB | ? | 2100 MB | ? | 6.5 t/s | ? | ? | ? |
| RTX 3060 12GB | Qwen2.5-7B-Q4_K | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| RX 6600 8GB | Qwen2.5-7B-Q4_K | ? | ? | ? | ? | ? | ? | ? | ? | ? |

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
