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
