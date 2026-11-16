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
