# Porting repack/kernels: what of the 12 forks is portable to vipera.rs

Date: 2026-11-23 · Branch: `perf/q4k-repack-8x8` · Status: the queue is measured and ranked, no workstream opened yet.

This document answers one question: of the serious CPU-inference projects in `~/llm-forks`, which pieces of
*kernel machinery* are portable to `crates/vipera-backend-cpu`, how much are they worth, and what would
refute them. Every background number was measured on this machine (i5-11300H, 4c/8t Tiger Lake,
AVX-512 VNNI); anything inferred is marked as such.

## 1. Conclusions first

1. **Closed by measurement:** llama.cpp's repack layout (`block_q4_Kx8` + 8-row gemv + 8x8 GEMM). Prefill
   with `GGML_CPU_REPACK=ON` is **0.74x** of OFF; the 8-row gemv is 1.34x at one thread and **1.06x at
   eight** (decode is bus-bound). Not ported; `q4k8_lab.rs` kept as evidence.
   *Update 2026-11-24:* the kernel **was** ported into `simd.rs` and wired into the engine, but is **off** by
   default (`VIPERA_Q4K8`, decided in `ADR-006`); the measurement conclusions above are unchanged.
2. **Still open, and the only big one:** unlock ik's epilogue of **one float op per 256-block** — but the
   way in is **not** "port an integer epilogue" (section 3.1: with our one activation scale per 32 values,
   four vector ops per `(row, sub-block)` is a **floor** that cannot be cut). The way in is **changing the
   activation quantization granularity**: llama/ik's `block_q8_K` has **one** scale for all 256 values, ours
   has eight. That is an arithmetic change, not a kernel change.
3. **That ceiling is measured, and so is the price** (2026-11-23): the kernel is **+1.24..1.36x** when the
   epilogue drops to one op per block (the `PER_BLOCK_EPI_PROBE` probe), implying **+14..20% chunk at eight
   threads**; in exchange NLL worsens by **+0.3..0.7% on every text** (per-128 and per-256 are of the same
   order). So the largest remaining prize *cannot* be bought with perf money — it is an accuracy decision for
   the owner (`ADR-005`), see W-1.
4. **Eight threads lose nothing to scaling.** Measured today: the q4k phase at 8T is **838.7 GFLOPS** =
   **209.7 GFLOPS/core**, against 218.4 for the isolated single-core band → 96% efficiency, and scaling from
   202 to 209.7/core is **4.15x on four physical cores**. So every improvement must come from *single-core
   kernel speed*, not from chunk splitting or the thread pool.
5. **What cannot be ported:** anything belonging to i-quants (IQ2/IQ3/IQ4/trellis), NVFP4, tensor
   parallelism/MoE expert caches, and rewrites in other languages. The model that runs is Q4_K/Q6_K, GGUF.

## 2. The project ledger

Scanned `~/llm-forks` (17 directories). The "machinery" column counts only CPU kernels for K-quants, not
GPU/MoE/quant types.

| project | CPU K-quant machinery | worth porting? |
|---|---|---|
| **ik_llama.cpp** `1180df7` | `iqk/iqk_gemm_kquants.cpp`, 5,337 lines: `mul_mat_q4_k_r4_q8_k` (:1497), `mul_mat_qX_K_q8_K_AVX512` (:381), `process_min_r4_b32`, `iqk_convert_q4_k_q8_1_r8` (:2335) | **Yes** — see section 3, M2/M3 |
| **Strata / PolyStrata** (`engine/third_party/strata-ggml`) | `iqp.cpp`, 1,253 lines: an 8-row decode panel into `int8` (`block_iqp_x8`), integer GEMM, `GGML_IQP_MIN_BATCH 8`; `arch/x86/repack.cpp` is a verbatim copy of llama.cpp's | **Partly** — M4 (staging panel), only if M2 is not enough |
| **llama.cpp** `a8f8294c4` | `arch/x86/repack.cpp`, 6,407 lines (`block_q4_Kx8`), `tiled/tiled-kernel.cpp` (`ggml_compute_forward_mul_mat_tiled`, **compiled but never called** in this build) | **No** — measured at 0.74x |
| **basalt** | `src/kernels/cpu/kq_avx2.cpp`: `q4k_dot` for `nt` tokens around **one** weight stream, claimed *bit-exact* with `ggml_vec_dot_q4_K_q8_K`; `expert.cpp`/`pool.cpp` for MoE | **No** — same idea as our band (16 columns per load), no new mechanism; but it is a precedent for "batching by token need not change the arithmetic" |
| **cpu-vega-llama.cpp** | carries `llamafile/sgemm.cpp` (tinyBLAS, f32) + `iqk_mul_mat` | **No** — f32 for K-quants means dequantize then GEMM; 2.62 GFLOP/token in f32 cannot beat `dpbusd` |
| koboldcpp, ktransformers | repackaged llama.cpp copies | No |
| llama.cpp-glm52, llama-optimize, llama-cpp-optimization-lab, llama-optimus, extended-strata-moe, ik_llama-numa-mirror | 0 hits for `block_q4_Kx8` / `dpbusd_epi32` / `tiled-kernel` | No |
| exllamav3, mlx-lm, tabbyAPI | GPU / Apple / API | No |

## 3. Seven mechanisms, priced by their measured ceiling

Ceiling notation: `q4k` = 74% of the chunk at 8T, `q6k` = 17%, `attention` = 6% (measured today, 128-token
chunk, 8T).

| # | mechanism | source | upstream number | ceiling in vipera.rs | state |
|---|---|---|---|---|---|
| **M1** | Repack weights at load: rows interleaved by 4/8 bytes, one load serving 4-8 rows | llama `repack.cpp`, ik `iqk_quantize` | llama: pp512 0.74x when on; ik: pp512 268.7 @8T | our `NO_WEIGHT` probe: +29% (8960x1536) / +56% (1536x8960) kernel → but needs an **on-disk layout** + loader | **Closed** for decode (bus-bound, 1.06x); prefill still possible, see W-3 |
| **M2** | 6-bit scales applied in the `i32` domain: `packs_epi32` + `dpwssd`, one `cvt`+`fma` for all 256 values | ik :381, :1497 | Q6_K dot 2.84 → 1.97 cycles per 32 values (#27590, different source) | needs **one activation scale per 256** to be usable (section 3.1); then the epilogue goes 21-23% → ~4% → **+11..15% chunk at 8T** | **Open — W-1, together with the granularity change** |
| **M3** | Min term hoisted out of the loop, using the activation's `bsums`, once per 256 block | ik `process_min_r4_b32` | inside the same kernel | folded into M2 (the same 21-23%) | **Open — W-1** |
| **M4** | `int8` staging: decode 8 rows into a panel, then run an integer GEMM over all columns | Strata `iqp.cpp` | `MIN_BATCH 8`, `SB_SIZE 16` | q4k, but needs batch ≥8 and 2 KB scratch per 8 rows; breaks bit-identity with the GGUF path | **Waiting** — only if W-1 lands and q4k is still < 240 GFLOPS/core |
| **M5** | `tiled/`'s 16x16 microkernel | llama `tiled-kernel.cpp` | no upstream number (never wired into `ops.cpp`) | different shape from the band (8-10 rows x 16 columns); no new mechanism | **Closed** — no evidence to port |
| **M6** | Tiled flash-attention | ik `iqk_flash_attn.cpp`, llama `ops.cpp` tiled FA | — | attention is 6% at 512, ~43% of a step at ctx 4096 (inferred from the pp1024→pp4096 slope) | **Open — W-4** (ported once already, see `idea-sources.md`; re-measure before porting again) |
| **M7** | tinyBLAS f32 GEMM for prefill | llamafile `sgemm.cpp` | — | requires dequantizing Q4_K→f32 (4 B/weight) then a GEMM; 7.5x the bytes | **Closed** |

A note on M2/M3, the part that is easy to misread: ik's kernel is *not* faster because it "does not
broadcast" — our band does not broadcast either (`vpbroadcastd` memory-source, one uop, serving 8 rows). What
ik does differently is **move the scale and min out of the float domain**: `packs_epi32` + `dpwssd` with an
`i16` scale, then *one* `fmadd` per `(row, 256 block)`.

### 3.1 Why our epilogue is a floor of four ops, and ik's is not

What makes that possible is **not** the instruction but the **activation granularity**:

| | activation scale | float ops per `(row, 256 block)` |
|---|---|---:|
| llama.cpp / ik `block_q8_K` = `{float d; float sum; int8_t qs[256]; int16_t bsums[16]}` | **1** | ~2 |
| vipera.rs `QuantizedX` = `{q: Vec<i8>, d: Vec<f32>}` — one `d` per 32 | **8** | 32 (fused: 4 x 8) |

The operand to compute for each `(row, s)` is
`facc += (d_row·scale_s·acc_s − dmin_row·min_s·sum_s) · y_d_s`, and `y_d_s` **changes per sub-block**. So
every `s` forces its own float step: `cvt` + `mul` (times `dxv`) + `fma` + `fnmadd` = 4 ops — exactly the
shape of the current `fused_epi`. Accumulating in `i32` and multiplying once at the end is **only correct
when `y_d` does not change with `s`**, i.e. when the activation has a single scale for all 256. That is why
ik can write one `cvt`+`fma` per block, and also why an "integer epilogue port" that keeps `QuantizedX`
would be **slower**: `mullo`+`add` (2 instructions) per `s` **plus** the float step that is still there.

Conclusion: to collect the 21-23% epilogue you must change the **activation quantization** for the tiled
path, not the kernel. The price is accuracy (activations 8x coarser in granularity), and that is an
arithmetic decision of the same kind as ADR-003, so it needs its own ADR + NLL numbers, not a perf PR.

## 4. Five workstreams

Each workstream follows the gates of the `vipera-perf-loop` skill exactly: code → review → release build →
codegen gate → bit/parity → bench → ledger. Land threshold: **≥3% chunk, median over ≥5 alternating pairs**.

### W-0 — Fix the 8T target before writing kernel code (half a day) — **deferred, no longer blocking**

Original purpose: measure ik on *our* scale to learn whether their +29% comes from q4k or from another phase
(`llama-bench -p 128,512,1024 -n 0 -t 8` on the ik build, then subtract the attention slope as was done with
llama).

This is **no longer a decision gate**, because the question it asked has been answered directly: the part
worth porting was isolated and *measured in place* (the `PER_BLOCK_EPI_PROBE` probe, +1.24..1.36x kernel), and
its price was measured too (NLL, W-1). Building ik's `llama-bench` is very slow on this machine
(`iqk_gemm_iquants.cpp` alone takes minutes; it stopped at 16% after ~8 minutes with `-j8`), so it was halted
and left as later confirmation work.

- Command used, should it need rerunning: `cmake -S ~/llm-forks/ik_llama.cpp -B build-bench
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON && cmake --build build-bench --target
  llama-bench -j8`.
- Remaining value: if ik beats us by <1.15x on q4k alone, then our probe is measuring the wrong place and
  `fill_rows_batch` (tile sharing, chunk width) must be re-examined before trusting W-1.

### W-1 — One activation scale per 256 for the band, then a one-op-per-block epilogue (2-3 days) — **priority 1**

**State: the measurement half is done (2026-11-23), the code half is unwritten.** Both halves of the
trade-off were measured without writing the kernel — see `docs/benchmarks.md`, "a per-256 activation scale is
worth 1.24-1.36x on the band, and costs 0.3-0.7% NLL":

| | measured |
|---|---|
| band kernel, one-op-per-block epilogue (probe) | **+1.24..1.36x** (152.8 → 189.5 GFLOPS on 8960x1536, 168.8 → 221.9 on 1536x8960) |
| chunk at 8T, inferred (q4k = 74%) | **+14..20%** |
| NLL, per-256 activations | +0.45..0.69% on every text (+0.013..0.020 nats) |
| NLL, per-128 activations | +0.33..0.53% (not meaningfully lighter) |

So this item **fails its own quality gate** (`|Δ| ≤ 0.5%`): three of four texts exceed 0.5%, none is below
0.3%, and the 128 mark does not rescue it. W-1 is therefore **no longer a perf PR**; it is an arithmetic
policy decision for the owner. The two code steps below happen only after the owner accepts that price, and
they must come with ADR-005.

Two steps, measured separately:

**W-1a — change the granularity (kernel untouched).** Add an activation quantizer for the tiled path: one
scale and one set of `bsums` per 256 values instead of the eight scales `QuantizedX` uses today. The band's
`dims` change from `dxs[(half, s, col)]` to `dx[(half, block)]`; the `i8` codes are still 256 values, and the
min term uses `bsums` per 16 like ik. Decode and the row kernel are **unchanged** (they still use per-32
`QuantizedX`), so the affected path is batch ≥ 16 only.

**W-1a — DONE (2026-11-24):** port — `TiledActivation` (quant.rs) + `q4k_matvec_tiled_range_a`/`try_q4k_matvec_tiled_range_a`
(simd.rs) + `fill_rows_batch` wiring + the `VIPERA_TILED_A` switch (OFF by default — numerics change → ADR-005). Chunk at 8T: **+2.6%**
(B7 baseline 33.83 → 34.72 ms, within the machine's ±3%; band ≈ unchanged by design). Parity:
`q4k_tiled_range_a_matches_batch_within_tolerance` — max diff 2.63 (tolerance `2.0 + 0.2·|want|`). The band lab bench was not
run (same as the missing `decode_ab.rs` — asked the user: keep the declaration).

**W-1b — ik's epilogue.** In `q4k_tiled_band` (`simd.rs:2660`): accumulate `i32` across the 8 sub-blocks with
`mullo` (or `packs_epi32` + `dpwssd` as in ik :381), with the min term also in `i32`, then **one** `cvt`+`fma`
per `(row, block)`. 32 float ops per `(row, block)` → 2.

- Why it cannot overflow: `|q| ≤ 15`, `|y| ≤ 127`, 32 values → `|Σ| ≤ 60960`; scale ≤ 63, eight sub-blocks
  added → `|isum| < 2^25`.
- Gates (in order; stop at the first failure): kernel `bench_prefill_band_gflops` ≥ +10%; chunk at 8T ≥3%
  median over 5 pairs; NLL on ADR-003's six texts — **a stricter threshold than ADR-003's**: this coarsens
  the activations rather than only reordering rounding, so it demands `|Δ| ≤ 0.5%` per text with no text worse
  by more than 0.01 nats; `q4k_tiled_range_matches_batch_vnni` green (existing tolerance `1e-3 + 1e-4·|ref|`).
  → **the NLL gate already failed on paper, before a line of code was written** (table above). Continue only
  if the owner relaxes that threshold deliberately, not by editing it to match the result.
- Expected (measured, not guessed): kernel +24..36%, chunk at 8T +14..20%.
- Rollback: W-1a is a new quantize function + one branch in `BatchActivation`; W-1b is a `bool` arm in the
  kernel like `fused_epi`. No on-disk layout, no loader, no decode touched.
- Refuted if: chunk <3%, **or** NLL worse than the threshold on any text — which already happened at
  0.3-0.7%. The current default is **keep per-32**, and in that case the question "collect the remaining
  21-23% epilogue" is answered: it cannot be collected without changing quality.
- Companion work: `ADR-005` (ADR-004 covered the attention q-block default) records the activation
  granularity for the tiled path, because it is user-visible policy (prefill and decode would quantize
  differently) and because it changes output on every prompt longer than 16 tokens.

### W-2 — Row kernel (decode) — **folded into W-1/ADR-005, no longer its own item**

Decode with one thread: llama 14.54 tok/s, ours 10.8 (1.34x). At `nrc = 1` the epilogue is divided by fewer
`dpbusd` than the band's, so the ratio is *higher*, and this used to look like an independent workstream.

Reading `q4k_dot_vnni` (`simd.rs:3311`, the real B=1 path — a batch of one does not enter the tiled kernel)
shows it has **the same dependency**: `coefficients = scale_at · dx · d` is an **8-lane vector** because the
weight scale is 6-bit *per sub-block*, not because of the activation. So even in decode, the per-sub-block
float multiply can only be merged when `dx` is one scale per block — exactly what ADR-005 must decide.
Otherwise W-2's remainder is bounded by bandwidth: Q4_K 1.1116 G × 4.5 bits + Q6_K 0.4316 G × 6.5625 bits ≈
0.98 GB/token, and 10.8 tok/s ⇒ 92 ms/token, i.e. the kernel runs at **~10.6 GB/s** while the bus measures
35.7 GB/s — the loss is not the epilogue but how weights are loaded, a different question (and possibly
bigger than W-1 if it can be measured).

- W-2's remaining work: if ADR-005 says *yes*, switch `q4k_dot_vnni` to int-scale + one `fma`/block in the
  same PR as W-1b, measuring the encode at 1T separately. If *no*, W-2 closes with W-1.

### W-3 — Q6_K has the same disease (0.5 day, only after W-1)

The Q6_K epilogue is 17.6% of the kernel = **2.8% chunk**, under the threshold. Do it only if W-1 has opened
the path (the same switch), and only so the repo does not carry two different epilogue policies — so it too
sits behind ADR-005 and does not run first.

### W-4 — Attention at long context (2-4 days)

At ctx 4096, one thread: vipera pp4096 43.5 tok/s, llama 60.48; attention is ~43% of the step time (inferred
from the pp1024/pp4096 slope, not from the profiler). ik has `iqk_flash_attn.cpp` + `fa/`; `idea-sources.md`
records that PR #65 ported part of it — **re-measure first**, do not port it twice.

- Refuted if: after subtracting attention from the slope equation, the rest of the 4096 gap is < 1.15x.

### W-5 — int8 staging panel (M4) (3-5 days) — only if W-1 is still < 240 GFLOPS/core

`iqp.cpp` is a *simplification* in exactly the place we need: decode 8 rows into `int8` (with bias), then run
an integer GEMM over all columns. It is equivalent to M1+M2 but pays with scratch and a second arithmetic
path. Not opened before W-1, because if W-1 collects the epilogue then what remains of M4 is only the load
port, which the band already optimises (2 uops per 64 MAC).

## 5. Order and reasoning

As it stands: **the owner decides W-1 (`ADR-005`)** → if yes, `W-1a → W-1b` → `(W-2 | W-3)` → `W-4` → `W-5`;
if no, `W-2` (probe first) → `W-4`. `W-0` is confirmation work, not blocking. Do **not** do M1/M5/M7. Why the
ranking:

- Ceiling = `phase share x kernel share`. q4k at 8T is 74% of the chunk and the epilogue is 21-23% of that →
  W-1 is the only item with a >3% ceiling that needs *no* on-disk layout, new loader or new dependency (it
  only changes how activations are quantized in memory, something rebuilt on every call) — but it has turned
  from "the biggest perf job" into "a job waiting on a decision", so if the owner says no, the next item must
  be W-4 (attention at long context), where the 43% of step time at 4096 is still untouched and nobody has
  re-priced it since ADR-004.
- M1 (repack) has a larger nominal ceiling (the `NO_WEIGHT` probe's +29/56%) but has failed measurement twice
  on this machine (decode 1.06x at 8T; llama's own GEMM 0.74x), and the price is a second on-disk layout.
- W-4 grows with context, not with chunk: at 512 it is 6%, at 4096 ~43%. Only long-chat users feel it, so it
  ranks after W-1 but before W-5.

## 6. What this document does not say

- It does not claim ik's kernel is *measurably* faster under identical conditions: the 268.7 figure is the
  whole engine, a different harness and a different ubatch. W-0 exists to close that gap.
- It says nothing about W-1's output quality: W-1 **coarsens** the activations (one scale per 256 instead of
  per 32), so it must go through ADR-003's measurement set (6 NLL texts + 9 prompts) with a stricter
  threshold, and needs its own ADR before it can become a default.
- It proposes no new dependency, no new `unsafe` outside `simd.rs`, and no `#[allow]` (`CONSTRAINTS.md`).
