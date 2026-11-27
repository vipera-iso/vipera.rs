# ADR-003: fused epilogue (`fma` + `fnmadd`) for the Q4_K tiled kernel

Status: **Accepted — stays opt-in, the default is still multiply-then-add.** Making it the default has
**not** been approved and is not part of this commit; the conditions to flip it are in section 8.
Date: 2026-11-22 · Owner: *to be filled in* · Supersedes: none

## 1. Context — the numbers

`q4k_matvec_tiled_range` (`crates/vipera-backend-cpu/src/simd.rs`) is the prefill path for every Q4_K
matrix. For each `(row, sub-block)` pair it finishes with six vector operations:

```
main    = (cvt(acc) · scale_f[row][s]) · dxv[s]
minterm = sumv[s] · min_f[row][s]
facc   += main - minterm
```

Three earlier probes (remove the weight broadcast, remove the float epilogue, remove both) put the
epilogue at **+33% (8960×1536)** and **+42% (1536×8960)** when removed entirely — but the epilogue is
mandatory arithmetic: `sum_s (cvt(acc_s)·scale_s·dx_s − sum_s·min_s)`. This ADR asks whether it can be
written in fewer operations without breaking the tiled path's published tolerance.

The fused form keeps `dxv` shared across all 16 columns, folds `scale_f` into one fma and `min_f` into
one fnmadd:

```
scaled  = cvt(acc) · dxv
sum     = fma(scaled, scale_f[row][s], facc)
facc    = fnmadd(sumv, min_f[row][s], sum)
```

Four operations instead of six. Measured on the i5-11300H, 1 thread, same binary, with
`VIPERA_Q4K_FMA_EPI` on and off:

| arm | 8960×1536 | 1536×8960 | chunk 4 (524 tokens, 8 threads), 5 alternating pairs |
|---|---:|---:|---:|
| multiply-then-add (default) | 153.4 / 149.5 GFLOPS | 165.2 / 173.2 GFLOPS | median **785.16 ms** |
| fused (`VIPERA_Q4K_FMA_EPI=1`) | 165.1 / 167.4 (**+8.6%**) | 193.4 / 189.3 (**+12%**) | median **746.04 ms** (**−5.0%**, 5/5 pairs) |

The added `if` does not slow the default arm: 153.4/149.5 falls inside the older baseline range
(151.7/152.6 in the previous run, same shapes).

## 2. Quality — the deciding part

### 2.1 Determinism control

Before comparing two arms, make sure the measurement is not thread noise. Same prompt, same seed, 8
threads, 24 greedy tokens:

| comparison | result |
|---|---|
| off vs off (3 runs) | **identical token by token** |
| fma vs fma (2 runs) | **identical token by token** |
| off vs fma | **different** |

Both arms are deterministic, so the differences below belong to the arithmetic, not to thread
scheduling.

### 2.2 Nine fixed prompts, 24 greedy tokens

Prompts: eight short questions/tasks (technical explanation, Vietnamese, writing Rust code, maths,
comparing concepts, page faults, benchmarking methodology, FMA) plus a 986-token excerpt cut from
`docs/benchmarks.md`. Every prompt is ≥ 16 tokens, so all of them take the tiled path.

| | |
|---|---|
| prompts producing **identical** output | **3 / 9** |
| positional agreement, average | **0.625** |
| first diverging position | 1, 4, 8, 11, 14, 23 (6 prompts); 3 prompts match to the end |
| Vietnamese prompt (27 tokens) | diverges at **token 1** |
| 986-token prompt | diverges at token 8 |

For the same Vietnamese prompt: the default arm opens with a sentence meaning "Prefill and decode are
two terms…", the fused arm with "Keywords: prefill, decode, hardware…". Both are plausible outputs from
a 1.5B model; this measurement says **nothing** about which is better, only that they differ — and
differ from the very first token.

### 2.3 Perplexity on fixed texts

Measured with `vipera score` (a new command in the same commit: the log-likelihood of a fixed text,
chunked at 64 tokens through the real prefill path, so the tiled kernel — and FMA — is part of the
measurement). Five texts cut from `docs/benchmarks.md` and `docs/design/*.md`, each run twice per arm,
with both arms reproducing every digit:

| text | tokens | mean NLL off | mean NLL fused | Δ |
|---|---:|---:|---:|---:|
| popcount, rays | 986 | 2.826631 | 2.823378 | **−0.00325** |
| gate/up shapes | 1697 | 3.084673 | 3.083967 | **−0.00071** |
| ggml bind | 2444 | 2.637364 | 2.639467 | **+0.00210** |
| ADR-001 opening | 1492 | 3.324126 | 3.324727 | **+0.00060** |
| ADR-003 section 2 | 1494 | 3.581916 | 3.582051 | **+0.00014** |
| ADR-001, first 60 lines (2026-11-23) | 1504 | 3.1906 | 3.1876 | **−0.00300** |

The last row was re-measured on 2026-11-23 on the `perf/q4k-repack-8x8` tree, two runs per arm, and
**reproduces every digit** (off 3.1906 / fused 3.1876, perplexity 24.3027 / 24.2306). It repeats the
table's conclusion: the per-text delta sits around 0.003 nats (±0.1%), changes sign with the text, and
shows no trend.

**A/B trap:** `tiled_fma_enabled` reads `var_os(..).is_some()`, so `VIPERA_Q4K_FMA_EPI=0` still **enables**
the fused path. The default arm must be run with `env -u VIPERA_Q4K_FMA_EPI`, otherwise both arms are the
same configuration and the measurement returns two identical numbers (which happened in this section's
first run).

Average **−0.000224 nats/token**, i.e. **−0.007%**; two texts improve, three get worse, and none moves
more than 0.003 nats (0.1%). Perplexity moves 16.888 → 16.834 on the first text. At this resolution the
measurement **cannot** tell the two arms apart — it rules out "fused makes things clearly worse", it does
not prove the two arms share a distribution.

### 2.4 Distance to the row kernel

The row kernel (`q4k_dot_batch_vnni`) is what decode uses for every Q4_K matrix, so "closer to it" is a
neutral criterion that can be measured in-repo. Test `q4k_tiled_error_against_row_kernel` (48 rows, 512
columns, B=16, fixed seed, 768 values):

| arm | max abs | max rel | mean rel |
|---|---:|---:|---:|
| multiply-then-add | 9.155e-5 | 4.784e-5 | 3.542e-7 |
| fused | 9.155e-5 | 4.784e-5 | **3.250e-7** (−8%) |

The worst case is identical (the same value, coming from quantization rather than the epilogue) and the
mean error **drops** — as theory says it should: one rounding instead of two cannot be worse.

### 2.5 The epilogue is prefill's only remaining lever

The single-thread prefill gap against llama.cpp (67.0 vs 87.53 tok/s) was split by phase (1 thread,
`VIPERA_PROFILE_STEP=1`, 128-token chunks; `docs/benchmarks.md`, 2026-11-23):

| phase | % chunk | GFLOPS |
|---|---:|---:|
| q4k | 79% | 202 |
| q6k | 16% | 179 |
| attention | 4% | — |
| whole prefill | 100% | 188 (llama: 229) |

The band isolated from the engine at B=128 reaches 218 GFLOPS, so the engine's q4k phase is already at 93%
of its own kernel — what is left is not in the plumbing. llama.cpp's dot kernel is AVX2 256-bit
(`vpmaddubsw`, 0 `zmm`) and spends more instructions per MAC than our band, so there is no kernel over
there left to port. Both engines run at 4.0 GHz, and the single-thread decode gap (1.34x) equals the
prefill gap (1.32x), so it does not come from chunking.

What that means for section 5: of the q4k phase's 22% epilogue, the fused form recovers 8.6% of chunk
time — the only figure above the 3% threshold. Q6_K has the same disease (17.6% of its kernel) but is only
2.8% of a chunk, below the gate; question 2 in section 11 stays a question about the switch, not about the
benefit.

## 3. What this ADR does NOT say

- It does **not** say fused gives better output. It only says fused is no worse arithmetically than the
  row kernel, and that it **does** change output. A quality conclusion needs a task-level measurement
  (section 8).
- It does **not** cover the rest of the proposed patch: `transpose_16x16` when building the activation
  tile, and `q4k_tile_rows::<R>` split out for band/tail. The default chunk's hot path goes through
  `SharedTile` (`SharedTile::fill_entries`), not the transpose branch, so the transpose only matters if
  `fill_entries` is ported too — a separate change, measured separately.
- It says nothing about Q6_K tiled (`q6k_matvec_tiled_range`). It is symmetric but has a different
  epilogue and takes 18% of a chunk; measured separately. Already measured (2026-11-23): its epilogue is
  **17.6%** of the kernel (`1.176x` when folded away), i.e. ~3% of a chunk, and the weight unpack only
  4.8% (`1.048x`) — so a Q6_K switch would add ~1% of a chunk to this same decision, not worth a change
  of its own.

## 4. Options

Updated once the perplexity measurement existed (section 2.3): both independent measurements — error
against the row kernel and NLL on fixed texts — say fused is no worse, so **B becomes a defensible
choice**. What still keeps A as the default is behaviour: 6/9 prompts produce different text, and
perplexity cannot see that. Flipping it or not is the owner's decision, and this ADR keeps the opt-in
state until that decision — no longer for lack of numbers.

| # | Option | Gains | Costs | Policy |
|---|---|---|---|---|
| A | **Keep multiply-then-add as default, fused behind a switch** (current) | no behaviour change; the numbers and the fast path are there to flip later | the −5% chunk goes unused | none |
| B | Flip fused to default, keep `VIPERA_Q4K_EXACT_EPI=1` for the old form | −5% chunk; follows the `VIPERA_EXACT_DOT` precedent | changes the output of every prefill of ≥16 tokens; users comparing against an earlier build see a difference | this ADR + conformance update |
| C | Drop it entirely, delete the switch | least code | loses the +8.6/+12% numbers for good; re-measuring next time | none |

## 5. Decision

**A.** Commit `bfb0fee` keeps the old form as the default and puts the fused form behind
`VIPERA_Q4K_FMA_EPI=1` (off by default), with the numbers and the reasoning in code and in
`docs/benchmarks.md`.

Why A rather than B: the fused form's benefit is real and its arithmetic is no worse, but it changes the
output of **6/9** prompts, from the first token through token 23, and the repo has no measurement that
answers "is that change better or worse". Five percent of prefill time on a 524-token prompt (≈39 ms per
request) is not enough on its own to carry an unmeasured behaviour change. The `VIPERA_EXACT_DOT`
precedent does not transfer directly: there the fast form is the **same** expression with a different
summation order, buying a 128-link dependency chain; here it is a different rounding of a term that every
later layer consumes.

## 6. Consequences

- Users can try it now: `VIPERA_Q4K_FMA_EPI=1`, no rebuild.
- The switch reads the environment on every call, so one process can measure both arms — which also means
  it must **not** be toggled between calls within a run if stable bits are needed; acceptable because both
  arms are deterministic within one configuration.
- `q4k_tiled_range_matches_batch_vnni` (tolerance `1e-3 + 1e-4·|ref|`) is green on **both** arms. That is
  why this is a policy decision rather than a bug.
- There is no golden-token test in the workspace, so the default path is not constrained by the test
  suite — the constraint is a human decision.

## 7. Test oracle

- `q4k_tiled_range_matches_batch_vnni`: parity tolerance against the row kernel, must be green on both
  arms.
- `q4k_tiled_error_against_row_kernel` (`#[ignore]`, measurement): prints max/mean relative error; reuse
  it whenever the epilogue is touched, and it is the data behind section 2.3.
- Token A/B: the nine fixed prompts from section 2.2; if the default is flipped, that set is what must be
  re-run and recorded.
- `cargo test -p vipera-backend-cpu --lib` (87 passed / 45 ignored) and `-p vipera-runtime --lib`.

## 8. Promote (the conditions for flipping to B)

| # | Criterion | Threshold |
|---|---|---|
| 1 | chunk 4 | **≥ 3%** faster at the median over ≥5 alternating pairs, idle box | met: −5.0%, 5/5 |
| 2 | kernel GFLOPS | no slower on either shape | met: +8.6% / +12% |
| 3 | error against the row kernel | mean rel does not rise | met: 3.250e-7 vs 3.542e-7 |
| 4 | determinism | each arm reproduces itself across runs | met |
| 5 | **task-level quality** | perplexity on fixed texts, "not worse" | measured: −0.007% average, 2 better / 3 worse, below the resolution floor (`vipera score`, section 2.3) |
| 6 | behaviour cost | the changelog states that prefill output changes | not done |

Rollback: delete one `if` branch or leave the switch at its default off; no layout, no row kernel, no
`SharedTile` touched.

## 9. Risks

- **Unmeasurable output drift**: a 5% prefill gain changes 6/9 prompts. If another model or prompt reacts
  more strongly, the symptom will be "the new build writes differently", not a red test.
- **Measured on one machine**: 153→165 GFLOPS is Tiger Lake with MUL+FMA on the same port. On a core with
  two FMA ports (Zen 4/5, Xeon server) the percentage may differ; do not extrapolate.
- **Heat/load**: chunk runs varied by up to ±5% between sessions (the same binary at 800 vs 1025 ms in an
  earlier run). So section 1 uses only within-pair ratios; the absolute numbers are not a promise.
- **Env switch on the hot path**: one `var_os` per kernel call; negligible for a 35-block × 144-call
  prefill, but if the kernel is later called on small shapes it should be cached.

## 10. Non-goals

- No change to the summation order across sub-blocks (`facc` still accumulates over increasing `s`).
- No touching `q6k_matvec_tiled_range`, `SharedTile`, `quant.rs`.
- No new dependency, no `unsafe` outside `simd.rs`.
- No default flip in this ADR.

## 11. Open questions

1. `vipera score` now exists; how many texts and what length make criterion 5 meaningful (one ~1000-token
   file gives ±0.1% NLL, but cannot see behavioural differences)?
2. Should the Q6_K part be folded in if it shows the same ratio, so that there is only one "fused
   epilogue" switch? Half answered: the ratios are not the same — 17.6% against Q4_K's 21-23% — and the
   Q6_K part is 2.8% of a chunk (16% of the chunk × 17.6% of the kernel, section 2.5), i.e. below the 3%
   gate: worth doing only together with the Q4_K switch, not on its own. What remains is a decision: fold
   it in for a tidier switch, or leave it because the gain is below the gate.
3. Owner and expiry of the "default off" commitment — if nobody measures for six months, keep it as it is
   or delete the switch so the dead code does not ride along?
