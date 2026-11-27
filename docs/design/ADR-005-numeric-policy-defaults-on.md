# ADR-005: three numeric-changing switches turned on by default

Status: **Partly superseded, 2026-11-25.** The paired end-to-end measurement (§5) shows that both
defaults are regressions on the reference machine: `VIPERA_Q4K8` at 0.49x decode (it repacks inside
every call) and `VIPERA_TILED_A` at 0.58x prefill. Both went back to **opt-in**; `VIPERA_ATTN_GEMM` was
opt-in anyway (§3). The original decision is kept below as a historical record.
Date: 2026-11-25 · Supersedes: ADR-006 (partly). ADR-004 still holds (keep it opt-in).

## 1. Context

Three fast paths were already coded but left off, because each changes output and that decision
belongs to the owner:

| Switch | What it changes | Price measured earlier | Gain measured earlier |
|---|---|---|---|
| `VIPERA_ATTN_GEMM` (`attention_qblock_enabled`) | attention in 32x64 tiles through GEMM, the softmax total accumulated | flips one greedy token (40-token probe: `1000 times` -> `100 times,`) | attention -48%, chunk -13% at 3.7k tokens; -6.7% at 1.5k |
| `VIPERA_TILED_A` | the tiled band's activations quantized with one scale per 256 instead of per 32 | NLL +0.45..0.69% on every text (W-1) | kernel probe +1.24..1.36x, 8T chunk inferred at +14..20% |
| `VIPERA_Q4K8` | Q4_K decode through the 8x8 repack and `block_q8_K` activations (also one scale per 256) | same kind as the row above | lab: 1.34x at 1 thread, **1.056x at 8 threads** (ADR-006 §3) |

The numbers in the table come from `docs/benchmarks.md`, ADR-004 and ADR-006. This ADR does **not**
re-measure them.

## 2. Decision

- Two switches **on by default**: `VIPERA_TILED_A`, `VIPERA_Q4K8`. `=0`, `false`, `off` or `no` turns
  each off (`tiled::env_default_on`).
- `VIPERA_ATTN_GEMM` **stays opt-in** (ADR-004 still holds): see §3, the gate stopped it.
- ADR-006 §5's reopening condition ("ADR-005 decides per-256 for the activation path") is satisfied by
  this decision.
- Two changes ship with it that do not change the arithmetic of the shipping path:
  - KV `q8_0` goes through the tile kernel (`attend_rows_shared`, decode via `KvLayout`) instead of
    falling back to the per-entry path when `head_dim` is divisible by 32. Test:
    `qblock_attention_matches_the_per_entry_kernel_on_a_q8_0_cache`.
  - `vipera serve` turns the prefix cache on. `VIPERA_PREFIX_CACHE=0` turns it off.

## 3. Not verified

- There is no paired measurement (one process, two alternating arms, `taskset`, median of >= 5 pairs)
  for this ADR. The sandbox that wrote it has no GGUF model and only one core.
- The tests that need a model were not run: `prefix_cache`, `logits_reference`, `llama_continuations`,
  `q4k_f32_continuation`. llama.cpp's continuation fixture may shift by a token after this change; if
  so, that is the consequence accepted in §1, and the fixture should be regenerated rather than the
  threshold loosened.
- ADR-003's 6-text NLL set has not been re-run with all three switches on at once; every number in §1
  is per switch.
- The prefix cache assumes a position has the same bits regardless of how the prompt was chunked.
- **Checked 2026-11-25, negative result — twice.** With `VIPERA_ATTN_GEMM` on, `cargo test
  --workspace` fails at `logits_each_matches_single_passes` (win_vs_llama.rs, "position 1 differs").
  The cause is not the GEMM width but path selection: `attend_entries` only takes the tile kernel when
  `entries.div_ceil(QBLOCK) > 1` (i.e. from the fifth token of a chunk), so the same position is
  computed by the tile kernel in an 8-token chunk and by the per-entry kernel in a 2-token pass — two
  different softmax summation orders (~1e-7). Ablating one switch at a time: `VIPERA_ATTN_GEMM=0`
  makes the test pass (76 s), while `VIPERA_TILED_A=0` and `VIPERA_Q4K8=0` stay red. An 11:40 patch
  claimed the owner had accepted this trade-off without fixing the test; re-running that exact test
  still fails at "position 1 differs", so the `VIPERA_ATTN_GEMM` row in §2 stays opt-in. Speculation
  (on by default for greedy turns) checks the draft with a batch pass, and the prefix cache in `serve`
  rests on the same assumption, so the tile cannot be the default until the two kernels share one
  summation order — or the owner changes the policy outright and fixes that test/invariant first,
  rather than loosening the threshold.

## 5. Re-measurement — 2026-11-25 (supersedes §2)

Two-arm A/B, two alternating rounds with the order reversed, `taskset -c 0-7`, `VIPERA_THREADS=8`, same
Qwen2.5-1.5B Q4_K_M GGUF. ik_llama.cpp `1180df7` in the same session: pp128 253.6 tok/s, tg64 29.2
tok/s.

| Arm | pp128 | decode ctx 32 | Q4_K kernel, 1T |
|---|---:|---:|---:|
| default (both on) | 125.3 | 16.4 | 3.88 GB/s |
| `VIPERA_TILED_A=0`, `VIPERA_Q4K8` default | 122.9 | 32.4 | — |
| both off | **213.2** | **32.5** | **12.97 GB/s** |

The wired Q4_K8 path repacks the weights inside every call (`quant.rs::fill_rows_q4k8_groups` calls
`make_blocks_q4k8`), so it never amortizes the repack; the lab's 1.056x was pre-packed. TILED_A's +2.6%
chunk does not reproduce at pp128. Both defaults went back to opt-in; details in
`docs/benchmarks.md`, sections "Q4K8 on by default — re-measured end-to-end" and "TILED_A on by default
— prefill re-measured".

## 4. Rollback

One at a time, no rebuild: `VIPERA_TILED_A=0`, `VIPERA_Q4K8=0`, `VIPERA_PREFIX_CACHE=0`. Flipping a
default back is a revert of two `*_enabled` functions and one line in `serve.rs`.
