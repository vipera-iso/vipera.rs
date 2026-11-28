# ADR-006: `VIPERA_Q4K8` (Q4_K 8x8 repack) — **off** by default

Status: **Accepted — default off.** *(Partly superseded by ADR-005 on 2026-11-25: `VIPERA_Q4K8` is now
on by default; `=0` turns it off.)* The kernel is ported and wired into the engine, but that path only
runs when `VIPERA_Q4K8=1`. Making it the default needs two things that did not exist: a paired
measurement and an arithmetic decision.
Date: 2026-11-24 · Owner: *to be filled in* · Supersedes: none · Related: ADR-003, ADR-004,
`docs/design/repack-port-plan.md`, `docs/benchmarks.md` section "Phase 1b".

Numbering: **ADR-005** was reserved for the *activation quantization granularity* decision (W-1 in
`repack-port-plan.md`), because that is the user-visible policy and the precondition for the
one-op-per-block epilogue to exist at all. This ADR-006 decides only the *repack switch*, and
deliberately does not merge the two questions.

## 1. Context — from "closed" to a switch that is on

PR #66 (`bf1f02d`) closed the repack direction with measurement: the 8-row gemv is **1.34x at one
thread** but only **1.06x at eight**, because 8-thread decode already saturates the bus, and
llama.cpp's `GGML_CPU_REPACK=ON` measured 0.74x of OFF. The conclusion then: do not port it, keep
`q4k8_lab.rs` as evidence.

Then the `perf/q4k-repack-8x8` branch did two things PR #66 had not:

1. Ported the kernel from the lab into the engine: `BlockQ4Kx8`, `make_blocks_q4k8`, `quantize_q8k`,
   `gemv_q4k8` (`crates/vipera-backend-cpu/src/simd.rs`), wired through `fill_rows_q4k8` /
   `fill_rows_q4k8_groups` (`quant.rs`).
2. Turned the switch **on by default** and recorded a **+53% tok/s** decode table in
   `docs/benchmarks.md`.

Step 2 is why this ADR exists: it made a new arithmetic path the default for every user, on the back of
a measurement whose two halves are not comparable (section 3).

## 2. Mechanism — and why this is an arithmetic change, not a kernel change

`block_q4_Kx8` (1152 B) is eight `block_q4_K` interleaved every eight bytes, so one `vpbroadcastd` of
the activation serves `vpdpbusd` for **eight rows** instead of one. The bytes do not change; the read
order does.

What changes with the layout is the **activation format**: the `ggml_gemv_q4_K_8x8_q8_K` kernel reads
`block_q8_K` — **one** scale per 256 activation values, plus `bsums` per 16. The engine today uses
`QuantizedX`: **one scale per 32** values. So enabling the repack does not just change summation
order: it makes activations 8x coarser in granularity, exactly the granularity that
`repack-port-plan.md` §3.1 and W-1 measured as costing **+0.3..0.7% NLL** on every text. That is the
owner's decision (ADR-005), not a perf PR's (`CONSTRAINTS.md`: "changing one bit is an arithmetic
change").

A second consequence, fixed within the same change: the 8-row group must be formed on the **absolute
row index**, not on the call's offset. Before the fix, `fill_rows` split rows across workers by
`rows/threads` while the pool split them differently, so the same row had two different results; the
test `a_pooled_matvec_matches_the_spawned_one` failed, and the batch path (B < `TILE_MIN_BATCH`) also
disagreed with "each sequence run alone" (`a_batched_step_matches_every_sequence_run_alone`). Now the
kernel is called for **every** width the tiled path does not take, packing once for all `B` entries,
and only the final rows of the matrix (when `rows % 8 != 0`) still go through the row kernel — so the
result is a function of (row, activation).

## 3. The evidence at hand — and where it fails to stand

**Lab, one process, two alternating arms** (`bench_q4k8_gemv_vs_row`, 2026-11-23, i5-11300H, 4c/8t,
45 MB):

| threads | row kernel | 8-row gemv | ratio |
|---:|---:|---:|---:|
| 1 | 4.02 ms / 11.2 GB/s | 3.01 ms / 15.0 GB/s | 1.338x |
| 2 | 2.10 / 21.5 | 1.65 / 27.4 | 1.273x |
| 4 | 1.20 / 37.7 | 1.12 / 40.3 | 1.070x |
| 8 | 1.54 / 29.3 | 1.46 / 31.0 | **1.056x** |

That is the only number that stands, and it agrees exactly with #66's conclusion: by eight threads the
two kernels read the same 45 MB through the same bus and differ within noise. 1.056x is **not** the
≥3% chunk the repo's gate requires.

**Engine, the "Phase 1b" table in `docs/benchmarks.md`:** step 53.47 → 34.85 ms, decode 18.7 → 28.7
tok/s. That table is **not a paired measurement**: two different sessions, and it records `q6k matvec`
at −35.8% while this change touches only Q4_K — a phase whose kernel did not change cannot really be
35.8% faster, so the two sessions ran under different conditions and the Q4_K "−34.8%" cannot be
attributed to the repack either. The paired measurement (one process, two alternating arms, same
bytes) **was never run** — the working tree while this ADR was written had W-1 half-finished and did
not build. It is the precondition for the default, not this ADR's job.

## 4. Decision

`VIPERA_Q4K8` is **off by default**, following the repo's `env_truthy` convention (unset / empty / `0`
is off; anything else is on). The kernel is not deleted: it is evidence, a bench for re-measuring, and
the only fully ported path for W-3/W-5 if the owner accepts per-256 granularity.

Reasons, in order of weight:

1. **It changes output on every prompt.** Per-256 activations are the same class of decision as
   ADR-004 (rejected) and ADR-005 (pending). Making it the default before that decision turns a quality
   choice into an unreviewed default.
2. **The only number that stands is 1.056x at eight threads**, below the repo's own 3% chunk threshold.
3. **The engine number that supports it is not comparable** (section 3), so it cannot justify the
   default.
4. **Turning it off costs nothing**: the switch exists, the old path is still the default, there is no
   on-disk layout, no new loader, no dependency.

## 5. Consequences

- Default = the row kernel as before #66. `VIPERA_Q4K8=1` enables the repack, and both arms must pass
  the same set: `q4k8_wire_matches_lab_kernel_within_tolerance`,
  `q4k_rows_do_not_depend_on_the_call_split`, `a_pooled_matvec_matches_the_spawned_one`,
  `a_batched_step_matches_every_sequence_run_alone`.
- Reopening conditions (either one, not both):
  - a **paired** measurement (one process, two alternating arms, same bytes, `taskset`, median of ≥5
    pairs) showing ≥3% chunk or ≥3% step at eight threads; or
  - ADR-005 decides per-256 for the activation path, at which point the repack is how to get that
    granularity without writing another kernel — and the NLL price is then no longer specific to the
    repack.
- If the owner says no to both, this section closes with #66 and `q4k8_lab.rs` stays as evidence; one row
  in `CONSTRAINTS.md` §Rejected is enough.

## 6. What this ADR does not say

- It does not say which kernel is faster on other machines: every number here is i5-11300H, 4c/8t Tiger
  Lake, AVX-512 VNNI. On a 1-2 thread machine (where the lab measured 1.27-1.34x) the conclusion may
  differ, which is why the switch still exists.
- It says nothing about the quality of `block_q8_K` itself: it reuses W-1's NLL measurement rather than
  re-measuring.
- It does not propose changing `TILE_MIN_BATCH`, changing the on-disk layout, or porting llama.cpp's
  8x8 GEMM (measured at 0.74x).
