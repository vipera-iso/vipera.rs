# Q3_K band: decode once per row band, not once per tile half

**Status:** plan, no branch yet. Threshold and falsification fixed up front (section 3).
**Measured on:** `3960eda` (main after PR #83/#84), release, i5-11300H, governor `powersave`.
**End-to-end already measured (2026-11-27):** the band is **2.887x** the row kernel at pp512 on this
model, five rounds interleaved in one process (`docs/benchmarks.md`). `VIPERA_Q3K_TILED=0` plus the
test `win_vs_llama::prefill_q3k_band_vs_row` is the harness for later A/B runs.

## 1. Per-block cost, measured today

`cargo test -p vipera-backend-cpu --release --lib bench_q3k_decode_codes -- --ignored --nocapture`
(1536 columns = 6 blocks, band of 4 rows, 16-activation tile, cache-warm, 200k iterations per round):

| | ns/row-block | share |
|---|---:|---:|
| `q3k_decode_codes` (codes + 16 scales) | **9.36** (8.83–9.56) | 19.6% |
| `q3k_band_rows::<4>` (decode + dot) | **47.67** (47.60–47.77) | 100% |
| remainder (dot + epilogue) | 38.3 | 80.4% |

Converted to one block (4 rows) at 3.5 GHz, the band is ~190 ns ≈ **665 cycles**:

| component | cycles | note |
|---|---:|---|
| mandatory `dpbusd` | **320** | 4 rows × 16 groups × 4, plus 16 groups × 4 for the bias; 1 instruction/cycle ⇒ floor |
| code + scale decode | **131** | 19.6%, paid **8 times** per block at B=128 (section 2) |
| weight broadcast, activation/`dxs` loads, epilogue | ~213 | not split yet; ceiling ≤32% |
| **measured total** | ~665 | 2.08× the `dpbusd` floor |

The conclusion comes from that table: the kernel is **not** geometrically limited — it issues the
minimum number of `dpbusd` (256 + 64 bias per block) but runs at 2.08× those cycles, and ~20% of them
are decode, i.e. work that does not have to be repeated.

**Prefill phase split** (same machine, same model, `VIPERA_THREADS=8 taskset -c 2-7`,
`VIPERA_PROFILE_STEP=1`, 540-token prompt, `--max-tokens 1`):

| phase | share of chunk |
|---|---:|
| `other_dot` (Q3_K + Q5_K) | **68–70%** |
| q4k | 24% |
| attention | 3–6% |
| q6k / proj / rope / kv_write / elem | 1% each |

So an X% improvement in the Q3_K band is worth **0.68·X** at pp512: H=2 (9.8% of the band) →
**~6.7%**, H=4 (14.7%) → ~10%. That is where the 5% threshold in section 3 comes from.

## 2. Mechanism: decode is paid 8 times

`try_q3k_matvec_tiled_range::<B>` (B ∈ 16..=128, `TILE_MIN_BATCH = 16`) splits each call into
`halves = B.div_ceil(16)` tile halves and calls `q3k_band_rows::<4>` **once per half**. The band
decodes each row-block's codes inside itself. At B=128 (`prefill_chunk()` default) every row-block is
therefore decoded **8 times**, although codes and scales are a pure function of `(row, block)` and do
not depend on the tile half.

Ceiling of hoisting the decode out of the half loop, from the table above:

| variant | decodes | saving on the band |
|---|---:|---:|
| current (one half per pass) | 8 | — |
| group **2 halves** per pass (H=2) | 4 | **9.8%** |
| group 4 halves per pass (H=4) | 2 | 14.7% |
| decode once, cache the whole row band | 1 | 17.2% (minus ~3–4 µs/band of re-reading codes from L2) |

Start with **H=2**: no added buffer, no added traffic, just 8 accumulator `zmm` registers
(4 rows × 2 halves) and decode amortised twice. H=4 (16 accumulators, 14.7%) is the second step once
H=2's codegen proves there is no spill. Caching the whole row band needs 44.8 KB for an 8960-wide ffn
(35 blocks) and re-reads it 8 times (~358 KB/band), so its net gain is only ~10% and it spends
bandwidth — last.

## 3. Hypothesis, threshold, falsification

> Group two tile halves into one band pass (`q3k_band_rows::<N, H>`, H=2 when `halves ≥ 2`) so each
> row-block is decoded once for 32 activations instead of twice for 16, expecting **≥5% pp512** on
> the Q3_K_M model; the band bench must drop **≥6% ns/row-block** (ceiling 9.8%, 60% of it).

Falsify if: pp512 < 3%, or the band bench < 6%, or the codegen shows vector spill to `%rsp` inside the
group loop, or parity differs by one bit. On falsification: record the numbers in `CONSTRAINTS.md`
§Rejected with the measurement, and do not open a PR.

The 5% threshold is 75% of the 6.7% end-to-end ceiling (9.8% of the band × 0.68 `other_dot` share);
6% on the bench is 61% of the 9.8% band ceiling. Both sit above this machine's 3% noise floor.

## 4. Scope

`crates/vipera-backend-cpu/src/simd.rs` only:

1. `q3k_band_rows<const N: usize>` → `q3k_band_rows<const N: usize, const H: usize>`, with `H = 1`
   monomorphising to exactly today's code (the old path is preserved, no arithmetic change).
   `Q3kBandView` gains `halves: usize` (to compute the half offset) or an array `[Q3kBandView; H]`.
2. Inner loop: decode the `N` rows once per block (as today), then `for h in 0..H` over the 16
   groups; accumulators split by `(row, half)`; write `out[half]` after the block loop, keeping each
   half's accumulation order ⇒ **bit-identical** to H=1 per half.
3. Driver `q3k_matvec_tiled_range`: `for half0 in (0..halves).step_by(H)`, with the odd trailing half
   running H=1; the `halves == 1` case (B=16) goes straight to H=1.
4. Switch `VIPERA_Q3K_PAIR` (default **off**, `env_truthy` like the other switches) — turned on only
   after measurement, following the path PR #82 took.

Not touched: `quant.rs`, `model.rs`, the GGUF layout, `q3k_dot_vnni`, the `count` tail (still the row
kernel).

## 5. Order of work

1. **B0, same session:** pp512 (tok/s) on `q8src-imbf-q3km.gguf` (6.9 GB) — the old absolute rows
   belong to the 876 MiB model and cannot be compared. The phase split is measured (section 1); only
   the pp512 baseline of the exact binary to be A/B'd is missing.
2. **B5 codegen gate** for both `::<4,1>` and `::<4,2>`: zmm ≠ 0 is by design (512-bit band); the
   condition is **0 `(%rsp)` inside the dot region**, no spilled accumulator, and `vpdpbusd` count =
   320 per block (no increase).
3. **B6 parity:** `q3k_tiled_range_matches_the_row_kernel` green for batches 4/16/32/40 **and add
   64/128** (they exercise the H=2 path and the odd trailing half); re-run with
   `VIPERA_Q3K_PAIR=0`.
4. **B7 measure:** band bench in one process, two arms alternating, best-of-6 → ns/row-block; then
   pp512 over 5 interleaved switch on/off pairs in the same binary. Report ratios, not absolute ms.
5. Ledger row in `docs/benchmarks.md` (with the section-1 cost table) → commit + push branch.

## 6. Known risks

- **Spill:** 8 accumulators + 4 activation vectors + weight broadcast + float temporaries ≈ 20 `zmm`.
  H=4 (16 accumulators) will be tight; if the H=2 gate already shows spill, stop and do not try H=4.
- **Thermals/frequency:** the host drifts 12% between runs six minutes apart. The evidence therefore
  has to be the one-process band bench and 5 interleaved pp512 pairs, not a before/after pair.
- **Different model:** the 7.3 GB `q8src-imbf-q3km.gguf` (6.9 GiB) has a different type mix from the
  876 MiB model; record the model, `VIPERA_THREADS`, and date in the ledger, and do not compare
  absolute numbers with the older 151/155 tok/s rows.

## 7. Queue after this lever (ranked by ceiling × cost)

| # | Idea | Mechanism | Measured ceiling | How to reject |
|---|---|---|---|---|
| 1 | **Price the weight broadcast** (probe, no kernel change) | replace the weight dword with a constant in a test-only build, like the Q4_K `NO_WEIGHT` probe (which gave +29–56%) | ≤32% of the band (the 213 unexplained cycles) | close if the probe is < 8% of the band |
| 2 | **Decode straight into `u32` order** | have `q3k_decode_codes` emit `[u32; 64]` and let the band read 16 bytes per group instead of four 4-byte loads overlapping a 32-byte store | inside the 32% above; store-forwarding is the prime suspect | open only if probe #1 is ≥ 8% |
| 3 | **Repack Q3_K at load** (like Q4K8) | remove decode from the hot loop, read codes in `dpbusd` form | ~19.6% (the decode share) | needs an ADR: 110 B → 320 B/block ⇒ 876 MiB → ~2.5 GiB, and the 7.3 GB model no longer fits |
