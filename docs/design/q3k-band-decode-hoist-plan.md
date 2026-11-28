# Q3_K band: decode once per row band, not once per tile half

**Status:** **shipped** on `perf/q3k-band-pair` (2026-11-28): H=2 gave **+6.7% pp512** (median of five
interleaved rounds) and **+10.5%** on the band bench, bit-identical (zero logit delta), so
`VIPERA_Q3K_PAIR` defaults on. Ledger row in `docs/benchmarks.md`. **H=4 was measured on
`perf/q3k-band-quad`:** it does not spill (the 16 accumulators stay in registers), but it buys only
~2.6% of the band, so `VIPERA_Q3K_QUAD` stays off. The weight probe from section 7 then priced the
code load and its broadcast at ~55% of the paired band; lever #2 was rejected at +0.1% because both
ways of writing it fold to the same `vpbroadcastd (mem)`, and lever #3 cleared its gate at +7.6% but
is deferred on density (ADR-008: 2.91x, ~20 GiB for the 6.9 GiB model). Threshold and falsification
were fixed up front (section 3).
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
| 1 | ~~**Price the weight broadcast**~~ **done 2026-11-28** | `NO_Q3K_WEIGHT_PROBE` replaces the weight dword with a constant in a test-only build | measured **~55%** of the paired band (21.5 vs 48.2 ns per half), above the ≤32% estimate because the hoisted broadcast disappears too | gate was < 8%: cleared, so #2 is open |
| 2 | ~~**Decode straight into `u32` order**~~ **rejected 2026-11-28** | one 16-byte load per group, dwords broadcast from the register (`VIPERA_Q3K_WIDE`, removed) | **+0.1%** band median, gate 3%: both formulations fold to `vpbroadcastd (mem)` per word (0 `vpermd`, 0 `vpextrd`), so the four "narrow loads" are already one fused instruction each | closed; `CONSTRAINTS.md` §Rejected has the row |
| 3 | ~~**Repack Q3_K at load**~~ **ADR-008: deferred, not built** | 320-byte pre-decoded block (`q3k_band_rows_packed`, test-only) instead of the in-band decode | measured **+7.6%** of the band (median of eight round pairs) — clears the 6% gate; density **2.91x** (110 → 320 B/block, 6.9 GiB → ~20 GiB) | deferred: the probe is cache-hot, the real block streams 2.91x the bytes, and no machine here can hold the repacked model. Needs a bigger box, not another kernel |

## 8. What is left, scored at real depth (2026-11-28)

Two facts from the ledger decide the queue, and neither was available when §7 was written:

1. **`other_dot` is flat with depth**: 6.39 s per 128-token chunk in the first chunk, 6.41 s at ~3.9k.
   Its *share* falls 69% → 45% only because attention grows 0.24 s → 5.0 s. So a band gain `g` is
   worth `g × 0.45` of a late chunk, not `g × 0.69`.
2. **`other_dot` is Q3_K for practical purposes.** The model's bytes are Q3_K 3756 MiB (53.8%),
   Q4_K 2486 (35.6%), Q6_K 609 (8.7%), Q5_K 124 (1.8%) — Q5_K is 3.2% of `Q3_K + Q5_K`.

| lever | band-level ceiling | value in a 4k chunk (× 0.45) | state |
|---|---:|---:|---|
| repack to 320-byte pre-decoded blocks (ADR-008) | **+7.6%** (measured) | **3.4%** | blocked: 2.91× density, ~20 GiB for this model |
| ~~H=8 over 2 rows~~ **measured 2026-11-28: −10.7%** | the decode saving is ~5.7% of the band, and halving the rows per pass costs more than that | — | rejected: 52.9–53.2 vs 47.9–48.3 ns per half. Two rows double the activation side per output and the passes' activation traffic; the decode-once-per-chunk saving does not survive it |
| H=4 | +2.6% (measured) | 1.2% | shipped off |
| 16-byte code load (lever #2) | +0.1% (measured) | 0.05% | rejected, folds to `vpbroadcastd (mem)` |
| Q5_K two-half hoist (Q5_K still decodes per half) | ~10% of Q5_K ≈ 0.3% of `other_dot` | 0.14% | dead on the byte split |
| the 55% weight side | — | — | needs a transposed tile and `asm!`; the unsigned `dpbusd` operand cannot come from memory (`CONSTRAINTS.md` §Rejected) |

**Decision: stop on the Q3_K band kernel.** No unblocked lever clears 5% of a real-depth chunk, and the
only one that clears it (the repack, 3.4%) is blocked on memory before it is blocked on anything else.
The lever with the better real-depth value is now **attention**: 35% of a late chunk and growing as
O(n²), against a band that is flat. Q4_K at 18% of a late chunk and flat is the second.

The guard that keeps this honest is `scripts/q3k-band-drift.sh`: one 128-token chunk, `PROFILEPASS`,
`other_dot` against `scripts/q3k-band-baseline.txt` within 8% on the recorded host. Run it before a
release; if the number is *meant* to move, update the baseline in the same commit.
