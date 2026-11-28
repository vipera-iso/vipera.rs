# ADR-008: repack Q3_K at load into a `dpbusd`-ready layout — density first

Status: **Deferred — the ceiling cleared the gate, the density did not clear the machine.** The band
probe in §4 measures **+7.6% of the band (median of eight round pairs, 6.3–10.8%)** for the
pre-decoded block, so option B is justified; it is not built, because nothing on this machine can hold
the repacked model to verify it (§6). Never a default.
Date: 2026-11-28 · Owner: *to be filled in* · Supersedes: none · Related: ADR-006 (Q4K8 repack),
`docs/design/q3k-band-decode-hoist-plan.md` §7 lever #3, `docs/benchmarks.md`.

## 1. Context

The Q3_K band has already hoisted its row-block decode twice: `VIPERA_Q3K_PAIR` decodes once per two
tile halves (shipped, on by default, +6.7% pp512) and `VIPERA_Q3K_QUAD` once per four (measured, +2.6%
of the band, off). Both leave the decode in the hot loop, paid `halves / width` times per row-block.
The remaining lever is to take it out entirely: repack every Q3_K block at load into the form the band
computes for itself, so the band only loads and multiplies.

`q3k_decode_codes` turns one 110-byte `block_q3_K` into:

| piece | size | what it is |
|---|---:|---|
| `codes` | 256 B | the 256 weights as bytes 0..15, the form `vpdpbusd` wants |
| `coef` | 64 B | 16 `f32`, `d · scale` per sixteen-value group |
| **total** | **320 B** | per row-block, versus the 110 B stored today |

## 2. The density cost

**2.91x the bytes for every Q3_K tensor in the file.** Nothing about the GGUF changes; this is a second
buffer built at load.

| model | Q3_K share (approx.) | today | repacked |
|---|---:|---:|---:|
| the 876 MiB test model | most of it | ~0.9 GiB | **~2.5 GiB** |
| `q8src-imbf-q3km.gguf` (6.9 GiB) | most of it | 6.9 GiB | **~20 GiB** |

This machine is a 4-core/8-thread laptop with **15 GB of RAM, about 3 GB of it free** at measurement
time. The 6.9 GiB model repacks to more than the machine has, so lever #3 **cannot be validated
end-to-end here at all**, and it cannot be a default for anyone whose model already fills their RAM.

## 3. Options

| option | what it costs | verdict |
|---|---|---|
| **A. Repack every Q3_K tensor at load, unconditionally** | 2.91x resident size; a 6.9 GiB model needs ~20 GiB | rejected: changes whether a model can run at all, for a promise that is not yet measured |
| **B. Opt-in (`VIPERA_Q3K_REPACK=1`) with a load-time budget check** | the same bytes, but only when asked, and refused with a printed reason when the projected size exceeds the budget | the only shape that can ship. Must fall back to the band decode, must not touch the GGUF, and must be inert when off |
| **C. Leave it closed** | nothing | correct if the ceiling probe in §4 is small |

`ADR-006` set the precedent that a repack is allowed to be a switch before it is a default. Q4_K's
repack is 144 → 1152 B per 8 rows, i.e. the bytes are unchanged and only the order is; Q3_K is
different in kind: `dpbusd` needs the weights as bytes, so the packing *is* the expansion. That is why
this ADR exists separately.

## 4. What has to be true before option B is built

The ceiling is the decode, which is the share the band bench already prints: 17–19% at one tile half,
about **10% at the shipped two-half arrangement** (the decode is paid once per two halves). A repack
that removes it completely cannot beat that, and it adds a 320-byte read per row-block where the
decode read 110 — so the realistic ceiling is lower than the decode share, not equal to it.

**Pre-registered gate:** band bench, H=2, one process, arms interleaved, **≥6% of the band** for the
pre-decoded arm against today's, plus an H=2 parity match. Under that, option C (close) and the ledger
gets the number. Over it, option B with the memory guard, and the end-to-end number has to come from a
machine that can hold the repacked model — not this one.

## 5. Consequences if B is built

- `CpuBackend::load` grows a repack pass for Q3_K tensors, with a projected-size check before the
  allocation and a printed fallback reason, in the style of the other load-time switches.
- The band driver takes the packed buffer instead of `data`; the row kernel keeps reading the GGUF
  layout, so both paths must stay bit-identical (`q3k_tiled_range_matches_the_row_kernel` already
  compares against the row kernel, so it covers this).
- The `codes` buffer stops being a stack array in the band; the 320-byte block is read from the repack
  buffer, so the 16-accumulator pressure that H=4 tested does not change.
- `CONSTRAINTS.md` §Rejected gets the row either way, so nobody re-derives the density.

## 6. Measurement (2026-11-28), and why it stops at "deferred"

`bench_q3k_decode_codes` now builds the exact repacked block the loader would store — 256 code bytes
plus sixteen `f32` per (row, block) — and times `q3k_band_rows_packed::<4, 2>` against the shipped
two-half band. Same arithmetic, same `dpbusd` chain, same fused broadcast-loads; the only difference
is that `q3k_decode_codes` does not run.

| run (4 rounds each) | pre-decoded ns/half | paired ns/half | gain |
|---|---:|---:|---:|
| 1 | 45.83–46.06, 44.97, 43.21 | 50.48, 48.97, 49.05, 46.44 | 6.4, 8.3, 7.0, 8.8% |
| 2 | 45.65, 44.25, 44.49, 44.98 | 48.92, 49.06, 49.90, 48.02 | 6.7, 9.8, 10.8, 6.3% |

**Median +7.6%**, against the pre-registered 6% gate. That is consistent with the decode share at two
halves (~10%), minus the wider read.

**But the probe is cache-hot.** Four rows × six blocks is 2.6 KB of packed blocks; the 320-byte block
that a real repack reads is 2.91x the bytes of the 110-byte block it replaces, and at pp512 those
blocks stream from RAM. The gain measured here is therefore an upper bound that ignores the only cost
the ADR is about — which is exactly the way the Q4_K `NO_WEIGHT` probe over-priced its lever earlier
(`CONSTRAINTS.md` §Rejected). Pricing the traffic needs a prefill measurement, and a prefill
measurement needs the model repacked, which is ~20 GiB on a box with 15 GB of RAM and ~3 GB free.

So: **option B is designed and justified, and deliberately not built.** The next step is not another
kernel change; it is a machine that can hold the repacked 6.9 GiB model, a `VIPERA_Q3K_REPACK` loader
with the projected-size guard from §5, and `prefill_vs_decode` run both ways there. Anyone with that
machine can implement it from this document and `q3k_band_rows_packed`, which is the kernel it needs.
Until then the shipped defaults stay as they are: band decode, `VIPERA_Q3K_PAIR=1`.

**A second probe in the same run answers a related question** (see `docs/benchmarks.md`): forcing one
opaque 16-byte code load with the decode still present is **~3% slower** (median of eight rounds,
−5.6 to +2.5), so the band's fused `vpbroadcastd (mem)` is not waiting on the decode's store. The ~7.6%
that removing the decode buys is the decode's own instructions and their store traffic, not a
store-forwarding stall in the band.
