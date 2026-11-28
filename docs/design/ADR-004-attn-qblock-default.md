# ADR-004: should `VIPERA_ATTN_QBLOCK` become the default

Status: **Rejected — stays opt-in.** The gain is only 3.2% on a short chunk and 0.87% at depth, in
exchange for a systematic likelihood regression across all five texts and different output. No line of
code changes in this commit; the switch stays as it is.
Date: 2026-11-22 · Owner: *to be filled in* · Supersedes: none

## 1. Context — why this question is open again

`VIPERA_ATTN_QBLOCK` (`crates/vipera-backend-cpu/src/model.rs`, `attention_qblock_enabled`,
`QBLOCK = 4`, `attend_rows_shared`) was implemented and measured before, and the conclusion then was
**"does not pay for itself end to end yet"**. That conclusion was right for the share of the work
**at the time**: attention was then 3% of a prefill chunk.

Two things changed:

1. The tiled kernel made q4k (69% of a chunk) and q6k (18%) faster while attention stayed put, so its
   slice became **10%** at chunk 4 of a 524-token prompt and **29-30%** at 2.4k positions
   (`docs/benchmarks.md`).
2. `vipera score` arrived the same day, so the quality price of a rounding change is now measurable
   instead of guessed.

Following lesson L8 (`docs/lessons/2026-11-22-lessons.md`), a rejected lever must be re-measured when
the share of the slice it cuts grows. This is that re-measurement.

## 2. Mechanism

The default kernel calls attention once per layer with **one query row per position** in the chunk, and
each row walks the whole history. A ~105-position chunk at ~370 positions of history reads

```
105 rows × 28 layers × 2 KV heads × 370 positions × 128 values × 2 bytes × 2 (K and V) ≈ 1.1 GB
```

KV — more than the 0.9 GB of weights the whole chunk reads. `attend_rows_shared` keeps `QBLOCK = 4`
query rows in registers and sends them on **one trip through history**, so row traffic drops 4x; the
arithmetic per row (not just the read) is unchanged.

## 3. Measurement — shallow and deep are not the same

Three to five alternating pairs, same binary, same box:

| context | arm | attention | chunk | Δ chunk |
|---|---|---:|---:|---:|
| chunk 4 of a 524-token prompt (~370 positions), 5 pairs | default | 66.25 ms median | 649.71 ms median | |
| | `QBLOCK=1` | 47.77 ms (**−27.9%**) | 629.16 ms | **−3.2%**, 5/5 |
| chunk 24 of a 2953-token prompt (~2.4k positions), 3 pairs | default | 254.07 ms median | 854.45 ms median | |
| | `QBLOCK=1` | 236.43 ms (**−6.9%**) | 847.02 ms | **−0.87%**, 3/3 |

Worth noting: the gain **shrinks** as the context deepens, even though attention is 30% of the chunk
there. Four times less row traffic buys only 6.9% at depth against 27.9% in short context. So at depth
attention is no longer dominated by reading K/V but by the per-row arithmetic — scores, `exp`, and the
per-position weighted sum — which sharing rows cannot cut. That is an inference from two measurement
points, not a decomposition that was carried out.

## 4. The price — a systematic likelihood regression

`vipera score` (64-token chunks through the real prefill path), five texts, mean NLL per token:

| text | tokens | default | `QBLOCK=1` | Δ | % |
|---|---:|---:|---:|---:|---:|
| popcount / rays | 986 | 2.826631 | 2.829815 | +0.003184 | +0.113% |
| gate and up shapes | 1697 | 3.084673 | 3.084879 | +0.000206 | +0.007% |
| the ggml bind | 2444 | 2.637364 | 2.638594 | +0.001230 | +0.047% |
| ADR-001 opening | 1492 | 3.324126 | 3.328024 | +0.003898 | +0.117% |
| ADR-003 §2 | 1494 | 3.581916 | 3.586679 | +0.004763 | +0.133% |

Average **+0.002656 nats/token (+0.086%)**, and **5/5 in the same direction**: every text gets less
likely. Unlike the FMA epilogue (ADR-003), where 2 texts improved and 3 got worse around zero, here no
text improves. A regression that small but that consistent is not noise, and it runs against the
engine's own goal.

On top of that, 16 greedy tokens from a 524-token prompt **differ** between the two arms. This is an
arithmetic change, not a pure optimisation: `attend_rows_shared` groups the softmax sums differently.

## 5. Options

| # | Option | Gains | Costs | Policy |
|---|---|---|---|---|
| A | **Keep opt-in** (current) | no behaviour change; the lever and its numbers stay for anyone who needs faster short-context prefill | the −3.2% chunk goes unused | none |
| B | Flip the default, `VIPERA_ATTN_QBLOCK=0` to disable | +3.2% short chunk / +0.87% deep | NLL +0.086% on 5/5 texts; output changes; the smallest gain lands exactly where prefill is most expensive | this ADR + changelog |
| C | Delete the switch and the `attend_rows_shared` code | least code | loses a measured lever; short context is still where prefill is cheapest, so deleting it is not worth it either | none |

## 6. Decision

**A — keep opt-in.** Reasons in order:

1. **A consistent regression**: 5/5 texts get worse, +0.086% NLL on average, none better. A change with
   a direction that clear is not justified by 3.2% of chunk time.
2. **The smallest gain sits where it is most expensive**: prefill at 2.4k positions costs 854 ms per
   chunk and gains only 0.87%; the 3.2% is on short chunks, where 650 ms per chunk is the day's lowest.
3. **It is not the only lever**: the per-row arithmetic the two points point to as the real limit
   (scores, `exp`, the per-position weighted sum) has not been cut. Flipping the default now spends a
   regression on a kernel that still has headroom — exactly backwards.

Reopening conditions: either of the following, measured.

- **Share more queries than `QBLOCK = 4`** (8 or 16 rows in registers) — if row traffic is still the
  cost, scaling up must make the gain grow, not shrink. Measured by raising the constant, no structural
  change needed.
- **Cut the per-row arithmetic** (for example a grouped `exp`, or accumulating weighted-V in position
  blocks). If that works, QBLOCK's likelihood regression must be re-measured, because the 0.086% comes
  from the grouping of the sums, not from the path.

## 7. Consequences

- Anyone who needs faster short-context prefill can turn it on now: `VIPERA_ATTN_QBLOCK=1`.
- The default path changes not one bit, so no test needs editing; `cargo test -p vipera-backend-cpu
  --lib` (87 passed / 47 ignored) and `-p vipera-runtime --lib` (69) stay green.
- The switch is read once (`OnceLock`), so toggling needs a new process — acceptable because attention
  is not bit-identical between the two arms.
- `quant::exact_dot_requested()` already bypasses this path, so `VIPERA_EXACT_DOT=1` remains the
  bit-exact reference; rollback is unsetting the variable.

## 8. Promote (if it is reopened)

| # | Criterion | Threshold |
|---|---|---|
| 1 | chunk 4, 5 alternating pairs | ≥ 3% | met: −3.2% |
| 2 | chunk 24 (2.4k positions), 3 pairs | ≥ 3% | **missed**: −0.87% |
| 3 | mean NLL over 5 texts | **does not rise** | **missed**: +0.086%, 5/5 worse |
| 4 | greedy tokens | stated if they change | they change |
| 5 | decode | not slower | not applicable: `entries == 1`, this path does not run |

Two missed criteria are enough to reject; criterion 1 does not compensate for 2 and 3.

## 9. Risks

- **A conclusion that expires**: this is the second time the same lever is assessed, and the conclusion
  flipped with the slice's share. If another kernel gets faster again, the numbers in section 3 must be
  re-measured before being quoted.
- **Two measurement points**: shallow (370 positions) and deep (2.4k). Nothing between them was measured;
  the hypothesis "the gain shrinks with depth" is an inference from two points, not a curve.
- **NLL over in-repo texts**: a narrow domain, and perplexity does not see behavioural differences. It
  is enough to reject a consistent regression, not enough to call the two arms equivalent.

## 10. Non-goals

- This ADR touches neither `attend_entries`, `attend_rows_shared` nor `QBLOCK`.
- It does not discuss `VIPERA_ATTN_KV_SPLIT` (currently on by default) or the window kernel.
- No new dependency, no new `unsafe`.

## 11. Open questions

1. Do `QBLOCK = 8/16` make the gain grow? **Measured (2026-11-22, the same day):** no. Two binaries,
   `QBLOCK = 4` and `= 8`, ran 3 pairs each at both places, chunk 4 (shallow) and chunk 24 (deep):
   shallow 697.86 ms median (Q4) against 706.43 (Q8), deep 876.09 against 883.82 — **Q8 is slower in
   0/3 pairs** in both contexts. `QBLOCK = 16` is worse still. Earlier rounds *looked like* Q8 won at
   depth only because the two binaries were measured at different times with the box drifting ±2%;
   the comparison has to alternate within one run. The constant stays 4.
2. How does the per-row arithmetic at 2.4k positions split between scores, `exp` and the weighted sum?
   No measurement separates those three — needed before promising any number.
3. Should this decision be merged with ADR-003 into one policy for "rounding changes in prefill: show
   NLL, not just ms"?
