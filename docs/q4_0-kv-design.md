# Q4_0 KV cache — design note

**Status: design only, not implemented.** No source file was changed for this note. Date: 2026-11-28.
Scope: the CPU backend's paged KV cache; the CUDA backend stores f16 only and refuses other types
(`bins/vipera/src/main.rs:709`). This note records the layout, the byte math, where dequantization
happens in attention, the quality risk, and the estimated work.

## 1. What exists today

- `KvCacheType` already has `Q4_0`: the CLI accepts `f16|q8_0|q4_0` for `plan`, `verify`, `run`, and
  `score`, and `KvCacheType::bytes_per_element()` already returns `18.0/32.0`
  (`crates/vipera-core/src/planner.rs:27-67`). `ModelShape::kv_bytes()` therefore already sizes a
  q4_0 cache correctly, and `GgmlType::Q4_0` already declares 32 values per 18-byte block
  (`crates/vipera-core/src/gguf/types.rs:116,126`). **No planner change is needed.**
- The CPU backend does not implement it. `KvLayout` has only `F16` and `Q8_0`;
  `KvLayout::of(KvCacheType::Q4_0)` returns `None` (`crates/vipera-backend-cpu/src/model.rs:249-261`),
  so `ModelConfig::kv_bytes_per_token_for(Q4_0)` returns `None` (model.rs:204-211). `run` and `score`
  turn that into "this model cannot store its KV cache as q4_0" (main.rs:691, 956).
- `kv_layout()` reconstructs the layout from `bytes_per_token` by comparing against the F16 and Q8_0
  candidates only (model.rs:217-240), so even a hand-fed q4_0 buffer would be refused.
- The consequence is already documented: `docs/benchmarks.md:853-859` reports the prefix-cache test
  skipping Q4_0 because the backend cannot carry it.

## 2. Layout and bytes per token

Match the block geometry of GGML's Q4_0 — the format `GgufType` already declares — rather than
inventing a new one: **18 bytes per 32 values** = one `f16` scale `d` (2 bytes) plus 16 bytes of
packed 4-bit codes. KV rows are backend-private, so byte-for-byte compatibility with GGML *weight*
blocks is not required; what must agree is `18/32` so the planner and the runtime cannot drift.
For value `i` in a block, the code is the low nibble of `qs[i/2]` when `i < 16` and the high nibble
when `i >= 16`;
dequantization is `(code as f32 - 8.0) * d`, i.e. a 16-code range centred on 8. Encoding takes
`d = max(|values|) / 8.0`, quantizes `round(v/d) + 8` clamped to `0..15`, and writes `d = 0` with all
codes zero for an all-zero block. This keeps encode/decode strictly a function of the row's values,
which prefix-copy and ring reuse rely on.

Each cached position owns one K row and one V row per layer, each row `width = kv_heads * head_dim`
values wide, laid out as `width/32` Q4_0 blocks:

```
bytes_per_row(width) = width / 32 * 18
bytes_per_token      = 2 * layers * bytes_per_row(width)
                     = 2 * layers * width * 18/32
```

That is exactly `ModelShape::kv_bytes()`'s `2 * layers * ctx * n_head_kv * head_dim * 18/32`, so the
planner's prediction and the runtime's allocation cannot drift (the same invariant Q8_0 maintains
with `34/32`).

| type | bytes / 32 values | bytes / value | B/token, 28 layers / 2 KV heads / head_dim 128 | at ctx 32768 |
|---|---:|---:|---:|---:|
| f16 | 64 | 2.0000 | 28,672 | 896 MiB |
| q8_0 | 34 | 1.0625 | 15,232 | 476 MiB |
| q4_0 | 18 | 0.5625 | 8,064 | 252 MiB |

Q4_0 saves **71.9%** of the f16 cache and **47.1%** of the q8_0 cache. The 252 MiB figure is not a
new prediction: it is `planner.rs:191-206` evaluated at `18/32`, the same arithmetic that reports
896 MiB for f16 today (`docs/benchmarks.md` uses 939 MB decimal for that figure).

**Alignment.** A row is a flat sequence of 32-value blocks, so the base encode/decode path needs only
`width % 32 == 0` — the same condition Q8_0 checks at model.rs:207. The qblock prefill kernel slices a
row *per head* (model.rs:4104, 4192), so the fast path additionally needs `head_dim % 32 == 0`, mirroring
the Q8_0 gate at model.rs:4485; otherwise the chunk stays on the per-entry path. The planner has no
divisibility check for either type today; run/score must keep refusing an unrepresentable width by
name rather than silently storing a partial block.

Nothing else in the cache changes: the block table, 16-token blocks, ring aliasing, layer-major vs
token-major row order, prefix parking/copying, and trim all move opaque `bytes_per_token` chunks and
never interpret a row's contents.

## 3. Codec and layout wiring (where the code would go)

- `crates/vipera-backend-cpu/src/compress.rs`: add `encode_q4_0(values, &mut [u8])` and
  `decode_q4_0(&[u8], &mut [f32])` next to the q8_0 pair (compress.rs:62-100), reusing
  `f32_to_f16` / `f16_to_f32` from `quant.rs` exactly as Q8_0 does. Add a `Q4Kv: KvCompression`
  policy with the same length and non-finite refusal contract as `Q8Kv::compress`
  (compress.rs:27-55): a row that is not a whole number of 32-value blocks, or holds a NaN/inf, is
  left unencoded rather than approximated, because a linear scale cannot express it.
- `crates/vipera-backend-cpu/src/model.rs`: add `Q4_0` to `KvLayout` and to all five of its methods —
  `of` (map the planner type), `name` (`"q4_0"`), `bytes_per_row` (`width/32*18`), and `encode` /
  `decode` (check divisibility, delegate to compress.rs). Add it to the `kv_layout()` candidate list
  (model.rs:220) and update the refusal text at model.rs:230-239 so it names all three types. With
  that, `kv_bytes_per_token_for` returns `Some`, and `run` / `score` work with no CLI change at all.

## 4. Dequantization in attention

KV rows are written once per token per layer (`KvView::write`, model.rs:1783-1818) but read once per
position per head per attention pass. So encoding cost is negligible and decoding cost is on the hot
path. The reader is only ever `KvRead` / `load_layer`; there are three arms:

1. **Decode and short-history attention — `attention_streaming_heads` (model.rs:2562-2897).** Its
   in-place F16 fast path is explicitly gated on `kv.layout == KvLayout::F16` (model.rs:2588), so a
   q4_0 cache naturally falls into the generic arm: `KvLayout::decode` fills the one-row `row`
   buffer in pass 1 for scoring (model.rs:2628-2631) and again in pass 2 for the V fold
   (model.rs:2851-2854). **No new branch is needed** — adding `Q4_0` to `KvLayout::decode` is enough.
   The same applies to the window / chunked variants (model.rs:2951, 3350, 3605), which only use the
   F16 gate to choose the in-place kernels, and to `load_layer` (model.rs:1979-2041), which decodes
   through `KvLayout::decode` too.
2. **Chunk-prefill qblock kernel — `attend_rows_shared` (model.rs:4085 onward).** The `match layout`
   at model.rs:4195 has F16 and Q8_0 arms. A Q4_0 arm decodes one head's `head_dim` values with
   `decode_q4_0` into the existing `head_row` scratch (allocated at model.rs:4126-4131), transposes
   them into `keys_f32[row + dim*fold_pad]` exactly as the Q8_0 arm does, and decodes V into
   `values_f32`. The dispatch gate at model.rs:4485 must admit `Q4_0` only when
   `head_dim % 32 == 0`; otherwise the chunk keeps using the per-entry path, which is correct via
   arm 1. This is the only place new attention code is required.
3. **Prefill writes.** A chunk writes its KV rows before any of its entries reads them
   (`KvView::write`); nothing about the quantizer depends on chunk boundaries, so the prefix-cache
   property (row = f(token, position)) survives.

**Cost model, honestly stated.** Q4_0 cuts cache traffic to 56% of f16 / 53% of q8_0 per byte read,
but dequant is now paid on every read, not once on write, and none of the in-place SIMD helpers
(`dot_f16_f32`, `axpy_f16_f32`, `dot_f16_group`, model.rs:2667, 2885, 2613) can consume a q4_0 row.
The 4-bit unpack (two nibbles plus an f16 scale per 32 values) is cheap *if* it stays in a tight
loop, but an unvectorized unpack can eat the bandwidth saving. The repo's precedent for this kind of
claim is ADR-006: a measured kernel win was not enough to change a default because it changed the
arithmetic. Accordingly, Q4_0 should stay **opt-in via the existing `--kv-type q4_0`**; no new
environment switch is warranted.

## 5. Quality risk

**Unmeasured on this repository — everything in this section is a proposed gate, not a result.**
The arithmetic risk is concrete: Q4_0 carries **4.5 bits per value** (18 bytes per 32 values,
versus q8_0's 8.5), and with `d = max_abs/8` its step is about `max_abs/8` — roughly **16x coarser**
than q8_0's `max_abs/127`. Because the code range is centred at 8 rather than
symmetric about the row's mean, one outlier lowers resolution for all 32 values in its block. K and V
both get this treatment, so the error enters the score dot and the value sum at every layer and every
position — it compounds across depth much more than a single weight-format change does. This is the
same class of decision ADR-004 rejected and ADR-005 reserved for the owner (see ADR-006 §2,
`docs/design/ADR-006-q4k8-repack-default.md:41-45`); it must not become a silent default.

The harness already exists:

- `vipera score --text-file … --kv-type f16|q8_0|q4_0` prints mean NLL and perplexity
  (`bins/vipera/src/main.rs:1023-1028`) using one fresh cache per text. Compare the same tokens,
  tokenizer, and `--ctx`, with f16 as baseline and q8_0 as the known-good quantized control.
- `prefix_cache_gives_the_same_tokens_for_every_kv_type`
  (`crates/vipera-backend-cpu/tests/prefix_cache.rs:142-172`) already loops over Q4_0 and will start
  executing once `kv_bytes_per_token_for` returns `Some`. It checks that copying a cached prefix
  gives the same tokens, **not** that quality matches f16.
- `win_vs_llama::kv_quantization_tradeoff` (tests/win_vs_llama.rs:234-288) is the existing paired
  speed/RAM harness; extend its two arms to three for the performance side.

Proposed acceptance gates, to be replaced by measurements:

| gate | threshold | falsification |
|---|---|---|
| mean NLL vs f16 | ≤ +0.5% on a fixed ≥1000-token corpus | revert the q4_0 default-to-opt-in claim and record the delta |
| mean NLL vs q8_0 | ≤ +0.2% on the same corpus | document q4_0 as memory-only, not quality-neutral |
| greedy continuation | 480-token shared prefix, 12 tokens, ≥10 identical to f16 | record first divergence and position |
| decode latency vs q8_0 | paired alternating arms at ctx 1024 and 4096, ≤ +5% ms/step | ship as memory-saving option only, with the penalty in `docs/benchmarks.md` |

These thresholds follow the repo's convention of fixing a falsifiable gate up front (see
`docs/design/q3k-band-decode-hoist-plan.md:77-88`) and sit above the machine's ~3% noise floor. No
claim about q4_0's actual NLL or speed should be made until they are run; a passing quality gate but
failing speed gate is still a useful opt-in for long contexts, where 47% less cache can be the
difference between fitting and not fitting.

## 6. Effort and sequence

Rough order, ~1–2 focused days total (estimate, not a measurement):

1. **Codec (~0.5 day):** `encode_q4_0` / `decode_q4_0` / `Q4Kv` in `compress.rs`, with round-trip,
   zero-row, partial-block, and non-finite tests mirroring compress.rs:113-170.
2. **Wiring (~0.25 day):** `KvLayout::Q4_0` and the `kv_layout()` candidate list in `model.rs`;
   run/score then accept `--kv-type q4_0` with no CLI edits.
3. **Attention dequant (~0.25 day):** the Q4_0 arm at model.rs:4195 and the gate at 4485;
   add Q4_0 to the streaming/oracle parity test at model.rs:6415 and the qblock test at 6194.
4. **Measurement (~0.5 day):** score/NLL triple comparison, prefix-cache test, and the paired
   latency harness; write the numbers into `docs/benchmarks.md` and this note's status line.

**Out of scope:** the CUDA backend (f16-only by explicit refusal); a new SIMD nibble-unpack kernel;
the MLA latent cache (`crates/vipera-backend-cpu/src/mla.rs`), which has its own layout and never
touches `KvLayout`; planner changes. Each of those would be a separate decision.

## 7. Open questions

- **Divisibility mismatch:** the planner will size q4_0 (and q8_0) for a model whose `head_dim` is
  not a multiple of 32, while run/score refuse it. Keep that explicit refusal, or add a planning-time
  check? Consistency argues for the latter, but it is a shared q8_0 issue, not a q4_0-only change.
- **Vectorizing the unpack:** start scalar and measure; only if the latency gate fails should a
  SIMD unpack be considered, and it must preserve the per-position summation order the attention
  tests pin (`model.rs:2097-2134`).
- **Long context:** the quality and speed gates above stop at 4096; at 32k the memory saving is
  largest but untested. If q4_0 is to be recommended for large contexts, re-run the corpus and
  paired decode there before saying so.
