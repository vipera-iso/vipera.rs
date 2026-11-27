# ADR-007: per-architecture RoPE pairing and `rope_freqs.weight`

Status: **Accepted, not yet verified on a real Llama file.** Date: 2026-11-25.

## 1. Problem

The CPU backend accepts the `llama` architecture but only implements NeoX-style RoPE (pairs
`(i, i + head_dim/2)`). llama.cpp uses `LLAMA_ROPE_TYPE_NORM` (adjacent pairs `(2i, 2i+1)`) for `llama`,
and the converter permutes Q/K of that architecture into exactly that layout. Rotating such a file with
NeoX pairs the wrong values.

Second, Llama 3.x GGUFs do not write `rope.scaling.type`; the scaling lives in the `rope_freqs.weight`
tensor (one factor per pair, which ggml divides the angle by). Nothing in this repo reads that tensor.

There is no evidence in the repo that a `llama` model has ever run through this backend; only Qwen2.5 has.

## 2. Change

- `RopeLayout { NeoX, Norm }`, chosen from `general.architecture`: `llama` is `Norm`, everything else
  `NeoX`. Qwen2/Qwen3 change no bits (the test
  `rope_without_scaling_is_bit_identical_to_the_original_table` still passes).
- `rope_freqs.weight` is read when present (F32, `head_dim/2` elements, finite and positive) and
  divides each pair's angle before the scaling rule sees it, as ggml's `rope_yarn(theta / ff, ...)`
  does. Heads wider than 128 values are rejected by name rather than rotated wrongly.
- Tests: factors against an f64 reference; all-1.0 factors bit-identical to having no tensor; `Norm`
  equal to `NeoX` on permuted data (bit comparison); the loader picks the layout from the architecture.

## 3. Not done / not known

- A real Llama 3.x file has not been run: there is no such model in the sandbox. Logits must be
  compared with llama.cpp before trusting this.
- The GPU backend still rejects every `rope.scaling.type` and does not read `rope_freqs.weight`.
- `longrope` (Phi-3, `rope_factors_long/short`) is still rejected.
