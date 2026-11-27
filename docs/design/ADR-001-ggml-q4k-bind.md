# ADR-001: Bind the ggml Q4_K GEMM behind a feature flag

Status: **Refuted by measurement for Q4_K (2026-11-15) — implementation closed.** The policy part
(EXC-001) remains in force until its expiry, and applies only to the Q6_K question.
Date: 2026-11-13 · Owner: vipera-iso · EXC-001 expiry: 2027-02-11 · Supersedes: none

> **Phase 0 conclusion (2026-11-15).** This ADR's central assumption — that the prefill gap is
> *kernel quality*, so bind `ggml_gemm_q4_K_8x8_q8_K` — was measured at kernel level and is **wrong**.
> Same shape, same B=128, same 1 thread, alternating pairs: the Rust tiled kernel gives **153.7 GFLOPS**
> (8960x1536) / **174.4** (1536x8960) against ggml repacked's **125.1** / **128.8**. The Rust kernel is
> 23-35% faster, so binding would make prefill slower. No Q4_K bind, no C++ toolchain added. Full
> numbers in `docs/benchmarks.md`, "The ggml repacked GEMM is slower than the tiled Rust kernel".
> Consequence: the remaining 18% prefill gap is not in the microkernel (see section 1, conclusion 2 —
> streaming/residency and threading are what must be measured next).

This document replaces the artifact draft of 2026-11-13: that draft did not build and bound the wrong
symbol. Section 10 records the five things that were fixed, and why.

## 1. Context — measurements, not feelings

Measured on this repo (i5-11300H, 8 threads, AVX-VNNI, `qwen2.5-1.5b-q4km.gguf` 1.117 GB),
`win_vs_llama::prefill_vs_decode` + `VIPERA_PROFILE_STEP=1`:

| Metric | Vipera | llama.cpp — measured 2026-11-13, build 5676dfb | Gap |
|---|---|---|---|
| pp64 | 51.3 tok/s | — | — |
| **pp128** | **44.4–45.0 tok/s** | **209.06 tok/s** (8T) / 211.26 (4T) | **4.6x** |
| pp512 | 40.9–41.1 tok/s | — | — |
| **tg (ctx 32)** | **28.1–32.0 tok/s** (31.3–35.2 ms/step) | **37.13** (8T) / **39.04** (4T) tok/s | **1.16–1.34x** |

That baseline already includes patch 0002 (streaming attention) sitting in the tree. It does **not**
slow prefill down: pp128 44.4 without the patch / 45.0 with the gate is run-to-run variation, and an
earlier 47.1 sample does not reproduce. What it does change is decode at depth: ctx 1024 from 56.1 to
51.0 ms/step (+9% tok/s). Every promote threshold in section 7 compares against the 44.4–45.0 / 28.1
row.

Where the time goes:

| Phase | Decode | Prefill (chunk) |
|---|---|---|
| Q4_K dots | 22.94 ms (**64%**) | **74%** |
| Q6_K dots | 11.41 ms (**32%**) | 19% |
| attention | 0.51 ms (1%) → 10–12% at depth ~400 | 3% |
| norm / rest | 0.00 ms / 1.17 ms | 0% / 4% |

Two conclusions shape this ADR:

1. **Decode is memory-bound, but NOT at the ceiling** — measured with STREAM 5.10
   (`docs/benchmarks.md`): read-only **42.8 GB/s** (4 threads) / 41.7 (8 threads), Copy 39.7, Triad 29.6;
   SMT adds nothing. One decode token reads **0.980 GB** (1.111 GB of tensors minus 125 MB of
   `token_embd`, which decode does not read), so the applicable ceiling is the read-only **~43 GB/s** →
   a floor of **22.8 ms/token**. Vipera: 28.8 GB/s (4T, 67%) and 31.3 GB/s (8T, 73%); llama.cpp 39.04
   tok/s (4T) = **38.3 GB/s (89%)**. So the ceiling is real and the headroom is 1.34x at the same thread
   count, but it is **not in the access pattern**: prefetch (`_MM_HINT_T0`, 31.5 vs 40.3 GB/s) and row
   blocking (0% at every thread count) were implemented and refuted by the numbers
   (`docs/benchmarks.md`, "Decode headroom"), and the thread curve shows the first thread already
   reaching ~10 GB/s, i.e. one core's ceiling. Same `vpdpbusd`, C++ does not stream faster than Rust.
2. **Prefill is where the ROI is.** 74% is Q4_K, and the 4.6x gap is tile/repack kernel quality. Three
   prototypes in the repo were refuted by measurement: tiled scalar **0.04 GB/s**, packed VNNI **5.00 vs
   5.52 GB/s**, `B>=32` spill. Rewriting in C++ with the same intrinsics yields the same instruction
   stream → ~0 gain. What already exists is in ggml, so the mechanism to buy it is **binding**.

Scope: **Q4_K only** (`gemv` for decode, `gemm` for prefill). Q6_K is 32% of decode and needs its own
ADR — not folded in here.

## 2. Decision

Feature `ggml-q4k` in `vipera-backend-cpu`, default **off**: bind ggml's Q4_K kernels through an
`extern "C"` shim we write, repack weights into `block_q4_Kx8` once at load, and quantize activations to
`block_q8_K` per batch.

Four rules are broken, and their recorded state:

| Rule broken | Recorded where | State |
|---|---|---|
| Stdlib-only foundation (R3) | `specs/001-vipera-rs-engine/research.md` **R11** | ✅ recorded (purpose, alternative, cost, admission conditions) |
| Every dependency needs approval (`CONSTRAINTS.md` rule 2) | `CONSTRAINTS.md` **Exceptions** table | ✅ **EXC-001**, owner `vipera-iso`, expiry 2027-02-11, rollback = feature off |
| `unsafe_code` only in `simd.rs` | workspace `Cargo.toml` still `forbid` | ✅ unchanged; only `vipera-backend-cpu` relaxes `deny` → `allow` in `ffi_ggml.rs` exactly (the `simd.rs:59` precedent) |
| Win-condition "vipera.rs > llama.cpp" | `HANDOFF.md`, `PLAN-tiled-prefill.md`, `SPEC-vipera-perf.md` | ⛔ **not changed** — needs your signature, see section 10 |

Without those four rows the code does not merge. The first three are done; row 4 is your decision, not
the agent's.

## 3. Seam spec

```
Boundary:        vipera-runtime::ForwardPass  ↔  C ABI shim  ↔  ggml Q4_K kernels, in-process
Mechanism:       C ABI (extern "C") over a .cpp shim; ggml is C++17 with no C linkage
                 → the toolchain is C++ (`CXX`), not `CC`; EXC-001 records this scope
Allocator:       Rust allocates every buffer (repacked weights, q8_K activations, output); ggml does not
Ownership:       weights are REPACKED once at load into a separate buffer (block_q4_Kx8), no longer
                 borrowed from the mmap; activations quantized per batch; output allocated by the caller
Error model:     ggml kernels return void and GGML_ASSERT → abort. Rust must validate shapes BEFORE the
                 call; there is no return code to map onto ViperaError
Threading:       the runtime still owns the stepping thread (one calling thread); the shim spawns none
Cancellation:    not needed — one call is ~1 ms, shorter than any request cancellation threshold
Marshalling:     ~20 ns/call; 28 layers × ~6 matmuls ≈ 168 calls/step over 35.2 ms → 3.4 µs ≈ 0.01%
Quantize:        `ggml_quantize_mat_q8_K_4x8` runs on EVERY batch before the GEMM — not included in the
                 3.4 µs above, must be measured separately (gate: ≤ 0.1% step, see 3.1 and section 6)
Test oracle:     vipera-conformance + a new parity test (argmax + max-abs ≤ 1e-3 against the pooled
                 path); a_batched_step_matches_every_sequence_run_alone stays bit-identical because it
                 compares the same path
```

The 5% rule: 0.01% ≪ 5% — the seam is not the bottleneck. The bottleneck is the kernel, and that is the
target.

### 3.1 Four constraints that must hold before implementing

Derived from fixing the seam (section 11, error #3). The first three are correctness, not optimisation —
getting one wrong gives wrong output or a dead process, not a slow one.

1. **C++ toolchain, not C.** `cc::Build` must use `.cpp(true)`; CI needs `CXX` (g++/clang++, or MSVC C++
   if Windows is opened). The gate will break at **link** time, not at typecheck — which is why section 6
   requires a `build --release` step.
2. **Activation quantization must be measured separately.** Each call also runs
   `ggml_quantize_mat_q8_K_4x8` on the activations before the GEMM. It is cheap, but if it exceeds 0.1%
   of a step then the seam spec's 0.01% is wrong and must be corrected. One bench line inside a
   `kv_attention_cost`-style test, no new infrastructure.
3. **Repack cache key = CPU features × ggml SHA × model hash.** Only applies if option B (repack to a
   file + mmap) is chosen. The `8x8` layout depends on the instruction set (AVX2 vs VNNI vs AVX512), the
   kernel depends on the ggml commit, and the size/shape depends on the model. Missing one component of
   the key → **silently wrong output**, with no crash and no log line. This is a correctness constraint,
   not an optimisation note.
4. **`-DNDEBUG` is mandatory in release.** ggml uses `GGML_ASSERT` → `abort()`: a failed assert inside the
   server kills the whole process, with no `Result` to return. Two layers: the shim/build defines
   `NDEBUG` in release, **and** Rust validates shapes strictly before the call (section 11). Do not rely
   on `if rc != 0`, because the kernels return `void`.

## 4. The real shape of ggml (unlike the draft)

Checked against `vipera.cpp/ggml` (the llama.cpp copy on this machine):

| What the draft assumed | Reality |
|---|---|
| `ggml.c` + `ggml-quants.c`, C | The kernels live in **C++**: `ggml/src/ggml-cpu/repack.cpp` (5,283 lines) + `arch/x86/repack.cpp` (6,407 lines), requiring C++17 and headers from `ggml-base` |
| `ggml_q4k_gemm(w, x, y, n, k, m) -> i32` | Does not exist. The real one: `void ggml_gemm_q4_K_8x8_q8_K(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc)` (`repack.h:174`); decode uses `ggml_gemv_q4_K_8x8_q8_K` (`repack.h:156`) |
| weights borrowed straight from the GGUF mmap | `vx` must be repacked `block_q4_Kx8` (`repack.h:48`); a repack pass at load is required |
| only one FFI function needed | Activation quantization is also needed: `ggml_quantize_mat_q8_K_4x8(const float *x, void *vy, int64_t k)` (`repack.h:148`) |
| link the ggml symbol directly | The symbol is C++ with no C linkage → an `extern "C"` shim (~40 lines) is required |

Cost consequence: add (a) the `block_q4_Kx8` repack — done **in Rust** (pure byte shuffling, no
`unsafe`), (b) `block_q8_K` quantization per batch, (c) the shim. That is why the estimate in section 8
is 2–4 dev-months rather than "1–2 weeks".

## 5. Which files it touches (when implemented)

| File | Work |
|---|---|
| `vendor/ggml/` (new) | pin the commit, ship the LICENSE; keep only the two translation units needed |
| `vendor/ggml/COMMIT` | the pinned SHA (section 10, condition 1) |
| `crates/vipera-backend-cpu/build.rs` (new) | `cc` with `.cpp(true)`, reading `CARGO_FEATURE_GGML_Q4K` (not `cfg!`) |
| `crates/vipera-backend-cpu/src/vipera_ggml_shim.cpp` (new) | the `extern "C"` wrapper, including `repack.h` |
| `crates/vipera-backend-cpu/src/ffi_ggml.rs` (new) | all the `unsafe`, `unsafe extern "C"` (edition 2024), shape validation |
| `crates/vipera-backend-cpu/src/q4k_repack.rs` (new) | `block_q4_Kx8` packing in Rust, with a layout test |
| `crates/vipera-backend-cpu/src/quant.rs` | `#[cfg(feature = "ggml-q4k")]` dispatch in the pooled Q4_K branch |
| `crates/vipera-backend-cpu/Cargo.toml` | `[features] ggml-q4k = []`, `build = "build.rs"`, `[build-dependencies] cc = "1"` (**not** `optional`) |
| `crates/vipera-backend-cpu/tests/ggml_parity.rs` (new) | argmax + max-abs against the pooled path, at both B=1 and B=16 |
| `.github/workflows/ci.yml` | add a `gate-ggml-q4k` job (section 6) |

`vipera-runtime` is untouched: `ForwardPass` is already the right seam.

Do **not** add `cc`/`ggml-q4k` to `Cargo.toml` before `build.rs` and a module that uses it exist —
config without a consumer violates `CONSTRAINTS.md` rule 5 and creates a dead feature.

## 6. CI checklist

The current gate (`.github/workflows/ci.yml`: fmt → clippy `-D warnings` → test `--locked`) must stay
green **unchanged with the feature off** — that is the proof that rollback still works.

- [ ] `cargo fmt --all --check` (unchanged)
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` (feature off — mandatory)
- [ ] `cargo test --workspace --locked` (feature off — mandatory, 0 ignored)
- [ ] `gate-ggml-q4k` job, `ubuntu-latest`, `timeout-minutes: 25`, `actions/checkout@v7` (the version in use):
  - [ ] `cargo clippy --workspace --all-targets --features ggml-q4k --locked -- -D warnings`
  - [ ] `cargo test --workspace --features ggml-q4k --locked`
- [ ] `cargo build --release -p vipera-backend-cpu --features ggml-q4k` — catches link errors (C++, `CXX`)
- [ ] That release build must define `NDEBUG`; verify with a deliberate assert in the shim on the test job
- [ ] A separate bench of `quantize_mat_q8_K_4x8` for B=1 and B=16 (gate ≤ 0.1% of a step)
- [ ] `vendor/ggml/LICENSE` + `COMMIT` present in the repo
- [ ] Windows does **not** enter the matrix in v1: it needs MSVC C++ and its own AVX guards — a separate decision
- [ ] `win_vs_llama` benches do **not** enter the gate (they need a model, `#[ignore]` as today)

## 7. Promote / rollback

Today's baseline (`win_vs_llama::prefill_vs_decode`, model and machine in section 1, including patch
0002): **pp128 44.4–45.0 · pp64 50.0–50.7 · pp512 40.9–41.1 · tg(ctx32) 28.1–32.0 tok/s (31.3–35.2
ms/step).** llama.cpp on the same machine: `pp128 209.06`, `tg64 37.13` (8T) — see
`docs/benchmarks.md`.

Promote (flip the default to on) only when **all** of these hold, over 3 consecutive runs, medians:

| # | Criterion | Threshold |
|---|---|---|
| 1 | prefill | `pp128` **≥ 100 tok/s** (≥2.1x baseline) |
| 2 | decode does not regress | `tg(ctx32)` **≥ 28.4 tok/s** |
| 3 | correctness | argmax matches the pooled path on ≥3 prompts; logits max-abs ≤ 1e-3; B=1 and B=16 |
| 4 | tests | `cargo test --workspace` green in **both** feature configurations, 0 ignored |
| 5 | memory | peak RSS **≤ baseline + 0.75 GB** (one repacked Q4_K copy, roughly the same bytes: 1,152 B per 8 rows vs 8×144 B). Above that is a fail; if even that is unacceptable, repack into a **file cache** and mmap it rather than keeping a second copy in RAM |
| 6 | p99 | p99 per token no worse than 5% above baseline |

Record all six numbers in `HANDOFF.md` before flipping the default. If `pp128` stops at 60–99: **keep the
default off**, close EXC-001, treat it as a CONSIDER, do not promote.

**Rollback:** turn `ggml-q4k` off (default off) and it is done. The code sits behind `#[cfg(feature)]`,
one revert commit, no `ForwardPass` change, no data migration.

## 8. The cost

- CI/build depends on a C++ toolchain in the job that enables the feature; with the feature off it still
  builds as pure Rust, but the `cc` crate is still compiled as a build-dependency (build-deps cannot be
  `optional`).
- `unsafe` surface: one new module with `unsafe extern "C"` — reviewed more carefully than `simd.rs`.
- Supply chain: ~11.7k lines of C++ outside the repo; must pin the commit, ship the LICENSE, and gate the
  object at < 5 MB.
- RSS: the repacked weights are a **second copy** of the Q4_K tensors (the mmap can no longer be
  borrowed) — the largest memory risk, and criterion 5 in section 7. If that is unacceptable, option B is
  to repack once into a file cache (`~/.vipera/repack/<model-hash>.q4k8`) and mmap that, at the price of
  an extra I/O step at load and a cache format that must be versioned.
- Debugging: a crash inside the C++ kernel gives no clean Rust backtrace; ggml `assert` → abort, with no
  error to return.
- Build time and binary size grow in the feature-on configuration.
- In substance the hot path becomes a wrapper around ggml — exactly `polyglot-architect`'s
  self-critique; that is the price accepted by changing the goal (section 10, condition 3).

## 9. Non-goals

- No swapping or rewriting the server, tokenizer, sampler, core or CLI (under 4% of the time, no ROI).
- No rewriting Q4_K in C++ from scratch (2–4 dev-months, ~0 gain against binding).
- No Q6_K binding in this ADR (32% of decode — its own ADR, with its own numbers).
- No binding of the whole ggml pipeline (graph, sampler, KV layout); the KV cache stays with the runtime.
- No default flip before all six criteria in section 7 hold.
- No replacing `vipera-conformance` — the suite remains the oracle.

## 10. Blocking implementation (your decision needed)

1. **Pin the ggml commit + audit license/size.** There is no SHA yet. Gate: the compiled subset's object
   is < 5 MB. Record it in `vendor/ggml/COMMIT` and in section 2 of this ADR.
2. ~~The llama.cpp baseline must be re-measured~~ — **DONE 2026-11-13.** Built upstream 5676dfb,
   CPU-only, `GGML_AVX512=ON`, same machine and quant: `pp128 209.06` (8T) / `211.26` (4T), `tg64 37.13`
   (8T) / `39.04` (4T). The 256 / 34.7 figures in `HANDOFF.md` do not reproduce and were replaced by
   these in `docs/benchmarks.md`. The `pp128 ≥ 100` threshold stands: it is 2.2x the current baseline.
3. **The new win-condition** ("fastest engine, binding allowed") must be written into `HANDOFF.md` /
   `SPEC-vipera-perf.md` in prose, with a date. Without that line, this ADR is only a proposal.

Measurement note: the governor is `powersave` and cannot be changed without root, so the clock swings
between 0.7 and 3.8 GHz and the same binary varies by up to 15% between runs. Every comparison must be a
best-of-N/median within one session, never a single sample.

## 11. Appendix: five things the draft got wrong, and the correct version

| # | Draft | Problem | Correct version |
|---|---|---|---|
| 1 | `[build-dependencies] cc = { optional = true }` | Cargo **forbids** `optional` in build-dependencies → manifest error | plain `cc = "1"`; skip the compile via `CARGO_FEATURE_GGML_Q4K` |
| 2 | `#[cfg(feature = "ggml-q4k")]` inside `build.rs` | Build scripts do not receive `cfg(feature)`; they must read the environment | `std::env::var_os("CARGO_FEATURE_GGML_Q4K").is_none()` → early return |
| 3 | `ggml_q4k_gemm(...) -> i32`, `crate::Error::internal(rc)` | The symbol does not exist; the kernel returns `void`; the repo uses `ViperaError::internal(impl Into<String>)` | an `extern "C"` shim + `ViperaError::internal("...")` |
| 4 | `#[cfg_attr(feature = ..., allow(unsafe_code))] mod ffi_ggml;` | Redundant and easy to get wrong; the file already carries `#![allow(unsafe_code)]` like `simd.rs:59` | `#[cfg(feature = "ggml-q4k")] pub(crate) mod ffi_ggml;` + an inner allow in the file |
| 5 | CI: `actions/checkout@v4`, Windows matrix, clippy missing `-D warnings`/`--locked` | Inconsistent with the existing gate (`@v7`, ubuntu-only, `-D warnings`, `--locked`); C++ on MSVC is separate scope | an ubuntu-only job matching the current gate |

### `build.rs` (the correct version)

```rust
fn main() {
    println!("cargo:rerun-if-changed=vendor/ggml");
    println!("cargo:rerun-if-changed=src/vipera_ggml_shim.cpp");
    println!("cargo:rerun-if-changed=src/ffi_ggml.rs");
    if std::env::var_os("CARGO_FEATURE_GGML_Q4K").is_none() {
        return; // feature off: no C++ toolchain needed
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/vipera_ggml_shim.cpp")
        .file("vendor/ggml/src/ggml-cpu/repack.cpp")
        .file("vendor/ggml/src/ggml-cpu/arch/x86/repack.cpp")
        .include("vendor/ggml/include")
        .include("vendor/ggml/src")
        .include("vendor/ggml/src/ggml-cpu")
        .warnings(false)
        .compile("vipera_ggml_q4k");
}
```

### `ffi_ggml.rs` (the skeleton, edition 2024)

```rust
//! EXC-001: all the `unsafe` of the `ggml-q4k` feature lives in this file.
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use vipera_errors::ViperaError;

unsafe extern "C" {
    // Our shim — not a ggml C++ symbol (those have no C linkage).
    fn vipera_ggml_gemm_q4k(
        n: c_int,
        out: *mut f32,
        out_row_bytes: usize,
        weights: *const c_void, // block_q4_Kx8, already repacked
        activations: *const c_void, // block_q8_K
        rows: c_int,
        tokens: c_int,
    );
}

/// There is no return code: ggml asserts and aborts, so shapes must be right before the call.
pub(crate) fn gemm_q4k(
    weights: &[u8],
    activations: &[u8],
    out: &mut [f32],
    n: usize,
    rows: usize,
    tokens: usize,
    out_row_bytes: usize,
) -> Result<(), ViperaError> {
    if n == 0 || rows == 0 || tokens == 0 || out_row_bytes < tokens * size_of::<f32>() {
        return Err(ViperaError::internal("ggml_q4k: shape mismatch"));
    }
    // SAFETY: the caller allocated the buffers for the shape checked above; the weights are the
    // backend-owned repacked copy, the activations the q8_K copy quantized in the same call.
    unsafe {
        vipera_ggml_gemm_q4k(
            n as c_int,
            out.as_mut_ptr(),
            out_row_bytes,
            weights.as_ptr().cast(),
            activations.as_ptr().cast(),
            rows as c_int,
            tokens as c_int,
        );
    }
    Ok(())
}
```
