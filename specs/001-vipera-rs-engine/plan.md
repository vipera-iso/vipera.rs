# Implementation Plan: vipera.rs Phase 1 — Local Inference Engine

**Branch**: `001-vipera-rs-engine` | **Date**: 2026-11-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-vipera-rs-engine/spec.md`

## Summary

Build Phase 1 of a Rust inference engine: load a GGUF model, plan its memory placement before
loading, verify the plan against measured memory, serve an OpenAI-compatible chat endpoint with
streaming, keep conversation context across turns with rollback, and operate it with health,
metrics, auth, and rate limits. Six algorithms and two backends ship with a shared conformance
suite, and every new algorithm or backend is added without editing core code.

The engine is the Cargo workspace at the repository root. The specification, plan, and tasks stay
under `specs/001-vipera-rs-engine/` so Spec Kit remains the single traceability surface; the
`vipera.cpp/` tree and the Python `vipera-opt` planner are separate and not part of this repository.

## Technical Context

**Language/Version**: Rust 1.99.1 (installed: `cargo 1.99.1`, `rustc 1.99.1`), edition 2024

**Primary Dependencies**: **stdlib only for the foundation slices.** Each additional crate is admitted
one slice at a time with its justification recorded in `research.md`. Planned admissions, not yet
approved: `memmap2` (model mapping), `axum`/`tokio` (HTTP), `clap` (CLI ergonomics), `serde_json`
(JSON responses), `prometheus`-style metrics. A dependency without a row in `research.md` does not
land (see `CONSTRAINTS.md`, agent rule 2).

**Storage**: model files on the local filesystem (GGUF); no database in Phase 1

**Testing**: `cargo test` (unit + integration), a shared conformance suite that every algorithm and
backend must pass; tests must run with no GPU present

**Target Platform**: Linux x86_64 (CUDA optional at runtime, CPU-only must work), no cloud dependency

**Project Type**: Rust Cargo workspace — libraries plus CLI and server binaries

**Performance Goals**: from spec SCs — predicted peak memory within 10% of measured; follow-up-turn
TTFT ≤ 40% of cold TTFT at 4K context; 24h run with error rate < 0.1% and < 5% memory growth

**Constraints**: no out-of-memory during load (infeasible plans refused up front), offline test suite,
no unsafe code outside the CUDA FFI boundary, single node and single process, `--json` output stable

**Scale/Scope**: 8 core crates, 6 algorithms, 2 backends, ~3 months; Phase 2+ (Metal, ROCm, Vulkan,
WASM, plugins, distributed, extra SDKs) explicitly out of scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution in `.specify/memory/constitution.md` (v1.1.1) governs `vipera-opt` at the repository
root. Its principles are written for a measurement-driven optimization tool, and they transfer to the
engine with one substitution (section 2 of the spec, Assumptions):

| Gate | Applies how | Status |
|------|-------------|--------|
| I. Measurement Before Claims | Memory and throughput claims come from a run and are recorded side by side with the prediction | PASS — SC-002/SC-007 make it checkable |
| II. Feasibility Outranks Throughput | Planner refuses infeasible placements instead of degrading silently | PASS — FR-003/FR-004 |
| III. Reproducible Benchmarks | Fingerprint + model identity + placement recorded with every number | PASS — FR-005, quickstart records the same fields |
| IV. Explicit CLI Contract | `plan`/`verify` keep stable `--json`; `profile` stays `key: value` | PASS — contracts/cli.md freezes it |
| V. Scope Discipline | Phase 2+ blueprint items are excluded, not deferred silently | PASS — spec Assumptions list the exclusions |
| VI. Tests Travel With Behavior Changes | GGUF parsing, estimation, and planner logic ship with tests | PASS — no GPU required by the suite |
| Governance → CONSTRAINTS.md | Floor + ratchet applies; new crates need justification | PASS — see project structure rules below |

**Known gap**: no constitution exists yet for `vipera.rs` itself. This plan defers creating one until
the workspace has its own root, and records the gap rather than hiding it. Nothing in this plan
depends on the engine having separate governance.

## Project Structure

### Documentation (this feature)

```text
specs/001-vipera-rs-engine/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── cli.md
│   └── http-chat.md
├── checklists/requirements.md
└── spec.md
```

### Source Code (repository root)

```text
.
├── Cargo.toml                     # workspace: library crates under crates/, the CLI under bins/
├── Cargo.lock
├── rust-toolchain.toml
├── .github/workflows/ci.yml       # fmt, clippy -D warnings, workspace tests
├── crates/
│   ├── vipera-errors/             # slice 1
│   ├── vipera-core/               # slice 1 (gguf), planner + kv_cache later
│   ├── vipera-traits/             # slice 2
│   ├── vipera-conformance/        # slice 2
│   ├── vipera-runtime/            # slice 4
│   ├── vipera-backend-cpu/        # slice 4b
│   ├── vipera-sampler/            # slice 3
│   ├── vipera-tokenizer/          # slice 3
│   └── vipera-server/             # slice 5 (HTTP surface, streaming, metrics)
├── bins/vipera/                   # model-info, plan, profile, tokenize, sample, verify, generate, serve
├── scripts/                       # parity-tokenize.sh
└── specs/001-vipera-rs-engine/    # spec, plan, tasks, contracts
```

**Structure Decision**: a Cargo workspace, one crate per concern, because the spec's extensibility
requirement (FR-018/FR-023) is satisfied by crate boundaries rather than by module conventions inside
a monolith. Algorithms and backends will be their own crates registered at startup; `vipera-core`
never depends on them.

### Delivery slices

Each slice compiles, passes `cargo test`, and leaves the workspace green. A slice that is not green
does not merge into the next.

1. **Slice 1 — model introspection**: workspace, `vipera-errors`, `vipera-core::gguf` (header, version,
   metadata, tensor info, validation), CLI `vipera model-info <path> [--json]`. Delivers FR-001.
2. **Slice 2 — hardware + planner**: `vipera-traits` (backend, tensor, tensor traits),
   `vipera-core::planner` (estimator, layer split, feasibility), CLI `vipera plan`. Delivers FR-002,
   FR-003, FR-004, and FR-021 partially.
3. **Slice 3 — tokenizer + sampler** *(delivered: `vipera-tokenizer`, `vipera-sampler`)*: three
   segmentation schemes (unigram over piece scores, greedy longest-match, and byte-level BPE with
   `tokenizer.ggml.merges` for GPT-2-derived models), atomic control tokens, byte fallback, sampling
   policies, deterministic seed, and chat template rendering for four recognised template families.
   Delivers FR-012 and FR-013. Chat rendering refuses templates it does not recognise instead of
   approximating them; see `contracts/cli.md`.
4. **Slice 4 — runtime** *(delivered: `vipera-runtime`, `vipera verify`)*: paged KV blocks backed by a
   real buffer, sequence lifecycle, the seven-step pipeline with per-stage timings and one structured
   log line per request, prefix reuse across turns, snapshots and rollback, release on finish, cancel,
   and failure, then `vipera verify` to close the prediction loop. Delivers FR-005, FR-009, FR-010,
   FR-011, FR-022.

   **Boundary**: this slice alone ships no compute backend; slice 4b adds one. `ForwardPass` is the
   seam, so the loop and its tests run against a deterministic model. `vipera verify` refuses a
   placement with layers on the device (exit 2) and measures a CPU placement, which is where `VmHWM`
   is meaningful. Steps 2, 3, 4, and 6 of the pipeline are identity strategies until the algorithm
   crates land; they are stepped and timed so their cost is visible.
4b. **Slice 4b — CPU compute backend** *(delivered: `vipera-backend-cpu`, `vipera generate`)*:
   F32/F16/Q8_0/Q4_K/Q6_K row dequantization, RMSNorm, NeoX rotary embeddings, causal grouped-query
   attention, SwiGLU, and K/V written into the runtime's own paged buffer as half precision. Config
   comes from the file's metadata (layer count, head counts, epsilon, rotary base), so a second model
   of the same family needs no code change. Delivers the compute half of FR-005 and turns the runtime
   loop into something a user can run: `vipera generate` reports text, tokens per second, and the
   stage breakdown.
5. **Slice 5 — service**: HTTP chat completions with streaming, health/readiness, metrics, auth, rate
   limits. Delivers FR-006, FR-007, FR-008, FR-014 to FR-017.
6. **Slice 6 — extensibility proof**: registry, two backends, six algorithms, conformance suite.
   Delivers FR-018, FR-019, FR-023, FR-024 and SC-006, SC-008.

## Complexity Tracking

| Decision | Why it is not simpler | Revisit when |
|----------|-----------------------|--------------|
| Workspace with one crate per concern instead of one crate | FR-018 requires adding algorithms without editing core; crate boundaries make that mechanical | never — this is the point |
| A shared conformance suite before the second algorithm exists | FR-019/FR-023 must be enforced by something other than the author's opinion | never |
| Stdlib-only start | `CONSTRAINTS.md` forbids unapproved dependencies; a GGUF parser needs none | as soon as a slice needs async, JSON, or CLI parsing |
| Planner carries a measured 16 MiB device reserve instead of modelling CUDA context and compute buffers | Phase 1 predicts before loading; a constant measured against a reference implementation is honest, a faked model of allocator behaviour is not | when `vipera verify` measures our own runtime, replace the constant with a measured term |
| `unsafe` is allowed in exactly one module | blocking over the reduction, precomputed activation sums and per-matvec threads are safe and took the 1.5B model from 22.7 s to 4.6 s of compute per twelve tokens; the Q4_K AVX2 kernel is another 1.29× end to end (3.56 s), and `std::arch` cannot be written without `unsafe` | the workspace keeps `unsafe_code = "forbid"`; `vipera-backend-cpu` declares its own `deny` and allows it only in `src/simd.rs`, which is tested against the scalar kernel. AVX2 for Q6_K/Q8_0/F32 measured slower than scalar on this CPU, so it is not wired |
| Backend implements one block layout (qwen2/llama) and one KV type (f16) | Every number it needs comes from metadata, and refusing another architecture by name is honest; a half-implemented second family would be silently wrong | when a second family is actually required, add it with its own conformance run |
| `vipera verify` measures host memory only | `VmHWM` is a real measurement; a device placement cannot be allocated without a device backend, and inventing a device number would be the decorative estimate FR-005 exists to prevent | when the CUDA allocator lands, measure the device side with the same delta method |
