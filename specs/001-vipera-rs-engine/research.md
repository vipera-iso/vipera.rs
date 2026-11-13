# Research — vipera.rs Phase 1

Phase 0 decisions. Each entry states what was chosen, why, and what was rejected. A dependency only
enters the workspace through an entry here (see `CONSTRAINTS.md`, agent rule 2).

## R1. Workspace layout

**Decision**: one Cargo workspace at the repository root, one crate per concern, `crates/*` for
libraries and `bins/vipera` for the CLI.

**Rationale**: spec FR-018 requires adding an algorithm or backend without editing core code. Crate
boundaries make that a build-system fact instead of a code-review convention, and they let the CUDA
backend be excluded from builds on machines without a toolchain.

**Alternatives**: single crate with modules (rejected: no enforced boundary); git submodules per crate
(rejected: version skew, no benefit at this size).

## R2. Toolchain and edition

**Decision**: Rust edition 2024, pinned via `rust-toolchain.toml` to the installed stable 1.99.1.

**Rationale**: edition 2024 is available on the installed compiler and gives `let`-chains and the
2018+ borrow rules without extra dependencies.

**Alternatives**: edition 2021 (rejected: no reason to start behind).

## R3. Stdlib-only foundation

**Decision**: slices 1–2 use only `std`.

**Rationale**: `CONSTRAINTS.md` requires explicit approval for every new dependency, and neither GGUF
parsing, error types, nor hardware detection needs one. Fewer crates means a smaller supply-chain
surface for a project that will run on user machines.

**Alternatives**: `thiserror`/`anyhow` (rejected for now: a small hand-written enum is enough);
`memmap2` (deferred to the loader slice, where mapping matters); `serde` (deferred to the HTTP slice,
where the JSON surface is large enough to justify it).

**Admission rule for later slices**: a dependency is admitted only with its purpose, the alternative
considered, and the size/audit cost recorded here.

## R4. GGUF parsing scope

**Decision**: parse GGUF v2 and v3 headers, metadata, and tensor info; reject v1 and any newer version
with a specific error. Little-endian reads only. Enforce limits while reading (string length, tensor
count, dimension count) so a hostile file cannot trigger a huge allocation.

**Rationale**: v2/v3 are what llama.cpp-era tooling emits; a bounded reader is the difference between a
parser and a `malloc` gadget. FR-001 requires a diagnostic naming the inconsistency, so errors carry
the file offset.

**Alternatives**: full v1 support (rejected: different length encoding, vanishingly rare);
third-party GGUF crate (rejected at this stage: unvetted and we need the metadata anyway).

## R5. Error strategy

**Decision**: `vipera-errors` exposes a single `ViperaError` enum with `Display`, `std::error::Error`,
and `From` conversions from I/O and parse errors. No panics in library code; `unwrap` only in tests.

**Rationale**: one error type keeps call sites honest and keeps the CLI able to map errors to exit
codes in one place.

**Alternatives**: per-crate error enums (rejected: conversion boilerplate with no benefit yet);
`anyhow` (rejected: erases the variant the CLI needs for exit codes).

## R6. Registry without a framework

**Decision**: algorithms and backends register through an explicit `register(...)` call into a
`OnceLock`-guarded table, wrapped by a macro. No `inventory`-style link-time magic.

**Rationale**: link-time registration does not survive in static linking or on all platforms, and its
failure mode is "silently absent". An explicit call is greppable and testable.

**Alternatives**: `inventory` crate (rejected: linker-dependent); `linkme` (rejected: same class of
problem, plus unsafe).

## R7. Conformance suite shape

**Decision**: a `conformance` test crate holds generic test functions; each algorithm/backend crate
invokes them for its implementation.

**Rationale**: SC-008 requires every component to pass "the shared suite". A generic function invoked
by the implementor keeps one definition of the contract while letting each crate own its fixtures.

**Alternatives**: separate integration tests per crate (rejected: contracts drift); one test binary
that imports all implementations (rejected: build coupling).

## R8. HTTP surface (deferred, decided now to avoid rework)

**Decision**: OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming),
`GET /v1/models`, `GET /health`, `GET /ready`, `GET /metrics`. Unknown request fields are ignored, not
rejected.

**Rationale**: spec Assumptions fix the compatibility meaning; ignoring unknown fields is what makes
"change only the base URL" true for real clients.

**Alternatives**: strict schema validation (rejected: breaks clients on their next minor version);
full OpenAI surface including embeddings and tools (out of Phase 1 scope).

## R9. Model file identity

**Decision**: identify a model by a hash of the GGUF metadata and tensor table (header identity), not
by file path or mtime.

**Rationale**: FR-003/FR-005 need a stable identity for caching plans and comparing runs; paths move.

**Alternatives**: full-file hash (rejected: reads gigabytes per plan); path + size (rejected: copies
collide).

## R10. Testing without a GPU

**Decision**: all unit and integration tests run on CPU with synthesized fixtures — tiny GGUF files
generated in the test, no binary fixtures in the repo.

**Rationale**: constitution Principle VI and `CONSTRAINTS.md` require an offline, GPU-free suite;
generated fixtures cannot rot or hide licensing problems.

**Alternatives**: committed binary fixtures (rejected: opaque diffs, size); mocking the parser
(rejected: tests the mock).

## R11. Admission of `cc` and the vendored ggml kernels (EXC-001)

**Decision**: `cc` may enter `[build-dependencies]` of `vipera-backend-cpu`, and a pinned ggml source
tree may live under `vendor/ggml/`, for exactly one purpose: bind `ggml_gemm_q4_K_8x8_q8_K` behind the
`ggml-q4k` feature, default **off**. Nothing else in the workspace may use either.

**Rationale**: Q4_K dots are 64% of a decode step and 74% of a prefill chunk on the measured model; the
prefill gap to llama.cpp is 4.6x (44.4-45.0 vs 209.06 tok/s on `pp128`; llama.cpp 5676dfb, same
machine, measured 2026-11-13) and three in-repo prototypes were falsified by measurement (tiled
scalar 0.04 GB/s, packed VNNI 5.00 vs 5.52 GB/s). The winning kernel is
ledgered as a library, so the mechanism that buys it is a bind, not a rewrite. Recorded as EXC-001 in
`CONSTRAINTS.md` and specified in `docs/design/ADR-001-ggml-q4k-bind.md`.

**Cost recorded before admission**: `cc` is a build-time-only crate (no runtime code, no `links`
namespace used by anything else). The vendored kernel is ~11.7k lines of C++ in two translation units
(`ggml/src/ggml-cpu/repack.cpp` 5 283, `ggml/src/ggml-cpu/arch/x86/repack.cpp` 6 407) plus headers it
includes from `ggml-base`; it needs a C++ compiler, not a C compiler.

**Alternatives**: rewrite the same kernels in Rust (rejected: same instruction stream, 2-4 dev-months,
measured VNNI-vs-maddubs spread is only 0.91-1.14x); bind the whole ggml graph (rejected: replaces the
runtime and the KV ownership the spec fixes); call llama.cpp over a pipe (rejected: ~1 ms per call vs a
35 ms step, and it is the project's benchmark rival, not a library dependency).

**Admission condition**: a dependency here is admitted only with an exception ID in `CONSTRAINTS.md`,
an owner, an expiry, a rollback path, and a test that the feature-**off** build stays green. Missing any
one of those, the change does not merge. This entry is not a blanket permission for other crates.

**Pending before first merge of code**: the pinned ggml commit SHA, the licence/size audit of the vendored
subset (gate: compiled object < 5 MB), and a re-measured llama.cpp baseline. Until those three exist, the
feature is design only.
