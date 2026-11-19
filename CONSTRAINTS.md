# Constraints

Last reviewed: 2026-11-06 · Owner: repo maintainer · Applies to: every human and every agent that edits this repo.

This file is the written bar for `vipera-opt` (Python, repo root) and for the `vipera.rs` Phase 1 work specified in
`specs/001-vipera-rs-engine/`. It exists because the agent writes far more code than anyone reads line by line, so
"good enough" has to be written down and checked, not remembered.

**Do not weaken this file to make a change pass.** Tightening is silent; loosening needs a row in Exceptions.

## Floor (always enforced)

These five are review-checked on every change. No tooling required, no exceptions without a row below.

- **No new suppression comments**: `# noqa`, `# type: ignore`, `# pyright: ignore`, `// eslint-disable`, `@ts-ignore`.
- **No unimplemented work presented as done**: `raise NotImplementedError` stubs, `pass  # TODO`, empty `except: pass`,
  functions that return a hardcoded value where the caller expects a computed one.
- **No skipped or deleted tests** without the reason in the same change (`@pytest.mark.skip`, `xfail`, removed assertions).
- **No secrets in source**: API keys, tokens, connection strings. Report the location, never the value.
- **This file is not edited in the same change that was failing a rule.**

## Rules for the agent — what "code linh tinh" means here

1. **Work from a written intent.** Every change traces to a spec requirement (`FR-xxx`), a constitution principle, or an
   explicit instruction in the conversation. If it traces to none of those, stop and ask instead of inventing scope.
2. **No unrequested features.** No extra flags, endpoints, config keys, abstractions, or "while I'm here" additions. A new
   file needs a reason the task stated; a new dependency needs explicit approval.
3. **No drive-by refactors.** Do not reformat, rename, reorder imports in, or "clean up" files the task did not name.
   Unrelated improvements are proposed in words, not committed as diff.
4. **Read before writing.** Reuse what exists under `src/vipera/` instead of adding a parallel helper. Never invent an API
   that is not in the tree — check the file, then call it.
5. **No speculative generality.** No interface with one implementation, no plugin hook with no plugin, no config knob with
   no consumer. The smallest change that fully solves the task wins.
6. **No weakened checks to reach green.** Fix the cause. Skipping a test, loosening an assertion, adding a suppression, or
   editing a threshold is a finding, not a fix — and is reported as such.
7. **State assumptions out loud.** When the request is ambiguous, pick the smallest defensible reading and say which one in
   the reply. Guess silently only when the guess is reversible and unobservable.
8. **Evidence before claims.** Run the exact check and quote its result. If a check could not run, say "not run" and why.
   Never describe compiled-but-unverified behavior as tested.
9. **One concern per change.** A bug fix contains no feature; a feature contains no refactor; a refactor changes no behavior.
10. **Respect the non-goals.** For `vipera-opt` these are binding (`constitution` Principle V): no quantization, pruning,
    distillation, speculative decoding, multi-tenant serving, UI, or cloud orchestration. For `vipera.rs`, Phase 2+ items
    in `plan.md` stay out of Phase 1.

## Enforced with numbers

Baselines are measured on 2026-11-06, not aspirational. A number may only move in the stricter direction.

| Dimension | Rule | Checked by | Runs at |
|-----------|------|------------|---------|
| Lint | ruff findings ≤ 154 (today's baseline) | `ruff check src/vipera tests` | task end |
| Types | `mypy src/vipera --ignore-missing-imports` output ≤ 8 lines (today's baseline) | `mypy src/vipera --ignore-missing-imports --no-error-summary` | task end |
| Tests | `pytest tests/` passes once the dev extras are installed; zero new skips | `python3 -m pytest tests/ -q` | task end |
| Rust tests | `cargo test --workspace` — 0 failures, 0 ignored except the release-mode measurements listed in EXC-004, which the gate runs with `--ignored` | `cargo test --workspace`, then the release step in §How to run the checks | task end |
| Rust lint | `cargo clippy --workspace --all-targets --features acme` — 0 warnings | `cargo clippy --workspace --all-targets --locked --features acme` | task end |
| Changed lines | New behavior ships with a test that fails without it | review of the diff | review |

Known debt at baseline, not to be added to: `src/vipera/planner/ladder.py` references an undefined `LadderStep`
(4 errors), `src/vipera/planner/cache.py:64` an undefined `Plan`, `src/vipera/gguf/reader.py:37` uses `any` where
`Any` is meant, and `src/vipera/runtime/verifier.py:26` assigns `None` to a `list[Any]`. Fixing them is allowed and
welcome; adding to the count is not.

## Measured, not yet enforced

| Metric | Today | Direction |
|--------|-------|-----------|
| Project coverage | unknown — `pytest` and `pytest-cov` are not installed in the system interpreter | must be measured before it is enforced |
| Planner unit tests | 0 (tests cover gguf, hardware, a risk gate only) | must increase with every planner change |
| Diff-scoped checks | available — this is a git repository, and every change lands through a PR whose CI run is the confirmation | every performance claim also carries a before/after in `docs/benchmarks.md` |

## Exceptions

| ID | Rule | Path | Reason | Owner | Expires |
|----|------|------|--------|-------|---------|
| EXC-001 | Stdlib-only (`research.md` R3), dependency approval (agent rule 2), `unsafe` confined to `simd.rs` | `crates/vipera-backend-cpu/src/ffi_ggml.rs`, `crates/vipera-backend-cpu/src/vipera_ggml_shim.cpp`, `crates/vipera-backend-cpu/build.rs`, `vendor/ggml/` | Bind the ggml Q4_K GEMM behind feature `ggml-q4k`, **default off**, to close the measured 4.6x prefill gap on `pp128` (45.0 tok/s vs llama.cpp 209.06, same machine, build 5676dfb). Scope of the toolchain exception is `cc` **plus a C++ compiler** (`CXX`): the kernels are C++17 with no C linkage, so a C-only toolchain fails at link, not at typecheck. Rollback: feature off, one revert. Full record: `docs/design/ADR-001-ggml-q4k-bind.md`. | vipera-iso | 2027-02-11 |
| EXC-002 | Stdlib-only (`research.md` R3), dependency approval (agent rule 2) | `crates/vipera-server/Cargo.toml`, `crates/vipera-server/src/serve.rs`, `crates/vipera-server/src/tls.rs`, `bins/vipera/Cargo.toml` (dev-dependency, client side of the TLS test) | Serve HTTPS. Writing TLS by hand is not something to ship, and the workspace had no alternative that was not one; `rustls` 0.23 with `ring`, no default features, is the whole of it. Everything else in the engine stays stdlib-only, and `vipera serve` without `--tls-cert`/`--tls-key` links nothing new at run time. Rollback: drop the two flags, one revert, and the plain-HTTP path is untouched. Full record: the `vipera-server` crate doc comment and the README. | vipera-iso | 2027-02-11 |

| EXC-003 | Stdlib-only (`research.md` R3), dependency approval (agent rule 2) | `crates/vipera-server/Cargo.toml` (`acme` feature), `crates/vipera-server/src/acme.rs` | ACME issuance (RFC 8555) behind feature `acme`, **default off**: `instant-acme` with `hyper-rustls`/`rcgen`/`ring`, plus `tokio` and `serde_json` for the client and the stored account. Writing an ACME client by hand is not something to ship. Nothing in the engine links these; `vipera serve` without `--acme-domain` compiles them out. Rollback: feature off, one revert. | vipera-iso | 2027-02-11 |
| EXC-004 | Rust tests: 0 ignored | `crates/vipera-backend-cpu/tests/prefix_cache.rs`, `crates/vipera-backend-cpu/tests/alignment_probe.rs`, `crates/vipera-backend-cpu/src/{model,quant,simd,tiled}.rs`, `crates/vipera-backend-cpu/tests/{batch,matvec_throughput,win_vs_llama}.rs`, `crates/vipera-ggml/tests/throughput.rs` | Measured on 2026-11-19: a default `cargo test --workspace` reports 52 ignored tests, and they are measurements rather than checks — `bench:`-style timings, plus model-dependent tests that time a prefill. A debug build cannot run them at a usable speed (measured: one 200 MB matvec takes over 300 s in debug), so they stay out of the default run. The two that do assert — prefix cache and the alignment probe — are run by the gate in release with `--ignored`; without a model they print the reason and pass. Tightening means giving the model-dependent ones a checked-in fixture model. | vipera-iso | 2027-02-11 |

An exception needs an owner and an expiry; without both it is a removal, not an exception.

## How to run the checks

```bash
# vipera-opt (Python, repo root)
ruff check src/vipera tests
mypy src/vipera --ignore-missing-imports --no-error-summary
pip install -e ".[dev]"        # once, provides pytest
python3 -m pytest tests/ -q

# vipera.rs (Rust workspace) — both baselines are zero and must stay zero
cd vipera.rs
cargo clippy --workspace --all-targets --locked --features acme
cargo test --workspace --locked
cargo test -p vipera-server --features acme --locked
# The two measurement tests that assert (EXC-004). Release because a debug build cannot time them.
cargo test -p vipera-backend-cpu --release --locked --test prefix_cache --test alignment_probe -- --ignored
```

The commands above are the canonical definition of "green". If `CONSTRAINTS.md` and a wrapper script disagree, this file wins.
