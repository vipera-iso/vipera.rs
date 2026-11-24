# Contributing

## The gate

Every change must pass these three commands, from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

CI runs the same three (`.github/workflows/ci.yml`). A change that passes locally with different
flags is not passing — `--locked` is there because a stale `Cargo.lock` fails on CI and nowhere else.

## Branches and commits

Work on a branch, never on `main`:

```sh
git switch -c <type>/<short-name>
```

Commits follow Conventional Commits: `<type>(<scope>): <what changed>` in the imperative, with a
body that says why. The types are the usual set: `feat`, `fix`, `docs`, `perf`, `refactor`, `test`,
`chore`, `build`, `ci`.

Merge through a pull request, and treat the CI run for the merge commit as the confirmation. `main`
is expected to be green at every commit, so a local run alone does not prove the merge is safe.

## Rules the code is held to

- No external crates. `std`, `core` and `alloc` only. The single recorded exception is `rustls`
  for TLS in `vipera serve` (`EXC-002` in `CONSTRAINTS.md`); anything else needs its own row there,
  with an owner and an expiry, or it is a removal rather than an exception.
- `unsafe` stays in `crates/vipera-backend-cpu/src/simd.rs`, the only file with
  `#![allow(unsafe_code)]`. Every `unsafe` block carries a `// SAFETY:` comment naming the invariant
  that makes it sound, and every `unsafe fn` has a `# Safety` section listing its preconditions.
- No `unwrap()`, `expect()` or `panic!()` in library code. Tests may use them.
- `#![warn(missing_docs)]` is on in every crate, so a public item without a doc block fails clippy.
- A behavior change ships with a test that fails without it.
- A performance change ships with a number, before and after, in `docs/benchmarks.md`. A change
  that cannot show one is reverted; a change that measures neutral is reverted too, and the number
  is recorded first. When the change only moves scheduling, a cache hint or an address, time both
  arms in one process — two separate runs compare two different amounts of interference, which is
  how the same switch once measured 20% and later 34%. `attention_combinations` and
  `kv_row_order_at_1024` are the examples to copy, and the ledger describes the recipe.

## What a performance claim carries

Evidence, not a verb. A `perf` change states, next to the number:

- the **measurement command**, copy-pasteable, with `--release` and any `--ignored` flag;
- the **date** and the **machine**: CPU model, physical cores, RAM, OS;
- the **model** (GGUF arch, quant, size) and the **thread count** (`VIPERA_THREADS`);
- the **baseline and the new number**, measured under the same conditions — a fresh arm against a
  remembered one is not a comparison;
- the **switch that gates it** (`VIPERA_*` variable, feature, or "none — default path"), so the
  change can be turned off.

The withdrawn claim in `README.md` is why this exists: it summed `PROFILE` running averages, which
undercount by ~11%, and recorded no command, date or thread count, so nothing in it could show the
number was wrong before it had already been published.

## Checkpoints

When a piece of work reaches a stopping point, a short note under
`specs/001-vipera-rs-engine/checkpoints/YYYY-MM-DD-<name>.md` — what changed, the branch and commit,
the gate output as it actually came back, and what is still open — is welcome but optional. Nothing
in the workflow depends on it.
