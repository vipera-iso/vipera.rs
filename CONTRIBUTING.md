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

- No external crates. `std`, `core` and `alloc` only.
- `unsafe` stays in `crates/vipera-backend-cpu/src/simd.rs`, the only file with
  `#![allow(unsafe_code)]`. Every `unsafe` block carries a `// SAFETY:` comment naming the invariant
  that makes it sound, and every `unsafe fn` has a `# Safety` section listing its preconditions.
- No `unwrap()`, `expect()` or `panic!()` in library code. Tests may use them.
- `#![warn(missing_docs)]` is on in every crate, so a public item without a doc block fails clippy.
- A behavior change ships with a test that fails without it.

## Checkpoints

When a piece of work reaches a stopping point, record it in
`specs/001-vipera-rs-engine/checkpoints/YYYY-MM-DD-<name>.md`: what changed, the branch and commit,
the gate output as it actually came back, and what is still open. Keep it short — it is the handoff
to the next session, not a changelog.
