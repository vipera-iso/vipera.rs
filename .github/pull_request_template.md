## What changed

<!-- One or two sentences: the behavior, not the file list. -->

## Why

<!-- The problem this solves, or the measurement that motivated it. -->

## Gate

Run from the repository root; CI runs the same three commands.

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo test --workspace --locked`

## Checks

- [ ] No new dependency (`std`, `core`, `alloc` only)
- [ ] New public items have rustdoc (`#![warn(missing_docs)]` is on in every crate)
- [ ] New `unsafe` has a `// SAFETY:` comment, and `unsafe fn` has a `# Safety` section
- [ ] A behavior change comes with a test that fails without it
- [ ] A performance claim comes with a before/after measurement

## Notes for the reviewer

<!-- What the diff does not show: tradeoffs, follow-ups, what is deliberately out of scope. -->
