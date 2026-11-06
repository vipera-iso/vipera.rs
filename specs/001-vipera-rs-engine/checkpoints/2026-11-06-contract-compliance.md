# Checkpoint 2026-11-06 — contract compliance

**Branch:** `chore/contract-compliance`
**Commit:** `b08022c` — `docs: document every public item and justify every unsafe block`
**Scope:** documentation and lint only; no behavior change.

## What changed

- `crates/vipera-backend-cpu/src/simd.rs`: `hsum` gained a `# Safety` section (AVX2 plus the two
  call sites that already guarantee it), and the blocks in `dot_nibbles` and `q4k_dot` gained
  `// SAFETY:` comments naming the bounds they rely on. The comparison test's `unsafe` call is
  annotated too.
- `crates/vipera-server/src/json.rs`: `Json`, its six variants, and `get`, `as_str`, `as_f64`,
  `as_bool`, `as_array` documented.
- `crates/vipera-server/src/serve.rs`: `command` documented.
- `crates/vipera-runtime/src/engine.rs`: `/// Live counters.` had been glued onto `set_sampler`'s
  doc block, leaving `stats` undocumented; split back apart and `stats` described.
- All nine library crates: `#![warn(missing_docs)]` at the crate root.

## Gate (as run)

```sh
cargo fmt --all -- --check                                  # exit 0
cargo clippy --workspace --all-targets --locked -D warnings  # 0 warnings
cargo test --workspace --locked                             # 187 passed, 0 failed, 0 ignored
```

The lint is live, not inert: `RUSTFLAGS="-D missing_docs" cargo check -p vipera-server -p
vipera-runtime` also passes, which it could not if a public item were still undocumented.

Earlier audit found no real `unwrap()`/`expect()`/`panic!()` outside tests; the four textual hits in
`json.rs` are calls to the parser's own `expect(byte)` method, which returns `Result`.

## Still open

- Not pushed, no PR yet.
- `main` is still the branch everything was committed to before this checkpoint; the branch rule in
  `CONTRIBUTING.md` starts applying from here.
- Next: the Q4_K throughput work, which needs a per-phase measurement before any optimization.
