# Capability Map: Vipera Perf Rewrite

| Module id | Responsibility | Depends on |
|---|---|---|
| prefill-tiled | Tiled mul_mat 256x256 for prefill B>=32, unpack once and reuse 256 times | the existing matvec |
| decode-stream | Prefetch + 8-row blocking for fill_rows (decode + prefill GEMV) | the existing simd |
| verify-parity | Parity tests tiled vs matvec, the win_vs_llama bench | both modules above |

Build order: decode-stream -> prefill-tiled -> verify-parity

# Spec: Vipera Perf Rewrite

## Objective
Catch up with llama.cpp on CPU: prefill 35 -> 150+ tok/s, decode holding 23+ tok/s at ctx 32.
Users: running Qwen2.5-1.5B Q4_K_M offline on an i5-11300H.
Success: prefill pp128 >= 100 tok/s, decode tg64 >= 30 tok/s, without breaking bit-exact batching.

## Tech Stack
Rust 2024, unsafe only inside simd.rs, AVX2 + AVX512-VNNI cpuid checks, GGUF Q4_K/Q6_K.

## Commands
Build: cargo build -p vipera-backend-cpu --release
Test: cargo test --workspace
Bench: VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf cargo test -p vipera-backend-cpu --release --test win_vs_llama -- --ignored --nocapture
Clippy: cargo clippy --workspace --all-targets -- -D warnings

## Project Structure
crates/vipera-backend-cpu/src/simd.rs  -> the only unsafe code, prefetch + tiled microkernel
crates/vipera-backend-cpu/src/tiled.rs -> NEW, portable scalar tiled first, AVX later
crates/vipera-backend-cpu/src/quant.rs -> fill_rows + matvec_pooled_batch take the tiled path when B>=32
crates/vipera-backend-cpu/tests/win_vs_llama.rs -> pp/tg bench

## Code Style
target_feature on every kernel that touches intrinsics, #[allow(unsafe_code)] only inside simd.rs,
bit-exact batching: entry e == scalar entry e.

## Testing Strategy
Unit parity of tiled vs dot_scalar for Q4_K/Q6_K, ignored GB/s + tok/s benches, no new dependencies.

## Boundaries
- Always: run cargo test workspace + clippy, keep unsafe forbidden everywhere else
- Ask first: changing the KV layout, raising MAX_BATCH above 8, adding a new dependency
- Never: commit secrets, edit vendor code, delete old tests

## Success Criteria
- [ ] cargo test --workspace passes
- [ ] cargo clippy --workspace --all-targets passes
- [ ] prefill pp128 measured with win_vs_llama, up from the old 35 tok/s
- [ ] decode tg64 no lower than the old 23 tok/s

## Open Questions
- Raise MAX_BATCH from 8 to 16/32 for prefill, or keep 8 to stay safe on registers?
- Do the 8x8 repack like llama.cpp, or only the tiled layout without a repack?
