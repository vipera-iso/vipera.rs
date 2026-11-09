# Capability Map: Vipera Perf Rewrite

| Module id | Responsibility | Depends on |
|---|---|---|
| prefill-tiled | Tiled mul_mat 256x256 cho prefill B>=32, unpack 1 lần dùng 256 lần | matvec hiện tại |
| decode-stream | Prefetch + 8-row blocking cho fill_rows (decode + prefill GEMV) | simd hiện tại |
| verify-parity | Parity test tiled vs matvec, bench win_vs_llama | cả hai module trên |

Build order: decode-stream -> prefill-tiled -> verify-parity

# Spec: Vipera Perf Rewrite

## Objective
Đuổi kịp llama.cpp trên CPU: prefill 35 -> 150+ tok/s, decode giữ 23+ tok/s ở ctx 32.
Người dùng: chạy Qwen2.5-1.5B Q4_K_M offline trên i5-11300H.
Success: prefill pp128 >= 100 tok/s, decode tg64 >= 30 tok/s, không vỡ bit-exact batch.

## Tech Stack
Rust 2024, unsafe chỉ trong simd.rs, AVX2 + AVX512-VNNI cpuid check, GGUF Q4_K/Q6_K.

## Commands
Build: cargo build -p vipera-backend-cpu --release
Test: cargo test --workspace
Bench: VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf cargo test -p vipera-backend-cpu --release --test win_vs_llama -- --ignored --nocapture
Clippy: cargo clippy --workspace --all-targets -- -D warnings

## Project Structure
crates/vipera-backend-cpu/src/simd.rs  -> chỉ chỗ được unsafe, prefetch + tiled microkernel
crates/vipera-backend-cpu/src/tiled.rs -> NEW, portable scalar tiled trước, AVX sau
crates/vipera-backend-cpu/src/quant.rs -> fill_rows + matvec_pooled_batch gọi tiled khi B>=32
crates/vipera-backend-cpu/tests/win_vs_llama.rs -> bench pp/tg

## Code Style
target_feature trên mọi kernel chạm intrinsics, #[allow(unsafe_code)] chỉ trong simd.rs, bit-exact batch: entry e == scalar entry e.

## Testing Strategy
Unit parity tiled vs dot_scalar cho Q4_K/Q6_K, ignored bench GB/s + tok/s, không thêm dependency.

## Boundaries
- Always: chạy cargo test workspace + clippy, giữ unsafe forbid nơi khác
- Ask first: đổi KV layout, đổi MAX_BATCH > 8, thêm dependency mới
- Never: commit secrets, sửa vendor, xóa test cũ

## Success Criteria
- [ ] cargo test --workspace pass
- [ ] cargo clippy --workspace --all-targets pass
- [ ] prefill pp128 đo được bằng win_vs_llama, tăng so với 35 tok/s cũ
- [ ] decode tg64 không giảm so với 23 tok/s cũ

## Open Questions
- Có mở MAX_BATCH 8 -> 16/32 cho prefill không, hay giữ 8 để an toàn register?
- Có làm repack 8x8 như llama.cpp không, hay chỉ tiled mà không repack?
