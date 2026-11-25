# ADR-005: ba switch đổi số học thành mặc định bật

Status: **Accepted, trừ `VIPERA_ATTN_GEMM`.** Chủ repo chọn chấp nhận các đánh đổi tốc độ-lấy-số-học
cho `VIPERA_TILED_A` và `VIPERA_Q4K8`; `VIPERA_ATTN_GEMM` bị cổng chặn và trở lại opt-in (§3).
Ngày: 2026-11-25 · Supersedes: ADR-006 (một phần). ADR-004 vẫn đúng (giữ opt-in).

## 1. Context

Ba đường nhanh đã code xong nhưng để tắt, vì mỗi cái đổi output và quyết định đó thuộc về owner:

| Switch | Đổi gì | Giá đã đo trước đó | Lợi đã đo trước đó |
|---|---|---|---|
| `VIPERA_ATTN_GEMM` (`attention_qblock_enabled`) | attention theo tile 32x64 qua GEMM, tổng softmax gom lại | lệch một greedy token (probe 40 token: `1000 times` -> `100 times,`) | attention -48%, chunk -13% ở 3.7k token; -6.7% ở 1.5k |
| `VIPERA_TILED_A` | activation của band tiled lượng tử hoá một scale mỗi 256 thay vì mỗi 32 | NLL +0.45..0.69% trên mọi văn bản (W-1) | probe kernel +1.24..1.36x, chunk 8T suy ra +14..20% |
| `VIPERA_Q4K8` | decode Q4_K qua repack 8x8 và activation `block_q8_K` (cũng một scale mỗi 256) | cùng loại với dòng trên | lab: 1.34x ở 1 thread, **1.056x ở 8 thread** (ADR-006 §3) |

Các số trong bảng lấy từ `docs/benchmarks.md`, ADR-004 và ADR-006. Bản ADR này **không đo lại** chúng.

## 2. Decision

- Hai switch **mặc định bật**: `VIPERA_TILED_A`, `VIPERA_Q4K8`. `=0`, `false`, `off` hoặc `no` tắt
từng cái (`tiled::env_default_on`).
- `VIPERA_ATTN_GEMM` **giữ opt-in** (ADR-004 vẫn đúng): xem §3, cổng đã chặn nó.
- Điều kiện mở lại của ADR-006 §5 ("ADR-005 quyết định per-256 cho đường activation") được thoả bằng quyết định này.
- Hai thay đổi đi kèm, không đổi số học của đường đang chạy:
  - KV `q8_0` đi qua kernel tile (`attend_rows_shared`, decode qua `KvLayout`) thay vì rơi về đường từng entry khi `head_dim` chia hết cho 32. Test: `qblock_attention_matches_the_per_entry_kernel_on_a_q8_0_cache`.
  - `vipera serve` bật prefix cache. `VIPERA_PREFIX_CACHE=0` tắt.

## 3. Chưa kiểm chứng

- Không có phép đo paired nào (một process, hai arm xen kẽ, `taskset`, median >= 5 cặp) cho bản này. Sandbox viết ra nó không có model GGUF và chỉ có 1 core.
- Các test cần model chưa chạy: `prefix_cache`, `logits_reference`, `llama_continuations`, `q4k_f32_continuation`. Fixture continuation của llama.cpp có thể lệch một token sau thay đổi này; nếu vậy đó là hệ quả đã chấp nhận ở §1, và fixture cần sinh lại thay vì nới ngưỡng.
- NLL 6 văn bản của ADR-003 chưa chạy lại với ba switch bật cùng lúc; mỗi số ở §1 là của từng switch riêng.
- Prefix cache giả định một vị trí có cùng bit bất kể prompt bị chia chunk thế nào.
- **Đã kiểm tra 2026-11-25, kết quả âm — hai lần.** Với `VIPERA_ATTN_GEMM` bật, `cargo test --workspace` đỏ ở
  `logits_each_matches_single_passes` (win_vs_llama.rs, "position 1 differs"). Nguyên nhân không phải
  độ rộng GEMM mà là chọn đường: `attend_entries` chỉ vào kernel tile khi
  `entries.div_ceil(QBLOCK) > 1` (tức từ token thứ năm của chunk), nên cùng một vị trí được tính bằng
  kernel tile trong chunk 8 token và bằng kernel từng entry trong pass 2 token — hai thứ tự cộng
  softmax khác nhau (~1e-7). Ablation từng switch: `VIPERA_ATTN_GEMM=0` test xanh (76 s),
  `VIPERA_TILED_A=0` và `VIPERA_Q4K8=0` vẫn đỏ. Bản patch 11:40 khẳng định owner đã chấp nhận đánh đổi
  này nhưng không sửa test; chạy lại đúng test đó vẫn đỏ ở "position 1 differs", nên dòng
  `VIPERA_ATTN_GEMM` ở §2 tiếp tục opt-in. Speculation (bật mặc định cho lượt greedy) kiểm draft bằng
  một pass batch và prefix cache trong `serve` cũng dựa vào cùng giả định đó, nên tile không thể là
  mặc định cho tới khi hai kernel chia sẻ một thứ tự cộng — hoặc owner đổi hẳn policy và sửa test/
  invariant đó trước, chứ không phải nới ngưỡng.

## 4. Rollback

Từng cái một, không build lại: `VIPERA_TILED_A=0`, `VIPERA_Q4K8=0`, `VIPERA_PREFIX_CACHE=0`.
Đảo mặc định là một revert của hai hàm `*_enabled` và một dòng trong `serve.rs`.
