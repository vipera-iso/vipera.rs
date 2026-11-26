# ADR-005: ba switch đổi số học thành mặc định bật

Status: **Superseded một phần, 2026-11-25.** Phép đo paired end-to-end (§5) cho thấy cả hai switch
mặc định bật là regression trên máy tham chiếu: `VIPERA_Q4K8` 0.49x decode (repack trong mỗi lần gọi)
và `VIPERA_TILED_A` 0.58x prefill. Cả hai trả về **opt-in**; `VIPERA_ATTN_GEMM` vẫn opt-in (§3).
Quyết định gốc giữ nguyên dưới đây làm bản ghi lịch sử.
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

## 5. Re-measurement — 2026-11-25 (supersedes §2)

A/B hai arm, hai vòng xen kẽ đảo thứ tự, `taskset -c 0-7`, `VIPERA_THREADS=8`, cùng GGUF Qwen2.5-1.5B
Q4_K_M. ik_llama.cpp `1180df7` cùng phiên: pp128 253.6 tok/s, tg64 29.2 tok/s.

| Arm | pp128 | decode ctx 32 | kernel Q4_K 1T |
|---|---:|---:|---:|
| default (cả hai bật) | 125.3 | 16.4 | 3.88 GB/s |
| `VIPERA_TILED_A=0`, `VIPERA_Q4K8` mặc định | 122.9 | 32.4 | — |
| cả hai tắt | **213.2** | **32.5** | **12.97 GB/s** |

The wired Q4_K8 path repacks weights inside every call (`quant.rs::fill_rows_q4k8_groups` calls
`make_blocks_q4k8`), so it never amortizes the repack; the lab 1.056x was pre-packed. TILED_A's
+2.6% chunk does not reproduce at pp128. Both defaults reverted to opt-in; chi tiết trong
`docs/benchmarks.md` mục "Q4K8 mặc định bật — đo lại end-to-end" và mục "TILED_A mặc định bật".

## 4. Rollback

Từng cái một, không build lại: `VIPERA_TILED_A=0`, `VIPERA_Q4K8=0`, `VIPERA_PREFIX_CACHE=0`.
Đảo mặc định là một revert của hai hàm `*_enabled` và một dòng trong `serve.rs`.
