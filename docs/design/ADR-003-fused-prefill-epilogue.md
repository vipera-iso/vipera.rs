# ADR-003: Fused epilogue (`fma` + `fnmadd`) cho kernel Q4_K tiled

Status: **Accepted — giữ opt-in, default vẫn là multiply-then-add.** Bật làm default **chưa** được
duyệt và không có trong commit này; điều kiện để lật nằm ở mục 8.
Ngày: 2026-11-22 · Owner: *cần điền* · Supersedes: none

## 1. Context — số đo

Kernel `q4k_matvec_tiled_range` (`crates/vipera-backend-cpu/src/simd.rs`) là đường prefill cho mọi
ma trận Q4_K. Mỗi cặp `(row, sub-block)` nó kết thúc bằng sáu phép vector:

```
main    = (cvt(acc) · scale_f[row][s]) · dxv[s]
minterm = sumv[s] · min_f[row][s]
facc   += main - minterm
```

Ba probe trước đó (bỏ broadcast weight, bỏ epilogue float, bỏ cả hai) đặt epilogue ở **+33%
(8960×1536)** và **+42% (1536×8960)** khi bỏ hẳn — nhưng epilogue là số học bắt buộc: `sum_s
(cvt(acc_s)·scale_s·dx_s − sum_s·min_s)`. Câu hỏi của ADR này là có viết nó bằng ít phép hơn mà
không phạm vào tolerance đã công bố của đường tiled hay không.

Bản fused giữ nguyên `dxv` chung cho cả 16 cột, đưa `scale_f` vào một fma và `min_f` vào một fnmadd:

```
scaled  = cvt(acc) · dxv
sum     = fma(scaled, scale_f[row][s], facc)
facc    = fnmadd(sumv, min_f[row][s], sum)
```

Bốn phép thay sáu. Đo trên i5-11300H, 1 thread, cùng binary, `VIPERA_Q4K_FMA_EPI` bật/tắt:

| arm | 8960×1536 | 1536×8960 | chunk 4 (524 token, 8 thread), 5 cặp xen kẽ |
|---|---:|---:|---:|
| multiply-then-add (default) | 153.4 / 149.5 GFLOPS | 165.2 / 173.2 GFLOPS | median **785.16 ms** |
| fused (`VIPERA_Q4K_FMA_EPI=1`) | 165.1 / 167.4 (**+8.6%**) | 193.4 / 189.3 (**+12%**) | median **746.04 ms** (**−5.0%**, 5/5 cặp) |

Nhánh `if` thêm vào không làm arm default chậm đi: 153.4/149.5 đo được nằm trong dải baseline cũ
(151.7/152.6 ở lần đo trước, cùng shape).

## 2. Chất lượng — phần quyết định

### 2.1 Kiểm soát tính tất định

Trước khi so hai arm phải chắc chắn phép đo không phải nhiễu thread. Cùng prompt, cùng seed, 8 thread,
24 token greedy:

| so sánh | kết quả |
|---|---|
| off vs off (3 lần) | **giống nhau từng token** |
| fma vs fma (2 lần) | **giống nhau từng token** |
| off vs fma | **khác** |

Cả hai arm đều tất định, nên khác biệt dưới đây là của số học chứ không của lịch thread.

### 2.2 Chín prompt cố định, 24 token greedy

Prompt: 8 câu hỏi/việc ngắn (giải thích kỹ thuật, tiếng Việt, sinh mã Rust, toán, so sánh khái niệm,
page fault, phương pháp benchmark, FMA) cộng một đoạn 986 token cắt từ `docs/benchmarks.md`. Mọi
prompt đều ≥ 16 token nên đều đi qua đường tiled.

| | |
|---|---|
| prompt cho ra **giống hệt** | **3 / 9** |
| độ đồng ý theo vị trí, trung bình | **0.625** |
| vị trí lệch đầu tiên | 1, 4, 8, 11, 14, 23 (6 prompt); 3 prompt khớp tới hết |
| prompt tiếng Việt (27 token) | lệch ngay **token thứ 1** |
| prompt dài 986 token | lệch ở token thứ 8 |

Ví dụ cùng prompt tiếng Việt: arm default mở đầu "Prefill và decode là hai thuật ngữ…", arm fused mở
đầu "Từ khóa: prefill, decode, phần cứng…". Cả hai đều là output hợp lý của một model 1.5B; phép đo
này **không** nói cái nào tốt hơn, chỉ nói rằng chúng khác nhau — và khác ngay từ token đầu.

### 2.3 Khoảng cách tới row kernel

Row kernel (`q4k_dot_batch_vnni`) là kernel mà decode dùng cho mọi ma trận Q4_K, nên "gần nó hơn"
là tiêu chí trung tính còn đo được trong repo. Test `q4k_tiled_error_against_row_kernel` (48 row,
512 cột, B=16, seed cố định, 768 giá trị):

| arm | max abs | max rel | mean rel |
|---|---:|---:|---:|
| multiply-then-add | 9.155e-5 | 4.784e-5 | 3.542e-7 |
| fused | 9.155e-5 | 4.784e-5 | **3.250e-7** (−8%) |

Worst case y hệt (cùng một giá trị, đến từ việc lượng tử hoá chứ không từ epilogue), sai số trung
bình **giảm** — đúng như lý thuyết: một lần làm tròn thay vì hai thì không tệ đi.

## 3. Cái ADR này KHÔNG nói

- **Không** nói fused cho output tốt hơn. Nó chỉ nói fused không tệ hơn về mặt số học so với row
  kernel, và **có** đổi output. Muốn kết luận về chất lượng phải có bộ đo theo tác vụ (mục 8).
- **Không** bao gồm phần còn lại của patch đề xuất: `transpose_16x16` khi dựng activation tile và
  `q4k_tile_rows::<R>` tách riêng cho band/tail. Đường nóng của chunk mặc định đi qua `SharedTile`
  (`SharedTile::fill_entries`), không qua nhánh transpose, nên transpose chỉ có nghĩa nếu port cả
  `fill_entries` — đó là thay đổi riêng, đo riêng.
- **Không** nói gì về Q6_K tiled (`q6k_matvec_tiled_range`). Nó đối xứng nhưng có epilogue khác và
  chiếm 18% chunk; đo riêng.

## 4. Options

| # | Cách | Được | Mất | Policy |
|---|---|---|---|---|
| A | **Giữ multiply-then-add làm default, fused sau switch** (hiện tại) | 0 thay đổi hành vi; số đo và đường tắt có sẵn để lật sau | −5% chunk không được dùng | none |
| B | Lật fused làm default, giữ `VIPERA_Q4K_EXACT_EPI=1` cho dạng cũ | −5% chunk; theo tiền lệ `VIPERA_EXACT_DOT` | Đổi output của mọi prefill ≥16 token; người dùng so kết quả với bản trước sẽ thấy khác | ADR này + cập nhật conformance |
| C | Bỏ hẳn, xoá switch | Ít code nhất | Mất số đo +8.6/+12% vĩnh viễn; phải đo lại lần sau | none |

## 5. Decision

**A.** Commit `bfb0fee` giữ dạng cũ làm default và đặt dạng fused sau `VIPERA_Q4K_FMA_EPI=1` (mặc
định tắt), kèm số đo và lý do trong code và trong `docs/benchmarks.md`.

Lý do chọn A thay vì B: hiệu quả của fused là thật và không tệ về số học, nhưng nó đổi output của
**6/9** prompt ở token đầu tiên tới token thứ 23, và repo không có phép đo nào trả lời được "đổi như
vậy là tốt lên hay xấu đi". Năm phần trăm thời gian prefill trên một prompt 524 token (≈39 ms trên
một request) không đủ để tự nó gánh một thay đổi hành vi chưa đo được. Tiền lệ `VIPERA_EXACT_DOT`
không chuyển thẳng sang được: ở đó dạng nhanh là **cùng** biểu thức với thứ tự cộng khác, đổi lại
được một chuỗi phụ thuộc 128-link; ở đây là một lần làm tròn khác trên số hạng mà mọi layer sau đều
tiêu thụ.

## 6. Hệ quả

- Người dùng có thể thử ngay: `VIPERA_Q4K_FMA_EPI=1`, không cần build lại.
- Switch đọc env mỗi lần gọi, nên một process đo được cả hai arm — nhưng nó cũng có nghĩa **không**
  được bật/tắt giữa hai lần gọi trong cùng một lần chạy nếu cần bit ổn định; chấp nhận được vì cả hai
  arm đều tất định trong một cấu hình.
- `q4k_tiled_range_matches_batch_vnni` (tolerance `1e-3 + 1e-4·|ref|`) xanh trên **cả hai** arm.
  Đây là lý do thay đổi này là quyết định policy chứ không phải bug.
- Không có test golden-token nào trong workspace, nên đường default không bị ràng buộc bởi bộ test —
  ràng buộc là quyết định của con người.

## 7. Test oracle

- `q4k_tiled_range_matches_batch_vnni`: parity tolerance với row kernel, phải xanh ở cả hai arm.
- `q4k_tiled_error_against_row_kernel` (`#[ignore]`, measurement): in max/mean relative error; dùng
  lại mỗi khi đụng epilogue, và là số liệu ở mục 2.3.
- A/B token: 9 prompt cố định ở mục 2.2; nếu lật default thì bộ này là thứ phải chạy lại và ghi lại.
- `cargo test -p vipera-backend-cpu --lib` (87 passed / 45 ignored) và `-p vipera-runtime --lib`.

## 8. Promote (điều kiện để lật sang B)

| # | Tiêu chí | Ngưỡng |
|---|---|---|
| 1 | chunk 4 | nhanh hơn **≥ 3%** median trên ≥5 cặp xen kẽ, box rảnh | đạt: −5.0%, 5/5 |
| 2 | kernel GFLOPS | không chậm hơn ở cả hai shape | đạt: +8.6% / +12% |
| 3 | sai số so với row kernel | mean rel không tăng | đạt: 3.250e-7 vs 3.542e-7 |
| 4 | tất định | mỗi arm tự khớp qua nhiều lần chạy | đạt |
| 5 | **chất lượng theo tác vụ** | **chưa có** — cần một bộ prompt có đáp án hoặc perplexity trên văn bản cố định, và tiêu chí "không tệ hơn" viết trước khi đo | **chặn** |
| 6 | chi phí hành vi | ghi rõ trong changelog rằng output prefill đổi | chưa làm |

Rollback: xoá một nhánh `if` hoặc để switch về default off; không đụng layout, không đụng kernel
row, không đụng `SharedTile`.

## 9. Rủi ro

- **Output drift không đo được**: 5% prefill đổi 6/9 prompt. Nếu model hoặc prompt khác phản ứng mạnh
  hơn, triệu chứng sẽ là "bản mới viết khác" chứ không phải một test đỏ.
- **Đo trên một máy**: 153→165 GFLOPS là của Tiger Lake với MUL+FMA trên cùng cổng. Trên core có hai
  cổng FMA (Zen 4/5, Xeon server) phần trăm có thể khác; không suy diễn.
- **Nhiệt/load**: các lần đo chunk chênh tới ±5% giữa các buổi (cùng binary 800 vs 1025 ms ở lần đo
  trước). Vì vậy mục 1 chỉ dùng tỉ lệ cùng cặp; con số tuyệt đối không phải cam kết.
- **Switch env trong đường nóng**: một `var_os` mỗi lần gọi kernel; với prefill 35 block × 144 call
  thì không đáng kể, nhưng nếu sau này kernel được gọi cho shape nhỏ thì nên cache.

## 10. Non-goals

- Không đổi tổng thứ tự cộng giữa các sub-block (`facc` vẫn cộng dồn theo `s` tăng dần).
- Không đụng `q6k_matvec_tiled_range`, `SharedTile`, `quant.rs`.
- Không thêm dependency, không thêm `unsafe` ngoài `simd.rs`.
- Không lật default trong ADR này.

## 11. Open questions

1. Bộ đo chất lượng theo tác vụ nào là đủ rẻ để chạy trong CI (perplexity trên một file văn bản cố
   định cần một lệnh `score` chưa có trong CLI)?
2. Có nên gộp luôn phần Q6_K nếu nó cho cùng tỉ lệ, để chỉ có một switch cho "epilogue fused"?
3. Owner và ngày hết hạn của cam kết "default off" — nếu 6 tháng nữa không ai đo, giữ nguyên hay
   xoá switch để khỏi mang code chết?
