# ADR-004: `VIPERA_ATTN_QBLOCK` có thành default không

Status: **Rejected — giữ opt-in.** Lợi ích chỉ 3.2% ở chunk ngắn, 0.87% ở độ sâu, đổi lại là một
hồi quy likelihood có hệ thống trên cả 5 văn bản và output khác đi. Không có dòng code nào đổi trong
commit này; switch giữ nguyên.
Ngày: 2026-11-22 · Owner: *cần điền* · Supersedes: none

## 1. Context — vì sao câu hỏi này mở lại

`VIPERA_ATTN_QBLOCK` (`crates/vipera-backend-cpu/src/model.rs`, `attention_qblock_enabled`,
`QBLOCK = 4`, `attend_rows_shared`) đã được implement và đo trước đây, kết luận khi đó là **"does not
pay for itself end to end yet"**. Kết luận đó đúng với tỉ lệ của phần việc **lúc đo**: attention khi
ấy là 3% của một prefill chunk.

Hai thứ đã đổi:

1. Kernel tiled làm q4k (69% chunk) và q6k (18%) nhanh lên, attention thì không — nên slice của nó
   thành **10%** ở chunk 4 của prompt 524 token và **29-30%** ở 2.4k vị trí (`docs/benchmarks.md`).
2. `vipera score` ra đời cùng ngày, nên cái giá về chất lượng của một thay đổi làm tròn giờ đo được
   thay vì phải suy đoán.

Theo đúng bài học L8 (`docs/lessons/2026-11-22-lessons.md`), một lever bị loại phải được đo lại khi
share của slice nó cắt lớn lên. Đây là lần đo lại đó.

## 2. Cơ chế

Kernel mặc định gọi attention một lần mỗi layer với **một hàng query cho mỗi vị trí** trong chunk, và
mỗi hàng đi hết history. Chunk ~105 vị trí ở ~370 vị trí lịch sử đọc

```
105 hàng × 28 layer × 2 KV head × 370 vị trí × 128 giá trị × 2 byte × 2 (K và V) ≈ 1.1 GB
```

KV — nhiều hơn 0.9 GB weight mà cả chunk đọc. `attend_rows_shared` giữ `QBLOCK = 4` hàng query trong
register và cho chúng **một lượt đi history**, nên traffic hàng giảm 4 lần; phần số học cho mỗi hàng
(không chỉ phần đọc) không đổi.

## 3. Số đo — ngắn và sâu không giống nhau

Ba/năm cặp xen kẽ, cùng binary, cùng box:

| ngữ cảnh | arm | attention | chunk | Δ chunk |
|---|---|---:|---:|---:|
| chunk 4 của prompt 524 token (~370 vị trí), 5 cặp | default | 66.25 ms median | 649.71 ms median | |
| | `QBLOCK=1` | 47.77 ms (**−27.9%**) | 629.16 ms | **−3.2%**, 5/5 |
| chunk 24 của prompt 2953 token (~2.4k vị trí), 3 cặp | default | 254.07 ms median | 854.45 ms median | |
| | `QBLOCK=1` | 236.43 ms (**−6.9%**) | 847.02 ms | **−0.87%**, 3/3 |

Điều đáng ghi: lợi ích **nhỏ đi** khi ngữ cảnh sâu lên, dù attention chiếm 30% chunk ở đó. Bốn lần ít
traffic hàng chỉ mua 6.9% ở độ sâu, so với 27.9% ở ngữ cảnh ngắn. Nghĩa là ở độ sâu attention không
còn bị chi phối bởi việc đọc K/V mà bởi phần số học trên mỗi hàng — scores, `exp`, và weighted sum
theo từng vị trí — mà chia sẻ hàng không cắt được. Đây là suy luận từ hai điểm đo, không phải một
phép phân rã đã làm.

## 4. Cái giá — hồi quy likelihood có hệ thống

`vipera score` (chunk 64 token qua đúng đường prefill), five texts, mean NLL mỗi token:

| văn bản | token | default | `QBLOCK=1` | Δ | % |
|---|---:|---:|---:|---:|---:|
| popcount / rays | 986 | 2.826631 | 2.829815 | +0.003184 | +0.113% |
| gate and up shapes | 1697 | 3.084673 | 3.084879 | +0.000206 | +0.007% |
| the ggml bind | 2444 | 2.637364 | 2.638594 | +0.001230 | +0.047% |
| ADR-001 opening | 1492 | 3.324126 | 3.328024 | +0.003898 | +0.117% |
| ADR-003 §2 | 1494 | 3.581916 | 3.586679 | +0.004763 | +0.133% |

Trung bình **+0.002656 nats/token (+0.086%)**, và **5/5 cùng chiều**: văn bản kém khả năng hơn. Khác
với FMA epilogue (ADR-003), ở đó 2 văn bản tốt lên và 3 xấu đi quanh 0 — ở đây không có văn bản nào
tốt lên. Một hồi quy nhỏ nhưng nhất quán như vậy không phải nhiễu, và nó đi ngược chiều với mục tiêu
của engine.

Ngoài ra 16 token greedy từ prompt 524 token **khác nhau** giữa hai arm. Đây là thay đổi số học, không
phải tối ưu thuần tuý: `attend_rows_shared` gộp lại các tổng softmax theo nhóm khác.

## 5. Options

| # | Cách | Được | Mất | Policy |
|---|---|---|---|---|
| A | **Giữ opt-in** (hiện tại) | 0 đổi hành vi; lever + số đo còn nguyên cho người cần prefill nhanh ở ngữ cảnh ngắn | −3.2% chunk không được dùng | none |
| B | Lật default, `VIPERA_ATTN_QBLOCK=0` để tắt | +3.2% chunk ngắn / +0.87% sâu | NLL +0.086% trên 5/5 văn bản; output đổi; lợi ích nhỏ nhất đúng chỗ prefill đắt nhất | ADR này + changelog |
| C | Xoá switch và code `attend_rows_shared` | ít code nhất | mất một lever đã đo; ngữ cảnh ngắn vẫn là chỗ prefill rẻ nhất nên cũng không đáng xoá | none |

## 6. Decision

**A — giữ opt-in.** Lý do theo thứ tự:

1. **Hồi quy nhất quán**: 5/5 văn bản xấu đi, +0.086% NLL trung bình, không văn bản nào tốt lên. Một
   thay đổi có hướng rõ ràng như vậy không được biện minh bằng 3.2% thời gian chunk.
2. **Lợi ích nhỏ nhất ở chỗ đắt nhất**: prefill ở 2.4k vị trí tốn 854 ms/chunk và chỉ được 0.87%;
   chỗ được 3.2% là chunk ngắn, nơi 650 ms/chunk là mức thấp nhất trong ngày.
3. **Không phải lever duy nhất**: phần số học mà hai điểm đo cho thấy là giới hạn thật (scores, `exp`,
   weighted sum cho từng vị trí) chưa được cắt. Lật default bây giờ là tiêu một hồi quy vào một
   kernel còn headroom, đúng thứ tự ngược.

Điều kiện để mở lại: một trong hai điều sau, đo được.

- **Chia sẻ nhiều query hơn `QBLOCK = 4`** (8 hoặc 16 hàng trong register) — nếu traffic hàng vẫn còn
  là chi phí, mở rộng phải cho lợi ích tăng, không phải giảm. Đo bằng cách tăng hằng số, không cần
  đổi cấu trúc.
- **Cắt phần số học mỗi hàng** (ví dụ `exp` theo nhóm, hoặc tích luỹ weighted-V theo block vị trí).
  Nếu làm được điều này thì hồi quy likelihood của QBLOCK phải được đo lại, vì cái giá 0.086% đến từ
  việc gộp tổng chứ không từ đường đi.

## 7. Hệ quả

- Người cần prefill nhanh ở ngữ cảnh ngắn có thể bật ngay: `VIPERA_ATTN_QBLOCK=1`.
- Đường default không đổi một bit nào, nên không có test nào phải sửa; `cargo test -p
  vipera-backend-cpu --lib` (87 passed / 47 ignored) và `-p vipera-runtime --lib` (69) vẫn xanh.
- Switch đọc một lần (`OnceLock`), nên bật/tắt cần process mới — chấp nhận được vì attention không
  bit-identical giữa hai arm.
- `quant::exact_dot_requested()` đã bỏ qua đường này, nên `VIPERA_EXACT_DOT=1` vẫn là đường tham
  chiếu bit-exact; rollback là bỏ biến môi trường.

## 8. Promote (nếu mở lại)

| # | Tiêu chí | Ngưỡng |
|---|---|---|
| 1 | chunk 4, 5 cặp xen kẽ | ≥ 3% | đạt: −3.2% |
| 2 | chunk 24 (2.4k vị trí), 3 cặp | ≥ 3% | **trượt**: −0.87% |
| 3 | mean NLL trên 5 văn bản | **không tăng** | **trượt**: +0.086%, 5/5 xấu |
| 4 | token greedy | ghi rõ nếu đổi | đổi |
| 5 | decode | không chậm hơn | không áp dụng: `entries == 1`, đường này không chạy |

Hai tiêu chí trượt là đủ để từ chối; không có chuyện lấy tiêu chí 1 bù cho 2 và 3.

## 9. Rủi ro

- **Kết luận lại hết hạn**: đây là lần thứ hai cùng lever này được đánh giá, và kết luận đổi chiều theo
  share của slice. Nếu một kernel khác nhanh lên lần nữa, số ở mục 3 phải đo lại trước khi trích dẫn.
- **Hai điểm đo**: ngắn (370 vị trí) và sâu (2.4k). Giữa hai điểm đó chưa đo; giả thuyết "lợi ích nhỏ
  dần theo độ sâu" là suy luận từ hai điểm, không phải đường cong.
- **NLL trên văn bản cùng repo**: domain hẹp, và perplexity không nhìn thấy khác biệt hành vi. Nó đủ
  để bác một hồi quy nhất quán, không đủ để kết luận hai arm tương đương.

## 10. Non-goals

- Không đụng `attend_entries`, `attend_rows_shared` hay `QBLOCK` trong ADR này.
- Không bàn tới `VIPERA_ATTN_KV_SPLIT` (đang default on) hay window kernel.
- Không thêm dependency, không thêm `unsafe`.

## 11. Open questions

1. `QBLOCK = 8/16` có làm lợi ích ở ngữ cảnh ngắn tăng không, và có làm hồi quy NLL tệ hơn không?
2. Phần số học mỗi hàng ở 2.4k vị trí phân bổ thế nào giữa scores, `exp` và weighted sum? Chưa có phép
   đo nào tách ba phần đó — cần trước khi hứa bất kỳ con số nào.
3. Có nên gộp quyết định này với ADR-003 thành một policy chung cho "thay đổi làm tròn trong prefill:
   cần số NLL, không chỉ số ms" không?
