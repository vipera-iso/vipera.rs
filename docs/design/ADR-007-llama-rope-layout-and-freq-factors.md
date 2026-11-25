# ADR-007: cặp RoPE theo kiến trúc và `rope_freqs.weight`

Status: **Accepted, chưa kiểm chứng trên file Llama thật.** Ngày: 2026-11-25.

## 1. Vấn đề

Backend CPU nhận kiến trúc `llama` nhưng chỉ cài RoPE kiểu NeoX (cặp `(i, i + head_dim/2)`).
llama.cpp dùng `LLAMA_ROPE_TYPE_NORM` (cặp kề nhau `(2i, 2i+1)`) cho `llama`, và converter hoán vị
Q/K của kiến trúc đó vào đúng layout ấy. Xoay một file như vậy theo NeoX ghép sai giá trị.

Thứ hai, GGUF của Llama 3.x không ghi `rope.scaling.type`; scaling nằm ở tensor `rope_freqs.weight`
(một hệ số mỗi cặp, ggml chia góc cho nó). Không chỗ nào trong repo đọc tensor này.

Không có bằng chứng trong repo rằng một model `llama` đã từng chạy qua backend này; chỉ có Qwen2.5.

## 2. Thay đổi

- `RopeLayout { NeoX, Norm }`, chọn theo `general.architecture`: `llama` là `Norm`, còn lại `NeoX`.
  Qwen2/Qwen3 không đổi bit nào (test `rope_without_scaling_is_bit_identical_to_the_original_table` vẫn qua).
- `rope_freqs.weight` được đọc nếu có (F32, `head_dim/2` phần tử, hữu hạn và dương) và chia góc của
  từng cặp trước khi luật scaling nhìn thấy nó, như `rope_yarn(theta / ff, ...)` của ggml.
  Head rộng hơn 128 giá trị bị từ chối bằng tên thay vì xoay sai.
- Test: hệ số so với tham chiếu f64; hệ số toàn 1.0 bit-identical với không có tensor; `Norm` bằng
  `NeoX` trên dữ liệu đã hoán vị (so bit); loader chọn layout theo kiến trúc.

## 3. Chưa làm / chưa biết

- Chưa chạy một file Llama 3.x thật: không có model trong sandbox. Cần so logits với llama.cpp trước khi tin.
- Backend GPU vẫn từ chối mọi `rope.scaling.type` và không đọc `rope_freqs.weight`.
- `longrope` (Phi-3, `rope_factors_long/short`) vẫn bị từ chối.
