# Port repack/kernel: cái gì trong 12 fork port được sang vipera.rs

Ngày: 2026-11-23 · Nhánh: `perf/q4k-repack-8x8` · Trạng thái: hàng đợi đã đo và xếp hạng, chưa mở workstream nào.

Tài liệu này trả lời một câu: trong các dự án CPU-inference nghiêm túc có trong `~/llm-forks`, phần *máy móc
kernel* nào port được sang `crates/vipera-backend-cpu`, đáng bao nhiêu, và bác bỏ bằng gì. Mọi con số nền
đều đã đo trên máy này (i5-11300H, 4c/8t Tiger Lake, AVX-512VNNI); con số nào là suy ra thì ghi rõ.

## 1. Kết luận trước

1. **Đã đóng bằng đo:** layout repack của llama.cpp (`block_q4_Kx8` + gemv 8 hàng + GEMM 8x8). Prefill
   `GGML_CPU_REPACK=ON` = **0.74x** của OFF; gemv 8 hàng 1.34x ở 1 thread và **1.06x ở 8 thread** (decode
   bus-bound). Không port, giữ `q4k8_lab.rs` làm bằng chứng.
   *Cập nhật 2026-11-24:* kernel **đã** được port vào `simd.rs` và nối vào engine, nhưng mặc định
   **tắt** (`VIPERA_Q4K8`, quyết định ở `ADR-006`); kết luận đo ở trên không đổi.
2. **Còn mở, và là việc lớn duy nhất:** mở khoá epilogue **một phép float cho mỗi 256 block** như ik — nhưng
   đường vào **không** phải "port epilogue integer" (mục 3.1: với scale activation mỗi 32 giá trị như ta,
   4 phép vector cho mỗi `(row, sub-block)` là **sàn**, không cắt được). Đường vào là **đổi granularity
   quantization của activation**: `block_q8_K` của llama/ik có **một** scale cho cả 256 giá trị, ta có tám.
   Đây là thay đổi số học, không phải thay đổi kernel.
3. **Trần đó đã đo xong, và cái giá cũng đã đo xong** (2026-11-23): kernel **+1.24..1.36x** khi epilogue
   về một phép/block (probe `PER_BLOCK_EPI_PROBE`), suy ra **+14..20% chunk ở 8 thread**; đổi lại NLL xấu
   đi **+0.3..0.7% trên mọi văn bản** (per-128 và per-256 như nhau về bậc). Tức món lời lớn nhất còn lại
   *không* mua được bằng tiền perf — nó là quyết định accuracy của owner (`ADR-005`), xem W-1.
4. **8 thread không mất vào scaling.** Đo hôm nay: pha q4k ở 8T = **838.7 GFLOPS** = **209.7 GFLOPS/core**,
   so với band cô lập 1 core = 218.4 → hiệu suất 96%, và scaling 202 → 209.7/core = **4.15x trên 4 core vật
   lý**. Nên mọi cải thiện phải đến từ *tốc độ kernel trên một core*, không từ chia chunk hay thread pool.
5. **Cái không port được:** mọi thứ thuộc về i-quants (IQ2/IQ3/IQ4/trellis), NVFP4, tensor song song/MoE
   expert cache, và các bản viết lại bằng ngôn ngữ khác. Model đang chạy là Q4_K/Q6_K, GGUF.

## 2. Sổ các dự án

Quét `~/llm-forks` (17 thư mục). Cột "máy móc" chỉ tính kernel CPU cho K-quant, không tính GPU/MoE/quant type.

| dự án | máy móc CPU K-quant | đáng port? |
|---|---|---|
| **ik_llama.cpp** `1180df7` | `iqk/iqk_gemm_kquants.cpp` 5337 dòng: `mul_mat_q4_k_r4_q8_k` (:1497), `mul_mat_qX_K_q8_K_AVX512` (:381), `process_min_r4_b32`, `iqk_convert_q4_k_q8_1_r8` (:2335) | **Có** — xem mục 3, M2/M3 |
| **Strata / PolyStrata** (`engine/third_party/strata-ggml`) | `iqp.cpp` 1253 dòng: panel 8 hàng decode ra `int8` (`block_iqp_x8`), integer GEMM, `GGML_IQP_MIN_BATCH 8`; `arch/x86/repack.cpp` là bản sao nguyên của llama.cpp | **Một phần** — M4 (staging panel), chỉ khi M2 không đủ |
| **llama.cpp** `a8f8294c4` | `arch/x86/repack.cpp` 6407 dòng (`block_q4_Kx8`), `tiled/tiled-kernel.cpp` (`ggml_compute_forward_mul_mat_tiled`, **đã biên dịch nhưng không được gọi** trong bản này) | **Không** — đã đo 0.74x |
| **basalt** | `src/kernels/cpu/kq_avx2.cpp`: `q4k_dot` cho `nt` token quanh **một** luồng weight, tuyên bố *bit-exact* với `ggml_vec_dot_q4_K_q8_K`; `expert.cpp`/`pool.cpp` cho MoE | **Không** — cùng ý tưởng với band của ta (16 cột/lần load), không có cơ chế mới; nhưng là tiền lệ cho lập luận "batch theo token không buộc phải đổi số học" |
| **cpu-vega-llama.cpp** | mang `llamafile/sgemm.cpp` (tinyBLAS, f32) + `iqk_mul_mat` | **Không** — f32 cho K-quant nghĩa là dequant rồi GEMM; 2.62 GFLOP/token ở f32 không thắng nổi `dpbusd` |
| koboldcpp, ktransformers | bản sao llama.cpp đóng gói | Không |
| llama.cpp-glm52, llama-optimize, llama-cpp-optimization-lab, llama-optimus, extended-strata-moe, ik_llama-numa-mirror | 0 hit cho `block_q4_Kx8` / `dpbusd_epi32` / `tiled-kernel` | Không |
| exllamav3, mlx-lm, tabbyAPI | GPU / Apple / API | Không |

## 3. Bảy cơ chế, định giá theo trần đo được

Ký hiệu trần: `q4k` = 74% chunk ở 8T, `q6k` = 17%, `attention` = 6% (đo hôm nay, chunk 128 token, 8T).

| # | cơ chế | nguồn | số upstream | trần trong vipera.rs | trạng thái |
|---|---|---|---|---|---|
| **M1** | Repack weight lúc load: hàng xen kẽ theo 4/8 byte, một load phục vụ 4-8 hàng | llama `repack.cpp`, ik `iqk_quantize` | llama: pp512 0.74x khi bật; ik: pp512 268.7 @8T | probe `NO_WEIGHT` của ta: +29% (8960x1536) / +56% (1536x8960) kernel → nhưng cần **layout trên đĩa** + loader | **Đóng** cho decode (bus-bound, 1.06x); prefill còn khả năng, xem W-3 |
| **M2** | Scale 6-bit áp trong miền `i32`: `packs_epi32` + `dpwssd`, một `cvt`+`fma` cho cả 256 giá trị | ik :381, :1497 | Q6_K dot 2.84 → 1.97 chu kỳ/32 giá trị (#27590, khác nguồn) | cần **activation một scale/256** mới dùng được (mục 3.1); khi đó epilogue 21-23% → ~4% → **+11..15% chunk 8T** | **Mở — W-1, cùng với đổi granularity** |
| **M3** | Số hạng min dồn ra khỏi vòng lặp, dùng `bsums` của activation, một lần cho mỗi 256 block | ik `process_min_r4_b32` | nằm trong cùng kernel | gộp vào M2 (cùng 21-23%) | **Mở — W-1** |
| **M4** | Staging `int8`: decode 8 hàng ra panel rồi chạy integer GEMM trên mọi cột | Strata `iqp.cpp` | `MIN_BATCH 8`, `SB_SIZE 16` | q4k, nhưng cần batch ≥8 và 2 KB scratch/8 hàng; vi phạm bit-identity với đường GGUF | **Chờ** — chỉ mở nếu W-1 xong mà q4k < 240 GFLOPS/core |
| **M5** | Microkernel 16x16 của `tiled/` | llama `tiled-kernel.cpp` | không có số upstream (chưa nối vào `ops.cpp`) | shape khác band (8-10 hàng x 16 cột); không có cơ chế mới | **Đóng** — không có bằng chứng để port |
| **M6** | Tiled flash-attention | ik `iqk_flash_attn.cpp`, llama `ops.cpp` FA tiled | — | attention 6% ở 512, ~43% bước ở ctx 4096 (suy ra từ độ dốc pp1024→pp4096) | **Mở — W-4** (đã port một lần, xem `idea-sources.md`; đo lại trước khi port lần nữa) |
| **M7** | tinyBLAS f32 GEMM cho prefill | llamafile `sgemm.cpp` | — | cần dequant Q4_K→f32 (4 B/weight) rồi GEMM; byte gấp 7.5x | **Đóng** |

Ghi chú về M2/M3, phần dễ hiểu sai: kernel của ik *không* nhanh hơn vì "không broadcast" — band của ta
cũng đã không broadcast (`vpbroadcastd` memory-source, một uop, phục vụ 8 hàng). Cái ik làm khác là **đưa
scale và min ra khỏi miền float**: `packs_epi32` + `dpwssd` với scale `i16`, rồi *một* `fmadd` cho mỗi
`(row, 256 block)`.

### 3.1 Vì sao epilogue của ta là sàn 4 phép, và ik thì không

Cái làm được điều đó **không phải** instruction mà là **granularity của activation**:

| | scale activation | phép float cho mỗi `(row, 256 block)` |
|---|---|---:|
| llama.cpp / ik `block_q8_K` = `{float d; float sum; int8_t qs[256]; int16_t bsums[16]}` | **1** | ~2 |
| vipera.rs `QuantizedX` = `{q: Vec<i8>, d: Vec<f32>}` — một `d` mỗi 32 | **8** | 32 (fused: 4 x 8) |

Toán hạng cần tính cho mỗi `(row, s)` là
`facc += (d_row·scale_s·acc_s − dmin_row·min_s·sum_s) · y_d_s`, và `y_d_s` **đổi theo từng sub-block**.
Vì thế mỗi `s` buộc phải có một bước float riêng: `cvt` + `mul` (nhân `dxv`) + `fma` + `fnmadd` = 4 phép —
đúng dạng `fused_epi` hiện tại. Tích luỹ trong `i32` rồi nhân một lần cuối **chỉ đúng khi `y_d` không đổi
theo `s`**, tức khi activation có một scale cho cả 256. Đó là lý do ik viết được `cvt`+`fma` một lần cho
mỗi block, và cũng là lý do một bản "port epilogue integer" giữ nguyên `QuantizedX` sẽ **chậm hơn**:
`mullo`+`add` (2 lệnh) cho mỗi `s` **cộng thêm** bước float vẫn còn đó.

Kết luận: muốn lấy 21-23% epilogue thì phải đổi **quantization của activation** cho đường tiled, không phải
đổi kernel. Giá phải trả là accuracy (activation thô hơn 8 lần về granularity), và đó là quyết định số học
cùng loại với ADR-003, nên nó cần ADR riêng + số NLL, không phải một PR perf.

## 4. Năm workstream

Mỗi workstream theo đúng gate của skill `vipera-perf-loop`: code → review → build release → codegen gate →
bit/parity → bench → ledger. Ngưỡng land: **≥3% chunk, median trên ≥5 cặp xen kẽ**.

### W-0 — Chốt mục tiêu 8T trước khi viết kernel (nửa ngày) — **hoãn, không còn chặn**

Mục đích ban đầu: đo ik bằng *cùng* thang của ta để biết +29% của họ đến từ q4k hay từ pha khác
(`llama-bench -p 128,512,1024 -n 0 -t 8` trên build ik, rồi trừ độ dốc attention như đã làm với llama).

Việc này **không còn là cổng quyết định**, vì câu hỏi nó đặt ra đã có câu trả lời trực tiếp: phần đáng
port đã được cô lập và *đo tại chỗ* (probe `PER_BLOCK_EPI_PROBE`, +1.24..1.36x kernel), và cái giá của nó
cũng đã đo (NLL, mục W-1). Build `llama-bench` của ik rất chậm trên máy này (iqk_gemm_iquants.cpp một file
đã vài phút; dừng ở 16% sau ~8 phút với `-j8`), nên nó bị dừng lại và để làm việc xác nhận về sau.

- Lệnh đã dùng, nếu cần chạy lại: `cmake -S ~/llm-forks/ik_llama.cpp -B build-bench -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF -DGGML_NATIVE=ON && cmake --build build-bench --target llama-bench -j8`.
- Giá trị còn lại: nếu ik hơn ta <1.15x ở riêng q4k thì probe của ta đang đo sai chỗ và phải xem lại
  `fill_rows_batch` (tile sharing, chunk width) trước khi tin W-1.

### W-1 — Activation một scale/256 cho band, rồi epilogue một phép/block (2-3 ngày) — **ưu tiên 1**

**Trạng thái: phần đo đã xong (2026-11-23), phần code chưa viết.** Cả hai nửa của đánh đổi đã được đo
không cần viết kernel — xem `docs/benchmarks.md` mục "a per-256 activation scale is worth 1.24-1.36x on the
band, and costs 0.3-0.7% NLL":

| | đo được |
|---|---|
| kernel band, epilogue một phép/block (probe) | **+1.24..1.36x** (152.8 → 189.5 GFLOPS trên 8960x1536, 168.8 → 221.9 trên 1536x8960) |
| chunk 8T suy ra (q4k = 74%) | **+14..20%** |
| NLL, activation per-256 | +0.45..0.69% mọi văn bản (+0.013..0.020 nats) |
| NLL, activation per-128 | +0.33..0.53% (không nhẹ hơn đáng kể) |

Nghĩa là mục này **trượt cổng chất lượng của chính nó** (`|Δ| ≤ 0.5%`): ba trong bốn văn bản vượt 0.5%,
không văn bản nào dưới 0.3%, và mốc 128 không cứu được. Vì thế W-1 **không còn là một PR perf**; nó là
quyết định policy số học của owner. Hai bước code dưới đây chỉ được làm sau khi owner chấp nhận cái giá
đó, và phải kèm ADR-005.

Hai bước, đo riêng từng bước:

**W-1a — đổi granularity (không đụng kernel).** Thêm một activation quantizer cho đường tiled: một scale và
một bộ `bsums` cho mỗi 256 giá trị, thay vì tám scale như `QuantizedX` hiện tại. `dims` của band đổi từ
`dxs[(half, s, col)]` sang `dx[(half, block)]`; mã `i8` vẫn 256 giá trị, min term dùng `bsums` mỗi 16 như
ik. Decode và row kernel **không đổi** (chúng vẫn dùng `QuantizedX` per-32), nên đường bị ảnh hưởng chỉ là
batch ≥ 16.

**W-1a — ĐÃ DONE (2026-11-24):** port — `TiledActivation` (quant.rs) + `q4k_matvec_tiled_range_a`/`try_q4k_matvec_tiled_range_a`
(simd.rs) + `fill_rows_batch` wiring + switch `VIPERA_TILED_A` (mặc định OFF — numerics change → ADR-005). Chunk 8T: **+2.6%**
(B7 baseline 33.83 → 34.72 ms, trong máy ±3%; band ≈ unchanged theo thiết kế). Parity:
`q4k_tiled_range_a_matches_batch_within_tolerance` — max diff 2.63 (tolerance `2.0 + 0.2·|want|`). Không chạy lab band bench
(cùng với missing `decode_ab.rs` — hỏi user: giữ declaration).

**W-1b — epilogue theo ik.** Trong `q4k_tiled_band` (`simd.rs:2660`): tích luỹ `i32` qua 8 sub-block bằng
`mullo` (hoặc `packs_epi32` + `dpwssd` như ik :381), min term cũng trong `i32`, rồi **một** `cvt`+`fma` cho
mỗi `(row, block)`. 32 phép float/`(row, block)` → 2.

- Vì sao không tràn: `|q| ≤ 15`, `|y| ≤ 127`, 32 giá trị → `|Σ| ≤ 60960`; nhân scale ≤ 63, cộng 8 sub-block
  → `|isum| < 2^25`.
- Gate (theo thứ tự, dừng ở bước nào trượt thì dừng): kernel `bench_prefill_band_gflops` ≥ +10%; chunk 8T
  ≥3% median trên 5 cặp; NLL 6 văn bản ADR-003 — **ngưỡng chặt hơn ADR-003**: đây là thô hoá activation
  chứ không chỉ đổi thứ tự làm tròn, nên đòi `|Δ| ≤ 0.5%` mỗi văn bản và không văn bản nào xấu đi quá
  0.01 nats; `q4k_tiled_range_matches_batch_vnni` xanh (tolerance hiện có `1e-3 + 1e-4·|ref|`).
  → **cổng NLL đã trượt trên giấy, trước khi viết dòng code nào** (bảng ở trên). Chỉ tiếp tục nếu owner
  nới ngưỡng này một cách có ý thức, không phải bằng cách sửa ngưỡng cho vừa kết quả.
- Kỳ vọng (đo rồi, không phải suy đoán): kernel +24..36%, chunk 8T +14..20%.
- Rollback: W-1a là một hàm quantize mới + một nhánh trong `BatchActivation`; W-1b là một arm `bool` trong
  kernel như `fused_epi`. Không đụng layout trên đĩa, không đụng loader, không đụng decode.
- Bác bỏ nếu: chunk <3%, **hoặc** NLL xấu quá ngưỡng trên bất kỳ văn bản nào — điều đã xảy ra ở mức
  0.3-0.7%. Trạng thái mặc định hiện tại là **giữ per-32**, và khi đó câu "lấy nốt 21-23% epilogue" coi như
  đã trả lời: không lấy được mà không đổi chất lượng.
- Việc phải làm kèm: `ADR-005` (ADR-004 đã là attention q-block default) ghi granularity của activation
  cho đường tiled, vì nó là policy người dùng nhìn thấy (prefill và decode sẽ dùng hai cách lượng tử hoá
  khác nhau) và vì nó đổi output trên mọi prompt dài hơn 16 token.

### W-2 — Row kernel (decode) — **gộp vào W-1/ADR-005, không còn là việc riêng**

Decode 1 thread: llama 14.54 tok/s, ta 10.8 (1.34x). Ở `nrc = 1` epilogue chia cho ít `dpbusd` hơn band,
nên tỉ lệ *cao hơn*, và trước đây tưởng đây là một workstream độc lập.

Đọc `q4k_dot_vnni` (`simd.rs:3311`, đường thật của B=1 — batch một không vào kernel tiled) thì thấy nó
**cùng một phụ thuộc**: `coefficients = scale_at · dx · d` là **vector 8 làn** vì scale weight là 6-bit
*theo từng sub-block*, không phải vì activation. Nên cả ở decode, phép nhân float theo sub-block chỉ gộp
được khi `dx` là một scale cho cả block — đúng cái ADR-005 phải quyết. Ngoài ra phần dư của W-2 bị kẹp
bởi băng thông: Q4_K 1.1116 G × 4.5 bit + Q6_K 0.4316 G × 6.5625 bit ≈ 0.98 GB/token, và 10.8 tok/s
⇒ 92 ms/token, tức kernel đang chạy **~10.6 GB/s** trong khi bus đo được 35.7 GB/s — chỗ mất không phải
epilogue mà là cách nạp weight, một câu hỏi khác (và có thể lớn hơn cả W-1 nếu đo được).

- Việc còn lại của W-2: nếu ADR-005 nói *có*, đổi `q4k_dot_vnni` sang int-scale + một `fma`/block trong
  cùng PR với W-1b, đo riêng encode ở 1T. Nếu *không*, W-2 đóng cùng W-1.

### W-3 — Q6_K cùng bệnh (0.5 ngày, chỉ sau W-1)

Epilogue Q6_K = 17.6% kernel = **2.8% chunk**, dưới ngưỡng. Chỉ làm nếu W-1 đã mở đường (cùng một switch),
và chỉ để khỏi hai policy epilogue khác nhau trong repo — nên nó cũng nằm sau ADR-005, không chạy trước.

### W-4 — Attention ở context dài (2-4 ngày)

Ở ctx 4096, một thread: vipera pp4096 43.5 tok/s, llama 60.48; attention chiếm ~43% thời gian bước (suy ra
từ độ dốc pp1024/pp4096, không phải từ profiler). ik có `iqk_flash_attn.cpp` + `fa/`; `idea-sources.md`
ghi PR #65 đã port một phần — **đo lại trước**, đừng port lần hai.

- Bác bỏ nếu: sau khi trừ attention khỏi phương trình độ dốc, phần còn lại của gap 4096 < 1.15x.

### W-5 — Staging panel int8 (M4) (3-5 ngày) — chỉ khi W-1 < 240 GFLOPS/core

`iqp.cpp` là bản *đơn giản hoá* đúng chỗ ta cần: decode 8 hàng ra `int8` (kèm bias), rồi integer GEMM trên
mọi cột. Nó tương đương M1+M2 nhưng trả giá bằng scratch và một đường số học thứ hai. Không mở trước W-1 vì
nếu W-1 đã lấy được phần epilogue thì phần còn lại của M4 chỉ là load-port, thứ band đã tối ưu (2 uop/64
MAC).

## 5. Thứ tự và lý do

Hiện tại: **owner quyết định W-1 (`ADR-005`)** → nếu đồng ý thì `W-1a → W-1b` → `(W-2 | W-3)` → `W-4` →
`W-5`; nếu không đồng ý thì `W-2` (đo probe trước) → `W-4`. `W-0` là việc xác nhận, không chặn.
**Không** làm M1/M5/M7. Lý do xếp hạng:

- Trần = `phase share x kernel share`. q4k ở 8T là 74% chunk và epilogue là 21-23% của nó → W-1 là mục duy
  nhất có trần >3% mà *không* cần layout trên đĩa, loader mới, hay dependency mới (chỉ đổi cách lượng tử
  hoá activation trong bộ nhớ, thứ được dựng lại mỗi lần gọi) — nhưng nó đã đổi từ "việc perf lớn nhất"
  thành "việc chờ quyết định", nên nếu owner nói không thì mục tiếp theo phải là W-4 (attention ở context
  dài), nơi 43% thời gian ở 4096 vẫn còn nguyên và chưa ai định giá lại kể từ ADR-004.
- M1 (repack) có trần danh nghĩa lớn hơn (probe `NO_WEIGHT` +29/56%) nhưng đã đo thất bại hai lần trên máy
  này (decode 1.06x ở 8T; llama's own GEMM 0.74x), và cái giá là một layout trên đĩa thứ hai.
- W-4 lớn dần theo context, không theo chunk: ở 512 nó là 6%, ở 4096 là ~43%. Người dùng chat dài mới là
  người thấy nó, nên nó xếp sau W-1 nhưng trước W-5.

## 6. Cái tài liệu này không nói

- Không nói kernel của ik *đo được* nhanh hơn ở cùng điều kiện: con số 268.7 là cả engine, khác harness,
  khác ubatch. W-0 tồn tại để đóng khoảng đó.
- Không nói gì về chất lượng output của W-1: nó **thô hoá** activation (một scale/256 thay vì /32), nên
  phải qua bộ đo của ADR-003 (6 văn bản NLL + 9 prompt) với ngưỡng chặt hơn, và cần ADR riêng trước khi
  thành default.
- Không đề xuất dependency mới, `unsafe` mới ngoài `simd.rs`, hay `#[allow]` nào (`CONSTRAINTS.md`).
