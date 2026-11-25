# ADR-006: `VIPERA_Q4K8` (Q4_K 8x8 repack) — mặc định **tắt**

Status: **Accepted — default off.** *(Bị thay thế một phần bởi ADR-005 ngày 2026-11-25: `VIPERA_Q4K8` giờ mặc định bật; `=0` tắt.)* Kernel đã port và nối vào engine, nhưng đường đó chỉ chạy khi
`VIPERA_Q4K8=1`. Bật default cần hai thứ chưa có: một phép đo paired và một quyết định số học.
Ngày: 2026-11-24 · Owner: *cần điền* · Supersedes: none · Liên quan: ADR-003, ADR-004,
`docs/design/repack-port-plan.md`, `docs/benchmarks.md` mục "Pha 1b".

Đánh số: **ADR-005** đã được giữ cho quyết định *granularity quantization của activation* (W-1 trong
`repack-port-plan.md`), vì đó là policy người dùng nhìn thấy và là điều kiện để epilogue một phép/block
tồn tại. ADR-006 này chỉ quyết định *switch của repack*, và cố ý không gộp hai câu hỏi.

## 1. Context — từ "đã đóng" tới một switch đang bật

PR #66 (`bf1f02d`) đóng hướng repack bằng đo: gemv 8 hàng nhanh **1.34x ở 1 thread** nhưng chỉ
**1.06x ở 8 thread**, vì decode 8 thread đã bão hoà bus, và `GGML_CPU_REPACK=ON` của llama.cpp đo được
0.74x của OFF. Kết luận khi đó: không port, giữ `q4k8_lab.rs` làm bằng chứng.

Sau đó nhánh `perf/q4k-repack-8x8` làm hai việc mà PR #66 không làm:

1. Port kernel từ lab vào engine: `BlockQ4Kx8`, `make_blocks_q4k8`, `quantize_q8k`, `gemv_q4k8`
   (`crates/vipera-backend-cpu/src/simd.rs`), nối qua `fill_rows_q4k8` / `fill_rows_q4k8_groups`
   (`quant.rs`).
2. Bật switch **default on** và ghi vào `docs/benchmarks.md` một bảng decode **+53% tok/s**.

Việc 2 là lý do ADR này tồn tại: nó bật một đường số học mới cho mặc định của mọi người dùng, dựa trên
một phép đo không so được với nhau (mục 3).

## 2. Cơ chế — và vì sao đây là thay đổi số học, không phải thay đổi kernel

`block_q4_Kx8` (1152 B) là tám `block_q4_K` xen kẽ theo 8 byte, nên một `vpbroadcastd` của activation
phục vụ `vpdpbusd` của **tám hàng** thay vì một. Byte không đổi, thứ tự đọc đổi.

Cái đổi theo layout là **định dạng activation**: kernel `ggml_gemv_q4_K_8x8_q8_K` đọc `block_q8_K` —
**một** scale cho mỗi 256 giá trị activation, cộng `bsums` mỗi 16. Engine hôm nay dùng `QuantizedX`:
**một scale cho mỗi 32** giá trị. Nên bật repack không chỉ đổi thứ tự cộng: nó làm activation thô hơn
8 lần về granularity, đúng cái granularity mà `repack-port-plan.md` §3.1 và W-1 đã đo là có giá
**+0.3..0.7% NLL** trên mọi văn bản. Đó là quyết định của owner (ADR-005), không phải của một PR perf
(`CONSTRAINTS.md`: "đổi một bit là thay đổi số học").

Hệ quả thứ hai, đã sửa trong cùng thay đổi này: nhóm 8 hàng phải hình thành trên **chỉ số hàng tuyệt
đối**, không trên offset của lời gọi. Trước khi sửa, `fill_rows` chia hàng cho worker theo
`rows/threads` còn pool chia theo cách khác, nên cùng một hàng có hai kết quả khác nhau; test
`a_pooled_matvec_matches_the_spawned_one` đỏ, và đường batch (B < `TILE_MIN_BATCH`) còn lệch với
"mỗi sequence chạy một mình" (`a_batched_step_matches_every_sequence_run_alone`). Giờ kernel được gọi
cho **mọi** width mà đường tiled không nhận, pack một lần dùng cho cả `B` entry, và chỉ các hàng cuối
của ma trận (khi `rows % 8 != 0`) còn đi row kernel — nên kết quả là hàm của (hàng, activation).

## 3. Bằng chứng đang có — và chỗ nó không đứng được

**Lab, một process, xen kẽ hai arm** (`bench_q4k8_gemv_vs_row`, 2026-11-23, i5-11300H, 4c/8t, 45 MB):

| threads | row kernel | 8-row gemv | ratio |
|---:|---:|---:|---:|
| 1 | 4.02 ms / 11.2 GB/s | 3.01 ms / 15.0 GB/s | 1.338x |
| 2 | 2.10 / 21.5 | 1.65 / 27.4 | 1.273x |
| 4 | 1.20 / 37.7 | 1.12 / 40.3 | 1.070x |
| 8 | 1.54 / 29.3 | 1.46 / 31.0 | **1.056x** |

Đó là con số duy nhất đứng được, và nó đúng bằng kết luận của #66: đến 8 thread thì hai kernel đọc cùng
45 MB qua cùng bus và khác nhau trong nhiễu. 1.056x **không phải** ≥3% chunk mà vòng gate của repo đòi.

**Engine, bảng "Pha 1b" trong `docs/benchmarks.md`:** step 53.47 → 34.85 ms, decode 18.7 → 28.7 tok/s.
Bảng đó **không phải phép đo paired**: hai phiên khác nhau, và nó ghi `q6k matvec` −35.8% trong khi
thay đổi này chỉ chạm Q4_K — một pha không đổi kernel mà "nhanh" 35.8% nghĩa là hai phiên khác điều kiện,
nên phần Q4_K "−34.8%" cũng chưa quy được cho repack. Phép đo paired (một process, hai arm xen kẽ,
cùng bytes) **chưa chạy** — cây làm việc lúc viết ADR này có W-1 đang sửa dở nên không build được.
Nó là điều kiện để bật default, không phải việc của ADR này.

## 4. Decision

`VIPERA_Q4K8` **mặc định tắt**, theo đúng convention `env_truthy` của repo (unset / rỗng / `0` là off;
bất kỳ giá trị khác là on). Không xoá kernel: nó là bằng chứng, là bệ để đo lại, và là đường duy nhất
đã port xong cho W-3/W-5 nếu owner chấp nhận granularity per-256.

Lý do, xếp theo sức nặng:

1. **Nó đổi output trên mọi prompt.** Activation per-256 là cùng loại quyết định với ADR-004 (bị
   reject) và ADR-005 (đang chờ). Bật default trước quyết định đó là biến một lựa chọn chất lượng
   thành mặc định không ai review.
2. **Số duy nhất đứng được là 1.056x ở 8 thread**, dưới ngưỡng 3% chunk của chính repo.
3. **Con số engine ủng hộ nó không so được** (mục 3), nên không thể dùng làm lý do bật.
4. **Tắt không mất gì**: switch đã có, đường cũ vẫn là đường mặc định, không layout trên đĩa, không
   loader mới, không dependency.

## 5. Consequences

- Default = đường row kernel như trước #66. `VIPERA_Q4K8=1` bật repack, và cả hai nhánh phải qua cùng
  bộ test: `q4k8_wire_matches_lab_kernel_within_tolerance`, `q4k_rows_do_not_depend_on_the_call_split`,
  `a_pooled_matvec_matches_the_spawned_one`, `a_batched_step_matches_every_sequence_run_alone`.
- Điều kiện để mở lại (một trong hai, không cần cả hai):
  - một phép đo **paired** (một process, hai arm xen kẽ, cùng bytes, `taskset`, median ≥5 cặp) cho
    ≥3% chunk hoặc ≥3% step ở 8 thread; hoặc
  - ADR-005 quyết định per-256 cho đường activation, khi đó repack là cách lấy granularity đó mà
    không phải viết thêm kernel — và khi đó cái giá NLL không còn tính riêng cho repack nữa.
- Nếu owner nói không với cả hai, mục này đóng cùng #66 và `q4k8_lab.rs` ở lại làm bằng chứng; một
  dòng vào `CONSTRAINTS.md` §Rejected là đủ.

## 6. Cái ADR này không nói

- Không nói kernel nào nhanh hơn trên máy khác: mọi số ở đây là i5-11300H, 4c/8t Tiger Lake, AVX-512VNNI.
  Trên máy 1-2 thread (nơi lab đo 1.27-1.34x) kết luận có thể khác, và đó là lý do switch còn đó.
- Không nói gì về chất lượng của chính `block_q8_K`: nó dùng lại đúng phép đo NLL của W-1, không đo lại.
- Không đề xuất đổi `TILE_MIN_BATCH`, đổi layout trên đĩa, hay port GEMM 8x8 của llama.cpp (đã đo 0.74x).
