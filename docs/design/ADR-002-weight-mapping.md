# ADR-002: Page-aligned hoặc mmap cho buffer weight

Status: **Rejected tại chính gate E1** — căn trang không mua gì trên máy này (mục 4). Không có dòng
code sản phẩm nào; `fs::read` giữ nguyên. Động cơ còn lại của mmap (RSS, cold start, model > RAM) là
mục tiêu bộ nhớ, không phải tốc độ, và cần một quyết định riêng.
Ngày: 2026-11-13 · Owner: *cần điền* · Supersedes: none

## 1. Context — hiện trạng và số đo

Model được nạp bằng `std::fs::read` — một `Vec<u8>` do allocator cấp, **không căn trang, không
hugepage, không mmap** (`bins/vipera/src/main.rs:167, 224, 334, 630, 874, 1091`). Toàn bộ 1.111 GB
tensor nằm trong heap riêng của process, cộng KV cache.

Số đo trên máy này (`docs/benchmarks.md`, 2026-11-13, i5-11300H, 4 core / 8 thread):

| Đại lượng | Giá trị |
|---|---|
| Read set mỗi token decode | **0.980 GB** (1.111 GB tensor − 125 MB `token_embd`) |
| Trần đọc đo được (STREAM-class, 4 thread) | **42.8 GB/s**; Copy 39.7 |
| Vipera decode | median-of-5 **35.1 ms/step** = **27.9 GB/s** (65% trần); best 31.3 ms = 31.3 GB/s |
| Vipera, tensor Q4_K lớn (8960×1536) | **32.6 GB/s** |
| Vipera, output Q6_K (191 MB) | **35.6 GB/s** |
| llama.cpp cùng máy (4 thread) | 39.04 tok/s = **38.3 GB/s** (89% trần) |
| Trang 4 KiB cho 1.111 GB | **~271 000 trang** |

## 2. Cơ chế giả thuyết

1. **TLB.** 271k trang 4 KiB so với ~2 000 entry L2 TLB của Tiger Lake: mỗi lần quét weight là một
   chuỗi page-walk không tái sử dụng được. Hugepage 2 MiB chia số trang đi 512 lần.
2. **Căn chỉnh / đường cấp phát.** Một thí nghiệm bên ngoài (Zig/Rust/C, 1 GiB, 1 thread) thấy Zig
   nhanh hơn Rust ~20% khi mỗi bên dùng allocator mặc định, và **khoảng cách biến mất** khi ép cả ba
   dùng vùng nhớ căn trang. Kết luận ở đó: chênh lệch đến từ đường cấp phát, không phải ngôn ngữ.
   Số tuyệt đối của thí nghiệm đó (~10 GB/s, VM 1 vCPU) **không** chuyển sang máy này.
3. **THP trên máy này:** `enabled=[always]`, `defrag=madvise`, `hpage_pmd_size=2 MiB`. Nhưng
   **`CONFIG_READ_ONLY_THP_FOR_FS` không có** trong `/proc/config.gz`, tức là một **file mapping
   read-only không được hugepage** trên kernel này; THP chỉ áp dụng cho **anonymous memory**. Đây là
   chỗ dễ hứa sai nhất: mmap file (option C) **không** mang lại hugepage, chỉ mang lại RSS dùng chung
   và paging lười. Muốn hugepage thì phải là buffer anonymous căn 2 MiB + `MADV_HUGEPAGE` (option B).

Đây là **giả thuyết**, chưa phải nguyên nhân. ADR này không được phép biến nó thành kết luận trước E1.

## 3. Cái ADR này KHÔNG giải thích

Bốn tensor attention nhỏ (1.33 MB và 0.22 MB) chạy 5.2–18.3 GB/s và chiếm 16% thời gian — nhưng đó
**không** phải TLB, không phải căn chỉnh, và cũng không phải handshake: một dispatch đo được **8.4 µs**
(`docs/benchmarks/beating-llamacpp.md` §E) và toàn bộ phần không-phải-dot của một step là 0.9–1.5 ms
trên 40–45 ms. Việc gộp dispatch đã implement, đo (paired median **0.999x**) và **revert** vì không có lợi ích
(`docs/benchmarks.md`). Nếu E1 thắng, nó thắng ở tensor lớn (≥1 MB) nhờ băng thông, không nhờ bớt lời gọi.

## 4. Ba thí nghiệm phải chạy trước khi đổi code

E1 **không cần `unsafe` và không cần dependency** — nó là điều kiện rẻ nhất và phải chạy trước. E2 cần
`unsafe`/`libc`, nên chỉ chạy sau khi E1 cho thấy có tín hiệu (≥ 3%).

- **E1 — căn trang, so trong cùng process.** Cấp một `Vec<u8>` dài `len + 2 MiB`, lấy subslice bắt đầu
  tại mốc 2 MiB kế tiếp (toàn bộ thao tác là safe Rust), copy tensor Q4_K 200 MB vào cả hai buffer,
  chạy cùng `matvec_pooled` 20 lần, xen kẽ A/B, lấy median. Đo GB/s của từng buffer.
  *Nếu chênh < 3%: giả thuyết căn chỉnh bị bác bỏ, dừng ADR này.*
- **E2 — hugepage, trên bộ nhớ anonymous.** Lặp E1 với buffer anonymous căn 2 MiB + `MADV_HUGEPAGE`
  (cần `unsafe`/`libc`, nên đây là thí nghiệm có điều kiện), rồi đọc `/proc/self/smaps` để xác nhận
  `AnonHugePages` thực sự xuất hiện. **Không** kỳ vọng hugepage từ mmap file trên máy này (mục 2.3);
  nếu `AnonHugePages` vẫn 0, ghi "chưa đo được hugepage" và không suy diễn.
- **E3 — cold start.** Thời gian tới token đầu tiên: `fs::read` (1.111 GB copy vào heap) so với mmap
  (page-fault theo nhu cầu). mmap thường thắng cold start và thua tail ở lần chạm đầu; cả hai phải đo.

Chỉ khi E1 hoặc E2 cho **≥ 5%** trên tensor lớn mới sang mục 6.

### Kết quả E1 (2026-11-13)

`crates/vipera-backend-cpu/tests/alignment_probe.rs`, 200 MB Q4_K, hai buffer cùng seed, gọi xen kẽ
đảo thứ tự 5 vòng, best-of-5:

| threads | plain (start % 2 MiB = 16) | aligned (start % 2 MiB = 0) | aligned / plain |
|---|---|---|---|
| 8 | 42.67 GB/s | 42.22 GB/s | **0.989x** |
| 4 | 39.84 GB/s | 39.91 GB/s | **1.002x** |

Cả hai đều dưới ngưỡng 1.03 → **căn trang bị bác bỏ**. Hai điều đáng ghi thêm:

- Buffer *không* căn đã đạt **42.67 GB/s**, tức 96% trần read-only 42.8 GB/s của máy. Kernel Q4_K
  không thiếu băng thông khi dữ liệu là một dải phẳng; chênh lệch in-situ (33 GB/s ở shape lớn,
  5.6 GB/s ở shape 0.22 MB) đến từ độ hạt của lời gọi và latency, không từ đường cấp phát.
- `transparent_hugepage/enabled = [always]` trên máy này, nên E1 thực chất so *độ lệch của allocator*
  chứ không so "có hugepage vs 4 KiB": cả hai buffer đều có thể đã được THP. Muốn tách đúng biến đó
  phải có nhánh ép 4 KiB (`madvise(MADV_NOHUGEPAGE)`) — việc đó cần `unsafe`, và chỉ nên làm nếu có
  lý do khác để mở exception EXC-002.
- E2 (hugepage trên bộ nhớ anonymous) **không chạy**: E1 không cho tín hiệu, nên theo mục 4 dừng lại.
  Ghi "chưa đo" thay vì suy diễn.

## 5. Options

| # | Cách | Được | Mất | Policy |
|---|---|---|---|---|
| A | Giữ `fs::read` | 0 thay đổi, portable, không `unsafe` | 1.111 GB private RSS, không căn trang, không THP | none |
| B | Heap anonymous căn 2 MiB (`std::alloc::alloc` với `Layout::from_size_align`) + `MADV_HUGEPAGE` | **Đường duy nhất có hugepage trên máy này**; bỏ được TLB pressure | Vẫn private RSS, vẫn copy 1.1 GB lúc load; `unsafe` ngoài `simd.rs` | EXC-002 |
| C | **mmap read-only**, fallback `fs::read` | RSS giảm (page cache dùng chung, evictable), cold start tốt hơn, chạy được model > RAM | **Không có hugepage** (kernel này không bật THP cho file read-only); `unsafe` hoặc dependency `memmap2`; SIGBUS nếu file bị truncate; Windows cần đường riêng | EXC-002 + R12 |
| D | C + `MAP_POPULATE` | Bớt page-fault lúc chạy | Mất lợi thế cold start, RSS lại đầy đủ, vẫn không hugepage | như C |

`GgufFile::parse(&[u8])` đã nhận slice, và `Pool` chỉ borrow `&'d [u8]` trong scope của một pass, nên
một owner `Mapped` sống trong `main` là đủ — **không cần đổi API của core, runtime hay backend**.

## 6. Decision (nếu E1/E2 đạt ngưỡng)

Quyết định phụ thuộc vào **cái nào thắng trong E1/E2**, không chọn trước:

**Kết quả: E1 dưới ngưỡng, E2 không chạy → giữ A (`fs::read`).** Không mở EXC-002, không thêm
`unsafe`, không thêm dependency. mmap quay lại bàn chỉ khi mục tiêu là bộ nhớ (RSS/cold start/model
lớn hơn RAM) và khi đó phải viết lại tiêu chí promote theo mục tiêu đó, không mượn ngưỡng tốc độ.

Đòn bẩy tốc độ còn lại không nằm ở bộ nhớ đệm weight: xem mục 4 (E1) và phần per-shape trong
`docs/benchmarks.md` — 25% thời gian Q4_K nằm ở các tensor 0.22–1.3 MB, và đó là latency của lời gọi.

Mọi lựa chọn đều có fallback `fs::read` và gate `VIPERA_MMAP=0` (hoặc `VIPERA_ALIGNED=0`) để tắt trong
một commit.

Sửa policy trước khi merge, đúng như ADR-001:

| Luật bị phá | Ghi ở đâu | Sửa gì |
|---|---|---|
| `unsafe` chỉ trong `simd.rs` | `Cargo.toml` workspace vẫn `forbid` | `allow(unsafe_code)` chỉ trong module mới `src/mapped.rs` của `bins/vipera` (hoặc `vipera-core`), cùng tiền lệ `simd.rs:59` |
| Mọi dependency cần approval | `CONSTRAINTS.md` | **EXC-002** với owner + expiry; nếu dùng `memmap2` thì thêm mục **R12** trong `research.md` theo admission rule của R3 |
| Không có nền tảng thứ hai | CI hiện ubuntu-only | Ghi rõ Windows/macOS dùng fallback `fs::read`; không mở matrix ở v1 |

## 7. Test oracle

- `vipera-conformance` + `a_batched_step_matches_every_sequence_run_alone` phải xanh và **bit-identical**
  — mmap không được đổi một bit nào.
- So `argmax` và logits giữa `VIPERA_MMAP=0` và `=1` trên ≥3 prompt.
- Test riêng: file bị truncate sau khi map → phải trả `ViperaError`, không SIGBUS giết process.
- `verify` (predicted vs actual RSS) phải khớp hơn hiện tại, không kém.

## 8. Promote / rollback

Baseline hiện tại: **decode median-of-5 35.1 ms (28.5 tok/s), best 31.3 ms; pp128 44.4–45.0 tok/s;
RSS ≈ 1.1 GB weight + KV**.

Promote (gate của phương án thắng ở mục 6 thành default) khi, đo median-of-5 trong cùng phiên:

| # | Tiêu chí | Ngưỡng |
|---|---|---|
| 1 | decode | nhanh hơn **≥ 5%** ở median (≤ 33.3 ms) |
| 2 | prefill | không chậm hơn 2% (`pp128 ≥ 43.5`) |
| 3 | cold start | tới token đầu **không chậm hơn** `fs::read` quá 10% |
| 4 | RSS | **giảm** so với baseline (mmap dùng page cache) — nếu tăng là hiểu sai cơ chế |
| 5 | đúng số | conformance + parity bit-identical, cả hai chế độ |
| 6 | ổn định | p99/token không tệ hơn 5% |

Rollback: tắt gate (`VIPERA_MMAP=0` / `VIPERA_ALIGNED=0`, fallback `fs::read` có sẵn) hoặc revert
một commit; không đụng parse/runtime.

## 9. Rủi ro

- **SIGBUS**: file bị truncate/thay trong lúc chạy. Phải giữ fd và/hoặc kiểm tra độ dài trước khi map;
  test ở mục 7.
- **Tail latency ở lần chạm đầu**: major fault cho mỗi 2 MiB; nếu không `MAP_POPULATE`, token đầu tiên
  vào một vùng mới có thể chậm hơn. Đo bằng p99, không chỉ median.
- **THP không đảm bảo**: `defrag=madvise` có thể từ chối hugepage dưới phân mảnh, và **chỉ áp dụng cho
  anonymous memory** trên kernel này. Ghi kết quả thật, không hứa.
- **RSS "giảm" nhưng thực tế là page cache**: báo cáo phải nói rõ shared/private, không so số tổng
  của `free`.
- **Windows/macOS**: fallback `fs::read`; đừng để build vỡ vì `libc`.
- **Đây là tối ưu bộ nhớ nhiều hơn là tốc độ**: nếu E1 chỉ cho 2–3%, giá trị chính vẫn là RSS và model
  > RAM — khi đó phải viết lại mục tiêu promote, không được hạ ngưỡng cho đủ.

## 10. Non-goals

- Không đổi layout KV cache, không đổi `ForwardPass`, không đổi thuật toán matvec.
- Không dùng `MAP_POPULATE` mặc định.
- Không thêm dependency nếu `libc`-free mmap bằng syscall trong một module `unsafe` là đủ (quyết định ở
  lúc implement, sau E1).
- Không hứa hugepage: chỉ báo cáo cái đo được.

## 11. Open questions

1. E1/E2 có đạt ≥5% không? (chặn toàn bộ ADR này)
2. `unsafe` mmap đặt ở `bins/vipera` hay `vipera-core`? `core` là nơi biết GGUF, nhưng `bins` là nơi
   đọc file — nghiêng về `bins` để giữ `core` không có `unsafe`.
3. Ai là owner và expiry của EXC-002?
4. Có cần `madvise(MADV_SEQUENTIAL)` cho vùng weight không, hay hardware prefetch đã đủ? (đo, đừng đoán)
