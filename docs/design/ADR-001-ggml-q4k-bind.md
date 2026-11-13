# ADR-001: Bind ggml Q4_K GEMM sau feature flag

Status: **Accepted cho phần policy** (EXC-001 đã ghi), **implementation chưa bắt đầu** — mục 10 còn
3 điều kiện chặn.
Ngày: 2026-11-13 · Owner: vipera-iso · Expiry của EXC-001: 2027-02-11 · Supersedes: none

Tài liệu này thay cho bản nháp artifact ngày 2026-11-13: bản nháp đó không build được và bind sai
symbol. Mục 10 ghi rõ 5 chỗ đã sửa và vì sao.

## 1. Context — số đo, không phải cảm giác

Đo trên chính repo này (i5-11300H, 8 thread, AVX-VNNI, `qwen2.5-1.5b-q4km.gguf` 1.117 GB),
`win_vs_llama::prefill_vs_decode` + `VIPERA_PROFILE_STEP=1`:

| Metric | Vipera | llama.cpp — đo 2026-11-13, build 5676dfb | Gap |
|---|---|---|---|
| pp64 | 51.3 tok/s | — | — |
| **pp128** | **44.4–45.0 tok/s** | **209.06 tok/s** (8T) / 211.26 (4T) | **4.6x** |
| pp512 | 40.9–41.1 tok/s | — | — |
| **tg (ctx 32)** | **28.1–32.0 tok/s** (31.3–35.2 ms/step) | **37.13** (8T) / **39.04** (4T) tok/s | **1.16–1.34x** |

Baseline này đã tính patch 0002 (streaming attention) đang nằm trong tree. Nó **không** làm prefill
chậm đi: pp128 44.4 không patch / 45.0 có gate là chênh lệch giữa các lần chạy, và mẫu 47.1 đo sớm
hơn không tái lặp. Cái nó thật sự đổi là decode ở depth: ctx 1024 56.1 → 51.0 ms/step (+9% tok/s).
Mọi ngưỡng promote ở mục 7 so với dòng 44.4–45.0 / 28.1 này.

Chia thời gian:

| Phase | Decode | Prefill (chunk) |
|---|---|---|
| Q4_K dots | 22.94 ms (**64%**) | **74%** |
| Q6_K dots | 11.41 ms (**32%**) | 19% |
| attention | 0.51 ms (1%) → 10–12% ở depth ~400 | 3% |
| norm / rest | 0.00 ms / 1.17 ms | 0% / 4% |

Hai kết luận định hình ADR này:

1. **Decode là memory-bound, nhưng KHÔNG ở trần** — đo bằng STREAM 5.10 (`docs/benchmarks.md`):
   read-only **42.8 GB/s** (4 thread) / 41.7 (8 thread), Copy 39.7, Triad 29.6; SMT không thêm gì.
   Một token decode đọc **0.980 GB** (1.111 GB tensor trừ 125 MB `token_embd`, decode không đọc),
   nên trần áp dụng là read-only **~43 GB/s** → sàn **22.8 ms/token**. Vipera: 28.8 GB/s (4T, 67%)
   và 31.3 GB/s (8T, 73%); llama.cpp 39.04 tok/s (4T) = **38.3 GB/s (89%)**. Vậy trần là thật, dư
   địa 1.34x ở cùng số thread, nhưng **không phải ở access pattern**: prefetch (`_MM_HINT_T0`, 31.5
   vs 40.3 GB/s) và row blocking (0% ở mọi thread count) đã implement và bị số liệu bác bỏ
   (`docs/benchmarks.md`, "Decode headroom"), và thread curve cho thấy thread đầu đã đạt ~10 GB/s,
   tức trần của một core. Cùng một `vpdpbusd`, C++ không stream nhanh hơn Rust.
2. **Prefill mới là chỗ có ROI.** 74% là Q4_K, gap 4.6x là chất lượng kernel tile/repack. Ba prototype
   trong repo đã bị số liệu bác bỏ: tiled scalar **0.04 GB/s**, packed VNNI **5.00 vs 5.52 GB/s**,
   `B>=32` spill. Viết lại C++ cùng intrinsics cho ra cùng instruction stream → gain ~0. Phần đã có
   sẵn nằm trong ggml, nên cơ chế mua được nó là **bind**.

Phạm vi: **chỉ Q4_K** (`gemv` cho decode, `gemm` cho prefill). Q6_K là 32% decode nên phải mở ADR
riêng — không nhét vào đây.

## 2. Decision

Feature `ggml-q4k` trong `vipera-backend-cpu`, default **off**: bind kernel Q4_K của ggml qua một shim
`extern "C"` do ta viết, weight repack sang `block_q4_Kx8` một lần lúc load, activation quantize sang
`block_q8_K` mỗi batch.

Bốn luật bị phá, và trạng thái ghi nhận:

| Luật bị phá | Ghi ở đâu | Trạng thái |
|---|---|---|
| Stdlib-only foundation (R3) | `specs/001-vipera-rs-engine/research.md` **R11** | ✅ đã ghi (purpose, alternative, cost, điều kiện admission) |
| Mọi dependency cần approval (`CONSTRAINTS.md` rule 2) | `CONSTRAINTS.md` bảng **Exceptions** | ✅ **EXC-001**, owner `vipera-iso`, expiry 2027-02-11, rollback = feature off |
| `unsafe_code` chỉ trong `simd.rs` | `Cargo.toml` workspace vẫn `forbid` | ✅ giữ nguyên; chỉ `vipera-backend-cpu` mở `deny` → `allow` trong đúng `ffi_ggml.rs` (tiền lệ `simd.rs:59`) |
| Win-condition "vipera.rs > llama.cpp" | `HANDOFF.md`, `PLAN-tiled-prefill.md`, `SPEC-vipera-perf.md` | ⛔ **chưa đổi** — cần chữ ký của bạn, xem mục 10 |

Không có 4 dòng này thì không merge code. Ba dòng đầu đã xong; dòng 4 là quyết định của bạn, không
phải của agent.

## 3. Seam spec

```
Boundary:        vipera-runtime::ForwardPass  ↔  shim C ABI  ↔  ggml Q4_K kernels, in-process
Mechanism:       C ABI (extern "C") trên một shim .cpp; ggml là C++17, không có C linkage
                 → toolchain là C++ (`CXX`), không phải `CC`; EXC-001 ghi rõ scope này
Allocator:       Rust cấp mọi buffer (weight repacked, activation q8_K, output); ggml không cấp phát
Ownership:       weight REPACK một lần lúc load sang buffer riêng (block_q4_Kx8), không borrow mmap
                 nữa; activation quantize per batch; output do caller cấp
Error model:     kernel ggml trả void và GGML_ASSERT → abort. Rust phải validate shape TRƯỚC khi gọi;
                 không có return code để map sang ViperaError
Threading:       runtime vẫn sở hữu stepping thread (1 thread gọi); shim không tự tạo thread
Cancellation:    không cần — 1 call ~1 ms, ngắn hơn mọi ngưỡng huỷ của request
Marshalling:     ~20 ns/call; 28 layer × ~6 matmul ≈ 168 call/step trên 35.2 ms → 3.4 µs ≈ 0.01%
Quantize:        `ggml_quantize_mat_q8_K_4x8` chạy MỖI batch trước GEMM — chưa nằm trong 3.4 µs ở
                 trên, phải đo riêng (gate: ≤ 0.1% step, xem 3.1 và mục 6)
Test oracle:     vipera-conformance + parity test mới (argmax + max-abs ≤ 1e-3 so với pooled path);
                 a_batched_step_matches_every_sequence_run_alone giữ bit-identical vì nó so cùng path
```

Luật 5%: 0.01% ≪ 5% — seam không phải nút thắt. Nút thắt là kernel và mục tiêu.

### 3.1 Bốn ràng buộc phải có trước khi implement

Rút ra từ việc sửa seam (mục 11, lỗi #3). Ba cái đầu là correctness, không phải optimization — sai một
cái là output sai hoặc process chết, không phải chậm.

1. **Toolchain C++, không phải C.** `cc::Build` phải `.cpp(true)`; CI cần `CXX` (g++/clang++ hoặc MSVC
   C++ nếu mở Windows). Gate sẽ vỡ ở **link**, không phải typecheck — nên mục 6 bắt buộc có bước
   `build --release`.
2. **Quantize activation phải đo riêng.** Mỗi call còn `ggml_quantize_mat_q8_K_4x8` trên activation
   trước GEMM. Rẻ, nhưng nếu > 0.1% step thì con số 0.01% của seam spec là sai và phải ghi lại. Một
   dòng bench trong `kv_attention_cost`-style test, không cần hạ tầng mới.
3. **Repack cache key = CPU features × ggml SHA × model hash.** Chỉ áp dụng nếu chọn phương án B
   (repack ra file + mmap). Layout `8x8` phụ thuộc tập lệnh (AVX2 vs VNNI vs AVX512), kernel phụ thuộc
   commit ggml, kích thước/shape phụ thuộc model. Thiếu một thành phần trong key → **output sai âm
   thầm**, không crash, không log. Đây là constraint correctness, không phải ghi chú tối ưu.
4. **`-DNDEBUG` bắt buộc trong release.** ggml dùng `GGML_ASSERT` → `abort()`: nếu assert fail trong
   server thì cả process chết, không có `Result` để trả về. Hai lớp: shim/build define `NDEBUG` ở
   release, **và** validation shape ở Rust chặt trước khi gọi (mục 11). Không được dựa vào `if rc != 0`
   vì kernel trả `void`.

## 4. Hình dạng thật của ggml (khác bản nháp)

Đối chiếu `vipera.cpp/ggml` (bản llama.cpp trong máy):

| Thứ bản nháp giả định | Thực tế |
|---|---|
| `ggml.c` + `ggml-quants.c`, C | Kernel nằm trong **C++**: `ggml/src/ggml-cpu/repack.cpp` (5 283 dòng) + `arch/x86/repack.cpp` (6 407 dòng), cần C++17 và include từ `ggml-base` |
| `ggml_q4k_gemm(w, x, y, n, k, m) -> i32` | Không tồn tại. Thật: `void ggml_gemm_q4_K_8x8_q8_K(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc)` (`repack.h:174`); decode dùng `ggml_gemv_q4_K_8x8_q8_K` (`repack.h:156`) |
| weight borrow thẳng từ mmap GGUF | `vx` phải là `block_q4_Kx8` đã repack (`repack.h:48`); phải có pass repack lúc load |
| chỉ cần 1 hàm FFI | Cần thêm quantize activation: `ggml_quantize_mat_q8_K_4x8(const float *x, void *vy, int64_t k)` (`repack.h:148`) |
| link trực tiếp symbol ggml | Symbol là C++, không có C linkage → phải có shim `extern "C"` (một file ~40 dòng) |

Hệ quả về chi phí: thêm (a) repack `block_q4_Kx8` — làm **trong Rust** (byte shuffle thuần, không
`unsafe`), (b) quantize `block_q8_K` mỗi batch, (c) shim. Đó là lý do cost ở mục 8 là 2–4 dev-month,
không phải "1–2 tuần".

## 5. File nào chạm (khi implement)

| File | Việc |
|---|---|
| `vendor/ggml/` (mới) | pin commit, kèm LICENSE; chỉ giữ phần cần cho 2 translation unit |
| `vendor/ggml/COMMIT` | SHA đã pin (mục 10, điều kiện 1) |
| `crates/vipera-backend-cpu/build.rs` (mới) | `cc` với `.cpp(true)`, đọc `CARGO_FEATURE_GGML_Q4K` (không phải `cfg!`) |
| `crates/vipera-backend-cpu/src/vipera_ggml_shim.cpp` (mới) | `extern "C"` wrapper, include `repack.h` |
| `crates/vipera-backend-cpu/src/ffi_ggml.rs` (mới) | toàn bộ `unsafe`, `unsafe extern "C"` (edition 2024), validate shape |
| `crates/vipera-backend-cpu/src/q4k_repack.rs` (mới) | `block_q4_Kx8` packing trong Rust, có test so layout |
| `crates/vipera-backend-cpu/src/quant.rs` | dispatch `#[cfg(feature = "ggml-q4k")]` ở nhánh pooled Q4_K |
| `crates/vipera-backend-cpu/Cargo.toml` | `[features] ggml-q4k = []`, `build = "build.rs"`, `[build-dependencies] cc = "1"` (**không** `optional`) |
| `crates/vipera-backend-cpu/tests/ggml_parity.rs` (mới) | argmax + max-abs vs pooled path, cả B=1 và B=16 |
| `.github/workflows/ci.yml` | thêm job `gate-ggml-q4k` (mục 6) |

Không chạm `vipera-runtime`: `ForwardPass` đã là seam đúng.

**Không** thêm `cc`/`ggml-q4k` vào `Cargo.toml` trước khi có `build.rs` + module dùng nó — config
không consumer vi phạm `CONSTRAINTS.md` rule 5 và tạo dead feature.

## 6. Checklist CI

Gate hiện tại (`.github/workflows/ci.yml`: fmt → clippy `-D warnings` → test `--locked`) phải xanh
**nguyên trạng với feature off** — đó là bằng chứng rollback còn sống.

- [ ] `cargo fmt --all --check` (không đổi)
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` (feature off — bắt buộc)
- [ ] `cargo test --workspace --locked` (feature off — bắt buộc, 0 ignored)
- [ ] Job `gate-ggml-q4k`, `ubuntu-latest`, `timeout-minutes: 25`, `actions/checkout@v7` (đúng version đang dùng):
  - [ ] `cargo clippy --workspace --all-targets --features ggml-q4k --locked -- -D warnings`
  - [ ] `cargo test --workspace --features ggml-q4k --locked`
  - [ ] `cargo build --release -p vipera-backend-cpu --features ggml-q4k` — bắt lỗi link (C++, `CXX`)
- [ ] Build release đó phải define `NDEBUG`; kiểm bằng một assert cố ý trong shim ở job test
- [ ] Bench riêng chi phí `quantize_mat_q8_K_4x8` cho B=1 và B=16 (gate ≤ 0.1% step)
- [ ] `vendor/ggml/LICENSE` + `COMMIT` có trong repo
- [ ] Windows **không** vào matrix ở v1: cần MSVC C++ và guard AVX riêng — quyết định riêng, không kèm
- [ ] Bench `win_vs_llama` **không** vào gate (cần model, `#[ignore]` như hiện tại)

## 7. Promote / rollback

Baseline hôm nay (`win_vs_llama::prefill_vs_decode`, model/máy ở mục 1, đã gồm patch 0002):
**pp128 44.4–45.0 · pp64 50.0–50.7 · pp512 40.9–41.1 · tg(ctx32) 28.1–32.0 tok/s (31.3–35.2
ms/step).** llama.cpp cùng máy: `pp128 209.06`, `tg64 37.13` (8T) — xem `docs/benchmarks.md`.

Promote (đổi default sang on) chỉ khi **tất cả** đúng, 3 lần chạy liên tiếp, lấy median:

| # | Tiêu chí | Ngưỡng |
|---|---|---|
| 1 | prefill | `pp128` **≥ 100 tok/s** (≥2.1x baseline) |
| 2 | decode không thoái | `tg(ctx32)` **≥ 28.4 tok/s** |
| 3 | đúng số | argmax trùng pooled path trên ≥3 prompt; logits max-abs ≤ 1e-3; B=1 và B=16 |
| 4 | test | `cargo test --workspace` xanh ở **cả hai** cấu hình feature, 0 ignored |
| 5 | bộ nhớ | RSS đỉnh **≤ baseline + 0.75 GB** (một bản Q4_K repack, ~cùng số byte: 1 152 B/8 hàng vs 8×144 B). Vượt mức đó = fail; nếu mức đó vẫn không chấp nhận được thì phải repack ra **file cache** rồi mmap, không giữ bản thứ hai trong RAM |
| 6 | p99 | p99/token không tệ hơn baseline quá 5% |

Ghi cả 6 số vào `HANDOFF.md` trước khi đổi default. `pp128` dừng ở 60–99 → **giữ default off**, đóng
EXC-001, coi là CONSIDER, không promote.

**Rollback:** tắt `ggml-q4k` (default off) là hết. Code nằm sau `#[cfg(feature)]`, một commit revert,
không đụng `ForwardPass`, không migrate dữ liệu.

## 8. Cái giá

- CI/build phụ thuộc C++ toolchain ở job bật feature; feature off vẫn build bằng Rust thuần, nhưng
  crate `cc` vẫn được compile như build-dependency (build-deps không thể `optional`).
- Bề mặt `unsafe`: một module mới có `unsafe extern "C"` — review kỹ hơn `simd.rs`.
- Supply chain: ~11.7k dòng C++ ngoài repo; phải pin commit, giữ LICENSE, gate object < 5 MB.
- RSS: weight repacked là **bản thứ hai** của tensor Q4_K (không borrow mmap được nữa) — đây là rủi ro
  lớn nhất về bộ nhớ và là tiêu chí 5 ở mục 7. Phương án B nếu không chấp nhận: repack một lần ra file
  cache (`~/.vipera/repack/<model-hash>.q4k8`) rồi mmap file đó, đổi lại là thêm một bước I/O lúc load
  và một định dạng cache phải version.
- Debug: crash trong kernel C++ không có Rust backtrace sạch; ggml `assert` → abort, không có lỗi trả về.
- Build time và binary size tăng ở cấu hình bật feature.
- Về bản chất hot path thành wrapper quanh ggml — đúng như self-critique của `polyglot-architect`;
  đây là cái giá đã được chấp nhận bằng việc đổi mục tiêu (mục 10, điều kiện 3).

## 9. Non-goals

- Không swap/rewrite server, tokenizer, sampler, core, CLI (dưới 4% thời gian, không ROI).
- Không viết lại Q4_K bằng C++ từ đầu (2–4 dev-month, gain ~0 so với bind).
- Không bind Q6_K trong ADR này (32% decode — ADR riêng, có số riêng).
- Không bind cả pipeline ggml (graph, sampler, KV layout); KV vẫn của runtime.
- Không bật default trước khi đủ 6 tiêu chí mục 7.
- Không thay `vipera-conformance` — suite vẫn là oracle.

## 10. Chặn implementation (cần bạn quyết)

1. **Pin commit ggml + audit license/size.** Chưa có SHA. Gate: subset compile ra object < 5 MB.
   Ghi vào `vendor/ggml/COMMIT` và mục 2 của ADR này.
2. ~~Baseline llama.cpp phải đo lại~~ — **XONG 2026-11-13.** Build upstream 5676dfb, CPU-only,
   `GGML_AVX512=ON`, cùng máy + cùng quant: `pp128 209.06` (8T) / `211.26` (4T), `tg64 37.13` (8T) /
   `39.04` (4T). Số 256 / 34.7 trong `HANDOFF.md` không tái lặp và đã bị thay bằng số này trong
   `docs/benchmarks.md`. Ngưỡng `pp128 ≥ 100` giữ nguyên: nó là 2.2x baseline hiện tại.
3. **Win-condition mới** ("engine nhanh nhất, được phép bind") phải được ghi vào `HANDOFF.md` /
   `SPEC-vipera-perf.md` bằng văn bản, có ngày. Không có dòng đó, ADR này chỉ là đề xuất.

Lưu ý đo: governor là `powersave` và không đổi được nếu không có root, nên tần số dao động 0.7–3.8
GHz và cùng một binary lệch tới 15% giữa các lần chạy. Mọi so sánh phải là best-of-N/median trong
cùng một phiên, không phải một mẫu.

## 11. Phụ lục: 5 chỗ bản nháp sai, và bản đúng

| # | Bản nháp | Vấn đề | Bản đúng |
|---|---|---|---|
| 1 | `[build-dependencies] cc = { optional = true }` | Cargo **cấm** `optional` trong build-dependencies → lỗi manifest | `cc = "1"` thường; bỏ compile bằng `CARGO_FEATURE_GGML_Q4K` |
| 2 | `#[cfg(feature = "ggml-q4k")]` trong `build.rs` | Build script không nhận `cfg(feature)`; phải đọc env | `std::env::var_os("CARGO_FEATURE_GGML_Q4K").is_none()` → return sớm |
| 3 | `ggml_q4k_gemm(...) -> i32`, `crate::Error::internal(rc)` | Symbol không tồn tại; kernel trả `void`; repo dùng `ViperaError::internal(impl Into<String>)` | Shim `extern "C"` + `ViperaError::internal("...")` |
| 4 | `#[cfg_attr(feature = ..., allow(unsafe_code))] mod ffi_ggml;` | Thừa và dễ sai; file đã tự `#![allow(unsafe_code)]` như `simd.rs:59` | `#[cfg(feature = "ggml-q4k")] pub(crate) mod ffi_ggml;` + inner allow trong file |
| 5 | CI: `actions/checkout@v4`, matrix Windows, clippy thiếu `-D warnings`/`--locked` | Lệch gate hiện có (`@v7`, ubuntu-only, `-D warnings`, `--locked`); C++ trên MSVC là scope riêng | Job ubuntu-only khớp gate hiện tại |

### `build.rs` (bản đúng)

```rust
fn main() {
    println!("cargo:rerun-if-changed=vendor/ggml");
    println!("cargo:rerun-if-changed=src/vipera_ggml_shim.cpp");
    println!("cargo:rerun-if-changed=src/ffi_ggml.rs");
    if std::env::var_os("CARGO_FEATURE_GGML_Q4K").is_none() {
        return; // feature off: không cần C++ toolchain
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/vipera_ggml_shim.cpp")
        .file("vendor/ggml/src/ggml-cpu/repack.cpp")
        .file("vendor/ggml/src/ggml-cpu/arch/x86/repack.cpp")
        .include("vendor/ggml/include")
        .include("vendor/ggml/src")
        .include("vendor/ggml/src/ggml-cpu")
        .warnings(false)
        .compile("vipera_ggml_q4k");
}
```

### `ffi_ggml.rs` (khung, edition 2024)

```rust
//! EXC-001: toàn bộ `unsafe` của feature `ggml-q4k` nằm trong file này.
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use vipera_errors::ViperaError;

unsafe extern "C" {
    // Shim của ta — không phải symbol C++ của ggml (chúng không có C linkage).
    fn vipera_ggml_gemm_q4k(
        n: c_int,
        out: *mut f32,
        out_row_bytes: usize,
        weights: *const c_void, // block_q4_Kx8, đã repack
        activations: *const c_void, // block_q8_K
        rows: c_int,
        tokens: c_int,
    );
}

/// Không có return code: ggml `assert` rồi abort, nên shape phải đúng trước khi gọi.
pub(crate) fn gemm_q4k(
    weights: &[u8],
    activations: &[u8],
    out: &mut [f32],
    n: usize,
    rows: usize,
    tokens: usize,
    out_row_bytes: usize,
) -> Result<(), ViperaError> {
    if n == 0 || rows == 0 || tokens == 0 || out_row_bytes < tokens * size_of::<f32>() {
        return Err(ViperaError::internal("ggml_q4k: shape mismatch"));
    }
    // SAFETY: caller đã cấp buffer theo đúng shape đã kiểm tra ở trên; weight là bản repack
    // thuộc sở hữu backend, activation là bản q8_K vừa quantize trong cùng call.
    unsafe {
        vipera_ggml_gemm_q4k(
            n as c_int,
            out.as_mut_ptr(),
            out_row_bytes,
            weights.as_ptr().cast(),
            activations.as_ptr().cast(),
            rows as c_int,
            tokens as c_int,
        );
    }
    Ok(())
}
```
