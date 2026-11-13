//! ADR-001 prototype: ggml's Q4_K x Q8_K kernels, behind a three-function surface.
//!
//! This crate exists to answer one question with a number: how much of the prefill gap
//! against llama.cpp is the kernel rather than the harness? It is not a dependency of the
//! engine yet, it is not in the workspace, and nothing here is `unsafe`-free by accident —
//! the callers are three `unsafe` functions whose contracts are one line each.
//!
//! Layouts are ggml's, unchanged:
//! - a Q4_K row is `cols / 256` blocks of 144 bytes;
//! - a Q8_K row is `cols / 256` blocks of 292 bytes (`f32 d`, then `i8` quants, then `i16`
//!   per-16 sums);
//! - both are written and read by the same ggml version that is compiled in.

use std::ffi::CStr;
use std::os::raw::c_char;

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q8_K_BLOCK_BYTES: usize = 292;

extern "C" {
    fn vp_init_tables();
    fn vp_gemv_q4_k_8x8_q8_k(
        cols: i32,
        out: *mut f32,
        repacked: *const u8,
        activations: *const u8,
        rows: i32,
    ) -> i32;
    fn vp_quantize_mat_q8_k_4x8(x: *const f32, out: *mut u8, cols: i64) -> i32;
    fn vp_gemm_q4_k_8x8_q8_k(
        cols: i32,
        out: *mut f32,
        stride: usize,
        repacked: *const u8,
        activations: *const u8,
        rows: i32,
        columns: i32,
    ) -> i32;
    fn vp_quantize_row_q8_k(cols: i64, x: *const f32, out: *mut u8) -> i32;
    fn vp_vec_dot_q4_k_q8_k(cols: i64, q4k_row: *const u8, q8k: *const u8, out: *mut f32) -> i32;
    fn vp_kernel_build() -> *const c_char;
}

/// Fill the lookups the kernel needs. Cheap once, required before any dot: the half table
/// is how the kernel reads a Q4_K block's scale.
fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe { vp_init_tables() });
}

/// Bytes one repacked block of eight rows occupies: `block_q4_Kx8`.
pub const Q4K8_BLOCK_BYTES: usize = 1152;

/// Rows a repacked block covers, and the group that must share one.
pub const REPACK_ROWS: usize = 8;

/// Repack `rows` Q4_K rows of `cols` weights into ggml's `block_q4_Kx8` layout.
///
/// The layout is what lets the x86 kernels skip the per-call unpack: for every 256-column
/// block, eight rows share one 1152-byte record of `d[8]`, `dmin[8]`, `scales[96]` and a
/// `qs[1024]` that interleaves the eight rows' nibbles. It is a permutation of the raw
/// bytes, so it costs one pass at load and nothing afterwards.
pub fn repack_q4_k_8x8(rows: usize, cols: usize, raw: &[u8]) -> Vec<u8> {
    assert!(
        rows.is_multiple_of(REPACK_ROWS),
        "rows must be a multiple of {REPACK_ROWS}"
    );
    assert!(
        cols.is_multiple_of(QK_K),
        "cols must be a multiple of {QK_K}"
    );
    let stride = q4_k_bytes(cols);
    assert_eq!(raw.len(), stride * rows, "raw must hold every row");
    let blocks = cols / QK_K;
    let mut out = vec![0u8; (rows / REPACK_ROWS) * blocks * Q4K8_BLOCK_BYTES];
    for group in 0..rows / REPACK_ROWS {
        for block in 0..blocks {
            let base = (group * blocks + block) * Q4K8_BLOCK_BYTES;
            let record = &mut out[base..base + Q4K8_BLOCK_BYTES];
            for row in 0..REPACK_ROWS {
                let source = &raw[(group * REPACK_ROWS + row) * stride + block * 144..];
                record[row * 2..row * 2 + 2].copy_from_slice(&source[0..2]);
                record[16 + row * 2..16 + row * 2 + 2].copy_from_slice(&source[2..4]);
                record[32 + row * 12..32 + row * 12 + 12].copy_from_slice(&source[4..16]);
                for k in 0..16 {
                    for i in 0..8 {
                        // `k` walks 8-value runs across the row, `j` the eight rows, so the
                        // record holds eight rows per run — the order `dpbusd` reads.
                        let from = (k / 4) * 32 + (k % 4) * 8 + i;
                        record[128 + k * 64 + row * 8 + i] = source[16 + from];
                    }
                }
            }
        }
    }
    out
}

/// One repacked weight block against `rows` activations.
///
/// # Safety
/// `repacked` must come from [`repack_q4_k_8x8`] for the same `cols` and `rows`, and
/// `activations` must be a Q8_K row for the same `cols`.
pub unsafe fn gemv_q4_k_8x8(
    cols: usize,
    rows: usize,
    repacked: &[u8],
    activations: &[u8],
) -> Result<Vec<f32>, i32> {
    init();
    assert_eq!(
        repacked.len(),
        rows / REPACK_ROWS * (cols / QK_K) * Q4K8_BLOCK_BYTES
    );
    assert_eq!(activations.len(), q8_k_bytes(cols));
    let mut out = vec![0.0f32; rows];
    let code = unsafe {
        vp_gemv_q4_k_8x8_q8_k(
            cols as i32,
            out.as_mut_ptr(),
            repacked.as_ptr(),
            activations.as_ptr(),
            rows as i32,
        )
    };
    (code == 0).then_some(out).ok_or(code)
}

/// Bytes four interleaved activation rows of `cols` values occupy: `block_q8_Kx4`.
pub fn q8_k_mat_bytes(rows: usize, cols: usize) -> usize {
    assert!(rows.is_multiple_of(4) && cols.is_multiple_of(QK_K));
    rows / 4 * (cols / QK_K) * 1168
}

/// Quantize `rows` activation rows into the interleaved layout the GEMM reads.
pub fn quantize_mat_q8_k_4x8(x: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>, i32> {
    init();
    assert_eq!(x.len(), rows * cols);
    let mut out = vec![0u8; q8_k_mat_bytes(rows, cols)];
    let code = unsafe { vp_quantize_mat_q8_k_4x8(x.as_ptr(), out.as_mut_ptr(), cols as i64) };
    (code == 0).then_some(out).ok_or(code)
}

/// `rows` activation rows against `columns` repacked weight rows.
///
/// # Safety
/// `repacked` must come from [`repack_q4_k_8x8`] with those dimensions, `activations` from
/// [`quantize_mat_q8_k_4x8`], and `rows` a multiple of four.
pub unsafe fn gemm_q4_k_8x8(
    cols: usize,
    rows: usize,
    columns: usize,
    repacked: &[u8],
    activations: &[u8],
) -> Result<Vec<f32>, i32> {
    init();
    assert_eq!(activations.len(), q8_k_mat_bytes(rows, cols));
    let mut out = vec![0.0f32; rows * columns];
    let code = unsafe {
        vp_gemm_q4_k_8x8_q8_k(
            cols as i32,
            out.as_mut_ptr(),
            columns,
            repacked.as_ptr(),
            activations.as_ptr(),
            rows as i32,
            columns as i32,
        )
    };
    (code == 0).then_some(out).ok_or(code)
}

/// Which ggml kernel this build selected, from its own preprocessor.
pub fn kernel_build() -> String {
    unsafe { CStr::from_ptr(vp_kernel_build()) }
        .to_string_lossy()
        .into_owned()
}

/// Bytes a Q8_K row of `cols` floats occupies.
pub fn q8_k_bytes(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(QK_K),
        "cols must be a multiple of {QK_K}"
    );
    cols / QK_K * Q8_K_BLOCK_BYTES
}

/// Bytes a Q4_K row of `cols` weights occupies.
pub fn q4_k_bytes(cols: usize) -> usize {
    assert!(
        cols.is_multiple_of(QK_K),
        "cols must be a multiple of {QK_K}"
    );
    cols / QK_K * Q4_K_BLOCK_BYTES
}

/// Quantize one f32 activation row to Q8_K.
///
/// # Safety
/// `dst` must be at least [`q8_k_bytes`]`(src.len())` bytes and 2-byte aligned.
pub unsafe fn quantize_q8_k(src: &[f32], dst: &mut [u8]) -> Result<(), i32> {
    init();
    assert_eq!(dst.len(), q8_k_bytes(src.len()));
    let code = unsafe { vp_quantize_row_q8_k(src.len() as i64, src.as_ptr(), dst.as_mut_ptr()) };
    (code == 0).then_some(()).ok_or(code)
}

/// One Q4_K weight row dotted with one Q8_K activation row.
///
/// # Safety
/// `row` must be exactly [`q4_k_bytes`]`(cols)` bytes, `q8k` exactly [`q8_k_bytes`]`(cols)`,
/// and both must come from a build with the same `QK_K`.
pub unsafe fn dot_q4_k_q8_k(cols: usize, row: &[u8], q8k: &[u8]) -> Result<f32, i32> {
    init();
    assert_eq!(row.len(), q4_k_bytes(cols));
    assert_eq!(q8k.len(), q8_k_bytes(cols));
    let mut out = 0.0f32;
    let code = unsafe {
        vp_vec_dot_q4_k_q8_k(
            cols as i64,
            row.as_ptr(),
            q8k.as_ptr(),
            &mut out as *mut f32,
        )
    };
    (code == 0).then_some(out).ok_or(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Q4_K block with `d = 1`, `dmin = 0`, all six-bit scales 1, and 4-bit quants
    /// cycling 0..15. Values are small and well separated, so a wrong stride or a wrong
    /// block size shows up as a number, not as a crash.
    fn synthetic_q4k(cols: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(q4_k_bytes(cols));
        for block in 0..cols / QK_K {
            let mut raw = [0u8; Q4_K_BLOCK_BYTES];
            raw[0..2].copy_from_slice(&half(1.0));
            raw[2..4].copy_from_slice(&half(0.0));
            // scales: 12 bytes of 6-bit fields; 0x01 decodes to scale 1 in every 6-bit field for the
            // eight scale/six sub-blocks that this layout uses.
            for byte in raw[4..16].iter_mut() {
                *byte = 0x01;
            }
            for (index, byte) in raw[16..].iter_mut().enumerate() {
                let low = ((index + block) % 16) as u8;
                let high = ((index + block + 7) % 16) as u8;
                *byte = low | (high << 4);
            }
            row.extend_from_slice(&raw);
        }
        row
    }

    fn half(value: f32) -> [u8; 2] {
        // 1.0 and 0.0 are exact in f16; avoid pulling a conversion in for the test.
        let bits = if value == 1.0 { 0x3c00u16 } else { 0x0000u16 };
        bits.to_le_bytes()
    }

    /// Dot the same synthetic row and the same quantized activations in Rust, following
    /// ggml's block layouts exactly (Q8_K: `f32 d`, then `i8 qs[256]`, then `i16 bsums`;
    /// Q4_K: 32-byte groups where the low nibble is `y[p]` and the high nibble `y[p + 32]`).
    #[test]
    fn dot_matches_a_rust_reference() {
        let cols = 512;
        let row = synthetic_q4k(cols);
        let x: Vec<f32> = (0..cols).map(|at| (at as f32 * 0.01).sin()).collect();

        let mut q8k = vec![0u8; q8_k_bytes(cols)];
        unsafe { quantize_q8_k(&x, &mut q8k).expect("quantize") };
        let bound = unsafe { dot_q4_k_q8_k(cols, &row, &q8k).expect("dot") };

        let reference: f32 = q8k
            .as_chunks::<Q8_K_BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
            .map(|(block, activation)| {
                let d_activation = f32::from_le_bytes(activation[0..4].try_into().unwrap());
                let quants = &activation[4..4 + QK_K];
                let q4 = &row[block * Q4_K_BLOCK_BYTES..(block + 1) * Q4_K_BLOCK_BYTES];
                let d_weight = half_to_f32([q4[0], q4[1]]);
                let weights = &q4[16..];
                let mut sum = 0.0f32;
                for (index, byte) in weights.iter().enumerate() {
                    let group = index / 32;
                    let low = 64 * group + index % 32;
                    // Every synthetic scale is 1 and dmin is 0, so `scale * quant` is the value.
                    sum += (byte & 0x0f) as f32 * quants[low] as i8 as f32;
                    sum += (byte >> 4) as f32 * quants[low + 32] as i8 as f32;
                }
                sum * d_weight * d_activation
            })
            .sum();

        let tolerance = bound.abs().max(reference.abs()) * 1e-3 + 1e-3;
        assert!(
            (bound - reference).abs() <= tolerance,
            "ggml dot {bound} vs reference {reference} (tolerance {tolerance})"
        );
    }

    #[test]
    fn repack_alone_has_the_expected_size() {
        let cols = 512;
        let rows = 16;
        let raw: Vec<u8> = (0..rows)
            .flat_map(|row| synthetic_q4k_with_salt(cols, row))
            .collect();
        let packed = repack_q4_k_8x8(rows, cols, &raw);
        assert_eq!(packed.len(), 2 * 2 * Q4K8_BLOCK_BYTES);
        assert_eq!(packed, repack_q4_k_8x8(rows, cols, &raw));
    }

    /// The repack is a permutation of the raw bytes: if it is right, eight repacked rows
    /// through ggml's own GEMV must equal eight separate dots through the row kernel.
    #[test]
    fn repacked_gemv_matches_row_dots() {
        let cols = 512;
        let rows = 16;
        let raw: Vec<u8> = (0..rows)
            .flat_map(|row| synthetic_q4k_with_salt(cols, row))
            .collect();
        let x: Vec<f32> = (0..cols).map(|at| (at as f32 * 0.017).cos()).collect();
        let mut q8k = vec![0u8; q8_k_bytes(cols)];
        unsafe { quantize_q8_k(&x, &mut q8k).expect("quantize") };

        let packed = repack_q4_k_8x8(rows, cols, &raw);
        let got = unsafe { gemv_q4_k_8x8(cols, rows, &packed, &q8k).expect("gemv") };

        let stride = q4_k_bytes(cols);
        for row in 0..rows {
            let want = unsafe {
                dot_q4_k_q8_k(cols, &raw[row * stride..(row + 1) * stride], &q8k).expect("dot")
            };
            let tolerance = want.abs().max(got[row].abs()) * 1e-3 + 1e-3;
            assert!(
                (got[row] - want).abs() <= tolerance,
                "row {row}: repacked {} vs row kernel {want}",
                got[row]
            );
        }
    }

    fn synthetic_q4k_with_salt(cols: usize, salt: usize) -> Vec<u8> {
        let mut row = synthetic_q4k(cols);
        for (block_index, block) in row.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            for (index, byte) in block[16..].iter_mut().enumerate() {
                let low = ((index + salt + block_index) % 15 + 1) as u8;
                let high = ((index + salt + block_index + 4) % 15 + 1) as u8;
                *byte = low | (high << 4);
            }
        }
        row
    }

    fn half_to_f32(bytes: [u8; 2]) -> f32 {
        let bits = u16::from_le_bytes(bytes) as u32;
        let sign = (bits & 0x8000) >> 15;
        let exponent = (bits >> 10) & 0x1f;
        let mantissa = bits & 0x3ff;
        let value = if exponent == 0 {
            (mantissa as f32) * 2f32.powi(-24)
        } else {
            (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(exponent as i32 - 15)
        };
        if sign == 1 {
            -value
        } else {
            value
        }
    }
}
