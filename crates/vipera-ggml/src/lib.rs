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
