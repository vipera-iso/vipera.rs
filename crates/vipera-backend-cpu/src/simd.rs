//! An AVX2 kernel for Q4_K, the format that dominates this model.
//!
//! This is the only module in the workspace allowed to use `unsafe`: `std::arch` intrinsics require it.
//! The kernel mirrors the scalar one in [`crate::quant`] operation for operation, takes exactly the
//! same inputs, and the test below runs both over randomised blocks and compares.
//!
//! Dispatch happens at runtime through `is_x86_feature_detected!("avx2")`; on a CPU without it the
//! scalar code is used and this module is never reached.
//!
//! Only Q4_K is wired up, because that is the only format where measuring said it was worth it. On an
//! i5-11300H this kernel is **2.5x** the scalar one over the same data (0.162 s to 0.065 s for 355M
//! multiply-adds), and the whole model went from 4.58 s to 3.55 s of compute for the same twelve
//! tokens, unchanged. Hand-written AVX2 for Q6_K, Q8_0 and F32 was written, checked, and measured at
//! 0.29x, 0.38x and 0.83x of the scalar speed on the same machine, so those versions were removed
//! rather than shipped as dead weight. Revisit them only with a measurement that says otherwise.
//!
//! # Safety
//!
//! The public functions are `unsafe` because they use AVX2 intrinsics. Besides CPU support, `row` must
//! be a whole number of 144-byte Q4_K blocks, `x` exactly 256 values per block, and `sums` at least 16
//! entries per block. [`crate::quant::matvec`] validates all of that before dispatching, and
//! [`try_dot`] is the safe entry point that checks the CPU.
#![allow(unsafe_code)]

use std::arch::x86_64::{
    __m128i, __m256, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_loadl_epi64, _mm_movehdup_ps,
    _mm_movehl_ps, _mm256_add_ps, _mm256_and_si256, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_epi32,
    _mm256_setzero_ps, _mm256_srli_epi32,
};

use vipera_core::gguf::GgmlType;

use crate::quant::{f16_to_f32, q4k_scale_min};

/// Whether this CPU can run the kernels below. Cached by the standard library.
pub fn available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// The AVX2 dot for a Q4_K row, or `None` when this CPU has no AVX2 or the format is another one.
///
/// Safe to call: it checks the feature itself, and the lengths were validated by the caller's
/// `matvec`, whose contract is the same one [`crate::quant::matvec`] enforces before dispatch.
pub fn try_dot(ty: GgmlType, row: &[u8], x: &[f32], sums: &[f32]) -> Option<f32> {
    if !available() {
        return None;
    }
    // SAFETY: `available()` established AVX2 support, and the caller's lengths match the scalar
    // kernel's expectations, which the kernels below mirror exactly.
    Some(unsafe {
        match ty {
            GgmlType::Q4K => q4k_dot(row, x, sums),
            _ => return None,
        }
    })
}

#[inline]
/// Horizontal sum of the eight lanes of `value`.
///
/// # Safety
///
/// The CPU must support AVX2. Both callers already satisfy that: `dot_nibbles` carries the same
/// `#[target_feature]`, and `try_dot` checked `available()` before entering the block.
#[target_feature(enable = "avx2")]
unsafe fn hsum(value: __m256) -> f32 {
    let high = _mm256_extractf128_ps(value, 1);
    let low = _mm256_castps256_ps128(value);
    let sum = _mm_add_ps(low, high);
    let pairs = _mm_add_ps(sum, _mm_movehdup_ps(sum));
    let total = _mm_add_ss(pairs, _mm_movehl_ps(pairs, pairs));
    _mm_cvtss_f32(total)
}

/// Dot of 32 nibbles (one Q4_K sub-block) with 32 activation values.
///
/// # Safety
///
/// `q` must point at 32 readable bytes, `x` at 32 readable floats, and the CPU must support AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_nibbles(q: *const u8, x: *const f32, high: bool) -> f32 {
    // SAFETY: the caller guarantees 32 readable bytes at `q`, 32 readable floats at `x`, and AVX2
    // support. The four 8-byte loads touch `q + 0..32` and the four vector loads touch `x + 0..32`
    // floats, so neither pointer leaves those bounds.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mask = _mm256_set1_epi32(0x0f);
        for chunk in 0..4 {
            let raw = _mm_loadl_epi64(q.add(chunk * 8) as *const __m128i);
            let bytes = _mm256_cvtepu8_epi32(raw);
            let nibbles = if high {
                _mm256_and_si256(_mm256_srli_epi32(bytes, 4), mask)
            } else {
                _mm256_and_si256(bytes, mask)
            };
            let values = _mm256_cvtepi32_ps(nibbles);
            let activations = _mm256_loadu_ps(x.add(chunk * 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(values, activations));
        }
        hsum(acc)
    }
}

/// Q4_K row dot. Mirrors [`crate::quant`]'s scalar version, including the minimum term.
///
/// # Safety
///
/// See the module documentation.
pub unsafe fn q4k_dot(row: &[u8], x: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the module documentation is the contract: `row` is whole 144-byte blocks, `x` holds
    // 256 floats per block, `sums` holds 16 entries per block, and the CPU has AVX2. Every read
    // stays inside those bounds, including `from_raw_parts(block + 4, 12)`, which covers exactly
    // the 12 scale/min bytes a Q4_K block reserves for them.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 144;
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let scales = std::slice::from_raw_parts(block.add(4), 12);
            for j in 0..4 {
                let q = block.add(16 + j * 32);
                let activations = x.as_ptr().add(b * 256 + j * 64);
                let low = dot_nibbles(q, activations, false);
                let high = dot_nibbles(q, activations.add(32), true);
                let (scale0, min0) = q4k_scale_min(scales, 2 * j);
                let (scale1, min1) = q4k_scale_min(scales, 2 * j + 1);
                let base = b * 16 + 4 * j;
                total += d * f32::from(scale0) * low
                    - dmin * f32::from(min0) * (sums[base] + sums[base + 1]);
                total += d * f32::from(scale1) * high
                    - dmin * f32::from(min1) * (sums[base + 2] + sums[base + 3]);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic bytes, so a failure is reproducible.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn byte(&mut self) -> u8 {
            self.next() as u8
        }

        fn unit(&mut self) -> f32 {
            (self.next() % 10_000) as f32 / 5_000.0 - 1.0
        }

        fn half(&mut self, low: f32, high: f32) -> [u8; 2] {
            let value = low + (high - low) * (self.next() % 1_000) as f32 / 1_000.0;
            crate::quant::f32_to_f16(value).to_le_bytes()
        }
    }

    fn close(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= 2e-4 * (1.0 + expected.abs())
    }

    #[test]
    fn avx2_q4k_matches_scalar() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for trial in 0..24 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let sums = crate::quant::x_sums(&x);
            let mut row = Vec::with_capacity(blocks * 144);
            for _ in 0..blocks {
                row.extend_from_slice(&rng.half(0.001, 0.06));
                row.extend_from_slice(&rng.half(0.0, 0.02));
                for _ in 0..12 {
                    row.push(rng.byte());
                }
                for _ in 0..128 {
                    row.push(rng.byte());
                }
            }
            let expected = crate::quant::q4k_dot(&row, &x, &sums);
            // SAFETY: `row` is whole blocks, `x` and `sums` are sized for them, and the test
            // returned early unless `available()` saw AVX2 on this CPU.
            let actual = unsafe { q4k_dot(&row, &x, &sums) };
            assert!(
                close(actual, expected),
                "trial {trial}: avx2 {actual} vs scalar {expected}"
            );
        }
    }
}
