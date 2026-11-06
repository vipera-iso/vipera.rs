//! AVX2 kernels for the two quantized formats a Q4_K_M file is made of.
//!
//! This is the only module in the workspace allowed to use `unsafe`: `std::arch` intrinsics require it.
//! Each kernel mirrors the scalar one in [`crate::quant`] operation for operation, takes exactly the
//! same inputs, and the tests below run both over randomised blocks and compare.
//!
//! Dispatch happens at runtime through `is_x86_feature_detected!("avx2")`; on a CPU without it the
//! scalar code is used and this module is never reached.
//!
//! # Why `#[target_feature]` is on every public kernel
//!
//! An `_mm256_*` call is itself a `#[target_feature(enable = "avx2")]` function, so a kernel that
//! lacks the attribute compiles to one out-of-line call per intrinsic. That mistake is not a paper
//! cut: a Q6_K kernel written that way measured **0.88 GB/s**, half the scalar path it replaced, and
//! ran only because the attribute was missing. With the attribute it measures 10.9 GB/s on the shapes
//! below (`ffn_down`) and 13.4 GB/s on `output.weight`, against 1.6-1.8 GB/s for the scalar version.
//!
//! Q4_K is the larger share of the bytes (625 MB of the 1.1 GB read per token, once the token
//! embedding — a row lookup, not a matvec — is excluded) and measures 5.2-6.7 GB/s across repeated
//! runs on an i5-11300H, where the numbers vary with thermal state. It is the remaining bottleneck:
//! end-to-end this model went from 2.57 to 3.3-5.3 tok/s, not the 10 tok/s target, and closing that
//! gap needs the Q4_K kernel to roughly double again.
//!
//! # Safety
//!
//! The public functions are `unsafe` because they use AVX2 intrinsics. Besides CPU support, `row` must
//! be a whole number of blocks of the format named by the function (144 bytes for Q4_K, 210 for Q6_K),
//! `x` exactly 256 values per block, and `sums` at least 16 entries per Q4_K block. Q6_K ignores
//! `sums`. [`crate::quant::matvec`] validates all of that before dispatching, and [`try_dot`] is the
//! safe entry point that checks the CPU.
#![allow(unsafe_code)]

use std::arch::x86_64::{
    __m128i, __m256, __m256i, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_loadl_epi64,
    _mm_movehdup_ps, _mm_movehl_ps, _mm256_add_ps, _mm256_and_si256, _mm256_castps256_ps128,
    _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_sub_epi32,
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
            GgmlType::Q6K => q6k_dot(row, x),
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

/// Q4_K row dot. Mirrors [`crate::quant`]'s scalar version, including the minimum term.
///
/// # Safety
///
/// See the module documentation.
#[target_feature(enable = "avx2")]
pub unsafe fn q4k_dot(row: &[u8], x: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the module documentation is the contract: `row` is whole 144-byte blocks, `x` holds
    // 256 floats per block, `sums` holds 16 entries per block, and the CPU has AVX2. Every read
    // stays inside those bounds, including `from_raw_parts(block + 4, 12)`, which covers exactly
    // the 12 scale/min bytes a Q4_K block reserves for them.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 144;
        let mask = _mm256_set1_epi32(0x0f);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let scales = std::slice::from_raw_parts(block.add(4), 12);
            for j in 0..4 {
                let q = block.add(16 + j * 32);
                let activations = x.as_ptr().add(b * 256 + j * 64);
                // One pass over the 32 bytes: the low nibbles are values 0..32 of the sub-block and
                // the high nibbles are 32..64, so one load of eight bytes feeds both halves. Two
                // accumulators keep the additions independent.
                let mut low = _mm256_setzero_ps();
                let mut high = _mm256_setzero_ps();
                for chunk in 0..4 {
                    let raw = _mm_loadl_epi64(q.add(chunk * 8) as *const __m128i);
                    let bytes = _mm256_cvtepu8_epi32(raw);
                    let low_values = _mm256_cvtepi32_ps(_mm256_and_si256(bytes, mask));
                    let high_values = _mm256_cvtepi32_ps(_mm256_srli_epi32(bytes, 4));
                    low = _mm256_add_ps(
                        low,
                        _mm256_mul_ps(low_values, _mm256_loadu_ps(activations.add(chunk * 8))),
                    );
                    high = _mm256_add_ps(
                        high,
                        _mm256_mul_ps(
                            high_values,
                            _mm256_loadu_ps(activations.add(32 + chunk * 8)),
                        ),
                    );
                }
                let (scale0, min0) = q4k_scale_min(scales, 2 * j);
                let (scale1, min1) = q4k_scale_min(scales, 2 * j + 1);
                let base = b * 16 + 4 * j;
                total += d * f32::from(scale0) * hsum(low)
                    - dmin * f32::from(min0) * (sums[base] + sums[base + 1]);
                total += d * f32::from(scale1) * hsum(high)
                    - dmin * f32::from(min1) * (sums[base + 2] + sums[base + 3]);
            }
        }
        total
    }
}

/// Q6_K row dot, eight values per lane group. Mirrors [`crate::quant::q6k_dot`].
///
/// A Q6_K block is 210 bytes: 128 bytes of low nibbles, 64 bytes of the two high bits, 16 signed
/// scales, and one `f16` super-scale. Each half of 128 activation values is produced by walking 32
/// byte positions four times, so each lane group of eight fills four separate 32-value runs.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 210-byte blocks, `x` exactly 256 floats per
/// block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn q6k_dot(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: the caller guarantees whole 210-byte blocks and 256 floats per block, and AVX2. Every
    // load below stays inside one block: `ql` and `ql + 32` read at most bytes 64 and 96 of the
    // 128-byte nibble area, `qh` reads at most bytes 32 of its 64, and the activation loads cover
    // the 128 floats of each half exactly once.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 210;
        let low_nibbles = _mm256_set1_epi32(0x0f);
        let two_bits = _mm256_set1_epi32(0x03);
        let bias = _mm256_set1_epi32(32);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 210);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            // One accumulator for all 256 values of the block: a horizontal sum costs more than the
            // eight values it would reduce, so the scale is applied per lane group and `hsum` runs
            // once per block instead of once per group.
            let mut accumulator = _mm256_setzero_ps();
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = block.add(128 + half * 32);
                let x_half = x.as_ptr().add(b * 256 + half * 128);
                for group in 0..4 {
                    let at = group * 8;
                    // The scale index flips every 16 positions, which is every two groups.
                    let is = at / 16;
                    let levels =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(at) as *const __m128i));
                    let levels_high =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(at + 32) as *const __m128i));
                    let highs = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh.add(at) as *const __m128i));
                    let first = _mm256_or_si256(
                        _mm256_and_si256(levels, low_nibbles),
                        _mm256_slli_epi32(_mm256_and_si256(highs, two_bits), 4),
                    );
                    let second = _mm256_or_si256(
                        _mm256_and_si256(levels_high, low_nibbles),
                        _mm256_slli_epi32(
                            _mm256_and_si256(_mm256_srli_epi32(highs, 2), two_bits),
                            4,
                        ),
                    );
                    let third = _mm256_or_si256(
                        _mm256_srli_epi32(levels, 4),
                        _mm256_slli_epi32(
                            _mm256_and_si256(_mm256_srli_epi32(highs, 4), two_bits),
                            4,
                        ),
                    );
                    let fourth = _mm256_or_si256(
                        _mm256_srli_epi32(levels_high, 4),
                        _mm256_slli_epi32(
                            _mm256_and_si256(_mm256_srli_epi32(highs, 6), two_bits),
                            4,
                        ),
                    );
                    let scale =
                        |index: usize| _mm256_set1_ps(f32::from(scales[half * 8 + index] as i8));
                    let weighted = |values: __m256i, offset: usize, scale: __m256| {
                        _mm256_mul_ps(
                            _mm256_mul_ps(
                                _mm256_cvtepi32_ps(_mm256_sub_epi32(values, bias)),
                                _mm256_loadu_ps(x_half.add(at + offset)),
                            ),
                            scale,
                        )
                    };
                    accumulator = _mm256_add_ps(accumulator, weighted(first, 0, scale(is)));
                    accumulator = _mm256_add_ps(accumulator, weighted(second, 32, scale(is + 2)));
                    accumulator = _mm256_add_ps(accumulator, weighted(third, 64, scale(is + 4)));
                    accumulator = _mm256_add_ps(accumulator, weighted(fourth, 96, scale(is + 6)));
                }
            }
            total += d * hsum(accumulator);
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

    #[test]
    fn avx2_q6k_matches_scalar() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x0bad_c0de_dead_beef);
        for trial in 0..24 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let sums = crate::quant::x_sums(&x);
            let mut row = Vec::with_capacity(blocks * 210);
            for _ in 0..blocks {
                // Block layout: 128 bytes of low nibbles, 64 bytes of high bits, 16 signed scales,
                // then the `f16` super-scale at offset 208.
                for _ in 0..192 {
                    row.push(rng.byte());
                }
                for _ in 0..16 {
                    row.push(rng.byte());
                }
                row.extend_from_slice(&rng.half(0.001, 0.06));
            }
            let expected = crate::quant::q6k_dot(&row, &x, &sums);
            // SAFETY: `row` is whole 210-byte blocks, `x` is sized for them, and the test returned
            // early unless `available()` saw AVX2 on this CPU.
            let actual = unsafe { q6k_dot(&row, &x) };
            assert!(
                close(actual, expected),
                "trial {trial}: avx2 {actual} vs scalar {expected}"
            );
        }
    }
}
