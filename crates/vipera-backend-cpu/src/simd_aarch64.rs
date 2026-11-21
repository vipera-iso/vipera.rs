//! NEON kernels for aarch64.
//!
//! NEON is mandatory on AArch64, so these paths run without a runtime feature check. The float
//! kernels below use 128-bit SIMD; the f16-group kernels fall through to the scalar reference
//! because widening half floats per lane keeps this correct on cores without the optional FP16
//! vector extension and because their structure is dominated by the reduction rather than the
//! conversion. The scalar module remains the source of truth for their semantics.
//!
//! Like the x86 module this is the sanctioned home for `unsafe`: `std::arch` intrinsics require
//! it, and each kernel stays inside the lengths its caller established.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub(crate) use crate::scalar::{
    GROUP_ROWS, axpy_f16_f32, axpy_f16_f32_heads, axpy_f16_f32_heads_exact,
    axpy_f16_f32_heads_exact_block, axpy_f16_group, dot_f16_f32, dot_f16_f32_heads,
    dot_f16_f32_heads_exact, dot_f16_group, exp_sum_shifted, silu_mul,
};

/// Whether an in-place `f16` row and a row decoded to `f32` produce the same bits.
///
/// They do not here yet. [`dot_f32`] adds into four accumulators and [`axpy_f32`] fuses its
/// multiply and add, while the `f16` kernels above are the scalar reference: one running sum, one
/// unfused multiply-add. The in-place path decides which of the two a row takes, so turning it on
/// would make two attention entry points disagree in the last bits instead of agreeing, which is
/// what the bit-identity test in `model` holds them to. It stays off until the `f16` kernels get
/// the same shape as the `f32` ones.
pub(crate) const IN_PLACE_KV_BIT_EXACT: bool = false;

/// Widen little-endian `f16` values into `f32`, stopping at whichever slice is shorter.
pub(crate) fn half_to_float(source: &[u8], values: &mut [f32]) {
    crate::scalar::half_to_float(source, values);
}

/// `sum(a[i] * b[i])` over the shorter length.
#[cfg(target_arch = "aarch64")]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());
    if length == 0 {
        return 0.0;
    }
    // SAFETY: NEON is baseline on aarch64; every load is inside `length`.
    unsafe {
        let (mut acc0, mut acc1, mut acc2, mut acc3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (left, right) = (a.as_ptr(), b.as_ptr());
        let mut at = 0;
        while at + 16 <= length {
            acc0 = vfmaq_f32(acc0, vld1q_f32(left.add(at)), vld1q_f32(right.add(at)));
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(left.add(at + 4)),
                vld1q_f32(right.add(at + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(left.add(at + 8)),
                vld1q_f32(right.add(at + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(left.add(at + 12)),
                vld1q_f32(right.add(at + 12)),
            );
            at += 16;
        }
        while at + 4 <= length {
            acc0 = vfmaq_f32(acc0, vld1q_f32(left.add(at)), vld1q_f32(right.add(at)));
            at += 4;
        }
        let sum = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut total = vaddvq_f32(sum);
        while at < length {
            total += *left.add(at) * *right.add(at);
            at += 1;
        }
        total
    }
}

/// `target[i] += weight * source[i]` over the shorter length.
#[cfg(target_arch = "aarch64")]
pub(crate) fn axpy_f32(target: &mut [f32], weight: f32, source: &[f32]) {
    let length = target.len().min(source.len());
    if length == 0 {
        return;
    }
    // SAFETY: NEON is baseline on aarch64; every access stays inside `length`.
    unsafe {
        let scale = vdupq_n_f32(weight);
        let (dst, src) = (target.as_mut_ptr(), source.as_ptr());
        let mut at = 0;
        while at + 16 <= length {
            let t0 = vld1q_f32(dst.add(at));
            let t1 = vld1q_f32(dst.add(at + 4));
            let t2 = vld1q_f32(dst.add(at + 8));
            let t3 = vld1q_f32(dst.add(at + 12));
            vst1q_f32(dst.add(at), vfmaq_f32(t0, scale, vld1q_f32(src.add(at))));
            vst1q_f32(
                dst.add(at + 4),
                vfmaq_f32(t1, scale, vld1q_f32(src.add(at + 4))),
            );
            vst1q_f32(
                dst.add(at + 8),
                vfmaq_f32(t2, scale, vld1q_f32(src.add(at + 8))),
            );
            vst1q_f32(
                dst.add(at + 12),
                vfmaq_f32(t3, scale, vld1q_f32(src.add(at + 12))),
            );
            at += 16;
        }
        while at + 4 <= length {
            let t = vld1q_f32(dst.add(at));
            vst1q_f32(dst.add(at), vfmaq_f32(t, scale, vld1q_f32(src.add(at))));
            at += 4;
        }
        while at < length {
            *dst.add(at) += weight * *src.add(at);
            at += 1;
        }
    }
}
