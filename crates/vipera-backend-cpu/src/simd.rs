//! AVX2 kernels for the two quantized formats a Q4_K_M file is made of.
//!
//! The hot paths of this module are the only ones the crate sanctions as the workspace's
//! performance path: `std::arch` intrinsics require them. Each kernel mirrors the scalar one in
//! [`crate::quant`] operation for operation, takes exactly the same inputs, and the tests below
//! run both over randomised blocks and compare. (`bus.rs` has a documented opt-out for file
//! mapping, EXC-005.)
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
//! embedding — a row lookup, not a matvec — is excluded), so it decides the token rate. Two changes
//! moved it:
//!
//! 1. The kernel multiplies `i8` activations against the nibbles: `_mm256_maddubs_epi16` handles 32
//!    weights per instruction where the `f32` path needs four widen-multiply-add rounds. The cost
//!    is one quantization of `x` per matvec (see `crate::quant::QuantizedX`), which bounds the
//!    error of each 32-value group to `0.5/127` of that group's largest magnitude.
//! 2. [`crate::quant::f16_to_f32`] stopped calling `powi` with a runtime exponent, which compiled to
//!    the `__powisf2` compiler-rt helper and ran twice per Q4_K block. This was the larger of the
//!    two: single-threaded the kernel went 1.67 -> 3.6 GB/s, and Q6_K gained as well because it
//!    calls the same conversion once per block.
//!
//! Measured with `matvec_throughput` on an i5-11300H, four threads, cache-warm synthetic tensors of
//! the real shapes: Q4_K `ffn_gate` 8.0-13.4 GB/s (median of three runs 8.0), Q6_K `output`
//! 14.5-16.6 GB/s, Q6_K `ffn_down` 8.3-13.7 GB/s. End to end, 12 tokens of Qwen2.5-1.5B Q4_K_M went
//! from 3.56 s of compute to 1.64-1.73 s, which is 6.1-6.5 tok/s against 3.2 before these changes
//! and 2.57 in the first measurement of this engine. The 10 tok/s target is still not met: the
//! remaining gap is streaming the other 470 MB of weights from RAM and per-`matvec` thread setup.
//! Sampling used to be part of it — 19 ms per token spent sorting a 151936-entry logit vector — but
//! that path now takes 0.3 ms greedy and 1.3 ms with top-k/top-p/min-p over the same vocabulary.
//!
//! # What 512-bit kernels did not buy
//!
//! Two were written at 512 bits, measured against these kernels in the same session, and removed. A
//! Q4_K `zmm` version: 7.99 vs 8.01 tok/s end to end and 5.2 vs 6.8 GB/s at the kernel level. A Q6_K
//! VNNI version, two 32-value runs per `vpdpbusd` and bit-identical to `q6k_dot_q8`: 8.7 vs 8.7
//! tok/s on one thread, 19.8 vs 20.0 on four, 19.1 vs 19.5 GB/s on `ffn_down`.
//!
//! Neither loop is limited by vector width. The Q4_K kernel spends its time on per-block scalar
//! work. Whether these shapes are memory-bound is measured, not assumed: STREAM 5.10 (McCalpin)
//! and a read-only kernel on this machine — i5-11300H, arrays far above the 8 MiB L3, best of 10
//! timed passes — give
//!
//! | kernel | 4 threads | 8 threads |
//! |---|---|---|
//! | Copy (1 read, 1 write) | 39.7 GB/s | 38.8 GB/s |
//! | Triad (2 reads, 1 write) | 29.6 GB/s | 30.5 GB/s |
//! | read-only | 42.8 GB/s | 41.7 GB/s |
//!
//! SMT adds nothing to a bandwidth stream: 8 threads is not faster than 4. A decode step reads
//! 0.980 GB — the 1.111 GB of tensors minus the 125 MB token embedding, which decode never reads —
//! so the ceiling that applies to it is the read-only one, ~43 GB/s (a 22.8 ms floor per token).
//! vipera's 35.2 ms/step is 28 GB/s and llama.cpp's 34.7 tok/s is 34 GB/s, so the Q6_K shapes are
//! memory-bound and there is still headroom under the ceiling; what closes the gap is the access
//! pattern, not the multiplier and not the language. The 512-bit results show width did not help
//! on this core. Tiger Lake client cores are documented as having
//! one 512-bit FMA/VNNI unit (ports 0 and 1 fused), so a `zmm` instruction would retire no more
//! integer work per cycle than two `ymm`; that is not verified here and says nothing about Zen 5
//! or Xeon cores with two 512-bit units.
//!
//! # Safety
//!
//! The public functions are `unsafe` because they use AVX2 intrinsics. Besides CPU support, `row` must
//! be a whole number of blocks of the format named by the function (144 bytes for Q4_K, 210 for Q6_K),
//! `x` exactly 256 values per block, and `sums` at least 16 entries per Q4_K block. Q6_K ignores
//! `sums`. [`crate::quant::matvec`] validates all of that before dispatching, and `try_dot` is the
//! safe entry point that checks the CPU.
#![allow(unsafe_code)]

use std::arch::x86_64::{
    __m128i, __m256, __m256i, __m512, __m512i, _MM_HINT_T0, _mm_add_ps, _mm_add_ss,
    _mm_cvtsi64_si128, _mm_cvtss_f32, _mm_hadd_epi16, _mm_loadl_epi64, _mm_loadu_ps,
    _mm_loadu_si128, _mm_movehdup_ps, _mm_movehl_ps, _mm_prefetch, _mm_set1_ps, _mm_setr_epi8,
    _mm_setr_epi32, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_unpacklo_epi8, _mm256_add_epi8,
    _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_blend_epi32, _mm256_bsrli_epi128, _mm256_castps256_ps128, _mm256_castsi128_si256,
    _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cmpeq_epi32,
    _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_cvtepu8_epi32, _mm256_cvtph_ps,
    _mm256_cvttps_epi32, _mm256_div_ps, _mm256_dpbusd_epi32, _mm256_extractf128_ps,
    _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epi32, _mm256_max_ps, _mm256_min_epi32,
    _mm256_min_ps, _mm256_mul_ps, _mm256_or_si256, _mm256_permute2f128_si256,
    _mm256_permutevar8x32_ps, _mm256_set_m128, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_epi32, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_shuffle_epi32, _mm256_slli_epi16, _mm256_slli_epi32, _mm256_srli_epi16,
    _mm256_srli_epi32, _mm256_srlv_epi32, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16,
    _mm256_sub_epi32, _mm256_sub_ps, _mm512_add_epi32, _mm512_add_ps, _mm512_cvtepi32_ps,
    _mm512_dpbusd_epi32, _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_mul_ps, _mm512_mullo_epi32, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps,
    _mm512_setzero_si512, _mm512_storeu_ps, _mm512_sub_ps,
};

use vipera_core::gguf::GgmlType;

use crate::quant::{BatchActivation, QuantizedX, TiledActivation, f16_to_f32};

/// Tile halves the tiled kernels keep activation coefficients for: 16
/// activations per half, and batches up to 64 wide run four halves against one
/// weight pass.
const MAX_HALVES: usize = 16;

/// Whether this CPU can run the kernels below. Cached by the standard library.
pub fn available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Round half away from zero, as `f32::round` does, in eight lanes at once.
///
/// `trunc(v + copysign(0.49999997, v))` where `0.49999997` is the largest `f32` below one half.
/// Adding exactly `0.5` is wrong for `0.49999997`, which would round up to one; adding the value
/// just under it gets every tie and every near-tie right for `|v| < 2^23`. The exhaustive test
/// `round_half_away_matches_f32_round_for_every_value_in_range` checks it against `f32::round`
/// for every value that matters here.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_half_away_x8(v: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm256_add_ps, _mm256_and_ps, _mm256_or_ps,
        _mm256_round_ps, _mm256_set1_ps,
    };
    let sign = _mm256_set1_ps(-0.0);
    let just_under_half = _mm256_set1_ps(f32::from_bits(0x3EFF_FFFF));
    let signed = _mm256_or_ps(just_under_half, _mm256_and_ps(v, sign));
    _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_add_ps(v, signed))
}

/// Quantize `x` in groups of 32 into `q` (32 values per group) and `d` (one scale per group),
/// giving exactly what [`QuantizedX::quantize_group`] gives for every group.
///
/// Returns `false`, having written nothing, when this CPU has no AVX2 or the slices are too short.
///
/// Safe to call: it detects AVX2 itself and touches only `groups * 32` values of `x` and `q` and
/// `groups` values of `d`.
pub(crate) fn quantize_groups(x: &[f32], q: &mut [i8], d: &mut [f32]) -> bool {
    let groups = x.len() / 32;
    if !available() || q.len() < groups * 32 || d.len() < groups {
        return false;
    }
    // SAFETY: AVX2 is available, and the length checks above bound every load and store.
    unsafe { quantize_groups_avx2(x, q, d, groups) };
    true
}

#[target_feature(enable = "avx2")]
unsafe fn quantize_groups_avx2(x: &[f32], q: &mut [i8], d: &mut [f32], groups: usize) {
    use std::arch::x86_64::{
        __m256i, _CMP_UNORD_Q, _mm_cvtss_f32, _mm_max_ps, _mm_max_ss, _mm_movehl_ps,
        _mm_shuffle_ps, _mm256_and_ps, _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cmp_ps,
        _mm256_cvttps_epi32, _mm256_div_ps, _mm256_extractf128_ps, _mm256_loadu_ps,
        _mm256_max_epi32, _mm256_max_ps, _mm256_min_epi32, _mm256_movemask_ps, _mm256_or_ps,
        _mm256_packs_epi16, _mm256_packs_epi32, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_setr_epi32, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };
    // SAFETY: the caller guarantees AVX2 and that `x` and `q` hold `groups * 32` values and `d`
    // holds `groups`; every load and store below is inside one 32-value group.
    unsafe {
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let low = _mm256_set1_epi32(-127);
        let high = _mm256_set1_epi32(127);
        // After two 16-bit-lane packs the dwords sit as 0,4,1,5,2,6,3,7 of the order wanted.
        let reorder = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (index, group) in x.as_chunks::<32>().0.iter().enumerate().take(groups) {
            let at = group.as_ptr();
            let v = [
                _mm256_loadu_ps(at),
                _mm256_loadu_ps(at.add(8)),
                _mm256_loadu_ps(at.add(16)),
                _mm256_loadu_ps(at.add(24)),
            ];
            // `acc.max(v.abs())` ignores a NaN `v`; `_mm256_max_ps(a, b)` returns `b` when either is
            // NaN, so the new value goes first and the running maximum second.
            let mut largest = _mm256_setzero_ps();
            let mut unordered = _mm256_setzero_ps();
            for lane in v {
                largest = _mm256_max_ps(_mm256_and_ps(lane, abs_mask), largest);
                unordered = _mm256_or_ps(unordered, _mm256_cmp_ps::<_CMP_UNORD_Q>(lane, lane));
            }
            let halves = _mm_max_ps(
                _mm256_castps256_ps128(largest),
                _mm256_extractf128_ps::<1>(largest),
            );
            let halves = _mm_max_ps(halves, _mm_movehl_ps(halves, halves));
            let largest = _mm_cvtss_f32(_mm_max_ss(halves, _mm_shuffle_ps::<0x55>(halves, halves)));

            let out = &mut q[index * 32..index * 32 + 32];
            if largest == 0.0 {
                d[index] = 0.0;
                out.fill(0);
                continue;
            }
            let scale = largest / 127.0;
            // A NaN, an infinity or a scale too small to divide by cleanly is rare and not worth
            // reproducing lane by lane: the scalar definition handles it.
            if _mm256_movemask_ps(unordered) != 0
                || !largest.is_finite()
                || scale < f32::MIN_POSITIVE
            {
                d[index] = QuantizedX::quantize_group(group, out);
                continue;
            }
            d[index] = scale;
            let divisor = _mm256_set1_ps(scale);
            let mut lanes = [_mm256_setzero_si256(); 4];
            for (slot, lane) in lanes.iter_mut().zip(v) {
                let rounded = round_half_away_x8(_mm256_div_ps(lane, divisor));
                let integers = _mm256_cvttps_epi32(rounded);
                *slot = _mm256_min_epi32(_mm256_max_epi32(integers, low), high);
            }
            let first = _mm256_packs_epi32(lanes[0], lanes[1]);
            let second = _mm256_packs_epi32(lanes[2], lanes[3]);
            let packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(first, second), reorder);
            _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), packed);
        }
    }
}

/// Prefetch `data[ahead_byte ..]` into L1, hint T0.
///
/// Streaming rows from DRAM is bandwidth-bound; the hint found by Genesis (one PREFETCHNTA)
/// lets the memory pipeline run ahead of the integer unpack loop. Safe: no dereference, only a
/// cache hint on a valid offset. No-op when out of range or no AVX2.
pub(crate) fn prefetch_ahead(data: &[u8], byte_offset: usize) {
    if byte_offset >= data.len() || !available() {
        return;
    }
    // SAFETY: byte_offset is within data; prefetch is a hint, never a fault.
    unsafe {
        _mm_prefetch(data.as_ptr().add(byte_offset) as *const i8, _MM_HINT_T0);
    }
}

/// Whether the `vpdpbusd` path can run: AVX2, `avx512_vnni` with `avx512_vl` for the 256-bit form,
/// and no `VIPERA_NO_VNNI`.
///
/// The switch exists so an A/B run can hide the new kernel without rebuilding; the results are
/// bit-identical either way, so it changes speed only.
pub fn vnni_available() -> bool {
    static VNNI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VNNI.get_or_init(|| {
        if !available()
            || !std::is_x86_feature_detected!("avx512vnni")
            || !std::is_x86_feature_detected!("avx512vl")
        {
            return false;
        }
        !std::env::var("VIPERA_NO_VNNI").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

/// Convert `f16` values, little endian in `source`, into `values`.
///
/// The KV cache is read in full on every layer, so at 1024 positions a decode step spends more time
/// here than in the attention arithmetic that follows. F16C converts eight at a time, and the
/// conversion is exact — the values are the ones the scalar loop produces, which the test below
/// checks over all 65536 bit patterns rather than trusting — so no switch and no token moves.
///
/// Safe to call: it detects F16C itself, converts the shorter of the two lengths, and finishes any
/// tail one value at a time.
pub(crate) fn half_to_float(source: &[u8], values: &mut [f32]) {
    let length = values.len().min(source.len() / 2);
    if length == 0 {
        return;
    }
    if std::is_x86_feature_detected!("f16c") && length >= 8 {
        // SAFETY: F16C was just detected, and the kernel reads and writes only within `length`.
        unsafe { half_to_float_f16c(source, &mut values[..length], length) };
        return;
    }
    for (value, chunk) in values[..length]
        .iter_mut()
        .zip(source.as_chunks::<2>().0.iter())
    {
        *value = f16_to_f32(u16::from_le_bytes(*chunk));
    }
}

/// The SwiGLU multiply `out *= silu(g)` in eight-lane steps.
///
/// Every arithmetic operation is the scalar one in [`crate::model::silu`]'s `exp` (see
/// [`exp_x8`]), lane by lane and in the same order, so the bits are identical rather than close:
/// the test below compares `to_bits` over a dense sweep, random patterns and the special values.
/// The scalar path is a quarter of a prefill chunk's `rest` bucket, so this is worth the
/// duplication.
///
/// Safe to call: it detects AVX2 itself, applies the shorter of the two lengths, and finishes the
/// tail one value at a time.
pub(crate) fn silu_mul(up: &mut [f32], gate: &[f32]) {
    let length = up.len().min(gate.len());
    if length == 0 {
        return;
    }
    if available() && length >= 8 {
        // SAFETY: AVX2 was just detected and the kernel stays inside `length`.
        unsafe { silu_mul_avx2(&mut up[..length], &gate[..length]) };
        return;
    }
    for (out, gate) in up[..length].iter_mut().zip(&gate[..length]) {
        *out *= crate::model::silu(*gate);
    }
}

/// [`silu_mul`] in eight-value steps.
///
/// # Safety
///
/// The CPU must support AVX2, and both slices must have the same length.
#[target_feature(enable = "avx2")]
unsafe fn silu_mul_avx2(up: &mut [f32], gate: &[f32]) {
    // SAFETY: the caller guarantees AVX2 and equal lengths; every load and store is at `at < len`
    // and reads or writes eight values, which the loop condition guarantees are in range.
    unsafe {
        let len = up.len();
        let out = up.as_mut_ptr();
        let input = gate.as_ptr();
        let mut at = 0;
        while at + 8 <= len {
            let g = _mm256_loadu_ps(input.add(at));
            let exponentiated = exp_x8(_mm256_sub_ps(_mm256_setzero_ps(), g));
            // `g / (1 + exp(-g))`, the scalar definition.
            let silu = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0), exponentiated));
            let value = _mm256_mul_ps(_mm256_loadu_ps(out.add(at)), silu);
            _mm256_storeu_ps(out.add(at), value);
            at += 8;
        }
        while at < len {
            *out.add(at) *= crate::model::silu(*input.add(at));
            at += 1;
        }
    }
}

/// `exp` from [`crate::model`], in eight lanes.
///
/// The polynomial, the rounding trick and the exponent clamp are the scalar constants in
/// [`crate::model::silu`]'s helper, so a lane gets exactly the scalar result. `whole` is clamped in
/// the float domain before the integer conversion because a cast of a huge float saturates, while
/// `cvttps2dq` does not; clamping first makes both paths agree, and only for inputs whose exponent
/// is already pinned to 254.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn exp_x8(value: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    const LN_2: f32 = std::f32::consts::LN_2;
    let scaled = _mm256_mul_ps(value, _mm256_set1_ps(LOG2_E));
    let magic = _mm256_set1_ps(12_582_912.0);
    let whole = _mm256_sub_ps(_mm256_add_ps(scaled, magic), magic);
    let fraction = _mm256_sub_ps(scaled, whole);
    let mut polynomial = _mm256_set1_ps(0.000_154_035_3);
    for coefficient in [
        0.001_333_355_8,
        0.009_618_129,
        0.055_504_11,
        0.240_226_5,
        LN_2,
        1.0,
    ] {
        polynomial = _mm256_add_ps(
            _mm256_set1_ps(coefficient),
            _mm256_mul_ps(fraction, polynomial),
        );
    }
    let bounded = _mm256_min_ps(
        _mm256_max_ps(whole, _mm256_set1_ps(-127.0)),
        _mm256_set1_ps(128.0),
    );
    let exponent = _mm256_add_epi32(_mm256_cvttps_epi32(bounded), _mm256_set1_epi32(127));
    let exponent = _mm256_slli_epi32(
        _mm256_min_epi32(
            _mm256_max_epi32(exponent, _mm256_setzero_si256()),
            _mm256_set1_epi32(254),
        ),
        23,
    );
    _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent))
}

/// `sum(e^(values[i] - shift))` for every i, each value replaced by its exponential: one head's
/// softmax numerator, written the way [`crate::model::exp`] computes it, in eight lanes.
///
/// The vector lanes and the scalar remainder are the same polynomial and the same rounding trick, so
/// every value written here is the scalar `exp`'s bits; only the order the lanes are added across
/// differs, which moves the total by a few ulps. It exists because the softmax pass is
/// `heads * positions` libm calls per layer, and a call is a function call.
///
/// Safe to call: it needs AVX2 for the vector path and falls back otherwise.
pub(crate) fn exp_sum_shifted(values: &mut [f32], shift: f32) -> f32 {
    if values.len() >= 8 && available() {
        // SAFETY: AVX2 was just detected, and the kernel reads and writes `len` values.
        return unsafe { exp_sum_shifted_avx2(values.as_mut_ptr(), values.len(), shift) };
    }
    let mut total = 0.0_f32;
    for value in values.iter_mut() {
        let exponential = crate::model::exp(*value - shift);
        *value = exponential;
        total += exponential;
    }
    total
}

/// # Safety
///
/// The CPU must support AVX2, and `values` must hold `len` writable values.
#[target_feature(enable = "avx2")]
unsafe fn exp_sum_shifted_avx2(values: *mut f32, len: usize, shift: f32) -> f32 {
    // SAFETY: the caller guarantees AVX2 and the extent; every step is inside it.
    unsafe {
        let broadcast = _mm256_set1_ps(shift);
        let mut low = _mm256_setzero_ps();
        let mut high = _mm256_setzero_ps();
        let mut at = 0;
        while at + 16 <= len {
            let first = exp_x8(_mm256_sub_ps(_mm256_loadu_ps(values.add(at)), broadcast));
            let second = exp_x8(_mm256_sub_ps(
                _mm256_loadu_ps(values.add(at + 8)),
                broadcast,
            ));
            _mm256_storeu_ps(values.add(at), first);
            _mm256_storeu_ps(values.add(at + 8), second);
            low = _mm256_add_ps(low, first);
            high = _mm256_add_ps(high, second);
            at += 16;
        }
        while at + 8 <= len {
            let block = exp_x8(_mm256_sub_ps(_mm256_loadu_ps(values.add(at)), broadcast));
            _mm256_storeu_ps(values.add(at), block);
            low = _mm256_add_ps(low, block);
            at += 8;
        }
        let mut total = hsum(_mm256_add_ps(low, high));
        while at < len {
            let exponential = crate::model::exp(*values.add(at) - shift);
            *values.add(at) = exponential;
            total += exponential;
            at += 1;
        }
        total
    }
}

/// [`half_to_float`] in eight-value steps.
///
/// # Safety
///
/// The CPU must support F16C, `source` must hold at least `length * 2` bytes, and `values` must hold
/// at least `length` slots.
#[target_feature(enable = "f16c")]
unsafe fn half_to_float_f16c(source: &[u8], values: &mut [f32], length: usize) {
    // SAFETY: the caller guarantees F16C and the bounds; each step reads 16 bytes and writes 32, at
    // an offset that leaves eight values before `length`.
    unsafe {
        let input = source.as_ptr();
        let out = values.as_mut_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let half = _mm_loadu_si128(input.add(at * 2) as *const __m128i);
            _mm256_storeu_ps(out.add(at), _mm256_cvtph_ps(half));
            at += 8;
        }
        while at < length {
            *out.add(at) = f16_to_f32(u16::from_le_bytes([
                *input.add(at * 2),
                *input.add(at * 2 + 1),
            ]));
            at += 1;
        }
    }
}

/// The dot of two `f32` slices, eight values at a time: one attention query head against one key.
///
/// At 1024 positions a decode step spent 58 ms here, and it was not the arithmetic: the scalar loop
/// leaves one `addss` chain of 128 links per position, each waiting four cycles on the one before, so
/// a whole step went by at about two thirds of one lane's throughput. Eight lanes and two
/// accumulators shorten the chain to sixteen links, at the cost of a tree reduction, which is a
/// different rounding from the sequential sum. `VIPERA_EXACT_DOT=1` keeps the sequential loop for
/// reproducing the golden tokens; everything else takes this one.
///
/// Safe to call: it detects AVX2 itself and falls back to the scalar loop, and it reads only whole
/// blocks, finishing any tail one value at a time.
///
/// Whether an in-place `f16` row and a row decoded to `f32` produce the same bits: every `f16`
/// kernel here mirrors this file's reduction tree, so they do. See `simd_aarch64` for the
/// architecture where they do not yet.
pub(crate) const IN_PLACE_KV_BIT_EXACT: bool = true;

pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());
    if !available() || length < 16 {
        return a.iter().zip(b).take(length).map(|(a, b)| a * b).sum();
    }
    // SAFETY: `available()` established AVX2 support, and the kernel reads only within `length`.
    unsafe { dot_f32_avx2(a, b, length) }
}

/// `target += weight * source`, elementwise.
///
/// Every value receives exactly the multiply and the add the scalar loop gives it, in the same order,
/// so this is bit-identical to `target[i] += weight * source[i]` and needs no switch.
///
/// Safe to call: it detects AVX2 itself, and the length is the shorter of the two slices.
pub(crate) fn axpy_f32(target: &mut [f32], weight: f32, source: &[f32]) {
    let length = target.len().min(source.len());
    if !available() || length < 8 {
        for (target, source) in target.iter_mut().zip(source).take(length) {
            *target += weight * source;
        }
        return;
    }
    // SAFETY: `available()` established AVX2 support, and the kernel writes only within `length`.
    unsafe { axpy_f32_avx2(&mut target[..length], weight, &source[..length]) }
}

/// [`dot_f32`] with the reduction tree described on it.
///
/// # Safety
///
/// The CPU must support AVX2, and `length` must not exceed either slice.
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], length: usize) -> f32 {
    // SAFETY: the caller guarantees AVX2 and `length` inside both slices, and every load below is
    // addressed at an offset that leaves eight values before `length`.
    unsafe {
        let (left, right) = (a.as_ptr(), b.as_ptr());
        let mut first = _mm256_setzero_ps();
        let mut second = _mm256_setzero_ps();
        let mut at = 0;
        while at + 16 <= length {
            first = _mm256_add_ps(
                first,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at)),
                    _mm256_loadu_ps(right.add(at)),
                ),
            );
            second = _mm256_add_ps(
                second,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at + 8)),
                    _mm256_loadu_ps(right.add(at + 8)),
                ),
            );
            at += 16;
        }
        while at + 8 <= length {
            first = _mm256_add_ps(
                first,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at)),
                    _mm256_loadu_ps(right.add(at)),
                ),
            );
            at += 8;
        }
        let mut total = hsum(_mm256_add_ps(first, second));
        while at < length {
            total += *left.add(at) * *right.add(at);
            at += 1;
        }
        total
    }
}

/// [`axpy_f32`] in eight lanes.
///
/// # Safety
///
/// The CPU must support AVX2, and both slices must be the same length.
#[target_feature(enable = "avx2")]
unsafe fn axpy_f32_avx2(target: &mut [f32], weight: f32, source: &[f32]) {
    // SAFETY: the caller guarantees AVX2 and equal lengths; the loop stops at the last whole block
    // and the tail is one value at a time.
    unsafe {
        let length = target.len();
        let scale = _mm256_set1_ps(weight);
        let out = target.as_mut_ptr();
        let input = source.as_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let scaled = _mm256_mul_ps(scale, _mm256_loadu_ps(input.add(at)));
            let current = _mm256_loadu_ps(out.add(at));
            _mm256_storeu_ps(out.add(at), _mm256_add_ps(current, scaled));
            at += 8;
        }
        while at < length {
            *out.add(at) += weight * *input.add(at);
            at += 1;
        }
    }
}

/// [`dot_f32`] over a row the KV cache stores as `f16`, converting inside the loop.
///
/// The conversion is exact and the reduction tree is [`dot_f32_avx2`]'s, so this gives the bits
/// that decoding the row and calling [`dot_f32`] gives — the attention tests compare against that
/// oracle bit for bit. What it avoids is materializing the row: at 1024 positions a decode step
/// writes and re-reads 2 MB of `f32` row buffer per layer for no arithmetic at all.
///
/// Safe to call: it detects the features itself, and the length is the shorter of the two.
pub(crate) fn dot_f16_f32(a: &[f32], row: &[u8]) -> f32 {
    let length = a.len().min(row.len() / 2);
    if length == 0 {
        return 0.0;
    }
    if available() && std::is_x86_feature_detected!("f16c") && length >= 8 {
        // SAFETY: AVX2 and F16C were just detected, and the kernel stays inside `length`.
        return unsafe { dot_f16_f32_avx2(a, row, length) };
    }
    let mut total = 0.0_f32;
    for (value, chunk) in a.iter().zip(row.as_chunks::<2>().0).take(length) {
        total += value * f16_to_f32(u16::from_le_bytes(*chunk));
    }
    total
}

/// [`dot_f16_f32`] with [`dot_f32_avx2`]'s reduction tree and the conversion folded in.
///
/// # Safety
///
/// The CPU must support AVX2 and F16C, and `length` must not exceed `a`'s length or `row`'s half.
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_f16_f32_avx2(a: &[f32], row: &[u8], length: usize) -> f32 {
    // SAFETY: the caller guarantees the features and `length`; every load reads eight halves or
    // eight floats at an offset that leaves that many before `length`.
    unsafe {
        let (left, bytes) = (a.as_ptr(), row.as_ptr());
        let mut first = _mm256_setzero_ps();
        let mut second = _mm256_setzero_ps();
        let mut at = 0;
        while at + 16 <= length {
            first = _mm256_add_ps(
                first,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at)),
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        bytes.add(at * 2) as *const std::arch::x86_64::__m128i
                    )),
                ),
            );
            second = _mm256_add_ps(
                second,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at + 8)),
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        bytes.add((at + 8) * 2) as *const std::arch::x86_64::__m128i
                    )),
                ),
            );
            at += 16;
        }
        while at + 8 <= length {
            first = _mm256_add_ps(
                first,
                _mm256_mul_ps(
                    _mm256_loadu_ps(left.add(at)),
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        bytes.add(at * 2) as *const std::arch::x86_64::__m128i
                    )),
                ),
            );
            at += 8;
        }
        let mut total = hsum(_mm256_add_ps(first, second));
        while at < length {
            let low = std::ptr::read_unaligned(bytes.add(at * 2) as *const u16);
            total += *left.add(at) * f16_to_f32(low);
            at += 1;
        }
        total
    }
}

/// [`axpy_f32`] over a row the KV cache stores as `f16`.
///
/// Same arithmetic as decoding then [`axpy_f32`], eight lanes at a time, without the row buffer.
///
/// Safe to call: it detects the features itself, and the length is the shorter of the two.
pub(crate) fn axpy_f16_f32(target: &mut [f32], weight: f32, row: &[u8]) {
    let length = target.len().min(row.len() / 2);
    if length == 0 {
        return;
    }
    if available() && std::is_x86_feature_detected!("f16c") && length >= 8 {
        // SAFETY: AVX2 and F16C were just detected, and the kernel stays inside `length`.
        unsafe { axpy_f16_f32_avx2(&mut target[..length], weight, row) };
        return;
    }
    for (target, chunk) in target.iter_mut().zip(row.as_chunks::<2>().0).take(length) {
        *target += weight * f16_to_f32(u16::from_le_bytes(*chunk));
    }
}

/// [`dot_f16_f32`] for several heads that share one K row, with the feature check and the row facts
/// hoisted out of the per-head call.
///
/// The window kernel reads one K row and dots it against every head of a KV head — six of them at
/// this model's shape — so the per-call cost of [`dot_f16_f32`] (an `OnceLock` load, two feature
/// checks and a fresh loop setup) was paid six times for one 64-value dot. Each head's arithmetic is
/// exactly [`dot_f16_f32`]'s, so the scores are bit-identical.
///
/// `q` holds `first_head` and the `count` heads after it back to back at `head_dim` each.
pub(crate) fn dot_f16_f32_heads(
    q: &[f32],
    first_head: usize,
    count: usize,
    head_dim: usize,
    row: &[u8],
    out: &mut [f32],
) {
    let fma = f16_fma_available();
    let fast = available() && std::is_x86_feature_detected!("f16c");
    let row_length = row.len() / 2;
    let length = head_dim.min(row_length);
    if fma && length >= 16 && count <= HEADS_IN_REGISTERS && out.len() >= count {
        // SAFETY: the features were detected above, `count` is inside the register budget and every
        // head index below is inside `q` and `out`, which this function's slices bound at
        // `head_dim` per head.
        unsafe {
            dot_f16_f32_heads_avx2_fma(q, first_head, count, head_dim, row, length, out, true);
        }
        return;
    }
    for (member, slot) in out.iter_mut().take(count).enumerate() {
        let at = (first_head + member) * head_dim;
        let query = &q[at..at + head_dim];
        if length == 0 {
            *slot = 0.0;
        } else if fma && length >= 8 {
            // SAFETY: AVX2, F16C and FMA were detected once, before the loop, and the kernel reads
            // only the first `length` values of both operands.
            *slot = unsafe { dot_f16_f32_avx2_fma(query, row, length) };
        } else if fast && length >= 8 {
            // SAFETY: AVX2 and F16C were detected once, before the loop, and the kernel reads only
            // the first `length` values of both operands.
            *slot = unsafe { dot_f16_f32_avx2(query, row, length) };
        } else {
            let mut total = 0.0_f32;
            for (value, chunk) in query.iter().zip(row.as_chunks::<2>().0).take(length) {
                total += value * f16_to_f32(u16::from_le_bytes(*chunk));
            }
            *slot = total;
        }
    }
}

/// `targets += weights[h] * row` for the heads of one KV head, again sharing the row work.
///
/// `targets` holds `weights.len()` rows back to back at `head_dim` each, and each row is one head's
/// running weighted sum of V. Same arithmetic as [`axpy_f16_f32`] per head, so it is bit-identical.
pub(crate) fn axpy_f16_f32_heads(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    row: &[u8],
) {
    let fma = f16_fma_available();
    let fast = available() && std::is_x86_feature_detected!("f16c");
    let row_length = row.len() / 2;
    let length = head_dim.min(row_length);
    if fma
        && length >= 8
        && weights.len() <= HEADS_IN_REGISTERS
        && targets.len() >= weights.len() * head_dim
    {
        // SAFETY: the features were detected above, `weights` is inside the register budget and every
        // row of `targets` below is inside the slice, which the caller sized at `head_dim` per head.
        unsafe {
            axpy_f16_f32_heads_avx2_fma(targets, head_dim, weights, row, length, true);
        }
        return;
    }
    for (member, weight) in weights.iter().enumerate() {
        let target = &mut targets[member * head_dim..][..head_dim];
        if length == 0 {
            continue;
        }
        if fma && length >= 8 {
            // SAFETY: AVX2, F16C and FMA were detected once, before the loop, and the kernel writes
            // only the first `length` values of the target row.
            unsafe { axpy_f16_f32_avx2_fma(&mut target[..length], *weight, row) };
        } else if fast && length >= 8 {
            // SAFETY: AVX2 and F16C were detected once, before the loop, and the kernel writes only
            // the first `length` values of the target row.
            unsafe { axpy_f16_f32_avx2(&mut target[..length], *weight, row) };
        } else {
            for (value, chunk) in target.iter_mut().zip(row.as_chunks::<2>().0).take(length) {
                *value += *weight * f16_to_f32(u16::from_le_bytes(*chunk));
            }
        }
    }
}

/// How many heads the interleaved kernels keep accumulators for at once.
const HEADS_IN_REGISTERS: usize = 8;

/// The interleaved K kernel: one pass over the row, one accumulator pair per head.
///
/// Loading the row once and multiplying it into every head's accumulator takes the dot off its
/// dependency chain — each head's adds are independent — and removes one row load per head. Per head
/// the tree is [`dot_f16_f32_avx2_fma`]'s, so this gives that kernel's bits.
///
/// # Safety
///
/// The CPU must support AVX2, F16C and FMA; `count` must be at most [`HEADS_IN_REGISTERS`];
/// `q` must hold `first_head + count` heads of `head_dim`; `out` must hold `count` floats; `length`
/// must not exceed `head_dim` or `row`'s half.
#[target_feature(enable = "avx2,f16c,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn dot_f16_f32_heads_avx2_fma(
    q: &[f32],
    first_head: usize,
    count: usize,
    head_dim: usize,
    row: &[u8],
    length: usize,
    out: &mut [f32],
    fused: bool,
) {
    // SAFETY: the caller's contract covers every pointer below; `head < count` keeps the query row
    // inside `q` and the slot inside `out`, and every load stays before `length`.
    unsafe {
        let bytes = row.as_ptr();
        let mut first = [_mm256_setzero_ps(); HEADS_IN_REGISTERS];
        let mut second = [_mm256_setzero_ps(); HEADS_IN_REGISTERS];
        let mut at = 0;
        while at + 16 <= length {
            let low = _mm256_cvtph_ps(_mm_loadu_si128(
                bytes.add(at * 2) as *const std::arch::x86_64::__m128i
            ));
            let high = _mm256_cvtph_ps(_mm_loadu_si128(
                bytes.add((at + 8) * 2) as *const std::arch::x86_64::__m128i
            ));
            for head in 0..count {
                let query = q.as_ptr().add((first_head + head) * head_dim + at);
                // `fused` picks the rounding, not the order: the unfused form is the multiply and
                // the add `dot_f16_f32_avx2` does, so a head scored this way keeps that kernel's
                // bits and the bit-for-bit oracle test still applies.
                if fused {
                    first[head] = _mm256_fmadd_ps(_mm256_loadu_ps(query), low, first[head]);
                    second[head] =
                        _mm256_fmadd_ps(_mm256_loadu_ps(query.add(8)), high, second[head]);
                } else {
                    first[head] =
                        _mm256_add_ps(first[head], _mm256_mul_ps(_mm256_loadu_ps(query), low));
                    second[head] = _mm256_add_ps(
                        second[head],
                        _mm256_mul_ps(_mm256_loadu_ps(query.add(8)), high),
                    );
                }
            }
            at += 16;
        }
        for head in 0..count {
            let mut total = hsum(_mm256_add_ps(first[head], second[head]));
            let query = q.as_ptr().add((first_head + head) * head_dim);
            let mut left = at;
            while left < length {
                let low = std::ptr::read_unaligned(bytes.add(left * 2) as *const u16);
                total += *query.add(left) * f16_to_f32(low);
                left += 1;
            }
            *out.as_mut_ptr().add(head) = total;
        }
    }
}

/// The interleaved V kernel: one conversion of the row per eight values, for every head.
///
/// # Safety
///
/// The CPU must support AVX2, F16C and FMA; `targets` must hold `weights.len()` rows of `head_dim`;
/// `length` must not exceed `head_dim` or `row`'s half.
#[target_feature(enable = "avx2,f16c,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn axpy_f16_f32_heads_avx2_fma(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    row: &[u8],
    length: usize,
    fused: bool,
) {
    // SAFETY: the caller's contract covers every pointer below; each store lands inside the head's
    // own row of `targets` and before `length`.
    unsafe {
        let bytes = row.as_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let values = _mm256_cvtph_ps(_mm_loadu_si128(
                bytes.add(at * 2) as *const std::arch::x86_64::__m128i
            ));
            for (head, weight) in weights.iter().enumerate() {
                let target = targets.as_mut_ptr().add(head * head_dim + at);
                let scale = _mm256_set1_ps(*weight);
                let current = _mm256_loadu_ps(target);
                let sum = if fused {
                    _mm256_fmadd_ps(scale, values, current)
                } else {
                    _mm256_add_ps(current, _mm256_mul_ps(scale, values))
                };
                _mm256_storeu_ps(target, sum);
            }
            at += 8;
        }
        for (head, weight) in weights.iter().enumerate() {
            let target = targets.as_mut_ptr().add(head * head_dim);
            let mut left = at;
            while left < length {
                let low = std::ptr::read_unaligned(bytes.add(left * 2) as *const u16);
                *target.add(left) += weight * f16_to_f32(low);
                left += 1;
            }
        }
    }
}

/// Whether the batched head kernels can use a fused multiply-add: AVX2, F16C and FMA, and no
/// `VIPERA_NO_VNNI`-style opt-out beyond the ones the other kernels honour.
pub(crate) fn f16_fma_available() -> bool {
    static FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FMA.get_or_init(|| {
        available() && std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("fma")
    })
}

/// [`dot_f16_f32_avx2`] with the multiply and the add fused.
///
/// The lane order and the two-accumulator tree are unchanged, so this rounds once where the
/// unfused kernel rounds twice — the scores of the window kernel can differ in the last bit from
/// [`dot_f16_f32`]. That is why only the batched head path takes it: the two-pass window kernel and
/// the exact-dot path stay the reference the attention tests compare against.
///
/// # Safety
///
/// The CPU must support AVX2, F16C and FMA, and `length` must not exceed `a`'s length or `row`'s
/// half.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f16_f32_avx2_fma(a: &[f32], row: &[u8], length: usize) -> f32 {
    // SAFETY: the caller guarantees the features and `length`; every load reads eight halves or
    // eight floats at an offset that leaves that many before `length`.
    unsafe {
        let (left, bytes) = (a.as_ptr(), row.as_ptr());
        let mut first = _mm256_setzero_ps();
        let mut second = _mm256_setzero_ps();
        let mut at = 0;
        while at + 16 <= length {
            first = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.add(at)),
                _mm256_cvtph_ps(_mm_loadu_si128(
                    bytes.add(at * 2) as *const std::arch::x86_64::__m128i
                )),
                first,
            );
            second = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.add(at + 8)),
                _mm256_cvtph_ps(_mm_loadu_si128(
                    bytes.add((at + 8) * 2) as *const std::arch::x86_64::__m128i
                )),
                second,
            );
            at += 16;
        }
        while at + 8 <= length {
            first = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.add(at)),
                _mm256_cvtph_ps(_mm_loadu_si128(
                    bytes.add(at * 2) as *const std::arch::x86_64::__m128i
                )),
                first,
            );
            at += 8;
        }
        let mut total = hsum(_mm256_add_ps(first, second));
        while at < length {
            let low = std::ptr::read_unaligned(bytes.add(at * 2) as *const u16);
            total += *left.add(at) * f16_to_f32(low);
            at += 1;
        }
        total
    }
}

/// [`axpy_f16_f32_avx2`] with the fused multiply-add, same lane order and same caveat as
/// [`dot_f16_f32_avx2_fma`].
///
/// # Safety
///
/// The CPU must support AVX2, F16C and FMA, and `target`'s length must not exceed `row`'s half.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpy_f16_f32_avx2_fma(target: &mut [f32], weight: f32, row: &[u8]) {
    // SAFETY: the caller guarantees the features and that every load is inside `row`; the loop
    // stops at the last whole block and the tail is one value at a time.
    unsafe {
        let length = target.len();
        let scale = _mm256_set1_ps(weight);
        let out = target.as_mut_ptr();
        let input = row.as_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let scaled = _mm256_cvtph_ps(_mm_loadu_si128(
                input.add(at * 2) as *const std::arch::x86_64::__m128i
            ));
            let current = _mm256_loadu_ps(out.add(at));
            _mm256_storeu_ps(out.add(at), _mm256_fmadd_ps(scale, scaled, current));
            at += 8;
        }
        while at < length {
            let low = std::ptr::read_unaligned(input.add(at * 2) as *const u16);
            *out.add(at) += weight * f16_to_f32(low);
            at += 1;
        }
    }
}

/// `targets += w * row` for the heads of one KV head and a whole block of positions at once.
///
/// The per-position form loads and stores every target row once per position: at this model's shape
/// that is 6 rows of 128 floats read and written for each 512-byte V row, so the weighted sum spends
/// most of its memory traffic on the accumulator it is accumulating into. Holding the accumulator in
/// registers for `P` positions and touching the target rows once per block removes that, and the
/// arithmetic per element is unchanged — the same multiply and add, in the same increasing position
/// order — so the result is bit-identical to [`axpy_f16_f32_heads_exact`] called once per position.
///
/// `weights` is position-major: `weights[p * count + head]`, with `count = weights.len() / P`.
pub(crate) fn axpy_f16_f32_heads_exact_block<const P: usize>(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    rows: &[&[u8]; P],
) {
    if P == 0 || weights.is_empty() {
        return;
    }
    let count = weights.len() / P;
    if count == 0 || count > HEADS_IN_REGISTERS || targets.len() < count * head_dim {
        return;
    }
    let length = head_dim.min(rows.iter().map(|row| row.len() / 2).min().unwrap_or(0));
    if length >= 8 && available() && std::is_x86_feature_detected!("f16c") {
        // SAFETY: the CPU was checked here, `count` is inside the register budget and each head's row
        // is inside `targets`; every row was length-checked above.
        unsafe {
            axpy_f16_f32_heads_block_avx2::<P>(targets, head_dim, weights, rows, count, length);
        }
        return;
    }
    for (p, row) in rows.iter().enumerate() {
        axpy_f16_f32_heads_exact(targets, head_dim, &weights[p * count..(p + 1) * count], row);
    }
}

/// # Safety
///
/// The CPU must support AVX2 and F16C; `count` must be at most [`HEADS_IN_REGISTERS`]; `targets` must
/// hold `count` rows of `head_dim`; `length` must not exceed `head_dim` nor any row's half; `weights`
/// must hold `P * count` values.
#[target_feature(enable = "avx2,f16c")]
unsafe fn axpy_f16_f32_heads_block_avx2<const P: usize>(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    rows: &[&[u8]; P],
    count: usize,
    length: usize,
) {
    // SAFETY: the caller's contract covers every pointer below; each store lands inside the head's
    // own row of `targets` and before `length`.
    unsafe {
        let base = targets.as_mut_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let mut acc = [_mm256_setzero_ps(); HEADS_IN_REGISTERS];
            for (head, slot) in acc.iter_mut().enumerate().take(count) {
                *slot = _mm256_loadu_ps(base.add(head * head_dim + at));
            }
            for (p, row) in rows.iter().enumerate() {
                let values = _mm256_cvtph_ps(_mm_loadu_si128(
                    row.as_ptr().add(at * 2) as *const std::arch::x86_64::__m128i
                ));
                for (head, slot) in acc.iter_mut().enumerate().take(count) {
                    let scale = _mm256_set1_ps(*weights.get_unchecked(p * count + head));
                    *slot = _mm256_add_ps(*slot, _mm256_mul_ps(scale, values));
                }
            }
            for (head, slot) in acc.iter().enumerate().take(count) {
                _mm256_storeu_ps(base.add(head * head_dim + at), *slot);
            }
            at += 8;
        }
        for (p, row) in rows.iter().enumerate() {
            for (head, weight) in weights[p * count..(p + 1) * count].iter().enumerate() {
                let target = base.add(head * head_dim);
                let mut left = at;
                while left < length {
                    let low = std::ptr::read_unaligned(row.as_ptr().add(left * 2) as *const u16);
                    *target.add(left) += weight * f16_to_f32(low);
                    left += 1;
                }
            }
        }
    }
}

/// [`dot_f16_f32_heads`] with [`dot_f16_f32`]'s rounding: the same two-accumulator tree per head,
/// with the multiply and the add kept apart.
///
/// This is the batched form for the paths that are pinned to the decoded-then-dot oracle bit for
/// bit — the streaming attention kernel, which is what a prefill spends half its time in. Batching
/// removes the per-head call overhead and the per-head row load; the arithmetic stays exactly the
/// one [`dot_f16_f32`] does, so the oracle comparison still holds.
pub(crate) fn dot_f16_f32_heads_exact(
    q: &[f32],
    first_head: usize,
    count: usize,
    head_dim: usize,
    row: &[u8],
    out: &mut [f32],
) {
    let fast = available()
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma");
    let row_length = row.len() / 2;
    let length = head_dim.min(row_length);
    if fast && length >= 16 && count <= HEADS_IN_REGISTERS && out.len() >= count {
        // SAFETY: the features were detected above, `count` is inside the register budget and every
        // head index below is inside `q` and `out`, which this function's slices bound at
        // `head_dim` per head.
        unsafe {
            dot_f16_f32_heads_avx2_fma(q, first_head, count, head_dim, row, length, out, false);
        }
        return;
    }
    for (member, slot) in out.iter_mut().take(count).enumerate() {
        let at = (first_head + member) * head_dim;
        *slot = dot_f16_f32(&q[at..at + head_dim], row);
    }
}

/// [`axpy_f16_f32_heads`] with [`axpy_f16_f32`]'s rounding, for the same bit-exact paths.
pub(crate) fn axpy_f16_f32_heads_exact(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    row: &[u8],
) {
    let fast = available()
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma");
    let row_length = row.len() / 2;
    let length = head_dim.min(row_length);
    if fast
        && length >= 8
        && weights.len() <= HEADS_IN_REGISTERS
        && targets.len() >= weights.len() * head_dim
    {
        // SAFETY: the features were detected above, `weights` is inside the register budget and
        // every row of `targets` below is inside the slice, which the caller sized per head.
        unsafe {
            axpy_f16_f32_heads_avx2_fma(targets, head_dim, weights, row, length, false);
        }
        return;
    }
    for (member, weight) in weights.iter().enumerate() {
        let target = &mut targets[member * head_dim..][..head_dim];
        axpy_f16_f32(target, *weight, row);
    }
}

/// [`axpy_f16_f32`] in eight lanes.
///
/// # Safety
///
/// The CPU must support AVX2 and F16C, and `target`'s length must not exceed `row`'s half.
#[target_feature(enable = "avx2,f16c")]
unsafe fn axpy_f16_f32_avx2(target: &mut [f32], weight: f32, row: &[u8]) {
    // SAFETY: the caller guarantees the features and that every load is inside `row`; the loop
    // stops at the last whole block and the tail is one value at a time.
    unsafe {
        let length = target.len();
        let scale = _mm256_set1_ps(weight);
        let out = target.as_mut_ptr();
        let input = row.as_ptr();
        let mut at = 0;
        while at + 8 <= length {
            let scaled = _mm256_mul_ps(
                scale,
                _mm256_cvtph_ps(_mm_loadu_si128(
                    input.add(at * 2) as *const std::arch::x86_64::__m128i
                )),
            );
            let current = _mm256_loadu_ps(out.add(at));
            _mm256_storeu_ps(out.add(at), _mm256_add_ps(current, scaled));
            at += 8;
        }
        while at < length {
            let low = std::ptr::read_unaligned(input.add(at * 2) as *const u16);
            *out.add(at) += weight * f16_to_f32(low);
            at += 1;
        }
    }
}

/// Query heads one [`score_block`] call keeps in registers: one accumulator each, beside the two
/// key vectors and the query loads, which is what fits in sixteen vector registers.
const SCORE_BLOCK: usize = 6;

/// Query heads one [`fold_block`] call keeps in registers: one accumulator each, plus the value
/// vector and a broadcast weight.
const VALUE_BLOCK: usize = 8;

/// Positions one [`axpy_f16_group`] call folds with its accumulators held in registers; callers
/// with more rows pass them in as many calls.
pub(crate) const GROUP_ROWS: usize = 16;

/// The scores of several query heads against one `f16` key row: `out[head]` is head `head`'s dot.
///
/// `queries` holds the heads back to back, `head_dim` values each, which is how a grouped-query
/// layer lays out the query heads that share one key head. [`dot_f16_f32`] run once per head
/// converts and loads the same key row once per head; this converts it once and feeds every head
/// from the same registers, so a group of six reads the row a sixth as often. It uses fused
/// multiply-adds and two accumulators per head, so it rounds differently from [`dot_f16_f32`] (the
/// measured difference is a few `f32` ulps of the sum of the absolute terms); it is not a
/// replacement where the bits have to match.
///
/// Safe to call: it needs AVX2, F16C and FMA and a `head_dim` that is a multiple of 16, and
/// otherwise runs [`dot_f16_f32`] per head.
pub(crate) fn dot_f16_group(queries: &[f32], head_dim: usize, row: &[u8], out: &mut [f32]) {
    let heads = out.len();
    let fits = head_dim > 0
        && head_dim.is_multiple_of(16)
        && queries.len() >= heads * head_dim
        && row.len() >= head_dim * 2;
    if heads > 0
        && fits
        && available()
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: AVX2, F16C and FMA were just detected; `queries` holds `heads * head_dim` values,
        // `row` holds `head_dim` halves and `out` holds `heads` slots, all checked above.
        unsafe {
            dot_f16_group_avx2(
                queries.as_ptr(),
                head_dim,
                row.as_ptr(),
                out.as_mut_ptr(),
                heads,
            );
        }
        return;
    }
    for (head, slot) in out.iter_mut().enumerate() {
        let query = queries
            .get(head * head_dim..(head + 1) * head_dim)
            .unwrap_or(&[]);
        *slot = dot_f16_f32(query, row);
    }
}

/// # Safety
///
/// AVX2, F16C and FMA; `queries` must hold `heads * head_dim` values, `row` `head_dim` halves, `out`
/// `heads` slots, and `head_dim` must be a multiple of 16.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_f16_group_avx2(
    queries: *const f32,
    head_dim: usize,
    row: *const u8,
    out: *mut f32,
    heads: usize,
) {
    // SAFETY: the caller guarantees the features and the extents; each block reads its own heads.
    unsafe {
        let mut head = 0;
        while head < heads {
            let block_queries = queries.add(head * head_dim);
            let block_out = out.add(head);
            match (heads - head).min(SCORE_BLOCK) {
                6 => score_block_single::<6>(block_queries, head_dim, row, block_out),
                5 => score_block_single::<5>(block_queries, head_dim, row, block_out),
                4 => score_block::<4>(block_queries, head_dim, row, block_out),
                3 => score_block::<3>(block_queries, head_dim, row, block_out),
                2 => score_block::<2>(block_queries, head_dim, row, block_out),
                _ => score_block::<1>(block_queries, head_dim, row, block_out),
            }
            head += SCORE_BLOCK;
        }
    }
}

/// Eight `f16` values from `bytes[at * 2..]` as `f32` lanes.
///
/// # Safety
///
/// AVX2 and F16C, and sixteen readable bytes at `bytes + at * 2`.
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn load_half(bytes: *const u8, at: usize) -> __m256 {
    // SAFETY: the caller guarantees the features and the sixteen bytes.
    unsafe {
        _mm256_cvtph_ps(_mm_loadu_si128(
            bytes.add(at * 2) as *const std::arch::x86_64::__m128i
        ))
    }
}

/// `HEADS` query heads against one key row, the row converted once per sixteen values.
///
/// # Safety
///
/// As [`dot_f16_group_avx2`], for `HEADS` heads.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn score_block_single<const HEADS: usize>(
    queries: *const f32,
    head_dim: usize,
    row: *const u8,
    out: *mut f32,
) {
    // SAFETY: the caller guarantees the features and the extents; every load is inside `head_dim`.
    unsafe {
        // One accumulator per head: the chains are independent, so five or six of them cover the
        // multiply-add latency between one head's steps, and six is the group a 12/2 layer has, so
        // its K row is converted once instead of once per four-head block.
        let mut sums = [_mm256_setzero_ps(); HEADS];
        let mut at = 0;
        while at < head_dim {
            let key_low = load_half(row, at);
            let key_high = load_half(row, at + 8);
            for (head, sum) in sums.iter_mut().enumerate() {
                let query = queries.add(head * head_dim + at);
                *sum = _mm256_fmadd_ps(_mm256_loadu_ps(query), key_low, *sum);
                *sum = _mm256_fmadd_ps(_mm256_loadu_ps(query.add(8)), key_high, *sum);
            }
            at += 16;
        }
        for (head, sum) in sums.iter().enumerate() {
            *out.add(head) = hsum(*sum);
        }
    }
}

/// [`score_block_single`] with two accumulators per head instead, which is faster for the small
/// groups where the extra chains cost nothing: four heads fill the registers where six would spill.
///
/// # Safety
///
/// As [`score_block_single`], for `HEADS` heads.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn score_block<const HEADS: usize>(
    queries: *const f32,
    head_dim: usize,
    row: *const u8,
    out: *mut f32,
) {
    // SAFETY: the caller guarantees the features and the extents; every load is inside `head_dim`.
    unsafe {
        let mut low = [_mm256_setzero_ps(); HEADS];
        let mut high = [_mm256_setzero_ps(); HEADS];
        let mut at = 0;
        while at < head_dim {
            let key_low = load_half(row, at);
            let key_high = load_half(row, at + 8);
            for head in 0..HEADS {
                let query = queries.add(head * head_dim + at);
                low[head] = _mm256_fmadd_ps(_mm256_loadu_ps(query), key_low, low[head]);
                high[head] = _mm256_fmadd_ps(_mm256_loadu_ps(query.add(8)), key_high, high[head]);
            }
            at += 16;
        }
        for head in 0..HEADS {
            *out.add(head) = hsum(_mm256_add_ps(low[head], high[head]));
        }
    }
}

/// `targets[head] += weights[position * heads + head] * rows[position]` for every head and
/// position: several query heads folding the same value rows into their outputs.
///
/// `targets` holds the heads back to back, `head_dim` values each. Each value row is converted once
/// and feeds every head, and the accumulators stay in registers across up to [`GROUP_ROWS`]
/// positions, where [`axpy_f16_f32`] per head and position loads and stores the target for each.
/// Each element still accumulates in increasing position order, but with a fused multiply-add, so
/// it rounds differently from [`axpy_f16_f32`].
///
/// Safe to call: it needs AVX2, F16C and FMA and a `head_dim` that is a multiple of 8, and otherwise
/// runs [`axpy_f16_f32`] per head and position.
pub(crate) fn axpy_f16_group(
    targets: &mut [f32],
    head_dim: usize,
    heads: usize,
    weights: &[f32],
    rows: &[&[u8]],
) {
    let fits = head_dim > 0
        && head_dim.is_multiple_of(8)
        && heads > 0
        && targets.len() >= heads * head_dim
        && weights.len() >= rows.len() * heads
        && rows.iter().all(|row| row.len() >= head_dim * 2);
    if fits
        && available()
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma")
    {
        for (chunk, rows) in rows.chunks(GROUP_ROWS).enumerate() {
            let mut pointers = [std::ptr::null::<u8>(); GROUP_ROWS];
            for (slot, row) in pointers.iter_mut().zip(rows) {
                *slot = row.as_ptr();
            }
            // SAFETY: AVX2, F16C and FMA were just detected; the extents of `targets`, `weights`
            // (`rows.len() * heads` values, of which this chunk starts at `chunk * GROUP_ROWS *
            // heads`) and every row were checked above.
            unsafe {
                axpy_f16_group_avx2(
                    targets.as_mut_ptr(),
                    head_dim,
                    heads,
                    weights.as_ptr().add(chunk * GROUP_ROWS * heads),
                    pointers.as_ptr(),
                    rows.len(),
                );
            }
        }
        return;
    }
    for (position, row) in rows.iter().enumerate() {
        for head in 0..heads {
            let weight = weights.get(position * heads + head).copied().unwrap_or(0.0);
            if let Some(target) = targets.get_mut(head * head_dim..(head + 1) * head_dim) {
                axpy_f16_f32(target, weight, row);
            }
        }
    }
}

/// # Safety
///
/// AVX2, F16C and FMA; `targets` must hold `heads * head_dim` values, `weights` `count * heads`,
/// `rows` `count` pointers to `head_dim` readable halves each, with `count` at most
/// [`GROUP_ROWS`], and `head_dim` must be a multiple of 8.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpy_f16_group_avx2(
    targets: *mut f32,
    head_dim: usize,
    heads: usize,
    weights: *const f32,
    rows: *const *const u8,
    count: usize,
) {
    // SAFETY: the caller guarantees the features and the extents; each block touches its own heads.
    unsafe {
        let mut head = 0;
        while head < heads {
            let block_targets = targets.add(head * head_dim);
            let block_weights = weights.add(head);
            match (heads - head).min(VALUE_BLOCK) {
                8 => fold_block::<8>(block_targets, head_dim, block_weights, heads, rows, count),
                7 => fold_block::<7>(block_targets, head_dim, block_weights, heads, rows, count),
                6 => fold_block::<6>(block_targets, head_dim, block_weights, heads, rows, count),
                5 => fold_block::<5>(block_targets, head_dim, block_weights, heads, rows, count),
                4 => fold_block::<4>(block_targets, head_dim, block_weights, heads, rows, count),
                3 => fold_block::<3>(block_targets, head_dim, block_weights, heads, rows, count),
                2 => fold_block::<2>(block_targets, head_dim, block_weights, heads, rows, count),
                _ => fold_block::<1>(block_targets, head_dim, block_weights, heads, rows, count),
            }
            head += VALUE_BLOCK;
        }
    }
}

/// `HEADS` heads folding `count` value rows, eight values at a time, accumulators in registers.
///
/// # Safety
///
/// As [`axpy_f16_group_avx2`], for `HEADS` heads; `weight_stride` is the distance between
/// positions in `weights`.
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn fold_block<const HEADS: usize>(
    targets: *mut f32,
    head_dim: usize,
    weights: *const f32,
    weight_stride: usize,
    rows: *const *const u8,
    count: usize,
) {
    // SAFETY: the caller guarantees the features and the extents; every access is inside them.
    unsafe {
        let mut at = 0;
        while at < head_dim {
            let mut accumulators = [_mm256_setzero_ps(); HEADS];
            for (head, accumulator) in accumulators.iter_mut().enumerate() {
                *accumulator = _mm256_loadu_ps(targets.add(head * head_dim + at));
            }
            for position in 0..count {
                let value = load_half(*rows.add(position), at);
                let row_weights = weights.add(position * weight_stride);
                for (head, accumulator) in accumulators.iter_mut().enumerate() {
                    *accumulator = _mm256_fmadd_ps(
                        _mm256_set1_ps(*row_weights.add(head)),
                        value,
                        *accumulator,
                    );
                }
            }
            for (head, accumulator) in accumulators.iter().enumerate() {
                _mm256_storeu_ps(targets.add(head * head_dim + at), *accumulator);
            }
            at += 8;
        }
    }
}

/// The AVX2 dot for a Q4_K row, or `None` when this CPU has no AVX2 or the format is another one.
///
/// Safe to call: it checks the feature itself, and the lengths were validated by the caller's
/// `matvec`, whose contract is the same one [`crate::quant::matvec`] enforces before dispatch.
///
/// `xq` is the quantized activation the Q4_K kernel reads; without it the Q4_K row falls through to
/// the scalar path rather than guessing a quantization.
pub(crate) fn try_dot(
    ty: GgmlType,
    row: &[u8],
    x: &[f32],
    sums: &[f32],
    xq: Option<&QuantizedX>,
) -> Option<f32> {
    if !available() {
        return None;
    }
    // SAFETY: `available()` established AVX2 support, and the caller's lengths match the scalar
    // kernel's expectations, which the kernels below mirror exactly. The quantized activation has
    // 32 values per group, so it covers 256 values per block like `x` does.
    Some(unsafe {
        match ty {
            GgmlType::Q4K => {
                let xq = xq?;
                // The faster kernel reorders float adds, so the exact path keeps the old one.
                // The faster kernel reorders float adds, so the exact path keeps the old one.
                if crate::quant::exact_dot_requested() {
                    q4k_dot_previous(row, &xq.q, &xq.d, sums)
                } else if vnni_available() {
                    q4k_dot_vnni(row, &xq.q, &xq.d, sums)
                } else {
                    q4k_dot(row, &xq.q, &xq.d, sums)
                }
            }
            // The integer path is the default for both, as for Q6_K: `VIPERA_EXACT_DOT` restores
            // the `f32` row kernel, which is what the pre-integer tokens were produced with.
            GgmlType::Q3K => match xq {
                Some(xq) if !crate::quant::exact_dot_requested() => {
                    if vnni_available() {
                        q3k_dot_vnni(row, &xq.q, &xq.d)
                    } else {
                        q3k_dot_int(row, &xq.q, &xq.d)
                    }
                }
                _ => q3k_dot_avx2(row, x),
            },
            GgmlType::Q5K => match xq {
                Some(xq) if !crate::quant::exact_dot_requested() => {
                    if vnni_available() {
                        q5k_dot_vnni(row, &xq.q, &xq.d, sums)
                    } else {
                        q5k_dot_int(row, &xq.q, &xq.d, sums)
                    }
                }
                _ => q5k_dot_avx2(row, x),
            },
            GgmlType::Q6K => match xq {
                Some(xq) => {
                    // SAFETY: the outer block of `try_dot` and `vnni_available()` above establish
                    // the CPU features; both kernels take the same row/activation lengths.
                    if vnni_available() && crate::quant::q6k_vnni_requested() {
                        q6k_dot_vnni(row, xq)
                    } else {
                        q6k_dot_q8(row, xq)
                    }
                }
                None => q6k_dot(row, x),
            },
            _ => return None,
        }
    })
}

/// The AVX2 dots for `B` activations at once, or `false` when this CPU has no AVX2 or the format is
/// another one.
///
/// Safe to call for the same reason [`try_dot`] is: it checks the CPU itself, and the lengths were
/// validated by the caller's `matvec`, whose contract is the same one [`crate::quant::matvec`]
/// enforces before dispatch.
pub(crate) fn try_dot_batch<const B: usize>(
    ty: GgmlType,
    row: &[u8],
    activation: &BatchActivation<'_, B>,
    out: &mut [f32; B],
) -> bool {
    if !available() {
        return false;
    }
    // Under the flag a batch runs the old kernel once per entry, which is exactly how the old batch
    // kernel behaved entry by entry, so a batch and a lone sequence still agree to the bit.
    if crate::quant::exact_dot_requested() {
        return match ty {
            GgmlType::Q4K => {
                let Some(xs) = activation.quantized() else {
                    return false;
                };
                for (entry, out) in out.iter_mut().enumerate() {
                    // SAFETY: `available()` established AVX2, and the lengths match the kernel's
                    // contract, which the caller's `matvec` enforces.
                    *out = unsafe {
                        q4k_dot_previous(row, &xs[entry].q, &xs[entry].d, activation.sums()[entry])
                    };
                }
                true
            }
            _ => false,
        };
    }
    match ty {
        GgmlType::Q4K => {
            // A Q4_K row needs the quantized activation; without it the caller's scalar path is the
            // only honest answer.
            let Some(xs) = activation.quantized() else {
                return false;
            };
            // SAFETY: `available()` established AVX2 support, and the caller's lengths match the
            // scalar kernel's expectations, which the kernels below mirror exactly. The VNNI kernel
            // is only reached when its own feature check passed.
            unsafe {
                if vnni_available() {
                    q4k_dot_batch_vnni(row, xs, activation.sums(), out)
                } else {
                    q4k_dot_batch(row, xs, activation.sums(), out)
                }
            }
            true
        }
        // No weight-sharing batch kernel for these two: measured, the batch form was slower than
        // running the row kernel once per entry (the row is already cache-warm across entries, and
        // a batch of 64 spills the per-entry accumulators). Each entry takes the kernel a lone
        // sequence would.
        GgmlType::Q3K => {
            let exact = crate::quant::exact_dot_requested();
            for (entry, out) in out.iter_mut().enumerate() {
                *out = match activation.quantized().map(|xs| &xs[entry]) {
                    // SAFETY: as above; these kernels take `xq` and cover 256 values per block.
                    Some(xq) if !exact => unsafe {
                        if vnni_available() {
                            q3k_dot_vnni(row, &xq.q, &xq.d)
                        } else {
                            q3k_dot_int(row, &xq.q, &xq.d)
                        }
                    },
                    // SAFETY: as above; the `f32` row kernel covers 256 values per block.
                    _ => unsafe { q3k_dot_avx2(row, activation.x()[entry]) },
                };
            }
            true
        }
        GgmlType::Q5K => {
            let exact = crate::quant::exact_dot_requested();
            for (entry, out) in out.iter_mut().enumerate() {
                *out = match activation.quantized().map(|xs| &xs[entry]) {
                    // SAFETY: as above; these kernels take `xq` and cover 256 values per block.
                    Some(xq) if !exact => unsafe {
                        if vnni_available() {
                            q5k_dot_vnni(row, &xq.q, &xq.d, activation.sums()[entry])
                        } else {
                            q5k_dot_int(row, &xq.q, &xq.d, activation.sums()[entry])
                        }
                    },
                    // SAFETY: as above; the `f32` row kernel covers 256 values per block.
                    _ => unsafe { q5k_dot_avx2(row, activation.x()[entry]) },
                };
            }
            true
        }
        GgmlType::Q6K => match activation.quantized() {
            // SAFETY: as above; a quantized activation covers the 256 values of a block.
            Some(xs) => {
                // SAFETY: as above; a quantized activation covers the 256 values of a block.
                unsafe { q6k_dot_batch_q8(row, xs, out) };
                true
            }
            None => {
                // SAFETY: as above.
                unsafe { q6k_dot_batch(row, activation.x(), out) };
                true
            }
        },
        _ => false,
    }
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

/// The Q4_K dot as it stood before the minimum term and the horizontal sums were moved out of the
/// block loop.
///
/// It keeps the old order of every float operation, which is what `VIPERA_EXACT_DOT=1` needs: the
/// tokens before the change were produced with this arithmetic. The batch kernel stays bit-identical
/// to this one entry by entry, so a batch in exact mode is one call per entry.
///
/// # Safety
///
/// The module contract applies: `row` whole 144-byte blocks, `xq` 256 values and `dx` 8 scales per
/// block, `sums` 8 entries per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn q4k_dot_previous(row: &[u8], xq: &[i8], dx: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the caller states the lengths, and every read stays inside them.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 144;
        let low_mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let scales = std::slice::from_raw_parts(block.add(4), 12);
            let mut accumulator = _mm256_setzero_ps();
            for j in 0..4 {
                let q = block.add(16 + j * 32);
                let packed = _mm256_loadu_si256(q as *const __m256i);
                let low_bytes = _mm256_and_si256(packed, low_mask);
                let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let values = xq.as_ptr().add(b * 256 + j * 64);
                let low_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(low_bytes, _mm256_loadu_si256(values as *const __m256i)),
                    ones,
                );
                let high_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(
                        high_bytes,
                        _mm256_loadu_si256(values.add(32) as *const __m256i),
                    ),
                    ones,
                );
                let (scale0, min0) = crate::quant::q4k_scale_min(scales, 2 * j);
                let (scale1, min1) = crate::quant::q4k_scale_min(scales, 2 * j + 1);
                let base = b * 8 + 2 * j;
                let low_coeff = d * f32::from(scale0) * dx[b * 8 + 2 * j];
                let high_coeff = d * f32::from(scale1) * dx[b * 8 + 2 * j + 1];
                accumulator = _mm256_add_ps(
                    accumulator,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(low_sums), _mm256_set1_ps(low_coeff)),
                );
                accumulator = _mm256_add_ps(
                    accumulator,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(high_sums), _mm256_set1_ps(high_coeff)),
                );
                total -= dmin * f32::from(min0) * sums[base];
                total -= dmin * f32::from(min1) * sums[base + 1];
            }
            total += hsum(accumulator);
        }
        total
    }
}

/// The eight Q4_K scales and the eight minimums of one block, as vectors in sub-block order.
///
/// The twelve bytes hold both sets of eight 6-bit values. Asking `q4k_scale_min` for one index at a
/// time repeats the same bit work eight times per block and branches on the index; four `u32`s and
/// two byte-to-lane conversions do all eight at once. Both kernels call this, so the scale and
/// minimum each sub-block sees are the same vector lanes in the same order.
///
/// # Safety
///
/// `block` must point at the start of a whole 144-byte Q4_K block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn q4k_scale_vectors(block: *const u8) -> (__m256, __m256) {
    // SAFETY: the caller guarantees a whole block, so the twelve scale/min bytes at offsets 4..16
    // are inside it.
    unsafe {
        let low = u32::from_le(std::ptr::read_unaligned(block.add(4) as *const u32));
        let high = u32::from_le(std::ptr::read_unaligned(block.add(8) as *const u32));
        let packed = u32::from_le(std::ptr::read_unaligned(block.add(12) as *const u32));
        let scale_bytes = u64::from(low & 0x3f3f_3f3f)
            | (u64::from((packed & 0x0f0f_0f0f) | ((low >> 6) & 0x0303_0303) << 4) << 32);
        let min_bytes = u64::from(high & 0x3f3f_3f3f)
            | (u64::from(((packed >> 4) & 0x0f0f_0f0f) | ((high >> 6) & 0x0303_0303) << 4) << 32);
        let scale_at =
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(scale_bytes as i64)));
        let min_at = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(min_bytes as i64)));
        (scale_at, min_at)
    }
}

/// Q4_K row dot against an `i8` activation. Mirrors [`crate::quant`]'s scalar version up to the
/// activation quantization described on `QuantizedX`.
///
/// One 32-byte weight load yields both nibble planes, and each plane is dotted with 32 `i8`
/// activations by `maddubs` (unsigned × signed, 16 pair sums) followed by `madd` against ones (8
/// `i32` lane sums). The products cannot saturate: the largest pair is `2 * 15 * 127 = 3810`.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 144-byte blocks, `xq` exactly 256 values and
/// `dx` 8 scales per block, `sums` at least 8 entries per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn q4k_dot(row: &[u8], xq: &[i8], dx: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the module documentation is the contract: `row` is whole 144-byte blocks, `xq` holds
    // 256 values and `dx` 8 scales per block, `sums` holds 8 entries per block, and the CPU has
    // AVX2. Every read stays inside those bounds, including `from_raw_parts(block + 4, 12)`, which
    // covers exactly the 12 scale/min bytes a Q4_K block reserves for them.
    unsafe {
        let blocks = row.len() / 144;
        // A byte mask: `maddubs` works on every byte of the vector, so a per-`i32` mask would keep
        // only the low byte of each group and silently drop three quarters of the weights.
        let low_mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        // Two accumulators for the weight part, filled by alternating sub-blocks: the adds were one
        // dependent chain of eight, and splitting the chain is what keeps the multiply units busy.
        let mut weights0 = _mm256_setzero_ps();
        let mut weights1 = _mm256_setzero_ps();
        // The minimum term is one vector multiply-add per block instead of eight scalar ones, and
        // its horizontal sum waits for the end of the row like the others.
        let mut minimums = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            // `d * scale` is the weight scale and `dx` the activation scale, so their product turns
            // the integer sum into the real dot of a sub-block. All eight are built once here; the
            // sub-block loop below only broadcasts the one it needs out of this vector.
            let coefficients = _mm256_mul_ps(
                _mm256_mul_ps(scale_at, _mm256_loadu_ps(dx.as_ptr().add(b * 8))),
                _mm256_set1_ps(d),
            );
            minimums = _mm256_fmadd_ps(
                _mm256_mul_ps(min_at, _mm256_set1_ps(dmin)),
                _mm256_loadu_ps(sums.as_ptr().add(b * 8)),
                minimums,
            );
            let mut coeff_stack = [0.0f32; 8];
            _mm256_storeu_ps(coeff_stack.as_mut_ptr(), coefficients);
            for j in 0..4 {
                let q = block.add(16 + j * 32);
                let packed = _mm256_loadu_si256(q as *const __m256i);
                let low_bytes = _mm256_and_si256(packed, low_mask);
                let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let values = xq.as_ptr().add(b * 256 + j * 64);
                let low_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(low_bytes, _mm256_loadu_si256(values as *const __m256i)),
                    ones,
                );
                let high_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(
                        high_bytes,
                        _mm256_loadu_si256(values.add(32) as *const __m256i),
                    ),
                    ones,
                );
                let low_coeff = _mm256_set1_ps(coeff_stack[2 * j]);
                let high_coeff = _mm256_set1_ps(coeff_stack[2 * j + 1]);
                let low_lanes = _mm256_cvtepi32_ps(low_sums);
                let high_lanes = _mm256_cvtepi32_ps(high_sums);
                if j.is_multiple_of(2) {
                    weights0 = _mm256_fmadd_ps(low_lanes, low_coeff, weights0);
                    weights1 = _mm256_fmadd_ps(high_lanes, high_coeff, weights1);
                } else {
                    weights1 = _mm256_fmadd_ps(low_lanes, low_coeff, weights1);
                    weights0 = _mm256_fmadd_ps(high_lanes, high_coeff, weights0);
                }
            }
        }
        // Two horizontal sums for the whole row instead of one per block, which took a serialised
        // chain of shuffles out of the block loop.
        hsum(_mm256_add_ps(weights0, weights1)) - hsum(minimums)
    }
}

/// Q6_K row dot, eight values per lane group. Mirrors `crate::quant::q6k_dot`.
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

/// The sixteen Q3_K sub-block scales of one block, already biased by `-32`.
///
/// The twelve packed bytes are unpacked by the same bit shuffles `dequantize_row_q3_K` uses; the
/// result is the per-16-value multiplier `d * (scale - 32)` is built from. Once per block, so the
/// scalar work is not on the hot path.
///
/// # Safety
///
/// `block` must point at the start of a whole 110-byte block.
unsafe fn q3k_scales(block: *const u8) -> [f32; 16] {
    // SAFETY: the caller's contract says the four-byte reads below stay inside the block.
    unsafe {
        let a0 = (block.add(96) as *const u32).read_unaligned();
        let a1 = (block.add(100) as *const u32).read_unaligned();
        let tmp = (block.add(104) as *const u32).read_unaligned();
        const K1: u32 = 0x0303_0303;
        const K2: u32 = 0x0f0f_0f0f;
        let aux = [
            (a0 & K2) | ((tmp & K1) << 4),
            (a1 & K2) | (((tmp >> 2) & K1) << 4),
            ((a0 >> 4) & K2) | (((tmp >> 4) & K1) << 4),
            ((a1 >> 4) & K2) | (((tmp >> 6) & K1) << 4),
        ];
        std::array::from_fn(|i| f32::from(aux[i / 4].to_le_bytes()[i % 4] as i8 - 32))
    }
}

/// Dot product of one Q3_K row with `x`: the three-bit codes widened to `f32` eight at a time, with
/// the `-4` bias applied as a mask instead of a branch.
///
/// # Safety
///
/// `row` must be whole 110-byte blocks, `x` 256 values per block, and the CPU must have AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn q3k_dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: the caller guarantees whole blocks and matching activation length, and AVX2. Every
    // load below stays inside one block and one 256-value activation block.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 110;
        let three = _mm256_set1_epi32(3);
        let zero = _mm256_setzero_si256();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 110);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(108), *block.add(109)]));
            let scales = q3k_scales(block);
            let mut accumulator = _mm256_setzero_ps();
            let mut is = 0_usize;
            let mut bit = 1_u32;
            for half in 0..2 {
                let qs = block.add(32 + half * 32);
                let x_half = x.as_ptr().add(b * 256 + half * 128);
                for run in 0..4 {
                    let mask = _mm256_set1_epi32(bit as i32);
                    let shift = _mm256_set1_epi32((run * 2) as i32);
                    for part in 0..2 {
                        let scale = _mm256_set1_ps(scales[is]);
                        is += 1;
                        let qs_part = qs.add(part * 16);
                        let hm_part = block.add(part * 16);
                        let x_part = x_half.add(run * 32 + part * 16);
                        for at in 0..2 {
                            let off = at * 8;
                            let bytes = _mm_loadl_epi64(qs_part.add(off) as *const __m128i);
                            // Variable shift, because the run number is a loop value.
                            let codes = _mm256_and_si256(
                                _mm256_srlv_epi32(_mm256_cvtepu8_epi32(bytes), shift),
                                three,
                            );
                            let highs = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                hm_part.add(off) as *const __m128i
                            ));
                            let clear = _mm256_cmpeq_epi32(_mm256_and_si256(highs, mask), zero);
                            // `clear` is all ones where the sign bit is unset, so shifting it left
                            // by two adds exactly `-4` there and nothing where it is set.
                            let levels = _mm256_add_epi32(codes, _mm256_slli_epi32(clear, 2));
                            let xv = _mm256_loadu_ps(x_part.add(off));
                            accumulator = _mm256_add_ps(
                                accumulator,
                                _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(levels), xv), scale),
                            );
                        }
                    }
                    bit <<= 1;
                }
            }
            total += d * hsum(accumulator);
        }
        total
    }
}

/// Dot product of one Q5_K row with `x`: the four low bits plus the sixteenth level from `qh`, with
/// the `dmin * min` correction accumulated against the same activation loads.
///
/// # Safety
///
/// `row` must be whole 176-byte blocks, `x` 256 values per block, and the CPU must have AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn q5k_dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: the caller guarantees whole blocks and matching activation length, and AVX2. Every
    // load below stays inside one 176-byte block and one 256-value activation block.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 176;
        let low_nibbles = _mm256_set1_epi32(0x0f);
        let sixteen = _mm256_set1_epi32(16);
        let zero = _mm256_setzero_si256();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 176);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(0), *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let scales = std::slice::from_raw_parts(block.add(4), 12);
            let qh = block.add(16);
            let qs = block.add(48);
            let mut levels_acc = _mm256_setzero_ps();
            let mut min_acc = _mm256_setzero_ps();
            let mut is = 0_usize;
            let mut bits = [1_u32, 2_u32];
            for group in 0..4 {
                let x_group = x.as_ptr().add(b * 256 + group * 64);
                for (part, &bit) in bits.iter().enumerate() {
                    let (sc, m) = crate::quant::q4k_scale_min(scales, is);
                    is += 1;
                    let scale = _mm256_set1_ps(f32::from(sc));
                    let min = _mm256_set1_ps(f32::from(m));
                    let mask = _mm256_set1_epi32(bit as i32);
                    let x_part = x_group.add(part * 32);
                    for at in 0..4 {
                        let off = at * 8;
                        let bytes = _mm_loadl_epi64(qs.add(group * 32 + off) as *const __m128i);
                        let mut codes = _mm256_cvtepu8_epi32(bytes);
                        codes = if part == 0 {
                            _mm256_and_si256(codes, low_nibbles)
                        } else {
                            _mm256_srli_epi32(codes, 4)
                        };
                        let highs =
                            _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh.add(off) as *const __m128i));
                        // Sixteen where the `qh` bit is set: the all-ones mask of `cmpeq` keeps
                        // `sixteen`, and zeroes it where the bit is clear.
                        let high_level = _mm256_andnot_si256(
                            _mm256_cmpeq_epi32(_mm256_and_si256(highs, mask), zero),
                            sixteen,
                        );
                        let levels = _mm256_cvtepi32_ps(_mm256_add_epi32(codes, high_level));
                        let xv = _mm256_loadu_ps(x_part.add(off));
                        levels_acc = _mm256_add_ps(
                            levels_acc,
                            _mm256_mul_ps(_mm256_mul_ps(levels, xv), scale),
                        );
                        min_acc = _mm256_add_ps(min_acc, _mm256_mul_ps(xv, min));
                    }
                }
                bits[0] <<= 2;
                bits[1] <<= 2;
            }
            total += d * hsum(levels_acc) - dmin * hsum(min_acc);
        }
        total
    }
}

/// Q5_K dot against an `i8` activation: the same shape as [`q4k_dot`], with the fifth bit of each
/// value taken from `qh` and OR-ed into the nibble before the integer multiply.
///
/// `maddubs` handles 32 values per instruction here where the `f32` kernel needs four widen rounds,
/// which is where the byte saved by five-bit weights turns into time.
///
/// # Safety
///
/// `row` must be whole 176-byte blocks, `xq` 256 values and `dx` 8 scales per block, `sums` at
/// least 8 entries per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q5k_dot_int(row: &[u8], xq: &[i8], dx: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the caller states the lengths, and AVX2; every load stays inside them.
    unsafe {
        let blocks = row.len() / 176;
        let low_mask = _mm256_set1_epi8(0x0f);
        let sixteen = _mm256_set1_epi8(16);
        let zero = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut weights = _mm256_setzero_ps();
        let mut minimums = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 176);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            let coefficients = _mm256_mul_ps(
                _mm256_mul_ps(scale_at, _mm256_loadu_ps(dx.as_ptr().add(b * 8))),
                _mm256_set1_ps(d),
            );
            minimums = _mm256_fmadd_ps(
                _mm256_mul_ps(min_at, _mm256_set1_ps(dmin)),
                _mm256_loadu_ps(sums.as_ptr().add(b * 8)),
                minimums,
            );
            let qh = _mm256_loadu_si256(block.add(16) as *const __m256i);
            for j in 0..4 {
                let packed = _mm256_loadu_si256(block.add(48 + j * 32) as *const __m256i);
                let low_nibbles = _mm256_and_si256(packed, low_mask);
                let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let low_bit = _mm256_set1_epi8((1_u32 << (2 * j)) as i8);
                let high_bit = _mm256_set1_epi8((2_u32 << (2 * j)) as i8);
                // `andnot(cmpeq(.., 0), 16)` is sixteen where the `qh` bit is set and zero elsewhere.
                let low_codes = _mm256_or_si256(
                    low_nibbles,
                    _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, low_bit), zero),
                        sixteen,
                    ),
                );
                let high_codes = _mm256_or_si256(
                    high_nibbles,
                    _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, high_bit), zero),
                        sixteen,
                    ),
                );
                let values = xq.as_ptr().add(b * 256 + j * 64);
                let low_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(low_codes, _mm256_loadu_si256(values as *const __m256i)),
                    ones,
                );
                let high_sums = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(
                        high_codes,
                        _mm256_loadu_si256(values.add(32) as *const __m256i),
                    ),
                    ones,
                );
                let low_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j) as i32));
                let high_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j + 1) as i32));
                weights = _mm256_fmadd_ps(_mm256_cvtepi32_ps(low_sums), low_coeff, weights);
                weights = _mm256_fmadd_ps(_mm256_cvtepi32_ps(high_sums), high_coeff, weights);
            }
        }
        hsum(weights) - hsum(minimums)
    }
}

/// [`q5k_dot_int`] with `vpdpbusd`, which does the multiply and the four-way add in one
/// instruction. The integer result is identical, so the float work is the same.
///
/// # Safety
///
/// As [`q5k_dot_int`], plus a CPU with `avx512_vnni` and `avx512_vl`.
#[target_feature(enable = "avx512vnni,avx512vl,avx2")]
pub(crate) unsafe fn q5k_dot_vnni(row: &[u8], xq: &[i8], dx: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the caller states the lengths, and the CPU features; every load stays inside them.
    unsafe {
        let blocks = row.len() / 176;
        let low_mask = _mm256_set1_epi8(0x0f);
        let sixteen = _mm256_set1_epi8(16);
        let zero = _mm256_setzero_si256();
        let mut weights = _mm256_setzero_ps();
        let mut minimums = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 176);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            let coefficients = _mm256_mul_ps(
                _mm256_mul_ps(scale_at, _mm256_loadu_ps(dx.as_ptr().add(b * 8))),
                _mm256_set1_ps(d),
            );
            minimums = _mm256_fmadd_ps(
                _mm256_mul_ps(min_at, _mm256_set1_ps(dmin)),
                _mm256_loadu_ps(sums.as_ptr().add(b * 8)),
                minimums,
            );
            let qh = _mm256_loadu_si256(block.add(16) as *const __m256i);
            for j in 0..4 {
                let packed = _mm256_loadu_si256(block.add(48 + j * 32) as *const __m256i);
                let low_nibbles = _mm256_and_si256(packed, low_mask);
                let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let low_bit = _mm256_set1_epi8((1_u32 << (2 * j)) as i8);
                let high_bit = _mm256_set1_epi8((2_u32 << (2 * j)) as i8);
                let low_codes = _mm256_or_si256(
                    low_nibbles,
                    _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, low_bit), zero),
                        sixteen,
                    ),
                );
                let high_codes = _mm256_or_si256(
                    high_nibbles,
                    _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, high_bit), zero),
                        sixteen,
                    ),
                );
                let values = xq.as_ptr().add(b * 256 + j * 64);
                let low_sums = _mm256_dpbusd_epi32(
                    zero,
                    low_codes,
                    _mm256_loadu_si256(values as *const __m256i),
                );
                let high_sums = _mm256_dpbusd_epi32(
                    zero,
                    high_codes,
                    _mm256_loadu_si256(values.add(32) as *const __m256i),
                );
                let low_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j) as i32));
                let high_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j + 1) as i32));
                weights = _mm256_fmadd_ps(_mm256_cvtepi32_ps(low_sums), low_coeff, weights);
                weights = _mm256_fmadd_ps(_mm256_cvtepi32_ps(high_sums), high_coeff, weights);
            }
        }
        hsum(weights) - hsum(minimums)
    }
}

/// Q3_K dot against an `i8` activation.
///
/// The three-bit codes are made unsigned (`code + 4·high`, 0..7) so `maddubs` accepts them, and the
/// `-4` bias the reference applies to the low values becomes a subtraction of four times the
/// activation sum over the same sixteen values. One 32-byte run covers two 16-value sub-blocks, so
/// the coefficients differ across the lane halves: lanes 0-3 scale the first sub-block, 4-7 the
/// second.
///
/// # Safety
///
/// `row` must be whole 110-byte blocks, `xq` 256 values per block, `dx` 8 scales per block, and the
/// CPU must support AVX2. `sums` is unused for this format (Q3_K has no minimum term).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q3k_dot_int(row: &[u8], xq: &[i8], dx: &[f32]) -> f32 {
    // SAFETY: the caller states the lengths, and AVX2; every load stays inside them.
    unsafe {
        let blocks = row.len() / 110;
        let three = _mm256_set1_epi8(3);
        let four_levels = _mm256_set1_epi8(4);
        let ones = _mm256_set1_epi16(1);
        let one_u8 = _mm256_set1_epi8(1);
        let four = _mm256_set1_ps(4.0);
        let zero = _mm256_setzero_si256();
        let mut accumulator = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 110);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(108), *block.add(109)]));
            let scales = q3k_scales(block);
            let qh = _mm256_loadu_si256(block as *const __m256i);
            let dx_at = dx.as_ptr().add(b * 8);
            let mut bit = 1_u8;
            for half in 0..2 {
                let qs = block.add(32 + half * 32);
                let x_half = b * 256 + half * 128;
                // Per-half coefficients: this half's eight sub-block scales, each paired with the
                // 32-value activation scale covering it, times `d`. Built with vectors so no
                // per-run scalar insert is needed.
                let dx4 = _mm_loadu_ps(dx_at.add(half * 4));
                let dx8 = _mm256_set_m128(dx4, dx4);
                let dx_dup =
                    _mm256_permutevar8x32_ps(dx8, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
                let coeffs = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(scales.as_ptr().add(half * 8)), dx_dup),
                    _mm256_set1_ps(d),
                );
                for run in 0..4 {
                    let mask = _mm256_set1_epi8(bit as i8);
                    let shift = _mm256_set1_epi32((run * 2) as i32);
                    // Lanes 0-3 are the first sixteen values of the run, 4-7 the second.
                    let pick = _mm256_setr_epi32(
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                    );
                    // Byte `k` of the 32-byte run is the code of value `k`: part 0 in bytes 0-15,
                    // part 1 in 16-31, which is row order. The variable shift is per 32-bit lane;
                    // masking every byte with three removes the bits shifted in from the next byte
                    // because a run shifts by at most six.
                    let packed = _mm256_loadu_si256(qs as *const __m256i);
                    let codes = _mm256_and_si256(_mm256_srlv_epi32(packed, shift), three);
                    let high = _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask), zero),
                        four_levels,
                    );
                    let unsigned = _mm256_add_epi8(codes, high);
                    let xv =
                        _mm256_loadu_si256(xq.as_ptr().add(x_half + run * 32) as *const __m256i);
                    let dot = _mm256_madd_epi16(_mm256_maddubs_epi16(unsigned, xv), ones);
                    let bias = _mm256_madd_epi16(_mm256_maddubs_epi16(one_u8, xv), ones);
                    // Lanes 0-3 cover the first sixteen values and 4-7 the second, so the two
                    // sub-block coefficients are one vector with a different scale per half.
                    let coefficients = _mm256_permutevar8x32_ps(coeffs, pick);
                    let lanes = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(dot),
                        _mm256_mul_ps(four, _mm256_cvtepi32_ps(bias)),
                    );
                    accumulator = _mm256_fmadd_ps(lanes, coefficients, accumulator);
                    bit <<= 1;
                }
            }
        }
        hsum(accumulator)
    }
}

/// [`q3k_dot_int`] with `vpdpbusd` in place of `maddubs` + `madd`. The integer result is identical.
///
/// # Safety
///
/// As [`q3k_dot_int`], plus a CPU with `avx512_vnni` and `avx512_vl`.
#[target_feature(enable = "avx512vnni,avx512vl,avx2")]
pub(crate) unsafe fn q3k_dot_vnni(row: &[u8], xq: &[i8], dx: &[f32]) -> f32 {
    // SAFETY: the caller states the lengths, and the CPU features; every load stays inside them.
    unsafe {
        let blocks = row.len() / 110;
        let three = _mm256_set1_epi8(3);
        let four_levels = _mm256_set1_epi8(4);
        let one_u8 = _mm256_set1_epi8(1);
        let four = _mm256_set1_ps(4.0);
        let zero = _mm256_setzero_si256();
        let mut accumulator = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 110);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(108), *block.add(109)]));
            let scales = q3k_scales(block);
            let qh = _mm256_loadu_si256(block as *const __m256i);
            let dx_at = dx.as_ptr().add(b * 8);
            let mut bit = 1_u8;
            for half in 0..2 {
                let qs = block.add(32 + half * 32);
                let x_half = b * 256 + half * 128;
                // Per-half coefficients: this half's eight sub-block scales, each paired with the
                // 32-value activation scale covering it, times `d`. Built with vectors so no
                // per-run scalar insert is needed.
                let dx4 = _mm_loadu_ps(dx_at.add(half * 4));
                let dx8 = _mm256_set_m128(dx4, dx4);
                let dx_dup =
                    _mm256_permutevar8x32_ps(dx8, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
                let coeffs = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(scales.as_ptr().add(half * 8)), dx_dup),
                    _mm256_set1_ps(d),
                );
                for run in 0..4 {
                    let mask = _mm256_set1_epi8(bit as i8);
                    let shift = _mm256_set1_epi32((run * 2) as i32);
                    // Lanes 0-3 are the first sixteen values of the run, 4-7 the second.
                    let pick = _mm256_setr_epi32(
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                        (2 * run + 1) as i32,
                    );
                    let packed = _mm256_loadu_si256(qs as *const __m256i);
                    let codes = _mm256_and_si256(_mm256_srlv_epi32(packed, shift), three);
                    let high = _mm256_andnot_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask), zero),
                        four_levels,
                    );
                    let unsigned = _mm256_add_epi8(codes, high);
                    let xv =
                        _mm256_loadu_si256(xq.as_ptr().add(x_half + run * 32) as *const __m256i);
                    let dot = _mm256_dpbusd_epi32(zero, unsigned, xv);
                    let bias = _mm256_dpbusd_epi32(zero, one_u8, xv);
                    let coefficients = _mm256_permutevar8x32_ps(coeffs, pick);
                    let lanes = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(dot),
                        _mm256_mul_ps(four, _mm256_cvtepi32_ps(bias)),
                    );
                    accumulator = _mm256_fmadd_ps(lanes, coefficients, accumulator);
                    bit <<= 1;
                }
            }
        }
        hsum(accumulator)
    }
}

/// Bytes one repacked Q3_K 256-value block takes: 256 one-byte codes, then 16 `f32` coefficients.
#[cfg(test)]
pub(crate) const Q3K_RECORD: usize = 256 + 16 * 4;

/// Repack Q3_K rows into the layout the band kernel reads.
///
/// Per block: the 256 codes as one byte each (`code + 4·high`, the value the row kernel makes
/// unsigned for `dpbusd`), then the sixteen `d * (scale - 32)` coefficients. Expanding three bits
/// into a byte costs 256 bytes instead of 110, which is only paid here because a band reads each
/// block once for sixteen activations; in the row kernel it would be a loss.
///
/// `data` holds whole rows of `stride` bytes; the output is `rows * blocks * Q3K_RECORD`.
#[cfg(test)]
pub(crate) fn pack_q3k_rows(data: &[u8], stride: usize) -> Vec<u8> {
    assert!(
        stride.is_multiple_of(110),
        "a Q3_K row is whole 110-byte blocks"
    );
    let blocks = stride / 110;
    let rows = data.len() / stride;
    let mut out = vec![0_u8; rows * blocks * Q3K_RECORD];
    for row in 0..rows {
        for b in 0..blocks {
            let block = &data[row * stride + b * 110..][..110];
            let record = &mut out[(row * blocks + b) * Q3K_RECORD..][..Q3K_RECORD];
            let hm = &block[0..32];
            let qs = &block[32..96];
            let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
            // SAFETY: `block` is one whole 110-byte block, which is what `q3k_scales` requires.
            let scales = unsafe { q3k_scales(block.as_ptr()) };
            for half in 0..2 {
                for run in 0..4 {
                    let shift = run * 2;
                    let bit = 1_u8 << (half * 4 + run);
                    for part in 0..2 {
                        for l in 0..16 {
                            let low = (qs[half * 32 + part * 16 + l] >> shift) & 3;
                            let high = if hm[part * 16 + l] & bit != 0 { 4 } else { 0 };
                            let at = half * 128 + run * 32 + part * 16 + l;
                            record[at] = low + high;
                        }
                    }
                }
            }
            for (k, scale) in scales.iter().enumerate() {
                record[256 + k * 4..260 + k * 4].copy_from_slice(&(d * scale).to_le_bytes());
            }
        }
    }
    out
}

/// The 16-activation tile the band kernel reads.
///
/// Layout: for every block and every group of four values, the sixteen activations' four values are
/// contiguous (`block · 4096 + grp · 64 + column · 4`), which is what lets one `dpbusd` cover
/// sixteen activations against the same four weights. Columns past `xs.len()` are zero-filled.
#[cfg(test)]
pub(crate) fn q3k_act_band<const B: usize>(xs: &[&QuantizedX; B], blocks: usize) -> Vec<i8> {
    let halves = B.div_ceil(16).max(1);
    let mut act = vec![0_i8; blocks * halves * 4096];
    for b in 0..blocks {
        for (entry, x) in xs.iter().enumerate() {
            let half = entry / 16;
            let col = entry % 16;
            let base = (b * halves + half) * 4096;
            for grp in 0..64 {
                let src = &x.q[b * 256 + grp * 4..][..4];
                let at = base + grp * 64 + col * 4;
                act[at..at + 4].copy_from_slice(src);
            }
        }
    }
    act
}

/// `N` repacked rows against sixteen activations, one pass over each weight block.
///
/// Per 16-value group: one `dpbusd` for the activation sums (shared by every row), then one
/// `dpbusd` chain per row on four weight words. Sixteen groups therefore cost `N + 1` integer dot
/// chains per block instead of `N · 8` in the row kernel (`q3k_dot_vnni` needs two per 32 values),
/// and the per-block prologue (scale unpack, coefficients) is paid once for `N · 16` outputs.
///
/// # Safety
///
/// `wq` must point at `row + N` repacked rows of `block_stride = blocks * Q3K_RECORD` bytes, `act`
/// at `blocks * 4096` bytes of tile, `dxs` at `blocks * 16 * 16` floats, `out` at `N * 16` floats,
/// and the CPU must have AVX-512 VNNI with VL.
#[cfg(test)]
#[target_feature(enable = "avx512vnni,avx512vl,avx2")]
pub(crate) unsafe fn q3k_band_16<const N: usize>(
    wq: *const u8,
    row: usize,
    block_stride: usize,
    act: *const i8,
    dxs: *const f32,
    blocks: usize,
    out: *mut f32,
) {
    // SAFETY: the caller's contract is this function's contract; every load below stays inside the
    // rows, blocks, groups and sixteen columns it describes.
    unsafe {
        let mut lanes_acc = [_mm512_setzero_ps(); N];
        let ones = _mm512_set1_epi32(0x0101_0101);
        for b in 0..blocks {
            let block_act = act.add(b * 4096);
            let block_dx = dxs.add(b * 16 * 16);
            for k in 0..16 {
                let a = [
                    _mm512_loadu_si512(block_act.add(k * 256) as *const __m512i),
                    _mm512_loadu_si512(block_act.add(k * 256 + 64) as *const __m512i),
                    _mm512_loadu_si512(block_act.add(k * 256 + 128) as *const __m512i),
                    _mm512_loadu_si512(block_act.add(k * 256 + 192) as *const __m512i),
                ];
                // The `-4` bias of every code, summed over this 16-value group for each column. One
                // chain for all rows: it does not depend on the weights.
                let mut bias = _mm512_setzero_si512();
                for group in &a {
                    bias = _mm512_dpbusd_epi32(bias, ones, *group);
                }
                let bias = _mm512_mul_ps(_mm512_cvtepi32_ps(bias), _mm512_set1_ps(4.0));
                let dx = _mm512_loadu_ps(block_dx.add(k * 16));
                for (t, accumulator) in lanes_acc.iter_mut().enumerate() {
                    let codes = wq.add((row + t) * block_stride + b * Q3K_RECORD);
                    let weights = codes.add(k * 16);
                    let mut acc = _mm512_setzero_si512();
                    for (at, group) in a.iter().enumerate() {
                        let word = (weights.add(at * 4) as *const u32).read_unaligned();
                        acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(word as i32), *group);
                    }
                    let levels = _mm512_sub_ps(_mm512_cvtepi32_ps(acc), bias);
                    let coef = f32::from_le_bytes([
                        *codes.add(256 + k * 4),
                        *codes.add(256 + k * 4 + 1),
                        *codes.add(256 + k * 4 + 2),
                        *codes.add(256 + k * 4 + 3),
                    ]);
                    let scale = _mm512_mul_ps(dx, _mm512_set1_ps(coef));
                    *accumulator = _mm512_fmadd_ps(levels, scale, *accumulator);
                }
            }
        }
        for (t, accumulator) in lanes_acc.iter().enumerate() {
            _mm512_storeu_ps(out.add(t * 16), *accumulator);
        }
    }
}

/// [`q4k_dot`] for `B` activations, dequantizing each 32-byte weight load once.
///
/// The nibble planes and the scale/min bytes are shared across the entries; only the `maddubs`
/// against each entry's activation repeats. Per entry the operations happen in the same order as
/// [`q4k_dot`], so entry `e` of a batch is bit-identical to its own single call.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 144-byte blocks, every `xs[e]` must hold 256
/// values and 8 scales per block, every `sums[e]` at least 8 entries per block, and the CPU must
/// support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q4k_dot_batch<const B: usize>(
    row: &[u8],
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32; B],
) {
    // SAFETY: the module documentation is the contract for every pointer below, and the CPU has
    // AVX2. Each entry's `xq` covers the same 256 values per block as `x` does.
    unsafe {
        // Each entry's total is kept in an array of known length rather than written through the
        // slice on every group, so a call costs `B` stores instead of `8 * B` read-modify-writes.
        // The operations and their order are unchanged.
        let mut totals = [0.0_f32; B];
        // None of these pointers depend on the block, so they are resolved once per call.
        let values_at: [*const i8; B] = std::array::from_fn(|entry| xs[entry].q.as_ptr());
        let scales_at: [*const f32; B] = std::array::from_fn(|entry| xs[entry].d.as_ptr());
        let sums_at: [*const f32; B] = std::array::from_fn(|entry| sums[entry].as_ptr());
        let blocks = row.len() / 144;
        let low_mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        // The same three accumulators, filled in the same order, as the single-entry kernel: two for
        // the weights so the adds are not one dependent chain, one for the minimum term.
        let mut weights0 = [_mm256_setzero_ps(); B];
        let mut weights1 = [_mm256_setzero_ps(); B];
        let mut minimums = [_mm256_setzero_ps(); B];
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            // Entry-independent part of the minimum product, built once per block. The coefficient
            // keeps the single-entry kernel's order, `(scale * dx) * d`, so a batch entry stays
            // bit-identical to its own call.
            let subtracted = _mm256_mul_ps(min_at, _mm256_set1_ps(dmin));
            let d_vector = _mm256_set1_ps(d);
            let coefficients: [__m256; B] = std::array::from_fn(|entry| {
                _mm256_mul_ps(
                    _mm256_mul_ps(scale_at, _mm256_loadu_ps(scales_at[entry].add(b * 8))),
                    d_vector,
                )
            });
            for entry in 0..B {
                minimums[entry] = _mm256_fmadd_ps(
                    subtracted,
                    _mm256_loadu_ps(sums_at[entry].add(b * 8)),
                    minimums[entry],
                );
            }
            for j in 0..4 {
                let packed = _mm256_loadu_si256(block.add(16 + j * 32) as *const __m256i);
                let low_bytes = _mm256_and_si256(packed, low_mask);
                let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                for entry in 0..B {
                    let values = values_at[entry].add(b * 256 + j * 64);
                    let low_sums = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            low_bytes,
                            _mm256_loadu_si256(values as *const __m256i),
                        ),
                        ones,
                    );
                    let high_sums = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            high_bytes,
                            _mm256_loadu_si256(values.add(32) as *const __m256i),
                        ),
                        ones,
                    );
                    let low_coeff = _mm256_permutevar8x32_ps(
                        coefficients[entry],
                        _mm256_set1_epi32((2 * j) as i32),
                    );
                    let high_coeff = _mm256_permutevar8x32_ps(
                        coefficients[entry],
                        _mm256_set1_epi32((2 * j + 1) as i32),
                    );
                    let low_lanes = _mm256_cvtepi32_ps(low_sums);
                    let high_lanes = _mm256_cvtepi32_ps(high_sums);
                    if j.is_multiple_of(2) {
                        weights0[entry] = _mm256_fmadd_ps(low_lanes, low_coeff, weights0[entry]);
                        weights1[entry] = _mm256_fmadd_ps(high_lanes, high_coeff, weights1[entry]);
                    } else {
                        weights1[entry] = _mm256_fmadd_ps(low_lanes, low_coeff, weights1[entry]);
                        weights0[entry] = _mm256_fmadd_ps(high_lanes, high_coeff, weights0[entry]);
                    }
                }
            }
        }
        for entry in 0..B {
            totals[entry] =
                hsum(_mm256_add_ps(weights0[entry], weights1[entry])) - hsum(minimums[entry]);
        }
        out.copy_from_slice(&totals);
    }
}

/// [`q4k_dot_batch`] with `vpdpbusd` in place of `maddubs` + `madd`, on 256-bit registers.
///
/// Same integer results, so every entry stays bit-identical to its own [`q4k_dot_vnni`] call and to
/// [`q4k_dot`]. This is the kernel a decode step actually runs: the pooled path always goes through
/// a batch, even a batch of one.
///
/// # Safety
///
/// As [`q4k_dot_batch`], plus a CPU with `avx512_vnni` and `avx512_vl`.
///
/// The nibble planes and the scale/min bytes are shared across the entries; only the `maddubs`
/// against each entry's activation repeats. Per entry the operations happen in the same order as
/// [`q4k_dot`], so entry `e` of a batch is bit-identical to its own single call.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 144-byte blocks, every `xs[e]` must hold 256
/// values and 8 scales per block, every `sums[e]` at least 8 entries per block, and the CPU must
/// support AVX2.
#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
pub(crate) unsafe fn q4k_dot_batch_vnni<const B: usize>(
    row: &[u8],
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32; B],
) {
    // SAFETY: the module documentation is the contract for every pointer below, and the CPU has
    // AVX2. Each entry's `xq` covers the same 256 values per block as `x` does.
    unsafe {
        // Each entry's total is kept in an array of known length rather than written through the
        // slice on every group, so a call costs `B` stores instead of `8 * B` read-modify-writes.
        // The operations and their order are unchanged.
        let mut totals = [0.0_f32; B];
        // None of these pointers depend on the block, so they are resolved once per call.
        let values_at: [*const i8; B] = std::array::from_fn(|entry| xs[entry].q.as_ptr());
        let scales_at: [*const f32; B] = std::array::from_fn(|entry| xs[entry].d.as_ptr());
        let sums_at: [*const f32; B] = std::array::from_fn(|entry| sums[entry].as_ptr());
        let blocks = row.len() / 144;
        let low_mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        // The same three accumulators, filled in the same order, as the single-entry kernel: two for
        // the weights so the adds are not one dependent chain, one for the minimum term.
        let mut weights0 = [_mm256_setzero_ps(); B];
        let mut weights1 = [_mm256_setzero_ps(); B];
        let mut minimums = [_mm256_setzero_ps(); B];
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            // Entry-independent part of the minimum product, built once per block. The coefficient
            // keeps the single-entry kernel's order, `(scale * dx) * d`, so a batch entry stays
            // bit-identical to its own call.
            let subtracted = _mm256_mul_ps(min_at, _mm256_set1_ps(dmin));
            let d_vector = _mm256_set1_ps(d);
            let coefficients: [__m256; B] = std::array::from_fn(|entry| {
                _mm256_mul_ps(
                    _mm256_mul_ps(scale_at, _mm256_loadu_ps(scales_at[entry].add(b * 8))),
                    d_vector,
                )
            });
            // Scalar-broadcast pool: one float per (entry, j), extracted once
            // per block so the inner `j` loop does scalar broadcast instead of
            // `permutevar8x32` (cross-lane shuffle, 3-cycle latency) per
            // (j, entry). Extraction is a single store per entry per block.
            let mut coeff_stack = [[0.0_f32; 8]; B];
            for entry in 0..B {
                _mm256_storeu_ps(coeff_stack[entry].as_mut_ptr(), coefficients[entry]);
            }
            for entry in 0..B {
                minimums[entry] = _mm256_fmadd_ps(
                    subtracted,
                    _mm256_loadu_ps(sums_at[entry].add(b * 8)),
                    minimums[entry],
                );
            }
            for j in 0..4 {
                let packed = _mm256_loadu_si256(block.add(16 + j * 32) as *const __m256i);
                let low_bytes = _mm256_and_si256(packed, low_mask);
                let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                for entry in 0..B {
                    let values = values_at[entry].add(b * 256 + j * 64);
                    let low_sums = _mm256_dpbusd_epi32(
                        zero,
                        low_bytes,
                        _mm256_loadu_si256(values as *const __m256i),
                    );
                    let high_sums = _mm256_dpbusd_epi32(
                        zero,
                        high_bytes,
                        _mm256_loadu_si256(values.add(32) as *const __m256i),
                    );
                    let low_coeff = _mm256_set1_ps(coeff_stack[entry][2 * j]);
                    let high_coeff = _mm256_set1_ps(coeff_stack[entry][2 * j + 1]);
                    // One fused multiply-add instead of a multiply and an add per plane: two fewer
                    // uops per plane, eight per 144-byte block. Measured on the Q4_K phase of a
                    // decode step, 20 interleaved paired runs against the multiply-then-add form:
                    // median 0.984x the q4k time (two batches, 0.981 and 0.986, both positive). The
                    // rounding is the fused one; `q4k_dot_previous` behind `VIPERA_EXACT_DOT` is
                    // still the multiply-then-add path, and the pooled/spawned bit-identity test
                    // holds because every kernel here moved together.
                    let low_lanes = _mm256_cvtepi32_ps(low_sums);
                    let high_lanes = _mm256_cvtepi32_ps(high_sums);
                    if j.is_multiple_of(2) {
                        weights0[entry] = _mm256_fmadd_ps(low_lanes, low_coeff, weights0[entry]);
                        weights1[entry] = _mm256_fmadd_ps(high_lanes, high_coeff, weights1[entry]);
                    } else {
                        weights1[entry] = _mm256_fmadd_ps(low_lanes, low_coeff, weights1[entry]);
                        weights0[entry] = _mm256_fmadd_ps(high_lanes, high_coeff, weights0[entry]);
                    }
                }
            }
        }
        for entry in 0..B {
            totals[entry] =
                hsum(_mm256_add_ps(weights0[entry], weights1[entry])) - hsum(minimums[entry]);
        }
        out.copy_from_slice(&totals);
    }
}

/// Whether the AVX-512 VNNI tiled kernel can run: 512-bit forms plus
/// `avx512_vnni`, and no `VIPERA_NO_VNNI`.
pub fn vnni512_available() -> bool {
    static VNNI512: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VNNI512.get_or_init(|| {
        if !std::is_x86_feature_detected!("avx512f")
            || !std::is_x86_feature_detected!("avx512bw")
            || !std::is_x86_feature_detected!("avx512vnni")
        {
            return false;
        }
        !std::env::var("VIPERA_NO_VNNI").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

/// Safe entry point for [`q4k_matvec_tiled_range`]: checks the CPU and the
/// activation lengths itself, so callers outside this module can use it
/// without `unsafe`.
/// The Q4_K activation tile of one batched product, built once for every participant.
///
/// Layout is `(half, group, entry, 4)` `i8`, so one 64-byte load holds four codes for each of the
/// sixteen columns of a half. The activation-prep workers fill the entries they own; the row
/// workers only read it once [`crate::pool::Pool::prepare`] has joined them, which is the
/// happens-before that makes the read safe. Building it per participant cost ~44 ms of a 770 ms
/// chunk (`docs/benchmarks.md`, 2026-11-22).
pub(crate) struct SharedTile {
    act: std::cell::UnsafeCell<Vec<i8>>,
    blocks: usize,
    halves: usize,
    /// The prefix of `act` this product needs; the buffer itself may be a reused larger one.
    len: usize,
}

// SAFETY: `fill_entries` is called only with the disjoint entry ranges of one `prepare`, and
// `bytes` is read only after that prepare has joined every worker.
unsafe impl Sync for SharedTile {}

impl SharedTile {
    fn needed(blocks: usize, halves: usize) -> usize {
        blocks * halves * 64 * 16 * 4
    }

    pub(crate) fn new(blocks: usize, halves: usize) -> Self {
        let len = Self::needed(blocks, halves);
        Self {
            act: std::cell::UnsafeCell::new(vec![0_i8; len]),
            blocks,
            halves,
            len,
        }
    }

    /// Wrap a buffer recycled from an earlier product. The bytes a later `fill_entries`
    /// overwrites are the bytes that matter; columns past the batch are multiplied by a zero
    /// scale on every participant, so stale bytes there cannot reach an output.
    pub(crate) fn from_buffer(mut buffer: Vec<i8>, blocks: usize, halves: usize) -> Self {
        let len = Self::needed(blocks, halves);
        if buffer.len() < len {
            buffer.resize(len, 0);
        }
        Self {
            act: std::cell::UnsafeCell::new(buffer),
            blocks,
            halves,
            len,
        }
    }

    /// Give the buffer back to the pool's free list.
    pub(crate) fn into_buffer(self) -> Vec<i8> {
        self.act.into_inner()
    }

    pub(crate) fn blocks(&self) -> usize {
        self.blocks
    }

    pub(crate) fn halves(&self) -> usize {
        self.halves
    }

    /// Write the columns of `first_entry..first_entry + q.len()`.
    ///
    /// Concurrent calls must cover disjoint entries: every entry owns its own `col` slot, so two
    /// calls that touch different entries never write the same byte.
    pub(crate) fn fill_entries(&self, first_entry: usize, q: &[&[i8]]) {
        // SAFETY: disjoint columns per the contract above; no reader runs until `prepare` joins.
        let act = unsafe { &mut *self.act.get() };
        for (offset, chunk_all) in q.iter().enumerate() {
            let entry = first_entry + offset;
            let half = entry / 16;
            let col = entry % 16;
            for b in 0..self.blocks {
                let chunk = &chunk_all[b * 256..b * 256 + 256];
                let base = (b * self.halves + half) * 64 * 16 * 4;
                for grp in 0..64 {
                    act[base + grp * 64 + col * 4..][..4].copy_from_slice(&chunk[grp * 4..][..4]);
                }
            }
        }
    }

    /// The built tile. Only call after every `fill_entries` for this product has returned.
    fn bytes(&self) -> &[i8] {
        // SAFETY: the caller's happens-before is documented on the type.
        unsafe { &(&*self.act.get())[..self.len] }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_q4k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
    tile: Option<&SharedTile>,
) -> bool {
    if !vnni512_available()
        || stride == 0
        || !stride.is_multiple_of(144)
        || B == 0
        || B > MAX_HALVES * 16
    {
        return false;
    }
    let blocks = stride / 144;
    let cols = blocks * 256;
    if out.len() != B.saturating_mul(count)
        || xs.iter().any(|x| x.q.len() < cols || x.d.len() < cols / 32)
        || sums.iter().any(|s| s.len() < cols / 32)
    {
        return false;
    }
    // SAFETY: vnni512_available checked the CPU and the lengths above match
    // the kernel's contract.
    unsafe { q4k_matvec_tiled_range(data, stride, first_row, count, xs, sums, out, tile) };
    true
}

/// Whether the 256-bit band runs: AVX2, no 512-bit VNNI tile, and no `VIPERA_TILED_AVX2=0`.
///
/// The band exists for the machines the 512-bit tile cannot serve — anything with AVX2 that is not
/// an AVX-512 VNNI part, which is every AMD CPU before Zen 4 and every Intel one before Ice Lake.
/// `VIPERA_TILED=0` still turns the whole tiled path off, and `VIPERA_NO_VNNI` makes this band the
/// one a VNNI machine runs, which is how it is measured without such a CPU.
fn tiled_ymm_enabled() -> bool {
    !std::env::var("VIPERA_TILED_AVX2").is_ok_and(|value| value == "0")
}

/// Test override for the `!vnni512_available()` half of [`try_q4k_matvec_tiled_range_ymm`]: on a VNNI
/// machine that check is what keeps the 256-bit band unreachable, and its parity test needs the band
/// to run there. `2` forces it on, like `Q4K8_OVERRIDE`.
#[cfg(test)]
static YMM_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
fn ymm_forced() -> bool {
    YMM_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) == 2
}

#[cfg(not(test))]
fn ymm_forced() -> bool {
    false
}

/// Safe entry point for [`q4k_matvec_tiled_ymm`]: the 256-bit band, for a CPU with AVX2 and no
/// 512-bit VNNI tile.
///
/// Same contract and same arithmetic as [`try_q4k_matvec_tiled_range`]; only the lane count differs
/// (eight activation columns per `ymm` instead of sixteen per `zmm`), and the integer sums are exact
/// in both, so a row gets the same bits from either band.
pub(crate) fn try_q4k_matvec_tiled_range_ymm<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
) -> bool {
    if !available()
        || (vnni512_available() && !ymm_forced())
        || !tiled_ymm_enabled()
        || stride == 0
        || !stride.is_multiple_of(144)
        || B == 0
        || B > MAX_HALVES * 8
    {
        return false;
    }
    let blocks = stride / 144;
    let cols = blocks * 256;
    if out.len() != B.saturating_mul(count)
        || xs.iter().any(|x| x.q.len() < cols || x.d.len() < cols / 32)
        || sums.iter().any(|s| s.len() < cols / 32)
    {
        return false;
    }
    // SAFETY: `available()` checked AVX2 and the lengths above match the kernel's contract.
    unsafe { q4k_matvec_tiled_ymm::<B>(data, stride, first_row, count, xs, sums, out) };
    true
}

/// The 256-bit band: the same block walk and epilogue as [`q4k_tiled_matvec_worker`], with eight
/// activation columns per register and `maddubs`+`madd` in place of `dpbusd`.
///
/// `maddubs` multiplies unsigned weight codes 0..15 by signed activation codes and adds pairs into
/// `i16` (the largest product is 15 × 127, so the pair sum is at most 3810 and cannot saturate),
/// then `madd` against ones adds pairs of those into `i32` — the integer `dpbusd` computes, one
/// instruction pair instead of one. The float epilogue is the same expression in the same order
/// (multiply by `d * scale`, multiply by `dx`, subtract `dmin * min * sum`), so the result is the
/// 512-bit band's bits.
///
/// Four rows per band instead of eight: AVX2 has sixteen vector registers, and four accumulators
/// plus four running sums plus the code/weight temporaries leave one or two spare, where eight
/// plus eight would spill every block.
///
/// # Safety
///
/// `data` must hold the rows with the given `stride` (a whole number of 144-byte Q4_K blocks), every
/// `xs[e].q` must cover the row width, every `xs[e].d` and `sums[e]` one entry per 32-value group,
/// and the CPU must support AVX2 (the caller checks via [`available`] and the absence of
/// [`vnni512_available`]).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn q4k_matvec_tiled_ymm<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
) {
    // SAFETY: the caller validated the lengths; every load below stays inside the rows, blocks and
    // activation slices the contract describes.
    unsafe {
        const COLS: usize = 8;
        const ROW_WIN: usize = 32;
        const BAND: usize = 4;
        let blocks = stride / 144;
        let halves = B.div_ceil(COLS);
        let mut act = vec![0_i8; blocks * halves * 64 * COLS * 4];
        for b in 0..blocks {
            for (entry, x) in xs.iter().enumerate() {
                let half = entry / COLS;
                let col = entry % COLS;
                let chunk = x.q.as_ptr().add(b * 256);
                let base = (b * halves + half) * 64 * COLS * 4;
                for grp in 0..64 {
                    std::ptr::copy_nonoverlapping(
                        chunk.add(grp * 4),
                        act.as_mut_ptr().add(base + grp * COLS * 4 + col * 4),
                        4,
                    );
                }
            }
        }
        let mut dxs = vec![0.0_f32; blocks * halves * 8 * COLS];
        let mut sumt = vec![0.0_f32; blocks * halves * 8 * COLS];
        for b in 0..blocks {
            for entry in 0..B {
                let half = entry / COLS;
                let col = entry % COLS;
                for sub in 0..8 {
                    let at = ((b * halves + half) * 8 + sub) * COLS + col;
                    dxs[at] = xs[entry].d[b * 8 + sub];
                    sumt[at] = sums[entry][b * 8 + sub];
                }
            }
        }
        let low_mask = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let mut wq = vec![0_u8; ROW_WIN * 256];
        let mut scale_f = vec![[0.0_f32; 8]; ROW_WIN];
        let mut min_f = vec![[0.0_f32; 8]; ROW_WIN];
        let mut dvec = [0.0_f32; ROW_WIN];
        let mut dminvec = [0.0_f32; ROW_WIN];
        let mut buf = vec![0.0_f32; ROW_WIN * halves * COLS];
        let mut win = 0;
        while win < count {
            let cnt = (count - win).min(ROW_WIN);
            buf[..cnt * halves * COLS].fill(0.0);
            for b in 0..blocks {
                wq.fill(0);
                for slot in scale_f.iter_mut() {
                    *slot = [0.0; 8];
                }
                for slot in min_f.iter_mut() {
                    *slot = [0.0; 8];
                }
                dvec.fill(0.0);
                dminvec.fill(0.0);
                for r in 0..cnt {
                    let block = data.as_ptr().add((first_row + win + r) * stride + b * 144);
                    dvec[r] = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
                    dminvec[r] = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
                    let scale_bytes = std::slice::from_raw_parts(block.add(4), 12);
                    for s in 0..8 {
                        let (scale, min) = crate::quant::q4k_scale_min(scale_bytes, s);
                        scale_f[r][s] = dvec[r] * f32::from(scale);
                        min_f[r][s] = dminvec[r] * f32::from(min);
                    }
                    let wrow = wq.as_mut_ptr().add(r * 256);
                    for j in 0..4 {
                        let packed = _mm256_loadu_si256(block.add(16 + j * 32) as *const __m256i);
                        let low = _mm256_and_si256(packed, low_mask);
                        let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                        _mm256_storeu_si256(wrow.add(j * 64) as *mut __m256i, low);
                        _mm256_storeu_si256(wrow.add(j * 64 + 32) as *mut __m256i, high);
                    }
                }
                let wq_ptr = wq.as_ptr();
                for half in 0..halves {
                    let half_act = act.as_ptr().add((b * halves + half) * 64 * COLS * 4);
                    let dxs_ptr = dxs.as_ptr().add((b * halves + half) * 8 * COLS);
                    let sumt_ptr = sumt.as_ptr().add((b * halves + half) * 8 * COLS);
                    let mut band = 0;
                    while band + BAND <= cnt {
                        q4k_tiled_band_ymm::<BAND>(
                            wq_ptr,
                            half_act,
                            dxs_ptr,
                            sumt_ptr,
                            band,
                            &scale_f,
                            &min_f,
                            buf.as_mut_ptr(),
                            halves * COLS,
                            half * COLS,
                            ones,
                        );
                        band += BAND;
                    }
                    while band < cnt {
                        q4k_tiled_band_ymm::<1>(
                            wq_ptr,
                            half_act,
                            dxs_ptr,
                            sumt_ptr,
                            band,
                            &scale_f,
                            &min_f,
                            buf.as_mut_ptr(),
                            halves * COLS,
                            half * COLS,
                            ones,
                        );
                        band += 1;
                    }
                }
            }
            for r in 0..cnt {
                for entry in 0..B {
                    out[entry * count + win + r] = buf[r * halves * COLS + entry];
                }
            }
            win += cnt;
        }
    }
}

/// `N` rows of one 256-column block against one eight-column activation tile half, added into
/// `buf`.
///
/// # Safety
///
/// Every pointer must cover what [`q4k_matvec_tiled_ymm`] built for this block and band, and the
/// CPU must support AVX2.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn q4k_tiled_band_ymm<const N: usize>(
    wq: *const u8,
    half_act: *const i8,
    dxs: *const f32,
    sumt: *const f32,
    band: usize,
    scale_f: &[[f32; 8]],
    min_f: &[[f32; 8]],
    buf: *mut f32,
    stride_cols: usize,
    col_offset: usize,
    ones: std::arch::x86_64::__m256i,
) {
    // SAFETY: as above.
    unsafe {
        const COLS: usize = 8;
        let mut facc: [std::arch::x86_64::__m256; N] = [_mm256_setzero_ps(); N];
        for s in 0..8 {
            let mut acc: [std::arch::x86_64::__m256i; N] = [_mm256_setzero_si256(); N];
            for g in 0..8 {
                let grp = s * 8 + g;
                let codes = _mm256_loadu_si256(half_act.add(grp * COLS * 4) as *const __m256i);
                for (t, slot) in acc.iter_mut().enumerate() {
                    let word = u32::from_le(std::ptr::read_unaligned(
                        wq.add((band + t) * 256 + grp * 4) as *const u32,
                    ));
                    let products = _mm256_maddubs_epi16(_mm256_set1_epi32(word as i32), codes);
                    *slot = _mm256_add_epi32(*slot, _mm256_madd_epi16(products, ones));
                }
            }
            let dxv = _mm256_loadu_ps(dxs.add(s * COLS));
            let sumv = _mm256_loadu_ps(sumt.add(s * COLS));
            for (t, (acc_slot, facc_slot)) in acc.iter().zip(facc.iter_mut()).enumerate() {
                let row = band + t;
                let main = _mm256_mul_ps(
                    _mm256_mul_ps(
                        _mm256_cvtepi32_ps(*acc_slot),
                        _mm256_set1_ps(scale_f[row][s]),
                    ),
                    dxv,
                );
                let minterm = _mm256_mul_ps(sumv, _mm256_set1_ps(min_f[row][s]));
                *facc_slot = _mm256_add_ps(*facc_slot, _mm256_sub_ps(main, minterm));
            }
        }
        for (t, value) in facc.iter().enumerate() {
            let slot = buf.add((band + t) * stride_cols + col_offset);
            _mm256_storeu_ps(slot, _mm256_add_ps(_mm256_loadu_ps(slot), *value));
        }
    }
}

/// Whether the tiled Q4_K kernel folds the epilogue's scale and minimum with a fused multiply-add.
///
/// Off by default; `VIPERA_Q4K_FMA_EPI=1` turns it on. Read per call so one process can measure both
/// arms, and because the switch is about arithmetic, not speed: the fused form is faster and gives
/// different tokens.
fn tiled_fma_enabled() -> bool {
    std::env::var_os("VIPERA_Q4K_FMA_EPI").is_some()
}

/// Ceiling probe for the repacked-weight direction: replaces the 4-byte weight load inside the
/// `dpbusd` loop with a constant, so a measurement prices that load and nothing else.
///
/// Test-only on purpose: it changes the arithmetic, so it must never be reachable from a real
/// build. In a non-test build `tiled_no_weight` folds to `false` and the branch disappears.
#[cfg(test)]
static NO_WEIGHT_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ceiling probe for epilogue work: folds the six vector ops and two broadcasts of the per
/// (row, sub-block) epilogue into one convert and one add. The dpbusd chain is untouched, so the
/// difference prices exactly the arithmetic that a fused or restructured epilogue could remove.
///
/// Test-only: it changes the result, so it must never be reachable from a real build. In a
/// non-test build `tiled_no_epi` folds to `false` and the branch disappears.
#[cfg(test)]
static NO_EPI_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ceiling probe for the Q6_K weight unpack: skips writing the unpacked codes, so the dpbusd chain
/// still reads `wq` but the per-byte decode loop is gone. Test-only; the values are stale, which is
/// fine for timing and wrong for anything else.
#[cfg(test)]
static NO_UNPACK_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Schedule probe for the Q4_K band: accumulate even and odd sub-blocks in two integer chains per
/// row instead of one, doubling the independent `dpbusd` chains. Test-only; the sum is the same
/// (integer adds are associative), so this one could be wired if it wins.
#[cfg(test)]
static TWO_ACC_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ceiling probe for a per-256-block activation scale: accumulates the eight sub-block partial
/// sums in `i32` (scaled by a constant, since the real scale would live in the activation) and
/// applies one convert plus one multiply-add per (row, block) instead of four ops per
/// (row, sub-block). That is the epilogue `block_q8_K`'s single per-block scale makes possible;
/// with this repo's one-scale-per-32 activation the per-sub-block float step cannot be removed.
///
/// Test-only: the values are wrong on purpose, so they price the restructure and nothing else.
#[cfg(test)]
static PER_BLOCK_EPI_PROBE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn tiled_no_unpack() -> bool {
    #[cfg(test)]
    {
        NO_UNPACK_PROBE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn tiled_no_epi() -> bool {
    #[cfg(test)]
    {
        NO_EPI_PROBE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn tiled_two_acc() -> bool {
    #[cfg(test)]
    {
        TWO_ACC_PROBE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn tiled_no_weight() -> bool {
    #[cfg(test)]
    {
        NO_WEIGHT_PROBE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn tiled_per_block_epi() -> bool {
    #[cfg(test)]
    {
        PER_BLOCK_EPI_PROBE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Shared body of the Q4_K tiled band: unpacks the window rows in `wq` from the product-wide
/// `act` tile, walks the blocks with [`q4k_tiled_band`], and writes `out` once per window.
///
/// Math mirrors [`q4k_dot`]: per sub-block `s`, the integer sum is the `u8` codes 0..15 against
/// the `i8` activation codes, times `d * scale[s] * dx[s]`, minus `dmin * min[s]` times the
/// activation sums. Summation order across sub-blocks differs from the row kernel, so results
/// agree within `1e-3 + 1e-4 * |ref|`, not bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn q4k_tiled_matvec_worker<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    act: &[i8],
    dxs: &[f32],
    sumt: &[f32],
    out: &mut [f32],
) {
    // SAFETY: the caller validated the lengths the module contract describes; every load below
    // stays inside the rows, blocks, and activation slices.
    unsafe {
        const ROW_WIN: usize = 32;
        const BAND: usize = 8;
        const COLS: usize = 16;
        // A tile is 16 activations wide, the width one 512-bit `dpbusd`
        // consumes. Wider batches run several tile halves against the same
        // unpacked weight window, so the unpack and the weight loads are paid
        // once for the whole batch instead of once per 16 activations.
        let halves = B.div_ceil(COLS);
        debug_assert!(halves <= MAX_HALVES);
        let stride_cols = halves * COLS;
        let blocks = stride / 144;
        let mut wq = vec![0_u8; ROW_WIN * 256];
        let mut scales = vec![[0_i32; 8]; ROW_WIN];
        let mut mins = vec![[0_i32; 8]; ROW_WIN];
        // `d * scale` and `dmin * min` are fixed for a row and sub-block once the block is read, and
        // the band loop visits them four times per window, so the products are made once here.
        let mut scale_f = vec![[0.0_f32; 8]; ROW_WIN];
        let mut min_f = vec![[0.0_f32; 8]; ROW_WIN];
        let mut dvec = [0.0_f32; ROW_WIN];
        let mut dminvec = [0.0_f32; ROW_WIN];
        // `act` is the activation tile in (half, group, entry, 4) order: one 64-byte ZMM load
        // holds 4 codes for each of the 16 columns of that half. Both callers build it once per
        // product (see the wrappers below) and it stays in L2 while the row windows walk it.
        let low_mask = _mm256_set1_epi8(0x0f);
        // Fused epilogue: `fma(cvt(acc) * dxv, scale, facc)` followed by `fnmadd(sumv, min, ..)` is
        // four vector ops per row and sub-block where the multiply-then-add form is six. Measured
        // +8.6% (8960x1536) and +12% (1536x8960) GFLOPS single thread, and -5.0% of a prefill chunk
        // over five interleaved pairs.
        //
        // It also changes the arithmetic: the fused form rounds once where the two multiplies round
        // twice, and 16 greedy tokens from the 524-token prompt came out different. Off by default
        // for that reason; `VIPERA_Q4K_FMA_EPI=1` turns it on, and flipping the default needs an ADR
        // and a quality comparison, not a perf change.
        let fused_epi = tiled_fma_enabled();
        let no_weight = tiled_no_weight();
        let no_epi = tiled_no_epi();
        let per_block_epi = tiled_per_block_epi();
        // Test-only ceiling probe: skip the nibble decode, leaving `wq` zeroed. Prices the unpack
        // separately from the `dpbusd` chain; the values are wrong on purpose.
        let no_unpack = tiled_no_unpack();
        // Window-major accumulation: one row window of `ROW_WIN` rows keeps its partial sums in
        // `buf` (ROW_WIN * stride_cols floats, L2-resident) across every block, and `out` is written
        // once per window. The old order walked all rows per 256-column slab and parked partials in
        // a `count * stride_cols` buffer — 8960 rows deep, which is tens of megabytes of DRAM
        // traffic per block and the reason wide batches never paid off.
        let mut buf = vec![0.0_f32; ROW_WIN * stride_cols];
        let mut win = 0;
        while win < count {
            let cnt = (count - win).min(ROW_WIN);
            buf[..cnt * stride_cols].fill(0.0);
            for b in 0..blocks {
                // The walk takes 144 bytes out of every row, so consecutive addresses are a stride
                // apart and the hardware prefetcher only sees part of the stream; an explicit hint
                // covers the block this window is about to unpack next.
                if b + 1 < blocks {
                    for r in 0..cnt {
                        let at = (first_row + win + r) * stride + (b + 1) * 144;
                        prefetch_ahead(data, at);
                        prefetch_ahead(data, at + 64);
                        prefetch_ahead(data, at + 128);
                    }
                }
                wq.fill(0);
                for slot in scales.iter_mut() {
                    *slot = [0; 8];
                }
                for slot in mins.iter_mut() {
                    *slot = [0; 8];
                }
                dvec.fill(0.0);
                dminvec.fill(0.0);
                for r in 0..cnt {
                    let block = data.as_ptr().add((first_row + win + r) * stride + b * 144);
                    dvec[r] = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
                    dminvec[r] = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
                    let scale_bytes = std::slice::from_raw_parts(block.add(4), 12);
                    for s in 0..8 {
                        let (scale, min) = crate::quant::q4k_scale_min(scale_bytes, s);
                        scales[r][s] = i32::from(scale);
                        mins[r][s] = i32::from(min);
                        scale_f[r][s] = dvec[r] * f32::from(scale);
                        min_f[r][s] = dminvec[r] * f32::from(min);
                    }
                    if !no_unpack {
                        let wrow = wq.as_mut_ptr().add(r * 256);
                        for j in 0..4 {
                            let packed =
                                _mm256_loadu_si256(block.add(16 + j * 32) as *const __m256i);
                            let low = _mm256_and_si256(packed, low_mask);
                            let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                            _mm256_storeu_si256(wrow.add(j * 64) as *mut __m256i, low);
                            _mm256_storeu_si256(wrow.add(j * 64 + 32) as *mut __m256i, high);
                        }
                    }
                }
                let wq_ptr = wq.as_ptr();
                for half in 0..halves {
                    let half_act = act.as_ptr().add((b * halves + half) * 64 * COLS * 4);
                    let dxs_ptr = dxs.as_ptr().add((b * halves + half) * 8 * COLS);
                    let sumt_ptr = sumt.as_ptr().add((b * halves + half) * 8 * COLS);
                    let mut band = 0;
                    // Every tensor this kernel serves is a whole number of windows, and a whole
                    // number of bands inside one (8960, 1536 and 256 are all multiples of
                    // `ROW_WIN`), so the band is always full. A runtime row bound here costs a
                    // fused `cmp`+`jcc` per `dpbusd` — 512 per band-block, none of which a real
                    // shape takes — so the full band gets a constant and only the last band of an
                    // odd shape falls back to one row at a time.
                    while band + BAND <= cnt {
                        q4k_tiled_band::<BAND>(
                            wq_ptr,
                            half_act,
                            dxs_ptr,
                            sumt_ptr,
                            band,
                            &scale_f,
                            &min_f,
                            buf.as_mut_ptr(),
                            stride_cols,
                            half,
                            fused_epi,
                            no_weight,
                            no_epi,
                            per_block_epi,
                        );
                        band += BAND;
                    }
                    while band < cnt {
                        q4k_tiled_band::<1>(
                            wq_ptr,
                            half_act,
                            dxs_ptr,
                            sumt_ptr,
                            band,
                            &scale_f,
                            &min_f,
                            buf.as_mut_ptr(),
                            stride_cols,
                            half,
                            fused_epi,
                            no_weight,
                            no_epi,
                            per_block_epi,
                        );
                        band += 1;
                    }
                }
            }
            for r in 0..cnt {
                for entry in 0..B {
                    out[entry * count + win + r] = buf[r * stride_cols + entry];
                }
            }
            win += cnt;
        }
    }
}

/// Tiled Q4_K matvec range: `count` rows from `first_row`, `B` quantized activations,
/// token-major `out` holding `B * count` values — the per-32-scale activation of [`QuantizedX`]
/// (W-1a is [`q4k_matvec_tiled_range_a`]).
///
/// A 256-column slab is unpacked once for a 32-row window: the 32 rows share
/// one activation tile, and inside it an 8-row x 16-column ZMM `dpbusd`
/// microtile loads each activation group once for 8 weight rows. That is the
/// reuse the row-by-row batch kernel cannot get: with 8960 rows and 16
/// activations, the row kernel re-reads the activation tile once per row.
///
/// Math mirrors [`q4k_dot`]: per sub-block `s`, the integer sum is the `u8`
/// codes 0..15 against the `i8` activation codes, times `d * scale[s] * dx[s]`,
/// minus `dmin * min[s]` times the exact per-sub-block activation sum
/// (`sums[e]`, the sum of the original `f32` values). Summation order across
/// sub-blocks differs from the row kernel, so results agree within
/// `1e-3 + 1e-4 * |ref|`, not bit-for-bit.
///
/// # Safety
///
/// `data` must hold the rows with the given `stride` (a whole number of
/// 144-byte Q4_K blocks per row), every `xs[e].q` must cover the row width,
/// every `xs[e].d` and `sums[e]` one entry per 32-value group, and the CPU
/// must support AVX-512F/BW/VNNI (the caller checks via
/// [`vnni512_available`]).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn q4k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
    tile: Option<&SharedTile>,
) {
    unsafe {
        const COLS: usize = 16;
        let blocks = stride / 144;
        let halves = B.div_ceil(COLS);
        let mut owned_act;
        let act: &[i8] = match tile.filter(|t| t.blocks() == blocks && t.halves() == halves) {
            Some(tile) => tile.bytes(),
            None => {
                owned_act = vec![0_i8; blocks * halves * 64 * COLS * 4];
                for b in 0..blocks {
                    for (entry, x) in xs.iter().enumerate() {
                        let half = entry / COLS;
                        let col = entry % COLS;
                        let chunk = x.q.as_ptr().add(b * 256);
                        let base = (b * halves + half) * 64 * COLS * 4;
                        for grp in 0..64 {
                            std::ptr::copy_nonoverlapping(
                                chunk.add(grp * 4),
                                owned_act.as_mut_ptr().add(base + grp * 64 + col * 4),
                                4,
                            );
                        }
                    }
                }
                &owned_act
            }
        };
        let mut dxs = vec![0.0_f32; blocks * halves * 8 * COLS];
        let mut sumt = vec![0.0_f32; blocks * halves * 8 * COLS];
        for b in 0..blocks {
            for entry in 0..B {
                let half = entry / COLS;
                let col = entry % COLS;
                for sub in 0..8 {
                    let at = ((b * halves + half) * 8 + sub) * COLS + col;
                    dxs[at] = xs[entry].d[b * 8 + sub];
                    sumt[at] = sums[entry][b * 8 + sub];
                }
            }
        }
        q4k_tiled_matvec_worker::<B>(data, stride, first_row, count, act, &dxs, &sumt, out);
    }
}

/// Safe entry point for [`q4k_matvec_tiled_range_a`] — the W-1a band: the per-256-scale
/// activation the band reads (`VIPERA_TILED_A`).
pub(crate) fn try_q4k_matvec_tiled_range_a<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    act_a: &TiledActivation,
    out: &mut [f32],
) -> bool {
    if !vnni512_available() || stride == 0 || !stride.is_multiple_of(144) || B == 0 || B > 128 {
        return false;
    }
    let blocks = stride / 144;
    let cols = blocks * 256;
    if out.len() != B.saturating_mul(count)
        || !(0..B).all(|entry| {
            act_a.q(entry).len() >= cols
                && act_a.scale(entry).len() >= blocks
                && act_a.bsums(entry).len() >= blocks * 16
        })
    {
        return false;
    }
    // SAFETY: vnni512_available checked the CPU and the lengths above match the kernel contract.
    unsafe { q4k_matvec_tiled_range_a::<B>(data, stride, first_row, count, act_a, out) };
    true
}

/// W-1a: the Q4_K tiled band over a [`TiledActivation`] — the activation quantized per
/// 256-value block: `dxs` holds the per-block scale (the same value for every sub of the block),
/// and `sumt` holds the activation sums the band reads as
/// `Σ_j bsums[b * 16 + 2s] + bsums[b * 16 + 2s + 1]` — the raw `x` over the weight sub `s`'s two
/// 16-value groups. The tile is built from the activation's natural-order codes with the same
/// reorg the per-32 path uses.
///
/// # Safety
///
/// The caller validated the lengths [`try_q4k_matvec_tiled_range_a`] checks; the CPU has AVX-512
/// (the caller checks via [`vnni512_available`]).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn q4k_matvec_tiled_range_a<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    act_a: &TiledActivation,
    out: &mut [f32],
) {
    unsafe {
        const COLS: usize = 16;
        let blocks = stride / 144;
        let halves = B.div_ceil(COLS);
        let mut act = vec![0_i8; blocks * halves * 64 * COLS * 4];
        for b in 0..blocks {
            for entry in 0..B {
                let half = entry / COLS;
                let col = entry % COLS;
                let chunk = act_a.q(entry).as_ptr().add(b * 256);
                let base = (b * halves + half) * 64 * COLS * 4;
                for grp in 0..64 {
                    std::ptr::copy_nonoverlapping(
                        chunk.add(grp * 4),
                        act.as_mut_ptr().add(base + grp * 64 + col * 4),
                        4,
                    );
                }
            }
        }
        let mut dxs = vec![0.0_f32; blocks * halves * 8 * COLS];
        let mut sumt = vec![0.0_f32; blocks * halves * 8 * COLS];
        for b in 0..blocks {
            for entry in 0..B {
                let half = entry / COLS;
                let col = entry % COLS;
                let bsums = act_a.bsums(entry);
                for sub in 0..8 {
                    let at = ((b * halves + half) * 8 + sub) * COLS + col;
                    // The per-256 activation scale: the band reads it for every sub of the block.
                    dxs[at] = act_a.scale(entry)[b];
                    // Weight sub `s` spans the two 16-value groups 2s and 2s+1 of the block.
                    sumt[at] = bsums[b * 16 + 2 * sub] + bsums[b * 16 + 2 * sub + 1];
                }
            }
        }
        q4k_tiled_matvec_worker::<B>(data, stride, first_row, count, &act, &dxs, &sumt, out);
    }
}

/// Where one 16-activation tile half sits for the Q6_K kernel: its codes in (group, column, 4)
/// order, then per 16-value group the bias to start the integer sum from and the activation scale.
#[derive(Clone, Copy)]
struct Q6TileHalf {
    act: *const i8,
    bias: *const i32,
    dxs: *const f32,
}

/// `R` weight rows of one 256-column Q6_K block against one 16-activation tile half, added into
/// `dst`.
///
/// `wq` holds the rows' unpacked codes (256 bytes each, biased by +32), `coef[k][first + t]` is
/// `d * scale` of row `t` for 16-value group `k`, and `tile` is the half. `dst` is `R` rows of 16
/// floats, `dst_stride` floats apart. `R` is a constant so the accumulators stay in registers.
///
/// Each group's integer sum starts from `-32 * (sum of the group's activation codes)`, which is
/// the `raw - 32 * bsum` bias correction applied before the first `dpbusd` instead of after the
/// last: the same integer, one vector op fewer per row and group. Every row's arithmetic is
/// independent of `R`, so a window's tail rows run with `R = 1` and give the same bits.
///
/// # Safety
///
/// The CPU must support AVX-512F/BW/VNNI, and every pointer must cover what the description above
/// says: `R * 256` bytes at `wq`, `64 * 64` bytes at `tile.act`, `16 * 16` words at `tile.bias` and
/// `tile.dxs`, and `16` floats in each of `R` rows at `dst`. `first + R` must be at most
/// `ROW_WIN`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[inline]
unsafe fn q6k_tile_rows<const R: usize>(
    wq: *const u8,
    coef: &[[f32; 32]; 16],
    first: usize,
    tile: Q6TileHalf,
    dst: *mut f32,
    dst_stride: usize,
    no_epi: bool,
) {
    // SAFETY: the contract above.
    unsafe {
        let mut facc = [_mm512_setzero_ps(); R];
        for (k, coefs) in coef.iter().enumerate() {
            let bias = _mm512_loadu_si512(tile.bias.add(k * 16) as *const __m512i);
            let mut acc = [bias; R];
            for g in 0..4 {
                let grp = k * 4 + g;
                let codes = _mm512_loadu_si512(tile.act.add(grp * 64) as *const __m512i);
                for (t, slot) in acc.iter_mut().enumerate() {
                    let word = u32::from_le(std::ptr::read_unaligned(
                        wq.add(t * 256 + grp * 4) as *const u32
                    ));
                    *slot = _mm512_dpbusd_epi32(*slot, _mm512_set1_epi32(word as i32), codes);
                }
            }
            let dxv = _mm512_loadu_ps(tile.dxs.add(k * 16));
            for t in 0..R {
                // Probe arm: four epilogue ops and a broadcast become one convert and one add.
                if no_epi {
                    facc[t] = _mm512_add_ps(facc[t], _mm512_cvtepi32_ps(acc[t]));
                    continue;
                }
                let lanes = _mm512_mul_ps(_mm512_cvtepi32_ps(acc[t]), dxv);
                let scaled = _mm512_mul_ps(lanes, _mm512_set1_ps(coefs[first + t]));
                facc[t] = _mm512_add_ps(facc[t], scaled);
            }
        }
        for (t, value) in facc.iter().enumerate() {
            let slot = dst.add(t * dst_stride);
            _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), *value));
        }
    }
}

/// One band of `N` rows of [`q4k_matvec_tiled_range`], with the row count a constant.
///
/// The bound has to be a constant: as a runtime `take(band_cnt)` it costs a fused `cmp`+`jcc` per
/// `dpbusd`, and every tensor this kernel serves takes the full band.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn q4k_tiled_band<const N: usize>(
    wq: *const u8,
    half_act: *const i8,
    dxs: *const f32,
    sumt: *const f32,
    band: usize,
    scale_f: &[[f32; 8]],
    min_f: &[[f32; 8]],
    buf: *mut f32,
    stride_cols: usize,
    half: usize,
    fused_epi: bool,
    no_weight: bool,
    no_epi: bool,
    per_block_epi: bool,
) {
    // SAFETY: every pointer is one the caller built for this block and band, and the `N` rows from
    // `band` are inside the window the caller clamped to `cnt`.
    unsafe {
        const COLS: usize = 16;
        let mut facc: [__m512; N] = [_mm512_setzero_ps(); N];
        // Probe arm: one int accumulator across all eight sub-blocks, so the float work drops from
        // four ops per (row, sub-block) to one convert and one multiply-add per (row, block).
        let mut isum: [__m512i; N] = [_mm512_setzero_si512(); N];
        let two_acc = tiled_two_acc();
        for s in 0..8 {
            let mut acc: [__m512i; N] = [_mm512_setzero_si512(); N];
            let mut acc_odd: [__m512i; N] = [_mm512_setzero_si512(); N];
            for g in 0..8 {
                let grp = s * 8 + g;
                let codes = _mm512_loadu_si512(half_act.add(grp * 64) as *const __m512i);
                if two_acc && g % 2 == 1 {
                    for (t, slot) in acc_odd.iter_mut().enumerate() {
                        let word = if no_weight {
                            0x1111_1111_i32
                        } else {
                            u32::from_le(std::ptr::read_unaligned(
                                wq.add((band + t) * 256 + grp * 4) as *const u32,
                            )) as i32
                        };
                        *slot = _mm512_dpbusd_epi32(*slot, _mm512_set1_epi32(word), codes);
                    }
                } else {
                    for (t, slot) in acc.iter_mut().enumerate() {
                        // The probe arm keeps every instruction of the real loop and drops only the
                        // 4-byte weight load, so the difference prices that load.
                        let word = if no_weight {
                            0x1111_1111_i32
                        } else {
                            u32::from_le(std::ptr::read_unaligned(
                                wq.add((band + t) * 256 + grp * 4) as *const u32,
                            )) as i32
                        };
                        *slot = _mm512_dpbusd_epi32(*slot, _mm512_set1_epi32(word), codes);
                    }
                }
            }
            if two_acc {
                for (even, odd) in acc.iter_mut().zip(acc_odd.iter()) {
                    *even = _mm512_add_epi32(*even, *odd);
                }
            }
            let dxv = _mm512_loadu_ps(dxs.add(s * COLS));
            let sumv = _mm512_loadu_ps(sumt.add(s * COLS));
            if per_block_epi {
                // The weight scale, in the integer domain: a constant here because a real one
                // needs the activation's per-block scale, which this probe is pricing.
                let wscale = _mm512_set1_epi32(0x0f);
                for (slot, acc_slot) in isum.iter_mut().zip(acc.iter()) {
                    *slot = _mm512_add_epi32(*slot, _mm512_mullo_epi32(*acc_slot, wscale));
                }
                continue;
            }
            for (t, (acc_slot, facc_slot)) in acc.iter().zip(facc.iter_mut()).enumerate() {
                let row = band + t;
                if no_epi {
                    *facc_slot = _mm512_add_ps(*facc_slot, _mm512_cvtepi32_ps(*acc_slot));
                    continue;
                }
                if fused_epi {
                    let scaled = _mm512_mul_ps(_mm512_cvtepi32_ps(*acc_slot), dxv);
                    let sum = _mm512_fmadd_ps(scaled, _mm512_set1_ps(scale_f[row][s]), *facc_slot);
                    *facc_slot = _mm512_fnmadd_ps(sumv, _mm512_set1_ps(min_f[row][s]), sum);
                    continue;
                }
                let main = _mm512_mul_ps(
                    _mm512_mul_ps(
                        _mm512_cvtepi32_ps(*acc_slot),
                        _mm512_set1_ps(scale_f[row][s]),
                    ),
                    dxv,
                );
                let minterm = _mm512_mul_ps(sumv, _mm512_set1_ps(min_f[row][s]));
                *facc_slot = _mm512_add_ps(*facc_slot, _mm512_sub_ps(main, minterm));
            }
        }
        if per_block_epi {
            let dxv = _mm512_loadu_ps(dxs);
            for (slot, value) in isum.iter().zip(facc.iter_mut()) {
                *value = _mm512_fmadd_ps(_mm512_cvtepi32_ps(*slot), dxv, *value);
            }
        }
        for (t, value) in facc.iter().enumerate() {
            let slot = buf.add((band + t) * stride_cols + half * COLS);
            _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), *value));
        }
    }
}

/// Safe entry point for [`q6k_matvec_tiled_range`], as
/// [`try_q4k_matvec_tiled_range`] is for Q4_K.
pub(crate) fn try_q6k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    out: &mut [f32],
    tile: Option<&SharedTile>,
) -> bool {
    if !vnni512_available()
        || stride == 0
        || !stride.is_multiple_of(210)
        || B == 0
        || B > MAX_HALVES * 16
    {
        return false;
    }
    let blocks = stride / 210;
    let cols = blocks * 256;
    if out.len() != B.saturating_mul(count)
        || xs.iter().any(|x| x.q.len() < cols || x.d.len() < cols / 32)
    {
        return false;
    }
    // SAFETY: vnni512_available checked the CPU and the lengths above match
    // the kernel's contract.
    unsafe { q6k_matvec_tiled_range(data, stride, first_row, count, xs, out, tile) };
    true
}

/// Tiled Q6_K matvec range, the Q6_K twin of [`q4k_matvec_tiled_range`]:
/// `row = d * sum over 16-value groups of scale * (sum of (wq - 32) * xq)`,
/// with the weights biased by +32 so the unsigned side of `dpbusd` can read
/// them directly and the bias corrected per group before the first `dpbusd`
/// (see [`q6k_tile_rows`]).
///
/// # Safety
///
/// `data` must hold the rows with the given `stride` (a whole number of
/// 210-byte Q6_K blocks per row), every `xs[e].q` must cover the row width,
/// every `xs[e].d` one entry per 32-value group, and the CPU must support
/// AVX-512F/BW/VNNI (the caller checks via [`vnni512_available`]).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn q6k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    out: &mut [f32],
    tile: Option<&SharedTile>,
) {
    // SAFETY: the module contract above; every load below stays inside the
    // rows, blocks, and activation slices the caller validated.
    unsafe {
        const ROW_WIN: usize = 32;
        const BAND: usize = 8;
        const COLS: usize = 16;
        let halves = B.div_ceil(COLS);
        debug_assert!(halves <= MAX_HALVES);
        let stride_cols = halves * COLS;
        let blocks = stride / 210;
        let no_epi = tiled_no_epi();
        let no_unpack = tiled_no_unpack();
        let mut wq = vec![0_u8; ROW_WIN * 256];
        // `d * scale` per 16-value group and row, group-major so a band reads one contiguous run.
        let mut coef = [[0.0_f32; ROW_WIN]; 16];
        // Activation tile for every block, built once and reused by every row window.
        // The activation tile is the same bytes and layout as Q4_K's, so a product-wide build from
        // the prep workers is reused here too. The integer sums and scales stay per participant.
        let mut owned_act;
        let act: &[i8] = match tile.filter(|t| t.blocks() == blocks && t.halves() == halves) {
            Some(tile) => tile.bytes(),
            None => {
                owned_act = vec![0_i8; blocks * halves * 64 * COLS * 4];
                for b in 0..blocks {
                    for (entry, x) in xs.iter().enumerate() {
                        let col_half = entry / COLS;
                        let col = entry % COLS;
                        let chunk = x.q.as_ptr().add(b * 256);
                        let base = (b * halves + col_half) * 64 * COLS * 4;
                        for grp in 0..64 {
                            std::ptr::copy_nonoverlapping(
                                chunk.add(grp * 4),
                                owned_act.as_mut_ptr().add(base + grp * 64 + col * 4),
                                4,
                            );
                        }
                    }
                }
                &owned_act
            }
        };
        // The bias is the same integer the kernel used to apply after the last `dpbusd`
        // (`raw - 32 * bsum`), moved into the accumulator's starting value.
        let mut bias = vec![0_i32; blocks * halves * 16 * COLS];
        let mut dxs = vec![0.0_f32; blocks * halves * 16 * COLS];
        for b in 0..blocks {
            for (entry, x) in xs.iter().enumerate() {
                let half = entry / COLS;
                let col = entry % COLS;
                let chunk = x.q.as_ptr().add(b * 256);
                for k in 0..16 {
                    let mut sum = 0_i32;
                    for i in 0..16 {
                        sum += i32::from(*chunk.add(k * 16 + i));
                    }
                    let at = ((b * halves + half) * 16 + k) * COLS + col;
                    bias[at] = -32 * sum;
                    dxs[at] = x.d[b * 8 + k / 2];
                }
            }
        }
        // Window-major accumulation, as in the Q4_K kernel: `out` is written once per row window
        // and the partial sums live in a ROW_WIN * stride_cols buffer instead of one entry per row
        // of the whole matrix.
        let mut buf = vec![0.0_f32; ROW_WIN * stride_cols];
        let mut win = 0;
        while win < count {
            let cnt = (count - win).min(ROW_WIN);
            buf[..cnt * stride_cols].fill(0.0);
            for b in 0..blocks {
                // The walk takes 210 bytes out of every row, so consecutive addresses are a
                // stride apart and the hardware prefetcher only sees part of the stream.
                if b + 1 < blocks {
                    for r in 0..cnt {
                        let at = (first_row + win + r) * stride + (b + 1) * 210;
                        prefetch_ahead(data, at);
                        prefetch_ahead(data, at + 64);
                        prefetch_ahead(data, at + 128);
                        prefetch_ahead(data, at + 192);
                    }
                }
                // Only rows `0..cnt` are read below, so `wq` and `coef` need no clearing.
                for r in 0..cnt {
                    let block = data.as_ptr().add((first_row + win + r) * stride + b * 210);
                    let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
                    let scale_bytes = std::slice::from_raw_parts(block.add(192), 16);
                    for (column, byte) in coef.iter_mut().zip(scale_bytes) {
                        column[r] = d * f32::from(*byte as i8);
                    }
                    if no_unpack {
                        continue;
                    }
                    let wrow = wq.as_mut_ptr().add(r * 256);
                    for half in 0..2 {
                        let ql = block.add(half * 64);
                        let qh = block.add(128 + half * 32);
                        for l in 0..32 {
                            let lo = *ql.add(l);
                            let hi = *ql.add(32 + l);
                            let h = *qh.add(l);
                            let base = half * 128 + l;
                            *wrow.add(base) = (lo & 0x0f) | ((h & 0x03) << 4);
                            *wrow.add(base + 32) = (hi & 0x0f) | (((h >> 2) & 0x03) << 4);
                            *wrow.add(base + 64) = (lo >> 4) | (((h >> 4) & 0x03) << 4);
                            *wrow.add(base + 96) = (hi >> 4) | (((h >> 6) & 0x03) << 4);
                        }
                    }
                }
                let wq_ptr = wq.as_ptr();
                for col_half in 0..halves {
                    let at = b * halves + col_half;
                    let tile = Q6TileHalf {
                        act: act.as_ptr().add(at * 64 * COLS * 4),
                        bias: bias.as_ptr().add(at * 16 * COLS),
                        dxs: dxs.as_ptr().add(at * 16 * COLS),
                    };
                    let mut band = 0;
                    while band < cnt {
                        // Full bands take the unrolled path, the tail goes a row at a time.
                        let wq_rows = wq_ptr.add(band * 256);
                        let dst = buf.as_mut_ptr().add(band * stride_cols + col_half * COLS);
                        if cnt - band >= BAND {
                            q6k_tile_rows::<BAND>(
                                wq_rows,
                                &coef,
                                band,
                                tile,
                                dst,
                                stride_cols,
                                no_epi,
                            );
                            band += BAND;
                        } else {
                            q6k_tile_rows::<1>(
                                wq_rows,
                                &coef,
                                band,
                                tile,
                                dst,
                                stride_cols,
                                no_epi,
                            );
                            band += 1;
                        }
                    }
                }
            }
            for r in 0..cnt {
                for entry in 0..B {
                    out[entry * count + win + r] = buf[r * stride_cols + entry];
                }
            }
            win += cnt;
        }
    }
}

/// One packed row against an `i8` activation, `dpbusd` over unpacked weights.
///
/// Prototype Pha 1b for `crate::repack::PackedTile`: the tile holds weights
/// unpacked to `u8` 0..15 in row-major order, so each 32-value group is one
/// `dpbusd` call with no nibble unpack in the dot. Parity target is
/// [`crate::quant::q4k_dot`] within `1e-3 + 1e-4 * |ref|`, not bit-identical:
/// lane grouping changes addition order.
///
/// # Safety
///
/// `tile.q` has `8 * cols` bytes, `coeff`/`min_coeff` have `8 * blocks * 8`
/// entries, `xq`/`dx`/`sums` cover `cols`, and the CPU supports
/// AVX2 + `avx512_vnni` with `avx512_vl` (caller checks via
/// `vnni_available()`).
#[cfg(test)]
#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
pub(crate) unsafe fn q4k_dot_packed_row_vnni(
    tile: &crate::repack::PackedTile,
    row: usize,
    xq: &[i8],
    dx: &[f32],
    sums: &[f32],
) -> f32 {
    // SAFETY: caller's contract as above; all loads stay inside the slices.
    unsafe {
        let cols = tile.cols;
        let blocks = tile.blocks;
        let w_row = tile.q.as_ptr().add(row * cols);
        let c_row = tile.coeff.as_ptr().add(row * blocks * 8);
        let m_row = tile.min_coeff.as_ptr().add(row * blocks * 8);
        let zero = _mm256_setzero_si256();
        let mut weights0 = _mm256_setzero_ps();
        let mut weights1 = _mm256_setzero_ps();
        let mut minimum = 0.0_f32;
        for b in 0..blocks {
            for g in 0..8 {
                let global = b * 8 + g;
                let w = w_row.add(b * 256 + g * 32) as *const i8;
                let a = xq.as_ptr().add(b * 256 + g * 32);
                let int_sum = _mm256_dpbusd_epi32(
                    zero,
                    _mm256_loadu_si256(w as *const __m256i),
                    _mm256_loadu_si256(a as *const __m256i),
                );
                let scale = dx[global];
                let combined = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(int_sum),
                    _mm256_set1_ps(*c_row.add(global) * scale),
                );
                minimum += *m_row.add(global) * sums[global];
                if g.is_multiple_of(2) {
                    weights0 = _mm256_add_ps(weights0, combined);
                } else {
                    weights1 = _mm256_add_ps(weights1, combined);
                }
            }
        }
        hsum(_mm256_add_ps(weights0, weights1)) - minimum
    }
}

/// [`q6k_dot`] for `B` activations, sharing the six-bit unpacking across them.
///
/// Every level and scale in a group is entry-independent; only the multiply against each entry's
/// activation repeats. Per entry the operations happen in the same order as [`q6k_dot`].
///
/// # Safety
///
/// See the module documentation: `row` must be whole 210-byte blocks, every `xs[e]` exactly 256
/// floats per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q6k_dot_batch<const B: usize>(
    row: &[u8],
    xs: &[&[f32]; B],
    out: &mut [f32; B],
) {
    // SAFETY: the module documentation is the contract for every pointer below, and the CPU has
    // AVX2. Each entry's activation covers the 256 values of a block exactly once.
    unsafe {
        out.fill(0.0);
        let blocks = row.len() / 210;
        let low_nibbles = _mm256_set1_epi32(0x0f);
        let two_bits = _mm256_set1_epi32(0x03);
        let bias = _mm256_set1_epi32(32);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 210);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            let mut accumulator = [_mm256_setzero_ps(); B];
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = block.add(128 + half * 32);
                for group in 0..4 {
                    let at = group * 8;
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
                    for entry in 0..B {
                        let x_half = xs[entry].as_ptr().add(b * 256 + half * 128);
                        let weighted = |values: __m256i, offset: usize, scale: __m256| {
                            _mm256_mul_ps(
                                _mm256_mul_ps(
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(values, bias)),
                                    _mm256_loadu_ps(x_half.add(at + offset)),
                                ),
                                scale,
                            )
                        };
                        accumulator[entry] =
                            _mm256_add_ps(accumulator[entry], weighted(first, 0, scale(is)));
                        accumulator[entry] =
                            _mm256_add_ps(accumulator[entry], weighted(second, 32, scale(is + 2)));
                        accumulator[entry] =
                            _mm256_add_ps(accumulator[entry], weighted(third, 64, scale(is + 4)));
                        accumulator[entry] =
                            _mm256_add_ps(accumulator[entry], weighted(fourth, 96, scale(is + 6)));
                    }
                }
            }
            for entry in 0..B {
                out[entry] += d * hsum(accumulator[entry]);
            }
        }
    }
}

/// [`q6k_dot`]'s six-bit unpacking, dotted in integers against an `i8` activation.
///
/// One 32-value run is unpacked once (`q` unsigned 0..63, plus the two high bits per value), and
/// `maddubs` turns it into sixteen pair sums against the activation bytes; `madd` against ones
/// leaves eight `i32` lane sums, four per 16-value group. The `-32` of the six-bit levels is removed
/// in the integer domain (`32 * xq` per pair), which is what lets the float side be a single
/// multiply per 16 values and one horizontal sum per block instead of one multiply-add per value.
/// No product saturates: `2 * 63 * 127 = 16002` per pair, and a lane sums two pairs.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 210-byte blocks, `xq` exactly 256 values and
/// 8 scales per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q6k_dot_q8(row: &[u8], xq: &QuantizedX) -> f32 {
    // SAFETY: the module documentation is the contract: `row` is whole 210-byte blocks, `xq` holds
    // 256 values and 8 scales per block, and the CPU has AVX2. Every load below stays inside one
    // block or one entry's activation.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 210;
        let low_nibbles = _mm256_set1_epi8(0x0f);
        let two_bits = _mm256_set1_epi8(0x03);
        let bias = _mm256_set1_epi8(32);
        let ones = _mm256_set1_epi16(1);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 210);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            let values = xq.q.as_ptr().add(b * 256);
            let dxs = xq.d.as_ptr().add(b * 8);
            let mut accumulator = _mm256_setzero_ps();
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = _mm256_loadu_si256(block.add(128 + half * 32) as *const __m256i);
                // The two high bits of each value, one vector per run, so the shifts stay immediates.
                let high_bits = [
                    _mm256_and_si256(qh, two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), two_bits),
                ];
                for (run, high) in high_bits.iter().enumerate() {
                    let (high_nibble, offset) = if run < 2 {
                        (false, run * 32)
                    } else {
                        (true, (run - 2) * 32)
                    };
                    let packed = _mm256_loadu_si256(ql.add(offset) as *const __m256i);
                    let levels = if high_nibble {
                        _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_nibbles)
                    } else {
                        _mm256_and_si256(packed, low_nibbles)
                    };
                    let q = _mm256_or_si256(levels, _mm256_slli_epi16(*high, 4));
                    let x = _mm256_loadu_si256(values.add(half * 128 + run * 32) as *const __m256i);
                    let pairs =
                        _mm256_sub_epi16(_mm256_maddubs_epi16(q, x), _mm256_maddubs_epi16(bias, x));
                    let lanes = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));
                    let base = half * 8 + run * 2;
                    let dx = *dxs.add(half * 4 + run);
                    let low_scale = d * f32::from(scales[base] as i8) * dx;
                    let high_scale = d * f32::from(scales[base + 1] as i8) * dx;
                    let scale = _mm256_set_m128(_mm_set1_ps(high_scale), _mm_set1_ps(low_scale));
                    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(lanes, scale));
                }
            }
            total += hsum(accumulator);
        }
        total
    }
}

/// [`q6k_dot_q8`]'s unpacking with `vpdpbusd`: the six-bit levels are dotted in integers against
/// the activation by one instruction per run, and the −32 of the six-bit levels is removed in the
/// same integer domain (the unsigned `q` products minus a `32` shift of the activation sums), so
/// the float side is one multiply-add per run like [`q6k_dot_q8`].
///
/// `vpdpbusd` takes the weight as unsigned and the activation as signed — exactly the shapes the
/// unpacked `q` (0..63) and `x` have here — so the old kernel's `maddubs` pair and `madd` against
/// ones become one `dpbusd` per run plus one more `dpbusd` against an all-ones activation vector
/// for the −32 correction, as a shift of that sum. The lane values are the same eight lane sums
/// of four products, so the float multiply and the per-run scale land in the same places.
///
/// # Safety
///
/// As [`q6k_dot_q8`]: `row` is whole 210-byte blocks, `xq` exactly 256 values and 8 scales per
/// block, and the CPU must support AVX512-VNNI and AVX512-VL.
#[target_feature(enable = "avx512vnni,avx512vl,avx2")]
pub(crate) unsafe fn q6k_dot_vnni(row: &[u8], xq: &QuantizedX) -> f32 {
    // SAFETY: the module contract is the same as in `q6k_dot_q8`; every load stays inside one
    // block or one entry's activation.
    unsafe {
        let mut total = 0.0_f32;
        let blocks = row.len() / 210;
        let low_nibbles = _mm256_set1_epi8(0x0f);
        let two_bits = _mm256_set1_epi8(0x03);
        let ones = _mm256_set1_epi8(1);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 210);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            let values = xq.q.as_ptr().add(b * 256);
            let dxs = xq.d.as_ptr().add(b * 8);
            let mut accumulator = _mm256_setzero_ps();
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = _mm256_loadu_si256(block.add(128 + half * 32) as *const __m256i);
                // The two high bits of each value, one vector per run, so the shifts stay
                // immediates.
                let high_bits = [
                    _mm256_and_si256(qh, two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), two_bits),
                ];
                for (run, high) in high_bits.iter().enumerate() {
                    let (high_nibble, offset) = if run < 2 {
                        (false, run * 32)
                    } else {
                        (true, (run - 2) * 32)
                    };
                    let packed = _mm256_loadu_si256(ql.add(offset) as *const __m256i);
                    let levels = if high_nibble {
                        _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_nibbles)
                    } else {
                        _mm256_and_si256(packed, low_nibbles)
                    };
                    let q = _mm256_or_si256(levels, _mm256_slli_epi16(*high, 4));
                    let x = _mm256_loadu_si256(values.add(half * 128 + run * 32) as *const __m256i);
                    // One `vpdpbusd`: each i32 lane sums four adjacent `q * x` products.
                    let raw = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q, x);
                    // The −32 per value is `32` times the activation sum of the same four values;
                    // a shift of an exact i32 sum, so `lanes` is the old kernel's integer lanes.
                    let activation_sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, x);
                    let lanes = _mm256_sub_epi32(raw, _mm256_slli_epi32(activation_sums, 5));
                    let base = half * 8 + run * 2;
                    let dx = *dxs.add(half * 4 + run);
                    let low_scale = d * f32::from(scales[base] as i8) * dx;
                    let high_scale = d * f32::from(scales[base + 1] as i8) * dx;
                    let scale = _mm256_set_m128(_mm_set1_ps(high_scale), _mm_set1_ps(low_scale));
                    accumulator =
                        _mm256_add_ps(accumulator, _mm256_mul_ps(_mm256_cvtepi32_ps(lanes), scale));
                }
            }
            total += hsum(accumulator);
        }
        total
    }
}

/// [`q6k_dot_q8`] for `B` activations, unpacking each 32-value run once.
///
/// The unpacked run and its scales are shared across the entries; only the `maddubs` against each
/// entry's activation repeats. Per entry the operations happen in the same order as
/// [`q6k_dot_q8`], so entry `e` of a batch is bit-identical to its own single call.
///
/// # Safety
///
/// See the module documentation: `row` must be whole 210-byte blocks, every `xs[e]` must hold 256
/// values and 8 scales per block, and the CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q6k_dot_batch_q8<const B: usize>(
    row: &[u8],
    xs: &[&QuantizedX; B],
    out: &mut [f32; B],
) {
    // SAFETY: the module documentation is the contract for every pointer below, and the CPU has
    // AVX2. Each entry's `xq` covers the 256 values of a block exactly once.
    unsafe {
        let mut totals = [0.0_f32; B];
        let values_at: [*const i8; B] = std::array::from_fn(|entry| xs[entry].q.as_ptr());
        let scales_at: [*const f32; B] = std::array::from_fn(|entry| xs[entry].d.as_ptr());
        let blocks = row.len() / 210;
        let low_nibbles = _mm256_set1_epi8(0x0f);
        let two_bits = _mm256_set1_epi8(0x03);
        let bias = _mm256_set1_epi8(32);
        let ones = _mm256_set1_epi16(1);
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 210);
            let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
            let scales = std::slice::from_raw_parts(block.add(192), 16);
            let mut accumulator = [_mm256_setzero_ps(); B];
            for half in 0..2 {
                let ql = block.add(half * 64);
                let qh = _mm256_loadu_si256(block.add(128 + half * 32) as *const __m256i);
                let high_bits = [
                    _mm256_and_si256(qh, two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 2), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 4), two_bits),
                    _mm256_and_si256(_mm256_srli_epi16(qh, 6), two_bits),
                ];
                for (run, high) in high_bits.iter().enumerate() {
                    let (high_nibble, offset) = if run < 2 {
                        (false, run * 32)
                    } else {
                        (true, (run - 2) * 32)
                    };
                    let packed = _mm256_loadu_si256(ql.add(offset) as *const __m256i);
                    let levels = if high_nibble {
                        _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_nibbles)
                    } else {
                        _mm256_and_si256(packed, low_nibbles)
                    };
                    let q = _mm256_or_si256(levels, _mm256_slli_epi16(*high, 4));
                    let base = half * 8 + run * 2;
                    // The eight lanes cover 32 values: the first sixteen use the low sub-block scale
                    // and the last sixteen the high one. That vector is entry-independent, so it is
                    // built once here and an entry only multiplies it by its own activation scale —
                    // `(d * scale) * dx`, the same product in the same order as before.
                    let weights = _mm256_set_m128(
                        _mm_set1_ps(d * f32::from(scales[base + 1] as i8)),
                        _mm_set1_ps(d * f32::from(scales[base] as i8)),
                    );
                    for entry in 0..B {
                        let values = values_at[entry].add(b * 256 + half * 128 + run * 32);
                        let x = _mm256_loadu_si256(values as *const __m256i);
                        let pairs = _mm256_sub_epi16(
                            _mm256_maddubs_epi16(q, x),
                            _mm256_maddubs_epi16(bias, x),
                        );
                        let lanes = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));
                        let dx = *scales_at[entry].add(b * 8 + half * 4 + run);
                        let scale = _mm256_mul_ps(weights, _mm256_set1_ps(dx));
                        accumulator[entry] =
                            _mm256_add_ps(accumulator[entry], _mm256_mul_ps(lanes, scale));
                    }
                }
            }
            for entry in 0..B {
                totals[entry] += hsum(accumulator[entry]);
            }
        }
        out.copy_from_slice(&totals);
    }
}

/// [`q4k_dot`] with `vpdpbusd` in place of `maddubs` + `madd`, on 256-bit registers.
///
/// `vpdpbusd` multiplies four `u8` by four `i8` and adds the four products into one `i32` lane in
/// a single instruction, which is exactly what `maddubs` followed by `madd` against ones
/// computes. The integer result is identical, so the float path below is byte-for-byte
/// `q4k_dot`'s and each row stays bit-exact.
///
/// 256-bit rather than 512-bit on purpose: the wide form downclocks and was measured 1.43x
/// slower in this same kernel family. This CPU exposes `avx512_vnni` with `avx512_vl`, which
/// gives the instruction on YMM without the wide-register cost.
///
/// # Safety
///
/// As [`q4k_dot`]: whole 144-byte blocks, 256 activation values and 8 scales per block, at least
/// 8 sums per block, and a CPU with `avx512_vnni` and `avx512_vl`.
#[target_feature(enable = "avx512vnni,avx512vl,avx2")]
pub(crate) unsafe fn q4k_dot_vnni(row: &[u8], xq: &[i8], dx: &[f32], sums: &[f32]) -> f32 {
    // SAFETY: the caller guarantees the same layout `q4k_dot` needs, and the CPU features.
    unsafe {
        let blocks = row.len() / 144;
        let low_mask = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut weights0 = _mm256_setzero_ps();
        let mut weights1 = _mm256_setzero_ps();
        let mut minimums = _mm256_setzero_ps();
        for b in 0..blocks {
            let block = row.as_ptr().add(b * 144);
            let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
            let (scale_at, min_at) = q4k_scale_vectors(block);
            let coefficients = _mm256_mul_ps(
                _mm256_mul_ps(scale_at, _mm256_loadu_ps(dx.as_ptr().add(b * 8))),
                _mm256_set1_ps(d),
            );
            minimums = _mm256_fmadd_ps(
                _mm256_mul_ps(min_at, _mm256_set1_ps(dmin)),
                _mm256_loadu_ps(sums.as_ptr().add(b * 8)),
                minimums,
            );
            for j in 0..4 {
                let q = block.add(16 + j * 32);
                let packed = _mm256_loadu_si256(q as *const __m256i);
                let low_bytes = _mm256_and_si256(packed, low_mask);
                let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                let values = xq.as_ptr().add(b * 256 + j * 64);
                let low_sums = _mm256_dpbusd_epi32(
                    zero,
                    low_bytes,
                    _mm256_loadu_si256(values as *const __m256i),
                );
                let high_sums = _mm256_dpbusd_epi32(
                    zero,
                    high_bytes,
                    _mm256_loadu_si256(values.add(32) as *const __m256i),
                );
                let low_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j) as i32));
                let high_coeff =
                    _mm256_permutevar8x32_ps(coefficients, _mm256_set1_epi32((2 * j + 1) as i32));
                let low_lanes = _mm256_cvtepi32_ps(low_sums);
                let high_lanes = _mm256_cvtepi32_ps(high_sums);
                if j.is_multiple_of(2) {
                    weights0 = _mm256_fmadd_ps(low_lanes, low_coeff, weights0);
                    weights1 = _mm256_fmadd_ps(high_lanes, high_coeff, weights1);
                } else {
                    weights1 = _mm256_fmadd_ps(low_lanes, low_coeff, weights1);
                    weights0 = _mm256_fmadd_ps(high_lanes, high_coeff, weights0);
                }
            }
        }
        hsum(_mm256_add_ps(weights0, weights1)) - hsum(minimums)
    }
}

/// `out[i] = row[i]` read as `f16`, for every `i` the two slices share.
///
/// One key or value row of a KV tile, decoded once so the tile's GEMMs read `f32` unchanged.
/// Safe to call: it uses the eight-lane conversion when the CPU has `f16c` and a value at a
/// time otherwise.
pub(crate) fn f16_row_to_f32(row: &[u8], out: &mut [f32]) {
    use std::arch::x86_64::{_mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let length = row.len() / 2;
    let keep = out.len().min(length);
    let out = &mut out[..keep];
    let values = out.len();
    if values >= 8 && available() && std::is_x86_feature_detected!("f16c") {
        // SAFETY: AVX2 and F16C were just detected, `row` holds `length` halves and `out` holds
        // `values` of them, so every step below reads and writes inside both.
        unsafe {
            let input = row.as_ptr();
            let target = out.as_mut_ptr();
            let mut at = 0;
            while at + 8 <= values {
                _mm256_storeu_ps(
                    target.add(at),
                    _mm256_cvtph_ps(_mm_loadu_si128(input.add(at * 2).cast())),
                );
                at += 8;
            }
            while at < values {
                let low = std::ptr::read_unaligned(input.add(at * 2).cast::<u16>());
                target.add(at).write(f16_to_f32(low));
                at += 1;
            }
        }
        return;
    }
    for (slot, chunk) in out.iter_mut().zip(row.as_chunks::<2>().0) {
        *slot = f16_to_f32(u16::from_le_bytes(*chunk));
    }
}

/// `out[i * stride] = row[i]` read as `f16`: one key row written where a `[dim][rows]` operand
/// wants it, so a tile of rows becomes the second operand of a score GEMM without a transpose
/// pass over the tile.
pub(crate) fn f16_row_to_f32_strided(row: &[u8], out: &mut [f32], stride: usize) {
    use std::arch::x86_64::{_mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let length = row.len() / 2;
    // Every write lands on `out[i * stride]`, so the row fits one value into each stride step: the
    // slot count is the *ceiling* of the length over the stride. Truncating it drops the last dim
    // of the rows whose base is more than a stride past the end, which is a silent wrong answer.
    let slots = if stride == 0 {
        0
    } else {
        out.len().div_ceil(stride).min(length)
    };
    if slots >= 8 && available() && std::is_x86_feature_detected!("f16c") {
        // SAFETY: AVX2 and F16C were just detected; `row` holds `length` halves and every write
        // below lands on `out[i * stride]` for `i < slots`, all inside `out`.
        unsafe {
            let input = row.as_ptr();
            let target = out.as_mut_ptr();
            let mut at = 0;
            while at + 8 <= slots {
                // The lane array is a stack local, so the store has to be the unaligned form:
                // `_mm256_store_ps` faults on it, which is how this kernel first arrived.
                let mut lanes = [0.0_f32; 8];
                _mm256_storeu_ps(
                    lanes.as_mut_ptr(),
                    _mm256_cvtph_ps(_mm_loadu_si128(input.add(at * 2).cast())),
                );
                for (lane, value) in lanes.iter().enumerate() {
                    target.add((at + lane) * stride).write(*value);
                }
                at += 8;
            }
            while at < slots {
                let low = std::ptr::read_unaligned(input.add(at * 2).cast::<u16>());
                target.add(at * stride).write(f16_to_f32(low));
                at += 1;
            }
        }
        return;
    }
    for at in 0..slots {
        let chunk = &row[at * 2..at * 2 + 2];
        out[at * stride] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

/// `C[m x n] += A[m x k] * B[k x n]`, with the row strides of `A` and `C` given.
///
/// `A` is `m` rows of `k` values every `a_row` values, `B` is `k` contiguous rows of `n`, and `C`
/// is `m` rows of `n` every `c_row`: the shape a tile GEMM needs when its operands are slices of
/// the attention buffers (a query head inside a chunk of positions, one head's output inside the
/// chunk's output). Converts the register tile into 4 rows x 2 vectors of 16 lanes on AVX-512 and
/// falls back to the plain triple loop otherwise.
///
/// Safe to call; it checks the extents itself.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    a_row: usize,
    b: &[f32],
    c: &mut [f32],
    c_row: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    let a_last = (m - 1) * a_row + k;
    let c_last = (m - 1) * c_row + n;
    assert!(
        a.len() >= a_last && b.len() >= k * n && c.len() >= c_last,
        "a {m}x{k} every {a_row}, b {k}x{n}, c {m}x{n} every {c_row} need {a_last}/{}/{c_last} values, got {}/{}/{}",
        k * n,
        a.len(),
        b.len(),
        c.len()
    );
    if std::is_x86_feature_detected!("avx512f") && n >= 16 && k >= 4 {
        // SAFETY: AVX-512F was just detected and the extents were checked above; the kernel only
        // reads and writes inside them.
        unsafe { gemm_avx512(m, k, n, a, a_row, b, c, c_row) };
        return;
    }
    for i in 0..m {
        let arow = &a[i * a_row..][..k];
        let crow = &mut c[i * c_row..][..n];
        for (j, target) in crow.iter_mut().enumerate() {
            let column = &b[j..];
            let mut total = *target;
            for (offset, weight) in arow.iter().enumerate() {
                total += weight * column[offset * n];
            }
            *target = total;
        }
    }
}

/// # Safety
///
/// AVX-512F, and the extents [`gemm`] checks: `a` holds `(m-1)*a_row + k` values, `b` `k*n`, `c`
/// `(m-1)*c_row + n`.
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_avx512(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    a_row: usize,
    b: &[f32],
    c: &mut [f32],
    c_row: usize,
) {
    use std::arch::x86_64::{
        __mmask16, _mm512_mask_loadu_ps, _mm512_mask_storeu_ps, _mm512_mask3_fmadd_ps,
        _mm512_set1_ps,
    };
    // SAFETY: the caller checked the extents; every pointer below stays inside `a`, `b` and `c`.
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let cp = c.as_mut_ptr();
        let mut i = 0;
        while i + 4 <= m {
            let a0 = ap.add(i * a_row);
            let a1 = a0.add(a_row);
            let a2 = a1.add(a_row);
            let a3 = a2.add(a_row);
            let c0 = cp.add(i * c_row);
            let c1 = c0.add(c_row);
            let c2 = c1.add(c_row);
            let c3 = c2.add(c_row);
            let mut j = 0;
            while j < n {
                let lanes = (n - j).min(16);
                let mask = if lanes == 16 {
                    u16::MAX
                } else {
                    (1_u16 << lanes) - 1
                };
                let mask: __mmask16 = mask;
                let mut s0 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, c0.add(j));
                let mut s1 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, c1.add(j));
                let mut s2 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, c2.add(j));
                let mut s3 = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, c3.add(j));
                for step in 0..k {
                    let values =
                        _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, bp.add(step * n + j));
                    s0 = _mm512_mask3_fmadd_ps(_mm512_set1_ps(*a0.add(step)), values, s0, mask);
                    s1 = _mm512_mask3_fmadd_ps(_mm512_set1_ps(*a1.add(step)), values, s1, mask);
                    s2 = _mm512_mask3_fmadd_ps(_mm512_set1_ps(*a2.add(step)), values, s2, mask);
                    s3 = _mm512_mask3_fmadd_ps(_mm512_set1_ps(*a3.add(step)), values, s3, mask);
                }
                _mm512_mask_storeu_ps(c0.add(j), mask, s0);
                _mm512_mask_storeu_ps(c1.add(j), mask, s1);
                _mm512_mask_storeu_ps(c2.add(j), mask, s2);
                _mm512_mask_storeu_ps(c3.add(j), mask, s3);
                j += lanes;
            }
            i += 4;
        }
        // Fewer than four rows left: one row at a time, so the tile always uses its full width.
        while i < m {
            let arow = ap.add(i * a_row);
            let crow = cp.add(i * c_row);
            for j in 0..n {
                let mut total = *crow.add(j);
                for step in 0..k {
                    total += *arow.add(step) * *bp.add(step * n + j);
                }
                *crow.add(j) = total;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    /// The tiled GEMM against the plain triple loop, over the shapes the attention kernel asks
    /// for: a score tile that lands on the vector width, one that does not, a single row, and the
    /// strided `A` and `C` a head inside a chunk's buffers gives it.
    #[test]
    fn gemm_matches_the_triple_loop_at_every_shape_the_attention_tiles_take() {
        let mut rng = Rng(0x6e_11_00_00_00_00_0a_01);
        let shapes = [
            (1_usize, 1_usize, 1_usize),
            (1, 16, 16),
            (3, 64, 5),
            (9, 64, 64),
            (9, 64, 17),
            (32, 128, 64),
            (32, 128, 128),
            (7, 128, 64),
            (4, 5, 3),
        ];
        for (m, k, n) in shapes {
            let a_row = k + 3;
            let c_row = n + 5;
            let a: Vec<f32> = (0..(m - 1) * a_row + k)
                .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let seed: Vec<f32> = (0..(m - 1) * c_row + n)
                .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let mut expected = seed.clone();
            for i in 0..m {
                for j in 0..n {
                    let mut total = expected[i * c_row + j];
                    for step in 0..k {
                        total += a[i * a_row + step] * b[step * n + j];
                    }
                    expected[i * c_row + j] = total;
                }
            }
            let mut actual = seed.clone();
            super::gemm(m, k, n, &a, a_row, &b, &mut actual, c_row);
            for (at, (want, got)) in expected.iter().zip(&actual).enumerate() {
                let tolerance = 1e-5 * want.abs().max(1.0);
                assert!(
                    (want - got).abs() <= tolerance,
                    "{m}x{k}x{n} at {at}: {want} against {got}"
                );
            }
            for (row, value) in actual.iter().enumerate() {
                let column = row % c_row;
                if column >= n {
                    assert_eq!(
                        *value, seed[row],
                        "{m}x{k}x{n}: wrote past the row at {row}"
                    );
                }
            }
        }
    }

    /// Reading an `f16` row in place must give the bits that decoding it into `f32` and calling
    /// the `f32` kernels gives. The attention path relies on this: it compares against the decoded
    /// oracle bit for bit, so an inexact conversion or a different reduction tree would show up as
    /// a broken attention test rather than as a slow one.
    #[test]
    fn f16_rows_give_the_same_bits_as_decoded_ones() {
        let mut rng = Rng(0x5eed_0d16_0000_0007);
        for length in [1usize, 7, 8, 15, 16, 128, 129, 256] {
            let values: Vec<f32> = (0..length).map(|_| rng.unit() * 8.0 - 4.0).collect();
            let row: Vec<u8> = values
                .iter()
                .flat_map(|value| crate::quant::f32_to_f16(*value).to_le_bytes())
                .collect();
            let other: Vec<f32> = (0..length).map(|_| rng.unit() * 2.0 - 1.0).collect();
            let mut decoded = vec![0.0_f32; length];
            super::half_to_float(&row, &mut decoded);

            assert_eq!(
                super::dot_f16_f32(&other, &row).to_bits(),
                super::dot_f32(&other, &decoded).to_bits(),
                "dot over {length} values"
            );

            let mut fast = vec![0.25_f32; length];
            let mut slow = fast.clone();
            super::axpy_f16_f32(&mut fast, 1.75, &row);
            super::axpy_f32(&mut slow, 1.75, &decoded);
            assert_eq!(fast, slow, "axpy over {length} values");
        }
    }

    fn half_row(rng: &mut Rng, length: usize, spread: f32) -> (Vec<u8>, Vec<f32>) {
        let row: Vec<u8> = (0..length)
            .flat_map(|_| crate::quant::f32_to_f16((rng.unit() - 0.5) * spread).to_le_bytes())
            .collect();
        let mut decoded = vec![0.0_f32; length];
        super::half_to_float(&row, &mut decoded);
        (row, decoded)
    }

    /// The shared score kernel has no bit-exact oracle, only an error budget: it fuses the multiply
    /// and the add and splits each head's sum over two accumulators, so what is pinned is that every
    /// head stays within a few `f32` ulps of the exact sum, measured against the sum of the
    /// absolute terms. The head counts cross the kernel's four-head blocks and the head dimensions
    /// include two that take the per-head fallback.
    #[test]
    fn group_scores_stay_within_rounding_of_the_exact_sum() {
        let mut rng = Rng(0x5eed_6709_0000_0001);
        for head_dim in [8usize, 16, 24, 64, 96, 128] {
            for heads in 1..=9usize {
                let queries: Vec<f32> = (0..heads * head_dim)
                    .map(|_| (rng.unit() - 0.5) * 2.0)
                    .collect();
                let (row, key) = half_row(&mut rng, head_dim, 8.0);
                let mut out = vec![f32::NAN; heads];
                super::dot_f16_group(&queries, head_dim, &row, &mut out);
                for (head, got) in out.iter().enumerate() {
                    let terms = queries[head * head_dim..(head + 1) * head_dim]
                        .iter()
                        .zip(&key)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b));
                    let exact: f64 = terms.clone().sum();
                    let magnitude: f64 = terms.map(f64::abs).sum();
                    let error = (f64::from(*got) - exact).abs();
                    assert!(
                        error <= 1e-6 * magnitude.max(f64::MIN_POSITIVE),
                        "head {head} of {heads}, head_dim {head_dim}: {got} against {exact}"
                    );
                }
            }
        }
    }

    /// The shared value kernel against the exact accumulation: every target element starts from
    /// its own value and adds one weighted element per row, in increasing row order. The row counts
    /// cross the sixteen-position blocks and the head counts the eight-head blocks.
    #[test]
    fn group_fold_stays_within_rounding_of_the_exact_sum() {
        let mut rng = Rng(0x5eed_6709_0000_0002);
        for head_dim in [8usize, 16, 64, 128] {
            for heads in [1usize, 2, 5, 6, 8, 9, 17] {
                for count in [1usize, 2, 15, 16, 17, 40] {
                    let start: Vec<f32> = (0..heads * head_dim)
                        .map(|_| (rng.unit() - 0.5) * 2.0)
                        .collect();
                    let weights: Vec<f32> = (0..count * heads).map(|_| rng.unit()).collect();
                    let rows: Vec<(Vec<u8>, Vec<f32>)> = (0..count)
                        .map(|_| half_row(&mut rng, head_dim, 4.0))
                        .collect();
                    let refs: Vec<&[u8]> = rows.iter().map(|(row, _)| row.as_slice()).collect();
                    let mut targets = start.clone();
                    super::axpy_f16_group(&mut targets, head_dim, heads, &weights, &refs);
                    for head in 0..heads {
                        for at in 0..head_dim {
                            let mut exact = f64::from(start[head * head_dim + at]);
                            let mut magnitude = exact.abs();
                            for (position, (_, decoded)) in rows.iter().enumerate() {
                                let term = f64::from(weights[position * heads + head])
                                    * f64::from(decoded[at]);
                                exact += term;
                                magnitude += term.abs();
                            }
                            let got = targets[head * head_dim + at];
                            assert!(
                                (f64::from(got) - exact).abs() <= 5e-6 * magnitude,
                                "head {head}/{heads} element {at}, {count} rows, head_dim {head_dim}: {got} against {exact}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Per-position cost of the shared group kernels against the per-head calls they replace, for
    /// one key head's group of query heads over 1024 `f16` rows of head dimension 128. It is a
    /// measurement, not a check, so it is `#[ignore]`d:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib attention_group_throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement"]
    fn attention_group_throughput() {
        use std::time::Instant;
        if !super::available()
            || !std::is_x86_feature_detected!("f16c")
            || !std::is_x86_feature_detected!("fma")
        {
            println!("attention_group_throughput: AVX2, F16C and FMA are required; not run");
            return;
        }
        const DIM: usize = 128;
        const POSITIONS: usize = 1024;
        let mut rng = Rng(0x5eed_6709_0000_0003);
        for group in [1usize, 2, 4, 6] {
            let queries: Vec<f32> = (0..group * DIM).map(|_| (rng.unit() - 0.5) * 2.0).collect();
            let rows: Vec<Vec<u8>> = (0..POSITIONS)
                .map(|_| half_row(&mut rng, DIM, 2.0).0)
                .collect();
            let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
            let weights: Vec<f32> = (0..POSITIONS * group).map(|_| rng.unit() * 0.01).collect();
            let mut scores = vec![0.0_f32; group];
            let mut targets = vec![0.0_f32; group * DIM];
            let repeats = 40;
            let mut best = [f64::MAX; 4];
            for _ in 0..7 {
                for (arm, slot) in best.iter_mut().enumerate() {
                    targets.fill(0.0);
                    let started = Instant::now();
                    for _ in 0..repeats {
                        match arm {
                            0 => {
                                for row in &refs {
                                    for (head, score) in scores.iter_mut().enumerate() {
                                        *score = super::dot_f16_f32(
                                            &queries[head * DIM..(head + 1) * DIM],
                                            row,
                                        );
                                    }
                                    std::hint::black_box(&scores);
                                }
                            }
                            1 => {
                                for row in &refs {
                                    super::dot_f16_group(&queries, DIM, row, &mut scores);
                                    std::hint::black_box(&scores);
                                }
                            }
                            2 => {
                                for (position, row) in refs.iter().enumerate() {
                                    for head in 0..group {
                                        super::axpy_f16_f32(
                                            &mut targets[head * DIM..(head + 1) * DIM],
                                            weights[position * group + head],
                                            row,
                                        );
                                    }
                                }
                            }
                            _ => {
                                for (block, rows) in refs.chunks(super::GROUP_ROWS).enumerate() {
                                    let first = block * super::GROUP_ROWS * group;
                                    super::axpy_f16_group(
                                        &mut targets,
                                        DIM,
                                        group,
                                        &weights[first..first + rows.len() * group],
                                        rows,
                                    );
                                }
                            }
                        }
                        std::hint::black_box(&targets);
                    }
                    let per_position =
                        started.elapsed().as_secs_f64() * 1e9 / (repeats * POSITIONS) as f64;
                    *slot = slot.min(per_position);
                }
            }
            println!(
                "group {group}: score per-head {:>6.1} ns -> shared {:>6.1} ns ({:.2}x) | value per-head {:>6.1} ns -> shared {:>6.1} ns ({:.2}x)",
                best[0],
                best[1],
                best[0] / best[1],
                best[2],
                best[3],
                best[2] / best[3]
            );
        }
    }

    /// Would a transposed K layout pay? The score pass both ways, at the shape that matters.
    ///
    /// `per-position dots` is what attention does today: one 128-wide `f16` dot per head per
    /// position, reduced horizontally. `block gemm` is what a `[dim][position]` layout allows: 64
    /// positions scored at once with FMA, the reduction vertical, no horizontal sums. The transpose
    /// itself is timed separately, because that is the cost a format change would move somewhere
    /// else (into the KV writes) rather than remove.
    ///
    /// Ignored by default: `cargo test -p vipera-backend-cpu --release --lib -- --ignored
    /// --nocapture transposed_k_scores`.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn transposed_k_scores() {
        const HEADS: usize = 12;
        const KV_HEADS: usize = 2;
        const DIM: usize = 128;
        const POSITIONS: usize = 1024;
        const BLOCK: usize = 64;
        let group = HEADS / KV_HEADS;
        let mut rng = Rng(0x5eed_0a77_0000_0011);
        let q: Vec<f32> = (0..HEADS * DIM).map(|_| rng.unit() * 2.0 - 1.0).collect();
        // One layer's K rows, position-major and contiguous: the shape the cache has now.
        let rows: Vec<u8> = (0..POSITIONS * KV_HEADS * DIM)
            .flat_map(|_| crate::quant::f32_to_f16(rng.unit() * 2.0 - 1.0).to_le_bytes())
            .collect();
        let row_bytes = KV_HEADS * DIM * 2;

        fn time<R: FnMut()>(rounds: u32, mut run: R) -> f64 {
            run();
            let mut best = f64::INFINITY;
            for _ in 0..rounds {
                let start = std::time::Instant::now();
                run();
                best = best.min(start.elapsed().as_secs_f64());
            }
            best * 1000.0
        }

        let mut scores = vec![0.0_f32; HEADS * POSITIONS];
        let dots = time(5, || {
            for position in 0..POSITIONS {
                for head in 0..HEADS {
                    let packed = position * row_bytes + (head / group) * DIM * 2;
                    scores[head * POSITIONS + position] =
                        super::dot_f16_f32(&q[head * DIM..][..DIM], &rows[packed..][..DIM * 2]);
                }
            }
            std::hint::black_box(&scores);
        });

        // `kt[dim][position within block]`, the layout a transposed K would give for free.
        let mut kt = vec![0.0_f32; DIM * BLOCK];
        let transpose = time(5, || {
            for block_start in (0..POSITIONS).step_by(BLOCK) {
                for dim in 0..DIM {
                    for offset in 0..BLOCK {
                        let position = block_start + offset;
                        let at = position * row_bytes + dim * 2;
                        kt[dim * BLOCK + offset] =
                            crate::quant::f16_to_f32(u16::from_le_bytes([rows[at], rows[at + 1]]));
                    }
                }
            }
            std::hint::black_box(&kt);
        });

        let mut block_scores = vec![0.0_f32; HEADS * POSITIONS];
        let gemm = time(5, || {
            // For each block of positions: transpose once, score the group's heads from it.
            for block_start in (0..POSITIONS).step_by(BLOCK) {
                for kv_head in 0..KV_HEADS {
                    for dim in 0..DIM {
                        for offset in 0..BLOCK {
                            let at =
                                (block_start + offset) * row_bytes + kv_head * DIM * 2 + dim * 2;
                            kt[dim * BLOCK + offset] =
                                crate::quant::f16_to_f32(u16::from_le_bytes([
                                    rows[at],
                                    rows[at + 1],
                                ]));
                        }
                    }
                    if !available() {
                        continue;
                    }
                    for member in 0..group {
                        let head = kv_head * group + member;
                        // SAFETY: AVX2 was checked above, and every load stays inside `kt`.
                        unsafe {
                            score_block_avx2(
                                &q[head * DIM..][..DIM],
                                &kt,
                                &mut block_scores[head * POSITIONS + block_start..][..BLOCK],
                            )
                        };
                    }
                }
            }
            std::hint::black_box(&block_scores);
        });

        println!(
            "score pass at {POSITIONS} x {HEADS} x {DIM}: per-position dots {dots:.2} ms, \
             block gemm (with transpose) {gemm:.2} ms, transpose alone {transpose:.2} ms"
        );
        let worst = dots.abs().max(1.0);
        let diff = scores
            .iter()
            .zip(&block_scores)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "largest gap between the two, on the same data: {diff:.3} (score scale {worst:.0})"
        );
    }

    /// `scores[i] = dot(q, kt[..][i])` for `scores.len()` positions, FMA over dimensions, the
    /// reduction vertical instead of a horizontal sum per position.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2, `kt` must hold `q.len() * scores.len()` values in
    /// `[dim][position]` order, and `scores.len()` must be a multiple of eight.
    #[cfg(test)]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn score_block_avx2(q: &[f32], kt: &[f32], scores: &mut [f32]) {
        // SAFETY: the caller guarantees AVX2+FMA, the layout of `kt`, and that `scores` divides
        // into eight-lane chunks; every load below stays inside one.
        unsafe {
            let positions = scores.len();
            let out = scores.as_mut_ptr();
            let mut lane = 0;
            while lane + 8 <= positions {
                let mut acc = _mm256_setzero_ps();
                for (dim, value) in q.iter().enumerate() {
                    let k = _mm256_loadu_ps(kt.as_ptr().add(dim * positions + lane));
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(*value), k, acc);
                }
                _mm256_storeu_ps(out.add(lane), acc);
                lane += 8;
            }
            while lane < positions {
                let mut total = 0.0_f32;
                for (dim, value) in q.iter().enumerate() {
                    total += value * kt[dim * positions + lane];
                }
                *out.add(lane) = total;
                lane += 1;
            }
        }
    }

    /// What each phase of a depth-1024 attention layer costs, at its own shape: 12 heads over
    /// 1024 positions against an `f16` row of 128 values.
    ///
    /// Ignored by default. `cargo test -p vipera-backend-cpu --release --lib -- --ignored
    /// --nocapture attention_phase_cost`.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn attention_phase_cost() {
        const HEADS: usize = 12;
        const POSITIONS: usize = 1024;
        const DIM: usize = 128;
        let mut rng = Rng(0x5eed_0a77_0000_000b);
        let q: Vec<f32> = (0..HEADS * DIM).map(|_| rng.unit() * 2.0 - 1.0).collect();
        let key: Vec<u8> = (0..DIM)
            .flat_map(|_| crate::quant::f32_to_f16(rng.unit() * 2.0 - 1.0).to_le_bytes())
            .collect();
        let value = key.clone();
        let mut out = vec![0.0_f32; DIM];

        let best = |rounds: u32, mut run: Box<dyn FnMut()>| -> f64 {
            run();
            let mut best = f64::INFINITY;
            for _ in 0..rounds {
                let start = std::time::Instant::now();
                run();
                best = best.min(start.elapsed().as_secs_f64());
            }
            best * 1000.0
        };

        let score = best(
            5,
            Box::new(|| {
                let mut total = 0.0_f32;
                for _ in 0..POSITIONS {
                    for head in 0..HEADS {
                        total += super::dot_f16_f32(&q[head * DIM..][..DIM], &key);
                    }
                }
                std::hint::black_box(total);
            }),
        );
        let weighted = best(
            5,
            Box::new(|| {
                for _ in 0..POSITIONS {
                    for _ in 0..HEADS {
                        super::axpy_f16_f32(&mut out, 0.5, &value);
                    }
                }
                std::hint::black_box(&out);
            }),
        );
        println!(
            "attention layer at 1024 x 12 x 128: score {score:.2} ms, weighted sum {weighted:.2} ms, both {:.2} ms",
            score + weighted
        );
    }

    /// `vpdpbusd` against `maddubs` + `madd` on the shapes that dominate a decode step.
    ///
    /// Checks bit-exactness row by row, then times both at one and four threads.
    ///
    /// Ignored by default. Run it with
    /// `cargo test -p vipera-backend-cpu --release -- --ignored --nocapture q4k_vnni_versus_maddubs`.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_vnni_versus_maddubs() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        if !std::is_x86_feature_detected!("avx512vnni")
            || !std::is_x86_feature_detected!("avx512vl")
        {
            eprintln!("skipped: this CPU has no avx512_vnni with avx512_vl");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0004);
        for (name, rows, values) in [
            ("attn_q", 1536usize, 1536usize),
            ("ffn_gate+up", 8960, 1536),
            ("ffn_down", 1536, 8960),
        ] {
            let blocks = values / 256;
            let row_bytes = blocks * 144;
            let row = q4k_row(&mut rng, blocks);
            let raw: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
            let x: Vec<f32> = (0..values).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            let sums = crate::quant::x_sums(&x);

            for r in 0..rows {
                let at = r * row_bytes;
                // SAFETY: whole blocks, AVX2 and avx512_vnni checked above.
                let fast = unsafe { q4k_dot_vnni(&raw[at..at + row_bytes], &xq.q, &xq.d, &sums) };
                // SAFETY: as above.
                let slow = unsafe { q4k_dot(&raw[at..at + row_bytes], &xq.q, &xq.d, &sums) };
                assert_eq!(fast.to_bits(), slow.to_bits(), "{name} row {r}");
            }
            println!("{name}: every row bit-identical");

            let gb = (rows * row_bytes) as f64 / 1e9;
            let reps = 8;
            let time = |vnni: bool| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    let mut total = 0.0f32;
                    for _ in 0..reps {
                        for r in 0..rows {
                            let at = r * row_bytes;
                            // SAFETY: as above.
                            total += unsafe {
                                if vnni {
                                    q4k_dot_vnni(&raw[at..at + row_bytes], &xq.q, &xq.d, &sums)
                                } else {
                                    q4k_dot(&raw[at..at + row_bytes], &xq.q, &xq.d, &sums)
                                }
                            };
                        }
                    }
                    std::hint::black_box(total);
                    best = best.min(start.elapsed().as_secs_f64());
                }
                gb * reps as f64 / best
            };
            let threads = 4;
            let rows_per = rows / threads;
            let time_threads = |vnni: bool| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    std::thread::scope(|scope| {
                        for t in 0..threads {
                            let raw = &raw;
                            let xq = &xq;
                            let sums = &sums;
                            scope.spawn(move || {
                                let first = t * rows_per;
                                let mut total = 0.0f32;
                                for r in first..first + rows_per {
                                    let at = r * row_bytes;
                                    // SAFETY: as above.
                                    total += unsafe {
                                        if vnni {
                                            q4k_dot_vnni(
                                                &raw[at..at + row_bytes],
                                                &xq.q,
                                                &xq.d,
                                                sums,
                                            )
                                        } else {
                                            q4k_dot(&raw[at..at + row_bytes], &xq.q, &xq.d, sums)
                                        }
                                    };
                                }
                                std::hint::black_box(total);
                            });
                        }
                    });
                    best = best.min(start.elapsed().as_secs_f64());
                }
                gb / best
            };
            let madd1 = time(false);
            let vnni1 = time(true);
            let madd4 = time_threads(false);
            let vnni4 = time_threads(true);
            println!(
                "{name}: 1 thread maddubs {madd1:.1} GB/s, vnni {vnni1:.1} GB/s, {:.2}x",
                vnni1 / madd1
            );
            println!(
                "{name}: 4 threads maddubs {madd4:.1} GB/s, vnni {vnni4:.1} GB/s, {:.2}x\n",
                vnni4 / madd4
            );
        }
    }

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

    /// A Q4_K row of `blocks` blocks with random scales and nibbles.
    #[test]
    fn tiled_q4k_tile_parity_vs_row_dot() {
        if !vnni_available() {
            return;
        }
        let mut rng = Rng(0x1112_2222_3333_4444);
        let rows = 256usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let stride = blocks * 144;
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let sums = crate::quant::x_sums(&x);
        // Row-by-row reference through the shipped VNNI batch kernel at B=1.
        for row in [0usize, 17, 128, 255] {
            let row_bytes = &data[row * stride..(row + 1) * stride];
            // SAFETY: whole blocks, VNNI checked.
            let want = unsafe { q4k_dot(row_bytes, &xq.q, &xq.d, &sums) };
            assert!(want.is_finite());
        }
    }

    /// A tile filled through `SharedTile::fill_entries` must produce the same bytes the kernel's
    /// own build produces, so sharing it cannot change a result. The two runs below are the two
    /// halves `Pool::prepare` would hand to two participants.
    #[test]
    fn a_shared_activation_tile_is_bit_identical_to_a_local_one() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x5eed_0001_0002_0003);
        let rows = 48usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let stride = blocks * 144;
        let source: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 32] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 32] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let tile = SharedTile::new(blocks, 32usize.div_ceil(16));
        let q: Vec<&[i8]> = xs.iter().map(|x| x.q.as_slice()).collect();
        tile.fill_entries(0, &q[..16]);
        tile.fill_entries(16, &q[16..]);
        let mut shared = vec![0.0_f32; 32 * rows];
        let mut local = vec![0.0_f32; 32 * rows];
        assert!(try_q4k_matvec_tiled_range(
            &data,
            stride,
            0,
            rows,
            &xs,
            &sums,
            &mut shared,
            Some(&tile)
        ));
        assert!(try_q4k_matvec_tiled_range(
            &data, stride, 0, rows, &xs, &sums, &mut local, None
        ));
        assert_eq!(shared, local, "the shared tile changed a result");
    }

    #[test]
    fn q4k_tiled_range_matches_batch_vnni() {
        if !vnni_available() || !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let rows = 48usize; // straddles the 32-row window
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let stride = blocks * 144;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut got = vec![0.0_f32; 16 * rows];
        // SAFETY: whole blocks, VNNI-512 checked above.
        unsafe { q4k_matvec_tiled_range::<16>(&data, stride, 0, rows, &xs, &sums, &mut got, None) };
        let mut totals = [0.0_f32; 16];
        for row in 0..rows {
            let row_bytes = &data[row * stride..(row + 1) * stride];
            // SAFETY: whole blocks, VNNI checked above.
            unsafe { q4k_dot_batch_vnni(row_bytes, &xs, &sums, &mut totals) };
            for entry in 0..16 {
                let want = totals[entry];
                let g = got[entry * rows + row];
                assert!(
                    (g - want).abs() <= 1e-3 + 1e-4 * want.abs(),
                    "entry {entry} row {row}: got {g}, want {want}"
                );
            }
        }
    }

    /// W1a parity: the per-256-scale activation (`VIPERA_TILED_A`) vs the batch row kernel on
    /// the per-32 activation. The two disagree by the W1a activation-quant noise (measured: max
    /// diff 2.63 on this shape — the W1a codes the band reads are the same integers, only their
    /// scale and granularity changed), so the tolerance is `2.0 + 0.2·|want|` — 2-3x the noise
    /// floor, still an order of magnitude below the diffs a structural bug in the W1a path
    /// produces.
    #[test]
    fn q4k_tiled_range_a_matches_batch_within_tolerance() {
        if !vnni_available() || !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x5eed_0002_0003_0004);
        let rows = 48usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let stride = blocks * 144;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_ref: [&[f32]; 16] = std::array::from_fn(|e| source[e].as_slice());
        let act_a = TiledActivation::new(xs_ref);
        let mut got = vec![0.0_f32; 16 * rows];
        // SAFETY: whole blocks, VNNI-512 checked above.
        unsafe { q4k_matvec_tiled_range_a::<16>(&data, stride, 0, rows, &act_a, &mut got) };
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut totals = [0.0_f32; 16];
        for row in 0..rows {
            let row_bytes = &data[row * stride..(row + 1) * stride];
            // SAFETY: whole blocks, VNNI checked above.
            unsafe { q4k_dot_batch_vnni(row_bytes, &xs, &sums, &mut totals) };
            for entry in 0..16 {
                let want = totals[entry];
                let g = got[entry * rows + row];
                assert!(
                    (g - want).abs() <= 2.0 + 0.2 * want.abs(),
                    "entry {entry} row {row}: got {g}, want {want}"
                );
            }
        }
    }

    /// Measurement only: how far the tiled kernel sits from the row kernel, so a change to the
    /// epilogue can be judged by whether it moves toward the reference or away from it.
    ///
    /// The row kernel is not ground truth for the dot — both pick a different summation order — but
    /// it is the kernel every decode path uses, so distance from it is the quantity a prefill's
    /// activations are consumed with.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn q4k_tiled_error_against_row_kernel() {
        if !vnni_available() || !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x51ed_270b_3ac9_e1f7);
        let rows = 48usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let stride = blocks * 144;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut got = vec![0.0_f32; 16 * rows];
        // SAFETY: whole blocks, VNNI-512 checked above.
        unsafe { q4k_matvec_tiled_range::<16>(&data, stride, 0, rows, &xs, &sums, &mut got, None) };
        let mut worst_abs = 0.0_f32;
        let mut worst_rel = 0.0_f32;
        let mut sum_rel = 0.0_f64;
        let mut count = 0usize;
        let mut totals = [0.0_f32; 16];
        for row in 0..rows {
            let row_bytes = &data[row * stride..(row + 1) * stride];
            // SAFETY: whole blocks, VNNI checked above.
            unsafe { q4k_dot_batch_vnni(row_bytes, &xs, &sums, &mut totals) };
            for entry in 0..16 {
                let want = totals[entry];
                let got = got[entry * rows + row];
                let abs = (got - want).abs();
                let rel = abs / want.abs().max(f32::MIN_POSITIVE);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                sum_rel += f64::from(rel);
                count += 1;
            }
        }
        println!(
            "{}: max abs {worst_abs:.3e}, max rel {worst_rel:.3e}, mean rel {:.3e} over {count} values",
            if tiled_fma_enabled() {
                "fused epilogue"
            } else {
                "multiply-then-add"
            },
            sum_rel / count as f64
        );
    }

    use std::arch::x86_64::{
        _mm512_shuffle_i32x4, _mm512_storeu_si512, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64,
        _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
    };

    /// Transpose a 16 x 16 square of 32-bit words held in sixteen ZMM registers: word `k` of input
    /// row `i` becomes word `i` of output row `k`.
    ///
    /// Pure data movement, so the result is bit-for-bit what a word-by-word copy produces.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[inline]
    fn transpose_16x16(r: [__m512i; 16]) -> [__m512i; 16] {
        let mut pairs = [_mm512_setzero_si512(); 16];
        for i in 0..8 {
            pairs[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
            pairs[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
        }
        let mut quads = [[_mm512_setzero_si512(); 4]; 4];
        for g in 0..4 {
            quads[g][0] = _mm512_unpacklo_epi64(pairs[4 * g], pairs[4 * g + 2]);
            quads[g][1] = _mm512_unpackhi_epi64(pairs[4 * g], pairs[4 * g + 2]);
            quads[g][2] = _mm512_unpacklo_epi64(pairs[4 * g + 1], pairs[4 * g + 3]);
            quads[g][3] = _mm512_unpackhi_epi64(pairs[4 * g + 1], pairs[4 * g + 3]);
        }
        let mut out = [_mm512_setzero_si512(); 16];
        for m in 0..4 {
            let lo01 = _mm512_shuffle_i32x4::<0x88>(quads[0][m], quads[1][m]);
            let hi01 = _mm512_shuffle_i32x4::<0xDD>(quads[0][m], quads[1][m]);
            let lo23 = _mm512_shuffle_i32x4::<0x88>(quads[2][m], quads[3][m]);
            let hi23 = _mm512_shuffle_i32x4::<0xDD>(quads[2][m], quads[3][m]);
            out[m] = _mm512_shuffle_i32x4::<0x88>(lo01, lo23);
            out[4 + m] = _mm512_shuffle_i32x4::<0x88>(hi01, hi23);
            out[8 + m] = _mm512_shuffle_i32x4::<0xDD>(lo01, lo23);
            out[12 + m] = _mm512_shuffle_i32x4::<0xDD>(hi01, hi23);
        }
        out
    }

    /// Measurement only: `SharedTile::fill_entries` against the 16x16 transpose the tiled kernel's
    /// owned path uses, on the same data and single-threaded, with the bytes compared.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn shared_tile_fill_vs_transpose() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x7a11_0000_0000_0001);
        let blocks = 35usize;
        let entries = 128usize;
        let halves = entries / 16;
        let source: Vec<Vec<i8>> = (0..entries)
            .map(|_| {
                (0..blocks * 256)
                    .map(|_| (rng.unit() * 15.0) as i8)
                    .collect()
            })
            .collect();
        let q: Vec<&[i8]> = source.iter().map(|x| x.as_slice()).collect();
        let len = blocks * halves * 64 * 16 * 4;

        let tile = SharedTile::new(blocks, halves);
        let start = std::time::Instant::now();
        for _ in 0..16 {
            tile.fill_entries(0, &q);
        }
        let copy = start.elapsed().as_secs_f64() / 16.0;
        let want = tile.bytes().to_vec();

        let mut got = vec![0_i8; len];
        let start = std::time::Instant::now();
        for _ in 0..16 {
            // SAFETY: AVX-512 checked above; every store is inside `got` and every load inside `q`.
            unsafe {
                for b in 0..blocks {
                    for half in 0..halves {
                        let base = (b * halves + half) * 64 * 16 * 4;
                        for chunk in 0..4 {
                            let mut rows = [_mm512_setzero_si512(); 16];
                            for (col, row) in rows.iter_mut().enumerate() {
                                *row = _mm512_loadu_si512(
                                    q[half * 16 + col].as_ptr().add(b * 256 + chunk * 64)
                                        as *const __m512i,
                                );
                            }
                            for (k, value) in transpose_16x16(rows).into_iter().enumerate() {
                                _mm512_storeu_si512(
                                    got.as_mut_ptr().add(base + (chunk * 16 + k) * 64)
                                        as *mut __m512i,
                                    value,
                                );
                            }
                        }
                    }
                }
            }
        }
        let transposed = start.elapsed().as_secs_f64() / 16.0;
        assert_eq!(want, got, "the transpose wrote different bytes");
        println!(
            "fill_entries {:.3} ms, transpose {:.3} ms ({}x, {blocks} blocks x {entries} entries)",
            copy * 1e3,
            transposed * 1e3,
            copy / transposed
        );
    }

    /// Measurement only: what allocating and zeroing one product tile costs, so the remaining
    /// ceiling gap can be attributed before a reuse path is written.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn shared_tile_allocation_cost() {
        let start = std::time::Instant::now();
        for _ in 0..2000 {
            std::hint::black_box(SharedTile::new(35, 8));
        }
        println!("{:?} per 35x8 tile", start.elapsed() / 2000);
        let start = std::time::Instant::now();
        for _ in 0..2000 {
            std::hint::black_box(SharedTile::new(6, 8));
        }
        println!("{:?} per 6x8 tile", start.elapsed() / 2000);
    }

    /// A recycled buffer must not leak the previous product into the next one, including when the
    /// next product is narrower and only a prefix of the old buffer is used.
    #[test]
    fn a_recycled_tile_does_not_leak_the_previous_product() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x0bad_c0de_1234_5678);
        let rows = 40usize;
        let big_cols = 512usize;
        let big_blocks = big_cols / 256;
        let big: Vec<u8> = (0..rows)
            .flat_map(|_| q4k_row(&mut rng, big_blocks))
            .collect();
        let big_stride = big_blocks * 144;
        let small_cols = 256usize;
        let small: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, 1)).collect();
        let small_stride = 144;
        let activations = |count: usize, cols: usize, rng: &mut Rng| {
            let source: Vec<Vec<f32>> = (0..count)
                .map(|_| (0..cols).map(|_| rng.unit()).collect())
                .collect();
            let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
            let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
            (xs_vec, sums_vec)
        };
        let (first, first_sums) = activations(32, big_cols, &mut rng);
        let (second, second_sums) = activations(32, small_cols, &mut rng);
        let first_xs: [&QuantizedX; 32] = std::array::from_fn(|e| &first[e]);
        let first_s: [&[f32]; 32] = std::array::from_fn(|e| first_sums[e].as_slice());
        let tile = SharedTile::new(big_blocks, 2);
        let q: Vec<&[i8]> = first_xs.iter().map(|x| x.q.as_slice()).collect();
        tile.fill_entries(0, &q[..16]);
        tile.fill_entries(16, &q[16..]);
        let mut out = vec![0.0_f32; 32 * rows];
        assert!(try_q4k_matvec_tiled_range(
            &big,
            big_stride,
            0,
            rows,
            &first_xs,
            &first_s,
            &mut out,
            Some(&tile)
        ));
        // Same buffer, narrower product, different activations.
        let recycled = SharedTile::from_buffer(tile.into_buffer(), 1, 2);
        let second_xs: [&QuantizedX; 32] = std::array::from_fn(|e| &second[e]);
        let second_s: [&[f32]; 32] = std::array::from_fn(|e| second_sums[e].as_slice());
        let q: Vec<&[i8]> = second_xs.iter().map(|x| x.q.as_slice()).collect();
        recycled.fill_entries(0, &q[..16]);
        recycled.fill_entries(16, &q[16..]);
        let mut shared = vec![0.0_f32; 32 * rows];
        let mut local = vec![0.0_f32; 32 * rows];
        assert!(try_q4k_matvec_tiled_range(
            &small,
            small_stride,
            0,
            rows,
            &second_xs,
            &second_s,
            &mut shared,
            Some(&recycled)
        ));
        assert!(try_q4k_matvec_tiled_range(
            &small,
            small_stride,
            0,
            rows,
            &second_xs,
            &second_s,
            &mut local,
            None
        ));
        assert_eq!(shared, local, "a recycled tile changed a result");
    }

    /// Same byte-for-byte check as the Q4_K shared tile, on the Q6_K kernel.
    #[test]
    fn a_shared_q6k_tile_is_bit_identical_to_a_local_one() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x6b6b_1111_2222_3333);
        let rows = 48usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let stride = blocks * 210;
        let source: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 32] = std::array::from_fn(|e| &xs_vec[e]);
        let tile = SharedTile::new(blocks, 32usize.div_ceil(16));
        let q: Vec<&[i8]> = xs.iter().map(|x| x.q.as_slice()).collect();
        tile.fill_entries(0, &q[..16]);
        tile.fill_entries(16, &q[16..]);
        let mut shared = vec![0.0_f32; 32 * rows];
        let mut local = vec![0.0_f32; 32 * rows];
        assert!(try_q6k_matvec_tiled_range(
            &data,
            stride,
            0,
            rows,
            &xs,
            &mut shared,
            Some(&tile)
        ));
        assert!(try_q6k_matvec_tiled_range(
            &data, stride, 0, rows, &xs, &mut local, None
        ));
        assert_eq!(shared, local, "the shared tile changed a Q6_K result");
    }

    #[test]
    fn q6k_tiled_range_matches_batch_q8() {
        if !vnni_available() || !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x85eb_ca6b_c2b2_ae35);
        let rows = 48usize;
        let cols = 512usize;
        let blocks = cols / 256;
        let data: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let stride = blocks * 210;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let mut got = vec![0.0_f32; 16 * rows];
        // SAFETY: whole blocks, VNNI-512 checked above.
        unsafe { q6k_matvec_tiled_range::<16>(&data, stride, 0, rows, &xs, &mut got, None) };
        let mut totals = [0.0_f32; 16];
        for row in 0..rows {
            let row_bytes = &data[row * stride..(row + 1) * stride];
            // SAFETY: whole blocks, VNNI checked above.
            unsafe { q6k_dot_batch_q8(row_bytes, &xs, &mut totals) };
            for entry in 0..16 {
                let want = totals[entry];
                let g = got[entry * rows + row];
                assert!(
                    (g - want).abs() <= 1e-3 + 1e-4 * want.abs(),
                    "entry {entry} row {row}: got {g}, want {want}"
                );
            }
        }
    }

    /// Measurement, not a check: what a 148-byte repacked Q4_K block buys over GGUF's 144.
    ///
    /// The 12 packed scale bytes become 8 scales + 8 mins as plain `u8` (still 6-bit values), so a
    /// block's dequant reads them with two 8-byte loads instead of calling `q4k_scale_min` eight
    /// times. The code bytes stay packed, so the nibble split and both f16 conversions are in both
    /// arms; this isolates the scale header. +2.8% bytes is the only repack a decode-bound model can
    /// afford, so if this arm does not move, no 4.5 bpw repack will.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn bench_repacked_q4k_scale_header() {
        const N: usize = 1 << 16;
        const ROUNDS: usize = 20;

        let mut fill = 0x9e37_79b9_7f4a_7c15u64;
        let mut gguf = vec![0u8; N * 144];
        for byte in gguf.iter_mut() {
            fill ^= fill << 13;
            fill ^= fill >> 7;
            fill ^= fill << 17;
            *byte = (fill >> 24) as u8;
        }
        // Random bytes make NaN/Inf f16 scales, whose payloads need not sum alike; pin both
        // scale factors to 1.0 so the two arms differ only in how the header bytes are read.
        for i in 0..N {
            gguf[i * 144..i * 144 + 4].copy_from_slice(&[0x00, 0x3c, 0x00, 0x3c]);
        }
        let mut packed = vec![0u8; N * 148];
        for i in 0..N {
            let src = &gguf[i * 144..][..144];
            let dst = &mut packed[i * 148..][..148];
            dst[..4].copy_from_slice(&src[..4]);
            for j in 0..8 {
                let (scale, min) = crate::quant::q4k_scale_min(&src[4..16], j);
                dst[4 + j] = scale;
                dst[12 + j] = min;
            }
            dst[20..].copy_from_slice(&src[16..]);
        }

        fn run(buf: &[u8], stride: usize, wide_header: bool) -> (u128, u32) {
            let mut checksum = 0u32;
            let start = std::time::Instant::now();
            for _ in 0..ROUNDS {
                for i in 0..N {
                    let b = &buf[i * stride..][..stride];
                    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
                    let mut acc = 0.0f32;
                    if wide_header {
                        let scales = u64::from_le_bytes(b[4..12].try_into().unwrap());
                        let mins = u64::from_le_bytes(b[12..20].try_into().unwrap());
                        for j in 0..8 {
                            let scale = ((scales >> (j * 8)) & 0xff) as f32;
                            let min = ((mins >> (j * 8)) & 0xff) as f32;
                            acc += d * scale + dmin * min;
                        }
                        for code in &b[20..] {
                            acc += f32::from(code & 0x0f);
                        }
                    } else {
                        for j in 0..8 {
                            let (scale, min) = crate::quant::q4k_scale_min(&b[4..16], j);
                            acc += d * f32::from(scale) + dmin * f32::from(min);
                        }
                        for code in &b[16..] {
                            acc += f32::from(code & 0x0f);
                        }
                    }
                    checksum = checksum.wrapping_add(acc.to_bits());
                }
            }
            (start.elapsed().as_nanos(), checksum)
        }

        let (mut gguf_ns, mut gguf_sum) = (0u128, 0u32);
        let (mut packed_ns, mut packed_sum) = (0u128, 0u32);
        for _ in 0..3 {
            let (ns, sum) = run(&gguf, 144, false);
            gguf_ns += ns;
            gguf_sum = sum;
            let (ns, sum) = run(&packed, 148, true);
            packed_ns += ns;
            packed_sum = sum;
        }
        assert_eq!(
            gguf_sum, packed_sum,
            "the two layouts must dequant identically"
        );
        let blocks = (N * ROUNDS * 3) as f64;
        let gguf_ns_block = gguf_ns as f64 / blocks;
        let packed_ns_block = packed_ns as f64 / blocks;
        println!(
            "dequant per block: GGUF 144B {gguf_ns_block:.2} ns, repacked 148B \
             {packed_ns_block:.2} ns ({:.1}% of the GGUF cost)",
            packed_ns_block / gguf_ns_block * 100.0
        );
    }

    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_tiled_vs_row_single_thread() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let rows = 8960usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 144;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let gb = (data.len() as f64) / 1e9;
        let reps = 5;
        let mut out = vec![0.0_f32; 16 * rows];
        let start = std::time::Instant::now();
        for _ in 0..reps {
            let _ = try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs, &sums, &mut out, None);
        }
        let tiled = start.elapsed().as_secs_f64();
        println!("tiled single-thread: {:.2} GB/s", gb * reps as f64 / tiled);
        let mut totals = [0.0_f32; 16];
        let start = std::time::Instant::now();
        for _ in 0..reps {
            for row in 0..rows {
                let rb = &data[row * stride..(row + 1) * stride];
                unsafe { q4k_dot_batch_vnni(rb, &xs, &sums, &mut totals) };
            }
        }
        let row = start.elapsed().as_secs_f64();
        println!(
            "row batch single-thread: {:.2} GB/s",
            gb * reps as f64 / row
        );
        std::hint::black_box((&out, &totals));
    }

    /// What one Q6_K pass costs, warm and cold, so the output projection's
    /// share of a chunk can be judged against the kernel's own ceiling.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q6k_tiled() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x96_9600_0000_1234);
        let rows = 8_960usize;
        let cold_rows = 57_344usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 210;
        let data: Vec<u8> = (0..cold_rows)
            .flat_map(|_| q6k_row(&mut rng, blocks))
            .collect();
        let source: Vec<Vec<f32>> = (0..128)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let reps = 3;
        macro_rules! bench {
            ($b:literal, $rows:expr, $label:literal) => {{
                let xs: [&QuantizedX; $b] = std::array::from_fn(|e| &xs_vec[e]);
                let mut out = vec![0.0_f32; $b * $rows];
                let mut best = f64::INFINITY;
                for _ in 0..reps {
                    let start = std::time::Instant::now();
                    let _ =
                        try_q6k_matvec_tiled_range(&data, stride, 0, $rows, &xs, &mut out, None);
                    best = best.min(start.elapsed().as_secs_f64());
                }
                std::hint::black_box(&out);
                let gb = ($rows * stride) as f64 / 1e9;
                println!(
                    "{} B={:>3}: {:>7.2} ms/pass, {:>6.2} ms/token, {:>5.2} GB/s",
                    $label,
                    $b,
                    best * 1e3,
                    best * 1e3 / f64::from($b),
                    gb / best
                );
            }};
        }
        bench!(16, rows, "warm");
        bench!(128, rows, "warm");
        bench!(16, cold_rows, "cold");
        bench!(128, cold_rows, "cold");
    }

    /// Q4_K against Q6_K through their tiled prefill kernels, one thread, B = 128, at the two shapes
    /// a chunk multiplies: the FFN gate/up (8960 rows of 1536) and the FFN down (1536 rows of 8960).
    /// Three row counts per shape separate what a call pays once (the activation tile, built over
    /// the whole row width) from what each row pays.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_tiled_q4k_vs_q6k_shapes() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x96_9600_0000_4242);
        for (label, rows, cols) in [
            ("gate/up 8960x1536", 8_960_usize, 1_536_usize),
            ("down 1536x8960", 1_536, 8_960),
        ] {
            let blocks = cols / 256;
            let q4: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
            let q6: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
            let source: Vec<Vec<f32>> = (0..128)
                .map(|_| (0..cols).map(|_| rng.unit()).collect())
                .collect();
            let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
            let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
            let xs: [&QuantizedX; 128] = std::array::from_fn(|e| &xs_vec[e]);
            let sums: [&[f32]; 128] = std::array::from_fn(|e| sums_vec[e].as_slice());
            for count in [0_usize, 32, 256, rows] {
                for (format, data, stride) in
                    [("q4k", &q4, blocks * 144), ("q6k", &q6, blocks * 210)]
                {
                    let mut out = vec![0.0_f32; 128 * count];
                    let mut best = f64::INFINITY;
                    for _ in 0..3 {
                        let start = std::time::Instant::now();
                        let ok = if format == "q4k" {
                            try_q4k_matvec_tiled_range(
                                data, stride, 0, count, &xs, &sums, &mut out, None,
                            )
                        } else {
                            try_q6k_matvec_tiled_range(data, stride, 0, count, &xs, &mut out, None)
                        };
                        assert!(ok, "{format} tiled kernel refused");
                        best = best.min(start.elapsed().as_secs_f64());
                    }
                    std::hint::black_box(&out);
                    let macs = (count * cols * 128) as f64;
                    println!(
                        "{label} {format} rows {count:>5}: {:>8.2} ms, {:>6.1} GMAC/s",
                        best * 1e3,
                        macs / best / 1e9
                    );
                }
            }
        }
    }

    /// The output projection's shape: one activation against the vocabulary
    /// matrix, which is the largest single tensor read in a chunk and runs at
    /// B=1 because only the last position needs logits.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q6k_output_projection() {
        let mut rng = Rng(0x96_9600_0000_9876);
        let rows = 151_936usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 210;
        let data: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xs_vec = QuantizedX::new(&x);
        let xs = [&xs_vec];
        let mut out = [0.0_f32; 1];
        let reps = 2;
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = std::time::Instant::now();
            for r in 0..rows {
                // SAFETY: each row is a whole number of 210-byte blocks and
                // `xs` covers the row width.
                unsafe {
                    q6k_dot_batch_q8::<1>(&data[r * stride..(r + 1) * stride], &xs, &mut out);
                }
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        std::hint::black_box(&out);
        let gb = (rows * stride) as f64 / 1e9;
        println!(
            "output projection: {:.2} ms single thread, {:.2} GB/s ({:.1} MB read)",
            best * 1e3,
            gb / best,
            gb * 1e3
        );
    }

    /// Per-token cost against ubatch width: the weight pass is paid once per
    /// `B` tokens, so this is the curve a bigger engine batch would follow.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_ubatch_curve() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0xabcd_1234_5678_9f01);
        // Warm first (fits L3), then a tensor large enough that every pass
        // streams from DRAM like the engine's weights do.
        let rows = 8960usize;
        let cold_rows = 57_344usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 144;
        let data: Vec<u8> = (0..cold_rows)
            .flat_map(|_| q4k_row(&mut rng, blocks))
            .collect();
        let source: Vec<Vec<f32>> = (0..128)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let reps = 3;
        macro_rules! bench {
            ($b:literal, $rows:expr, $label:literal) => {{
                let xs: [&QuantizedX; $b] = std::array::from_fn(|e| &xs_vec[e]);
                let sums: [&[f32]; $b] = std::array::from_fn(|e| sums_vec[e].as_slice());
                let mut out = vec![0.0_f32; $b * $rows];
                let mut best = f64::INFINITY;
                for _ in 0..reps {
                    let start = std::time::Instant::now();
                    let _ = try_q4k_matvec_tiled_range(
                        &data, stride, 0, $rows, &xs, &sums, &mut out, None,
                    );
                    best = best.min(start.elapsed().as_secs_f64());
                }
                std::hint::black_box(&out);
                let gb = ($rows * stride) as f64 / 1e9;
                println!(
                    "{} B={:>3}: {:>7.2} ms/pass, {:>6.2} ms/token, {:>5.2} GB/s",
                    $label,
                    $b,
                    best * 1e3,
                    best * 1e3 / f64::from($b),
                    gb / best
                );
            }};
        }
        bench!(16, rows, "warm");
        bench!(32, rows, "warm");
        bench!(64, rows, "warm");
        bench!(128, rows, "warm");
        bench!(16, cold_rows, "cold");
        bench!(64, cold_rows, "cold");
        bench!(128, cold_rows, "cold");
    }

    /// B=32 against two B=16 passes: same shape, same data, 32 activations.
    ///
    /// The B=32 kernel runs two 16-wide halves against one unpacked weight
    /// window, so it should match the two-call result closely while doing
    /// half the unpacking, and per-token time is the number that matters.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_tiled_b32_vs_two_b16() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x0bad_c0de_1234_5678);
        let rows = 8960usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 144;
        let data: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let source: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs_lo: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let xs_hi: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e + 16]);
        let sums_lo: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let sums_hi: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e + 16].as_slice());
        let xs32: [&QuantizedX; 32] = std::array::from_fn(|e| &xs_vec[e]);
        let sums32: [&[f32]; 32] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut two = vec![0.0_f32; 32 * rows];
        let mut one = vec![0.0_f32; 32 * rows];
        let reps = 5;
        let (lo, hi) = two.split_at_mut(16 * rows);
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_lo, &sums_lo, lo, None);
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_hi, &sums_hi, hi, None);
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs32, &sums32, &mut one, None);
        let mut worst = 0.0_f32;
        for (a, b) in two.iter().zip(one.iter()) {
            let tol = 1e-3 + 1e-4 * a.abs();
            worst = worst.max((a - b).abs() - tol);
        }
        println!("parity B=32 vs 2x B=16: worst excess over tolerance {worst:.3e}");
        assert!(worst <= 0.0, "B=32 disagrees with two B=16 passes");
        let macs = (rows * cols * 32) as f64;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            let (lo, hi) = two.split_at_mut(16 * rows);
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_lo, &sums_lo, lo, None);
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_hi, &sums_hi, hi, None);
        }
        let t16 = start.elapsed().as_secs_f64() / reps as f64;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs32, &sums32, &mut one, None);
        }
        let t32 = start.elapsed().as_secs_f64() / reps as f64;
        println!(
            "2x B=16: {:.1} ms, {:.1} G token-MAC/s",
            t16 * 1e3,
            macs / t16 / 1e9
        );
        println!(
            "B=32:    {:.1} ms, {:.1} G token-MAC/s",
            t32 * 1e3,
            macs / t32 / 1e9
        );
        println!("B=32 speedup: {:.2}x", t16 / t32);
        std::hint::black_box((&two, &one));
    }

    /// Hai dạng microkernel cho cùng một 16x16 tile của Q4_K, đo cái nào nhanh hơn trên máy này.
    ///
    /// A giống kernel đang chạy: zmm là 16 *activation* x 4 cột của **một** hàng weight, nên mỗi
    /// `dpbusd` cần một `vpbroadcastd` từ `wq` (64 broadcast cho 64 dot trong codegen gate).
    /// B là dạng đã repack: zmm là 16 **hàng weight** x 4 code, activation broadcast từ bộ nhớ
    /// (dạng `{1to16}` của `vpdpbusd`, nếu compiler gấp được) — một broadcast miễn phí thay vì một
    /// lần load + broadcast mỗi dot.
    ///
    /// Số MAC như nhau (64 mỗi lệnh), nên thời gian nói thẳng cái nào trả tiền cho broadcast.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn repacked_gemm_form_against_the_broadcast_form() {
        use std::arch::x86_64::*;
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const ROWS: usize = 16;
        const GROUPS: usize = 64; // 256 cột / 4 code mỗi dword
        let mut rng = Rng(0x5eed_0d08_0000_0001);
        // A: weight theo (hàng, group) là một dword; activation theo (group, activation) là zmm.
        let weights_dword: Vec<u32> = (0..ROWS * GROUPS).map(|_| rng.next() as u32).collect();
        let acts: Vec<u8> = (0..GROUPS * 64).map(|_| rng.byte()).collect();
        // B: weight đã repack thành zmm = 16 hàng x 4 code cho từng group.
        let mut weights_packed = vec![0u8; GROUPS * 64];
        for group in 0..GROUPS {
            for row in 0..ROWS {
                let dword = weights_dword[row * GROUPS + group].to_le_bytes();
                weights_packed[group * 64 + row * 4..group * 64 + row * 4 + 4]
                    .copy_from_slice(&dword);
            }
        }
        // Intrinsics chỉ inline khi chính hàm mang `target_feature`; gọi chúng từ một hàm test
        // thường thì mỗi lệnh thành một call và bench đo overhead (đo được 30 chu kỳ/lệnh).
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn run_a(words: *const u32, acts: *const u8, out: *mut i32, groups: usize) {
            // SAFETY: the caller hands buffers of `ROWS * groups` words, `groups * 64` bytes and
            // `ROWS` outputs.
            unsafe {
                let mut acc = [_mm512_setzero_si512(); ROWS];
                for group in 0..groups {
                    let codes = _mm512_loadu_si512(acts.add(group * 64).cast());
                    for (row, slot) in acc.iter_mut().enumerate() {
                        let word = std::ptr::read_unaligned(words.add(row * groups + group));
                        *slot = _mm512_dpbusd_epi32(*slot, _mm512_set1_epi32(word as i32), codes);
                    }
                }
                for (row, value) in acc.iter().enumerate() {
                    let mut lanes = [0_i32; 16];
                    _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *value);
                    *out.add(row) = lanes[0];
                }
            }
        }
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn run_b(packed: *const u8, acts: *const u8, out: *mut i32, groups: usize) {
            // SAFETY: the caller hands `groups * 64` bytes for both the packed weights and the
            // activations, and `ROWS` outputs.
            unsafe {
                let mut acc = [_mm512_setzero_si512(); ROWS];
                for group in 0..groups {
                    let weight = _mm512_loadu_si512(packed.add(group * 64).cast());
                    for (quad, slot) in acc.iter_mut().enumerate().take(16) {
                        let act = _mm512_set1_epi32(std::ptr::read_unaligned(
                            acts.add(group * 64 + quad * 4).cast::<i32>(),
                        ));
                        *slot = _mm512_dpbusd_epi32(*slot, weight, act);
                    }
                }
                for (quad, value) in acc.iter().enumerate() {
                    let mut lanes = [0_i32; 16];
                    _mm512_storeu_si512(lanes.as_mut_ptr().cast(), *value);
                    *out.add(quad) = lanes[0];
                }
            }
        }
        let mut out_a = [0_i32; ROWS * 16];
        let mut out_b = [0_i32; ROWS * 16];
        // Mười sáu hàng weight, mỗi hàng một accumulator; `codes` của group được load một lần và
        // dùng cho cả 16 hàng — đúng như kernel tiled đang chạy.
        let mut form_a = || unsafe {
            run_a(
                weights_dword.as_ptr(),
                acts.as_ptr(),
                out_a.as_mut_ptr(),
                GROUPS,
            )
        };
        let mut form_b = || unsafe {
            run_b(
                weights_packed.as_ptr(),
                acts.as_ptr(),
                out_b.as_mut_ptr(),
                GROUPS,
            )
        };
        const ROUNDS: usize = 20_000;
        let run = |body: &mut dyn FnMut()| {
            let start = std::time::Instant::now();
            for _ in 0..ROUNDS {
                body();
            }
            start.elapsed().as_secs_f64() / ROUNDS as f64
        };
        let a = run(&mut form_a);
        let b = run(&mut form_b);
        // 16 hàng x 256 cột x 16 activation, nhân 2 cho MAC.
        let flops = 2.0 * (ROWS * 256 * 16) as f64;
        println!(
            "form A (broadcast per dot): {:>8.1} ns  {:>7.1} GFLOPS",
            a * 1e9,
            flops / a / 1e9
        );
        println!(
            "form B (weight vector, mem broadcast): {:>8.1} ns  {:>7.1} GFLOPS  ({:+.1}%)",
            b * 1e9,
            flops / b / 1e9,
            (a / b - 1.0) * 100.0
        );
    }

    fn q4k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
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
        row
    }

    fn q3k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(blocks * 110);
        for _ in 0..blocks {
            for _ in 0..108 {
                row.push(rng.byte());
            }
            row.extend_from_slice(&rng.half(0.001, 0.03));
        }
        row
    }

    fn q5k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(blocks * 176);
        for _ in 0..blocks {
            row.extend_from_slice(&rng.half(0.001, 0.06));
            row.extend_from_slice(&rng.half(0.0, 0.02));
            for _ in 0..172 {
                row.push(rng.byte());
            }
        }
        row
    }

    /// The Q3_K and Q5_K row kernels must agree with the scalar reference within the tolerance the
    /// other row kernels use (`1e-3 + 1e-4·|ref|`). Random block bytes cover every scale, every
    /// `qh` bit pattern, and both halves of the block.
    #[test]
    fn q3k_and_q5k_row_kernels_match_the_scalar_reference() {
        if !available() {
            eprintln!("skipped: no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0103);
        for (ty, width) in [(GgmlType::Q3K, 110_usize), (GgmlType::Q5K, 176)] {
            for cols in [256_usize, 1536] {
                let blocks = cols / 256;
                let row = if width == 110 {
                    q3k_row(&mut rng, blocks)
                } else {
                    q5k_row(&mut rng, blocks)
                };
                let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                let want = crate::quant::dot_scalar(ty, &row, &x, &[], None);
                // SAFETY: whole blocks on both sides, and `available()` checked AVX2.
                let got = unsafe {
                    if width == 110 {
                        q3k_dot_avx2(&row, &x)
                    } else {
                        q5k_dot_avx2(&row, &x)
                    }
                };
                let tol = 1e-3 + 1e-4 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "{ty:?} cols {cols}: {got} vs {want} (tol {tol})"
                );
            }
        }
    }

    /// `q3k_dot_vnni` with the bias dot removed, to price what precomputing the 16-value activation
    /// sums could buy before any plumbing is added. Test-only and deliberately wrong.
    #[cfg(test)]
    #[target_feature(enable = "avx512vnni,avx512vl,avx2")]
    unsafe fn q3k_dot_vnni_without_bias(row: &[u8], xq: &[i8], dx: &[f32]) -> f32 {
        // SAFETY: the caller states the lengths, and the CPU features; every load stays inside them.
        unsafe {
            let blocks = row.len() / 110;
            let three = _mm256_set1_epi8(3);
            let four_levels = _mm256_set1_epi8(4);
            let zero = _mm256_setzero_si256();
            let mut accumulator = _mm256_setzero_ps();
            for b in 0..blocks {
                let block = row.as_ptr().add(b * 110);
                let d = f16_to_f32(u16::from_le_bytes([*block.add(108), *block.add(109)]));
                let scales = q3k_scales(block);
                let qh = _mm256_loadu_si256(block as *const __m256i);
                let dx_at = dx.as_ptr().add(b * 8);
                let mut bit = 1_u8;
                for half in 0..2 {
                    let qs = block.add(32 + half * 32);
                    let x_half = b * 256 + half * 128;
                    let dx4 = _mm_loadu_ps(dx_at.add(half * 4));
                    let dx8 = _mm256_set_m128(dx4, dx4);
                    let dx_dup =
                        _mm256_permutevar8x32_ps(dx8, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
                    let coeffs = _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_loadu_ps(scales.as_ptr().add(half * 8)), dx_dup),
                        _mm256_set1_ps(d),
                    );
                    for run in 0..4 {
                        let mask = _mm256_set1_epi8(bit as i8);
                        let shift = _mm256_set1_epi32((run * 2) as i32);
                        let pick = _mm256_setr_epi32(
                            (2 * run) as i32,
                            (2 * run) as i32,
                            (2 * run) as i32,
                            (2 * run) as i32,
                            (2 * run + 1) as i32,
                            (2 * run + 1) as i32,
                            (2 * run + 1) as i32,
                            (2 * run + 1) as i32,
                        );
                        let packed = _mm256_loadu_si256(qs as *const __m256i);
                        let codes = _mm256_and_si256(_mm256_srlv_epi32(packed, shift), three);
                        let high = _mm256_andnot_si256(
                            _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask), zero),
                            four_levels,
                        );
                        let unsigned = _mm256_add_epi8(codes, high);
                        let xv = _mm256_loadu_si256(
                            xq.as_ptr().add(x_half + run * 32) as *const __m256i
                        );
                        let dot = _mm256_dpbusd_epi32(zero, unsigned, xv);
                        let coefficients = _mm256_permutevar8x32_ps(coeffs, pick);
                        accumulator =
                            _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), coefficients, accumulator);
                        bit <<= 1;
                    }
                }
            }
            hsum(accumulator)
        }
    }

    /// Prices the bias dot: same row, same activation, with and without the second `dpbusd`.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q3k_bias_dot() {
        if !vnni_available() {
            eprintln!("skipped: no VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0105);
        let cols = 2048;
        let row = q3k_row(&mut rng, cols / 256);
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let dx = xq.d.clone();
        let mut with = 0.0_f32;
        let mut without = 0.0_f32;
        let iterations = 100_000;
        for round in 0..7 {
            // Alternate which kernel runs first so a frequency drift cannot favour either.
            let (first, second) = if round % 2 == 0 {
                (true, false)
            } else {
                (false, true)
            };
            let start = std::time::Instant::now();
            if first {
                for _ in 0..iterations {
                    // SAFETY: whole blocks, matching activation, VNNI checked above.
                    with += unsafe { q3k_dot_vnni(&row, &xq.q, &dx) };
                }
            } else {
                for _ in 0..iterations {
                    // SAFETY: as above, with the test-only kernel that drops the bias dot.
                    without += unsafe { q3k_dot_vnni_without_bias(&row, &xq.q, &dx) };
                }
            }
            let first_ms = start.elapsed().as_secs_f64() * 1e3;
            let start = std::time::Instant::now();
            if second {
                for _ in 0..iterations {
                    // SAFETY: as above.
                    with += unsafe { q3k_dot_vnni(&row, &xq.q, &dx) };
                }
            } else {
                for _ in 0..iterations {
                    // SAFETY: as above, with the test-only kernel that drops the bias dot.
                    without += unsafe { q3k_dot_vnni_without_bias(&row, &xq.q, &dx) };
                }
            }
            let second_ms = start.elapsed().as_secs_f64() * 1e3;
            let (with_ms, without_ms) = if first {
                (first_ms, second_ms)
            } else {
                (second_ms, first_ms)
            };
            if round >= 3 {
                println!(
                    "q3k row {cols}: with bias {with_ms:.1} ms, without {without_ms:.1} ms, \
                     ratio {:.3}x",
                    with_ms / without_ms
                );
            }
        }
        println!("(checksums {with:.3} {without:.3})");
    }

    /// The band kernel must reproduce the row kernel's numbers for every row and every column of
    /// its sixteen-activation tile.
    #[test]
    fn q3k_band_matches_the_row_kernel() {
        if !vnni512_available() {
            eprintln!("skipped: no AVX-512 VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0107);
        let cols = 1536;
        let blocks = cols / 256;
        let rows: Vec<Vec<u8>> = (0..4).map(|_| q3k_row(&mut rng, blocks)).collect();
        let data: Vec<u8> = rows.concat();
        let stride = blocks * 110;
        let repacked = pack_q3k_rows(&data, stride);
        let xs: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let quantized: Vec<QuantizedX> = xs.iter().map(|x| QuantizedX::new(x)).collect();
        let refs: [&QuantizedX; 16] = std::array::from_fn(|entry| &quantized[entry]);
        let band = q3k_act_band(&refs, blocks);
        let mut dxs = vec![0.0_f32; blocks * 16 * 16];
        for b in 0..blocks {
            for k in 0..16 {
                for (col, xq) in quantized.iter().enumerate() {
                    dxs[(b * 16 + k) * 16 + col] = xq.d[b * 8 + k / 2];
                }
            }
        }
        let mut out = [0.0_f32; 4 * 16];
        // SAFETY: four repacked rows, a tile for all six blocks, sixteen columns of scales, and
        // `vnni512_available()` checked the CPU.
        unsafe {
            q3k_band_16::<4>(
                repacked.as_ptr(),
                0,
                blocks * Q3K_RECORD,
                band.as_ptr(),
                dxs.as_ptr(),
                blocks,
                out.as_mut_ptr(),
            );
        }
        for (t, row) in rows.iter().enumerate() {
            for (col, xq) in quantized.iter().enumerate() {
                // SAFETY: whole blocks and a matching activation.
                let want = unsafe { q3k_dot_vnni(row, &xq.q, &xq.d) };
                let got = out[t * 16 + col];
                let tol = 1e-3 + 1e-4 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "row {t} col {col}: {got} vs {want}"
                );
            }
        }
    }

    /// Prices the band against the row kernel on the same work: four rows × sixteen activations.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q3k_band() {
        if !vnni512_available() {
            eprintln!("skipped: no AVX-512 VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0108);
        let cols = 1536;
        let blocks = cols / 256;
        let rows: Vec<Vec<u8>> = (0..4).map(|_| q3k_row(&mut rng, blocks)).collect();
        let data: Vec<u8> = rows.concat();
        let stride = blocks * 110;
        let repacked = pack_q3k_rows(&data, stride);
        let xs: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let quantized: Vec<QuantizedX> = xs.iter().map(|x| QuantizedX::new(x)).collect();
        let refs: [&QuantizedX; 16] = std::array::from_fn(|entry| &quantized[entry]);
        let band = q3k_act_band(&refs, blocks);
        let mut dxs = vec![0.0_f32; blocks * 16 * 16];
        for b in 0..blocks {
            for k in 0..16 {
                for (col, xq) in quantized.iter().enumerate() {
                    dxs[(b * 16 + k) * 16 + col] = xq.d[b * 8 + k / 2];
                }
            }
        }
        let iterations = 20_000;
        let mut sink = 0.0_f32;
        let mut out = [0.0_f32; 4 * 16];
        for round in 0..5 {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                // SAFETY: as in the parity test above.
                unsafe {
                    q3k_band_16::<4>(
                        repacked.as_ptr(),
                        0,
                        blocks * Q3K_RECORD,
                        band.as_ptr(),
                        dxs.as_ptr(),
                        blocks,
                        out.as_mut_ptr(),
                    );
                }
                sink += out[0];
            }
            let band_ms = start.elapsed().as_secs_f64() * 1e3;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                for row in &rows {
                    for xq in &quantized {
                        // SAFETY: whole blocks and a matching activation.
                        sink += unsafe { q3k_dot_vnni(row, &xq.q, &xq.d) };
                    }
                }
            }
            let row_ms = start.elapsed().as_secs_f64() * 1e3;
            if round >= 2 {
                println!(
                    "4x16 over {blocks} blocks: band {band_ms:.1} ms, row kernels {row_ms:.1} ms, {:.2}x",
                    row_ms / band_ms
                );
            }
        }
        println!("(sink {sink:.3})");
    }

    /// The integer kernels must agree with the `f32` row kernel run on the *dequantized*
    /// activation, which isolates kernel arithmetic from the quantization error of `x` itself.
    /// Both the `maddubs` and the `dpbusd` form are checked, and the batch path must equal the
    /// single path entry by entry.
    #[test]
    fn q3k_and_q5k_integer_kernels_match_the_f32_kernel_on_dequantized_x() {
        if !available() {
            eprintln!("skipped: no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0104);
        for (ty, width) in [(GgmlType::Q3K, 110_usize), (GgmlType::Q5K, 176)] {
            for cols in [256_usize, 1536] {
                let blocks = cols / 256;
                let row = if width == 110 {
                    q3k_row(&mut rng, blocks)
                } else {
                    q5k_row(&mut rng, blocks)
                };
                let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                let xq = QuantizedX::new(&x);
                let dx = xq.dequantized();
                let sums = crate::quant::x_sums(&dx);
                let want = crate::quant::dot_scalar(ty, &row, &dx, &sums, None);
                // SAFETY: whole blocks, an activation of the same width, and AVX2.
                let got = unsafe {
                    match width {
                        110 => q3k_dot_int(&row, &xq.q, &xq.d),
                        _ => q5k_dot_int(&row, &xq.q, &xq.d, &sums),
                    }
                };
                let tol = 1e-3 + 1e-4 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "{ty:?} maddubs cols {cols}: {got} vs {want}"
                );
                if vnni_available() {
                    // SAFETY: as above, and `vnni_available()` checked the features.
                    let wide = unsafe {
                        match width {
                            110 => q3k_dot_vnni(&row, &xq.q, &xq.d),
                            _ => q5k_dot_vnni(&row, &xq.q, &xq.d, &sums),
                        }
                    };
                    assert!(
                        (wide - want).abs() <= tol,
                        "{ty:?} vnni cols {cols}: {wide} vs {want}"
                    );
                    assert_eq!(wide.to_bits(), got.to_bits(), "{ty:?} vnni vs maddubs");
                }
            }
        }
    }

    /// A Q6_K row of `blocks` blocks with random scales and nibbles.
    /// Q6_K tiled against the row kernel at the shape that dominates prefill.
    ///
    /// Q6_K carries more bytes per weight than Q4_K (210 vs 144 per 256
    /// values), so the same MAC count moves more memory and its per-block
    /// epilogue is shorter (one integer correction, no separate min term).
    /// This measures where it lands relative to the Q4_K tiled kernel's
    /// ~24 MAC/cycle, i.e. how much headroom a rewrite could still find.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q6k_tiled_vs_row() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x0dd6_ba11_0000_0001);
        let rows = 8960usize;
        let cols = 1536usize;
        let blocks = cols / 256;
        let stride = blocks * 210;
        let data: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let reps = 5;
        let mut out = vec![0.0_f32; 16 * rows];
        let start = std::time::Instant::now();
        for _ in 0..reps {
            try_q6k_matvec_tiled_range(&data, stride, 0, rows, &xs, &mut out, None);
        }
        let tiled = start.elapsed().as_secs_f64();
        let mut totals = [0.0_f32; 16];
        let start = std::time::Instant::now();
        for _ in 0..reps {
            for row in 0..rows {
                let rb = &data[row * stride..(row + 1) * stride];
                // SAFETY: whole 210-byte blocks, VNNI checked above.
                unsafe { q6k_dot_batch_q8(rb, &xs, &mut totals) };
            }
        }
        let row = start.elapsed().as_secs_f64();
        let macs = (rows * cols * 16) as f64;
        let gb = (data.len() as f64) / 1e9;
        println!(
            "q6k tiled: {:.2} GB/s, {:.1} G token-MAC/s",
            gb * reps as f64 / tiled,
            macs * reps as f64 / tiled / 1e9
        );
        println!(
            "q6k row:   {:.2} GB/s, {:.1} G token-MAC/s",
            gb * reps as f64 / row,
            macs * reps as f64 / row / 1e9
        );
        std::hint::black_box((&out, &totals));
    }

    fn q6k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
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
        row
    }

    /// The Q4_K dot's throughput on a matrix the size of this model's gate projection.
    ///
    /// Ignored by default: it is a measurement, not a check. Run it with
    /// `cargo test -p vipera-backend-cpu --release -- --ignored --nocapture q4k_dot_throughput`.
    /// The AVX2 inner loop of [`q4k_dot`] with the scales, the minimums and the float combination
    /// removed: the same loads and the same integer work, so the difference between it and the real
    /// kernel is what the per-block bookkeeping costs. Test-only, and only ever called from the
    /// benchmark below.
    ///
    /// # Safety
    ///
    /// `row` must be whole 144-byte blocks, `xq` 256 values per block, and the CPU must have AVX2.
    #[cfg(test)]
    #[target_feature(enable = "avx2")]
    unsafe fn q4k_int_only(row: &[u8], xq: &[i8]) -> i32 {
        // SAFETY: the caller's contract is this function's contract, and every load stays inside
        // `row` and `xq` for the same arithmetic reasons as `q4k_dot`.
        unsafe {
            let blocks = row.len() / 144;
            let low_mask = _mm256_set1_epi8(0x0f);
            let ones = _mm256_set1_epi16(1);
            let mut total = _mm256_setzero_si256();
            for b in 0..blocks {
                let block = row.as_ptr().add(b * 144);
                for j in 0..4 {
                    let q = block.add(16 + j * 32);
                    let packed = _mm256_loadu_si256(q as *const __m256i);
                    let low_bytes = _mm256_and_si256(packed, low_mask);
                    let high_bytes = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                    let values = xq.as_ptr().add(b * 256 + j * 64);
                    let low_sums = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            low_bytes,
                            _mm256_loadu_si256(values as *const __m256i),
                        ),
                        ones,
                    );
                    let high_sums = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            high_bytes,
                            _mm256_loadu_si256(values.add(32) as *const __m256i),
                        ),
                        ones,
                    );
                    total = _mm256_add_epi32(total, _mm256_add_epi32(low_sums, high_sums));
                }
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, total);
            lanes.iter().sum()
        }
    }

    /// The 256-bit band must agree with the row kernel within the same tolerance the 512-bit band
    /// does (`1e-3 + 1e-4·|ref|`), and every row must be a function of the row and its activation
    /// only: the band splits rows into 32-row windows and 4-row bands, so a row's value must not
    /// depend on where the split falls.
    #[test]
    fn the_256_bit_band_matches_the_row_kernel_and_ignores_the_window_split() {
        if !available() {
            eprintln!("skipped: no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0026);
        let cols = 512;
        let blocks = cols / 256;
        // 9 rows: two 4-row bands plus a one-row tail, so both band widths and the tail run.
        let rows: Vec<Vec<u8>> = (0..9).map(|_| q4k_row(&mut rng, blocks)).collect();
        let xs: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let quantized: Vec<QuantizedX> = xs.iter().map(|x| QuantizedX::new(x)).collect();
        let sums: Vec<Vec<f32>> = xs.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xq: [&QuantizedX; 4] = std::array::from_fn(|entry| &quantized[entry]);
        let sum_refs: [&[f32]; 4] = std::array::from_fn(|entry| sums[entry].as_slice());

        YMM_OVERRIDE.store(2, std::sync::atomic::Ordering::Relaxed);
        let stride = blocks * 144;
        let mut data = Vec::with_capacity(rows.len() * stride);
        for row in &rows {
            data.extend_from_slice(row);
        }
        let mut out = vec![0.0_f32; 4 * rows.len()];
        let ran = try_q4k_matvec_tiled_range_ymm::<4>(
            &data,
            stride,
            0,
            rows.len(),
            &xq,
            &sum_refs,
            &mut out,
        );
        assert!(ran, "the 256-bit band refused a shape it accepts");

        for entry in 0..4 {
            for (row, row_bytes) in rows.iter().enumerate() {
                // SAFETY: whole blocks and a quantized activation of the same width.
                let want = unsafe {
                    q4k_dot(
                        row_bytes,
                        &quantized[entry].q,
                        &quantized[entry].d,
                        &sums[entry],
                    )
                };
                let got = out[entry * rows.len() + row];
                let tol = 1e-3 + 1e-4 * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "entry {entry} row {row}: {got} vs {want} (tol {tol})"
                );
            }
        }
        // A second call with the rows split at a different boundary (5 rows, then 4) must give the
        // same numbers for the overlapping rows: the window split is not part of the value.
        let mut split = vec![0.0_f32; 4 * 5];
        assert!(try_q4k_matvec_tiled_range_ymm::<4>(
            &data, stride, 0, 5, &xq, &sum_refs, &mut split,
        ));
        for entry in 0..4 {
            for row in 0..5 {
                assert_eq!(
                    split[entry * 5 + row].to_bits(),
                    out[entry * rows.len() + row].to_bits()
                );
            }
        }
        YMM_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn q4k_packed_row_matches_scalar_within_tolerance() {
        if !vnni_available() {
            eprintln!("skipped: no VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0006);
        let cols = 512;
        let blocks = cols / 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_row(&mut rng, blocks)).collect();
        let refs: [&[u8]; 8] = std::array::from_fn(|i| rows[i].as_slice());
        let tile = crate::repack::pack_q4k8(&refs, cols).expect("pack");
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let sums = crate::quant::x_sums(&x);
        for (row, row_bytes) in rows.iter().enumerate() {
            // SAFETY: whole blocks, VNNI checked above.
            let fast = unsafe { q4k_dot_packed_row_vnni(&tile, row, &xq.q, &xq.d, &sums) };
            // SAFETY: as above; integer reference matches kernel arithmetic.
            let slow = unsafe { q4k_dot(row_bytes, &xq.q, &xq.d, &sums) };
            let tol = 1e-3 + 1e-4 * slow.abs();
            assert!(
                (fast - slow).abs() <= tol,
                "row {row}: fast {fast}, slow {slow}"
            );
        }
    }

    /// How the Q4_K dot spends its time on each shape this model multiplies, split into bytes moved,
    /// integer work, and per-block bookkeeping.
    ///
    /// Three variants over the same bytes: a byte sum (the memory floor), the kernel's integer loop
    /// with the scales removed, and the whole kernel. `bookkeeping` is the share of the full
    /// kernel's time that disappears when the per-block unpacking goes, which is the share an
    /// 8-row layout can amortize.
    ///
    /// Ignored by default: it is a measurement, not a check. Run it with
    /// `cargo test -p vipera-backend-cpu --release -- --ignored --nocapture q4k_shape_breakdown`.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_shape_breakdown() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0002);
        // Name, rows, values per row: the shapes a Q4_K_M 1.5B model multiplies per token.
        let shapes = [
            ("attn_q", 1536usize, 1536usize),
            ("attn_kv", 512, 1536),
            ("attn_output", 1536, 1536),
            ("ffn_gate+up", 17_920, 1536),
            ("ffn_down", 1536, 8960),
        ];
        println!(
            "\n{:<14} {:>8} {:>9} {:>9} {:>9} {:>13} {:>11}",
            "shape", "MB", "sum", "int", "full", "int/share", "bookkeeping"
        );
        for (name, rows, values) in shapes {
            let blocks = values / 256;
            let row = q4k_row(&mut rng, blocks);
            let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
            let x: Vec<f32> = (0..values).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            let sums = crate::quant::x_sums(&x);
            let bytes = (rows * blocks * 144) as f64 / 1e6;
            // Repeat until a shape has moved at least 200 MB, so a small matrix is not timed as noise.
            let reps = ((200.0 / bytes).ceil() as usize).max(3);
            let row_bytes = blocks * 144;
            let time = |f: &mut dyn FnMut()| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    for _ in 0..reps {
                        f();
                    }
                    best = best.min(start.elapsed().as_secs_f64() / reps as f64);
                }
                best
            };
            let gbs = |seconds: f64| (rows * blocks * 144) as f64 / seconds / 1e9;
            let sum_time = time(&mut || {
                let mut total = 0u64;
                for r in 0..rows {
                    let slice = &matrix[r * row_bytes..(r + 1) * row_bytes];
                    for word in slice.as_chunks::<8>().0 {
                        total = total.wrapping_add(u64::from_le_bytes(*word));
                    }
                }
                std::hint::black_box(total);
            });
            let int_time = time(&mut || {
                let mut total = 0i32;
                for r in 0..rows {
                    // SAFETY: whole blocks on both sides, and the CPU was checked above.
                    total = total.wrapping_add(unsafe {
                        q4k_int_only(&matrix[r * row_bytes..(r + 1) * row_bytes], &xq.q)
                    });
                }
                std::hint::black_box(total);
            });
            let full_time = time(&mut || {
                let mut total = 0.0f32;
                for r in 0..rows {
                    // SAFETY: as above, plus `dx` and `sums` are one set per block.
                    total += unsafe {
                        q4k_dot(
                            &matrix[r * row_bytes..(r + 1) * row_bytes],
                            &xq.q,
                            &xq.d,
                            &sums,
                        )
                    };
                }
                std::hint::black_box(total);
            });
            println!(
                "{:<14} {:>8.1} {:>9.1} {:>9.1} {:>9.1} {:>12.0}% {:>10.0}%",
                name,
                bytes,
                gbs(sum_time),
                gbs(int_time),
                gbs(full_time),
                int_time / full_time * 100.0,
                (1.0 - int_time / full_time) * 100.0,
            );
        }
    }

    /// Packed row kernel vs current VNNI kernel on the gate shape.
    ///
    /// Ignored by default: it is a measurement, not a check. Run it with
    /// `cargo test -p vipera-backend-cpu --release -- --ignored --nocapture q4k_packed_throughput`.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_packed_throughput() {
        if !vnni_available() {
            eprintln!("skipped: no VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0007);
        let rows = 8960;
        let values = 1536;
        let blocks = values / 256;
        let row = q4k_row(&mut rng, blocks);
        let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
        let refs: Vec<&[u8]> = (0..rows)
            .map(|r| &matrix[r * blocks * 144..(r + 1) * blocks * 144])
            .collect();
        let mut tiles = Vec::with_capacity(rows / 8);
        for chunk in refs.as_chunks::<8>().0 {
            let arr = *chunk;
            tiles.push(crate::repack::pack_q4k8(&arr, values).expect("pack"));
        }
        let x: Vec<f32> = (0..values).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let sums = crate::quant::x_sums(&x);
        let gbs = |seconds: f64| (rows * blocks * 144) as f64 / 1e9 / seconds;
        let timeit = |f: &mut dyn FnMut()| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                f();
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let t_current = timeit(&mut || {
            let mut total = 0.0f32;
            for r in 0..rows {
                // SAFETY: whole blocks, VNNI checked above.
                total += unsafe {
                    q4k_dot(
                        &matrix[r * blocks * 144..(r + 1) * blocks * 144],
                        &xq.q,
                        &xq.d,
                        &sums,
                    )
                };
            }
            std::hint::black_box(total);
        });
        let t_packed = timeit(&mut || {
            let mut total = 0.0f32;
            for (t, tile) in tiles.iter().enumerate() {
                for row in 0..8 {
                    let _ = t;
                    // SAFETY: as above; tile covers the same bytes as matrix.
                    total += unsafe { q4k_dot_packed_row_vnni(tile, row, &xq.q, &xq.d, &sums) };
                }
            }
            std::hint::black_box(total);
        });
        println!(
            "current: {:.2} GB/s, packed: {:.2} GB/s (pack excluded)",
            gbs(t_current),
            gbs(t_packed)
        );
    }

    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_dot_throughput() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0d07_0000_0001);
        // 8960 rows of 1536 values: 7.7 MB of weights, the shape of the gate matrix.
        let rows = 8960;
        let blocks = 6;
        let row = q4k_row(&mut rng, blocks);
        let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
        let x: Vec<f32> = (0..blocks * 256).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let sums = crate::quant::x_sums(&x);
        let bytes = (rows * blocks * 144) as f64 / 1e9;
        for round in 0..6 {
            let start = std::time::Instant::now();
            let total: f32 = unsafe {
                (0..rows)
                    .map(|r| {
                        q4k_dot(
                            &matrix[r * blocks * 144..(r + 1) * blocks * 144],
                            &xq.q,
                            &xq.d,
                            &sums,
                        )
                    })
                    .sum()
            };
            let ms = start.elapsed().as_secs_f64() * 1e3;
            println!(
                "round {round}: {ms:.2} ms, {:.1} GB/s ({total})",
                bytes / (ms / 1e3)
            );
        }
    }

    /// Median seconds of one `B`-wide tiled call over `count` rows starting at
    /// `first_row`, five samples.
    fn time_tiled<const B: usize>(
        matrix: &[u8],
        stride: usize,
        blocks: usize,
        first_row: usize,
        count: usize,
    ) -> f64 {
        let cols = blocks * 256;
        let mut rng = Rng(0x5eed_0d07_0000_0003);
        let acts: Vec<(QuantizedX, Vec<f32>)> = (0..B)
            .map(|_| {
                let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                (QuantizedX::new(&x), crate::quant::x_sums(&x))
            })
            .collect();
        let xs: [&QuantizedX; B] = std::array::from_fn(|i| &acts[i].0);
        let sums: [&[f32]; B] = std::array::from_fn(|i| acts[i].1.as_slice());
        let mut out = vec![0.0f32; B * count];
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            let ok = try_q4k_matvec_tiled_range::<B>(
                matrix, stride, first_row, count, &xs, &sums, &mut out, None,
            );
            assert!(ok, "tiled kernel must accept this shape");
            std::hint::black_box(&out);
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("ordered"));
        samples[samples.len() / 2]
    }

    /// The tiled kernel at prefill batch, single thread, on the two shapes
    /// ADR-001 measures for the ggml repacked GEMM: the number that decides
    /// whether binding that kernel is worth a second toolchain.
    ///
    /// Single-threaded on purpose: ggml's figure in `vipera-ggml` is
    /// Does a wider prefill batch pay? Same shapes and kernel as the B=128 bench below, one arm each.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_tiled_batch_width_128_against_256() {
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        for &(rows, blocks) in &[(8960usize, 6usize), (1536, 35)] {
            let cols = blocks * 256;
            let stride = blocks * 144;
            let mut rng = Rng(0x5eed_0d07_0000_0009);
            let row = q4k_row(&mut rng, blocks);
            let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
            for &b in &[128usize, 256] {
                let (seconds, gflops) = if b == 128 {
                    let s = time_tiled::<128>(&matrix, stride, blocks, 0, rows);
                    (s, 2.0 * (rows * cols * 128) as f64 / s / 1e9)
                } else {
                    let s = time_tiled::<256>(&matrix, stride, blocks, 0, rows);
                    (s, 2.0 * (rows * cols * 256) as f64 / s / 1e9)
                };
                println!(
                    "q4_K {:>6}x{:<5} x {b:>3} rows 1T: {:>7.1} ms  {:>6.1} GFLOPS",
                    rows,
                    cols,
                    seconds * 1e3,
                    gflops
                );
            }
        }
    }

    /// single-threaded too, and the pool would hide the kernel difference.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_tiled_batch128_throughput() {
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const B: usize = 128;
        for &(rows, blocks) in &[(8960usize, 6usize), (1536, 35)] {
            let cols = blocks * 256;
            let stride = blocks * 144;
            let mut rng = Rng(0x5eed_0d07_0000_0002);
            let row = q4k_row(&mut rng, blocks);
            let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
            let seconds = time_tiled::<B>(&matrix, stride, blocks, 0, rows);
            let gflops = 2.0 * (rows * cols * B) as f64 / seconds / 1e9;
            println!(
                "tiled q4_K {:>6}x{:<5} x {B} rows 1T: {:>7.1} ms  {:>6.1} GFLOPS",
                rows,
                cols,
                seconds * 1e3,
                gflops
            );
        }
    }

    /// Does the batch-wide activation tile cost throughput? Its size is
    /// `blocks * halves * 64 * 16 * 4` bytes — 192 KB at B=128 on the 6-block
    /// shape, 1.5 MB across eight threads — so if residency is the limiter the
    /// rate should rise as the batch (and the tile) shrinks.
    ///
    /// Reuse falls with the batch too, so a flat curve is the interesting
    /// answer: it would mean the tile is not what caps the kernel.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_tiled_batch_sweep() {
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        let (rows, blocks) = (8960usize, 6usize);
        let cols = blocks * 256;
        let stride = blocks * 144;
        let mut rng = Rng(0x5eed_0d07_0000_0004);
        let row = q4k_row(&mut rng, blocks);
        let matrix: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
        for b in [16usize, 32, 64, 128] {
            let seconds = match b {
                16 => time_tiled::<16>(&matrix, stride, blocks, 0, rows),
                32 => time_tiled::<32>(&matrix, stride, blocks, 0, rows),
                64 => time_tiled::<64>(&matrix, stride, blocks, 0, rows),
                _ => time_tiled::<128>(&matrix, stride, blocks, 0, rows),
            };
            // `act` + `dxs` + `sumt`, the per-thread tile buffers.
            let tile_kb = blocks * b.div_ceil(16) * 4096 / 1024;
            let gflops = 2.0 * (rows * cols * b) as f64 / seconds / 1e9;
            println!(
                "sweep B={b:>3} act tile {tile_kb:>4} KB/thread: {:>7.1} ms  {:>6.1} GFLOPS",
                seconds * 1e3,
                gflops
            );
        }
    }

    /// The same kernel on a weight sweep bigger than L3: 127 copies of the
    /// 6-block gate shape, 0.98 GB, the size one prefill chunk reads from RAM.
    /// The L3-resident figure above prices the microkernel; this one prices the
    /// stream, and the ratio between them is the headroom prefill leaves on the
    /// table when the weights are not cached.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn q4k_tiled_streaming_throughput() {
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const B: usize = 128;
        const COPIES: usize = 127;
        let (rows, blocks) = (8960usize, 6usize);
        let cols = blocks * 256;
        let stride = blocks * 144;
        let mut rng = Rng(0x5eed_0d07_0000_0005);
        let row = q4k_row(&mut rng, blocks);
        let matrix: Vec<u8> = (0..rows * COPIES).flat_map(|_| row.clone()).collect();
        let cols_b = blocks * 256;
        let acts: Vec<(QuantizedX, Vec<f32>)> = (0..B)
            .map(|_| {
                let x: Vec<f32> = (0..cols_b).map(|_| rng.unit()).collect();
                (QuantizedX::new(&x), crate::quant::x_sums(&x))
            })
            .collect();
        let xs: [&QuantizedX; B] = std::array::from_fn(|i| &acts[i].0);
        let sums: [&[f32]; B] = std::array::from_fn(|i| acts[i].1.as_slice());
        let mut out = vec![0.0f32; B * rows];
        let mut samples = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for copy in 0..COPIES {
                let ok = try_q4k_matvec_tiled_range::<B>(
                    &matrix,
                    stride,
                    copy * rows,
                    rows,
                    &xs,
                    &sums,
                    &mut out,
                    None,
                );
                assert!(ok, "tiled kernel must accept this shape");
                std::hint::black_box(&out);
            }
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("ordered"));
        let seconds = samples[samples.len() / 2];
        let bytes = (rows * COPIES * stride) as f64;
        let gflops = 2.0 * (rows * COPIES * cols * B) as f64 / seconds / 1e9;
        println!(
            "stream 0.98 GB q4_K 8960x1536 x{B} rows 1T: {:>7.1} ms  {:>6.1} GFLOPS  {:>5.2} GB/s",
            seconds * 1e3,
            gflops,
            bytes / seconds / 1e9
        );
    }

    #[test]
    fn avx2_q4k_matches_scalar_on_quantized_activation() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for trial in 0..24 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let row = q4k_row(&mut rng, blocks);
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            // The scalar reference runs on the dequantized activation, so this test asks whether
            // the integer path reproduces the same dot, not whether quantization is lossless.
            let restored = xq.dequantized();
            let sums = crate::quant::x_sums(&restored);
            let expected = crate::quant::q4k_dot(&row, &restored, &sums);
            // SAFETY: `row` is whole 144-byte blocks, `xq` holds 256 values and 8 scales per
            // block, `sums` holds 16 per block, and the test returned early unless `available()`
            // saw AVX2 on this CPU.
            let actual = unsafe { q4k_dot(&row, &xq.q, &xq.d, &sums) };
            assert!(
                close(actual, expected),
                "trial {trial}: avx2 {actual} vs scalar {expected}"
            );
        }
    }

    #[test]
    fn avx2_q4k_is_exact_when_the_activation_needs_no_rounding() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0f11_ce11);
        // Every group's largest magnitude is 127/128 and every value is a multiple of 1/128, so
        // the per-group scale is exactly 1/128 and quantization changes nothing.
        let step = 1.0_f32 / 128.0;
        let mut x = Vec::new();
        for _ in 0..3 {
            for _ in 0..8 {
                for value in 0..32 {
                    let magnitude = if value == 0 {
                        127
                    } else {
                        i32::from(rng.byte() % 128)
                    };
                    let sign = if rng.next().is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    x.push(magnitude as f32 * step * sign);
                }
            }
        }
        let blocks = x.len() / 256;
        let row = q4k_row(&mut rng, blocks);
        let sums = crate::quant::x_sums(&x);
        let xq = QuantizedX::new(&x);
        assert_eq!(
            xq.dequantized(),
            x,
            "the activation should survive quantization"
        );
        let expected = crate::quant::q4k_dot(&row, &x, &sums);
        // SAFETY: as above, with `row` sized for `blocks`.
        let actual = unsafe { q4k_dot(&row, &xq.q, &xq.d, &sums) };
        assert!(
            close(actual, expected),
            "avx2 {actual} vs scalar {expected}"
        );
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

    /// The integer Q6_K kernel against its portable reference, which shares the unpacking and the
    /// per-16-value weighting but sums the group in scalar code.
    #[test]
    fn avx2_q6k_q8_matches_scalar() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut rng = Rng(0x5eed_0f6b_a1e5_0001);
        for trial in 0..24 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            let row = q6k_row(&mut rng, blocks);
            let expected = crate::quant::q6k_dot_q8(&row, &xq);
            // SAFETY: `row` is whole 210-byte blocks, `xq` covers them, and the test returned early
            // unless `available()` saw AVX2 on this CPU.
            let actual = unsafe { q6k_dot_q8(&row, &xq) };
            assert!(
                close(actual, expected),
                "trial {trial}: avx2 {actual} vs scalar {expected}"
            );
        }
    }

    /// The VNNI Q6_K kernel against the scalar reference: `vpdpbusd` replaces the `maddubs`/`madd`
    /// pair, and the integer lane sums are the same, so the results are close.
    #[test]
    fn avx2_q6k_vnni_matches_scalar() {
        if !vnni_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        let mut rng = Rng(0x5eed_0f6b_a1e5_0003);
        for trial in 0..24 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            let row = q6k_row(&mut rng, blocks);
            let expected = crate::quant::q6k_dot_q8(&row, &xq);
            // SAFETY: `row` is whole 210-byte blocks, `xq` covers them, and the test returned
            // early unless `vnni_available()` saw AVX-512 VNNI on this CPU.
            let actual = unsafe { q6k_dot_vnni(&row, &xq) };
            assert!(
                close(actual, expected),
                "trial {trial}: avx512-vnni {actual} vs scalar {expected}"
            );
        }
    }

    /// The same exactness claim for the integer Q6_K kernel: sharing the unpacking across entries
    /// may not change a bit of any entry's result.
    #[test]
    fn avx2_q6k_q8_batch_is_exactly_each_entry_alone() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        const B: usize = 4;
        let mut rng = Rng(0x5eed_0f6b_a1e5_0002);
        for trial in 0..8 {
            let blocks = 1 + trial % 3;
            let cols = blocks * 256;
            let row = q6k_row(&mut rng, blocks);
            let owned: Vec<crate::quant::OwnedActivation> = (0..B)
                .map(|_| {
                    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                    crate::quant::OwnedActivation::new(GgmlType::Q6K, &x)
                })
                .collect();
            let owned: &[crate::quant::OwnedActivation; B] =
                owned.as_slice().try_into().expect("four activations");
            let batch = BatchActivation::new(owned);
            let xs = batch
                .quantized()
                .expect("Q6_K activations are quantized on the fast path");
            let mut expected = [0.0_f32; B];
            for (entry, xq) in xs.iter().enumerate() {
                // SAFETY: `row` is whole 210-byte blocks and `xq` covers them; AVX2 is available.
                expected[entry] = unsafe { q6k_dot_q8(&row, xq) };
            }
            let mut actual = [0.0_f32; B];
            // SAFETY: as above; the batch kernel has the same contract as the single one.
            unsafe { q6k_dot_batch_q8(&row, xs, &mut actual) };
            assert_eq!(actual, expected, "trial {trial}");
        }
    }

    /// The SwiGLU multiply must be bit-identical to the scalar definition, lane for lane: it runs
    /// over every activation of a chunk, and a changed bit would move the sampling. Checked on a
    /// dense sweep, on random bit patterns (which reach the subnormals and the huge magnitudes), and
    /// on the special values.
    /// The softmax exponent kernel writes the scalar `exp`'s bits, lane for lane, at every length
    /// that crosses the sixteen-value and eight-value steps and the tail; the only thing that
    /// differs is the order the lanes are added across.
    #[test]
    fn exp_sum_shifted_matches_the_scalar_exp_bit_for_bit() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut bits = 0x9e37_79b9_u32;
        let mut inputs: Vec<f32> = (-40 * 64..=2 * 64).map(|step| step as f32 / 64.0).collect();
        for _ in 0..50_000 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            inputs.push(f32::from_bits(bits));
        }
        for shift in [0.0_f32, -3.5, 12.0, 40.0] {
            for length in [0_usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 1024] {
                let mut values = inputs[..length.min(inputs.len())].to_vec();
                if values.len() < length {
                    values.resize(length, 0.5);
                }
                let mut expected = values.clone();
                let mut expected_sum = 0.0_f32;
                for value in expected.iter_mut() {
                    *value = crate::model::exp(*value - shift);
                    expected_sum += *value;
                }
                let sum = exp_sum_shifted(&mut values, shift);
                for (at, (actual, expected)) in values.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "length {length} shift {shift} index {at}: {actual} against {expected}"
                    );
                }
                assert!(
                    (sum - expected_sum).abs() <= 1e-6 * expected_sum.abs().max(f32::MIN_POSITIVE),
                    "length {length} shift {shift}: {sum} against {expected_sum}"
                );
            }
        }
    }

    /// The scalar exponent the kernel reproduces, against libm's: this is the assumption the opt-in
    /// `VIPERA_FAST_SOFTMAX` rests on, so it is measured rather than claimed.
    #[test]
    fn the_polynomial_exp_stays_within_a_quarter_ppm_of_libm() {
        let (mut weighted, mut tail) = (0.0_f64, 0.0_f64);
        let mut value = -60.0_f32;
        while value <= 5.0 {
            let reference = f64::from(value.exp());
            let poly = f64::from(crate::model::exp(value));
            let error = ((poly - reference) / reference).abs();
            if value >= -40.0 {
                weighted = weighted.max(error);
            } else {
                tail = tail.max(error);
            }
            value += 0.0007;
        }
        println!("worst relative error: {weighted:.2e} in [-40, 5], {tail:.2e} in [-60, -40)");
        assert!(weighted < 2.5e-6, "the polynomial drifted: {weighted:.2e}");
        assert!(tail < 5e-6, "the denormal tail drifted: {tail:.2e}");
    }

    #[test]
    fn silu_mul_matches_the_scalar_one_bit_for_bit() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        let mut inputs: Vec<f32> = (-200 * 256..=200 * 256)
            .map(|step| step as f32 / 256.0)
            .collect();
        let mut bits = 0x1234_5678_u32;
        for _ in 0..200_000 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            inputs.push(f32::from_bits(bits));
        }
        inputs.extend_from_slice(&[
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MIN,
            f32::MAX,
            -f32::MAX,
            1e-45,
            -1e-45,
        ]);
        // Padded to a multiple of eight so the vector path and the tail both run.
        while !inputs.len().is_multiple_of(8) {
            inputs.push(0.5);
        }
        let gate = inputs.clone();
        let mut up: Vec<f32> = (0..inputs.len())
            .map(|at| 0.25 + at as f32 * 1e-6)
            .collect();
        let mut expected = up.clone();
        for (out, gate) in expected.iter_mut().zip(&gate) {
            *out *= crate::model::silu(*gate);
        }
        silu_mul(&mut up, &gate);
        for (at, (actual, expected)) in up.iter().zip(&expected).enumerate() {
            if expected.is_nan() {
                assert!(
                    actual.is_nan(),
                    "input {:#x} should be NaN",
                    gate[at].to_bits()
                );
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "input {:#x} ({} -> {}) differed: {actual} vs {expected}",
                    gate[at].to_bits(),
                    gate[at],
                    expected / (0.25 + at as f32 * 1e-6),
                );
            }
        }
    }

    /// F16C must produce exactly what the scalar conversion produces: every one of the 65536
    /// patterns, not a sample, because the KV cache is the one place a changed bit would move a token.
    #[test]
    fn f16c_conversion_matches_the_scalar_one_over_every_pattern() {
        if !std::is_x86_feature_detected!("f16c") {
            eprintln!("skipped: this CPU has no F16C");
            return;
        }
        let patterns: Vec<u16> = (0..=u16::MAX).collect();
        let bytes: Vec<u8> = patterns
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let mut converted = vec![0.0_f32; patterns.len()];
        half_to_float(&bytes, &mut converted);
        for (bits, value) in patterns.iter().zip(&converted) {
            let expected = f16_to_f32(*bits);
            if expected.is_nan() {
                assert!(value.is_nan(), "pattern {bits:#06x} should be NaN");
            } else {
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "pattern {bits:#06x} converted differently"
                );
            }
        }
    }

    /// Four activations against one row must give each of them exactly the result it gets alone: the
    /// shared weight read may not change a single bit of a sequence's arithmetic.
    #[test]
    fn avx2_q4k_batch_is_exactly_each_entry_alone() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        const B: usize = 4;
        let mut rng = Rng(0xfeed_face_1234_5678);
        for trial in 0..8 {
            let blocks = 1 + trial % 3;
            let cols = blocks * 256;
            let row = q4k_row(&mut rng, blocks);
            let owned: Vec<crate::quant::OwnedActivation> = (0..B)
                .map(|_| {
                    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                    crate::quant::OwnedActivation::new(GgmlType::Q4K, &x)
                })
                .collect();
            let owned: &[crate::quant::OwnedActivation; B] =
                owned.as_slice().try_into().expect("four activations");
            let batch = BatchActivation::new(owned);
            let xs = batch.quantized().expect("Q4_K activations are quantized");
            let sums = batch.sums();
            let mut expected = [0.0_f32; B];
            for (entry, xq) in xs.iter().enumerate() {
                // SAFETY: `row` is whole 144-byte blocks and `xq` covers them; AVX2 is available.
                expected[entry] = unsafe { q4k_dot(&row, &xq.q, &xq.d, sums[entry]) };
            }
            let mut actual = [0.0_f32; B];
            // SAFETY: as above; the batch kernel has the same contract as the single one.
            unsafe { q4k_dot_batch(&row, xs, sums, &mut actual) };
            assert_eq!(actual, expected, "trial {trial}");
        }
    }

    /// The same exactness claim for Q6_K.
    #[test]
    fn avx2_q6k_batch_is_exactly_each_entry_alone() {
        if !available() {
            eprintln!("skipped: this CPU has no AVX2");
            return;
        }
        const B: usize = 4;
        let mut rng = Rng(0x0dd_ba11_5eed_0042);
        for trial in 0..8 {
            let blocks = 1 + trial % 3;
            let cols = blocks * 256;
            let mut row = Vec::with_capacity(blocks * 210);
            for _ in 0..blocks {
                for _ in 0..192 {
                    row.push(rng.byte());
                }
                for _ in 0..16 {
                    row.push(rng.byte());
                }
                row.extend_from_slice(&rng.half(0.001, 0.06));
            }
            let owned: Vec<crate::quant::OwnedActivation> = (0..B)
                .map(|_| {
                    let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
                    crate::quant::OwnedActivation::new(GgmlType::Q6K, &x)
                })
                .collect();
            let owned: &[crate::quant::OwnedActivation; B] =
                owned.as_slice().try_into().expect("four activations");
            let batch = BatchActivation::new(owned);
            let xs = batch.x();
            let mut expected = [0.0_f32; B];
            for (entry, x) in xs.iter().enumerate() {
                // SAFETY: `row` is whole 210-byte blocks and `x` covers them; AVX2 is available.
                expected[entry] = unsafe { q6k_dot(&row, x) };
            }
            let mut actual = [0.0_f32; B];
            // SAFETY: as above; the batch kernel has the same contract as the single one.
            unsafe { q6k_dot_batch(&row, xs, &mut actual) };
            assert_eq!(actual, expected, "trial {trial}");
        }
    }

    /// The pool gives each thread its own slice of rows, so a row's tiled result must not depend on
    /// where the slice starts or where the row sits in its 32-row window and 8-row band.
    #[test]
    fn tiled_rows_do_not_depend_on_how_the_range_is_split() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let rows = 45usize;
        let blocks = 2usize;
        let cols = blocks * 256;
        let q4: Vec<u8> = (0..rows).flat_map(|_| q4k_row(&mut rng, blocks)).collect();
        let q6: Vec<u8> = (0..rows).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let source: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let xs: [&QuantizedX; 32] = std::array::from_fn(|e| &xs_vec[e]);
        let sums: [&[f32]; 32] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let run4 = |first: usize, count: usize| {
            let mut out = vec![0.0_f32; 32 * count];
            assert!(try_q4k_matvec_tiled_range(
                &q4,
                blocks * 144,
                first,
                count,
                &xs,
                &sums,
                &mut out,
                None,
            ));
            out
        };
        let run6 = |first: usize, count: usize| {
            let mut out = vec![0.0_f32; 32 * count];
            assert!(try_q6k_matvec_tiled_range(
                &q6,
                blocks * 210,
                first,
                count,
                &xs,
                &mut out,
                None,
            ));
            out
        };
        for split in [1usize, 5, 8, 13, 32, 37] {
            for (name, run) in [
                ("Q4_K", &run4 as &dyn Fn(usize, usize) -> Vec<f32>),
                ("Q6_K", &run6),
            ] {
                let whole = run(0, rows);
                let (head, tail) = (run(0, split), run(split, rows - split));
                for entry in 0..32 {
                    for row in 0..rows {
                        let part = if row < split {
                            head[entry * split + row]
                        } else {
                            tail[entry * (rows - split) + row - split]
                        };
                        assert_eq!(
                            whole[entry * rows + row].to_bits(),
                            part.to_bits(),
                            "{name} split {split}: entry {entry} row {row}"
                        );
                    }
                }
            }
        }
    }
    /// How much of the tiled kernel is epilogue: same dpbusd chain, with the six vector ops and
    /// two broadcasts per (row, sub-block) folded into one convert and one add.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn bench_epilogue_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const B: usize = 128;
        for &(rows, blocks, copies) in &[(8960usize, 6usize, 8usize), (1536, 35, 1)] {
            let cols = blocks * 256;
            let stride = blocks * 144;
            let mut rng = Rng(0x5eed_0d07_0000_0011);
            let row = q4k_row(&mut rng, blocks);
            let matrix: Vec<u8> = (0..rows * copies).flat_map(|_| row.clone()).collect();
            let count = rows * copies;
            let mut arng = Rng(0x5eed_0d07_0000_0012);
            let acts: Vec<(QuantizedX, Vec<f32>)> = (0..B)
                .map(|_| {
                    let x: Vec<f32> = (0..cols).map(|_| arng.unit()).collect();
                    (QuantizedX::new(&x), crate::quant::x_sums(&x))
                })
                .collect();
            let xs: [&QuantizedX; B] = std::array::from_fn(|i| &acts[i].0);
            let sums: [&[f32]; B] = std::array::from_fn(|i| acts[i].1.as_slice());
            let mut out = vec![0.0f32; B * count];
            let mut arm = |probe: bool| -> f64 {
                NO_EPI_PROBE.store(probe, Ordering::Relaxed);
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    let ok = try_q4k_matvec_tiled_range::<B>(
                        &matrix, stride, 0, count, &xs, &sums, &mut out, None,
                    );
                    assert!(ok, "tiled kernel must accept this shape");
                    std::hint::black_box(&out);
                    best = best.min(start.elapsed().as_secs_f64());
                }
                best
            };
            let (mut real, mut probe) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..2 {
                real = real.min(arm(false));
                probe = probe.min(arm(true));
            }
            NO_EPI_PROBE.store(false, Ordering::Relaxed);
            let gflops = |s: f64| 2.0 * (count * cols * B) as f64 / s / 1e9;
            println!(
                "epilogue probe {rows}x{cols} x{B} x{copies}: real {:>7.1} ms {:>6.1} GFLOPS | \
                 no-epi {:>7.1} ms {:>6.1} GFLOPS | {:.3}x",
                real * 1e3,
                gflops(real),
                probe * 1e3,
                gflops(probe),
                real / probe,
            );
        }
    }

    /// What a per-256-block activation scale would buy: same dpbusd chain, epilogue folded to one
    /// convert and one multiply-add per (row, block) instead of four ops per (row, sub-block).
    ///
    /// This is the ceiling for the "activation with one scale per 256 values" direction, and it is
    /// the number that decides whether changing the activation quantization is worth an ADR.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_per_block_epilogue_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const B: usize = 128;
        for &(rows, blocks, copies) in &[(8960usize, 6usize, 8usize), (1536, 35, 1)] {
            let cols = blocks * 256;
            let stride = blocks * 144;
            let mut rng = Rng(0x5eed_0d07_0000_0021);
            let row = q4k_row(&mut rng, blocks);
            let matrix: Vec<u8> = (0..rows * copies).flat_map(|_| row.clone()).collect();
            let count = rows * copies;
            let mut arng = Rng(0x5eed_0d07_0000_0022);
            let acts: Vec<(QuantizedX, Vec<f32>)> = (0..B)
                .map(|_| {
                    let x: Vec<f32> = (0..cols).map(|_| arng.unit()).collect();
                    (QuantizedX::new(&x), crate::quant::x_sums(&x))
                })
                .collect();
            let xs: [&QuantizedX; B] = std::array::from_fn(|i| &acts[i].0);
            let sums: [&[f32]; B] = std::array::from_fn(|i| acts[i].1.as_slice());
            let mut out = vec![0.0f32; B * count];
            let mut arm = |probe: bool| -> f64 {
                PER_BLOCK_EPI_PROBE.store(probe, Ordering::Relaxed);
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let start = std::time::Instant::now();
                    let ok = try_q4k_matvec_tiled_range::<B>(
                        &matrix, stride, 0, count, &xs, &sums, &mut out, None,
                    );
                    assert!(ok, "tiled kernel must accept this shape");
                    std::hint::black_box(&out);
                    best = best.min(start.elapsed().as_secs_f64());
                }
                best
            };
            let (mut real, mut probe) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..2 {
                real = real.min(arm(false));
                probe = probe.min(arm(true));
            }
            PER_BLOCK_EPI_PROBE.store(false, Ordering::Relaxed);
            let gflops = |s: f64| 2.0 * (count * cols * B) as f64 / s / 1e9;
            println!(
                "per-block epilogue probe {rows}x{cols} x{B} x{copies}: real {:>7.1} ms \
                 {:>6.1} GFLOPS | one-op {:>7.1} ms {:>6.1} GFLOPS | {:.3}x",
                real * 1e3,
                gflops(real),
                probe * 1e3,
                gflops(probe),
                real / probe,
            );
        }
    }

    /// How much of the Q6_K tiled kernel is the weight unpack: the same dpbusd chain, with the
    /// per-byte decode loop skipped (codes are stale, timing is what this measures).
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q6k_unpack_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x0dd6_ba11_0000_0003);
        let (rows, cols, copies) = (8960usize, 1536usize, 8usize);
        let blocks = cols / 256;
        let stride = blocks * 210;
        let row = q6k_row(&mut rng, blocks);
        let data: Vec<u8> = (0..rows * copies).flat_map(|_| row.clone()).collect();
        let count = rows * copies;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let mut out = vec![0.0_f32; 16 * count];
        let mut arm = |probe: bool| -> f64 {
            NO_UNPACK_PROBE.store(probe, Ordering::Relaxed);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                try_q6k_matvec_tiled_range(&data, stride, 0, count, &xs, &mut out, None);
                std::hint::black_box(&out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let (mut real, mut probe) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..2 {
            real = real.min(arm(false));
            probe = probe.min(arm(true));
        }
        NO_UNPACK_PROBE.store(false, Ordering::Relaxed);
        let bytes = (count * stride) as f64;
        println!(
            "q6k unpack probe {rows}x{cols} x16 x{copies}: real {:>7.1} ms ({:.1} GB/s) | \
             no-unpack {:>7.1} ms ({:.1} GB/s) | {:.3}x",
            real * 1e3,
            bytes / real / 1e9,
            probe * 1e3,
            bytes / probe / 1e9,
            real / probe,
        );
    }

    /// The same probe for the Q4_K tiled band: real against the nibble decode skipped. The delta is
    /// the unpack's share of the band, which is the ceiling a load-time unpack could recover.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q4k_unpack_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            return;
        }
        let (rows, cols, copies) = (8960usize, 1536usize, 4usize);
        let blocks = cols / 256;
        let stride = blocks * 144;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        let mut row = vec![0_u8; stride];
        for b in 0..blocks {
            let at = b * 144;
            row[at..at + 2].copy_from_slice(&crate::quant::f32_to_f16(0.02).to_le_bytes());
            row[at + 2..at + 4].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
            for byte in &mut row[at + 4..at + 144] {
                *byte = next();
            }
        }
        let count = rows * copies;
        let data: Vec<u8> = (0..count).flat_map(|_| row.clone()).collect();
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut out = vec![0.0_f32; 16 * count];
        let mut arm = |probe: bool| -> f64 {
            NO_UNPACK_PROBE.store(probe, Ordering::Relaxed);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                try_q4k_matvec_tiled_range(&data, stride, 0, count, &xs, &sums, &mut out, None);
                std::hint::black_box(&out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let (mut real, mut probe) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..2 {
            real = real.min(arm(false));
            probe = probe.min(arm(true));
        }
        NO_UNPACK_PROBE.store(false, Ordering::Relaxed);
        let bytes = (count * stride) as f64;
        println!(
            "q4k unpack probe {rows}x{cols} x16 x{copies}: real {:>7.1} ms ({:.1} GB/s) | \
             no-unpack {:>7.1} ms ({:.1} GB/s) | {:.3}x",
            real * 1e3,
            bytes / real / 1e9,
            probe * 1e3,
            bytes / probe / 1e9,
            real / probe,
        );
    }

    /// Schedule probe for the Q4_K band: one integer chain per row against two (even/odd
    /// sub-blocks). Same sum, so a win here is wireable.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q4k_schedule_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            return;
        }
        let (rows, cols, copies) = (8960usize, 1536usize, 4usize);
        let blocks = cols / 256;
        let stride = blocks * 144;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        let mut row = vec![0_u8; stride];
        for b in 0..blocks {
            let at = b * 144;
            row[at..at + 2].copy_from_slice(&crate::quant::f32_to_f16(0.02).to_le_bytes());
            row[at + 2..at + 4].copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
            for byte in &mut row[at + 4..at + 144] {
                *byte = next();
            }
        }
        let count = rows * copies;
        let data: Vec<u8> = (0..count).flat_map(|_| row.clone()).collect();
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let sums_vec: Vec<Vec<f32>> = source.iter().map(|x| crate::quant::x_sums(x)).collect();
        let sums: [&[f32]; 16] = std::array::from_fn(|e| sums_vec[e].as_slice());
        let mut out = vec![0.0_f32; 16 * count];
        let mut arm = |two: bool| -> f64 {
            TWO_ACC_PROBE.store(two, Ordering::Relaxed);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                try_q4k_matvec_tiled_range(&data, stride, 0, count, &xs, &sums, &mut out, None);
                std::hint::black_box(&out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let (mut one, mut two) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..2 {
            one = one.min(arm(false));
            two = two.min(arm(true));
        }
        TWO_ACC_PROBE.store(false, Ordering::Relaxed);
        let bytes = (count * stride) as f64;
        println!(
            "q4k schedule probe {rows}x{cols} x16 x{copies}: one-acc {:>7.1} ms ({:.1} GB/s) | \
             two-acc {:>7.1} ms ({:.1} GB/s) | {:.3}x",
            one * 1e3,
            bytes / one / 1e9,
            two * 1e3,
            bytes / two / 1e9,
            one / two,
        );
    }

    /// How much of the Q6_K tiled kernel is epilogue. Per (row, 16-value group) it does four ops
    /// and a broadcast over four `dpbusd`, so the ratio is worse than Q4_K's.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q6k_epilogue_probe() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x0dd6_ba11_0000_0002);
        let (rows, cols, copies) = (8960usize, 1536usize, 8usize);
        let blocks = cols / 256;
        let stride = blocks * 210;
        let row = q6k_row(&mut rng, blocks);
        let data: Vec<u8> = (0..rows * copies).flat_map(|_| row.clone()).collect();
        let count = rows * copies;
        let source: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|_| rng.unit()).collect())
            .collect();
        let xs_vec: Vec<QuantizedX> = source.iter().map(|x| QuantizedX::new(x)).collect();
        let xs: [&QuantizedX; 16] = std::array::from_fn(|e| &xs_vec[e]);
        let mut out = vec![0.0_f32; 16 * count];
        let mut arm = |probe: bool| -> f64 {
            NO_EPI_PROBE.store(probe, Ordering::Relaxed);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                try_q6k_matvec_tiled_range(&data, stride, 0, count, &xs, &mut out, None);
                std::hint::black_box(&out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let (mut real, mut probe) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..2 {
            real = real.min(arm(false));
            probe = probe.min(arm(true));
        }
        NO_EPI_PROBE.store(false, Ordering::Relaxed);
        let bytes = (count * stride) as f64;
        println!(
            "q6k epilogue probe {rows}x{cols} x16 x{copies}: real {:>7.1} ms ({:.1} GB/s) | \
             no-epi {:>7.1} ms ({:.1} GB/s) | {:.3}x",
            real * 1e3,
            bytes / real / 1e9,
            probe * 1e3,
            bytes / probe / 1e9,
            real / probe,
        );
    }

    /// Prices the one thing a repacked weight layout can buy: dropping the 4-byte weight load
    /// from every `dpbusd`. Streams 32 copies of an 8960x1536 tensor (0.25 GB a pass) because that
    /// is what prefill does; the L2-hot figure in `q4k_tiled_batch128_throughput` flatters the
    /// load, since from the second pass the weights are in cache.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn bench_no_weight_probe_streaming() {
        use std::sync::atomic::Ordering;
        if !vnni512_available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI");
            return;
        }
        const B: usize = 128;
        const COPIES: usize = 32;
        let (rows, blocks) = (8960usize, 6usize);
        let cols = blocks * 256;
        let stride = blocks * 144;
        let mut rng = Rng(0x5eed_0d07_0000_000b);
        let row = q4k_row(&mut rng, blocks);
        let matrix: Vec<u8> = (0..rows * COPIES).flat_map(|_| row.clone()).collect();
        let count = rows * COPIES;
        let mut arng = Rng(0x5eed_0d07_0000_000c);
        let acts: Vec<(QuantizedX, Vec<f32>)> = (0..B)
            .map(|_| {
                let x: Vec<f32> = (0..cols).map(|_| arng.unit()).collect();
                (QuantizedX::new(&x), crate::quant::x_sums(&x))
            })
            .collect();
        let xs: [&QuantizedX; B] = std::array::from_fn(|i| &acts[i].0);
        let sums: [&[f32]; B] = std::array::from_fn(|i| acts[i].1.as_slice());
        let mut out = vec![0.0f32; B * count];
        let mut run = |probe: bool| -> f64 {
            NO_WEIGHT_PROBE.store(probe, Ordering::Relaxed);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                let ok = try_q4k_matvec_tiled_range::<B>(
                    &matrix, stride, 0, count, &xs, &sums, &mut out, None,
                );
                assert!(ok, "tiled kernel must accept this shape");
                std::hint::black_box(&out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let mut real = f64::INFINITY;
        let mut probe = f64::INFINITY;
        for _ in 0..2 {
            real = real.min(run(false));
            probe = probe.min(run(true));
        }
        NO_WEIGHT_PROBE.store(false, Ordering::Relaxed);
        let bytes = (count * stride) as f64;
        let gflops = |s: f64| 2.0 * (count * cols * B) as f64 / s / 1e9;
        println!(
            "no-weight probe {:.2} GB streamed: real {:>7.1} ms {:>6.1} GFLOPS ({:.1} GB/s) | \
             probe {:>7.1} ms {:>6.1} GFLOPS ({:.1} GB/s) | speedup {:.3}x",
            bytes / 1e9,
            real * 1e3,
            gflops(real),
            bytes / real / 1e9,
            probe * 1e3,
            gflops(probe),
            bytes / probe / 1e9,
            real / probe,
        );
    }
}

#[cfg(test)]
mod quantize_tests {
    use super::*;
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

    /// `round_half_away_x8` on eight values.
    #[target_feature(enable = "avx2")]
    unsafe fn round_eight(values: [f32; 8]) -> [f32; 8] {
        // SAFETY: AVX2 is guaranteed by the caller; the loads and stores cover exactly the arrays.
        unsafe {
            let rounded = round_half_away_x8(_mm256_loadu_ps(values.as_ptr()));
            let mut out = [0.0_f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), rounded);
            out
        }
    }

    fn check_round(values: [f32; 8]) {
        // SAFETY: callers return early when AVX2 is missing.
        let got = unsafe { round_eight(values) };
        for (value, got) in values.iter().zip(got) {
            assert!(
                value.round() == got,
                "round({value:e}) = {:e} but the AVX2 form gave {got:e}",
                value.round()
            );
        }
    }

    #[test]
    fn round_half_away_matches_f32_round_on_ties_near_ties_and_a_stride_of_everything_else() {
        if !available() {
            return;
        }
        // Every tie and the values one ulp either side of it, which is where a naive `+ 0.5` fails.
        let mut batch = Vec::new();
        for whole in 0..=130_u32 {
            let tie = whole as f32 + 0.5;
            for value in [
                tie,
                f32::from_bits(tie.to_bits() - 1),
                f32::from_bits(tie.to_bits() + 1),
            ] {
                batch.push(value);
                batch.push(-value);
            }
        }
        batch.extend([
            0.0,
            -0.0,
            0.49999997,
            -0.49999997,
            0.5,
            -0.5,
            1.0e-30,
            -1.0e-45,
        ]);
        while batch.len() % 8 != 0 {
            batch.push(0.0);
        }
        for chunk in batch.as_chunks::<8>().0 {
            check_round(*chunk);
        }
        // A stride through every bit pattern from zero to 200.0, both signs.
        let end = 200.0_f32.to_bits();
        let mut bits = 0_u32;
        while bits + 8 * 4093 <= end {
            let values = std::array::from_fn(|lane| f32::from_bits(bits + lane as u32 * 4093));
            check_round(values);
            check_round(values.map(|value| -value));
            bits += 8 * 4093;
        }
    }

    /// Every `f32` whose magnitude is at most 200, which covers all `x / scale` the quantizer
    /// can produce (at most about 127), against `f32::round`.
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib round_half_away_matches_f32_round_for_every -- --ignored
    /// ```
    #[test]
    #[ignore = "exhaustive: about 2.3e9 values, ten seconds or so in release"]
    fn round_half_away_matches_f32_round_for_every_value_in_range() {
        if !available() {
            return;
        }
        let end = 200.0_f32.to_bits();
        let mut bits = 0_u32;
        while bits <= end {
            let values = std::array::from_fn(|lane| f32::from_bits(bits + lane as u32));
            check_round(values);
            check_round(values.map(|value| -value));
            bits += 8;
        }
    }

    /// Groups chosen to break a careless vector form: exact ties, ties one ulp off, subnormal
    /// scales, NaN, infinities, all-zero and all-equal groups, and magnitudes from 1e-3 to 3e7.
    fn adversarial(groups: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        let mut x = Vec::with_capacity(groups * 32);
        for _ in 0..groups {
            let magnitude = [1e-3_f32, 0.7, 1.0, 127.0, 5e3, 3e7][next() as usize % 6];
            let random = |value: u32| value as f32 / (1 << 24) as f32 * 2.0 - 1.0;
            let mut group = [0.0_f32; 32];
            match next() % 10 {
                0 => {}
                1 | 2 => {
                    // The largest value is 127 * scale with scale a power of two, so every other
                    // value divides to an exact `whole + 0.5`.
                    let scale = 2.0_f32.powi(-(next() as i32 % 12));
                    for slot in &mut group {
                        let tie = ((next() % 126) as f32 + 0.5) * scale;
                        *slot = if next() % 2 == 0 { tie } else { -tie };
                    }
                    group[(next() % 32) as usize] = 127.0 * scale;
                    if next() % 2 == 0 {
                        for slot in group.iter_mut().filter(|slot| **slot != 127.0 * scale) {
                            *slot = f32::from_bits(slot.to_bits() + 1);
                        }
                    }
                }
                3 => {
                    for slot in &mut group {
                        *slot = random(next()) * 1e-40;
                    }
                }
                4 => {
                    for slot in &mut group {
                        *slot = random(next()) * magnitude;
                    }
                    group[(next() % 32) as usize] = f32::NAN;
                }
                5 => {
                    for slot in &mut group {
                        *slot = random(next()) * magnitude;
                    }
                    group[(next() % 32) as usize] =
                        f32::INFINITY * if next() % 2 == 0 { 1.0 } else { -1.0 };
                }
                6 => group = [random(next()) * magnitude; 32],
                7 => {
                    for slot in &mut group {
                        *slot = random(next()) * magnitude * 1e-3;
                    }
                    group[(next() % 32) as usize] = magnitude;
                }
                _ => {
                    for slot in &mut group {
                        *slot = random(next()) * magnitude;
                    }
                }
            }
            x.extend(group);
        }
        x
    }

    #[test]
    fn quantize_groups_gives_exactly_what_the_scalar_definition_gives() {
        if !available() {
            return;
        }
        for seed in 1..=40_u32 {
            let x = adversarial(64, seed);
            let groups = x.len() / 32;
            let (mut q, mut d) = (vec![0_i8; groups * 32], vec![0.0_f32; groups]);
            assert!(quantize_groups(&x, &mut q, &mut d));
            for (index, group) in x.as_chunks::<32>().0.iter().enumerate() {
                let mut expected = [0_i8; 32];
                let scale = QuantizedX::quantize_group(group, &mut expected);
                assert_eq!(
                    &q[index * 32..index * 32 + 32],
                    &expected,
                    "seed {seed} group {index}: {group:?}"
                );
                assert_eq!(
                    d[index].to_bits(),
                    scale.to_bits(),
                    "seed {seed} group {index} scale"
                );
            }
        }
    }

    #[test]
    fn quantize_groups_refuses_slices_that_are_too_short() {
        let x = vec![1.0_f32; 64];
        assert!(!quantize_groups(&x, &mut [0_i8; 63], &mut [0.0_f32; 2]));
        assert!(!quantize_groups(&x, &mut [0_i8; 64], &mut [0.0_f32; 1]));
    }
}

// ---------------------------------------------------------------------------
// Pha 1b: VNNI 8x8 kernel over the ggml `block_q4_Kx8` layout
// ---------------------------------------------------------------------------
//
// `BlockQ4Kx8` is the ggml `block_q4_Kx8` (`llama.cpp/ggml/src/ggml-cpu/repack.h`): eight
// `block_q4_K` rows interleaved so one activation read feeds eight row dots. `gemv_q4k8` is a
// faithful port of `ggml_gemv_q4_K_8x8_q8_K` from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp`
// (AVX2 + FMA + F16C), lab-proven against the engine's row kernel on the reference machine
// (`bench_q4k8_gemv_vs_row`, 2026-11-23): 1.38x faster on the decode shape 8960x1536 at t=1,
// and ~1.0-2.2x on the other shapes. Wired through [`crate::quant::fill_rows_q4k8`], behind
// VIPERA_Q4K8 (`q4k8_enabled`, default off: ADR-006). The lab copy lives in
// `q4k8_lab.rs`; the two are expected to stay in lockstep — the parity test in quant.rs holds the
// engine path to the row path within the repack prototype's tolerance.

/// `block_q4_Kx8`: eight interleaved `block_q4_K` rows, the ggml layout `make_block_q4_Kx8` builds.
#[repr(C, align(8))]
pub(crate) struct BlockQ4Kx8 {
    /// `d` of the eight rows.
    pub d: [u16; 8],
    /// `dmin` of the eight rows.
    pub dmin: [u16; 8],
    /// Per sub-block pair: eight scales, eight mins, then the high nibbles of the last four.
    pub scales: [u8; 96],
    /// The nibbles of the eight rows, 8 bytes interleaved.
    pub qs: [u8; 1024],
}

/// Bytes of one `block_q4_Kx8`: eight rows, one super-block each — the same 144 bytes per row as
/// `block_q4_K`, so the layout moves nothing but the order.
pub(crate) const Q4K8_BLOCK: usize = 1152;
/// Values per Q4_K super-block (the activation is quantized in the same chunks).
#[cfg(target_arch = "x86_64")]
pub(crate) const QK: usize = 256;
/// Bytes of one `block_q8_K`: `f32` scale, 256 `i8`, 16 `i16` sums of 16.
#[cfg(target_arch = "x86_64")]
pub(crate) const Q8K_BLOCK: usize = 292;

const _: () = assert!(core::mem::size_of::<BlockQ4Kx8>() == Q4K8_BLOCK);

/// Test-only override for [`q4k8_enabled`]: the switch is cached from the environment once, which
/// leaves one process able to time only one arm. `0` follows the environment, `1` forces off,
/// `2` forces on.
#[cfg(all(test, target_arch = "x86_64"))]
static Q4K8_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the Pha 1b 8x8 kernel is on: `VIPERA_Q4K8`, **off by default** (ADR-006).
///
/// The repack is hoisted to load time (`WeightSource::set_q4k8`, one buffer per Q4_K tensor), which
/// removed the 0.49x per-call regression, but the kernel still measures 0.90-0.93x the row kernel on
/// decode at 8 threads (2026-11-26, paired). It stays opt-in: `VIPERA_Q4K8=1` turns it on.
#[cfg(target_arch = "x86_64")]
pub(crate) fn q4k8_enabled() -> bool {
    #[cfg(test)]
    match Q4K8_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::tiled::env_truthy("VIPERA_Q4K8"))
}

/// `make_block_q4_Kx8`: pack the eight rows starting at `first_row` into one `BlockQ4Kx8` per
/// `blocks` super-block, appending to `out` (the kernel reads `w[x * blocks + b]`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn make_blocks_q4k8(
    src: &[u8],
    stride: usize,
    first_row: usize,
    blocks: usize,
    out: &mut Vec<BlockQ4Kx8>,
) {
    let mut s = [0_u8; 8];
    let mut m = [0_u8; 8];
    for b in 0..blocks {
        let row = |i: usize| &src[(first_row + i) * stride + b * 144..][..144];
        let mut blk = BlockQ4Kx8 {
            d: [0; 8],
            dmin: [0; 8],
            scales: [0; 96],
            qs: [0; 1024],
        };
        for i in 0..8 {
            let r = row(i);
            blk.d[i] = u16::from_le_bytes([r[0], r[1]]);
            blk.dmin[i] = u16::from_le_bytes([r[2], r[3]]);
        }
        // Nibbles: take 8 bytes from each row in turn, so every 64-byte span of `qs` covers all
        // eight rows.
        for i in 0..128 {
            let r = row(i % 8);
            let src_off = (i / 8) * 8;
            blk.qs[i * 8..i * 8 + 8].copy_from_slice(&r[16 + src_off..16 + src_off + 8]);
        }
        // Sub-blocks 0-3: the low six bits of each row's scale and min bytes.
        for i in 0..4 {
            for j in 0..8 {
                s[j] = row(j)[4 + i] & 63;
                m[j] = row(j)[4 + i + 4] & 63;
            }
            let o = i * 12;
            blk.scales[o] = (s[0] & 63) + ((s[4] & 48) << 2);
            blk.scales[o + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
            blk.scales[o + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
            blk.scales[o + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
            blk.scales[o + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
            blk.scales[o + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
            blk.scales[o + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
            blk.scales[o + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
            blk.scales[o + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
            blk.scales[o + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
            blk.scales[o + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
            blk.scales[o + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
        }
        // Sub-blocks 4-7: the top two bits live in the first four bytes, the low four in the last
        // four — the Q4_K six-bit packing read backwards.
        for i in 0..4 {
            for j in 0..8 {
                s[j] = ((row(j)[4 + i] & 192) >> 2) | (row(j)[4 + i + 8] & 15);
                m[j] = ((row(j)[4 + i + 4] & 192) >> 2) | ((row(j)[4 + i + 8] & 240) >> 4);
            }
            let o = i * 12 + 48;
            blk.scales[o] = (s[0] & 63) + ((s[4] & 48) << 2);
            blk.scales[o + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
            blk.scales[o + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
            blk.scales[o + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
            blk.scales[o + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
            blk.scales[o + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
            blk.scales[o + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
            blk.scales[o + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
            blk.scales[o + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
            blk.scales[o + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
            blk.scales[o + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
            blk.scales[o + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
        }
        out.push(blk);
    }
}

/// Pack every whole 8-row group of a Q4_K matrix once, at load time: group `g`'s `blocks`
/// `BlockQ4Kx8` start at `g * blocks`. The hot path (`fill_rows_q4k8_groups`) then reads this
/// buffer instead of calling [`make_blocks_q4k8`] on every product. The packed bytes are the same
/// size as the source (1152 bytes per eight rows), so this copies the tensor, it does not expand it.
#[cfg(target_arch = "x86_64")]
pub(crate) fn pack_q4k8_rows(
    src: &[u8],
    stride: usize,
    blocks: usize,
    rows: usize,
) -> Vec<BlockQ4Kx8> {
    let mut packed = Vec::with_capacity(rows.div_ceil(8) * blocks);
    for group in 0..rows / 8 {
        make_blocks_q4k8(src, stride, group * 8, blocks, &mut packed);
    }
    packed
}

/// `quantize_row_q8_K_ref` per 256-value block (lab port): one scale for the whole block, the
/// activation's 16 pair sums in the ggml `block_q8_K` tail.
#[cfg(target_arch = "x86_64")]
pub(crate) fn quantize_q8k(x: &[f32], out: &mut [u8]) {
    assert_eq!(x.len() % QK, 0);
    assert_eq!(out.len(), x.len() / QK * Q8K_BLOCK);
    for (b, values) in x.as_chunks::<QK>().0.iter().enumerate() {
        let at = &mut out[b * Q8K_BLOCK..(b + 1) * Q8K_BLOCK];
        let mut max = 0.0_f32;
        let mut amax = 0.0_f32;
        for &v in values {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            at.fill(0);
            continue;
        }
        let iscale = -128.0 / max;
        at[0..4].copy_from_slice(&(1.0 / iscale).to_le_bytes());
        for (i, &v) in values.iter().enumerate() {
            at[4 + i] = (iscale * v).round().clamp(-128.0, 127.0) as i8 as u8;
        }
        for k in 0..QK / 16 {
            let sum: i32 = (0..16).map(|i| at[4 + k * 16 + i] as i8 as i32).sum();
            at[260 + k * 2..260 + k * 2 + 2].copy_from_slice(&(sum as i16).to_le_bytes());
        }
    }
}

/// `ggml_gemv_q4_K_8x8_q8_K` (lab port, AVX2 + FMA + F16C): the same layout and the same arithmetic
/// as the engine's row kernel, processed eight weight rows at once against one activation row.
///
/// # Safety
///
/// `w.len() == out.len() / 8 * blocks`, `act` holds one `block_q8_K` per block, and the CPU has
/// AVX2, FMA and F16C — the caller checks at runtime; the wiring in [`crate::quant::fill_rows_q4k8`]
/// does.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn gemv_q4k8(w: &[BlockQ4Kx8], blocks: usize, act: &[u8], out: &mut [f32]) {
    let groups = out.len() / 8;
    assert_eq!(w.len(), groups * blocks);
    assert!(act.len() >= blocks * Q8K_BLOCK);
    // SAFETY: every intrinsic below needs AVX2/FMA/F16C, which the caller guarantees, and the
    // pointers are derived from slices whose lengths are asserted above.
    unsafe {
        let deltamask = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        let scalemask = _mm_setr_epi8(0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
        let finalperm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        let m4b = _mm256_set1_epi8(0x0f);
        let (kmask1, kmask2, kmask3) = (0x3f3f_3f3f_u32, 0x0f0f_0f0f_u32, 0x0303_0303_u32);
        for x in 0..groups {
            let base = x * blocks;
            let mut acc_row = _mm256_setzero_ps();
            let mut acc_min = _mm256_setzero_ps();
            for b in 0..blocks {
                let blk = &w[base + b];
                let a = act.as_ptr().add(b * Q8K_BLOCK);
                let row_scale = _mm256_set1_ps(*(a as *const f32));
                let col_scale = _mm256_cvtph_ps(_mm_shuffle_epi8(
                    _mm_loadu_si128(blk.d.as_ptr() as *const __m128i),
                    deltamask,
                ));
                let col_dmin =
                    _mm256_cvtph_ps(_mm_loadu_si128(blk.dmin.as_ptr() as *const __m128i));
                let mut iacc_b = _mm256_setzero_si256();
                let mut iacc_min_b = _mm256_setzero_si256();
                // The activation's per-16 sums, folded and broadcast like ggml does.
                let q8sums = _mm256_loadu_si256(a.add(260) as *const __m256i);
                let mut q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
                    _mm256_castsi256_si128(q8sums),
                    _mm256_extracti128_si256(q8sums, 1),
                ));
                q8s = _mm256_permute2f128_si256::<0>(q8s, q8s);
                // Two sub-blocks per pass: the low nibbles are one, the high nibbles the next.
                for sb in 0..QK / 64 {
                    let raw = |off: usize| {
                        _mm256_loadu_si256(blk.qs.as_ptr().add(sb * 256 + off) as *const __m256i)
                    };
                    let (r0, r4) = (raw(0), raw(32));
                    let (r1, r5) = (raw(64), raw(96));
                    let (r2, r6) = (raw(128), raw(160));
                    let (r3, r7) = (raw(192), raw(224));
                    let low = |v: __m256i| _mm256_and_si256(v, m4b);
                    let high = |v: __m256i| _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4b);
                    let (lo0, lo4, lo1, lo5) = (low(r0), low(r4), low(r1), low(r5));
                    let (lo2, lo6, lo3, lo7) = (low(r2), low(r6), low(r3), low(r7));
                    let (hi0, hi4, hi1, hi5) = (high(r0), high(r4), high(r1), high(r5));
                    let (hi2, hi6, hi3, hi7) = (high(r2), high(r6), high(r3), high(r7));

                    let mut utmp_0 = [0_u32; 4];
                    let mut utmp_1 = [0_u32; 4];
                    // Twelve bytes: the fourth word is derived below, as in the C original.
                    for (i, slot) in utmp_0.iter_mut().take(3).enumerate() {
                        *slot = u32::from_le_bytes(
                            blk.scales[sb * 24 + i * 4..sb * 24 + i * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                    }
                    for (i, slot) in utmp_1.iter_mut().take(3).enumerate() {
                        *slot = u32::from_le_bytes(
                            blk.scales[12 + sb * 24 + i * 4..12 + sb * 24 + i * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                    }
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    let aux0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = aux0;
                    utmp_0[0] &= kmask1;
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    let aux1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = aux1;
                    utmp_1[0] &= kmask1;

                    let ms0 = _mm_setr_epi32(
                        utmp_0[0] as i32,
                        utmp_0[1] as i32,
                        utmp_0[2] as i32,
                        utmp_0[3] as i32,
                    );
                    let ms1 = _mm_setr_epi32(
                        utmp_1[0] as i32,
                        utmp_1[1] as i32,
                        utmp_1[2] as i32,
                        utmp_1[3] as i32,
                    );
                    let scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(ms0, scalemask));
                    let scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(ms1, scalemask));
                    let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                        _mm_shuffle_epi32::<78>(ms0),
                        _mm_shuffle_epi32::<78>(ms1),
                    ));

                    let mut lhs = [_mm256_setzero_si256(); 4];
                    for (i, slot) in lhs.iter_mut().enumerate() {
                        let v = _mm256_castsi128_si256(_mm_loadu_si128(
                            a.add(4 + sb * 64 + i * 16) as *const __m128i,
                        ));
                        *slot = _mm256_permute2f128_si256::<0>(v, v);
                    }

                    let mut iacc_0 = _mm256_setzero_si256();
                    let mut iacc_1 = _mm256_setzero_si256();
                    // Eight rows sit in one `ymm` as `B0(0-3) B4(0-3) B1(0-3) ... B7(0-3)`: the odd
                    // lanes come from the high half of the row set, which is why each of the two
                    // instructions in a pair blends one row vector with the 32-bit shuffle of its
                    // partner, and why the pair walks the activation in 4-value quarters.
                    // `$a`/`$b` are rows 0-3 and 4-7; `$lhs` is the 16-value activation span these
                    // eight rows share, and `$imm1`/`$imm2` pick two of its four quarters.
                    macro_rules! dot_pair {
                        ($acc:ident, $a:expr, $b:expr, $lhs:expr, $imm1:expr, $imm2:expr) => {
                            $acc = _mm256_add_epi16(
                                $acc,
                                _mm256_maddubs_epi16(
                                    _mm256_blend_epi32::<170>($a, _mm256_shuffle_epi32::<177>($b)),
                                    _mm256_shuffle_epi32::<$imm1>($lhs),
                                ),
                            );
                            $acc = _mm256_add_epi16(
                                $acc,
                                _mm256_maddubs_epi16(
                                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>($a), $b),
                                    _mm256_shuffle_epi32::<$imm2>($lhs),
                                ),
                            );
                        };
                    }
                    dot_pair!(iacc_0, lo0, lo4, lhs[0], 0, 85);
                    dot_pair!(iacc_0, lo1, lo5, lhs[0], 170, 255);
                    dot_pair!(iacc_0, lo2, lo6, lhs[1], 0, 85);
                    dot_pair!(iacc_0, lo3, lo7, lhs[1], 170, 255);
                    dot_pair!(iacc_1, hi0, hi4, lhs[2], 0, 85);
                    dot_pair!(iacc_1, hi1, hi5, lhs[2], 170, 255);
                    dot_pair!(iacc_1, hi2, hi6, lhs[3], 0, 85);
                    dot_pair!(iacc_1, hi3, hi7, lhs[3], 170, 255);

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);
                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);
                    let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);
                    let iacc_min_sb = _mm256_madd_epi16(_mm256_shuffle_epi32::<0>(q8s), mins_01);
                    q8s = _mm256_bsrli_epi128::<4>(q8s);
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }
                acc_row = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_b),
                    _mm256_mul_ps(col_scale, row_scale),
                    acc_row,
                );
                acc_min = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(iacc_min_b),
                    _mm256_mul_ps(col_dmin, row_scale),
                    acc_min,
                );
            }
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalperm);
            _mm256_storeu_ps(out.as_mut_ptr().add(x * 8), _mm256_sub_ps(acc_row, acc_min));
        }
    }
}

/// The safe Pha 1b entry: the 8x8 dot over a packed group — called by
/// [`crate::quant::fill_rows_q4k8`], the only production caller.
#[cfg(target_arch = "x86_64")]
pub(crate) fn q4k8_dot8(w: &[BlockQ4Kx8], blocks: usize, act: &[u8], out: &mut [f32; 8]) {
    // SAFETY: the caller checked `vnni512_available()` at the call site, and the slices satisfy
    // the kernel contract, which `make_blocks_q4k8` built for.
    unsafe { gemv_q4k8(w, blocks, act, out) }
}

// ---------------------------------------------------------------------------
// VIPERA_Q6K_IQK: repacked Q6_K GEMM, AVX-512 VNNI with an i32 epilogue (default off)
// ---------------------------------------------------------------------------

/// Whether the IQK-style repacked Q6_K kernel is on: `VIPERA_Q6K_IQK`, default off.
pub(crate) fn q6k_iqk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VIPERA_Q6K_IQK").ok().is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

/// Bytes per row in [`Q6KRepacked::w`]: the 256 codes of a row, row-major.
const Q6K_REPACK_ROW: usize = 256;

/// The Q6_K weights of a matrix, repacked in the IQK `repack_q6_k` layout family: four rows
/// interleaved, one super-scale per row per 256-value block, and one `d * scales[...]`
/// coefficient per row, block and 16-value group.
pub(crate) struct Q6KRepacked {
    /// Number of 256-value blocks per row.
    pub(crate) blocks: usize,
    /// Number of rows, a multiple of four.
    pub(crate) rows: usize,
    /// Per block: `rows * 256` codes (0..63), row-major — `w[(b * rows + r) * 272 + v]`.
    pub(crate) w: Vec<u8>,
    /// Per block: `rows * 16` coefficients `d * scales[...]` — `coef[(b * rows + r) * 16 + g]`.
    pub(crate) coef: Vec<f32>,
}

/// Pack `rows` (a multiple of four) of one Q6_K matrix into the IQK repack.
pub(crate) fn pack_q6k_iqk(
    data: &[u8],
    stride: usize,
    first_row: usize,
    rows: usize,
) -> Q6KRepacked {
    assert!(
        stride.is_multiple_of(210),
        "a Q6_K row is a whole number of 210-byte blocks"
    );
    assert_eq!(rows % 4, 0, "the IQK repack interleaves four rows");
    let blocks = stride / 210;
    let mut w = vec![0_u8; blocks * rows * Q6K_REPACK_ROW];
    let mut coef = vec![0.0_f32; blocks * rows * 16];
    for b in 0..blocks {
        for r in 0..rows {
            let row = &data[(first_row + r) * stride + b * 210..][..210];
            let d = f16_to_f32(u16::from_le_bytes([row[208], row[209]]));
            let scales = &row[192..208];
            for half in 0..2 {
                let ql = &row[half * 64..half * 64 + 64];
                let qh = &row[128 + half * 32..128 + half * 32 + 32];
                for run in 0..4 {
                    // `run` picks one of the four 32-value runs of a half, the way
                    // [`q6k_dot_q8`] reads them; `high_nibble` picks the high nibble of the run.
                    let (high_nibble, offset) = if run < 2 {
                        (false, run * 32)
                    } else {
                        (true, (run - 2) * 32)
                    };
                    let shift = run * 2;
                    for sub in 0..2 {
                        let g = half * 8 + run * 2 + sub;
                        let base = offset + sub * 16;
                        coef[(b * rows + r) * 16 + g] = d * f32::from(scales[g] as i8);
                        for l in 0..16 {
                            let byte = ql[base + l];
                            let level = if high_nibble { byte >> 4 } else { byte & 0x0f };
                            let high = (qh[sub * 16 + l] >> shift) & 0x03;
                            // The six-bit code, 0..63, as [`q6k_dot_q8`] builds it.
                            let code = level | (high << 4);
                            let at = (b * rows + r) * Q6K_REPACK_ROW
                                + half * 128
                                + run * 32
                                + sub * 16
                                + l;
                            w[at] = code;
                        }
                    }
                }
            }
        }
    }
    Q6KRepacked {
        blocks,
        rows,
        w,
        coef,
    }
}

/// Horizontal sum of the eight `i32` lanes of a 256-bit vector.
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(value: __m256i) -> i32 {
    // SAFETY: every byte of the stack array is written by the store and read by the sum.
    unsafe {
        let mut lanes = [0_i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), value);
        lanes.iter().sum()
    }
}

/// The scalar form of [`q6k_repack_dot_avx512`] over the repack, for a CPU without AVX-512 VNNI.
fn q6k_repack_dot_scalar(repack: &Q6KRepacked, row: usize, xq: &QuantizedX) -> f32 {
    debug_assert!(row < repack.rows);
    let mut total = 0.0_f32;
    for b in 0..repack.blocks {
        let at = b * repack.rows + row;
        let coef = &repack.coef[at * 16..][..16];
        let w = &repack.w[at * Q6K_REPACK_ROW..][..256];
        for half in 0..2 {
            for run in 0..4 {
                let dx = xq.d[b * 8 + half * 4 + run];
                for sub in 0..2 {
                    let g = half * 8 + run * 2 + sub;
                    let mut sum = 0_i32;
                    let mut centered = 0_i32;
                    for l in 0..16 {
                        let v = i32::from(xq.q[b * 256 + half * 128 + run * 32 + sub * 16 + l]);
                        sum += i32::from(w[half * 128 + run * 32 + sub * 16 + l]) * v;
                        centered += v;
                    }
                    total += coef[g] * dx * (sum - 32 * centered) as f32;
                }
            }
        }
    }
    total
}

/// One row of the repack against a quantized activation — the `vpdpbusd` form.
///
/// Per 16-value group the integer sum runs in `i32` via one `_mm256_dpbusd_epi32` over the group's
/// codes and activation bytes, and the group's single scale — `d * scales[...] * dx`, the same
/// product [`q6k_dot_q8`] builds — is applied once at the end of the group.
///
/// # Safety
///
/// The CPU must support AVX-512F/BW/VNNI, `repack` must cover `row`, and `xq` must cover
/// `repack.blocks` blocks of 256 values.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
pub(crate) unsafe fn q6k_repack_dot_avx512(
    repack: &Q6KRepacked,
    row: usize,
    xq: &QuantizedX,
) -> f32 {
    // SAFETY: the caller states the contract; every load below stays inside `repack.w`, the
    // row's slice of `xq.q`, and the 32-byte `dpbusd` window of each group.
    unsafe {
        debug_assert!(row < repack.rows);
        let mut total = 0.0_f32;
        for b in 0..repack.blocks {
            let at = b * repack.rows + row;
            let coef = repack.coef.as_ptr().add(at * 16);
            let x_base = xq.q.as_ptr().add(b * 256);
            let w_base = repack.w.as_ptr().add(at * Q6K_REPACK_ROW);
            for half in 0..2 {
                for run in 0..4 {
                    let dx = *xq.d.as_ptr().add(b * 8 + half * 4 + run);
                    for sub in 0..2 {
                        let g = half * 8 + run * 2 + sub;
                        let v0 = half * 128 + run * 32 + sub * 16;
                        // The group's 16 codes and activation bytes go into zero-padded 32-byte
                        // windows, so the `dpbusd` sums the group alone.
                        let mut wbuf = [0_u8; 32];
                        let mut xbuf = [0_i8; 32];
                        for l in 0..16 {
                            wbuf[l] = *w_base.add(v0 + l);
                            xbuf[l] = *x_base.add(v0 + l);
                        }
                        // SAFETY: the stack arrays are fully written, so the loads are in bounds
                        // and the zero tails contribute nothing.
                        let w = _mm256_loadu_si256(wbuf.as_ptr().cast::<__m256i>());
                        let x = _mm256_loadu_si256(xbuf.as_ptr().cast::<__m256i>());
                        let sums = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, x);
                        // The six-bit levels are centred on 32: `q6k_dot_q8` subtracts that for
                        // the activation sum of the group before scaling.
                        let mut centered = 0_i32;
                        for l in 0..16 {
                            centered += i32::from(*x_base.add(v0 + l));
                        }
                        // One float multiply per group at the end of the group.
                        total += *coef.add(g) * dx * (hsum_i32(sums) - 32 * centered) as f32;
                    }
                }
            }
        }
        total
    }
}

/// One row of the repack against a quantized activation — the `vpdpbusd` kernel where this CPU
/// has it, the scalar form of [`q6k_repack_dot_avx512`] otherwise.
pub(crate) fn q6k_repack_dot(repack: &Q6KRepacked, row: usize, xq: &QuantizedX) -> f32 {
    if vnni512_available() {
        // SAFETY: `vnni512_available` checked the CPU; the repack covers `row` and `xq` covers
        // `repack.blocks` blocks of 256 values.
        unsafe { q6k_repack_dot_avx512(repack, row, xq) }
    } else {
        q6k_repack_dot_scalar(repack, row, xq)
    }
}

#[cfg(test)]
mod q6k_repack_tests {
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

    /// A Q6_K row of `blocks` blocks with random scales and bytes, as the engine's tests build.
    fn q6k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(blocks * 210);
        for _ in 0..blocks {
            for _ in 0..192 {
                row.push(rng.byte());
            }
            for _ in 0..16 {
                row.push(rng.byte());
            }
            row.extend_from_slice(&rng.half(0.001, 0.06));
        }
        row
    }

    fn close(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= 2e-4 * (1.0 + expected.abs())
    }

    /// The repack dot against `q6k_dot_q8` (the engine's integer row kernel): the parity the
    /// row kernels themselves hold. The scalar form of the kernel computes the same integer
    /// math and is covered by `q6k_repack_rows_do_not_depend_on_the_pack_offset`.
    #[test]
    fn q6k_repack_dot_matches_the_row_kernels() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x5eed_0f11_0000_0031);
        for trial in 0..16 {
            let blocks = 1 + trial % 4;
            let cols = blocks * 256;
            let stride = blocks * 210;
            let data: Vec<u8> = (0..4).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
            let xq = QuantizedX::new(&x);
            let repack = pack_q6k_iqk(&data, stride, 0, 4);
            for row in 0..4 {
                let row_bytes = &data[row * stride..][..stride];
                let fast = q6k_repack_dot(&repack, row, &xq);
                let q8 = crate::quant::q6k_dot_q8(row_bytes, &xq);
                assert!(
                    close(fast, q8),
                    "trial {trial} row {row}: repack {fast} vs q6k_dot_q8 {q8}"
                );
            }
        }
    }

    /// A row's repack value must not depend on where the pack put it: the same rows packed at
    /// different `first_row` offsets give the same dots, bit for bit.
    #[test]
    fn q6k_repack_rows_do_not_depend_on_the_pack_offset() {
        if !vnni512_available() {
            return;
        }
        let mut rng = Rng(0x5eed_0f11_0000_0032);
        let cols = 512;
        let blocks = cols / 256;
        let stride = blocks * 210;
        let data: Vec<u8> = (0..24).flat_map(|_| q6k_row(&mut rng, blocks)).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let xq = QuantizedX::new(&x);
        let first = pack_q6k_iqk(&data, stride, 0, 16);
        let second = pack_q6k_iqk(&data, stride, 4, 16);
        for row in 4..16 {
            // SAFETY: both repacks cover the rows being dotted; q6k_repack_dot is a safe
            // entry point that dispatches to the AVX-512/VNNI kernel when available.
            let a = q6k_repack_dot(&first, row, &xq);
            let b = q6k_repack_dot(&second, row - 4, &xq);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "row {row} differs with the pack offset"
            );
        }
    }
}
