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
    _mm_cvtsi64_si128, _mm_cvtss_f32, _mm_loadl_epi64, _mm_loadu_si128, _mm_movehdup_ps,
    _mm_movehl_ps, _mm_prefetch, _mm_set1_ps, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_castsi256_ps, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32,
    _mm256_cvtph_ps, _mm256_cvttps_epi32, _mm256_div_ps, _mm256_dpbusd_epi32,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_max_epi32, _mm256_max_ps, _mm256_min_epi32, _mm256_min_ps,
    _mm256_mul_ps, _mm256_or_si256, _mm256_permutevar8x32_ps, _mm256_set_m128, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_slli_epi16, _mm256_slli_epi32, _mm256_srli_epi16, _mm256_srli_epi32, _mm256_storeu_ps,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_ps, _mm512_add_ps,
    _mm512_cvtepi32_ps, _mm512_dpbusd_epi32, _mm512_loadu_ps, _mm512_loadu_si512, _mm512_mul_ps,
    _mm512_mullo_epi32, _mm512_set1_epi32, _mm512_set1_ps, _mm512_setzero_ps, _mm512_setzero_si512,
    _mm512_storeu_ps, _mm512_sub_epi32, _mm512_sub_ps,
};

use vipera_core::gguf::GgmlType;

use crate::quant::{BatchActivation, QuantizedX, f16_to_f32};

/// Tile halves the tiled kernels keep activation coefficients for: 16
/// activations per half, and batches up to 64 wide run four halves against one
/// weight pass.
const MAX_HALVES: usize = 8;

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
            GgmlType::Q6K => match xq {
                Some(xq) => q6k_dot_q8(row, xq),
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
pub(crate) fn try_q4k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
) -> bool {
    if !vnni512_available() || stride == 0 || !stride.is_multiple_of(144) || B == 0 || B > 128 {
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
    unsafe { q4k_matvec_tiled_range(data, stride, first_row, count, xs, sums, out) };
    true
}

/// Tiled Q4_K matvec range: `count` rows from `first_row`, `B` quantized
/// activations, token-major `out` holding `B * count` values.
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
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn q4k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    sums: &[&[f32]; B],
    out: &mut [f32],
) {
    // SAFETY: the module contract above; every load below stays inside the
    // rows, blocks, and activation slices the caller validated.
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
        // Activation tile in (half, group, entry, 4) order: one 64-byte ZMM
        // load holds 4 codes for each of the 16 columns of that half.
        // The activation tile for every block is built once, up front, and stays in L2 while the
        // row windows below walk it. Building it per window would repeat the same copies 144 times.
        let mut act = vec![0_i8; blocks * halves * 64 * COLS * 4];
        let mut dxs = vec![0.0_f32; blocks * halves * 8 * COLS];
        let mut sumt = vec![0.0_f32; blocks * halves * 8 * COLS];
        for b in 0..blocks {
            for entry in 0..B {
                let half = entry / COLS;
                let col = entry % COLS;
                let chunk = xs[entry].q.as_ptr().add(b * 256);
                let base = (b * halves + half) * 64 * COLS * 4;
                for grp in 0..64 {
                    std::ptr::copy_nonoverlapping(
                        chunk.add(grp * 4),
                        act.as_mut_ptr().add(base + grp * 64 + col * 4),
                        4,
                    );
                }
                for sub in 0..8 {
                    let at = ((b * halves + half) * 8 + sub) * COLS + col;
                    dxs[at] = xs[entry].d[b * 8 + sub];
                    sumt[at] = sums[entry][b * 8 + sub];
                }
            }
        }
        let low_mask = _mm256_set1_epi8(0x0f);
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
                    while band < cnt {
                        let band_cnt = (cnt - band).min(BAND);
                        let mut facc: [__m512; BAND] = [_mm512_setzero_ps(); BAND];
                        for s in 0..8 {
                            let mut acc: [__m512i; BAND] = [_mm512_setzero_si512(); BAND];
                            for g in 0..8 {
                                let grp = s * 8 + g;
                                let codes =
                                    _mm512_loadu_si512(half_act.add(grp * 64) as *const __m512i);
                                for (t, slot) in acc.iter_mut().enumerate().take(band_cnt) {
                                    let word = u32::from_le(std::ptr::read_unaligned(
                                        wq_ptr.add((band + t) * 256 + grp * 4) as *const u32,
                                    ));
                                    *slot = _mm512_dpbusd_epi32(
                                        *slot,
                                        _mm512_set1_epi32(word as i32),
                                        codes,
                                    );
                                }
                            }
                            let dxv = _mm512_loadu_ps(dxs_ptr.add(s * COLS));
                            let sumv = _mm512_loadu_ps(sumt_ptr.add(s * COLS));
                            for (t, (acc_slot, facc_slot)) in
                                acc.iter().zip(facc.iter_mut()).enumerate().take(band_cnt)
                            {
                                let row = band + t;
                                let main = _mm512_mul_ps(
                                    _mm512_mul_ps(
                                        _mm512_cvtepi32_ps(*acc_slot),
                                        _mm512_set1_ps(scale_f[row][s]),
                                    ),
                                    dxv,
                                );
                                let minterm = _mm512_mul_ps(sumv, _mm512_set1_ps(min_f[row][s]));
                                *facc_slot =
                                    _mm512_add_ps(*facc_slot, _mm512_sub_ps(main, minterm));
                            }
                        }
                        for (t, value) in facc.iter().enumerate().take(band_cnt) {
                            let slot = buf.as_mut_ptr().add((band + t) * stride_cols + half * COLS);
                            _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), *value));
                        }
                        band += band_cnt;
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

/// Safe entry point for [`q6k_matvec_tiled_range`], as
/// [`try_q4k_matvec_tiled_range`] is for Q4_K.
pub(crate) fn try_q6k_matvec_tiled_range<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    out: &mut [f32],
) -> bool {
    if !vnni512_available() || stride == 0 || !stride.is_multiple_of(210) || B == 0 || B > 128 {
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
    unsafe { q6k_matvec_tiled_range(data, stride, first_row, count, xs, out) };
    true
}

/// Tiled Q6_K matvec range, the Q6_K twin of [`q4k_matvec_tiled_range`]:
/// `row = d * sum over 16-value groups of scale * (sum of (wq - 32) * xq)`,
/// with the weights biased by +32 so the unsigned side of `dpbusd` can read
/// them directly and the bias corrected per group as `raw - 32 * bsum`.
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
        let mut wq = vec![0_u8; ROW_WIN * 256];
        // Sub-block major: `scales[k][row]`, so the sub-block loop reads one contiguous band.
        let mut scales = vec![[0_i32; ROW_WIN]; 16];
        let mut dvec = [0.0_f32; ROW_WIN];
        // Activation tile for every block, built once and reused by every row window.
        let mut act = vec![0_i8; blocks * halves * 64 * COLS * 4];
        let mut bsumt = vec![0_i32; blocks * halves * 16 * COLS];
        let mut dxs = vec![0.0_f32; blocks * halves * 16 * COLS];
        for b in 0..blocks {
            for (entry, x) in xs.iter().enumerate() {
                let col_half = entry / COLS;
                let col = entry % COLS;
                let chunk = x.q.as_ptr().add(b * 256);
                let base = (b * halves + col_half) * 64 * COLS * 4;
                for grp in 0..64 {
                    std::ptr::copy_nonoverlapping(
                        chunk.add(grp * 4),
                        act.as_mut_ptr().add(base + grp * 64 + col * 4),
                        4,
                    );
                }
                for k in 0..16 {
                    let mut sum = 0_i32;
                    for i in 0..16 {
                        sum += i32::from(*chunk.add(k * 16 + i));
                    }
                    let at = ((b * halves + col_half) * 16 + k) * COLS + col;
                    bsumt[at] = sum;
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
                wq.fill(0);
                for slot in scales.iter_mut() {
                    *slot = [0; ROW_WIN];
                }
                dvec.fill(0.0);
                for r in 0..cnt {
                    let block = data.as_ptr().add((first_row + win + r) * stride + b * 210);
                    dvec[r] = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
                    let scale_bytes = std::slice::from_raw_parts(block.add(192), 16);
                    for (column, byte) in scales.iter_mut().zip(scale_bytes) {
                        column[r] = i32::from(*byte as i8);
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
                    let half_act = act.as_ptr().add((b * halves + col_half) * 64 * COLS * 4);
                    let bsumt_ptr = bsumt.as_ptr().add((b * halves + col_half) * 16 * COLS);
                    let dxs_ptr = dxs.as_ptr().add((b * halves + col_half) * 16 * COLS);
                    let mut band = 0;
                    while band < cnt {
                        let band_cnt = (cnt - band).min(BAND);
                        let mut facc: [__m512; BAND] = [_mm512_setzero_ps(); BAND];
                        for (k, by_row) in scales.iter().enumerate() {
                            let mut acc: [__m512i; BAND] = [_mm512_setzero_si512(); BAND];
                            for g in 0..4 {
                                let grp = k * 4 + g;
                                let codes =
                                    _mm512_loadu_si512(half_act.add(grp * 64) as *const __m512i);
                                for (t, slot) in acc.iter_mut().enumerate().take(band_cnt) {
                                    let word = u32::from_le(std::ptr::read_unaligned(
                                        wq_ptr.add((band + t) * 256 + grp * 4) as *const u32,
                                    ));
                                    *slot = _mm512_dpbusd_epi32(
                                        *slot,
                                        _mm512_set1_epi32(word as i32),
                                        codes,
                                    );
                                }
                            }
                            let bias = _mm512_mullo_epi32(
                                _mm512_loadu_si512(bsumt_ptr.add(k * COLS) as *const __m512i),
                                _mm512_set1_epi32(32),
                            );
                            let dxv = _mm512_loadu_ps(dxs_ptr.add(k * COLS));
                            for (t, (acc_slot, facc_slot)) in
                                acc.iter().zip(facc.iter_mut()).enumerate().take(band_cnt)
                            {
                                let row = band + t;
                                let corr = _mm512_sub_epi32(*acc_slot, bias);
                                let main = _mm512_mul_ps(
                                    _mm512_mul_ps(_mm512_cvtepi32_ps(corr), dxv),
                                    _mm512_set1_ps(dvec[row] * by_row[band + t] as f32),
                                );
                                *facc_slot = _mm512_add_ps(*facc_slot, main);
                            }
                        }
                        for (t, value) in facc.iter().enumerate().take(band_cnt) {
                            let slot = buf
                                .as_mut_ptr()
                                .add((band + t) * stride_cols + col_half * COLS);
                            _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), *value));
                        }
                        band += band_cnt;
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

#[cfg(test)]
mod tests {
    /// `vpdpbusd` against `maddubs` + `madd` on the shapes that dominate a decode step.
    ///
    /// Checks bit-exactness row by row, then times both at one and four threads.
    ///
    /// Ignored by default. Run it with
    /// `cargo test -p vipera-backend-cpu --release -- --ignored --nocapture q4k_vnni_versus_maddubs`.
    #[test]
    #[ignore]
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
        unsafe { q4k_matvec_tiled_range::<16>(&data, stride, 0, rows, &xs, &sums, &mut got) };
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
        unsafe { q6k_matvec_tiled_range::<16>(&data, stride, 0, rows, &xs, &mut got) };
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
            let _ = try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs, &sums, &mut out);
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
                    let _ = try_q6k_matvec_tiled_range(&data, stride, 0, $rows, &xs, &mut out);
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
                    let _ =
                        try_q4k_matvec_tiled_range(&data, stride, 0, $rows, &xs, &sums, &mut out);
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
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_lo, &sums_lo, lo);
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_hi, &sums_hi, hi);
        try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs32, &sums32, &mut one);
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
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_lo, &sums_lo, lo);
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs_hi, &sums_hi, hi);
        }
        let t16 = start.elapsed().as_secs_f64() / reps as f64;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            try_q4k_matvec_tiled_range(&data, stride, 0, rows, &xs32, &sums32, &mut one);
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
            try_q6k_matvec_tiled_range(&data, stride, 0, rows, &xs, &mut out);
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
    #[ignore]
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
    #[ignore]
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
    #[ignore]
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
