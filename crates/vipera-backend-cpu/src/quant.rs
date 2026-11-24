//! Block formats and the quantized matrix-vector product.
//!
//! A GGUF matrix is stored row-major over `dims[0]`: row `r` starts at `r * row_bytes` and holds
//! `cols = dims[0]` values. Every block size here divides the row length of a real model.
//!
//! Only the formats a Q4_K_M file can contain are implemented (F32, F16, Q8_0, Q4_K, Q6_K). Anything
//! else is refused by name rather than dequantized approximately: a wrong scale convention produces
//! fluent-looking nonsense, which is harder to notice than an error.

use std::sync::Arc;
use std::thread;

use vipera_core::gguf::GgmlType;
use vipera_errors::ViperaError;

/// Values per block and bytes per block, for the formats this backend understands.
const fn block(ty: GgmlType) -> Option<(usize, usize)> {
    match ty {
        GgmlType::F32 => Some((1, 4)),
        GgmlType::F16 => Some((1, 2)),
        GgmlType::Q8_0 => Some((32, 34)),
        GgmlType::Q4K => Some((256, 144)),
        GgmlType::Q6K => Some((256, 210)),
        _ => None,
    }
}

/// Bytes one row of `cols` values occupies.
pub fn row_bytes(ty: GgmlType, cols: usize) -> Result<usize, ViperaError> {
    let (values, bytes) = block(ty).ok_or_else(|| unsupported(ty))?;
    if !cols.is_multiple_of(values) {
        return Err(ViperaError::unsupported(format!(
            "a row of {cols} {} values is not a whole number of {values}-value blocks",
            ty.name()
        )));
    }
    (cols / values)
        .checked_mul(bytes)
        .ok_or_else(|| ViperaError::invalid_argument("row byte count overflows"))
}

/// Whole blocks of `N` bytes, with any remainder ignored. Callers validated the length first.
fn blocks<const N: usize>(data: &[u8]) -> &[[u8; N]] {
    data.as_chunks::<N>().0
}

fn unsupported(ty: GgmlType) -> ViperaError {
    ViperaError::unsupported(format!(
        "the CPU backend cannot read {} weights; supported formats are F32, F16, Q8_0, Q4_K, and Q6_K",
        ty.name()
    ))
}

/// Half-precision bits to `f32`, by moving bits rather than multiplying by a power.
///
/// This is on the hot path: a Q4_K block calls it twice and a Q6_K block once, so 12 to 14 calls
/// per `matvec` row. The obvious implementation, `(1024 + mantissa) * 2f32.powi(exponent - 25)`,
/// passes a runtime exponent to `powi`, which compiles to the `__powisf2` compiler-rt helper — a
/// loop — and not to a multiply. That cost showed up as the Q4_K kernel measuring 1.67 GB/s
/// single-threaded with the nibble arithmetic accounting for a fraction of the time.
///
/// Subnormal inputs are normalized by shifting the mantissa left until its top bit reaches the
/// explicit-bit position, which is why they cost a `leading_zeros` instead of a special case.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let out = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Value is `mantissa * 2^-24`; put the leading bit at 2^10 and count the shift down.
            let shift = mantissa.leading_zeros() - 21;
            let normalized = (mantissa << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (normalized << 13)
        }
    } else if exponent == 0x1f {
        // Infinity when the mantissa is zero, otherwise a NaN; both keep the exponent all ones.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// `f32` to half-precision bits, rounding to nearest with the low bits dropped.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | ((mantissa >> 13) as u16 & 0x3ff);
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    if biased <= 0 {
        if biased < -10 {
            return sign;
        }
        let shifted = (mantissa | 0x80_0000) >> (1 - biased);
        return sign | ((shifted + 0x1000) >> 13) as u16;
    }
    let rounded = (mantissa + 0x1000) >> 13;
    if rounded >= 0x400 {
        sign | (((biased + 1) as u16) << 10)
    } else {
        sign | ((biased as u16) << 10) | rounded as u16
    }
}

fn read_f16(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// The 6-bit scale and minimum of one Q4_K sub-block.
///
/// Eight sub-blocks share twelve bytes: four low six-bit scales, four low six-bit minimums, then the
/// high two bits of both packed into the last four bytes.
pub(crate) fn q4k_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Sums of `x` over every 32 values: the eight Q4_K sub-block sums of each 256-value block.
///
/// A k-quant minimum is constant inside one 32-value sub-block, so the term `-dmin * m * Σx` needs
/// only these sums: computing them once per matvec removes one multiply and one subtract from every
/// weight, and one add per sub-block from every row. Each sum is the pair of 16-value partial sums
/// added in the order the kernels used before, so the value is unchanged to the bit.
pub(crate) fn x_sums(x: &[f32]) -> Vec<f32> {
    x_sums_iter(x).collect()
}

/// [`x_sums`] without the `Vec`, so a caller that already owns a buffer can extend it directly.
fn x_sums_iter(x: &[f32]) -> impl Iterator<Item = f32> + '_ {
    x.as_chunks::<16>()
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| pair[0].iter().sum::<f32>() + pair[1].iter().sum::<f32>())
}

/// `x` rounded to `i8`, one scale per 32 values.
///
/// The AVX2 Q4_K kernel multiplies weights by `i8` for a reason that is arithmetic, not taste:
/// `_mm256_maddubs_epi16` turns 32 unsigned weights and 32 signed activations into 16 pair sums in
/// one instruction, where the `f32` path needs four widen-multiply-add rounds for the same 32
/// weights. The cost is that `x` gains a quantization error of at most `0.5/127` of its largest
/// magnitude in each 32-value group.
///
/// The scales are per 32 values because a Q4_K sub-block is 32 values wide: one scale per Q4_K
/// sub-block means the kernel multiplies by a single constant instead of feeding the scale through
/// the integer part.
pub(crate) struct QuantizedX {
    /// The rounded values, `32` per group, in row order.
    pub(crate) q: Vec<i8>,
    /// One scale per 32-value group; a group of zeros gets scale `0`.
    pub(crate) d: Vec<f32>,
}

/// Whether this format's dot reads an `i8` activation.
///
/// `VIPERA_EXACT_DOT=1` takes Q6_K back off that path, which restores the `f32` kernel and with it
/// the tokens the engine produced before the q8 kernel existed; it does not apply to Q4_K.
/// `VIPERA_Q4K_F32=1` opts Q4_K into `f32` activations instead (bit-identical to the `f32` path on
/// the checked prompts, slower).
fn quantizes(ty: GgmlType) -> bool {
    quantizes_for(ty, q4k_f32_activations_requested(), exact_dot_requested())
}

/// [`quantizes`] with the switches passed in, so the mapping is testable without touching the
/// process environment (each switch is cached in a `OnceLock`).
fn quantizes_for(ty: GgmlType, q4k_f32: bool, exact: bool) -> bool {
    match ty {
        GgmlType::Q4K => !q4k_f32,
        GgmlType::Q6K => !exact,
        _ => false,
    }
}

/// Whether an environment value asks for a switch: unset, empty, and `0` mean no; anything else
/// means yes.
fn env_truthy(value: Option<&str>) -> bool {
    value
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
        .unwrap_or(false)
}

/// Whether `VIPERA_Q4K_F32` asked for `f32` activations on Q4_K rows, so the dot runs on the
/// dequantized `f32` path instead of `i8` activations (bit-identical to the `f32` path, slower).
pub(crate) fn q4k_f32_activations_requested() -> bool {
    static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F32.get_or_init(|| env_truthy(std::env::var("VIPERA_Q4K_F32").ok().as_deref()))
}

/// Whether `VIPERA_EXACT_DOT` asked for the pre-q8 Q6_K kernel, and for the sequential attention
/// score dot, so the tokens before either change can still be reproduced.
pub(crate) fn exact_dot_requested() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| env_truthy(std::env::var("VIPERA_EXACT_DOT").ok().as_deref()))
}

/// Whether the `vpdpbusd` Q6_K kernel is on: it is the default, and `VIPERA_Q6K_VNNI=0` turns it
/// off so the `maddubs` kernel can still be reproduced.
pub(crate) fn q6k_vnni_requested() -> bool {
    static Q6K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Q6K.get_or_init(|| {
        std::env::var("VIPERA_Q6K_VNNI").map_or(true, |value| {
            let value = value.trim();
            !value.is_empty() && !env_truthy(Some(value))
        })
    })
}

impl QuantizedX {
    /// Quantize `x` in groups of 32. A trailing partial group is ignored, and every caller in this
    /// module passes a length that is a whole number of groups.
    pub(crate) fn new(x: &[f32]) -> Self {
        let groups = x.len() / 32;
        let mut q = vec![0_i8; groups * 32];
        let mut d = vec![0.0_f32; groups];
        // The AVX2 path produces exactly the values `quantize_group` does, so which one ran cannot
        // change a result; it only changes how long the activation takes to build. At 1536 values
        // the scalar loop is ~8 us, which a decode step pays on every one of its ~200 matvecs.
        #[cfg(target_arch = "x86_64")]
        if crate::simd::quantize_groups(x, &mut q, &mut d) {
            return Self { q, d };
        }
        for ((group, scale), values) in x
            .as_chunks::<32>()
            .0
            .iter()
            .zip(&mut d)
            .zip(q.as_chunks_mut::<32>().0.iter_mut())
        {
            *scale = Self::quantize_group(group, values);
        }
        Self { q, d }
    }

    /// Quantize one group of 32 into `out` and return its scale (0 for an all-zero group).
    ///
    /// The definition the AVX2 path in [`crate::simd::quantize_groups`] is tested against, and the
    /// path it falls back to for a group holding a NaN, an infinity or a subnormal scale.
    pub(crate) fn quantize_group(group: &[f32; 32], out: &mut [i8]) -> f32 {
        let largest = group.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        if largest == 0.0 {
            out.fill(0);
            return 0.0;
        }
        let scale = largest / 127.0;
        for (slot, value) in out.iter_mut().zip(group) {
            *slot = (value / scale).round().clamp(-127.0, 127.0) as i8;
        }
        scale
    }

    /// The dot this struct stands for, evaluated in `f32` against the original `x`.
    ///
    /// Used by tests to bound what quantizing the activation costs; the kernels never call it.
    #[cfg(test)]
    pub(crate) fn dequantized(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(at, value)| f32::from(*value) * self.d[at / 32])
            .collect()
    }
}

/// The W-1a tiled-band activation (`VIPERA_TILED_A`, `docs/design/repack-port-plan.md`, W-1a): the
/// activation the band quantizes **per 256-value block** — one `f32` scale per block instead of
/// the per-32 scales of [`QuantizedX`] — plus the block's 16 raw-`x` sums the band's minimum term
/// reads (ik `block_q8_K`-style `bsums`). The integer codes are the same values the band's
/// `dpbusd` has always multiplied; only their scale and granularity changed, so the band itself
/// is unchanged — W-1a is a numerics change (hence the default-off switch), not a speed one.
///
/// Built per product by [`TiledActivation::new`]; consumed by
/// [`crate::simd::q4k_matvec_tiled_range_a`].
pub(crate) struct TiledActivation {
    /// Per entry: the activation's codes, `blocks * 256` values in natural order.
    q: Vec<Vec<i8>>,
    /// Per entry: one scale per block.
    scale: Vec<Vec<f32>>,
    /// Per entry: `blocks * 16` sums — group `j` of block `b` is the raw `x` over
    /// `[b * 256 + 16j, b * 256 + 16(j + 1))`.
    bsums: Vec<Vec<f32>>,
}

impl TiledActivation {
    /// Quantize every entry of `x`: per block, `scale = max|x| / 127` and
    /// `codes = round(x / scale)` (the `QuantizedX::quantize_group` definition with 256-value
    /// granularity), and the block's 16 `bsums` — `Σ raw x` over each 16-value group.
    pub(crate) fn new<const B: usize>(x: [&[f32]; B]) -> Self {
        let mut q = Vec::with_capacity(x.len());
        let mut scale = Vec::with_capacity(x.len());
        let mut bsums = Vec::with_capacity(x.len());
        for values in x {
            let mut codes = Vec::with_capacity(values.len());
            let mut block_scales = Vec::with_capacity(values.len() / 256);
            let mut sums = Vec::with_capacity(values.len() / 16);
            for chunk in values.as_chunks::<256>().0 {
                let largest = chunk
                    .iter()
                    .fold(0.0_f32, |acc, value| acc.max(value.abs()));
                let block_scale = if largest == 0.0 { 0.0 } else { largest / 127.0 };
                block_scales.push(block_scale);
                for value in chunk {
                    let code = if block_scale == 0.0 {
                        0
                    } else {
                        (value / block_scale).round().clamp(-127.0, 127.0) as i8
                    };
                    codes.push(code);
                }
                for group in 0..16 {
                    let sum: f32 = chunk[group * 16..(group + 1) * 16].iter().sum();
                    sums.push(sum);
                }
            }
            q.push(codes);
            scale.push(block_scales);
            bsums.push(sums);
        }
        Self { q, scale, bsums }
    }

    /// The entry's codes, `blocks * 256` values in natural order.
    pub(crate) fn q(&self, entry: usize) -> &[i8] {
        &self.q[entry]
    }

    /// The entry's per-block scales, one per 256 values.
    pub(crate) fn scale(&self, entry: usize) -> &[f32] {
        &self.scale[entry]
    }

    /// The entry's per-block sums: `bsums[entry][block * 16 + group]` — the raw `x` over the
    /// group's 16 values, the exact `f32` values the band's minimum term reads.
    pub(crate) fn bsums(&self, entry: usize) -> &[f32] {
        &self.bsums[entry]
    }
}

/// Dot product of one Q4_K row with `x`.
pub(crate) fn q4k_dot(row: &[u8], x: &[f32], sums: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (b, (block, xb)) in blocks::<144>(row)
        .iter()
        .zip(x.as_chunks::<256>().0)
        .enumerate()
    {
        let d = read_f16(block, 0);
        let dmin = read_f16(block, 2);
        let scales = &block[4..16];
        let sums = &sums[b * 8..b * 8 + 8];
        for (j, (q, x64)) in blocks::<32>(&block[16..144])
            .iter()
            .zip(xb.as_slice().as_chunks::<64>().0)
            .enumerate()
        {
            let (scale0, min0) = q4k_scale_min(scales, 2 * j);
            let (scale1, min1) = q4k_scale_min(scales, 2 * j + 1);
            let d0 = d * f32::from(scale0);
            let m0 = dmin * f32::from(min0);
            let d1 = d * f32::from(scale1);
            let m1 = dmin * f32::from(min1);
            let (low, high) = x64.split_at(32);
            let mut a0 = 0.0_f32;
            let mut a1 = 0.0_f32;
            for (byte, value) in q.iter().zip(low) {
                a0 += f32::from(byte & 0x0f) * value;
            }
            for (byte, value) in q.iter().zip(high) {
                a1 += f32::from(byte >> 4) * value;
            }
            let sum0 = sums[2 * j];
            let sum1 = sums[2 * j + 1];
            sum += (d0 * a0 - m0 * sum0) + (d1 * a1 - m1 * sum1);
        }
    }
    sum
}

/// Dot product of one Q6_K row with `x`.
///
/// The block packs 256 six-bit values: low nibbles in `ql`, the two high bits in `qh`, sixteen signed
/// sub-block scales, and one half-precision multiplier.
pub(crate) fn q6k_dot(row: &[u8], x: &[f32], _sums: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (block, xb) in blocks::<210>(row).iter().zip(x.as_chunks::<256>().0) {
        let d = read_f16(block, 208);
        let scales = &block[192..208];
        for half in 0..2 {
            let ql = &block[half * 64..half * 64 + 64];
            let qh = &block[128 + half * 32..128 + half * 32 + 32];
            let x128 = &xb[half * 128..half * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let scale = |index: usize| d * f32::from(scales[half * 8 + index] as i8);
                let q1 = ((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
                sum += scale(is) * (q1 as f32 * x128[l]);
                sum += scale(is + 2) * (q2 as f32 * x128[l + 32]);
                sum += scale(is + 4) * (q3 as f32 * x128[l + 64]);
                sum += scale(is + 6) * (q4 as f32 * x128[l + 96]);
            }
        }
    }
    sum
}

/// Dot product of one Q6_K row against an `i8` activation.
///
/// The same six-bit unpacking as [`q6k_dot`], but each 16-value group is summed in integers: the
/// unsigned level `q` (0..63) against the activation byte, with `32 * sum(xq)` subtracted for the
/// `-32` offset of the six-bit levels. Every product fits `i16` and every group sum fits `i32`, so
/// the integer side is exact; the float side is one multiply per 16 values.
pub(crate) fn q6k_dot_q8(row: &[u8], xq: &QuantizedX) -> f32 {
    let mut total = 0.0_f32;
    for ((block, q), dxs) in blocks::<210>(row)
        .iter()
        .zip(xq.q.as_chunks::<256>().0)
        .zip(xq.d.as_chunks::<8>().0)
    {
        let d = read_f16(block, 208);
        let scales = &block[192..208];
        let mut accumulator = [0.0_f32; 16];
        for half in 0..2 {
            let ql = &block[half * 64..half * 64 + 64];
            let qh = &block[128 + half * 32..128 + half * 32 + 32];
            let values = &q[half * 128..half * 128 + 128];
            for run in 0..4 {
                // `run` picks one of the four 32-value runs of a half: the first two take the low
                // nibbles of the two `ql` planes, the last two their high nibbles, and the two high
                // bits of `qh` move up by two for each run.
                let (high_nibble, offset) = if run < 2 {
                    (false, run * 32)
                } else {
                    (true, (run - 2) * 32)
                };
                let shift = run * 2;
                let dx = dxs[half * 4 + run];
                for sub in 0..2 {
                    let mut sum = 0_i32;
                    for l in sub * 16..sub * 16 + 16 {
                        let byte = ql[offset + l];
                        let level = if high_nibble { byte >> 4 } else { byte & 0x0f };
                        let high = (qh[l] >> shift) & 0x03;
                        let q6 = i32::from(level) | (i32::from(high) << 4);
                        sum += (q6 - 32) * i32::from(values[run * 32 + l]);
                    }
                    let scale = d * f32::from(scales[half * 8 + run * 2 + sub] as i8) * dx;
                    accumulator[half * 8 + run * 2 + sub] += scale * sum as f32;
                }
            }
        }
        total += accumulator.iter().sum::<f32>();
    }
    total
}

/// Dot product of one row with `x` in portable Rust, for any supported format.
///
/// This is the reference the AVX2 kernels are checked against, and the path taken on CPUs without it.
/// `xq`, when present, is the quantization of `x`; the formats with an integer kernel read it.
pub(crate) fn dot_scalar(
    ty: GgmlType,
    row: &[u8],
    x: &[f32],
    sums: &[f32],
    xq: Option<&QuantizedX>,
) -> f32 {
    match ty {
        GgmlType::F32 => blocks::<4>(row)
            .iter()
            .zip(x)
            .map(|(bytes, value)| f32::from_le_bytes(*bytes) * value)
            .sum(),
        GgmlType::F16 => blocks::<2>(row)
            .iter()
            .zip(x)
            .map(|(bytes, value)| read_f16(bytes, 0) * value)
            .sum(),
        GgmlType::Q8_0 => {
            let mut sum = 0.0_f32;
            for (block, xb) in blocks::<34>(row).iter().zip(x.as_chunks::<32>().0) {
                let d = read_f16(block, 0);
                let mut inner = 0.0_f32;
                for (byte, value) in block[2..34].iter().zip(xb.iter()) {
                    inner += f32::from(*byte as i8) * value;
                }
                sum += d * inner;
            }
            sum
        }
        GgmlType::Q4K => q4k_dot(row, x, sums),
        GgmlType::Q6K => match xq {
            Some(xq) => q6k_dot_q8(row, xq),
            None => q6k_dot(row, x, sums),
        },
        other => {
            let _ = other;
            0.0
        }
    }
}

/// Dot product of one row with `x`, using the best kernel this CPU has.
///
/// Only the AVX2 module is allowed to use `unsafe`, and it re-checks the CPU itself, so this stays a
/// safe function and a CPU without AVX2 simply takes the portable path.
fn dot(ty: GgmlType, row: &[u8], activation: Activation<'_>) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if let Some(value) = crate::simd::try_dot(ty, row, activation.x, activation.sums, activation.q)
    {
        return value;
    }
    dot_scalar(ty, row, activation.x, activation.sums, activation.q)
}

/// Dot products of one row with `B` activations, sharing the row read.
///
/// This is [`dot`] for a batch: the AVX2 module replaces it when the CPU has it, and every other
/// format or CPU takes one [`dot_scalar`] per entry. Each entry's arithmetic is the same as its own
/// single call, in the same order, so a batch reproduces the unbatched results exactly.
fn dot_batch<const B: usize>(
    ty: GgmlType,
    row: &[u8],
    activation: &BatchActivation<'_, B>,
    out: &mut [f32; B],
) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::try_dot_batch(ty, row, activation, out) {
        return;
    }
    let quantized = activation.quantized();
    for (entry, out) in out.iter_mut().enumerate() {
        *out = dot_scalar(
            ty,
            row,
            activation.x[entry],
            activation.sums[entry],
            quantized.map(|xs| xs[entry]),
        );
    }
}

/// The activation vector, with the state `matvec` derives from it once per call.
///
/// Bundling the three slices keeps the per-row plumbing to one argument, and makes it obvious that
/// the quantized copy and the sums always come from the same `x`.
#[derive(Clone, Copy)]
pub(crate) struct Activation<'a> {
    /// The `cols` values being multiplied.
    x: &'a [f32],
    /// Sums of `x` over every 32 values, for the minimum term.
    sums: &'a [f32],
    /// `x` rounded to `i8`; only built for Q4_K rows.
    q: Option<&'a QuantizedX>,
}

/// The same state, copied so a worker thread can own it.
///
/// A pool of workers outlives one `matvec`, and its jobs cannot borrow the caller's scratch
/// buffers, so the activation travels by value. For this model that is a few kilobytes per call
/// against a tensor read of megabytes.
pub(crate) struct OwnedActivation {
    x: Vec<f32>,
    sums_off: usize,
    sums_len: usize,
    q: Option<QuantizedX>,
}

impl OwnedActivation {
    /// Kept for the tests that compare a fused activation against the plain build.
    #[cfg(test)]
    pub(crate) fn new(ty: GgmlType, x: &[f32]) -> Self {
        let groups = x.len() / 32;
        // `x` and its group sums share one buffer: one allocation instead of two.
        let mut buf = Vec::with_capacity(x.len() + groups);
        buf.extend_from_slice(x);
        let sums_off = buf.len();
        buf.extend(x_sums_iter(x));
        Self {
            x: buf,
            sums_off,
            sums_len: groups,
            q: quantizes(ty).then(|| QuantizedX::new(x)),
        }
    }

    /// [`OwnedActivation::new`] over a buffer that already holds `x` in its first `values` slots,
    /// finished in place: the group sums are appended and the quantiser reads the same values, so
    /// the result is `new(ty, &buffer[..values])` bit for bit without a second copy of `x`.
    ///
    /// `fusion` is applied to those values first. That is the whole point of the transform living
    /// in this module: the buffer was just filled from the caller's activation, and a worker that
    /// already holds it can transform, sum and quantise without the caller running the transform
    /// pass on its own thread.
    pub(crate) fn from_buffer_fused(
        ty: GgmlType,
        mut buffer: Vec<f32>,
        values: usize,
        fusion: &Fusion,
        entry: usize,
    ) -> Self {
        buffer.truncate(values);
        fusion.apply(entry, &mut buffer);
        let groups = values / 32;
        let sums: Vec<f32> = x_sums_iter(&buffer).collect();
        buffer.extend_from_slice(&sums);
        let q = quantizes(ty).then(|| QuantizedX::new(&buffer[..values]));
        Self {
            x: buffer,
            sums_off: values,
            sums_len: groups,
            q,
        }
    }

    /// The activation `x`, as the batched kernels read it.
    pub(crate) fn x(&self) -> &[f32] {
        &self.x[..self.sums_off]
    }

    /// The 32-value sums of `x`.
    pub(crate) fn sums(&self) -> &[f32] {
        &self.x[self.sums_off..self.sums_off + self.sums_len]
    }

    /// `x` rounded to `i8`, when this activation was built for Q4_K.
    pub(crate) fn quantized(&self) -> Option<&QuantizedX> {
        self.q.as_ref()
    }
}

/// A transform a batched product applies to each activation before it is summed and quantised.
///
/// It lives here rather than in the model because the only place it can run for free is the copy
/// into a worker's buffer: a chunk spends 12 ms on the feed-forward norm and 30 ms on the SwiGLU
/// multiply on the calling thread, between products that leave every worker idle. Both transforms
/// are elementwise and read nothing but their own inputs, so a worker computing them produces the
/// same bits the caller's pass produced.
///
/// The inputs a transform needs are owned (`Arc`), because a job travels to a worker with the
/// pool's data lifetime and the caller's scratch buffers are shorter-lived than that.
#[derive(Clone)]
pub(crate) enum Fusion {
    /// Nothing; the activation is copied as given.
    None,
    /// `rms_norm(x, weight, epsilon)`, in place.
    Norm { weight: Arc<[f32]>, epsilon: f32 },
    /// `x * silu(gate[entry])`, the SwiGLU multiply, entry `entry` of a `B`-entry batch.
    Silu { gate: Arc<[f32]>, width: usize },
}

impl Fusion {
    /// Apply the transform to entry `entry`'s values.
    fn apply(&self, entry: usize, values: &mut [f32]) {
        match self {
            Fusion::None => {}
            Fusion::Norm { weight, epsilon } => {
                crate::model::rms_norm_in_place(values, weight, *epsilon);
            }
            Fusion::Silu { gate, width } => {
                if let Some(gate) = gate.get(entry * width..(entry + 1) * width) {
                    crate::simd::silu_mul(values, gate);
                }
            }
        }
    }

    /// Whether this transform does nothing, in which case the paths that do not copy an activation
    /// (the tiled product) still apply.
    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Fusion::None)
    }
}

/// `B` activations borrowed together, so one row read serves all of them.
///
/// Built once per batched matvec rather than once per row: rebuilding it inside the row loop would
/// put an allocation back into the very place the batched kernel exists to remove.
pub(crate) struct BatchActivation<'a, const B: usize> {
    x: [&'a [f32]; B],
    sums: [&'a [f32]; B],
    /// Present only when every entry was quantized, which is the Q4_K case.
    q: Option<[&'a QuantizedX; B]>,
}

impl<'a, const B: usize> BatchActivation<'a, B> {
    /// Borrow the pieces of `B` owned activations that came from the same format.
    pub(crate) fn new(activations: &'a [OwnedActivation; B]) -> Self {
        let q = activations
            .iter()
            .map(OwnedActivation::quantized)
            .collect::<Option<Vec<_>>>()
            .and_then(|every| every.try_into().ok());
        Self {
            x: std::array::from_fn(|entry| activations[entry].x()),
            sums: std::array::from_fn(|entry| activations[entry].sums()),
            q,
        }
    }

    /// The activations themselves.
    ///
    /// The scalar and NEON kernels read the f32 form directly, so only the x86 batch kernels need
    /// these accessors; the aarch64 build would otherwise carry two dead methods.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn x(&self) -> &[&'a [f32]; B] {
        &self.x
    }

    /// The 32-value sums of each activation.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn sums(&self) -> &[&'a [f32]; B] {
        &self.sums
    }

    /// The quantized activations, when every entry has one.
    pub(crate) fn quantized(&self) -> Option<&[&'a QuantizedX; B]> {
        self.q.as_ref()
    }
}

/// Dequantize one row into `out`.
pub fn read_row(
    ty: GgmlType,
    data: &[u8],
    cols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let stride = row_bytes(ty, cols)?;
    let start = row
        .checked_mul(stride)
        .ok_or_else(|| ViperaError::invalid_argument("row offset overflows"))?;
    if out.len() != cols {
        return Err(ViperaError::internal(format!(
            "row buffer is {} values, expected {cols}",
            out.len()
        )));
    }
    let end = start
        .checked_add(stride)
        .ok_or_else(|| ViperaError::invalid_argument("row offset overflows"))?;
    let bytes = data
        .get(start..end)
        .ok_or_else(|| ViperaError::parse(start as u64, "row lies outside the tensor data"))?;
    match ty {
        GgmlType::F32 => {
            for (value, chunk) in out.iter_mut().zip(blocks::<4>(bytes).iter()) {
                *value = f32::from_le_bytes(*chunk);
            }
        }
        GgmlType::F16 => {
            for (value, chunk) in out.iter_mut().zip(blocks::<2>(bytes).iter()) {
                *value = read_f16(chunk, 0);
            }
        }
        GgmlType::Q8_0 => {
            for (block, chunk) in blocks::<34>(bytes).iter().zip(out.as_chunks_mut::<32>().0) {
                let d = read_f16(block, 0);
                for (value, byte) in chunk.iter_mut().zip(block[2..34].iter()) {
                    *value = d * f32::from(*byte as i8);
                }
            }
        }
        GgmlType::Q4K => {
            for (block, chunk) in blocks::<144>(bytes)
                .iter()
                .zip(out.as_chunks_mut::<256>().0)
            {
                let d = read_f16(block, 0);
                let dmin = read_f16(block, 2);
                let scales = &block[4..16];
                for (j, (q, x64)) in blocks::<32>(&block[16..144])
                    .iter()
                    .zip(chunk.as_mut_slice().as_chunks_mut::<64>().0)
                    .enumerate()
                {
                    let (scale0, min0) = q4k_scale_min(scales, 2 * j);
                    let (scale1, min1) = q4k_scale_min(scales, 2 * j + 1);
                    let d0 = d * f32::from(scale0);
                    let m0 = dmin * f32::from(min0);
                    let d1 = d * f32::from(scale1);
                    let m1 = dmin * f32::from(min1);
                    let (low, high) = x64.split_at_mut(32);
                    for (value, byte) in low.iter_mut().zip(q) {
                        *value = d0 * f32::from(byte & 0x0f) - m0;
                    }
                    for (value, byte) in high.iter_mut().zip(q) {
                        *value = d1 * f32::from(byte >> 4) - m1;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (block, chunk) in blocks::<210>(bytes)
                .iter()
                .zip(out.as_chunks_mut::<256>().0)
            {
                let d = read_f16(block, 208);
                let scales = &block[192..208];
                for half in 0..2 {
                    let ql = &block[half * 64..half * 64 + 64];
                    let qh = &block[128 + half * 32..128 + half * 32 + 32];
                    let out128 = &mut chunk[half * 128..half * 128 + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let scale = |index: usize| d * f32::from(scales[half * 8 + index] as i8);
                        let q1 = ((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
                        out128[l] = scale(is) * q1 as f32;
                        out128[l + 32] = scale(is + 2) * q2 as f32;
                        out128[l + 64] = scale(is + 4) * q3 as f32;
                        out128[l + 96] = scale(is + 6) * q4 as f32;
                    }
                }
            }
        }
        other => return Err(unsupported(other)),
    }
    Ok(())
}

/// Dequantize a whole tensor, used by tests and diagnostics.
pub fn dequantize(
    ty: GgmlType,
    data: &[u8],
    cols: usize,
    rows: usize,
) -> Result<Vec<f32>, ViperaError> {
    if cols == 0 {
        return Err(ViperaError::invalid_argument(
            "cannot dequantize zero columns",
        ));
    }
    let stride = row_bytes(ty, cols)?;
    let needed = stride
        .checked_mul(rows)
        .ok_or_else(|| ViperaError::invalid_argument("dequantize byte count overflows"))?;
    if data.len() < needed {
        return Err(ViperaError::parse(
            data.len() as u64,
            format!(
                "{} bytes are too few for {rows} rows of {cols} values",
                data.len()
            ),
        ));
    }
    let total = cols
        .checked_mul(rows)
        .ok_or_else(|| ViperaError::invalid_argument("dequantize value count overflows"))?;
    let mut out = vec![0.0; total];
    for (row, chunk) in out.chunks_exact_mut(cols).enumerate() {
        read_row(ty, data, cols, row, chunk)?;
    }
    Ok(out)
}

/// Multiply a `rows x cols` matrix by `x`, writing `rows` results.
///
/// The work is blocked over the reduction: one 256-value block of activations and its precomputed
/// sub-block sums stay hot while the matching weight blocks are consumed. Rows are independent, so a
/// large product is split across threads; each row's sum order never changes, which keeps the result
/// bit-for-bit identical to a single-threaded run.
pub fn matvec(
    ty: GgmlType,
    data: &[u8],
    cols: usize,
    rows: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let stride = checked_stride(ty, data, cols, rows, x, out)?;
    // One quantization of the activation, shared by every row. Only the formats with an integer
    // kernel use it; building it for the others would be a wasted pass over `x`.
    let xq = quantizes(ty).then(|| QuantizedX::new(x));
    let sums = x_sums(x);
    let product = rows.saturating_mul(cols);
    let threads = thread_budget(rows, product);

    let activation = Activation {
        x,
        sums: &sums,
        q: xq.as_ref(),
    };

    if threads <= 1 {
        fill_rows(ty, data, stride, 0, activation, out);
        return Ok(());
    }

    let per_thread = rows.div_ceil(threads);
    thread::scope(|scope| {
        for (chunk, slice) in out.chunks_mut(per_thread).enumerate() {
            let first_row = chunk * per_thread;
            scope.spawn(move || fill_rows(ty, data, stride, first_row, activation, slice));
        }
    });
    Ok(())
}

/// Validate the shapes of a matvec call and return the bytes one row occupies.
fn checked_stride(
    ty: GgmlType,
    data: &[u8],
    cols: usize,
    rows: usize,
    x: &[f32],
    out: &[f32],
) -> Result<usize, ViperaError> {
    // `row_bytes` accepts zero columns (`0` is a whole number of blocks), but a zero-width row has
    // no stride. The 8x8 Q4_K kernel divides the tensor length by that stride, so letting this
    // through would panic there. `dequantize` refuses zero columns for the same reason.
    if cols == 0 {
        return Err(ViperaError::invalid_argument(
            "cannot multiply a matrix with zero columns",
        ));
    }
    if x.len() != cols || out.len() != rows {
        return Err(ViperaError::internal(format!(
            "matvec expected a {cols}-value input and {rows} outputs, got {} and {}",
            x.len(),
            out.len()
        )));
    }
    let stride = row_bytes(ty, cols)?;
    let needed = stride
        .checked_mul(rows)
        .ok_or_else(|| ViperaError::invalid_argument("tensor size overflows"))?;
    if data.len() < needed {
        return Err(ViperaError::parse(
            data.len() as u64,
            format!("{needed} bytes are needed for a {rows}x{cols} matrix"),
        ));
    }
    Ok(stride)
}

/// The same product, using threads that already exist.
///
/// The rows are handed to [`crate::pool::Pool`] rather than to freshly spawned threads; `data` is
/// tied to the pool's lifetime because a worker holds it for as long as the pool lives.
pub(crate) fn matvec_pooled<'d>(
    pool: &crate::pool::Pool<'d, 1>,
    ty: GgmlType,
    data: &'d [u8],
    cols: usize,
    rows: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let matrix = Matrix { ty, cols, rows };
    matvec_pooled_batch(pool, matrix, data, &[x], &Fusion::None, out)
}

/// The layout of one weight matrix: everything a product needs to know about the bytes it reads.
///
/// Grouped as a unit because a batched product would otherwise take `ty`, `cols` and `rows` beside
/// the data they describe, which is one argument past what a call to read.
#[derive(Clone, Copy)]
pub(crate) struct Matrix {
    pub(crate) ty: GgmlType,
    pub(crate) cols: usize,
    pub(crate) rows: usize,
}

/// [`matvec_pooled`] for `B` activations, reading each row once for all of them.
///
/// `out` holds `B * rows` values, token-major: entry `entry`'s `rows` results start at
/// `entry * rows`.
///
/// When `VIPERA_TILED=1` and `B >= 8` and `ty` is `F32`/`F16`, this takes the scalar tiled path ([`crate::tiled::matvec_tiled_batch`]) on dequantized weights, which changes
/// summation order and is approximate (documented parity is
/// `max_abs <= 1e-3 + 1e-4 * max|ref|`); with the gate off the result stays
/// bit-exact via the pooled kernel. `Q4_K` with `VIPERA_TILED=1` on an
/// AVX-512 VNNI CPU takes the integer tiled kernel in
/// `fill_rows_batch` (same approximate parity, activation tile shared
/// across 32 rows); `Q8_0`/`Q6_K` stay on the pooled kernel.
pub(crate) fn matvec_pooled_batch<'d, const B: usize>(
    pool: &crate::pool::Pool<'d, B>,
    matrix: Matrix,
    data: &'d [u8],
    activations: &[&[f32]; B],
    fusion: &Fusion,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let Matrix { ty, cols, rows } = matrix;
    let stride = row_bytes(ty, cols)?;
    for (entry, x) in activations.iter().enumerate() {
        if x.len() != cols {
            return Err(ViperaError::internal(format!(
                "batch entry {entry} has {} values, expected {cols}",
                x.len()
            )));
        }
    }
    if out.len() != B.saturating_mul(rows) {
        return Err(ViperaError::internal(format!(
            "a {B}-entry batch of {rows} rows needs {} outputs, got {}",
            B.saturating_mul(rows),
            out.len()
        )));
    }
    let needed = stride
        .checked_mul(rows)
        .ok_or_else(|| ViperaError::invalid_argument("tensor size overflows"))?;
    if data.len() < needed {
        return Err(ViperaError::parse(
            data.len() as u64,
            format!("{needed} bytes are needed for a {rows}x{cols} matrix"),
        ));
    }
    // The float tile is a diagnostic and stays behind `VIPERA_TILED=1`: the integer prefill
    // path below is the default, and it should not change how the few float tensors sum.
    if fusion.is_none()
        && crate::tiled::scalar_tiled_enabled()
        && crate::tiled::tiled_serves::<B>()
        && matches!(ty, GgmlType::F32 | GgmlType::F16)
    {
        return crate::tiled::matvec_tiled_batch(ty, data, stride, cols, rows, activations, out);
    }
    let threads = thread_budget(
        rows.saturating_mul(B),
        rows.saturating_mul(cols).saturating_mul(B),
    );
    // A wide Q4_K batch builds its activation tile once, in the prep workers, and every row
    // participant reads that copy instead of building its own (measured at ~44 ms of a 770 ms
    // chunk, `docs/benchmarks.md` 2026-11-22). Narrow batches and the serial path keep the old
    // per-participant build, so a decode/speculative batch is untouched.
    let block = match ty {
        GgmlType::Q4K => Some(144),
        GgmlType::Q6K => Some(210),
        _ => None,
    };
    let tiled = block
        .filter(|block| {
            B >= PARALLEL_PREP_MIN_BATCH
                && threads > 1
                && parallel_prep_enabled()
                && shared_tile_enabled()
                && stride.is_multiple_of(*block)
        })
        .map(|block| {
            let (blocks, halves) = (stride / block, B.div_ceil(16));
            if tile_reuse_enabled() {
                pool.take_tile(blocks, halves)
            } else {
                std::sync::Arc::new(crate::simd::SharedTile::new(blocks, halves))
            }
        });
    let owned: std::sync::Arc<[OwnedActivation; B]> =
        if B >= PARALLEL_PREP_MIN_BATCH && threads > 1 && parallel_prep_enabled() {
            pool.prepare(ty, activations, fusion, threads, tiled.clone())?
        } else {
            std::sync::Arc::new(std::array::from_fn(|entry| {
                let x = activations[entry];
                let mut buffer = Vec::with_capacity(x.len() + x.len() / 32);
                buffer.extend_from_slice(x);
                OwnedActivation::from_buffer_fused(ty, buffer, cols, fusion, entry)
            }))
        };
    let result = pool.fill_batch(ty, data, stride, threads, &owned, out, tiled.clone());
    if tile_reuse_enabled()
        && let Some(tile) = tiled
    {
        pool.recycle_tile(tile);
    }
    result
}

/// Whether a wide Q4_K batch builds its activation tile once in the prep workers.
///
/// On by default; `VIPERA_SHARED_TILE=0` restores the per-participant build. The two paths write
/// the same bytes, so the result is bit-identical either way.
/// Whether a product's tile buffer is taken from and returned to the pool's free list.
///
/// On by default; `VIPERA_TILE_REUSE=0` allocates a fresh tile per product, which is what the
/// one-per-product build did before the free list existed. The two are numerically identical.
fn tile_reuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VIPERA_TILE_REUSE").is_ok_and(|value| value.trim() == "0"))
}

fn shared_tile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VIPERA_SHARED_TILE").is_ok_and(|value| value.trim() == "0"))
}

/// Entries a batched product needs before its activations are built by the pool instead of by the
/// calling thread: a decode step or a short draft is one to a handful of entries and a wake-up per
/// product would cost more than the work it spreads.
const PARALLEL_PREP_MIN_BATCH: usize = 16;

/// Whether a batched product's activations are prepared by the pool's workers.
///
/// On by default: the workers build the batched activations instead of leaving them to the calling
/// thread. Set `VIPERA_PARALLEL_PREP=0` (or `false`/`off`/`no`) to fall back to the serial build;
/// the result is bit-identical either way, only who does the work changes. Read once.
fn parallel_prep_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VIPERA_PARALLEL_PREP") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => true,
    })
}

/// Multiply-adds below this stay on the calling thread: a spawn costs more than the work saves.
const PARALLEL_MIN_PRODUCT: usize = 1 << 18;

/// Threads to split one `matvec` across, capped by the rows there are to fill.
///
/// `VIPERA_THREADS` overrides the detected count. Hyper-threading reports twice the physical cores,
/// and these kernels are memory-bound enough that the sibling threads do not always pay for their
/// spawn, so being able to pin the count is worth one `std::env` read per process. The override is
/// parsed once and cached; a value of `0` or nonsense falls back to the detected count.
pub(crate) fn thread_budget(rows: usize, product: usize) -> usize {
    if product < PARALLEL_MIN_PRODUCT {
        return 1;
    }
    max_threads().min(rows.max(1))
}

/// The thread count this process was configured with: `VIPERA_THREADS` when set, else what the OS
/// reports.
///
/// Attention takes its worker count from here too, so one knob describes the whole backend.
pub(crate) fn max_threads() -> usize {
    static REQUESTED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let requested = *REQUESTED.get_or_init(|| {
        std::env::var("VIPERA_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    });
    requested
        .or_else(|| {
            thread::available_parallelism()
                .ok()
                .map(|count| count.get())
        })
        .unwrap_or(1)
}

pub(crate) fn fill_rows(
    ty: GgmlType,
    data: &[u8],
    stride: usize,
    first_row: usize,
    activation: Activation<'_>,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if ty == GgmlType::Q4K
        && !exact_dot_requested()
        && crate::simd::q4k8_enabled()
        && crate::simd::vnni512_available()
    {
        fill_rows_q4k8(data, stride, first_row, activation, out);
        return;
    }
    // VIPERA_Q6K_IQK: the repacked Q6_K GEMM (`q6k_repack_dot`), behind its switch. Rows of a
    // four-row group run through the repack; any tail row stays on the row kernel.
    #[cfg(target_arch = "x86_64")]
    if ty == GgmlType::Q6K
        && activation.q.is_some()
        && crate::simd::q6k_iqk_enabled()
        && crate::simd::vnni512_available()
    {
        let xq = activation
            .q
            .as_ref()
            .unwrap_or_else(|| panic!("Q6_K builds a quantized activation"));
        let whole = out.len() - out.len() % 4;
        if whole > 0 {
            let repack = crate::simd::pack_q6k_iqk(data, stride, first_row, whole);
            let out_rows = &mut out[..whole];
            for (offset, slot) in out_rows.iter_mut().enumerate() {
                // SAFETY: the caller checked `vnni512_available`; the repack covers
                // `first_row .. first_row + whole` and the activation covers the row.
                *slot = crate::simd::q6k_repack_dot(&repack, offset, xq);
            }
        }
        for (offset, slot) in out.iter_mut().enumerate().skip(whole) {
            let start = (first_row + offset) * stride;
            *slot = dot(ty, &data[start..start + stride], activation);
        }
        return;
    }
    for (offset, slot) in out.iter_mut().enumerate() {
        let start = (first_row + offset) * stride;
        *slot = dot(ty, &data[start..start + stride], activation);
    }
}

/// [`fill_rows`] for Q4_K with the Pha 1b 8x8 kernel (`VIPERA_Q4K8`, default off).
///
/// Eight rows per call: pack them with [`crate::simd::make_blocks_q4k8`], quantize the activation
/// to the lab `block_q8_K` format with [`crate::simd::quantize_q8k`], and dot with
/// [`crate::simd::q4k8_dot8`] — the path lab-proven faster than the row kernel on the reference
/// machine (`bench_q4k8_gemv_vs_row`, 2026-11-23).
///
/// Groups are formed on the **absolute** row index (`rows 8g..8g+8`), never on this call's own
/// offsets, and a group that is only partly inside `out` is still computed from its aligned start.
/// That is what keeps a row's value a function of the row and the activation alone: the threaded
/// `matvec` hands a worker `rows/threads` rows and the pool hands its workers a different split, so
/// grouping at offset 0 would give the same row two different answers — and the two paths are
/// required to agree (`a_pooled_matvec_matches_the_spawned_one`). Only the trailing rows of the
/// matrix, when its row count is not a multiple of eight, keep the row kernel's result.
#[cfg(target_arch = "x86_64")]
fn fill_rows_q4k8(
    data: &[u8],
    stride: usize,
    first_row: usize,
    activation: Activation<'_>,
    out: &mut [f32],
) {
    let end_row = first_row + out.len();
    let row = fill_rows_q4k8_groups(data, stride, first_row, out.len(), &[activation.x], out);
    let readable_end = end_row.min(data.len().checked_div(stride).unwrap_or(0));
    for absolute in row..readable_end {
        let start = absolute * stride;
        out[absolute - first_row] = dot(GgmlType::Q4K, &data[start..start + stride], activation);
    }
    for absolute in readable_end.max(row)..end_row {
        out[absolute - first_row] = 0.0;
    }
}

/// Compute every whole 8-row group touching `[first_row, first_row + count)` for `B` activations,
/// writing `out[entry * count + offset]`, and return the absolute row from which nothing was
/// written.
///
/// Groups are formed on the **absolute** row index (`rows 8g..8g+8`), never on this call's own
/// offsets, and a group only partly inside the range is still computed from its aligned start.
/// That is what keeps a row's value a function of the row and the activation alone: the threaded
/// `matvec` hands a worker `rows/threads` rows and the pool hands its workers a different split, so
/// grouping at offset 0 would give the same row two different answers — and the two paths are
/// required to agree (`a_pooled_matvec_matches_the_spawned_one`), as are a batch and the same
/// sequences decoded alone.
#[cfg(target_arch = "x86_64")]
fn fill_rows_q4k8_groups<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&[f32]; B],
    out: &mut [f32],
) -> usize {
    // A zero-width row has no stride and no blocks. `matvec` rejects zero columns, but a pooled
    // batch can reach here directly, so leave every output to the row-kernel tail below. Also
    // fall back when the stride is not a whole number of Q4_K blocks or does not match the
    // activation width: `quantize_q8k` and the packer assume `stride / 144 == x.len() / 256`.
    if stride == 0 || !stride.is_multiple_of(144) {
        return first_row;
    }
    let blocks = stride / 144;
    if xs.iter().any(|x| x.len() != blocks * crate::simd::QK) {
        return first_row;
    }
    // Rows this slice of `data` can read; a group is only computed whole when all eight fit.
    let readable_rows = data.len() / stride;
    let per_entry = blocks * crate::simd::Q8K_BLOCK;
    let mut act = vec![0_u8; B * per_entry];
    for (entry, x) in xs.iter().enumerate() {
        crate::simd::quantize_q8k(x, &mut act[entry * per_entry..][..per_entry]);
    }
    let mut packed: Vec<crate::simd::BlockQ4Kx8> = Vec::with_capacity(blocks);
    let mut values = [0.0_f32; 8];
    let end_row = first_row + count;
    let mut row = first_row;
    while row < end_row {
        let group_start = row / 8 * 8;
        if group_start + 8 > readable_rows {
            break;
        }
        packed.clear();
        crate::simd::make_blocks_q4k8(data, stride, group_start, blocks, &mut packed);
        let first = row.max(group_start);
        let last = end_row.min(group_start + 8);
        for entry in 0..B {
            let at = entry * per_entry;
            crate::simd::q4k8_dot8(&packed, blocks, &act[at..at + per_entry], &mut values);
            for absolute in first..last {
                out[entry * count + absolute - first_row] = values[absolute - group_start];
            }
        }
        row = group_start + 8;
    }
    row
}

/// Fill `B * rows` results, reading each row once for all `B` activations.
///
/// Rows are filled one at a time: two rows in flight was measured at 0% on this machine, because the
/// loop is bound by instruction throughput rather than by load latency at every thread count.
pub(crate) fn fill_rows_batch<const B: usize>(
    ty: GgmlType,
    data: &[u8],
    stride: usize,
    first_row: usize,
    activation: &BatchActivation<'_, B>,
    out: &mut [f32],
    tile: Option<&crate::simd::SharedTile>,
) {
    let count = out.len() / B;
    // VIPERA_Q8R16: the 16-row-interleaved Q8_0 activation repack (default off). Q8_0 is the one
    // format here that still multiplies `f32` activations — Q4_K and Q6_K already run an integer
    // activation through their row and tiled kernels — so it is the only place the repack can
    // replace a kernel. Each weight row is read once for sixteen activation rows, and `i32`
    // arithmetic replaces sixteen separate `f32` dots. The activation is quantized to one scale
    // per 32 values, so the numbers move: `VIPERA_EXACT_DOT` keeps it off too, and a chunk that is
    // not a whole number of 16-row groups at least 32 wide stays on the row kernel.
    if crate::simd::q8r16_applies_for(ty, B, crate::simd::q8r16_enabled()) && !exact_dot_requested()
    {
        crate::simd::fill_rows_q8r16(data, stride, first_row, count, &activation.x, out);
        return;
    }
    // Integer tiled path, the prefill default: on an AVX-512 VNNI CPU a Q4_K
    // batch reuses one activation tile across 32 weight rows (8-row bands
    // inside). Measured on the reference machine at B=16 it cuts a prefill
    // chunk from 360 to 229 ms. `VIPERA_TILED=0` and exact mode keep the old
    // summation order and stay on the row kernel. A batch of one keeps
    // the row kernel: its VNNI-256 path is already the faster one here (a
    // column-parallel 512-bit port measured 6.95 vs 11.87 GB/s), because the
    // sixteen-entry tile wastes fifteen lanes but the 512-bit form is
    // double-pumped on this CPU.
    // The 256-bit band, for a CPU with AVX2 and no 512-bit VNNI tile: same reuse, half the columns
    // per register. It returns false when the 512-bit tile is available, so the two never fight
    // over the same batch.
    #[cfg(target_arch = "x86_64")]
    if !exact_dot_requested()
        && crate::tiled::tiled_enabled()
        && ty == GgmlType::Q4K
        && (crate::tiled::TILE_MIN_BATCH..=128).contains(&B)
        && let Some(xs) = activation.quantized()
        && crate::simd::try_q4k_matvec_tiled_range_ymm(
            data,
            stride,
            first_row,
            count,
            xs,
            activation.sums(),
            out,
        )
    {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if !exact_dot_requested()
        && crate::tiled::tiled_enabled()
        && crate::simd::vnni512_available()
        && (crate::tiled::TILE_MIN_BATCH..=128).contains(&B)
    {
        if ty == GgmlType::Q4K {
            // W-1a: the band's activation is quantized per 256 values (`VIPERA_TILED_A`, default
            // off — this changes the engine's numbers, so flipping the default is an ADR call).
            let tiled_a = crate::tiled::tiled_a_enabled()
                .then(|| crate::quant::TiledActivation::new(*activation.x()));
            if let Some(tiled_a) = tiled_a
                && crate::simd::try_q4k_matvec_tiled_range_a::<B>(
                    data, stride, first_row, count, &tiled_a, out,
                )
            {
                return;
            }
        }
        if ty == GgmlType::Q4K
            && let Some(xs) = activation.quantized()
            && crate::simd::try_q4k_matvec_tiled_range(
                data,
                stride,
                first_row,
                count,
                xs,
                activation.sums(),
                out,
                tile,
            )
        {
            return;
        }
        if ty == GgmlType::Q6K
            && let Some(xs) = activation.quantized()
            && crate::simd::try_q6k_matvec_tiled_range(
                data, stride, first_row, count, xs, out, tile,
            )
        {
            return;
        }
    }
    // The Q4_K 8x8 kernel, for every width the tiled path above did not take. It has to be every
    // width, not just the one `matvec` takes: the pool runs a batch of one exactly like a batch of
    // three, and a sequence decoded alone must return the same bits batched
    // (`a_batched_step_matches_every_sequence_run_alone`). One pack per group serves all `B`
    // entries, so the weights are still read once.
    #[cfg(target_arch = "x86_64")]
    if ty == GgmlType::Q4K
        && !exact_dot_requested()
        && crate::simd::q4k8_enabled()
        && crate::simd::vnni512_available()
    {
        let row = fill_rows_q4k8_groups(data, stride, first_row, count, &activation.x, out);
        let readable_end = (first_row + count).min(data.len().checked_div(stride).unwrap_or(0));
        let mut values = [0.0_f32; B];
        let tail_start = row.saturating_sub(first_row);
        let tail_dot_end = readable_end.saturating_sub(first_row).min(count);
        for offset in tail_start..tail_dot_end.max(tail_start) {
            let start = (first_row + offset) * stride;
            dot_batch(ty, &data[start..start + stride], activation, &mut values);
            for (entry, value) in values.iter().enumerate() {
                out[entry * count + offset] = *value;
            }
        }
        let tail_zero_start = readable_end.max(row).saturating_sub(first_row);
        for offset in tail_zero_start..count {
            for entry in 0..B {
                out[entry * count + offset] = 0.0;
            }
        }
        return;
    }
    // No gap-blocked prefetch here: a `_MM_HINT_T0` four rows ahead made a 200 MB stream slower,
    // not faster (31.5 against 40.3 GB/s at eight threads), because the hardware streamer already
    // runs ahead and the hint evicts the activation block the kernel rereads every row.
    //
    // The small-tensor case was tried too, on the theory that 0.22 MB is 55 pages read once and
    // cannot build memory-level parallelism: rows two ahead, two lines each, gated on
    // `data.len() < 1 MB`. It measured 1.006x the q4k phase and 1.017x the q6k phase — no help, so
    // the hint stays out. A first attempt gated on the worker's chunk instead of the whole tensor
    // and cost 15% on q4k, which is the same eviction effect as the 200 MB run.
    // VIPERA_Q6K_IQK: the repacked Q6_K GEMM, per entry, through
    // [`fill_rows_q6k_iqk_batch`]; the offsets it leaves fall to the row loop below.
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if !exact_dot_requested()
        && ty == GgmlType::Q6K
        && crate::simd::q6k_iqk_enabled()
        && crate::simd::vnni512_available()
        && let Some(xs) = activation.quantized()
    {
        done = fill_rows_q6k_iqk_batch(data, stride, first_row, count, xs, out);
    }
    let mut row = [0.0_f32; B];
    for offset in done..count {
        let start = (first_row + offset) * stride;
        dot_batch(ty, &data[start..start + stride], activation, &mut row);
        for (entry, value) in row.iter().enumerate() {
            out[entry * count + offset] = *value;
        }
    }
}

/// [`fill_rows_batch`]'s IQK-repacked Q6_K path: pack the whole-row range once and dot every
/// entry against the repack. Returns the first offset the row loop still has to fill.
#[cfg(target_arch = "x86_64")]
fn fill_rows_q6k_iqk_batch<const B: usize>(
    data: &[u8],
    stride: usize,
    first_row: usize,
    count: usize,
    xs: &[&QuantizedX; B],
    out: &mut [f32],
) -> usize {
    let whole = count - count % 4;
    if whole == 0 {
        return 0;
    }
    let repack = crate::simd::pack_q6k_iqk(data, stride, first_row, whole);
    for offset in 0..whole {
        for entry in 0..B {
            // SAFETY: the caller checked `vnni512_available` and the lengths; the repack covers
            // the range and `xs[entry]` the activation of each row.
            out[entry * count + offset] = crate::simd::q6k_repack_dot(&repack, offset, xs[entry]);
        }
    }
    whole
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16(value: f32) -> [u8; 2] {
        f32_to_f16(value).to_le_bytes()
    }

    #[test]
    fn half_precision_round_trips_the_values_a_kv_cache_holds() {
        for value in [0.0, 1.0, -1.0, 0.5, -2.5, 0.125, 1.5] {
            let bits = f32_to_f16(value);
            assert_eq!(f16_to_f32(bits), value, "{value} did not round trip");
        }
        assert_eq!(f16_to_f32(f32_to_f16(1.000_1)), 1.0);
    }

    #[test]
    fn the_q4k_f32_switch_takes_q4k_off_the_i8_activation_path() {
        assert!(
            quantizes_for(GgmlType::Q4K, false, false),
            "the default multiplies i8 activations"
        );
        assert!(
            !quantizes_for(GgmlType::Q4K, true, false),
            "VIPERA_Q4K_F32 restores the f32 dot for Q4_K"
        );
        assert!(
            quantizes_for(GgmlType::Q6K, false, false),
            "Q6_K keeps its q8 dot by default"
        );
        assert!(
            !quantizes_for(GgmlType::Q6K, false, true),
            "VIPERA_EXACT_DOT restores the f32 dot for Q6_K"
        );
        assert!(
            !quantizes_for(GgmlType::F32, true, true),
            "formats with no quantized dot never build one"
        );
    }

    #[test]
    fn only_an_explicit_nonzero_switch_value_turns_it_on() {
        for on in ["1", "true", "on"] {
            assert!(env_truthy(Some(on)), "{on:?} turns a switch on");
        }
        for off in ["", "0", " 0 "] {
            assert!(!env_truthy(Some(off)), "{off:?} leaves a switch off");
        }
        assert!(!env_truthy(None), "unset leaves a switch off");
    }

    #[test]
    fn half_precision_covers_subnormals_and_the_special_values() {
        // The small scales of a real file land in the subnormal range, where the bit manipulation
        // has to agree with the arithmetic definition `mantissa * 2^-24`.
        for mantissa in [1_u16, 2, 512, 1023] {
            let expected = f64::from(mantissa) * 2f64.powi(-24);
            let actual = f64::from(f16_to_f32(mantissa));
            assert!(
                (actual - expected).abs() <= expected * 1e-6,
                "subnormal {mantissa}: {actual} vs {expected}"
            );
            assert_eq!(f16_to_f32(mantissa | 0x8000), -f16_to_f32(mantissa));
        }
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0, "the largest finite half");
    }

    #[test]
    fn q8_0_scales_every_value_by_its_block_scale() {
        let mut block = Vec::new();
        block.extend_from_slice(&f16(0.5));
        block.extend((0..32).map(|value| value as i8 as u8));
        let mut out = vec![0.0; 32];
        read_row(GgmlType::Q8_0, &block, 32, 0, &mut out).expect("row");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 1.5);
        assert_eq!(out[31], 15.5);
    }

    #[test]
    fn q4_k_applies_both_scale_and_minimum() {
        // d = 1, dmin = 0, every scale 1, every minimum 0; nibbles are 1 (low) and 2 (high).
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16(1.0));
        block[2..4].copy_from_slice(&f16(0.0));
        block[4..8].fill(1); // scales 0..3
        block[8..12].fill(0); // minimums 0..3, and the high bits of 4..7
        for byte in &mut block[12..16] {
            *byte = 0x01; // scale 4..7 = 1, minimum 4..7 = 0
        }
        block[16..144].fill(0x21);
        let mut out = vec![0.0; 256];
        read_row(GgmlType::Q4K, &block, 256, 0, &mut out).expect("row");
        assert_eq!(&out[0..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&out[32..36], &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(&out[64..68], &[1.0, 1.0, 1.0, 1.0]);
        let ones = vec![1.0_f32; 256];
        assert_eq!(q4k_dot(&block, &ones, &x_sums(&ones)), 384.0);
    }

    #[test]
    fn q4_k_minimum_is_subtracted() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16(1.0));
        block[2..4].copy_from_slice(&f16(1.0)); // dmin = 1
        block[4..8].fill(1);
        block[8..12].fill(0);
        for byte in &mut block[12..16] {
            *byte = 0x02; // scale 4..7 = 2, minimum 4..7 = 0
        }
        let mut out = vec![0.0; 256];
        read_row(GgmlType::Q4K, &block, 256, 0, &mut out).expect("row");
        // With minimum 0 the values are the scaled nibbles, which are zero here.
        assert!(out.iter().all(|value| *value == 0.0));
        block[8..12].fill(2); // minimums 0..3 = 2
        read_row(GgmlType::Q4K, &block, 256, 0, &mut out).expect("row");
        assert_eq!(out[0], -2.0);
        assert_eq!(out[255], 0.0, "sub-block 7 keeps its own minimum");
    }

    #[test]
    fn q6_k_centres_six_bit_values_on_thirty_two() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16(1.0));
        for byte in &mut block[192..208] {
            *byte = 1;
        }
        let mut out = vec![0.0; 256];
        read_row(GgmlType::Q6K, &block, 256, 0, &mut out).expect("row");
        assert!(out.iter().all(|value| *value == -32.0));
        let ones = vec![1.0_f32; 256];
        assert_eq!(q6k_dot(&block, &ones, &x_sums(&ones)), -8192.0);
    }

    #[test]
    fn q6_k_reads_all_four_value_groups() {
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&f16(1.0));
        block[192..208].fill(1);
        // qh[0] carries the high bits of all four groups for the first column.
        block[128] = 0b11_11_11_11;
        block[0] = 0x0f; // ql low nibble for group 1
        block[32] = 0x0f; // ql low nibble for group 2
        let mut out = vec![0.0; 256];
        read_row(GgmlType::Q6K, &block, 256, 0, &mut out).expect("row");
        // Each group is (nibble | high bits) - 32, with the high bits set to 3 in all four groups.
        assert_eq!(out[0], 31.0, "low nibble 15 plus 48, centred");
        assert_eq!(out[32], 31.0);
        assert_eq!(out[64], 16.0, "group 3 uses the high nibble of ql[0]");
        assert_eq!(out[96], 16.0);
    }

    /// A fused activation has to be the plain build of the transformed values, bit for bit: the
    /// transform moves to a worker, not the arithmetic.
    #[test]
    fn a_fused_activation_matches_the_transform_built_serially() {
        let width = 64;
        let weight: Vec<f32> = (0..width)
            .map(|at| ((at % 13) as f32 - 6.0) / 7.0)
            .collect();
        let gate_values: Vec<f32> = (0..2 * width)
            .map(|at| ((at * 37 % 41) as f32 - 20.0) / 11.0)
            .collect();
        for entry in 0..2 {
            let x: Vec<f32> = (0..width)
                .map(|at| ((at * 7919 % 61) as f32 - 30.0) / 13.0)
                .collect();
            for fusion in [
                Fusion::Norm {
                    weight: weight.clone().into(),
                    epsilon: 1e-6,
                },
                Fusion::Silu {
                    gate: gate_values.clone().into(),
                    width,
                },
            ] {
                let mut buffer = Vec::with_capacity(x.len() + x.len() / 32);
                buffer.extend_from_slice(&x);
                let fused = OwnedActivation::from_buffer_fused(
                    GgmlType::Q4K,
                    buffer,
                    width,
                    &fusion,
                    entry,
                );
                let mut expected = x.clone();
                match &fusion {
                    Fusion::Norm { weight, epsilon } => {
                        crate::model::rms_norm_in_place(&mut expected, weight, *epsilon);
                    }
                    Fusion::Silu { gate, width } => {
                        let gate = &gate[entry * width..(entry + 1) * width];
                        crate::simd::silu_mul(&mut expected, gate);
                    }
                    Fusion::None => unreachable!(),
                }
                let plain = OwnedActivation::new(GgmlType::Q4K, &expected);
                assert_eq!(fused.x(), plain.x(), "entry {entry} values");
                assert_eq!(fused.sums(), plain.sums(), "entry {entry} sums");
                assert_eq!(
                    fused.quantized().map(|q| &q.q),
                    plain.quantized().map(|q| &q.q),
                    "entry {entry} quantized activations"
                );
            }
        }
    }

    #[test]
    fn quantizing_an_activation_keeps_each_value_within_half_a_step() {
        let mut x: Vec<f32> = (0..256).map(|at| (at as f32 - 128.0) / 37.0).collect();
        let quantized = QuantizedX::new(&x);
        assert_eq!(quantized.d.len(), 8);
        for (at, (original, restored)) in x.iter().zip(quantized.dequantized()).enumerate() {
            let step = quantized.d[at / 32];
            assert!(
                (original - restored).abs() <= step * 0.5 + 1e-6,
                "value {at}: {original} became {restored} with step {step}"
            );
        }
        // A group of zeros keeps zero rather than dividing by a zero scale.
        x[32..64].fill(0.0);
        let zeros = QuantizedX::new(&x);
        assert_eq!(zeros.d[1], 0.0);
        assert!(zeros.q[32..64].iter().all(|value| *value == 0));
    }

    /// What one pooled product costs the caller, from handing the work out to the last reply being
    /// in, for several amounts of other work between two products.
    ///
    /// A decode step issues one pooled product per weight matrix, 196 of them for this model, with
    /// rope, norms and attention between them. The gap column stands in for that work: it is spent
    /// on the calling thread, so the workers are idle for as long as it lasts and the next product
    /// pays to wake them.
    ///
    /// This replaces a version that timed an 8-row product through `matvec_pooled`. `thread_budget`
    /// gives a product that small one thread, so that version never woke a worker: what it timed
    /// was building the activation, which is a real cost but not the cost of a dispatch. Here the
    /// thread count goes straight to `fill_batch`, and the activation is built outside the timing.
    ///
    /// Ignored by default. Run it with
    /// `cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture pool_dispatch_cost`.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn pool_dispatch_cost() {
        use std::time::{Duration, Instant};
        let blocks = 6;
        let rows = 64;
        let stride = blocks * 144;
        // 144 bytes per superblock; the contents do not matter for a dispatch measurement.
        let data = vec![0_u8; rows * stride];
        let x = vec![0.01_f32; blocks * 256];
        let activations = std::sync::Arc::new([OwnedActivation::new(GgmlType::Q4K, &x)]);
        let mut out = vec![0.0_f32; rows];
        let participants = max_threads().min(8);
        println!("{participants} participants");
        std::thread::scope(|scope| {
            let pool = crate::pool::Pool::<1>::new(scope, participants);
            for gap_us in [0_u64, 30, 300] {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let mut inside = Duration::ZERO;
                    for _ in 0..200 {
                        let until = Instant::now() + Duration::from_micros(gap_us);
                        while Instant::now() < until {
                            std::hint::spin_loop();
                        }
                        let started = Instant::now();
                        pool.fill_batch(
                            GgmlType::Q4K,
                            &data,
                            stride,
                            participants,
                            &activations,
                            &mut out,
                            None,
                        )
                        .expect("dispatch");
                        inside += started.elapsed();
                    }
                    best = best.min(inside.as_secs_f64() / 200.0);
                }
                println!(
                    "{gap_us:>3} us between products: {:>6.1} us per product",
                    best * 1e6
                );
            }
        });
    }

    /// Whether a 512-bit VNNI Q4_K kernel is worth having for a decode batch of one.
    ///
    /// The comment on `fill_rows_batch` says a 512-bit port was already measured and lost at B=1
    /// (6.95 against 11.87 GB/s, because this CPU double-pumps AVX-512 and a sixteen-entry tile
    /// wastes fifteen lanes for one row). This re-measures it on the kernel that exists, at the shape
    /// a decode step actually runs, so the claim is checked rather than inherited.
    ///
    /// Ignored by default:
    /// `cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture a_512_bit_decode_row`
    // The kernel it measures is the AVX-512 VNNI one, so the test only exists where that module
    // does; on aarch64 the row kernel it would compare against is the only kernel there is.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn a_512_bit_decode_row_is_measured_against_the_row_kernel() {
        use std::time::Instant;
        if !crate::simd::vnni512_available() {
            eprintln!("skipped: no AVX-512 VNNI");
            return;
        }
        let cols = 1536usize;
        let rows = 8960usize;
        let stride = row_bytes(GgmlType::Q4K, cols).expect("stride");
        let mut state = 0x9E37_79B9_u32;
        let mut byte = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let data: Vec<u8> = (0..stride * rows).map(|_| byte()).collect();
        let x: Vec<f32> = (0..cols).map(|at| (at % 97) as f32 / 97.0 - 0.5).collect();
        let xq = QuantizedX::new(&x);
        let sums = x_sums(&x);
        let mut row_out = vec![0.0_f32; rows];
        let mut tile_out = vec![0.0_f32; rows];

        let run_row = || {
            let mut out = vec![0.0_f32; rows];
            fill_rows(
                GgmlType::Q4K,
                &data,
                stride,
                0,
                Activation {
                    x: &x,
                    sums: &sums,
                    q: Some(&xq),
                },
                &mut out,
            );
            out
        };
        let run_tile = || {
            let mut out = vec![0.0_f32; rows];
            let xs = [&xq];
            let sums = [sums.as_slice()];
            assert!(crate::simd::try_q4k_matvec_tiled_range::<1>(
                &data, stride, 0, rows, &xs, &sums, &mut out, None
            ));
            out
        };

        let mut best = [f64::MAX; 2];
        for _ in 0..4 {
            let start = Instant::now();
            row_out = run_row();
            best[0] = best[0].min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            tile_out = run_tile();
            best[1] = best[1].min(start.elapsed().as_secs_f64());
        }
        let megabytes = (stride * rows) as f64 / 1e6;
        let worst = row_out
            .iter()
            .zip(&tile_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "q4k {}x{}: row kernel {:.2} GB/s, 512-bit tile {:.2} GB/s ({:+.0}%), max |delta| {worst:.3e}",
            rows,
            cols,
            megabytes / 1e3 / best[0],
            megabytes / 1e3 / best[1],
            100.0 * (best[1] - best[0]) / best[0],
        );
    }

    #[test]
    fn matvec_matches_a_hand_computed_product() {
        let mut data = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let mut out = vec![0.0; 2];
        matvec(GgmlType::F32, &data, 3, 2, &[1.0, 1.0, 1.0], &mut out).expect("matvec");
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn a_pooled_matvec_matches_the_spawned_one() {
        // Four chunks over eight rows, so every reply has to land in the right slice.
        const ROWS: usize = 8;
        let stride = row_bytes(GgmlType::Q4K, 512).expect("stride");
        let mut data = vec![0u8; stride * ROWS];
        for (at, byte) in data.iter_mut().enumerate() {
            *byte = (at.wrapping_mul(37) % 251) as u8;
        }
        let x: Vec<f32> = (0..512).map(|at| ((at % 13) as f32 - 6.0) / 7.0).collect();

        let mut spawned = vec![0.0_f32; ROWS];
        matvec(GgmlType::Q4K, &data, 512, ROWS, &x, &mut spawned).expect("matvec");

        thread::scope(|scope| {
            let pool = crate::pool::Pool::new(scope, 4);
            let mut pooled = vec![0.0_f32; ROWS];
            matvec_pooled(&pool, GgmlType::Q4K, &data, 512, ROWS, &x, &mut pooled)
                .expect("pooled matvec");
            assert_eq!(pooled, spawned, "a reused worker must give the same rows");
        });
    }

    #[test]
    fn a_short_row_is_refused_instead_of_padded() {
        let error = matvec(
            GgmlType::Q4K,
            &[0u8; 144],
            512,
            1,
            &[0.0; 512],
            &mut [0.0; 1],
        )
        .expect_err("a 144-byte row cannot hold 512 values");
        assert!(error.to_string().contains("288"), "{error}");
    }

    #[test]
    fn unknown_formats_are_named_in_the_error() {
        let error = row_bytes(GgmlType::Bf16, 32).expect_err("bf16 is not implemented");
        assert!(error.to_string().contains("BF16"), "{error}");
    }

    /// Measurement, not a correctness check: the same tensors, read by the spawn-per-call path and
    /// then by the pool a forward pass reuses, at the shapes the Qwen2.5-1.5B file uses. The
    /// difference is what the thread startup cost, so the pooled column is the kernel's own
    /// bandwidth. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn pooled_matvec_throughput_by_tensor_type() {
        fn measure(label: &str, ty: GgmlType, cols: usize, rows: usize, iterations: u32) {
            let stride = row_bytes(ty, cols).expect("stride");
            // Deterministic bytes, so two runs measure the same data; all zeroes would let the FP
            // units cut corners.
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            let mut byte = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            };
            let data: Vec<u8> = (0..stride * rows).map(|_| byte()).collect();
            let x: Vec<f32> = (0..cols).map(|at| (at % 17) as f32 / 17.0 - 0.5).collect();
            let mut out = vec![0.0_f32; rows];

            // One warm-up call per path: the first pass also faults in the allocation.
            matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes are consistent");
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes are consistent");
            }
            let spawned = started.elapsed().as_secs_f64();

            thread::scope(|scope| {
                let pool = crate::pool::Pool::new(scope, thread_budget(rows, rows * cols));
                matvec_pooled(&pool, ty, &data, cols, rows, &x, &mut out)
                    .expect("shapes are consistent");
                let started = std::time::Instant::now();
                for _ in 0..iterations {
                    matvec_pooled(&pool, ty, &data, cols, rows, &x, &mut out)
                        .expect("shapes are consistent");
                }
                let pooled = started.elapsed().as_secs_f64();

                let calls = f64::from(iterations);
                let megabytes = (stride * rows) as f64 / 1e6;
                println!(
                    "{label:>18}: {rows}x{cols} {ty:?}, {megabytes:>6.1} MB per call, \
                     spawned {:>6.1} ms/call {:>5.2} GB/s, pooled {:>6.1} ms/call {:>5.2} GB/s",
                    spawned * 1000.0 / calls,
                    megabytes / 1e3 * calls / spawned,
                    pooled * 1000.0 / calls,
                    megabytes / 1e3 * calls / pooled,
                );
            });
        }

        measure("wq (Q4_K)", GgmlType::Q4K, 1536, 1536, 200);
        measure("ffn_gate (Q4_K)", GgmlType::Q4K, 1536, 8960, 60);
        measure("ffn_down (Q6_K)", GgmlType::Q6K, 8960, 1536, 60);
        measure("output (Q6_K)", GgmlType::Q6K, 1536, 151_936, 8);
    }

    /// Measurement, not a correctness check: where the 39 us of a 0.22 MB call goes.
    ///
    /// The same pooled call is run twice: over 280 distinct copies of the 256x1536 shape (cold, the
    /// way a decode step reads it) and over one copy 280 times (hot — 0.22 MB stays in the 8 MB L3).
    /// Cold minus hot is the DRAM share; hot is the dispatch-and-wake floor. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture small_shape
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn small_shape_call_cost_hot_and_cold() {
        fn run(rows: usize, cols: usize, windows: usize, mode: &str) {
            let stride = row_bytes(GgmlType::Q4K, cols).expect("stride");
            let per_window = rows * stride;
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            let data: Vec<u8> = (0..per_window * windows)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 24) as u8
                })
                .collect();
            let x: Vec<f32> = (0..cols).map(|at| (at % 17) as f32 / 17.0 - 0.5).collect();
            let mut out = vec![0.0_f32; rows];
            let threads = thread_budget(rows, rows * cols);
            thread::scope(|scope| {
                let pool = crate::pool::Pool::new(scope, threads);
                // Warm-up: faults the pages in and starts the workers.
                matvec_pooled(
                    &pool,
                    GgmlType::Q4K,
                    &data[..per_window],
                    cols,
                    rows,
                    &x,
                    &mut out,
                )
                .expect("shapes are consistent");
                let started = std::time::Instant::now();
                for window in data.chunks_exact(per_window) {
                    let slice = if mode == "hot" {
                        &data[..per_window]
                    } else {
                        window
                    };
                    matvec_pooled(&pool, GgmlType::Q4K, slice, cols, rows, &x, &mut out)
                        .expect("shapes are consistent");
                }
                let elapsed = started.elapsed().as_secs_f64();
                println!(
                    "{rows}x{cols} {mode:>4}: {:>6.2} us/call, {:>5.2} GB/s (threads={threads}, {} MB of data)",
                    elapsed * 1e6 / windows as f64,
                    (per_window * windows) as f64 / elapsed / 1e9,
                    data.len() >> 20
                );
            });
        }

        for mode in ["hot", "cold"] {
            run(256, 1536, 280, mode);
        }
        // A call with almost no work at all: the dispatch-and-wake floor.
        run(4, 1536, 280, "hot");
    }

    /// Measurement, not a correctness check: Q6_K at the two shapes a decode step calls it with.
    ///
    /// `wd` is 1536x8960 read once per layer (355 MB a token, 28 calls) and measured 28.7 GB/s in
    /// the step; `output` is 151936x1536 in one call and measured 37.1. This runs both through the
    /// pool on a flat buffer, so the difference is the shape and the call size rather than where the
    /// bytes live. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture q6k_shape
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn q6k_shape_call_throughput() {
        fn run(rows: usize, cols: usize, windows: usize, label: &str) {
            let stride = row_bytes(GgmlType::Q6K, cols).expect("stride");
            let per_window = rows * stride;
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let data: Vec<u8> = (0..per_window * windows)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 24) as u8
                })
                .collect();
            let x: Vec<f32> = (0..cols).map(|at| (at % 17) as f32 / 17.0 - 0.5).collect();
            let mut out = vec![0.0_f32; rows];
            let threads = thread_budget(rows, rows * cols);
            thread::scope(|scope| {
                let pool = crate::pool::Pool::new(scope, threads);
                matvec_pooled(
                    &pool,
                    GgmlType::Q6K,
                    &data[..per_window],
                    cols,
                    rows,
                    &x,
                    &mut out,
                )
                .expect("shapes are consistent");
                let started = std::time::Instant::now();
                for window in data.chunks_exact(per_window) {
                    matvec_pooled(&pool, GgmlType::Q6K, window, cols, rows, &x, &mut out)
                        .expect("shapes are consistent");
                }
                let elapsed = started.elapsed().as_secs_f64();
                println!(
                    "{label:>16} {rows}x{cols}: {:>6.2} ms/call, {:>5.2} GB/s (threads={threads}, {} MB)",
                    elapsed * 1e3 / windows as f64,
                    (per_window * windows) as f64 / elapsed / 1e9,
                    data.len() >> 20
                );
            });
        }

        run(1536, 8960, 28, "wd (28 calls)");
        run(151_936, 1536, 1, "output (1 call)");
    }

    /// Measurement, not a correctness check: what one weight read shared by `B` activations buys at
    /// the kernel level. Each line is a batch of `B` entries, its time, and the throughput relative
    /// to running the same `B` entries one at a time; the design note's gate is a factor of two at
    /// `B = 4`. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn pooled_matvec_batch_throughput() {
        /// Seconds per call for a batch of `B` activations.
        fn per_call<const B: usize>(
            ty: GgmlType,
            data: &[u8],
            cols: usize,
            rows: usize,
            activations: &[&[f32]; B],
            iterations: u32,
        ) -> f64 {
            let mut out = vec![0.0_f32; B * rows];
            thread::scope(|scope| {
                let pool = crate::pool::Pool::new(scope, thread_budget(rows, rows * cols * B));
                // One warm-up call: the first pass also faults in the allocation.
                let matrix = Matrix { ty, cols, rows };
                matvec_pooled_batch(&pool, matrix, data, activations, &Fusion::None, &mut out)
                    .expect("shapes are consistent");
                let started = std::time::Instant::now();
                for _ in 0..iterations {
                    matvec_pooled_batch(&pool, matrix, data, activations, &Fusion::None, &mut out)
                        .expect("shapes are consistent");
                }
                started.elapsed().as_secs_f64() / f64::from(iterations)
            })
        }

        /// The fastest of three rounds: a single round on a shared machine measures the scheduler
        /// as much as the kernel.
        fn best<const B: usize>(
            ty: GgmlType,
            data: &[u8],
            cols: usize,
            rows: usize,
            activations: &[&[f32]; B],
            iterations: u32,
        ) -> f64 {
            (0..3).fold(f64::INFINITY, |best, _| {
                best.min(per_call::<B>(ty, data, cols, rows, activations, iterations))
            })
        }

        fn measure(label: &str, ty: GgmlType, cols: usize, rows: usize, iterations: u32) {
            let stride = row_bytes(ty, cols).expect("stride");
            // Deterministic bytes, so two runs measure the same data; all zeroes would let the FP
            // units cut corners.
            let mut state = 0x2545_f491_4f6c_dd1d_u64;
            let mut byte = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            };
            let data: Vec<u8> = (0..stride * rows).map(|_| byte()).collect();
            // One activation per entry, each a different vector so nothing is cached between them.
            let xs: Vec<Vec<f32>> = (0..8)
                .map(|entry| {
                    (0..cols)
                        .map(|at| ((at + entry * 7) % 17) as f32 / 17.0 - 0.5)
                        .collect()
                })
                .collect();
            let rows_of = |entry: usize| xs[entry].as_slice();
            let one = [rows_of(0)];
            let two = [rows_of(0), rows_of(1)];
            let four = [rows_of(0), rows_of(1), rows_of(2), rows_of(3)];
            let eight = [
                rows_of(0),
                rows_of(1),
                rows_of(2),
                rows_of(3),
                rows_of(4),
                rows_of(5),
                rows_of(6),
                rows_of(7),
            ];

            let single = best::<1>(ty, &data, cols, rows, &one, iterations);
            let measured: [(usize, f64); 3] = [
                (2, best::<2>(ty, &data, cols, rows, &two, iterations)),
                (4, best::<4>(ty, &data, cols, rows, &four, iterations)),
                (8, best::<8>(ty, &data, cols, rows, &eight, iterations)),
            ];
            let megabytes = (stride * rows) as f64 / 1e6;
            let mut line = format!(
                "{label:>18}: {rows}x{cols} {ty:?}, {megabytes:>6.1} MB per read, \
                 B=1 {:>5.2} ms ({:>5.2} GB/s)",
                single * 1000.0,
                megabytes / 1e3 / single,
            );
            for (size, seconds) in measured {
                line.push_str(&format!(
                    ", B={size} {:>5.2} ms ({:>5.2}x)",
                    seconds * 1000.0,
                    size as f64 * single / seconds,
                ));
            }
            println!("{line}");
        }

        measure("wq (Q4_K)", GgmlType::Q4K, 1536, 1536, 100);
        measure("ffn_gate (Q4_K)", GgmlType::Q4K, 1536, 8960, 20);
        measure("ffn_down (Q6_K)", GgmlType::Q6K, 8960, 1536, 20);
        measure("output (Q6_K)", GgmlType::Q6K, 1536, 151_936, 4);
    }

    #[test]
    fn huge_cols_is_an_error_not_a_panic() {
        assert!(row_bytes(GgmlType::F32, usize::MAX / 2).is_err());
    }

    #[test]
    fn huge_row_is_an_error_not_a_panic() {
        let mut out = vec![0.0; 3];
        assert!(read_row(GgmlType::F32, &[], 3, usize::MAX / 12, &mut out).is_err());
    }

    #[test]
    fn zero_cols_is_an_error_not_a_panic() {
        assert!(dequantize(GgmlType::F32, &[], 0, 1).is_err());
        assert!(dequantize(GgmlType::F32, &[], 0, 0).is_err());
        assert!(dequantize(GgmlType::F32, &[], 3, usize::MAX).is_err());
    }

    #[test]
    fn dequantize_fills_every_row() {
        let mut data = Vec::new();
        for value in 0..6 {
            data.extend_from_slice(&(value as f32).to_le_bytes());
        }
        let values = dequantize(GgmlType::F32, &data, 3, 2).expect("dequantize");
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// What building the owned activations of one batched product costs on the calling thread, per
    /// part, at the two widths a prefill chunk multiplies (1536 and 8960) with 128 entries. It is a
    /// measurement, not a check, so it is `#[ignore]`d.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_activation_prep() {
        let mut seed = 0x1234_5678_9abc_def1_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for cols in [1536_usize, 8960] {
            let xs: Vec<Vec<f32>> = (0..128)
                .map(|_| (0..cols).map(|_| next()).collect())
                .collect();
            let time = |label: &str, run: &dyn Fn()| {
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let started = std::time::Instant::now();
                    run();
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                println!("cols {cols:>5}, 128 entries, {label:<22} {best:>7.3} ms");
            };
            time("OwnedActivation (Q4K)", &|| {
                for x in &xs {
                    std::hint::black_box(OwnedActivation::new(GgmlType::Q4K, x));
                }
            });
            time("  x_sums alone", &|| {
                for x in &xs {
                    std::hint::black_box(x_sums(x));
                }
            });
            time("  QuantizedX alone", &|| {
                for x in &xs {
                    std::hint::black_box(QuantizedX::new(x));
                }
            });
            time("  copy alone", &|| {
                for x in &xs {
                    std::hint::black_box(x.to_vec());
                }
            });
        }
    }

    /// The Pha 1b 8x8 kernel (`simd::q4k8_dot8`, wired into `fill_rows_q4k8`) must agree with the
    /// lab's original kernel over the same packed weights and the same activation: the port is
    /// correct when the wired kernel matches `q4k8_lab::gemv_q4k8_generic` within the repack
    /// prototype's tolerance (`packed_dot_matches_scalar_q4k_within_tolerance`).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k8_wire_matches_lab_kernel_within_tolerance() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn byte(&mut self) -> u8 {
                (self.next() >> 24) as u8
            }
            fn unit(&mut self) -> f32 {
                (self.next() % 10_000) as f32 / 5_000.0 - 1.0
            }
            fn half(&mut self, low: f32, high: f32) -> [u8; 2] {
                let v = low + (high - low) * ((self.next() % 1_000) as f32 / 1_000.0);
                f32_to_f16(v).to_le_bytes()
            }
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

        let mut rng = Rng(0x9a7b_0001);
        let cols = 512;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_row(&mut rng, cols / 256)).collect();
        let data: Vec<u8> = rows.concat();
        let stride = row_bytes(GgmlType::Q4K, cols).unwrap();
        let blocks = stride / 144;
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let mut act = vec![0_u8; blocks * crate::simd::Q8K_BLOCK];
        crate::simd::quantize_q8k(&x, &mut act);

        // Reference: the lab's original kernel over the lab's own pack of the same rows.
        let packed_lab = crate::q4k8_lab::pack_q4k8(&data, 8, blocks);
        let mut out_ref = vec![0.0_f32; 8];
        crate::q4k8_lab::gemv_q4k8_generic(&packed_lab, blocks, &act, &mut out_ref);

        // Wired: the simd port of the same kernel over the engine's packer.
        let mut packed: Vec<crate::simd::BlockQ4Kx8> = Vec::new();
        crate::simd::make_blocks_q4k8(&data, stride, 0, blocks, &mut packed);
        let mut out8 = [0.0_f32; 8];
        crate::simd::q4k8_dot8(&packed, blocks, &act, &mut out8);

        for (i, (got, want)) in out8.iter().zip(out_ref.iter()).enumerate() {
            let tol = 1e-3 + 1e-4 * want.abs();
            assert!(
                (got - want).abs() <= tol,
                "row {i}: 8x8 {got} vs lab generic {want}"
            );
        }
    }

    /// A row's value must not depend on where a thread split put it: the spawned path and the pool
    /// cut the same tensor differently, so an 8-row group formed from this call's own offsets gives
    /// one row two answers. Twenty rows (not a multiple of eight) also keeps the trailing rows on
    /// the row kernel in every split.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k_rows_do_not_depend_on_the_call_split() {
        if !crate::simd::vnni512_available() || !crate::simd::q4k8_enabled() {
            return;
        }
        const ROWS: usize = 20;
        const COLS: usize = 512;
        let stride = row_bytes(GgmlType::Q4K, COLS).expect("stride");
        let blocks = stride / 144;
        let mut data = vec![0u8; stride * ROWS];
        for row in 0..ROWS {
            for block in 0..blocks {
                let at = row * stride + block * 144;
                // Finite half scales, so a mismatch cannot hide behind a NaN.
                data[at..at + 2].copy_from_slice(&f16(0.02 + row as f32 * 0.001));
                data[at + 2..at + 4].copy_from_slice(&f16(0.01));
                for i in 0..12 {
                    data[at + 4 + i] = ((row * 7 + block * 3 + i) % 251) as u8;
                }
                for i in 0..128 {
                    data[at + 16 + i] = ((row * 13 + block * 5 + i * 3) % 251) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..COLS).map(|at| ((at % 13) as f32 - 6.0) / 7.0).collect();
        let xq = QuantizedX::new(&x);
        let sums = x_sums(&x);
        let activation = Activation {
            x: &x,
            sums: &sums,
            q: Some(&xq),
        };

        let mut whole = vec![0.0_f32; ROWS];
        fill_rows(GgmlType::Q4K, &data, stride, 0, activation, &mut whole);

        let mut split = vec![0.0_f32; ROWS];
        for (first_row, end_row) in [(0, 3), (3, 11), (11, 20)] {
            fill_rows(
                GgmlType::Q4K,
                &data,
                stride,
                first_row,
                activation,
                &mut split[first_row..end_row],
            );
        }
        assert_eq!(
            split, whole,
            "a row must not change when the split moves its group boundary"
        );
    }

    /// A pooled batch shares one 8-row group fill across entries, and a row's value must not depend
    /// on whether it landed in a group or in the tail: the same rows through a B=2 pooled batch
    /// must match each activation run alone, including a row count that is not a multiple of eight.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k8_pooled_batch_matches_single_runs_across_group_boundaries() {
        if !crate::simd::vnni512_available() || !crate::simd::q4k8_enabled() {
            return;
        }
        const ROWS: usize = 20;
        const COLS: usize = 512;
        let stride = row_bytes(GgmlType::Q4K, COLS).expect("stride");
        let blocks = stride / 144;
        let mut data = vec![0u8; stride * ROWS];
        for row in 0..ROWS {
            for block in 0..blocks {
                let at = row * stride + block * 144;
                data[at..at + 2].copy_from_slice(&f16(0.02 + row as f32 * 0.001));
                data[at + 2..at + 4].copy_from_slice(&f16(0.01));
                for i in 0..12 {
                    data[at + 4 + i] = ((row * 7 + block * 3 + i) % 251) as u8;
                }
                for i in 0..128 {
                    data[at + 16 + i] = ((row * 13 + block * 5 + i * 3) % 251) as u8;
                }
            }
        }
        let x0: Vec<f32> = (0..COLS).map(|at| ((at % 13) as f32 - 6.0) / 7.0).collect();
        let x1: Vec<f32> = (0..COLS).map(|at| ((at % 17) as f32 - 8.0) / 9.0).collect();

        let mut single0 = vec![0.0_f32; ROWS];
        let mut single1 = vec![0.0_f32; ROWS];
        matvec(GgmlType::Q4K, &data, COLS, ROWS, &x0, &mut single0).expect("single 0");
        matvec(GgmlType::Q4K, &data, COLS, ROWS, &x1, &mut single1).expect("single 1");

        thread::scope(|scope| {
            let pool = crate::pool::Pool::new(scope, 4);
            let mut batched = vec![0.0_f32; 2 * ROWS];
            matvec_pooled_batch(
                &pool,
                Matrix {
                    ty: GgmlType::Q4K,
                    cols: COLS,
                    rows: ROWS,
                },
                &data,
                &[&x0, &x1],
                &Fusion::None,
                &mut batched,
            )
            .expect("pooled batch");
            assert_eq!(
                &batched[..ROWS],
                single0.as_slice(),
                "batch entry 0 must match its single run"
            );
            assert_eq!(
                &batched[ROWS..],
                single1.as_slice(),
                "batch entry 1 must match its single run"
            );
        });
    }

    /// An unusual stride (not a whole number of Q4_K blocks) or an activation width that does not
    /// match the stride cannot be consumed by the 8x8 group fill; it must fall back to the row
    /// kernel instead of dividing or indexing out of bounds.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k8_group_fill_rejects_unusual_stride_without_oob() {
        const COLS: usize = 256;
        let stride = row_bytes(GgmlType::Q4K, COLS).expect("stride");
        assert_eq!(stride, 144);
        let mut data = vec![0u8; stride * 8];
        for row in 0..8 {
            let at = row * stride;
            data[at..at + 2].copy_from_slice(&f16(0.03));
            data[at + 2..at + 4].copy_from_slice(&f16(0.0));
            for i in 0..12 {
                data[at + 4 + i] = 1;
            }
            for i in 0..128 {
                data[at + 16 + i] = 0x21;
            }
        }
        let x: Vec<f32> = (0..COLS).map(|i| (i % 7) as f32 * 0.1).collect();
        let xq = QuantizedX::new(&x);
        let sums = x_sums(&x);
        let activation = Activation {
            x: &x,
            sums: &sums,
            q: Some(&xq),
        };

        let mut reference = vec![0.0_f32; 8];
        fill_rows(GgmlType::Q4K, &data, stride, 0, activation, &mut reference);

        // A stride not divisible by 144 is unusual for Q4_K; the group fill must not panic or read
        // out of bounds, and it must leave the output to the row-kernel tail. Pad each row so the
        // tail can read a full (unusual) stride.
        let unusual_stride = stride + 16;
        let mut padded = vec![0u8; unusual_stride * 8];
        for row in 0..8 {
            padded[row * unusual_stride..row * unusual_stride + stride]
                .copy_from_slice(&data[row * stride..(row + 1) * stride]);
        }
        let mut unusual = vec![0.0_f32; 8];
        let wrote =
            fill_rows_q4k8_groups(&padded, unusual_stride, 0, 8, &[x.as_slice()], &mut unusual);
        assert_eq!(wrote, 0, "unusual stride must fall back to the tail");
        for (absolute, slot) in unusual.iter_mut().enumerate() {
            let start = absolute * unusual_stride;
            *slot = dot(
                GgmlType::Q4K,
                &padded[start..start + unusual_stride],
                activation,
            );
        }
        assert_eq!(
            unusual, reference,
            "padded layout must match the canonical row"
        );

        // A stride/activation mismatch must also fall back, not panic inside quantize_q8k.
        let x_short: Vec<f32> = vec![0.1; 256];
        let stride_two_blocks = 288;
        let mut data_two = vec![0u8; stride_two_blocks * 8];
        for row in 0..8 {
            data_two[row * stride_two_blocks..row * stride_two_blocks + stride]
                .copy_from_slice(&data[row * stride..(row + 1) * stride]);
        }
        let mut mismatch = vec![0.0_f32; 8];
        let wrote = fill_rows_q4k8_groups(
            &data_two,
            stride_two_blocks,
            0,
            8,
            &[x_short.as_slice()],
            &mut mismatch,
        );
        assert_eq!(
            wrote, 0,
            "mismatched activation width must fall back to the tail"
        );
    }

    /// A zero-column matrix has no stride, yet zero is a whole number of blocks, so `row_bytes`
    /// accepts it. `matvec` used to pass it through and the 8x8 Q4_K kernel then divided the tensor
    /// length by that zero stride. Refuse it at the public boundary, as `dequantize` already does.
    #[test]
    fn zero_columns_is_an_error_not_a_division_by_zero() {
        for ty in [GgmlType::F32, GgmlType::Q4K, GgmlType::Q6K] {
            let error = matvec(ty, &[], 0, 2, &[], &mut [0.0, 0.0])
                .expect_err("a zero-column matrix has no stride");
            assert!(error.to_string().contains("zero columns"), "{error}");
        }
    }

    /// Encode two whole 256-value blocks as Q4_K. Every sub-block gets scale 1 and minimum 0, so
    /// the block scale is `max|x| / 15` and each unsigned nibble is `round(x / d)`. That is the
    /// widest code step the format can give one value and keeps the minimum term out of the way.
    fn encode_q4k_pair(values: &[f32]) -> Vec<u8> {
        let mut row = vec![0_u8; 2 * 144];
        for (b, chunk) in values.as_chunks::<256>().0.iter().enumerate() {
            let block = &mut row[b * 144..(b + 1) * 144];
            let largest = chunk.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
            let d = if largest == 0.0 { 0.0 } else { largest / 15.0 };
            block[0..2].copy_from_slice(&f16(d));
            block[4..8].fill(1); // scales 0..3
            block[8..12].fill(0); // minimums 0..3
            block[12..16].fill(0x01); // high two bits of scales 4..7; minimums 4..7 stay zero
            for (j, q) in block[16..144]
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .enumerate()
            {
                for (i, slot) in q.iter_mut().enumerate() {
                    let low = (chunk[j * 64 + i] / d).round().clamp(0.0, 15.0) as u8;
                    let high = (chunk[j * 64 + i + 32] / d).round().clamp(0.0, 15.0) as u8;
                    *slot = low | (high << 4);
                }
            }
        }
        row
    }

    /// Encode two whole blocks as Q6_K: every signed sub-block scale is 1 and the block scale is
    /// `max|x| / 31`, so each six-bit level is `round(x / d) + 32` (32 is the format's centre).
    fn encode_q6k_pair(values: &[f32]) -> Vec<u8> {
        let mut row = vec![0_u8; 2 * 210];
        for (b, chunk) in values.as_chunks::<256>().0.iter().enumerate() {
            let block = &mut row[b * 210..(b + 1) * 210];
            let largest = chunk.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
            let d = if largest == 0.0 { 0.0 } else { largest / 31.0 };
            block[208..210].copy_from_slice(&f16(d));
            block[192..208].fill(1);
            for half in 0..2 {
                for l in 0..32 {
                    for run in 0..4 {
                        let value = chunk[half * 128 + run * 32 + l];
                        let level = if d == 0.0 {
                            32
                        } else {
                            (value / d).round().clamp(-31.0, 31.0) as i32 + 32
                        };
                        let (low, high) = ((level & 15) as u8, ((level >> 4) & 3) as u8);
                        let plane = if run % 2 == 0 { l } else { 32 + l };
                        if run < 2 {
                            block[half * 64 + plane] |= low;
                        } else {
                            block[half * 64 + plane] |= low << 4;
                        }
                        block[128 + half * 32 + l] |= high << (run * 2);
                    }
                }
            }
        }
        row
    }

    /// A two-block row crosses a real format boundary at 255/256. Encode a known f32 row in each
    /// format, dequantize two copies of it, and check both the round-trip limit (half a code step
    /// plus the f16 rounding of the block scale) and that the row stride leaves the blocks intact.
    #[test]
    fn q4k_and_q6k_round_trip_within_limits_at_block_boundaries() {
        let q4_values: Vec<f32> = (0..512)
            .map(|at| 0.02 + (at % 31) as f32 * 0.03 + (at / 256) as f32 * 0.05)
            .collect();
        let q6_values: Vec<f32> = (0..512)
            .map(|at| {
                let sign = if at % 2 == 0 { 1.0 } else { -1.0 };
                sign * (0.02 + (at % 31) as f32 * 0.03 + (at / 256) as f32 * 0.05)
            })
            .collect();

        for (ty, values, row) in [
            (
                GgmlType::Q4K,
                q4_values.clone(),
                encode_q4k_pair(&q4_values),
            ),
            (
                GgmlType::Q6K,
                q6_values.clone(),
                encode_q6k_pair(&q6_values),
            ),
        ] {
            let denom = if ty == GgmlType::Q4K { 15.0 } else { 31.0 };
            let largest = values.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
            let step = largest / denom;
            let tolerance = step * 0.5 + step * 0.001 + 1e-6;
            let mut data = row.clone();
            data.extend_from_slice(&row);
            let restored = dequantize(ty, &data, 512, 2).expect("two whole rows");
            assert_eq!(restored.len(), 1024);
            for at in 0..512 {
                let (want, got) = (values[at], restored[at]);
                assert!(
                    (want - got).abs() <= tolerance,
                    "{ty:?} value {at}: {want} became {got}, tolerance {tolerance}"
                );
                assert_eq!(got, restored[512 + at], "{ty:?} row 1 differs at {at}");
            }
            assert!(
                (restored[255] - values[255]).abs() <= tolerance,
                "{ty:?} last value of block 0"
            );
            assert!(
                (restored[256] - values[256]).abs() <= tolerance,
                "{ty:?} first value of block 1"
            );
            assert!(
                restored.iter().all(|value| value.is_finite()),
                "{ty:?} produced a non-finite value"
            );
        }
    }
}
