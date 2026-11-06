//! Block formats and the quantized matrix-vector product.
//!
//! A GGUF matrix is stored row-major over `dims[0]`: row `r` starts at `r * row_bytes` and holds
//! `cols = dims[0]` values. Every block size here divides the row length of a real model.
//!
//! Only the formats a Q4_K_M file can contain are implemented (F32, F16, Q8_0, Q4_K, Q6_K). Anything
//! else is refused by name rather than dequantized approximately: a wrong scale convention produces
//! fluent-looking nonsense, which is harder to notice than an error.

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
    Ok(cols / values * bytes)
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

/// Sums of `x` over every 16 values.
///
/// Every k-quant scale is constant inside 16 or 32 values, so the minimum term
/// `-dmin * m * Σx` needs only these sums: computing them once per matvec removes one multiply and
/// one subtract from every weight, which is where most of the time went.
pub(crate) fn x_sums(x: &[f32]) -> Vec<f32> {
    x.as_chunks::<16>()
        .0
        .iter()
        .map(|chunk| chunk.iter().sum())
        .collect()
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

impl QuantizedX {
    /// Quantize `x` in groups of 32. A trailing partial group is ignored, and every caller in this
    /// module passes a length that is a whole number of groups.
    pub(crate) fn new(x: &[f32]) -> Self {
        let mut q = Vec::with_capacity(x.len());
        let mut d = Vec::with_capacity(x.len() / 32);
        for group in x.as_chunks::<32>().0 {
            let largest = group.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if largest == 0.0 {
                d.push(0.0);
                q.extend(std::iter::repeat_n(0, 32));
                continue;
            }
            let scale = largest / 127.0;
            d.push(scale);
            for value in group {
                q.push((value / scale).round().clamp(-127.0, 127.0) as i8);
            }
        }
        Self { q, d }
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
        let sums = &sums[b * 16..b * 16 + 16];
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
            let sum0 = sums[4 * j] + sums[4 * j + 1];
            let sum1 = sums[4 * j + 2] + sums[4 * j + 3];
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

/// Dot product of one row with `x` in portable Rust, for any supported format.
///
/// This is the reference the AVX2 kernels are checked against, and the path taken on CPUs without it.
pub(crate) fn dot_scalar(ty: GgmlType, row: &[u8], x: &[f32], sums: &[f32]) -> f32 {
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
        GgmlType::Q6K => q6k_dot(row, x, sums),
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
    dot_scalar(ty, row, activation.x, activation.sums)
}

/// The activation vector, with the state `matvec` derives from it once per call.
///
/// Bundling the three slices keeps the per-row plumbing to one argument, and makes it obvious that
/// the quantized copy and the sums always come from the same `x`.
#[derive(Clone, Copy)]
pub(crate) struct Activation<'a> {
    /// The `cols` values being multiplied.
    x: &'a [f32],
    /// Sums of `x` over every 16 values, for the minimum term.
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
    sums: Vec<f32>,
    q: Option<QuantizedX>,
}

impl OwnedActivation {
    pub(crate) fn new(ty: GgmlType, x: &[f32]) -> Self {
        Self {
            x: x.to_vec(),
            sums: x_sums(x),
            q: (ty == GgmlType::Q4K).then(|| QuantizedX::new(x)),
        }
    }

    pub(crate) fn borrowed(&self) -> Activation<'_> {
        Activation {
            x: &self.x,
            sums: &self.sums,
            q: self.q.as_ref(),
        }
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
    let bytes = data
        .get(start..start + stride)
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
    let stride = row_bytes(ty, cols)?;
    if data.len() < stride * rows {
        return Err(ViperaError::parse(
            data.len() as u64,
            format!(
                "{} bytes are too few for {rows} rows of {cols} values",
                data.len()
            ),
        ));
    }
    let mut out = vec![0.0; cols * rows];
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
    // One quantization of the activation, shared by every row. Only Q4_K uses it; building it for
    // the other formats would be a wasted pass over `x`.
    let xq = (ty == GgmlType::Q4K).then(|| QuantizedX::new(x));
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
    pool: &crate::pool::Pool<'d>,
    ty: GgmlType,
    data: &'d [u8],
    cols: usize,
    rows: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let stride = checked_stride(ty, data, cols, rows, x, out)?;
    let threads = thread_budget(rows, rows.saturating_mul(cols));
    let activation = std::sync::Arc::new(OwnedActivation::new(ty, x));
    pool.fill(ty, data, stride, threads, &activation, out)
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
        .min(rows.max(1))
}

pub(crate) fn fill_rows(
    ty: GgmlType,
    data: &[u8],
    stride: usize,
    first_row: usize,
    activation: Activation<'_>,
    out: &mut [f32],
) {
    for (offset, slot) in out.iter_mut().enumerate() {
        let start = (first_row + offset) * stride;
        *slot = dot(ty, &data[start..start + stride], activation);
    }
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
            let activation = std::sync::Arc::new(OwnedActivation::new(GgmlType::Q4K, &x));
            let mut pooled = vec![0.0_f32; ROWS];
            pool.fill(GgmlType::Q4K, &data, stride, 4, &activation, &mut pooled)
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

    #[test]
    fn dequantize_fills_every_row() {
        let mut data = Vec::new();
        for value in 0..6 {
            data.extend_from_slice(&(value as f32).to_le_bytes());
        }
        let values = dequantize(GgmlType::F32, &data, 3, 2).expect("dequantize");
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
