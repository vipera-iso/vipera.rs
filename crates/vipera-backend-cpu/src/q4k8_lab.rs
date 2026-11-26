//! Lab port of llama.cpp's `block_q4_Kx8` weight layout and its 8x8 gemv, kept as evidence and not
//! wired into the engine: the question it answers is whether interleaving eight weight rows into one
//! block — so a single activation broadcast feeds eight row dots — beats the row kernel that ships.
//!
//! The port is faithful to `ggml/src/ggml-cpu/repack.cpp`: `make_block_q4_Kx8` for the layout
//! (eight rows, 8-byte interleave in `qs`, a 96-byte scale area holding one sub-block of all eight
//! rows per 12 bytes) and `ggml_gemv_q4_K_8x8_q8_K` for the kernel, including its AVX2 register
//! blocking: eight rows live in one `ymm` as eight four-value lanes, `maddubs` against a broadcast
//! activation sums them, and the per-sub-block scales and mins ride along as `madd` operands.
//!
//! Two contracts differ from the engine's, both belonging to the format and not to the kernel: the
//! activation is `block_q8_K` (one scale per 256 values, `|q| <= 127`) where the engine quantizes per
//! 32 values, and the kernel emits `f32` per row. Both arms below quantize the same `f32` activation
//! their own way, so their results agree only to a tolerance — the numbers that matter here are
//! times, and the tolerance check exists to catch a broken port.

#![allow(dead_code)]
// `std::arch` intrinsics need `unsafe`, and the crate denies it outside `simd`. This module is the
// same exception, test-only: it is declared under `#[cfg(test)]` in `lib.rs` and no production path
// reaches it.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

/// Values per Q4_K super-block.
pub(crate) const QK: usize = 256;
/// Bytes of one `block_q4_K`.
pub(crate) const Q4K_BLOCK: usize = 144;
/// Bytes of one `block_q8_K`: `f32` scale, 256 `i8`, 16 `i16` sums of 16.
pub(crate) const Q8K_BLOCK: usize = 292;
/// Bytes of one `block_q4_Kx8`: eight rows, one super-block each — the same 144 bytes per row as
/// `block_q4_K`, so the layout moves nothing but the order.
pub(crate) const Q4K8_BLOCK: usize = 1152;

/// `struct block_q4_Kx8` from `repack.h`, field for field.
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

const _: () = assert!(core::mem::size_of::<BlockQ4Kx8>() == Q4K8_BLOCK);

/// `make_block_q4_Kx8` with `blck_size_interleave = 8`, for a matrix of `rows` rows (a multiple of
/// eight) whose rows sit `blocks * 144` bytes apart.
pub(crate) fn pack_q4k8(src: &[u8], rows: usize, blocks: usize) -> Vec<BlockQ4Kx8> {
    assert_eq!(rows % 8, 0, "block_q4_Kx8 interleaves eight rows");
    assert_eq!(src.len(), rows * blocks * Q4K_BLOCK);
    let stride = blocks * Q4K_BLOCK;
    let mut out = Vec::with_capacity(rows / 8 * blocks);
    let mut s = [0_u8; 8];
    let mut m = [0_u8; 8];
    for g in 0..rows / 8 {
        for b in 0..blocks {
            let row = |i: usize| &src[(g * 8 + i) * stride + b * Q4K_BLOCK..][..Q4K_BLOCK];
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
            // Sub-blocks 4-7: the top two bits live in the first four bytes, the low four in the
            // last four, which is the Q4_K six-bit packing read backwards.
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
    out
}

/// `quantize_row_q8_K_ref` for each 256-value block: one scale for the whole block, which is the
/// coarse contract the llama.cpp kernel multiplies by.
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

/// `ggml_gemv_q4_K_8x8_q8_K_generic`: the same layout and the same arithmetic as the vector kernel,
/// written as straight loops. It is the reference the port is checked against, and the place to read
/// what the layout means: `qs[k*64 + row*8 + i]` holds the low nibble for column `k*8+i` and the high
/// nibble for column `k*8+i+32`, and the 6-bit scale area holds, per 12 bytes, eight scales then
/// eight mins of one sub-block across the eight rows.
#[allow(clippy::needless_range_loop)]
pub(crate) fn gemv_q4k8_generic(w: &[BlockQ4Kx8], blocks: usize, act: &[u8], out: &mut [f32]) {
    let groups = out.len() / 8;
    assert_eq!(w.len(), groups * blocks);
    let (kmask1, kmask2, kmask3) = (0x3f3f_3f3f_u32, 0x0f0f_0f0f_u32, 0x0303_0303_u32);
    let half = |bits: u16| crate::quant::f16_to_f32(bits);
    for x in 0..groups {
        let mut sumf = [0.0_f32; 8];
        let mut sum_minf = [0.0_f32; 8];
        for l in 0..blocks {
            let blk = &w[x * blocks + l];
            let a = &act[l * Q8K_BLOCK..(l + 1) * Q8K_BLOCK];
            let a_d = f32::from_le_bytes(a[0..4].try_into().unwrap());
            let mut utmp = [0_u8; 128];
            for sb in 0..8 {
                utmp[sb * 16..sb * 16 + 12].copy_from_slice(&blk.scales[sb * 12..sb * 12 + 12]);
                let at = sb * 16;
                let mut u = [0_u32; 4];
                for (i, slot) in u.iter_mut().enumerate() {
                    *slot =
                        u32::from_le_bytes(utmp[at + i * 4..at + i * 4 + 4].try_into().unwrap());
                }
                u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
                let aux = u[1] & kmask1;
                u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
                u[2] = aux;
                u[0] &= kmask1;
                for (i, v) in u.iter().enumerate() {
                    utmp[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            for k in 0..QK / 16 {
                let s0 = (k / 4) * 32;
                let s1 = s0 + 16;
                for j in 0..8 {
                    let mut sumi = 0_i32;
                    for i in 0..8 {
                        let q = blk.qs[k * 64 + j * 8 + i];
                        let lo = (q & 0xf) as i32;
                        let hi = (q >> 4) as i32;
                        let base = 4 + (k >> 2) * 64 + (k % 4) * 8 + i;
                        let a_lo = a[base] as i8 as i32;
                        let a_hi = a[base + 32] as i8 as i32;
                        sumi += lo * a_lo * utmp[s0 + j] as i32 + hi * a_hi * utmp[s1 + j] as i32;
                    }
                    sumf[j] += sumi as f32 * half(blk.d[j]) * a_d;
                }
            }
            for sb in 0..8 {
                let mins = &utmp[8 + sb * 16..16 + sb * 16];
                let bsums =
                    i16::from_le_bytes(a[260 + sb * 4..260 + sb * 4 + 2].try_into().unwrap())
                        as i32
                        + i16::from_le_bytes(a[262 + sb * 4..262 + sb * 4 + 2].try_into().unwrap())
                            as i32;
                for j in 0..8 {
                    sum_minf[j] += mins[j] as f32 * bsums as f32 * half(blk.dmin[j]) * a_d;
                }
            }
        }
        for j in 0..8 {
            out[x * 8 + j] = sumf[j] - sum_minf[j];
        }
    }
}

/// `ggml_gemv_q4_K_8x8_q8_K` for one activation row. `out` holds eight results per weight group and
/// `blocks` is the super-block count of the row length.
///
/// # Safety
///
/// `w.len() == out.len() / 8 * blocks`, `act` holds one `block_q8_K` per block, and the CPU has
/// AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn gemv_q4k8(w: &[BlockQ4Kx8], blocks: usize, act: &[u8], out: &mut [f32]) {
    let groups = out.len() / 8;
    assert_eq!(w.len(), groups * blocks);
    assert!(act.len() >= blocks * Q8K_BLOCK);
    // SAFETY: every intrinsic below needs AVX2/FMA/F16C, which the caller guarantees, and the
    // pointers are derived from slices whose lengths are asserted above.
    unsafe {
        // Row order of the six-bit scale area, and the broadcast of `acc_row` into row order.
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
                // Sums of 16 are folded to sums of 32, then broadcast one pair of sub-blocks at a
                // time; the signed broadcast makes the `madd` pick the right half of each pair.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{QuantizedX, x_sums};
    use crate::simd::{q4k_dot_vnni, vnni512_available};

    /// Small deterministic source, so the lab shares the shapes the other benches use without
    /// reaching into the test-only helpers in `simd`.
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
            (self.next() >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
        }

        /// A binary16 with the given magnitude range, as two little-endian bytes.
        fn half(&mut self, low: f32, high: f32) -> [u8; 2] {
            let v = low + (high - low) * (self.unit() + 0.5);
            let bits = v.to_bits();
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let mant = ((bits >> 13) & 0x3ff) as u16;
            let half = if exp > 0 && exp < 31 {
                (((bits >> 16) & 1) as u16) << 15 | (exp as u16) << 10 | mant
            } else {
                0
            };
            half.to_le_bytes()
        }
    }

    /// The 6-bit scale and min of sub-block `g`, per the Q4_K spec.
    pub(super) fn scale_min_pub(scales: &[u8], g: usize) -> (f32, f32) {
        let get = |i: usize| -> u8 {
            if i < 4 {
                scales[i] & 63
            } else {
                (scales[i + 4] & 0x0f) | ((scales[i - 4] >> 6) << 4)
            }
        };
        let m = |i: usize| -> u8 {
            if i < 4 {
                scales[i + 4] & 63
            } else {
                (scales[i + 4] >> 4) | ((scales[i] >> 6) << 4)
            }
        };
        (get(g) as f32, m(g) as f32)
    }

    fn q4k_row(rng: &mut Rng, blocks: usize) -> Vec<u8> {
        let mut row = Vec::with_capacity(blocks * Q4K_BLOCK);
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

    struct Lab {
        packed: Vec<BlockQ4Kx8>,
        raw: Vec<u8>,
        act: Vec<u8>,
        x: Vec<f32>,
        rows: usize,
        blocks: usize,
    }

    fn lab(rows: usize, cols: usize, seed: u64) -> Lab {
        let mut rng = Rng(seed);
        let blocks = cols / QK;
        let row = q4k_row(&mut rng, blocks);
        let raw: Vec<u8> = (0..rows).flat_map(|_| row.clone()).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let mut act = vec![0_u8; blocks * Q8K_BLOCK];
        quantize_q8k(&x, &mut act);
        let packed = pack_q4k8(&raw, rows, blocks);
        Lab {
            packed,
            raw,
            act,
            x,
            rows,
            blocks,
        }
    }

    /// The port has to land on the shipping row kernel within the formats' quantization gap, which
    /// is small: both quantize the same activation to `i8`, one per 256 values, one per 32.
    #[test]
    fn q4k8_gemv_matches_the_row_kernel_within_quantization() {
        if !vnni512_available() {
            return;
        }
        let Lab {
            packed,
            raw,
            act,
            x,
            rows,
            blocks,
        } = lab(64, 1536, 0x9a7b_0001);
        let stride = blocks * Q4K_BLOCK;
        let mut got = vec![0.0_f32; rows];
        // SAFETY: the CPU was checked above and the layout comes from `pack_q4k8`.
        unsafe { gemv_q4k8(&packed, blocks, &act, &mut got) };
        let xq = QuantizedX::new(&x);
        let sums = x_sums(&x);
        for r in 0..rows {
            // SAFETY: whole 144-byte blocks, VNNI checked above.
            let want =
                unsafe { q4k_dot_vnni(&raw[r * stride..(r + 1) * stride], &xq.q, &xq.d, &sums) };
            let scale = want.abs().max(1e-6);
            let error = (got[r] - want).abs() / scale;
            assert!(error < 0.02, "row {r}: {} vs {want} ({error:.4})", got[r]);
        }
    }

    /// The interleaved layout, read by the scalar reference, against the shipping row kernel: this
    /// is the layout question with no vector code in between.
    #[test]
    fn q4k8_layout_matches_the_row_kernel_within_quantization() {
        let Lab {
            packed,
            raw,
            act,
            x,
            rows: _,
            blocks,
        } = lab(64, 1536, 0x9a7b_0003);
        let stride = blocks * Q4K_BLOCK;
        let mut got = vec![0.0_f32; 64];
        gemv_q4k8_generic(&packed, blocks, &act, &mut got);
        let xq = QuantizedX::new(&x);
        let sums = x_sums(&x);
        for r in 0..64 {
            // SAFETY: whole 144-byte blocks, and the CPU is x86-64 with AVX2 whenever this runs
            // there; `q4k_dot_vnni` is only called when VNNI is present.
            if !vnni512_available() {
                return;
            }
            let want =
                unsafe { q4k_dot_vnni(&raw[r * stride..(r + 1) * stride], &xq.q, &xq.d, &sums) };
            let scale = want.abs().max(1e-6);
            let error = (got[r] - want).abs() / scale;
            assert!(error < 0.02, "row {r}: {} vs {want} ({error:.4})", got[r]);
        }
    }

    /// The vector kernel against the scalar reference of the same layout.
    #[test]
    fn q4k8_gemv_matches_the_generic_reference() {
        if !vnni512_available() {
            return;
        }
        let Lab {
            packed,
            act,
            blocks,
            ..
        } = lab(64, 1536, 0x9a7b_0004);
        let mut want = vec![0.0_f32; 64];
        let mut got = vec![0.0_f32; 64];
        gemv_q4k8_generic(&packed, blocks, &act, &mut want);
        // SAFETY: the CPU was checked above and the layout comes from `pack_q4k8`.
        unsafe { gemv_q4k8(&packed, blocks, &act, &mut got) };
        for r in 0..64 {
            let scale = want[r].abs().max(1e-6);
            let error = (got[r] - want[r]).abs() / scale;
            assert!(
                error < 0.01,
                "row {r}: {} vs {} ({error:.4})",
                got[r],
                want[r]
            );
        }
    }

    /// Every nibble 1, every scale 1, every min 0, `d == 1`, activation all ones: the 256 products
    /// are 256, so the number that comes out names the step that is wrong. The random rows only
    /// prove the two kernels agree; this one proves the layout itself puts each scale, min and
    /// nibble where the kernel reads it.
    #[test]
    fn q4k8_crafted_block_dots_256() {
        let mut row = Vec::new();
        row.extend_from_slice(&0x3c00_u16.to_le_bytes()); // d = 1
        row.extend_from_slice(&0x0000_u16.to_le_bytes()); // dmin = 0
        row.extend_from_slice(&[1, 1, 1, 1]); // s0..s3 = 1, high bits 0
        row.extend_from_slice(&[0, 0, 0, 0]); // m0..m3 = 0
        row.extend_from_slice(&[1, 1, 1, 1]); // s4..s7 = 1, m4..m7 = 0
        row.extend_from_slice(&[0x11; 128]);
        let raw: Vec<u8> = (0..8).flat_map(|_| row.clone()).collect();
        let x = vec![1.0_f32; QK];
        let mut act = vec![0_u8; Q8K_BLOCK];
        quantize_q8k(&x, &mut act);
        let packed = pack_q4k8(&raw, 8, 1);
        let mut want = vec![0.0_f32; 8];
        gemv_q4k8_generic(&packed, 1, &act, &mut want);
        let mut got = vec![0.0_f32; 8];
        // SAFETY: the CPU check above stands and the layout comes from `pack_q4k8`.
        unsafe { gemv_q4k8(&packed, 1, &act, &mut got) };
        for r in 0..8 {
            assert!(
                (want[r] - 256.0).abs() < 1e-3,
                "row {r}: generic {}",
                want[r]
            );
            assert!((got[r] - 256.0).abs() < 1e-3, "row {r}: simd {}", got[r]);
        }
    }

    /// Eight rows per weight block against one row per block, at the shapes decode reads, with the
    /// row split the decode pool uses so a single-thread lead has to survive the cache and the bus.
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn bench_q4k8_gemv_vs_row() {
        if !vnni512_available() {
            return;
        }
        let physical = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads: Vec<usize> = [1, 2, 4, 8]
            .into_iter()
            .filter(|&t| t <= physical)
            .collect();
        for (rows, cols) in [(8960usize, 1536usize), (1536, 8960), (8960, 8960)] {
            let Lab {
                packed,
                raw,
                act,
                x,
                rows,
                blocks,
            } = lab(rows, cols, 0x9a7b_0002);
            let stride = blocks * Q4K_BLOCK;
            let bytes = (rows * stride) as f64;
            let xq = QuantizedX::new(&x);
            let sums = x_sums(&x);
            let mut totals = vec![0.0_f32; rows];
            let mut out = vec![0.0_f32; rows];
            let sample = |threads: usize,
                          eight: bool,
                          act: &[u8],
                          xq: &QuantizedX,
                          sums: &[f32],
                          totals: &mut [f32],
                          out: &mut [f32]| {
                let per = ((rows / 8) / threads) * 8;
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let start = std::time::Instant::now();
                    std::thread::scope(|s| {
                        for (idx, (o, t)) in
                            out.chunks_mut(per).zip(totals.chunks_mut(per)).enumerate()
                        {
                            let r0 = idx * per;
                            let (packed, raw) = (&packed, &raw);
                            // SAFETY: `vnni512_available` was checked, the CPU feature is process
                            // wide, and the row split keeps every slice a whole number of blocks
                            // and weight groups, so no worker reads past its own rows.
                            s.spawn(move || unsafe {
                                if eight {
                                    gemv_q4k8(
                                        &packed[r0 / 8 * blocks..(r0 + per) / 8 * blocks],
                                        blocks,
                                        act,
                                        o,
                                    );
                                } else {
                                    let raw = &raw[r0 * stride..(r0 + per) * stride];
                                    for (i, t) in t.iter_mut().enumerate() {
                                        *t = q4k_dot_vnni(
                                            &raw[i * stride..(i + 1) * stride],
                                            &xq.q,
                                            &xq.d,
                                            sums,
                                        );
                                    }
                                }
                            });
                        }
                    });
                    best = best.min(start.elapsed().as_secs_f64());
                }
                best
            };
            for &t in &threads {
                let (act, xq, sums) = (&act[..], &xq, &sums[..]);
                let row_time = sample(t, false, act, xq, sums, &mut totals, &mut out);
                let eight_time = sample(t, true, act, xq, sums, &mut totals, &mut out);
                println!(
                    "q4k8 {rows}x{cols} t={t}: row {:>7.2} ms ({:.1} GB/s) | 8x8 {:>7.2} ms \
                     ({:.1} GB/s) | {:.3}x",
                    row_time * 1e3,
                    bytes / row_time / 1e9,
                    eight_time * 1e3,
                    bytes / eight_time / 1e9,
                    row_time / eight_time,
                );
            }
        }
    }

    /// What the prefill matmul actually achieves per MAC, single thread, once the engine's
    /// plumbing is out of the picture: the shipping tiled band at the batch width a prefill chunk
    /// uses (128), at the minimum width that picks the tiled path (16), and the row kernel.
    ///
    /// The numbers to compare against are whole-engine rates on the same machine: vipera's prefill
    /// is 176 GFLOPS single thread, llama.cpp's is 229 (`docs/benchmarks.md` 2026-11-23). A rate
    /// here well above 176 would put the gap in the engine, not in the kernel.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn bench_prefill_band_gflops() {
        use crate::quant::{BatchActivation, OwnedActivation, fill_rows_batch};
        use vipera_core::gguf::types::GgmlType;

        if !vnni512_available() {
            return;
        }
        let (rows, cols, widest) = (16384usize, 1536usize, 128usize);
        let Lab { raw, blocks, x, .. } = lab(rows, cols, 0x9a7b_0003);
        let stride = blocks * Q4K_BLOCK;
        let mut rng = Rng(0x51ee_0001);
        let mut sources = Vec::with_capacity(widest);
        sources.push(x.clone());
        for _ in 1..widest {
            sources.push((0..cols).map(|_| rng.unit()).collect::<Vec<f32>>());
        }
        let owned: Vec<OwnedActivation> = sources
            .iter()
            .map(|row| OwnedActivation::new(GgmlType::Q4K, row))
            .collect();
        let flops = 2.0 * (rows * cols) as f64;

        let band = |entries: usize, out: &mut [f32]| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                match entries {
                    128 => {
                        let batch: &[OwnedActivation; 128] =
                            owned[..128].try_into().expect("exact width");
                        let act = BatchActivation::<128>::new(batch);
                        fill_rows_batch::<128>(
                            GgmlType::Q4K,
                            &raw,
                            stride,
                            0,
                            &act,
                            None,
                            out,
                            None,
                        );
                    }
                    16 => {
                        let batch: &[OwnedActivation; 16] =
                            owned[..16].try_into().expect("exact width");
                        let act = BatchActivation::<16>::new(batch);
                        fill_rows_batch::<16>(
                            GgmlType::Q4K,
                            &raw,
                            stride,
                            0,
                            &act,
                            None,
                            out,
                            None,
                        );
                    }
                    other => panic!("no arm for {other} entries"),
                }
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let row_kernel = |out: &mut [f32]| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                for entry in &owned {
                    let batch: &[OwnedActivation; 1] =
                        std::slice::from_ref(entry).try_into().expect("one");
                    let act = BatchActivation::<1>::new(batch);
                    fill_rows_batch::<1>(GgmlType::Q4K, &raw, stride, 0, &act, None, out, None);
                }
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };

        let mut out = vec![0.0_f32; rows * widest];
        let full = flops * widest as f64;
        let wide = band(128, &mut out);
        println!(
            "prefill band {rows}x{cols} B=128: {:>7.2} ms  {:.1} GFLOPS",
            wide * 1e3,
            full / wide / 1e9
        );
        let narrow = band(16, &mut out[..rows * 16]);
        println!(
            "prefill band {rows}x{cols} B=16:  {:>7.2} ms  {:.1} GFLOPS",
            narrow * 1e3,
            flops * 16.0 / narrow / 1e9
        );
        let row = row_kernel(&mut out[..rows]);
        println!(
            "prefill row  {rows}x{cols} B=1x128: {:>7.2} ms  {:.1} GFLOPS",
            row * 1e3,
            full / row / 1e9
        );
    }
}
