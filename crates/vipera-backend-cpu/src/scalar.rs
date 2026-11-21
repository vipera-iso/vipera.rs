//! Portable scalar kernels.
//!
//! Two jobs. On a non-x86_64 target this is the whole `simd` module: `lib.rs` re-exports it under
//! that name, so `model.rs` and everything above it builds and runs unchanged. On x86_64 it is
//! compiled for the tests only, where it is the reference the AVX2 kernels are held to — the tests
//! at the bottom of this file run both and compare.
//!
//! Every function here mirrors the scalar branch inside its AVX2 counterpart, down to the order of
//! the additions, so a target without AVX2 produces the same numbers rather than merely similar
//! ones. Where the x86 kernel changes the reduction order (the tiled Q4_K matvec, the grouped
//! exponent) the difference is already documented as rounding elsewhere, and this module is the
//! side that keeps the plain order.

use crate::quant::f16_to_f32;

/// Positions folded into one accumulator pass by [`axpy_f16_group`]; the AVX2 kernel uses the same
/// number so the two paths walk the rows in the same order.
pub(crate) const GROUP_ROWS: usize = 16;

/// Heads the AVX2 kernels keep in registers. The scalar path does not need the bound, but callers
/// pass the same shapes and the block kernel uses it to reject what the vector path would.
const HEADS_IN_REGISTERS: usize = 8;

/// Widen little-endian `f16` values into `f32`, stopping at whichever slice is shorter.
pub(crate) fn half_to_float(source: &[u8], values: &mut [f32]) {
    let length = values.len().min(source.len() / 2);
    for (value, chunk) in values[..length].iter_mut().zip(source.as_chunks::<2>().0) {
        *value = f16_to_f32(u16::from_le_bytes(*chunk));
    }
}

/// `sum(a[i] * b[i])` over the shorter length.
#[allow(dead_code)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());
    let mut total = 0.0_f32;
    for (a, b) in a.iter().zip(b).take(length) {
        total += a * b;
    }
    total
}

/// `target[i] += weight * source[i]` over the shorter length.
#[allow(dead_code)]
pub(crate) fn axpy_f32(target: &mut [f32], weight: f32, source: &[f32]) {
    let length = target.len().min(source.len());
    for (target, source) in target.iter_mut().zip(source).take(length) {
        *target += weight * source;
    }
}

/// `up[i] *= silu(gate[i])`, the FFN's gated product.
pub(crate) fn silu_mul(up: &mut [f32], gate: &[f32]) {
    let length = up.len().min(gate.len());
    for (out, gate) in up[..length].iter_mut().zip(&gate[..length]) {
        *out *= crate::model::silu(*gate);
    }
}

/// `values[i] = exp(values[i] - shift)`, returning their sum.
pub(crate) fn exp_sum_shifted(values: &mut [f32], shift: f32) -> f32 {
    let mut total = 0.0_f32;
    for value in values.iter_mut() {
        let exponential = crate::model::exp(*value - shift);
        *value = exponential;
        total += exponential;
    }
    total
}

/// Dot of an `f32` vector against an `f16` row stored as little-endian bytes.
pub(crate) fn dot_f16_f32(a: &[f32], row: &[u8]) -> f32 {
    let length = a.len().min(row.len() / 2);
    let mut total = 0.0_f32;
    for (value, chunk) in a.iter().zip(row.as_chunks::<2>().0).take(length) {
        total += value * f16_to_f32(u16::from_le_bytes(*chunk));
    }
    total
}

/// `target[i] += weight * f16(row[i])` over the shorter length.
pub(crate) fn axpy_f16_f32(target: &mut [f32], weight: f32, row: &[u8]) {
    let length = target.len().min(row.len() / 2);
    for (target, chunk) in target.iter_mut().zip(row.as_chunks::<2>().0).take(length) {
        *target += weight * f16_to_f32(u16::from_le_bytes(*chunk));
    }
}

/// The scores of `count` query heads against one `f16` key row. `q` holds the heads back to back
/// at `head_dim` each, starting at `first_head`.
pub(crate) fn dot_f16_f32_heads(
    q: &[f32],
    first_head: usize,
    count: usize,
    head_dim: usize,
    row: &[u8],
    out: &mut [f32],
) {
    for (member, slot) in out.iter_mut().take(count).enumerate() {
        let at = (first_head + member) * head_dim;
        *slot = dot_f16_f32(&q[at..at + head_dim], row);
    }
}

/// The same scores with [`dot_f16_f32`]'s rounding, for the paths that ask for bit-exactness.
pub(crate) fn dot_f16_f32_heads_exact(
    q: &[f32],
    first_head: usize,
    count: usize,
    head_dim: usize,
    row: &[u8],
    out: &mut [f32],
) {
    dot_f16_f32_heads(q, first_head, count, head_dim, row, out);
}

/// `targets += weights[h] * f16(row)` for each head, `head_dim` values per head.
pub(crate) fn axpy_f16_f32_heads(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    row: &[u8],
) {
    for (member, weight) in weights.iter().enumerate() {
        let target = &mut targets[member * head_dim..][..head_dim];
        axpy_f16_f32(target, *weight, row);
    }
}

/// The same accumulation with [`axpy_f16_f32`]'s rounding.
pub(crate) fn axpy_f16_f32_heads_exact(
    targets: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    row: &[u8],
) {
    axpy_f16_f32_heads(targets, head_dim, weights, row);
}

/// `P` positions folded into the same targets, `weights` position-major as `weights[p * count + head]`.
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
    for (position, row) in rows.iter().enumerate() {
        let weights = &weights[position * count..(position + 1) * count];
        axpy_f16_f32_heads_exact(targets, head_dim, weights, row);
    }
}

/// The heads of one KV group scored against one key row: `out[head]` is that head's dot.
pub(crate) fn dot_f16_group(queries: &[f32], head_dim: usize, row: &[u8], out: &mut [f32]) {
    for (head, slot) in out.iter_mut().enumerate() {
        let query = queries
            .get(head * head_dim..(head + 1) * head_dim)
            .unwrap_or(&[]);
        *slot = dot_f16_f32(query, row);
    }
}

/// The heads of one KV group accumulating one `f16` value row each, `weights` position-major over
/// `rows`, `heads` values per position.
pub(crate) fn axpy_f16_group(
    targets: &mut [f32],
    head_dim: usize,
    heads: usize,
    weights: &[f32],
    rows: &[&[u8]],
) {
    for (position, row) in rows.iter().enumerate() {
        for head in 0..heads {
            let weight = weights.get(position * heads + head).copied().unwrap_or(0.0);
            if let Some(target) = targets.get_mut(head * head_dim..(head + 1) * head_dim) {
                axpy_f16_f32(target, weight, row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator: the comparisons below only need varied bits, not randomness
    /// with a distribution.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        /// `-1..1`, small enough that a sum of 260 of them stays inside f32's exact range for the
        /// tolerance below.
        fn unit(&mut self) -> f32 {
            (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
        }

        fn row(&mut self, values: usize) -> Vec<u8> {
            (0..values)
                .flat_map(|_| crate::quant::f32_to_f16(self.unit()).to_le_bytes())
                .collect()
        }
    }

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs()))
    }

    /// The comparisons that matter are between this module and the AVX2 kernels it stands in for;
    /// the tolerance covers the reduction order, which the vector path is documented to change.
    const TOLERANCE: f32 = 1e-5;

    #[test]
    fn the_scalar_dots_match_the_kernels_they_stand_in_for() {
        assert_eq!(GROUP_ROWS, crate::simd::GROUP_ROWS);
        let mut rng = Rng(0x005c_a1a2);
        for length in [0_usize, 1, 7, 8, 15, 16, 63, 64, 127, 260] {
            let a: Vec<f32> = (0..length).map(|_| rng.unit()).collect();
            let row = rng.row(length);
            let mut widened = vec![0.0_f32; length];
            let mut widened_vector = vec![0.0_f32; length];
            half_to_float(&row, &mut widened);
            crate::simd::half_to_float(&row, &mut widened_vector);
            for (lane, (scalar, vector)) in widened.iter().zip(&widened_vector).enumerate() {
                assert_eq!(
                    scalar.to_bits(),
                    vector.to_bits(),
                    "half_to_float, length {length}, lane {lane}"
                );
            }
            assert!(
                close(
                    dot_f16_f32(&a, &row),
                    crate::simd::dot_f16_f32(&a, &row),
                    TOLERANCE
                ),
                "dot_f16_f32, length {length}"
            );

            let mut left = a.clone();
            let mut right = a.clone();
            axpy_f16_f32(&mut left, 0.375, &row);
            crate::simd::axpy_f16_f32(&mut right, 0.375, &row);
            for (lane, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, TOLERANCE),
                    "axpy_f16_f32, length {length}, lane {lane}"
                );
            }

            let b: Vec<f32> = (0..length).map(|_| rng.unit()).collect();
            assert!(close(
                dot_f32(&a, &b),
                crate::simd::dot_f32(&a, &b),
                TOLERANCE
            ));
        }
    }

    #[test]
    fn the_scalar_zero_short_and_exponent_paths_match() {
        let mut rng = Rng(0x005c_a1b3);
        for length in [0_usize, 3, 8, 9, 16, 17, 260] {
            let values: Vec<f32> = (0..length).map(|_| rng.unit() * 4.0).collect();

            let mut left = values.clone();
            let mut right = values.clone();
            let left_sum = exp_sum_shifted(&mut left, 0.5);
            let right_sum = crate::simd::exp_sum_shifted(&mut right, 0.5);
            assert!(
                close(left_sum, right_sum, TOLERANCE),
                "exp_sum_shifted sum, length {length}"
            );
            for (lane, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, 1e-6),
                    "exp_sum_shifted value, length {length}, lane {lane}"
                );
            }

            let gate: Vec<f32> = (0..length).map(|_| rng.unit() * 3.0).collect();
            let mut left = values.clone();
            let mut right = values.clone();
            silu_mul(&mut left, &gate);
            crate::simd::silu_mul(&mut right, &gate);
            for (lane, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, 1e-6),
                    "silu_mul, length {length}, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn the_scalar_group_kernels_match_the_kernels_they_stand_in_for() {
        let mut rng = Rng(0x005c_a1c4);
        let head_dim = 64_usize;
        let row = rng.row(head_dim);
        for heads in [1_usize, 2, 6, 8] {
            let queries: Vec<f32> = (0..heads * head_dim).map(|_| rng.unit()).collect();
            let mut left = vec![0.0_f32; heads];
            let mut right = vec![0.0_f32; heads];
            dot_f16_group(&queries, head_dim, &row, &mut left);
            crate::simd::dot_f16_group(&queries, head_dim, &row, &mut right);
            for (head, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, TOLERANCE),
                    "dot_f16_group, head {head}"
                );
            }

            let first_head = 2_usize;
            let mut wide = vec![0.0_f32; (first_head + heads) * head_dim];
            for value in wide.iter_mut() {
                *value = rng.unit();
            }
            let mut left = vec![0.0_f32; heads];
            let mut right = vec![0.0_f32; heads];
            let mut exact = vec![0.0_f32; heads];
            dot_f16_f32_heads(&wide, first_head, heads, head_dim, &row, &mut left);
            dot_f16_f32_heads_exact(&wide, first_head, heads, head_dim, &row, &mut exact);
            crate::simd::dot_f16_f32_heads(&wide, first_head, heads, head_dim, &row, &mut right);
            assert_eq!(left, exact, "the exact path is the plain one");
            for (head, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, TOLERANCE),
                    "dot_f16_f32_heads, head {head}"
                );
            }

            let weights: Vec<f32> = (0..heads).map(|_| rng.unit()).collect();
            let mut left = queries.clone();
            let mut right = queries.clone();
            axpy_f16_f32_heads(&mut left, head_dim, &weights, &row);
            crate::simd::axpy_f16_f32_heads(&mut right, head_dim, &weights, &row);
            for (lane, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, TOLERANCE),
                    "axpy_f16_f32_heads, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn the_scalar_row_group_accumulation_matches_the_kernel() {
        let mut rng = Rng(0x005c_a1d5);
        let head_dim = 64_usize;
        let heads = 6_usize;
        for positions in [0_usize, 1, 5, 16, 17, 40] {
            let rows: Vec<Vec<u8>> = (0..positions).map(|_| rng.row(head_dim)).collect();
            let row_refs: Vec<&[u8]> = rows.iter().map(|row| row.as_slice()).collect();
            let weights: Vec<f32> = (0..positions * heads).map(|_| rng.unit()).collect();
            let base: Vec<f32> = (0..heads * head_dim).map(|_| rng.unit()).collect();

            let mut left = base.clone();
            let mut right = base.clone();
            axpy_f16_group(&mut left, head_dim, heads, &weights, &row_refs);
            crate::simd::axpy_f16_group(&mut right, head_dim, heads, &weights, &row_refs);
            for (lane, (left, right)) in left.iter().zip(&right).enumerate() {
                assert!(
                    close(*left, *right, TOLERANCE),
                    "axpy_f16_group, positions {positions}, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn the_scalar_block_accumulation_matches_one_call_per_position() {
        let mut rng = Rng(0x005c_a1e6);
        let head_dim = 64_usize;
        let heads = 6_usize;
        const P: usize = 4;
        let rows: Vec<Vec<u8>> = (0..P).map(|_| rng.row(head_dim)).collect();
        let row_refs: [&[u8]; P] = std::array::from_fn(|at| rows[at].as_slice());
        let weights: Vec<f32> = (0..P * heads).map(|_| rng.unit()).collect();

        let mut block = vec![0.0_f32; heads * head_dim];
        axpy_f16_f32_heads_exact_block::<P>(&mut block, head_dim, &weights, &row_refs);

        let mut per_position = vec![0.0_f32; heads * head_dim];
        for (position, row) in row_refs.iter().enumerate() {
            let slice = &weights[position * heads..(position + 1) * heads];
            axpy_f16_f32_heads_exact(&mut per_position, head_dim, slice, row);
        }
        assert_eq!(block.len(), per_position.len());
        for (lane, (block, single)) in block.iter().zip(&per_position).enumerate() {
            assert!(close(*block, *single, 1e-6), "block, lane {lane}");
        }
    }
}
