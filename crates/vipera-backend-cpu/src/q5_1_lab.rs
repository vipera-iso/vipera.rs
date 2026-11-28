//! Scalar Q5_1 matvec prototype, kept as evidence and not wired into the engine.
//!
//! A `block_q5_1` is 24 bytes holding 32 values: `f16 d`, `f16 m`, `u32 qh`, and 16 bytes `qs` of
//! packed nibbles. Value `i` is `d * (nibble_i | high_bit_i << 4) + m`, where `nibble_i` is the low
//! half of `qs[i]` for `i < 16` and the high half of `qs[i - 16]` otherwise, and `high_bit_i` is bit
//! `i` of `qh`. The kernel decodes on the fly and accumulates in value order.

use crate::quant::f16_to_f32;

/// Values per Q5_1 block.
pub(crate) const QK5_1: usize = 32;
/// Bytes per Q5_1 block.
pub(crate) const BLOCK: usize = 24;

/// Dot one 24-byte Q5_1 block against 32 activations.
fn q5_1_block(block: &[u8; BLOCK], x: &[f32; QK5_1]) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let mut sum = 0.0_f32;
    for i in 0..QK5_1 {
        let byte = block[8 + i % 16];
        let nibble = if i < 16 { byte & 0x0f } else { byte >> 4 };
        let level = u32::from(nibble) | ((qh >> i) & 1) << 4;
        sum += (d * level as f32 + m) * x[i];
    }
    sum
}

/// Dot one Q5_1 row of a whole number of 32-value blocks with `x`.
pub(crate) fn q5_1_dot(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BLOCK, 0);
    debug_assert_eq!(x.len() * BLOCK, row.len() * QK5_1);
    let mut sum = 0.0_f32;
    for (block, xb) in row
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(x.as_chunks::<QK5_1>().0)
    {
        sum += q5_1_block(block, xb);
    }
    sum
}

/// Multiply a `rows x cols` Q5_1 matrix by `x`, writing one result per row.
pub(crate) fn q5_1_matvec(data: &[u8], cols: usize, rows: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cols % QK5_1, 0);
    debug_assert_eq!(out.len(), rows);
    let stride = cols / QK5_1 * BLOCK;
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = q5_1_dot(&data[row * stride..(row + 1) * stride], x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::f32_to_f16;

    /// Encode 32 known 5-bit levels into one Q5_1 block.
    fn encode(levels: &[i32; QK5_1], d: f32, m: f32) -> [u8; BLOCK] {
        let mut block = [0_u8; BLOCK];
        block[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(m).to_le_bytes());
        let mut qh = 0_u32;
        for (i, &level) in levels.iter().enumerate() {
            assert!((0..32).contains(&level));
            if level & 0x10 != 0 {
                qh |= 1 << i;
            }
            let nibble = (level & 0x0f) as u8;
            let at = 8 + i % 16;
            if i < 16 {
                block[at] |= nibble;
            } else {
                block[at] |= nibble << 4;
            }
        }
        block[4..8].copy_from_slice(&qh.to_le_bytes());
        block
    }

    fn expected(levels: &[i32; QK5_1], d: f32, m: f32, x: &[f32; QK5_1]) -> f32 {
        levels
            .iter()
            .zip(x)
            .map(|(&level, &xv)| (d * level as f32 + m) * xv)
            .sum()
    }

    #[test]
    fn q5_1_block_decodes_scale_min_nibbles_and_high_bits() {
        // Covers both nibble halves and a mix of set and clear high bits across `qh`.
        let levels: [i32; QK5_1] = std::array::from_fn(|i| ((i * 5 + 3) % 32) as i32);
        let (d, m) = (0.25_f32, -1.5_f32); // both exactly representable in f16
        let block = encode(&levels, d, m);
        let x: [f32; QK5_1] = std::array::from_fn(|i| (i as f32 - 8.0) / 4.0);
        let got = q5_1_block(&block, &x);
        let want = expected(&levels, d, m, &x);
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn q5_1_matvec_matches_per_row_dot() {
        let cols = QK5_1 * 3;
        let rows = 5;
        let xs: Vec<f32> = (0..cols).map(|i| (i as f32 - 40.0) / 8.0).collect();
        let mut data = Vec::new();
        for r in 0..rows {
            for b in 0..cols / QK5_1 {
                let levels: [i32; QK5_1] =
                    std::array::from_fn(|i| (((i + r * 3 + b * 7) * 5 + 3) % 32) as i32);
                data.extend_from_slice(&encode(&levels, 0.5, -2.0));
            }
        }
        let mut out = vec![0.0_f32; rows];
        q5_1_matvec(&data, cols, rows, &xs, &mut out);
        let stride = cols / QK5_1 * BLOCK;
        for (row, got) in out.iter().enumerate() {
            let want = q5_1_dot(&data[row * stride..(row + 1) * stride], &xs);
            assert!(
                (got - want).abs() < 1e-6,
                "row {row}: got {got}, want {want}"
            );
        }
    }
}
