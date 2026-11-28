//! Scalar Q8_1 matvec prototype, kept as evidence and not wired into the engine.
//!
//! A 36-byte block holds 32 signed 8-bit values preceded by two little-endian `f16` values: a scale
//! `d` and a minimum/offset `m`. Value `i` is `d * qs[i] + m`; `m` carries its own sign, so a
//! Q4_K-derived block stores a negated minimum and subtracts it. The kernel decodes on the fly and
//! accumulates in value order, giving a later vector path a bit-exact reference.

use crate::quant::f16_to_f32;

/// Values per Q8_1 block.
pub(crate) const QK8_1: usize = 32;
/// Bytes per Q8_1 block: two `f16` plus 32 `i8`.
pub(crate) const BLOCK: usize = 36;

const _: () = assert!(BLOCK == 2 * 2 + QK8_1);

/// Dot one 36-byte Q8_1 block against 32 activations.
fn q8_1_block(block: &[u8; BLOCK], x: &[f32; QK8_1]) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut sum = 0.0_f32;
    for (byte, value) in block[4..].iter().zip(x) {
        sum += (d * f32::from(*byte as i8) + m) * *value;
    }
    sum
}

/// Dot one Q8_1 row of a whole number of 32-value blocks with `x`.
pub(crate) fn q8_1_dot(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BLOCK, 0);
    debug_assert_eq!(x.len() * BLOCK, row.len() * QK8_1);
    let mut sum = 0.0_f32;
    for (block, xb) in row
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(x.as_chunks::<QK8_1>().0)
    {
        sum += q8_1_block(block, xb);
    }
    sum
}

/// Multiply a `rows x cols` Q8_1 matrix by `x`, writing one result per row.
pub(crate) fn q8_1_matvec(data: &[u8], cols: usize, rows: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cols % QK8_1, 0);
    debug_assert_eq!(out.len(), rows);
    let stride = cols / QK8_1 * BLOCK;
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = q8_1_dot(&data[row * stride..(row + 1) * stride], x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::f32_to_f16;

    /// Encode 32 signed codes into one Q8_1 block.
    fn encode(codes: &[i8; QK8_1], d: f32, m: f32) -> [u8; BLOCK] {
        let mut block = [0_u8; BLOCK];
        block[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(m).to_le_bytes());
        for (slot, code) in block[4..].iter_mut().zip(codes) {
            *slot = *code as u8;
        }
        block
    }

    fn expected(codes: &[i8; QK8_1], d: f32, m: f32, x: &[f32; QK8_1]) -> f32 {
        codes
            .iter()
            .zip(x)
            .map(|(&code, &xv)| (d * f32::from(code) + m) * xv)
            .sum()
    }

    #[test]
    fn q8_1_block_decodes_scale_min_and_signed_codes() {
        let codes: [i8; QK8_1] = std::array::from_fn(|i| (i as i8 % 11) - 5);
        let (d, m) = (0.25_f32, -1.5_f32); // both exactly representable in f16
        let block = encode(&codes, d, m);
        let x: [f32; QK8_1] = std::array::from_fn(|i| (i as f32 - 16.0) / 4.0);
        assert_eq!(q8_1_block(&block, &x), expected(&codes, d, m, &x));
        assert!(codes.iter().any(|&code| code < 0));
        assert_eq!(q8_1_block(&block, &x), q8_1_dot(&block, &x));
    }

    #[test]
    fn q8_1_matvec_matches_per_row_dots() {
        const COLS: usize = QK8_1 * 3;
        const ROWS: usize = 5;
        let xs: Vec<f32> = (0..COLS).map(|i| (i as f32 - 40.0) / 8.0).collect();
        let mut data = Vec::new();
        for r in 0..ROWS {
            for b in 0..COLS / QK8_1 {
                let codes: [i8; QK8_1] =
                    std::array::from_fn(|i| ((i + r * 3 + b * 7) % 17) as i8 - 8);
                data.extend_from_slice(&encode(&codes, 0.5, -2.0));
            }
        }
        let mut out = vec![0.0_f32; ROWS];
        q8_1_matvec(&data, COLS, ROWS, &xs, &mut out);
        let stride = COLS / QK8_1 * BLOCK;
        for (row, got) in out.iter().enumerate() {
            let want = q8_1_dot(&data[row * stride..(row + 1) * stride], &xs);
            assert_eq!(*got, want, "row {row}");
        }
        assert!(out.iter().any(|&value| value != out[0]), "rows differ");
    }
}
