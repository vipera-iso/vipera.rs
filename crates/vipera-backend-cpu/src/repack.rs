//! Pha 1 repack Q4_K prototype: pack 8 rows into blocked layout, dot scalar parity test.
//!
//! Layout: `q` unpacked to `u8` 0..15 per value, row-major `8 * cols`; per-row
//! per-block coefficients precomputed (`coeff`/`min_coeff`, 8 groups of 32 per
//! 256 block). Parity reference is [`crate::quant::q4k_dot`].

use vipera_errors::ViperaError;

/// Packed 8-row tile: unpacked nibbles + per-group float coefficients.
#[derive(Debug, Clone)]
pub(crate) struct PackedTile {
    /// Unpacked weights, `8 * cols`, row-major.
    pub q: Vec<u8>,
    /// `8 * blocks * 8` floats: `d * scale` per 32-value group.
    pub coeff: Vec<f32>,
    /// `8 * blocks * 8` floats: `dmin * min` per 32-value group.
    pub min_coeff: Vec<f32>,
    /// Number of 256-value blocks per row.
    pub blocks: usize,
    /// Columns per row (`blocks * 256`).
    pub cols: usize,
}

impl PackedTile {
    /// Interleaved view (phòng thí nghiệm VNNI #27851): sắp xếp lại `q` thành
    /// `k*8 + r` để 8 hàng cùng cột nằm liên tục — activation `x` load 1 lần
    /// dùng cho 8 hàng (reuse 8x giống microkernel 16x16 của llama.cpp).
    /// `coeff`/`min_coeff` giữ nguyên row-major để dễ parity.
    pub fn interleaved_q(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.cols * 8];
        for row in 0..8 {
            for k in 0..self.cols {
                out[k * 8 + row] = self.q[row * self.cols + k];
            }
        }
        out
    }
}

/// Pack 8 rows of Q4_K bytes into the prototype blocked layout.
pub(crate) fn pack_q4k8(rows: &[&[u8]; 8], cols: usize) -> Result<PackedTile, ViperaError> {
    if cols == 0 || !cols.is_multiple_of(256) {
        return Err(ViperaError::invalid_argument(
            "repack needs cols to be a multiple of 256",
        ));
    }
    let blocks = cols / 256;
    let mut q = vec![0u8; 8 * cols];
    let mut coeff = vec![0.0_f32; 8 * blocks * 8];
    let mut min_coeff = vec![0.0_f32; 8 * blocks * 8];
    for (row, bytes) in rows.iter().enumerate() {
        if bytes.len() < blocks * 144 {
            return Err(ViperaError::internal("repack row too short"));
        }
        for b in 0..blocks {
            let block = &bytes[b * 144..b * 144 + 144];
            let d = crate::quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = crate::quant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            for j in 0..8 {
                let (s, m) = crate::quant::q4k_scale_min(scales, j);
                coeff[(row * blocks + b) * 8 + j] = d * f32::from(s);
                min_coeff[(row * blocks + b) * 8 + j] = dmin * f32::from(m);
            }
            let nibbles = &block[16..144];
            let dst_base = row * cols + b * 256;
            for (j, chunk) in nibbles.as_chunks::<32>().0.iter().enumerate() {
                let out = &mut q[dst_base + j * 64..dst_base + j * 64 + 64];
                for (i, &byte) in chunk.iter().enumerate() {
                    out[i] = byte & 0x0f;
                    out[i + 32] = byte >> 4;
                }
            }
        }
    }
    Ok(PackedTile {
        q,
        coeff,
        min_coeff,
        blocks,
        cols,
    })
}

/// Scalar dot of packed tile against `x` (256-wide block sums `sums`).
pub(crate) fn dot_packed(
    tile: &PackedTile,
    x: &[f32],
    sums: &[f32],
) -> Result<[f32; 8], ViperaError> {
    if x.len() != tile.cols {
        return Err(ViperaError::invalid_argument(
            "dot_packed row width mismatch",
        ));
    }
    if sums.len() < tile.blocks * 8 {
        return Err(ViperaError::invalid_argument("dot_packed sums too short"));
    }
    let mut out = [0.0_f32; 8];
    for (row, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for b in 0..tile.blocks {
            let base = row * tile.cols + b * 256;
            let coe = &tile.coeff[(row * tile.blocks + b) * 8..(row * tile.blocks + b) * 8 + 8];
            let min = &tile.min_coeff[(row * tile.blocks + b) * 8..(row * tile.blocks + b) * 8 + 8];
            for j in 0..8 {
                let start = base + j * 32;
                let mut acc = 0.0_f32;
                for k in 0..32 {
                    acc += f32::from(tile.q[start + k]) * x[b * 256 + j * 32 + k];
                }
                sum += coe[j] * acc - min[j] * sums[b * 8 + j];
            }
        }
        *slot = sum;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{q4k_dot, x_sums};

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
            crate::quant::f32_to_f16(v).to_le_bytes()
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

    #[test]
    fn packed_dot_matches_scalar_q4k_within_tolerance() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let cols = 512;
        let blocks = cols / 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_row(&mut rng, blocks)).collect();
        let refs: [&[u8]; 8] = std::array::from_fn(|i| rows[i].as_slice());
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let sums = x_sums(&x);
        let packed = pack_q4k8(&refs, cols).expect("pack");
        let got = dot_packed(&packed, &x, &sums).expect("dot");
        for (row, g) in got.iter().enumerate() {
            let want = q4k_dot(&rows[row], &x, &sums);
            let tol = 1e-3 + 1e-4 * want.abs();
            assert!((g - want).abs() <= tol, "row {row}: got {g}, want {want}");
        }
    }

    #[test]
    fn interleaved_layout_matches_row_major() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let cols = 256;
        let rows: Vec<Vec<u8>> = (0..8).map(|_| q4k_row(&mut rng, 1)).collect();
        let refs: [&[u8]; 8] = std::array::from_fn(|i| rows[i].as_slice());
        let packed = pack_q4k8(&refs, cols).expect("pack");
        let interleaved = packed.interleaved_q();
        // Verify: interleaved[k*8 + r] == row-major q[r*cols + k]
        for row in 0..8 {
            for k in 0..cols {
                assert_eq!(interleaved[k * 8 + row], packed.q[row * cols + k]);
            }
        }
        // Dot qua layout interleaved phải bằng dot_packed (reuse activation 8x về logic).
        let x: Vec<f32> = (0..cols).map(|_| rng.unit()).collect();
        let sums = x_sums(&x);
        let got = dot_packed(&packed, &x, &sums).expect("dot");
        for row in 0..8 {
            let want = q4k_dot(&rows[row], &x, &sums);
            let tol = 1e-3 + 1e-4 * want.abs();
            assert!((got[row] - want).abs() <= tol, "row {row}");
        }
    }

    #[test]
    fn pack_refuses_bad_width() {
        let rows: Vec<Vec<u8>> = (0..8).map(|_| vec![0u8; 144]).collect();
        let refs: [&[u8]; 8] = std::array::from_fn(|i| rows[i].as_slice());
        assert!(pack_q4k8(&refs, 100).is_err());
    }
}
