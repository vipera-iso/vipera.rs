//! Tiled scalar matmul for prefill, gated by `VIPERA_TILED=1`.
//!
//! Portable Rust only (no `unsafe`): a K-tile of 256 columns is unpacked once
//! into a `[16][256]` `f32` tile and reused for every activation in the batch,
//! with a 16x16 micro-tile accumulation inside. Two flavours live here:
//!
//! - [`tiled_matvec_batch`] works on dequantized `f32` weights (tests, diagnostics).
//! - [`matvec_tiled_batch`] is the real prefill path: it unpacks the quantized
//!   bytes tile by tile and never materializes the whole tensor as `f32`
//!   (the old version allocated ~5 GB for the 191 MB `output` tensor).
//!
//! The default path (gate off) stays bit-exact via `dot`/`dot_batch`.
//! The tiled path changes summation order and is approximate
//! (`max_abs <= 1e-3 + 1e-4 * max|ref|`).

use vipera_core::gguf::GgmlType;
use vipera_errors::ViperaError;

use crate::quant::{f16_to_f32, q4k_scale_min};

/// Whether the tiled path is requested. `VIPERA_TILED=1` turns it on;
/// anything else (unset included) keeps the existing pooled-bit-exact path.
pub(crate) fn tiled_enabled() -> bool {
    std::env::var("VIPERA_TILED").is_ok_and(|value| value == "1")
}

/// Whether `B`-entry batches may take the tiled path (`B >= 8`).
///
/// `MAX_BATCH` is 16, so 32 would never fire; the point of the tile is to
/// share one unpack across many activations, which 8 already gives, and the
/// gate keeps the bit-exact pooled kernel as the default.
pub(crate) fn tiled_serves<const B: usize>() -> bool {
    tiled_enabled() && B >= 8
}

fn read_f16(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Dequantize one 256-wide slice of a row into `dst` (length >= 256).
///
/// `row` is the 256-block at its start (caller already slices to the K-tile).
/// Matches [`crate::quant::read_row`] value for value.
fn dequant_slice(ty: GgmlType, row: &[u8], _k0: usize, dst: &mut [f32]) -> Result<(), ViperaError> {
    match ty {
        GgmlType::F32 => {
            for (i, v) in dst.iter_mut().enumerate() {
                let at = i * 4;
                *v = f32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]);
            }
        }
        GgmlType::F16 => {
            for (i, v) in dst.iter_mut().enumerate() {
                let at = i * 2;
                *v = read_f16(row, at);
            }
        }
        GgmlType::Q8_0 => {
            // 256 values = 8 blocks of 32.
            for (b, chunk) in dst.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                let block = &row[b * 34..b * 34 + 34];
                let d = read_f16(block, 0);
                for (v, byte) in chunk.iter_mut().zip(block[2..34].iter()) {
                    *v = d * f32::from(*byte as i8);
                }
            }
        }
        GgmlType::Q4K => {
            let block = &row[..144];
            let d = read_f16(block, 0);
            let dmin = read_f16(block, 2);
            let scales = &block[4..16];
            for (j, x64) in dst.as_chunks_mut::<64>().0.iter_mut().enumerate() {
                let q = &block[16 + j * 32..16 + j * 32 + 32];
                let (s0, m0) = q4k_scale_min(scales, 2 * j);
                let (s1, m1) = q4k_scale_min(scales, 2 * j + 1);
                let d0 = d * f32::from(s0);
                let mm0 = dmin * f32::from(m0);
                let d1 = d * f32::from(s1);
                let mm1 = dmin * f32::from(m1);
                for (i, &byte) in q.iter().enumerate() {
                    x64[i] = d0 * f32::from(byte & 0x0f) - mm0;
                    x64[i + 32] = d1 * f32::from(byte >> 4) - mm1;
                }
            }
        }
        GgmlType::Q6K => {
            let block = &row[..210];
            let d = read_f16(block, 208);
            let scales = &block[192..208];
            for half in 0..2 {
                let ql = &block[half * 64..half * 64 + 64];
                let qh = &block[128 + half * 32..128 + half * 32 + 32];
                let out128 = &mut dst[half * 128..half * 128 + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let scale = |idx: usize| d * f32::from(scales[half * 8 + idx] as i8);
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
        other => {
            return Err(ViperaError::internal(format!(
                "tiled: unsupported {other:?}"
            )));
        }
    }
    Ok(())
}

/// F32 tiled GEMM: `out[entry * rows + row] = sum_c weights[row * cols + c] * activations[entry][c]`.
#[allow(dead_code)] // kept for diagnostics/tests; real path is matvec_tiled_batch
pub(crate) fn tiled_matvec_batch<const B: usize>(
    weights: &[f32],
    cols: usize,
    rows: usize,
    activations: &[&[f32]; B],
    out: &mut [f32],
) -> Result<(), ViperaError> {
    if cols == 0 || rows == 0 {
        return Err(ViperaError::invalid_argument(
            "tiled matvec needs non-zero cols and rows",
        ));
    }
    for (entry, x) in activations.iter().enumerate() {
        if x.len() != cols {
            return Err(ViperaError::internal(format!(
                "tiled batch entry {entry} has {} values, expected {cols}",
                x.len()
            )));
        }
    }
    if out.len() != B.saturating_mul(rows) {
        return Err(ViperaError::internal(format!(
            "tiled batch of {rows} rows needs {} outputs, got {}",
            B.saturating_mul(rows),
            out.len()
        )));
    }
    if weights.len() < cols.saturating_mul(rows) {
        return Err(ViperaError::invalid_argument("tiled weights are too few"));
    }
    for value in out.iter_mut() {
        *value = 0.0;
    }
    let mut tile = [[0.0_f32; 256]; 16];
    let mut acc = [[0.0_f32; 16]; B];
    let mut row0 = 0;
    while row0 < rows {
        let row_lim = (row0 + 16).min(rows);
        for acc_row in acc.iter_mut() {
            acc_row.fill(0.0);
        }
        let mut k0 = 0;
        while k0 < cols {
            let k_lim = (k0 + 256).min(cols);
            let k_len = k_lim - k0;
            for row in tile.iter_mut() {
                row.fill(0.0);
            }
            for (slot, row) in tile.iter_mut().enumerate().take(row_lim - row0) {
                let base = (row0 + slot) * cols + k0;
                row[..k_len].copy_from_slice(&weights[base..base + k_len]);
            }
            let mut k1 = 0;
            while k1 < k_len {
                let k1_lim = (k1 + 16).min(k_len);
                for (slot, row) in tile.iter().enumerate().take(row_lim - row0) {
                    let weights16 = &row[k1..k1_lim];
                    for (entry, x) in activations.iter().enumerate() {
                        let slice = &x[k0 + k1..k0 + k1_lim];
                        let partial: f32 =
                            weights16.iter().zip(slice.iter()).map(|(w, v)| w * v).sum();
                        acc[entry][slot] += partial;
                    }
                }
                k1 = k1_lim;
            }
            k0 = k_lim;
        }
        for (slot, _) in tile.iter().enumerate().take(row_lim - row0) {
            for (entry, acc_row) in acc.iter().enumerate() {
                out[entry * rows + row0 + slot] = acc_row[slot];
            }
        }
        row0 = row_lim;
    }
    Ok(())
}

/// Quantized tiled GEMM, token-major out. Same contract as
/// [`crate::quant::matvec_pooled_batch`] but tiled.
///
/// K-tile 256 outer, row-tile 16, weights unpacked once per (row,k) tile and
/// reused across all `B` activations. Never expands the matrix to `f32`.
/// Caller validates shapes; this checks the buffer sizes again.
///
/// Optimizer note: the per-slot loops below previously recomputed
/// `row0 + slot` / `k0 + k1` and called `.iter().zip(...).sum()` for every
/// partial (breaking auto-vectorization and defeating the 16×256 tile reuse).
/// The hot path now uses raw pointers + offset arithmetic so the compiler
/// can hoist base pointers and vectorize the inner 16-wide FMA; results are
/// identical within the documented approximate tolerance.
#[allow(clippy::needless_range_loop)]
pub(crate) fn matvec_tiled_batch<const B: usize>(
    ty: GgmlType,
    data: &[u8],
    stride: usize,
    cols: usize,
    rows: usize,
    activations: &[&[f32]; B],
    out: &mut [f32],
) -> Result<(), ViperaError> {
    if cols == 0 || rows == 0 {
        return Err(ViperaError::invalid_argument(
            "tiled matvec needs non-zero cols and rows",
        ));
    }
    if !cols.is_multiple_of(256) {
        return Err(ViperaError::invalid_argument(
            "tiled matvec needs cols to be a multiple of 256",
        ));
    }
    if B == 0 || B > 16 {
        return Err(ViperaError::invalid_argument(
            "tiled matvec needs a batch of 1 to 16 entries",
        ));
    }
    let expected_stride = crate::quant::row_bytes(ty, cols)?;
    if stride != expected_stride {
        return Err(ViperaError::invalid_argument(
            "tiled matvec stride does not match the weight type",
        ));
    }
    let needed = stride
        .checked_mul(rows)
        .ok_or_else(|| ViperaError::invalid_argument("tiled matvec byte count overflows"))?;
    if data.len() < needed {
        return Err(ViperaError::invalid_argument(
            "tiled matvec data is too short",
        ));
    }
    if out.len() != B.saturating_mul(rows) {
        return Err(ViperaError::invalid_argument(
            "tiled matvec output is the wrong size",
        ));
    }
    for (entry, x) in activations.iter().enumerate() {
        if x.len() != cols {
            return Err(ViperaError::invalid_argument(format!(
                "tiled matvec entry {entry} has {} values, expected {cols}",
                x.len()
            )));
        }
    }
    for value in out.iter_mut() {
        *value = 0.0;
    }
    let mut w_tile = [[0.0_f32; 256]; 16];
    let mut x_tile = [[0.0_f32; 256]; 16];
    let mut acc = [[0.0_f32; 16]; 16];
    let n_entries = B.min(16);
    let mut row0 = 0;
    let block_bytes = stride / (cols / 256);
    while row0 < rows {
        let row_cnt = (rows - row0).min(16);
        for entry in 0..n_entries {
            acc[entry][..row_cnt].fill(0.0);
        }
        let mut k0 = 0;
        while k0 < cols {
            // Unpack weight tile [row_cnt][256] once: the 256-block at (k0/256).
            for slot in 0..row_cnt {
                let row_offset = (row0 + slot) * stride;
                let row = &data[row_offset..row_offset + stride];
                // dequant_slice expects the 256-block at its start; for k0>0
                // take the block at (k0/256)*block_bytes.
                let block_off = (k0 / 256) * block_bytes;
                dequant_slice(ty, &row[block_off..], k0, &mut w_tile[slot])?;
            }
            // Pack activation tile.
            for entry in 0..n_entries {
                x_tile[entry][..256].copy_from_slice(&activations[entry][k0..k0 + 256]);
            }
            // 16x16 micro-tile accumulate: safe indexing lets LLVM
            // auto-vectorize the 16-wide inner loop (was `.iter().zip().sum()`
            // closure, which forced scalar + opaque call per partial).
            let mut k1 = 0;
            while k1 < 256 {
                let k1_lim = (k1 + 16).min(256);
                for slot in 0..row_cnt {
                    for entry in 0..n_entries {
                        let mut partial = 0.0_f32;
                        for k in k1..k1_lim {
                            partial += w_tile[slot][k] * x_tile[entry][k];
                        }
                        acc[entry][slot] += partial;
                    }
                }
                k1 = k1_lim;
            }
            k0 += 256;
        }
        for slot in 0..row_cnt {
            for entry in 0..n_entries {
                out[entry * rows + row0 + slot] = acc[entry][slot];
            }
        }
        row0 += row_cnt;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_batch<const B: usize>(
        weights: &[f32],
        cols: usize,
        rows: usize,
        activations: &[&[f32]; B],
        out: &mut [f32],
    ) {
        for (entry, x) in activations.iter().enumerate() {
            for row in 0..rows {
                let mut sum = 0.0_f32;
                for c in 0..cols {
                    sum += weights[row * cols + c] * x[c];
                }
                out[entry * rows + row] = sum;
            }
        }
    }

    #[test]
    fn tiled_matches_naive_within_tolerance() {
        let (cols, rows) = (512, 32);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let weights: Vec<f32> = (0..cols * rows).map(|_| next()).collect();
        let xs: Vec<Vec<f32>> = (0..32)
            .map(|_| (0..cols).map(|_| next()).collect())
            .collect();
        let activations: [&[f32]; 32] = std::array::from_fn(|entry| xs[entry].as_slice());
        let mut got = vec![0.0; 32 * rows];
        let mut want = vec![0.0; 32 * rows];
        tiled_matvec_batch(&weights, cols, rows, &activations, &mut got).expect("tiled");
        naive_batch(&weights, cols, rows, &activations, &mut want);
        let max_ref = want.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let tol = 1e-3 + 1e-4 * max_ref;
        for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "at {index}: got {g}, want {w}, tol {tol}"
            );
        }
    }

    #[test]
    fn tiled_quant_refuses_bad_shapes_not_silent_zero() {
        // B > 16 previously silently left entries 16.. as 0.0 with Ok.
        let cols = 256;
        let rows = 2;
        let stride = crate::quant::row_bytes(GgmlType::F32, cols).expect("stride");
        let data = vec![0u8; stride * rows];
        let xs: Vec<Vec<f32>> = (0..20).map(|_| vec![1.0; cols]).collect();
        let activations: [&[f32]; 20] = std::array::from_fn(|e| xs[e].as_slice());
        let mut out = vec![0.0; 20 * rows];
        assert!(
            super::matvec_tiled_batch(
                GgmlType::F32,
                &data,
                stride,
                cols,
                rows,
                &activations,
                &mut out
            )
            .is_err()
        );
        // Short buffers must error, not panic.
        let xs8: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0; cols]).collect();
        let act8: [&[f32]; 8] = std::array::from_fn(|e| xs8[e].as_slice());
        let mut out8 = vec![0.0; 8 * rows];
        assert!(
            super::matvec_tiled_batch(GgmlType::F32, &[], stride, cols, rows, &act8, &mut out8)
                .is_err()
        );
        assert!(
            super::matvec_tiled_batch(
                GgmlType::F32,
                &data,
                stride,
                cols,
                rows,
                &act8,
                &mut [0.0; 4]
            )
            .is_err()
        );
        let bad: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0; 10]).collect();
        let bad_act: [&[f32]; 8] = std::array::from_fn(|e| bad[e].as_slice());
        assert!(
            super::matvec_tiled_batch(
                GgmlType::F32,
                &data,
                stride,
                cols,
                rows,
                &bad_act,
                &mut out8
            )
            .is_err()
        );
    }

    #[test]
    fn tiled_refuses_bad_shapes() {
        let cases: [&[f32]; 2] = [&[1.0], &[1.0]];
        let mut out = vec![0.0; 4];
        assert!(tiled_matvec_batch(&[1.0], 2, 2, &cases, &mut out).is_err());
    }

    #[test]
    fn quantized_tiled_matches_read_row() {
        // Q8_0: compare tile path against read_row dot for 256 cols.
        let cols = 256;
        let rows = 2;
        let stride = crate::quant::row_bytes(GgmlType::Q8_0, cols).expect("stride");
        let mut data = vec![0u8; stride * rows];
        for row in 0..rows {
            // d = 0.5, values 0..31
            data[row * stride..row * stride + 2]
                .copy_from_slice(&crate::quant::f32_to_f16(0.5).to_le_bytes());
            for i in 0..32 {
                data[row * stride + 2 + i] = i as u8;
            }
        }
        let xs: Vec<Vec<f32>> = (0..8).map(|_| vec![1.0; cols]).collect();
        let activations: [&[f32]; 8] = std::array::from_fn(|e| xs[e].as_slice());
        let mut got = vec![0.0; 8 * rows];
        matvec_tiled_batch(
            GgmlType::Q8_0,
            &data,
            stride,
            cols,
            rows,
            &activations,
            &mut got,
        )
        .expect("tiled quant");
        // Expected: sum of d * i for i in 0..32 = 0.5 * 496 = 248 per row.
        for entry in 0..8 {
            for row in 0..rows {
                let v = got[entry * rows + row];
                assert!((v - 248.0).abs() < 1e-3, "entry {entry} row {row}: got {v}");
            }
        }
    }

    #[test]
    fn f32_tiled_wide_cols_matches_read_row() {
        // cols=512 crosses two K-tiles: second tile must read block 1, not
        // index k0 into the already-sliced row (would panic out of bounds).
        let cols = 512;
        let rows = 2;
        let stride = crate::quant::row_bytes(GgmlType::F32, cols).expect("stride");
        let mut data = vec![0u8; stride * rows];
        for row in 0..rows {
            for c in 0..cols {
                let v = (c % 7) as f32 / 7.0 - 0.5;
                let at = row * stride + c * 4;
                data[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let xs: Vec<Vec<f32>> = (0..8)
            .map(|e| {
                (0..cols)
                    .map(|c| ((c + e) % 5) as f32 / 5.0 - 0.4)
                    .collect()
            })
            .collect();
        let activations: [&[f32]; 8] = std::array::from_fn(|e| xs[e].as_slice());
        let mut got = vec![0.0; 8 * rows];
        super::matvec_tiled_batch(
            GgmlType::F32,
            &data,
            stride,
            cols,
            rows,
            &activations,
            &mut got,
        )
        .expect("tiled f32");
        for (entry, x) in xs.iter().enumerate() {
            for row in 0..rows {
                let mut want_row = vec![0.0; cols];
                crate::quant::read_row(GgmlType::F32, &data, cols, row, &mut want_row)
                    .expect("read_row");
                let want: f32 = want_row.iter().zip(x).map(|(w, v)| w * v).sum();
                let g = got[entry * rows + row];
                assert!(
                    (g - want).abs() < 1e-3,
                    "entry {entry} row {row}: got {g} want {want}"
                );
            }
        }
    }

    #[test]
    fn f16_tiled_wide_cols_matches_read_row() {
        let cols = 512;
        let rows = 2;
        let stride = crate::quant::row_bytes(GgmlType::F16, cols).expect("stride");
        let mut data = vec![0u8; stride * rows];
        for row in 0..rows {
            for c in 0..cols {
                let v = (c % 7) as f32 / 7.0 - 0.5;
                let at = row * stride + c * 2;
                data[at..at + 2].copy_from_slice(&crate::quant::f32_to_f16(v).to_le_bytes());
            }
        }
        let xs: Vec<Vec<f32>> = (0..8)
            .map(|e| {
                (0..cols)
                    .map(|c| ((c + e) % 5) as f32 / 5.0 - 0.4)
                    .collect()
            })
            .collect();
        let activations: [&[f32]; 8] = std::array::from_fn(|e| xs[e].as_slice());
        let mut got = vec![0.0; 8 * rows];
        super::matvec_tiled_batch(
            GgmlType::F16,
            &data,
            stride,
            cols,
            rows,
            &activations,
            &mut got,
        )
        .expect("tiled f16");
        for (entry, x) in xs.iter().enumerate() {
            for row in 0..rows {
                let mut want_row = vec![0.0; cols];
                crate::quant::read_row(GgmlType::F16, &data, cols, row, &mut want_row)
                    .expect("read_row");
                let want: f32 = want_row.iter().zip(x).map(|(w, v)| w * v).sum();
                let g = got[entry * rows + row];
                assert!(
                    (g - want).abs() < 1e-3,
                    "entry {entry} row {row}: got {g} want {want}"
                );
            }
        }
    }

    #[test]
    fn q4k_tiled_matches_scalar_q4k_dot_within_tolerance() {
        use crate::quant::{q4k_dot, x_sums};
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 7;
            state ^= state >> 3;
            state ^= state << 8;
            state
        };
        let cols = 512;
        let rows = 4;
        let stride = crate::quant::row_bytes(GgmlType::Q4K, cols).expect("stride");
        let mut data = vec![0u8; stride * rows];
        for row in 0..rows {
            for b in 0..(cols / 256) {
                let off = row * stride + b * 144;
                data[off..off + 2].copy_from_slice(&crate::quant::f32_to_f16(0.02).to_le_bytes());
                data[off + 2..off + 4]
                    .copy_from_slice(&crate::quant::f32_to_f16(0.01).to_le_bytes());
                for i in 0..12 {
                    data[off + 4 + i] = (next() % 64) as u8;
                }
                for i in 0..128 {
                    data[off + 16 + i] = next() as u8;
                }
            }
        }
        let xs: Vec<Vec<f32>> = (0..8)
            .map(|_| {
                (0..cols)
                    .map(|_| (((next() % 2000) as f32) / 1000.0) - 1.0)
                    .collect()
            })
            .collect();
        let activations: [&[f32]; 8] = std::array::from_fn(|e| xs[e].as_slice());
        let mut got = vec![0.0; 8 * rows];
        matvec_tiled_batch(
            GgmlType::Q4K,
            &data,
            stride,
            cols,
            rows,
            &activations,
            &mut got,
        )
        .expect("tiled q4k");
        for (entry, x) in xs.iter().enumerate() {
            let sums = x_sums(x);
            for row in 0..rows {
                let want = q4k_dot(&data[row * stride..(row + 1) * stride], x, &sums);
                let g = got[entry * rows + row];
                let tol = 1e-3 + 1e-4 * want.abs();
                assert!(
                    (g - want).abs() <= tol,
                    "entry {entry} row {row}: got {g} want {want}"
                );
            }
        }
    }

    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn q4k_tiled_gbs_bench() {
        use std::time::Instant;
        // Same LCG fixture as pooled throughput tests: real shape ffn_gate 8960 x 1536.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        let cols = 1536;
        let rows = 8960;
        let stride = crate::quant::row_bytes(GgmlType::Q4K, cols).expect("stride");
        let data: Vec<u8> = (0..stride * rows).map(|_| next()).collect();
        let xs: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..cols).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect())
            .collect();
        let activations: [&[f32]; 16] = std::array::from_fn(|e| xs[e].as_slice());
        let mut out = vec![0.0; 16 * rows];
        let start = Instant::now();
        let reps = 3;
        for _ in 0..reps {
            out.fill(0.0);
            matvec_tiled_batch(
                GgmlType::Q4K,
                &data,
                stride,
                cols,
                rows,
                &activations,
                &mut out,
            )
            .expect("bench");
        }
        let secs = start.elapsed().as_secs_f64();
        let gb = (stride * rows) as f64 * reps as f64 / 1e9;
        println!(
            "tiled q4k scalar: {:.2} GB/s (pack-free, scalar)",
            gb / secs
        );
        std::hint::black_box(&out);
    }
}
