//! KV compression in the format this backend's cache already stores.
//!
//! `--kv-type q8_0` keeps a cached position as one half-precision scale per 32 values plus their
//! signed codes: 34 bytes per 32 values, a quarter of the `f32` they came from. [`Q8Kv`] is the
//! same arithmetic behind `KvCompression`, so a build can register the cache's own format as its
//! compression policy instead of a second, differently-rounded one.

use vipera_runtime::engine::KvCompression;

use crate::quant::{f16_to_f32, f32_to_f16};

/// Values per q8_0 block: one scale, one code each.
const VALUES_PER_BLOCK: usize = 32;

/// Bytes one q8_0 block occupies: the half-precision scale, then the codes.
const BYTES_PER_BLOCK: usize = 34;

/// The CPU backend's [`KvCompression`]: the format its KV cache already stores at `--kv-type q8_0`.
///
/// One scale per 32 values and one signed byte per value, so a block costs a quarter of the `f32`
/// row plus two bytes, and a value comes back within half a step. A block whose length is not a
/// whole number of 32-value blocks is left as it came (`0`), because the format cannot express a
/// partial block, and so is a block holding a non-finite value, because a linear scale cannot.
#[derive(Debug, Default, Clone, Copy)]
pub struct Q8Kv;

impl KvCompression for Q8Kv {
    fn name(&self) -> &'static str {
        "q8_0"
    }

    fn compress(&self, block: &[f32], out: &mut Vec<u8>) -> usize {
        if block.is_empty()
            || !block.len().is_multiple_of(VALUES_PER_BLOCK)
            || block.iter().any(|value| !value.is_finite())
        {
            return 0;
        }
        let start = out.len();
        out.resize(start + block.len() / VALUES_PER_BLOCK * BYTES_PER_BLOCK, 0);
        encode_q8_0(block, &mut out[start..]);
        out.len() - start
    }

    fn decompress(&self, compressed: &[u8], out: &mut [f32]) -> usize {
        if out.is_empty()
            || !out.len().is_multiple_of(VALUES_PER_BLOCK)
            || compressed.len() < out.len() / VALUES_PER_BLOCK * BYTES_PER_BLOCK
        {
            return 0;
        }
        decode_q8_0(compressed, out);
        out.len()
    }
}

/// Write `values` as q8_0 into `out`: one `f16` scale per 32 values, then their signed codes.
///
/// `values.len()` must be a whole number of 32-value blocks and `out` must hold 34 bytes per block.
/// Callers check both, since the cache reports a bad row as an error and the compression policy as
/// a block it will not encode.
pub(crate) fn encode_q8_0(values: &[f32], out: &mut [u8]) {
    debug_assert!(values.len().is_multiple_of(VALUES_PER_BLOCK));
    debug_assert!(out.len() >= values.len() / VALUES_PER_BLOCK * BYTES_PER_BLOCK);
    for (block, chunk) in out
        .as_chunks_mut::<BYTES_PER_BLOCK>()
        .0
        .iter_mut()
        .zip(values.as_chunks::<VALUES_PER_BLOCK>().0)
    {
        let largest = chunk
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        let scale = if largest == 0.0 { 0.0 } else { largest / 127.0 };
        block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        for (slot, value) in block[2..].iter_mut().zip(chunk) {
            *slot = (value * inverse).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

/// Read q8_0 blocks from `source` into `out`: `34` bytes of source per 32 values.
///
/// The same length contract as [`encode_q8_0`], checked by the callers.
pub(crate) fn decode_q8_0(source: &[u8], out: &mut [f32]) {
    debug_assert!(out.len().is_multiple_of(VALUES_PER_BLOCK));
    debug_assert!(source.len() >= out.len() / VALUES_PER_BLOCK * BYTES_PER_BLOCK);
    for (chunk, block) in out
        .as_chunks_mut::<VALUES_PER_BLOCK>()
        .0
        .iter_mut()
        .zip(source.as_chunks::<BYTES_PER_BLOCK>().0)
    {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (value, quantized) in chunk.iter_mut().zip(&block[2..]) {
            *value = scale * f32::from(*quantized as i8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values of both signs, spread wide enough that one q8_0 step is not a rounding artefact.
    fn row(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| (index as f32 * 0.37).sin() * 3.0)
            .collect()
    }

    #[test]
    fn a_row_round_trips_within_half_a_step() {
        let values = row(64);
        let mut bytes = Vec::new();
        assert_eq!(Q8Kv.compress(&values, &mut bytes), 34 * 2);
        assert_eq!(bytes.len(), 34 * 2);

        let mut restored = vec![f32::NAN; values.len()];
        assert_eq!(Q8Kv.decompress(&bytes, &mut restored), values.len());
        let largest = values
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        // Half a step from the rounding to a code, plus what rounding the scale to `f16` costs.
        let tolerance = largest / 127.0 / 2.0 + largest * 0.001;
        for (value, restored) in values.iter().zip(&restored) {
            assert!(
                (value - restored).abs() <= tolerance,
                "{value} came back as {restored}"
            );
        }
    }

    #[test]
    fn a_zero_row_is_exact_and_still_costs_a_quarter() {
        let values = vec![0.0_f32; 32];
        let mut bytes = Vec::new();
        assert_eq!(Q8Kv.compress(&values, &mut bytes), 34);
        let mut restored = vec![f32::NAN; 32];
        assert_eq!(Q8Kv.decompress(&bytes, &mut restored), 32);
        assert!(restored.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn a_trailing_partial_block_is_left_as_it_came() {
        let mut bytes = Vec::new();
        assert_eq!(Q8Kv.compress(&row(40), &mut bytes), 0);
        assert!(bytes.is_empty());
        assert_eq!(Q8Kv.decompress(&bytes, &mut [0.0_f32; 40]), 0);
    }

    #[test]
    fn a_row_holding_a_non_finite_value_is_left_as_it_came() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = row(32);
            values[5] = bad;
            let mut bytes = Vec::new();
            assert_eq!(Q8Kv.compress(&values, &mut bytes), 0, "{bad} was encoded");
            assert!(bytes.is_empty());
        }
    }

    #[test]
    fn a_short_buffer_is_not_restored() {
        let values = row(32);
        let mut bytes = Vec::new();
        assert_eq!(Q8Kv.compress(&values, &mut bytes), 34);
        assert_eq!(Q8Kv.decompress(&bytes[..33], &mut [0.0_f32; 32]), 0);
    }

    #[test]
    fn the_policy_names_the_cache_format_and_is_a_trait_object() {
        let policy: &dyn KvCompression = &Q8Kv;
        assert_eq!(policy.name(), "q8_0");
    }
}
