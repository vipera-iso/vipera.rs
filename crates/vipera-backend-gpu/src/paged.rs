//! The runtime's paged KV cache, as the device backend has to see it.
//!
//! The runtime owns one byte buffer and hands a backend a block table into it; a position `p` lives
//! in block `table[p / tokens_per_block]` at byte offset `(block * tokens_per_block + p %
//! tokens_per_block) * bytes_per_token`. This module turns that into the two byte offsets a layer
//! writes (K, then V), and the `f16` encode/decode the CPU backend writes there, so the device
//! backend's rows are the same rows: a block the runtime copies for a prefix-cache hit is meaningful
//! to either backend.
//!
//! Both row orders are implemented, resolved from the same `VIPERA_KV_LAYER_MAJOR` switch the CPU
//! backend reads: layer-major by default, token-major when the switch is off. The order is a property
//! of the buffer, not of the backend, so the two have to agree or a prefix-cache block stops meaning
//! the same thing on both sides.
//!
//! Everything here is host-side and allocation-free on the hot path, which is what makes it
//! testable without a CUDA device: the layout arithmetic is the part a GPU can silently get wrong.

use vipera_errors::ViperaError;

/// Bytes one `f16` KV value occupies.
const VALUE_BYTES: usize = 2;

/// Order the bytes of one block are laid out in.
///
/// Mirrors the CPU backend's `RowOrder`, including its default: `VIPERA_KV_LAYER_MAJOR` unset means
/// layer-major, and `0`/`false`/`off`/`no` selects token-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowOrder {
    /// Every layer of a token together, walked with a stride of a whole position.
    TokenMajor,
    /// Each layer's rows for the whole block together.
    LayerMajor,
}

impl RowOrder {
    /// The order the runtime's buffer is written in, from the same switch the CPU backend reads.
    pub(crate) fn from_env() -> Self {
        let layer_major = std::env::var("VIPERA_KV_LAYER_MAJOR").map_or(true, |value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        });
        if layer_major {
            Self::LayerMajor
        } else {
            Self::TokenMajor
        }
    }
}

/// Geometry of one position in the runtime's KV buffer, for this model.
///
/// Built per call from the [`vipera_runtime::ForwardContext`] the runtime passes, because the same
/// backend can be handed caches with different block sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvRows {
    /// Decoder layers whose K and V rows the buffer interleaves.
    layers: usize,
    /// Values in one K or V row: `kv_heads * head_dim`.
    width: usize,
    /// Positions per block.
    tokens_per_block: usize,
    /// Bytes one whole position occupies, every layer and both K and V.
    bytes_per_token: usize,
    /// Order the rows sit in.
    order: RowOrder,
}

impl KvRows {
    /// Check that the runtime's geometry is the `f16` token-major layout this backend writes.
    ///
    /// The error names the arithmetic rather than saying "unsupported": a mismatched
    /// `bytes_per_token` is either a `q8_0` plan or a different model, and the two need different
    /// fixes.
    pub(crate) fn new(
        layers: usize,
        width: usize,
        tokens_per_block: u64,
        bytes_per_token: u64,
    ) -> Result<Self, ViperaError> {
        Self::with_order(
            layers,
            width,
            tokens_per_block,
            bytes_per_token,
            RowOrder::from_env(),
        )
    }

    /// The same geometry in a named order, for the tests that check both.
    pub(crate) fn with_order(
        layers: usize,
        width: usize,
        tokens_per_block: u64,
        bytes_per_token: u64,
        order: RowOrder,
    ) -> Result<Self, ViperaError> {
        if layers == 0 || width == 0 {
            return Err(ViperaError::internal(
                "the device model has no layers or no KV width",
            ));
        }
        if tokens_per_block == 0 {
            return Err(ViperaError::invalid_argument(
                "a KV block must hold at least one token",
            ));
        }
        let expected = 2 * layers * Self::row_bytes(width);
        if bytes_per_token as usize != expected {
            return Err(ViperaError::unsupported(format!(
                "the device backend writes f16 KV: {layers} layers of {width} values, K and V, is \
                 {expected} bytes per position, and the cache holds {bytes_per_token}. Plan the \
                 cache with --kv-type f16 for the device backend."
            )));
        }
        Ok(Self {
            layers,
            width,
            tokens_per_block: tokens_per_block as usize,
            bytes_per_token: expected,
            order,
        })
    }

    /// Bytes one K or V row of `width` values occupies.
    pub(crate) const fn row_bytes(width: usize) -> usize {
        width * VALUE_BYTES
    }

    /// Values in one row.
    pub(crate) const fn width(self) -> usize {
        self.width
    }

    /// Layers the buffer covers.
    pub(crate) const fn layers(self) -> usize {
        self.layers
    }

    /// Byte offsets of layer `layer`'s K and V rows for `position`.
    ///
    /// Both orders start the same way — the position's block, and how far into it it is — and differ
    /// only in where the layer's rows sit inside that block.
    pub(crate) fn offsets(
        &self,
        layer: usize,
        position: usize,
        table: &[u32],
    ) -> Result<(usize, usize), ViperaError> {
        if layer >= self.layers {
            return Err(ViperaError::internal(format!(
                "layer {layer} is past the {0} layers the buffer holds",
                self.layers
            )));
        }
        let block = *table
            .get(position / self.tokens_per_block)
            .ok_or_else(|| ViperaError::internal("position has no KV block"))?
            as usize;
        let block_start = block * self.tokens_per_block * self.bytes_per_token;
        let in_block = position % self.tokens_per_block;
        let row_bytes = Self::row_bytes(self.width);
        let (k, v) = match self.order {
            RowOrder::TokenMajor => {
                let k = block_start + in_block * self.bytes_per_token + layer * 2 * row_bytes;
                (k, k + row_bytes)
            }
            RowOrder::LayerMajor => {
                let k = block_start
                    + layer * 2 * row_bytes * self.tokens_per_block
                    + in_block * row_bytes;
                (k, k + row_bytes * self.tokens_per_block)
            }
        };
        Ok((k, v))
    }

    /// Blocks a table needs to address `positions`.
    pub(crate) fn blocks(&self, positions: usize) -> usize {
        positions.div_ceil(self.tokens_per_block)
    }

    /// Bytes the buffer must have to cover every block in `table`.
    pub(crate) fn reach(&self, table: &[u32]) -> usize {
        table.iter().copied().max().map_or(0, |block| {
            (block as usize + 1) * self.tokens_per_block * self.bytes_per_token
        })
    }

    /// Whether the table names the same block twice, which only a ring cache does.
    ///
    /// The device attention kernels walk positions `0..n` in one dense cache, so a table whose
    /// entries repeat would have this backend read rows the window has already overwritten. It is
    /// refused by name instead of run approximately.
    pub(crate) fn repeats(table: &[u32]) -> bool {
        let mut seen: Vec<u32> = table.to_vec();
        seen.sort_unstable();
        seen.dedup();
        seen.len() != table.len()
    }
}

/// Whether the device cache already holds the KV of `tokens[..first_position]`.
///
/// The check is the token history itself, not a length: the runtime hands a backend the same KV
/// buffer for a new sequence when the prefix cache matched, and the rows in it are then the ones the
/// device wrote for those tokens. A different history — even of the same length — must be uploaded.
pub(crate) fn prefix_matches(synced: &[u32], tokens: &[u32], first_position: usize) -> bool {
    first_position == 0
        || (tokens.len() >= first_position
            && synced.len() >= first_position
            && synced[..first_position] == tokens[..first_position])
}

/// Decode `count` f16 rows out of the runtime's buffer.
pub(crate) fn decode_rows(source: &[u8], values: &mut [f32]) {
    let count = values.len().min(source.len() / VALUE_BYTES);
    for (value, chunk) in values[..count]
        .iter_mut()
        .zip(source.as_chunks::<VALUE_BYTES>().0)
    {
        *value = crate::layer::f16(u16::from_le_bytes(*chunk));
    }
}

/// Encode `values` as f16 into the runtime's buffer.
pub(crate) fn encode_rows(values: &[f32], destination: &mut [u8]) {
    let count = values.len().min(destination.len() / VALUE_BYTES);
    for (chunk, value) in destination
        .as_chunks_mut::<VALUE_BYTES>()
        .0
        .iter_mut()
        .zip(values)
        .take(count)
    {
        *chunk = f32_to_f16(*value).to_le_bytes();
    }
}

/// `f32` to IEEE 754 half, round to nearest even.
///
/// The same arithmetic the CPU backend's `quant::f32_to_f16` performs, so a row the device writes
/// and the CPU later reads back is the value the CPU would have written itself.
pub(crate) fn f32_to_f16(value: f32) -> u16 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(order: RowOrder) -> KvRows {
        // Two layers of six values: 2 * 2 * 6 * 2 = 48 bytes per position.
        KvRows::with_order(2, 6, 4, 48, order).expect("geometry")
    }

    #[test]
    fn a_position_maps_to_the_token_major_layout() {
        let rows = rows(RowOrder::TokenMajor);
        let table = [4, 9];
        // Position 5 is in block 9, one position into it, layer 1 two twelve-byte rows in.
        let (k, v) = rows.offsets(1, 5, &table).expect("offsets");
        let block_start = 9 * 4 * 48;
        assert_eq!(k, block_start + 48 + 2 * 12);
        assert_eq!(v, k + 12);
        // Position 4 opens the second block, and the table says that block is 9, not 5.
        let (k4, _) = rows.offsets(0, 4, &table).expect("offsets");
        assert_eq!(k4, 9 * 4 * 48);
        // Position 3 is the last of block 4.
        let (k3, _) = rows.offsets(0, 3, &table).expect("offsets");
        assert_eq!(k3, 4 * 4 * 48 + 3 * 48);
    }

    #[test]
    fn a_position_maps_to_the_layer_major_layout() {
        let rows = rows(RowOrder::LayerMajor);
        let table = [4, 9];
        // Layer 1 owns the whole block's K and V, so its rows are a block long and its V row is four
        // positions past its K row.
        let (k, v) = rows.offsets(1, 5, &table).expect("offsets");
        let block_start = 9 * 4 * 48;
        assert_eq!(k, block_start + 2 * 12 * 4 + 12);
        assert_eq!(v, k + 12 * 4);
        let (k3, _) = rows.offsets(0, 3, &table).expect("offsets");
        assert_eq!(k3, 4 * 4 * 48 + 3 * 12);
    }

    #[test]
    fn the_two_orders_agree_on_which_block_a_position_is_in() {
        let table = [4, 9];
        let token = rows(RowOrder::TokenMajor);
        let layer = rows(RowOrder::LayerMajor);
        for position in 0..8 {
            let (k_token, _) = token.offsets(0, position, &table).expect("offsets");
            let (k_layer, _) = layer.offsets(0, position, &table).expect("offsets");
            assert_eq!(
                k_token / 192,
                k_layer / 192,
                "position {position} landed in a different block"
            );
        }
    }

    #[test]
    fn a_position_without_a_block_is_refused() {
        let rows = rows(RowOrder::TokenMajor);
        assert!(rows.offsets(0, 8, &[1]).is_err());
    }

    #[test]
    fn the_geometry_must_be_f16() {
        let error = KvRows::new(2, 6, 4, 24).expect_err("q8_0 bytes");
        assert!(error.to_string().contains("f16"), "{error}");
        assert!(KvRows::new(0, 6, 4, 48).is_err());
        assert!(KvRows::new(2, 6, 0, 48).is_err());
    }

    #[test]
    fn a_block_table_can_repeat_only_in_a_ring() {
        assert!(!KvRows::repeats(&[0, 1, 2, 3]));
        assert!(KvRows::repeats(&[0, 1, 0, 1]));
    }

    #[test]
    fn the_buffer_reach_is_the_highest_named_block() {
        let rows = rows(RowOrder::TokenMajor);
        assert_eq!(rows.reach(&[0, 1, 2]), 3 * 4 * 48);
        assert_eq!(rows.reach(&[]), 0);
        assert_eq!(rows.blocks(9), 3);
    }

    #[test]
    fn f16_rows_round_trip() {
        let values: [f32; 6] = [0.0, 1.0, -2.5, 0.333_333_34, 65_504.0, -0.000_061];
        let mut buffer = [0u8; 12];
        encode_rows(&values, &mut buffer);
        let mut back = [0f32; 6];
        decode_rows(&buffer, &mut back);
        for (before, after) in values.iter().zip(back.iter()) {
            assert!(
                (before - after).abs() <= before.abs() * 1e-3 + 1e-6,
                "{before} came back as {after}"
            );
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
    }

    #[test]
    fn the_prefix_is_the_token_history() {
        let synced = [11u32, 22, 33];
        assert!(prefix_matches(&synced, &[11, 22, 33, 44], 0));
        assert!(prefix_matches(&synced, &[11, 22, 33, 44], 3));
        assert!(prefix_matches(&synced, &[11, 22, 33, 44], 2));
        assert!(!prefix_matches(&synced, &[11, 22, 99, 44], 3));
        assert!(!prefix_matches(&synced, &[11, 22], 3));
        assert!(!prefix_matches(&[], &[11, 22, 33], 1));
    }
}
