//! The device backend as the runtime's [`ForwardPass`].
//!
//! The runtime owns sequencing, the paged KV buffer and sampling. This type is the other half: it
//! computes a step on the card and keeps the runtime's buffer truthful, so a block the runtime parks
//! and hands to another sequence is the same block the CPU backend would have written.
//!
//! Two things make that more than a wrapper:
//!
//! - **The runtime's buffer is the authority.** After a step, every K and V row it computed is copied
//!   back into the runtime's block at the position's slot. Before a step whose prefix the device has
//!   not computed, those rows are read back out of the buffer and uploaded. The check is the token
//!   history, not a length, so a prefix-cache hit is reused and an unrelated history is not.
//! - **The device cache is dense.** The attention kernels walk positions `0..n`, so the backend
//!   refuses a table that names a block twice (only a ring cache does) instead of reading rows the
//!   window has overwritten.
//!
//! What it is not: batched. `logits_batch` keeps the trait's default, which runs the entries one at
//! a time through `logits`, and prefill is one position per weight read. Making the batched kernel
//! pay needs per-row norms, RoPE and a batched attention pass — a separate change with its own
//! measurement, not something a trait implementation is allowed to assume.

use std::path::Path;

use vipera_errors::ViperaError;
use vipera_runtime::{ForwardContext, ForwardPass};

use crate::device::Device;
use crate::model::GpuModel;
use crate::paged::{KvRows, decode_rows, encode_rows, prefix_matches};

/// A model on one CUDA device, addressed the way the runtime addresses a backend.
pub struct GpuBackend {
    model: GpuModel,
    /// Values in one K or V row.
    width: usize,
    /// Layers whose rows the runtime's buffer interleaves.
    layers: usize,
    /// Tokens whose KV the device caches hold, positionally: `synced[p]` is the token at `p`.
    synced: Vec<u32>,
}

impl GpuBackend {
    /// Open device `ordinal`, read `path`, and size the KV cache for `positions`.
    ///
    /// `positions` is the context the runtime planned; a step that would reach past it is refused
    /// rather than silently wrapping the cache.
    pub fn load(device: Device, path: &Path, positions: usize) -> Result<Self, ViperaError> {
        let model = GpuModel::load_with_positions(device, path, positions)?;
        Ok(Self {
            width: model.kv_width(),
            layers: model.depth(),
            model,
            synced: Vec::new(),
        })
    }

    /// The model, for callers that want the device or the shape.
    pub fn model(&self) -> &GpuModel {
        &self.model
    }

    /// Positions the device KV cache holds.
    pub fn positions(&self) -> usize {
        self.model.kv_positions()
    }

    /// Copy the runtime's rows for positions `0..first` onto the device.
    ///
    /// One upload per layer and per K/V, not one per row: the device cache is dense and the rows are
    /// contiguous, so a prefix costs two copies a layer however long it is.
    fn upload_prefix(
        &mut self,
        kv: &[u8],
        table: &[u32],
        rows: KvRows,
        first: usize,
    ) -> Result<(), ViperaError> {
        if first == 0 {
            return Ok(());
        }
        let width = rows.width();
        let row_bytes = KvRows::row_bytes(width);
        let mut k = vec![0f32; first * width];
        let mut v = vec![0f32; first * width];
        for layer in 0..rows.layers() {
            for position in 0..first {
                let (k_offset, v_offset) = rows.offsets(layer, position, table)?;
                let row = position * width;
                let k_source = kv.get(k_offset..k_offset + row_bytes).ok_or_else(|| {
                    ViperaError::internal("a K row is past the end of the KV buffer")
                })?;
                let v_source = kv.get(v_offset..v_offset + row_bytes).ok_or_else(|| {
                    ViperaError::internal("a V row is past the end of the KV buffer")
                })?;
                decode_rows(k_source, &mut k[row..row + width]);
                decode_rows(v_source, &mut v[row..row + width]);
            }
            self.model.upload_kv(layer, 0, &k, &v)?;
        }
        Ok(())
    }

    /// Copy the rows a step just computed into the runtime's blocks.
    fn store_rows(
        &mut self,
        kv: &mut [u8],
        table: &[u32],
        rows: KvRows,
        first: usize,
        count: usize,
    ) -> Result<(), ViperaError> {
        let width = rows.width();
        let row_bytes = KvRows::row_bytes(width);
        let mut encoded = vec![0u8; row_bytes];
        for layer in 0..rows.layers() {
            let (k, v) = self.model.download_kv(layer, first, count)?;
            for position in first..first + count {
                let (k_offset, v_offset) = rows.offsets(layer, position, table)?;
                let row = (position - first) * width;
                for (values, offset) in [(&k, k_offset), (&v, v_offset)] {
                    encode_rows(&values[row..row + width], &mut encoded);
                    let target = kv.get_mut(offset..offset + row_bytes).ok_or_else(|| {
                        ViperaError::internal("a KV row is past the end of the KV buffer")
                    })?;
                    target.copy_from_slice(&encoded);
                }
            }
        }
        Ok(())
    }
}

impl ForwardPass for GpuBackend {
    fn logits(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        let first = context.first_position as usize;
        let count = context.new_tokens.len();
        if count == 0 {
            return Err(ViperaError::invalid_argument(
                "a forward pass needs at least one token",
            ));
        }
        if context.tokens.len() < first {
            return Err(ViperaError::internal(format!(
                "position {first} is past the {} tokens the sequence retains",
                context.tokens.len()
            )));
        }
        let rows = KvRows::new(
            self.layers,
            self.width,
            context.tokens_per_block,
            context.bytes_per_token,
        )?;
        let last = first + count;
        if last > self.model.kv_positions() {
            return Err(ViperaError::unsupported(format!(
                "the device KV cache holds {} positions and this step reaches {last}; load the \
                 backend with the context the runtime planned",
                self.model.kv_positions()
            )));
        }
        let blocks = rows.blocks(last);
        if blocks > context.block_table.len() {
            return Err(ViperaError::internal(format!(
                "{blocks} KV blocks are needed for {last} positions but the sequence has {}",
                context.block_table.len()
            )));
        }
        let table = &context.block_table[..blocks];
        if KvRows::repeats(table) {
            return Err(ViperaError::unsupported(
                "the device backend walks positions densely, so a ring cache (a block table that \
                 names one block twice) is not supported; run the sliding window on the CPU backend",
            ));
        }
        let reach = rows.reach(table);
        if context.kv.len() < reach {
            return Err(ViperaError::internal(
                "the KV buffer is smaller than its own block table implies",
            ));
        }

        if !prefix_matches(&self.synced, context.tokens, first) {
            self.upload_prefix(context.kv, table, rows, first)?;
        }

        let mut logits = Vec::new();
        for (offset, token) in context.new_tokens.iter().enumerate() {
            logits = self.model.step(*token, first + offset)?;
        }

        self.store_rows(context.kv, table, rows, first, count)?;
        self.synced.clear();
        self.synced.extend_from_slice(&context.tokens[..first]);
        self.synced.extend_from_slice(context.new_tokens);
        Ok(logits)
    }
}
