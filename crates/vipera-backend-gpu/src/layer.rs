//! One decoder layer, end to end on the device.
//!
//! The shape is the Qwen2/Llama block the CPU backend runs: pre-norm residual, grouped-query
//! attention with NeoX rotary embeddings, SwiGLU feed-forward, and both norms. Every intermediate is a
//! device buffer, so a step launches kernels and copies nothing but the input row up and the output
//! row down — the point of this module is to measure a layer without PCIe in the middle.
//!
//! What it is not: batched, paged, or wired into `ForwardPass`. K/V live in one flat per-layer cache
//! sized to `max_positions`, which is what a comparison against the CPU needs and nothing more.

use std::sync::atomic::{AtomicU8, Ordering};

use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};
use crate::pool::{Format, PooledMatrix, VramPool};
use crate::q4k::Q4kMatrix;
use crate::q6k::Q6kMatrix;
use crate::q8::Q8Device;

pub(crate) const SOURCE: &str = include_str!("layer.cu");

/// Environment switch for the shared-attention arm, for the one measurement that wants both arms in
/// one process. `on`/`1`/`true` shares, anything else keeps the per-head kernel.
pub const SHARE_ENV: &str = "VIPERA_GPU_ATTN_SHARE";

/// `2` means "not forced"; `0` and `1` are the arms [`force_shared`] pins.
const UNSET: u8 = 2;
static SHARE: AtomicU8 = AtomicU8::new(UNSET);

/// Pin the attention arm for the next steps, or hand control back to the environment with `None`.
///
/// Both arms are the same arithmetic over the same rows, so a measurement can time them in one
/// process — which is the only way two timings here are comparable, since the clock on this card
/// moves with the load before it.
pub fn force_shared(arm: Option<bool>) {
    SHARE.store(
        match arm {
            Some(true) => 1,
            Some(false) => 0,
            None => UNSET,
        },
        Ordering::Relaxed,
    );
}

/// Whether a step should use the shared-group attention kernel.
///
/// Off unless asked for: the per-head kernel is the one every earlier number was taken with, and a
/// switch that defaults on turns a measurement into a change of behaviour.
pub fn shared_enabled() -> bool {
    match SHARE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => matches!(
            std::env::var(SHARE_ENV).as_deref(),
            Ok("1" | "on" | "true" | "yes")
        ),
    }
}

/// Environment switch for the split arm. The split kernel is the default past
/// [`SPLIT_MIN`] positions because it is the same arithmetic at 3.4x the block count; `off`/`0`
/// pins the older arms for a measurement.
pub const SPLIT_ENV: &str = "VIPERA_GPU_ATTN_SPLIT";

/// Unset leaves the split arm on (see [`split_enabled`]); `0`/`1` pin it.
static SPLIT: AtomicU8 = AtomicU8::new(UNSET);

/// Pin the split arm for the next steps (`None` hands control back to the environment).
pub fn force_split(arm: Option<bool>) {
    SPLIT.store(
        match arm {
            Some(true) => 1,
            Some(false) => 0,
            None => UNSET,
        },
        Ordering::Relaxed,
    );
}

/// Whether a step should cut the history into chunks instead of walking it in one block.
///
/// On by default. Unlike the shared-group arm this is not a different accounting of the same reads
/// — it is the same reads with eight times the blocks and no per-position `__syncthreads()` on a
/// 4096-long critical path, and it is held to the per-head arm within rounding by
/// `attention_split_matches_the_per_head_kernel`.
pub fn split_enabled() -> bool {
    match SPLIT.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => !matches!(
            std::env::var(SPLIT_ENV).as_deref(),
            Ok("0" | "off" | "false" | "no")
        ),
    }
}

/// Positions per block in the split kernel.
pub(crate) const SPLIT_CHUNK: usize = 128;

/// Shortest history worth a second launch (the combine pass costs a few microseconds).
pub(crate) const SPLIT_MIN: usize = 512;

/// Whether a shape can take the shared kernel at all: a group of more than one head, a block that
/// fits, and a head width of whole warps (the reduction is a warp shuffle per head).
fn shares(shape: LayerShape) -> bool {
    let group = shape.heads / shape.kv_heads.max(1);
    shape.kv_heads > 0
        && shape.heads.is_multiple_of(shape.kv_heads)
        && group > 1
        && shape.head_dim.is_multiple_of(32)
        && group * shape.head_dim <= 1024
}

/// Floats the shared kernel needs for its softmax weights, zero when the shape cannot use it.
fn score_scratch(shape: LayerShape) -> usize {
    if shares(shape) {
        shape.heads * shape.max_positions
    } else {
        0
    }
}

/// Chunks the split kernel uses at this shape's longest history.
fn split_chunks(shape: LayerShape) -> usize {
    if shares(shape) {
        shape.max_positions.div_ceil(SPLIT_CHUNK)
    } else {
        0
    }
}

/// Whether a step of `positions` positions takes the split arm.
///
/// A shape the shared kernel cannot take (no group, odd head width, oversized block) has no split
/// arm either: it is the same block shape, only cut along the history.
fn uses_split(shape: LayerShape, positions: usize) -> bool {
    split_enabled() && shares(shape) && positions >= SPLIT_MIN
}

/// Floats the split kernel needs: per head, each chunk's max, denominator and V row.
fn partial_scratch(shape: LayerShape) -> usize {
    let chunks = split_chunks(shape);
    if chunks == 0 {
        0
    } else {
        shape.heads * (2 * chunks + chunks * shape.head_dim)
    }
}

/// f16 by bit pattern, for the tensors the file stores as halves.
pub fn f16(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    let value = if exp == 0 {
        man * 2f32.powi(-24)
    } else if exp == 31 {
        if man == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (man + 1024.0) * 2f32.powi(exp - 25)
    };
    if sign == 1 { -value } else { value }
}

/// Everything the layer's kernels need to know about the model.
#[derive(Clone, Copy, Debug)]
pub struct LayerShape {
    /// Hidden width.
    pub hidden: usize,
    /// Feed-forward width.
    pub ffn: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Rotary base.
    pub rope_base: f32,
    /// Norm epsilon.
    pub epsilon: f32,
    /// Positions the KV cache can hold.
    pub max_positions: usize,
}

impl LayerShape {
    /// Width of a KV row.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Width of a query row.
    pub fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// One packed tensor, in whichever format the file uses.
#[derive(Clone, Copy)]
pub enum Packed<'a> {
    /// Q4_K, the format most of a Q4_K_M file uses.
    Q4(&'a [u8]),
    /// Q6_K, which a Q4_K_M file keeps for `attn_v` and `ffn_down`.
    Q6(&'a [u8]),
}

/// A projection on the device, in either format.
pub enum Matrix {
    /// Q4_K, fed int8 activations.
    Q4(Q4kMatrix),
    /// Q6_K, fed float activations.
    Q6(Q6kMatrix),
    /// A view into a [`VramPool`] slot: the weights were copied up once, with the layer.
    Pooled(PooledMatrix),
}

impl Matrix {
    /// Bytes of weights held on the device.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Q4(matrix) => matrix.bytes(),
            Self::Q6(matrix) => matrix.bytes(),
            Self::Pooled(matrix) => matrix.bytes(),
        }
    }

    fn upload(
        device: &Device,
        packed: Packed<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, ViperaError> {
        match packed {
            Packed::Q4(bytes) => Ok(Self::Q4(Q4kMatrix::upload(device, bytes, rows, cols)?)),
            Packed::Q6(bytes) => Ok(Self::Q6(Q6kMatrix::upload(device, bytes, rows, cols)?)),
        }
    }

    /// `out = W · input`, quantizing the input first when the format wants int8.
    fn project(
        &self,
        device: &Device,
        input: &DeviceBuffer<f32>,
        q8: &mut Q8Device,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        match self {
            Self::Q4(matrix) => {
                device.quantize_into(input, q8, input.len() / 32)?;
                matrix.matvec_q8_into(device, q8, out)
            }
            Self::Q6(matrix) => matrix.matvec_into(device, input, out),
            Self::Pooled(matrix) => matrix.project(device, input, q8, out),
        }
    }
}

/// Where one tensor sits inside a pooled layer, and what shape it has.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Bytes from the start of the layer's span.
    pub offset: usize,
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
    /// Packed format.
    pub format: Format,
}

/// The Q4_K bytes and norms of one layer, as they sit in the file.
pub struct LayerWeights<'a> {
    /// `attn_norm.weight`.
    pub attn_norm: &'a [f32],
    /// `ffn_norm.weight`.
    pub ffn_norm: &'a [f32],
    /// `wq`, `wk`, `wv`, `wo`, `wg`, `wu`, `wd` in that order.
    pub matrices: [Packed<'a>; 7],
    /// `bq`, `bk`, `bv`; absent on Llama, present on Qwen2.
    pub biases: [Option<&'a [f32]>; 3],
}

/// The weights of one layer, resident on the device, plus the scratch a step reuses.
pub struct GpuLayer {
    shape: LayerShape,
    attn_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    wg: Matrix,
    wu: Matrix,
    wd: Matrix,
    bq: Option<DeviceBuffer<f32>>,
    bk: Option<DeviceBuffer<f32>>,
    bv: Option<DeviceBuffer<f32>>,
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    /// Softmax weights, one row per query head, between the two halves of the shared kernel:
    /// `heads * max_positions` floats.
    attn_scores: DeviceBuffer<f32>,
    /// Per-chunk partial softmax of the split kernel: `heads * (2 * chunks + chunks * head_dim)`.
    attn_partial: DeviceBuffer<f32>,
    projection: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    q8: Q8Device,
}

impl GpuLayer {
    /// Copy a layer's weights up and allocate its caches and scratch.
    pub fn upload(
        device: &Device,
        shape: LayerShape,
        weights: LayerWeights<'_>,
    ) -> Result<Self, ViperaError> {
        let [wq, wk, wv, wo, wg, wu, wd] = weights.matrices;
        let query = shape.query_width();
        let kv = shape.kv_width();
        Ok(Self {
            shape,
            attn_norm: device.upload(weights.attn_norm)?,
            ffn_norm: device.upload(weights.ffn_norm)?,
            wq: Matrix::upload(device, wq, query, shape.hidden)?,
            wk: Matrix::upload(device, wk, kv, shape.hidden)?,
            wv: Matrix::upload(device, wv, kv, shape.hidden)?,
            wo: Matrix::upload(device, wo, shape.hidden, query)?,
            wg: Matrix::upload(device, wg, shape.ffn, shape.hidden)?,
            wu: Matrix::upload(device, wu, shape.ffn, shape.hidden)?,
            wd: Matrix::upload(device, wd, shape.hidden, shape.ffn)?,
            bq: weights.biases[0].map(|b| device.upload(b)).transpose()?,
            bk: weights.biases[1].map(|b| device.upload(b)).transpose()?,
            bv: weights.biases[2].map(|b| device.upload(b)).transpose()?,
            k_cache: device.alloc(shape.max_positions * kv)?,
            v_cache: device.alloc(shape.max_positions * kv)?,
            residual: device.alloc(shape.hidden)?,
            normed: device.alloc(shape.hidden)?,
            q: device.alloc(query)?,
            k: device.alloc(kv)?,
            v: device.alloc(kv)?,
            attn: device.alloc(query)?,
            attn_scores: device.alloc(score_scratch(shape).max(1))?,
            attn_partial: device.alloc(partial_scratch(shape).max(1))?,
            projection: device.alloc(shape.hidden)?,
            gate: device.alloc(shape.ffn)?,
            up: device.alloc(shape.ffn)?,
            activated: device.alloc(shape.ffn)?,
            q8: Q8Device::allocate(device, shape.hidden.max(shape.ffn).max(query))?,
        })
    }

    /// Build the same layer out of a pool slot instead of seven uploads.
    ///
    /// `specs` are `wq, wk, wv, wo, wg, wu, wd`, with offsets measured from the start of the
    /// layer's span in the file (the bytes the bus coped up). Nothing is copied here: every
    /// projection is a view into the one buffer the bus already filled.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pool(
        device: &Device,
        shape: LayerShape,
        pool: &VramPool,
        layer: usize,
        specs: [Spec; 7],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        biases: [Option<&[f32]>; 3],
    ) -> Result<Self, ViperaError> {
        let mut matrices = Vec::with_capacity(7);
        for spec in specs {
            matrices.push(Matrix::Pooled(pool.matrix(
                layer,
                spec.offset,
                spec.rows,
                spec.cols,
                spec.format,
            )?));
        }
        let mut drain = matrices.into_iter();
        let (wq, wk, wv, wo, wg, wu, wd) = (
            drain.next().expect("wq"),
            drain.next().expect("wk"),
            drain.next().expect("wv"),
            drain.next().expect("wo"),
            drain.next().expect("wg"),
            drain.next().expect("wu"),
            drain.next().expect("wd"),
        );
        let query = shape.query_width();
        let kv = shape.kv_width();
        Ok(Self {
            shape,
            attn_norm: device.upload(attn_norm)?,
            ffn_norm: device.upload(ffn_norm)?,
            wq,
            wk,
            wv,
            wo,
            wg,
            wu,
            wd,
            bq: biases[0].map(|b| device.upload(b)).transpose()?,
            bk: biases[1].map(|b| device.upload(b)).transpose()?,
            bv: biases[2].map(|b| device.upload(b)).transpose()?,
            k_cache: device.alloc(shape.max_positions * kv)?,
            v_cache: device.alloc(shape.max_positions * kv)?,
            residual: device.alloc(shape.hidden)?,
            normed: device.alloc(shape.hidden)?,
            q: device.alloc(query)?,
            k: device.alloc(kv)?,
            v: device.alloc(kv)?,
            attn: device.alloc(query)?,
            attn_scores: device.alloc(score_scratch(shape).max(1))?,
            attn_partial: device.alloc(partial_scratch(shape).max(1))?,
            projection: device.alloc(shape.hidden)?,
            gate: device.alloc(shape.ffn)?,
            up: device.alloc(shape.ffn)?,
            activated: device.alloc(shape.ffn)?,
            q8: Q8Device::allocate(device, shape.hidden.max(shape.ffn).max(query))?,
        })
    }

    /// The shape this layer was built for.
    pub fn shape(&self) -> LayerShape {
        self.shape
    }

    /// The layer's K and V caches, which the paged backend mirrors into the runtime's buffer.
    pub fn kv_caches_mut(&mut self) -> (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>) {
        (&mut self.k_cache, &mut self.v_cache)
    }

    /// The layer's K cache, for a read-only copy back to the host.
    pub fn k_cache(&self) -> &DeviceBuffer<f32> {
        &self.k_cache
    }

    /// The layer's V cache.
    pub fn v_cache(&self) -> &DeviceBuffer<f32> {
        &self.v_cache
    }

    /// Bytes of Q4_K weights held on the device.
    pub fn weight_bytes(&self) -> usize {
        [
            &self.wq, &self.wk, &self.wv, &self.wo, &self.wg, &self.wu, &self.wd,
        ]
        .iter()
        .map(|matrix| matrix.bytes())
        .sum()
    }

    /// One decode step: `x` is the residual row for a single position.
    ///
    /// Mirrors the CPU `run` for one token: norm → qkv → rope → cache → attention → out projection →
    /// residual → norm → gate/up → SwiGLU → down → residual.
    pub fn step(
        &mut self,
        device: &Device,
        x: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, ViperaError> {
        device.copy_into(x, &mut self.residual)?;
        self.step_device(device, position)?;
        self.residual.download(device)
    }

    /// The same step with the residual row already on the device: what a model loop wants, so a
    /// layer's output never travels to the host and back.
    pub fn step_device(&mut self, device: &Device, position: usize) -> Result<(), ViperaError> {
        let shape = self.shape;
        if position >= shape.max_positions {
            return Err(ViperaError::invalid_argument(format!(
                "position {position} is past the cache of {}",
                shape.max_positions
            )));
        }
        device.rmsnorm(
            &self.residual,
            &self.attn_norm,
            &mut self.normed,
            shape.epsilon,
        )?;
        self.wq
            .project(device, &self.normed, &mut self.q8, &mut self.q)?;
        self.wk
            .project(device, &self.normed, &mut self.q8, &mut self.k)?;
        self.wv
            .project(device, &self.normed, &mut self.q8, &mut self.v)?;
        if let Some(bias) = &self.bq {
            device.add_inplace(&mut self.q, bias, shape.query_width())?;
        }
        if let Some(bias) = &self.bk {
            device.add_inplace(&mut self.k, bias, shape.kv_width())?;
        }
        if let Some(bias) = &self.bv {
            device.add_inplace(&mut self.v, bias, shape.kv_width())?;
        }
        device.rope(
            &mut self.q,
            position,
            shape.heads,
            shape.head_dim,
            shape.rope_base,
        )?;
        device.rope(
            &mut self.k,
            position,
            shape.kv_heads,
            shape.head_dim,
            shape.rope_base,
        )?;
        device.write_kv(
            position,
            &self.k,
            &self.v,
            &mut self.k_cache,
            &mut self.v_cache,
            shape.kv_width(),
        )?;

        let scale = 1.0 / (shape.head_dim as f32).sqrt();
        if !shared_enabled() && uses_split(shape, position + 1) {
            let chunks = device.attention_split(
                &self.q,
                &self.k_cache,
                &self.v_cache,
                &mut self.attn_partial,
                position + 1,
                SPLIT_CHUNK,
                shape,
                scale,
            )?;
            device.attention_combine(&self.attn_partial, &mut self.attn, chunks, shape)?;
        } else if shared_enabled() && shares(shape) {
            device.attention_shared(
                &self.q,
                &self.k_cache,
                &self.v_cache,
                &mut self.attn,
                &mut self.attn_scores,
                position + 1,
                shape,
                scale,
            )?;
        } else {
            device.attention(
                &self.q,
                &self.k_cache,
                &self.v_cache,
                &mut self.attn,
                position + 1,
                shape,
                scale,
            )?;
        }
        self.wo
            .project(device, &self.attn, &mut self.q8, &mut self.projection)?;
        device.add_inplace(&mut self.residual, &self.projection, shape.hidden)?;

        device.rmsnorm(
            &self.residual,
            &self.ffn_norm,
            &mut self.normed,
            shape.epsilon,
        )?;
        self.wg
            .project(device, &self.normed, &mut self.q8, &mut self.gate)?;
        self.wu
            .project(device, &self.normed, &mut self.q8, &mut self.up)?;
        device.swiglu(&self.gate, &self.up, &mut self.activated, shape.ffn)?;
        self.wd
            .project(device, &self.activated, &mut self.q8, &mut self.projection)?;
        device.add_inplace(&mut self.residual, &self.projection, shape.hidden)?;
        Ok(())
    }

    /// The layer's output row, still on the device.
    pub fn residual(&self) -> &DeviceBuffer<f32> {
        &self.residual
    }

    /// Load a residual row that is already on the device.
    pub fn load_residual(
        &mut self,
        device: &Device,
        x: &DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        device.copy_buffer(x, &mut self.residual)
    }

    /// Load a residual row into the layer before [`Self::step_device`].
    pub fn set_residual(&mut self, device: &Device, x: &[f32]) -> Result<(), ViperaError> {
        device.copy_into(x, &mut self.residual)
    }

    /// Whether the model's QKV projections carry biases.
    pub fn has_biases(&self) -> bool {
        self.bq.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_14b(max_positions: usize) -> LayerShape {
        LayerShape {
            hidden: 5120,
            ffn: 17408,
            heads: 40,
            kv_heads: 8,
            head_dim: 128,
            rope_base: 1.0e6,
            epsilon: 1e-6,
            max_positions,
        }
    }

    #[test]
    fn the_split_arm_needs_a_group_and_whole_warps() {
        assert!(shares(qwen3_14b(4096)));
        // One KV head for every query head: there is nothing to share, and the split block would be
        // one head's width.
        let ungrouped = LayerShape {
            kv_heads: 40,
            ..qwen3_14b(4096)
        };
        assert!(!shares(ungrouped));
        // A group of five heads at head width 96 is 480 threads of three warps each: fine. At 100 it
        // is not a whole number of warps, and the shuffle reduction would read across heads.
        let ragged = LayerShape {
            head_dim: 100,
            ..qwen3_14b(4096)
        };
        assert!(!shares(ragged));
    }

    #[test]
    fn the_split_arm_starts_at_its_threshold() {
        let shape = qwen3_14b(4096);
        force_split(Some(false));
        assert!(!uses_split(shape, SPLIT_MIN));
        force_split(Some(true));
        assert!(!uses_split(shape, SPLIT_MIN - 1));
        assert!(uses_split(shape, SPLIT_MIN));
        force_split(None);
    }

    #[test]
    fn the_scratch_fits_the_shape() {
        let shape = qwen3_14b(1024);
        let chunks = 1024_usize.div_ceil(SPLIT_CHUNK);
        assert_eq!(split_chunks(shape), chunks);
        assert_eq!(
            partial_scratch(shape),
            shape.heads * (2 * chunks + chunks * shape.head_dim)
        );
        // The shared kernel's score rows are per head, not per group: every KV block writes its own.
        assert_eq!(score_scratch(shape), shape.heads * shape.max_positions);
    }
}
