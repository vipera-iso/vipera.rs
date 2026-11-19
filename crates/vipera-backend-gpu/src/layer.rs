//! One decoder layer, end to end on the device.
//!
//! The shape is the Qwen2/Llama block the CPU backend runs: pre-norm residual, grouped-query
//! attention with NeoX rotary embeddings, SwiGLU feed-forward, and both norms. Every intermediate is a
//! device buffer, so a step launches kernels and copies nothing but the input row up and the output
//! row down — the point of this module is to measure a layer without PCIe in the middle.
//!
//! What it is not: batched, paged, or wired into `ForwardPass`. K/V live in one flat per-layer cache
//! sized to `max_positions`, which is what a comparison against the CPU needs and nothing more.

use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};
use crate::q4k::Q4kMatrix;
use crate::q6k::Q6kMatrix;
use crate::q8::Q8Device;

pub(crate) const SOURCE: &str = include_str!("layer.cu");

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
}

impl Matrix {
    /// Bytes of weights held on the device.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Q4(matrix) => matrix.bytes(),
            Self::Q6(matrix) => matrix.bytes(),
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
        }
    }
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
        let shape = self.shape;
        if x.len() != shape.hidden {
            return Err(ViperaError::invalid_argument(format!(
                "the residual row is {} values, the layer takes {}",
                x.len(),
                shape.hidden
            )));
        }
        if position >= shape.max_positions {
            return Err(ViperaError::invalid_argument(format!(
                "position {position} is past the cache of {}",
                shape.max_positions
            )));
        }
        device.copy_into(x, &mut self.residual)?;

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
        device.attention(
            &self.q,
            &self.k_cache,
            &self.v_cache,
            &mut self.attn,
            position + 1,
            shape,
            scale,
        )?;
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

        self.residual.download(device)
    }

    /// Whether the model's QKV projections carry biases.
    pub fn has_biases(&self) -> bool {
        self.bq.is_some()
    }
}
