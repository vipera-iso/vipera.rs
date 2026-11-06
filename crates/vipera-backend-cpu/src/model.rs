//! A GGUF model loaded for the CPU backend, and the forward pass over it.
//!
//! The block layout is the one Qwen2 and Llama share: pre-norm residual blocks, RMSNorm, grouped-query
//! attention with NeoX-style rotary embeddings, SwiGLU feed-forward, and a final norm before the output
//! projection. Weights stay quantized in the file's own format; every matvec dequantizes one row at a
//! time, so loading a model does not multiply its memory by four.
//!
//! K/V are written into the runtime's KV buffer as half-precision pairs, which is what the planner
//! sized the cache for (`f16`). A cache with a different geometry is refused rather than silently
//! overrunning.

use std::path::Path;

use vipera_core::gguf::{GgmlType, GgufFile, TensorInfo, Value};
use vipera_core::planner::KvCacheType;
use vipera_errors::ViperaError;
use vipera_runtime::{ForwardContext, ForwardPass};

use crate::quant::{self, f16_to_f32, f32_to_f16};

/// The numbers a forward pass needs, read from the file rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    /// Transformer blocks.
    pub layers: usize,
    /// Residual stream width.
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads; fewer than `heads` means grouped-query attention.
    pub kv_heads: usize,
    /// Width of one head, `hidden / heads`.
    pub head_dim: usize,
    /// Feed-forward width.
    pub ffn: usize,
    /// Vocabulary size, from the embedding's row count.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Rotary base frequency.
    pub rope_base: f32,
}

impl ModelConfig {
    /// Bytes one position occupies in an `f16` KV cache: K and V for every layer.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token_for(KvCacheType::F16).unwrap_or(0)
    }

    /// Bytes one position occupies at a given cache type, or `None` when this model's head width
    /// cannot be stored that way (a block quant needs a width divisible by 32).
    pub fn kv_bytes_per_token_for(&self, ty: KvCacheType) -> Option<u64> {
        let layout = KvLayout::of(ty)?;
        let width = self.kv_heads * self.head_dim;
        if layout == KvLayout::Q8_0 && !width.is_multiple_of(32) {
            return None;
        }
        Some((2 * self.layers) as u64 * layout.bytes_per_row(width) as u64)
    }

    /// Derive the layout from the number of bytes the runtime reserved per position.
    ///
    /// The runtime only carries the geometry; comparing it with both layouts this backend can write
    /// means a cache planned as `q8_0` is written as `q8_0`, and anything else is refused by name.
    fn kv_layout(&self, bytes_per_token: usize) -> Result<KvLayout, ViperaError> {
        let width = self.kv_heads * self.head_dim;
        let mut expected = Vec::new();
        for layout in [KvLayout::F16, KvLayout::Q8_0] {
            if layout == KvLayout::Q8_0 && !width.is_multiple_of(32) {
                continue;
            }
            let bytes = 2 * self.layers * layout.bytes_per_row(width);
            if bytes == bytes_per_token {
                return Ok(layout);
            }
            expected.push(format!("{bytes} for {}", layout.name()));
        }
        Err(ViperaError::unsupported(format!(
            "the CPU backend writes {} bytes per position ({}), but the KV cache holds \
             {bytes_per_token}; plan the cache with --kv-type f16 or q8_0",
            expected.join(" or "),
            if expected.len() > 1 {
                "f16 or q8_0"
            } else {
                "q8_0"
            }
        )))
    }
}

/// How one cached position is stored.
///
/// `f16` is two bytes per value; `q8_0` is one byte plus a half-precision scale per 32 values, which
/// is 34 bytes per 32 values — the same arithmetic the planner sizes `--kv-type q8_0` with, so the two
/// cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvLayout {
    F16,
    Q8_0,
}

impl KvLayout {
    const fn of(ty: KvCacheType) -> Option<Self> {
        match ty {
            KvCacheType::F16 => Some(Self::F16),
            KvCacheType::Q8_0 => Some(Self::Q8_0),
            KvCacheType::Q4_0 => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    /// Bytes one K (or V) row of `width` values occupies.
    fn bytes_per_row(self, width: usize) -> usize {
        match self {
            Self::F16 => width * 2,
            Self::Q8_0 => width / 32 * 34,
        }
    }

    fn encode(self, destination: &mut [u8], values: &[f32]) -> Result<(), ViperaError> {
        match self {
            Self::F16 => {
                for (chunk, value) in destination.as_chunks_mut::<2>().0.iter_mut().zip(values) {
                    *chunk = f32_to_f16(*value).to_le_bytes();
                }
                Ok(())
            }
            Self::Q8_0 => {
                if !values.len().is_multiple_of(32) || destination.len() < values.len() / 32 * 34 {
                    return Err(ViperaError::internal(
                        "a q8_0 KV row needs a whole number of 32-value blocks",
                    ));
                }
                for (block, chunk) in destination
                    .as_chunks_mut::<34>()
                    .0
                    .iter_mut()
                    .zip(values.as_chunks::<32>().0)
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
                Ok(())
            }
        }
    }

    fn decode(self, source: &[u8], values: &mut [f32]) -> Result<(), ViperaError> {
        match self {
            Self::F16 => {
                for (value, chunk) in values.iter_mut().zip(source.as_chunks::<2>().0.iter()) {
                    *value = f16_to_f32(u16::from_le_bytes(*chunk));
                }
                Ok(())
            }
            Self::Q8_0 => {
                if !values.len().is_multiple_of(32) || source.len() < values.len() / 32 * 34 {
                    return Err(ViperaError::internal(
                        "a q8_0 KV row needs a whole number of 32-value blocks",
                    ));
                }
                for (chunk, block) in values
                    .as_chunks_mut::<32>()
                    .0
                    .iter_mut()
                    .zip(source.as_chunks::<34>().0)
                {
                    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    for (value, quantized) in chunk.iter_mut().zip(&block[2..]) {
                        *value = scale * f32::from(*quantized as i8);
                    }
                }
                Ok(())
            }
        }
    }
}

/// Where one tensor lives inside the file's bytes.
#[derive(Debug, Clone, Copy)]
struct Tensor {
    ty: GgmlType,
    cols: usize,
    rows: usize,
    start: usize,
    len: usize,
}

impl Tensor {
    fn data<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        &file[self.start..self.start + self.len]
    }

    fn matvec(&self, file: &[u8], x: &[f32], out: &mut [f32]) -> Result<(), ViperaError> {
        quant::matvec(self.ty, self.data(file), self.cols, self.rows, x, out)
    }

    fn read_row(&self, file: &[u8], row: usize, out: &mut [f32]) -> Result<(), ViperaError> {
        quant::read_row(self.ty, self.data(file), self.cols, row, out)
    }
}

/// One transformer block's weights.
struct Layer {
    attn_norm: Vec<f32>,
    wq: Tensor,
    bq: Vec<f32>,
    wk: Tensor,
    bk: Vec<f32>,
    wv: Tensor,
    bv: Vec<f32>,
    wo: Tensor,
    ffn_norm: Vec<f32>,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
}

/// Qwen2/Llama weights plus the scratch buffers a forward pass reuses.
pub struct CpuBackend {
    bytes: Vec<u8>,
    config: ModelConfig,
    tok_embd: Tensor,
    output: Tensor,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    k_all: Vec<f32>,
    v_all: Vec<f32>,
    logits: Vec<f32>,
}

impl CpuBackend {
    /// Read a model file and prepare it for inference.
    pub fn load(path: &Path) -> Result<Self, ViperaError> {
        let bytes = std::fs::read(path)?;
        let file = GgufFile::parse(&bytes)?;
        Self::from_bytes(bytes, &file)
    }

    /// Build from a file that is already in memory, so a caller that also needs the metadata does not
    /// read the model twice.
    pub fn from_bytes(bytes: Vec<u8>, file: &GgufFile) -> Result<Self, ViperaError> {
        let config = ModelConfig::from_gguf(file)?;
        if config.kv_bytes_per_token() == 0 {
            return Err(ViperaError::unsupported("model declares zero KV heads"));
        }

        let tok_embd = matrix(file, "token_embd.weight", config.hidden, config.vocab)?;
        let output = matrix(file, "output.weight", config.hidden, config.vocab).unwrap_or(tok_embd);
        let output_norm = vector(file, &bytes, "output_norm.weight", config.hidden)?;

        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            layers.push(Layer {
                attn_norm: vector(file, &bytes, &name("attn_norm.weight"), config.hidden)?,
                wq: matrix(
                    file,
                    &name("attn_q.weight"),
                    config.hidden,
                    config.heads * config.head_dim,
                )?,
                bq: vector(
                    file,
                    &bytes,
                    &name("attn_q.bias"),
                    config.heads * config.head_dim,
                )?,
                wk: matrix(
                    file,
                    &name("attn_k.weight"),
                    config.hidden,
                    config.kv_heads * config.head_dim,
                )?,
                bk: vector(
                    file,
                    &bytes,
                    &name("attn_k.bias"),
                    config.kv_heads * config.head_dim,
                )?,
                wv: matrix(
                    file,
                    &name("attn_v.weight"),
                    config.hidden,
                    config.kv_heads * config.head_dim,
                )?,
                bv: vector(
                    file,
                    &bytes,
                    &name("attn_v.bias"),
                    config.kv_heads * config.head_dim,
                )?,
                wo: matrix(
                    file,
                    &name("attn_output.weight"),
                    config.hidden,
                    config.hidden,
                )?,
                ffn_norm: vector(file, &bytes, &name("ffn_norm.weight"), config.hidden)?,
                wg: matrix(file, &name("ffn_gate.weight"), config.hidden, config.ffn)?,
                wu: matrix(file, &name("ffn_up.weight"), config.hidden, config.ffn)?,
                wd: matrix(file, &name("ffn_down.weight"), config.ffn, config.hidden)?,
            });
        }

        Ok(Self {
            config,
            bytes,
            tok_embd,
            output,
            output_norm,
            layers,
            x: vec![0.0; config.hidden],
            h: vec![0.0; config.hidden],
            q: vec![0.0; config.heads * config.head_dim],
            k: vec![0.0; config.kv_heads * config.head_dim],
            v: vec![0.0; config.kv_heads * config.head_dim],
            attn: vec![0.0; config.hidden],
            gate: vec![0.0; config.ffn],
            up: vec![0.0; config.ffn],
            scores: Vec::new(),
            k_all: Vec::new(),
            v_all: Vec::new(),
            logits: vec![0.0; config.vocab],
        })
    }

    /// The model's shape.
    pub const fn config(&self) -> ModelConfig {
        self.config
    }

    /// Run the transformer over `context.new_tokens`, returning the logits of the last one.
    fn run(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        let config = self.config;
        if context.new_tokens.is_empty() {
            return Err(ViperaError::invalid_argument(
                "a forward pass needs at least one token",
            ));
        }
        let layout = config.kv_layout(context.bytes_per_token as usize)?;
        let last = context.first_position as usize + context.new_tokens.len();
        let blocks = last.div_ceil(context.tokens_per_block as usize);
        if blocks > context.block_table.len() {
            return Err(ViperaError::internal(format!(
                "{blocks} KV blocks are needed for {last} positions but the sequence has {}",
                context.block_table.len()
            )));
        }
        if context.kv.len()
            < blocks * context.bytes_per_token as usize * context.tokens_per_block as usize
        {
            return Err(ViperaError::internal(
                "the KV buffer is smaller than its own block table implies",
            ));
        }

        let Self {
            bytes,
            tok_embd,
            output,
            output_norm,
            layers,
            x,
            h,
            q,
            k,
            v,
            attn,
            gate,
            up,
            scores,
            k_all,
            v_all,
            logits,
            ..
        } = self;
        let mut kv = KvView {
            storage: context.kv,
            table: context.block_table,
            tokens_per_block: context.tokens_per_block as usize,
            bytes_per_token: context.bytes_per_token as usize,
            layout,
        };
        let shape = AttnShape {
            heads: config.heads,
            kv_heads: config.kv_heads,
            head_dim: config.head_dim,
        };

        for (offset, token) in context.new_tokens.iter().enumerate() {
            let position = context.first_position as usize + offset;
            let row = usize::try_from(*token)
                .map_err(|_| ViperaError::invalid_argument("token id does not fit this build"))?;
            if row >= config.vocab {
                return Err(ViperaError::invalid_argument(format!(
                    "token {row} is outside the vocabulary of {}",
                    config.vocab
                )));
            }
            tok_embd.read_row(bytes, row, x)?;

            for (index, layer) in layers.iter().enumerate() {
                rms_norm(x, &layer.attn_norm, config.rms_eps, h);
                layer.wq.matvec(bytes, h, q)?;
                add_assign(q, &layer.bq);
                layer.wk.matvec(bytes, h, k)?;
                add_assign(k, &layer.bk);
                layer.wv.matvec(bytes, h, v)?;
                add_assign(v, &layer.bv);

                rope(q, position, shape.heads, shape.head_dim, config.rope_base);
                rope(
                    k,
                    position,
                    shape.kv_heads,
                    shape.head_dim,
                    config.rope_base,
                );
                kv.write(position, index, shape, k, v)?;

                kv.load_layer(index, shape, position + 1, k_all, v_all)?;
                attention(q, k_all, v_all, position + 1, shape, scores, attn);

                layer.wo.matvec(bytes, attn, h)?;
                add_assign(x, h);

                rms_norm(x, &layer.ffn_norm, config.rms_eps, h);
                layer.wg.matvec(bytes, h, gate)?;
                layer.wu.matvec(bytes, h, up)?;
                for (value, gate) in up.iter_mut().zip(gate.iter()) {
                    *value *= silu(*gate);
                }
                layer.wd.matvec(bytes, up, h)?;
                add_assign(x, h);
            }
        }

        rms_norm(x, output_norm, config.rms_eps, h);
        output.matvec(bytes, h, logits)?;
        Ok(logits.clone())
    }
}

impl ForwardPass for CpuBackend {
    fn logits(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        self.run(context)
    }
}

impl ModelConfig {
    fn from_gguf(file: &GgufFile) -> Result<Self, ViperaError> {
        let architecture = file
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or_else(|| ViperaError::unsupported("model does not declare general.architecture"))?
            .to_owned();
        if !matches!(architecture.as_str(), "qwen2" | "llama") {
            return Err(ViperaError::unsupported(format!(
                "the CPU backend implements the qwen2/llama block layout, and this model is \
                 \"{architecture}\""
            )));
        }
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let u64_of = |suffix: &str| {
            file.metadata
                .get(&key(suffix))
                .and_then(Value::as_u64)
                .map(|value| value as usize)
        };
        let f32_of = |suffix: &str, fallback: f32| {
            file.metadata
                .get(&key(suffix))
                .and_then(Value::as_f32)
                .unwrap_or(fallback)
        };

        let layers = u64_of("block_count").ok_or_else(|| {
            ViperaError::unsupported(format!("{architecture}.block_count is missing"))
        })?;
        let hidden = u64_of("embedding_length").ok_or_else(|| {
            ViperaError::unsupported(format!("{architecture}.embedding_length is missing"))
        })?;
        let heads = u64_of("attention.head_count").ok_or_else(|| {
            ViperaError::unsupported(format!("{architecture}.attention.head_count is missing"))
        })?;
        let kv_heads = u64_of("attention.head_count_kv").unwrap_or(heads);
        let ffn = u64_of("feed_forward_length").ok_or_else(|| {
            ViperaError::unsupported(format!("{architecture}.feed_forward_length is missing"))
        })?;
        let vocab = file
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .and_then(|tensor| tensor.dims.get(1))
            .map(|value| *value as usize)
            .ok_or_else(|| ViperaError::unsupported("model has no token_embd.weight"))?;

        if heads == 0 || hidden % heads != 0 {
            return Err(ViperaError::unsupported(format!(
                "{hidden} hidden values do not divide into {heads} heads"
            )));
        }
        if kv_heads == 0 || heads % kv_heads != 0 {
            return Err(ViperaError::unsupported(format!(
                "{heads} query heads do not divide into {kv_heads} key/value heads"
            )));
        }

        Ok(Self {
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim: hidden / heads,
            ffn,
            vocab,
            rms_eps: f32_of("attention.layer_norm_rms_epsilon", 1e-6),
            rope_base: f32_of("rope.freq_base", 10_000.0),
        })
    }
}

/// The K/V buffer viewed as the runtime's blocks.
struct KvView<'a> {
    storage: &'a mut [u8],
    table: &'a [u32],
    tokens_per_block: usize,
    bytes_per_token: usize,
    layout: KvLayout,
}

impl KvView<'_> {
    fn slot(&self, position: usize) -> Result<usize, ViperaError> {
        let block = *self
            .table
            .get(position / self.tokens_per_block)
            .ok_or_else(|| ViperaError::internal("position has no KV block"))?;
        Ok(
            (block as usize * self.tokens_per_block + position % self.tokens_per_block)
                * self.bytes_per_token,
        )
    }

    fn write(
        &mut self,
        position: usize,
        layer: usize,
        shape: AttnShape,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), ViperaError> {
        let layer_bytes = self.layout.bytes_per_row(shape.kv_heads * shape.head_dim);
        let base = self.slot(position)? + layer * 2 * layer_bytes;
        let end = base + 2 * layer_bytes;
        if end > self.storage.len() {
            return Err(ViperaError::internal(
                "KV write would fall outside the cache buffer",
            ));
        }
        self.layout
            .encode(&mut self.storage[base..base + layer_bytes], k)?;
        self.layout
            .encode(&mut self.storage[base + layer_bytes..end], v)
    }

    fn load_layer(
        &self,
        layer: usize,
        shape: AttnShape,
        positions: usize,
        k_all: &mut Vec<f32>,
        v_all: &mut Vec<f32>,
    ) -> Result<(), ViperaError> {
        let width = shape.kv_heads * shape.head_dim;
        let layer_bytes = self.layout.bytes_per_row(width);
        k_all.resize(positions * width, 0.0);
        v_all.resize(positions * width, 0.0);
        for position in 0..positions {
            let base = self.slot(position)? + layer * 2 * layer_bytes;
            let end = base + 2 * layer_bytes;
            if end > self.storage.len() {
                return Err(ViperaError::internal(
                    "KV read would fall outside the cache buffer",
                ));
            }
            self.layout.decode(
                &self.storage[base..base + layer_bytes],
                &mut k_all[position * width..][..width],
            )?;
            self.layout.decode(
                &self.storage[base + layer_bytes..end],
                &mut v_all[position * width..][..width],
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct AttnShape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// Causal grouped-query attention over every position up to and including `positions - 1`.
fn attention(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    positions: usize,
    shape: AttnShape,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    scores.resize(positions, 0.0);
    for head in 0..shape.heads {
        let kv_head = head / group;
        let query = &q[head * head_dim..(head + 1) * head_dim];
        let mut best = f32::NEG_INFINITY;
        for (position, score) in scores.iter_mut().enumerate().take(positions) {
            let base = (position * shape.kv_heads + kv_head) * head_dim;
            let mut dot = 0.0_f32;
            for (q, k) in query.iter().zip(&k_all[base..base + head_dim]) {
                dot += q * k;
            }
            *score = dot * scale;
            best = best.max(*score);
        }
        let mut total = 0.0_f32;
        for score in scores.iter_mut().take(positions) {
            *score = (*score - best).exp();
            total += *score;
        }
        let inverse = 1.0 / total;
        let output = &mut out[head * head_dim..(head + 1) * head_dim];
        output.fill(0.0);
        for (position, weight) in scores.iter().take(positions).enumerate() {
            let weight = weight * inverse;
            let base = (position * shape.kv_heads + kv_head) * head_dim;
            for (value, v) in output.iter_mut().zip(&v_all[base..base + head_dim]) {
                *value += weight * v;
            }
        }
    }
}

/// Rotary embedding, NeoX layout: rotate pairs `(i, i + head_dim / 2)`.
fn rope(vector: &mut [f32], position: usize, heads: usize, head_dim: usize, base: f32) {
    let half = head_dim / 2;
    for head in 0..heads {
        let head = &mut vector[head * head_dim..(head + 1) * head_dim];
        for index in 0..half {
            let exponent = 2.0 * index as f32 / head_dim as f32;
            let angle = position as f32 / base.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let first = head[index];
            let second = head[index + half];
            head[index] = first * cos - second * sin;
            head[index + half] = first * sin + second * cos;
        }
    }
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse = 1.0 / (mean + epsilon).sqrt();
    for ((value, weight), input) in out.iter_mut().zip(weight).zip(input) {
        *value = input * inverse * weight;
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn add_assign(target: &mut [f32], addition: &[f32]) {
    for (target, addition) in target.iter_mut().zip(addition) {
        *target += addition;
    }
}

fn find<'a>(file: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, ViperaError> {
    file.tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| ViperaError::unsupported(format!("model has no tensor \"{name}\"")))
}

fn locate(file: &GgufFile, tensor: &TensorInfo) -> Result<(usize, usize), ViperaError> {
    let start = usize::try_from(file.data_offset + tensor.offset)
        .map_err(|_| ViperaError::parse(0, "tensor offset does not fit this build"))?;
    let len = usize::try_from(
        tensor
            .byte_len()
            .ok_or_else(|| ViperaError::parse(0, "tensor is not a whole number of blocks"))?,
    )
    .map_err(|_| ViperaError::parse(0, "tensor length does not fit this build"))?;
    Ok((start, len))
}

fn matrix(file: &GgufFile, name: &str, cols: usize, rows: usize) -> Result<Tensor, ViperaError> {
    let info = find(file, name)?;
    if info.dims.len() != 2 || info.dims[0] as usize != cols || info.dims[1] as usize != rows {
        return Err(ViperaError::unsupported(format!(
            "tensor \"{name}\" is {:?}, expected [{cols}, {rows}]",
            info.dims
        )));
    }
    let (start, len) = locate(file, info)?;
    Ok(Tensor {
        ty: info.ty,
        cols,
        rows,
        start,
        len,
    })
}

fn vector(file: &GgufFile, data: &[u8], name: &str, len: usize) -> Result<Vec<f32>, ViperaError> {
    let info = find(file, name)?;
    if info.dims.len() != 1 || info.dims[0] as usize != len {
        return Err(ViperaError::unsupported(format!(
            "tensor \"{name}\" is {:?}, expected [{len}]",
            info.dims
        )));
    }
    if info.ty != GgmlType::F32 {
        return Err(ViperaError::unsupported(format!(
            "tensor \"{name}\" is {}, and only F32 norms and biases are implemented",
            info.ty.name()
        )));
    }
    let (start, bytes) = locate(file, info)?;
    let slice = data
        .get(start..start + bytes)
        .ok_or_else(|| ViperaError::parse(start as u64, "tensor lies outside the file"))?;
    Ok(slice
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_the_definition() {
        let mut out = [0.0; 2];
        rms_norm(&[3.0, 4.0], &[1.0, 1.0], 0.0, &mut out);
        let expected = 1.0 / (12.5_f32).sqrt();
        assert!((out[0] - 3.0 * expected).abs() < 1e-6);
        assert!((out[1] - 4.0 * expected).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_a_pair_by_the_position_angle() {
        let mut vector = [1.0, 0.0];
        rope(&mut vector, 1, 1, 2, 1.0);
        assert!((vector[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((vector[1] - 1.0_f32.sin()).abs() < 1e-6);
        let mut at_zero = [2.0, 5.0];
        rope(&mut at_zero, 0, 1, 2, 10_000.0);
        assert_eq!(at_zero, [2.0, 5.0], "position zero is the identity");
    }

    #[test]
    fn attention_is_causal_and_ignores_later_positions() {
        let shape = AttnShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
        };
        let q = [1.0, 0.0];
        let k_all = [1.0, 0.0, 0.0, 1.0];
        let v_all = [1.0, 0.0, 0.0, 1.0];
        let mut scores = Vec::new();
        let mut out = [0.0; 2];
        attention(&q, &k_all, &v_all, 1, shape, &mut scores, &mut out);
        assert_eq!(out, [1.0, 0.0], "only position 0 exists yet");
        attention(&q, &k_all, &v_all, 2, shape, &mut scores, &mut out);
        // Scores are (1 / sqrt(2)) and 0, so the first position's softmax weight is borrow-free:
        // it depends only on that difference, not on the raw magnitudes.
        let first = 1.0 / (1.0 + (-(1.0_f32 / 2.0_f32.sqrt())).exp());
        assert!((out[0] - first).abs() < 1e-6);
        assert!((out[1] - (1.0 - first)).abs() < 1e-6);
    }

    #[test]
    fn kv_slots_follow_the_block_table_across_block_boundaries() {
        // One layer: K and V are 4 bytes each, so a position costs 8 bytes in this toy layout.
        let per_token = 8;
        let mut storage = vec![0u8; 2 * 4 * per_token];
        let table = [1u32, 0u32];
        let kv = KvView {
            storage: &mut storage,
            table: &table,
            tokens_per_block: 4,
            bytes_per_token: per_token,
            layout: KvLayout::F16,
        };
        assert_eq!(kv.slot(0).expect("slot"), 4 * per_token);
        assert_eq!(kv.slot(3).expect("slot"), 7 * per_token);
        assert_eq!(
            kv.slot(4).expect("slot"),
            0,
            "block 0 holds the second half"
        );
        assert_eq!(kv.slot(7).expect("slot"), 3 * per_token);
        assert!(
            kv.slot(8).is_err(),
            "a position beyond the block table has no slot"
        );
    }

    #[test]
    fn kv_round_trips_through_half_precision() {
        let shape = AttnShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
        };
        let per_token = 8;
        let mut storage = vec![0u8; 4 * per_token];
        let table = [0u32];
        let mut kv = KvView {
            storage: &mut storage,
            table: &table,
            tokens_per_block: 4,
            bytes_per_token: per_token,
            layout: KvLayout::F16,
        };
        kv.write(0, 0, shape, &[3.0, 4.0], &[5.0, 6.0])
            .expect("write");
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        kv.load_layer(0, shape, 1, &mut k_all, &mut v_all)
            .expect("load");
        assert_eq!(k_all, vec![3.0, 4.0]);
        assert_eq!(v_all, vec![5.0, 6.0]);
    }

    #[test]
    fn kv_round_trips_through_q8_0() {
        let shape = AttnShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 32,
        };
        // One layer, K and V, one 32-value block each: 34 bytes per row.
        let per_token = 2 * 34;
        let mut storage = vec![0u8; 4 * per_token];
        let table = [0u32];
        let mut kv = KvView {
            storage: &mut storage,
            table: &table,
            tokens_per_block: 4,
            bytes_per_token: per_token,
            layout: KvLayout::Q8_0,
        };
        let k: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.125 - 2.0).collect();
        kv.write(0, 0, shape, &k, &v).expect("write");
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        kv.load_layer(0, shape, 1, &mut k_all, &mut v_all)
            .expect("load");
        // A byte per value is lossy by construction; the largest value sets the scale, so the
        // error is bounded by half a step of max/127.
        for (stored, original) in k_all.iter().zip(&k).chain(v_all.iter().zip(&v)) {
            assert!(
                (stored - original).abs() <= 4.0 / 127.0 / 2.0 + 1e-3,
                "q8_0 KV changed {original} into {stored}"
            );
        }
    }

    #[test]
    fn a_cache_whose_per_position_size_matches_no_layout_is_refused() {
        let config = ModelConfig {
            layers: 1,
            hidden: 32,
            heads: 1,
            kv_heads: 1,
            head_dim: 32,
            ffn: 64,
            vocab: 8,
            rms_eps: 1e-6,
            rope_base: 10_000.0,
        };
        assert_eq!(config.kv_layout(68).expect("q8_0"), KvLayout::Q8_0);
        assert_eq!(config.kv_layout(128).expect("f16"), KvLayout::F16);
        let error = config.kv_layout(96).expect_err("no layout is 96 bytes");
        assert!(error.to_string().contains("--kv-type"), "{error}");
    }

    #[test]
    fn query_heads_share_the_key_value_head_they_are_grouped_with() {
        let shape = AttnShape {
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
        };
        let q = [1.0, 0.0, 1.0, 0.0];
        let k_all = [1.0, 0.0];
        let v_all = [2.0, 3.0];
        let mut scores = Vec::new();
        let mut out = [0.0; 4];
        attention(&q, &k_all, &v_all, 1, shape, &mut scores, &mut out);
        assert_eq!(out, [2.0, 3.0, 2.0, 3.0]);
    }
}
