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
use std::thread;

use vipera_core::gguf::{GgmlType, GgufFile, TensorInfo, Value};
use vipera_core::planner::KvCacheType;
use vipera_errors::ViperaError;
use vipera_runtime::{BatchContext, BatchEntry, ForwardContext, ForwardPass};

use crate::pool::Pool;
use crate::quant::{self, Fusion, f16_to_f32, f32_to_f16};

/// The most sequences one batched step computes at once.
///
/// The const-generic kernels keep one accumulator per entry in registers, so the batch size is a
/// compile-time choice rather than a runtime one. Eight was the old stop point; sixteen doubles
/// prefill weight reuse (`ceil(n/16)` reads vs `ceil(n/8)`) — the linear-streaming win from the
/// llama.cpp tiled mul_mat paper's first step — for a few extra megabytes of scratch. The kernels
/// spill registers at 16 but the weight read saved dominates on prefill.
const MAX_BATCH: usize = 128;

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
                crate::simd::half_to_float(source, values);
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

    fn matvec<'d>(
        &self,
        pool: &Pool<'d, 1>,
        file: &'d [u8],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), ViperaError> {
        let at = crate::profile::stamp();
        let result =
            quant::matvec_pooled(pool, self.ty, self.data(file), self.cols, self.rows, x, out);
        if let Some(at) = at {
            crate::profile::record_shape(
                shape_name(self.ty),
                1,
                self.rows,
                self.cols,
                at.elapsed(),
            );
        }
        crate::profile::note(dot_phase(self.ty), at);
        result
    }

    /// The same product for `B` activations, reading each row once for all of them.
    fn matvec_batch<'d, const B: usize>(
        &self,
        pool: &Pool<'d, B>,
        file: &'d [u8],
        activations: &[&[f32]; B],
        fusion: &Fusion,
        out: &mut [f32],
    ) -> Result<(), ViperaError> {
        let at = crate::profile::stamp();
        let matrix = quant::Matrix {
            ty: self.ty,
            cols: self.cols,
            rows: self.rows,
        };
        let result =
            quant::matvec_pooled_batch(pool, matrix, self.data(file), activations, fusion, out);
        if let Some(at) = at {
            crate::profile::record_shape(
                shape_name(self.ty),
                B,
                self.rows,
                self.cols,
                at.elapsed(),
            );
        }
        crate::profile::note(dot_phase(self.ty), at);
        result
    }

    fn read_row(&self, file: &[u8], row: usize, out: &mut [f32]) -> Result<(), ViperaError> {
        quant::read_row(self.ty, self.data(file), self.cols, row, out)
    }
}

/// The phase a matrix product belongs to, by weight format.
/// The weight format's name, for the per-shape profile table. Borrowed, because it is recorded on a
/// per-call path.
fn shape_name(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::Bf16 => "Bf16",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q6K => "Q6_K",
        _ => "other",
    }
}

fn dot_phase(ty: GgmlType) -> crate::profile::Phase {
    use crate::profile::Phase;
    match ty {
        GgmlType::Q4K => Phase::Q4K,
        GgmlType::Q6K => Phase::Q6K,
        _ => Phase::OtherDot,
    }
}

/// One transformer block's weights.
///
/// `bq`/`bk`/`bv` are empty where the architecture has no attention bias (llama, qwen3); adding an
/// empty slice is a no-op. `q_norm`/`k_norm` are empty everywhere but qwen3, which normalises each
/// head's query and key vector on its own after the projection and before RoPE.
struct Layer {
    attn_norm: Vec<f32>,
    wq: Tensor,
    bq: Vec<f32>,
    wk: Tensor,
    bk: Vec<f32>,
    wv: Tensor,
    bv: Vec<f32>,
    wo: Tensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
}

/// Scratch for a batched step, sized for [`MAX_BATCH`] entries when the model is loaded.
///
/// One flat buffer per stage, token-major: entry `e`'s row of width `w` starts at `e * w`. Reusing
/// them for the life of the model keeps a decode step free of allocation.
#[derive(Debug)]
struct BatchScratch {
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    /// Shared rather than borrowed: a fused product hands its worker the gate rows, and a job
    /// travels with the pool's lifetime, which is longer than this scratch's borrow.
    gate: std::sync::Arc<[f32]>,
    up: Vec<f32>,
    /// The last entry's logits: the only row a prefill chunk or a single decode step asks for.
    logits: Vec<f32>,
    /// One logits row per entry, for the callers that want every row (`logits_batch`,
    /// `logits_range`). Empty until the first such call: a 128-row buffer is
    /// `MAX_BATCH * vocab * 4` bytes (77.8 MB for a 151,936-token vocabulary), and the chunked
    /// prefill path never looks past the last row.
    logits_each: Vec<f32>,
    attn_scratch: AttnScratch,
}

impl BatchScratch {
    fn new(config: ModelConfig) -> Self {
        let query = config.heads * config.head_dim;
        let kv = config.kv_heads * config.head_dim;
        Self {
            x: vec![0.0; MAX_BATCH * config.hidden],
            h: vec![0.0; MAX_BATCH * config.hidden],
            q: vec![0.0; MAX_BATCH * query],
            k: vec![0.0; MAX_BATCH * kv],
            v: vec![0.0; MAX_BATCH * kv],
            attn: vec![0.0; MAX_BATCH * config.hidden],
            gate: vec![0.0; MAX_BATCH * config.ffn].into(),
            up: vec![0.0; MAX_BATCH * config.ffn],
            logits: vec![0.0; config.vocab],
            logits_each: Vec::new(),
            attn_scratch: AttnScratch::default(),
        }
    }
}

/// Qwen2/Qwen3/Llama weights plus the scratch buffers a forward pass reuses.
pub struct CpuBackend {
    /// The weights and where they live: mapped from the file, or a caller's buffer.
    weights: crate::bus::WeightSource,
    config: ModelConfig,
    /// The architecture family the model declared; reported, not recomputed.
    architecture: String,
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
    attn_scratch: AttnScratch,
    logits: Vec<f32>,
    batch: BatchScratch,
    /// Set while a prompt chunk runs through the batch path, so the profile can tell a chunk from a
    /// decode step. Only meaningful when the phase timers are on.
    chunk_pass: bool,
}

impl CpuBackend {
    /// Read a model file and prepare it for inference.
    /// The file is mapped, not read: the weights stay on disk until a pass touches them, and a
    /// narrowed RAM window (`VIPERA_RAM_LAYERS`) pages them back out behind the pass.
    pub fn load(path: &Path) -> Result<Self, ViperaError> {
        let (weights, file) = crate::bus::WeightSource::open(path)?;
        Self::from_weights(weights, &file)
    }

    /// Build from a file that is already in memory, so a caller that also needs the metadata does not
    /// read the model twice.
    pub fn from_bytes(bytes: Vec<u8>, file: &GgufFile) -> Result<Self, ViperaError> {
        Self::from_weights(crate::bus::WeightSource::from_bytes(bytes, file), file)
    }

    fn from_weights(
        weights: crate::bus::WeightSource,
        file: &GgufFile,
    ) -> Result<Self, ViperaError> {
        let bytes = weights.bytes();
        let config = ModelConfig::from_gguf(file)?;
        let architecture = file
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        if config.kv_bytes_per_token() == 0 {
            return Err(ViperaError::unsupported("model declares zero KV heads"));
        }

        let tok_embd = matrix(file, "token_embd.weight", config.hidden, config.vocab)?;
        let output = matrix(file, "output.weight", config.hidden, config.vocab).unwrap_or(tok_embd);
        let output_norm = vector(file, bytes, "output_norm.weight", config.hidden)?;

        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            layers.push(Layer {
                attn_norm: vector(file, bytes, &name("attn_norm.weight"), config.hidden)?,
                wq: matrix(
                    file,
                    &name("attn_q.weight"),
                    config.hidden,
                    config.heads * config.head_dim,
                )?,
                bq: optional_vector(
                    file,
                    bytes,
                    &name("attn_q.bias"),
                    config.heads * config.head_dim,
                )?,
                wk: matrix(
                    file,
                    &name("attn_k.weight"),
                    config.hidden,
                    config.kv_heads * config.head_dim,
                )?,
                bk: optional_vector(
                    file,
                    bytes,
                    &name("attn_k.bias"),
                    config.kv_heads * config.head_dim,
                )?,
                wv: matrix(
                    file,
                    &name("attn_v.weight"),
                    config.hidden,
                    config.kv_heads * config.head_dim,
                )?,
                bv: optional_vector(
                    file,
                    bytes,
                    &name("attn_v.bias"),
                    config.kv_heads * config.head_dim,
                )?,
                wo: matrix(
                    file,
                    &name("attn_output.weight"),
                    config.hidden,
                    config.hidden,
                )?,
                q_norm: optional_vector(file, bytes, &name("attn_q_norm.weight"), config.head_dim)?,
                k_norm: optional_vector(file, bytes, &name("attn_k_norm.weight"), config.head_dim)?,
                ffn_norm: vector(file, bytes, &name("ffn_norm.weight"), config.hidden)?,
                wg: matrix(file, &name("ffn_gate.weight"), config.hidden, config.ffn)?,
                wu: matrix(file, &name("ffn_up.weight"), config.hidden, config.ffn)?,
                wd: matrix(file, &name("ffn_down.weight"), config.ffn, config.hidden)?,
            });
        }

        Ok(Self {
            config,
            weights,
            architecture,
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
            attn_scratch: AttnScratch::default(),
            logits: vec![0.0; config.vocab],
            chunk_pass: false,
            batch: BatchScratch::new(config),
        })
    }

    /// The model's shape.
    pub const fn config(&self) -> ModelConfig {
        self.config
    }

    /// The model's declared architecture family, e.g. `qwen2` or `llama`.
    ///
    /// Read once at construction from the file's `general.architecture`; a backend that reports a
    /// different name here is lying about which block layout it computed over.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Run the transformer over `context.new_tokens`, returning the logits of the last one.
    fn run(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        if context.tokens_per_block == 0 {
            return Err(ViperaError::invalid_argument(
                "a KV block must hold at least one token",
            ));
        }
        let pass = crate::profile::begin(false);
        let config = self.config;
        // A prompt chunk is a group of positions that share every weight read, which is exactly what
        // the batch path already does: one entry per position, each with a single new token, ordered
        // by position so the KV it writes is in place before the position that reads it. Handing the
        // chunk to that path turns `n` reads of 1.117 GB into `ceil(n / 8)`.
        if context.new_tokens.len() > 1 {
            // The batch path records the pass itself, so this one does not.
            let _ = pass;
            self.chunk_pass = true;
            return self.run_positions(context);
        }
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
        // The table does not have to name one distinct block per position: a ring cache repeats
        // its blocks, so what the buffer must cover is the highest block any position names.
        let block_bytes = context.bytes_per_token as usize * context.tokens_per_block as usize;
        let reach = context.block_table[..blocks]
            .iter()
            .copied()
            .max()
            .map_or(0, |block| (block as usize + 1) * block_bytes);
        if context.kv.len() < reach {
            return Err(ViperaError::internal(
                "the KV buffer is smaller than its own block table implies",
            ));
        }

        let Self {
            weights,
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
            attn_scratch,
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

        // The bytes are borrowed for the whole pass so the pool's workers can hold them; the
        // scratch buffers stay mutable because they are separate fields.
        let file: &[u8] = weights.bytes();
        let threads =
            quant::thread_budget(config.vocab, config.hidden.saturating_mul(config.vocab));
        thread::scope(|scope| {
            let pool = Pool::new(scope, threads);
            for (offset, token) in context.new_tokens.iter().enumerate() {
                let position = context.first_position as usize + offset;
                let row = usize::try_from(*token).map_err(|_| {
                    ViperaError::invalid_argument("token id does not fit this build")
                })?;
                if row >= config.vocab {
                    return Err(ViperaError::invalid_argument(format!(
                        "token {row} is outside the vocabulary of {}",
                        config.vocab
                    )));
                }
                tok_embd.read_row(file, row, x)?;

                for (index, layer) in layers.iter().enumerate() {
                    weights.advance(index);
                    let at = crate::profile::stamp();
                    rms_norm(x, &layer.attn_norm, config.rms_eps, h);
                    crate::profile::note(crate::profile::Phase::Norm, at);
                    layer.wq.matvec(&pool, file, h, q)?;
                    layer.wk.matvec(&pool, file, h, k)?;
                    layer.wv.matvec(&pool, file, h, v)?;
                    let at = crate::profile::stamp();
                    add_assign(q, &layer.bq);
                    add_assign(k, &layer.bk);
                    add_assign(v, &layer.bv);
                    if !layer.q_norm.is_empty() {
                        per_head_rms_norm(
                            q,
                            &layer.q_norm,
                            config.rms_eps,
                            shape.heads,
                            shape.head_dim,
                        );
                        per_head_rms_norm(
                            k,
                            &layer.k_norm,
                            config.rms_eps,
                            shape.kv_heads,
                            shape.head_dim,
                        );
                    }
                    crate::profile::note(crate::profile::Phase::Proj, at);

                    let at = crate::profile::stamp();
                    rope(q, position, shape.heads, shape.head_dim, config.rope_base);
                    rope(
                        k,
                        position,
                        shape.kv_heads,
                        shape.head_dim,
                        config.rope_base,
                    );
                    crate::profile::note(crate::profile::Phase::Rope, at);
                    let at = crate::profile::stamp();
                    kv.write(position, index, shape, k, v)?;
                    crate::profile::note(crate::profile::Phase::KvWrite, at);

                    let at = crate::profile::stamp();
                    attend(q, &kv, index, position + 1, shape, attn_scratch, attn)?;
                    crate::profile::note(crate::profile::Phase::Attention, at);

                    layer.wo.matvec(&pool, file, attn, h)?;
                    let at = crate::profile::stamp();
                    add_assign(x, h);
                    crate::profile::note(crate::profile::Phase::Elem, at);

                    rms_norm(x, &layer.ffn_norm, config.rms_eps, h);
                    layer.wg.matvec(&pool, file, h, gate)?;
                    layer.wu.matvec(&pool, file, h, up)?;
                    let at = crate::profile::stamp();
                    silu_mul(up, gate);
                    crate::profile::note(crate::profile::Phase::Elem, at);
                    layer.wd.matvec(&pool, file, up, h)?;
                    let at = crate::profile::stamp();
                    add_assign(x, h);
                    crate::profile::note(crate::profile::Phase::Elem, at);
                }
            }

            rms_norm(x, output_norm, config.rms_eps, h);
            output.matvec(&pool, file, h, logits)?;
            crate::profile::finish_pass(pass);
            Ok(logits.clone())
        })
    }
}

impl CpuBackend {
    /// Compute a run of consecutive positions of one sequence through the batch path.
    ///
    /// The entries are the same sequence at `first_position`, `first_position + 1`, ...; the batch
    /// kernel processes each entry with its own single new token and its own causal limit, and writes
    /// each entry's K/V before the next entry reads it. Every entry's arithmetic is the batch kernel's
    /// unchanged per-entry arithmetic, so the logits of the last position are bit-identical to what
    /// the sequential path produced.
    fn run_positions(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        let tokens = context.new_tokens;
        let first = context.first_position;
        let kv = std::mem::take(&mut context.kv);
        let mut logits = Vec::new();
        let mut start = 0;
        while start < tokens.len() {
            let end = (start + MAX_BATCH).min(tokens.len());
            let mut entries: Vec<BatchEntry<'_>> = (start..end)
                .map(|index| BatchEntry {
                    tokens: context.tokens,
                    new_tokens: &tokens[index..index + 1],
                    first_position: first + index as u32,
                    block_table: context.block_table,
                    tokens_per_block: context.tokens_per_block,
                    bytes_per_token: context.bytes_per_token,
                })
                .collect();
            let need_logits = end == tokens.len();
            let results = self.run_batch(
                &mut BatchContext {
                    kv: &mut *kv,
                    entries: &mut entries,
                },
                need_logits,
                need_logits,
            )?;
            if need_logits {
                // The projection only runs for the final chunk; intermediate
                // chunk logits are computed nowhere and would be discarded.
                logits = results.into_iter().next_back().unwrap_or_default();
            }
            start = end;
        }
        context.kv = kv;
        Ok(logits)
    }

    /// Every position's logits from one batched pass over `context.new_tokens`.
    ///
    /// Prompt-lookup speculation needs the target model's answer after each token it is checking,
    /// not only after the last one. The batch path already computes every entry; this asks it not
    /// to throw them away. Entry `i` gets its own single new token, exactly as prefill hands it
    /// over, so each answer is the same one a separate pass would produce.
    pub fn logits_each(
        &mut self,
        context: &mut ForwardContext<'_>,
    ) -> Result<Vec<Vec<f32>>, ViperaError> {
        let tokens = context.new_tokens;
        if tokens.is_empty() {
            return Err(ViperaError::invalid_argument(
                "a chunk needs at least one token",
            ));
        }
        let first = context.first_position;
        let kv = std::mem::take(&mut context.kv);
        let mut entries: Vec<BatchEntry<'_>> = (0..tokens.len())
            .map(|index| BatchEntry {
                tokens: context.tokens,
                new_tokens: &tokens[index..index + 1],
                first_position: first + index as u32,
                block_table: context.block_table,
                tokens_per_block: context.tokens_per_block,
                bytes_per_token: context.bytes_per_token,
            })
            .collect();
        let results = self.run_batch(
            &mut BatchContext {
                kv: &mut *kv,
                entries: &mut entries,
            },
            true,
            false,
        );
        context.kv = kv;
        results
    }

    /// A batched decode step, one new token per sequence.
    ///
    /// Anything that is not a single new token per context — a prefill, or a draft run — falls back
    /// to [`Self::run`] one sequence at a time. Prefill stays sequential on purpose: it is
    /// compute-heavy, and mixing it into a decode batch complicates every kernel for little gain on a
    /// machine whose decode step is bandwidth-bound.
    fn run_batch(
        &mut self,
        batch: &mut BatchContext<'_, '_>,
        need_logits: bool,
        only_last: bool,
    ) -> Result<Vec<Vec<f32>>, ViperaError> {
        let size = batch.entries.len();
        if size == 0 {
            return Err(ViperaError::invalid_argument(
                "a batched step needs at least one sequence",
            ));
        }
        if batch
            .entries
            .iter()
            .any(|entry| entry.new_tokens.len() != 1)
        {
            if only_last {
                // Fallback only ever runs for multi-token entries; a prefill
                // chunk is single-token per entry, so this is unreachable
                // there. Keep full compute for safety: run every entry (so
                // KV is written for all), return logits for the last only.
                let mut all = batch
                    .entries
                    .iter()
                    .map(|entry| {
                        let mut context = entry.context(&mut *batch.kv);
                        self.run(&mut context)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let last = all.pop().unwrap_or_default();
                let mut rest = vec![Vec::new(); size - 1];
                rest.push(last);
                return Ok(rest);
            }
            return batch
                .entries
                .iter()
                .map(|entry| {
                    let mut context = entry.context(&mut *batch.kv);
                    self.run(&mut context)
                })
                .collect();
        }
        if size > MAX_BATCH {
            return Err(ViperaError::invalid_argument(format!(
                "this backend batches up to {MAX_BATCH} sequences, not {size}"
            )));
        }
        // `run_batch_n` is monomorphised per const width, so every width it is instantiated at costs
        // a copy of the whole layer stack in compile time and code size — 128 of them for 128 widths
        // of nearly identical kernels. Keep every width 1..=8 (a decode batch never exceeds it here)
        // and the multiples of 8 up to MAX_BATCH; any other size runs as a multiple of 8 followed by
        // a remainder under 8, which is one extra weight read of at most seven tokens. Each pass
        // writes its entries' K/V before the next begins, so the order of the batch is preserved.
        let BatchContext { kv, entries } = batch;
        let mut rest: &mut [BatchEntry<'_>] = entries;
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(size);
        while !rest.is_empty() {
            let width = if rest.len() <= 8 {
                rest.len()
            } else {
                (rest.len() & !7).min(MAX_BATCH)
            };
            let is_last_piece = rest.len() == width;
            let piece_need_logits = need_logits && (!only_last || is_last_piece);
            let piece_only_last = only_last && is_last_piece;
            let (head, tail) = rest.split_at_mut(width);
            rest = tail;
            macro_rules! dispatch {
                ($($width:literal),* $(,)?) => {
                    match width {
                        $(
                            $width => self.run_batch_n(
                                kv,
                                exactly::<$width, _>(head)?,
                                piece_need_logits,
                                piece_only_last,
                            )
                            .map(|piece| results.extend(piece)),
                        )*
                        other => Err(ViperaError::invalid_argument(format!(
                            "this backend has no kernel for a batch of {other} sequences"
                        ))),
                    }
                };
            }
            dispatch!(
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
                128,
            )?;
        }
        Ok(results)
    }

    /// One decode step for `B` sequences, reading each weight tensor once for all of them.
    ///
    /// Each entry's arithmetic is the same as its own single-sequence pass, so entry `e`'s logits are
    /// bit-identical to what [`Self::logits`] returns for that context alone.
    fn run_batch_n<const B: usize>(
        &mut self,
        kv: &mut [u8],
        contexts: &mut [BatchEntry<'_>; B],
        need_logits: bool,
        only_last: bool,
    ) -> Result<Vec<Vec<f32>>, ViperaError> {
        if contexts.iter().any(|context| context.tokens_per_block == 0) {
            return Err(ViperaError::invalid_argument(
                "a KV block must hold at least one token",
            ));
        }
        let pass = crate::profile::begin(std::mem::take(&mut self.chunk_pass));
        let config = self.config;
        let mut positions = [0_usize; B];
        let mut tokens = [0_u32; B];
        for (entry, context) in contexts.iter().enumerate() {
            if context.new_tokens.len() != 1 {
                return Err(ViperaError::invalid_argument(format!(
                    "a batched decode step takes one token per sequence, entry {entry} has {}",
                    context.new_tokens.len()
                )));
            }
            let last = context.first_position as usize + 1;
            let blocks = last.div_ceil(context.tokens_per_block as usize);
            if blocks > context.block_table.len() {
                return Err(ViperaError::internal(format!(
                    "{blocks} KV blocks are needed for {last} positions but sequence {entry} has {}",
                    context.block_table.len()
                )));
            }
            let block_bytes = context.bytes_per_token as usize * context.tokens_per_block as usize;
            let reach = context.block_table[..blocks]
                .iter()
                .copied()
                .max()
                .map_or(0, |block| (block as usize + 1) * block_bytes);
            if kv.len() < reach {
                return Err(ViperaError::internal(
                    "the KV buffer is smaller than its own block table implies",
                ));
            }
            positions[entry] = last - 1;
            tokens[entry] = context.new_tokens[0];
        }

        let counts: Vec<usize> = positions.iter().map(|position| position + 1).collect();
        let Self {
            weights,
            tok_embd,
            output,
            output_norm,
            layers,
            batch,
            ..
        } = self;
        let hidden = config.hidden;
        let ffn = config.ffn;
        let query = config.heads * config.head_dim;
        let kv_width = config.kv_heads * config.head_dim;
        let vocab = config.vocab;
        let epsilon = config.rms_eps;
        let shape = AttnShape {
            heads: config.heads,
            kv_heads: config.kv_heads,
            head_dim: config.head_dim,
        };
        let file: &[u8] = weights.bytes();
        let threads = quant::thread_budget(vocab, hidden.saturating_mul(vocab).saturating_mul(B));

        thread::scope(|scope| {
            let pool = Pool::new(scope, threads);

            for (entry, token) in tokens.iter().enumerate() {
                let row = usize::try_from(*token).map_err(|_| {
                    ViperaError::invalid_argument("token id does not fit this build")
                })?;
                if row >= vocab {
                    return Err(ViperaError::invalid_argument(format!(
                        "token {row} is outside the vocabulary of {vocab}"
                    )));
                }
                tok_embd.read_row(file, row, &mut batch.x[entry * hidden..][..hidden])?;
            }

            // The chunk's positions are fixed before the first layer, and a rotary angle belongs to
            // the position alone, so the tables are built once here and shared by all 28 layers.
            let rope_angles: Vec<RopeAngles> = positions
                .iter()
                .map(|position| RopeAngles::new(*position, shape.head_dim, config.rope_base))
                .collect();

            for (index, layer) in layers.iter().enumerate() {
                weights.advance(index);
                let at = crate::profile::stamp();
                let fused = prep_fusion_enabled();
                if !fused {
                    for entry in 0..B {
                        rms_norm(
                            &batch.x[entry * hidden..][..hidden],
                            &layer.attn_norm,
                            epsilon,
                            &mut batch.h[entry * hidden..][..hidden],
                        );
                    }
                }
                crate::profile::note(crate::profile::Phase::Norm, at);
                let norm = if fused {
                    Fusion::Norm {
                        weight: std::sync::Arc::from(&layer.attn_norm[..]),
                        epsilon,
                    }
                } else {
                    Fusion::None
                };
                // With the norm fused the projections read the residual stream itself: each of them
                // runs the norm inside the copy its prep already makes.
                let activations = if fused {
                    rows::<B>(&batch.x, hidden)
                } else {
                    rows::<B>(&batch.h, hidden)
                };
                layer.wq.matvec_batch::<B>(
                    &pool,
                    file,
                    &activations,
                    &norm,
                    &mut batch.q[..B * query],
                )?;
                layer.wk.matvec_batch::<B>(
                    &pool,
                    file,
                    &activations,
                    &norm,
                    &mut batch.k[..B * kv_width],
                )?;
                layer.wv.matvec_batch::<B>(
                    &pool,
                    file,
                    &activations,
                    &norm,
                    &mut batch.v[..B * kv_width],
                )?;
                let per_head_norm = !layer.q_norm.is_empty();
                let at = crate::profile::stamp();
                for entry in 0..B {
                    add_assign(&mut batch.q[entry * query..][..query], &layer.bq);
                    add_assign(&mut batch.k[entry * kv_width..][..kv_width], &layer.bk);
                    add_assign(&mut batch.v[entry * kv_width..][..kv_width], &layer.bv);
                    if per_head_norm {
                        per_head_rms_norm(
                            &mut batch.q[entry * query..][..query],
                            &layer.q_norm,
                            epsilon,
                            shape.heads,
                            shape.head_dim,
                        );
                        per_head_rms_norm(
                            &mut batch.k[entry * kv_width..][..kv_width],
                            &layer.k_norm,
                            epsilon,
                            shape.kv_heads,
                            shape.head_dim,
                        );
                    }
                }
                crate::profile::note(crate::profile::Phase::Proj, at);

                for entry in 0..B {
                    let position = positions[entry];
                    let at = crate::profile::stamp();
                    rope_with(
                        &mut batch.q[entry * query..][..query],
                        &rope_angles[entry],
                        shape.heads,
                        shape.head_dim,
                    );
                    rope_with(
                        &mut batch.k[entry * kv_width..][..kv_width],
                        &rope_angles[entry],
                        shape.kv_heads,
                        shape.head_dim,
                    );
                    crate::profile::note(crate::profile::Phase::Rope, at);
                    let at = crate::profile::stamp();
                    let context = &contexts[entry];
                    let mut view = KvView {
                        storage: kv,
                        table: context.block_table,
                        tokens_per_block: context.tokens_per_block as usize,
                        bytes_per_token: context.bytes_per_token as usize,
                        layout: config.kv_layout(context.bytes_per_token as usize)?,
                    };
                    view.write(
                        position,
                        index,
                        shape,
                        &batch.k[entry * kv_width..][..kv_width],
                        &batch.v[entry * kv_width..][..kv_width],
                    )?;
                    crate::profile::note(crate::profile::Phase::KvWrite, at);
                }

                // Every row of the chunk is in the cache now, so the whole chunk's attention is one
                // call that can use every core instead of one entry at a time on the caller.
                let at = crate::profile::stamp();
                let storage: &[u8] = kv;
                let reads = contexts
                    .iter()
                    .map(|context| {
                        Ok(KvRead {
                            storage,
                            table: context.block_table,
                            tokens_per_block: context.tokens_per_block as usize,
                            bytes_per_token: context.bytes_per_token as usize,
                            layout: config.kv_layout(context.bytes_per_token as usize)?,
                        })
                    })
                    .collect::<Result<Vec<_>, ViperaError>>()?;
                attend_entries(
                    &batch.q[..B * query],
                    &reads,
                    &counts,
                    index,
                    shape,
                    &mut batch.attn_scratch,
                    &mut batch.attn[..B * hidden],
                )?;
                crate::profile::note(crate::profile::Phase::Attention, at);

                layer.wo.matvec_batch::<B>(
                    &pool,
                    file,
                    &rows::<B>(&batch.attn, hidden),
                    &Fusion::None,
                    &mut batch.h[..B * hidden],
                )?;
                let at = crate::profile::stamp();
                for entry in 0..B {
                    add_assign(
                        &mut batch.x[entry * hidden..][..hidden],
                        &batch.h[entry * hidden..][..hidden],
                    );
                }
                crate::profile::note(crate::profile::Phase::Elem, at);

                let at = crate::profile::stamp();
                if !fused {
                    for entry in 0..B {
                        rms_norm(
                            &batch.x[entry * hidden..][..hidden],
                            &layer.ffn_norm,
                            epsilon,
                            &mut batch.h[entry * hidden..][..hidden],
                        );
                    }
                }
                crate::profile::note(crate::profile::Phase::Norm, at);
                let norm = if fused {
                    Fusion::Norm {
                        weight: std::sync::Arc::from(&layer.ffn_norm[..]),
                        epsilon,
                    }
                } else {
                    Fusion::None
                };
                let activations = if fused {
                    rows::<B>(&batch.x, hidden)
                } else {
                    rows::<B>(&batch.h, hidden)
                };
                let gate = std::sync::Arc::get_mut(&mut batch.gate).ok_or_else(|| {
                    ViperaError::internal(
                        "the gate buffer is still shared while a product writes it",
                    )
                })?;
                layer.wg.matvec_batch::<B>(
                    &pool,
                    file,
                    &activations,
                    &norm,
                    &mut gate[..B * ffn],
                )?;
                layer.wu.matvec_batch::<B>(
                    &pool,
                    file,
                    &activations,
                    &norm,
                    &mut batch.up[..B * ffn],
                )?;
                let at = crate::profile::stamp();
                let silu = if fused {
                    Fusion::Silu {
                        gate: std::sync::Arc::clone(&batch.gate),
                        width: ffn,
                    }
                } else {
                    silu_mul(&mut batch.up[..B * ffn], &batch.gate[..B * ffn]);
                    Fusion::None
                };
                crate::profile::note(crate::profile::Phase::Elem, at);
                layer.wd.matvec_batch::<B>(
                    &pool,
                    file,
                    &rows::<B>(&batch.up, ffn),
                    &silu,
                    &mut batch.h[..B * hidden],
                )?;
                let at = crate::profile::stamp();
                for entry in 0..B {
                    add_assign(
                        &mut batch.x[entry * hidden..][..hidden],
                        &batch.h[entry * hidden..][..hidden],
                    );
                }
                crate::profile::note(crate::profile::Phase::Elem, at);
            }

            if need_logits && only_last {
                let entry = B - 1;
                rms_norm(
                    &batch.x[entry * hidden..][..hidden],
                    output_norm,
                    epsilon,
                    &mut batch.h[entry * hidden..][..hidden],
                );
                let pool1 = Pool::new(
                    scope,
                    quant::thread_budget(vocab, hidden.saturating_mul(vocab)),
                );
                output.matvec(
                    &pool1,
                    file,
                    rows::<1>(&batch.h[entry * hidden..][..hidden], hidden)[0],
                    &mut batch.logits[..vocab],
                )?;
            } else if need_logits {
                for entry in 0..B {
                    rms_norm(
                        &batch.x[entry * hidden..][..hidden],
                        output_norm,
                        epsilon,
                        &mut batch.h[entry * hidden..][..hidden],
                    );
                }
                if batch.logits_each.len() < B * vocab {
                    batch.logits_each.resize(B * vocab, 0.0);
                }
                output.matvec_batch::<B>(
                    &pool,
                    file,
                    &rows::<B>(&batch.h, hidden),
                    &Fusion::None,
                    &mut batch.logits_each[..B * vocab],
                )?;
            }
            crate::profile::finish_pass(pass);
            if !need_logits {
                return Ok(vec![Vec::new(); B]);
            }
            if only_last {
                let mut results = vec![Vec::new(); B];
                results[B - 1] = batch.logits[..vocab].to_vec();
                return Ok(results);
            }
            Ok((0..B)
                .map(|entry| batch.logits_each[entry * vocab..][..vocab].to_vec())
                .collect())
        })
    }
}

/// `B` rows of `width` values out of one flat, token-major buffer.
fn rows<const B: usize>(data: &[f32], width: usize) -> [&[f32]; B] {
    std::array::from_fn(|entry| &data[entry * width..][..width])
}

/// A mutable array of exactly `B` entries, or an error naming the batch it was given.
fn exactly<const B: usize, T>(slice: &mut [T]) -> Result<&mut [T; B], ViperaError> {
    let len = slice.len();
    slice.try_into().map_err(|_| {
        ViperaError::invalid_argument(format!(
            "this backend batches up to {MAX_BATCH} sequences, not {len}"
        ))
    })
}

impl ForwardPass for CpuBackend {
    fn logits(&mut self, context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
        self.run(context)
    }

    fn logits_batch(
        &mut self,
        batch: &mut BatchContext<'_, '_>,
    ) -> Result<Vec<Vec<f32>>, ViperaError> {
        self.run_batch(batch, true, false)
    }

    fn logits_range(
        &mut self,
        context: &mut ForwardContext<'_>,
    ) -> Result<Vec<Vec<f32>>, ViperaError> {
        // Batch path already computes per-token logits as separate entries,
        // one per position in the chunk. Reuse it so speculative verify is
        // one batched forward instead of k serial forwards.
        let tokens = context.new_tokens;
        let first = context.first_position;
        let kv = std::mem::take(&mut context.kv);
        let mut entries: Vec<BatchEntry<'_>> = (0..tokens.len())
            .map(|index| BatchEntry {
                tokens: context.tokens,
                new_tokens: &tokens[index..index + 1],
                first_position: first + index as u32,
                block_table: context.block_table,
                tokens_per_block: context.tokens_per_block,
                bytes_per_token: context.bytes_per_token,
            })
            .collect();
        let results = self.run_batch(
            &mut BatchContext {
                kv: &mut *kv,
                entries: &mut entries,
            },
            true,
            false,
        )?;
        context.kv = kv;
        Ok(results)
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
        if !matches!(architecture.as_str(), "qwen2" | "qwen3" | "llama") {
            return Err(ViperaError::unsupported(format!(
                "the CPU backend implements the qwen2/qwen3/llama block layouts, and this model is \
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

        // qwen3 states the head width explicitly (`attention.key_length`); qwen2 and llama leave it
        // implicit as `hidden / heads`. They agree on every model here, and reading it lets a head
        // narrower than the average still load.
        let head_dim = match u64_of("attention.key_length") {
            Some(explicit) => explicit,
            None => {
                if heads == 0 || hidden % heads != 0 {
                    return Err(ViperaError::unsupported(format!(
                        "{hidden} hidden values do not divide into {heads} heads"
                    )));
                }
                hidden / heads
            }
        };
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(ViperaError::unsupported(format!(
                "a head width of {head_dim} cannot be rotated"
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
            head_dim,
            ffn,
            vocab,
            rms_eps: f32_of("attention.layer_norm_rms_epsilon", 1e-6),
            rope_base: f32_of("rope.freq_base", 10_000.0),
        })
    }
}

/// The order the bytes of one block are laid out in.
///
/// `TokenMajor` puts every layer of a token together, so reading one layer walks the block with a
/// stride of a whole token (`bytes_per_token`, 28 KB for this model). `LayerMajor` puts each
/// layer's rows for the whole block together, so the same walk moves one row at a time and only
/// crosses into another block every `tokens_per_block` positions. The values a layer holds are
/// identical either way; only the addresses differ.
///
/// Layer-major is the default — it is 28-46% off the walk at depth 1024 and 34-59% at 4096, see
/// `docs/benchmarks.md`. `VIPERA_KV_LAYER_MAJOR=0` (or `false`, `off`, `no`) walks token-major
/// instead, for an A/B in one binary. It is a switch, not a format: the block table, the block
/// size and the row encoding all stay as they are, which is what makes the two arms comparable
/// and the change revertible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowOrder {
    TokenMajor,
    LayerMajor,
}

impl RowOrder {
    fn from_env() -> Self {
        #[cfg(test)]
        if let Some(forced) = forced_order() {
            return forced;
        }
        if flag("VIPERA_KV_LAYER_MAJOR") {
            Self::LayerMajor
        } else {
            Self::TokenMajor
        }
    }

    /// Where a token's K row starts, how far the next token's row is, and how far from K that
    /// same token's V row is. The three numbers are the whole difference between the orders.
    fn row_walk(
        self,
        block_start: usize,
        in_block: usize,
        per_block: usize,
        bytes_per_token: usize,
        layer_bytes: usize,
        layer: usize,
    ) -> RowWalk {
        let k_and_stride = match self {
            Self::TokenMajor => (
                block_start + in_block * bytes_per_token + layer * 2 * layer_bytes,
                bytes_per_token,
            ),
            Self::LayerMajor => (
                block_start + layer * 2 * layer_bytes * per_block + in_block * layer_bytes,
                layer_bytes,
            ),
        };
        RowWalk {
            k: k_and_stride.0,
            stride: k_and_stride.1,
            v: self.v_delta(per_block, layer_bytes),
        }
    }

    /// How far a token's V row sits from its K row.
    const fn v_delta(self, per_block: usize, layer_bytes: usize) -> usize {
        match self {
            Self::TokenMajor => layer_bytes,
            Self::LayerMajor => layer_bytes * per_block,
        }
    }

    /// The V row that pairs with the K row at `k_base`.
    const fn value_row(self, k_base: usize, per_block: usize, layer_bytes: usize) -> usize {
        k_base + self.v_delta(per_block, layer_bytes)
    }
}

/// One block's walk over one layer, as [`RowOrder::row_walk`] describes it.
struct RowWalk {
    k: usize,
    stride: usize,
    v: usize,
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
    /// The physical block `position` lives in, and how far into it `position` is.
    fn token_in_block(&self, position: usize) -> Result<(usize, usize), ViperaError> {
        let block = *self
            .table
            .get(position / self.tokens_per_block)
            .ok_or_else(|| ViperaError::internal("position has no KV block"))?;
        let per_block = self.tokens_per_block;
        Ok((
            block as usize * per_block * self.bytes_per_token,
            position % per_block,
        ))
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
        let per_block = self.tokens_per_block.max(1);
        let (block_start, in_block) = self.token_in_block(position)?;
        let RowWalk {
            k: base,
            v: v_delta,
            ..
        } = RowOrder::from_env().row_walk(
            block_start,
            in_block,
            per_block,
            self.bytes_per_token,
            layer_bytes,
            layer,
        );
        let v_base = base + v_delta;
        for end in [base + layer_bytes, v_base + layer_bytes] {
            if end > self.storage.len() {
                return Err(ViperaError::internal(
                    "KV write would fall outside the cache buffer",
                ));
            }
        }
        self.layout
            .encode(&mut self.storage[base..base + layer_bytes], k)?;
        self.layout
            .encode(&mut self.storage[v_base..v_base + layer_bytes], v)
    }

    /// The whole layer decoded to `f32`. Used for short histories (see [`STREAMING_MIN_POSITIONS`]),
    /// where the buffers are small, and kept as the oracle that [`attention_streaming`] is tested
    /// against, bit for bit. The forward pass reads through [`KvRead`]; this form is what the tests
    /// that build a `KvView` directly use.
    #[cfg(test)]
    fn load_layer(
        &self,
        layer: usize,
        shape: AttnShape,
        positions: usize,
        k_all: &mut Vec<f32>,
        v_all: &mut Vec<f32>,
    ) -> Result<(), ViperaError> {
        load_layer(
            self.storage,
            self.table,
            self.tokens_per_block,
            self.bytes_per_token,
            self.layout,
            layer,
            shape,
            positions,
            k_all,
            v_all,
        )
    }
}

/// [`KvView`]'s rows without the write permission, so the attention workers can read the same
/// buffer from several threads at once.
struct KvRead<'a> {
    storage: &'a [u8],
    table: &'a [u32],
    tokens_per_block: usize,
    bytes_per_token: usize,
    layout: KvLayout,
}

impl<'a> KvRead<'a> {
    /// A read-only view of the same blocks `view` describes.
    fn of(view: &'a KvView<'a>) -> Self {
        Self {
            storage: view.storage,
            table: view.table,
            tokens_per_block: view.tokens_per_block,
            bytes_per_token: view.bytes_per_token,
            layout: view.layout,
        }
    }

    fn for_each_row(
        &self,
        order: RowOrder,
        layer: usize,
        layer_bytes: usize,
        positions: usize,
        visit: impl FnMut(usize, usize) -> Result<(), ViperaError>,
    ) -> Result<(), ViperaError> {
        self.for_each_row_range(order, layer, layer_bytes, 0, positions, visit)
    }

    fn for_each_row_range(
        &self,
        order: RowOrder,
        layer: usize,
        layer_bytes: usize,
        start: usize,
        end: usize,
        visit: impl FnMut(usize, usize) -> Result<(), ViperaError>,
    ) -> Result<(), ViperaError> {
        for_each_row_range(
            self.storage,
            self.table,
            self.tokens_per_block,
            self.bytes_per_token,
            order,
            layer,
            layer_bytes,
            start,
            end,
            visit,
        )
    }

    /// The V row that pairs with the K row `for_each_row` handed the visitor.
    fn value_row(&self, order: RowOrder, k_base: usize, layer_bytes: usize) -> usize {
        order.value_row(k_base, self.tokens_per_block.max(1), layer_bytes)
    }

    fn load_layer(
        &self,
        layer: usize,
        shape: AttnShape,
        positions: usize,
        k_all: &mut Vec<f32>,
        v_all: &mut Vec<f32>,
    ) -> Result<(), ViperaError> {
        load_layer(
            self.storage,
            self.table,
            self.tokens_per_block,
            self.bytes_per_token,
            self.layout,
            layer,
            shape,
            positions,
            k_all,
            v_all,
        )
    }
}

/// [`for_each_row`] restricted to positions `[start, end)`.
#[allow(clippy::too_many_arguments)]
fn for_each_row_range(
    storage: &[u8],
    table: &[u32],
    tokens_per_block: usize,
    bytes_per_token: usize,
    order: RowOrder,
    layer: usize,
    layer_bytes: usize,
    start: usize,
    end: usize,
    mut visit: impl FnMut(usize, usize) -> Result<(), ViperaError>,
) -> Result<(), ViperaError> {
    let per_block = tokens_per_block.max(1);
    let mut position = start.min(end);
    while position < end {
        let block = *table
            .get(position / per_block)
            .ok_or_else(|| ViperaError::internal("position has no KV block"))?
            as usize;
        let block_start = block * per_block * bytes_per_token;
        let in_block = (per_block - position % per_block).min(end - position);
        let RowWalk { mut k, stride, v } = order.row_walk(
            block_start,
            position % per_block,
            per_block,
            bytes_per_token,
            layer_bytes,
            layer,
        );
        for offset in 0..in_block {
            if k + v + layer_bytes > storage.len() {
                return Err(ViperaError::internal(
                    "KV read would fall outside the cache buffer",
                ));
            }
            visit(position + offset, k)?;
            k += stride;
        }
        position += in_block;
    }
    Ok(())
}

/// [`KvView::load_layer`], over a buffer both views can reach.
#[allow(clippy::too_many_arguments)]
fn load_layer(
    storage: &[u8],
    table: &[u32],
    tokens_per_block: usize,
    bytes_per_token: usize,
    layout: KvLayout,
    layer: usize,
    shape: AttnShape,
    positions: usize,
    k_all: &mut Vec<f32>,
    v_all: &mut Vec<f32>,
) -> Result<(), ViperaError> {
    let width = shape.kv_heads * shape.head_dim;
    let layer_bytes = layout.bytes_per_row(width);
    k_all.resize(positions * width, 0.0);
    v_all.resize(positions * width, 0.0);
    // Positions inside one block are a fixed distance apart, so the block table is consulted once
    // per block instead of once per position: at 1024 positions that was a division and a
    // bounds-checked lookup for every row of every layer.
    let per_block = tokens_per_block.max(1);
    let order = RowOrder::from_env();
    let mut position = 0;
    while position < positions {
        let slot = position / per_block;
        // The table holds the physical block, which is not the index whenever two sequences share
        // one buffer.
        let block = *table
            .get(slot)
            .ok_or_else(|| ViperaError::internal("position has no KV block"))?
            as usize;
        let in_block = (per_block - position % per_block).min(positions - position);
        let walk = order.row_walk(
            block * per_block * bytes_per_token,
            position % per_block,
            per_block,
            bytes_per_token,
            layer_bytes,
            layer,
        );
        let mut base = walk.k;
        let (stride, v) = (walk.stride, walk.v);
        for offset in 0..in_block {
            let v_base = base + v;
            if v_base + layer_bytes > storage.len() {
                return Err(ViperaError::internal(
                    "KV read would fall outside the cache buffer",
                ));
            }
            let at = (position + offset) * width;
            layout.decode(
                &storage[base..base + layer_bytes],
                &mut k_all[at..][..width],
            )?;
            layout.decode(
                &storage[v_base..v_base + layer_bytes],
                &mut v_all[at..][..width],
            )?;
            base += stride;
        }
        position += in_block;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct AttnShape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// Causal grouped-query attention over every position up to and including `positions - 1`, on
/// K/V already decoded to `f32`.
///
/// This is the oracle the streaming kernels are tested against, bit for bit.
#[cfg(test)]
fn attention(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    positions: usize,
    shape: AttnShape,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    attention_heads(
        q,
        k_all,
        v_all,
        positions,
        shape,
        0..shape.heads,
        scores,
        out,
    );
}

/// [`attention`] for `heads` alone: `q` is indexed by the absolute head number and `out` starts at
/// head `heads.start`, so the per-head arithmetic and its order are the ones the whole-heads form
/// uses.
#[allow(clippy::too_many_arguments)]
fn attention_heads(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    positions: usize,
    shape: AttnShape,
    heads: std::ops::Range<usize>,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // The score dot runs in eight lanes unless the exact tokens were asked for; the value
    // accumulation is eight lanes either way, because it changes no rounding.
    let sequential = quant::exact_dot_requested();
    scores.resize(positions, 0.0);
    // Fused softmax+normalize into one pass over the window after scoring:
    // exp and divide in the same loop eliminates one full O(positions) pass.
    // Total must be accumulated in the same (increasing position) order as
    // before so rounding is unchanged.
    for head in heads.clone() {
        let kv_head = head / group;
        let query = &q[head * head_dim..(head + 1) * head_dim];
        let mut best = f32::NEG_INFINITY;
        for (position, score) in scores.iter_mut().enumerate().take(positions) {
            let base = (position * shape.kv_heads + kv_head) * head_dim;
            let key = &k_all[base..base + head_dim];
            let dot = if sequential {
                query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>()
            } else {
                crate::simd::dot_f32(query, key)
            };
            *score = dot * scale;
            best = best.max(*score);
        }
        let window = &mut scores[..positions];
        let total = if fast_softmax_enabled() {
            crate::simd::exp_sum_shifted(window, best)
        } else {
            let mut total = 0.0_f32;
            for score in window.iter_mut() {
                *score = (*score - best).exp();
                total += *score;
            }
            total
        };
        let inverse = 1.0 / total;
        let output = &mut out[(head - heads.start) * head_dim..][..head_dim];
        output.fill(0.0);
        for (position, weight) in scores.iter().take(positions).enumerate() {
            let base = (position * shape.kv_heads + kv_head) * head_dim;
            crate::simd::axpy_f32(output, weight * inverse, &v_all[base..base + head_dim]);
        }
    }
}

/// Working memory for [`attention_streaming`], kept between calls so a token allocates nothing.
#[derive(Debug, Default)]
struct AttnScratch {
    /// `heads * positions` scores, head-major.
    scores: Vec<f32>,
    /// One decoded K or V row, `kv_heads * head_dim` values.
    row: Vec<f32>,
    /// Per head: the running maximum, then its softmax normalizer.
    head_state: Vec<f32>,
    /// The whole layer decoded to `f32`, for [`attend`]'s short-history path. Empty until a rule
    /// with fewer than [`STREAMING_MIN_POSITIONS`] positions runs.
    k_all: Vec<f32>,
    /// The matching V rows. Same lifecycle as [`Self::k_all`].
    v_all: Vec<f32>,
}

/// Below this many positions, [`attend`] decodes the layer once and uses the buffered kernel;
/// at or above it, it reads the KV blocks twice through [`attention_streaming`].
///
/// The threshold is a memory decision, not a speed one. Speed was measured and the two arms are
/// the same end to end below it: on the Qwen2.5-1.5B / i5-11300H rig, `pp128` runs 45.0 tok/s with
/// the gate and 44.4 tok/s with the buffered kernel alone, which is run-to-run spread (a 47.1
/// sample taken earlier does not reproduce). What the gate buys is that the buffered arm's two
/// `positions * width` buffers stay small: end to end decode at ctx 1024 is 51.0 ms/step through
/// [`attention_streaming`] against 56.1 ms/step through the buffered path, and the streaming arm
/// needs no history buffer at all.
const STREAMING_MIN_POSITIONS: usize = 256;

/// Score lanes (entries times heads times positions) below which attention stays on the calling
/// thread, because starting the workers costs more than the split saves.
const PARALLEL_ATTENTION_MIN: usize = 4096;

/// Attention's switches, read once per call rather than per row.
///
/// A switch here cannot change a bit of the result — the head split only decides which thread
/// computes which head — so one process can time both arms on the same bytes. `VIPERA_ATTN_HEAD_
/// SPLIT=0` (or `false`, `off`, `no`) keeps a single entry's heads on the calling thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AttentionSwitches {
    /// A single entry's heads go to the pool instead of the calling thread.
    split_heads: bool,
    /// Split a single entry's KV history across workers and merge the partials (approximate,
    /// not bit-identical to the serial kernel). On by default; `VIPERA_ATTN_KV_SPLIT=0` restores
    /// the bit-exact serial/head-split path.
    kv_split: bool,
    /// Which order the KV bytes are walked in.
    order: RowOrder,
    /// At most this many workers share a single entry's heads; `None` is one per head up to the
    /// physical core count (see [`physical_cores`]). Each worker is a thread created and joined
    /// for the one layer, so more workers divide the arithmetic finer and pay for one more thread;
    /// past the physical core count the siblings contend instead. `VIPERA_ATTN_WORKERS=N`.
    workers: Option<usize>,
}

impl AttentionSwitches {
    fn from_env() -> Self {
        #[cfg(test)]
        if let Some(forced) = forced_switches() {
            return forced;
        }
        Self {
            split_heads: flag("VIPERA_ATTN_HEAD_SPLIT"),
            kv_split: flag("VIPERA_ATTN_KV_SPLIT"),
            order: RowOrder::from_env(),
            workers: std::env::var("VIPERA_ATTN_WORKERS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|workers| *workers > 0),
        }
    }
}

/// `VIPERA_<NAME>=0|false|off|no` turns a switch off; unset or anything else leaves it on.
fn flag(name: &str) -> bool {
    std::env::var(name).map_or(true, |value| {
        let value = value.trim().to_ascii_lowercase();
        !matches!(value.as_str(), "0" | "false" | "off" | "no")
    })
}

/// The switches a measurement wants, so one process can time all four combinations. Only the
/// benchmark sets it, and it only reaches performance code: unset means "ask the environment".
#[cfg(test)]
fn forced_switches() -> Option<AttentionSwitches> {
    use std::sync::atomic::Ordering;
    let forced = BENCH_SWITCHES.load(Ordering::Relaxed);
    let workers = BENCH_WORKERS.load(Ordering::Relaxed);
    (forced & 4 != 0).then_some(AttentionSwitches {
        split_heads: forced & 1 != 0,
        kv_split: forced & 8 != 0,
        order: if forced & 2 != 0 {
            RowOrder::LayerMajor
        } else {
            RowOrder::TokenMajor
        },
        workers: (workers != 0).then_some(workers),
    })
}

#[cfg(test)]
fn forced_order() -> Option<RowOrder> {
    forced_switches().map(|switches| switches.order)
}

#[cfg(test)]
fn force_switches(switches: AttentionSwitches) {
    use std::sync::atomic::Ordering;
    let bits = 4
        | u8::from(switches.split_heads)
        | (u8::from(switches.order == RowOrder::LayerMajor) << 1)
        | (u8::from(switches.kv_split) << 3);
    BENCH_SWITCHES.store(bits, Ordering::Relaxed);
    BENCH_WORKERS.store(switches.workers.unwrap_or(0), Ordering::Relaxed);
}

#[cfg(test)]
fn clear_forced_switches() {
    use std::sync::atomic::Ordering;
    BENCH_SWITCHES.store(0, Ordering::Relaxed);
    BENCH_WORKERS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
static BENCH_SWITCHES: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The worker cap a measurement forces, `0` for none.
#[cfg(test)]
static BENCH_WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many workers share one entry's heads: one per head, up to the thread count and the cap, and
/// never fewer than one.
#[inline]
fn attention_workers(threads: usize, heads: usize, cap: Option<usize>) -> usize {
    threads.min(heads).min(cap.unwrap_or(usize::MAX)).max(1)
}

/// The heads worker `part` of `workers` computes: contiguous runs that cover `0..heads` exactly
/// once and differ in length by at most one.
#[inline]
fn head_range(heads: usize, workers: usize, part: usize) -> std::ops::Range<usize> {
    heads * part / workers..heads * (part + 1) / workers
}

/// Physical cores, read once.
///
/// The matvec pool is memory-bound and runs on every logical processor the OS reports, but this
/// per-layer head split is compute-bound and builds its workers per layer: on an SMT machine the
/// sibling threads contend for the same core and the extra spawns cost more than the finer split
/// saves. Measured on the 4-core/8-thread i5-11300H: 4 workers beat 8 in every sweep at depth 1024
/// and 4096 (-2% to -20%) and were mixed at 8192. Falls back to the configured thread count when topology
/// is unavailable.
fn physical_cores() -> usize {
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CORES.get_or_init(|| {
        #[cfg(target_os = "linux")]
        if let Some(cores) = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| vipera_core::hardware::parse_physical_cores(&text))
        {
            return usize::try_from(cores).unwrap_or(1).max(1);
        }
        crate::quant::max_threads()
    })
}

fn in_place_kv_enabled() -> bool {
    std::env::var("VIPERA_INPLACE_KV").map_or(true, |value| {
        let value = value.trim().to_ascii_lowercase();
        !matches!(value.as_str(), "0" | "false" | "off" | "no")
    })
}

/// Whether the streaming and window kernels let the query heads of one key head share each K and V
/// row.
///
/// On unless `VIPERA_ATTN_GQA_SHARE` is set to `0`, `false`, `off` or `no`. It fuses the multiply-add
/// and regroups each head's sum, so it is not bit-identical to the per-head kernel: the tests hold
/// the two within rounding of each other rather than equal. Read once.
fn gqa_share_enabled() -> bool {
    #[cfg(test)]
    match BENCH_SHARE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static SHARE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| {
        std::env::var("VIPERA_ATTN_GQA_SHARE").map_or(true, |value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
    })
}

/// What a measurement forces [`gqa_share_enabled`] to: `0` leaves the environment in charge, `1`
/// forces it off and `2` on. Like [`BENCH_SWITCHES`], it exists so one process can time both arms.
#[cfg(test)]
static BENCH_SHARE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
fn force_share(share: Option<bool>) {
    let bits = match share {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    BENCH_SHARE.store(bits, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the softmax pass takes [`crate::simd::exp_sum_shifted`] instead of `f32::exp`.
///
/// On unless `VIPERA_FAST_SOFTMAX` is set to `0`, `false`, `off` or `no`. It evaluates the exponent
/// with the polynomial [`crate::model::exp`] rather than libm's `expf`: the two agree to about 2e-6
/// relative, which is not the same bits, so a run's tokens are not bit-identical to the libm path.
/// Read once.
fn fast_softmax_enabled() -> bool {
    static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FAST.get_or_init(|| {
        std::env::var("VIPERA_FAST_SOFTMAX").map_or(true, |value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
    })
}

/// The most query heads per key head the shared kernels take; a layer with a larger group keeps the
/// per-head kernels. Sizes the stack arrays below.
const MAX_SHARED_GROUP: usize = 16;

/// Causal grouped-query attention read straight from the KV blocks.
///
/// This computes exactly what `load_layer` followed by `attention` computes, bit for bit: every
/// head sees the same values, and sums them in the same order. What changes is the traffic. Each
/// K and V row is decoded once into a one-row buffer that stays in L1 while every query head of
/// its group uses it, instead of the whole history being decoded into two `positions * width`
/// buffers that every head then re-reads from L2 or L3.
#[cfg(test)]
fn attention_streaming(
    q: &[f32],
    kv: &KvView<'_>,
    layer: usize,
    positions: usize,
    shape: AttnShape,
    scratch: &mut AttnScratch,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let AttnScratch {
        scores,
        row,
        head_state,
        ..
    } = scratch;
    attention_streaming_heads(
        q,
        &KvRead::of(kv),
        AttentionSwitches::from_env().order,
        gqa_share_enabled(),
        layer,
        positions,
        shape,
        0..shape.heads,
        scores,
        row,
        head_state,
        out,
    )
}

/// The weighted sum of V for a block of positions, all heads at once: one call per KV head into
/// [`axpy_f16_f32_heads_exact_block`](crate::simd::axpy_f16_f32_heads_exact_block), so each head's
/// output row is read and written once per block rather than once per position.
#[allow(clippy::too_many_arguments)]
fn sum_v_block<const P: usize>(
    out: &mut [f32],
    storage: &[u8],
    scores: &[f32],
    head_state: &[f32],
    positions: usize,
    kv_heads: usize,
    group: usize,
    head_dim: usize,
    bases: &[usize; P],
    at_positions: &[usize; P],
) {
    for kv_head in 0..kv_heads {
        let first = kv_head * group;
        let mut weights = [0.0_f32; 128];
        for (p, position) in at_positions.iter().enumerate() {
            for member in 0..group {
                let local = first + member;
                weights[p * group + member] =
                    scores[local * positions + position] * head_state[local];
            }
        }
        let mut rows: [&[u8]; P] = [&[]; P];
        for (p, base) in bases.iter().enumerate() {
            rows[p] = &storage[base + kv_head * head_dim * 2..];
        }
        crate::simd::axpy_f16_f32_heads_exact_block::<P>(
            &mut out[first * head_dim..(first + group) * head_dim],
            head_dim,
            &weights[..P * group],
            &rows,
        );
    }
}

/// [`attention_streaming`] for `heads` alone, writing their own scratch buffers so several
/// workers can run at once. Everything each head sums is unchanged, in the same order, so a head
/// computes the same bits whichever worker holds it.
///
/// With `share` and `f16` rows read in place, the heads of one key head score each K row together
/// and fold each block of V rows together ([`crate::simd::dot_f16_group`],
/// [`crate::simd::axpy_f16_group`]): the row is converted once per group instead of once per head,
/// and the output accumulators stay in registers across a block of positions. That is the same sum
/// in the same position order, fused and regrouped, so it is within rounding of the per-head
/// kernel and not bit-identical to it. Every other layout, and `share` off, runs the per-head
/// kernel unchanged.
#[allow(clippy::too_many_arguments)]
fn attention_streaming_heads(
    q: &[f32],
    kv: &KvRead<'_>,
    order: RowOrder,
    share: bool,
    layer: usize,
    positions: usize,
    shape: AttnShape,
    heads: std::ops::Range<usize>,
    scores: &mut Vec<f32>,
    row: &mut Vec<f32>,
    head_state: &mut Vec<f32>,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let width = shape.kv_heads * head_dim;
    let layer_bytes = kv.layout.bytes_per_row(width);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sequential = quant::exact_dot_requested();
    // An f16 row is already the value the kernel reads, so the attention passes read it in place
    // instead of decoding it into `row` first: same bits (the conversion is exact and the dot tree
    // is the same one), no 2 MB of row buffer per layer per pass. `VIPERA_INPLACE_KV=0` keeps the
    // decoded arm for an A/B in one binary.
    let in_place = in_place_kv_enabled()
        && crate::simd::IN_PLACE_KV_BIT_EXACT
        && kv.layout == KvLayout::F16
        && !sequential;
    let shared = share && in_place && group <= MAX_SHARED_GROUP;
    let count = heads.end - heads.start;
    scores.clear();
    scores.resize(count * positions, 0.0);
    row.clear();
    row.resize(width, 0.0);
    head_state.clear();
    head_state.resize(count, f32::NEG_INFINITY);
    // A decode or prefill step asks for every head, and the heads of one KV head share its row.
    let whole = heads.start == 0 && heads.end == shape.heads && group <= 16;

    // Pass 1: every score, and each head's maximum, in increasing position order.
    kv.for_each_row(order, layer, layer_bytes, positions, |position, base| {
        if shared {
            for kv_head in 0..shape.kv_heads {
                let first = (kv_head * group).max(heads.start);
                let last = ((kv_head + 1) * group).min(heads.end);
                if first >= last {
                    continue;
                }
                let used = last - first;
                let key = &kv.storage[base + kv_head * head_dim * 2..][..head_dim * 2];
                let mut dots = [0.0_f32; MAX_SHARED_GROUP];
                crate::simd::dot_f16_group(
                    &q[first * head_dim..last * head_dim],
                    head_dim,
                    key,
                    &mut dots[..used],
                );
                for (member, dot) in dots[..used].iter().enumerate() {
                    let local = first + member - heads.start;
                    let score = dot * scale;
                    scores[local * positions + position] = score;
                    head_state[local] = head_state[local].max(score);
                }
            }
            return Ok(());
        }
        if !in_place {
            kv.layout
                .decode(&kv.storage[base..base + layer_bytes], row)?;
        }
        // Heads that share a KV head read the same row, so they are scored in one call: same
        // arithmetic as `dot_f16_f32` per head (the bit-for-bit oracle test still holds), one row
        // load and one feature check instead of `group` of them.
        for kv_head in 0..shape.kv_heads {
            let key = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
            let packed = (base + kv_head * head_dim * 2)..;
            if whole && in_place && !sequential {
                let first = kv_head * group;
                let mut dots = [0.0_f32; 16];
                crate::simd::dot_f16_f32_heads_exact(
                    q,
                    first,
                    group,
                    head_dim,
                    &kv.storage[packed.start..],
                    &mut dots[..group],
                );
                for (member, dot) in dots[..group].iter().enumerate() {
                    let local = first + member;
                    let score = *dot * scale;
                    scores[local * positions + position] = score;
                    head_state[local] = head_state[local].max(score);
                }
                continue;
            }
            for member in 0..group {
                let head = kv_head * group + member;
                if head < heads.start || head >= heads.end {
                    continue;
                }
                let local = head - heads.start;
                let query = &q[head * head_dim..(head + 1) * head_dim];
                let dot = if sequential {
                    query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>()
                } else if in_place {
                    crate::simd::dot_f16_f32(query, &kv.storage[packed.start..][..head_dim * 2])
                } else {
                    crate::simd::dot_f32(query, key)
                };
                let score = dot * scale;
                scores[local * positions + position] = score;
                head_state[local] = head_state[local].max(score);
            }
        }
        Ok(())
    })?;

    // Softmax per head, summed in increasing position order.
    let fast_softmax = fast_softmax_enabled();
    for local in 0..count {
        let best = head_state[local];
        let window = &mut scores[local * positions..(local + 1) * positions];
        let total = if fast_softmax {
            crate::simd::exp_sum_shifted(window, best)
        } else {
            let mut total = 0.0_f32;
            for score in window.iter_mut() {
                *score = (*score - best).exp();
                total += *score;
            }
            total
        };
        head_state[local] = 1.0 / total;
    }

    // Pass 2: the weighted sum of V, each head accumulating in increasing position order.
    out[..count * head_dim].fill(0.0);
    if shared {
        let mut value_bases = [0_usize; crate::simd::GROUP_ROWS];
        let mut value_positions = [0_usize; crate::simd::GROUP_ROWS];
        let mut pending = 0;
        kv.for_each_row(order, layer, layer_bytes, positions, |position, base| {
            value_bases[pending] = kv.value_row(order, base, layer_bytes);
            value_positions[pending] = position;
            pending += 1;
            if pending == crate::simd::GROUP_ROWS {
                fold_value_block(
                    kv.storage,
                    &value_bases,
                    &value_positions,
                    shape,
                    &heads,
                    positions,
                    scores,
                    head_state,
                    out,
                );
                pending = 0;
            }
            Ok(())
        })?;
        if pending > 0 {
            fold_value_block(
                kv.storage,
                &value_bases[..pending],
                &value_positions[..pending],
                shape,
                &heads,
                positions,
                scores,
                head_state,
                out,
            );
        }
        return Ok(());
    }
    // Eight positions at a time keeps each head's accumulator in registers between positions, so the
    // targets are loaded and stored once per eight rows instead of once per row. Per element the
    // additions still run in increasing position order, so the result is the per-row kernel's bits.
    if whole && in_place && !sequential && group <= 8 {
        const BATCH: usize = 8;
        let mut bases = [0usize; BATCH];
        let mut at_positions = [0usize; BATCH];
        let mut held = 0usize;
        kv.for_each_row(order, layer, layer_bytes, positions, |position, base| {
            bases[held] = kv.value_row(order, base, layer_bytes);
            at_positions[held] = position;
            held += 1;
            if held == BATCH {
                sum_v_block(
                    out,
                    kv.storage,
                    scores,
                    head_state,
                    positions,
                    shape.kv_heads,
                    group,
                    head_dim,
                    &bases,
                    &at_positions,
                );
                held = 0;
            }
            Ok(())
        })?;
        // A remainder shorter than a block keeps the per-row kernel, which is the same arithmetic.
        for slot in 0..held {
            let position = at_positions[slot];
            let row = &kv.storage[bases[slot]..];
            for kv_head in 0..shape.kv_heads {
                let first = kv_head * group;
                let mut weights = [0.0_f32; 16];
                for (member, weight) in weights[..group].iter_mut().enumerate() {
                    let local = first + member;
                    *weight = scores[local * positions + position] * head_state[local];
                }
                crate::simd::axpy_f16_f32_heads_exact(
                    &mut out[first * head_dim..(first + group) * head_dim],
                    head_dim,
                    &weights[..group],
                    &row[kv_head * head_dim * 2..],
                );
            }
        }
        return Ok(());
    }
    kv.for_each_row(order, layer, layer_bytes, positions, |position, base| {
        let v_base = kv.value_row(order, base, layer_bytes);
        if !in_place {
            kv.layout
                .decode(&kv.storage[v_base..v_base + layer_bytes], row)?;
        }
        for kv_head in 0..shape.kv_heads {
            let value = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
            let packed = (v_base + kv_head * head_dim * 2)..;
            if whole && in_place {
                let first = kv_head * group;
                let mut weights = [0.0_f32; 16];
                for (member, weight) in weights[..group].iter_mut().enumerate() {
                    let local = first + member;
                    *weight = scores[local * positions + position] * head_state[local];
                }
                crate::simd::axpy_f16_f32_heads_exact(
                    &mut out[first * head_dim..(first + group) * head_dim],
                    head_dim,
                    &weights[..group],
                    &kv.storage[packed.start..],
                );
                continue;
            }
            for member in 0..group {
                let head = kv_head * group + member;
                if head < heads.start || head >= heads.end {
                    continue;
                }
                let local = head - heads.start;
                let weight = scores[local * positions + position] * head_state[local];
                let target = &mut out[local * head_dim..(local + 1) * head_dim];
                if in_place {
                    crate::simd::axpy_f16_f32(
                        target,
                        weight,
                        &kv.storage[packed.start..][..head_dim * 2],
                    );
                } else {
                    crate::simd::axpy_f32(target, weight, value);
                }
            }
        }
        Ok(())
    })
}

/// Tokens kept at the front of the history and positions kept before the current one, read from the
/// environment: `VIPERA_ATTN_WINDOW=<positions>` (0 or unset leaves the full history) and
/// `VIPERA_ATTN_SINKS=<tokens>` (default 4).
///
/// This is the only lever that makes a decode step cost the same at every depth, and it is lossy: the
/// model stops seeing the middle of its history. It exists because attention at depth is O(positions)
/// of real arithmetic (see `docs/benchmarks.md`, 2026-11-18) and no exact loop shape removes it.
fn attention_window() -> Option<(usize, usize)> {
    vipera_core::window::attention_window()
        .map(|window| (window.tokens as usize, window.sinks as usize))
}

/// Attention over the sinks and the last `window` positions, instead of the whole history.
///
/// The positions kept are the union of `[0, sinks)` and `[positions - window, positions)`; their
/// scores are collected side by side and go through one softmax, then one weighted sum of V. That is
/// exactly full causal attention over a history whose middle has been dropped, so the arithmetic per
/// kept position is the streaming kernel's, and the cost stops growing once `positions` passes
/// `window + sinks`.
///
/// With `share` and `f16` rows read in place, the kept rows are scored and folded exactly as
/// [`attention_streaming_heads`] does it: each K row against a key head's whole group of query heads
/// in one call, and each block of up to [`crate::simd::GROUP_ROWS`] V rows folded into the outputs
/// together. The kept positions stay numbered `[0, kept)` across the gap the dropped middle leaves,
/// which is the index the score matrix uses, and that index is what [`fold_value_block`] is given.
#[allow(clippy::too_many_arguments)]
fn attention_window_heads(
    q: &[f32],
    kv: &KvRead<'_>,
    order: RowOrder,
    share: bool,
    layer: usize,
    positions: usize,
    sinks: usize,
    window: usize,
    shape: AttnShape,
    heads: std::ops::Range<usize>,
    scratch: &mut AttnScratch,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let width = shape.kv_heads * head_dim;
    let layer_bytes = kv.layout.bytes_per_row(width);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sequential = quant::exact_dot_requested();
    let count = heads.end - heads.start;
    if count == 0 || positions == 0 {
        return Ok(());
    }
    let in_place = in_place_kv_enabled()
        && crate::simd::IN_PLACE_KV_BIT_EXACT
        && kv.layout == KvLayout::F16
        && !sequential;
    let shared = share && in_place && group <= MAX_SHARED_GROUP;
    let sinks_end = sinks.min(positions);
    let window_start = positions.saturating_sub(window).max(sinks_end);
    let kept = sinks_end + (positions - window_start);
    let ranges = [(0, sinks_end), (window_start, positions)];

    let scores = &mut scratch.scores;
    let row = &mut scratch.row;
    let head_state = &mut scratch.head_state;
    scores.clear();
    scores.resize(count * kept, 0.0);
    row.clear();
    row.resize(width, 0.0);
    head_state.clear();
    head_state.resize(count, f32::NEG_INFINITY);

    // Pass 1: the kept positions' scores, in increasing position order.
    let mut slot = 0;
    for (from, to) in ranges {
        if from >= to {
            continue;
        }
        let base_slot = slot;
        kv.for_each_row_range(order, layer, layer_bytes, from, to, |position, base| {
            if shared {
                let at_slot = base_slot + position - from;
                for kv_head in 0..shape.kv_heads {
                    let first = (kv_head * group).max(heads.start);
                    let last = ((kv_head + 1) * group).min(heads.end);
                    if first >= last {
                        continue;
                    }
                    let used = last - first;
                    let key = &kv.storage[base + kv_head * head_dim * 2..][..head_dim * 2];
                    let mut dots = [0.0_f32; MAX_SHARED_GROUP];
                    crate::simd::dot_f16_group(
                        &q[first * head_dim..last * head_dim],
                        head_dim,
                        key,
                        &mut dots[..used],
                    );
                    for (member, dot) in dots[..used].iter().enumerate() {
                        let local = first + member - heads.start;
                        let score = dot * scale;
                        scores[local * kept + at_slot] = score;
                        head_state[local] = head_state[local].max(score);
                    }
                }
                return Ok(());
            }
            if !in_place {
                kv.layout
                    .decode(&kv.storage[base..base + layer_bytes], row)?;
            }
            for kv_head in 0..shape.kv_heads {
                let key = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                let packed = (base + kv_head * head_dim * 2)..;
                for member in 0..group {
                    let head = kv_head * group + member;
                    if head < heads.start || head >= heads.end {
                        continue;
                    }
                    let local = head - heads.start;
                    let query = &q[head * head_dim..(head + 1) * head_dim];
                    let dot = if sequential {
                        query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>()
                    } else if in_place {
                        crate::simd::dot_f16_f32(query, &kv.storage[packed.start..][..head_dim * 2])
                    } else {
                        crate::simd::dot_f32(query, key)
                    };
                    let score = dot * scale;
                    scores[local * kept + base_slot + position - from] = score;
                    head_state[local] = head_state[local].max(score);
                }
            }
            Ok(())
        })?;
        slot += to - from;
    }

    // Softmax over the kept positions, in the order they collected.
    for head in 0..count {
        let best = head_state[head];
        let window_scores = &mut scores[head * kept..(head + 1) * kept];
        let mut total = 0.0_f32;
        for score in window_scores.iter_mut() {
            *score = (*score - best).exp();
            total += *score;
        }
        head_state[head] = 1.0 / total;
    }

    // Pass 2: the weighted sum of V over the same positions.
    out[..count * head_dim].fill(0.0);
    if shared {
        // Kept rows arrive in kept order across the two runs, so a block can carry over the gap.
        let mut bases = [0_usize; crate::simd::GROUP_ROWS];
        let mut slots = [0_usize; crate::simd::GROUP_ROWS];
        let mut pending = 0;
        let mut next = 0;
        for (from, to) in ranges {
            if from >= to {
                continue;
            }
            kv.for_each_row_range(order, layer, layer_bytes, from, to, |_position, base| {
                bases[pending] = kv.value_row(order, base, layer_bytes);
                slots[pending] = next;
                next += 1;
                pending += 1;
                if pending == crate::simd::GROUP_ROWS {
                    fold_value_block(
                        kv.storage, &bases, &slots, shape, &heads, kept, scores, head_state, out,
                    );
                    pending = 0;
                }
                Ok(())
            })?;
        }
        if pending > 0 {
            fold_value_block(
                kv.storage,
                &bases[..pending],
                &slots[..pending],
                shape,
                &heads,
                kept,
                scores,
                head_state,
                out,
            );
        }
        return Ok(());
    }
    slot = 0;
    for (from, to) in ranges {
        if from >= to {
            continue;
        }
        let base_slot = slot;
        kv.for_each_row_range(order, layer, layer_bytes, from, to, |position, base| {
            let value_base = kv.value_row(order, base, layer_bytes);
            if !in_place {
                kv.layout
                    .decode(&kv.storage[value_base..value_base + layer_bytes], row)?;
            }
            for kv_head in 0..shape.kv_heads {
                let value = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                let packed = (value_base + kv_head * head_dim * 2)..;
                for member in 0..group {
                    let head = kv_head * group + member;
                    if head < heads.start || head >= heads.end {
                        continue;
                    }
                    let local = head - heads.start;
                    let weight =
                        scores[local * kept + base_slot + position - from] * head_state[local];
                    let target = &mut out[local * head_dim..(local + 1) * head_dim];
                    if in_place {
                        crate::simd::axpy_f16_f32(
                            target,
                            weight,
                            &kv.storage[packed.start..][..head_dim * 2],
                        );
                    } else {
                        crate::simd::axpy_f32(target, weight, value);
                    }
                }
            }
            Ok(())
        })?;
        slot += to - from;
    }
    Ok(())
}

/// Folds a block of V rows into the outputs of `heads`, every key head's group of query heads
/// together. `bases[i]` is the byte offset of the V row whose score slot is `slots[i]`, in a score
/// matrix of `stride` slots per head, and the weight of a head at a position is its softmax
/// numerator times its normalizer, as the per-head pass computes it.
#[allow(clippy::too_many_arguments)]
fn fold_value_block(
    storage: &[u8],
    bases: &[usize],
    slots: &[usize],
    shape: AttnShape,
    heads: &std::ops::Range<usize>,
    stride: usize,
    scores: &[f32],
    head_state: &[f32],
    out: &mut [f32],
) {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let rows_in_block = bases.len().min(slots.len());
    // The buffers below are sized from these, so a caller that outgrows them is a bug here and not
    // an index panic two loops later.
    assert!(
        rows_in_block <= crate::simd::GROUP_ROWS && group <= MAX_SHARED_GROUP,
        "a value block of {rows_in_block} rows and {group} heads exceeds the \
         {} x {MAX_SHARED_GROUP} arrays",
        crate::simd::GROUP_ROWS
    );
    for kv_head in 0..shape.kv_heads {
        let first = (kv_head * group).max(heads.start);
        let last = ((kv_head + 1) * group).min(heads.end);
        if first >= last {
            continue;
        }
        let used = last - first;
        let mut weights = [0.0_f32; crate::simd::GROUP_ROWS * MAX_SHARED_GROUP];
        let mut rows: [&[u8]; crate::simd::GROUP_ROWS] = [&[]; crate::simd::GROUP_ROWS];
        for slot in 0..rows_in_block {
            rows[slot] = &storage[bases[slot] + kv_head * head_dim * 2..][..head_dim * 2];
            for member in 0..used {
                let local = first + member - heads.start;
                weights[slot * used + member] =
                    scores[local * stride + slots[slot]] * head_state[local];
            }
        }
        crate::simd::axpy_f16_group(
            &mut out[(first - heads.start) * head_dim..(last - heads.start) * head_dim],
            head_dim,
            used,
            &weights[..rows_in_block * used],
            &rows[..rows_in_block],
        );
    }
}

/// One block of up to [`crate::simd::GROUP_ROWS`] consecutive positions folded into a KV chunk's
/// running `(max, sum_exp, weighted_v)` for every head in `heads`, the query heads of each key head
/// together.
///
/// The online softmax advances a block at a time instead of a position at a time: the block's scores
/// are computed first ([`crate::simd::dot_f16_group`]), each head's running maximum is raised to the
/// block's once and the accumulator and the sum rescaled once if it moved, then every position's
/// weight `exp(score - max)` is taken (one `exp` per head per position, as before) and the block's
/// V rows are folded in together ([`crate::simd::axpy_f16_group`]). The same softmax, regrouped, so
/// it is within rounding of the per-position update and not bit-identical to it.
///
/// `key_bases[i]` and `value_bases[i]` are the byte offsets of position `i`'s K and V rows;
/// `block_scores` holds `GROUP_ROWS` slots per head; `m`, `d` and `o` are the chunk's running state,
/// `o` with `head_dim` values per head.
#[allow(clippy::too_many_arguments)]
fn fold_chunk_block(
    storage: &[u8],
    key_bases: &[usize],
    value_bases: &[usize],
    q: &[f32],
    shape: AttnShape,
    heads: &std::ops::Range<usize>,
    scale: f32,
    block_scores: &mut [f32],
    m: &mut [f32],
    d: &mut [f32],
    o: &mut [f32],
) {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let count = heads.end - heads.start;
    let rows_in_block = key_bases.len().min(value_bases.len());
    // As in [`fold_value_block`]: a block larger than the stack arrays or the score buffer is a
    // caller bug, and it should say so instead of indexing past the end.
    assert!(
        rows_in_block <= crate::simd::GROUP_ROWS
            && block_scores.len() >= count * crate::simd::GROUP_ROWS,
        "a block of {rows_in_block} rows for {count} heads needs {} slots, got {}",
        count * crate::simd::GROUP_ROWS,
        block_scores.len()
    );
    for (slot, &base) in key_bases[..rows_in_block].iter().enumerate() {
        for kv_head in 0..shape.kv_heads {
            let first = (kv_head * group).max(heads.start);
            let last = ((kv_head + 1) * group).min(heads.end);
            if first >= last {
                continue;
            }
            let used = last - first;
            let key = &storage[base + kv_head * head_dim * 2..][..head_dim * 2];
            let mut dots = [0.0_f32; MAX_SHARED_GROUP];
            crate::simd::dot_f16_group(
                &q[first * head_dim..last * head_dim],
                head_dim,
                key,
                &mut dots[..used],
            );
            for (member, dot) in dots[..used].iter().enumerate() {
                let local = first + member - heads.start;
                block_scores[local * crate::simd::GROUP_ROWS + slot] = dot * scale;
            }
        }
    }
    for local in 0..count {
        let scores = &mut block_scores[local * crate::simd::GROUP_ROWS..][..rows_in_block];
        let block_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if block_max > m[local] {
            // A new maximum rescales the running sum and the accumulator by `exp(old_max - max)`.
            let alpha = (m[local] - block_max).exp();
            m[local] = block_max;
            d[local] *= alpha;
            for value in &mut o[local * head_dim..(local + 1) * head_dim] {
                *value *= alpha;
            }
        }
        let sum = if fast_softmax_enabled() {
            crate::simd::exp_sum_shifted(scores, m[local])
        } else {
            let mut sum = 0.0_f32;
            for score in scores.iter_mut() {
                *score = (*score - m[local]).exp();
                sum += *score;
            }
            sum
        };
        d[local] += sum;
    }
    for kv_head in 0..shape.kv_heads {
        let first = (kv_head * group).max(heads.start);
        let last = ((kv_head + 1) * group).min(heads.end);
        if first >= last {
            continue;
        }
        let used = last - first;
        let mut weights = [0.0_f32; crate::simd::GROUP_ROWS * MAX_SHARED_GROUP];
        let mut rows: [&[u8]; crate::simd::GROUP_ROWS] = [&[]; crate::simd::GROUP_ROWS];
        for slot in 0..rows_in_block {
            rows[slot] = &storage[value_bases[slot] + kv_head * head_dim * 2..][..head_dim * 2];
            for member in 0..used {
                let local = first + member - heads.start;
                weights[slot * used + member] =
                    block_scores[local * crate::simd::GROUP_ROWS + slot];
            }
        }
        crate::simd::axpy_f16_group(
            &mut o[(first - heads.start) * head_dim..(last - heads.start) * head_dim],
            head_dim,
            used,
            &weights[..rows_in_block * used],
            &rows[..rows_in_block],
        );
    }
}

/// Decode attention split by KV position chunks: each worker computes a partial `(max, sum_exp,
/// weighted_v)` over a disjoint range of positions for every requested head, then the caller merges
/// the partials in increasing chunk order. The merge is associative in exact math but groups
/// floating-point sums differently from the serial two-pass kernel, so the result can differ by
/// rounding (measured ~1e-7 max-abs on the 12×128 decode shape) and is not bit-identical to
/// [`attention_streaming_heads`].
#[allow(clippy::too_many_arguments)]
fn attention_kv_chunked_heads(
    q: &[f32],
    kv: &KvRead<'_>,
    order: RowOrder,
    share: bool,
    layer: usize,
    positions: usize,
    shape: AttnShape,
    heads: std::ops::Range<usize>,
    out: &mut [f32],
    workers: usize,
) -> Result<(), ViperaError> {
    let count = heads.end - heads.start;
    if count == 0 || positions == 0 {
        out[..count * shape.head_dim].fill(0.0);
        return Ok(());
    }
    let workers = workers.min(positions).max(1);
    if workers <= 1 {
        let mut scratch = AttnScratch::default();
        return attention_streaming_heads(
            q,
            kv,
            order,
            share,
            layer,
            positions,
            shape,
            heads,
            &mut scratch.scores,
            &mut scratch.row,
            &mut scratch.head_state,
            out,
        );
    }

    let head_dim = shape.head_dim;
    let heads_total = shape.heads;
    let kv_heads = shape.kv_heads;
    let group = heads_total / kv_heads;
    let width = kv_heads * head_dim;
    let layer_bytes = kv.layout.bytes_per_row(width);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sequential = quant::exact_dot_requested();
    let in_place = in_place_kv_enabled()
        && crate::simd::IN_PLACE_KV_BIT_EXACT
        && kv.layout == KvLayout::F16
        && !sequential;
    let shared = share && in_place && group <= MAX_SHARED_GROUP;

    #[derive(Clone)]
    struct Partial {
        m: Vec<f32>,
        d: Vec<f32>,
        o: Vec<f32>,
    }

    let mut partials: Vec<Partial> = (0..workers)
        .map(|_| Partial {
            m: vec![f32::NEG_INFINITY; count],
            d: vec![0.0_f32; count],
            o: vec![0.0_f32; count * head_dim],
        })
        .collect();

    // One traversal per chunk: the K row yields every head's score, then the V row is folded into a
    // running (max, sum, weighted-v) with an online-softmax update. Reading K and V once instead of
    // in two passes also removes the per-worker `heads x positions` score buffer.
    let compute_chunk = |part: &mut Partial, start: usize, end: usize| -> Result<(), ViperaError> {
        if start >= end {
            return Ok(());
        }
        if shared {
            let mut key_bases = [0_usize; crate::simd::GROUP_ROWS];
            let mut value_bases = [0_usize; crate::simd::GROUP_ROWS];
            let mut block_scores = vec![0.0_f32; count * crate::simd::GROUP_ROWS];
            let mut pending = 0;
            kv.for_each_row_range(order, layer, layer_bytes, start, end, |_position, base| {
                key_bases[pending] = base;
                value_bases[pending] = kv.value_row(order, base, layer_bytes);
                pending += 1;
                if pending == crate::simd::GROUP_ROWS {
                    fold_chunk_block(
                        kv.storage,
                        &key_bases,
                        &value_bases,
                        q,
                        shape,
                        &heads,
                        scale,
                        &mut block_scores,
                        &mut part.m,
                        &mut part.d,
                        &mut part.o,
                    );
                    pending = 0;
                }
                Ok(())
            })?;
            if pending > 0 {
                fold_chunk_block(
                    kv.storage,
                    &key_bases[..pending],
                    &value_bases[..pending],
                    q,
                    shape,
                    &heads,
                    scale,
                    &mut block_scores,
                    &mut part.m,
                    &mut part.d,
                    &mut part.o,
                );
            }
            return Ok(());
        }
        let mut scores = vec![0.0_f32; count];
        let mut row = vec![0.0_f32; width];

        kv.for_each_row_range(order, layer, layer_bytes, start, end, |_position, base| {
            if !in_place {
                kv.layout
                    .decode(&kv.storage[base..base + layer_bytes], &mut row)?;
            }
            for kv_head in 0..kv_heads {
                let packed_k = base + kv_head * head_dim * 2;
                let key = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                for member in 0..group {
                    let head = kv_head * group + member;
                    if head < heads.start || head >= heads.end {
                        continue;
                    }
                    let local = head - heads.start;
                    let query = &q[head * head_dim..(head + 1) * head_dim];
                    let dot = if sequential {
                        query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>()
                    } else if in_place {
                        crate::simd::dot_f16_f32(query, &kv.storage[packed_k..][..head_dim * 2])
                    } else {
                        crate::simd::dot_f32(query, key)
                    };
                    scores[local] = dot * scale;
                }
            }

            let v_base = kv.value_row(order, base, layer_bytes);
            if !in_place {
                kv.layout
                    .decode(&kv.storage[v_base..v_base + layer_bytes], &mut row)?;
            }
            for kv_head in 0..kv_heads {
                let packed_v = v_base + kv_head * head_dim * 2;
                let value = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                for member in 0..group {
                    let head = kv_head * group + member;
                    if head < heads.start || head >= heads.end {
                        continue;
                    }
                    let local = head - heads.start;
                    let score = scores[local];
                    let target = &mut part.o[local * head_dim..(local + 1) * head_dim];
                    let weight = if score > part.m[local] {
                        // A new maximum rescales the running sum and the accumulator by
                        // `exp(old_max - score)`; the new row then enters with weight one.
                        let alpha = (part.m[local] - score).exp();
                        part.m[local] = score;
                        part.d[local] = part.d[local] * alpha + 1.0;
                        for value in target.iter_mut() {
                            *value *= alpha;
                        }
                        1.0_f32
                    } else {
                        let beta = (score - part.m[local]).exp();
                        part.d[local] += beta;
                        beta
                    };
                    if in_place {
                        crate::simd::axpy_f16_f32(
                            target,
                            weight,
                            &kv.storage[packed_v..][..head_dim * 2],
                        );
                    } else {
                        crate::simd::axpy_f32(target, weight, value);
                    }
                }
            }
            Ok(())
        })
    };

    // The caller takes the last chunk; spawn one worker fewer like the head split.
    let spawned = workers - 1;
    let caller_range = {
        let start = positions * spawned / workers;
        start..(positions * (spawned + 1) / workers)
    };
    let (spawn_parts, tail) = partials.split_at_mut(spawned);
    let caller_part = &mut tail[0];
    let chunk_error: std::sync::Mutex<Option<ViperaError>> = std::sync::Mutex::new(None);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(spawned);
        for (worker, part) in spawn_parts.iter_mut().enumerate() {
            let start = positions * worker / workers;
            let end = positions * (worker + 1) / workers;
            let err = &chunk_error;
            let f = &compute_chunk;
            handles.push(scope.spawn(move || {
                if let Err(e) = f(part, start, end) {
                    *err.lock().expect("chunk error") = Some(e);
                }
            }));
        }

        if let Err(e) = compute_chunk(caller_part, caller_range.start, caller_range.end) {
            *chunk_error.lock().expect("chunk error") = Some(e);
        }
        for handle in handles {
            handle.join().expect("an attention chunk worker panicked");
        }
    });

    if let Some(err) = chunk_error.into_inner().expect("chunk error") {
        return Err(err);
    }

    // Merge partials in increasing chunk order.
    out[..count * head_dim].fill(0.0);
    let mut global_m = vec![f32::NEG_INFINITY; count];
    let mut global_d = vec![0.0_f32; count];
    for part in &partials {
        for local in 0..count {
            let m_new = global_m[local].max(part.m[local]);
            let scale_a = if global_m[local] == f32::NEG_INFINITY {
                0.0
            } else {
                (global_m[local] - m_new).exp()
            };
            let scale_b = if part.m[local] == f32::NEG_INFINITY {
                0.0
            } else {
                (part.m[local] - m_new).exp()
            };
            global_d[local] = global_d[local] * scale_a + part.d[local] * scale_b;
            let dst = &mut out[local * head_dim..(local + 1) * head_dim];
            let src = &part.o[local * head_dim..(local + 1) * head_dim];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = *d * scale_a + *s * scale_b;
            }
            global_m[local] = m_new;
        }
    }
    for local in 0..count {
        let inv = 1.0 / global_d[local];
        for value in &mut out[local * head_dim..(local + 1) * head_dim] {
            *value *= inv;
        }
    }
    Ok(())
}

/// Decode attention over a sliding window, split by kept-position chunks.
///
/// [`attention_window_heads`] partitions the work by head, which makes every worker walk every
/// kept row and decode it again: the four query heads that share a KV head read the same K row, so
/// the cache is read four times over. It also materialises the whole `heads x kept` score block and
/// walks the rows twice, once for K and once for V. At a 2048-position window that cost 13.4 ms of
/// a 44 ms step (see `docs/benchmarks.md`, 2026-11-18), against 3.9 ms per 16k positions for the
/// chunked full-history kernel.
///
/// This kernel partitions the kept positions instead, so each row is read once and feeds every
/// head, and it folds K and V into a running `(max, sum, weighted_v)` in one traversal. The merge
/// groups the floating-point sums by chunk rather than strictly by position, so the result differs
/// from the two-pass kernel by rounding (~1e-7); `exact_dot` mode therefore does not use it.
///
/// With `share` and `f16` rows read in place, each worker folds its chunk sixteen kept rows at a
/// time ([`fold_chunk_block`]), the query heads of one key head sharing the conversion of every K
/// and V row — the same online softmax as [`attention_kv_chunked_heads`], regrouped, so it is within
/// rounding of the per-position update and not bit-identical to it.
#[allow(clippy::too_many_arguments)]
fn attention_window_chunked_heads(
    q: &[f32],
    kv: &KvRead<'_>,
    order: RowOrder,
    share: bool,
    layer: usize,
    ranges: &[(usize, usize)],
    shape: AttnShape,
    heads: std::ops::Range<usize>,
    out: &mut [f32],
    workers: usize,
) -> Result<(), ViperaError> {
    let head_dim = shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let width = shape.kv_heads * head_dim;
    let layer_bytes = kv.layout.bytes_per_row(width);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sequential = quant::exact_dot_requested();
    let in_place = in_place_kv_enabled()
        && crate::simd::IN_PLACE_KV_BIT_EXACT
        && kv.layout == KvLayout::F16
        && !sequential;
    let shared = share && in_place && group <= MAX_SHARED_GROUP;
    let count = heads.end - heads.start;
    let kept: usize = ranges
        .iter()
        .map(|(from, to)| to.saturating_sub(*from))
        .sum();
    if count == 0 || kept == 0 {
        out[..count * head_dim].fill(0.0);
        return Ok(());
    }
    let workers = workers.min(kept).max(1);

    #[derive(Clone)]
    struct Partial {
        m: Vec<f32>,
        d: Vec<f32>,
        o: Vec<f32>,
    }

    let mut partials: Vec<Partial> = (0..workers)
        .map(|_| Partial {
            m: vec![f32::NEG_INFINITY; count],
            d: vec![0.0_f32; count],
            o: vec![0.0_f32; count * head_dim],
        })
        .collect();

    // Kept indices are numbered `[0, kept)` across the ranges in order, so a chunk is a slice of
    // that space and may cross the gap between the sinks and the window.
    let compute_chunk = |part: &mut Partial, start: usize, end: usize| -> Result<(), ViperaError> {
        if start >= end {
            return Ok(());
        }
        if shared {
            let mut key_bases = [0_usize; crate::simd::GROUP_ROWS];
            let mut value_bases = [0_usize; crate::simd::GROUP_ROWS];
            let mut block_scores = vec![0.0_f32; count * crate::simd::GROUP_ROWS];
            let mut pending = 0;
            // The chunk is a slice of the kept-index space, so it is walked per range with the
            // ranges' lengths as offsets, exactly as the per-head path below does.
            let mut offset = 0;
            for (from, to) in ranges {
                let len = to.saturating_sub(*from);
                let base = offset;
                offset += len;
                let low = start.max(base);
                let high = end.min(base + len);
                if low >= high {
                    continue;
                }
                let run_from = *from + (low - base);
                let run_to = *from + (high - base);
                kv.for_each_row_range(
                    order,
                    layer,
                    layer_bytes,
                    run_from,
                    run_to,
                    |_position, k_base| {
                        key_bases[pending] = k_base;
                        value_bases[pending] = kv.value_row(order, k_base, layer_bytes);
                        pending += 1;
                        if pending == crate::simd::GROUP_ROWS {
                            fold_chunk_block(
                                kv.storage,
                                &key_bases,
                                &value_bases,
                                q,
                                shape,
                                &heads,
                                scale,
                                &mut block_scores,
                                &mut part.m,
                                &mut part.d,
                                &mut part.o,
                            );
                            pending = 0;
                        }
                        Ok(())
                    },
                )?;
            }
            if pending > 0 {
                fold_chunk_block(
                    kv.storage,
                    &key_bases[..pending],
                    &value_bases[..pending],
                    q,
                    shape,
                    &heads,
                    scale,
                    &mut block_scores,
                    &mut part.m,
                    &mut part.d,
                    &mut part.o,
                );
            }
            return Ok(());
        }
        let mut scores = vec![0.0_f32; count];
        let mut row = vec![0.0_f32; width];

        // Walk the chunk as it lies in the ranges: contiguous where the chunk stays inside one
        // range, two runs where it steps over the dropped middle.
        let mut base = 0;
        for (from, to) in ranges {
            let len = to.saturating_sub(*from);
            let offset = base;
            base += len;
            let low = start.max(offset);
            let high = end.min(offset + len);
            if low >= high {
                continue;
            }
            let run_from = *from + (low - offset);
            let run_to = *from + (high - offset);
            kv.for_each_row_range(
                order,
                layer,
                layer_bytes,
                run_from,
                run_to,
                |_position, k_base| {
                    if !in_place {
                        kv.layout
                            .decode(&kv.storage[k_base..k_base + layer_bytes], &mut row)?;
                    }
                    // A decode step asks for every head, and the heads of one KV head share its
                    // K row: batching them pays the per-call overhead (an `OnceLock` load, two
                    // feature checks, a fresh loop setup) once per KV head instead of once per
                    // head, for the same 64-value dot and the same `* scale`.
                    let whole = heads.start == 0 && heads.end == shape.heads && group <= 16;
                    if whole && in_place && !sequential {
                        for kv_head in 0..shape.kv_heads {
                            let packed_k = k_base + kv_head * head_dim * 2;
                            let first = kv_head * group;
                            crate::simd::dot_f16_f32_heads(
                                q,
                                first,
                                group,
                                head_dim,
                                &kv.storage[packed_k..],
                                &mut scores[first..first + group],
                            );
                        }
                        for score in scores[..count].iter_mut() {
                            *score *= scale;
                        }
                    } else {
                        for kv_head in 0..shape.kv_heads {
                            let key = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                            let packed_k = k_base + kv_head * head_dim * 2;
                            for member in 0..group {
                                let head = kv_head * group + member;
                                if head < heads.start || head >= heads.end {
                                    continue;
                                }
                                let local = head - heads.start;
                                let query = &q[head * head_dim..(head + 1) * head_dim];
                                let dot = if sequential {
                                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>()
                                } else if in_place {
                                    crate::simd::dot_f16_f32(
                                        query,
                                        &kv.storage[packed_k..][..head_dim * 2],
                                    )
                                } else {
                                    crate::simd::dot_f32(query, key)
                                };
                                scores[local] = dot * scale;
                            }
                        }
                    }

                    let v_base = kv.value_row(order, k_base, layer_bytes);
                    if !in_place {
                        kv.layout
                            .decode(&kv.storage[v_base..v_base + layer_bytes], &mut row)?;
                    }
                    if whole && in_place {
                        // Same online-softmax update per head, with the V row added for the whole
                        // group in one call: a head that takes a new maximum has its row scaled
                        // first, so the shared call can use its own weight unchanged.
                        for kv_head in 0..shape.kv_heads {
                            let packed_v = v_base + kv_head * head_dim * 2;
                            let first = kv_head * group;
                            let mut weights = [0.0_f32; 16];
                            for (member, weight) in weights[..group].iter_mut().enumerate() {
                                let local = first + member;
                                let score = scores[local];
                                let target = &mut part.o[local * head_dim..(local + 1) * head_dim];
                                *weight = if score > part.m[local] {
                                    let alpha = (part.m[local] - score).exp();
                                    part.m[local] = score;
                                    part.d[local] = part.d[local] * alpha + 1.0;
                                    for value in target.iter_mut() {
                                        *value *= alpha;
                                    }
                                    1.0_f32
                                } else {
                                    let beta = (score - part.m[local]).exp();
                                    part.d[local] += beta;
                                    beta
                                };
                            }
                            crate::simd::axpy_f16_f32_heads(
                                &mut part.o[first * head_dim..(first + group) * head_dim],
                                head_dim,
                                &weights[..group],
                                &kv.storage[packed_v..],
                            );
                        }
                    } else {
                        for kv_head in 0..shape.kv_heads {
                            let packed_v = v_base + kv_head * head_dim * 2;
                            let value = &row[kv_head * head_dim..(kv_head + 1) * head_dim];
                            for member in 0..group {
                                let head = kv_head * group + member;
                                if head < heads.start || head >= heads.end {
                                    continue;
                                }
                                let local = head - heads.start;
                                let score = scores[local];
                                let target = &mut part.o[local * head_dim..(local + 1) * head_dim];
                                let weight = if score > part.m[local] {
                                    let alpha = (part.m[local] - score).exp();
                                    part.m[local] = score;
                                    part.d[local] = part.d[local] * alpha + 1.0;
                                    for value in target.iter_mut() {
                                        *value *= alpha;
                                    }
                                    1.0_f32
                                } else {
                                    let beta = (score - part.m[local]).exp();
                                    part.d[local] += beta;
                                    beta
                                };
                                if in_place {
                                    crate::simd::axpy_f16_f32(
                                        target,
                                        weight,
                                        &kv.storage[packed_v..][..head_dim * 2],
                                    );
                                } else {
                                    crate::simd::axpy_f32(target, weight, value);
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    };

    if workers == 1 {
        compute_chunk(&mut partials[0], 0, kept)?;
    } else {
        // The caller takes the last chunk, so one fewer thread is spawned per layer.
        let spawned = workers - 1;
        let caller_range = (kept * spawned / workers)..(kept * (spawned + 1) / workers);
        let (spawn_parts, tail) = partials.split_at_mut(spawned);
        let caller_part = &mut tail[0];
        let chunk_error: std::sync::Mutex<Option<ViperaError>> = std::sync::Mutex::new(None);
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(spawned);
            for (worker, part) in spawn_parts.iter_mut().enumerate() {
                let start = kept * worker / workers;
                let end = kept * (worker + 1) / workers;
                let err = &chunk_error;
                let f = &compute_chunk;
                handles.push(scope.spawn(move || {
                    if let Err(e) = f(part, start, end) {
                        *err.lock().expect("chunk error") = Some(e);
                    }
                }));
            }
            if let Err(e) = compute_chunk(caller_part, caller_range.start, caller_range.end) {
                *chunk_error.lock().expect("chunk error") = Some(e);
            }
            for handle in handles {
                handle.join().expect("an attention chunk worker panicked");
            }
        });
        if let Some(err) = chunk_error.into_inner().expect("chunk error") {
            return Err(err);
        }
    }

    // Merge in increasing chunk order, then normalise.
    out[..count * head_dim].fill(0.0);
    let mut global_m = vec![f32::NEG_INFINITY; count];
    let mut global_d = vec![0.0_f32; count];
    for part in &partials {
        for local in 0..count {
            let m_new = global_m[local].max(part.m[local]);
            let scale_a = if global_m[local] == f32::NEG_INFINITY {
                0.0
            } else {
                (global_m[local] - m_new).exp()
            };
            let scale_b = if part.m[local] == f32::NEG_INFINITY {
                0.0
            } else {
                (part.m[local] - m_new).exp()
            };
            global_d[local] = global_d[local] * scale_a + part.d[local] * scale_b;
            let dst = &mut out[local * head_dim..(local + 1) * head_dim];
            let src = &part.o[local * head_dim..(local + 1) * head_dim];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = *d * scale_a + *s * scale_b;
            }
            global_m[local] = m_new;
        }
    }
    for local in 0..count {
        let inv = 1.0 / global_d[local];
        for value in &mut out[local * head_dim..(local + 1) * head_dim] {
            *value *= inv;
        }
    }
    Ok(())
}

/// Attention for one layer, choosing the kernel by history length.
///
/// Both arms compute the same values, bit for bit: they see the same K/V rows in the same order.
/// Only the traffic differs. [`STREAMING_MIN_POSITIONS`] records why the choice is by length and
/// what the two measured ends are.
fn attend(
    q: &[f32],
    kv: &KvView<'_>,
    layer: usize,
    positions: usize,
    shape: AttnShape,
    scratch: &mut AttnScratch,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let read = KvRead::of(kv);
    attend_entries(
        q,
        std::slice::from_ref(&read),
        std::slice::from_ref(&positions),
        layer,
        shape,
        scratch,
        out,
    )
}

/// Queries one block of a prompt chunk shares a K/V row with, in [`attend_rows_shared`].
///
/// Four keeps the per-block state (four sets of `heads x head_dim` accumulators, 24 KiB at twelve
/// heads of 128) inside L1 while still dividing the row traffic by four. Eight doubles the win and
/// spills the accumulators to L2, where the V pass pays for them again.
const QBLOCK: usize = 4;

/// Whether a chunk's attention shares each K/V row across a block of queries.
///
/// **Off unless `VIPERA_ATTN_QBLOCK` is set to `1`, `true`, `on` or `yes`.** Measured at 4k on the
/// i5-11300H, `VIPERA_ATTN_QBLOCK=1` takes attention from 435-445 ms to 399-411 ms of a chunk, 7.7%,
/// and a chunk from 1280 ms to 1261-1273 ms, which is inside this ledger's own 8-15% window spread:
/// the sharing removes the row traffic and leaves the arithmetic, and attention is already bound by
/// that (see the entry of 2026-11-20 in `docs/benchmarks.md`). Off by default because it regroups
/// the softmax sums - within rounding (~1e-7) of the per-entry kernel, not bit-identical, the policy
/// the position-chunked kernel and `VIPERA_ATTN_GQA_SHARE` follow - and does not pay for itself end
/// to end yet.
fn attention_qblock_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VIPERA_ATTN_QBLOCK").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    })
}

/// Whether every entry of a chunk reads the same KV blocks, which is what lets one traversal serve
/// all of them.
fn one_kv_for_all(kv: &[KvRead<'_>]) -> bool {
    kv.iter().all(|view| {
        std::ptr::eq(view.storage, kv[0].storage)
            && std::ptr::eq(view.table, kv[0].table)
            && view.tokens_per_block == kv[0].tokens_per_block
            && view.bytes_per_token == kv[0].bytes_per_token
            && view.layout == kv[0].layout
    })
}

/// A block of a prompt chunk's attention, folded in one traversal of the history.
///
/// A chunk calls [`attend_entries`] once per layer with one query row per position, and every row
/// walks the whole history: at 3.3k positions that is 128 passes over the same K and V rows per
/// layer, 11 GB of reads for a 128-token chunk. The rows are the same rows for every query in the
/// block and the positions are consecutive, so one walk feeds [`QBLOCK`] queries: each K row is
/// scored against the block and each V row is folded into the block's accumulators before the next
/// row replaces it in L1.
///
/// The queries of a block have different history lengths. Entry `slot` sees positions below
/// `positions[first + slot]` only, so the rows past that are a prefix of the block it never folds —
/// the same causal answer with the same values, but folded a block of [`GROUP_ROWS`] positions at a
/// time ([`fold_chunk_block`], the kernel the position-chunked path already uses) rather than
/// scored, then normalised row by row. That regrouping is within rounding (~1e-7), not
/// bit-identical.
///
/// The first version of this kernel folded one position at a time and was **twice as slow** as the
/// per-entry path (677 ms against 325 ms of a 3k chunk, measured 2026-11-20): it saved three
/// quarters of the K/V reads and paid for them by touching every block accumulator once per
/// position instead of once per sixteen, which the V fold dominates. The row blocking is what makes
/// the sharing pay.
fn attend_rows_shared(
    q: &[f32],
    kv: &[KvRead<'_>],
    positions: &[usize],
    layer: usize,
    shape: AttnShape,
    out: &mut [f32],
    workers: usize,
) -> Result<(), ViperaError> {
    let entries = positions.len();
    let heads = shape.heads;
    let head_dim = shape.head_dim;
    let kv_heads = shape.kv_heads;
    let query = heads * head_dim;
    let width = kv_heads * head_dim;
    let layer_bytes = kv[0].layout.bytes_per_row(width);
    let order = AttentionSwitches::from_env().order;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let blocks = entries.div_ceil(QBLOCK);
    let workers = workers.min(blocks).max(1);

    let block_rows = crate::simd::GROUP_ROWS;
    let run = |first: usize, end: usize, slot_out: &mut [f32]| -> Result<(), ViperaError> {
        let rows = end - first;
        let mut m = vec![f32::NEG_INFINITY; rows * heads];
        let mut d = vec![0.0_f32; rows * heads];
        let mut o = vec![0.0_f32; rows * query];
        let mut block_scores = vec![0.0_f32; heads * block_rows];
        let mut key_bases = [0_usize; crate::simd::GROUP_ROWS];
        let mut value_bases = [0_usize; crate::simd::GROUP_ROWS];
        let history = positions[first..end].iter().copied().max().unwrap_or(0);
        let storage = kv[0].storage;

        // One walk of the history per block of positions, feeding every query of the block: the
        // K and V rows are decoded and read once, and each query folds the same sixteen rows into
        // its own running (max, sum, weighted V).
        let mut start = 0;
        while start < history {
            let stop = (start + block_rows).min(history);
            let mut pending = 0;
            kv[0].for_each_row_range(
                order,
                layer,
                layer_bytes,
                start,
                stop,
                |_position, base| {
                    key_bases[pending] = base;
                    value_bases[pending] = kv[0].value_row(order, base, layer_bytes);
                    pending += 1;
                    Ok(())
                },
            )?;
            for slot in 0..rows {
                // The rows at or past this query's own position are its future: the block is a
                // prefix of what it may see, so the allowed rows are a slice of the gathered ones.
                let seen = positions[first + slot];
                let allowed = (0..pending).take_while(|row| start + row < seen).count();
                if allowed == 0 {
                    continue;
                }
                let heads_slice = slot * heads..(slot + 1) * heads;
                fold_chunk_block(
                    storage,
                    &key_bases[..allowed],
                    &value_bases[..allowed],
                    &q[(first + slot) * query..][..query],
                    shape,
                    &(0..heads),
                    scale,
                    &mut block_scores,
                    &mut m[heads_slice.clone()],
                    &mut d[heads_slice.clone()],
                    &mut o[slot * query..][..query],
                );
            }
            start = stop;
        }

        for slot in 0..rows {
            for head in 0..heads {
                let at = slot * heads + head;
                let inverse = 1.0 / d[at];
                let dst = &mut slot_out[slot * query + head * head_dim..][..head_dim];
                for (out, value) in dst.iter_mut().zip(&o[at * head_dim..(at + 1) * head_dim]) {
                    *out = *value * inverse;
                }
            }
        }
        Ok(())
    };

    if workers == 1 {
        return run(0, entries, out);
    }

    let mut jobs: Vec<(usize, usize, &mut [f32])> = Vec::with_capacity(workers);
    let mut cursor: &mut [f32] = out;
    for part in 0..workers {
        let first = blocks * part / workers;
        let end = blocks * (part + 1) / workers;
        let row_start = first * QBLOCK;
        let row_end = (end * QBLOCK).min(entries);
        let (slot, rest) = cursor.split_at_mut((row_end - row_start) * query);
        cursor = rest;
        jobs.push((row_start, row_end, slot));
    }

    let results: Vec<Result<(), ViperaError>> = thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .into_iter()
            .map(|(first, end, slot)| scope.spawn(move || run(first, end, slot)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a qblock attention worker panicked"))
            .collect()
    });
    results.into_iter().collect()
}

/// A chunk's worth of attention: one output row per entry, each with its own history length.
///
/// The rows are independent, and so are the heads inside one row, so the work goes to
/// `min(VIPERA_THREADS, rows or heads)` workers. Each worker writes a disjoint output slice and
/// keeps its own scratch, and every row does the same arithmetic in the same order as a lone
/// [`attend`] call, so a chunk's logits still match a token-by-token run bit for bit.
///
/// Until this, attention was the one part of a prompt chunk that ran on a single thread while the
/// matvecs used the pool: at a 512-token prompt it is 11-12% of a chunk's time.
fn attend_entries(
    q: &[f32],
    kv: &[KvRead<'_>],
    positions: &[usize],
    layer: usize,
    shape: AttnShape,
    scratch: &mut AttnScratch,
    out: &mut [f32],
) -> Result<(), ViperaError> {
    let entries = positions.len();
    let query = shape.heads * shape.head_dim;
    if entries == 0 {
        return Ok(());
    }
    if cfg!(debug_assertions) {
        assert_eq!(kv.len(), entries, "one KV view per entry");
        assert_eq!(out.len(), entries * query, "one output row per entry");
    }
    let switches = AttentionSwitches::from_env();
    let last = positions.iter().copied().max().unwrap_or(0);
    let lanes = entries.saturating_mul(shape.heads).saturating_mul(last);

    // A prompt chunk asks the same history once per token, with consecutive positions and one KV
    // view between them. One walk of the history over a block of queries reads every K/V row
    // `QBLOCK` times fewer; the rows are then the traffic of the pass rather than its arithmetic.
    if entries > 1
        && kv[0].layout == KvLayout::F16
        && one_kv_for_all(kv)
        && !quant::exact_dot_requested()
        && attention_window().is_none()
        && attention_qblock_enabled()
    {
        let workers = quant::max_threads().min(entries.div_ceil(QBLOCK));
        if workers > 1 {
            return attend_rows_shared(q, kv, positions, layer, shape, out, workers);
        }
    }
    // Only the entries are split. Splitting a single entry's heads was measured too: at ctx 1024 a
    // decode step went from 45.3 to 48.7 ms, because every worker decodes the same KV rows again and
    // the workers cost more than the twelve heads they divide.
    let wanted = if lanes < PARALLEL_ATTENTION_MIN {
        1
    } else {
        quant::max_threads().min(entries)
    };

    // A window keeps only a slice of the history, and that slice is best split by position: the
    // head split below walks every kept row once per head, so the four query heads that share a KV
    // head read the same K row four times over. Splitting the kept positions reads each row once
    // and folds K and V in a single traversal, and it is what makes the window cost a third of
    // what the two-pass kernel did. `exact_dot` keeps the two-pass kernel, whose summation order is
    // the reference.
    if entries == 1
        && !quant::exact_dot_requested()
        && let Some((window, sinks)) = attention_window()
        && positions[0] > window.saturating_add(sinks)
    {
        let sinks_end = sinks.min(positions[0]);
        let window_start = positions[0].saturating_sub(window).max(sinks_end);
        let kept = sinks_end + (positions[0] - window_start);
        let threads = quant::max_threads().min(physical_cores());
        let workers = threads.min(kept);
        if workers > 1 {
            let ranges = [(0, sinks_end), (window_start, positions[0])];
            return attention_window_chunked_heads(
                &q[..query],
                &kv[0],
                switches.order,
                gqa_share_enabled(),
                layer,
                &ranges,
                shape,
                0..shape.heads,
                &mut out[..query],
                workers,
            );
        }
    }

    // One entry with a long history has nothing to split by entry. Splitting the KV history into
    // chunks and merging partial (max, sum, weighted-v) results parallelizes the memory-bound
    // scan well at depth, at the cost of floating-point merge grouping (~1e-7 max-abs vs serial),
    // so it is opt-in (`VIPERA_ATTN_KV_SPLIT=1`) and skipped in exact-dot mode. A window needs
    // none of it: the chunked window kernel above already split the kept positions.
    if entries == 1
        && lanes >= PARALLEL_ATTENTION_MIN
        && switches.kv_split
        && attention_window().is_none()
        && !quant::exact_dot_requested()
    {
        let threads = quant::max_threads().min(physical_cores());
        let workers =
            attention_workers(threads, threads.max(1), switches.workers).min(positions[0].max(1));
        if workers > 1 {
            return attention_kv_chunked_heads(
                &q[..query],
                &kv[0],
                switches.order,
                gqa_share_enabled(),
                layer,
                positions[0],
                shape,
                0..shape.heads,
                &mut out[..query],
                workers,
            );
        }
    }

    // One entry with a long history has nothing to split by entry, but its heads are independent
    // through the whole kernel, so they go to the pool instead. This was measured before with the
    // row buffer in place and lost (each worker decoded every row again); with the rows read in
    // place a worker repeats only the arithmetic, which is what the split is for.
    if entries == 1 && lanes >= PARALLEL_ATTENTION_MIN && switches.split_heads {
        let threads = quant::max_threads().min(physical_cores());
        let workers = attention_workers(threads, shape.heads, switches.workers);
        if workers > 1 {
            // The caller takes the last part itself, so the split still puts one thread on each
            // core but creates and joins one thread fewer per layer. The head/slice bookkeeping stays
            // on the stack: decode runs this once per layer and the old Vecs showed up in profiles as
            // avoidable churn.
            let spawned = workers - 1;
            let caller_heads = head_range(shape.heads, workers, spawned);
            if spawned <= 16 {
                let mut rest: &mut [f32] = out;
                let result = thread::scope(|scope| {
                    let mut handles: [Option<
                        std::thread::ScopedJoinHandle<'_, Result<(), ViperaError>>,
                    >; 16] = [const { None }; 16];
                    for (part, slot_handle) in handles.iter_mut().enumerate().take(spawned) {
                        let heads = head_range(shape.heads, workers, part);
                        let len = (heads.end - heads.start) * shape.head_dim;
                        let (slot, tail) = rest.split_at_mut(len);
                        rest = tail;
                        *slot_handle = Some(scope.spawn(move || {
                            let mut local = AttnScratch::default();
                            attend_entry(
                                AttentionCall {
                                    q: &q[..query],
                                    kv: &kv[0],
                                    layer,
                                    positions: positions[0],
                                    shape,
                                    heads: heads.start..heads.end,
                                    out: slot,
                                },
                                &mut local,
                            )
                        }));
                    }
                    let caller = attend_entry(
                        AttentionCall {
                            q: &q[..query],
                            kv: &kv[0],
                            layer,
                            positions: positions[0],
                            shape,
                            heads: caller_heads,
                            out: rest,
                        },
                        scratch,
                    );
                    for slot_handle in handles.iter_mut().take(spawned) {
                        slot_handle
                            .take()
                            .expect("a spawned attention worker")
                            .join()
                            .expect("an attention worker panicked")?;
                    }
                    caller
                });
                return result;
            }

            let mut slots: Vec<&mut [f32]> = Vec::with_capacity(workers);
            let mut cursor: &mut [f32] = out;
            for part in 0..workers {
                let heads = head_range(shape.heads, workers, part);
                let (slot, rest) = cursor.split_at_mut((heads.end - heads.start) * shape.head_dim);
                slots.push(slot);
                cursor = rest;
            }
            let caller_slot = slots.pop().expect("one slot per worker");
            let results: Vec<Result<(), ViperaError>> = thread::scope(|scope| {
                let mut handles = Vec::with_capacity(spawned);
                for (part, slot) in slots.into_iter().enumerate() {
                    let heads = head_range(shape.heads, workers, part);
                    let (first, end) = (heads.start, heads.end);
                    handles.push(scope.spawn(move || {
                        let mut local = AttnScratch::default();
                        attend_entry(
                            AttentionCall {
                                q: &q[..query],
                                kv: &kv[0],
                                layer,
                                positions: positions[0],
                                shape,
                                heads: first..end,
                                out: slot,
                            },
                            &mut local,
                        )
                    }));
                }
                let caller = attend_entry(
                    AttentionCall {
                        q: &q[..query],
                        kv: &kv[0],
                        layer,
                        positions: positions[0],
                        shape,
                        heads: caller_heads,
                        out: caller_slot,
                    },
                    scratch,
                );
                let mut results: Vec<Result<(), ViperaError>> = handles
                    .into_iter()
                    .map(|handle| handle.join().expect("an attention worker panicked"))
                    .collect();
                results.push(caller);
                results
            });
            return results.into_iter().collect();
        }
    }

    if wanted <= 1 {
        for entry in 0..entries {
            attend_entry(
                AttentionCall {
                    q: &q[entry * query..][..query],
                    kv: &kv[entry],
                    layer,
                    positions: positions[entry],
                    shape,
                    heads: 0..shape.heads,
                    out: &mut out[entry * query..][..query],
                },
                scratch,
            )?;
        }
        return Ok(());
    }

    // One worker per contiguous run of entries: the output rows are already in that order.
    if wanted <= 16 {
        let mut rest: &mut [f32] = out;
        return thread::scope(|scope| {
            let mut handles: [Option<std::thread::ScopedJoinHandle<'_, Result<(), ViperaError>>>;
                16] = [const { None }; 16];
            for (part, slot_handle) in handles.iter_mut().enumerate().take(wanted) {
                let first = entries * part / wanted;
                let end = entries * (part + 1) / wanted;
                let (slot, tail) = rest.split_at_mut((end - first) * query);
                rest = tail;
                *slot_handle = Some(scope.spawn(move || -> Result<(), ViperaError> {
                    let mut local = AttnScratch::default();
                    for (offset, row) in slot.chunks_mut(query).enumerate() {
                        let entry = first + offset;
                        attend_entry(
                            AttentionCall {
                                q: &q[entry * query..][..query],
                                kv: &kv[entry],
                                layer,
                                positions: positions[entry],
                                shape,
                                heads: 0..shape.heads,
                                out: row,
                            },
                            &mut local,
                        )?;
                    }
                    Ok(())
                }));
            }
            for slot_handle in handles.iter_mut().take(wanted) {
                slot_handle
                    .take()
                    .expect("a spawned attention worker")
                    .join()
                    .expect("an attention worker panicked")?;
            }
            Ok(())
        });
    }

    let mut jobs = Vec::with_capacity(wanted);
    let mut cursor: &mut [f32] = out;
    for part in 0..wanted {
        let first = entries * part / wanted;
        let end = entries * (part + 1) / wanted;
        let (slot, rest) = cursor.split_at_mut((end - first) * query);
        cursor = rest;
        jobs.push((first, slot));
    }
    let results: Vec<Result<(), ViperaError>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(jobs.len());
        for (first, slot) in jobs {
            handles.push(scope.spawn(move || -> Result<(), ViperaError> {
                let mut local = AttnScratch::default();
                for (offset, row) in slot.chunks_mut(query).enumerate() {
                    let entry = first + offset;
                    attend_entry(
                        AttentionCall {
                            q: &q[entry * query..][..query],
                            kv: &kv[entry],
                            layer,
                            positions: positions[entry],
                            shape,
                            heads: 0..shape.heads,
                            out: row,
                        },
                        &mut local,
                    )?;
                }
                Ok(())
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("an attention worker panicked"))
            .collect()
    });
    results.into_iter().collect()
}

/// One entry's attention, over `heads` only, choosing the kernel by history length.
#[allow(clippy::too_many_arguments)]
/// One attention kernel, and everything the forward pass asks of it.
///
/// There are two, and [`algorithm_for`] picks between them by history length. They sit behind a
/// trait so the choice is one function rather than a branch in the forward pass, and so another
/// kernel — a block-parallel one, or a device one — is an implementation rather than an edit here.
/// Both must compute the oracle's bits: the tests compare them to each other and to [`attention`].
/// One attention call: the query rows, the history to read, and where the answer goes.
struct AttentionCall<'q, 'k> {
    /// The query heads, `shape.heads * head_dim` values.
    q: &'q [f32],
    /// The KV cache to read.
    kv: &'q KvRead<'k>,
    /// Layer of the cache this pass reads.
    layer: usize,
    /// Positions of history, including the one `q` is.
    positions: usize,
    /// Head geometry.
    shape: AttnShape,
    /// Heads to compute, in absolute head numbers.
    heads: std::ops::Range<usize>,
    /// Where `heads.len()` rows of `head_dim` values are written, in head order from `heads.start`.
    out: &'q mut [f32],
}

trait AttentionAlgorithm {
    /// Fill `call.out` from `call`.
    fn attend(
        &self,
        call: AttentionCall<'_, '_>,
        scratch: &mut AttnScratch,
    ) -> Result<(), ViperaError>;

    /// Which kernel this is, for the test that the selection follows [`STREAMING_MIN_POSITIONS`].
    #[cfg(test)]
    fn name(&self) -> &'static str;
}

/// Decode the layer once and score the whole window from the decoded buffers.
///
/// Cheapest below [`STREAMING_MIN_POSITIONS`], where the two `positions * width` buffers are still
/// small enough that decoding each row once beats decoding it twice.
struct BufferedAttention;

/// Read each K/V row out of the cache where it lies, keeping one row of scratch.
struct StreamingAttention;

impl AttentionAlgorithm for BufferedAttention {
    fn attend(
        &self,
        call: AttentionCall<'_, '_>,
        scratch: &mut AttnScratch,
    ) -> Result<(), ViperaError> {
        call.kv.load_layer(
            call.layer,
            call.shape,
            call.positions,
            &mut scratch.k_all,
            &mut scratch.v_all,
        )?;
        attention_heads(
            call.q,
            &scratch.k_all,
            &scratch.v_all,
            call.positions,
            call.shape,
            call.heads,
            &mut scratch.scores,
            call.out,
        );
        Ok(())
    }

    #[cfg(test)]
    fn name(&self) -> &'static str {
        "buffered"
    }
}

impl AttentionAlgorithm for StreamingAttention {
    fn attend(
        &self,
        call: AttentionCall<'_, '_>,
        scratch: &mut AttnScratch,
    ) -> Result<(), ViperaError> {
        attention_streaming_heads(
            call.q,
            call.kv,
            AttentionSwitches::from_env().order,
            gqa_share_enabled(),
            call.layer,
            call.positions,
            call.shape,
            call.heads,
            &mut scratch.scores,
            &mut scratch.row,
            &mut scratch.head_state,
            call.out,
        )
    }

    #[cfg(test)]
    fn name(&self) -> &'static str {
        "streaming"
    }
}

/// The kernel for a history of `positions`: the only place the choice is made.
fn algorithm_for(positions: usize) -> &'static dyn AttentionAlgorithm {
    if positions >= STREAMING_MIN_POSITIONS {
        &StreamingAttention
    } else {
        &BufferedAttention
    }
}

fn attend_entry(call: AttentionCall<'_, '_>, scratch: &mut AttnScratch) -> Result<(), ViperaError> {
    // A window only changes the answer once the history is longer than what it keeps.
    if let Some((window, sinks)) = attention_window()
        && call.positions > window.saturating_add(sinks)
    {
        return attention_window_heads(
            call.q,
            call.kv,
            AttentionSwitches::from_env().order,
            gqa_share_enabled(),
            call.layer,
            call.positions,
            sinks,
            window,
            call.shape,
            call.heads,
            scratch,
            call.out,
        );
    }
    algorithm_for(call.positions).attend(call, scratch)
}

/// The rotary angles of one position: `sin_cos(position * inv_freq)` for every pair index.
///
/// An angle depends on the position and the pair index, not on the layer or the tensor that is
/// being rotated, so a prompt chunk builds one table per position and every layer turns its Q and
/// K from it. Building it inside each call instead repeats the same 64 transcendental values 56
/// times per position — once per Q and once per K, in each of the 28 layers — for 10.3 ms of a
/// 128-token chunk, against 3.8 ms for the rotation itself (see `norm_phase_cost`).
#[derive(Clone, Copy)]
struct RopeAngles {
    position: usize,
    head_dim: usize,
    base: f32,
    /// `sin_cos(position * inv_freq[index])`, filled when `head_dim / 2 <= 64`.
    pairs: [(f32, f32); 64],
    /// Whether `pairs` holds every index, i.e. `head_dim / 2 <= 64`.
    tabled: bool,
}

impl RopeAngles {
    fn new(position: usize, head_dim: usize, base: f32) -> Self {
        let half = head_dim / 2;
        let mut angles = Self {
            position,
            head_dim,
            base,
            pairs: [(0.0, 0.0); 64],
            tabled: half <= 64,
        };
        if angles.tabled {
            for (index, slot) in angles.pairs.iter_mut().enumerate().take(half) {
                let exponent = 2.0 * index as f32 / head_dim as f32;
                let inv = 1.0 / base.powf(exponent);
                *slot = (position as f32 * inv).sin_cos();
            }
        }
        angles
    }

    /// The `(sin, cos)` of pair `index`, computed on the fly for a width the table does not cover.
    fn at(&self, index: usize) -> (f32, f32) {
        if self.tabled {
            return self.pairs[index];
        }
        let exponent = 2.0 * index as f32 / self.head_dim as f32;
        let inv = 1.0 / self.base.powf(exponent);
        (self.position as f32 * inv).sin_cos()
    }
}

/// Rotate every head of `vector` by one position's angles, NeoX layout: pairs
/// `(i, i + head_dim / 2)`.
fn rope_with(vector: &mut [f32], angles: &RopeAngles, heads: usize, head_dim: usize) {
    let half = head_dim / 2;
    if angles.tabled {
        // The two halves are rotated together but live apart, so each is reached through its own
        // slice: a pair at a time through one slice would carry a bounds check and a shifted index
        // per element, and neither loop below vectorises.
        let pairs = &angles.pairs[..half];
        for head in vector.chunks_exact_mut(head_dim).take(heads) {
            let (first, second) = head.split_at_mut(half);
            for (index, (sin, cos)) in pairs.iter().enumerate() {
                let (low, high) = (first[index], second[index]);
                first[index] = low * cos - high * sin;
                second[index] = low * sin + high * cos;
            }
        }
        return;
    }
    for head in vector.chunks_exact_mut(head_dim).take(heads) {
        for index in 0..half {
            let (sin, cos) = angles.at(index);
            let first = head[index];
            let second = head[index + half];
            head[index] = first * cos - second * sin;
            head[index + half] = first * sin + second * cos;
        }
    }
}

/// Rotary embedding, NeoX layout: rotate pairs `(i, i + head_dim / 2)`.
fn rope(vector: &mut [f32], position: usize, heads: usize, head_dim: usize, base: f32) {
    rope_with(
        vector,
        &RopeAngles::new(position, head_dim, base),
        heads,
        head_dim,
    )
}

/// Whether the chunk path folds the norms and the SwiGLU multiply into the activation preparation.
///
/// Off unless `VIPERA_PREP_FUSION` is set to something other than `0`, `false`, `off` or `no`.
/// A chunk spends 12 ms on the two norms and 30 ms on the SwiGLU multiply on the calling thread,
/// between products that leave every worker idle; folding them into the copy a [`PrepJob`] already
/// makes moves that work onto the workers. The result is bit-identical either way — the transforms
/// are elementwise and read nothing but their own inputs. Read once.
fn prep_fusion_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VIPERA_PREP_FUSION") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => false,
    })
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse = 1.0 / (mean + epsilon).sqrt();
    for ((value, weight), input) in out.iter_mut().zip(weight).zip(input) {
        *value = input * inverse * weight;
    }
}

/// [`rms_norm`] over a buffer that already holds the input, in place.
///
/// The arithmetic is `rms_norm`'s, element for element, so a caller that can hand a pool worker its
/// own buffer gets the same bits as the pass on the calling thread (`rms_norm_in_place_is_rms_norm`
/// checks that). It exists because a worker cannot borrow a separate output slice.
pub(crate) fn rms_norm_in_place(values: &mut [f32], weight: &[f32], epsilon: f32) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inverse = 1.0 / (mean + epsilon).sqrt();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value = *value * inverse * weight;
    }
}

/// [`rms_norm`] over each head's vector on its own, in place: qwen3's `attn_q_norm` and
/// `attn_k_norm`, applied to the projected queries and keys before RoPE.
///
/// The per-element arithmetic is `rms_norm`'s, so a head here is the same bits as the same values
/// pushed through the whole-vector kernel when there is one head.
fn per_head_rms_norm(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    heads: usize,
    head_dim: usize,
) {
    for head in 0..heads {
        let Some(slice) = values.get_mut(head * head_dim..(head + 1) * head_dim) else {
            return;
        };
        let mean = slice.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let inverse = 1.0 / (mean + epsilon).sqrt();
        for (value, weight) in slice.iter_mut().zip(weight) {
            *value = *value * inverse * weight;
        }
    }
}

/// `e^value` without the libm call.
///
/// The feed-forward block evaluates 11.0M sigmoids per 128-token chunk, every one of them a
/// `f32::exp` call into libm at 4.5 ns. Splitting `e^x` into `2^(x log2 e)` turns the fractional
/// part into a degree-five polynomial and the integer part into an exponent-byte shift, both of
/// which stay in registers. The split is exact in the sense that matters here: the polynomial's
/// argument is the true rounded fraction, so the error is the polynomial's own, under 1e-7.
#[inline]
pub(crate) fn exp(value: f32) -> f32 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    // Round to nearest by adding and subtracting 1.5 * 2^23, which forces the
    // low mantissa bits to hold the integer part without a branch.
    let scaled = value * LOG2_E;
    let whole = (scaled + 12_582_912.0) - 12_582_912.0;
    let fraction = scaled - whole;
    let polynomial = 1.0
        + fraction
            * (std::f32::consts::LN_2
                + fraction
                    * (0.240_226_5
                        + fraction
                            * (0.055_504_11
                                + fraction
                                    * (0.009_618_129
                                        + fraction
                                            * (0.001_333_355_8 + fraction * 0.000_154_035_3)))));
    // Exponents outside this range are already zero or infinity in `f32`. The bound is applied to
    // `whole` in the float domain rather than left to the integer clamp: above 2^22 the cast
    // saturates to `i32::MAX` and the `+ 127` then wraps to a large negative, which turned a huge
    // positive exponent into a zero. Clamping here is a no-op for every input whose exponent is
    // already pinned at 0 or 254, and it is what the vector kernel in `simd` does, so both paths
    // stay bit-identical.
    let whole = whole.clamp(-127.0, 128.0);
    let exponent = (whole as i32 + 127).clamp(0, 254) as u32;
    polynomial * f32::from_bits(exponent << 23)
}

/// `x * sigmoid(x)`, the scalar definition the vector kernel reproduces bit for bit.
///
/// Visible to the crate so `simd`'s test can compare against it; the vectorized path lives in
/// [`crate::simd::silu_mul`], which is where the profile's `rest` bucket goes.
pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + exp(-value))
}

fn silu_mul(up: &mut [f32], gate: &[f32]) {
    crate::simd::silu_mul(up, gate);
}

#[inline]
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

/// [`vector`] for a tensor the architecture may not have: empty if it is absent, which is what the
/// bias and per-head-norm slots of an architecture without them read as.
fn optional_vector(
    file: &GgufFile,
    data: &[u8],
    name: &str,
    len: usize,
) -> Result<Vec<f32>, ViperaError> {
    if !file.tensors.iter().any(|tensor| tensor.name == name) {
        return Ok(Vec::new());
    }
    vector(file, data, name, len)
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

    /// The in-place kernel a fused product runs on a worker is `rms_norm`'s arithmetic, bit for
    /// bit, so folding the norm into the activation copy cannot move a token.
    #[test]
    fn rms_norm_in_place_is_rms_norm() {
        let weight: Vec<f32> = (0..1536)
            .map(|at| ((at % 37) as f32 - 18.0) / 23.0)
            .collect();
        let input: Vec<f32> = (0..1536)
            .map(|at| ((at * 7919 % 1013) as f32 - 500.0) / 97.0)
            .collect();
        let mut expected = vec![0.0_f32; input.len()];
        rms_norm(&input, &weight, 1e-6, &mut expected);
        let mut in_place = input.clone();
        rms_norm_in_place(&mut in_place, &weight, 1e-6);
        for (at, (expected, got)) in expected.iter().zip(&in_place).enumerate() {
            assert_eq!(
                expected.to_bits(),
                got.to_bits(),
                "value {at}: {expected} vs {got}"
            );
        }
    }

    /// qwen3's per-head norms must be `rms_norm` on each head's own vector, with the same weight
    /// for every head: two heads of different magnitude scale independently, and the result is the
    /// whole-vector kernel's bits when there is one head.
    #[test]
    fn per_head_rms_norm_is_rms_norm_on_each_head() {
        let weight = [0.5_f32, 1.0, 2.0, -1.0];
        let start = [3.0_f32, 4.0, 0.0, 0.0, -1.0, 2.0, 2.0, 1.0];
        let mut got = start;
        per_head_rms_norm(&mut got, &weight, 1e-6, 2, 4);
        let mut want = [0.0_f32; 8];
        for head in 0..2 {
            rms_norm(
                &start[head * 4..(head + 1) * 4],
                &weight,
                1e-6,
                &mut want[head * 4..(head + 1) * 4],
            );
        }
        assert_eq!(got, want);

        let mut single = start[..4].to_vec();
        let mut whole = [0.0_f32; 4];
        rms_norm(&start[..4], &weight, 1e-6, &mut whole);
        per_head_rms_norm(&mut single, &weight, 1e-6, 1, 4);
        assert_eq!(single, whole);
    }

    /// The in-register exponential has to stay inside libm's shadow on the value the model actually
    /// uses, `x * sigmoid(x)`. Its own error is `x * log2 e` rounded in `f32`, which is largest where
    /// the sigmoid has already saturated, so checking the exponential alone overstates the damage.
    #[test]
    fn fast_silu_stays_close_to_libm() {
        let mut worst = 0.0_f32;
        let mut value = -30.0_f32;
        while value < 30.0 {
            let got = silu(value);
            let want = value / (1.0 + (-value).exp());
            let scale = want.abs().max(1e-30);
            worst = worst.max((got - want).abs() / scale);
            value += 0.013;
        }
        assert!(worst < 2e-6, "worst relative error {worst}");
    }

    /// Measurement: what the SwiGLU multiply costs for one 128-token chunk, which is the part of the
    /// profile's `rest` a faster sigmoid would remove. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib silu_mul_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn silu_mul_cost() {
        let layers = 28usize;
        let per_layer = 128 * 3072;
        let mut up: Vec<f32> = (0..per_layer)
            .map(|i| (i % 97) as f32 * 0.05 - 2.0)
            .collect();
        let gate: Vec<f32> = (0..per_layer)
            .map(|i| (i % 89) as f32 * 0.04 - 1.8)
            .collect();
        let start = std::time::Instant::now();
        for _ in 0..layers {
            silu_mul(&mut up, &gate);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "silu_mul: {} elements in {:.2} ms ({:.1} ns/element, single thread)",
            layers * per_layer,
            elapsed,
            elapsed * 1e6 / (layers * per_layer) as f64
        );
        std::hint::black_box(&up);
    }

    /// Measurement: the two parts of the profile's `Norm` phase for one 128-token chunk, the norm
    /// itself and the rotary embedding beside it, plus what the hoisted rotary path costs in the
    /// chunk pass. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib norm_phase_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn norm_phase_cost() {
        let layers = 28usize;
        let tokens = 128usize;
        let (heads, kv_heads, head_dim) = (12usize, 2usize, 128usize);
        let mut q = vec![0.1_f32; heads * head_dim];
        let mut k = vec![0.1_f32; kv_heads * head_dim];
        let start = std::time::Instant::now();
        for _ in 0..layers {
            for token in 0..tokens {
                rope(&mut q, token, heads, head_dim, 1_000_000.0);
                rope(&mut k, token, kv_heads, head_dim, 1_000_000.0);
            }
        }
        let rope_ms = start.elapsed().as_secs_f64() * 1e3;

        // The chunk pass builds one table per position and rotates from it in every layer.
        let start = std::time::Instant::now();
        let angles: Vec<RopeAngles> = (0..tokens)
            .map(|token| RopeAngles::new(token, head_dim, 1_000_000.0))
            .collect();
        for _ in 0..layers {
            for table in &angles {
                rope_with(&mut q, table, heads, head_dim);
                rope_with(&mut k, table, kv_heads, head_dim);
            }
        }
        let hoisted_ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box((&q, &k));

        let hidden = 1536usize;
        let x: Vec<f32> = (0..hidden).map(|i| (i % 71) as f32 * 0.03 - 1.0).collect();
        let weight = vec![1.0_f32; hidden];
        let mut out = vec![0.0_f32; hidden];
        let start = std::time::Instant::now();
        for _ in 0..layers * tokens * 2 {
            rms_norm(&x, &weight, 1e-6, &mut out);
        }
        let norm_ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&out);

        // The four-accumulator sum that was measured and reverted (see `docs/benchmarks.md`,
        // 2026-11-19): it breaks the serial add chain, and it moves the logits by ~0.1 because the
        // per-block i8 activation quantization downstream turns a last-bit change into a discrete
        // jump. Kept here as the number, not as a path.
        fn rms_norm_four_lanes(input: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
            let mut sums = [0.0_f32; 4];
            let (chunks, remainder) = input.as_chunks::<4>();
            for chunk in chunks {
                for (slot, value) in sums.iter_mut().zip(chunk) {
                    *slot += value * value;
                }
            }
            let mut total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            for value in remainder {
                total += value * value;
            }
            let inverse = 1.0 / (total / input.len() as f32 + epsilon).sqrt();
            for ((dst, weight), input) in out.iter_mut().zip(weight).zip(input) {
                *dst = input * inverse * weight;
            }
        }
        let start = std::time::Instant::now();
        for _ in 0..layers * tokens * 2 {
            rms_norm_four_lanes(&x, &weight, 1e-6, &mut out);
        }
        let lanes_ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&out);
        println!(
            "norm phase per 128-token chunk: rope per call {rope_ms:.2} ms, rotary hoisted {hoisted_ms:.2} ms, rms_norm {norm_ms:.2} ms, rms_norm four lanes {lanes_ms:.2} ms"
        );
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

    /// Measurement: what `attention` costs on its own, split into the score dot, the softmax and the
    /// value accumulation. Run with:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib attention_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn attention_cost_per_position() {
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let round = |positions: usize| {
            let q: Vec<f32> = (0..shape.heads * shape.head_dim)
                .map(|at| (at % 17) as f32 * 0.01)
                .collect();
            let k: Vec<f32> = (0..positions * shape.kv_heads * shape.head_dim)
                .map(|at| (at % 13) as f32 * 0.02)
                .collect();
            let v: Vec<f32> = (0..positions * shape.kv_heads * shape.head_dim)
                .map(|at| (at % 11) as f32 * 0.03)
                .collect();
            let mut scores = Vec::new();
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            let rounds = 200_u32;
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                attention(&q, &k, &v, positions, shape, &mut scores, &mut out);
            }
            started.elapsed().as_secs_f64() / f64::from(rounds) * 1e3
        };
        // The score dot alone, at the same shape and count: heads * positions dots of 128 values.
        let dot_round = |positions: usize| -> f64 {
            let q: Vec<f32> = (0..shape.head_dim)
                .map(|at| (at % 17) as f32 * 0.01)
                .collect();
            let k: Vec<f32> = (0..positions * shape.kv_heads * shape.head_dim)
                .map(|at| (at % 13) as f32 * 0.02)
                .collect();
            let rounds = 20_u32;
            let started = std::time::Instant::now();
            let mut sink = 0.0_f32;
            for _ in 0..rounds {
                for head in 0..shape.heads {
                    let kv_head = head % shape.kv_heads;
                    for position in 0..positions {
                        let base = (position * shape.kv_heads + kv_head) * shape.head_dim;
                        sink += crate::simd::dot_f32(&q, &k[base..base + shape.head_dim]);
                    }
                }
            }
            let elapsed = started.elapsed().as_secs_f64() / f64::from(rounds) * 1e3;
            assert!(sink.is_finite());
            elapsed
        };
        for positions in [32_usize, 256, 1024] {
            println!(
                "positions {positions:>5}: attention {:.2} ms, score dots only {:.2} ms",
                round(positions),
                dot_round(positions)
            );
        }
    }

    /// Where a prefill's attention time actually goes, in the three parts it is built from.
    ///
    /// A 4096-token chunk profiles as ~41% attention, and the phases inside it are the score dots,
    /// the softmax's `exp`, and the weighted sum of V. This times each on the shape the model uses
    /// (12 heads, 2 KV heads, 128 values per head, f16 KV in place) at a deep history, so a change
    /// can be aimed at the part that is actually there rather than at the one that looks expensive.
    ///
    /// `cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture prefill_attention_parts`
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn prefill_attention_parts() {
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let positions = 2048usize;
        let tokens_per_block = 16;
        let table: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            positions,
        );
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(shape.kv_heads * shape.head_dim),
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            layout: KvLayout::F16,
        };
        let read = KvRead::of(&kv);
        let layer = 0usize;
        let layer_bytes = KvLayout::F16.bytes_per_row(shape.kv_heads * shape.head_dim);
        let group = shape.heads / shape.kv_heads;
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.005)
            .collect();
        let mut scores = vec![0.0_f32; shape.heads * positions];

        // Pass 1: one call per KV head, the interleaved exact kernel.
        let round = |name: &str, body: &mut dyn FnMut()| {
            let rounds = 40;
            body();
            let mut best = f64::MAX;
            for _ in 0..rounds {
                let started = std::time::Instant::now();
                body();
                best = best.min(started.elapsed().as_secs_f64());
            }
            println!("  {name:<16} {:>6.2} ms per history", best * 1e3);
        };

        round("score dots", &mut || {
            let mut index = 0;
            read.for_each_row(
                order_for_test(),
                layer,
                layer_bytes,
                positions,
                |_at, base| {
                    for kv_head in 0..shape.kv_heads {
                        let packed = base + kv_head * shape.head_dim * 2;
                        let first = kv_head * group;
                        crate::simd::dot_f16_f32_heads_exact(
                            &q,
                            first,
                            group,
                            shape.head_dim,
                            &read.storage[packed..],
                            &mut scores[first..first + group],
                        );
                    }
                    index += 1;
                    let _ = index;
                    Ok(())
                },
            )
            .expect("dots");
        });

        // The softmax without its dots: exp and sum over every score.
        round("exp and sum", &mut || {
            let mut total = 0.0_f32;
            for local in 0..shape.heads {
                let window = &mut scores[local * positions..(local + 1) * positions];
                for score in window.iter_mut() {
                    *score = score.exp();
                    total += *score;
                }
            }
            std::hint::black_box(total);
        });

        // Pass 2: V rows weighted into twelve accumulators, one call per KV head.
        let mut bases = Vec::new();
        read.for_each_row(
            order_for_test(),
            layer,
            layer_bytes,
            positions,
            |_at, base| {
                bases.push(read.value_row(order_for_test(), base, layer_bytes));
                Ok(())
            },
        )
        .expect("bases");

        round("V axpy row", &mut || {
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            let weights = [0.001_f32; 16];
            for &v_base in &bases {
                for kv_head in 0..shape.kv_heads {
                    let packed = v_base + kv_head * shape.head_dim * 2;
                    let first = kv_head * group;
                    crate::simd::axpy_f16_f32_heads_exact(
                        &mut out[first * shape.head_dim..(first + group) * shape.head_dim],
                        shape.head_dim,
                        &weights[..group],
                        &read.storage[packed..],
                    );
                }
            }
            std::hint::black_box(out[0]);
        });

        let block_weights = vec![0.001_f32; 8 * group];
        round("V axpy block", &mut || {
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            for chunk in bases.as_chunks::<8>().0 {
                for kv_head in 0..shape.kv_heads {
                    let mut rows: [&[u8]; 8] = [&[]; 8];
                    for (slot, &v_base) in rows.iter_mut().zip(chunk) {
                        *slot = &read.storage[v_base + kv_head * shape.head_dim * 2..];
                    }
                    let first = kv_head * group;
                    crate::simd::axpy_f16_f32_heads_exact_block::<8>(
                        &mut out[first * shape.head_dim..(first + group) * shape.head_dim],
                        shape.head_dim,
                        &block_weights,
                        &rows,
                    );
                }
            }
            std::hint::black_box(out[0]);
        });
    }

    /// The row-sharing kernel against the per-entry one a chunk takes today: same answer within
    /// rounding, at a history length that uses the streaming kernel and with a causal tail where
    /// the queries of one block see different amounts of history.
    #[test]
    fn qblock_attention_matches_the_per_entry_kernel_within_rounding() {
        let shape = AttnShape {
            heads: 6,
            kv_heads: 2,
            head_dim: 64,
        };
        let query = shape.heads * shape.head_dim;
        let entries = 9usize;
        let first = 260usize;
        let tokens_per_block = 16;
        let table: Vec<u32> = (0..(first + entries).div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            first + entries,
        );
        let bytes_per_token = 2 * 2 * KvLayout::F16.bytes_per_row(shape.kv_heads * shape.head_dim);
        let kv = KvView {
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            bytes_per_token,
            layout: KvLayout::F16,
        };
        let q: Vec<f32> = (0..entries * query)
            .map(|at| ((at * 13 % 29) as f32 - 14.0) * 0.07)
            .collect();
        let positions: Vec<usize> = (0..entries).map(|slot| first + slot + 1).collect();

        // One call per entry, as a chunk does it today.
        let mut expected = vec![0.0_f32; entries * query];
        for entry in 0..entries {
            let read = KvRead::of(&kv);
            let mut scratch = AttnScratch::default();
            attend_entry(
                AttentionCall {
                    q: &q[entry * query..][..query],
                    kv: &read,
                    layer: 0,
                    positions: positions[entry],
                    shape,
                    heads: 0..shape.heads,
                    out: &mut expected[entry * query..][..query],
                },
                &mut scratch,
            )
            .expect("one entry");
        }

        let reads: Vec<KvRead<'_>> = (0..entries).map(|_| KvRead::of(&kv)).collect();
        let mut shared = vec![0.0_f32; entries * query];
        attend_rows_shared(&q, &reads, &positions, 0, shape, &mut shared, 1).expect("shared");

        let worst = expected
            .iter()
            .zip(&shared)
            .map(|(one, many)| (one - many).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-5, "row sharing moved an output by {worst}");
    }

    fn order_for_test() -> RowOrder {
        RowOrder::from_env()
    }

    /// A KV cache of `blocks` physical blocks, two layers, filled with a fixed pseudo-random
    /// stream through the same `write` the engine uses.
    fn filled_kv(
        storage: &mut Vec<u8>,
        layout: KvLayout,
        shape: AttnShape,
        tokens_per_block: usize,
        table: &[u32],
        positions: usize,
    ) {
        let width = shape.kv_heads * shape.head_dim;
        let layers = 2;
        let bytes_per_token = layers * 2 * layout.bytes_per_row(width);
        let blocks = table
            .iter()
            .copied()
            .max()
            .map_or(0, |max| max as usize + 1);
        storage.clear();
        storage.resize(blocks * tokens_per_block * bytes_per_token, 0);
        let mut state = 0x9E37_79B9_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        let mut kv = KvView {
            storage,
            table,
            tokens_per_block,
            bytes_per_token,
            layout,
        };
        for position in 0..positions {
            for layer in 0..layers {
                let k: Vec<f32> = (0..width).map(|_| next()).collect();
                let v: Vec<f32> = (0..width).map(|_| next()).collect();
                kv.write(position, layer, shape, &k, &v).expect("write");
            }
        }
    }

    /// The buffered kernel against the streaming one with sharing off: same bits, at every block
    /// boundary. Sharing is pinned off here because the shared kernel regroups each head's sum and
    /// is only held within rounding, by the test below.
    #[test]
    fn streaming_attention_without_sharing_is_bit_identical_to_decoding_the_layer_first() {
        let shape = AttnShape {
            heads: 6,
            kv_heads: 2,
            head_dim: 64,
        };
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        // Physical blocks are not in logical order, as when sequences share one buffer.
        let table = [3_u32, 0, 2, 1];
        for layout in [KvLayout::F16, KvLayout::Q8_0] {
            let mut storage = Vec::new();
            filled_kv(&mut storage, layout, shape, 16, &table, 61);
            let width = shape.kv_heads * shape.head_dim;
            let kv = KvView {
                bytes_per_token: 2 * 2 * layout.bytes_per_row(width),
                storage: &mut storage,
                table: &table,
                tokens_per_block: 16,
                layout,
            };
            for layer in 0..2 {
                // One position, a block boundary on each side, and a partial last block.
                for positions in [1, 15, 16, 17, 32, 61] {
                    let (mut k_all, mut v_all, mut scores) = (Vec::new(), Vec::new(), Vec::new());
                    kv.load_layer(layer, shape, positions, &mut k_all, &mut v_all)
                        .expect("load");
                    let mut expected = vec![0.0_f32; shape.heads * shape.head_dim];
                    attention(
                        &q,
                        &k_all,
                        &v_all,
                        positions,
                        shape,
                        &mut scores,
                        &mut expected,
                    );

                    let actual =
                        streaming_with(false, &q, &kv, layer, positions, shape, 0..shape.heads);
                    let bits = |values: &[f32]| -> Vec<u32> {
                        values.iter().map(|value| value.to_bits()).collect()
                    };
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "{layout:?} layer {layer} positions {positions}"
                    );
                }
            }
        }
    }

    /// Runs the streaming kernel with the sharing switch passed explicitly, so a test never
    /// depends on the process environment.
    #[allow(clippy::too_many_arguments)]
    fn streaming_with(
        share: bool,
        q: &[f32],
        kv: &KvView<'_>,
        layer: usize,
        positions: usize,
        shape: AttnShape,
        heads: std::ops::Range<usize>,
    ) -> Vec<f32> {
        let mut out = vec![f32::NAN; (heads.end - heads.start) * shape.head_dim];
        let (mut scores, mut row, mut head_state) = (Vec::new(), Vec::new(), Vec::new());
        attention_streaming_heads(
            q,
            &KvRead::of(kv),
            AttentionSwitches::from_env().order,
            share,
            layer,
            positions,
            shape,
            heads,
            &mut scores,
            &mut row,
            &mut head_state,
            &mut out,
        )
        .expect("attention");
        out
    }

    /// With sharing on, the streaming kernel is within rounding of the per-head kernel and not
    /// equal to it, so that is what is pinned: for group sizes of 1, 3, 6 and 7, whole layers and
    /// head ranges that cut a group, at position counts around the sixteen-row value blocks and
    /// the KV block boundaries, over a block table that is not in logical order. Rows that are
    /// not read in place (`q8_0`) take the per-head kernel whatever the switch says, bit for bit.
    #[test]
    fn shared_group_attention_matches_the_per_head_kernel_within_rounding() {
        for (heads, kv_heads, head_dim) in [
            (6_usize, 2_usize, 64_usize),
            (12, 2, 128),
            (4, 4, 64),
            (14, 2, 64),
        ] {
            let shape = AttnShape {
                heads,
                kv_heads,
                head_dim,
            };
            let q: Vec<f32> = (0..heads * head_dim)
                .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
                .collect();
            let table: Vec<u32> = (0..20_u32).map(|block| block * 7 % 20).collect();
            for layout in [KvLayout::F16, KvLayout::Q8_0] {
                let mut storage = Vec::new();
                filled_kv(&mut storage, layout, shape, 16, &table, 300);
                let width = shape.kv_heads * shape.head_dim;
                let kv = KvView {
                    bytes_per_token: 2 * 2 * layout.bytes_per_row(width),
                    storage: &mut storage,
                    table: &table,
                    tokens_per_block: 16,
                    layout,
                };
                let mut ranges = vec![0..heads, 0..1];
                if heads > 2 {
                    ranges.push(1..heads - 1);
                }
                for layer in 0..2 {
                    for positions in [1_usize, 15, 16, 17, 100, 300] {
                        for range in &ranges {
                            let exact = streaming_with(
                                false,
                                &q,
                                &kv,
                                layer,
                                positions,
                                shape,
                                range.clone(),
                            );
                            let shared = streaming_with(
                                true,
                                &q,
                                &kv,
                                layer,
                                positions,
                                shape,
                                range.clone(),
                            );
                            for (at, (shared, exact)) in shared.iter().zip(&exact).enumerate() {
                                let context = format!(
                                    "{layout:?} {heads}/{kv_heads}x{head_dim} layer {layer} positions {positions} heads {range:?} element {at}"
                                );
                                if layout == KvLayout::F16 {
                                    assert!(
                                        (shared - exact).abs() <= 1e-5 + 1e-5 * exact.abs(),
                                        "{context}: {shared} against {exact}"
                                    );
                                } else {
                                    assert_eq!(shared.to_bits(), exact.to_bits(), "{context}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// One layer's streaming attention for one entry, per-head kernel against shared groups, on
    /// the Qwen2.5-1.5B shape (12 query heads over 2 key heads of 128), `f16` rows read in place.
    /// Both arms run in one process, interleaved, so the ratio is what is worth reading. It is a
    /// measurement, not a check, so it is `#[ignore]`d:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib attention_gqa_share_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn attention_gqa_share_cost() {
        use std::time::Instant;
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        for positions in [256_usize, 1024, 4096] {
            let blocks = positions.div_ceil(16);
            let mut table: Vec<u32> = (0..blocks as u32).collect();
            // A fixed shuffle, so the rows are not walked in physical order.
            for at in (1..table.len()).rev() {
                table.swap(at, at * 2654435761 % (at + 1));
            }
            let mut storage = Vec::new();
            filled_kv(&mut storage, KvLayout::F16, shape, 16, &table, positions);
            let width = shape.kv_heads * shape.head_dim;
            let kv = KvView {
                bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
                storage: &mut storage,
                table: &table,
                tokens_per_block: 16,
                layout: KvLayout::F16,
            };
            let calls = (2_000_000 / positions).max(4);
            let mut best = [f64::INFINITY; 2];
            for _ in 0..7 {
                for (arm, slot) in best.iter_mut().enumerate() {
                    let started = Instant::now();
                    for _ in 0..calls {
                        let out =
                            streaming_with(arm == 1, &q, &kv, 0, positions, shape, 0..shape.heads);
                        std::hint::black_box(out);
                    }
                    *slot = slot.min(started.elapsed().as_secs_f64() * 1e3 / calls as f64);
                }
            }
            println!(
                "positions {positions:>5}: per-head {:.3} ms, shared {:.3} ms per layer ({:.2}x)",
                best[0],
                best[1],
                best[0] / best[1]
            );
        }
    }

    #[test]
    fn streaming_attention_reports_a_position_with_no_block() {
        let shape = AttnShape {
            heads: 2,
            kv_heads: 1,
            head_dim: 32,
        };
        let table = [0_u32];
        let mut storage = Vec::new();
        filled_kv(&mut storage, KvLayout::F16, shape, 4, &table, 4);
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(32),
            storage: &mut storage,
            table: &table,
            tokens_per_block: 4,
            layout: KvLayout::F16,
        };
        let q = vec![0.5_f32; 64];
        let mut out = vec![0.0_f32; 64];
        let error =
            attention_streaming(&q, &kv, 0, 5, shape, &mut AttnScratch::default(), &mut out);
        assert!(
            error.is_err(),
            "position 4 has no block in a one-block table"
        );
    }

    /// The gate in [`attend`] must not change a number beyond the regrouping the shared kernel
    /// does, including when one scratch crosses the threshold: the short-history arm fills
    /// `k_all`/`v_all`, and the streaming arm must not leave anything behind that the next short
    /// write reads.
    #[test]
    fn attend_crosses_the_streaming_threshold_without_changing_the_result() {
        let shape = AttnShape {
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
        };
        let width = shape.kv_heads * shape.head_dim;
        let tokens_per_block = 16;
        let deepest = STREAMING_MIN_POSITIONS + 5;
        let table: Vec<u32> = (0..deepest.div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            deepest,
        );
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            layout: KvLayout::F16,
        };
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 5 % 19) as f32 - 9.0) * 0.1)
            .collect();
        let mut scratch = AttnScratch::default();
        let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
        for positions in [1, 8, STREAMING_MIN_POSITIONS - 1, deepest] {
            attend(&q, &kv, 0, positions, shape, &mut scratch, &mut out).expect("attention");
        }

        let (mut k_all, mut v_all, mut scores) = (Vec::new(), Vec::new(), Vec::new());
        kv.load_layer(0, shape, deepest, &mut k_all, &mut v_all)
            .expect("load");
        let mut expected = vec![0.0_f32; shape.heads * shape.head_dim];
        attention(
            &q,
            &k_all,
            &v_all,
            deepest,
            shape,
            &mut scores,
            &mut expected,
        );
        let magnitude = expected
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        for (at, (streamed, buffered)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (streamed - buffered).abs() <= 1e-5 * magnitude.max(1.0),
                "element {at}: streaming {streamed} against buffered {buffered}"
            );
        }
    }

    /// The kernel choice is one function, so this pins it to the documented threshold — and the
    /// two kernels it chooses between have to agree bit for bit at the position where they swap.
    #[test]
    fn the_attention_kernel_follows_the_position_threshold() {
        assert_eq!(algorithm_for(1).name(), "buffered");
        assert_eq!(
            algorithm_for(STREAMING_MIN_POSITIONS - 1).name(),
            "buffered"
        );
        assert_eq!(algorithm_for(STREAMING_MIN_POSITIONS).name(), "streaming");

        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let positions = STREAMING_MIN_POSITIONS;
        let tokens_per_block = 16;
        let table: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            positions,
        );
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            layout: KvLayout::F16,
        };
        let read = KvRead::of(&kv);
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        let mut outputs = Vec::new();
        for algorithm in [
            &BufferedAttention as &dyn AttentionAlgorithm,
            &StreamingAttention,
        ] {
            let mut scratch = AttnScratch::default();
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            algorithm
                .attend(
                    AttentionCall {
                        q: &q,
                        kv: &read,
                        layer: 0,
                        positions,
                        shape,
                        heads: 0..shape.heads,
                        out: &mut out,
                    },
                    &mut scratch,
                )
                .expect("attention");
            outputs.push(out);
        }
        let magnitude = outputs[0]
            .iter()
            .fold(0.0_f32, |max, value| max.max(value.abs()));
        for (at, (streamed, buffered)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
            assert!(
                (streamed - buffered).abs() <= 1e-5 * magnitude.max(1.0),
                "the kernels disagree at the boundary, element {at}: {streamed} against {buffered}"
            );
        }
    }

    /// What one layer of one decode step pays for its K/V: decode the history, then attend.
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib kv_attention_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn kv_attention_cost_per_layer() {
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        for layout in [KvLayout::F16, KvLayout::Q8_0] {
            for positions in [32_usize, 256, 1024, 4096] {
                let tokens_per_block = 16;
                let table: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();
                let mut storage = Vec::new();
                filled_kv(
                    &mut storage,
                    layout,
                    shape,
                    tokens_per_block,
                    &table,
                    positions,
                );
                let kv = KvView {
                    bytes_per_token: 2 * 2 * layout.bytes_per_row(width),
                    storage: &mut storage,
                    table: &table,
                    tokens_per_block,
                    layout,
                };
                let rounds = (2_000_000 / positions).clamp(20, 400) as u32;
                let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
                let (mut k_all, mut v_all, mut scores) = (Vec::new(), Vec::new(), Vec::new());
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    kv.load_layer(0, shape, positions, &mut k_all, &mut v_all)
                        .expect("load");
                    attention(&q, &k_all, &v_all, positions, shape, &mut scores, &mut out);
                }
                let before = started.elapsed().as_secs_f64() / f64::from(rounds) * 1e3;
                let mut scratch = AttnScratch::default();
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    attention_streaming(&q, &kv, 0, positions, shape, &mut scratch, &mut out)
                        .expect("attention");
                }
                let after = started.elapsed().as_secs_f64() / f64::from(rounds) * 1e3;
                println!(
                    "{layout:?} positions {positions:>5}: load+attention {before:.3} ms, \
                     streaming {after:.3} ms, {:.2}x",
                    before / after
                );
            }
        }
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
    fn kv_positions_map_through_the_block_table() {
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
        assert_eq!(kv.token_in_block(0).expect("block"), (4 * per_token, 0));
        assert_eq!(kv.token_in_block(3).expect("block"), (4 * per_token, 3));
        assert_eq!(
            kv.token_in_block(4).expect("block"),
            (0, 0),
            "block 0 holds the second half"
        );
        assert_eq!(kv.token_in_block(7).expect("block"), (0, 3));
        assert!(
            kv.token_in_block(8).is_err(),
            "a position beyond the block table has no block"
        );
    }

    /// The strided walk at 1024 positions, both orders, one process: same values at the same
    /// logical positions, only the addresses differ. Prints the cost and asserts the two arms
    /// compute the same bits, which is what makes the comparison meaningful.
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib kv_row_order_at_1024 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: prints a number, run with --ignored --nocapture"]
    fn kv_row_order_at_1024() {
        // The model this repository benchmarks: 28 layers, 12 heads over 2 KV heads of 128.
        let layers = 28;
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let row = KvLayout::F16.bytes_per_row(width);
        let bytes_per_token = layers * 2 * row;
        // `DEFAULT_TOKENS_PER_BLOCK`, so the walk is the one the binary runs.
        let per_block = 16;
        let positions = 1024usize;
        let blocks = positions.div_ceil(per_block);
        let table: Vec<u32> = (0..blocks as u32).collect();
        let mut values = vec![0.0_f32; width];
        let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7) % 23) as f32 * 0.03 - 0.3)
            .collect();

        let mut layout_for = |order: RowOrder| {
            let mut storage = vec![0u8; blocks * per_block * bytes_per_token];
            for layer in 0..layers {
                for position in 0..positions {
                    let walk = order.row_walk(
                        position / per_block * per_block * bytes_per_token,
                        position % per_block,
                        per_block,
                        bytes_per_token,
                        row,
                        layer,
                    );
                    for (at, value) in values.iter_mut().enumerate() {
                        *value = ((position * 31 + layer * 7 + at * 13) % 97) as f32 * 0.01 - 0.5;
                    }
                    KvLayout::F16
                        .encode(&mut storage[walk.k..walk.k + row], &values)
                        .expect("write K");
                    let v_base = walk.k + walk.v;
                    KvLayout::F16
                        .encode(&mut storage[v_base..v_base + row], &values)
                        .expect("write V");
                }
            }
            storage
        };

        let mut arms = Vec::new();
        for order in [RowOrder::TokenMajor, RowOrder::LayerMajor] {
            let mut storage = layout_for(order);
            let kv = KvView {
                storage: &mut storage,
                table: &table,
                tokens_per_block: per_block,
                bytes_per_token,
                layout: KvLayout::F16,
            };
            let read = KvRead::of(&kv);
            let run = |read: &KvRead<'_>, out: &mut [f32]| {
                let mut scores = Vec::new();
                let mut row_buffer = Vec::new();
                let mut head_state = Vec::new();
                for layer in 0..layers {
                    attention_streaming_heads(
                        &q,
                        read,
                        order,
                        false,
                        layer,
                        positions,
                        shape,
                        0..shape.heads,
                        &mut scores,
                        &mut row_buffer,
                        &mut head_state,
                        out,
                    )
                    .expect("attention");
                }
            };
            run(&read, &mut out);
            let mut best = f64::INFINITY;
            let mut rounds = 0;
            while rounds < 5 {
                let started = std::time::Instant::now();
                run(&read, &mut out);
                best = best.min(started.elapsed().as_secs_f64() * 1000.0);
                rounds += 1;
            }
            println!(
                "{order:?}: {best:6.2} ms for {layers} layers x {positions} positions ({:.3} ms/layer)",
                best / layers as f64
            );
            arms.push((order, best, out.clone()));
        }
        assert_eq!(
            arms[0].2, arms[1].2,
            "the two orders must compute the same output bits"
        );
    }

    #[test]
    fn the_head_split_never_has_more_workers_than_heads_threads_or_the_cap() {
        assert_eq!(attention_workers(8, 12, None), 8);
        assert_eq!(
            attention_workers(16, 12, None),
            12,
            "one worker per head at most"
        );
        assert_eq!(attention_workers(8, 12, Some(3)), 3);
        assert_eq!(
            attention_workers(2, 12, Some(6)),
            2,
            "the cap cannot add threads"
        );
        assert_eq!(attention_workers(1, 12, None), 1);
        assert_eq!(
            attention_workers(8, 12, Some(1)),
            1,
            "a cap of one is the calling thread's share"
        );
    }

    #[test]
    fn the_head_ranges_cover_every_head_once_whatever_the_worker_count() {
        for heads in [1_usize, 2, 7, 12, 28, 64] {
            for workers in 1..=heads {
                let mut next = 0;
                let (mut shortest, mut longest) = (usize::MAX, 0);
                for part in 0..workers {
                    let range = head_range(heads, workers, part);
                    assert_eq!(
                        range.start, next,
                        "{heads} heads, {workers} workers, part {part}"
                    );
                    next = range.end;
                    shortest = shortest.min(range.len());
                    longest = longest.max(range.len());
                }
                assert_eq!(next, heads, "{heads} heads over {workers} workers");
                assert!(
                    longest - shortest <= 1,
                    "{heads} heads over {workers} workers"
                );
                assert!(shortest >= 1, "no worker is left without a head");
            }
        }
    }

    /// What creating and joining the threads of one `thread::scope` costs, per scope and per thread,
    /// for the worker counts the head split uses. Attention pays this once per layer, 28 times a
    /// token, on top of the arithmetic it divides.
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib scoped_spawn_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: prints a table, run with --ignored --nocapture"]
    fn scoped_spawn_cost() {
        for workers in [1_usize, 2, 3, 4, 6, 8] {
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let rounds = 300;
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    thread::scope(|scope| {
                        let handles: Vec<_> = (0..workers)
                            .map(|_| scope.spawn(|| std::hint::black_box(1_u64)))
                            .collect();
                        for handle in handles {
                            std::hint::black_box(handle.join().expect("worker"));
                        }
                    });
                }
                best = best.min(started.elapsed().as_secs_f64() / f64::from(rounds));
            }
            println!(
                "{workers} threads per scope: {:6.1} us per scope, {:5.1} us per thread, x28 layers = {:5.2} ms a token",
                best * 1e6,
                best * 1e6 / workers as f64,
                best * 28.0 * 1e3
            );
        }
    }

    /// The attention switches in every combination, at two depths, all in one process on the same
    /// bytes. Prints the table and asserts every arm at a depth computes the same output, which is
    /// what makes a timing difference a scheduling or locality difference and nothing else.
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib attention_combinations -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: prints a table, run with --ignored --nocapture"]
    fn attention_combinations() {
        let layers = 28;
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let row = KvLayout::F16.bytes_per_row(width);
        let bytes_per_token = layers * 2 * row;
        let per_block = 16;
        let query = shape.heads * shape.head_dim;
        let q: Vec<f32> = (0..query)
            .map(|at| ((at * 7) % 23) as f32 * 0.03 - 0.3)
            .collect();
        let mut out = vec![0.0_f32; query];
        let orders = [RowOrder::TokenMajor, RowOrder::LayerMajor];

        for positions in [1024usize, 4096, 8192] {
            let blocks = positions.div_ceil(per_block);
            let table: Vec<u32> = (0..blocks as u32).collect();
            let lengths = [positions];
            let mut arms: Vec<(AttentionSwitches, f64, Vec<f32>)> = Vec::new();
            for order in orders {
                let mut storage = vec![0u8; blocks * per_block * bytes_per_token];
                let mut values = vec![0.0_f32; width];
                for layer in 0..layers {
                    for position in 0..positions {
                        let walk = order.row_walk(
                            position / per_block * per_block * bytes_per_token,
                            position % per_block,
                            per_block,
                            bytes_per_token,
                            row,
                            layer,
                        );
                        for (at, value) in values.iter_mut().enumerate() {
                            *value =
                                ((position * 31 + layer * 7 + at * 13) % 97) as f32 * 0.01 - 0.5;
                        }
                        KvLayout::F16
                            .encode(&mut storage[walk.k..walk.k + row], &values)
                            .expect("write K");
                        let v_base = walk.k + walk.v;
                        KvLayout::F16
                            .encode(&mut storage[v_base..v_base + row], &values)
                            .expect("write V");
                    }
                }
                let kv = KvView {
                    storage: &mut storage,
                    table: &table,
                    tokens_per_block: per_block,
                    bytes_per_token,
                    layout: KvLayout::F16,
                };
                let read = KvRead::of(&kv);
                let views = std::slice::from_ref(&read);

                for (split_heads, workers) in [
                    (false, None),
                    (true, None),
                    (true, Some(2)),
                    (true, Some(3)),
                    (true, Some(4)),
                    (true, Some(6)),
                    (true, Some(8)),
                ] {
                    let switches = AttentionSwitches {
                        split_heads,
                        kv_split: false,
                        order,
                        workers,
                    };
                    let mut scratch = AttnScratch::default();
                    let pass = |scratch: &mut AttnScratch, out: &mut Vec<f32>| {
                        out.fill(0.0);
                        for layer in 0..layers {
                            attend_entries(&q, views, &lengths, layer, shape, scratch, out)
                                .expect("attention");
                        }
                    };
                    // Four sweeps, direction alternating, so no arm keeps a warmer position in the
                    // sequence than another.
                    let mut best = f64::INFINITY;
                    for sweep in 0..4 {
                        force_switches(switches);
                        if sweep < 2 {
                            pass(&mut scratch, &mut out);
                            continue;
                        }
                        let started = std::time::Instant::now();
                        pass(&mut scratch, &mut out);
                        best = best.min(started.elapsed().as_secs_f64() * 1000.0);
                    }
                    arms.push((switches, best, out.clone()));
                }
            }
            clear_forced_switches();
            let baseline = arms
                .iter()
                .find(|(switches, ..)| {
                    switches.order == RowOrder::TokenMajor && !switches.split_heads
                })
                .map(|(.., out)| out.clone())
                .expect("a baseline arm");
            println!("depth {positions}:");
            for (switches, best, output) in &arms {
                let share = 100.0 * (1.0 - best / arms[0].1);
                println!(
                    "  {:>11} split {:<5} workers {:<4}: {best:6.2} ms ({:.3} ms/layer, {:+.0}% vs token-major/split-off)",
                    if switches.order == RowOrder::TokenMajor {
                        "token-major"
                    } else {
                        "layer-major"
                    },
                    switches.split_heads,
                    switches
                        .workers
                        .map_or_else(|| "all".to_string(), |workers| workers.to_string()),
                    best / layers as f64,
                    if switches.order == RowOrder::TokenMajor && !switches.split_heads {
                        0.0
                    } else {
                        share
                    }
                );
                assert_eq!(
                    output, &baseline,
                    "{:?} must compute what the baseline computes",
                    switches
                );
            }
        }
    }

    /// The write path at a prefill-sized batch: 128 tokens x 28 layers, both orders, one process.
    /// The read side is what the layer-major change is for, so this is the cost it must not add.
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib kv_write_batch_at_128 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: prints a number, run with --ignored --nocapture"]
    fn kv_write_batch_at_128() {
        let layers = 28;
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let row = KvLayout::F16.bytes_per_row(width);
        let bytes_per_token = layers * 2 * row;
        let per_block = 16;
        let batch = 128usize;
        let blocks = batch.div_ceil(per_block);
        let table: Vec<u32> = (0..blocks as u32).collect();
        let k: Vec<f32> = (0..width).map(|at| at as f32 * 0.01 - 1.0).collect();
        let v: Vec<f32> = (0..width).map(|at| 1.0 - at as f32 * 0.01).collect();

        let mut arms = Vec::new();
        for order in [RowOrder::TokenMajor, RowOrder::LayerMajor] {
            let mut storage = vec![0u8; blocks * per_block * bytes_per_token];
            let mut best = f64::INFINITY;
            let mut loaded = (Vec::new(), Vec::new());
            for sweep in 0..4 {
                force_switches(AttentionSwitches {
                    workers: None,
                    split_heads: true,
                    kv_split: false,
                    order,
                });
                let mut kv = KvView {
                    storage: &mut storage,
                    table: &table,
                    tokens_per_block: per_block,
                    bytes_per_token,
                    layout: KvLayout::F16,
                };
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    for position in 0..batch {
                        for layer in 0..layers {
                            kv.write(position, layer, shape, &k, &v).expect("write");
                        }
                    }
                }
                let elapsed = started.elapsed().as_secs_f64() * 100.0;
                if sweep >= 2 {
                    best = best.min(elapsed);
                }
                kv.load_layer(0, shape, batch, &mut loaded.0, &mut loaded.1)
                    .expect("load");
            }
            println!(
                "{order:?}: {best:6.2} ms for {batch} tokens x {layers} layers ({:.3} us/token-layer)",
                best * 1000.0 / (batch * layers) as f64
            );
            arms.push((order, best, loaded));
        }
        clear_forced_switches();
        assert_eq!(
            arms[0].2.0, arms[1].2.0,
            "the two orders must store the same K"
        );
        assert_eq!(
            arms[0].2.1, arms[1].2.1,
            "the two orders must store the same V"
        );
    }

    #[test]
    fn the_row_orders_differ_only_in_addresses() {
        // Two layers of one 4-value row, four tokens per block, three blocks in a permuted table:
        // enough to cross a block boundary and to see the table's indirection.
        let row = KvLayout::F16.bytes_per_row(4);
        let per_block = 4;
        let layers = 2;
        let bytes_per_token = layers * 2 * row;
        let table = [2u32, 0, 1];
        let positions = 10;
        let storage = vec![0u8; 3 * per_block * bytes_per_token];

        let rows = |order: RowOrder, layer: usize| -> Vec<(usize, usize, usize)> {
            let mut out = Vec::new();
            for_each_row_range(
                &storage,
                &table,
                per_block,
                bytes_per_token,
                order,
                layer,
                row,
                0,
                positions,
                |position, k| {
                    out.push((position, k, order.value_row(k, per_block, row)));
                    Ok(())
                },
            )
            .expect("walk");
            out
        };

        let token = rows(RowOrder::TokenMajor, 1);
        let layer_major = rows(RowOrder::LayerMajor, 1);
        assert_eq!(
            token
                .iter()
                .map(|(position, ..)| *position)
                .collect::<Vec<_>>(),
            layer_major
                .iter()
                .map(|(position, ..)| *position)
                .collect::<Vec<_>>(),
            "both orders visit the positions in the same order"
        );
        assert_ne!(
            token.iter().map(|(_, k, _)| *k).collect::<Vec<_>>(),
            layer_major.iter().map(|(_, k, _)| *k).collect::<Vec<_>>(),
            "the switch has to move the rows, or it measures nothing"
        );
        assert_eq!(
            token[1].1 - token[0].1,
            bytes_per_token,
            "token-major stride"
        );
        assert_eq!(
            layer_major[1].1 - layer_major[0].1,
            row,
            "layer-major strides one row, so a block's rows for one layer are contiguous"
        );

        // Every row of every layer, in either order, lands somewhere unique and inside its block:
        // no two K/V rows share bytes, and layer-major fills each block exactly.
        for order in [RowOrder::TokenMajor, RowOrder::LayerMajor] {
            let mut seen = std::collections::BTreeSet::new();
            for layer in 0..layers {
                for (position, k, v) in rows(order, layer) {
                    for (which, start) in [("K", k), ("V", v)] {
                        assert!(
                            seen.insert(start),
                            "{order:?}: layer {layer}, {which} of position {position} overlaps another row"
                        );
                        assert!(
                            start + row <= storage.len(),
                            "{order:?}: row past the buffer"
                        );
                    }
                }
            }
            assert_eq!(
                seen.len(),
                layers * 2 * positions,
                "{order:?}: every row has its own bytes"
            );
        }

        // Layer-major packs each block without gaps: the rows of one block span its whole size.
        let mut block_bytes = Vec::new();
        for layer in 0..layers {
            let start = rows(RowOrder::LayerMajor, layer)[0].1;
            block_bytes.push(start);
        }
        assert_eq!(block_bytes[0] % (per_block * bytes_per_token), 0);
        assert_eq!(block_bytes[1] - block_bytes[0], 2 * row * per_block);
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

    /// Both attention arms on the same decode steps, in one process: the cap is worth ~1 ms of a
    /// ~42 ms step at ctx 1024, which two processes cannot resolve on this machine.
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib decode_arm_switch -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: prints a number, run with --ignored --nocapture"]
    fn decode_arm_switch() {
        let Some(path) = std::env::var_os("VIPERA_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                let path = std::path::PathBuf::from(std::env::var_os("HOME")?)
                    .join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
                path.exists().then_some(path)
            })
        else {
            eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
            return;
        };
        let mut backend = CpuBackend::load(&path).expect("load model");
        let config = backend.config();
        let bytes_per_token = config
            .kv_bytes_per_token_for(KvCacheType::F16)
            .expect("the width is a multiple of 32");
        const BLOCK: u64 = 16;
        const STEPS: usize = 100;
        let ctx: usize = 1024;
        // Two decode steps a round, one per arm, so the sequence grows by `2 * STEPS`.
        let blocks = (ctx + 2 * STEPS + 64).div_ceil(BLOCK as usize);
        let table: Vec<u32> = (0..blocks as u32).collect();
        let mut kv = vec![0u8; blocks * bytes_per_token as usize * BLOCK as usize];
        let prompt: Vec<u32> = (0..ctx).map(|at| 1000 + (at % 400) as u32).collect();
        let mut tokens: Vec<u32> = Vec::new();
        let mut next = {
            let logits = backend
                .logits(&mut ForwardContext {
                    tokens: &tokens,
                    new_tokens: &prompt,
                    first_position: 0,
                    block_table: &table,
                    kv: &mut kv,
                    tokens_per_block: BLOCK,
                    bytes_per_token,
                })
                .expect("prefill");
            argmax(&logits)
        };
        tokens.extend_from_slice(&prompt);

        let order = RowOrder::from_env();
        let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut deltas: Vec<f64> = Vec::new();
        for round in 0..STEPS {
            // Alternate the direction so neither arm always runs on the warmer cache.
            let arms = if round % 2 == 0 { [4, 8] } else { [8, 4] };
            let mut pair = [0.0_f64; 2];
            for workers in arms {
                force_switches(AttentionSwitches {
                    split_heads: true,
                    kv_split: false,
                    order,
                    workers: Some(workers),
                });
                let at = usize::from(workers == 8);
                let started = std::time::Instant::now();
                let logits = backend
                    .logits(&mut ForwardContext {
                        tokens: &tokens,
                        new_tokens: &[next],
                        first_position: tokens.len() as u32,
                        block_table: &table,
                        kv: &mut kv,
                        tokens_per_block: BLOCK,
                        bytes_per_token,
                    })
                    .expect("decode");
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                samples[at].push(elapsed);
                pair[at] = elapsed;
                tokens.push(next);
                next = argmax(&logits);
            }
            deltas.push(pair[1] - pair[0]);
        }
        clear_forced_switches();

        let mut medians = [0.0; 2];
        for (at, values) in samples.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            medians[at] = values[values.len() / 2];
            println!(
                "{} workers: median {:.2} ms, best {:.2} ms, over {} steps",
                if at == 0 { 4 } else { 8 },
                medians[at],
                values[0],
                values.len()
            );
        }
        let slower = deltas.iter().filter(|delta| **delta > 0.0).count();
        let paired = median(&mut deltas);
        println!(
            "8 vs 4 at ctx {ctx}: {:+.2} ms/step ({:+.1}%), paired median {paired:+.2} ms, \
             eight slower in {slower}/{} rounds",
            medians[1] - medians[0],
            100.0 * (medians[1] - medians[0]) / medians[0],
            STEPS
        );

        fn median(values: &mut [f64]) -> f64 {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        }

        fn argmax(values: &[f32]) -> u32 {
            let mut best = (0, f32::NEG_INFINITY);
            for (at, value) in values.iter().enumerate() {
                if *value > best.1 {
                    best = (at, *value);
                }
            }
            best.0 as u32
        }
    }

    /// With sharing on, the KV-chunk decode kernel is within rounding of the serial per-head kernel
    /// it already was within rounding of: whole layers and head ranges that cut a group, worker
    /// counts that give chunks shorter and longer than the sixteen-row blocks, and a block table
    /// that is not in logical order.
    #[test]
    fn shared_kv_chunk_matches_the_serial_kernel_within_rounding() {
        for (heads, kv_heads, head_dim) in [
            (12_usize, 2_usize, 128_usize),
            (6, 2, 64),
            (4, 4, 64),
            (14, 2, 64),
        ] {
            let shape = AttnShape {
                heads,
                kv_heads,
                head_dim,
            };
            let width = kv_heads * head_dim;
            let q: Vec<f32> = (0..heads * head_dim)
                .map(|at| ((at * 13 % 29) as f32 - 14.0) * 0.05)
                .collect();
            let table: Vec<u32> = (0..20_u32).map(|block| block * 7 % 20).collect();
            let mut storage = Vec::new();
            filled_kv(&mut storage, KvLayout::F16, shape, 16, &table, 300);
            let kv = KvView {
                bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
                storage: &mut storage,
                table: &table,
                tokens_per_block: 16,
                layout: KvLayout::F16,
            };
            let read = KvRead::of(&kv);
            let mut ranges = vec![0..heads, 0..1];
            if heads > 2 {
                ranges.push(1..heads - 1);
            }
            for positions in [17_usize, 100, 300] {
                for range in &ranges {
                    let count = range.end - range.start;
                    let mut expected = vec![0.0_f32; count * head_dim];
                    let mut scratch = AttnScratch::default();
                    attention_streaming_heads(
                        &q,
                        &read,
                        RowOrder::LayerMajor,
                        false,
                        0,
                        positions,
                        shape,
                        range.clone(),
                        &mut scratch.scores,
                        &mut scratch.row,
                        &mut scratch.head_state,
                        &mut expected,
                    )
                    .expect("serial attention");
                    let magnitude = expected.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
                    for workers in [2_usize, 3, 4] {
                        let mut shared = vec![f32::NAN; count * head_dim];
                        attention_kv_chunked_heads(
                            &q,
                            &read,
                            RowOrder::LayerMajor,
                            true,
                            0,
                            positions,
                            shape,
                            range.clone(),
                            &mut shared,
                            workers,
                        )
                        .expect("shared chunked attention");
                        for (at, (shared, serial)) in shared.iter().zip(&expected).enumerate() {
                            assert!(
                                (shared - serial).abs() <= 1e-5 * magnitude.max(1.0),
                                "{heads}/{kv_heads}x{head_dim} positions {positions} heads {range:?} workers {workers} element {at}: {shared} against {serial}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// One layer of single-entry decode attention through the KV-chunk kernel, per-head against
    /// shared groups, on the Qwen2.5-1.5B shape. Both arms run in one process, interleaved, with the
    /// same worker count; with fewer cores than workers the arms time-slice, so the ratio measures
    /// the work and not the parallel speedup. It is a measurement, not a check, so it is
    /// `#[ignore]`d:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib attention_kv_chunk_share_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn attention_kv_chunk_share_cost() {
        use std::time::Instant;
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 13 % 29) as f32 - 14.0) * 0.05)
            .collect();
        let workers = 4;
        for positions in [1024_usize, 4096, 8192] {
            let table: Vec<u32> = (0..positions.div_ceil(16) as u32).collect();
            let mut storage = Vec::new();
            filled_kv(&mut storage, KvLayout::F16, shape, 16, &table, positions);
            let width = shape.kv_heads * shape.head_dim;
            let kv = KvView {
                bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
                storage: &mut storage,
                table: &table,
                tokens_per_block: 16,
                layout: KvLayout::F16,
            };
            let read = KvRead::of(&kv);
            let calls = (4_000_000 / positions).max(4);
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            let mut best = [f64::INFINITY; 2];
            for _ in 0..7 {
                for (arm, slot) in best.iter_mut().enumerate() {
                    let started = Instant::now();
                    for _ in 0..calls {
                        attention_kv_chunked_heads(
                            &q,
                            &read,
                            RowOrder::LayerMajor,
                            arm == 1,
                            0,
                            positions,
                            shape,
                            0..shape.heads,
                            &mut out,
                            workers,
                        )
                        .expect("chunked attention");
                        std::hint::black_box(&out);
                    }
                    *slot = slot.min(started.elapsed().as_secs_f64() * 1e3 / calls as f64);
                }
            }
            println!(
                "positions {positions:>5}: per-head {:.3} ms, shared {:.3} ms per layer, {workers} workers ({:.2}x)",
                best[0],
                best[1],
                best[0] / best[1]
            );
        }
    }

    /// The KV-chunk arm regroups the softmax sums, so it is held to a tight tolerance against the
    /// serial kernel instead of bit equality; one worker must take the serial kernel unchanged.
    #[test]
    fn kv_chunk_split_matches_the_serial_kernel_within_rounding() {
        let shape = AttnShape {
            heads: 12,
            kv_heads: 2,
            head_dim: 128,
        };
        let width = shape.kv_heads * shape.head_dim;
        let tokens_per_block = 16;
        let positions = 1024_usize;
        let table: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            positions,
        );
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            layout: KvLayout::F16,
        };
        let read = KvRead::of(&kv);
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 13 % 29) as f32 - 14.0) * 0.05)
            .collect();

        let mut expected = vec![0.0_f32; shape.heads * shape.head_dim];
        let mut scratch = AttnScratch::default();
        attend_entry(
            AttentionCall {
                q: &q,
                kv: &read,
                layer: 0,
                positions,
                shape,
                heads: 0..shape.heads,
                out: &mut expected,
            },
            &mut scratch,
        )
        .expect("serial attention");

        let mut chunked = vec![0.0_f32; shape.heads * shape.head_dim];
        attention_kv_chunked_heads(
            &q,
            &read,
            RowOrder::LayerMajor,
            false,
            0,
            positions,
            shape,
            0..shape.heads,
            &mut chunked,
            4,
        )
        .expect("chunked attention");

        let magnitude = expected.iter().fold(0.0_f32, |max, v| max.max(v.abs()));
        for (serial, split) in expected.iter().zip(chunked.iter()) {
            assert!(
                (serial - split).abs() <= 1e-5 * magnitude.max(1.0),
                "chunked {split} is not within rounding of serial {serial}"
            );
        }

        let mut one = vec![0.0_f32; shape.heads * shape.head_dim];
        // One worker must fall through to the serial kernel the caller itself would pick, so it
        // takes the sharing switch from the same place [`attend_entry`] does.
        attention_kv_chunked_heads(
            &q,
            &read,
            RowOrder::LayerMajor,
            gqa_share_enabled(),
            0,
            positions,
            shape,
            0..shape.heads,
            &mut one,
            1,
        )
        .expect("one-worker attention");
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&one),
            bits(&expected),
            "one worker must fall back to the serial kernel"
        );
    }

    /// Pairs the head split against the KV chunk on the same decode step, alternating the arm order
    /// every round so neither arm always runs on the warmer cache. Needs a model:
    /// `cargo test --release -p vipera-backend-cpu --lib decode_kv_split_switch -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement: needs a model"]
    fn decode_kv_split_switch() {
        let Some(path) = std::env::var_os("VIPERA_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                let path = std::path::PathBuf::from(std::env::var_os("HOME")?)
                    .join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
                path.exists().then_some(path)
            })
        else {
            eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
            return;
        };
        let mut backend = CpuBackend::load(&path).expect("load model");
        let config = backend.config();
        let bytes_per_token = config
            .kv_bytes_per_token_for(KvCacheType::F16)
            .expect("the width is a multiple of 32");
        const BLOCK: u64 = 16;
        const STEPS: usize = 60;
        let order = RowOrder::from_env();

        for ctx in [1024usize, 4096] {
            let blocks = (ctx + 2 * STEPS + 64).div_ceil(BLOCK as usize);
            let table: Vec<u32> = (0..blocks as u32).collect();
            let mut kv = vec![0u8; blocks * bytes_per_token as usize * BLOCK as usize];
            let prompt: Vec<u32> = (0..ctx).map(|at| 1000 + (at % 400) as u32).collect();
            let mut tokens: Vec<u32> = Vec::new();
            let mut next = {
                let logits = backend
                    .logits(&mut ForwardContext {
                        tokens: &tokens,
                        new_tokens: &prompt,
                        first_position: 0,
                        block_table: &table,
                        kv: &mut kv,
                        tokens_per_block: BLOCK,
                        bytes_per_token,
                    })
                    .expect("prefill");
                argmax_t(&logits)
            };
            tokens.extend_from_slice(&prompt);

            let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut deltas: Vec<f64> = Vec::new();
            for round in 0..STEPS {
                let arms = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                let mut pair = [0.0_f64; 2];
                for kv_split in arms {
                    force_switches(AttentionSwitches {
                        split_heads: true,
                        kv_split,
                        order,
                        workers: Some(4),
                    });
                    let at = usize::from(kv_split);
                    let started = std::time::Instant::now();
                    let logits = backend
                        .logits(&mut ForwardContext {
                            tokens: &tokens,
                            new_tokens: &[next],
                            first_position: tokens.len() as u32,
                            block_table: &table,
                            kv: &mut kv,
                            tokens_per_block: BLOCK,
                            bytes_per_token,
                        })
                        .expect("decode");
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    samples[at].push(elapsed);
                    pair[at] = elapsed;
                    tokens.push(next);
                    next = argmax_t(&logits);
                }
                deltas.push(pair[1] - pair[0]);
            }
            clear_forced_switches();

            let mut medians = [0.0; 2];
            for (at, values) in samples.iter_mut().enumerate() {
                values.sort_by(f64::total_cmp);
                medians[at] = values[values.len() / 2];
            }
            let slower = deltas.iter().filter(|delta| **delta > 0.0).count();
            deltas.sort_by(f64::total_cmp);
            let paired = deltas[deltas.len() / 2];
            println!(
                "ctx {ctx}: head-split {:.2} ms/step, kv-split {:.2} ms/step ({:+.1}%), paired median {paired:+.2} ms, kv-split slower in {slower}/{STEPS} rounds",
                medians[0],
                medians[1],
                100.0 * (medians[1] - medians[0]) / medians[0],
            );
        }
    }

    /// The window kernel must compute attention over exactly the positions it keeps: the sinks and
    /// the last `window`, in increasing order, through one softmax and one weighted sum of V.
    #[test]
    fn the_window_kernel_matches_attention_over_the_positions_it_keeps() {
        let shape = AttnShape {
            heads: 6,
            kv_heads: 2,
            head_dim: 64,
        };
        let width = shape.kv_heads * shape.head_dim;
        let positions = 200usize;
        let sinks = 4usize;
        let window = 32usize;
        let tokens_per_block = 16;
        let table: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();
        let mut storage = Vec::new();
        filled_kv(
            &mut storage,
            KvLayout::F16,
            shape,
            tokens_per_block,
            &table,
            positions,
        );
        let kv = KvView {
            bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
            storage: &mut storage,
            table: &table,
            tokens_per_block,
            layout: KvLayout::F16,
        };
        let read = KvRead::of(&kv);
        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        let mut got = vec![0.0_f32; shape.heads * shape.head_dim];
        attention_window_heads(
            &q,
            &read,
            RowOrder::from_env(),
            false,
            0,
            positions,
            sinks,
            window,
            shape,
            0..shape.heads,
            &mut AttnScratch::default(),
            &mut got,
        )
        .expect("window kernel");

        // The reference decodes the layer and sums the kept positions itself.
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        read.load_layer(0, shape, positions, &mut k_all, &mut v_all)
            .expect("load layer");
        let keep: Vec<usize> = (0..sinks).chain(positions - window..positions).collect();
        let scale = 1.0 / (shape.head_dim as f32).sqrt();
        let group = shape.heads / shape.kv_heads;
        let mut want = vec![0.0_f32; shape.heads * shape.head_dim];
        for head in 0..shape.heads {
            let kv_head = head / group;
            let query = &q[head * shape.head_dim..(head + 1) * shape.head_dim];
            let mut scores: Vec<f32> = keep
                .iter()
                .map(|position| {
                    let base = (position * shape.kv_heads + kv_head) * shape.head_dim;
                    let dot: f32 = query
                        .iter()
                        .zip(&k_all[base..base + shape.head_dim])
                        .map(|(q, k)| q * k)
                        .sum();
                    dot * scale
                })
                .collect();
            let best = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0_f32;
            for score in scores.iter_mut() {
                *score = (*score - best).exp();
                total += *score;
            }
            let target = &mut want[head * shape.head_dim..(head + 1) * shape.head_dim];
            for (position, weight) in keep.iter().zip(&scores) {
                let base = (position * shape.kv_heads + kv_head) * shape.head_dim;
                for (target, value) in target.iter_mut().zip(&v_all[base..base + shape.head_dim]) {
                    *target += weight / total * value;
                }
            }
        }
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst <= 1e-5, "the window kernel differs by {worst}");

        // The chunked kernel is what a decode step runs, and it sums by chunk instead of strictly
        // by position, so it is checked against the same reference at worker counts that do and do
        // not split the sinks from the window: 1, 3 (chunk boundary inside the gap), and 8.
        let ranges = [(0, sinks), (positions - window, positions)];
        for workers in [1, 3, 8] {
            let mut chunked = vec![0.0_f32; shape.heads * shape.head_dim];
            attention_window_chunked_heads(
                &q,
                &read,
                RowOrder::from_env(),
                false,
                0,
                &ranges,
                shape,
                0..shape.heads,
                &mut chunked,
                workers,
            )
            .expect("chunked window kernel");
            let worst = chunked
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst <= 1e-4,
                "the chunked window kernel with {workers} workers differs by {worst}"
            );
        }
    }

    /// A ring KV keeps only the sinks and the window, so the block table points dropped positions
    /// at the same physical slots as the positions that replaced them. The kernel must not notice:
    /// what it reads for a kept position is that position's row, and what it computes must equal the
    /// same attention over a cache that still held every position.
    #[test]
    fn the_window_kernel_reads_a_ring_table_like_a_linear_one() {
        let shape = AttnShape {
            heads: 6,
            kv_heads: 2,
            head_dim: 64,
        };
        let width = shape.kv_heads * shape.head_dim;
        let positions = 200usize;
        let sinks = 4usize;
        let window = 32usize;
        let tokens_per_block = 16usize;
        // One pinned block holds the sinks; four rolling blocks cover 64 positions, twice the
        // window, so the window's own blocks are the ones that wrote last into each physical block.
        let pinned_blocks = 1usize;
        let rolling_blocks = 4usize;
        let ring: Vec<u32> = (0..positions.div_ceil(tokens_per_block))
            .map(|block| {
                if block < pinned_blocks {
                    block as u32
                } else {
                    (pinned_blocks + (block - pinned_blocks) % rolling_blocks) as u32
                }
            })
            .collect();
        let linear: Vec<u32> = (0..positions.div_ceil(tokens_per_block) as u32).collect();

        let q: Vec<f32> = (0..shape.heads * shape.head_dim)
            .map(|at| ((at * 7 % 23) as f32 - 11.0) * 0.05)
            .collect();
        let run = |table: &[u32]| {
            let mut storage = Vec::new();
            filled_kv(
                &mut storage,
                KvLayout::F16,
                shape,
                tokens_per_block,
                table,
                positions,
            );
            let kv = KvView {
                bytes_per_token: 2 * 2 * KvLayout::F16.bytes_per_row(width),
                storage: &mut storage,
                table,
                tokens_per_block,
                layout: KvLayout::F16,
            };
            let read = KvRead::of(&kv);
            let mut out = vec![0.0_f32; shape.heads * shape.head_dim];
            attention_window_heads(
                &q,
                &read,
                RowOrder::from_env(),
                false,
                0,
                positions,
                sinks,
                window,
                shape,
                0..shape.heads,
                &mut AttnScratch::default(),
                &mut out,
            )
            .expect("window kernel over a ring table");
            out
        };
        let want = run(&linear);
        let got = run(&ring);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst <= 1e-5,
            "a ring table changed the kept positions by {worst}"
        );
    }

    /// With sharing on, both window kernels are within rounding of their per-head selves: whole
    /// layers and head ranges that cut a group, kept-position counts around the sixteen-row value
    /// blocks, and worker counts that give chunks shorter and longer than a block. `q8_0` rows are
    /// not read in place, so they keep the per-head kernel bit for bit whatever the switch says.
    #[test]
    fn shared_window_matches_the_per_head_kernel_within_rounding() {
        for (heads, kv_heads, head_dim) in [
            (6_usize, 2_usize, 64_usize),
            (12, 2, 128),
            (14, 2, 64),
            (4, 4, 64),
        ] {
            let shape = AttnShape {
                heads,
                kv_heads,
                head_dim,
            };
            let width = kv_heads * head_dim;
            let q: Vec<f32> = (0..heads * head_dim)
                .map(|at| ((at * 11 % 31) as f32 - 15.0) * 0.05)
                .collect();
            let table: Vec<u32> = (0..20_u32).map(|block| block * 7 % 20).collect();
            let sinks = 4usize;
            let mut ranges_at: Vec<std::ops::Range<usize>> = vec![0..heads, 0..1];
            if heads > 2 {
                ranges_at.push(1..heads - 1);
            }
            for layout in [KvLayout::F16, KvLayout::Q8_0] {
                let mut storage = Vec::new();
                filled_kv(&mut storage, layout, shape, 16, &table, 300);
                let kv = KvView {
                    bytes_per_token: 2 * 2 * layout.bytes_per_row(width),
                    storage: &mut storage,
                    table: &table,
                    tokens_per_block: 16,
                    layout,
                };
                let read = KvRead::of(&kv);
                for window in [8usize, 15, 16, 17, 40] {
                    let positions = sinks + window;
                    let ranges = [(0, sinks.min(positions)), (positions - window, positions)];
                    for range in &ranges_at {
                        let count = range.end - range.start;
                        let mut exact = vec![0.0_f32; count * head_dim];
                        let mut shared = vec![0.0_f32; count * head_dim];
                        attention_window_heads(
                            &q,
                            &read,
                            RowOrder::LayerMajor,
                            false,
                            0,
                            positions,
                            sinks,
                            window,
                            shape,
                            range.clone(),
                            &mut AttnScratch::default(),
                            &mut exact,
                        )
                        .expect("per-head window kernel");
                        attention_window_heads(
                            &q,
                            &read,
                            RowOrder::LayerMajor,
                            true,
                            0,
                            positions,
                            sinks,
                            window,
                            shape,
                            range.clone(),
                            &mut AttnScratch::default(),
                            &mut shared,
                        )
                        .expect("shared window kernel");
                        for (at, (shared, exact)) in shared.iter().zip(&exact).enumerate() {
                            let context = format!(
                                "two-pass {layout:?} {heads}/{kv_heads}x{head_dim} positions {positions} window {window} heads {range:?} element {at}"
                            );
                            if layout == KvLayout::F16 {
                                assert!(
                                    (shared - exact).abs() <= 1e-5 + 1e-5 * exact.abs(),
                                    "{context}: {shared} against {exact}"
                                );
                            } else {
                                assert_eq!(shared.to_bits(), exact.to_bits(), "{context}");
                            }
                        }
                        // The chunked kernel already regroups the sums, so it is compared with
                        // its own per-head arm rather than with the two-pass one.
                        for workers in [2usize, 3, 4] {
                            let mut chunked_exact = vec![0.0_f32; count * head_dim];
                            attention_window_chunked_heads(
                                &q,
                                &read,
                                RowOrder::LayerMajor,
                                false,
                                0,
                                &ranges,
                                shape,
                                range.clone(),
                                &mut chunked_exact,
                                workers,
                            )
                            .expect("per-head chunked window kernel");
                            let mut chunked = vec![0.0_f32; count * head_dim];
                            attention_window_chunked_heads(
                                &q,
                                &read,
                                RowOrder::LayerMajor,
                                true,
                                0,
                                &ranges,
                                shape,
                                range.clone(),
                                &mut chunked,
                                workers,
                            )
                            .expect("shared chunked window kernel");
                            for (at, (chunked, chunked_exact)) in
                                chunked.iter().zip(&chunked_exact).enumerate()
                            {
                                let context = format!(
                                    "chunked {layout:?} {heads}/{kv_heads}x{head_dim} positions {positions} window {window} heads {range:?} workers {workers} element {at}"
                                );
                                if layout == KvLayout::F16 {
                                    assert!(
                                        (chunked - chunked_exact).abs()
                                            <= 1e-5 + 1e-5 * chunked_exact.abs(),
                                        "{context}: {chunked} against {chunked_exact}"
                                    );
                                } else {
                                    assert_eq!(
                                        chunked.to_bits(),
                                        chunked_exact.to_bits(),
                                        "{context}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The model the end-to-end measurements run on, or `None` (after saying so) without one.
    fn measurement_backend() -> Option<CpuBackend> {
        let path = std::env::var_os("VIPERA_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .filter(|path| path.exists())
            .or_else(|| {
                let path = std::path::PathBuf::from(std::env::var_os("HOME")?)
                    .join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
                path.exists().then_some(path)
            });
        let Some(path) = path else {
            eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
            return None;
        };
        Some(CpuBackend::load(&path).expect("load model"))
    }

    fn median_of(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    /// Pairs the per-head attention kernels against the shared-group ones on the same decode step,
    /// the KV chunk and four workers on in both arms, alternating the arm order every round so
    /// neither arm always runs on the warmer cache. The switch is forced per call instead of read
    /// from the environment, so one process times both. Needs a model:
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib decode_gqa_share_switch -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: needs a model"]
    fn decode_gqa_share_switch() {
        let Some(mut backend) = measurement_backend() else {
            return;
        };
        let config = backend.config();
        let bytes_per_token = config
            .kv_bytes_per_token_for(KvCacheType::F16)
            .expect("the width is a multiple of 32");
        const BLOCK: u64 = 16;
        const STEPS: usize = 60;
        force_switches(AttentionSwitches {
            split_heads: true,
            kv_split: true,
            order: RowOrder::from_env(),
            workers: Some(4),
        });
        for ctx in [1024usize, 4096] {
            let blocks = (ctx + 2 * STEPS + 64).div_ceil(BLOCK as usize);
            let table: Vec<u32> = (0..blocks as u32).collect();
            let mut kv = vec![0u8; blocks * bytes_per_token as usize * BLOCK as usize];
            let prompt: Vec<u32> = (0..ctx).map(|at| 1000 + (at % 400) as u32).collect();
            let mut tokens: Vec<u32> = Vec::new();
            force_share(Some(false));
            let mut next = {
                let logits = backend
                    .logits(&mut ForwardContext {
                        tokens: &tokens,
                        new_tokens: &prompt,
                        first_position: 0,
                        block_table: &table,
                        kv: &mut kv,
                        tokens_per_block: BLOCK,
                        bytes_per_token,
                    })
                    .expect("prefill");
                argmax_t(&logits)
            };
            tokens.extend_from_slice(&prompt);

            let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut deltas: Vec<f64> = Vec::new();
            for round in 0..STEPS {
                let arms = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                let mut pair = [0.0_f64; 2];
                for share in arms {
                    force_share(Some(share));
                    let at = usize::from(share);
                    let started = std::time::Instant::now();
                    let logits = backend
                        .logits(&mut ForwardContext {
                            tokens: &tokens,
                            new_tokens: &[next],
                            first_position: tokens.len() as u32,
                            block_table: &table,
                            kv: &mut kv,
                            tokens_per_block: BLOCK,
                            bytes_per_token,
                        })
                        .expect("decode");
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    samples[at].push(elapsed);
                    pair[at] = elapsed;
                    tokens.push(next);
                    next = argmax_t(&logits);
                }
                deltas.push(pair[1] - pair[0]);
            }
            let medians = [median_of(&mut samples[0]), median_of(&mut samples[1])];
            let slower = deltas.iter().filter(|delta| **delta > 0.0).count();
            let paired = median_of(&mut deltas);
            println!(
                "ctx {ctx}: per-head {:.2} ms/step, shared {:.2} ms/step ({:+.1}%), paired median {paired:+.2} ms, shared slower in {slower}/{STEPS} rounds",
                medians[0],
                medians[1],
                100.0 * (medians[1] - medians[0]) / medians[0],
            );
        }
        force_share(None);
        clear_forced_switches();
    }

    /// Times the same prompt prefilled with the per-head and the shared-group attention kernels,
    /// alternating the arm order every round. A 4096-token prefill takes about a minute, so the
    /// rounds are few. Needs a model:
    ///
    /// ```text
    /// cargo test --release -p vipera-backend-cpu --lib prefill_gqa_share_switch -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement: needs a model"]
    fn prefill_gqa_share_switch() {
        let Some(mut backend) = measurement_backend() else {
            return;
        };
        let config = backend.config();
        let bytes_per_token = config
            .kv_bytes_per_token_for(KvCacheType::F16)
            .expect("the width is a multiple of 32");
        const BLOCK: u64 = 16;
        for (ctx, rounds) in [(1024usize, 3usize), (4096, 2)] {
            let blocks = (ctx + 64).div_ceil(BLOCK as usize);
            let table: Vec<u32> = (0..blocks as u32).collect();
            let mut kv = vec![0u8; blocks * bytes_per_token as usize * BLOCK as usize];
            let prompt: Vec<u32> = (0..ctx).map(|at| 1000 + (at % 400) as u32).collect();
            let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for round in 0..rounds {
                let arms = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for share in arms {
                    force_share(Some(share));
                    let started = std::time::Instant::now();
                    backend
                        .logits(&mut ForwardContext {
                            tokens: &[],
                            new_tokens: &prompt,
                            first_position: 0,
                            block_table: &table,
                            kv: &mut kv,
                            tokens_per_block: BLOCK,
                            bytes_per_token,
                        })
                        .expect("prefill");
                    samples[usize::from(share)].push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            let best = [
                samples[0].iter().copied().fold(f64::INFINITY, f64::min),
                samples[1].iter().copied().fold(f64::INFINITY, f64::min),
            ];
            println!(
                "prefill {ctx}: per-head {:.0} ms ({:.1} tok/s), shared {:.0} ms ({:.1} tok/s), {:+.1}% (best of {rounds})",
                best[0],
                ctx as f64 / (best[0] / 1000.0),
                best[1],
                ctx as f64 / (best[1] / 1000.0),
                100.0 * (best[1] - best[0]) / best[0],
            );
        }
        force_share(None);
    }

    fn argmax_t(values: &[f32]) -> u32 {
        let mut best = (0, f32::NEG_INFINITY);
        for (at, value) in values.iter().enumerate() {
            if *value > best.1 {
                best = (at, *value);
            }
        }
        best.0 as u32
    }
}
