//! The whole model on one device: embedding lookup, every layer, the final norm, the output
//! projection.
//!
//! Each layer is a [`GpuLayer`], so the weights are read once per token and the residual row never
//! leaves the card until logits are asked for. The KV cache is the layer's own device buffer, which is
//! why this is not (yet) a `vipera_runtime::ForwardPass`: that trait hands a backend the runtime's
//! host KV buffer and expects it written in the runtime's block layout (`KvLayout::encode`, block
//! table, row order), and mirroring that is a separate piece of work from making the arithmetic fast.
//!
//! Prefill is sequential here: one position at a time, which costs a full weight read per token. The
//! batched kernel exists (`q4k_matvec_q8_batch`, measured below) but wiring it through the layer needs
//! per-row norms, RoPE and a batched attention pass.

use std::path::Path;

use vipera_core::gguf::{GgmlType, GgufFile, Value};
use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};
use crate::layer::{GpuLayer, LayerShape, LayerWeights, Packed};
use crate::q4k::Q4kMatrix;
use crate::q6k::Q6kMatrix;

/// A model loaded for the device.
pub struct GpuModel {
    device: Device,
    shape: LayerShape,
    vocab: usize,
    tok_embd: Q4kMatrix,
    output: Q6kMatrix,
    output_norm: DeviceBuffer<f32>,
    layers: Vec<GpuLayer>,
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
}

fn tensor_bytes<'a>(
    file: &GgufFile,
    bytes: &'a [u8],
    name: &str,
) -> Result<(&'a [u8], GgmlType), ViperaError> {
    let info = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| ViperaError::invalid_argument(format!("the model has no {name}")))?;
    let start = file.data_offset as usize + info.offset as usize;
    let len = info
        .byte_len()
        .ok_or_else(|| ViperaError::invalid_argument(format!("{name} has no length")))?
        as usize;
    Ok((&bytes[start..start + len], info.ty))
}

fn f32_tensor(file: &GgufFile, bytes: &[u8], name: &str) -> Result<Vec<f32>, ViperaError> {
    let (raw, ty) = tensor_bytes(file, bytes, name)?;
    match ty {
        GgmlType::F32 => Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()),
        GgmlType::F16 => Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| crate::layer::f16(u16::from_le_bytes(*chunk)))
            .collect()),
        other => Err(ViperaError::unsupported(format!(
            "{name} is {other:?}, not a float vector"
        ))),
    }
}

fn packed<'a>(
    file: &GgufFile,
    bytes: &'a [u8],
    name: &str,
) -> Result<(Packed<'a>, usize, usize), ViperaError> {
    let info = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| ViperaError::invalid_argument(format!("the model has no {name}")))?;
    let start = file.data_offset as usize + info.offset as usize;
    let len = info
        .byte_len()
        .ok_or_else(|| ViperaError::invalid_argument(format!("{name} has no length")))?
        as usize;
    let raw = &bytes[start..start + len];
    let cols = *info
        .dims
        .first()
        .ok_or_else(|| ViperaError::invalid_argument(format!("{name} has no dims")))?
        as usize;
    let block = match info.ty {
        GgmlType::Q4K => crate::q4k::BLOCK_BYTES,
        GgmlType::Q6K => crate::q6k::BLOCK_BYTES,
        other => {
            return Err(ViperaError::unsupported(format!(
                "{name} is {other:?}; the device path knows Q4_K and Q6_K"
            )));
        }
    };
    let rows = len / (cols / 256) / block;
    let packed = match info.ty {
        GgmlType::Q4K => Packed::Q4(raw),
        _ => Packed::Q6(raw),
    };
    Ok((packed, rows, cols))
}

fn meta_usize(file: &GgufFile, key: &str, fallback: usize) -> usize {
    file.metadata
        .get(key)
        .and_then(Value::as_u64)
        .map_or(fallback, |value| value as usize)
}

fn meta_f32(file: &GgufFile, key: &str, fallback: f32) -> f32 {
    file.metadata
        .get(key)
        .and_then(Value::as_f32)
        .or_else(|| {
            file.metadata
                .get(key)
                .and_then(Value::as_u64)
                .map(|value| value as f32)
        })
        .unwrap_or(fallback)
}

impl GpuModel {
    /// Read a GGUF file and copy the tensors the device path knows onto the card.
    ///
    /// Sized for the 512 positions the measurements use; [`Self::load_with_positions`] is what a
    /// runtime that plans a context asks for.
    pub fn load(device: Device, path: &Path) -> Result<Self, ViperaError> {
        Self::load_with_positions(device, path, 512)
    }

    /// The same load with a KV cache sized to `positions`.
    pub fn load_with_positions(
        device: Device,
        path: &Path,
        positions: usize,
    ) -> Result<Self, ViperaError> {
        if positions == 0 {
            return Err(ViperaError::invalid_argument(
                "the device KV cache needs at least one position",
            ));
        }
        let bytes = std::fs::read(path)?;
        let file = GgufFile::parse(&bytes)?;
        let arch = file
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("qwen2")
            .to_owned();
        let key = |name: &str| format!("{arch}.{name}");
        let hidden = meta_usize(&file, &key("embedding_length"), 1536);
        let heads = meta_usize(&file, &key("attention.head_count"), 12);
        let shape = LayerShape {
            hidden,
            ffn: meta_usize(&file, &key("feed_forward_length"), 8960),
            heads,
            kv_heads: meta_usize(&file, &key("attention.head_count_kv"), 2),
            head_dim: hidden / heads.max(1),
            rope_base: meta_f32(&file, &key("rope.freq_base"), 1_000_000.0),
            epsilon: meta_f32(&file, &key("attention.layer_norm_rms_epsilon"), 1e-6),
            max_positions: positions,
        };
        let layers = meta_usize(&file, &key("block_count"), 28);
        let vocab = meta_usize(&file, "tokenizer.ggml.tokens", 0);

        let (embed_bytes, embed_ty) = tensor_bytes(&file, &bytes, "token_embd.weight")?;
        if embed_ty != GgmlType::Q4K {
            return Err(ViperaError::unsupported(
                "the device path needs a Q4_K token embedding",
            ));
        }
        let vocab = if vocab == 0 {
            embed_bytes.len() / (hidden / 256) / crate::q4k::BLOCK_BYTES
        } else {
            vocab
        };
        let tok_embd = Q4kMatrix::upload(&device, embed_bytes, vocab, hidden)?;
        let (output_bytes, _) = tensor_bytes(&file, &bytes, "output.weight")?;
        let output = Q6kMatrix::upload(&device, output_bytes, vocab, hidden)?;
        let output_norm = device.upload(&f32_tensor(&file, &bytes, "output_norm.weight")?)?;

        let mut built = Vec::with_capacity(layers);
        for index in 0..layers {
            let name = |tensor: &str| format!("blk.{index}.{tensor}");
            let matrices = [
                packed(&file, &bytes, &name("attn_q.weight"))?.0,
                packed(&file, &bytes, &name("attn_k.weight"))?.0,
                packed(&file, &bytes, &name("attn_v.weight"))?.0,
                packed(&file, &bytes, &name("attn_output.weight"))?.0,
                packed(&file, &bytes, &name("ffn_gate.weight"))?.0,
                packed(&file, &bytes, &name("ffn_up.weight"))?.0,
                packed(&file, &bytes, &name("ffn_down.weight"))?.0,
            ];
            let attn_norm = f32_tensor(&file, &bytes, &name("attn_norm.weight"))?;
            let ffn_norm = f32_tensor(&file, &bytes, &name("ffn_norm.weight"))?;
            let bias = |tensor: &str| {
                if file.tensors.iter().any(|info| info.name == name(tensor)) {
                    f32_tensor(&file, &bytes, &name(tensor)).ok()
                } else {
                    None
                }
            };
            let biases = [
                bias("attn_q.bias"),
                bias("attn_k.bias"),
                bias("attn_v.bias"),
            ];
            built.push(GpuLayer::upload(
                &device,
                shape,
                LayerWeights {
                    attn_norm: &attn_norm,
                    ffn_norm: &ffn_norm,
                    matrices,
                    biases: [
                        biases[0].as_deref(),
                        biases[1].as_deref(),
                        biases[2].as_deref(),
                    ],
                },
            )?);
        }

        Ok(Self {
            hidden: device.alloc(hidden)?,
            normed: device.alloc(hidden)?,
            logits: device.alloc(vocab)?,
            tok_embd,
            output,
            output_norm,
            layers: built,
            shape,
            vocab,
            device,
        })
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Hidden width.
    pub fn hidden(&self) -> usize {
        self.shape.hidden
    }

    /// Vocabulary size.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Layers loaded.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Positions the device KV cache holds.
    pub fn kv_positions(&self) -> usize {
        self.shape.max_positions
    }

    /// Values in one K or V row, `kv_heads * head_dim`.
    pub fn kv_width(&self) -> usize {
        self.shape.kv_width()
    }

    /// Copy `count` rows from layer `layer`'s caches into two host vectors.
    ///
    /// This is the write-back half of the paged seam: the runtime's buffer is what a prefix-cache
    /// copy moves, so every row a step computed has to reach it.
    pub fn download_kv(
        &self,
        layer: usize,
        first: usize,
        count: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), ViperaError> {
        let width = self.shape.kv_width();
        let rows = self
            .layers
            .get(layer)
            .ok_or_else(|| ViperaError::internal(format!("no layer {layer}")))?;
        if first
            .checked_add(count)
            .is_none_or(|end| end > self.kv_positions())
        {
            return Err(ViperaError::invalid_argument(format!(
                "rows {first}..{} are past the cache of {}",
                first + count,
                self.kv_positions()
            )));
        }
        let mut k = vec![0f32; count * width];
        let mut v = vec![0f32; count * width];
        self.device
            .download_at(rows.k_cache(), first * width, &mut k)?;
        self.device
            .download_at(rows.v_cache(), first * width, &mut v)?;
        Ok((k, v))
    }

    /// Copy host rows into layer `layer`'s caches, starting at position `first`.
    pub fn upload_kv(
        &mut self,
        layer: usize,
        first: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), ViperaError> {
        let width = self.shape.kv_width();
        let positions = self.shape.max_positions;
        if k.len() != v.len() || !k.len().is_multiple_of(width) {
            return Err(ViperaError::invalid_argument(
                "KV rows must be two equal vectors of whole rows",
            ));
        }
        if first
            .checked_add(k.len() / width)
            .is_none_or(|end| end > positions)
        {
            return Err(ViperaError::invalid_argument(format!(
                "{} rows at {first} are past the cache of {positions}",
                k.len() / width,
            )));
        }
        let rows = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| ViperaError::internal(format!("no layer {layer}")))?;
        let (k_cache, v_cache) = rows.kv_caches_mut();
        self.device.upload_at(k, k_cache, first * width)?;
        self.device.upload_at(v, v_cache, first * width)
    }

    /// Run `token` at `position` and return its logits.
    pub fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>, ViperaError> {
        let row = token as usize;
        if row >= self.vocab {
            return Err(ViperaError::invalid_argument(format!(
                "token {token} is outside the vocabulary of {}",
                self.vocab
            )));
        }
        let shape = self.shape;
        let view = self.tok_embd.row_view(row, shape.hidden)?;
        self.device.q4k_row(&view, shape.hidden, &mut self.hidden)?;

        for layer in &mut self.layers {
            layer.load_residual(&self.device, &self.hidden)?;
            layer.step_device(&self.device, position)?;
            self.device
                .copy_buffer(layer.residual(), &mut self.hidden)?;
        }
        self.device.rmsnorm(
            &self.hidden,
            &self.output_norm,
            &mut self.normed,
            shape.epsilon,
        )?;
        self.output
            .matvec_into(&self.device, &self.normed, &mut self.logits)?;
        self.logits.download(&self.device)
    }
}
