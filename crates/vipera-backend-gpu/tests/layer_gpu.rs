//! One decoder layer, all of it on the device, timed against the CPU's share of a decode step.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-gpu --release --test layer_gpu -- --ignored --nocapture
//! ```

use std::time::Instant;

use vipera_backend_gpu::Device;
use vipera_backend_gpu::layer::{GpuLayer, LayerShape, LayerWeights, Packed};
use vipera_core::gguf::{GgmlType, GgufFile, Value};

fn model_path() -> String {
    std::env::var("VIPERA_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            std::env::var("HOME").unwrap_or_default()
        )
    })
}

fn f32s(file: &GgufFile, bytes: &[u8], name: &str) -> Vec<f32> {
    let info = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    assert!(
        matches!(info.ty, GgmlType::F32 | GgmlType::F16),
        "{name} is {:?}",
        info.ty
    );
    let start = file.data_offset as usize + info.offset as usize;
    let len = info.byte_len().expect("length") as usize;
    let raw = &bytes[start..start + len];
    match info.ty {
        GgmlType::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        _ => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| {
                let h = u16::from_le_bytes([c[0], c[1]]);
                let sign = (h >> 15) & 1;
                let exp = ((h >> 10) & 0x1f) as i32;
                let man = (h & 0x3ff) as f32;
                let v = if exp == 0 {
                    man * 2f32.powi(-24)
                } else if exp == 31 {
                    if man == 0.0 { f32::INFINITY } else { f32::NAN }
                } else {
                    (man + 1024.0) * 2f32.powi(exp - 25)
                };
                if sign == 1 { -v } else { v }
            })
            .collect(),
    }
}

fn quantized<'a>(file: &GgufFile, bytes: &'a [u8], name: &str) -> Packed<'a> {
    let info = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let start = file.data_offset as usize + info.offset as usize;
    let len = info.byte_len().expect("length") as usize;
    let raw = &bytes[start..start + len];
    match info.ty {
        GgmlType::Q4K => Packed::Q4(raw),
        GgmlType::Q6K => Packed::Q6(raw),
        other => panic!("{name} is {other:?}"),
    }
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

#[test]
#[ignore = "needs a CUDA device and a model"]
fn one_layer_on_the_gpu() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let path = model_path();
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let file = GgufFile::parse(&bytes).expect("gguf");
    let arch = file
        .metadata
        .get("general.architecture")
        .and_then(Value::as_str)
        .unwrap_or("qwen2")
        .to_owned();
    let key = |name: &str| format!("{arch}.{name}");
    let shape = LayerShape {
        hidden: meta_usize(&file, &key("embedding_length"), 1536),
        ffn: meta_usize(&file, &key("feed_forward_length"), 8960),
        heads: meta_usize(&file, &key("attention.head_count"), 12),
        kv_heads: meta_usize(&file, &key("attention.head_count_kv"), 2),
        head_dim: meta_usize(&file, &key("embedding_length"), 1536)
            / meta_usize(&file, &key("attention.head_count"), 12),
        rope_base: meta_f32(&file, &key("rope.freq_base"), 1_000_000.0),
        epsilon: meta_f32(&file, &key("attention.layer_norm_rms_epsilon"), 1e-6),
        max_positions: 256,
    };
    println!(
        "shape: hidden {} ffn {} heads {}/{} head_dim {}",
        shape.hidden, shape.ffn, shape.heads, shape.kv_heads, shape.head_dim
    );

    let attn_norm = f32s(&file, &bytes, "blk.0.attn_norm.weight");
    let ffn_norm = f32s(&file, &bytes, "blk.0.ffn_norm.weight");
    let matrices = [
        quantized(&file, &bytes, "blk.0.attn_q.weight"),
        quantized(&file, &bytes, "blk.0.attn_k.weight"),
        quantized(&file, &bytes, "blk.0.attn_v.weight"),
        quantized(&file, &bytes, "blk.0.attn_output.weight"),
        quantized(&file, &bytes, "blk.0.ffn_gate.weight"),
        quantized(&file, &bytes, "blk.0.ffn_up.weight"),
        quantized(&file, &bytes, "blk.0.ffn_down.weight"),
    ];
    let has_bias = file
        .tensors
        .iter()
        .any(|tensor| tensor.name == "blk.0.attn_q.bias");
    let (bq, bk, bv) = if has_bias {
        (
            Some(f32s(&file, &bytes, "blk.0.attn_q.bias")),
            Some(f32s(&file, &bytes, "blk.0.attn_k.bias")),
            Some(f32s(&file, &bytes, "blk.0.attn_v.bias")),
        )
    } else {
        (None, None, None)
    };

    let start = Instant::now();
    let mut layer = GpuLayer::upload(
        &device,
        shape,
        LayerWeights {
            attn_norm: &attn_norm,
            ffn_norm: &ffn_norm,
            matrices,
            biases: [bq.as_deref(), bk.as_deref(), bv.as_deref()],
        },
    )
    .expect("upload layer");
    println!(
        "upload: {:.1} ms, {} MB of weights on the device",
        start.elapsed().as_secs_f64() * 1e3,
        layer.weight_bytes() / 1_000_000
    );

    let x: Vec<f32> = (0..shape.hidden)
        .map(|i| ((i as f32 * 0.013).sin()) * 0.5)
        .collect();
    // Warm up three positions (the first launches compile nothing but still fault the weights in).
    for position in 0..3 {
        let _ = layer.step(&device, &x, position).expect("warm up");
    }

    let rounds = 20;
    let start = Instant::now();
    let mut out = Vec::new();
    for position in 3..3 + rounds {
        out = layer.step(&device, &x, position).expect("step");
    }
    let per_layer = start.elapsed().as_secs_f64() / rounds as f64;
    println!(
        "gpu layer: {:.3} ms/step ({} of {} values, e.g. {:.4})",
        per_layer * 1e3,
        shape.hidden,
        out.len(),
        out[0]
    );

    // The CPU figure is the engine's own measured decode step on this machine, amortised over the
    // model's 28 layers: 30.6 ms/step over 28 layers.
    let cpu_per_layer = 30.6 / 28.0;
    println!(
        "cpu layer (amortised from a 30.6 ms decode step over 28 layers): {:.3} ms",
        cpu_per_layer
    );
    println!("speedup: {:.2}x", cpu_per_layer / (per_layer * 1e3));
    assert!(
        out.iter().all(|value| value.is_finite()),
        "non-finite output"
    );
}
