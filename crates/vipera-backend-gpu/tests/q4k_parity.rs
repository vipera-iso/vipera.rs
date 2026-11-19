//! The first GPU kernel, held to the CPU.
//!
//! Ignored by default: these need a CUDA device and a model. They print the reason and pass when
//! either is missing, the same contract the CPU measurement tests follow.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-gpu --release --test q4k_parity -- --ignored --nocapture
//! ```

use std::time::Instant;

use vipera_backend_cpu::quant;
use vipera_backend_gpu::{Device, q4k};
use vipera_core::gguf::{GgmlType, GgufFile};

/// Read the first Q4_K tensor of the model and return `(bytes, rows, cols)`.
fn tensor(path: &str, name: &str) -> Option<(Vec<u8>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    let file = GgufFile::parse(&bytes).ok()?;
    let info = file.tensors.iter().find(|tensor| tensor.name == name)?;
    if info.ty != GgmlType::Q4K {
        return None;
    }
    let start = file.data_offset as usize + info.offset as usize;
    let len = info.byte_len()? as usize;
    let cols = *info.dims.first()? as usize;
    let rows = len / (cols / q4k::QK_K) / q4k::BLOCK_BYTES;
    Some((bytes[start..start + len].to_vec(), rows, cols))
}

fn model_path() -> String {
    std::env::var("VIPERA_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            std::env::var("HOME").unwrap_or_default()
        )
    })
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_matvec_matches_the_cpu() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let Some((weights, rows, cols)) = tensor(&model_path(), "blk.0.ffn_gate.weight") else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    println!("device: {}", device.name().unwrap_or_default());
    println!(
        "tensor: {rows}x{cols} Q4_K, {} MB",
        weights.len() / 1_000_000
    );

    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.017).sin() + 1.0) * 0.5)
        .collect();
    let matrix = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");
    let gpu = matrix.matvec(&device, &x).expect("gpu matvec");
    assert_eq!(gpu.len(), rows);

    // The CPU reference dequantizes the same bytes to float and dots them in the same order the
    // kernel does not (the kernel splits each super-block across 32 lanes), so only the rounding
    // order differs.
    let row_bytes = cols / q4k::QK_K * q4k::BLOCK_BYTES;
    let mut worst: f32 = 0.0;
    let mut row_buffer = vec![0.0f32; cols];
    for row in 0..rows.min(64) {
        let data = &weights[row * row_bytes..(row + 1) * row_bytes];
        quant::read_row(GgmlType::Q4K, data, cols, 0, &mut row_buffer).expect("cpu dequant");
        let reference: f32 = row_buffer.iter().zip(&x).map(|(w, x)| w * x).sum();
        let scale = reference.abs().max(1.0);
        let error = (gpu[row] - reference).abs() / scale;
        worst = worst.max(error);
        if row < 3 {
            println!("row {row}: gpu {:.6} cpu {:.6}", gpu[row], reference);
        }
    }
    println!("worst relative error over 64 rows: {worst:.2e}");
    assert!(
        worst < 2e-4,
        "the kernel disagrees with the CPU by {worst:.2e}"
    );
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_matvec_throughput() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let Some((weights, rows, cols)) = tensor(&model_path(), "blk.0.ffn_gate.weight") else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).cos()).collect();
    println!("device: {}", device.name().unwrap_or_default());

    let matrix = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");
    // Warm up: the first call pays the module's lazy setup.
    let _ = matrix.matvec(&device, &x).expect("gpu matvec");

    let rounds = 30;
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = matrix.matvec(&device, &x).expect("gpu matvec");
    }
    let elapsed = start.elapsed().as_secs_f64() / rounds as f64;
    let gbps = matrix.bytes() as f64 / elapsed / 1e9;
    println!(
        "{rows}x{cols} Q4_K: {:.2} ms/call, {gbps:.1} GB/s (weights resident, one row of activations up)",
        elapsed * 1e3
    );
    println!("[cpu] the same shape measured 28-31 GB/s through the AVX-512 tile");
}
