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
use vipera_backend_gpu::pool::VramPool;
use vipera_backend_gpu::q8::Q8;
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

/// f16 by bit pattern, matching the kernel's own conversion.
fn h2f(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    let v = if exp == 0 {
        f32::from_bits(0) + man * 2f32.powi(-24)
    } else if exp == 31 {
        if man == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (man + 1024.0) * 2f32.powi(exp - 25)
    };
    if sign == 1 { -v } else { v }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// The int8 dot the kernel computes, in Rust: exact integers, one float multiply per sub-block.
fn reference_q4k_q8(row: &[u8], x: &Q8, cols: usize) -> f32 {
    let mut acc = 0.0_f32;
    for b in 0..cols / q4k::QK_K {
        let blk = &row[b * q4k::BLOCK_BYTES..][..q4k::BLOCK_BYTES];
        let d = h2f(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = h2f(u16::from_le_bytes([blk[2], blk[3]]));
        for j in 0..8 {
            let (sc, m) = scale_min_k4(j, &blk[4..16]);
            let mut dot = 0_i32;
            for v in 0..32 {
                let byte = blk[16 + (j / 2) * 32 + v];
                let q = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let at = b * q4k::QK_K + j * 32 + v;
                dot += i32::from(q) * i32::from(x.qs[at]);
            }
            let g = b * 8 + j;
            acc += d * f32::from(sc) * x.d[g] * dot as f32 - dmin * f32::from(m) * x.d[g] * x.s[g];
        }
    }
    acc
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_matvec_q8_matches_the_integer_reference() {
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
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.031).sin() * 0.8) + ((i as f32 * 0.11).cos() * 0.3))
        .collect();
    let quantized = Q8::quantize(&x);
    let matrix = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");
    let gpu = matrix.matvec_q8(&device, &quantized).expect("gpu matvec");
    let float_gpu = matrix.matvec(&device, &x).expect("gpu float matvec");

    let row_bytes = cols / q4k::QK_K * q4k::BLOCK_BYTES;
    let mut worst_int: f32 = 0.0;
    let mut worst_float: f32 = 0.0;
    for row in 0..rows.min(128) {
        let reference =
            reference_q4k_q8(&weights[row * row_bytes..][..row_bytes], &quantized, cols);
        let scale = reference.abs().max(1.0);
        worst_int = worst_int.max((gpu[row] - reference).abs() / scale);
        worst_float =
            worst_float.max((gpu[row] - float_gpu[row]).abs() / float_gpu[row].abs().max(1.0));
    }
    println!("int8 vs integer reference: {worst_int:.2e}");
    println!("int8 vs float kernel: {worst_float:.2e} (quantization cost)");
    assert!(
        worst_int < 2e-5,
        "the dp4a kernel disagrees: {worst_int:.2e}"
    );
    assert!(worst_float < 1e-2, "int8 moved too far: {worst_float:.2e}");
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q6k_matvec_matches_the_cpu() {
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
    let Some(info) = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == "blk.0.ffn_down.weight")
    else {
        eprintln!("skipped: no ffn_down");
        return;
    };
    assert_eq!(info.ty, GgmlType::Q6K);
    let start = file.data_offset as usize + info.offset as usize;
    let len = info.byte_len().expect("length") as usize;
    let cols = *info.dims.first().expect("dims") as usize;
    let rows = len / (cols / vipera_backend_gpu::q6k::QK_K) / vipera_backend_gpu::q6k::BLOCK_BYTES;
    let weights = &bytes[start..start + len];
    println!("q6k tensor: {rows}x{cols}, {} MB", len / 1_000_000);

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.007).sin()).collect();
    let matrix =
        vipera_backend_gpu::q6k::Q6kMatrix::upload(&device, weights, rows, cols).expect("upload");
    let dx = device.upload(&x).expect("upload x");
    let mut out = device.alloc::<f32>(rows).expect("alloc");
    matrix.matvec_into(&device, &dx, &mut out).expect("matvec");
    let gpu = out.download(&device).expect("download");

    let row_bytes = cols / vipera_backend_gpu::q6k::QK_K * vipera_backend_gpu::q6k::BLOCK_BYTES;
    let mut buffer = vec![0.0_f32; cols];
    let mut worst: f32 = 0.0;
    for row in 0..rows.min(32) {
        quant::read_row(
            GgmlType::Q6K,
            &weights[row * row_bytes..][..row_bytes],
            cols,
            0,
            &mut buffer,
        )
        .expect("cpu dequant");
        let reference: f32 = buffer.iter().zip(&x).map(|(w, x)| w * x).sum();
        let error = (gpu[row] - reference).abs() / reference.abs().max(1.0);
        worst = worst.max(error);
    }
    println!("q6k worst relative error: {worst:.2e}");
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "q6k produced non-finite values"
    );
    assert!(worst < 2e-4, "q6k disagrees: {worst:.2e}");
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_matvec_q8_throughput() {
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
    let quantized = Q8::quantize(&x);
    let matrix = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");
    let _ = matrix.matvec_q8(&device, &quantized).expect("warm up");

    let rounds = 50;
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = matrix.matvec_q8(&device, &quantized).expect("gpu matvec");
    }
    let elapsed = start.elapsed().as_secs_f64() / rounds as f64;
    let gbps = matrix.bytes() as f64 / elapsed / 1e9;
    println!(
        "{rows}x{cols} Q4_K int8: {:.3} ms/call, {gbps:.1} GB/s (weights resident)",
        elapsed * 1e3
    );
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_matvec_reads_weights_from_the_pool() {
    use std::sync::Arc;

    use vipera_backend_gpu::pool::Format;
    use vipera_core::vram::VramSink;

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
    let owned = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");

    // The pool slot is exactly what the bus hands over: one layer's bytes, copied once.
    let pool = Arc::new(VramPool::new(device.clone()));
    pool.bring(3, &weights);
    let pooled = pool
        .matrix(3, 0, rows, cols, Format::Q4)
        .expect("a pooled view");
    assert_eq!(pooled.bytes(), weights.len());

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.004).sin()).collect();
    let quantized = Q8::quantize(&x);
    let mut via_pool = device.alloc::<f32>(rows).expect("alloc");
    let mut q8 = vipera_backend_gpu::q8::Q8Device::allocate(&device, cols).expect("alloc");
    let device_x = device.upload(&x).expect("upload");
    device
        .quantize_into(&device_x, &mut q8, cols / 32)
        .expect("quantize");
    pooled
        .project(&device, &device_x, &mut q8, &mut via_pool)
        .expect("pooled matvec");
    let pooled_out = via_pool.download(&device).expect("download");
    let owned_out = owned.matvec_q8(&device, &quantized).expect("owned matvec");

    let worst = pooled_out
        .iter()
        .zip(&owned_out)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0_f32, f32::max);
    println!("pooled vs owned matvec: {worst:.2e} over {rows} rows");
    assert!(worst < 1e-5, "the pooled read differs: {worst:.2e}");
}

#[test]
#[ignore = "needs a CUDA device and a model"]
fn q4k_batch_reuses_one_weight_read() {
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
    let matrix = q4k::Q4kMatrix::upload(&device, &weights, rows, cols).expect("upload");

    let batch = 8usize;
    let rows_x: Vec<f32> = (0..batch * cols)
        .map(|i| ((i as f32) * 0.0031).sin() * 0.5)
        .collect();
    let mut quantized =
        vipera_backend_gpu::q8::Q8Device::allocate(&device, batch * cols).expect("alloc");
    let device_x = device.upload(&rows_x).expect("upload");
    device
        .quantize_into(&device_x, &mut quantized, batch * cols / 32)
        .expect("quantize");
    let mut out = device.alloc::<f32>(batch * rows).expect("alloc");

    q4k::matvec_q8_batch_view(
        &device,
        &matrix.view(),
        rows,
        cols,
        &quantized,
        batch,
        &mut out,
    )
    .expect("batch");
    let batched = out.download(&device).expect("download");

    // Every entry must equal the single-row kernel's answer for that row.
    let mut worst: f32 = 0.0;
    for entry in 0..batch {
        let single = matrix
            .matvec_q8(
                &device,
                &vipera_backend_gpu::q8::Q8::quantize(&rows_x[entry * cols..][..cols]),
            )
            .expect("single");
        for row in 0..rows.min(16) {
            let reference = single[row];
            worst = worst
                .max((batched[entry * rows + row] - reference).abs() / reference.abs().max(1.0));
        }
    }
    println!("batch vs single worst relative difference: {worst:.2e}");
    assert!(worst < 1e-4, "the batch kernel disagrees: {worst:.2e}");

    let rounds = 30;
    let start = Instant::now();
    for _ in 0..rounds {
        q4k::matvec_q8_batch_view(
            &device,
            &matrix.view(),
            rows,
            cols,
            &quantized,
            batch,
            &mut out,
        )
        .expect("batch");
    }
    // Launches are asynchronous: without this the loop only measures how fast the host can enqueue.
    let _ = out.download(&device).expect("download");
    let per_call = start.elapsed().as_secs_f64() / rounds as f64;
    println!(
        "{rows}x{cols} Q4_K int8 x{batch}: {:.3} ms/call, {:.1} GB/s of weights ({:.3} ms/token)",
        per_call * 1e3,
        matrix.bytes() as f64 / per_call / 1e9,
        per_call * 1e3 / batch as f64
    );
    println!("[cpu] the same shape reads 28-31 GB/s and batches 128 tokens per read");
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
