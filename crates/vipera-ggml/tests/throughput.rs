//! Kernel throughput, the number ADR-001 gate E3 asks for.
//!
//! Single-threaded on purpose: the question is kernel quality, not the pool. The engine's
//! own 8-thread in-situ figure for the same shape is 28.7-33.0 GB/s, so a single thread can
//! only be compared to a single-thread expectation, not to that number directly.
//!
//! Run with:
//! `VIPERA_GGML_SRC=/tmp/llamacpp cargo test --release --test throughput -- --ignored --nocapture`

use std::time::Instant;
use vipera_ggml::{dot_q4_k_q8_k, q4_k_bytes, q8_k_bytes, quantize_q8_k};

/// A Q4_K row with every scale 1, `dmin = 0`, `d = 1`, and 4-bit quants that vary by row, so
/// the kernel cannot constant-fold anything.
fn q4k_row(cols: usize, salt: usize) -> Vec<u8> {
    let mut row = vec![0u8; q4_k_bytes(cols)];
    for block in row.as_chunks_mut::<144>().0 {
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        block[2..4].copy_from_slice(&0u16.to_le_bytes()); // dmin = 0.0
        for byte in block[4..16].iter_mut() {
            *byte = 0x01;
        }
        for (index, byte) in block[16..].iter_mut().enumerate() {
            let low = ((index + salt) % 15 + 1) as u8;
            let high = ((index + salt + 5) % 15 + 1) as u8;
            *byte = low | (high << 4);
        }
    }
    row
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn q4k_gemv_throughput() {
    for &(rows, cols) in &[(8960usize, 1536usize), (1536, 8960), (151936, 1536)] {
        let weights: Vec<u8> = (0..rows).flat_map(|row| q4k_row(cols, row)).collect();
        let x: Vec<f32> = (0..cols).map(|at| (at as f32 * 0.013).sin()).collect();
        let mut q8k = vec![0u8; q8_k_bytes(cols)];
        unsafe { quantize_q8_k(&x, &mut q8k).expect("quantize") };

        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let mut checksum = 0.0f32;
            for row in weights.chunks_exact(q4_k_bytes(cols)) {
                checksum += unsafe { dot_q4_k_q8_k(cols, row, &q8k).expect("dot") };
            }
            std::hint::black_box(checksum);
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("ordered"));
        let seconds = samples[samples.len() / 2];
        let gigabytes = weights.len() as f64 / 1e9;
        println!(
            "q4_K {:>6}x{:<5} gemv 1T: {:>7.1} ms  {:>5.2} GB/s  (kernel {})",
            rows,
            cols,
            seconds * 1e3,
            gigabytes / seconds,
            vipera_ggml::kernel_build()
        );
    }
}
