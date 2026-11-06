//! Throughput of `matvec` per tensor type, on synthetic data of the shapes the real model uses.
//!
//! This exists to answer one question with a number instead of a guess: Q6_K is 32% of the bytes in
//! a Q4_K_M file and has no AVX2 kernel, so how much of a token's time does it take? It is
//! `#[ignore]`d because it is a measurement, not a correctness check — run it with:
//!
//! ```text
//! cargo test -p vipera-backend-cpu --release --test matvec_throughput -- --ignored --nocapture
//! ```
//!
//! Timing only reads bytes, so the contents do not need to be a valid quantization; a cheap LCG
//! fills them with something that is not all zeroes (which would let the FP units cut corners).
//!
//! This measures the spawn-per-call path. The pooled path a forward pass uses, measured against it
//! on the same data, is `quant::tests::pooled_matvec_throughput_by_tensor_type`:
//!
//! ```text
//! cargo test -p vipera-backend-cpu --release --lib -- --ignored --nocapture
//! ```

use std::time::Instant;

use vipera_backend_cpu::quant::matvec;
use vipera_core::gguf::GgmlType;

/// Deterministic bytes, so two runs measure the same data.
struct Fill(u64);

impl Fill {
    fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 24) as u8
    }
}

fn run(label: &str, ty: GgmlType, cols: usize, rows: usize, iterations: u32) {
    let mut fill = Fill(0x2545_f491_4f6c_dd1d);
    let bytes_per_row = match ty {
        GgmlType::Q4K => cols / 256 * 144,
        GgmlType::Q6K => cols / 256 * 210,
        _ => unreachable!("this benchmark only covers the two formats the model is made of"),
    };
    let data: Vec<u8> = (0..bytes_per_row * rows).map(|_| fill.byte()).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
    let mut out = vec![0.0_f32; rows];

    // One warm-up call: the first pass also faults in the allocation.
    matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes are consistent by construction");

    let started = Instant::now();
    for _ in 0..iterations {
        matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes are consistent by construction");
    }
    let elapsed = started.elapsed();
    let gigabytes = (bytes_per_row * rows) as f64 * f64::from(iterations) / 1e9;
    let seconds = elapsed.as_secs_f64();
    println!(
        "{label:>18}: {rows}x{cols} {ty:?}, {:>6.1} MB per call, {:>6.1} ms/call, {:>5.2} GB/s",
        (bytes_per_row * rows) as f64 / 1e6,
        seconds * 1000.0 / f64::from(iterations),
        gigabytes / seconds
    );
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn matvec_throughput_by_tensor_type() {
    // Shapes taken from the Qwen2.5-1.5B-Instruct Q4_K_M file this backend is measured on.
    run("ffn_gate (Q4_K)", GgmlType::Q4K, 1536, 8960, 20);
    run("ffn_down (Q6_K)", GgmlType::Q6K, 8960, 1536, 20);
    run("output (Q6_K)", GgmlType::Q6K, 1536, 151936, 3);
}
