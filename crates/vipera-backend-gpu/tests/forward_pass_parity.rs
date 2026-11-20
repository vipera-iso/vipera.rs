//! The CUDA seam, held to the runtime's own contract.
//!
//! The device backend is not a second engine if the runtime's paged KV buffer means the same bytes
//! to both. This test is the proof: the CPU backend writes a prompt's rows into the buffer, a fresh
//! device backend is handed that buffer and reads the prefix back out of it, and then the CPU backend
//! reads the rows the device wrote. Every comparison is on logits, not on internals.
//!
//! It runs wherever a CUDA device and a model are present and prints a skip reason otherwise
//! (`VIPERA_TEST_MODEL`, default `~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`).
//! `VIPERA_REQUIRE_GPU=1` turns the skip into a failure, which is what a GPU runner sets.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-gpu --release --test forward_pass_parity -- --nocapture
//! ```

use std::path::PathBuf;

use vipera_backend_cpu::CpuBackend;
use vipera_backend_gpu::{Device, GpuBackend};
use vipera_runtime::{ForwardContext, ForwardPass};

/// Blocks in the test cache, and the block size the runtime uses by default.
const TOKENS_PER_BLOCK: u64 = 16;

/// A prompt the tokenizer would produce; the backends only need ids inside the vocabulary. Long
/// enough to reach a second KV block, which is what the ring-table check needs.
const TOKENS: [u32; 18] = [
    9707, 11, 1879, 1234, 5678, 91011, 1213, 1415, 374, 279, 9707, 11, 1879, 1234, 5678, 91011,
    1213, 1415,
];

fn model_path() -> Option<PathBuf> {
    let path = std::env::var("VIPERA_TEST_MODEL").map_or_else(
        |_| {
            PathBuf::from(format!(
                "{}/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or_default()
            ))
        },
        PathBuf::from,
    );
    path.exists().then_some(path)
}

fn gpu_required() -> bool {
    matches!(
        std::env::var("VIPERA_REQUIRE_GPU").as_deref(),
        Ok("1" | "on" | "true" | "yes")
    )
}

/// One forward pass, with the buffer and geometry the runtime would hand a backend.
#[allow(clippy::too_many_arguments)]
fn logits(
    backend: &mut dyn ForwardPass,
    kv: &mut [u8],
    tokens: &[u32],
    first_position: usize,
    count: usize,
    bytes_per_token: u64,
    table: &[u32],
) -> Result<Vec<f32>, vipera_errors::ViperaError> {
    backend.logits(&mut ForwardContext {
        tokens,
        new_tokens: &tokens[first_position..first_position + count],
        first_position: first_position as u32,
        block_table: table,
        kv,
        tokens_per_block: TOKENS_PER_BLOCK,
        bytes_per_token,
    })
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn compare(label: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: vocabulary size differs");
    assert!(
        gpu.iter().chain(cpu).all(|value| value.is_finite()),
        "{label}: a logit is not finite"
    );
    let similarity = cosine(gpu, cpu);
    let worst = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{label}: argmax {} vs {}, cosine {similarity:.6}, max |diff| {worst:.3e}",
        argmax(gpu),
        argmax(cpu)
    );
    assert_eq!(
        argmax(gpu),
        argmax(cpu),
        "{label}: the two backends pick different tokens"
    );
    assert!(
        similarity > 0.99,
        "{label}: cosine {similarity:.6} is not the same distribution"
    );
}

#[test]
fn the_device_backend_matches_the_cpu_across_the_kv_buffer() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            if gpu_required() {
                panic!("VIPERA_REQUIRE_GPU is set and the device did not open: {error}");
            }
            eprintln!("skipped: {error}");
            return;
        }
    };

    let mut cpu = CpuBackend::load(&path).expect("the CPU backend loads the same file");
    let bytes_per_token = cpu.config().kv_bytes_per_token();
    assert!(bytes_per_token > 0, "the model has no f16 KV geometry");
    // Two blocks, and the table names them out of order so a hard-coded position-to-block map
    // cannot pass: positions 0..16 are block 1, positions 16..32 are block 0.
    let table = [1u32, 0];
    let mut kv = vec![0u8; (2 * TOKENS_PER_BLOCK * bytes_per_token) as usize];
    let mut gpu =
        GpuBackend::load(device, &path, 32).expect("the device backend loads and opens the card");

    // The CPU computes the prompt into the runtime's buffer; this is the state a prefix cache would
    // have been holding.
    logits(
        &mut cpu,
        &mut kv,
        &TOKENS[..8],
        0,
        8,
        bytes_per_token,
        &table,
    )
    .expect("cpu prefill");
    let cpu_eight = logits(
        &mut cpu,
        &mut kv,
        &TOKENS[..9],
        8,
        1,
        bytes_per_token,
        &table,
    )
    .expect("cpu decode");
    // A fresh device backend has computed nothing, so it has to read that prefix out of the buffer.
    let gpu_eight = logits(
        &mut gpu,
        &mut kv,
        &TOKENS[..9],
        8,
        1,
        bytes_per_token,
        &table,
    )
    .expect("gpu decode after uploading the CPU's prefix");
    compare("position 8, device after an upload", &gpu_eight, &cpu_eight);

    // Now the direction that matters more: the CPU reads the row the device wrote.
    let gpu_nine = logits(
        &mut gpu,
        &mut kv,
        &TOKENS[..10],
        9,
        1,
        bytes_per_token,
        &table,
    )
    .expect("gpu decode from its own cache");
    let cpu_nine = logits(
        &mut cpu,
        &mut kv,
        &TOKENS[..10],
        9,
        1,
        bytes_per_token,
        &table,
    )
    .expect("cpu decode over the device's row");
    compare(
        "position 9, CPU reading the device's row",
        &gpu_nine,
        &cpu_nine,
    );

    // What the seam refuses, and why: a q8_0 cache and a ring table are both wrong here, and both
    // have to be said out loud rather than computed approximately.
    let q8 = match logits(
        &mut gpu,
        &mut kv,
        &TOKENS[..1],
        0,
        1,
        bytes_per_token / 2,
        &table,
    ) {
        Ok(_) => panic!("a q8_0 geometry was accepted as f16"),
        Err(error) => error,
    };
    assert!(q8.to_string().contains("f16"), "{q8}");
    // Position 16 is the first of the second block, so the table is read for two entries and the
    // repeat is visible.
    let ring = [1u32, 1];
    let refused = match logits(
        &mut gpu,
        &mut kv,
        &TOKENS[..17],
        16,
        1,
        bytes_per_token,
        &ring,
    ) {
        Ok(_) => panic!("a ring table was run on a dense device cache"),
        Err(error) => error,
    };
    assert!(refused.to_string().contains("ring"), "{refused}");
}
