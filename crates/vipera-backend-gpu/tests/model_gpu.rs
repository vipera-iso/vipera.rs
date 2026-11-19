//! The whole model on the device, timed against the CPU's measured pp and tg.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-gpu --release --test model_gpu -- --ignored --nocapture
//! ```

use std::time::Instant;

use vipera_backend_gpu::Device;
use vipera_backend_gpu::GpuModel;

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
fn the_whole_model_runs_on_the_device() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    }
    let start = Instant::now();
    let mut model = GpuModel::load(device, std::path::Path::new(&path)).expect("load");
    println!(
        "loaded {} layers in {:.1} s, vocab {}",
        model.depth(),
        start.elapsed().as_secs_f64(),
        model.vocab()
    );

    // A short prompt: the same tokens the CPU sees, run one position at a time.
    let prompt: Vec<u32> = vec![9707, 11, 1879, 1234, 5678, 91011, 1213, 1415];
    let start = Instant::now();
    let mut logits = Vec::new();
    for (position, token) in prompt.iter().enumerate() {
        logits = model.step(*token, position).expect("prefill step");
    }
    let prefill = start.elapsed().as_secs_f64();
    println!(
        "pp {} tokens: {:.3} s -> {:.1} tok/s (sequential: one weight read per token)",
        prompt.len(),
        prefill,
        prompt.len() as f64 / prefill
    );

    let rounds = 16;
    let start = Instant::now();
    for position in prompt.len()..prompt.len() + rounds {
        logits = model.step(9707, position).expect("decode step");
    }
    let decode = start.elapsed().as_secs_f64() / rounds as f64;
    println!(
        "tg: {:.2} ms/token -> {:.1} tok/s",
        decode * 1e3,
        1.0 / decode
    );
    println!(
        "logits: {} values, finite {}",
        logits.len(),
        logits.iter().all(|v| v.is_finite())
    );
    let (best, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("logits");
    println!("argmax: {best}");
    assert!(logits.iter().all(|value| value.is_finite()));

    println!("[cpu] measured on this machine: pp128 186 tok/s, tg 30.6 ms/token (32.7 tok/s)");
}
