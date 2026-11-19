//! The weight bus pushing its window onto the device.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-gpu --release --test vram_window -- --ignored --nocapture
//! ```

use std::sync::Arc;

use vipera_backend_cpu::bus::{Tier, WeightSource};
use vipera_backend_gpu::Device;
use vipera_backend_gpu::pool::VramPool;

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
fn the_bus_moves_its_window_onto_the_device() {
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
    // Two layers resident, everything else on disk.
    // SAFETY: this test is the only one in the binary that reads the variable.
    unsafe { std::env::set_var("VIPERA_RAM_LAYERS", "2") };
    let (mut source, _file) = WeightSource::open(std::path::Path::new(&path)).expect("open");
    println!("layers: {}", source.layers());

    let pool = Arc::new(VramPool::new(device));
    source.install_vram(pool.clone());
    assert_eq!(source.vram_bytes(), 0, "nothing is resident before a pass");

    source.advance(0);
    let after_first = source.vram_bytes();
    println!("after layer 0: {after_first} bytes on the device");
    assert!(after_first > 0, "the first advance brought nothing");
    assert!(pool.holds(0) && pool.holds(1), "the window is two layers");
    assert!(!pool.holds(2), "layer 2 is outside the window");
    assert_eq!(source.tier_of(0), Tier::Vram);
    assert_eq!(source.tier_of(7), Tier::Ssd);

    // Walk the pass: each layer brings the next one in and releases the one two behind.
    for layer in 1..6 {
        source.advance(layer);
    }
    let held = source.vram_bytes();
    println!(
        "after six layers: {held} bytes on the device (peak {}), holds 5={} 0={}",
        pool.peak_bytes(),
        pool.holds(5),
        pool.holds(0)
    );
    assert!(
        pool.holds(5) && pool.holds(6),
        "the window followed the pass"
    );
    assert!(
        !pool.holds(0),
        "a layer that left the window stays on the device"
    );
    assert_eq!(source.tier_of(5), Tier::Vram);
    assert_ne!(source.tier_of(0), Tier::Vram);
    // The pool holds a window, not the model.
    let model_bytes = std::fs::metadata(&path).expect("stat").len() as usize;
    assert!(
        held * 8 < model_bytes,
        "the pool held {held} of {model_bytes}"
    );
}
