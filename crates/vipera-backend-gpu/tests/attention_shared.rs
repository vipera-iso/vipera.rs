//! The two attention arms: the per-head kernel and the KV-head-shared one.
//!
//! ```text
//! cargo test -p vipera-backend-gpu --release --test attention_shared -- --ignored --nocapture
//! ```
//!
//! The first test is correctness: one block per KV head computes the same attention as one block per
//! query head, to the rounding of a different summation order. The second is the A/B the kernel was
//! added for: cache traffic against the per-head arm at the context lengths a long prompt uses.
//!
//! Synthetic rows, not a model. This measures the kernel, and a GGUF would put a 4 GB file and a
//! quantizer in front of the thing being timed; the rows cost the same to read either way.

use std::time::Instant;

use vipera_backend_gpu::Device;
use vipera_backend_gpu::layer::LayerShape;

/// Qwen3-14B's attention shape: 40 query heads, 8 KV heads, head width 128.
fn shape(max_positions: usize) -> LayerShape {
    LayerShape {
        hidden: 5120,
        ffn: 17408,
        heads: 40,
        kv_heads: 8,
        head_dim: 128,
        rope_base: 1.0e6,
        epsilon: 1e-6,
        max_positions,
    }
}

fn rows(count: usize, seed: u32) -> Vec<f32> {
    // Small deterministic values; `sin` keeps them uncorrelated enough that a wrong index shows up
    // as a wrong score rather than as a plausible one.
    let mut state = seed as f32;
    (0..count)
        .map(|i| {
            state = state * 1.000_017 + 0.37;
            (state + i as f32 * 0.0011).sin() * 0.5
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device"]
fn shared_attention_matches_the_per_head_kernel() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let shape = shape(512);
    let kv_width = shape.kv_width();
    let q = rows(shape.query_width(), 1);
    let k = rows(shape.max_positions * kv_width, 2);
    let v = rows(shape.max_positions * kv_width, 3);

    let dq = device.upload(&q).unwrap();
    let dk = device.upload(&k).unwrap();
    let dv = device.upload(&v).unwrap();
    let mut per_head = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut shared = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut scores = device
        .alloc::<f32>(shape.heads * shape.max_positions)
        .unwrap();
    let scale = 1.0 / (shape.head_dim as f32).sqrt();

    for positions in [1, 7, 64, 511] {
        device
            .attention(&dq, &dk, &dv, &mut per_head, positions, shape, scale)
            .unwrap();
        device
            .attention_shared(
                &dq,
                &dk,
                &dv,
                &mut shared,
                &mut scores,
                positions,
                shape,
                scale,
            )
            .unwrap();
        let a = per_head.download(&device).unwrap();
        let b = shared.download(&device).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let largest = a.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        println!(
            "positions {positions}: worst |per_head - shared| = {worst:.2e} on outputs up to {largest:.3}"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "the shared kernel produced non-finite values"
        );
        assert!(
            worst < 1e-5 * largest.max(1.0),
            "positions {positions}: {worst:.2e} is past rounding"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn split_attention_matches_the_per_head_kernel() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let shape = shape(1024);
    let kv_width = shape.kv_width();
    let chunk = 512;
    let chunks = shape.max_positions.div_ceil(chunk);
    let q = rows(shape.query_width(), 1);
    let k = rows(shape.max_positions * kv_width, 2);
    let v = rows(shape.max_positions * kv_width, 3);

    let dq = device.upload(&q).unwrap();
    let dk = device.upload(&k).unwrap();
    let dv = device.upload(&v).unwrap();
    let mut per_head = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut split = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut partial = device
        .alloc::<f32>(shape.heads * (2 * chunks + chunks * shape.head_dim))
        .unwrap();
    let scale = 1.0 / (shape.head_dim as f32).sqrt();

    // 1 and 7 are shorter than one chunk, 256/257/511/512 straddle chunk and thread edges, 1023 is
    // the last position that fits, and 1024 is the chunk with an empty tail past it.
    for positions in [1, 7, 255, 256, 257, 511, 512, 513, 1023, 1024] {
        device
            .attention(&dq, &dk, &dv, &mut per_head, positions, shape, scale)
            .unwrap();
        let written = device
            .attention_split(&dq, &dk, &dv, &mut partial, positions, chunk, shape, scale)
            .unwrap();
        device
            .attention_combine(&partial, &mut split, written, shape)
            .unwrap();
        let a = per_head.download(&device).unwrap();
        let b = split.download(&device).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let largest = a.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        println!(
            "positions {positions} in {written} chunks: worst |per_head - split| = {worst:.2e} on outputs up to {largest:.3}"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "the split kernel produced non-finite values"
        );
        assert!(
            worst < 1e-5 * largest.max(1.0),
            "positions {positions}: {worst:.2e} is past rounding"
        );
    }
}

#[test]
#[ignore = "needs a CUDA device"]
fn attention_arms_at_long_context() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    // Two shapes: the 14B one, where attention is 40 blocks of work per step, and the 1.5B one this
    // card actually serves, where there are 12 query heads and 2 KV heads.
    for (label, shape) in [
        ("qwen3-14b 40/8", shape(4096)),
        (
            "qwen2.5-1.5b 12/2",
            LayerShape {
                hidden: 1536,
                ffn: 8960,
                heads: 12,
                kv_heads: 2,
                head_dim: 128,
                rope_base: 1.0e6,
                epsilon: 1e-6,
                max_positions: 4096,
            },
        ),
    ] {
        sweep(&device, label, shape);
    }
}

fn sweep(device: &Device, label: &str, shape: LayerShape) {
    let kv_width = shape.kv_width();
    let scale = 1.0 / (shape.head_dim as f32).sqrt();
    let chunk = 128;
    let chunks = shape.max_positions.div_ceil(chunk);
    let dq = device.upload(&rows(shape.query_width(), 1)).unwrap();
    let dk = device
        .upload(&rows(shape.max_positions * kv_width, 2))
        .unwrap();
    let dv = device
        .upload(&rows(shape.max_positions * kv_width, 3))
        .unwrap();
    let mut per_head = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut shared = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut split = device.alloc::<f32>(shape.query_width()).unwrap();
    let mut scores = device
        .alloc::<f32>(shape.heads * shape.max_positions)
        .unwrap();
    let mut partial = device
        .alloc::<f32>(shape.heads * (2 * chunks + chunks * shape.head_dim))
        .unwrap();

    println!(
        "{label}: {} query heads / {} kv heads, head_dim {}, {kv_width} floats per K row",
        shape.heads, shape.kv_heads, shape.head_dim
    );
    let rounds = 20;
    for positions in [512, 1024, 2048, 4096] {
        // Warm every arm so none pays the first-launch cost inside a round.
        for _ in 0..3 {
            device
                .attention(&dq, &dk, &dv, &mut per_head, positions, shape, scale)
                .unwrap();
            device
                .attention_shared(
                    &dq,
                    &dk,
                    &dv,
                    &mut shared,
                    &mut scores,
                    positions,
                    shape,
                    scale,
                )
                .unwrap();
            device
                .attention_split(&dq, &dk, &dv, &mut partial, positions, chunk, shape, scale)
                .unwrap();
            device
                .attention_combine(&partial, &mut split, positions.div_ceil(chunk), shape)
                .unwrap();
        }
        device.synchronize().unwrap();

        let mut per = 0.0_f64;
        let mut share = 0.0_f64;
        let mut cut = 0.0_f64;
        // Interleaved: the arms follow each other inside every round, so a clock that drifts with
        // the card's temperature charges all three the same.
        for _ in 0..rounds {
            let start = Instant::now();
            device
                .attention(&dq, &dk, &dv, &mut per_head, positions, shape, scale)
                .unwrap();
            device.synchronize().unwrap();
            per += start.elapsed().as_secs_f64();

            let start = Instant::now();
            device
                .attention_shared(
                    &dq,
                    &dk,
                    &dv,
                    &mut shared,
                    &mut scores,
                    positions,
                    shape,
                    scale,
                )
                .unwrap();
            device.synchronize().unwrap();
            share += start.elapsed().as_secs_f64();

            let start = Instant::now();
            device
                .attention_split(&dq, &dk, &dv, &mut partial, positions, chunk, shape, scale)
                .unwrap();
            device
                .attention_combine(&partial, &mut split, positions.div_ceil(chunk), shape)
                .unwrap();
            device.synchronize().unwrap();
            cut += start.elapsed().as_secs_f64();
        }
        let per = per / rounds as f64 * 1.0e3;
        let share = share / rounds as f64 * 1.0e3;
        let cut = cut / rounds as f64 * 1.0e3;
        println!(
            "  {positions:5} positions: per-head {per:.3} ms, shared {share:.3} ms ({:+.1}%), split {cut:.3} ms ({:+.1}%), {:.1}x",
            (per - share) / per * 100.0,
            (per - cut) / per * 100.0,
            per / cut
        );
    }
}
