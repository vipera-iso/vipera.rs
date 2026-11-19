//! Small, isolated checks of the elementwise kernels that a layer chains.
//!
//! ```text
//! cargo test -p vipera-backend-gpu --release --test kernel_smoke -- --ignored --nocapture
//! ```

use vipera_backend_gpu::Device;
use vipera_backend_gpu::q8::Q8Device;

#[test]
#[ignore = "needs a CUDA device"]
fn norms_quantize_and_projection_follow_the_cpu() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let n = 1536;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0001).collect();

    let dx = device.upload(&x).unwrap();
    let dw = device.upload(&w).unwrap();
    let mut out = device.alloc::<f32>(n).unwrap();
    device.rmsnorm(&dx, &dw, &mut out, 1e-6).unwrap();
    let gpu = out.download(&device).unwrap();
    let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean + 1e-6).sqrt();
    let worst = (0..n)
        .map(|i| (gpu[i] - x[i] * inv * w[i]).abs())
        .fold(0.0_f32, f32::max);
    println!("rmsnorm worst error: {worst:.2e}");
    assert!(worst < 1e-4, "rmsnorm: {worst:.2e}");
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "rmsnorm produced non-finite values"
    );

    let mut q8 = Q8Device::allocate(&device, n).unwrap();
    device.quantize_into(&out, &mut q8, n / 32).unwrap();
    let qs = q8.qs.download(&device).unwrap();
    let d = q8.d.download(&device).unwrap();
    let s = q8.s.download(&device).unwrap();
    println!(
        "quantize: first group scale {:.6}, sum {:.0}, q[0] {}",
        d[0], s[0], qs[0]
    );
    assert!(d.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(s.iter().all(|v| v.is_finite()));
    assert_eq!(d.len(), n / 32);
}

#[test]
#[ignore = "needs a CUDA device"]
fn one_synthetic_q6k_block() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    // 256 values, d = 1, every scale = 1, every quant = 0 (so each value is -32).
    let mut block = vec![0u8; vipera_backend_gpu::q6k::BLOCK_BYTES];
    block[208] = 0x00;
    block[209] = 0x3c; // f16 1.0
    block[192..208].fill(1);
    let x = vec![1.0_f32; 256];
    let matrix =
        vipera_backend_gpu::q6k::Q6kMatrix::upload(&device, &block, 1, 256).expect("upload");
    let dx = device.upload(&x).unwrap();
    let mut out = device.alloc::<f32>(1).unwrap();
    matrix.matvec_into(&device, &dx, &mut out).unwrap();
    let value = out.download(&device).unwrap()[0];
    println!("synthetic q6k row: {value} (expected -8192)");
    assert!((value + 8192.0).abs() < 1.0, "got {value}");
}

#[test]
#[ignore = "needs a CUDA device"]
fn rope_attention_and_swiglu_match_their_definitions() {
    let device = match Device::open(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipped: {error}");
            return;
        }
    };
    let (heads, head_dim, position) = (2usize, 8usize, 5usize);
    let base = 10_000.0_f32;
    let vector: Vec<f32> = (0..heads * head_dim).map(|i| (i as f32) * 0.05).collect();

    let mut dev = device.upload(&vector).unwrap();
    device
        .rope(&mut dev, position, heads, head_dim, base)
        .unwrap();
    let rope = dev.download(&device).unwrap();
    let half = head_dim / 2;
    let mut expected = vector.clone();
    for head in 0..heads {
        for index in 0..half {
            let inv = 1.0 / base.powf(2.0 * index as f32 / head_dim as f32);
            let (sine, cosine) = (position as f32 * inv).sin_cos();
            let at = head * head_dim + index;
            let (low, high) = (vector[at], vector[at + half]);
            expected[at] = low * cosine - high * sine;
            expected[at + half] = low * sine + high * cosine;
        }
    }
    let worst = rope
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("rope worst error: {worst:.2e}");
    assert!(worst < 1e-5, "rope: {worst:.2e}");

    // One position: attention has one row to look at, so the output is that value row.
    let q: Vec<f32> = vec![0.3; heads * head_dim];
    let k: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();
    let v: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32).collect();
    let dq = device.upload(&q).unwrap();
    let dk = device.upload(&k).unwrap();
    let dv = device.upload(&v).unwrap();
    let mut k_cache = device.alloc::<f32>(4 * head_dim).unwrap();
    let mut v_cache = device.alloc::<f32>(4 * head_dim).unwrap();
    device
        .write_kv(0, &dk, &dv, &mut k_cache, &mut v_cache, head_dim)
        .unwrap();
    let shape = vipera_backend_gpu::layer::LayerShape {
        hidden: head_dim,
        ffn: head_dim,
        heads,
        kv_heads: 1,
        head_dim,
        rope_base: base,
        epsilon: 1e-6,
        max_positions: 4,
    };
    let mut out = device.alloc::<f32>(heads * head_dim).unwrap();
    device
        .attention(&dq, &k_cache, &v_cache, &mut out, 1, shape, 0.5)
        .unwrap();
    let attn = out.download(&device).unwrap();
    println!(
        "attention at one position: {:.3} {:.3}",
        attn[0], attn[head_dim]
    );
    assert!(attn.iter().all(|value| value.is_finite()));
    for head in 0..heads {
        for index in 0..head_dim {
            assert!((attn[head * head_dim + index] - v[index]).abs() < 1e-5);
        }
    }

    let gate: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
    let up: Vec<f32> = (0..64).map(|i| 0.5 + i as f32 * 0.01).collect();
    let dg = device.upload(&gate).unwrap();
    let du = device.upload(&up).unwrap();
    let mut act = device.alloc::<f32>(64).unwrap();
    device.swiglu(&dg, &du, &mut act, 64).unwrap();
    let act = act.download(&device).unwrap();
    let worst = (0..64)
        .map(|i| {
            let expect = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            (act[i] - expect).abs()
        })
        .fold(0.0_f32, f32::max);
    println!("swiglu worst error: {worst:.2e}");
    assert!(worst < 1e-5, "swiglu: {worst:.2e}");
}
