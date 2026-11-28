//! Vipera vs llama.cpp — the four tricks from the deep-research piece, turned into measurements.
//!
//! What the original piece claims:
//! * **Fused dequant+GEMM** removes 128 MB of intermediate traffic; ik_llama.cpp's Q4_K prefill is
//!   1.87x and Q6_K 2.43x faster than llama.cpp thanks to AVX512-VNNI `vpdpbusd` and 64/128-byte
//!   reuse. Vipera does the same: `QuantizedX` (i8 activations) + `simd::try_dot` with a VNNI path
//!   and an AVX2 fallback.
//! * **KV-cache tradeoff with depth**: q8_0 saves ~47% RAM (58.7 MB f16 -> 31.2 MB at ctx 2048 for
//!   Qwen2.5-1.5B) but the decode penalty grows with depth; at 64K only 56% of f16 remains. This
//!   test measures the penalty at 32 / 256 / 1024 (kept light so CI can run it).
//! * **Prefill and decode measured separately** (Mistral.rs CPU decode 1.81x at 16K depth): prefill
//!   is compute-bound, decode is bandwidth-bound. Vipera chunks the prompt through the batch path
//!   (`run_positions`, `ceil(n/8)` weight reads) while decode reads the weights once per token. The
//!   test reports pp tok/s and tg tok/s separately.
//! * **Batched decode** (continuous batching): one weight read serves B activations. llama.cpp does
//!   the same via `n_parallel`; vipera has `logits_batch` with `MAX_BATCH = 8`.
//!
//! All of these are measurements, not correctness checks — all `#[ignore]`d.
//! To measure vipera:
//!   VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
//!   cargo test -p vipera-backend-cpu --release --test win_vs_llama -- --ignored --nocapture
//! To compare against llama.cpp on the same machine (same model, ctx and threads): build CPU-only
//! llama.cpp, run `llama-bench -m <model> -p 512 -n 128 -t 8`, and read it against the `llama.cpp`
//! column this test prints. If `llama-bench` is on PATH the test runs it itself and prints both side
//! by side.
//!
//! To see how much VNNI wins: run twice, once with `VIPERA_NO_VNNI=1`.
//! To see the thread budget: `VIPERA_THREADS=4 cargo test ...`.

use std::path::PathBuf;
use std::time::Instant;

use vipera_backend_cpu::quant::matvec;
use vipera_backend_cpu::{CpuBackend, ModelConfig};
use vipera_core::gguf::GgmlType;
use vipera_core::planner::KvCacheType;
use vipera_core::window::{PREFILL_HEADROOM, attention_window};
use vipera_runtime::{ForwardContext, ForwardPass};

const TOKENS_PER_BLOCK: u64 = 16;

fn model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VIPERA_TEST_MODEL") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    path.exists().then_some(path)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(best, top), (at, value)| {
            if *value > top {
                (at, *value)
            } else {
                (best, top)
            }
        })
        .0
}

/// One sequence plus its own KV buffer, with the selected KV type.
#[derive(Clone)]
struct State {
    tokens: Vec<u32>,
    kv: Vec<u8>,
    table: Vec<u32>,
    bytes_per_token: u64,
}

impl State {
    fn with_room(config: ModelConfig, positions: usize, ty: KvCacheType) -> Self {
        let bytes_per_token = config
            .kv_bytes_per_token_for(ty)
            .expect("qwen2.5-1.5b width is a multiple of 32");
        let blocks = positions.div_ceil(TOKENS_PER_BLOCK as usize);
        Self {
            tokens: Vec::new(),
            kv: vec![0; blocks * bytes_per_token as usize * TOKENS_PER_BLOCK as usize],
            table: (0..blocks as u32).collect(),
            bytes_per_token,
        }
    }

    /// The same sequence in a ring cache: one pinned block for the sinks, then a ring just long
    /// enough for the window, so `kv` is the window's size instead of the context's.
    fn with_ring(
        config: ModelConfig,
        positions: usize,
        ty: KvCacheType,
        window: usize,
        sinks: usize,
    ) -> Self {
        let bytes_per_token = config
            .kv_bytes_per_token_for(ty)
            .expect("qwen2.5-1.5b width is a multiple of 32");
        let tpb = TOKENS_PER_BLOCK as usize;
        let pinned = sinks.div_ceil(tpb).max(1);
        let rolling = (window + PREFILL_HEADROOM as usize).div_ceil(tpb) + 1;
        let blocks = pinned + rolling;
        Self {
            tokens: Vec::new(),
            kv: vec![0; blocks * bytes_per_token as usize * tpb],
            table: (0..positions.div_ceil(tpb))
                .map(|block| {
                    if block < pinned {
                        block as u32
                    } else {
                        (pinned + (block - pinned) % rolling) as u32
                    }
                })
                .collect(),
            bytes_per_token,
        }
    }

    fn context<'a>(&'a mut self, new_tokens: &'a [u32]) -> ForwardContext<'a> {
        let first_position = self.tokens.len() as u32;
        ForwardContext {
            tokens: &self.tokens,
            new_tokens,
            first_position,
            block_table: &self.table,
            kv: &mut self.kv,
            tokens_per_block: TOKENS_PER_BLOCK,
            bytes_per_token: self.bytes_per_token,
        }
    }
}

fn prefill_chunked(
    backend: &mut CpuBackend,
    config: ModelConfig,
    prompt: &[u32],
    ty: KvCacheType,
) -> (State, u32) {
    let mut state = State::with_room(config, prompt.len() + 4, ty);
    let logits = backend
        .logits(&mut state.context(prompt))
        .expect("prompt fits");
    state.tokens.extend_from_slice(prompt);
    (state, argmax(&logits) as u32)
}

/// Runs `rounds` times and returns the best time per run, in ms.
fn timed_best<F: FnMut()>(rounds: u32, mut run: F) -> f64 {
    // One warm-up: the first run faults in the allocation and the cache.
    run();
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let start = Instant::now();
        run();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best * 1000.0
}

// ---------------------------------------------------------------------------
// 1. Fused kernel: GB/s theo tensor type (Q4_K fused i8 + VNNI vs Q6_K)
// ---------------------------------------------------------------------------
//
// Vipera fuses it: dequant row + dot in one pass, with the activation already reduced to i8
// (`QuantizedX`), so memory only reads the weight bytes (7.7 MB for ffn_gate, 191 MB for output)
// and writes no intermediate f32. llama.cpp's GGML does the same, but its Q4_K lacked VNNI
// `vpdpbusd` before that was added. This measures whether vipera reaches the machine's bandwidth.

#[test]
#[ignore = "measurement, not a correctness check"]
fn kernel_gbps_fused_q4k_q6k() {
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
            _ => unreachable!(),
        };
        let data: Vec<u8> = (0..bytes_per_row * rows).map(|_| fill.byte()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
        let mut out = vec![0.0_f32; rows];
        matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes by construction");
        let started = Instant::now();
        for _ in 0..iterations {
            matvec(ty, &data, cols, rows, &x, &mut out).expect("shapes by construction");
        }
        let elapsed = started.elapsed();
        let gigabytes = (bytes_per_row * rows) as f64 * f64::from(iterations) / 1e9;
        let seconds = elapsed.as_secs_f64();
        println!(
            "{label:>18}: {rows}x{cols} {ty:?}, {:>6.1} MB/call, {:>6.1} ms/call, {:>5.2} GB/s{}",
            (bytes_per_row * rows) as f64 / 1e6,
            seconds * 1000.0 / f64::from(iterations),
            gigabytes / seconds,
            if std::env::var("VIPERA_NO_VNNI").is_ok() {
                " [NO_VNNI]"
            } else {
                ""
            }
        );
    }

    // The real Qwen2.5-1.5B shapes (memory summary: 1.117 GB of weights).
    run("ffn_gate (Q4_K)", GgmlType::Q4K, 1536, 8960, 30);
    run("ffn_down (Q6_K)", GgmlType::Q6K, 8960, 1536, 30);
    run("wq (Q4_K)", GgmlType::Q4K, 1536, 1536, 200);
    run("output (Q6_K)", GgmlType::Q6K, 1536, 151_936, 5);
    println!(
        "Note: fused i8+VNNI reads only the weight bytes; llama.cpp's Q4_K without VNNI is ~1.8x slower (ik_llama.cpp Q4_K prefill 1.87x)."
    );
    print_llama_hint("kernel");
}

// ---------------------------------------------------------------------------
// 2. KV-cache tradeoff with depth: q8_0 saves RAM but the penalty grows
// ---------------------------------------------------------------------------
//
// The research piece: q8_0 saves 47% RAM at ctx 2048, but at 64K decode runs at only 56% of f16.
// Vipera stores both through `KvLayout` (f16: 2 bytes/value, q8_0: 34/32 bytes/value). This test
// prefills at each depth, measures 3 decode steps, and prints the penalty % and the RAM saved %.

#[test]
#[ignore = "measurement, not a correctness check"]
fn kv_quantization_tradeoff() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();

    println!("ctx     f16 ms/step   q8_0 ms/step   penalty   RAM saved");
    for &ctx in &[32usize, 128] {
        // Fake tokens: they go through MHA/GQA but do not need to be meaningful text.
        let prompt: Vec<u32> = (0..ctx).map(|at| 1000 + (at % 400) as u32).collect();

        let mut ms = [f64::INFINITY; 2];
        for (slot, ty) in [KvCacheType::F16, KvCacheType::Q8_0]
            .into_iter()
            .enumerate()
        {
            // Prefill chunked through the batch path for speed, then keep the state to decode.
            let (mut state, mut next) = prefill_chunked(&mut backend, config, &prompt, ty);
            // One warm-up decode step, then the best of 3.
            let logits = backend.logits(&mut state.context(&[next])).expect("decode");
            next = argmax(&logits) as u32;
            state.tokens.push(next);
            let step = timed_best(3, || {
                let logits = backend.logits(&mut state.context(&[next])).expect("decode");
                next = argmax(&logits) as u32;
            });
            // The measured closure must not push to tokens or it would change ctx; the logits are
            // still computed at the same position, and the KV simply overwrites the same slot.
            ms[slot] = step;
        }
        let f16 = ms[0];
        let q8 = ms[1];
        let penalty = 100.0 * (q8 - f16) / f16;
        let f16_bytes = config.kv_bytes_per_token_for(KvCacheType::F16).unwrap_or(0);
        let q8_bytes = config
            .kv_bytes_per_token_for(KvCacheType::Q8_0)
            .unwrap_or(0);
        let saved = if f16_bytes > 0 {
            100.0 * (f16_bytes - q8_bytes) as f64 / f16_bytes as f64
        } else {
            0.0
        };
        println!(
            "{ctx:>5}  {f16:>7.1}        {q8:>7.1}        {penalty:>5.1}%    {saved:>5.1}%  (f16 {f16_bytes} B/tok, q8_0 {q8_bytes} B/tok)"
        );
    }
    println!(
        "Expected from the piece: the penalty grows with ctx; at small ctx q8_0 is barely slower, at large ctx it shows."
    );
    print_llama_hint("kv");
}

// ---------------------------------------------------------------------------
// 3. Prefill vs decode: chunked prefill (compute-bound) vs decode (bandwidth-bound)
// ---------------------------------------------------------------------------
//
// llama.cpp separates `pp` (prompt processing) from `tg` (token generation) because they are
// completely different phases: prefill computes many positions per weight read, decode one. Vipera
// chunks the prompt through the batch path (`ceil(n/8)` reads) while decode stays at one read per
// token. This test measures both and compares against the ~19-22 tok/s decode measured on this machine.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_vs_decode() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();

    for &n in &[64usize, 128, 512] {
        let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();
        // Chunked: one ForwardContext with n new_tokens -> run_positions takes the batch path.
        let chunk_ms = {
            let mut state = State::with_room(config, n + 4, KvCacheType::F16);
            timed_best(2, || {
                // Reset the tokens to measure from scratch; the KV is overwritten.
                state.tokens.clear();
                backend
                    .logits(&mut state.context(&prompt))
                    .expect("prefill");
            })
        };
        let seq_ms = {
            let mut backend2 = CpuBackend::load(&path).expect("reload for sequential");
            let mut state = State::with_room(config, n + 4, KvCacheType::F16);
            let start = Instant::now();
            let mut next = 0u32;
            for &tok in &prompt {
                let logits = backend2.logits(&mut state.context(&[tok])).expect("decode");
                state.tokens.push(tok);
                next = argmax(&logits) as u32;
            }
            let _ = next;
            start.elapsed().as_secs_f64() * 1000.0
        };
        let chunk_tok = n as f64 / (chunk_ms / 1000.0);
        let seq_tok = n as f64 / (seq_ms / 1000.0);
        println!(
            "prefill n={n:>4}: chunked {chunk_ms:>7.1} ms ({chunk_tok:>6.1} tok/s), sequential {seq_ms:>7.1} ms ({seq_tok:>6.1} tok/s), chunk speedup {:>4.2}x",
            seq_ms / chunk_ms
        );
    }

    // Decode tok/s at ctx 32 (essentially weights only, attention ~0%).
    let prompt: Vec<u32> = (0..32).map(|at| 1000 + (at % 400) as u32).collect();
    let (mut state, mut next) = prefill_chunked(&mut backend, config, &prompt, KvCacheType::F16);
    let decode_ms = timed_best(5, || {
        let logits = backend.logits(&mut state.context(&[next])).expect("decode");
        next = argmax(&logits) as u32;
    });
    println!(
        "decode ctx 32: {decode_ms:>6.1} ms/step ({:.1} tok/s)",
        1000.0 / decode_ms
    );
    println!(
        "Expected: prefill >> decode; report `pp` and `tg` separately like llama-bench, never one average."
    );
    print_llama_hint("prefill");
}

// ---------------------------------------------------------------------------
// 3a. Q3_K tiled band vs its row kernel, one process, arms interleaved
// ---------------------------------------------------------------------------
//
// `VIPERA_Q3K_TILED=0` sends Q3_K back to the row kernel; everything else (Q4_K/Q6_K tiles, chunk
// width, defaults) stays identical, so the only difference is the kernel under test. The two arms
// are different summation orders (documented numeric policy, not bit-identity), so the check is
// top-1 agreement plus the largest logit delta, and the number reported is the ratio of medians
// from interleaved rounds — a before/after pair minutes apart cannot resolve 5% on this machine.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_q3k_band_vs_row() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let n = 512usize;
    let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();

    let run = |backend: &mut CpuBackend, band: bool| {
        vipera_backend_cpu::pin_q3k_tiled(Some(band));
        let mut state = State::with_room(config, n + 4, KvCacheType::F16);
        let start = Instant::now();
        let logits = backend
            .logits(&mut state.context(&prompt))
            .expect("prefill");
        (start.elapsed().as_secs_f64() * 1e3, logits)
    };

    // One warm-up per arm, also the pair the numeric check compares.
    let (_, band_logits) = run(&mut backend, true);
    let (_, row_logits) = run(&mut backend, false);
    let band_top = argmax(&band_logits);
    let row_top = argmax(&row_logits);
    let max_delta = band_logits
        .iter()
        .zip(&row_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("numeric: top-1 band {band_top} vs row {row_top}, max |logit delta| {max_delta:.5}");
    assert_eq!(
        band_top, row_top,
        "the two Q3_K paths must pick the same token"
    );

    let rounds = 5usize;
    let mut band_ms = Vec::with_capacity(rounds);
    let mut row_ms = Vec::with_capacity(rounds);
    println!("round   band ms   row ms   row/band");
    for round in 0..rounds {
        // Alternate which arm runs first, so a monotonic host drift hits both arms equally.
        let (first, second) = if round % 2 == 0 {
            (run(&mut backend, true), run(&mut backend, false))
        } else {
            let row = run(&mut backend, false);
            let band = run(&mut backend, true);
            (band, row)
        };
        let (band, row) = (first.0, second.0);
        assert_eq!(first.1.len(), second.1.len());
        println!("{round:>5}  {band:>8.1}  {row:>7.1}   {:>7.3}", row / band);
        band_ms.push(band);
        row_ms.push(row);
    }
    vipera_backend_cpu::pin_q3k_tiled(None);
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let band = median(&mut band_ms);
    let row = median(&mut row_ms);
    println!(
        "median: band {band:.1} ms ({:.1} tok/s), row {row:.1} ms ({:.1} tok/s), row/band {:>6.3}x ({:+.1}% for the band)",
        n as f64 / (band / 1e3),
        n as f64 / (row / 1e3),
        row / band,
        100.0 * (row / band - 1.0)
    );
}

// ---------------------------------------------------------------------------
// 3a'. Q3_K band: two tile halves per pass vs one, one process, arms interleaved
// ---------------------------------------------------------------------------
//
// `VIPERA_Q3K_PAIR` groups two tile halves into one band pass so each row-block is decoded once
// for 32 activations instead of twice for 16. Each half keeps its own accumulator and summation
// order, so this is bit-identity, not a numeric-policy change: the test asserts the logits match
// exactly, and reports the ratio of medians from interleaved rounds.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_q3k_pair_vs_single() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let n = 512usize;
    let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();

    let run = |backend: &mut CpuBackend, paired: bool| {
        vipera_backend_cpu::pin_q3k_pair(Some(paired));
        let mut state = State::with_room(config, n + 4, KvCacheType::F16);
        let start = Instant::now();
        let logits = backend
            .logits(&mut state.context(&prompt))
            .expect("prefill");
        (start.elapsed().as_secs_f64() * 1e3, logits)
    };

    // One warm-up per arm, also the pair the bit-identity check compares.
    let (_, paired_logits) = run(&mut backend, true);
    let (_, single_logits) = run(&mut backend, false);
    let max_delta = paired_logits
        .iter()
        .zip(&single_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("numeric: max |logit delta| {max_delta:.9} (must be 0)");
    assert_eq!(
        paired_logits.len(),
        single_logits.len(),
        "both arms must produce one logit vector"
    );
    assert_eq!(max_delta, 0.0, "the paired band must be bit-identical");

    let rounds = 5usize;
    let mut paired_ms = Vec::with_capacity(rounds);
    let mut single_ms = Vec::with_capacity(rounds);
    println!("round  paired ms  single ms  single/paired");
    for round in 0..rounds {
        // Alternate which arm runs first, so a monotonic host drift hits both arms equally.
        let (first, second) = if round % 2 == 0 {
            (run(&mut backend, true), run(&mut backend, false))
        } else {
            let single = run(&mut backend, false);
            let paired = run(&mut backend, true);
            (paired, single)
        };
        let (paired, single) = (first.0, second.0);
        println!(
            "{round:>5}  {paired:>9.1}  {single:>9.1}   {:>7.3}",
            single / paired
        );
        paired_ms.push(paired);
        single_ms.push(single);
    }
    vipera_backend_cpu::pin_q3k_pair(None);
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let paired = median(&mut paired_ms);
    let single = median(&mut single_ms);
    println!(
        "median: paired {paired:.1} ms ({:.1} tok/s), single {single:.1} ms ({:.1} tok/s), single/paired {:>6.3}x ({:+.1}% for the pair)",
        n as f64 / (paired / 1e3),
        n as f64 / (single / 1e3),
        single / paired,
        100.0 * (single / paired - 1.0)
    );
}

// ---------------------------------------------------------------------------
// 3a''. Q3_K band: four tile halves per pass vs two, one process, arms interleaved
// ---------------------------------------------------------------------------
//
// `VIPERA_Q3K_QUAD` groups four tile halves into one band pass, so each row-block decode is paid
// once for 64 activations instead of twice for 32. It buys less decode than the pair did and costs
// sixteen live accumulators instead of eight, which is the spill risk: the test asserts the logits
// are unchanged (each half keeps its own accumulator) and reports the ratio of medians, so the
// measurement says whether the decode saving survived the register pressure.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_q3k_quad_vs_pair() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let n = 512usize;
    let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();

    // `quad` picks four halves per pass; the pair is pinned on in both arms.
    let run = |backend: &mut CpuBackend, quad: bool| {
        vipera_backend_cpu::pin_q3k_pair(Some(true));
        vipera_backend_cpu::pin_q3k_quad(Some(quad));
        let mut state = State::with_room(config, n + 4, KvCacheType::F16);
        let start = Instant::now();
        let logits = backend
            .logits(&mut state.context(&prompt))
            .expect("prefill");
        (start.elapsed().as_secs_f64() * 1e3, logits)
    };

    // One warm-up per arm, also the bit-identity check.
    let (_, quad_logits) = run(&mut backend, true);
    let (_, pair_logits) = run(&mut backend, false);
    let max_delta = quad_logits
        .iter()
        .zip(&pair_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("numeric: max |logit delta| {max_delta:.9} (must be 0)");
    assert_eq!(
        quad_logits.len(),
        pair_logits.len(),
        "both arms must produce one logit vector"
    );
    assert_eq!(max_delta, 0.0, "the quad band must be bit-identical");

    let rounds = 5usize;
    let mut quad_ms = Vec::with_capacity(rounds);
    let mut pair_ms = Vec::with_capacity(rounds);
    println!("round   quad ms   pair ms  pair/quad");
    for round in 0..rounds {
        // Alternate which arm runs first, so a monotonic host drift hits both arms equally.
        let (first, second) = if round % 2 == 0 {
            (run(&mut backend, true), run(&mut backend, false))
        } else {
            let pair = run(&mut backend, false);
            let quad = run(&mut backend, true);
            (quad, pair)
        };
        let (quad, pair) = (first.0, second.0);
        println!(
            "{round:>5}  {quad:>9.1}  {pair:>9.1}   {:>8.3}",
            pair / quad
        );
        quad_ms.push(quad);
        pair_ms.push(pair);
    }
    vipera_backend_cpu::pin_q3k_quad(None);
    vipera_backend_cpu::pin_q3k_pair(None);
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let quad = median(&mut quad_ms);
    let pair = median(&mut pair_ms);
    println!(
        "median: quad {quad:.1} ms ({:.1} tok/s), pair {pair:.1} ms ({:.1} tok/s), pair/quad {:>6.3}x ({:+.1}% for the quad)",
        n as f64 / (quad / 1e3),
        n as f64 / (pair / 1e3),
        pair / quad,
        100.0 * (pair / quad - 1.0)
    );
}

// ---------------------------------------------------------------------------
// 3b. Long prefill: 1024 and 4096 tokens, to read against `llama-bench -p 1024,4096`
// ---------------------------------------------------------------------------
//
// Batch path only: the sequential version above costs O(n) decode steps, so at 4k each round takes
// minutes. `timed_best` warms up once and keeps the fastest round — the KV is allocated by then, so
// no page faults remain.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_long_prompt() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();

    for (n, rounds) in [(1024usize, 2u32), (4096, 1)] {
        let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();
        let mut state = State::with_room(config, n + 4, KvCacheType::F16);
        let ms = timed_best(rounds, || {
            state.tokens.clear();
            backend
                .logits(&mut state.context(&prompt))
                .expect("prefill");
        });
        println!(
            "prefill n={n:>4}: {ms:>8.1} ms ({:>6.1} tok/s)",
            n as f64 / (ms / 1000.0)
        );
    }
    print_llama_hint("prefill 1024/4096");
}

/// The two attention kernels in one process, on the same 1024-token prompt.
///
/// The per-entry walk is the default; the tiled kernel (`Q_TILE=32`, `KV_TILE=64`) is measured at
/// −48% attention and −13% of a chunk set at 3.7k but stays opt-in, because its entry condition
/// makes a position's logits depend on chunk length (ADR-005). This is the instrument that can price
/// it per release: the arms have to alternate inside one process, since on this host a two-process
/// comparison moved a matvec phase the switch cannot touch by 2.5×.
///
/// The wall clock alone is a poor instrument at 1024 tokens (attention is a quarter of a chunk), so
/// run it with `VIPERA_PROFILE_STEP=1`: each `PROFILEPASS chunk` line carries the attention
/// milliseconds, and the `arm:` markers printed here say which kernel produced which lines. Roughly
/// 80 s per prefill, so two rounds is about 5 minutes.
#[test]
#[ignore]
fn prefill_attention_tile_vs_entry() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let n = 1024usize;
    let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();

    for round in 0..2 {
        for (label, tile) in [("entry", false), ("tile", true)] {
            vipera_backend_cpu::pin_attention_qblock(Some(tile));
            let mut state = State::with_room(config, n + 4, KvCacheType::F16);
            println!("arm: {label} round {round}");
            let start = Instant::now();
            backend
                .logits(&mut state.context(&prompt))
                .expect("prefill");
            println!(
                "arm total: {label} round {round}: {:.1} ms",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    vipera_backend_cpu::pin_attention_qblock(None);
}

// ---------------------------------------------------------------------------
// 4. Batched decode: one weight read for B activations (continuous batching)
// ---------------------------------------------------------------------------
//
// The piece's point: batching is how a bandwidth-bound decode wins, because the weight read is
// divided by B. Vipera batches up to 8; llama.cpp has `n_parallel` and batches too. Measures B=1,2,4,8
// through `logits_batch`.

#[test]
#[ignore = "measurement, not a correctness check"]
fn batched_decode_throughput() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();

    // Eight short prompts, each prefilled on its own, then batched decode.
    let prompts: [&[u32]; 8] = [
        &[12095, 13, 576],
        &[9906, 11, 1879, 315],
        &[17, 1234, 576, 6722, 315, 9625],
        &[315, 279, 6722],
        &[13, 576, 6722, 315],
        &[9625, 374, 1083],
        &[279, 6722, 315, 892],
        &[576, 6722, 315],
    ];
    let mut states: Vec<State> = Vec::new();
    let mut next = [0_u32; 8];
    for (i, p) in prompts.iter().enumerate() {
        let (s, t) = prefill_chunked(&mut backend, config, p, KvCacheType::F16);
        states.push(s);
        next[i] = t;
    }

    // Shared buffer like batch.rs: one allocation, one table per entry.
    let blocks_per_seq = states[0].table.len();
    let per_seq = states[0].kv.len();
    let mut shared_kv = vec![0u8; per_seq * states.len()];
    for (i, s) in states.iter().enumerate() {
        shared_kv[i * per_seq..(i + 1) * per_seq].copy_from_slice(&s.kv);
    }
    let tables: Vec<Vec<u32>> = (0..states.len())
        .map(|i| {
            (0..blocks_per_seq as u32)
                .map(|b| (i * blocks_per_seq) as u32 + b)
                .collect()
        })
        .collect();

    for &b in &[1usize, 2, 4, 8] {
        let toks: Vec<[u32; 1]> = (0..b).map(|i| [next[i]]).collect();
        let mut entries: Vec<vipera_runtime::BatchEntry<'_>> = (0..b)
            .map(|i| vipera_runtime::BatchEntry {
                tokens: &states[i].tokens,
                new_tokens: &toks[i],
                first_position: states[i].tokens.len() as u32,
                block_table: &tables[i],
                tokens_per_block: TOKENS_PER_BLOCK,
                bytes_per_token: states[i].bytes_per_token,
            })
            .collect();
        let mut batch = vipera_runtime::BatchContext {
            kv: &mut shared_kv,
            entries: &mut entries,
        };
        let ms = timed_best(3, || {
            backend.logits_batch(&mut batch).expect("batched decode");
        });
        println!(
            "B={b}: {ms:>6.1} ms/batch ({:.2} tok/s)",
            b as f64 * 1000.0 / ms
        );
    }
    print_llama_hint("batch");
}

// ---------------------------------------------------------------------------
// 5. Decode attention scaling: weights + attention * ctx (Mistral.rs 1.81x at large depth)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "measurement, not a correctness check"]
fn decode_attention_scaling() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    for ty in [KvCacheType::F16, KvCacheType::Q8_0] {
        if config.kv_bytes_per_token_for(ty).is_none() {
            println!("{ty:?} KV: this backend cannot store it, skipped");
            continue;
        }
        let mut steps: Vec<(usize, f64)> = Vec::new();
        for &length in &[32usize, 256, 1024, 2048, 4096, 8192] {
            let prompt: Vec<u32> = (0..length).map(|at| 1000 + (at % 400) as u32).collect();
            let (mut state, mut next) = prefill_chunked(&mut backend, config, &prompt, ty);
            let ms = timed_best(3, || {
                let logits = backend.logits(&mut state.context(&[next])).expect("decode");
                next = argmax(&logits) as u32;
            });
            steps.push((length, ms));
        }
        let base = steps[0].1;
        for (ctx, ms) in &steps {
            println!(
                "{ty:?} decode ctx {ctx:>5}: {ms:>6.1} ms/step ({:.1} tok/s), attention+KV {:>5.1} ms ({:.0}%)",
                1000.0 / ms,
                ms - base,
                100.0 * (ms - base) / ms
            );
        }
    }
    println!(
        "The Mistral.rs point: decode slows down as KV reads grow with ctx; vipera measures the same path against llama.cpp tg."
    );
    print_llama_hint("decode");
}

/// Decode at depth without paying for a prefill: the KV cache is allocated for the context and left
/// at zero (zeros are a legal f16 row, so the softmax stays finite), and the step is timed at that
/// position. Prefilling 32k takes close to an hour; this measures the same step in seconds.
///
/// ```text
/// cargo test -p vipera-backend-cpu --release --test win_vs_llama -- --ignored --nocapture \
///     decode_cost_at_depth_without_prefill
/// VIPERA_ATTN_WINDOW=4096 ...   # the same measurement over a sliding window
/// ```
#[test]
#[ignore = "measurement, not a correctness check"]
fn decode_cost_at_depth_without_prefill() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    for ctx in [4096usize, 8192, 16384, 32768] {
        let mut state = State::with_room(config, ctx + 4, KvCacheType::F16);
        state.tokens = vec![1000_u32; ctx];
        let mut next = 1000_u32;
        let ms = timed_best(3, || {
            let logits = backend
                .logits(&mut state.context(&[next]))
                .expect("decode at depth");
            next = argmax(&logits) as u32;
        });
        println!(
            "ctx {ctx:>6}: {ms:>7.1} ms/step ({:>6.1} tok/s)",
            1000.0 / ms
        );
    }
}

// ---------------------------------------------------------------------------
// Compare against llama.cpp when it is available
// ---------------------------------------------------------------------------

/// The same 32k decode step with the KV in a ring cache instead of a linear one.
///
/// The window kernel reads the sinks and the last `window` positions either way, so this measures
/// only what the ring changes: which physical block a row lives in. Run as
/// `VIPERA_ATTN_WINDOW=2048 cargo test --release -p vipera-backend-cpu --test win_vs_llama
///   -- --ignored --nocapture decode_cost_at_depth_with_ring_cache`.
#[test]
#[ignore = "measurement, not a correctness check"]
fn decode_cost_at_depth_with_ring_cache() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let Some(window) = attention_window() else {
        eprintln!("skipped: set VIPERA_ATTN_WINDOW to measure a ring cache");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let ctx = 32_768usize;
    for ring in [false, true] {
        let mut state = if ring {
            State::with_ring(
                config,
                ctx + 4,
                KvCacheType::F16,
                window.tokens as usize,
                window.sinks as usize,
            )
        } else {
            State::with_room(config, ctx + 4, KvCacheType::F16)
        };
        let kv_mb = state.kv.len() as f64 / (1 << 20) as f64;
        state.tokens = vec![1000_u32; ctx];
        let mut next = 1000_u32;
        let ms = timed_best(3, || {
            let logits = backend
                .logits(&mut state.context(&[next]))
                .expect("decode at depth");
            next = argmax(&logits) as u32;
        });
        println!(
            "ctx {ctx}: {} cache {kv_mb:>7.1} MB, {ms:>7.1} ms/step ({:>6.1} tok/s)",
            if ring { "ring  " } else { "linear" },
            1000.0 / ms
        );
    }
}

/// How far a verify pass's first row is from the same token decoded alone.
///
/// `logits_each` runs one entry per token of the chunk, all of them batched, while plain decoding
/// runs the first token through `logits` on its own. The `run_batch_n` doc says an entry's logits are
/// bit-identical to `logits` for that context alone; this measures that claim at depth. It holds
/// with `VIPERA_THREADS=1` and does not hold with more: the batched kernels reduce across threads
/// differently from the single-sequence ones, and by depth the difference is large enough to move an
/// argmax, which is why speculative decoding cannot be claimed at a deep context yet.
#[test]
#[ignore = "measurement, not a correctness check"]
fn a_verify_pass_matches_a_single_token_pass() {
    let Some(path) = model_path() else {
        return;
    };
    let depth: usize = std::env::var("VIPERA_SPEC_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);
    const CHUNK: usize = 128;
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let window = attention_window();
    let room = depth + 4 * 32 + CHUNK;
    let mut state = match window {
        Some(window) => State::with_ring(
            config,
            room,
            KvCacheType::F16,
            window.tokens as usize,
            window.sinks as usize,
        ),
        None => State::with_room(config, room, KvCacheType::F16),
    };
    let pattern: [u32; 10] = [9707, 11, 1879, 264, 402, 5297, 13, 576, 682, 304];
    let mut haystack = Vec::with_capacity(depth);
    while haystack.len() < depth {
        haystack.extend_from_slice(&pattern);
    }
    haystack.truncate(depth);
    let mut logits = Vec::new();
    for chunk in haystack.chunks(CHUNK) {
        logits = backend
            .logits(&mut state.context(chunk))
            .expect("prefill fits");
    }
    state.tokens.extend_from_slice(&haystack);
    let first = argmax(&logits) as u32;
    let chunk = [first, 13, 576, 682, 304];

    let mut alone = state.clone();
    let single = backend
        .logits(&mut alone.context(&chunk[..1]))
        .expect("single pass");

    for len in 1..=chunk.len() {
        let mut batched = state.clone();
        let each = backend
            .logits_each(&mut batched.context(&chunk[..len]))
            .expect("verify pass");
        let worst = single
            .iter()
            .zip(&each[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "depth {depth}: chunk {len}, row 0 vs single pass, max |delta| {worst:.3e}, argmax {} vs {} ({} threads)",
            argmax(&single),
            argmax(&each[0]),
            std::env::var("VIPERA_THREADS").unwrap_or_else(|_| "default".into()),
        );
    }
}

/// Prompt-lookup speculation at a deep context, where the step is dominated by the weight stream.
///
/// Unlike `decode_cost_at_depth_without_prefill`, this really fills the KV: the history is prefilled
/// in 128-token chunks, so a ring cache has to survive a chunk that reads a window still being
/// written. Run as `VIPERA_ATTN_WINDOW=2048 cargo test --release -p vipera-backend-cpu --test
/// win_vs_llama -- --ignored --nocapture speculative_decode_at_depth`.
///
/// It prints instead of asserting: at a deep context the verify pass is not bit-identical to plain
/// decoding with more than one thread (`a_verify_pass_matches_a_single_token_pass`), so the stream
/// it reports is not the stream plain decoding would produce, and that has to be fixed before a
/// speedup here counts.
#[test]
#[ignore = "measurement, not a correctness check"]
fn speculative_decode_at_depth() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let depth: usize = std::env::var("VIPERA_SPEC_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32_768);
    const STEPS: usize = 32;
    const CHUNK: usize = 128;
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let window = attention_window();
    let room = depth + 4 * STEPS + CHUNK;
    let mut state = match window {
        Some(window) => State::with_ring(
            config,
            room,
            KvCacheType::F16,
            window.tokens as usize,
            window.sinks as usize,
        ),
        None => State::with_room(config, room, KvCacheType::F16),
    };
    // A ten-token period, repeated to the depth being measured: prompt-lookup has something to
    // find, and the history is long enough for the window to be doing its work.
    let pattern: [u32; 10] = [9707, 11, 1879, 264, 402, 5297, 13, 576, 682, 304];
    let mut haystack = Vec::with_capacity(depth);
    while haystack.len() < depth {
        haystack.extend_from_slice(&pattern);
    }
    haystack.truncate(depth);

    let start = Instant::now();
    let mut logits = Vec::new();
    for chunk in haystack.chunks(CHUNK) {
        logits = backend
            .logits(&mut state.context(chunk))
            .expect("prefill fits");
    }
    let prefill = start.elapsed().as_secs_f64();
    state.tokens.extend_from_slice(&haystack);
    println!(
        "prefill {depth} tokens: {prefill:.1} s ({:.1} tok/s), KV {:.1} MB in {} blocks",
        depth as f64 / prefill,
        state.kv.len() as f64 / f64::from(1 << 20),
        state.table.len(),
    );
    let first = argmax(&logits) as u32;

    let mut plain_state = state.clone();
    let start = Instant::now();
    let plain = decode_plain(&mut backend, &mut plain_state, first, STEPS);
    let plain_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Each round counts its accepted drafts, because at depth the number that matters is how many
    // tokens one weight pass returns, not how fast the pass is.
    let mut runs = Vec::new();
    for (n, k) in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 4)] {
        let mut spec_state = state.clone();
        let mut out: Vec<u32> = Vec::with_capacity(STEPS);
        let mut next = first;
        let (mut drafts, mut accepted_total, mut rounds) = (0_usize, 0_usize, 0_usize);
        let start = Instant::now();
        while out.len() < STEPS {
            let draft = draft_from_ngram(&spec_state.tokens, n, k);
            let mut chunk = Vec::with_capacity(draft.len() + 1);
            chunk.push(next);
            chunk.extend_from_slice(&draft);
            let answers = backend
                .logits_each(&mut spec_state.context(&chunk))
                .expect("verify fits");
            rounds += 1;
            drafts += draft.len();
            let mut accepted = 0;
            while accepted < draft.len() && accepted + 1 < answers.len() {
                let pick = argmax(&answers[accepted]) as u32;
                if pick != draft[accepted] {
                    break;
                }
                accepted += 1;
                out.push(pick);
                if out.len() == STEPS {
                    break;
                }
            }
            accepted_total += accepted;
            let tail = argmax(&answers[accepted]) as u32;
            spec_state.tokens.extend_from_slice(&chunk[..=accepted]);
            out.push(tail);
            next = tail;
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  depth {depth}, draft n={n} k={k}: {ms:.1} ms ({:.1} tok/s), speedup {:.2}x, \
             {rounds} passes for {STEPS} tokens, {accepted_total}/{drafts} drafts accepted, identical {}",
            STEPS as f64 * 1000.0 / ms,
            plain_ms / ms,
            out[..out.len().min(STEPS)] == plain[..plain.len().min(STEPS)]
        );
        runs.push((ms, n, k));
    }

    let (spec_ms, n, k) = runs
        .iter()
        .copied()
        .fold((f64::INFINITY, 0, 0), |best, run| {
            if run.0 < best.0 { run } else { best }
        });
    println!(
        "depth {depth}: plain {:.1} tok/s, best draft n={n} k={k} {:.1} tok/s, speedup {:.2}x",
        STEPS as f64 * 1000.0 / plain_ms,
        STEPS as f64 * 1000.0 / spec_ms,
        plain_ms / spec_ms
    );
}

fn print_llama_hint(what: &str) {
    let bench = std::env::var_os("LLAMA_BENCH")
        .map(PathBuf::from)
        .filter(|p| p.exists());
    let in_path = std::process::Command::new("which")
        .arg("llama-bench")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if bench.is_some() || in_path {
        println!(
            "[llama.cpp] llama-bench is available; run: llama-bench -m $VIPERA_TEST_MODEL -p 512 -n 128 -t 8 --json | jq .  then compare the pp/tg columns with vipera's (test: {what})."
        );
    } else {
        println!(
            "[llama.cpp] chua cai; build CPU-only: git clone https://github.com/ggerganov/llama.cpp && cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON && cmake --build build -j8 --target llama-bench"
        );
    }
    // If LLAMA_BENCH points at a binary, actually run it and print both side by side.
    if let Some(bin) = bench {
        let model =
            model_path().map_or_else(|| "<no-model>".to_string(), |p| p.display().to_string());
        let out = std::process::Command::new(&bin)
            .args(["-m", &model, "-p", "512", "-n", "128", "-t", "8"])
            .output();
        match out {
            Ok(o) if o.status.success() => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                // llama-bench prints its results to stderr; filter the pp/tg lines.
                for line in stderr.lines() {
                    if line.contains("pp512")
                        || line.contains("tg128")
                        || line.contains("pp")
                        || line.contains("tg")
                    {
                        println!("[llama-bench] {line}");
                    }
                }
                if stderr.chars().count() > 0 && !stderr.contains("pp") && !stderr.contains("tg") {
                    println!(
                        "[llama-bench] {}",
                        stderr.lines().take(20).collect::<Vec<_>>().join("\n")
                    );
                }
            }
            Ok(o) => println!(
                "[llama-bench] exit {}: {}",
                o.status,
                String::from_utf8_lossy(&o.stderr)
                    .lines()
                    .take(5)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Err(e) => println!("[llama-bench] failed to run: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 9. Speculation groundwork: every position's logits must match a separate pass
// ---------------------------------------------------------------------------
//
// Prompt-lookup speculation verifies several proposed tokens in one batch pass, which is only
// sound if entry `i` of that pass answers exactly what a pass over that token alone would.

#[test]
fn logits_each_matches_single_passes() {
    let Some(path) = model_path() else {
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let prompt: Vec<u32> = (0..8).map(|at| 1000 + (at % 400) as u32).collect();

    let mut chunk = State::with_room(config, prompt.len() + 4, KvCacheType::F16);
    let each = backend
        .logits_each(&mut chunk.context(&prompt))
        .expect("chunk fits");
    assert_eq!(each.len(), prompt.len(), "one answer per position");

    for at in 0..prompt.len() {
        let mut alone = State::with_room(config, prompt.len() + 4, KvCacheType::F16);
        let expected = backend
            .logits(&mut alone.context(&prompt[..=at]))
            .expect("prefix fits");
        assert_eq!(each[at], expected, "position {at} differs");
    }
}

// ---------------------------------------------------------------------------
// 10. Speculation payoff: what a 5-token verify pass costs, and how often it pays
// ---------------------------------------------------------------------------

#[test]
#[ignore = "measurement, not a correctness check"]
fn speculative_payoff() {
    let Some(path) = model_path() else {
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    // A repetitive prompt: the kind of text prompt-lookup speculation exists for.
    let prompt: Vec<u32> = (0..64).map(|at| 1000 + (at % 12) as u32).collect();
    let (state, first) = prefill_chunked(&mut backend, config, &prompt, KvCacheType::F16);

    // Reference greedy stream, one token per pass, exactly today's decode.
    let mut greedy = vec![first];
    let mut probe = state.clone();
    for _ in 0..6 {
        let token = *greedy.last().expect("non-empty");
        let logits = backend
            .logits(&mut probe.context(&[token]))
            .expect("decode fits");
        probe.tokens.push(token);
        greedy.push(argmax(&logits) as u32);
    }

    // One verify pass over the same 6 tokens must answer the same thing, and its wall time is
    // what a speculation round pays.
    let single = timed_best(3, || {
        let mut s = state.clone();
        for token in &greedy[..5] {
            let _ = backend.logits(&mut s.context(&[*token]));
            s.tokens.push(*token);
        }
    });
    let verify = timed_best(3, || {
        let mut s = state.clone();
        let _ = backend
            .logits_each(&mut s.context(&greedy[..5]))
            .expect("verify fits");
    });

    // Draft from the n-gram: repeat of the last 3 tokens found earlier in the prompt.
    let context: Vec<u32> = state
        .tokens
        .iter()
        .copied()
        .chain(greedy.iter().copied())
        .collect();
    let mut accepted = 0;
    let mut tried = 0;
    for at in 3..context.len().saturating_sub(4) {
        let needle = &context[at - 3..at];
        if let Some(pos) = (0..at - 3)
            .rev()
            .find(|start| &context[*start..*start + 3] == needle)
        {
            let draft = &context[pos + 3..(pos + 6).min(context.len())];
            if !draft.is_empty() {
                tried += 1;
                let hit = draft
                    .iter()
                    .zip(&context[at..])
                    .take_while(|(d, t)| d == t)
                    .count();
                accepted += hit;
            }
        }
    }
    let rate = if tried == 0 {
        0.0
    } else {
        accepted as f64 / tried as f64
    };
    println!(
        "speculation: 5 single passes {single:.1} ms, one 5-token verify {verify:.1} ms, \
         verify/5-single {:.2}x, n-gram acceptance {rate:.2} ({} drafts)",
        verify / single,
        tried
    );
}

// ---------------------------------------------------------------------------
// 11. Prompt-lookup speculation end to end: same tokens, fewer passes
// ---------------------------------------------------------------------------

fn draft_from_ngram(context: &[u32], n: usize, k: usize) -> Vec<u32> {
    if context.len() < n + 1 {
        return Vec::new();
    }
    let needle = &context[context.len() - n..];
    for start in (0..context.len() - n).rev() {
        if &context[start..start + n] == needle {
            let after = start + n;
            if after < context.len() {
                return context[after..(after + k).min(context.len())].to_vec();
            }
        }
    }
    Vec::new()
}

/// Greedy continuation, one forward pass per token.
fn decode_plain(backend: &mut CpuBackend, state: &mut State, start: u32, steps: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(steps);
    let mut next = start;
    for _ in 0..steps {
        let logits = backend
            .logits(&mut state.context(&[next]))
            .expect("decode fits");
        state.tokens.push(next);
        next = argmax(&logits) as u32;
        out.push(next);
    }
    out
}

/// Greedy continuation with prompt-lookup drafts verified one pass per round.
fn decode_speculative(
    backend: &mut CpuBackend,
    state: &mut State,
    start: u32,
    steps: usize,
    n: usize,
    k: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(steps);
    let mut next = start;
    while out.len() < steps {
        let draft = draft_from_ngram(&state.tokens, n, k);
        let mut chunk = Vec::with_capacity(draft.len() + 1);
        chunk.push(next);
        chunk.extend_from_slice(&draft);
        let answers = backend
            .logits_each(&mut state.context(&chunk))
            .expect("verify fits");
        // answers[i] is the model's pick after chunk[..=i]; it must equal chunk[i + 1] to accept.
        let mut accepted = 0;
        while accepted < draft.len() && accepted + 1 < answers.len() {
            let pick = argmax(&answers[accepted]) as u32;
            if pick != draft[accepted] {
                break;
            }
            accepted += 1;
            out.push(pick);
            if out.len() == steps {
                break;
            }
        }
        // The token after the last accepted draft is the model's own answer, always usable.
        let tail = argmax(&answers[accepted]) as u32;
        // `chunk[0]` is `next`, which the previous round already emitted: only the accepted part
        // of the chunk joins the context, and `tail` stays pending until the next round feeds it
        // as that round's `chunk[0]`. Pushing it here would append it twice — the context would
        // then be one token longer than the KV that was filled for it, and logits drift.
        state.tokens.extend_from_slice(&chunk[..=accepted]);
        out.push(tail);
        next = tail;
    }
    out.truncate(steps);
    out
}

/// Same loop, but the draft length follows the last round: shrink on a miss, grow on a full hit.
fn decode_adaptive(
    backend: &mut CpuBackend,
    state: &mut State,
    start: u32,
    steps: usize,
) -> Vec<u32> {
    const N: usize = 2;
    let mut out = Vec::with_capacity(steps);
    let mut next = start;
    let mut k = 4;
    while out.len() < steps {
        let draft = draft_from_ngram(&state.tokens, N, k);
        let mut chunk = Vec::with_capacity(draft.len() + 1);
        chunk.push(next);
        chunk.extend_from_slice(&draft);
        let answers = backend
            .logits_each(&mut state.context(&chunk))
            .expect("verify fits");
        let mut accepted = 0;
        while accepted < draft.len() && accepted + 1 < answers.len() {
            let pick = argmax(&answers[accepted]) as u32;
            if pick != draft[accepted] {
                break;
            }
            accepted += 1;
            out.push(pick);
            if out.len() == steps {
                break;
            }
        }
        let tail = argmax(&answers[accepted]) as u32;
        // As in `decode_speculative`: `next` is already in the context, so only the accepted part
        // of the chunk is appended, and `tail` is fed as the next round's first token.
        state.tokens.extend_from_slice(&chunk[..=accepted]);
        out.push(tail);
        next = tail;
        k = if accepted == 0 {
            k.saturating_sub(1).max(1)
        } else if accepted == draft.len() {
            (k + 1).min(8)
        } else {
            k
        };
    }
    out.truncate(steps);
    out
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn speculative_decode_wins() {
    let Some(path) = model_path() else {
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let prompt: Vec<u32> = (0..64).map(|at| 1000 + (at % 12) as u32).collect();
    const STEPS: usize = 32;
    // Room for the prompt plus every token both loops may compute, including drafts that are
    // thrown away.
    let mut state = State::with_room(config, prompt.len() + 4 * STEPS, KvCacheType::F16);
    let logits = backend
        .logits(&mut state.context(&prompt))
        .expect("prompt fits");
    state.tokens.extend_from_slice(&prompt);
    let first = argmax(&logits) as u32;

    let mut plain_state = state.clone();
    let start = Instant::now();
    let plain = decode_plain(&mut backend, &mut plain_state, first, STEPS);
    let plain_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut best = (f64::INFINITY, 0, 0, Vec::new());
    for (n, k) in [(1, 1), (1, 2), (1, 3), (2, 2)] {
        let mut spec_state = state.clone();
        let start = Instant::now();
        let stream = decode_speculative(&mut backend, &mut spec_state, first, STEPS, n, k);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  draft n={n} k={k}: {ms:.1} ms ({:.1} t/s), identical {}",
            STEPS as f64 * 1000.0 / ms,
            stream == plain
        );
        assert_eq!(stream, plain, "speculation changed the greedy stream");
        if ms < best.0 {
            best = (ms, n, k, stream);
        }
    }
    let mut adaptive_state = state.clone();
    let start = Instant::now();
    let adaptive = decode_adaptive(&mut backend, &mut adaptive_state, first, STEPS);
    let adaptive_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  adaptive: {adaptive_ms:.1} ms ({:.1} t/s), identical {}",
        STEPS as f64 * 1000.0 / adaptive_ms,
        adaptive == plain
    );
    assert_eq!(
        adaptive, plain,
        "adaptive speculation changed the greedy stream"
    );

    let (spec_ms, n, k, speculative) = best;
    println!("  best draft n={n} k={k}");

    println!(
        "decode {STEPS} tokens: plain {plain_ms:.1} ms ({:.1} t/s), speculative {spec_ms:.1} ms \
         ({:.1} t/s), speedup {:.2}x, identical tokens {}",
        STEPS as f64 * 1000.0 / plain_ms,
        STEPS as f64 * 1000.0 / spec_ms,
        plain_ms / spec_ms,
        plain == speculative
    );
    assert_eq!(plain, speculative, "speculation changed the greedy stream");
}

// ---------------------------------------------------------------------------
// 3c. One 128-token chunk, for `scripts/q3k-band-drift.sh`
// ---------------------------------------------------------------------------
//
// The drift check needs exactly one prompt chunk with the phase timers on: a 512-token prefill is
// four chunks and takes four times as long, and `prefill_long_prompt` is minutes. The timers are
// switched on by the script through `VIPERA_PROFILE_STEP`, not here — the workspace forbids
// `set_var`, and the engine reads the switch once per process anyway.

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefill_one_chunk() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load model");
    let config = backend.config();
    let n = 128usize;
    let prompt: Vec<u32> = (0..n).map(|at| 1000 + (at % 400) as u32).collect();
    let mut state = State::with_room(config, n + 4, KvCacheType::F16);
    let start = Instant::now();
    backend
        .logits(&mut state.context(&prompt))
        .expect("prefill one chunk");
    println!(
        "one chunk: {n} tokens in {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
}
