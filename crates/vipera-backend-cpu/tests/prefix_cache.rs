//! What a repeated prompt costs: two requests that share a prompt, one engine with the prefix
//! cache on and one with it off.
//!
//! This is a measurement, not a correctness check; `#[ignore]`d like the other benches here. With
//! the cache on, the second request copies the KV of the tokens it shares with the first and only
//! computes the tail, so the two columns should differ by the size of the shared prefix.
//!
//!   VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test -p vipera-backend-cpu --release --test prefix_cache -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use vipera_backend_cpu::CpuBackend;
use vipera_core::gguf::GgufFile;
use vipera_core::planner::KvCacheType;
use vipera_runtime::engine::{Engine, EngineConfig};
use vipera_runtime::kv::{DEFAULT_TOKENS_PER_BLOCK, KvCache, KvConfig};

fn model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VIPERA_TEST_MODEL") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    path.exists().then_some(path)
}

fn engine(path: &PathBuf, cache: bool) -> Engine<CpuBackend> {
    let bytes = std::fs::read(path).expect("model bytes");
    let file = GgufFile::parse(&bytes).expect("gguf");
    let backend = CpuBackend::from_bytes(bytes, &file).expect("backend");
    let bytes_per_token = backend
        .config()
        .kv_bytes_per_token_for(KvCacheType::F16)
        .expect("f16 kv");
    let context = 2048;
    let kv = KvCache::new(
        KvConfig::from_bytes(
            context * bytes_per_token,
            bytes_per_token,
            DEFAULT_TOKENS_PER_BLOCK,
        )
        .expect("kv config"),
    )
    .expect("kv cache");
    let mut engine = Engine::new(
        kv,
        backend,
        EngineConfig {
            recent_window: 128,
            max_batch: 1,
            speculative: true,
            ..EngineConfig::default()
        },
    );
    engine.set_prefix_cache(cache);
    engine
}

/// One request of `shared` + `tail` prompt tokens: how long it took, how many tokens it really
/// computed, and what it produced.
struct Measured {
    ms: f64,
    processed: u64,
    reused: usize,
    tokens: Vec<u32>,
}

fn request(
    engine: &mut Engine<CpuBackend>,
    shared: &[u32],
    tail: &[u32],
    max_new: u32,
) -> Measured {
    let mut prompt = shared.to_vec();
    prompt.extend_from_slice(tail);
    let id = engine.admit(&prompt, max_new).expect("admit");
    let started = Instant::now();
    let generation = engine.generate(id).expect("generate");
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    engine.release(id).expect("release");
    Measured {
        ms,
        processed: generation.processed_tokens,
        reused: generation.reused_tokens,
        tokens: generation.tokens,
    }
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn prefix_cache_reuses_a_shared_prompt() {
    let Some(path) = model_path() else {
        eprintln!("no model: set VIPERA_TEST_MODEL");
        return;
    };
    // 480 tokens that are a plausible shared system prompt plus 4 that differ.
    let shared: Vec<u32> = (0..480u32).map(|i| 1000 + (i * 37) % 20_000).collect();
    let tail = [11u32, 22, 33, 44];

    let mut produced = Vec::new();
    for cache in [false, true] {
        let mut engine = engine(&path, cache);
        // Request one seeds the KV (parked when the cache is on); its time is not measured.
        request(&mut engine, &shared, &[500, 501, 502, 503], 1);
        // One generated token keeps the measurement to the prompt pass plus a single decode.
        let quick = request(&mut engine, &shared, &tail, 1);
        println!(
            "prefix cache {:<3}: {:>8.1} ms, processed {:>4} tokens, reused {:>4}",
            if cache { "on" } else { "off" },
            quick.ms,
            quick.processed,
            quick.reused,
        );
        // A turn that generates: the copied prefix must give the tokens a computed one gives.
        produced.push(request(&mut engine, &shared, &tail, 12).tokens);
    }
    assert_eq!(
        produced[0], produced[1],
        "a copied prefix must not change the tokens"
    );
    println!(
        "twelve tokens from a copied prefix are the ones a computed prefix gives: {:?}",
        produced[1]
    );
}
