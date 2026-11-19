//! What a repeated prompt costs: two requests that share a prompt, one engine with the prefix
//! cache on and one with it off.
//!
//! Both tests assert — a copied prefix must produce the tokens a computed prefix produces, for every
//! KV type — so they are not skipped for being measurements. They are `#[ignore]`d only because a
//! debug build cannot prefill 480 tokens in a usable time: the gate runs them in release with
//! `--ignored`, and without a model they print the reason and return, the way the other
//! model-dependent tests here do. With a model, the printed time and `processed` columns show the
//! reuse as well as checking it:
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

fn engine(path: &PathBuf, cache: bool, kv_type: KvCacheType) -> Engine<CpuBackend> {
    let bytes = std::fs::read(path).expect("model bytes");
    let file = GgufFile::parse(&bytes).expect("gguf");
    let backend = CpuBackend::from_bytes(bytes, &file).expect("backend");
    let bytes_per_token = backend
        .config()
        .kv_bytes_per_token_for(kv_type)
        .expect("kv bytes per token");
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
#[ignore = "prefill is a release-mode measurement; the gate runs it with --ignored --release"]
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
        let mut engine = engine(&path, cache, KvCacheType::F16);
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

/// The cache is a byte copy of another request's KV, so it is only sound if a position's stored
/// KV depends on the token and the position and not on the chunk it arrived in. F16 KV stores the
/// row as computed; Q8_0 and Q4_0 quantize it, so this asks the same question of each type.
fn supported(path: &PathBuf, kv_type: KvCacheType) -> bool {
    let bytes = std::fs::read(path).expect("model bytes");
    let file = GgufFile::parse(&bytes).expect("gguf");
    let backend = CpuBackend::from_bytes(bytes, &file).expect("backend");
    backend.config().kv_bytes_per_token_for(kv_type).is_some()
}
#[test]
#[ignore = "prefill is a release-mode measurement; the gate runs it with --ignored --release"]
fn prefix_cache_gives_the_same_tokens_for_every_kv_type() {
    let Some(path) = model_path() else {
        eprintln!("no model: set VIPERA_TEST_MODEL");
        return;
    };
    let shared: Vec<u32> = (0..480u32).map(|i| 1000 + (i * 37) % 20_000).collect();
    let tail = [11u32, 22, 33, 44];

    for kv_type in [KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0] {
        if !supported(&path, kv_type) {
            println!("{kv_type:?} KV: this backend cannot store it, skipped");
            continue;
        }
        let mut produced = Vec::new();
        for cache in [false, true] {
            let mut engine = engine(&path, cache, kv_type);
            request(&mut engine, &shared, &[500, 501, 502, 503], 1);
            produced.push(request(&mut engine, &shared, &tail, 12));
        }
        assert_eq!(
            produced[0].tokens, produced[1].tokens,
            "{kv_type:?}: a copied prefix changed the tokens"
        );
        println!(
            "{kv_type:?} KV: reused {:>3}, tokens {:?}",
            produced[1].reused, produced[1].tokens
        );
    }
}
