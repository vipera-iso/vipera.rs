//! The batched decode step must produce, for each sequence, exactly the logits that sequence gets
//! when it runs alone.
//!
//! That is the claim the batched kernels rest on: one weight read shared across `B` activations may
//! not change a single bit of any sequence's arithmetic. The check needs a real model, so it is
//! skipped — with the reason printed — when none is available. Run it with `--nocapture` to see
//! whether it ran:
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! cargo test -p vipera-backend-cpu --test batch -- --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use vipera_backend_cpu::CpuBackend;
use vipera_backend_cpu::ModelConfig;
use vipera_runtime::{BatchContext, BatchEntry, ForwardContext, ForwardPass};

/// Tokens per KV block; the runtime's default, and any positive value works here.
const TOKENS_PER_BLOCK: u64 = 16;
/// Blocks one sequence owns in these tests: room for 64 positions at 16 tokens a block.
const BLOCKS: usize = 4;

/// A sequence and the KV state it has written so far, in a buffer of its own.
#[derive(Clone)]
struct State {
    tokens: Vec<u32>,
    kv: Vec<u8>,
    table: Vec<u32>,
    bytes_per_token: u64,
}

impl State {
    fn new(config: ModelConfig) -> Self {
        let bytes_per_token = config.kv_bytes_per_token();
        Self {
            tokens: Vec::new(),
            kv: vec![0; BLOCKS * bytes_per_token as usize * TOKENS_PER_BLOCK as usize],
            table: (0..BLOCKS as u32).collect(),
            bytes_per_token,
        }
    }

    /// A state with room for `positions` tokens, for measurements that need a long context.
    fn with_room(config: ModelConfig, positions: usize) -> Self {
        let blocks = positions.div_ceil(TOKENS_PER_BLOCK as usize);
        let bytes_per_token = config.kv_bytes_per_token();
        Self {
            tokens: Vec::new(),
            kv: vec![0; blocks * bytes_per_token as usize * TOKENS_PER_BLOCK as usize],
            table: (0..blocks as u32).collect(),
            bytes_per_token,
        }
    }

    /// A context that computes `new_tokens` starting after everything already retained.
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

/// One buffer holding every entry's KV in its own block range, plus the block tables that point into
/// it. That is the shape a runtime has — one allocation, one table per sequence — and the shape a
/// batch has to work against: entries cannot each own a `&mut [u8]` of the same allocation.
struct Shared {
    kv: Vec<u8>,
    tables: Vec<Vec<u32>>,
}

impl Shared {
    /// Concatenate each state's KV into one buffer, giving entry `e` blocks `e * BLOCKS..`.
    fn over(states: &[State]) -> Self {
        let per_sequence = states.first().map_or(0, |state| state.kv.len());
        let mut kv = vec![0_u8; per_sequence * states.len()];
        for (entry, state) in states.iter().enumerate() {
            kv[entry * per_sequence..(entry + 1) * per_sequence].copy_from_slice(&state.kv);
        }
        let tables = (0..states.len())
            .map(|entry| {
                let base = (entry * BLOCKS) as u32;
                (0..BLOCKS as u32).map(|block| base + block).collect()
            })
            .collect();
        Self { kv, tables }
    }
}

/// One batch entry per state, each decoding the tokens it is handed at its own position.
fn entries<'a>(
    states: &'a [State],
    tables: &'a [Vec<u32>],
    new_tokens: &[&'a [u32]],
) -> Vec<BatchEntry<'a>> {
    (0..states.len())
        .map(|entry| BatchEntry {
            tokens: &states[entry].tokens,
            new_tokens: new_tokens[entry],
            first_position: states[entry].tokens.len() as u32,
            block_table: &tables[entry],
            tokens_per_block: TOKENS_PER_BLOCK,
            bytes_per_token: states[entry].bytes_per_token,
        })
        .collect()
}

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

/// Prefill `prompt` on a fresh sequence and return its state plus the next token.
fn prefill(backend: &mut CpuBackend, config: ModelConfig, prompt: &[u32]) -> (State, u32) {
    let mut state = State::new(config);
    let logits = backend
        .logits(&mut state.context(prompt))
        .expect("the prompt fits the sequence and this model");
    state.tokens.extend_from_slice(prompt);
    (state, argmax(&logits) as u32)
}

#[test]
fn a_batched_step_matches_every_sequence_run_alone() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load the model");
    let config = backend.config();

    let prompts: [&[u32]; 3] = [
        &[12095, 13, 576],
        &[9906, 11, 1879, 315],
        &[17, 1234, 576, 6722, 315, 9625],
    ];
    let (base_a, next_a) = prefill(&mut backend, config, prompts[0]);
    let (base_b, next_b) = prefill(&mut backend, config, prompts[1]);
    let (base_c, next_c) = prefill(&mut backend, config, prompts[2]);

    // The same one-token decode, first one at a time, then as a batch of three.
    let tokens: [&[u32]; 3] = [&[next_a], &[next_b], &[next_c]];
    let mut alone_a = base_a.clone();
    let mut alone_b = base_b.clone();
    let mut alone_c = base_c.clone();
    let expected = [
        backend
            .logits(&mut alone_a.context(tokens[0]))
            .expect("decode"),
        backend
            .logits(&mut alone_b.context(tokens[1]))
            .expect("decode"),
        backend
            .logits(&mut alone_c.context(tokens[2]))
            .expect("decode"),
    ];

    let states = [base_a, base_b, base_c];
    let mut shared = Shared::over(&states);
    let mut batch_entries = entries(&states, &shared.tables, &tokens);
    let mut batch = BatchContext {
        kv: shared.kv.as_mut_slice(),
        entries: batch_entries.as_mut_slice(),
    };
    let actual = backend.logits_batch(&mut batch).expect("batched decode");

    assert_eq!(actual.len(), expected.len());
    for (entry, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "entry {entry} returned a different vocabulary size"
        );
        for (at, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "entry {entry}, logit {at}: batched {actual} vs alone {expected}"
            );
        }
    }
}

#[test]
fn a_batch_of_one_is_the_single_sequence_path() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load the model");
    let config = backend.config();
    let prompt: &[u32] = &[12095, 13, 576, 6722];

    let mut single = State::new(config);
    let alone = backend
        .logits(&mut single.context(prompt))
        .expect("a prompt is one sequence");

    let states = [State::new(config)];
    let mut shared = Shared::over(&states);
    let mut batch_entries = entries(&states, &shared.tables, &[prompt]);
    let mut batch = BatchContext {
        kv: shared.kv.as_mut_slice(),
        entries: batch_entries.as_mut_slice(),
    };
    let batched = backend
        .logits_batch(&mut batch)
        .expect("a batch of one is a single sequence");

    assert_eq!(batched.len(), 1);
    assert_eq!(
        batched[0]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        alone
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "a batch of one must not move a single bit"
    );
}

/// Measurement, not a correctness check: what a batched decode step is worth on the real model,
/// batched against the same number of sequences run one after another. The kernel bench answers the
/// per-tensor question; this answers the per-step one, attention and the KV gather included. Run
/// with:
///
/// ```text
/// VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
/// cargo test -p vipera-backend-cpu --release --test batch -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measurement, not a correctness check"]
fn batched_step_speedup_on_the_real_model() {
    const ROUNDS: u32 = 3;

    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load the model");
    let config = backend.config();

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
    // Each sequence is prefilled alone, then every step below decodes one token per entry.
    let mut next = [0_u32; 8];
    let mut states: Vec<State> = Vec::new();
    for (entry, prompt) in prompts.iter().enumerate() {
        let (state, token) = prefill(&mut backend, config, prompt);
        next[entry] = token;
        states.push(state);
    }
    let mut shared = Shared::over(&states);

    let mut batched_ms = [f64::INFINITY; 4];
    let mut serial_ms = [f64::INFINITY; 4];
    for (index, size) in [1_usize, 2, 4, 8].into_iter().enumerate() {
        // Three rounds, fastest wins: a shared machine measures the scheduler as much as the step.
        for _ in 0..3 {
            let new_tokens: Vec<&[u32]> = (0..size).map(|entry| &next[entry..=entry]).collect();
            let mut batch_entries = entries(&states[..size], &shared.tables[..size], &new_tokens);
            let mut batch = BatchContext {
                kv: shared.kv.as_mut_slice(),
                entries: batch_entries.as_mut_slice(),
            };
            let started = Instant::now();
            for _ in 0..ROUNDS {
                backend.logits_batch(&mut batch).expect("batched decode");
            }
            batched_ms[index] = batched_ms[index].min(started.elapsed().as_secs_f64());
        }

        for _ in 0..3 {
            let started = Instant::now();
            for _ in 0..ROUNDS {
                for entry in 0..size {
                    backend
                        .logits(&mut states[entry].context(&next[entry..=entry]))
                        .expect("single decode");
                }
            }
            serial_ms[index] = serial_ms[index].min(started.elapsed().as_secs_f64());
        }

        let batched = batched_ms[index] / f64::from(ROUNDS) * 1000.0;
        let serial = serial_ms[index] / f64::from(ROUNDS) * 1000.0;
        println!(
            "B={size}: batched {batched:>7.1} ms/step, serial {serial:>7.1} ms/step, \
             {:>5.2}x, {:>6.2} tok/s",
            serial / batched,
            size as f64 * 1000.0 / batched,
        );
    }
}

/// Where a decode step's time goes. A step streams the same weights whatever the context length, so
/// the gap between a step at 32 keys and a step at 1024 keys is attention and the KV it reads — which
/// is the whole budget any flash-attention kernel could claim. Run with:
///
/// ```text
/// VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
/// cargo test -p vipera-backend-cpu --release --test batch -- --ignored --nocapture
/// ```
#[test]
#[ignore = "measurement, not a correctness check"]
fn what_a_decode_step_spends_on_attention() {
    const ROUNDS: u32 = 4;

    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL");
        return;
    };
    let mut backend = CpuBackend::load(&path).expect("load the model");
    let config = backend.config();

    let mut steps: Vec<(usize, f64)> = Vec::new();
    for length in [32_usize, 256, 1024] {
        let prompt: Vec<u32> = (0..length).map(|at| 1000 + (at % 400) as u32).collect();
        // Room for the prompt plus the warm-up and measured steps below.
        let mut state = State::with_room(config, length + ROUNDS as usize + 1);
        let logits = backend
            .logits(&mut state.context(&prompt))
            .expect("the prompt fits the sequence and this model");
        state.tokens.extend_from_slice(&prompt);
        let mut next = [argmax(&logits) as u32];
        // One warm-up step, so the first round does not carry a cold cache.
        let logits = backend.logits(&mut state.context(&next)).expect("decode");
        next[0] = argmax(&logits) as u32;
        state.tokens.push(next[0]);

        let started = Instant::now();
        for _ in 0..ROUNDS {
            let logits = backend.logits(&mut state.context(&next)).expect("decode");
            next[0] = argmax(&logits) as u32;
            state.tokens.push(next[0]);
        }
        let per_step = started.elapsed().as_secs_f64() / f64::from(ROUNDS) * 1000.0;
        steps.push((length, per_step));
    }

    let shortest = steps[0].1;
    for (length, per_step) in &steps {
        println!(
            "ctx {length:>5}: {per_step:>6.1} ms/step, attention and KV {:.1} ms ({:.0}% of the step), \
             weights {shortest:.1} ms",
            per_step - shortest,
            100.0 * (per_step - shortest) / per_step,
        );
    }
}
