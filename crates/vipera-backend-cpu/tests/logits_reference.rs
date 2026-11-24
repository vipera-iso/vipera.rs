//! The qwen3 block layout against llama.cpp's own logits.
//!
//! qwen3 adds a per-head RMSNorm on the projected queries and keys before RoPE; get the order or the
//! weight sharing wrong and the model still emits fluent text, which is why the shell-to-shell
//! comparison is not enough. This compares the logits of one fixed token sequence against a run of
//! llama.cpp on the same file and prints the difference, so a wrong norm shows up as a rank-1
//! disagreement rather than as prose that happens to read well.
//!
//! The reference is produced, not transcribed:
//!
//! ```bash
//! # one fixed sequence, no sampling, no chat template
//! scripts/logits-dump.sh "$VIPERA_TEST_MODEL" 16 12095 13 3555 374 279 6722 315 9856 30 \
//!     > crates/vipera-backend-cpu/tests/data/qwen3-logits.json
//! VIPERA_TEST_MODEL=$HOME/.vipera/models/hermes4-14b-q4km.gguf \
//!   cargo test -p vipera-backend-cpu --release --test logits_reference -- --ignored --nocapture
//! ```
//!
//! Both sides load the same GGUF, so the only difference left is arithmetic: this backend sums the
//! Q4_K dots in its own order, so the values agree to a tolerance rather than bit for bit. The test
//! holds the argmax exactly and the top rows to `LOGIT_TOLERANCE`.

use std::path::{Path, PathBuf};

use vipera_backend_cpu::CpuBackend;
use vipera_core::planner::KvCacheType;
use vipera_runtime::{ForwardContext, ForwardPass};

const TOKENS_PER_BLOCK: u64 = 16;

/// How far a shared logit may sit from llama.cpp's. Both sides run f32 dots over the same Q4_K
/// bytes, but llama.cpp sums a row 256 values at a time through its own SIMD reductions and this
/// backend sums in its own order, so the totals differ in the last few digits of a dot that has
/// already been through 40 layers. On Hermes 4 14B the worst of the top sixteen moved 0.26 against a
/// top logit of 22.6, which is 1% of the margin the distribution actually has; 0.3 leaves room for
/// that and is still far below the tens of logits a wrong per-head norm shifts things by.
const LOGIT_TOLERANCE: f32 = 0.3;

/// The same measure for a scaled model over a long sequence. Measured on the synthetic Qwen2.5-1.5B
/// YaRN file (factor 4, 32768 trained positions) at 64 and 6001 tokens: the unscaled control on the
/// same sequence moved 0.333 and 0.194, the scaled one 0.357 and 0.566, with the same argmax. The
/// extra is the two ways of building the table: llama.cpp multiplies `theta_scale` pair by pair and
/// vipera raises the base per pair, which is 6.7e-4 rad apart at position 6000 (pair 6), amplified
/// by 28 layers of a sharper distribution. A wrong ramp or a missing magnitude shows up as a
/// different argmax, not as this.
const SCALED_LOGIT_TOLERANCE: f32 = 0.75;

fn model_path() -> Option<PathBuf> {
    let path = std::env::var_os("VIPERA_TEST_MODEL").map(PathBuf::from)?;
    path.exists().then_some(path)
}

fn reference_path() -> PathBuf {
    if let Some(path) = std::env::var_os("VIPERA_REFERENCE_LOGITS") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/qwen3-logits.json")
        .to_owned()
}

/// The `tokens` and `top` arrays of the dumper's one-line JSON. The shape is fixed and tiny, so this
/// reads it rather than pulling a JSON dependency into the backend's test build.
fn parse_reference(text: &str) -> (Vec<u32>, Vec<(u32, f32)>) {
    let tokens_at = text.find("\"tokens\":[").expect("a tokens array");
    let tokens_end = tokens_at + text[tokens_at..].find(']').expect("the tokens end");
    let tokens = text[tokens_at + 10..tokens_end]
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().expect("a token id"))
        .collect();

    let rows_at = text.find("\"top\":[").expect("a top array");
    let rows = text[rows_at + 7..]
        .split("],[")
        .filter_map(|row| {
            let row = row.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-');
            let (id, value) = row.split_once(',')?;
            Some((
                id.trim().parse::<u32>().ok()?,
                value.trim().parse::<f32>().ok()?,
            ))
        })
        .collect();
    (tokens, rows)
}

fn top_rows(logits: &[f32], count: usize) -> Vec<(u32, f32)> {
    let mut rows: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(at, value)| (at as u32, *value))
        .collect();
    rows.sort_by(|left, right| right.1.total_cmp(&left.1));
    rows.truncate(count);
    rows
}

/// The architecture the reference run recorded, so the same comparison serves whichever model the
/// dump was taken from. Dumps written before the field existed are the qwen3 ones.
fn reference_arch(text: &str) -> String {
    let at = match text.find("\"arch\":\"") {
        Some(at) => at + 8,
        None => return "qwen3".to_owned(),
    };
    let rest = &text[at..];
    rest[..rest.find('"').expect("the arch value ends")].to_owned()
}

#[test]
#[ignore = "needs a qwen3 GGUF and a reference run of llama.cpp on it"]
fn qwen3_logits_match_the_reference_run() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL to a qwen3 GGUF");
        return;
    };
    let reference = reference_path();
    let Ok(text) = std::fs::read_to_string(&reference) else {
        eprintln!(
            "skipped: no reference at {}; run scripts/logits-dump.sh",
            reference.display()
        );
        return;
    };
    compare_with_reference(&path, &text, LOGIT_TOLERANCE);
}

/// The same comparison for a model whose rotary table is scaled (linear or YaRN), on a sequence long
/// enough for the rule to matter:
///
/// ```bash
/// scripts/logits-dump.sh /tmp/qwen2.5-1.5b-yarn.gguf $(python3 -c 'print(" ".join(map(str, IDS)))') \
///     > /tmp/yarn-logits.json
/// VIPERA_TEST_MODEL=/tmp/qwen2.5-1.5b-yarn.gguf VIPERA_REFERENCE_LOGITS=/tmp/yarn-logits.json \
///   cargo test -p vipera-backend-cpu --release --test logits_reference -- --ignored scaled --nocapture
/// ```
#[test]
#[ignore = "needs a scaled GGUF and a reference run of llama.cpp on it"]
fn scaled_model_logits_match_the_reference_run() {
    let Some(path) = model_path() else {
        eprintln!("skipped: no model; set VIPERA_TEST_MODEL to the scaled GGUF");
        return;
    };
    let reference = reference_path();
    let Ok(text) = std::fs::read_to_string(&reference) else {
        eprintln!(
            "skipped: no reference at {}; run scripts/logits-dump.sh",
            reference.display()
        );
        return;
    };
    compare_with_reference(&path, &text, SCALED_LOGIT_TOLERANCE);
}

fn compare_with_reference(path: &Path, text: &str, tolerance: f32) {
    let (tokens, expected) = parse_reference(text);
    let mut backend = CpuBackend::load(path).expect("load model");
    assert_eq!(
        backend.architecture(),
        reference_arch(text),
        "the reference was dumped from a different architecture; point VIPERA_TEST_MODEL at that model"
    );

    let config = backend.config();
    let bytes_per_token = config
        .kv_bytes_per_token_for(KvCacheType::F16)
        .expect("the width is a multiple of 32");
    let blocks = (tokens.len() as u64 + 64).div_ceil(TOKENS_PER_BLOCK);
    let table: Vec<u32> = (0..blocks as u32).collect();
    let mut kv = vec![0u8; blocks as usize * bytes_per_token as usize * TOKENS_PER_BLOCK as usize];
    let logits = backend
        .logits(&mut ForwardContext {
            tokens: &[],
            new_tokens: &tokens,
            first_position: 0,
            block_table: &table,
            kv: &mut kv,
            tokens_per_block: TOKENS_PER_BLOCK,
            bytes_per_token,
        })
        .expect("prefill the reference sequence");

    let got = top_rows(&logits, expected.len());
    let mut worst = 0.0f32;
    println!(
        "{:<8} {:>10} {:>10} {:>8}",
        "token", "llama.cpp", "vipera", "diff"
    );
    for (position, ((want_id, want), (got_id, _))) in expected.iter().zip(got.iter()).enumerate() {
        /* The rows are compared by the id llama.cpp ranks, so a swapped rank shows up as a large
         * difference in a value rather than as two lists that quietly disagree. */
        let ours = *logits
            .get(*want_id as usize)
            .expect("a logit for the reference id");
        let diff = (ours - want).abs();
        worst = worst.max(diff);
        println!(
            "{:<8} {want:>10.4} {ours:>10.4} {diff:>8.4}  #{} {}",
            want_id,
            position + 1,
            if got_id == want_id { "same" } else { "moved" },
        );
    }

    assert_eq!(
        got[0].0, expected[0].0,
        "argmax: llamacpp {} ({:.4}) vs vipera {} ({:.4})",
        expected[0].0, expected[0].1, got[0].0, got[0].1
    );
    assert!(
        worst <= tolerance,
        "worst top-row logit differs by {worst:.4}, over the {tolerance} tolerance"
    );
    /* The order inside the tail of the top sixteen flips on this run, and one row at the very edge of
     * it (rank 15 against rank 16) trades places with a neighbour 0.2 away, because those rows sit
     * within the accumulation noise of each other. The top eight have real margin, so those are held:
     * a wrong norm does not shuffle one neighbourhood, it replaces it. */
    const PINNED: usize = 8;
    let got_ids: Vec<u32> = got.iter().map(|(id, _)| *id).collect();
    let missing: Vec<u32> = expected
        .iter()
        .take(PINNED)
        .map(|(id, _)| *id)
        .filter(|id| !got_ids.contains(id))
        .collect();
    assert!(
        missing.is_empty(),
        "reference tokens {missing:?} are not in the top {} this backend ranks",
        got.len()
    );
}
