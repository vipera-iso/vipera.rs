//! Pin the Q4_K `f32`-activation continuation the README's `VIPERA_Q4K_F32` switch promises.
//!
//! With the switch on, the Q4_K dot runs on `f32` activations instead of the default `i8` ones. This
//! test reruns the CLI on a prompt where the two paths diverge inside the budget and compares the
//! greedy continuation to `tests/fixtures/q4k_f32_continuation.json`, so a change that stops
//! honouring the switch (or that changes the `f32` arithmetic) fails at the prompt level, not only in
//! the kernel tests.
//!
//! Skipped — with the reason printed — when there is no model, or when the fixture was recorded on a
//! different model file.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! cargo test -p vipera --test q4k_f32_continuation -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";
const MAX_NEW: u32 = 24;

#[test]
fn q4k_f32_reproduces_the_recorded_prompt_continuation() {
    let Some(model) = model_path() else {
        eprintln!(
            "skipped: no model; set VIPERA_TEST_MODEL or place qwen2.5-1.5b-instruct-q4_k_m.gguf in ~/.vipera/models"
        );
        return;
    };

    let fixture = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/q4k_f32_continuation.json"),
    )
    .expect("the fixture is checked in");
    let recorded_model = field(&fixture, "model").expect("fixture names its model");
    let name = model
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if recorded_model != name {
        eprintln!(
            "skipped: the fixture was recorded on {recorded_model}, not {name}; re-record it with \
             VIPERA_Q4K_F32=1"
        );
        return;
    }
    assert_eq!(
        number(&fixture, "max_new"),
        Some(MAX_NEW),
        "the fixture is recorded for MAX_NEW tokens"
    );
    assert_eq!(
        field(&fixture, "prompt").as_deref(),
        Some(PROMPT),
        "the fixture is recorded for this prompt"
    );
    let expected = numbers(&fixture, "tokens").expect("fixture records tokens");
    assert_eq!(
        expected.len(),
        MAX_NEW as usize,
        "a short fixture checks less"
    );

    let output = Command::new(vipera_binary())
        .args([
            "generate",
            model.to_str().expect("model path is UTF-8"),
            "--prompt",
            PROMPT,
            "--max-tokens",
            &MAX_NEW.to_string(),
            "--temperature",
            "0",
            "--json",
        ])
        .env("VIPERA_Q4K_F32", "1")
        .output()
        .expect("run vipera generate");
    assert!(
        output.status.success(),
        "vipera generate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body = String::from_utf8_lossy(&output.stdout).to_string();
    let ours = numbers(&body, "tokens").expect("vipera returned no tokens");

    assert_eq!(
        ours, expected,
        "the f32-activation continuation moved for prompt {PROMPT:?}\n  now: {ours:?}\n  was: {expected:?}"
    );
}

fn model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VIPERA_TEST_MODEL") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    path.is_file().then_some(path)
}

/// The release binary when it exists: a debug build runs the same loops about an order slower.
fn vipera_binary() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let release = manifest.join("../../target/release/vipera");
    if release.is_file() {
        return release;
    }
    PathBuf::from(env!("CARGO_BIN_EXE_vipera"))
}

/// The JSON string value of `"key": "value"` — all this test needs from the fixture.
fn field(body: &str, key: &str) -> Option<String> {
    let start = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = &body[start..];
    let open = rest.find('"')? + 1;
    let close = rest[open..].find('"')? + open;
    Some(rest[open..close].to_owned())
}

/// The JSON number value of `"key": 12`.
fn number(body: &str, key: &str) -> Option<u32> {
    let start = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = &body[start..];
    let colon = rest.find(':')? + 1;
    rest[colon..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// The JSON array of integers after `"key"`.
fn numbers(body: &str, key: &str) -> Option<Vec<u32>> {
    let start = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = &body[start..];
    let open = rest.find('[')? + 1;
    let close = rest[open..].find(']')? + open;
    rest[open..close]
        .split(',')
        .map(|part| part.trim().parse().ok())
        .collect()
}
