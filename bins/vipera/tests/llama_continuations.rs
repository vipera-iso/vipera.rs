//! Diff a greedy continuation against llama.cpp on the same model file.
//!
//! This is the check that says the dequantization, rotary embedding, attention, and KV layout all
//! agree with the reference implementation, not merely with themselves.
//!
//! The reference comes from one of two places. `tests/fixtures/llama_cpp_continuation.json` holds a
//! continuation recorded from llama.cpp on the model this repository develops against, so the check
//! runs on any machine that has the model — no llama.cpp install needed. When `LLAMA_SERVER` points
//! at a `llama-server` binary, the test asks it live instead and the fixture is not consulted; that
//! is how the fixture gets re-recorded after a reference build or a model change.
//!
//! Skipped — with the reason printed — when there is no model, or when there is no server and the
//! fixture was recorded on a different model. Run it with `--nocapture` to see which arm ran:
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! LLAMA_SERVER=/path/to/llama-server cargo test -p vipera --test llama_continuations -- --nocapture
//! ```

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PROMPT: &str = "The capital of France is";
const MAX_NEW: u32 = 12;

#[test]
fn greedy_continuation_matches_llama_cpp() {
    let Some(model) = model_path() else {
        eprintln!(
            "skipped: no model; set VIPERA_TEST_MODEL or place qwen2.5-1.5b-instruct-q4_k_m.gguf in ~/.vipera/models"
        );
        return;
    };
    let vipera = vipera_binary();

    // Live reference when a server binary is available, the recorded one otherwise.
    let reference = match server_binary() {
        Some(binary) => {
            let port = free_port();
            let mut server = Server::start(&binary, &model, port).expect("start llama-server");
            let reference = ask_llama_server(port, &mut server).expect("llama.cpp completion");
            eprintln!("compared against a live llama-server on port {port}");
            reference
        }
        None => match fixture_tokens(&model) {
            Some(tokens) => {
                eprintln!(
                    "no llama-server; compared against tests/fixtures/llama_cpp_continuation.json"
                );
                tokens
            }
            None => {
                eprintln!(
                    "skipped: no llama-server and the fixture was recorded on another model; set LLAMA_SERVER"
                );
                return;
            }
        },
    };
    assert!(
        !reference.is_empty(),
        "llama.cpp generated nothing for {PROMPT:?}"
    );

    let output = Command::new(&vipera)
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
        .output()
        .expect("run vipera generate");
    assert!(
        output.status.success(),
        "vipera generate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ours_body = String::from_utf8_lossy(&output.stdout).to_string();
    let ours = json_numbers(&ours_body, "tokens").expect("vipera returned no tokens");

    assert_eq!(
        ours.len(),
        reference.len().min(ours.len()),
        "different token counts for the same budget\n  vipera: {ours:?}\n  llama.cpp: {reference:?}"
    );
    for (index, (mine, theirs)) in ours.iter().zip(&reference).enumerate() {
        assert_eq!(
            mine, theirs,
            "token {index} differs for prompt {PROMPT:?}\n  vipera: {ours:?}\n  llama.cpp: {reference:?}"
        );
    }
}

/// Ask the server for the reference continuation, waiting for it to finish loading.
///
/// Not every build answers `/health` only once the model is ready: the llama.cpp inside Ollama
/// reports healthy while the weights are still mmapping, and a completion sent then comes back
/// without a `tokens` array. So this retries the completion itself until the array appears, which
/// covers both behaviours and needs no version check.
fn ask_llama_server(port: u16, server: &mut Server) -> Result<Vec<u32>, String> {
    wait_until_healthy(port, server)?;
    let request = format!(
        r#"{{"prompt":{},"n_predict":{MAX_NEW},"temperature":0,"top_k":1,"cache_prompt":false,"return_tokens":true}}"#,
        escape(PROMPT)
    );
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut last = String::new();
    while Instant::now() < deadline {
        match http(port, "/completion", &request) {
            Ok(body) => match json_numbers(&body, "tokens") {
                Some(tokens) if !tokens.is_empty() => return Ok(tokens),
                _ => last = body,
            },
            Err(error) => last = error,
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "no tokens array from llama.cpp within the deadline: {last}"
    ))
}

/// The recorded reference continuation, when it was recorded on this model file.
fn fixture_tokens(model: &Path) -> Option<Vec<u32>> {
    let name = model.file_name()?.to_str()?;
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama_cpp_continuation.json");
    let text = std::fs::read_to_string(path).ok()?;
    let recorded = field(&text, "model")?;
    let prompt = field(&text, "prompt")?;
    if recorded != name || prompt != PROMPT {
        return None;
    }
    json_numbers(&text, "tokens")
}

/// The JSON string value of `"key": "value"` — all this test needs from the fixture.
fn field(body: &str, key: &str) -> Option<String> {
    let start = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = &body[start..];
    let open = rest.find('"')? + 1;
    let close = rest[open..].find('"')? + open;
    Some(rest[open..close].to_owned())
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

fn server_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LLAMA_SERVER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join("llama-server"))
        .find(|candidate| candidate.is_file())
}

/// The release binary when it exists: a debug build runs the same loops about an order slower, which
/// would make this test wait minutes for twelve tokens.
fn vipera_binary() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let release = manifest.join("../../target/release/vipera");
    if release.is_file() {
        return release;
    }
    PathBuf::from(env!("CARGO_BIN_EXE_vipera"))
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    listener.local_addr().expect("local address").port()
}

struct Server {
    child: Child,
}

impl Server {
    fn start(binary: &Path, model: &Path, port: u16) -> std::io::Result<Self> {
        let child = Command::new(binary)
            .args([
                "-m",
                model.to_str().expect("model path is UTF-8"),
                "--port",
                &port.to_string(),
                "-c",
                "512",
                "-ngl",
                "0",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(Self { child })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn wait_until_healthy(port: u16, server: &mut Server) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if http(port, "/health", "").is_ok() {
            return Ok(());
        }
        if let Ok(Some(status)) = server.child.try_wait() {
            return Err(format!("llama-server exited early with {status}"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("timed out waiting for /health".to_owned())
}

/// One request per connection, closed by the server: enough for two endpoints and no HTTP client
/// dependency.
fn http(port: u16, path: &str, body: &str) -> Result<String, String> {
    let mut stream =
        TcpStream::connect(("127.0.0.1", port)).map_err(|error| format!("connect: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .map_err(|error| format!("timeout: {error}"))?;
    let method = if body.is_empty() { "GET" } else { "POST" };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("write: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("read: {error}"))?;
    let text = String::from_utf8_lossy(&response).to_string();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("malformed response: {text}"))?;
    if head.contains("Transfer-Encoding: chunked") {
        return Err("chunked responses are not handled by this test".to_owned());
    }
    Ok(body.to_owned())
}

/// Every integer in the array that follows `"key"`, which is all this test needs from either side's
/// JSON.
fn json_numbers(body: &str, key: &str) -> Option<Vec<u32>> {
    let start = body.find(&format!("\"{key}\""))?;
    let open = body[start..].find('[')? + start;
    let close = body[open..].find(']')? + open;
    let mut numbers = Vec::new();
    for piece in body[open + 1..close].split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        numbers.push(piece.parse().ok()?);
    }
    Some(numbers)
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::{escape, json_numbers};

    #[test]
    fn extracts_numbers_after_a_key() {
        let body = r#"{"tokens": [1, 22, 3], "text": "x"}"#;
        assert_eq!(json_numbers(body, "tokens"), Some(vec![1, 22, 3]));
        assert_eq!(json_numbers(body, "missing"), None);
    }

    #[test]
    fn escapes_a_prompt_for_json() {
        assert_eq!(escape("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
    }

    #[test]
    fn reads_a_string_field_out_of_the_fixture() {
        let fixture = r#"{"model": "m.gguf", "prompt": "x", "max_new": 1}"#;
        assert_eq!(super::field(fixture, "model").as_deref(), Some("m.gguf"));
        assert_eq!(super::field(fixture, "prompt").as_deref(), Some("x"));
        assert_eq!(super::field(fixture, "absent"), None);
    }

    #[test]
    fn the_fixture_records_a_whole_continuation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/llama_cpp_continuation.json");
        let text = std::fs::read_to_string(path).expect("fixture is checked in");
        let tokens = super::json_numbers(&text, "tokens").expect("tokens array");
        assert!(
            !tokens.is_empty(),
            "a fixture with no tokens checks nothing"
        );
        assert_eq!(
            tokens.len(),
            12,
            "the fixture is recorded for MAX_NEW tokens"
        );
        assert_eq!(
            super::field(&text, "prompt").as_deref(),
            Some(super::PROMPT)
        );
    }
}
