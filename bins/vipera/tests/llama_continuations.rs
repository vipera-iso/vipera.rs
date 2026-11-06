//! Diff a greedy continuation against llama.cpp on the same model file.
//!
//! This is the check that says the dequantization, rotary embedding, attention, and KV layout all
//! agree with the reference implementation, not merely with themselves. It is skipped — with the
//! reason printed — when the model or `llama-server` is not available on the machine, because neither
//! belongs in the repository. Run it with `--nocapture` to see whether it ran:
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
    let Some(server) = server_binary() else {
        eprintln!("skipped: no llama-server; set LLAMA_SERVER to the llama.cpp server binary");
        return;
    };
    let vipera = vipera_binary();

    let port = free_port();
    let mut server = Server::start(&server, &model, port).expect("start llama-server");
    wait_until_healthy(port, &mut server).expect("llama-server never became healthy");

    let request = format!(
        r#"{{"prompt":{},"n_predict":{MAX_NEW},"temperature":0,"top_k":1,"cache_prompt":false,"return_tokens":true}}"#,
        escape(PROMPT)
    );
    let reference_body = http(port, "/completion", &request).expect("llama.cpp completion");
    let reference = json_numbers(&reference_body, "tokens")
        .expect("llama.cpp did not return a tokens array; check its version");
    assert!(
        !reference.is_empty(),
        "llama.cpp generated nothing: {reference_body}"
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
}
