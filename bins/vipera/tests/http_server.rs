//! End-to-end check of `vipera serve` against `contracts/http-chat.md`.
//!
//! It starts the real binary on a free port, then speaks HTTP to it by hand: health, an auth rejection,
//! a non-streaming completion, a streamed one that must end with `data: [DONE]`, and the metrics
//! endpoint. Skipped — with the reason printed — when no model is on the machine.
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! cargo test -p vipera --test http_server -- --nocapture
//! ```

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const API_KEY: &str = "test-key";

#[test]
fn serve_answers_chat_completions_and_streams() {
    let Some(model) = model_path() else {
        eprintln!(
            "skipped: no model; set VIPERA_TEST_MODEL or place qwen2.5-1.5b-instruct-q4_k_m.gguf in ~/.vipera/models"
        );
        return;
    };
    let port = free_port();
    let mut server = Server::start(&model, port).expect("start vipera serve");
    wait_until_listening(port, &mut server).expect("server never accepted a connection");

    let health = http(port, "GET", "/health", "").expect("health");
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(
        health.contains("Access-Control-Allow-Origin: *"),
        "{health}"
    );

    // A browser sends the preflight without the API key; it must still succeed and advertise the
    // methods and headers the endpoint accepts.
    let preflight =
        http_with_key(port, "OPTIONS", "/v1/chat/completions", "", None).expect("preflight");
    assert!(preflight.starts_with("HTTP/1.1 204"), "{preflight}");
    assert!(
        preflight.contains("Access-Control-Allow-Origin: *"),
        "{preflight}"
    );
    assert!(
        preflight.contains("Access-Control-Allow-Headers: authorization, content-type"),
        "{preflight}"
    );
    assert!(health.contains(r#""status":"ok""#), "{health}");

    let unauthorized = http_with_key(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        None,
    )
    .expect("unauthorized request");
    assert!(unauthorized.starts_with("HTTP/1.1 401"), "{unauthorized}");
    assert!(
        unauthorized.contains(r#""code":"unauthorized""#),
        "{unauthorized}"
    );

    let body = r#"{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":2,"temperature":0}"#;
    let answer = http(port, "POST", "/v1/chat/completions", body).expect("completion");
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(answer.contains(r#""object":"chat.completion""#), "{answer}");
    assert!(answer.contains(r#""role":"assistant""#), "{answer}");
    assert!(answer.contains(r#""usage":{"prompt_tokens":"#), "{answer}");
    assert!(
        answer.contains(r#""finish_reason":"stop""#)
            || answer.contains(r#""finish_reason":"length""#),
        "{answer}"
    );

    let streamed = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"test","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Say hi"}],"max_tokens":3,"temperature":0}"#,
    )
    .expect("stream");
    assert!(streamed.starts_with("HTTP/1.1 200"), "{streamed}");
    assert!(
        streamed.contains("Content-Type: text/event-stream"),
        "{streamed}"
    );
    assert!(
        streamed.contains(r#""object":"chat.completion.chunk""#),
        "{streamed}"
    );
    assert!(
        streamed.trim_end().ends_with("data: [DONE]"),
        "stream must end with the terminator: {streamed}"
    );
    assert!(
        streamed.contains(r#""finish_reason":"stop""#)
            || streamed.contains(r#""finish_reason":"length""#),
        "{streamed}"
    );
    assert!(
        streamed.contains(r#""choices":[],"usage":{"prompt_tokens""#),
        "include_usage must add a usage chunk: {streamed}"
    );

    let with_logprobs = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":2,"temperature":0,"logprobs":true,"top_logprobs":2}"#,
    )
    .expect("logprobs");
    assert!(with_logprobs.starts_with("HTTP/1.1 200"), "{with_logprobs}");
    assert!(
        with_logprobs.contains(r#""logprobs":{"content":["#),
        "{with_logprobs}"
    );
    assert!(with_logprobs.contains(r#""logprob":"#), "{with_logprobs}");
    assert!(
        with_logprobs.contains(r#""top_logprobs":[{"token""#),
        "{with_logprobs}"
    );

    let streamed_logprobs = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":3,"temperature":0,"stream":true,"logprobs":true,"top_logprobs":2}"#,
    )
    .expect("streamed logprobs");
    assert!(
        streamed_logprobs.starts_with("HTTP/1.1 200"),
        "{streamed_logprobs}"
    );
    assert!(
        streamed_logprobs.contains(r#""logprobs":{"content":["#),
        "{streamed_logprobs}"
    );
    assert!(
        streamed_logprobs.contains(r#""top_logprobs":[{"token""#),
        "{streamed_logprobs}"
    );

    let streamed_completion_logprobs = http(
        port,
        "POST",
        "/v1/completions",
        r#"{"model":"test","prompt":"Say hi","max_tokens":3,"temperature":0,"stream":true,"logprobs":true,"top_logprobs":2}"#,
    )
    .expect("streamed completion logprobs");
    assert!(
        streamed_completion_logprobs.starts_with("HTTP/1.1 200"),
        "{streamed_completion_logprobs}"
    );
    assert!(
        streamed_completion_logprobs.contains(r#""logprobs":{"tokens":["#),
        "{streamed_completion_logprobs}"
    );
    assert!(
        streamed_completion_logprobs.contains(r#""token_logprobs":[-0."#),
        "{streamed_completion_logprobs}"
    );

    let two_choices = http(
        port,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":1,"temperature":0,"n":2}"#,
    )
    .expect("two choices");
    assert!(two_choices.starts_with("HTTP/1.1 200"), "{two_choices}");
    assert!(two_choices.contains(r#""index":0"#), "{two_choices}");
    assert!(two_choices.contains(r#""index":1"#), "{two_choices}");

    let completion = http(
        port,
        "POST",
        "/v1/completions",
        r#"{"model":"test","prompt":"The capital of France is","max_tokens":2,"temperature":0,"stop":["\n"]}"#,
    )
    .expect("completion endpoint");
    assert!(completion.starts_with("HTTP/1.1 200"), "{completion}");
    assert!(
        completion.contains(r#""object":"text_completion""#),
        "{completion}"
    );
    assert!(
        completion.contains(r#""finish_reason":"stop""#)
            || completion.contains(r#""finish_reason":"length""#),
        "{completion}"
    );

    let metrics = http(port, "GET", "/metrics", "").expect("metrics");
    assert!(metrics.contains("vipera_requests_total{"), "{metrics}");
    assert!(metrics.contains("vipera_kv_blocks_in_use"), "{metrics}");
    assert!(
        metrics.contains("vipera_time_to_first_token_seconds_bucket"),
        "{metrics}"
    );

    server.stop();
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
    fn start(model: &Path, port: u16) -> std::io::Result<Self> {
        let child = Command::new(vipera_binary())
            .args([
                "serve",
                model.to_str().expect("model path is UTF-8"),
                "--port",
                &port.to_string(),
                "--ctx",
                "512",
                "--model-id",
                "test",
                "--api-key",
                API_KEY,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(Server { child })
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_until_listening(port: u16, server: &mut Server) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child.try_wait() {
            return Err(format!("server exited early with {status}"));
        }
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Err("timed out waiting for the port".to_owned())
}

fn http(port: u16, method: &str, path: &str, body: &str) -> Result<String, String> {
    http_with_key(port, method, path, body, Some(API_KEY))
}

fn http_with_key(
    port: u16,
    method: &str,
    path: &str,
    body: &str,
    key: Option<&str>,
) -> Result<String, String> {
    let mut stream =
        TcpStream::connect(("127.0.0.1", port)).map_err(|error| format!("connect: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .map_err(|error| format!("timeout: {error}"))?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\n"
    );
    if let Some(key) = key {
        request.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("write: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("read: {error}"))?;
    String::from_utf8(response).map_err(|error| format!("response was not UTF-8: {error}"))
}
