//! `vipera serve --tls-cert/--tls-key`, as an HTTP client sees it: the real binary, the real
//! certificate fixture, and a client that trusts only that certificate.
//!
//! Skipped — with the reason printed — when no model is on the machine, because the server cannot
//! start without one. Run with `--nocapture` to see which arm ran:
//!
//! ```text
//! VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//! cargo test -p vipera --test tls_serve -- --nocapture
//! ```

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The throwaway pair `crates/vipera-server/tests/tls.rs` uses; `ca.der` is the same authority in
/// DER form, so this test can trust it without parsing PEM a third time.
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/vipera-server/tests/fixtures")
        .join(name)
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

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    listener.local_addr().expect("local address").port()
}

struct Server {
    child: Child,
}

impl Server {
    fn start(model: &Path, port: u16) -> std::io::Result<Self> {
        let child = Command::new(env!("CARGO_BIN_EXE_vipera"))
            .args([
                "serve",
                model.to_str().expect("model path is UTF-8"),
                "--port",
                &port.to_string(),
                "--ctx",
                "512",
                "--tls-cert",
                fixture("chain.crt").to_str().expect("fixture path"),
                "--tls-key",
                fixture("server.key").to_str().expect("fixture path"),
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

fn client_config(trust_authority: bool) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    if trust_authority {
        let der = std::fs::read(fixture("ca.der")).expect("ca.der fixture");
        roots
            .add(rustls::pki_types::CertificateDer::from(der))
            .expect("a usable root");
    }
    Arc::new(
        rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth(),
    )
}

/// One HTTPS GET, returning status line and body. `Ok` means the exchange completed.
fn https_get(port: u16, path: &str, trust_authority: bool) -> Result<(String, String), String> {
    let name = "localhost".try_into().expect("server name");
    let connection = rustls::ClientConnection::new(client_config(trust_authority), name)
        .map_err(|error| error.to_string())?;
    let stream =
        TcpStream::connect(("127.0.0.1", port)).map_err(|error| format!("connect: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    let mut tls = rustls::StreamOwned::new(connection, stream);
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    tls.write_all(request.as_bytes())
        .and_then(|()| tls.flush())
        .map_err(|error| format!("write: {error}"))?;
    let mut response = Vec::new();
    tls.read_to_end(&mut response)
        .map_err(|error| format!("read: {error}"))?;
    let text = String::from_utf8_lossy(&response).to_string();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("no response head: {text:?}"))?;
    Ok((
        head.lines().next().unwrap_or_default().to_owned(),
        body.to_owned(),
    ))
}

/// A plain HTTP GET on the TLS port: the server expects a handshake, so this must not answer 200.
fn plain_http_get(port: u16) -> String {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return String::new();
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ =
        stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    String::from_utf8_lossy(&response).to_string()
}

#[test]
fn the_served_binary_answers_https_and_only_https() {
    let Some(model) = model_path() else {
        eprintln!(
            "skipped: no model; set VIPERA_TEST_MODEL or place qwen2.5-1.5b-instruct-q4_k_m.gguf in ~/.vipera/models"
        );
        return;
    };
    let port = free_port();
    let _server = Server::start(&model, port).expect("start vipera serve");

    // The model takes a moment to load; until then the port refuses or the handshake fails.
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut last = String::new();
    let mut status = String::new();
    while Instant::now() < deadline {
        match https_get(port, "/health", true) {
            Ok((line, body)) => {
                status = line;
                if body.contains("\"status\":\"ok\"") {
                    break;
                }
                last = body;
            }
            Err(error) => last = error,
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert!(
        status.contains(" 200"),
        "HTTPS /health never answered 200 (last: {last})"
    );

    // A client that trusts nothing must not complete the handshake.
    let untrusted = https_get(port, "/health", false).expect_err("untrusted client must fail");
    assert!(!untrusted.is_empty(), "the client should say what failed");

    // Plain HTTP on the TLS port is not a request the server can answer.
    let plain = plain_http_get(port);
    assert!(
        !plain.contains(" 200"),
        "plain HTTP got an HTTP answer from the TLS port: {plain:?}"
    );

    // And a real completion goes through the same TLS stream.
    let (line, body) = https_get(port, "/v1/models", true).expect("models over TLS");
    assert!(line.contains(" 200"), "GET /v1/models: {line}");
    assert!(body.contains("\"object\":\"list\""), "body: {body}");
}
