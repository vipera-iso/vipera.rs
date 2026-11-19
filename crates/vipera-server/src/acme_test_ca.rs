//! A fake ACME CA, for the provisioning test.
//!
//! Test-only (`#[cfg(test)]` in `acme.rs`), never part of the shipped server. It speaks enough of
//! RFC 8555 for `instant-acme`'s real client to walk a whole order — directory, nonce, account,
//! order, authorization, http-01 challenge, finalize, certificate — over the same TLS client path
//! `vipera serve` uses. The one thing it does not do is verify JWS signatures: it reads the token
//! out of the order it handed out. Everything else is the real flow.
//!
//! The challenge is validated the way a CA validates it: a fresh `GET` to
//! `/.well-known/acme-challenge/<token>` on the listener `vipera-acme-http` runs.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::http::{Incoming, Transport, read_request};

/// The token this CA hands out in its one authorization.
pub(super) const TOKEN: &str = "vipera-fake-token";

/// What the CA observed, for the test to assert on.
#[derive(Default)]
pub(super) struct Observed {
    /// The body the CA got back for the challenge, `None` when the fetch was refused.
    pub(super) challenge_answer: Option<String>,
    /// Whether that fetch completed the validation.
    pub(super) validated: bool,
    /// Whether the order has been finalized, which is what makes it `valid`.
    pub(super) finalized: bool,
    /// Requests the client made, in order, so the test can prove the whole walk happened.
    pub(super) paths: Vec<String>,
    /// Size of the CSR that arrived at `finalize`.
    pub(super) csr_len: Option<usize>,
}

pub(super) struct FakeCa {
    /// The authority's own `host:port`, used to write absolute URLs.
    pub(super) address: String,
    /// Where the challenge listener answers: what this CA validates against.
    pub(super) challenge_address: String,
    pub(super) observed: Mutex<Observed>,
    /// The fixture chain returned once the order is finalized.
    pub(super) chain: Vec<u8>,
}

impl FakeCa {
    /// Start serving on a loopback port, and return once it is bound.
    pub(super) fn spawn(
        challenge_address: String,
        chain: Vec<u8>,
        config: Arc<rustls::ServerConfig>,
    ) -> Arc<FakeCa> {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the fake CA");
        let address = format!("localhost:{}", listener.local_addr().expect("addr").port());
        let ca = Arc::new(FakeCa {
            address,
            challenge_address,
            observed: Mutex::new(Observed::default()),
            chain,
        });
        let serving = Arc::clone(&ca);
        std::thread::spawn(move || {
            for incoming in listener.incoming() {
                let Ok(stream) = incoming else {
                    continue;
                };
                let ca = Arc::clone(&serving);
                let config = Arc::clone(&config);
                std::thread::spawn(move || {
                    let Ok(connection) = rustls::ServerConnection::new(config) else {
                        return;
                    };
                    let mut tls = rustls::StreamOwned::new(connection, stream);
                    let _ = answer(&ca, &mut tls);
                });
            }
        });
        ca
    }

    pub(super) fn directory(&self) -> String {
        self.url("/directory")
    }

    fn url(&self, path: &str) -> String {
        format!("https://{}{path}", self.address)
    }
}

fn answer(ca: &FakeCa, stream: &mut dyn Transport) -> std::io::Result<()> {
    let Ok(Incoming::Request(request)) = read_request(stream) else {
        return Ok(());
    };
    let (status, body, location) = route(ca, &request.method, &request.path, &request.body);
    let mut head = format!(
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nReplay-Nonce: \
         fake-nonce-{}\r\nContent-Length: {}\r\nConnection: close\r\n",
        nonce(),
        body.len()
    );
    if let Some(location) = location {
        head.push_str(&format!("Location: {location}\r\n"));
    }
    head.push_str("\r\n");
    if request.method == "HEAD" {
        return stream
            .write_all(head.as_bytes())
            .and_then(|()| stream.flush());
    }
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

fn nonce() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0)
}

fn route(ca: &FakeCa, method: &str, path: &str, body: &[u8]) -> (u16, String, Option<String>) {
    ca.observed
        .lock()
        .expect("observed")
        .paths
        .push(format!("{method} {path}"));
    match (method, path) {
        ("GET", "/directory") => (
            200,
            format!(
                r#"{{"newNonce":{},"newAccount":{},"newOrder":{},"revokeCert":{},"keyChange":{}}}"#,
                quote(&ca.url("/new-nonce")),
                quote(&ca.url("/new-account")),
                quote(&ca.url("/new-order")),
                quote(&ca.url("/revoke")),
                quote(&ca.url("/key-change")),
            ),
            None,
        ),
        ("HEAD", "/new-nonce") | ("GET", "/new-nonce") => (200, String::new(), None),
        ("POST", "/new-account") => (
            201,
            format!(
                r#"{{"status":"valid","contact":[],"orders":{}}}"#,
                quote(&ca.url("/account/1"))
            ),
            Some(ca.url("/account/1")),
        ),
        ("POST", "/new-order") => (201, order(ca, "pending", false), Some(ca.url("/order/1"))),
        ("POST", "/authz/1") => (200, authorization(ca), None),
        ("POST", "/challenge/1") => {
            validate(ca);
            (200, challenge(ca, "valid"), None)
        }
        ("POST", "/order/1") => {
            // A validated order is `ready`: finalize has not happened yet. Only after finalize does
            // it become `valid` and carry a certificate URL.
            let observed = ca.observed.lock().expect("observed");
            let status = match (observed.finalized, observed.validated) {
                (true, _) => "valid",
                (false, true) => "ready",
                (false, false) => "pending",
            };
            (200, order(ca, status, observed.finalized), None)
        }
        ("POST", "/finalize/1") => {
            let mut observed = ca.observed.lock().expect("observed");
            observed.csr_len.get_or_insert(body.len());
            observed.finalized = true;
            (200, order(ca, "valid", true), None)
        }
        ("POST", "/cert/1") => (200, String::from_utf8_lossy(&ca.chain).into_owned(), None),
        _ => (
            404,
            r#"{"type":"urn:ietf:params:acme:error:malformed","detail":"no such endpoint"}"#
                .to_owned(),
            None,
        ),
    }
}

fn order(ca: &FakeCa, status: &str, certificate: bool) -> String {
    let certificate = if certificate {
        quote(&ca.url("/cert/1"))
    } else {
        "null".to_owned()
    };
    format!(
        r#"{{"status":{},"identifiers":[{{"type":"dns","value":"localhost"}}],"authorizations":[{}],"finalize":{},"certificate":{certificate}}}"#,
        quote(status),
        quote(&ca.url("/authz/1")),
        quote(&ca.url("/finalize/1")),
    )
}

fn authorization(ca: &FakeCa) -> String {
    let status = if ca.observed.lock().expect("observed").validated {
        "valid"
    } else {
        "pending"
    };
    format!(
        r#"{{"identifier":{{"type":"dns","value":"localhost"}},"status":{},"challenges":[{}],"wildcard":false}}"#,
        quote(status),
        challenge(ca, status),
    )
}

fn challenge(ca: &FakeCa, status: &str) -> String {
    format!(
        r#"{{"type":"http-01","url":{},"token":{},"status":{},"error":null}}"#,
        quote(&ca.url("/challenge/1")),
        quote(TOKEN),
        quote(status),
    )
}

/// Fetch the answer the way a CA does, and remember whether it proved the token.
fn validate(ca: &FakeCa) {
    let mut observed = ca.observed.lock().expect("observed");
    if observed.validated {
        return;
    }
    let answer = fetch(ca);
    observed.validated = answer
        .as_deref()
        .is_some_and(|body| body.starts_with(&format!("{TOKEN}.")));
    observed.challenge_answer = answer;
}

fn fetch(ca: &FakeCa) -> Option<String> {
    let mut stream = TcpStream::connect(&ca.challenge_address).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .ok()?;
    write!(
        stream,
        "GET /.well-known/acme-challenge/{TOKEN} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        ca.challenge_address
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (head, body) = response.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_owned())
}

fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
