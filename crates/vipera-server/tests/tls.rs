//! TLS end to end, over loopback: the server configuration `vipera serve` builds from
//! `--tls-cert` / `--tls-key`, and a client that trusts only that certificate.
//!
//! The certificates in `tests/fixtures` are a throwaway pair generated for this test: `ca.crt` is a
//! self-signed authority, `server.crt`/`server.key` are a leaf for `localhost` and `127.0.0.1` that
//! it signed (a CA used directly as a server certificate is refused by webpki, `CaUsedAsEndEntity`),
//! and `chain.crt` is the leaf followed by the authority. They are checked in rather than generated
//! at test time so the test takes no certificate-generation dependency.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use vipera_server::serve::tls_config;

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn server_config() -> Arc<rustls::ServerConfig> {
    tls_config(&fixture("chain.crt"), &fixture("server.key")).expect("fixture pair loads")
}

/// A client that trusts exactly the certificates in `trust` — nothing else.
fn client_config(trust: &[u8]) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    for der in certificates(trust) {
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

/// One TLS request and response over one connection, or the error the client saw.
fn round_trip(trust: &[u8]) -> Result<Vec<u8>, String> {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("address");
    let config = server_config();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let connection = rustls::ServerConnection::new(config).expect("server connection");
        let mut tls = rustls::StreamOwned::new(connection, stream);
        let mut request = [0u8; 64];
        // A client that does not trust this certificate fails the handshake here; that is the
        // error path the second test relies on, so it is not a panic.
        if let Ok(count) = tls.read(&mut request)
            && count > 0
        {
            let _ = tls.write_all(b"200 OK\n");
            // Without this the client sees an EOF where it expected close_notify, and treats the
            // exchange as truncated.
            tls.conn.send_close_notify();
            let _ = tls.flush();
        }
    });

    let name = "localhost".try_into().expect("server name");
    let connection = rustls::ClientConnection::new(client_config(trust), name)
        .map_err(|error| error.to_string())?;
    let stream = TcpStream::connect(address).expect("connect");
    let mut tls = rustls::StreamOwned::new(connection, stream);

    let mut response = [0u8; 7];
    let result = tls
        .write_all(b"GET /health\n")
        .and_then(|()| tls.flush())
        .and_then(|()| tls.read_exact(&mut response));
    let _ = server.join();
    result
        .map(|()| response.to_vec())
        .map_err(|error| error.to_string())
}

#[test]
fn a_client_that_trusts_the_certificate_completes_a_request() {
    let response = round_trip(&fixture("ca.crt")).expect("handshake and exchange");
    assert_eq!(response, b"200 OK\n");
}

#[test]
fn a_client_that_trusts_nothing_is_refused() {
    let error = round_trip(&[]).expect_err("the handshake must fail");
    assert!(
        !error.is_empty(),
        "the client should say why the handshake failed"
    );
}

#[test]
fn a_pair_that_does_not_load_is_an_error_rather_than_a_panic() {
    assert!(tls_config(b"not a certificate", &fixture("server.key")).is_err());
    assert!(tls_config(&fixture("chain.crt"), b"not a key").is_err());
    // A certificate is not a key, and rustls says so instead of accepting the bytes.
    assert!(tls_config(&fixture("chain.crt"), &fixture("chain.crt")).is_err());
}

/// The DER bytes of every `CERTIFICATE` block in a PEM file.
fn certificates(pem: &[u8]) -> Vec<Vec<u8>> {
    let text = String::from_utf8_lossy(pem);
    let mut blocks = Vec::new();
    let mut inside = false;
    let mut body = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("-----BEGIN CERTIFICATE") {
            inside = true;
            body.clear();
        } else if line.starts_with("-----END ") {
            if inside && let Some(der) = base64(&body) {
                blocks.push(der);
            }
            inside = false;
        } else if inside {
            body.push_str(line);
        }
    }
    blocks
}

/// Standard base64 with optional padding, ignoring anything outside the alphabet.
fn base64(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for byte in text.bytes() {
        if byte == b'=' {
            break;
        }
        let value = ALPHABET.iter().position(|known| *known == byte)? as u32;
        acc = (acc << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[test]
fn the_fixture_chain_parses() {
    assert_eq!(
        certificates(&fixture("chain.crt")).len(),
        2,
        "leaf then authority"
    );
}
