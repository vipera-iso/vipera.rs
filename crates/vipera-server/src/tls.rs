//! TLS for `vipera serve`: the `rustls` configuration built from `--tls-cert`/`--tls-key`, the PEM
//! reader behind it, and the file stamps that turn a renewed certificate into a reload instead of a
//! restart.

use std::sync::Arc;

/// A TLS configuration that follows the files it was built from.
///
/// Renewing a certificate is a file write here, not a restart: the accept loop compares the pair's
/// length and modification time before each connection and rebuilds the configuration when they
/// move. A file that does not parse keeps the previous configuration, so a half-written renewal
/// cannot take a working server down, and the bad stamp is remembered so it is reported once.
pub(crate) struct TlsSource {
    cert: std::path::PathBuf,
    key: std::path::PathBuf,
    current: std::sync::Mutex<TlsCurrent>,
}

struct TlsCurrent {
    config: Arc<rustls::ServerConfig>,
    stamp: StampPair,
}

/// The pair of stamps a configuration was built from, `None` when a file is missing.
type StampPair = (Option<FileStamp>, Option<FileStamp>);

/// Just enough of a file's metadata to notice a rotation, without reading it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
}

impl FileStamp {
    fn of(path: &std::path::Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

impl TlsSource {
    pub(crate) fn new(cert: &str, key: &str) -> Result<Self, String> {
        let cert = std::path::PathBuf::from(cert);
        let key = std::path::PathBuf::from(key);
        let (config, stamp) = Self::build(&cert, &key)?;
        Ok(Self {
            cert,
            key,
            current: std::sync::Mutex::new(TlsCurrent { config, stamp }),
        })
    }

    fn build(
        cert: &std::path::Path,
        key: &std::path::Path,
    ) -> Result<(Arc<rustls::ServerConfig>, StampPair), String> {
        let stamp = (FileStamp::of(cert), FileStamp::of(key));
        let cert_pem = std::fs::read(cert)
            .map_err(|error| format!("cannot read {}: {error}", cert.display()))?;
        let key_pem = std::fs::read(key)
            .map_err(|error| format!("cannot read {}: {error}", key.display()))?;
        Ok((tls_config(&cert_pem, &key_pem)?, stamp))
    }

    /// The configuration to hand the next handshake.
    pub(crate) fn config(&self) -> Arc<rustls::ServerConfig> {
        Arc::clone(&self.current.lock().expect("TLS configuration").config)
    }

    /// Rebuilds the configuration when either file moved. Returns whether it was replaced.
    pub(crate) fn refresh(&self) -> bool {
        let stamp = (FileStamp::of(&self.cert), FileStamp::of(&self.key));
        if self.current.lock().expect("TLS configuration").stamp == stamp {
            return false;
        }
        match Self::build(&self.cert, &self.key) {
            Ok((config, stamp)) => {
                let mut current = self.current.lock().expect("TLS configuration");
                current.config = config;
                current.stamp = stamp;
                true
            }
            Err(error) => {
                let mut current = self.current.lock().expect("TLS configuration");
                current.stamp = stamp;
                eprintln!("vipera: keeping the previous certificate: {error}");
                false
            }
        }
    }
}

/// A server configuration from PEM certificate and key bytes.
///
/// The certificate chain comes first in the file, leaf first; the key may be PKCS#8, SEC1 or
/// PKCS#1, because every tool writes one of the three and the tag says which.
pub fn tls_config(cert_pem: &[u8], key_pem: &[u8]) -> Result<Arc<rustls::ServerConfig>, String> {
    let chain: Vec<rustls::pki_types::CertificateDer<'static>> = pem_blocks(cert_pem)
        .into_iter()
        .filter(|(tag, _)| tag.ends_with("CERTIFICATE"))
        .map(|(_, der)| rustls::pki_types::CertificateDer::from(der))
        .collect();
    if chain.is_empty() {
        return Err("no CERTIFICATE block in the certificate file".to_owned());
    }
    let (tag, der) = pem_blocks(key_pem)
        .into_iter()
        .find(|(tag, _)| tag.ends_with("PRIVATE KEY"))
        .ok_or_else(|| "no PRIVATE KEY block in the key file".to_owned())?;
    let key = rustls::pki_types::PrivateKeyDer::try_from(der)
        .map_err(|error| format!("{tag} is not a key rustls accepts: {error}"))?;
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|error| format!("certificate and key do not go together: {error}"))?;
    Ok(Arc::new(config))
}

/// The DER bytes of every PEM block, with its tag (`CERTIFICATE`, `PRIVATE KEY`, …).
///
/// Hand-rolled on purpose: the workspace takes one third-party crate for TLS and no base64 crate
/// for twenty lines of decoding, and the tests below cover the padding and the line breaks.
pub(crate) fn pem_blocks(pem: &[u8]) -> Vec<(String, Vec<u8>)> {
    let text = String::from_utf8_lossy(pem);
    let mut blocks = Vec::new();
    let mut tag = None;
    let mut body = String::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("-----BEGIN ") {
            tag = Some(rest.trim_end_matches("-----").to_owned());
            body.clear();
        } else if line.starts_with("-----END ") {
            if let Some(tag) = tag.take()
                && let Some(der) = base64_decode(&body)
            {
                blocks.push((tag, der));
            }
        } else if tag.is_some() {
            body.push_str(line);
        }
    }
    blocks
}

/// Standard base64 with optional padding, ignoring anything outside the alphabet.
pub(crate) fn base64_decode(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for byte in text.bytes() {
        if byte == b'=' {
            break;
        }
        if byte.is_ascii_whitespace() {
            continue;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pem_blocks_carry_their_tags_in_order() {
        let pem = b"-----BEGIN CERTIFICATE-----\naGk=\n-----END CERTIFICATE-----\n\
                    -----BEGIN PRIVATE KEY-----\nAQID\n-----END PRIVATE KEY-----\n";
        let blocks = pem_blocks(pem);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "CERTIFICATE");
        assert_eq!(blocks[0].1, b"hi");
        assert_eq!(blocks[1].0, "PRIVATE KEY");
        assert_eq!(blocks[1].1, [1u8, 2, 3]);
        assert!(pem_blocks(b"no blocks here").is_empty());
    }

    /// The certificate pair is read once at startup; a renewal is a file write after that.
    #[test]
    fn a_rotated_certificate_is_reloaded_without_a_restart() {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = std::env::temp_dir().join(format!("vipera-tls-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let cert = dir.join("chain.crt");
        let key = dir.join("server.key");
        std::fs::copy(fixtures.join("chain.crt"), &cert).expect("chain");
        std::fs::copy(fixtures.join("server.key"), &key).expect("key");
        let source = TlsSource::new(
            cert.to_str().expect("utf-8 path"),
            key.to_str().expect("utf-8 path"),
        )
        .expect("the fixture pair loads");

        assert!(!source.refresh(), "nothing moved since startup");
        let serving = source.config();

        // A half-written renewal keeps the certificate that works: a broken file must not take
        // the server down, and the bad stamp is remembered instead of retried every connection.
        std::fs::write(&cert, b"not a certificate").expect("write");
        assert!(!source.refresh());
        assert!(Arc::ptr_eq(&serving, &source.config()));
        assert!(!source.refresh(), "the bad stamp is remembered");

        std::fs::write(
            &cert,
            std::fs::read(fixtures.join("server.crt")).expect("leaf"),
        )
        .expect("write");
        assert!(
            source.refresh(),
            "a usable renewal replaces the configuration"
        );
        assert!(!Arc::ptr_eq(&serving, &source.config()));
        assert!(!source.refresh(), "the new stamp is remembered");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn base64_decodes_padding_and_line_breaks() {
        assert_eq!(base64_decode("aGk=").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64_decode("aGk").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64_decode("a\nG\r\nk").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64_decode("AQID").as_deref(), Some(&[1u8, 2, 3][..]));
        assert_eq!(base64_decode(""), Some(Vec::new()));
        assert_eq!(base64_decode("aGk*"), None, "a stray byte is not skipped");
    }
}
