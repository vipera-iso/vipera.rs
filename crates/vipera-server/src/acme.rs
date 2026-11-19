//! ACME (RFC 8555) certificate provisioning for `vipera serve`, behind the `acme` feature.
//!
//! This is HTTP-01 only, and it is deliberately small: the CA writes a token into a file, something
//! on port 80 answers for it, and the issued chain is written to `--tls-cert`/`--tls-key`. The
//! existing [`crate::serve::TlsSource`] reload picks a renewal up without a restart, so there is no
//! second TLS configuration path here.
//!
//! The two threads are the whole design:
//!
//! * `vipera-acme-http` listens on `--acme-http-address` and answers
//!   `GET /.well-known/acme-challenge/<token>` from a map the ACME thread fills in. On a real
//!   deployment something must forward the domain's port 80 there; the listener needs the privilege
//!   to bind it.
//! * `vipera-acme` runs a current-thread Tokio runtime: create (or restore) the account, order the
//!   certificate, provision the challenge, poll until ready, finalize, write both files atomically,
//!   then renew monthly.
//!
//! No ACME behaviour is claimed beyond this: the client is `instant-acme`, the account key is
//! persisted next to the certificate, and failures are retried every five minutes rather than
//! bringing the server down.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use instant_acme::{
    Account, AccountCredentials, AuthorizationStatus, ChallengeType, Identifier, NewAccount,
    NewOrder, OrderStatus, RetryPolicy,
};
use vipera_errors::ViperaError;

use crate::serve::ConnectionBudget;
use crate::tls::pem_blocks;

/// Renew this long before the certificate lapses. Let's Encrypt certificates last ninety days, so a
/// failed renewal still leaves a month of retries.
const RENEW_BEFORE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// Wait this long before re-ordering when the certificate's expiry cannot be read, or when it was
/// issued with less than [`RENEW_BEFORE`] of life left.
const RENEW_AFTER: Duration = RENEW_BEFORE;
/// How long to wait before retrying a failed issuance.
const RETRY_AFTER: Duration = Duration::from_secs(5 * 60);
/// The CA will not wait around; a challenge answer must appear quickly.
const CHALLENGE_PATH: &str = "/.well-known/acme-challenge/";

/// How many challenge requests may be answered at once. Port 80 is scanned by everything on the
/// internet, so this listener gets the same bound the main server has.
const MAX_CHALLENGE_CONNECTIONS: usize = 8;

/// What `vipera serve` was told about the certificate it should hold.
pub struct Options {
    /// The DNS name to order.
    pub domain: String,
    /// Optional contact URI (`mailto:...`) sent with the account.
    pub contact: Option<String>,
    /// ACME directory URL.
    pub directory: String,
    /// Address the plain-HTTP challenge listener binds.
    pub http_address: String,
    /// Extra root the client trusts, for a CA that is not publicly trusted.
    ///
    /// `None` in `vipera serve`, which talks to a CA the platform already trusts. The test suite
    /// points it at the repository's throwaway authority so the order flow runs against a fake CA
    /// over the real HTTPS client.
    pub root_cert: Option<PathBuf>,
}

/// Start the challenge listener and the renewal thread.
///
/// `cert` and `key` are the same paths `--tls-cert`/`--tls-key` name: this writes them, the server's
/// existing reload reads them.
pub fn spawn(options: Options, cert: &Path, key: &Path) -> Result<(), ViperaError> {
    let listener = TcpListener::bind(&options.http_address).map_err(|error| {
        ViperaError::invalid_argument(format!(
            "the ACME challenge listener cannot bind {}: {error}; point --acme-http-address at a \
             port something can forward the domain's port 80 to",
            options.http_address
        ))
    })?;
    let challenges: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

    let store = Arc::clone(&challenges);
    std::thread::Builder::new()
        .name("vipera-acme-http".to_owned())
        .spawn(move || answer_challenges(listener, store))
        .map_err(|error| {
            ViperaError::internal(format!("cannot start the challenge thread: {error}"))
        })?;

    let cert = cert.to_path_buf();
    let key = key.to_path_buf();
    std::thread::Builder::new()
        .name("vipera-acme".to_owned())
        .spawn(move || renew_forever(options, cert, key, challenges))
        .map_err(|error| ViperaError::internal(format!("cannot start the ACME thread: {error}")))?;
    Ok(())
}

/// Wait until both certificate files exist, so the TLS listener can start on a first issuance.
pub fn wait_for_certificate(cert: &Path, key: &Path, timeout: Duration) -> Result<(), ViperaError> {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if readable(cert) && readable(key) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(ViperaError::internal(format!(
        "no ACME certificate was issued within {}s; check that the domain's port 80 forwards to \
         the challenge listener and that it answers /.well-known/acme-challenge/, then start again",
        timeout.as_secs()
    )))
}

fn readable(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() > 0)
        .unwrap_or(false)
}

fn answer_challenges(listener: TcpListener, challenges: Arc<Mutex<HashMap<String, String>>>) {
    // One thread per connection, and a connection past the limit is dropped rather than spawning a
    // thread the box cannot back: a flood on port 80 must not take the process holding the key
    // down with it.
    let budget = Arc::new(ConnectionBudget::new(MAX_CHALLENGE_CONNECTIONS));
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else {
            continue;
        };
        let Some(slot) = budget.try_acquire() else {
            continue;
        };
        let store = Arc::clone(&challenges);
        // The thread owns the stream; nobody joins it, and that is fine for a reply this small.
        // Dropping the handle detaches it. The slot lives as long as the thread and frees itself
        // when the reply has been written.
        let _ = std::thread::Builder::new().spawn(move || {
            let _slot = slot;
            let _ = answer_one(&mut stream, &store);
        });
    }
}

fn answer_one(
    stream: &mut TcpStream,
    challenges: &Mutex<HashMap<String, String>>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut head = Vec::new();
    let mut byte = [0_u8; 1];
    while head.len() < 8192 && !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            break;
        }
        head.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&head);
    let mut parts = text.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    let answer = if method == "GET" {
        path.strip_prefix(CHALLENGE_PATH)
            .and_then(|token| challenges.lock().ok()?.get(token).cloned())
    } else {
        None
    };
    match answer {
        Some(answer) => write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{answer}",
            answer.len()
        ),
        None => write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
    }
}

/// The instant the certificate at `cert` stops being valid, from its own `notAfter`.
///
/// Read rather than stored so a certificate written by another tool, or by an earlier run of this
/// one, is treated the same as a certificate it wrote itself.
fn expiry_of(cert: &Path) -> Result<SystemTime, String> {
    let pem =
        std::fs::read(cert).map_err(|error| format!("cannot read {}: {error}", cert.display()))?;
    let der = pem_blocks(&pem)
        .into_iter()
        .find(|(tag, _)| tag.ends_with("CERTIFICATE"))
        .map(|(_, der)| der)
        .ok_or_else(|| format!("{} has no CERTIFICATE block", cert.display()))?;
    not_after(&der)
}

/// How long to wait before the certificate that expires at `expiry` has to be renewed.
///
/// A certificate with more than [`RENEW_BEFORE`] left is reused until a month before it lapses. One
/// issued with less than that left is re-ordered after half its remaining life, at least an hour
/// from now, so a short lifetime becomes an early renewal rather than a stream of orders.
fn wait_before_renewing(expiry: Option<SystemTime>, now: SystemTime) -> Duration {
    let Some(expiry) = expiry else {
        return RENEW_AFTER;
    };
    let left = expiry.duration_since(now).unwrap_or_default();
    if left > RENEW_BEFORE {
        left - RENEW_BEFORE
    } else {
        (left / 2).max(Duration::from_secs(3600))
    }
}

/// Whether a certificate expiring at `expiry` has to be ordered again now.
fn is_due(expiry: Option<SystemTime>, now: SystemTime) -> bool {
    expiry.is_none_or(|expiry| expiry.duration_since(now).unwrap_or_default() <= RENEW_BEFORE)
}

fn renew_forever(
    options: Options,
    cert: PathBuf,
    key: PathBuf,
    challenges: Arc<Mutex<HashMap<String, String>>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("vipera: ACME needs a runtime and none could start: {error}");
            return;
        }
    };
    loop {
        // A restart must not cost a certificate. Let's Encrypt allows five duplicate certificates a
        // week per name, so a crash-loop would spend them all and leave the domain unable to renew
        // for a week. An existing certificate that still has a month left is reused, and the thread
        // sleeps until it is actually due.
        let existing = expiry_of(&cert).ok();
        if !is_due(existing, SystemTime::now()) {
            let wait = wait_before_renewing(existing, SystemTime::now());
            eprintln!(
                "vipera: ACME reusing the certificate for {} (renewing in {}h)",
                options.domain,
                wait.as_secs() / 3600
            );
            std::thread::sleep(wait);
            continue;
        }
        if let Some(expiry) = existing {
            eprintln!(
                "vipera: ACME the certificate for {} expires {}s from now, renewing it",
                options.domain,
                expiry
                    .duration_since(SystemTime::now())
                    .unwrap_or_default()
                    .as_secs()
            );
        } else {
            eprintln!(
                "vipera: ACME no usable certificate at {} yet; ordering one for {}",
                cert.display(),
                options.domain
            );
        }
        match runtime.block_on(issue(&options, &cert, &key, &challenges)) {
            Ok(()) => {
                eprintln!(
                    "vipera: ACME wrote a certificate for {} to {}",
                    options.domain,
                    cert.display()
                );
                // Sleep on the certificate that was just written, not on a fixed month: a CA that
                // issued a shorter lifetime is renewed before it lapses, and one that issued a
                // longer one is not re-ordered early.
                let wait = wait_before_renewing(expiry_of(&cert).ok(), SystemTime::now());
                std::thread::sleep(wait);
            }
            Err(error) => {
                eprintln!("vipera: ACME: {error}");
                std::thread::sleep(RETRY_AFTER);
            }
        }
    }
}

const SEQUENCE: u8 = 0x30;
const UTC_TIME: u8 = 0x17;
const GENERALIZED_TIME: u8 = 0x18;
const EXPLICIT_VERSION: u8 = 0xa0;

/// Split one DER tag-length-value off the front: the tag, its value, and what follows.
///
/// Lengths are the definite form only, which is all X.509 uses.
fn tlv(data: &[u8]) -> Result<(u8, &[u8], &[u8]), String> {
    let Some((&tag, rest)) = data.split_first() else {
        return Err("truncated DER".to_owned());
    };
    let Some((&first, rest)) = rest.split_first() else {
        return Err("truncated DER length".to_owned());
    };
    let (len, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let count = usize::from(first & 0x7f);
        if count == 0 || count > 4 || rest.len() < count {
            return Err("bad DER length".to_owned());
        }
        let mut len = 0usize;
        for byte in &rest[..count] {
            len = (len << 8) | usize::from(*byte);
        }
        (len, &rest[count..])
    };
    if rest.len() < len {
        return Err("DER value runs past the certificate".to_owned());
    }
    let (value, tail) = rest.split_at(len);
    Ok((tag, value, tail))
}

/// The `notAfter` of a DER certificate, without an X.509 dependency.
///
/// Walks the fixed prefix of `tbsCertificate` — optional version, serial number, signature algorithm,
/// issuer — to the validity block, and reads its second `Time`. Every other field is skipped by its
/// length, so nothing is parsed that is not needed.
fn not_after(der: &[u8]) -> Result<SystemTime, String> {
    let (tag, certificate, _) = tlv(der)?;
    if tag != SEQUENCE {
        return Err("certificate is not a SEQUENCE".to_owned());
    }
    let (tag, tbs, _) = tlv(certificate)?;
    if tag != SEQUENCE {
        return Err("tbsCertificate is not a SEQUENCE".to_owned());
    }
    let mut rest = tbs;
    let (tag, _, tail) = tlv(rest)?;
    if tag == EXPLICIT_VERSION {
        rest = tail;
    }
    for field in ["serialNumber", "signature", "issuer"] {
        let (_, _, tail) = tlv(rest).map_err(|error| format!("{field}: {error}"))?;
        rest = tail;
    }
    let (tag, validity, _) = tlv(rest)?;
    if tag != SEQUENCE {
        return Err("validity is not a SEQUENCE".to_owned());
    }
    let (_, _, after_not_before) = tlv(validity)?;
    let (tag, time, _) = tlv(after_not_before)?;
    parse_time(tag, time)
}

fn parse_time(tag: u8, text: &[u8]) -> Result<SystemTime, String> {
    let text = std::str::from_utf8(text).map_err(|_| "certificate time is not ASCII".to_owned())?;
    let text = text
        .strip_suffix('Z')
        .ok_or_else(|| format!("certificate time {text:?} is not UTC"))?;
    let (year, rest) = match tag {
        UTC_TIME if text.len() == 12 => {
            let year: i64 = text[..2]
                .parse()
                .map_err(|_| format!("certificate year {text:?} is not a number"))?;
            (
                if year < 50 { 2000 + year } else { 1900 + year },
                &text[2..],
            )
        }
        GENERALIZED_TIME if text.len() == 14 => {
            let year: i64 = text[..4]
                .parse()
                .map_err(|_| format!("certificate year {text:?} is not a number"))?;
            (year, &text[4..])
        }
        _ => {
            return Err(format!(
                "certificate time {text:?} is not a form this reader parses"
            ));
        }
    };
    let field = |at: usize| -> Result<i64, String> {
        rest.get(at..at + 2)
            .and_then(|field| field.parse().ok())
            .ok_or_else(|| format!("certificate time {text:?} has a bad field"))
    };
    let (month, day, hour, minute, second) =
        (field(0)?, field(2)?, field(4)?, field(6)?, field(8)?);
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    if seconds < 0 {
        return Err(format!("certificate time {text:?} is before 1970"));
    }
    Ok(UNIX_EPOCH + Duration::from_secs(seconds as u64))
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's civil calendar).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_prime = (month + 9) % 12;
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

async fn issue(
    options: &Options,
    cert: &Path,
    key: &Path,
    challenges: &Mutex<HashMap<String, String>>,
) -> Result<(), String> {
    let account = account(options, cert).await?;
    let identifiers = [Identifier::Dns(options.domain.clone())];
    let mut order = account
        .new_order(&NewOrder::new(&identifiers))
        .await
        .map_err(|error| format!("cannot create an order: {error}"))?;

    // Every answer published for this order, so the one exit path that owns them takes them all
    // down again.
    let mut provisioned = Provisioned {
        map: challenges,
        tokens: Vec::new(),
    };
    let mut authorizations = order.authorizations();
    while let Some(result) = authorizations.next().await {
        let mut authorization =
            result.map_err(|error| format!("cannot read an authorization: {error}"))?;
        if authorization.status != AuthorizationStatus::Pending {
            continue;
        }
        let Some(mut challenge) = authorization.challenge(ChallengeType::Http01) else {
            return Err(format!(
                "the CA offered no http-01 challenge for {}",
                options.domain
            ));
        };
        let value = challenge.key_authorization().as_str().to_owned();
        // The answer has to stay readable until the CA has finished fetching it. That is after
        // `poll_ready` below, not after `set_ready`: the CA validates asynchronously, so dropping
        // the token here would race the fetch and turn a good challenge into a 404.
        provisioned.insert(&challenge.token, value)?;
        let ready = challenge.set_ready().await;
        ready.map_err(|error| format!("the CA refused the challenge: {error}"))?;
    }

    let status = order
        .poll_ready(&RetryPolicy::default())
        .await
        .map_err(|error| format!("the order never became ready: {error}"))?;
    if status != OrderStatus::Ready {
        return Err(format!("the order ended {status:?}"));
    }
    let private_key = order
        .finalize()
        .await
        .map_err(|error| format!("cannot finalize the order: {error}"))?;
    let chain = order
        .poll_certificate(&RetryPolicy::default())
        .await
        .map_err(|error| format!("the CA never returned the certificate: {error}"))?;

    write_atomically(key, private_key.as_bytes(), true)?;
    write_atomically(cert, chain.as_bytes(), false)?;
    Ok(())
}

/// An account builder that trusts what the platform trusts, plus the root `options` names when a
/// test points at a CA that is not publicly trusted.
fn builder(options: &Options) -> Result<instant_acme::AccountBuilder, String> {
    match &options.root_cert {
        Some(root) => Account::builder_with_root(root)
            .map_err(|error| format!("cannot trust {}: {error}", root.display())),
        None => {
            Account::builder().map_err(|error| format!("cannot build an ACME account: {error}"))
        }
    }
}

/// The stored account, or a fresh one. The credentials live next to the certificate so a restart
/// reuses the same ACME account instead of creating one per boot. A stored file the CA no longer
/// knows (a directory change) is replaced by a new account.
async fn account(options: &Options, cert: &Path) -> Result<Account, String> {
    let path = credentials_path(cert);
    if let Ok(bytes) = std::fs::read(&path)
        && let Ok(credentials) = serde_json::from_slice::<AccountCredentials>(&bytes)
        && let Ok(account) = builder(options)?.from_credentials(credentials).await
    {
        return Ok(account);
    }

    let contacts: Vec<&str> = options.contact.iter().map(String::as_str).collect();
    let (account, credentials) = builder(options)?
        .create(
            &NewAccount {
                contact: &contacts,
                terms_of_service_agreed: true,
                only_return_existing: false,
            },
            options.directory.clone(),
            None,
        )
        .await
        .map_err(|error| format!("cannot create an ACME account: {error}"))?;
    let bytes = serde_json::to_vec_pretty(&credentials).map_err(|error| error.to_string())?;
    write_atomically(&path, &bytes, true)?;
    Ok(account)
}

fn credentials_path(cert: &Path) -> PathBuf {
    cert.with_extension("account.json")
}

/// The HTTP-01 answers published for one order, taken down together when `issue` returns.
///
/// A `Drop` rather than a call at the end, so an early error still removes the answers instead of
/// leaving them readable for the life of the process.
struct Provisioned<'a> {
    map: &'a Mutex<HashMap<String, String>>,
    tokens: Vec<String>,
}

impl Provisioned<'_> {
    fn insert(&mut self, token: &str, value: String) -> Result<(), String> {
        self.map
            .lock()
            .map_err(|_| "the challenge map is poisoned".to_owned())?
            .insert(token.to_owned(), value);
        self.tokens.push(token.to_owned());
        Ok(())
    }
}

impl Drop for Provisioned<'_> {
    fn drop(&mut self) {
        if let Ok(mut map) = self.map.lock() {
            for token in &self.tokens {
                map.remove(token);
            }
        }
    }
}

fn write_atomically(path: &Path, bytes: &[u8], secret: bool) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = create(&temporary, secret)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", temporary.display()))?;
    drop(file);
    std::fs::rename(&temporary, path)
        .map_err(|error| format!("cannot replace {}: {error}", path.display()))
}

/// Create the temporary file, owner-only when it holds a key.
///
/// `std::fs::write` would leave a private key at the umask default, which on a normal box is
/// world-readable; the account key and the certificate key are both secrets.
fn create(path: &Path, secret: bool) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        if secret {
            options.mode(0o600);
        }
    }
    #[cfg(not(unix))]
    let _ = secret;
    let file = options.open(path)?;
    // `mode` is masked by the umask, so set it again: an operator with an odd umask still gets
    // exactly owner-only on a key.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if secret {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
    }
    Ok(file)
}

#[cfg(test)]
#[path = "acme_test_ca.rs"]
mod test_ca;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_challenge_answer_is_served_until_the_order_is_over() {
        let map = Mutex::new(HashMap::new());
        {
            let mut provisioned = Provisioned {
                map: &map,
                tokens: Vec::new(),
            };
            provisioned
                .insert("token-a", "token-a.thumbprint".to_owned())
                .expect("insert");
            // While the order is in flight the listener has to answer: the CA fetches after
            // `set_ready`, which is inside this scope.
            assert_eq!(
                map.lock().expect("map").get("token-a").map(String::as_str),
                Some("token-a.thumbprint")
            );
        }
        // Once the order is done nothing published is still readable; the listener is back to 404.
        assert!(map.lock().expect("map").is_empty());
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    /// The whole order flow, against a CA that speaks RFC 8555 over TLS and validates the
    /// http-01 answer by fetching it from the real challenge listener.
    ///
    /// This is the test the shape of `issue` is for: a token that is published, fetched by the CA
    /// while the order is in flight, and taken down afterwards; a CSR at finalize; both files
    /// written; and nothing left readable on the challenge port.
    #[test]
    fn a_fake_ca_walks_a_whole_order_and_the_challenge_is_fetched() {
        use crate::acme::test_ca::{FakeCa, TOKEN};

        let chain = std::fs::read(fixture("chain.crt")).expect("read the fixture chain");
        let config = crate::tls::tls_config(
            &std::fs::read(fixture("chain.crt")).expect("chain"),
            &std::fs::read(fixture("server.key")).expect("key"),
        )
        .expect("serve TLS with the fixture pair");

        // The challenge listener `vipera-acme-http` runs, on a port the fake CA is told about.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the challenge listener");
        let challenge_address = listener.local_addr().expect("addr").to_string();
        let challenges: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let server = Arc::clone(&challenges);
        std::thread::spawn(move || answer_challenges(listener, server));

        let ca = FakeCa::spawn(challenge_address, chain.clone(), config);

        let dir = std::env::temp_dir().join(format!(
            "vipera-acme-order-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let cert = dir.join("issued.crt");
        let key = dir.join("issued.key");
        let options = Options {
            domain: "localhost".to_owned(),
            contact: None,
            directory: ca.directory(),
            http_address: "127.0.0.1:0".to_owned(),
            root_cert: Some(fixture("ca.crt")),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        if let Err(error) = runtime.block_on(issue(&options, &cert, &key, &challenges)) {
            let observed = ca.observed.lock().expect("observed");
            panic!(
                "the order did not complete: {error}\n  asked: {:?}\n  validated: {}\n  answer: {:?}",
                observed.paths, observed.validated, observed.challenge_answer
            );
        }

        // What the CA saw: the answer proved the token it handed out, and the CSR arrived.
        let observed = ca.observed.lock().expect("observed");
        assert!(observed.validated, "the CA never validated the challenge");
        let answer = observed.challenge_answer.as_deref().unwrap_or_default();
        assert!(answer.starts_with(TOKEN), "the CA fetched {answer:?}");
        assert!(
            answer.len() > TOKEN.len() + 1,
            "no thumbprint after the token"
        );
        assert!(
            observed.csr_len.unwrap_or(0) > 100,
            "finalize carried no CSR: {:?}",
            observed.csr_len
        );
        for path in [
            "POST /new-account",
            "POST /new-order",
            "POST /challenge/1",
            "POST /finalize/1",
            "POST /cert/1",
        ] {
            assert!(
                observed.paths.iter().any(|asked| asked == path),
                "{path} was never asked for: {:?}",
                observed.paths
            );
        }
        drop(observed);

        // Both files written, the key owner-only, the chain byte for byte what the CA returned.
        assert_eq!(std::fs::read(&cert).expect("certificate"), chain);
        let written_key = std::fs::read_to_string(&key).expect("key");
        assert!(written_key.contains("PRIVATE KEY"), "{written_key}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&key)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "the key is {mode:o}, not 0600");
        }

        // The answers are gone once the order is over: port 80 is back to 404.
        assert!(challenges.lock().expect("map").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_certificates_expiry_is_read_from_its_own_bytes() {
        // `openssl x509 -enddate -noout -in server.crt`: notAfter=Nov 13 01:35:44 2036 GMT.
        let expiry = expiry_of(&fixture("server.crt")).expect("expiry");
        assert_eq!(
            expiry
                .duration_since(UNIX_EPOCH)
                .expect("after 1970")
                .as_secs(),
            2_110_152_944
        );
    }

    #[test]
    fn a_chain_is_read_leaf_first() {
        // `--tls-cert` holds the leaf followed by the authority; the leaf's expiry is the one that
        // decides a renewal, and it is the first CERTIFICATE block.
        let expiry = expiry_of(&fixture("chain.crt")).expect("expiry");
        assert_eq!(expiry, expiry_of(&fixture("server.crt")).expect("expiry"));
    }

    #[test]
    fn a_file_that_is_not_a_certificate_says_so() {
        let error = expiry_of(&fixture("server.key")).expect_err("a key is not a certificate");
        assert!(error.contains("CERTIFICATE"), "{error}");
    }

    #[test]
    fn a_certificate_with_a_month_left_is_reused_until_it_is_due() {
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let ninety_days = Duration::from_secs(90 * 24 * 60 * 60);
        // Ninety days out: wait sixty, so the renewal lands thirty days before it lapses.
        assert_eq!(
            wait_before_renewing(Some(now + ninety_days), now),
            Duration::from_secs(60 * 24 * 60 * 60)
        );
        // Ten days out: due now, but re-ordered only after half of what is left, not in a loop.
        assert_eq!(
            wait_before_renewing(Some(now + Duration::from_secs(10 * 24 * 60 * 60)), now),
            Duration::from_secs(5 * 24 * 60 * 60)
        );
        // Already expired: an hour, so a broken clock cannot spin on the CA.
        assert_eq!(
            wait_before_renewing(Some(now), now),
            Duration::from_secs(3600)
        );
        // No readable expiry: the fixed month.
        assert_eq!(wait_before_renewing(None, now), RENEW_AFTER);
    }

    #[test]
    fn a_restart_reuses_a_certificate_with_time_left() {
        // The fixture is valid until 2036; a process starting today must not order anything.
        let expiry = expiry_of(&fixture("server.crt")).expect("expiry");
        let left = expiry
            .duration_since(SystemTime::now())
            .expect("the fixture has not expired");
        assert!(
            left > RENEW_BEFORE,
            "{left:?} left is not more than a month"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_private_key_is_written_owner_only() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = std::env::temp_dir().join(format!("vipera-acme-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let key = dir.join("server.key");
        write_atomically(&key, b"secret", true).expect("write the key");
        let mode = std::fs::metadata(&key)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "the key is {mode:o}, not 0600");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
