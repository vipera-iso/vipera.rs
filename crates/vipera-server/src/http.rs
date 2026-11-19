//! The HTTP/1.1 surface `vipera serve` speaks: reading one request, writing one response, and the
//! limits that keep a hostile body from costing more than it is worth.
//!
//! Server-only, and the smallest thing the chat contract needs: one request per connection,
//! `Connection: close`, no keep-alive, no chunked bodies. It is not a general HTTP implementation.

use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::json::quote;

/// Bodies larger than this are refused instead of buffered.
pub(crate) const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Request line plus headers must fit here; anything larger is not a chat request.
pub(crate) const MAX_HEAD_BYTES: usize = 64 * 1024;

/// The request path only ever reads and writes bytes, so it is written against this rather than
/// against `TcpStream`; the address and the socket timeouts stay at the accept loop, where the
/// socket still is.
pub(crate) trait Transport: std::io::Read + std::io::Write {}

impl Transport for TcpStream {}

impl Transport for rustls::StreamOwned<rustls::ServerConnection, TcpStream> {}

/// Headers that let a page call the server directly.
///
/// Authentication is a bearer header rather than a cookie, so a wildcard origin does not expose
/// credentials; `authorization` stays an allowed request header.
pub(crate) const CORS_HEADERS: &str = "Access-Control-Allow-Origin: *\r\n\
    Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
    Access-Control-Allow-Headers: authorization, content-type\r\n";

pub(crate) fn sse(stream: &mut dyn Transport, data: &str) -> std::io::Result<()> {
    stream.write_all(format!("data: {data}\n\n").as_bytes())?;
    stream.flush()
}

pub(crate) fn respond(
    stream: &mut dyn Transport,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n{CORS_HEADERS}\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// Answer a CORS preflight: no credentials, no body, and a day-long cache.
pub(crate) fn respond_preflight(stream: &mut dyn Transport) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 204 No Content\r\n{CORS_HEADERS}Access-Control-Max-Age: 86400\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

pub(crate) fn respond_error(
    stream: &mut dyn Transport,
    status: u16,
    code: &str,
    message: &str,
) -> std::io::Result<u16> {
    let kind = match status {
        401 => "authentication_error",
        404 => "invalid_request_error",
        429 => "rate_limit_error",
        500 | 503 => "server_error",
        _ => "invalid_request_error",
    };
    let body = format!(
        r#"{{"error":{{"message":{},"type":"{kind}","code":{}}}}}"#,
        quote(message),
        quote(code)
    );
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    };
    respond(stream, status, reason, "application/json", &body)?;
    Ok(status)
}

pub(crate) fn respond_error_retry(
    stream: &mut dyn Transport,
    status: u16,
    code: &str,
    message: &str,
    retry_after: u64,
) -> std::io::Result<u16> {
    let kind = "rate_limit_error";
    let body = format!(
        r#"{{"error":{{"message":{},"type":"{kind}","code":{}}}}}"#,
        quote(message),
        quote(code)
    );
    let head = format!(
        "HTTP/1.1 {status} Too Many Requests\r\nContent-Type: application/json\r\n{CORS_HEADERS}\
         Content-Length: {}\r\nRetry-After: {retry_after}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(status)
}

pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body: Vec<u8>,
}

impl Request {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub(crate) fn authorized(&self, expected: &str) -> bool {
        match self.header("authorization") {
            Some(value) => value
                .strip_prefix("Bearer ")
                .map(str::trim)
                .is_some_and(|key| constant_time_eq(key.as_bytes(), expected.as_bytes())),
            None => false,
        }
    }
}

/// Compare two secrets without stopping at the first differing byte.
///
/// `==` on `&str` returns at the first mismatch, which tells a peer that can time the response how
/// much of its guess matched. This walks every byte whichever way the comparison goes. The length is
/// not treated as a secret — the response time of a length check is not the key's content.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut differs = 0u8;
    for (left, right) in left.iter().zip(right) {
        differs |= left ^ right;
    }
    differs == 0
}

/// What reading one request produced.
///
/// "The peer left" and "the request was refused" are separate outcomes so that a refusal the
/// server wrote while parsing still reaches the log and the metrics.
pub(crate) enum Incoming {
    /// A complete request, ready to route.
    Request(Request),
    /// The request was refused while it was being read; the peer has been answered with `status`.
    Rejected {
        /// The status that was written back.
        status: u16,
    },
    /// The peer closed before sending a request line; nothing was written, nothing to count.
    Closed,
}

/// Read one request.
pub(crate) fn read_request(stream: &mut dyn Transport) -> std::io::Result<Incoming> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(position) = find(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > MAX_HEAD_BYTES {
            respond_error(
                stream,
                400,
                "invalid_request",
                "request headers are too large",
            )?;
            return Ok(Incoming::Rejected { status: 400 });
        }
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Ok(Incoming::Closed);
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let Some(line) = lines.next() else {
        respond_error(stream, 400, "invalid_request", "malformed request line")?;
        return Ok(Incoming::Rejected { status: 400 });
    };
    let mut parts = line.split(' ');
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        respond_error(stream, 400, "invalid_request", "malformed request line")?;
        return Ok(Incoming::Rejected { status: 400 });
    };
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
    }

    // A header that decides framing or identity may appear once. Two `Content-Length` values are the
    // classic request-smuggling split: this server would read the first, a proxy in front the last,
    // and the two would disagree about where the next request starts.
    for single in ["content-length", "transfer-encoding", "authorization"] {
        if headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case(single))
            .count()
            > 1
        {
            respond_error(
                stream,
                400,
                "invalid_request",
                &format!("{single} appears more than once"),
            )?;
            return Ok(Incoming::Rejected { status: 400 });
        }
    }

    let mut request = Request {
        method: method.to_owned(),
        path: path.split('?').next().unwrap_or(path).to_owned(),
        headers,
        body: Vec::new(),
    };

    if request
        .header("transfer-encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        respond_error(
            stream,
            400,
            "invalid_request",
            "chunked request bodies are not supported; send Content-Length",
        )?;
        return Ok(Incoming::Rejected { status: 400 });
    }
    let length = match request.header("content-length") {
        Some(value) => match value.parse::<usize>() {
            Ok(length) => length,
            Err(_) => {
                respond_error(stream, 400, "invalid_request", "bad Content-Length")?;
                return Ok(Incoming::Rejected { status: 400 });
            }
        },
        None => 0,
    };
    if length > MAX_BODY_BYTES {
        respond_error(stream, 413, "invalid_request", "request body is too large")?;
        return Ok(Incoming::Rejected { status: 413 });
    }
    if request
        .header("expect")
        .is_some_and(|value| value.eq_ignore_ascii_case("100-continue"))
    {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }

    let mut body = buffer[head_end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    request.body = body;
    Ok(Incoming::Request(request))
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn request_id() -> String {
    // Time in milliseconds plus a per-process counter: unique enough for logs, no dependency.
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    base32(millis << 12 | u128::from(counter & 0xfff))
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn base32(mut value: u128) -> String {
    const ALPHABET: &[u8] = b"0123456789abcdefghjkmnpqrstvwxyz";
    let mut out = Vec::new();
    while value > 0 {
        out.push(ALPHABET[(value % 32) as usize]);
        value /= 32;
    }
    if out.is_empty() {
        out.push(b'0');
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

pub(crate) fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use crate::handler::map_engine_error;
    use vipera_errors::ViperaError;

    use super::*;

    /// A peer that has already sent `bytes` and then holds the connection open.
    struct Sent {
        bytes: Vec<u8>,
        at: usize,
        written: Vec<u8>,
    }

    impl Transport for Sent {}

    impl Sent {
        fn new(bytes: &[u8]) -> Self {
            Self {
                bytes: bytes.to_vec(),
                at: 0,
                written: Vec::new(),
            }
        }
    }

    impl std::io::Read for Sent {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let left = &self.bytes[self.at..];
            let take = left.len().min(out.len());
            out[..take].copy_from_slice(&left[..take]);
            self.at += take;
            Ok(take)
        }
    }

    impl std::io::Write for Sent {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.written.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_refused_request_is_reported_as_answered_so_it_can_be_logged() {
        // A body the limit refuses before a byte of it is read. The status has to come back to
        // `serve_connection`, or the refusal is answered and then leaves no log line and no metric.
        let mut peer =
            Sent::new(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 9437184\r\n\r\n");
        match read_request(&mut peer).expect("read") {
            Incoming::Rejected { status } => assert_eq!(status, 413),
            Incoming::Request(_) => panic!("an oversized body must not become a request"),
            Incoming::Closed => panic!("the peer did not close"),
        }
        let answer = String::from_utf8_lossy(&peer.written);
        assert!(answer.starts_with("HTTP/1.1 413"), "{answer}");
    }

    #[test]
    fn two_content_lengths_are_refused_rather_than_guessed() {
        let mut peer = Sent::new(
            b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 500\r\n\r\n",
        );
        match read_request(&mut peer).expect("read") {
            Incoming::Rejected { status } => assert_eq!(status, 400),
            Incoming::Request(_) => panic!("two lengths must not become a request"),
            Incoming::Closed => panic!("the peer did not close"),
        }
        let answer = String::from_utf8_lossy(&peer.written);
        assert!(
            answer.contains("content-length appears more than once"),
            "{answer}"
        );
    }

    #[test]
    fn the_key_comparison_does_not_stop_at_the_first_differing_byte() {
        assert!(constant_time_eq(b"sk-local", b"sk-local"));
        assert!(!constant_time_eq(b"sk-local", b"sk-locaX"));
        assert!(!constant_time_eq(b"Xk-local", b"sk-local"));
        assert!(!constant_time_eq(b"sk-local", b"sk-local-and-more"));
        assert!(!constant_time_eq(b"", b"sk-local"));
    }

    #[test]
    fn a_peer_that_leaves_before_a_request_line_is_not_a_refusal() {
        let mut peer = Sent::new(b"");
        assert!(
            matches!(read_request(&mut peer).expect("read"), Incoming::Closed),
            "an empty connection is a close, not something to count"
        );
        assert!(peer.written.is_empty(), "nothing should be written back");
    }

    #[test]
    fn bearer_tokens_are_compared_exactly() {
        let request = Request {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: vec![("Authorization".to_owned(), "Bearer secret".to_owned())],
            body: Vec::new(),
        };
        assert!(request.authorized("secret"));
        assert!(!request.authorized("secre"));
        assert!(!request.authorized(""));
    }

    #[test]
    fn error_bodies_use_the_contract_envelope() {
        let error = map_engine_error(&ViperaError::infeasible("too big"));
        assert_eq!(error, (422, "context_too_long"));
    }
}
