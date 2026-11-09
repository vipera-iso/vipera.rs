//! A small JSON reader and writer for the HTTP surface.
//!
//! The project takes no serialization dependency (see `research.md`, R3: manual parsing for a handful
//! of commands), so the server parses request bodies and writes responses with this module. It is
//! deliberately strict: malformed input is an error, never a guess, and nesting is bounded so a
//! hostile body cannot exhaust the stack.
//!
//! Only what the chat contract needs is supported: objects, arrays, strings, numbers, booleans and
//! `null`. Numbers become `f64`, which is exact for the token counts and ranges this API accepts.

use std::fmt::Write as _;

/// The deepest nesting accepted; enough for any chat request, shallow enough to be safe.
const MAX_DEPTH: usize = 64;

/// One JSON value, as parsed from a request body.
///
/// Objects keep their entries in document order and allow duplicate keys; [`Json::get`] returns the
/// first match. Order matters here because responses are written in the order the contract lists
/// their fields, and nothing in the chat schema depends on sorting.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// The `null` literal.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// Any number; fractions and exponents were already folded into an `f64`.
    Number(f64),
    /// A string with escapes decoded and the surrounding quotes removed.
    String(String),
    /// An ordered list of values.
    Array(Vec<Json>),
    /// Key/value pairs in document order, duplicate keys included.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse one JSON document. Trailing non-whitespace is an error.
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut parser = Parser {
            text,
            bytes: text.as_bytes(),
            at: 0,
        };
        parser.skip_whitespace();
        let value = parser.value(0)?;
        parser.skip_whitespace();
        if parser.at != parser.bytes.len() {
            return Err(format!("unexpected trailing data at byte {}", parser.at));
        }
        Ok(value)
    }

    /// The value under `key`, when this is an object that has one.
    ///
    /// Duplicate keys resolve to the first occurrence. Anything that is not an object is `None`.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// This value as a string slice, when it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(value) => Some(value),
            _ => None,
        }
    }

    /// This value as an `f64`, when it is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// This value as a `bool`, when it is a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// This value as a slice, when it is an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    /// A non-negative integer field, refusing fractions and negatives.
    pub fn as_u32(&self) -> Option<u32> {
        let value = self.as_f64()?;
        if value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
            return None;
        }
        Some(value as u32)
    }
}

/// A JSON string literal, quotes and escapes included.
pub fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if (ch as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.bytes.get(self.at) {
            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected {} at byte {}", byte as char, self.at))
        }
    }

    fn literal(&mut self, text: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(text.as_bytes()) {
            self.at += text.len();
            Ok(value)
        } else {
            Err(format!("unexpected token at byte {}", self.at))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {MAX_DEPTH} levels"));
        }
        match self.peek() {
            None => Err("unexpected end of input".to_owned()),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(byte) => Err(format!("unexpected byte {} at {}", byte as char, self.at)),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected , or ] at byte {}", self.at)),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(entries));
                }
                _ => return Err(format!("expected , or }} at byte {}", self.at)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| "unterminated string".to_owned())?;
            match byte {
                b'"' => {
                    self.at += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.at += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_owned())?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode()?),
                        other => {
                            return Err(format!(
                                "unknown escape \\{} at byte {}",
                                other as char, self.at
                            ));
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(format!("raw control byte {} in string", byte));
                }
                byte if byte < 0x80 => {
                    out.push(char::from(byte));
                    self.at += 1;
                }
                _ => {
                    // Copy one whole UTF-8 scalar. The input is already a
                    // &str, so this slice is on a char boundary and costs O(1)
                    // (the previous from_utf8 validated the whole remainder
                    // per character — O(n^2) on large bodies).
                    let ch = self.text[self.at..]
                        .chars()
                        .next()
                        .ok_or_else(|| "unterminated string".to_owned())?;
                    out.push(ch);
                    self.at += ch.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or_else(|| "truncated \\u escape".to_owned())?;
        let text = std::str::from_utf8(digits).map_err(|_| "bad \\u escape".to_owned())?;
        let value = u32::from_str_radix(text, 16).map_err(|_| format!("bad \\u{text} escape"))?;
        self.at += 4;
        Ok(value)
    }

    fn unicode(&mut self) -> Result<char, String> {
        let first = self.hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            if self.bytes.get(self.at..self.at + 2) != Some(b"\\u") {
                return Err("lone high surrogate".to_owned());
            }
            self.at += 2;
            let second = self.hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err("bad low surrogate".to_owned());
            }
            let combined = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
            char::from_u32(combined).ok_or_else(|| "bad surrogate pair".to_owned())
        } else {
            char::from_u32(first).ok_or_else(|| format!("bad code point {first:04x}"))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at])
            .map_err(|_| "bad number".to_owned())?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| format!("bad number \"{text}\" at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_chat_request_shape() {
        let body = r#"{"model":"qwen2","stream":true,"max_tokens":16,
            "messages":[{"role":"system","content":"Be terse."},{"role":"user","content":"Hi"}]}"#;
        let request = Json::parse(body).expect("valid");
        assert_eq!(request.get("model").and_then(Json::as_str), Some("qwen2"));
        assert_eq!(request.get("stream").and_then(Json::as_bool), Some(true));
        assert_eq!(request.get("max_tokens").and_then(Json::as_u32), Some(16));
        let messages = request
            .get("messages")
            .and_then(Json::as_array)
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[1].get("content").and_then(Json::as_str),
            Some("Hi")
        );
    }

    #[test]
    fn reads_escapes_and_surrogate_pairs() {
        let value = Json::parse(r#""a\"b\\c\nd\u00e9\ud83d\ude00""#).expect("valid");
        assert_eq!(value.as_str(), Some("a\"b\\c\ndé😀"));
    }

    #[test]
    fn a_large_string_parses_without_quadratic_revalidation() {
        let big = "a".repeat(200_000) + "é😀" + &"b".repeat(20_000);
        let body = format!("\"{big}\"");
        let start = std::time::Instant::now();
        let parsed = Json::parse(&body).expect("valid");
        assert_eq!(parsed.as_str(), Some(big.as_str()));
        assert!(
            start.elapsed().as_secs() < 5,
            "parse took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn writes_a_string_that_parses_back() {
        let original = "line\n\"quoted\"\ttab \u{1} end";
        let literal = quote(original);
        assert_eq!(Json::parse(&literal).unwrap().as_str(), Some(original));
    }

    #[test]
    fn refuses_malformed_bodies() {
        for body in [
            "",
            "{",
            "{\"a\":}",
            "{\"a\" 1}",
            "[1,]",
            "\"unterminated",
            "{\"a\":1} extra",
            "nul",
        ] {
            assert!(Json::parse(body).is_err(), "{body} should be rejected");
        }
    }

    #[test]
    fn refuses_deep_nesting_instead_of_overflowing_the_stack() {
        let deep = "[".repeat(MAX_DEPTH + 2);
        let error = Json::parse(&deep).expect_err("too deep");
        assert!(error.contains("nesting deeper"), "{error}");
    }

    #[test]
    fn integer_fields_reject_fractions_and_negatives() {
        let value = Json::parse(r#"{"a":-1,"b":1.5,"c":3}"#).expect("valid");
        assert_eq!(value.get("a").and_then(Json::as_u32), None);
        assert_eq!(value.get("b").and_then(Json::as_u32), None);
        assert_eq!(value.get("c").and_then(Json::as_u32), Some(3));
    }
}
