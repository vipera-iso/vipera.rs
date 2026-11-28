//! Minimal safetensors header reader.
//!
//! A safetensors file starts with an eight-byte little-endian header length, followed by that many
//! bytes of UTF-8 JSON describing the tensors; the weights come after the header. This module reads
//! only the header, so a model's tensor list is available without loading its data.

use vipera_errors::ViperaError;

/// Deepest JSON nesting accepted in a header.
const MAX_DEPTH: usize = 32;

/// One tensor declared by a safetensors header, without its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name as stored in the header.
    pub name: String,
    /// Element type string, e.g. `F16`.
    pub dtype: String,
    /// Shape as stored in the header.
    pub shape: Vec<u64>,
    /// Half-open byte range this tensor occupies in the data section.
    pub data_offsets: [u64; 2],
}

/// A parsed safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorsFile {
    /// Header length declared by the first eight bytes.
    pub header_len: u64,
    /// Tensors in header order; `__metadata__` is not a tensor.
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where tensor data begins.
    pub data_offset: usize,
}

impl SafeTensorsFile {
    /// Parse a safetensors file from memory, listing the tensors its header declares.
    ///
    /// The data section is checked for bounds but its bytes are never read.
    pub fn parse(data: &[u8]) -> Result<Self, ViperaError> {
        let prefix = data.get(..8).ok_or_else(|| {
            ViperaError::parse(
                0,
                "safetensors file is shorter than the 8-byte length prefix",
            )
        })?;
        let header_len = u64::from_le_bytes([
            prefix[0], prefix[1], prefix[2], prefix[3], prefix[4], prefix[5], prefix[6], prefix[7],
        ]);
        let declared = usize::try_from(header_len).map_err(|_| {
            ViperaError::parse(0, format!("header length {header_len} is too large"))
        })?;
        let header_end = 8usize
            .checked_add(declared)
            .ok_or_else(|| ViperaError::parse(0, "header length overflows"))?;
        let header_bytes = data.get(8..header_end).ok_or_else(|| {
            ViperaError::parse(
                8,
                format!(
                    "header declares {header_len} bytes, only {} available",
                    data.len().saturating_sub(8)
                ),
            )
        })?;
        let text = std::str::from_utf8(header_bytes)
            .map_err(|e| ViperaError::parse(8, format!("header is not valid UTF-8: {e}")))?;

        let root = Json::parse(text).map_err(|message| ViperaError::parse(8, message))?;
        let Json::Object(entries) = &root else {
            return Err(ViperaError::parse(
                8,
                "safetensors header is not a JSON object",
            ));
        };

        let data_len = data.len() - header_end;
        let mut tensors = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            if name == "__metadata__" {
                continue;
            }
            let Json::Object(fields) = value else {
                return Err(ViperaError::parse(
                    8,
                    format!("tensor \"{name}\" is not an object"),
                ));
            };
            let field = |key: &str| {
                fields
                    .iter()
                    .find(|(field_name, _)| field_name.as_str() == key)
                    .map(|(_, field_value)| field_value)
            };

            let dtype = field("dtype")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    ViperaError::parse(8, format!("tensor \"{name}\" has no string dtype"))
                })?
                .to_owned();
            let shape = field("shape").and_then(Json::as_array).ok_or_else(|| {
                ViperaError::parse(8, format!("tensor \"{name}\" has no array shape"))
            })?;
            let mut dims = Vec::with_capacity(shape.len());
            for dim in shape {
                let dim = dim.as_u64().ok_or_else(|| {
                    ViperaError::parse(8, format!("tensor \"{name}\" has a non-integer dimension"))
                })?;
                dims.push(dim);
            }
            let offsets = field("data_offsets")
                .and_then(Json::as_array)
                .ok_or_else(|| {
                    ViperaError::parse(8, format!("tensor \"{name}\" has no array data_offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(ViperaError::parse(
                    8,
                    format!("tensor \"{name}\" data_offsets must have two entries"),
                ));
            }
            let begin = offsets[0].as_u64().ok_or_else(|| {
                ViperaError::parse(
                    8,
                    format!("tensor \"{name}\" has a non-integer data offset"),
                )
            })?;
            let end = offsets[1].as_u64().ok_or_else(|| {
                ViperaError::parse(
                    8,
                    format!("tensor \"{name}\" has a non-integer data offset"),
                )
            })?;
            if begin > end || end > data_len as u64 {
                return Err(ViperaError::parse(
                    8,
                    format!(
                        "tensor \"{name}\" data offsets [{begin}, {end}) fall outside the {data_len} bytes of data"
                    ),
                ));
            }

            tensors.push(TensorInfo {
                name: name.clone(),
                dtype,
                shape: dims,
                data_offsets: [begin, end],
            });
        }

        Ok(Self {
            header_len,
            tensors,
            data_offset: header_end,
        })
    }
}

/// The subset of JSON a header can contain.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool,
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Result<Self, String> {
        let mut parser = Parser {
            text,
            bytes: text.as_bytes(),
            at: 0,
        };
        parser.ws();
        let value = parser.value(0)?;
        parser.ws();
        if parser.at != parser.bytes.len() {
            return Err(format!("unexpected trailing data at byte {}", parser.at));
        }
        Ok(value)
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(raw) => raw.parse().ok(),
            _ => None,
        }
    }
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
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
            Err(format!("expected '{}' at byte {}", byte as char, self.at))
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("unexpected token at byte {}", self.at))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {MAX_DEPTH} levels"));
        }
        match self.peek() {
            Some(b'n') => self.literal("null").map(|()| Json::Null),
            Some(b't') => self.literal("true").map(|()| Json::Bool),
            Some(b'f') => self.literal("false").map(|()| Json::Bool),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number().map(Json::Number),
            Some(byte) => Err(format!("unexpected byte '{}' at {}", byte as char, self.at)),
            None => Err("unexpected end of header".to_owned()),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(entries));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
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
                                "unknown escape '{}' at byte {}",
                                other as char, self.at
                            ));
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(format!("raw control byte {byte} in string"));
                }
                byte if byte < 0x80 => {
                    out.push(char::from(byte));
                    self.at += 1;
                }
                _ => {
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
            .ok_or_else(|| "truncated '\\u' escape".to_owned())?;
        let text = std::str::from_utf8(digits).map_err(|_| "bad '\\u' escape".to_owned())?;
        let value = u32::from_str_radix(text, 16).map_err(|_| format!("bad '\\u{text}' escape"))?;
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

    fn number(&mut self) -> Result<String, String> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.at += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!("bad number at byte {start}"));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.at += 1;
                }
            }
            _ => return Err(format!("bad number at byte {start}")),
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("bad number at byte {start}"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("bad number at byte {start}"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.at += 1;
            }
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .map(str::to_owned)
            .map_err(|_| format!("bad number at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Length prefix + JSON header + zero-filled data section, all in memory.
    fn fixture(header: &str, data_len: usize) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + data_len, 0);
        bytes
    }

    #[test]
    fn lists_tensors_from_a_synthetic_buffer() {
        let header = r#"{"weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},"bias":{"dtype":"F32","shape":[4],"data_offsets":[64,80]},"__metadata__":{"format":"pt"}}"#;
        let file = SafeTensorsFile::parse(&fixture(header, 80)).expect("valid fixture");

        assert_eq!(file.header_len, header.len() as u64);
        assert_eq!(file.data_offset, 8 + header.len());
        assert_eq!(file.tensors.len(), 2);
        assert_eq!(file.tensors[0].name, "weight");
        assert_eq!(file.tensors[0].dtype, "F16");
        assert_eq!(file.tensors[0].shape, vec![4, 8]);
        assert_eq!(file.tensors[0].data_offsets, [0, 64]);
        assert_eq!(file.tensors[1].name, "bias");
        assert_eq!(file.tensors[1].dtype, "F32");
        assert_eq!(file.tensors[1].shape, vec![4]);
        assert_eq!(file.tensors[1].data_offsets, [64, 80]);
    }

    #[test]
    fn rejects_a_truncated_header() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = fixture(header, 4);
        bytes.truncate(8 + header.len() / 2);

        let err = SafeTensorsFile::parse(&bytes).unwrap_err();
        assert!(err.to_string().contains("only"), "{err}");
    }

    #[test]
    fn rejects_offsets_outside_the_data_section() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#;
        let err = SafeTensorsFile::parse(&fixture(header, 4)).unwrap_err();
        assert!(err.to_string().contains("outside"), "{err}");
    }

    #[test]
    fn rejects_malformed_json() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
        let err = SafeTensorsFile::parse(&fixture(header, 4)).unwrap_err();
        assert!(err.to_string().contains("expected ',' or '}'"), "{err}");
    }
}
