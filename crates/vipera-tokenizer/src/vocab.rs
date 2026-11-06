//! Vocabulary as stored in GGUF metadata.

use std::collections::HashMap;

use vipera_core::gguf::{GgufFile, Value};
use vipera_errors::ViperaError;

/// Role a token plays, from `tokenizer.ggml.token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Ordinary vocabulary entry.
    Normal,
    /// Unknown token.
    Unknown,
    /// Control token such as `<s>`.
    Control,
    /// User-defined token, used by some chat formats.
    UserDefined,
    /// Entry reserved by the vocabulary but never emitted.
    Unused,
    /// Single-byte fallback token, spelled `<0xNN>`.
    Byte,
}

impl TokenType {
    fn from_code(code: i32) -> Self {
        match code {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

/// Segmentation scheme a vocabulary declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabModel {
    /// SentencePiece: pieces carry `▁` word markers.
    SentencePiece,
    /// GPT-2 byte-level BPE: text bytes are mapped to printable characters and merged by rank.
    ByteLevelBpe,
}

/// Vocabulary loaded from a model file.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<String>,
    scores: Option<Vec<f32>>,
    types: Option<Vec<TokenType>>,
    lookup: HashMap<String, u32>,
    unknown_id: Option<u32>,
    byte_tokens: HashMap<u8, u32>,
    longest_piece_chars: usize,
    has_word_markers: bool,
    model: Option<VocabModel>,
    pre: Option<String>,
    merges: Vec<(String, String)>,
}

impl Vocabulary {
    /// Read the vocabulary from `tokenizer.ggml.*` metadata.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ViperaError> {
        let tokens = string_array(file, "tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(ViperaError::unsupported(
                "model declares an empty tokenizer vocabulary",
            ));
        }

        let scores = number_array(file, "tokenizer.ggml.scores").ok();
        if let Some(scores) = &scores
            && scores.len() != tokens.len()
        {
            return Err(ViperaError::unsupported(format!(
                "tokenizer has {} tokens but {} scores",
                tokens.len(),
                scores.len()
            )));
        }

        let types = int_array(file, "tokenizer.ggml.token_type")
            .ok()
            .map(|codes| {
                codes
                    .into_iter()
                    .map(TokenType::from_code)
                    .collect::<Vec<_>>()
            });
        if let Some(types) = &types
            && types.len() != tokens.len()
        {
            return Err(ViperaError::unsupported(format!(
                "tokenizer has {} tokens but {} token types",
                tokens.len(),
                types.len()
            )));
        }

        let mut lookup = HashMap::with_capacity(tokens.len());
        let mut unknown_id = None;
        let mut byte_tokens = HashMap::new();
        let mut longest_piece_chars = 0;
        let mut has_word_markers = false;

        for (id, piece) in tokens.iter().enumerate() {
            let id = id as u32;
            lookup.entry(piece.clone()).or_insert(id);
            longest_piece_chars = longest_piece_chars.max(piece.chars().count());
            if matches!(
                types.as_ref().and_then(|t| t.get(id as usize)),
                Some(TokenType::Unknown)
            ) {
                unknown_id = Some(id);
            }
            if piece.contains('\u{2581}') {
                has_word_markers = true;
            }
            if let Some(byte) = parse_byte_token(piece) {
                byte_tokens.entry(byte).or_insert(id);
            }
        }
        if unknown_id.is_none() {
            unknown_id = lookup.get("<unk>").copied();
        }

        let model = match file
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
        {
            Some("gpt2") => Some(VocabModel::ByteLevelBpe),
            Some("llama") | Some("spm") => Some(VocabModel::SentencePiece),
            _ => None,
        };
        let pre = file
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let merges = file
            .metadata
            .get("tokenizer.ggml.merges")
            .map(parse_merges)
            .transpose()?
            .unwrap_or_default();

        Ok(Self {
            tokens,
            scores,
            types,
            lookup,
            unknown_id,
            byte_tokens,
            longest_piece_chars,
            has_word_markers,
            model,
            pre,
            merges,
        })
    }

    /// Segmentation scheme declared by the file, when it declares one.
    pub const fn model(&self) -> Option<VocabModel> {
        self.model
    }

    /// Number of merge rules, zero when the file carries none.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// The `tokenizer.ggml.pre` value, which selects the pre-tokenizer.
    pub fn pre(&self) -> Option<&str> {
        self.pre.as_deref()
    }

    /// Merge rules in priority order, as (left, right) piece pairs.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Number of pieces.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty. Never true for a loaded vocabulary.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Piece for an id.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Id for an exact piece.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.lookup.get(piece).copied()
    }

    /// Whether piece scores are available, which is what makes unigram segmentation possible.
    pub const fn has_scores(&self) -> bool {
        self.scores.is_some()
    }

    /// Score for a piece, or `None` when the vocabulary carries no scores.
    pub fn score(&self, id: u32) -> Option<f32> {
        self.scores.as_ref()?.get(id as usize).copied()
    }

    /// Type of a token.
    pub fn token_type(&self, id: u32) -> TokenType {
        self.types
            .as_ref()
            .and_then(|types| types.get(id as usize).copied())
            .unwrap_or(TokenType::Normal)
    }

    /// Whether the vocabulary uses the SentencePiece word marker, which means text must be
    /// normalised (spaces to `▁`, dummy prefix) before matching.
    pub const fn has_word_markers(&self) -> bool {
        self.has_word_markers
    }

    /// Longest piece length in characters, used to bound the match loop.
    pub const fn longest_piece_chars(&self) -> usize {
        self.longest_piece_chars
    }

    /// Unknown token id, when the vocabulary declares one.
    pub const fn unknown_id(&self) -> Option<u32> {
        self.unknown_id
    }

    /// Byte-fallback id for a raw byte, when the vocabulary declares one.
    pub fn byte_token(&self, byte: u8) -> Option<u32> {
        self.byte_tokens.get(&byte).copied()
    }
}

/// Parse `<0xNN>` byte-fallback pieces. Returns `None` for any other piece, including normal pieces
/// that merely start with `<`.
pub fn parse_byte_token(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn parse_merges(value: &Value) -> Result<Vec<(String, String)>, ViperaError> {
    let Value::Array(items) = value else {
        return Err(ViperaError::unsupported(format!(
            "tokenizer.ggml.merges is a {}, not an array",
            value.type_name()
        )));
    };
    let mut merges = Vec::with_capacity(items.len());
    for item in items {
        let Value::Str(rule) = item else {
            return Err(ViperaError::unsupported(format!(
                "tokenizer.ggml.merges contains a {} where a rule was expected",
                item.type_name()
            )));
        };
        let (left, right) = rule.split_once(' ').ok_or_else(|| {
            ViperaError::unsupported(format!("merge rule {rule:?} is not a pair of pieces"))
        })?;
        merges.push((left.to_owned(), right.to_owned()));
    }
    Ok(merges)
}

fn string_array(file: &GgufFile, key: &str) -> Result<Vec<String>, ViperaError> {
    match file.metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::Str(value) => Ok(value.clone()),
                other => Err(ViperaError::unsupported(format!(
                    "{key} contains a {} where a string was expected",
                    other.type_name()
                ))),
            })
            .collect(),
        Some(other) => Err(ViperaError::unsupported(format!(
            "{key} is a {}, not an array",
            other.type_name()
        ))),
        None => Err(ViperaError::unsupported(format!(
            "model does not declare {key}"
        ))),
    }
}

fn number_array(file: &GgufFile, key: &str) -> Result<Vec<f32>, ViperaError> {
    match file.metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::F32(value) => Ok(*value),
                Value::F64(value) => Ok(*value as f32),
                other => Err(ViperaError::unsupported(format!(
                    "{key} contains a {} where a score was expected",
                    other.type_name()
                ))),
            })
            .collect(),
        Some(other) => Err(ViperaError::unsupported(format!(
            "{key} is a {}, not an array",
            other.type_name()
        ))),
        None => Err(ViperaError::unsupported(format!(
            "model does not declare {key}"
        ))),
    }
}

fn int_array(file: &GgufFile, key: &str) -> Result<Vec<i32>, ViperaError> {
    match file.metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::I32(value) => Ok(*value),
                Value::U32(value) => Ok(*value as i32),
                Value::I8(value) => Ok(i32::from(*value)),
                Value::I16(value) => Ok(i32::from(*value)),
                Value::U8(value) => Ok(i32::from(*value)),
                Value::U16(value) => Ok(i32::from(*value)),
                other => Err(ViperaError::unsupported(format!(
                    "{key} contains a {} where a token type was expected",
                    other.type_name()
                ))),
            })
            .collect(),
        Some(other) => Err(ViperaError::unsupported(format!(
            "{key} is a {}, not an array",
            other.type_name()
        ))),
        None => Err(ViperaError::unsupported(format!(
            "model does not declare {key}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_byte_tokens_only_in_the_exact_shape() {
        assert_eq!(parse_byte_token("<0x0A>"), Some(0x0A));
        assert_eq!(parse_byte_token("<0xff>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0xGG>"), None);
        assert_eq!(parse_byte_token("<0xA>"), None);
        assert_eq!(parse_byte_token("<s>"), None);
        assert_eq!(parse_byte_token("▁hello"), None);
    }

    #[test]
    fn token_type_codes_map_to_roles() {
        assert_eq!(TokenType::from_code(1), TokenType::Normal);
        assert_eq!(TokenType::from_code(2), TokenType::Unknown);
        assert_eq!(TokenType::from_code(3), TokenType::Control);
        assert_eq!(TokenType::from_code(6), TokenType::Byte);
        assert_eq!(TokenType::from_code(99), TokenType::Normal);
    }
}
