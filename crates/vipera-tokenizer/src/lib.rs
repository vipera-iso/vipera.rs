//! Tokenization from GGUF metadata.
//!
//! Three segmentation strategies, chosen by what the file declares: byte-level BPE (merge table plus
//! pre-tokenizer) for `tokenizer.ggml.model = gpt2`, unigram (Viterbi over piece scores) when
//! `tokenizer.ggml.scores` is present, and greedy longest-match otherwise. Byte-fallback pieces
//! (`<0xNN>`) are used for characters the vocabulary does not cover, which is what keeps an unknown
//! emoji from failing a request.
//!
//! Chat template rendering lives in [`chat`]. It renders the families it recognises exactly and
//! refuses the rest: a template decides the prompt text a model was trained on, and an approximated
//! template is worse than a refusal.

#![warn(missing_docs)]

pub mod bpe;
pub mod chat;
pub mod vocab;

use vipera_core::gguf::GgufFile;
use vipera_errors::ViperaError;

use crate::bpe::BpeEncoder;

pub use bpe::PreTokenizer;
pub use chat::{ChatMessage, Role, TemplateFamily, detect_family, render, render_family};
pub use vocab::{TokenType, VocabModel, Vocabulary, parse_byte_token};

/// Score penalty applied when a character is emitted as the unknown token.
pub const UNKNOWN_PENALTY: f32 = -100.0;

/// SentencePiece marks a word boundary with U+2581 instead of a space.
const SPIECE_SPACE: char = '\u{2581}';

/// SentencePiece preprocessing: a dummy prefix, and every space becomes a word marker. Without
/// this, no piece that starts with `▁` can ever match raw input.
fn normalise_sentencepiece(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 3);
    out.push(SPIECE_SPACE);
    for ch in text.chars() {
        out.push(if ch == ' ' { SPIECE_SPACE } else { ch });
    }
    out
}

/// Tokenizer bound to one model's vocabulary.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vocabulary,
    bpe: Option<BpeEncoder>,
    /// Control and user-defined tokens, longest first, matched before segmentation. Chat templates
    /// render markers such as `<|im_start|>` that must stay single control tokens, not characters.
    specials: Vec<(String, u32)>,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Build a tokenizer from a parsed model file.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ViperaError> {
        let vocab = Vocabulary::from_gguf(file)?;
        let chat_template = file
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|value| value.as_str())
            .map(str::to_owned);
        let bpe = (vocab.model() == Some(VocabModel::ByteLevelBpe) && vocab.merge_count() > 0)
            .then(|| BpeEncoder::from_vocab(&vocab));
        // Control and user-defined tokens, plus bare `<...>` markers such as `</s>`: the reference
        // tokenizer treats any piece shaped like an angle-bracket marker as atomic, whatever its type.
        let mut specials: Vec<(String, u32)> = (0..vocab.len() as u32)
            .filter_map(|id| vocab.piece(id).map(|piece| (piece.to_owned(), id)))
            .filter(|(piece, id)| {
                !piece.is_empty()
                    && vocab.token_type(*id) != TokenType::Byte
                    && (matches!(
                        vocab.token_type(*id),
                        TokenType::Control | TokenType::UserDefined
                    ) || (piece.starts_with('<') && piece.ends_with('>') && piece.len() > 2))
            })
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
        Ok(Self {
            vocab,
            bpe,
            specials,
            chat_template,
        })
    }

    /// Number of control and user-defined tokens that are matched as single tokens.
    pub fn special_count(&self) -> usize {
        self.specials.len()
    }

    /// Id of an exact vocabulary piece, if the vocabulary has it. Used to find end-of-turn markers a
    /// chat template emits, which are never produced by encoding text.
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.vocab.id(piece)
    }

    /// Pre-tokenizer in use, when the vocabulary is byte-level BPE.
    pub fn pre_tokenizer(&self) -> Option<PreTokenizer> {
        self.bpe.as_ref().map(BpeEncoder::pre_tokenizer)
    }

    /// The vocabulary in use.
    pub const fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Raw chat template from the model, when it declares one.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Template family this build can render for the model, or `None`.
    pub fn template_family(&self) -> Option<TemplateFamily> {
        self.chat_template.as_deref().and_then(detect_family)
    }

    /// Render a conversation with the model's own template, then encode it.
    ///
    /// Fails when the model declares no template or declares one this build does not render, rather
    /// than guessing a format the model has never seen.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, ViperaError> {
        let template = self.chat_template.as_deref().ok_or_else(|| {
            ViperaError::unsupported("model does not declare tokenizer.chat_template")
        })?;
        self.encode(&chat::render(template, messages, add_generation_prompt)?)
    }

    /// Encode text into token ids.
    ///
    /// Control and user-defined tokens (chat markers, `<s>`, `[INST]`) are matched first and kept as
    /// single ids; everything between them is segmented normally.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ViperaError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if self.specials.is_empty() {
            return self.encode_plain(text);
        }

        let mut ids = Vec::new();
        let mut span_start = 0;
        let mut position = 0;
        while position < text.len() {
            if let Some((length, id)) = self.match_special(&text[position..]) {
                if span_start < position {
                    ids.extend(self.encode_plain(&text[span_start..position])?);
                }
                ids.push(id);
                position += length;
                span_start = position;
            } else {
                let ch = text[position..].chars().next().unwrap_or('\u{fffd}');
                position += ch.len_utf8();
            }
        }
        if span_start < text.len() {
            ids.extend(self.encode_plain(&text[span_start..])?);
        }
        Ok(ids)
    }

    fn match_special(&self, text: &str) -> Option<(usize, u32)> {
        self.specials
            .iter()
            .find(|(piece, _)| text.starts_with(piece.as_str()))
            .map(|(piece, id)| (piece.len(), *id))
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>, ViperaError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let normalised;
        let text = if self.vocab.has_word_markers() {
            normalised = normalise_sentencepiece(text);
            normalised.as_str()
        } else {
            text
        };
        if let Some(bpe) = &self.bpe {
            bpe.encode(&self.vocab, text)
        } else if self.vocab.has_scores() {
            self.encode_unigram(text)
        } else {
            self.encode_greedy(text)
        }
    }

    /// Decode token ids back to text.
    ///
    /// Control tokens are skipped, as they are not part of the model's visible output. Leading
    /// SentencePiece space is preserved; callers that want display text trim it.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ViperaError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self.vocab.piece(id).ok_or_else(|| {
                ViperaError::unsupported(format!("token id {id} is outside the vocabulary"))
            })?;
            if self.vocab.token_type(id) == TokenType::Control {
                continue;
            }
            if let Some(byte) = parse_byte_token(piece) {
                bytes.push(byte);
                continue;
            }
            for ch in piece.chars() {
                if let Some(byte) = self.bpe.as_ref().and_then(|bpe| bpe.byte_of(ch)) {
                    // Byte-level pieces store raw bytes behind printable characters.
                    bytes.push(byte);
                } else if ch == SPIECE_SPACE {
                    bytes.push(b' ');
                } else {
                    let mut buffer = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|e| ViperaError::unsupported(format!("decoded text is not valid UTF-8: {e}")))
    }

    fn char_offsets(text: &str) -> Vec<usize> {
        let mut offsets: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
        offsets.push(text.len());
        offsets
    }

    fn encode_greedy(&self, text: &str) -> Result<Vec<u32>, ViperaError> {
        let offsets = Self::char_offsets(text);
        let max_chars = self.vocab.longest_piece_chars().max(1);
        let mut ids = Vec::new();
        let mut position = 0;

        while position + 1 < offsets.len() {
            let mut matched = None;
            let last = (position + max_chars).min(offsets.len() - 1);
            for end in (position + 1..=last).rev() {
                let piece = &text[offsets[position]..offsets[end]];
                if let Some(id) = self.vocab.id(piece) {
                    matched = Some((end, id));
                    break;
                }
            }
            if let Some((end, id)) = matched {
                ids.push(id);
                position = end;
            } else {
                position = self.emit_byte_fallback(text, &offsets, position, &mut ids)?;
            }
        }
        Ok(ids)
    }

    fn encode_unigram(&self, text: &str) -> Result<Vec<u32>, ViperaError> {
        let offsets = Self::char_offsets(text);
        let count = offsets.len() - 1;
        let max_chars = self.vocab.longest_piece_chars().max(1);

        let mut best = vec![f32::NEG_INFINITY; count + 1];
        let mut from: Vec<Option<(usize, u32)>> = vec![None; count + 1];
        best[0] = 0.0;

        for start in 0..count {
            if best[start] == f32::NEG_INFINITY {
                continue;
            }
            let last = (start + max_chars).min(count);
            for end in start + 1..=last {
                let piece = &text[offsets[start]..offsets[end]];
                let Some(id) = self.vocab.id(piece) else {
                    continue;
                };
                let score = self.vocab.score(id).unwrap_or(0.0);
                let candidate = best[start] + score;
                // `>=` prefers the longer piece on a tie, which keeps segmentation stable.
                if candidate >= best[end] {
                    best[end] = candidate;
                    from[end] = Some((start, id));
                }
            }
            // Unknown single character, when the vocabulary has an unknown token.
            if let Some(unknown) = self.vocab.unknown_id() {
                let end = start + 1;
                let candidate = best[start] + UNKNOWN_PENALTY;
                if candidate >= best[end] {
                    best[end] = candidate;
                    from[end] = Some((start, unknown));
                }
            }
        }

        if best[count] == f32::NEG_INFINITY {
            return self.encode_greedy(text);
        }

        let mut ids = Vec::new();
        let mut cursor = count;
        while cursor > 0 {
            let (start, id) = from[cursor].ok_or_else(|| {
                ViperaError::internal(format!("unigram path broken at character {cursor}"))
            })?;
            ids.push(id);
            cursor = start;
        }
        ids.reverse();
        Ok(ids)
    }

    fn emit_byte_fallback(
        &self,
        text: &str,
        offsets: &[usize],
        position: usize,
        ids: &mut Vec<u32>,
    ) -> Result<usize, ViperaError> {
        let ch = text[offsets[position]..offsets[position + 1]]
            .chars()
            .next()
            .ok_or_else(|| ViperaError::internal("character boundary missing"))?;
        let mut buffer = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buffer).as_bytes();
        let mut fallback = Vec::with_capacity(encoded.len());
        for byte in encoded {
            let id = self.vocab.byte_token(*byte).ok_or_else(|| {
                ViperaError::unsupported(format!(
                    "character {ch:?} is not in the vocabulary and the model has no <0x{byte:02X}> byte token"
                ))
            })?;
            fallback.push(id);
        }
        ids.extend(fallback);
        Ok(position + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vipera_conformance::{GgufBuilder, MetaValue};
    use vipera_core::gguf::GgufFile;

    struct VocabSpec {
        tokens: Vec<String>,
        scores: Option<Vec<f32>>,
        types: Option<Vec<i32>>,
        merges: Option<Vec<String>>,
        model: Option<String>,
        pre: Option<String>,
    }

    fn build(spec: VocabSpec) -> Tokenizer {
        let mut builder = GgufBuilder::minimal();
        builder.metadata.push((
            "tokenizer.ggml.tokens".to_owned(),
            MetaValue::StrArray(spec.tokens),
        ));
        if let Some(scores) = spec.scores {
            builder.metadata.push((
                "tokenizer.ggml.scores".to_owned(),
                MetaValue::F32Array(scores),
            ));
        }
        if let Some(types) = spec.types {
            builder.metadata.push((
                "tokenizer.ggml.token_type".to_owned(),
                MetaValue::I32Array(types),
            ));
        }
        if let Some(merges) = spec.merges {
            builder.metadata.push((
                "tokenizer.ggml.merges".to_owned(),
                MetaValue::StrArray(merges),
            ));
        }
        if let Some(model) = spec.model {
            builder
                .metadata
                .push(("tokenizer.ggml.model".to_owned(), MetaValue::Str(model)));
        }
        if let Some(pre) = spec.pre {
            builder
                .metadata
                .push(("tokenizer.ggml.pre".to_owned(), MetaValue::Str(pre)));
        }
        let file = GgufFile::parse(&builder.build()).expect("fixture parses");
        Tokenizer::from_gguf(&file).expect("vocabulary loads")
    }

    fn greedy_ascii() -> Tokenizer {
        build(VocabSpec {
            tokens: ["a", "b", "ab", " "]
                .iter()
                .map(|p| (*p).to_owned())
                .collect(),
            scores: None,
            types: None,
            merges: None,
            model: None,
            pre: None,
        })
    }

    fn unigram_ascii() -> Tokenizer {
        build(VocabSpec {
            tokens: ["a", "b", "ab", " ", "<unk>"]
                .iter()
                .map(|p| (*p).to_owned())
                .collect(),
            scores: Some(vec![0.0, 0.0, 1.0, 0.0, -10.0]),
            types: Some(vec![1, 1, 1, 1, 2]),
            merges: None,
            model: None,
            pre: None,
        })
    }

    fn byte_bpe() -> Tokenizer {
        // Minimal byte-level BPE vocab: byte tokens for all bytes plus a few merges.
        let mut tokens: Vec<String> = Vec::new();
        let mut types: Vec<i32> = Vec::new();
        // Add byte tokens <0x00>..=<0xFF> as byte type (6)
        for b in 0u16..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        // Add some printable mapped tokens used by GPT-2 byte encoder.
        // The BPE encoder maps bytes to chars via its alphabet; those chars become pieces.
        // For a simple test, include single-char pieces for 'a','b',' ' in the byte alphabet.
        // We can instead rely on byte fallback tokens? BPE encoder looks up pieces in vocab by the
        // mapped string form; if not found, it errors. So we need the mapped pieces.
        // Easier: build from the actual alphabet by adding all single-char mapped pieces.
        let mut merges = Vec::new();
        // Use build helper by constructing via real vocab: easier to just make a BPE tokenizer
        // with tokens for the byte-alphabet chars. But we need to know the mapping.
        // Simpler approach: add tokens for common chars as they appear in GPT2 alphabet.
        // 'a' -> 'a', 'b' -> 'b', space -> Ġ (U+0120)
        tokens.push("a".to_owned());
        types.push(1);
        tokens.push("b".to_owned());
        types.push(1);
        tokens.push("\u{0120}".to_owned()); // space in GPT2 alphabet
        types.push(1);
        merges.push("a b".to_owned());
        tokens.push("ab".to_owned());
        types.push(1);

        build(VocabSpec {
            tokens,
            scores: None,
            types: Some(types),
            merges: Some(merges),
            model: Some("gpt2".to_owned()),
            pre: Some("qwen2".to_owned()),
        })
    }

    #[test]
    fn empty_input_encode_decode_greedy() {
        let tok = greedy_ascii();
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn empty_input_encode_decode_unigram() {
        let tok = unigram_ascii();
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn empty_input_encode_decode_bpe() {
        let tok = byte_bpe();
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn utf8_multibyte_greedy_with_byte_fallback() {
        // Greedy tokenizer with byte tokens should handle multi-byte UTF-8 via fallback.
        let mut tokens: Vec<String> = vec!["a".to_owned()];
        let mut types: Vec<i32> = vec![1];
        for b in 0u16..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        let tok = build(VocabSpec {
            tokens,
            scores: None,
            types: Some(types),
            merges: None,
            model: None,
            pre: None,
        });
        // 'é' is U+00E9, UTF-8 C3 A9 - 2 bytes
        let ids = tok.encode("aé").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "aé");
        // 3-byte char: '中' E4 B8 AD
        let ids = tok.encode("中").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "中");
        // 4-byte char: '😀' F0 9F 98 80
        let ids = tok.encode("😀").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "😀");
    }

    #[test]
    fn utf8_multibyte_unigram_with_byte_fallback() {
        // No unknown token: unigram falls back to greedy, which uses byte tokens.
        let mut tokens: Vec<String> = vec!["a".to_owned()];
        let mut types: Vec<i32> = vec![1];
        let mut scores: Vec<f32> = vec![0.0];
        for b in 0u16..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
            scores.push(-10.0);
        }
        let tok = build(VocabSpec {
            tokens,
            scores: Some(scores),
            types: Some(types),
            merges: None,
            model: None,
            pre: None,
        });
        for text in ["é", "aéz", "中", "中文", "😀", "a😀b", "é😀中"] {
            let ids = tok.encode(text).unwrap();
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(decoded, text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn utf8_multibyte_unigram_with_unknown_token_does_not_panic() {
        let tok = build(VocabSpec {
            tokens: vec!["a".to_owned(), "<unk>".to_owned()],
            scores: Some(vec![0.0, -1.0]),
            types: Some(vec![1, 2]),
            merges: None,
            model: None,
            pre: None,
        });
        for text in ["é", "中", "😀", "aé中😀b"] {
            let ids = tok.encode(text).unwrap();
            let decoded = tok.decode(&ids).unwrap();
            assert!(!decoded.is_empty() || text.is_empty());
            let _ = (ids, decoded);
        }
    }

    #[test]
    fn utf8_boundaries_with_special_tokens() {
        let tok = build(VocabSpec {
            tokens: ["a", "<|end|>", "é"]
                .iter()
                .map(|p| (*p).to_owned())
                .collect(),
            scores: None,
            types: Some(vec![1, 3, 1]),
            merges: None,
            model: None,
            pre: None,
        });
        let ids = tok.encode("a<|end|>é").unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(tok.decode(&ids).unwrap(), "aé");

        // Multi-byte UTF-8 adjacent to special
        let ids = tok.encode("é<|end|>a").unwrap();
        assert_eq!(ids, vec![2, 1, 0]);
    }

    #[test]
    fn utf8_only_multibyte_string_greedy() {
        // All chars are multi-byte, none in vocab except byte fallback
        let mut tokens: Vec<String> = Vec::new();
        let mut types: Vec<i32> = Vec::new();
        for b in 0u16..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        let tok = build(VocabSpec {
            tokens,
            scores: None,
            types: Some(types),
            merges: None,
            model: None,
            pre: None,
        });
        let text = "café naïve — 日本語";
        let ids = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    #[test]
    fn empty_and_whitespace_only_inputs() {
        let tok = greedy_ascii();
        assert!(tok.encode("").unwrap().is_empty());
        let ids = tok.encode(" ").unwrap();
        let _ = tok.decode(&ids).unwrap();
    }

    #[test]
    fn utf8_bpe_multibyte_does_not_panic() {
        let tok = byte_bpe();
        for text in ["é", "中", "😀", "aé", "café"] {
            let result = tok.encode(text);
            if let Ok(ids) = result {
                let _ = tok.decode(&ids);
            }
        }
    }

    #[test]
    fn utf8_multibyte_at_boundaries_with_word_markers() {
        let mut tokens: Vec<String> = vec!["▁".to_owned(), "▁a".to_owned()];
        let mut types: Vec<i32> = vec![1, 1];
        let mut scores: Vec<f32> = vec![-1.0, -1.0];
        for b in 0u16..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
            scores.push(-10.0);
        }
        let tok = build(VocabSpec {
            tokens,
            scores: Some(scores),
            types: Some(types),
            merges: None,
            model: Some("llama".to_owned()),
            pre: None,
        });
        for text in ["é", "a é", "é a", "中文 test", "😀!"] {
            let result = tok.encode(text);
            if let Ok(ids) = result {
                let _ = tok.decode(&ids);
            }
        }
    }

    #[test]
    fn empty_segments_around_special_tokens() {
        let tok = build(VocabSpec {
            tokens: ["<|a|>", "<|b|>"].iter().map(|p| (*p).to_owned()).collect(),
            scores: None,
            types: Some(vec![3, 3]),
            merges: None,
            model: None,
            pre: None,
        });
        assert_eq!(tok.encode("<|a|>").unwrap(), vec![0]);
        assert_eq!(tok.encode("<|a|><|b|>").unwrap(), vec![0, 1]);
        assert_eq!(tok.encode("<|b|><|a|><|b|>").unwrap(), vec![1, 0, 1]);
    }
}
