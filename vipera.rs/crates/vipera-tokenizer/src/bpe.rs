//! Byte-level BPE, the scheme GPT-2-derived models (Qwen, Hermes, Llama 3) use.
//!
//! Two steps, both taken from the reference implementations:
//!
//! 1. **Pre-tokenization**: the raw text is cut into chunks that BPE may not merge across. The chunk
//!    boundaries come from the `tokenizer.ggml.pre` pattern; this build implements the Qwen 2 pattern
//!    and falls back to treating the whole text as one chunk for patterns it does not implement.
//! 2. **Merge by rank**: every byte is mapped to a printable character (space becomes `Ġ`), then the
//!    lowest-ranked adjacent pair from `tokenizer.ggml.merges` is merged until no rule applies.
//!
//! The mapping is also needed in reverse when decoding, which is why it lives here rather than in the
//! encoder.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use vipera_errors::ViperaError;

use crate::vocab::Vocabulary;

/// GPT-2's byte alphabet: printable bytes keep their value, the rest move into U+0100 and above.
struct ByteAlphabet {
    to_char: [char; 256],
    to_byte: HashMap<char, u8>,
}

fn alphabet() -> &'static ByteAlphabet {
    static TABLE: OnceLock<ByteAlphabet> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut to_char = ['\0'; 256];
        let mut to_byte = HashMap::with_capacity(256);
        let mut next = 256u32;
        for byte in 0u32..256 {
            let printable =
                (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
            let mapped = if printable {
                char::from_u32(byte).unwrap_or('\u{fffd}')
            } else {
                let ch = char::from_u32(next).unwrap_or('\u{fffd}');
                next += 1;
                ch
            };
            to_char[byte as usize] = mapped;
            to_byte.insert(mapped, byte as u8);
        }
        ByteAlphabet { to_char, to_byte }
    })
}

/// Pre-tokenizer pattern family, from `tokenizer.ggml.pre`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// Qwen 2/3: contractions, optional single prefix character before letters, one digit per chunk,
    /// punctuation runs, and whitespace rules that leave one space attached to the following word.
    Qwen2,
    /// No splitting: BPE runs over the whole text. Used for patterns this build does not implement.
    WholeText,
}

impl PreTokenizer {
    /// Pick the pre-tokenizer a model declares.
    pub fn from_pre(pre: Option<&str>) -> Self {
        match pre {
            Some("qwen2") | Some("qwen3") => Self::Qwen2,
            _ => Self::WholeText,
        }
    }

    /// Name used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::WholeText => "whole-text",
        }
    }

    fn split(self, text: &str) -> Vec<&str> {
        match self {
            Self::Qwen2 => split_qwen2(text),
            Self::WholeText => vec![text],
        }
    }
}

/// Separator for a merge key. A byte-level piece is a printable character sequence, so U+0001 never
/// appears inside one and the key cannot collide.
const MERGE_SEPARATOR: char = '\u{1}';

/// Byte-level encoder bound to one vocabulary.
#[derive(Debug, Clone)]
pub struct BpeEncoder {
    ranks: HashMap<String, u32>,
    pre: PreTokenizer,
}

impl BpeEncoder {
    /// Build the merge table from a vocabulary's metadata.
    pub fn from_vocab(vocab: &Vocabulary) -> Self {
        let mut ranks = HashMap::with_capacity(vocab.merge_count());
        for (rank, (left, right)) in vocab.merges().iter().enumerate() {
            ranks
                .entry(format!("{left}{MERGE_SEPARATOR}{right}"))
                .or_insert(rank as u32);
        }
        Self {
            ranks,
            pre: PreTokenizer::from_pre(vocab.pre()),
        }
    }

    /// Pre-tokenizer in use.
    pub const fn pre_tokenizer(&self) -> PreTokenizer {
        self.pre
    }

    /// Map a piece's characters back to raw bytes, which is how decoding reads a byte-level piece.
    pub fn byte_of(&self, ch: char) -> Option<u8> {
        alphabet().to_byte.get(&ch).copied()
    }

    /// Encode text, returning token ids.
    pub fn encode(&self, vocab: &Vocabulary, text: &str) -> Result<Vec<u32>, ViperaError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for chunk in self.pre.split(text) {
            if chunk.is_empty() {
                continue;
            }
            let mapped: Vec<char> = chunk
                .as_bytes()
                .iter()
                .map(|byte| alphabet().to_char[*byte as usize])
                .collect();
            for piece in self.merge(mapped) {
                let id = vocab.id(&piece).ok_or_else(|| {
                    ViperaError::unsupported(format!(
                        "byte-level piece {piece:?} is not in the vocabulary"
                    ))
                })?;
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// Apply merge rules to one chunk, lowest rank first, never crossing the chunk's characters.
    fn merge(&self, chars: Vec<char>) -> Vec<String> {
        let count = chars.len();
        if count == 0 {
            return Vec::new();
        }
        let mut parts: Vec<String> = chars.iter().map(char::to_string).collect();
        let mut alive = vec![true; count];
        let mut next: Vec<usize> = (1..count).chain(std::iter::once(usize::MAX)).collect();
        let mut prev: Vec<usize> = (0..count)
            .map(|index| if index == 0 { usize::MAX } else { index - 1 })
            .collect();

        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(count);
        for index in 0..count.saturating_sub(1) {
            if let Some(rank) = self.rank(&parts[index], &parts[index + 1]) {
                heap.push(Reverse((rank, index)));
            }
        }

        while let Some(Reverse((rank, index))) = heap.pop() {
            if !alive[index] || next[index] == usize::MAX {
                continue;
            }
            let right = next[index];
            if !alive[right] {
                continue;
            }
            let Some(current) = self.rank(&parts[index], &parts[right]) else {
                continue;
            };
            if current != rank {
                continue;
            }
            // Merge right into index, then stitch the links around it.
            let merged = format!("{}{}", parts[index], parts[right]);
            parts[index] = merged;
            alive[right] = false;
            let after = next[right];
            next[index] = after;
            if after != usize::MAX {
                prev[after] = index;
            }
            let before = prev[index];
            if before != usize::MAX
                && let Some(rank) = self.rank(&parts[before], &parts[index])
            {
                heap.push(Reverse((rank, before)));
            }
            if after != usize::MAX
                && let Some(rank) = self.rank(&parts[index], &parts[after])
            {
                heap.push(Reverse((rank, index)));
            }
        }

        let mut out = Vec::with_capacity(count);
        let mut cursor = 0;
        while cursor != usize::MAX {
            if alive[cursor] {
                out.push(parts[cursor].clone());
            }
            cursor = next[cursor];
        }
        out
    }

    fn rank(&self, left: &str, right: &str) -> Option<u32> {
        self.ranks
            .get(&format!("{left}{MERGE_SEPARATOR}{right}"))
            .copied()
    }
}

fn split_qwen2(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end_of = |index: usize| -> usize {
        if index < chars.len() {
            chars[index].0
        } else {
            text.len()
        }
    };

    let mut out = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let start = chars[index].0;
        let current = chars[index].1;

        if current == '\''
            && let Some(length) = contraction_len(&chars, index)
        {
            out.push(&text[start..end_of(index + length)]);
            index += length;
            continue;
        }

        // A single space glues onto a following punctuation run: ` (`, ` —`. This is the
        // ` ?[^\s\p{L}\p{N}]+` alternative of the reference pre-tokenizer.
        if current == ' '
            && index + 1 < chars.len()
            && !chars[index + 1].1.is_whitespace()
            && !chars[index + 1].1.is_alphanumeric()
        {
            let mut end = index + 2;
            while end < chars.len()
                && !chars[end].1.is_whitespace()
                && !chars[end].1.is_alphanumeric()
            {
                end += 1;
            }
            while end < chars.len() && matches!(chars[end].1, '\r' | '\n') {
                end += 1;
            }
            out.push(&text[start..end_of(end)]);
            index = end;
            continue;
        }

        // One non-letter, non-digit, non-newline character glues onto the word that follows it: ` x`,
        // `=y`, `(x`, `\ttabs`. This is the `[^\r\n\p{L}\p{N}]?\p{L}+` rule of the reference
        // pre-tokenizer, and it is also how the last space of a run attaches to the next word.
        if !matches!(current, '\r' | '\n')
            && !current.is_alphanumeric()
            && index + 1 < chars.len()
            && chars[index + 1].1.is_alphabetic()
        {
            let mut end = index + 2;
            while end < chars.len() && chars[end].1.is_alphabetic() {
                end += 1;
            }
            out.push(&text[start..end_of(end)]);
            index = end;
            continue;
        }

        if current.is_alphabetic() {
            let mut end = index + 1;
            while end < chars.len() && chars[end].1.is_alphabetic() {
                end += 1;
            }
            out.push(&text[start..end_of(end)]);
            index = end;
            continue;
        }

        if current.is_numeric() {
            out.push(&text[start..end_of(index + 1)]);
            index += 1;
            continue;
        }

        if !current.is_whitespace() {
            let mut end = index + 1;
            while end < chars.len()
                && !chars[end].1.is_whitespace()
                && !chars[end].1.is_alphanumeric()
            {
                end += 1;
            }
            while end < chars.len() && matches!(chars[end].1, '\r' | '\n') {
                end += 1;
            }
            out.push(&text[start..end_of(end)]);
            index = end;
            continue;
        }

        // Whitespace: a run containing a newline ends at its last newline; a run of spaces leaves one
        // space for the following word, matching the reference pre-tokenizer.
        let mut end = index;
        while end < chars.len() && chars[end].1.is_whitespace() {
            end += 1;
        }
        let last_newline = (index..end)
            .rev()
            .find(|position| matches!(chars[*position].1, '\r' | '\n'));
        if let Some(position) = last_newline {
            out.push(&text[start..end_of(position + 1)]);
            index = position + 1;
        } else if end < chars.len() {
            let cut = (end - 1).max(index + 1);
            out.push(&text[start..end_of(cut)]);
            index = cut;
        } else {
            out.push(&text[start..end_of(end)]);
            index = end;
        }
    }
    out
}

/// Length of an English contraction (`'s`, `'re`, ...) at `index`, if one starts there.
fn contraction_len(chars: &[(usize, char)], index: usize) -> Option<usize> {
    let rest: String = chars[index..]
        .iter()
        .take(3)
        .map(|(_, ch)| ch.to_ascii_lowercase())
        .collect();
    for suffix in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if rest.starts_with(suffix) {
            return Some(suffix.chars().count());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_alphabet_round_trips_and_keeps_printable_bytes() {
        let table = alphabet();
        for byte in 0u8..=255 {
            let ch = table.to_char[byte as usize];
            assert_eq!(table.to_byte.get(&ch).copied(), Some(byte));
        }
        assert_eq!(table.to_char[b' ' as usize], '\u{120}');
        assert_eq!(table.to_char[b'A' as usize], 'A');
    }

    #[test]
    fn qwen2_splits_leave_one_space_on_the_word() {
        assert_eq!(split_qwen2("a b"), vec!["a", " b"]);
        assert_eq!(split_qwen2("  two"), vec![" ", " two"]);
        assert_eq!(split_qwen2("    return"), vec!["   ", " return"]);
        assert_eq!(split_qwen2("hi!"), vec!["hi", "!"]);
        assert_eq!(split_qwen2("a, b"), vec!["a", ",", " b"]);
        assert_eq!(split_qwen2("123"), vec!["1", "2", "3"]);
        assert_eq!(split_qwen2("x\ny"), vec!["x", "\n", "y"]);
    }

    #[test]
    fn contractions_are_their_own_chunk() {
        assert_eq!(split_qwen2("I'm"), vec!["I", "'m"]);
        assert_eq!(split_qwen2("they'RE"), vec!["they", "'RE"]);
    }

    #[test]
    fn unknown_pre_patterns_fall_back_to_the_whole_text() {
        assert_eq!(
            PreTokenizer::from_pre(Some("llama3")),
            PreTokenizer::WholeText
        );
        assert_eq!(PreTokenizer::from_pre(Some("qwen2")), PreTokenizer::Qwen2);
        assert_eq!(PreTokenizer::from_pre(None), PreTokenizer::WholeText);
    }
}
