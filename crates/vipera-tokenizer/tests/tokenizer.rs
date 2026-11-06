//! Tokenizer behaviour against synthetic vocabularies built in code.

use vipera_conformance::{GgufBuilder, MetaValue};
use vipera_core::gguf::GgufFile;
use vipera_tokenizer::{TokenType, Tokenizer};

struct VocabSpec {
    tokens: Vec<String>,
    scores: Option<Vec<f32>>,
    types: Option<Vec<i32>>,
    template: Option<String>,
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
    if let Some(template) = spec.template {
        builder.metadata.push((
            "tokenizer.chat_template".to_owned(),
            MetaValue::Str(template),
        ));
    }
    let file = GgufFile::parse(&builder.build()).expect("fixture parses");
    Tokenizer::from_gguf(&file).expect("vocabulary loads")
}

fn sentencepiece() -> Tokenizer {
    build(VocabSpec {
        tokens: [
            "▁hello", "▁world", "hello", "world", " ", "▁", "<unk>", "<s>",
        ]
        .iter()
        .map(|piece| (*piece).to_owned())
        .collect(),
        scores: Some(vec![0.0, -1.0, -2.0, -3.0, -1.5, -0.5, 0.0, 0.0]),
        types: Some(vec![1, 1, 1, 1, 1, 1, 2, 3]),
        template: None,
    })
}

#[test]
fn unigram_segmentation_prefers_the_higher_scoring_path() {
    let tokenizer = sentencepiece();
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![0, 1]);
    assert_eq!(tokenizer.vocab().piece(0), Some("▁hello"));
}

#[test]
fn a_leading_sentencepiece_marker_decodes_to_a_space() {
    let tokenizer = sentencepiece();
    assert_eq!(tokenizer.decode(&[0, 1]).unwrap(), " hello world");
}

#[test]
fn control_tokens_are_not_part_of_the_visible_text() {
    let tokenizer = sentencepiece();
    assert_eq!(tokenizer.vocab().token_type(7), TokenType::Control);
    assert_eq!(tokenizer.decode(&[7, 0, 1]).unwrap(), " hello world");
}

#[test]
fn without_scores_the_longest_piece_wins() {
    let tokenizer = build(VocabSpec {
        tokens: ["hello", "world", " ", "he", "llo"]
            .iter()
            .map(|piece| (*piece).to_owned())
            .collect(),
        scores: None,
        types: None,
        template: None,
    });
    assert_eq!(tokenizer.encode("hello world").unwrap(), vec![0, 2, 1]);
}

#[test]
fn unknown_characters_fall_back_to_byte_tokens() {
    let tokenizer = build(VocabSpec {
        tokens: ["a", "<0xE2>", "<0x9C>", "<0xA8>"]
            .iter()
            .map(|piece| (*piece).to_owned())
            .collect(),
        scores: None,
        types: Some(vec![1, 6, 6, 6]),
        template: None,
    });
    let ids = tokenizer.encode("a✨").unwrap();
    assert_eq!(ids, vec![0, 1, 2, 3]);
    assert_eq!(tokenizer.decode(&ids).unwrap(), "a✨");
}

#[test]
fn a_character_outside_the_vocabulary_without_byte_tokens_is_an_error() {
    let tokenizer = build(VocabSpec {
        tokens: vec!["a".to_owned()],
        scores: None,
        types: None,
        template: None,
    });
    let error = tokenizer
        .encode("a😀")
        .expect_err("no byte tokens")
        .to_string();
    assert!(error.contains("byte token"), "unexpected error: {error}");
}

#[test]
fn unigram_falls_back_to_the_unknown_token() {
    let tokenizer = build(VocabSpec {
        tokens: ["ab", "<unk>"].iter().map(|p| (*p).to_owned()).collect(),
        scores: Some(vec![0.0, 0.0]),
        types: Some(vec![1, 2]),
        template: None,
    });
    assert_eq!(tokenizer.encode("zz").unwrap(), vec![1, 1]);
}

#[test]
fn an_id_outside_the_vocabulary_is_an_error() {
    let tokenizer = sentencepiece();
    let error = tokenizer
        .decode(&[9_999])
        .expect_err("id is out of range")
        .to_string();
    assert!(
        error.contains("outside the vocabulary"),
        "unexpected: {error}"
    );
}

#[test]
fn an_empty_vocabulary_is_refused() {
    let mut builder = GgufBuilder::minimal();
    builder.metadata.push((
        "tokenizer.ggml.tokens".to_owned(),
        MetaValue::StrArray(Vec::new()),
    ));
    let file = GgufFile::parse(&builder.build()).expect("fixture parses");
    assert!(Tokenizer::from_gguf(&file).is_err());
}

#[test]
fn a_model_without_tokens_is_refused() {
    let file = GgufFile::parse(&GgufBuilder::minimal().build()).expect("fixture parses");
    let error = Tokenizer::from_gguf(&file)
        .expect_err("no vocabulary")
        .to_string();
    assert!(
        error.contains("tokenizer.ggml.tokens"),
        "unexpected: {error}"
    );
}

#[test]
fn scores_must_match_the_vocabulary_length() {
    let mut builder = GgufBuilder::minimal();
    builder.metadata.push((
        "tokenizer.ggml.tokens".to_owned(),
        MetaValue::StrArray(vec!["a".to_owned(), "b".to_owned()]),
    ));
    builder.metadata.push((
        "tokenizer.ggml.scores".to_owned(),
        MetaValue::F32Array(vec![0.0]),
    ));
    let file = GgufFile::parse(&builder.build()).expect("fixture parses");
    let error = Tokenizer::from_gguf(&file)
        .expect_err("mismatch")
        .to_string();
    assert!(error.contains("scores"), "unexpected: {error}");
}

#[test]
fn the_chat_template_is_exposed_but_not_rendered() {
    let template = "{% for m in messages %}{{ m['role'] }}{% endfor %}";
    let tokenizer = build(VocabSpec {
        tokens: vec!["▁hi".to_owned()],
        scores: Some(vec![0.0]),
        types: None,
        template: Some(template.to_owned()),
    });
    assert_eq!(tokenizer.chat_template(), Some(template));
}

#[test]
fn empty_text_encodes_to_no_tokens() {
    let tokenizer = sentencepiece();
    assert!(tokenizer.encode("").unwrap().is_empty());
    assert_eq!(tokenizer.decode(&[]).unwrap(), "");
}
