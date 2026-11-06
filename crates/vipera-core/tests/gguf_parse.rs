//! GGUF parsing contract tests.
//!
//! Fixtures are built in memory, never committed, so they cannot rot and every byte is visible in the
//! test that depends on it.

use vipera_core::gguf::{GgmlType, GgufFile};

/// One metadata entry, already encoded as type tag + value bytes.
struct Entry {
    key: &'static str,
    encoded: Vec<u8>,
}

impl Entry {
    fn u32(key: &'static str, value: u32) -> Self {
        let mut encoded = 4u32.to_le_bytes().to_vec();
        encoded.extend_from_slice(&value.to_le_bytes());
        Self { key, encoded }
    }

    fn string(key: &'static str, value: &str) -> Self {
        let mut encoded = 8u32.to_le_bytes().to_vec();
        encoded.extend_from_slice(&encode_string(value));
        Self { key, encoded }
    }

    fn string_array(key: &'static str, values: &[&str]) -> Self {
        let mut encoded = 9u32.to_le_bytes().to_vec();
        encoded.extend_from_slice(&8u32.to_le_bytes()); // element type
        encoded.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            encoded.extend_from_slice(&encode_string(value));
        }
        Self { key, encoded }
    }
}

struct Tensor {
    name: &'static str,
    dims: Vec<u64>,
    type_id: u32,
    offset: u64,
}

impl Tensor {
    fn f32(name: &'static str, dims: &[u64]) -> Self {
        Self {
            name,
            dims: dims.to_vec(),
            type_id: GgmlType::F32.id(),
            offset: 0,
        }
    }
}

fn encode_string(value: &str) -> Vec<u8> {
    let mut out = (value.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(value.as_bytes());
    out
}

struct Fixture {
    version: u32,
    metadata: Vec<Entry>,
    tensors: Vec<Tensor>,
    truncate_to: Option<usize>,
    corrupt_magic: bool,
    data_len: usize,
}

impl Fixture {
    fn minimal() -> Self {
        Self {
            version: 3,
            metadata: vec![
                Entry::string("general.architecture", "llama"),
                Entry::u32("llama.context_length", 4096),
                Entry::u32("llama.block_count", 32),
                Entry::string_array("tokenizer.ggml.tokens", &["<unk>", "<s>", "</s>"]),
            ],
            tensors: vec![Tensor::f32("token_embd.weight", &[4])],
            truncate_to: None,
            corrupt_magic: false,
            data_len: 16, // 4 f32 values
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(if self.corrupt_magic { b"GGUX" } else { b"GGUF" });
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for entry in &self.metadata {
            out.extend_from_slice(&encode_string(entry.key));
            out.extend_from_slice(&entry.encoded);
        }

        for tensor in &self.tensors {
            out.extend_from_slice(&encode_string(tensor.name));
            out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.type_id.to_le_bytes());
            out.extend_from_slice(&tensor.offset.to_le_bytes());
        }

        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend(std::iter::repeat_n(0u8, self.data_len));

        if let Some(len) = self.truncate_to {
            out.truncate(len);
        }
        out
    }
}

#[test]
fn parses_header_metadata_and_tensors() {
    let bytes = Fixture::minimal().build();
    let file = GgufFile::parse(&bytes).expect("fixture is valid");

    assert_eq!(file.header.version, 3);
    assert_eq!(file.header.tensor_count, 1);
    assert_eq!(file.header.metadata_count, 4);
    assert_eq!(file.tensors.len(), 1);
    assert_eq!(file.tensors[0].name, "token_embd.weight");
    assert_eq!(file.tensors[0].dims, vec![4]);
    assert_eq!(file.tensors[0].ty, GgmlType::F32);
    assert_eq!(file.tensors[0].element_count(), Some(4));
    assert_eq!(file.tensors[0].byte_len(), Some(16));
    assert_eq!(file.data_offset % 32, 0);
}

#[test]
fn describes_the_model_for_model_info() {
    let bytes = Fixture::minimal().build();
    let described = GgufFile::parse(&bytes).unwrap().describe();

    assert_eq!(described.name, "llama");
    assert_eq!(described.gguf_version, 3);
    assert_eq!(described.tensor_count, 1);
    assert_eq!(described.parameter_count, 4);
    assert_eq!(described.quantization, "F32");
    assert_eq!(described.context_limit, Some(4096));
    assert_eq!(described.layer_count, Some(32));
    assert_eq!(described.vocab_size, Some(3));
    assert_eq!(described.tensor_types.get("F32"), Some(&1));
    assert!(described.identity.starts_with("fnv1a64:"));
}

#[test]
fn identity_is_stable_and_content_addressed() {
    let first = GgufFile::parse(&Fixture::minimal().build()).unwrap();
    let second = GgufFile::parse(&Fixture::minimal().build()).unwrap();
    assert_eq!(first.identity(), second.identity());

    let mut changed = Fixture::minimal();
    changed.metadata[1] = Entry::u32("llama.context_length", 8192);
    let third = GgufFile::parse(&changed.build()).unwrap();
    assert_ne!(first.identity(), third.identity());
}

#[test]
fn absent_optional_metadata_is_omitted_not_guessed() {
    let mut fixture = Fixture::minimal();
    fixture.metadata = vec![Entry::string("general.architecture", "llama")];
    fixture.tensors = vec![Tensor::f32("blk.0.weight", &[2])];
    fixture.data_len = 8;

    let described = GgufFile::parse(&fixture.build()).unwrap().describe();
    assert_eq!(described.context_limit, None);
    assert_eq!(described.layer_count, None);
    assert_eq!(described.vocab_size, None);
}

#[test]
fn rejects_wrong_magic() {
    let mut fixture = Fixture::minimal();
    fixture.corrupt_magic = true;
    let err = GgufFile::parse(&fixture.build()).unwrap_err();
    assert!(err.to_string().contains("magic"), "{err}");
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn rejects_v1_and_future_versions_by_name() {
    for (version, needle) in [(1u32, "v1"), (4u32, "version 4")] {
        let mut fixture = Fixture::minimal();
        fixture.version = version;
        let err = GgufFile::parse(&fixture.build()).unwrap_err();
        assert!(
            err.to_string().contains(needle),
            "version {version} error was: {err}"
        );
        assert_eq!(err.exit_code(), 2);
    }
}

#[test]
fn rejects_a_truncated_file() {
    let mut fixture = Fixture::minimal();
    fixture.truncate_to = Some(20);
    let err = GgufFile::parse(&fixture.build()).unwrap_err();
    assert!(err.to_string().contains("unexpected end of file"), "{err}");
}

#[test]
fn rejects_a_tensor_that_overruns_the_file() {
    let mut fixture = Fixture::minimal();
    fixture.tensors = vec![Tensor {
        name: "too_big.weight",
        dims: vec![1024],
        type_id: GgmlType::F32.id(),
        offset: 0,
    }];
    let err = GgufFile::parse(&fixture.build()).unwrap_err();
    assert!(err.to_string().contains("ends at byte"), "{err}");
}

#[test]
fn rejects_an_unimplemented_element_type() {
    let mut fixture = Fixture::minimal();
    fixture.tensors = vec![Tensor {
        name: "mystery.weight",
        dims: vec![32],
        type_id: 99,
        offset: 0,
    }];
    let err = GgufFile::parse(&fixture.build()).unwrap_err();
    assert!(err.to_string().contains("99"), "{err}");
}

#[test]
fn rejects_rank_zero_and_rank_above_four() {
    for dims in [vec![], vec![1, 2, 3, 4, 5]] {
        let mut fixture = Fixture::minimal();
        fixture.tensors = vec![Tensor {
            name: "bad_rank.weight",
            dims,
            type_id: GgmlType::F32.id(),
            offset: 0,
        }];
        let err = GgufFile::parse(&fixture.build()).unwrap_err();
        assert!(err.to_string().contains("rank"), "{err}");
    }
}

#[test]
fn rejects_a_duplicate_metadata_key() {
    let mut fixture = Fixture::minimal();
    fixture
        .metadata
        .push(Entry::string("general.architecture", "qwen2"));
    let err = GgufFile::parse(&fixture.build()).unwrap_err();
    assert!(err.to_string().contains("more than once"), "{err}");
}
