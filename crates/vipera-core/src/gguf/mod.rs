//! GGUF v2/v3 reader.
//!
//! Parses the header, metadata, and tensor table, then validates that every tensor fits inside the
//! file. Weights are never read here: this module is what makes `vipera model-info` possible without
//! loading a model.

pub mod metadata;
pub mod reader;
pub mod types;

use std::collections::BTreeMap;

use vipera_errors::ViperaError;
use vipera_traits::Fnv1a64;

pub use metadata::Value;
pub use reader::Reader;
pub use types::GgmlType;

/// The bytes every GGUF file starts with.
pub const MAGIC: &[u8; 4] = b"GGUF";

/// Highest tensor rank GGUF defines.
pub const MAX_DIMS: u32 = 4;

/// Refuse files that declare more tensors than this, before allocating for them.
pub const MAX_TENSORS: u64 = 1_000_000;

/// Refuse files that declare more metadata keys than this.
pub const MAX_METADATA: u64 = 1_000_000;

/// Default alignment between the tensor table and the data section.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// File header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Format version; Phase 1 supports 2 and 3.
    pub version: u32,
    /// Tensor entries declared by the header.
    pub tensor_count: u64,
    /// Metadata entries declared by the header.
    pub metadata_count: u64,
}

/// Typed metadata, keyed by the file's own keys.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: BTreeMap<String, Value>,
}

impl Metadata {
    /// Look up one key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Number of keys present.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no metadata was present.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate in key order, which is also the order used for identity hashing.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter()
    }

    /// Alignment between the tensor table and the data section.
    pub fn alignment(&self) -> Result<u64, ViperaError> {
        let alignment = self
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ViperaError::parse(
                0,
                format!("general.alignment is {alignment}, which is not a power of two"),
            ));
        }
        Ok(alignment)
    }
}

/// One tensor's location and shape, without its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name as stored in the file.
    pub name: String,
    /// Dimensions as stored: `dims[0]` varies fastest, so a matrix reads `[columns, rows]`.
    pub dims: Vec<u64>,
    /// Element type.
    pub ty: GgmlType,
    /// Offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of values in this tensor.
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
    }

    /// Bytes this tensor occupies, when its element count is a whole number of storage blocks.
    pub fn byte_len(&self) -> Option<u64> {
        self.ty.byte_len(self.element_count()?)
    }
}

/// A parsed model file.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// Header as declared by the file.
    pub header: Header,
    /// Metadata entries.
    pub metadata: Metadata,
    /// Tensor table, in file order.
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where tensor data begins.
    pub data_offset: u64,
    /// Size of the file that was parsed.
    pub file_len: u64,
}

impl GgufFile {
    /// Parse a complete GGUF file from memory.
    ///
    /// Reads the header, metadata, and tensor table; validates that every tensor lies inside `data`.
    pub fn parse(data: &[u8]) -> Result<Self, ViperaError> {
        let file_len = data.len() as u64;
        let mut reader = Reader::new(data);

        let mut magic = [0u8; 4];
        for byte in &mut magic {
            *byte = reader.u8()?;
        }
        if &magic != MAGIC {
            return Err(ViperaError::parse(
                0,
                format!("wrong magic bytes {magic:?}, expected \"GGUF\""),
            ));
        }

        let version = reader.u32()?;
        match version {
            2 | 3 => {}
            1 => {
                return Err(ViperaError::unsupported(
                    "GGUF v1 uses a different length encoding; re-convert the model to v2 or v3",
                ));
            }
            other => {
                return Err(ViperaError::unsupported(format!(
                    "GGUF version {other} is newer than this build supports (2 or 3)"
                )));
            }
        }

        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(ViperaError::parse(
                reader.offset(),
                format!("header declares {tensor_count} tensors, limit is {MAX_TENSORS}"),
            ));
        }
        if metadata_count > MAX_METADATA {
            return Err(ViperaError::parse(
                reader.offset(),
                format!("header declares {metadata_count} metadata keys, limit is {MAX_METADATA}"),
            ));
        }

        let mut entries = BTreeMap::new();
        for _ in 0..metadata_count {
            let key_offset = reader.offset();
            let key = reader.string()?;
            let type_id = reader.u32()?;
            let value = metadata::read_value(&mut reader, type_id, 0)?;
            if entries.insert(key.clone(), value).is_some() {
                return Err(ViperaError::parse(
                    key_offset,
                    format!("metadata key \"{key}\" appears more than once"),
                ));
            }
        }
        let metadata = Metadata { entries };
        let alignment = metadata.alignment()?;

        let capacity = usize::try_from(tensor_count.min(4096)).unwrap_or(0);
        let mut tensors = Vec::with_capacity(capacity);
        for _ in 0..tensor_count {
            let info_offset = reader.offset();
            let name = reader.string()?;
            let rank = reader.u32()?;
            if rank == 0 || rank > MAX_DIMS {
                return Err(ViperaError::parse(
                    info_offset,
                    format!("tensor \"{name}\" has rank {rank}; GGUF allows 1..={MAX_DIMS}"),
                ));
            }
            let mut dims = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                let dim = reader.u64()?;
                if dim == 0 {
                    return Err(ViperaError::parse(
                        info_offset,
                        format!("tensor \"{name}\" has a zero dimension"),
                    ));
                }
                dims.push(dim);
            }
            let ty = GgmlType::from_id(reader.u32()?)?;
            let offset = reader.u64()?;
            if tensors
                .iter()
                .any(|existing: &TensorInfo| existing.name == name)
            {
                return Err(ViperaError::parse(
                    info_offset,
                    format!("tensor \"{name}\" appears more than once"),
                ));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        let data_offset = align_up(reader.offset(), alignment)?;

        for tensor in &tensors {
            let byte_len = tensor.byte_len().ok_or_else(|| {
                ViperaError::parse(
                    data_offset,
                    format!(
                        "tensor \"{}\" has {} elements, not a whole number of {} blocks",
                        tensor.name,
                        tensor.element_count().unwrap_or(0),
                        tensor.ty.name()
                    ),
                )
            })?;
            let end = data_offset
                .checked_add(tensor.offset)
                .and_then(|start| start.checked_add(byte_len))
                .ok_or_else(|| {
                    ViperaError::parse(
                        data_offset,
                        format!("tensor \"{}\" offset overflows", tensor.name),
                    )
                })?;
            if end > file_len {
                return Err(ViperaError::parse(
                    data_offset,
                    format!(
                        "tensor \"{}\" ends at byte {end} but the file is {file_len} bytes",
                        tensor.name
                    ),
                ));
            }
        }

        let mut intervals: Vec<(u64, u64, &str)> = tensors
            .iter()
            .map(|tensor| {
                let byte_len = tensor.byte_len().unwrap_or(0);
                (
                    tensor.offset,
                    tensor.offset.saturating_add(byte_len),
                    tensor.name.as_str(),
                )
            })
            .collect();
        intervals.sort_by_key(|(start, _, _)| *start);
        for pair in intervals.windows(2) {
            let (_, prev_end, prev_name) = pair[0];
            let (next_start, _, next_name) = pair[1];
            if next_start < prev_end {
                return Err(ViperaError::parse(
                    data_offset,
                    format!("tensors \"{prev_name}\" and \"{next_name}\" overlap in the file"),
                ));
            }
        }

        Ok(Self {
            header: Header {
                version,
                tensor_count,
                metadata_count,
            },
            metadata,
            tensors,
            data_offset,
            file_len,
        })
    }

    /// Derive the public description used by `model-info` and by plan caching.
    pub fn describe(&self) -> ModelDescriptor {
        let architecture = self
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();

        let lookup_u64 = |keys: &[String]| -> Option<u64> {
            keys.iter()
                .find_map(|key| self.metadata.get(key).and_then(Value::as_u64))
        };

        let context_limit = lookup_u64(&[
            format!("{architecture}.context_length"),
            "general.context_length".to_owned(),
        ]);
        let layer_count = lookup_u64(&[format!("{architecture}.block_count")]);
        let vocab_size = self
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::array_len)
            .map(|n| n as u64);

        let mut tensor_types: BTreeMap<String, u64> = BTreeMap::new();
        for tensor in &self.tensors {
            *tensor_types.entry(tensor.ty.name().to_owned()).or_insert(0) += 1;
        }
        let quantization = tensor_types
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "unknown".to_owned());

        let parameter_count = self
            .tensors
            .iter()
            .filter_map(TensorInfo::element_count)
            .fold(0u64, |acc, count| acc.saturating_add(count));

        ModelDescriptor {
            name: architecture,
            identity: self.identity(),
            gguf_version: self.header.version,
            tensor_count: self.header.tensor_count,
            parameter_count,
            quantization,
            context_limit,
            layer_count,
            vocab_size,
            tensor_types,
        }
    }

    /// Stable identity for this model's structure.
    ///
    /// FNV-1a over the version, the metadata in key order, and the tensor table. This identifies
    /// content; it is not a cryptographic digest and must not be used for integrity checking.
    pub fn identity(&self) -> String {
        let mut hash = Fnv1a64::new();
        hash.write(b"vipera-gguf-identity-v1");
        hash.write(&self.header.version.to_le_bytes());
        for (key, value) in self.metadata.iter() {
            hash.write(key.as_bytes());
            hash.write(&[0]);
            hash_value(&mut hash, value);
        }
        for tensor in &self.tensors {
            hash.write(tensor.name.as_bytes());
            hash.write(&[0]);
            hash.write(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                hash.write(&dim.to_le_bytes());
            }
            hash.write(&tensor.ty.id().to_le_bytes());
            hash.write(&tensor.offset.to_le_bytes());
        }
        format!("fnv1a64:{:016x}", hash.finish())
    }
}

/// Public description of a model, as produced by `vipera model-info --json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Architecture family, e.g. `qwen2`.
    pub name: String,
    /// Stable structure hash, see [`GgufFile::identity`].
    pub identity: String,
    /// GGUF version from the header.
    pub gguf_version: u32,
    /// Tensor count from the header.
    pub tensor_count: u64,
    /// Total parameters derived from tensor shapes.
    pub parameter_count: u64,
    /// Dominant tensor type, e.g. `Q4_K`.
    pub quantization: String,
    /// Declared context limit, when the file declares one.
    pub context_limit: Option<u64>,
    /// Declared layer count, when the file declares one.
    pub layer_count: Option<u64>,
    /// Tokenizer vocabulary size, when the file carries one.
    pub vocab_size: Option<u64>,
    /// Tensor count per element type.
    pub tensor_types: BTreeMap<String, u64>,
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ViperaError> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or_else(|| ViperaError::internal("alignment overflow"))
}

fn hash_value(hash: &mut Fnv1a64, value: &Value) {
    match value {
        Value::Bool(v) => hash.write(&[0, u8::from(*v)]),
        Value::U8(v) => hash.write(&[1, *v]),
        Value::I8(v) => hash.write(&[2, *v as u8]),
        Value::U16(v) => {
            hash.write(&[3]);
            hash.write(&v.to_le_bytes());
        }
        Value::I16(v) => {
            hash.write(&[4]);
            hash.write(&v.to_le_bytes());
        }
        Value::U32(v) => {
            hash.write(&[5]);
            hash.write(&v.to_le_bytes());
        }
        Value::I32(v) => {
            hash.write(&[6]);
            hash.write(&v.to_le_bytes());
        }
        Value::U64(v) => {
            hash.write(&[7]);
            hash.write(&v.to_le_bytes());
        }
        Value::I64(v) => {
            hash.write(&[8]);
            hash.write(&v.to_le_bytes());
        }
        Value::F32(v) => {
            hash.write(&[9]);
            hash.write(&v.to_bits().to_le_bytes());
        }
        Value::F64(v) => {
            hash.write(&[10]);
            hash.write(&v.to_bits().to_le_bytes());
        }
        Value::Str(s) => {
            hash.write(&[11]);
            hash.write(&(s.len() as u64).to_le_bytes());
            hash.write(s.as_bytes());
        }
        Value::Array(items) => {
            hash.write(&[12]);
            hash.write(&(items.len() as u64).to_le_bytes());
            for item in items {
                hash_value(hash, item);
            }
        }
    }
}
