//! GGUF metadata values.
//!
//! The type tag in the file decides how a value is decoded; there is no schema per key, so the value
//! type is carried in the enum rather than assumed.

use vipera_errors::ViperaError;

use super::reader::{MAX_ALLOCATION, Reader};

/// How deep arrays of arrays may nest before the file is rejected.
pub const MAX_ARRAY_DEPTH: u8 = 2;

/// One metadata value, tagged by its on-disk type.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `bool` (stored as one byte).
    Bool(bool),
    /// Unsigned 8-bit.
    U8(u8),
    /// Signed 8-bit.
    I8(i8),
    /// Unsigned 16-bit.
    U16(u16),
    /// Signed 16-bit.
    I16(i16),
    /// Unsigned 32-bit.
    U32(u32),
    /// Signed 32-bit.
    I32(i32),
    /// Unsigned 64-bit.
    U64(u64),
    /// Signed 64-bit.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// UTF-8 string.
    Str(String),
    /// Homogeneous array.
    Array(Vec<Value>),
}

impl Value {
    /// Interpret an integer-valued entry as `u64`. Floats and strings return `None`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Interpret a real-valued entry as `f32`. Integers are converted, strings return `None`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            other => other.as_u64().map(|v| v as f32),
        }
    }

    /// Borrow the string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Number of elements when this is an array.
    pub fn array_len(&self) -> Option<usize> {
        match self {
            Self::Array(items) => Some(items.len()),
            _ => None,
        }
    }

    /// Short type name, used when reporting a mismatched metadata type.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Str(_) => "string",
            Self::Array(_) => "array",
        }
    }
}

/// Read one value whose type tag has already been consumed by the caller.
pub fn read_value(reader: &mut Reader<'_>, type_id: u32, depth: u8) -> Result<Value, ViperaError> {
    Ok(match type_id {
        0 => Value::U8(reader.u8()?),
        1 => Value::I8(reader.i8()?),
        2 => Value::U16(reader.u16()?),
        3 => Value::I16(reader.i16()?),
        4 => Value::U32(reader.u32()?),
        5 => Value::I32(reader.i32()?),
        6 => Value::F32(reader.f32()?),
        7 => Value::Bool(reader.u8()? != 0),
        8 => Value::Str(reader.string()?),
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(ViperaError::parse(
                    reader.offset(),
                    "nested arrays are deeper than GGUF allows",
                ));
            }
            let start = reader.offset();
            let element_type = reader.u32()?;
            let count = reader.u64()?;
            if count > MAX_ALLOCATION {
                return Err(ViperaError::parse(
                    start,
                    format!("array of {count} elements exceeds the {MAX_ALLOCATION} limit"),
                ));
            }
            let mut items = Vec::with_capacity(count as usize);
            for _ in 0..count {
                items.push(read_value(reader, element_type, depth + 1)?);
            }
            Value::Array(items)
        }
        10 => Value::U64(reader.u64()?),
        11 => Value::I64(reader.i64()?),
        12 => Value::F64(reader.f64()?),
        other => {
            return Err(ViperaError::parse(
                reader.offset(),
                format!("unknown metadata value type {other}"),
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_string_value() {
        let mut bytes = 3u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"llm");
        let mut r = Reader::new(&bytes);
        assert_eq!(
            read_value(&mut r, 8, 0).unwrap(),
            Value::Str("llm".to_owned())
        );
    }

    #[test]
    fn reads_an_integer_array() {
        let mut bytes = 4u32.to_le_bytes().to_vec(); // element type u32
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        let mut r = Reader::new(&bytes);
        let v = read_value(&mut r, 9, 0).unwrap();
        assert_eq!(v.array_len(), Some(2));
        assert_eq!(v.as_u64(), None, "an array is not an integer");
    }

    #[test]
    fn rejects_unknown_value_types() {
        let mut r = Reader::new(&[]);
        let err = read_value(&mut r, 77, 0).unwrap_err();
        assert!(err.to_string().contains("77"));
    }

    #[test]
    fn coerces_non_negative_signed_integers() {
        assert_eq!(Value::I32(5).as_u64(), Some(5));
        assert_eq!(Value::I32(-1).as_u64(), None);
        assert_eq!(Value::F32(1.0).as_u64(), None);
    }
}
