//! GGML element types that Phase 1 can size correctly.
//!
//! A type this build cannot size is rejected rather than guessed: an unknown block layout would make
//! the tensor bounds check meaningless.

use vipera_errors::ViperaError;

/// GGML tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// Brain float 16.
    Bf16,
    /// 4-bit, one scale per block of 32.
    Q4_0,
    /// 4-bit, scale and minimum per block of 32.
    Q4_1,
    /// 5-bit, one scale per block of 32.
    Q5_0,
    /// 5-bit, scale and minimum per block of 32.
    Q5_1,
    /// 8-bit with scale.
    Q8_0,
    /// 8-bit with scale and sum.
    Q8_1,
    /// 2-bit k-quant, block of 256.
    Q2K,
    /// 3-bit k-quant, block of 256.
    Q3K,
    /// 4-bit k-quant, block of 256.
    Q4K,
    /// 5-bit k-quant, block of 256.
    Q5K,
    /// 6-bit k-quant, block of 256.
    Q6K,
    /// 8-bit k-quant, block of 256.
    Q8K,
}

impl GgmlType {
    /// Map the on-disk type id to a known type.
    pub fn from_id(id: u32) -> Result<Self, ViperaError> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::Bf16,
            other => {
                return Err(ViperaError::unsupported(format!(
                    "ggml type id {other} is not implemented in this build"
                )));
            }
        })
    }

    /// On-disk type id.
    pub const fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Bf16 => 30,
        }
    }

    /// Name used in JSON output and in `tensor_types`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
        }
    }

    /// Elements per storage block.
    pub const fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per storage block.
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Bytes needed for `elements` values, or `None` when `elements` is not a whole number of blocks.
    pub const fn byte_len(self, elements: u64) -> Option<u64> {
        let block = self.block_size();
        if !elements.is_multiple_of(block) {
            return None;
        }
        Some(elements / block * self.block_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip() {
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q4_0,
            GgmlType::Q4K,
            GgmlType::Q8K,
        ] {
            assert_eq!(GgmlType::from_id(ty.id()).unwrap(), ty);
        }
    }

    #[test]
    fn unknown_ids_are_unsupported_not_guessed() {
        let err = GgmlType::from_id(99).unwrap_err();
        assert!(err.to_string().contains("99"));
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn sizes_match_the_block_layout() {
        assert_eq!(GgmlType::F32.byte_len(10), Some(40));
        assert_eq!(GgmlType::Q4K.byte_len(256), Some(144));
        assert_eq!(GgmlType::Q4K.byte_len(1000), None, "not a whole block");
    }
}
