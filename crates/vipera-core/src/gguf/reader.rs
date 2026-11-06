//! Bounded little-endian reader.
//!
//! Every read is checked against the remaining bytes, so a truncated or hostile file produces a parse
//! error with an offset instead of an out-of-bounds read or a large allocation.

use vipera_errors::ViperaError;

/// Largest string or array this reader will accept, as a guard against a length field that lies.
pub const MAX_ALLOCATION: u64 = 1 << 20; // 1 MiB

/// Cursor over a model file's bytes.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Start reading at the beginning of `data`.
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset, as reported in [`ViperaError::Parse`].
    pub const fn offset(&self) -> u64 {
        self.pos as u64
    }

    /// Bytes left to read.
    pub const fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ViperaError> {
        let remaining = self.remaining();
        if n > remaining {
            return Err(ViperaError::parse(
                self.offset(),
                format!("unexpected end of file: needed {n} bytes, {remaining} left"),
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read one byte.
    pub fn u8(&mut self) -> Result<u8, ViperaError> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16, ViperaError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32, ViperaError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64, ViperaError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a little-endian `i8`.
    pub fn i8(&mut self) -> Result<i8, ViperaError> {
        Ok(self.u8()? as i8)
    }

    /// Read a little-endian `i16`.
    pub fn i16(&mut self) -> Result<i16, ViperaError> {
        Ok(self.u16()? as i16)
    }

    /// Read a little-endian `i32`.
    pub fn i32(&mut self) -> Result<i32, ViperaError> {
        Ok(self.u32()? as i32)
    }

    /// Read a little-endian `i64`.
    pub fn i64(&mut self) -> Result<i64, ViperaError> {
        Ok(self.u64()? as i64)
    }

    /// Read a little-endian `f32`.
    pub fn f32(&mut self) -> Result<f32, ViperaError> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// Read a little-endian `f64`.
    pub fn f64(&mut self) -> Result<f64, ViperaError> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Read a GGUF string: `u64` byte length followed by UTF-8 bytes.
    pub fn string(&mut self) -> Result<String, ViperaError> {
        let start = self.offset();
        let len = self.u64()?;
        if len > MAX_ALLOCATION {
            return Err(ViperaError::parse(
                start,
                format!("string length {len} exceeds the {MAX_ALLOCATION} byte limit"),
            ));
        }
        let len = len as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|e| ViperaError::parse(start, format!("string is not valid UTF-8: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_integers() {
        let mut r = Reader::new(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        assert_eq!(r.u32().unwrap(), 0x0403_0201);
        assert_eq!(r.u32().unwrap(), 0x0807_0605);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn refuses_to_read_past_the_end() {
        let mut r = Reader::new(&[0x01]);
        let err = r.u32().unwrap_err();
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("unexpected end of file"));
    }

    #[test]
    fn refuses_a_lying_string_length() {
        // Length says 1 MiB + 1 but only one byte follows.
        let mut bytes = (MAX_ALLOCATION + 1).to_le_bytes().to_vec();
        bytes.push(b'x');
        let err = Reader::new(&bytes).string().unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }

    #[test]
    fn rejects_invalid_utf8() {
        let mut bytes = 1u64.to_le_bytes().to_vec();
        bytes.push(0xFF);
        let err = Reader::new(&bytes).string().unwrap_err();
        assert!(err.to_string().contains("UTF-8"));
    }
}
