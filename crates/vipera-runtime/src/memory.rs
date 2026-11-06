//! Measuring this process's own memory.
//!
//! `vipera verify` has to report what the runtime really used, not what it meant to use. On Linux
//! that is `VmHWM` (peak resident set) from `/proc/self/status`; the loader here allocates the model
//! data section for real, so the peak is a measurement of committed memory rather than an estimate.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use vipera_errors::ViperaError;

/// Read one `kB` field from the text of a `status` file.
pub fn parse_status_kib(status: &str, field: &str) -> Option<u64> {
    for line in status.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if name.trim() != field {
            continue;
        }
        let value = rest.split_whitespace().next()?;
        return value.parse::<u64>().ok().map(|kib| kib * 1024);
    }
    None
}

fn status_field(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_status_kib(&status, field)
}

/// Peak resident set size of this process, in bytes.
pub fn peak_resident_bytes() -> Option<u64> {
    status_field("VmHWM")
}

/// Current resident set size of this process, in bytes.
pub fn resident_bytes() -> Option<u64> {
    status_field("VmRSS")
}

/// A model's tensor data, held in host memory.
///
/// GGUF stores quantized tensors as-is, so "loading" is reading the data section: no decompression,
/// no reinterpretation. The buffer is the host half of a real load, which is what makes the measured
/// byte count comparable to the plan's `predicted_host_bytes` for a CPU placement.
#[derive(Debug)]
pub struct WeightBuffer {
    data: Vec<u8>,
}

impl WeightBuffer {
    /// Read `len` bytes starting at `offset` into a freshly allocated buffer.
    pub fn load(path: &Path, offset: u64, len: u64) -> Result<Self, ViperaError> {
        let len = usize::try_from(len).map_err(|_| {
            ViperaError::infeasible(format!(
                "the model's data section is {len} bytes, too large for this address space"
            ))
        })?;
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| ViperaError::invalid_argument("model data range overflows"))?;
        if end > file_len {
            return Err(ViperaError::parse(
                offset,
                format!(
                    "model data section ends at {end} but the file is {file_len} bytes; the file is truncated"
                ),
            ));
        }

        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(len).map_err(|_| {
            ViperaError::infeasible(format!(
                "cannot allocate {len} bytes for the model weights; free host memory first"
            ))
        })?;
        data.resize(len, 0);
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut data)?;
        Ok(Self { data })
    }

    /// Bytes held.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer holds nothing.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The bytes, for a loader that wants to inspect or copy them.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS: &str =
        "Name:\tvipera\nVmPeak:\t 2000000 kB\nVmHWM:\t   123456 kB\nVmRSS:\t    65536 kB\n";

    #[test]
    fn status_fields_are_read_in_bytes() {
        assert_eq!(parse_status_kib(STATUS, "VmHWM"), Some(123_456 * 1024));
        assert_eq!(parse_status_kib(STATUS, "VmRSS"), Some(65_536 * 1024));
        assert_eq!(parse_status_kib(STATUS, "VmSwap"), None);
    }

    #[test]
    fn a_malformed_status_line_is_not_guessed() {
        assert_eq!(parse_status_kib("VmHWM:\tnot-a-number kB\n", "VmHWM"), None);
        assert_eq!(parse_status_kib("", "VmHWM"), None);
    }

    #[test]
    fn weights_are_read_from_the_requested_offset() {
        let path = std::env::temp_dir().join("vipera-runtime-weights.bin");
        std::fs::write(&path, [0u8, 1, 2, 3, 4, 5, 6, 7]).unwrap();

        let weights = WeightBuffer::load(&path, 2, 4).unwrap();
        assert_eq!(weights.as_slice(), &[2, 3, 4, 5]);
        assert_eq!(weights.len(), 4);

        let truncated = WeightBuffer::load(&path, 6, 4).unwrap_err();
        assert!(truncated.to_string().contains("truncated"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn this_process_reports_its_own_memory() {
        // Present on Linux; the assertion is only that the numbers parse.
        if let Some(peak) = peak_resident_bytes() {
            assert!(peak > 0);
        }
    }
}
