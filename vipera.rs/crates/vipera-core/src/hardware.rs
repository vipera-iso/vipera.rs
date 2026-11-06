//! Hardware detection.
//!
//! Reading the OS is separated from parsing, so every rule here is testable on a machine with no GPU.
//! Detection never fails: a missing file means "unknown", and the caller decides what that permits.

use std::path::{Path, PathBuf};
use std::process::Command;

use vipera_traits::{Device, DeviceKind, HardwareProbe, HardwareProfile};

/// Probe that reads `/proc`, `/sys`, and `nvidia-smi` when present.
pub struct SystemProbe {
    proc_root: PathBuf,
    smi: Option<&'static str>,
}

impl SystemProbe {
    /// Probe the machine this process is running on.
    pub fn new() -> Self {
        Self {
            proc_root: PathBuf::from("/proc"),
            smi: Some("nvidia-smi"),
        }
    }

    fn read(&self, path: &Path) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    fn cuda_devices(&self) -> Vec<Device> {
        let Some(smi) = self.smi else {
            return Vec::new();
        };
        let output = Command::new(smi)
            .args([
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ])
            .output();
        match output {
            Ok(out) if out.status.success() => {
                parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
            }
            _ => Vec::new(),
        }
    }
}

impl Default for SystemProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareProbe for SystemProbe {
    fn profile(&self) -> HardwareProfile {
        let meminfo = self.read(&self.proc_root.join("meminfo"));
        let cpuinfo = self.read(&self.proc_root.join("cpuinfo"));

        let ram_total_bytes = meminfo
            .as_deref()
            .and_then(parse_meminfo_total)
            .unwrap_or(0);
        let cpu_cores_physical = cpuinfo
            .as_deref()
            .and_then(parse_physical_cores)
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(|n| n.get() as u32)
            })
            .unwrap_or(1);

        HardwareProfile {
            cpu_cores_physical,
            ram_total_bytes,
            devices: self.cuda_devices(),
        }
    }
}

/// `MemTotal:` from `/proc/meminfo`, in bytes.
pub fn parse_meminfo_total(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let field = line.split_whitespace().nth(1)?;
    let kb: u64 = field.parse().ok()?;
    Some(kb.saturating_mul(1024))
}

/// Physical core count from `/proc/cpuinfo`.
///
/// Counts distinct `(physical id, core id)` pairs; falls back to counting `processor` entries when the
/// kernel does not expose topology (common in containers).
pub fn parse_physical_cores(text: &str) -> Option<u32> {
    let mut pairs = std::collections::BTreeSet::new();
    let mut processors = 0u32;
    let mut physical_id: Option<u32> = None;
    let mut core_id: Option<u32> = None;

    for line in text.lines().chain(std::iter::once("")) {
        let (key, value) = match line.split_once(':') {
            Some((key, value)) => (key.trim(), value.trim()),
            None => ("", ""),
        };
        match key {
            "physical id" => physical_id = value.parse().ok(),
            "core id" => core_id = value.parse().ok(),
            "processor" => {
                processors += 1;
                if let (Some(package), Some(core)) = (physical_id, core_id) {
                    pairs.insert((package, core));
                }
                physical_id = None;
                core_id = None;
            }
            _ => {}
        }
    }

    if !pairs.is_empty() {
        return Some(pairs.len() as u32);
    }
    if processors > 0 {
        return Some(processors);
    }
    None
}

/// Parse `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap`.
///
/// Memory is reported in MiB. A malformed row is skipped rather than treated as a device with zero
/// memory, because planning on a zero-capacity device would refuse every model for the wrong reason.
pub fn parse_nvidia_smi(text: &str) -> Vec<Device> {
    let mut devices = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 4 {
            continue;
        }
        let (Ok(total_mib), Ok(free_mib)) = (fields[1].parse::<u64>(), fields[2].parse::<u64>())
        else {
            continue;
        };
        if fields[0].is_empty() || total_mib == 0 {
            continue;
        }
        devices.push(Device {
            kind: DeviceKind::Cuda,
            name: fields[0].to_owned(),
            memory_total_bytes: total_mib.saturating_mul(1024 * 1024),
            memory_free_bytes: free_mib.saturating_mul(1024 * 1024),
            driver_version: (!fields[3].is_empty()).then(|| fields[3].to_owned()),
        });
    }
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_total_memory_in_bytes() {
        let text = "MemTotal:       16316384 kB\nMemFree:         1234567 kB\n";
        assert_eq!(parse_meminfo_total(text), Some(16_316_384 * 1024));
        assert_eq!(parse_meminfo_total("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn counts_physical_cores_not_hyperthreads() {
        let text = "\
processor\t: 0
physical id\t: 0
core id\t\t: 0

processor\t: 1
physical id\t: 0
core id\t\t: 0
";
        // Two logical CPUs, one physical core.
        assert_eq!(parse_physical_cores(text), Some(1));
    }

    #[test]
    fn falls_back_to_logical_count_without_topology() {
        let text = "processor\t: 0\n\nprocessor\t: 1\n\nprocessor\t: 2\n";
        assert_eq!(parse_physical_cores(text), Some(3));
    }

    #[test]
    fn parses_smi_rows_into_devices() {
        let text = "\
NVIDIA GeForce RTX 3060, 12288, 11000, 560.35.03, 8.6
";
        let devices = parse_nvidia_smi(text);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].memory_total_bytes, 12288 * 1024 * 1024);
        assert_eq!(devices[0].driver_version.as_deref(), Some("560.35.03"));
    }

    #[test]
    fn skips_malformed_or_zero_capacity_rows() {
        let text = "\
broken row
, 12288, 11000, 560.35.03, 8.6
GPU, unknown, 11000, 560.35.03, 8.6
GPU, 0, 0, 560.35.03, 8.6
";
        assert!(parse_nvidia_smi(text).is_empty());
    }
}
