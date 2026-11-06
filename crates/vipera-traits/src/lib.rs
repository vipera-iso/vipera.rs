//! Contracts shared across crates.
//!
//! Slice 2 needs one contract: how the engine learns what hardware it is running on. Backend and
//! tensor traits arrive with the backends themselves, so this crate stays small on purpose.

#![warn(missing_docs)]

use std::fmt;

/// Kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceKind {
    /// NVIDIA GPU reachable through CUDA.
    Cuda,
    /// The host CPU.
    Cpu,
}

impl DeviceKind {
    /// Stable name used in JSON output and fingerprints.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One compute device, as far as planning is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// Backend this device is reached through.
    pub kind: DeviceKind,
    /// Human-readable device name, e.g. `NVIDIA GeForce RTX 3060`.
    pub name: String,
    /// Total memory in bytes.
    pub memory_total_bytes: u64,
    /// Memory free at probe time, in bytes. Never part of the fingerprint, since it changes.
    pub memory_free_bytes: u64,
    /// Driver version, when the platform reports one.
    pub driver_version: Option<String>,
}

/// The machine the engine is planning for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareProfile {
    /// Physical CPU cores.
    pub cpu_cores_physical: u32,
    /// Total host memory in bytes.
    pub ram_total_bytes: u64,
    /// Discrete devices. Empty on a CPU-only machine.
    pub devices: Vec<Device>,
}

impl HardwareProfile {
    /// First device of `kind`, if the machine has one.
    pub fn device(&self, kind: DeviceKind) -> Option<&Device> {
        self.devices.iter().find(|device| device.kind == kind)
    }

    /// Stable identity for this hardware configuration.
    ///
    /// Excludes free memory and driver versions: a fingerprint that changes between probes is not an
    /// identity, and plan caches keyed on it would never hit.
    pub fn fingerprint(&self) -> String {
        let mut hash = Fnv1a64::new();
        hash.write(b"vipera-hw-v1");
        hash.write(&self.cpu_cores_physical.to_le_bytes());
        hash.write(&self.ram_total_bytes.to_le_bytes());
        let mut devices: Vec<&Device> = self.devices.iter().collect();
        devices.sort_by(|a, b| {
            a.kind
                .cmp(&b.kind)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.memory_total_bytes.cmp(&b.memory_total_bytes))
        });
        for device in devices {
            hash.write(device.kind.as_str().as_bytes());
            hash.write(&[0]);
            hash.write(device.name.as_bytes());
            hash.write(&[0]);
            hash.write(&device.memory_total_bytes.to_le_bytes());
        }
        format!("hw_{:016x}", hash.finish())
    }
}

/// Source of hardware facts. The real implementation reads the OS; tests substitute a fixed profile.
pub trait HardwareProbe {
    /// Current profile of the machine.
    fn profile(&self) -> HardwareProfile;
}

/// FNV-1a, 64-bit: small, deterministic, dependency-free.
///
/// This identifies content. It is not a cryptographic digest.
pub struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// Start a new hash.
    pub const fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    /// Feed bytes.
    pub fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    /// Final value.
    pub const fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(free: u64) -> HardwareProfile {
        HardwareProfile {
            cpu_cores_physical: 8,
            ram_total_bytes: 16 * 1024 * 1024 * 1024,
            devices: vec![Device {
                kind: DeviceKind::Cuda,
                name: "NVIDIA GeForce RTX 3060".to_owned(),
                memory_total_bytes: 12 * 1024 * 1024 * 1024,
                memory_free_bytes: free,
                driver_version: Some("560.35".to_owned()),
            }],
        }
    }

    #[test]
    fn fingerprint_ignores_free_memory() {
        assert_eq!(machine(1).fingerprint(), machine(11_000).fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_capacity() {
        let mut bigger = machine(1);
        bigger.devices[0].memory_total_bytes *= 2;
        assert_ne!(machine(1).fingerprint(), bigger.fingerprint());
    }

    #[test]
    fn fingerprint_is_order_independent() {
        let mut reversed = machine(1);
        reversed.devices.reverse();
        let mut two = machine(1);
        two.devices.push(Device {
            kind: DeviceKind::Cpu,
            name: "host".to_owned(),
            memory_total_bytes: 1,
            memory_free_bytes: 1,
            driver_version: None,
        });
        reversed.devices.push(two.devices[1].clone());
        assert_eq!(two.fingerprint(), reversed.fingerprint());
    }

    #[test]
    fn cpu_only_machine_has_no_cuda_device() {
        let profile = HardwareProfile {
            cpu_cores_physical: 4,
            ram_total_bytes: 8,
            devices: Vec::new(),
        };
        assert!(profile.device(DeviceKind::Cuda).is_none());
        assert!(profile.fingerprint().starts_with("hw_"));
    }
}
