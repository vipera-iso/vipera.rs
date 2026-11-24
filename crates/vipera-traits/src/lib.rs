//! Contracts shared across crates.
//!
//! Three groups so far: how the engine learns what hardware it is running on, what a sampling
//! strategy has to look like so a build can list the ones it has, and the four algorithm slots a
//! build fills at startup — KV compression, sparse attention, speculative decoding and MoE
//! offloading. Backend and tensor traits arrive with the backends themselves, so this crate stays
//! small on purpose. Every trait here is object safe, and the implementations live in their own
//! crates, which `vipera-core` never depends on.

#![warn(missing_docs)]

pub mod sampler;

use std::fmt;

pub use sampler::{LogitPenalty, NoPenalty, Random, Registry, SampleError, SampleRequest, Sampler};

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

/// A strategy that compresses retained KV cache content before a compute pass.
///
/// The runtime already has a `compress` stage and reports 0 µs in it; a build that registers an
/// algorithm here is what fills it. Implementations own their math and their buffers, take shared
/// references only, and are `Send + Sync` like a [`Sampler`] because the engine holds one per
/// process rather than one per turn.
pub trait KvCompression: Send + Sync {
    /// The name the runtime reports this algorithm under.
    fn name(&self) -> &'static str;

    /// Compress one block of cache values into `out`, returning the bytes appended.
    ///
    /// `0` means the block was left as it was: an algorithm that only wins on part of its input
    /// says so per block instead of failing the pass, and the caller stores the raw values instead.
    fn compress(&self, block: &[f32], out: &mut Vec<u8>) -> usize;

    /// Restore a block written by [`KvCompression::compress`] into `out`.
    ///
    /// Returns the number of values written, which is short of `out.len()` when the block was
    /// stored uncompressed. The trait fixes no byte layout: only the algorithm that wrote a block
    /// reads it back.
    fn decompress(&self, compressed: &[u8], out: &mut [f32]) -> usize;
}

/// A policy that chooses which earlier positions a query is computed against.
///
/// The `sparse` stage uses it to shrink the attention batch; returning fewer positions is the whole
/// point, and returning all of them is the valid policy the runtime runs today.
pub trait SparseAttention: Send + Sync {
    /// The name the runtime reports this policy under.
    fn name(&self) -> &'static str;

    /// Positions `query` attends to, appended to `out` in ascending order.
    ///
    /// `query` is the position being computed and `context` the number of positions retained, so a
    /// policy can depend on how far into the sequence it is without being handed the sequence
    /// itself. Every position written must be below `context`.
    fn positions(&self, query: u32, context: u32, out: &mut Vec<u32>);
}

/// A policy that proposes draft tokens for the `speculative` stage to verify.
///
/// The draft is only a guess: correctness stays with the verifier, which accepts a prefix of the
/// proposal and recomputes the rest, so a policy here trades acceptance rate for latency and never
/// changes what the model would have produced.
pub trait SpeculativeDecoder: Send + Sync {
    /// The name the runtime reports this policy under.
    fn name(&self) -> &'static str;

    /// Up to `count` tokens to try after `context`, appended to `out` in generation order.
    ///
    /// Proposing nothing is a valid answer: a build with no draft model registers a decoder that
    /// does exactly that, and the stage then costs nothing.
    fn draft(&self, context: &[u32], count: usize, out: &mut Vec<u32>);
}

/// A policy that decides which experts of a mixture-of-experts layer stay resident.
///
/// Offloading moves expert weights out of device memory between passes; this trait only says which
/// ones the next pass needs, so the policy can be swapped without the runtime knowing how the move
/// itself happens.
pub trait MoeOffloader: Send + Sync {
    /// The name the runtime reports this policy under.
    fn name(&self) -> &'static str;

    /// Indices of the experts `router_logits` selects, at most `capacity` of them, appended to
    /// `out` in the order they should be fetched.
    fn select_experts(&self, router_logits: &[f32], capacity: usize, out: &mut Vec<u32>);
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

    struct Inert;

    impl KvCompression for Inert {
        fn name(&self) -> &'static str {
            "inert"
        }

        fn compress(&self, _block: &[f32], _out: &mut Vec<u8>) -> usize {
            0
        }

        fn decompress(&self, _compressed: &[u8], _out: &mut [f32]) -> usize {
            0
        }
    }

    impl SparseAttention for Inert {
        fn name(&self) -> &'static str {
            "inert"
        }

        fn positions(&self, _query: u32, context: u32, out: &mut Vec<u32>) {
            out.extend(0..context);
        }
    }

    impl SpeculativeDecoder for Inert {
        fn name(&self) -> &'static str {
            "inert"
        }

        fn draft(&self, _context: &[u32], _count: usize, _out: &mut Vec<u32>) {}
    }

    impl MoeOffloader for Inert {
        fn name(&self) -> &'static str {
            "inert"
        }

        fn select_experts(&self, _router_logits: &[f32], _capacity: usize, _out: &mut Vec<u32>) {}
    }

    #[test]
    fn the_algorithm_traits_can_be_used_as_trait_objects() {
        let compression: &dyn KvCompression = &Inert;
        let sparse: &dyn SparseAttention = &Inert;
        let speculative: &dyn SpeculativeDecoder = &Inert;
        let offloader: &dyn MoeOffloader = &Inert;

        let mut bytes = Vec::new();
        assert_eq!(compression.compress(&[1.0, 2.0], &mut bytes), 0);
        let mut positions = Vec::new();
        sparse.positions(3, 3, &mut positions);
        assert_eq!(positions, vec![0, 1, 2]);
        let mut drafts = Vec::new();
        speculative.draft(&[1, 2], 4, &mut drafts);
        assert!(drafts.is_empty());
        let mut experts = Vec::new();
        offloader.select_experts(&[0.5], 1, &mut experts);
        assert!(experts.is_empty());
        assert_eq!(
            [
                compression.name(),
                sparse.name(),
                speculative.name(),
                offloader.name()
            ],
            ["inert"; 4]
        );
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
