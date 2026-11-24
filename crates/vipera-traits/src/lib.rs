//! Contracts shared across crates.
//!
//! Three groups so far: how the engine learns what hardware it is running on, what a sampling
//! strategy has to look like so a build can list the ones it has, and the four algorithm slots a
//! build fills at startup — KV compression, sparse attention, speculative decoding and MoE
//! offloading — the registry that lists what a build has, and the plan that picks from it.
//! Backend and tensor traits arrive with the backends themselves, so this crate stays
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

/// The algorithm implementations a build has, one list per slot.
///
/// A registry says what exists; an [`AlgorithmPlan`] says what to run. Keeping the two apart is
/// what makes a name checkable before any work starts: a plan resolves against the registry once,
/// and a name this build does not have fails there instead of leaving a stage quietly on its
/// identity strategy. Everything here is keyed on the name an implementation reports — the same
/// string the pipeline prints — so a log line and a flag cannot disagree about what ran.
#[derive(Clone, Copy)]
pub struct AlgorithmRegistry {
    compression: &'static [&'static dyn KvCompression],
    sparse: &'static [&'static dyn SparseAttention],
    speculative: &'static [&'static dyn SpeculativeDecoder],
    offloading: &'static [&'static dyn MoeOffloader],
}

impl fmt::Debug for AlgorithmRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

impl AlgorithmRegistry {
    /// Wrap the build's lists.
    ///
    /// The lists are `'static` because every entry is a shared value that lives for the process;
    /// that is also what lets a resolved plan hand the engine a `&'static dyn` policy.
    pub const fn new(
        compression: &'static [&'static dyn KvCompression],
        sparse: &'static [&'static dyn SparseAttention],
        speculative: &'static [&'static dyn SpeculativeDecoder],
        offloading: &'static [&'static dyn MoeOffloader],
    ) -> Self {
        Self {
            compression,
            sparse,
            speculative,
            offloading,
        }
    }

    /// The KV compression named `name`, or `None` if this build has none by that name.
    pub fn compression(&self, name: &str) -> Option<&'static dyn KvCompression> {
        self.compression
            .iter()
            .copied()
            .find(|entry| entry.name() == name)
    }

    /// The sparse-attention policy named `name`, or `None`.
    pub fn sparse(&self, name: &str) -> Option<&'static dyn SparseAttention> {
        self.sparse
            .iter()
            .copied()
            .find(|entry| entry.name() == name)
    }

    /// The speculative decoder named `name`, or `None`.
    pub fn speculative(&self, name: &str) -> Option<&'static dyn SpeculativeDecoder> {
        self.speculative
            .iter()
            .copied()
            .find(|entry| entry.name() == name)
    }

    /// The MoE offloader named `name`, or `None`.
    pub fn offloading(&self, name: &str) -> Option<&'static dyn MoeOffloader> {
        self.offloading
            .iter()
            .copied()
            .find(|entry| entry.name() == name)
    }

    /// Every name in the registry, slot by slot, in list order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        let compression = self.compression.iter().map(|entry| entry.name());
        let sparse = self.sparse.iter().map(|entry| entry.name());
        let speculative = self.speculative.iter().map(|entry| entry.name());
        let offloading = self.offloading.iter().map(|entry| entry.name());
        compression
            .chain(sparse)
            .chain(speculative)
            .chain(offloading)
    }

    /// Resolve `plan` against this registry.
    ///
    /// A plan that names something this build does not have fails whole: every slot is looked up
    /// before any policy is returned, so a caller never acts on half a plan.
    pub fn plan(&self, plan: &AlgorithmPlan) -> Result<PlannedAlgorithms, MissingAlgorithm> {
        Ok(PlannedAlgorithms {
            compression: resolve(plan.compression, "compression", |name| {
                self.compression(name)
            })?,
            sparse: resolve(plan.sparse, "sparse", |name| self.sparse(name))?,
            speculative: resolve(plan.speculative, "speculative", |name| {
                self.speculative(name)
            })?,
            offloading: resolve(plan.offloading, "offloading", |name| self.offloading(name))?,
        })
    }
}

/// Look up one planned slot, naming the slot and the missing name when the build lacks it.
fn resolve<T: ?Sized>(
    name: Option<&'static str>,
    slot: &'static str,
    lookup: impl Fn(&str) -> Option<&'static T>,
) -> Result<Option<&'static T>, MissingAlgorithm> {
    match name {
        Some(name) => lookup(name)
            .map(Some)
            .ok_or(MissingAlgorithm { slot, name }),
        None => Ok(None),
    }
}

/// Which implementations a build was asked to run, by the names a registry lists.
///
/// Fields are names rather than references so a plan can be built from configuration — a flag, a
/// file, a default — before the registry that has to satisfy it is in hand. `None` asks for the
/// slot's identity strategy, which is what a build without an algorithm does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AlgorithmPlan {
    /// [`KvCompression`] to run in the compress stage.
    pub compression: Option<&'static str>,
    /// [`SparseAttention`] to run in the sparse stage.
    pub sparse: Option<&'static str>,
    /// [`SpeculativeDecoder`] to run in the speculative stage.
    pub speculative: Option<&'static str>,
    /// [`MoeOffloader`] to consult about which experts stay resident.
    pub offloading: Option<&'static str>,
}

/// A plan resolved against a registry: every slot it named is now a policy.
#[derive(Clone, Copy)]
pub struct PlannedAlgorithms {
    /// The compression policy to run, if the plan named one.
    pub compression: Option<&'static dyn KvCompression>,
    /// The sparse-attention policy to run, if the plan named one.
    pub sparse: Option<&'static dyn SparseAttention>,
    /// The speculative decoder to run, if the plan named one.
    pub speculative: Option<&'static dyn SpeculativeDecoder>,
    /// The MoE offloader to run, if the plan named one.
    pub offloading: Option<&'static dyn MoeOffloader>,
}

impl fmt::Debug for PlannedAlgorithms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlannedAlgorithms")
            .field("compression", &self.compression.map(|policy| policy.name()))
            .field("sparse", &self.sparse.map(|policy| policy.name()))
            .field("speculative", &self.speculative.map(|policy| policy.name()))
            .field("offloading", &self.offloading.map(|policy| policy.name()))
            .finish()
    }
}

/// A plan named an implementation the build does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingAlgorithm {
    /// Slot the name was looked up in: `compression`, `sparse`, `speculative`, or `offloading`.
    pub slot: &'static str,
    /// The name the plan asked for.
    pub name: &'static str,
}

impl fmt::Display for MissingAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no {} algorithm named '{}'", self.slot, self.name)
    }
}

impl std::error::Error for MissingAlgorithm {}

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

    #[test]
    fn fnv1a64_of_an_empty_slice_is_the_offset_basis() {
        let mut hash = Fnv1a64::new();
        hash.write(&[]);
        assert_eq!(hash.finish(), Fnv1a64::default().finish());
    }

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
    fn a_machine_without_devices_still_fingerprints() {
        let bare = HardwareProfile {
            cpu_cores_physical: 4,
            ram_total_bytes: 8,
            devices: Vec::new(),
        };
        assert_eq!(bare.device(DeviceKind::Cuda), None);
        assert_eq!(bare.device(DeviceKind::Cpu), None);
        assert!(bare.fingerprint().starts_with("hw_"));
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
    fn empty_algorithm_inputs_stay_empty() {
        let compression: &dyn KvCompression = &Inert;
        let mut bytes = Vec::new();
        assert_eq!(compression.compress(&[], &mut bytes), 0);
        let mut values = vec![0.0_f32; 4];
        assert_eq!(compression.decompress(&[], &mut values), 0);

        let speculative: &dyn SpeculativeDecoder = &Inert;
        let mut drafts = Vec::new();
        speculative.draft(&[], 0, &mut drafts);
        assert!(drafts.is_empty());

        let offloader: &dyn MoeOffloader = &Inert;
        let mut experts = Vec::new();
        offloader.select_experts(&[], 0, &mut experts);
        assert!(experts.is_empty());
    }

    #[test]
    fn a_sparse_policy_with_zero_retained_context_writes_nothing() {
        let sparse: &dyn SparseAttention = &Inert;
        let mut positions = Vec::new();
        sparse.positions(0, 0, &mut positions);
        assert!(positions.is_empty());

        let every_other: &dyn SparseAttention = &EveryOther;
        let mut every_other_positions = Vec::new();
        every_other.positions(0, 0, &mut every_other_positions);
        assert!(every_other_positions.is_empty());
    }

    #[test]
    fn sparse_positions_written_are_all_below_the_context() {
        let sparse: &dyn SparseAttention = &Inert;
        let mut positions = Vec::new();
        sparse.positions(2, 4, &mut positions);
        assert_eq!(positions, vec![0, 1, 2, 3]);
        assert!(positions.iter().all(|&position| position < 4));
    }

    /// A sparse policy with a name of its own, so a slot mix-up is observable.
    struct EveryOther;

    impl SparseAttention for EveryOther {
        fn name(&self) -> &'static str {
            "every-other"
        }

        fn positions(&self, _query: u32, context: u32, out: &mut Vec<u32>) {
            out.extend((0..context).step_by(2));
        }
    }

    static ONLY_SPARSE: [&dyn SparseAttention; 1] = [&EveryOther];

    fn sparse_registry() -> AlgorithmRegistry {
        AlgorithmRegistry::new(&[], &ONLY_SPARSE, &[], &[])
    }

    #[test]
    fn a_registry_finds_an_algorithm_in_its_own_slot() {
        let registry = sparse_registry();
        assert_eq!(
            registry.sparse("every-other").map(|policy| policy.name()),
            Some("every-other")
        );
        // The same name in another slot is a different question, and the answer is no.
        assert!(registry.compression("every-other").is_none());
        assert_eq!(registry.names().collect::<Vec<_>>(), ["every-other"]);
    }

    #[test]
    fn a_plan_resolves_each_slot_it_names() {
        let planned = sparse_registry()
            .plan(&AlgorithmPlan {
                sparse: Some("every-other"),
                ..AlgorithmPlan::default()
            })
            .expect("the plan names what this build has");
        assert_eq!(
            planned.sparse.map(|policy| policy.name()),
            Some("every-other")
        );
        assert!(planned.compression.is_none());
        assert!(planned.speculative.is_none());
        assert!(planned.offloading.is_none());
    }

    #[test]
    fn an_empty_plan_resolves_to_no_policies() {
        let planned = sparse_registry()
            .plan(&AlgorithmPlan::default())
            .expect("an empty plan asks for nothing");
        assert!(planned.sparse.is_none());
        assert!(planned.compression.is_none());
    }

    #[test]
    fn a_plan_naming_an_absent_algorithm_says_which_slot_and_name() {
        let missing = sparse_registry()
            .plan(&AlgorithmPlan {
                sparse: Some("absorbed"),
                ..AlgorithmPlan::default()
            })
            .expect_err("this build has no such sparse policy");
        assert_eq!(missing.slot, "sparse");
        assert_eq!(missing.name, "absorbed");
        assert_eq!(missing.to_string(), "no sparse algorithm named 'absorbed'");
    }

    #[test]
    fn a_plan_is_checked_whole_before_any_slot_resolves() {
        let missing = sparse_registry()
            .plan(&AlgorithmPlan {
                sparse: Some("every-other"),
                offloading: Some("moe"),
                ..AlgorithmPlan::default()
            })
            .expect_err("the offloading name is absent even though sparse resolves");
        assert_eq!(missing.slot, "offloading");
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
