//! Memory placement planning.
//!
//! Answers one question before a model is loaded: how many layers fit on the GPU, and how much device
//! and host memory that placement needs. The math is pure and separately testable; refusing an
//! infeasible placement is a feature, not an error path (spec FR-004).

use vipera_errors::ViperaError;
use vipera_traits::{DeviceKind, HardwareProfile};

use crate::gguf::{GgufFile, Value};

/// Device overhead that is neither model weights nor KV cache: the CUDA context, allocator slack, and
/// the compute buffer.
///
/// Measured, not assumed. On a GTX 1650 (4 GiB) with llama.cpp b24 loading
/// `qwen2.5-1.5b-instruct-q4_k_m` at 2048 and 4096 token contexts, observed device use was 1128 and
/// 1186 MiB against a predicted 1116 and 1172 MiB, a residual of 12.1 and 14.1 MiB. 16 MiB is the
/// rounded-up worst case, so a plan errs on the side of refusing rather than overcommitting.
///
/// This value tracks a reference implementation on one device class; re-measure when calibrating a
/// different backend (see `specs/001-vipera-rs-engine/quickstart.md`).
pub const DEVICE_RESERVE_BYTES: u64 = 16 * 1024 * 1024;

/// KV cache storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheType {
    /// Half precision, 2 bytes per element.
    F16,
    /// 8-bit blocks, 34 bytes per 32 elements.
    Q8_0,
    /// 4-bit blocks, 18 bytes per 32 elements.
    Q4_0,
}

impl KvCacheType {
    /// Parse the value accepted by `--kv-type`.
    pub fn parse(name: &str) -> Result<Self, ViperaError> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Self::F16,
            "q8_0" | "q8" => Self::Q8_0,
            "q4_0" | "q4" => Self::Q4_0,
            other => {
                return Err(ViperaError::unsupported(format!(
                    "kv cache type \"{other}\" is not supported; use f16, q8_0, or q4_0"
                )));
            }
        })
    }

    /// Name used in JSON output.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    /// Average bytes per cached element.
    pub const fn bytes_per_element(self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q4_0 => 18.0 / 32.0,
        }
    }
}

/// The parts of a model that planning needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// Transformer layers.
    pub n_layer: u64,
    /// Embedding width.
    pub n_embd: u64,
    /// Attention heads.
    pub n_head: u64,
    /// Key/value heads (grouped-query attention); equals `n_head` without GQA.
    pub n_head_kv: u64,
    /// Declared maximum context, when the file declares one.
    pub max_context: Option<u64>,
    /// Bytes of all weights.
    pub weights_bytes: u64,
    /// Weight bytes that never move to the device (token embeddings, output projection).
    pub non_layer_bytes: u64,
}

impl ModelShape {
    /// Read the shape from a parsed model file.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, ViperaError> {
        let architecture = file
            .metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or_else(|| ViperaError::unsupported("model does not declare general.architecture"))?
            .to_owned();

        let require = |key: String| -> Result<u64, ViperaError> {
            file.metadata
                .get(&key)
                .and_then(Value::as_u64)
                .ok_or_else(|| ViperaError::unsupported(format!("model does not declare {key}")))
        };

        let n_layer = require(format!("{architecture}.block_count"))?;
        let n_embd = require(format!("{architecture}.embedding_length"))?;
        let n_head = require(format!("{architecture}.attention.head_count"))?;
        let n_head_kv = file
            .metadata
            .get(&format!("{architecture}.attention.head_count_kv"))
            .and_then(Value::as_u64)
            .unwrap_or(n_head);
        let max_context = file
            .metadata
            .get(&format!("{architecture}.context_length"))
            .and_then(Value::as_u64)
            .or_else(|| {
                file.metadata
                    .get("general.context_length")
                    .and_then(Value::as_u64)
            });

        if n_head == 0 || n_head_kv == 0 {
            return Err(ViperaError::unsupported(
                "model declares zero attention heads, which cannot be planned",
            ));
        }

        let mut weights_bytes = 0u64;
        let mut non_layer_bytes = 0u64;
        for tensor in &file.tensors {
            let bytes = tensor.byte_len().ok_or_else(|| {
                ViperaError::unsupported(format!(
                    "tensor \"{}\" is not a whole number of {} blocks",
                    tensor.name,
                    tensor.ty.name()
                ))
            })?;
            weights_bytes = weights_bytes.saturating_add(bytes);
            if is_non_layer_tensor(&tensor.name) {
                non_layer_bytes = non_layer_bytes.saturating_add(bytes);
            }
        }

        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            max_context,
            weights_bytes,
            // A file whose "non-layer" tensors exceed the total would be nonsense; clamp instead of
            // wrapping so planning stays arithmetic instead of panicking.
            non_layer_bytes: non_layer_bytes.min(weights_bytes),
        })
    }

    /// Weight bytes per transformer layer.
    pub fn per_layer_bytes(&self) -> u64 {
        if self.n_layer == 0 {
            return 0;
        }
        self.weights_bytes.saturating_sub(self.non_layer_bytes) / self.n_layer
    }

    /// KV cache bytes for a full context at the given type.
    pub fn kv_bytes(&self, ctx: u64, kv_type: KvCacheType) -> Result<u64, ViperaError> {
        if self.n_head == 0 {
            return Err(ViperaError::internal("n_head is zero"));
        }
        let head_dim = self.n_embd / self.n_head;
        let elements = 2.0 // keys and values
            * self.n_layer as f64
            * ctx as f64
            * self.n_head_kv as f64
            * head_dim as f64;
        let bytes = elements * kv_type.bytes_per_element();
        if !bytes.is_finite() || bytes < 0.0 {
            return Err(ViperaError::internal("kv cache size is not representable"));
        }
        Ok(bytes.ceil() as u64)
    }
}

/// One constraint that makes a placement impossible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// `device_memory`, `host_memory`, or `context_limit`.
    pub constraint: &'static str,
    /// What the placement needs.
    pub required_bytes: u64,
    /// What is available.
    pub available_bytes: u64,
    /// Human-readable explanation.
    pub message: String,
}

/// What the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanRequest {
    /// Context length in tokens.
    pub ctx: u64,
    /// KV cache storage type.
    pub kv_type: KvCacheType,
    /// Batch size. The KV estimate covers this many sequences.
    pub batch_size: u64,
}

/// The answer: a placement and its predicted footprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    /// Identity of the model this plan is for.
    pub model_identity: String,
    /// Identity of the hardware this plan is for.
    pub hardware_fingerprint: String,
    /// Layers placed on the device; 0 means CPU-only.
    pub layers_on_device: u64,
    /// KV cache type used for the estimate.
    pub kv_cache_type: String,
    /// Context length the estimate covers.
    pub context_length: u64,
    /// Batch size recorded from the request.
    pub batch_size: u64,
    /// Predicted peak device memory, including [`DEVICE_RESERVE_BYTES`].
    pub predicted_device_bytes: u64,
    /// Part of `predicted_device_bytes` that is measured overhead rather than weights or KV.
    ///
    /// Zero when no layer is placed on the device.
    pub device_reserve_bytes: u64,
    /// Predicted peak host memory.
    pub predicted_host_bytes: u64,
    /// Whether this placement may be loaded.
    pub feasible: bool,
    /// Constraints that failed, when `feasible` is false.
    pub violations: Vec<Violation>,
}

/// Plan a placement for a parsed model file on a probed machine.
pub fn plan(
    file: &GgufFile,
    profile: &HardwareProfile,
    request: &PlanRequest,
) -> Result<PlacementPlan, ViperaError> {
    let shape = ModelShape::from_gguf(file)?;
    plan_shape(&file.identity(), profile, &shape, request)
}

/// Plan a placement from a shape, without a file. Pure arithmetic.
pub fn plan_shape(
    model_identity: &str,
    profile: &HardwareProfile,
    shape: &ModelShape,
    request: &PlanRequest,
) -> Result<PlacementPlan, ViperaError> {
    if request.ctx == 0 {
        return Err(ViperaError::unsupported(
            "context length must be at least 1",
        ));
    }
    let kv_per_sequence = shape.kv_bytes(request.ctx, request.kv_type)?;
    let kv_bytes = kv_per_sequence.saturating_mul(request.batch_size.max(1));
    let per_layer = shape.per_layer_bytes();

    let mut violations = Vec::new();
    if let Some(max) = shape.max_context
        && request.ctx > max
    {
        violations.push(Violation {
            constraint: "context_limit",
            required_bytes: request.ctx,
            available_bytes: max,
            message: format!(
                "requested context {} tokens exceeds the model's maximum of {max}",
                request.ctx
            ),
        });
    }

    let device = profile.device(DeviceKind::Cuda);

    let Some(device) = device else {
        let host_bytes = shape.weights_bytes.saturating_add(kv_bytes);
        if host_bytes > profile.ram_total_bytes {
            violations.push(Violation {
                constraint: "host_memory",
                required_bytes: host_bytes,
                available_bytes: profile.ram_total_bytes,
                message: format!(
                    "CPU-only placement needs {host_bytes} bytes of host memory, {0} available",
                    profile.ram_total_bytes
                ),
            });
        }
        return Ok(PlacementPlan {
            model_identity: model_identity.to_owned(),
            hardware_fingerprint: profile.fingerprint(),
            layers_on_device: 0,
            kv_cache_type: request.kv_type.as_str().to_owned(),
            context_length: request.ctx,
            batch_size: request.batch_size,
            predicted_device_bytes: 0,
            device_reserve_bytes: 0,
            predicted_host_bytes: host_bytes,
            feasible: violations.is_empty(),
            violations,
        });
    };

    // Prefer the most layers that fit. Device use grows monotonically with
    // layers while host use shrinks, so the largest device-feasible split is
    // the optimum when it also fits the host; a linear scan over n_layer
    // (file-controlled) would let a crafted file hang the planner, so binary
    // search the device bound instead.
    let device_fits = |layers: u64| {
        let (device_bytes, _) = footprint(shape, per_layer, kv_bytes, layers);
        device_bytes.saturating_add(reserve(layers)) <= device.memory_free_bytes
    };
    let mut lo = 0u64;
    let mut hi = shape.n_layer;
    let mut best_device_layers = 0u64;
    loop {
        let mid = lo + (hi.saturating_sub(lo) / 2);
        if device_fits(mid) {
            best_device_layers = mid;
            if mid == u64::MAX || mid == hi {
                break;
            }
            lo = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        }
        if lo > hi {
            break;
        }
    }
    {
        let (device_bytes, host_bytes) = footprint(shape, per_layer, kv_bytes, best_device_layers);
        let required = device_bytes.saturating_add(reserve(best_device_layers));
        if required <= device.memory_free_bytes && host_bytes <= profile.ram_total_bytes {
            return Ok(PlacementPlan {
                model_identity: model_identity.to_owned(),
                hardware_fingerprint: profile.fingerprint(),
                layers_on_device: best_device_layers,
                kv_cache_type: request.kv_type.as_str().to_owned(),
                context_length: request.ctx,
                batch_size: request.batch_size,
                predicted_device_bytes: required,
                device_reserve_bytes: reserve(best_device_layers),
                predicted_host_bytes: host_bytes,
                feasible: violations.is_empty(),
                violations,
            });
        }
    }

    // Nothing fits. Report the closest attempt — the most layers the device alone could hold — so the
    // message names the constraint that is actually binding instead of one that only looks binding
    // when every layer is on the GPU.
    let (device_bytes, host_bytes) = footprint(shape, per_layer, kv_bytes, best_device_layers);
    let device_required = device_bytes.saturating_add(reserve(best_device_layers));
    if device_required > device.memory_free_bytes {
        violations.push(Violation {
            constraint: "device_memory",
            required_bytes: device_required,
            available_bytes: device.memory_free_bytes,
            message: format!(
                "no layer split fits: {} needs {device_required} bytes of device memory, {} free",
                device.name, device.memory_free_bytes
            ),
        });
    }
    if host_bytes > profile.ram_total_bytes {
        violations.push(Violation {
            constraint: "host_memory",
            required_bytes: host_bytes,
            available_bytes: profile.ram_total_bytes,
            message: format!(
                "no layer split fits: {host_bytes} bytes of host memory needed, {0} total",
                profile.ram_total_bytes
            ),
        });
    }

    Ok(PlacementPlan {
        model_identity: model_identity.to_owned(),
        hardware_fingerprint: profile.fingerprint(),
        layers_on_device: best_device_layers,
        kv_cache_type: request.kv_type.as_str().to_owned(),
        context_length: request.ctx,
        batch_size: request.batch_size,
        predicted_device_bytes: device_required,
        device_reserve_bytes: reserve(best_device_layers),
        predicted_host_bytes: host_bytes,
        feasible: false,
        violations,
    })
}

/// Overhead counted once at least one layer is on the device.
fn reserve(layers: u64) -> u64 {
    if layers == 0 { 0 } else { DEVICE_RESERVE_BYTES }
}

fn footprint(shape: &ModelShape, per_layer: u64, kv_bytes: u64, layers: u64) -> (u64, u64) {
    if shape.n_layer == 0 {
        return (0, shape.weights_bytes.saturating_add(kv_bytes));
    }
    let layers = layers.min(shape.n_layer);
    // Full offload moves every tensor, including embeddings and the output projection. Splitting the
    // per-layer average instead would silently leave those on the host and understate device use.
    let device_weights = if layers == shape.n_layer {
        shape.weights_bytes
    } else {
        per_layer.saturating_mul(layers)
    };
    let host_weights = shape.weights_bytes.saturating_sub(device_weights);
    let device_kv = scale(kv_bytes, layers, shape.n_layer);
    let host_kv = kv_bytes.saturating_sub(device_kv);
    (
        device_weights.saturating_add(device_kv),
        host_weights.saturating_add(host_kv),
    )
}

fn scale(total: u64, part: u64, whole: u64) -> u64 {
    if whole == 0 {
        return 0;
    }
    ((total as u128 * part as u128) / whole as u128) as u64
}

fn is_non_layer_tensor(name: &str) -> bool {
    name.contains("token_embd") || name.starts_with("output") || name.contains("output.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vipera_traits::{Device, HardwareProfile};

    fn shape() -> ModelShape {
        ModelShape {
            n_layer: 32,
            n_embd: 4096,
            n_head: 32,
            n_head_kv: 8,
            max_context: Some(32768),
            weights_bytes: 8 * 1024 * 1024 * 1024,
            non_layer_bytes: 1024 * 1024 * 1024,
        }
    }

    fn machine(vram_free: u64, ram: u64) -> HardwareProfile {
        HardwareProfile {
            cpu_cores_physical: 8,
            ram_total_bytes: ram,
            devices: vec![Device {
                kind: DeviceKind::Cuda,
                name: "Test GPU".to_owned(),
                memory_total_bytes: vram_free,
                memory_free_bytes: vram_free,
                driver_version: None,
            }],
        }
    }

    fn cpu_only(ram: u64) -> HardwareProfile {
        HardwareProfile {
            cpu_cores_physical: 8,
            ram_total_bytes: ram,
            devices: Vec::new(),
        }
    }

    fn request(ctx: u64) -> PlanRequest {
        PlanRequest {
            ctx,
            kv_type: KvCacheType::F16,
            batch_size: 1,
        }
    }

    #[test]
    fn batch_size_scales_the_kv_estimate() {
        let one = plan_shape(
            "m",
            &machine(48 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024),
            &shape(),
            &request(4096),
        )
        .unwrap();
        let mut req = request(4096);
        req.batch_size = 8;
        let eight = plan_shape(
            "m",
            &machine(48 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024),
            &shape(),
            &req,
        )
        .unwrap();
        assert!(
            eight.predicted_host_bytes > one.predicted_host_bytes
                || eight.predicted_device_bytes > one.predicted_device_bytes
        );
        assert!(eight.batch_size == 8);
    }

    #[test]
    fn huge_layer_count_returns_quickly() {
        let mut big = shape();
        big.n_layer = 10_000_000;
        big.weights_bytes = big.n_layer * 1024;
        let start = std::time::Instant::now();
        let plan = plan_shape("m", &machine(1024, 1024), &big, &request(4096)).unwrap();
        assert!(!plan.feasible);
        assert!(
            start.elapsed().as_secs() < 5,
            "planner took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn full_offload_moves_every_weight_to_the_device() {
        let kv = shape().kv_bytes(4096, KvCacheType::F16).unwrap();
        let (device, host) = footprint(&shape(), shape().per_layer_bytes(), kv, shape().n_layer);
        assert_eq!(device, shape().weights_bytes + kv);
        assert_eq!(host, 0);
    }

    #[test]
    fn a_partial_split_leaves_the_non_layer_weights_on_the_host() {
        let kv = shape().kv_bytes(4096, KvCacheType::F16).unwrap();
        let (device, host) = footprint(&shape(), shape().per_layer_bytes(), kv, 8);
        assert!(host > 0);
        assert!(device < shape().weights_bytes);
    }

    #[test]
    fn the_device_reserve_is_included_in_the_prediction() {
        let profile = machine(48 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(4096)).unwrap();
        assert_eq!(plan.layers_on_device, shape().n_layer);
        assert_eq!(plan.device_reserve_bytes, DEVICE_RESERVE_BYTES);
        assert_eq!(
            plan.predicted_device_bytes,
            shape().weights_bytes
                + shape().kv_bytes(4096, KvCacheType::F16).unwrap()
                + DEVICE_RESERVE_BYTES
        );
    }

    #[test]
    fn a_device_that_only_fits_without_the_reserve_is_refused() {
        // Free memory exactly equals weights plus KV; the measured overhead is the difference
        // between a plan that loads and one that runs out of memory.
        let kv = shape().kv_bytes(4096, KvCacheType::F16).unwrap();
        let profile = machine(shape().weights_bytes + kv, 128 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(4096)).unwrap();
        assert!(plan.layers_on_device < shape().n_layer);
        assert!(plan.predicted_device_bytes <= profile.devices[0].memory_free_bytes);
    }

    #[test]
    fn cpu_only_plans_carry_no_device_reserve() {
        let plan = plan_shape(
            "m",
            &cpu_only(128 * 1024 * 1024 * 1024),
            &shape(),
            &request(4096),
        )
        .unwrap();
        assert_eq!(plan.device_reserve_bytes, 0);
    }

    #[test]
    fn kv_cache_math_is_exact_for_f16() {
        // 2 (K and V) * 32 layers * 4096 tokens * 8 kv heads * 128 head dim * 2 bytes
        let bytes = shape().kv_bytes(4096, KvCacheType::F16).unwrap();
        assert_eq!(bytes, 536_870_912);
    }

    #[test]
    fn quantized_kv_is_smaller_than_f16() {
        let f16 = shape().kv_bytes(4096, KvCacheType::F16).unwrap();
        let q8 = shape().kv_bytes(4096, KvCacheType::Q8_0).unwrap();
        let q4 = shape().kv_bytes(4096, KvCacheType::Q4_0).unwrap();
        assert!(q8 < f16 && q4 < q8);
    }

    #[test]
    fn picks_the_most_layers_that_fit() {
        let profile = machine(4 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(4096)).unwrap();

        assert!(plan.feasible);
        assert!(plan.violations.is_empty());
        assert!(plan.layers_on_device > 0 && plan.layers_on_device <= 32);
        assert!(plan.predicted_device_bytes <= 4 * 1024 * 1024 * 1024);

        // One more layer must not fit, otherwise the ladder stopped early.
        if plan.layers_on_device < 32 {
            let (next_device, _) = footprint(
                &shape(),
                shape().per_layer_bytes(),
                shape().kv_bytes(4096, KvCacheType::F16).unwrap(),
                plan.layers_on_device + 1,
            );
            assert!(next_device + DEVICE_RESERVE_BYTES > 4 * 1024 * 1024 * 1024);
        }
    }

    #[test]
    fn a_model_larger_than_ram_is_infeasible_with_a_reason() {
        // 4 GiB of VRAM can hold roughly half the 8 GiB of weights, but the remainder plus the KV
        // cache does not fit in 2 GiB of host memory, so no layer split works.
        let profile = machine(4 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(4096)).unwrap();

        assert!(!plan.feasible);
        assert!(
            plan.violations
                .iter()
                .any(|v| v.constraint == "host_memory")
        );
        assert!(plan.violations.iter().all(|v| !v.message.is_empty()));
    }

    #[test]
    fn requesting_more_context_than_the_model_supports_is_refused() {
        let profile = machine(48 * 1024 * 1024 * 1024, 128 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(65_536)).unwrap();

        assert!(!plan.feasible);
        let violation = plan
            .violations
            .iter()
            .find(|v| v.constraint == "context_limit")
            .expect("context violation");
        assert_eq!(violation.required_bytes, 65_536);
        assert_eq!(violation.available_bytes, 32_768);
    }

    #[test]
    fn cpu_only_machines_keep_every_layer_on_the_host() {
        let profile = cpu_only(128 * 1024 * 1024 * 1024);
        let plan = plan_shape("m", &profile, &shape(), &request(4096)).unwrap();

        assert!(plan.feasible);
        assert_eq!(plan.layers_on_device, 0);
        assert_eq!(plan.predicted_device_bytes, 0);
        assert!(plan.predicted_host_bytes >= shape().weights_bytes);
    }

    #[test]
    fn zero_context_is_rejected() {
        let err = plan_shape("m", &cpu_only(1 << 30), &shape(), &request(0)).unwrap_err();
        assert!(err.to_string().contains("context"));
    }

    #[test]
    fn kv_type_names_round_trip() {
        for name in ["f16", "q8_0", "q4_0", "Q4_0"] {
            assert_eq!(
                KvCacheType::parse(name).unwrap().as_str(),
                KvCacheType::parse(name).unwrap().as_str()
            );
        }
        assert!(KvCacheType::parse("q2_k").is_err());
    }
}
