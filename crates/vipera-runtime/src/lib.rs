//! Runtime loop: KV blocks, sequence lifecycle, the seven-step pipeline, snapshots, and timings.
//!
//! This crate owns everything between a prompt and a generated token except the matrix
//! multiplications. It keeps a real KV cache, reuses retained context across turns, restores
//! snapshots, releases memory when a conversation ends, and reports per-stage timings.
//!
//! Several sequences can share one step: [`Engine::step`] batches the turns that need a forward pass
//! into one [`BatchContext`], so several requests share a weight read.
//!
//! What it does **not** contain is a compute backend. [`ForwardPass`] is the seam a backend plugs
//! into: `vipera-backend-cpu` reads GGUF weights and writes K/V into this crate's paged buffer, while
//! the tests here drive the loop with a deterministic model and need neither weights nor a device.

#![warn(missing_docs)]

pub mod engine;
pub mod kv;
pub mod memory;
pub mod moe;
pub mod pipeline;
pub mod sequence;

pub use engine::{
    Engine, EngineConfig, Generation, Released, RuntimeStats, StepReport, ngram_draft,
    speculative_accept,
};
pub use kv::{DEFAULT_TOKENS_PER_BLOCK, KvCache, KvConfig};
pub use memory::{WeightBuffer, parse_status_kib, peak_resident_bytes, resident_bytes};
pub use moe::MoEExpertCache;
pub use pipeline::{
    BatchContext, BatchEntry, ForwardContext, ForwardPass, RequestLog, Stage, StageTimings,
};
pub use sequence::{FinishReason, Sequence, SequenceState, Snapshot};
pub use vipera_algo_keepkv::KeepKv;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::kv::KvCache;
    use pipeline::ForwardPass;

    /// Helper: a minimal ForwardPass implementation for testing.
    struct ToyForward {
        vocab: usize,
    }

    impl ToyForward {
        fn new(vocab: usize) -> Self {
            Self { vocab }
        }
    }

    use vipera_errors::ViperaError;

    impl ForwardPass for ToyForward {
        fn logits(&mut self, _context: &mut ForwardContext<'_>) -> Result<Vec<f32>, ViperaError> {
            Ok(vec![0.0f32; self.vocab])
        }
    }

    /// Regression test: `AlgorithmRegistry::plan` with an empty plan (`AlgorithmPlan::default()`)
    /// should resolve to identity strategies (all `None`) without panicking.
    #[test]
    fn plan_algorithms_with_empty_plan() {
        let registry = vipera_traits::AlgorithmRegistry::new(&[], &[], &[], &[]);
        let plan = vipera_traits::AlgorithmPlan::default();
        let result = registry.plan(&plan);
        assert!(result.is_ok(), "empty plan should resolve successfully");
        let planned = result.unwrap();
        assert!(planned.compression.is_none());
        assert!(planned.sparse.is_none());
        assert!(planned.speculative.is_none());
        assert!(planned.offloading.is_none());
    }

    /// Regression test: `Engine::admit` with an empty prompt should return an error
    /// rather than panicking.
    #[test]
    fn admit_with_empty_prompt() {
        let config = KvConfig::new(64, 4, 8).unwrap();
        let kv = KvCache::new(config).unwrap();
        let mut engine = Engine::new(kv, ToyForward::new(32), EngineConfig::default());
        let result = engine.admit(&[], 5);
        assert!(result.is_err(), "admit with empty prompt should fail");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("a prompt needs at least one token"),
            "error message should mention empty prompt"
        );
    }

    /// Regression test: `Engine::admit` with `max_new_tokens == 0` should return an error
    /// rather than panicking.
    #[test]
    fn admit_with_zero_max_new_tokens() {
        let config = KvConfig::new(64, 4, 8).unwrap();
        let kv = KvCache::new(config).unwrap();
        let mut engine = Engine::new(kv, ToyForward::new(32), EngineConfig::default());
        let result = engine.admit(&[1], 0);
        assert!(
            result.is_err(),
            "admit with zero max_new_tokens should fail"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("max_new_tokens must be at least 1"),
            "error message should mention zero max_new_tokens"
        );
    }

    /// Regression test: `Engine::generate` on a sequence with valid tokens should succeed.
    #[test]
    fn generate_with_valid_sequence() {
        let config = KvConfig::new(64, 4, 8).unwrap();
        let kv = KvCache::new(config).unwrap();
        let mut engine = Engine::new(kv, ToyForward::new(32), EngineConfig::default());
        let id = engine.admit(&[1, 2, 3], 5).expect("admit should succeed");
        let r#gen = engine.generate(id);
        assert!(
            r#gen.is_ok(),
            "generate on a sequence with tokens should succeed"
        );
    }

    /// Regression test: `Engine::step` on an empty engine (no sequences, no turns)
    /// should return a default `StepReport` rather than panicking.
    #[test]
    fn step_on_empty_engine() {
        let config = KvConfig::new(64, 4, 8).unwrap();
        let kv = KvCache::new(config).unwrap();
        let mut engine = Engine::new(kv, ToyForward::new(32), EngineConfig::default());
        let result = engine.step();
        // Should return Ok with default StepReport, not panic
        assert!(result.is_ok(), "step on empty engine should not panic");
        let report = result.unwrap();
        assert!(report.tokens.is_empty());
        assert!(report.finished.is_empty());
        assert!(report.failed.is_empty());
    }
}
