//! Runtime loop: KV blocks, sequence lifecycle, the seven-step pipeline, snapshots, and timings.
//!
//! This crate owns everything between a prompt and a generated token except the matrix
//! multiplications. It keeps a real KV cache, reuses retained context across turns, restores
//! snapshots, releases memory when a conversation ends, and reports per-stage timings.
//!
//! What it does **not** contain is a compute backend. [`ForwardPass`] is the seam a backend plugs
//! into: `vipera-backend-cpu` reads GGUF weights and writes K/V into this crate's paged buffer, while
//! the tests here drive the loop with a deterministic model and need neither weights nor a device.

pub mod engine;
pub mod kv;
pub mod memory;
pub mod pipeline;
pub mod sequence;

pub use engine::{Engine, EngineConfig, Generation, Released, RuntimeStats};
pub use kv::{DEFAULT_TOKENS_PER_BLOCK, KvCache, KvConfig};
pub use memory::{WeightBuffer, parse_status_kib, peak_resident_bytes, resident_bytes};
pub use pipeline::{ForwardContext, ForwardPass, RequestLog, Stage, StageTimings};
pub use sequence::{FinishReason, Sequence, SequenceState, Snapshot};
