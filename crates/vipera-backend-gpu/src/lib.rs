//! A CUDA backend for vipera: the whole bounded model on one device, behind the runtime's
//! `ForwardPass`.
//!
//! Scope, stated plainly: one device (ordinal 0), the Qwen2/Llama block the CPU backend runs, Q4_K
//! and Q6_K weights, `f16` KV, and one sequence at a time. [`GpuBackend`] is the trait implementation:
//! it computes a step on the card and keeps the runtime's paged KV buffer truthful, so a prefix-cache
//! block the runtime moves is meaningful to either backend. Prefill is sequential — one position per
//! weight read — until the batched kernel is wired through the layer, and a ring cache (sliding
//! window) is refused by name because the device attention kernels walk positions densely.
//!
//! The kernels are compiled by NVRTC at open time, so no `.ptx` is checked in and the architecture is
//! the device's own (`compute_75` on the reference GTX 1650).
//!
//! Why this is a separate crate: `cudarc` is the only third-party dependency in the workspace that
//! binds a C API, and every call into it is `unsafe`. Keeping it here means the CPU backend and the
//! server keep their `forbid`-style rules untouched.

#![warn(missing_docs)]

pub mod backend;
pub mod device;
pub mod layer;
pub mod model;
mod paged;
pub mod pool;
pub mod q4k;
pub mod q6k;
pub mod q8;

pub use backend::GpuBackend;
pub use device::{Device, DeviceBuffer};
pub use model::GpuModel;
