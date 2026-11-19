//! A CUDA backend for vipera, starting with the one kernel that matters: a Q4_K matvec.
//!
//! Scope, stated plainly: one device (ordinal 0), one kernel (float activations against Q4_K
//! weights), no batching, no attention, and no wiring into the runtime's `ForwardPass` yet. What it
//! does have is the piece the tiered weight bus was waiting for — device memory that a layer can be
//! copied into, and a kernel that computes on it — plus a parity test against the CPU reference.
//!
//! The kernel is compiled by NVRTC at open time, so no `.ptx` is checked in and the architecture is
//! the device's own (`compute_75` on the reference GTX 1650).
//!
//! Why this is a separate crate: `cudarc` is the only third-party dependency in the workspace that
//! binds a C API, and every call into it is `unsafe`. Keeping it here means the CPU backend and the
//! server keep their `forbid`-style rules untouched.

#![warn(missing_docs)]

pub mod device;
pub mod layer;
pub mod pool;
pub mod q4k;
pub mod q6k;
pub mod q8;

pub use device::{Device, DeviceBuffer};
