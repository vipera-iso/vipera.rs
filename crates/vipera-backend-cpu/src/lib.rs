//! CPU forward pass over GGUF weights.
//!
//! This is the backend the runtime was built to accept: [`CpuBackend`] implements
//! [`vipera_runtime::ForwardPass`] over the weights as they sit in the file, writes K/V into the
//! runtime's own KV buffer, and reads its rotary and norm parameters from the file's metadata rather
//! than from constants that would quietly be wrong for another model.
//!
//! Scope, stated plainly: the qwen2/llama block layout, F32/F16/Q8_0/Q4_K/Q6_K weights, and an `f16`
//! KV cache. On x86-64 with AVX2 the hot dots run through `simd`, which the tests hold to the scalar
//! result; every other CPU falls back to the same portable code. A model outside that is refused by name instead of being run approximately.

#![warn(missing_docs)]

pub mod model;
mod pool;
pub mod quant;
#[cfg(target_arch = "x86_64")]
pub mod simd;

pub use model::{CpuBackend, ModelConfig};
pub use quant::{dequantize, f16_to_f32, f32_to_f16, matvec, read_row, row_bytes};
