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
pub mod profile;
pub mod quant;
#[cfg(test)]
pub(crate) mod repack;
#[cfg(target_arch = "x86_64")]
pub mod simd;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) mod simd {
    //! Portable scalar fallbacks for non-x86_64: same semantics as the scalar
    //! branches inside the x86 kernels, without any intrinsics.
    use crate::quant::f16_to_f32;

    pub(crate) fn half_to_float(source: &[u8], values: &mut [f32]) {
        let length = values.len().min(source.len() / 2);
        for (value, chunk) in values[..length]
            .iter_mut()
            .zip(source.as_chunks::<2>().0.iter())
        {
            *value = f16_to_f32(u16::from_le_bytes(*chunk));
        }
    }

    pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let length = a.len().min(b.len());
        a.iter().zip(b).take(length).map(|(a, b)| a * b).sum()
    }

    pub(crate) fn axpy_f32(target: &mut [f32], weight: f32, source: &[f32]) {
        let length = target.len().min(source.len());
        for (target, source) in target.iter_mut().zip(source).take(length) {
            *target += weight * source;
        }
    }
}
pub(crate) mod tiled;

pub use model::{CpuBackend, ModelConfig};
pub use quant::{dequantize, f16_to_f32, f32_to_f16, matvec, read_row, row_bytes};
