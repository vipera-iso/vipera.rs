//! CPU forward pass over GGUF weights.
//!
//! This is the backend the runtime was built to accept: [`CpuBackend`] implements
//! [`vipera_runtime::ForwardPass`] over the weights as they sit in the file, writes K/V into the
//! runtime's own KV buffer, and reads its rotary and norm parameters from the file's metadata rather
//! than from constants that would quietly be wrong for another model.
//!
//! Scope, stated plainly: the qwen2, qwen3 and llama block layouts, F32/F16/Q8_0/Q4_K/Q6_K weights,
//! and an `f16` or `q8_0` KV cache. On x86-64 with AVX2 the hot dots run through `simd`, which the tests hold to the scalar
//! result; every other CPU falls back to the same portable code. A model outside that is refused by name instead of being run approximately.

#![warn(missing_docs)]

pub mod bus;
pub mod compress;
pub mod model;
mod pool;
pub mod profile;
#[cfg(test)]
pub(crate) mod q4k8_lab;
pub mod quant;
#[cfg(test)]
pub(crate) mod repack;

// The scalar kernels are the reference for the x86 SIMD paths and the portable implementation on
// architectures without a SIMD module, so both sides can be built and compared on one machine.
#[cfg(any(test, target_arch = "aarch64", not(target_arch = "x86_64")))]
pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod simd;

#[cfg(target_arch = "aarch64")]
#[path = "simd_aarch64.rs"]
pub(crate) mod simd;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) mod simd {
    pub(crate) use crate::scalar::*;

    /// The `f16` and `f32` kernels are the same scalar code here, so an in-place row is the
    /// decoded row's bits.
    pub(crate) const IN_PLACE_KV_BIT_EXACT: bool = true;
}
pub(crate) mod tiled;

pub use compress::Q8Kv;
pub use model::{CpuBackend, ModelConfig};
pub use quant::{dequantize, f16_to_f32, f32_to_f16, matvec, read_row, row_bytes};

/// The fastest dot-product level this process can use, by name.
///
/// One of `avx512-vnni`, `vnni256`, `avx2` (x86-64), `neon` (aarch64), or `scalar`. The level is
/// what the runtime feature checks found — and `VIPERA_NO_VNNI` lowers it, because the checks
/// honour that switch — so a deployment can read this to see which machine fell back. It is not
/// an exact statement of the kernel a given tensor takes: `VIPERA_TILED`, `VIPERA_Q4K8` and
/// `VIPERA_EXACT_DOT` still choose between paths inside a level, and the Q4_K prefill tile is
/// only reached at `avx512-vnni` (a batch below `TILE_MIN_BATCH` keeps the row kernel).
pub fn simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::vnni512_available() {
            "avx512-vnni"
        } else if simd::vnni_available() {
            "vnni256"
        } else if simd::available() {
            "avx2"
        } else {
            "scalar"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}
