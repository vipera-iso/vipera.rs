//! Q6_K, the format a Q4_K_M file keeps `attn_v` and `ffn_down` in.
//!
//! Float activations for now: the kernel exists so a whole layer can run, and the int8 (`dp4a`)
//! version is a separate step once the layer's numbers are right. Q6_K is not affine the way Q4_K is —
//! it is `d * sc * (q - 32)` with no minimum term — so the int8 form needs only the dot, which makes
//! it a smaller job than Q4_K's.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};

/// Values per block.
pub const QK_K: usize = 256;

/// Bytes per block: 128 low nibbles, 64 high pairs, 16 scales, one f16.
pub const BLOCK_BYTES: usize = 210;

// Four, not eight: `ffn_down` is 35 blocks per row, so eight rows of staging would ask for
// 58.8 KB of shared memory and the launch is refused above 48 KB.
const ROWS_PER_BLOCK: usize = 4;
const THREADS: usize = 32 * ROWS_PER_BLOCK;
pub(crate) const SOURCE: &str = include_str!("q6k_matvec.cu");

/// Q6_K weights resident on the device.
pub struct Q6kMatrix {
    rows: usize,
    cols: usize,
    weights: DeviceBuffer<u8>,
}

impl Q6kMatrix {
    /// Copy a Q6_K tensor up, checking the byte count against the shape.
    pub fn upload(
        device: &Device,
        weights: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self, ViperaError> {
        if !cols.is_multiple_of(QK_K) {
            return Err(ViperaError::invalid_argument(format!(
                "Q6_K needs a column count that is a multiple of {QK_K}, got {cols}"
            )));
        }
        let blocks_per_row = cols / QK_K;
        let expected = rows * blocks_per_row * BLOCK_BYTES;
        if weights.len() != expected {
            return Err(ViperaError::invalid_argument(format!(
                "Q6_K weights are {} bytes, {rows}x{cols} needs {expected}",
                weights.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            weights: device.upload(weights)?,
        })
    }

    /// Rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Bytes of weights held on the device.
    pub fn bytes(&self) -> usize {
        self.weights.len()
    }

    /// `y = W · x` into a caller's buffer.
    pub fn matvec_into(
        &self,
        device: &Device,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        matvec_view(
            device,
            &self.weights.slice.slice(..),
            self.rows,
            self.cols,
            x,
            out,
        )
    }
}

/// `out = W · x` for Q6_K weights that live inside a larger buffer, such as a pool slot.
pub fn matvec_view(
    device: &Device,
    weights: &cudarc::driver::CudaView<'_, u8>,
    rows: usize,
    cols: usize,
    x: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), ViperaError> {
    if x.len() != cols {
        return Err(ViperaError::invalid_argument(format!(
            "the activation row is {} values, the matrix has {cols} columns",
            x.len()
        )));
    }
    if out.len() != rows {
        return Err(ViperaError::internal(format!(
            "the output buffer is {} rows, the matrix has {rows}",
            out.len()
        )));
    }
    let blocks_per_row = cols / QK_K;
    if weights.len() < rows * blocks_per_row * BLOCK_BYTES {
        return Err(ViperaError::invalid_argument(
            "the weight view is smaller than the matrix",
        ));
    }
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: ((ROWS_PER_BLOCK * (row_bytes + 3)) & !3) as u32,
    };
    let rows_arg = rows as i32;
    let blocks = blocks_per_row as i32;
    let mut launch = device.stream.launch_builder(&device.q6k_matvec);
    launch.arg(weights);
    launch.arg(&x.slice);
    launch.arg(&mut out.slice);
    launch.arg(&rows_arg);
    launch.arg(&blocks);
    // SAFETY: the view is at least `rows*blocks_per_row*210` bytes (checked above), `x` is exactly
    // `cols` and `out` exactly `rows`; the kernel writes only `out[0..rows]`.
    unsafe { launch.launch(config) }
        .map_err(|error| ViperaError::internal(format!("Q6_K launch failed: {error}")))?;
    Ok(())
}
