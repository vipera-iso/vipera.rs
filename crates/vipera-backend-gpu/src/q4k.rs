//! The first kernel: a Q4_K matrix against one float activation vector.
//!
//! This is the same shape the CPU backend spends 68% of a prefill chunk on, so it is the one worth
//! porting first. It dequantizes in float rather than through the integer `dpbusd` path the CPU uses,
//! which is a deliberate first step: parity against the CPU's float reference is exact enough to test,
//! and the tensor cores and int8 activations can come once the plumbing is known to be right.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};

/// Values per Q4_K super-block.
pub const QK_K: usize = 256;

/// Bytes one super-block occupies: `d`, `dmin`, twelve packed scales, 128 packed quants.
pub const BLOCK_BYTES: usize = 144;

/// Rows one thread block handles: one warp each.
const ROWS_PER_BLOCK: usize = 8;

/// Threads per block: one warp per row.
const THREADS: usize = 32 * ROWS_PER_BLOCK;

const SOURCE: &str = include_str!("q4k_matvec.cu");

/// Compile the kernel for `context`'s device.
///
/// PTX for an older architecture runs on a newer one through the driver's JIT, so the exact
/// capability is an optimisation rather than a requirement; it is still asked for so a Turing card
/// does not pay for codegen aimed at Hopper.
pub(crate) fn compile(context: &cudarc::driver::CudaContext) -> Result<Ptx, String> {
    use cudarc::driver::sys::{CUdevice_attribute, CUdevice_attribute_enum};
    let major = context
        .attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                as CUdevice_attribute_enum,
        )
        .unwrap_or(7);
    let minor = context
        .attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                as CUdevice_attribute_enum,
        )
        .unwrap_or(5);
    let options = CompileOptions {
        // The arch has to go through `options` because the typed field only takes a `'static` str.
        options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
        ..Default::default()
    };
    compile_ptx_with_opts(SOURCE, options).map_err(|error| error.to_string())
}

/// A Q4_K matrix that lives on the device between calls.
///
/// The point of the tiered bus: a layer is copied up once, then every pass reads it on the card.
/// Uploading per call, which is what a naive port does, pays 7.7 MB over PCIe each time and measured
/// 4.4 GB/s against the CPU's 30.
pub struct Q4kMatrix {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    weights: DeviceBuffer<u8>,
}

impl Q4kMatrix {
    /// Copy a GGUF Q4_K tensor to the device.
    ///
    /// `weights` must be the tensor exactly as it sits in the file: `rows * (cols / 256)` super-blocks
    /// of [`BLOCK_BYTES`] bytes, row-major.
    pub fn upload(
        device: &Device,
        weights: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self, ViperaError> {
        if !cols.is_multiple_of(QK_K) {
            return Err(ViperaError::invalid_argument(format!(
                "Q4_K needs a column count that is a multiple of {QK_K}, got {cols}"
            )));
        }
        let blocks_per_row = cols / QK_K;
        let expected = rows * blocks_per_row * BLOCK_BYTES;
        if weights.len() != expected {
            return Err(ViperaError::invalid_argument(format!(
                "Q4_K weights are {} bytes, {rows}x{cols} needs {expected}",
                weights.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            blocks_per_row,
            weights: device.upload(weights)?,
        })
    }

    /// Rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes of weights held on the device.
    pub fn bytes(&self) -> usize {
        self.weights.len()
    }

    /// `y = W · x`, with the weights already resident.
    pub fn matvec(&self, device: &Device, x: &[f32]) -> Result<Vec<f32>, ViperaError> {
        if x.len() != self.cols {
            return Err(ViperaError::invalid_argument(format!(
                "the activation vector is {} values, the matrix has {} columns",
                x.len(),
                self.cols
            )));
        }
        let activations: DeviceBuffer<f32> = device.upload(x)?;
        let mut out: DeviceBuffer<f32> = device.alloc(self.rows)?;

        let config = LaunchConfig {
            grid_dim: (self.rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: (ROWS_PER_BLOCK * self.blocks_per_row * BLOCK_BYTES) as u32,
        };
        let rows_arg = self.rows as i32;
        let blocks_arg = self.blocks_per_row as i32;
        let mut launch = device.stream.launch_builder(&device.q4k_matvec);
        launch.arg(&self.weights.slice);
        launch.arg(&activations.slice);
        launch.arg(&mut out.slice);
        launch.arg(&rows_arg);
        launch.arg(&blocks_arg);
        // SAFETY: the buffers live on this device's context; the kernel reads
        // `rows*blocks_per_row*144` bytes from `self.weights` (its own allocation, sized at upload)
        // and `cols` floats from `activations` (checked above), and writes exactly `rows` floats into
        // `out`. The argument types match the kernel's signature.
        unsafe { launch.launch(config) }.map_err(|error| {
            ViperaError::internal(format!("Q4_K kernel launch failed: {error}"))
        })?;

        out.download(device)
    }
}
