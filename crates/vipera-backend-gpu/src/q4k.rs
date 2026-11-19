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
use crate::q8::{Q8, Q8Device};

/// Values per Q4_K super-block.
pub const QK_K: usize = 256;

/// Bytes one super-block occupies: `d`, `dmin`, twelve packed scales, 128 packed quants.
pub const BLOCK_BYTES: usize = 144;

/// Rows one thread block handles: one warp each.
const ROWS_PER_BLOCK: usize = 8;

/// Threads per block: one warp per row.
const THREADS: usize = 32 * ROWS_PER_BLOCK;

const SOURCE: &str = include_str!("q4k_matvec.cu");
const SOURCE_Q8: &str = include_str!("q4k_matvec_q8.cu");
const SOURCE_Q8_BATCH: &str = include_str!("q4k_matvec_q8_batch.cu");

/// The kernels this module compiles, one NVRTC program each: they share helper functions, so
/// concatenating the sources would redefine them.
pub(crate) fn sources() -> [&'static str; 4] {
    [SOURCE, SOURCE_Q8, crate::q6k::SOURCE, SOURCE_Q8_BATCH]
}

/// Names in the order [`sources`] returns them.
pub(crate) const FUNCTIONS: [&str; 4] = [
    "q4k_matvec",
    "q4k_matvec_q8",
    "q6k_matvec",
    "q4k_matvec_q8_batch",
];

/// Compile the kernel for `context`'s device.
///
/// PTX for an older architecture runs on a newer one through the driver's JIT, so the exact
/// capability is an optimisation rather than a requirement; it is still asked for so a Turing card
/// does not pay for codegen aimed at Hopper.
pub(crate) fn compile(context: &cudarc::driver::CudaContext, source: &str) -> Result<Ptx, String> {
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
    compile_ptx_with_opts(source, options).map_err(|error| error.to_string())
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

    /// A view of one row, for the embedding lookup.
    pub fn row_view(
        &self,
        row: usize,
        cols: usize,
    ) -> Result<cudarc::driver::CudaView<'_, u8>, ViperaError> {
        let stride = (cols / QK_K) * BLOCK_BYTES;
        let start = row * stride;
        if start + stride > self.weights.len() {
            return Err(ViperaError::invalid_argument("row outside the matrix"));
        }
        Ok(self.weights.slice.slice(start..start + stride))
    }

    /// A view of the whole matrix, for the free functions that take one.
    pub fn view(&self) -> cudarc::driver::CudaView<'_, u8> {
        self.weights.slice.slice(..)
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

    /// `y = W · x` on the int8 path.
    ///
    /// The weights stay as they are on the device; the activations are quantized to [`Q8`] by the
    /// caller (or with [`Q8::quantize`]) and meet the 4-bit quants in `__dp4a`, four products per
    /// instruction. This is the path that can beat the CPU's SIMD tile, because there a value costs a
    /// dequantize and a multiply while here it costs a quarter of one instruction.
    pub fn matvec_q8(&self, device: &Device, x: &Q8) -> Result<Vec<f32>, ViperaError> {
        if x.qs.len() != self.cols {
            return Err(ViperaError::invalid_argument(format!(
                "the activation vector is {} values, the matrix has {} columns",
                x.qs.len(),
                self.cols
            )));
        }
        if x.groups() != self.cols / 32 {
            return Err(ViperaError::invalid_argument(format!(
                "{} activation groups for {} columns",
                x.groups(),
                self.cols
            )));
        }
        let quants: DeviceBuffer<i8> = device.upload(&x.qs)?;
        let scales: DeviceBuffer<f32> = device.upload(&x.d)?;
        let sums: DeviceBuffer<f32> = device.upload(&x.s)?;
        let mut out: DeviceBuffer<f32> = device.alloc(self.rows)?;

        let config = LaunchConfig {
            grid_dim: (self.rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: (ROWS_PER_BLOCK * self.blocks_per_row * BLOCK_BYTES) as u32,
        };
        let rows_arg = self.rows as i32;
        let blocks_arg = self.blocks_per_row as i32;
        let mut launch = device.stream.launch_builder(&device.q4k_matvec_q8);
        launch.arg(&self.weights.slice);
        launch.arg(&quants.slice);
        launch.arg(&scales.slice);
        launch.arg(&sums.slice);
        launch.arg(&mut out.slice);
        launch.arg(&rows_arg);
        launch.arg(&blocks_arg);
        // SAFETY: same contract as `matvec`; the two extra buffers hold exactly `cols` quants and
        // `cols/32` scales and sums, which is what the kernel indexes.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("Q4_K int8 launch failed: {error}")))?;

        out.download(device)
    }

    /// `y = W · x` writing into a caller's buffer, so a layer's chain stays on the device.
    pub fn matvec_q8_into(
        &self,
        device: &Device,
        x: &Q8Device,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        matvec_q8_view(
            device,
            &self.weights.slice.slice(..),
            self.rows,
            self.cols,
            x,
            out,
        )
    }
}

/// `out = W · X` for `batch` activations at once: one weight read for the whole chunk.
///
/// `x` holds `batch * cols` quants and `batch * cols/32` scales and sums, entry-major; `out` holds
/// `batch * rows`, also entry-major.
pub fn matvec_q8_batch_view(
    device: &Device,
    weights: &cudarc::driver::CudaView<'_, u8>,
    rows: usize,
    cols: usize,
    x: &Q8Device,
    batch: usize,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), ViperaError> {
    if batch == 0 || batch > 8 {
        return Err(ViperaError::invalid_argument(format!(
            "the batch kernel takes 1 to 8 entries, got {batch}"
        )));
    }
    if out.len() < batch * rows {
        return Err(ViperaError::internal(format!(
            "the output buffer holds {} values, {batch}x{rows} needs {}",
            out.len(),
            batch * rows
        )));
    }
    let blocks_per_row = cols / QK_K;
    if weights.len() < rows * blocks_per_row * BLOCK_BYTES {
        return Err(ViperaError::invalid_argument(
            "the weight view is smaller than the matrix",
        ));
    }
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: (ROWS_PER_BLOCK * blocks_per_row * BLOCK_BYTES) as u32,
    };
    let rows_arg = rows as i32;
    let blocks_arg = blocks_per_row as i32;
    let batch_arg = batch as i32;
    let mut launch = device.stream.launch_builder(&device.q4k_matvec_q8_batch);
    launch.arg(weights);
    launch.arg(&x.qs.slice);
    launch.arg(&x.d.slice);
    launch.arg(&x.s.slice);
    launch.arg(&mut out.slice);
    launch.arg(&rows_arg);
    launch.arg(&blocks_arg);
    launch.arg(&batch_arg);
    // SAFETY: the activations hold `batch*cols` quants and `batch*cols/32` groups (the caller
    // quantized a `batch x cols` buffer), `weights` was checked, and `out` holds `batch*rows`.
    unsafe { launch.launch(config) }
        .map_err(|error| ViperaError::internal(format!("Q4_K batch launch failed: {error}")))?;
    Ok(())
}

/// `out = W · x` for a matrix that lives somewhere in a buffer — a pool slot, or an owned upload.
///
/// This is the shape a [`crate::pool::VramPool`] slot needs: the weights are a view into one big
/// per-layer allocation, so the kernel starts at `weights` and reads `rows * blocks * 144` bytes.
pub fn matvec_q8_view(
    device: &Device,
    weights: &cudarc::driver::CudaView<'_, u8>,
    rows: usize,
    cols: usize,
    x: &Q8Device,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), ViperaError> {
    if out.len() != rows {
        return Err(ViperaError::internal(format!(
            "the output buffer is {} rows, the matrix has {rows}",
            out.len()
        )));
    }
    let blocks_per_row = cols / QK_K;
    if weights.len() < rows * blocks_per_row * BLOCK_BYTES {
        return Err(ViperaError::invalid_argument(format!(
            "the weight view is {} bytes, {rows}x{cols} needs {}",
            weights.len(),
            rows * blocks_per_row * BLOCK_BYTES
        )));
    }
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: (ROWS_PER_BLOCK * blocks_per_row * BLOCK_BYTES) as u32,
    };
    let rows_arg = rows as i32;
    let blocks_arg = blocks_per_row as i32;
    let mut launch = device.stream.launch_builder(&device.q4k_matvec_q8);
    launch.arg(weights);
    launch.arg(&x.qs.slice);
    launch.arg(&x.d.slice);
    launch.arg(&x.s.slice);
    launch.arg(&mut out.slice);
    launch.arg(&rows_arg);
    launch.arg(&blocks_arg);
    // SAFETY: the activations hold at least `cols` quants and `cols/32` groups (checked by the
    // caller's `quantize_into`), `weights` was checked above, and `out` is exactly `rows`.
    unsafe { launch.launch(config) }
        .map_err(|error| ViperaError::internal(format!("Q4_K int8 launch failed: {error}")))?;
    Ok(())
}
