//! The device and its allocator.
//!
//! [`Device`] owns the CUDA context, the stream every copy and launch runs on, and the compiled
//! module. [`DeviceBuffer`] is a typed allocation that knows how to come back to the host.
//!
//! Contract this module upholds (the reason its `unsafe` calls are sound): a buffer is allocated on
//! the device of the context that made it, is only ever passed to kernels loaded from that context's
//! module, and is not touched from the host except through the copy methods below, which synchronise
//! the stream first.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use vipera_errors::ViperaError;

use crate::q4k;

/// A CUDA context with the backend's kernels already compiled into it.
#[derive(Clone)]
pub struct Device {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    /// `q4k_matvec`, loaded once so a call does not walk the module's symbol table.
    pub(crate) q4k_matvec: Arc<CudaFunction>,
    /// `q4k_matvec_q8`, the int8 activation path.
    pub(crate) q4k_matvec_q8: Arc<CudaFunction>,
    /// `q6k_matvec`, for the tensors a Q4_K_M file keeps at 6 bits.
    pub(crate) q6k_matvec: Arc<CudaFunction>,
    /// `q4k_matvec_q8_batch`, the prefill path: one weight read for up to eight tokens.
    pub(crate) q4k_matvec_q8_batch: Arc<CudaFunction>,
    /// The elementwise and attention kernels that make a whole layer possible.
    pub(crate) ops: Ops,
}

/// One handle per kernel in `layer.cu`.
#[derive(Clone)]
pub(crate) struct Ops {
    pub(crate) rmsnorm: Arc<CudaFunction>,
    pub(crate) rope: Arc<CudaFunction>,
    pub(crate) attention_decode: Arc<CudaFunction>,
    pub(crate) swiglu: Arc<CudaFunction>,
    pub(crate) add_inplace: Arc<CudaFunction>,
    pub(crate) quantize_q8: Arc<CudaFunction>,
    pub(crate) q4k_row: Arc<CudaFunction>,
}

impl Device {
    /// Open device `ordinal` (0 is the only one this backend picks) and compile the kernels.
    ///
    /// Returns an error rather than panicking when the driver is missing, so a build with the GPU
    /// backend compiled in still runs on a machine without one.
    pub fn open(ordinal: usize) -> Result<Self, ViperaError> {
        let context = CudaContext::new(ordinal).map_err(|error| {
            ViperaError::unsupported(format!("no CUDA device {ordinal}: {error}"))
        })?;
        let stream = context.default_stream();
        let mut loaded = Vec::new();
        for (index, source) in q4k::sources().iter().enumerate() {
            let name = q4k::FUNCTIONS[index];
            let ptx = q4k::compile(&context, source).map_err(|error| {
                ViperaError::unsupported(format!("the {name} kernel did not compile: {error}"))
            })?;
            let module = context.load_module(ptx).map_err(|error| {
                ViperaError::unsupported(format!("the {name} module did not load: {error}"))
            })?;
            let function = module.load_function(name).map_err(|error| {
                ViperaError::internal(format!("the module has no {name}: {error}"))
            })?;
            loaded.push(Arc::new(function));
        }
        let batch = loaded.pop().expect("kernels compiled");
        let q6k = loaded.pop().expect("kernels compiled");
        let int8 = loaded.pop().expect("kernels compiled");
        let float = loaded.pop().expect("kernels compiled");

        let ptx = q4k::compile(&context, crate::layer::SOURCE).map_err(|error| {
            ViperaError::unsupported(format!("the layer kernels did not compile: {error}"))
        })?;
        let module = context.load_module(ptx).map_err(|error| {
            ViperaError::unsupported(format!("the layer module did not load: {error}"))
        })?;
        let op = |name: &str| -> Result<Arc<CudaFunction>, ViperaError> {
            module
                .load_function(name)
                .map(Arc::new)
                .map_err(|error| ViperaError::internal(format!("no {name}: {error}")))
        };
        let ops = Ops {
            rmsnorm: op("rmsnorm")?,
            rope: op("rope")?,
            attention_decode: op("attention_decode")?,
            swiglu: op("swiglu")?,
            add_inplace: op("add_inplace")?,
            quantize_q8: op("quantize_q8")?,
            q4k_row: op("q4k_row")?,
        };
        Ok(Self {
            context,
            stream,
            q4k_matvec: float,
            q4k_matvec_q8: int8,
            q6k_matvec: q6k,
            q4k_matvec_q8_batch: batch,
            ops,
        })
    }

    /// The name the driver reports for this device, for the startup line.
    pub fn name(&self) -> Result<String, ViperaError> {
        self.context
            .name()
            .map_err(|error| ViperaError::internal(format!("the device has no name: {error}")))
    }

    /// Bytes free and total, as the driver sees them.
    pub fn memory(&self) -> Result<(usize, usize), ViperaError> {
        self.context
            .mem_get_info()
            .map_err(|error| ViperaError::internal(format!("no memory info: {error}")))
    }

    /// Allocate `len` zeroed values on the device.
    pub fn alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>, ViperaError> {
        let slice = self
            .stream
            .alloc_zeros::<T>(len)
            .map_err(|error| ViperaError::internal(format!("device allocation failed: {error}")))?;
        Ok(DeviceBuffer { slice })
    }

    /// Copy `values` to a fresh device buffer.
    pub fn upload<T: cudarc::driver::DeviceRepr>(
        &self,
        values: &[T],
    ) -> Result<DeviceBuffer<T>, ViperaError> {
        let slice = self.stream.clone_htod(values).map_err(|error| {
            ViperaError::internal(format!("copy to the device failed: {error}"))
        })?;
        Ok(DeviceBuffer { slice })
    }
}

impl Device {
    /// Copy a host row into an existing device buffer.
    pub fn copy_into<T: cudarc::driver::DeviceRepr>(
        &self,
        host: &[T],
        dst: &mut DeviceBuffer<T>,
    ) -> Result<(), ViperaError> {
        if host.len() != dst.len() {
            return Err(ViperaError::invalid_argument(format!(
                "copy of {} values into {} slots",
                host.len(),
                dst.len()
            )));
        }
        self.stream
            .memcpy_htod(host, &mut dst.slice)
            .map_err(|error| ViperaError::internal(format!("copy to the device failed: {error}")))
    }

    /// Copy one device buffer into another of the same length.
    pub fn copy_buffer(
        &self,
        src: &DeviceBuffer<f32>,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        if src.len() != dst.len() {
            return Err(ViperaError::internal("a buffer copy changed length"));
        }
        self.stream
            .memcpy_dtod(&src.slice, &mut dst.slice)
            .map_err(|error| ViperaError::internal(format!("device copy failed: {error}")))
    }

    /// `rms_norm` for one row.
    pub fn rmsnorm(
        &self,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        epsilon: f32,
    ) -> Result<(), ViperaError> {
        let n = x.len() as i32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let mut launch = self.stream.launch_builder(&self.ops.rmsnorm);
        launch.arg(&x.slice);
        launch.arg(&weight.slice);
        launch.arg(&mut out.slice);
        launch.arg(&n);
        launch.arg(&epsilon);
        // SAFETY: one block reduces `n` squares, then writes `n` values on every thread; buffers are
        // all `n` long (the caller sizes them from the model's hidden width).
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("rmsnorm launch failed: {error}")))?;
        Ok(())
    }

    /// `rope_with` over `heads` heads of `head_dim`, in place.
    pub fn rope(
        &self,
        vector: &mut DeviceBuffer<f32>,
        position: usize,
        heads: usize,
        head_dim: usize,
        base: f32,
    ) -> Result<(), ViperaError> {
        let config = LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: ((head_dim / 2).min(128) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let position = position as i32;
        let heads = heads as i32;
        let head_dim = head_dim as i32;
        let mut launch = self.stream.launch_builder(&self.ops.rope);
        launch.arg(&mut vector.slice);
        launch.arg(&position);
        launch.arg(&heads);
        launch.arg(&head_dim);
        launch.arg(&base);
        // SAFETY: one block per head, each thread touching two values inside its own head.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("rope launch failed: {error}")))?;
        Ok(())
    }

    /// Put the new K and V rows into the cache slot for `position`.
    pub fn write_kv(
        &self,
        position: usize,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        k_cache: &mut DeviceBuffer<f32>,
        v_cache: &mut DeviceBuffer<f32>,
        width: usize,
    ) -> Result<(), ViperaError> {
        let src = position * width;
        if src + width > k_cache.len() {
            return Err(ViperaError::invalid_argument(
                "the KV cache has no slot for this position",
            ));
        }
        self.stream
            .memcpy_dtod(&k.slice, &mut k_cache.slice.slice_mut(src..src + width))
            .map_err(|error| ViperaError::internal(format!("K write failed: {error}")))?;
        self.stream
            .memcpy_dtod(&v.slice, &mut v_cache.slice.slice_mut(src..src + width))
            .map_err(|error| ViperaError::internal(format!("V write failed: {error}")))
    }

    /// Causal grouped-query attention over `positions` cached rows.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &DeviceBuffer<f32>,
        k_cache: &DeviceBuffer<f32>,
        v_cache: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        positions: usize,
        shape: crate::layer::LayerShape,
        scale: f32,
    ) -> Result<(), ViperaError> {
        let config = LaunchConfig {
            grid_dim: (shape.heads as u32, 1, 1),
            block_dim: (shape.head_dim as u32, 1, 1),
            // The scores, then one reduction slot per thread: the softmax pass writes
            // `partials[tid]` for every thread in the block, not only the warps.
            shared_mem_bytes: ((positions + shape.head_dim) * 4) as u32,
        };
        let positions = positions as i32;
        let heads = shape.heads as i32;
        let kv_heads = shape.kv_heads as i32;
        let head_dim = shape.head_dim as i32;
        let mut launch = self.stream.launch_builder(&self.ops.attention_decode);
        launch.arg(&q.slice);
        launch.arg(&k_cache.slice);
        launch.arg(&v_cache.slice);
        launch.arg(&mut out.slice);
        launch.arg(&positions);
        launch.arg(&heads);
        launch.arg(&kv_heads);
        launch.arg(&head_dim);
        launch.arg(&scale);
        // SAFETY: one block per head, `head_dim` threads; the cache holds `max_positions` rows of
        // `kv_heads*head_dim` and `positions` is checked against it by the caller's slot check.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("attention launch failed: {error}")))?;
        Ok(())
    }

    /// Dequantize one Q4_K row into `out`.
    pub fn q4k_row(
        &self,
        row: &cudarc::driver::CudaView<'_, u8>,
        cols: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        if row.len() < (cols / 256) * crate::q4k::BLOCK_BYTES || out.len() < cols {
            return Err(ViperaError::invalid_argument(
                "the row view or the output is too small for the width",
            ));
        }
        let config = LaunchConfig {
            grid_dim: ((cols / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cols_arg = cols as i32;
        let mut launch = self.stream.launch_builder(&self.ops.q4k_row);
        launch.arg(row);
        launch.arg(&mut out.slice);
        launch.arg(&cols_arg);
        // SAFETY: one block per 256 values; the view holds `cols/256*144` bytes (checked) and the
        // kernel writes `cols` floats into a buffer at least that long.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("q4k_row launch failed: {error}")))?;
        Ok(())
    }

    /// `swiglu` over `n` values.
    pub fn swiglu(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), ViperaError> {
        self.elementwise(&self.ops.swiglu, &[&gate.slice, &up.slice], out, n)
    }

    /// `a += b` over `n` values.
    pub fn add_inplace(
        &self,
        a: &mut DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), ViperaError> {
        let config = LaunchConfig::for_num_elems(n as u32);
        let n_arg = n as i32;
        let mut launch = self.stream.launch_builder(&self.ops.add_inplace);
        launch.arg(&mut a.slice);
        launch.arg(&b.slice);
        launch.arg(&n_arg);
        // SAFETY: both buffers are at least `n` long and the kernel touches only `[0, n)`.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("add launch failed: {error}")))?;
        Ok(())
    }

    fn elementwise(
        &self,
        function: &std::sync::Arc<cudarc::driver::CudaFunction>,
        inputs: &[&cudarc::driver::CudaSlice<f32>],
        out: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), ViperaError> {
        let config = LaunchConfig::for_num_elems(n as u32);
        let n_arg = n as i32;
        let mut launch = self.stream.launch_builder(function);
        for input in inputs {
            launch.arg(*input);
        }
        launch.arg(&mut out.slice);
        launch.arg(&n_arg);
        // SAFETY: every buffer is at least `n` long and the kernel touches only `[0, n)`.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("kernel launch failed: {error}")))?;
        Ok(())
    }

    /// Quantize `x` into the activation arrays, on the device.
    pub fn quantize_into(
        &self,
        x: &DeviceBuffer<f32>,
        q8: &mut crate::q8::Q8Device,
        groups: usize,
    ) -> Result<(), ViperaError> {
        if groups * 32 > x.len() {
            return Err(ViperaError::internal(
                "quantizing more values than the buffer holds",
            ));
        }
        self.launch_quantize(x, &mut q8.qs, &mut q8.d, &mut q8.s, groups)
    }

    /// Quantize `x` into the three activation arrays, on the device.
    pub(crate) fn launch_quantize(
        &self,
        x: &DeviceBuffer<f32>,
        qs: &mut DeviceBuffer<i8>,
        d: &mut DeviceBuffer<f32>,
        s: &mut DeviceBuffer<f32>,
        groups: usize,
    ) -> Result<(), ViperaError> {
        use cudarc::driver::PushKernelArg;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (groups as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let groups_arg = groups as i32;
        let mut launch = self.stream.launch_builder(&self.ops.quantize_q8);
        launch.arg(&x.slice);
        launch.arg(&mut qs.slice);
        launch.arg(&mut d.slice);
        launch.arg(&mut s.slice);
        launch.arg(&groups_arg);
        // SAFETY: one block per group of 32; the kernel reads `groups*32` floats and writes
        // `groups*32` quants plus one scale and one sum per group, which is the size of each buffer.
        unsafe { launch.launch(config) }
            .map_err(|error| ViperaError::internal(format!("quantize launch failed: {error}")))?;
        Ok(())
    }
}

/// A typed allocation on the device.
pub struct DeviceBuffer<T> {
    pub(crate) slice: CudaSlice<T>,
}

impl<T: cudarc::driver::DeviceRepr> DeviceBuffer<T> {
    /// Number of values held.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

impl<T: cudarc::driver::DeviceRepr + Clone> DeviceBuffer<T> {
    /// Copy the buffer back to the host.
    pub fn download(&self, device: &Device) -> Result<Vec<T>, ViperaError> {
        device
            .stream
            .clone_dtoh(&self.slice)
            .map_err(|error| ViperaError::internal(format!("copy to the host failed: {error}")))
    }
}
