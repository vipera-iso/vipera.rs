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

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
use vipera_errors::ViperaError;

use crate::q4k;

/// A CUDA context with the backend's kernels already compiled into it.
#[derive(Clone)]
pub struct Device {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    /// `q4k_matvec`, loaded once so a call does not walk the module's symbol table.
    pub(crate) q4k_matvec: Arc<CudaFunction>,
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
        let ptx = q4k::compile(&context).map_err(|error| {
            ViperaError::unsupported(format!("the Q4_K kernel did not compile: {error}"))
        })?;
        let module = context.load_module(ptx).map_err(|error| {
            ViperaError::unsupported(format!("the Q4_K module did not load: {error}"))
        })?;
        let q4k_matvec = module.load_function("q4k_matvec").map_err(|error| {
            ViperaError::internal(format!("the module has no q4k_matvec: {error}"))
        })?;
        Ok(Self {
            context,
            stream,
            q4k_matvec: q4k_matvec.into(),
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
