//! Device memory for a window of layers, driven by the weight bus.
//!
//! [`VramPool`] implements [`vipera_core::vram::VramSink`], so `WeightSource` in the CPU backend can
//! hand it the layers that enter its resident window without either crate knowing the other's
//! internals. The pool keeps one buffer per layer and nothing else: a layer that leaves the window is
//! freed, and the next pass that needs it is uploaded again.
//!
//! What this is *not*: it is not yet how the CPU engine computes. The kernels in this crate operate on
//! [`crate::q4k::Q4kMatrix`] handles; making the engine's matvecs read from the pool is the step after
//! this one, and until then the pool measures only what a window costs to hold.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use vipera_core::vram::VramSink;
use vipera_errors::ViperaError;

use crate::device::{Device, DeviceBuffer};

/// Layers resident on the device, by index.
pub struct VramPool {
    device: Device,
    layers: Mutex<HashMap<usize, Arc<DeviceBuffer<u8>>>>,
    /// Kept for the startup line and the tests: the largest the pool has ever been.
    peak: Mutex<usize>,
}

impl VramPool {
    /// Make a pool on `device`.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            layers: Mutex::new(HashMap::new()),
            peak: Mutex::new(0),
        }
    }

    /// The device this pool copies to.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether a layer is currently held.
    pub fn holds(&self, layer: usize) -> bool {
        self.layers
            .lock()
            .map(|layers| layers.contains_key(&layer))
            .unwrap_or(false)
    }

    /// The most bytes the pool has held at once.
    pub fn peak_bytes(&self) -> usize {
        self.peak.lock().map(|peak| *peak).unwrap_or(0)
    }

    /// The buffer holding `layer`, shared so a matrix can outlive the pool's own lock.
    ///
    /// A [`PooledMatrix`] keeps this `Arc`, so a layer that the bus releases mid-pass cannot free
    /// memory out from under a kernel.
    pub fn layer_buffer(&self, layer: usize) -> Option<Arc<DeviceBuffer<u8>>> {
        let layers = self.layers.lock().ok()?;
        layers.get(&layer).map(Arc::clone)
    }

    /// A Q4_K or Q6_K tensor that starts `offset` bytes into `layer`'s buffer.
    pub fn matrix(
        &self,
        layer: usize,
        offset: usize,
        rows: usize,
        cols: usize,
        format: Format,
    ) -> Result<PooledMatrix, ViperaError> {
        let buffer = self
            .layer_buffer(layer)
            .ok_or_else(|| ViperaError::internal(format!("layer {layer} is not on the device")))?;
        PooledMatrix::new(buffer, offset, rows, cols, format)
    }
}

/// Which packed format a pooled tensor is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Q4_K, fed int8 activations.
    Q4,
    /// Q6_K, fed float activations.
    Q6,
}

/// A tensor inside a pool slot: a view, not a copy.
///
/// The point of the pool is that a layer is copied up once; a matrix made this way reads straight out
/// of that copy, so a model that fits the pool never uploads a weight twice.
pub struct PooledMatrix {
    buffer: Arc<DeviceBuffer<u8>>,
    offset: usize,
    len: usize,
    rows: usize,
    cols: usize,
    format: Format,
}

impl PooledMatrix {
    /// Describe the tensor at `offset` and check that the slot is long enough for it.
    pub fn new(
        buffer: Arc<DeviceBuffer<u8>>,
        offset: usize,
        rows: usize,
        cols: usize,
        format: Format,
    ) -> Result<Self, ViperaError> {
        let block = match format {
            Format::Q4 => crate::q4k::BLOCK_BYTES,
            Format::Q6 => crate::q6k::BLOCK_BYTES,
        };
        let len = rows * (cols / 256) * block;
        if offset + len > buffer.len() {
            return Err(ViperaError::invalid_argument(format!(
                "a {rows}x{cols} tensor at {offset} needs {} bytes, the layer holds {}",
                len,
                buffer.len()
            )));
        }
        Ok(Self {
            buffer,
            offset,
            len,
            rows,
            cols,
            format,
        })
    }

    /// Rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Bytes the tensor occupies inside the slot.
    pub fn bytes(&self) -> usize {
        self.len
    }

    /// Which format the weights are in.
    pub fn format(&self) -> Format {
        self.format
    }

    /// `out = W · x`, quantizing `x` first when the format wants int8.
    pub fn project(
        &self,
        device: &Device,
        input: &DeviceBuffer<f32>,
        q8: &mut crate::q8::Q8Device,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), ViperaError> {
        let view = self.buffer.slice.slice(self.offset..self.offset + self.len);
        match self.format {
            Format::Q4 => {
                device.quantize_into(input, q8, input.len() / 32)?;
                crate::q4k::matvec_q8_view(device, &view, self.rows, self.cols, q8, out)
            }
            Format::Q6 => crate::q6k::matvec_view(device, &view, self.rows, self.cols, input, out),
        }
    }
}

impl VramSink for VramPool {
    fn bring(&self, layer: usize, bytes: &[u8]) {
        let Ok(mut layers) = self.layers.lock() else {
            return;
        };
        if layers.contains_key(&layer) {
            return;
        }
        let Ok(buffer) = self.device.upload(bytes).map(Arc::new) else {
            // Out of memory or a lost device: the bus keeps its mapping, so the layer is still
            // readable and the pass merely loses the speed-up.
            return;
        };
        layers.insert(layer, buffer);
        if let Ok(mut peak) = self.peak.lock() {
            let held: usize = layers.values().map(|buffer| buffer.len()).sum();
            *peak = (*peak).max(held);
        }
    }

    fn release(&self, layer: usize) {
        if let Ok(mut layers) = self.layers.lock() {
            layers.remove(&layer);
        }
    }

    fn held_bytes(&self) -> usize {
        self.layers
            .lock()
            .map(|layers| layers.values().map(|buffer| buffer.len()).sum())
            .unwrap_or(0)
    }
}
