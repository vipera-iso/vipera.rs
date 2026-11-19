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
use std::sync::Mutex;

use vipera_core::vram::VramSink;

use crate::device::{Device, DeviceBuffer};

/// Layers resident on the device, by index.
pub struct VramPool {
    device: Device,
    layers: Mutex<HashMap<usize, DeviceBuffer<u8>>>,
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
}

impl VramSink for VramPool {
    fn bring(&self, layer: usize, bytes: &[u8]) {
        let Ok(mut layers) = self.layers.lock() else {
            return;
        };
        if layers.contains_key(&layer) {
            return;
        }
        let Ok(buffer) = self.device.upload(bytes) else {
            // Out of memory or a lost device: the bus keeps its mapping, so the layer is still
            // readable and the pass merely loses the speed-up.
            return;
        };
        layers.insert(layer, buffer);
        if let Ok(mut peak) = self.peak.lock() {
            let held: usize = layers.values().map(DeviceBuffer::len).sum();
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
            .map(|layers| layers.values().map(DeviceBuffer::len).sum())
            .unwrap_or(0)
    }
}
