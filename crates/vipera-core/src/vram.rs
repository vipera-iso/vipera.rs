//! Where a layer's bytes go when a GPU is present.
//!
//! The weight bus lives in the CPU backend and knows nothing about CUDA; the GPU backend knows
//! nothing about GGUF offsets. This trait is the join: the bus says "layer `n` entered the window"
//! and hands over the bytes, a backend that has device memory copies them up, and a backend that does
//! not never installs a sink at all.
//!
//! Implementations must be callable from the thread that owns the forward pass and must not block
//! longer than a copy: a step calls [`VramSink::bring`] once per layer that enters the window.

/// A pool that can hold layers on the device.
pub trait VramSink: Send + Sync {
    /// Take layer `n`'s bytes. Called when the layer enters the resident window.
    ///
    /// Returning nothing is deliberate: a copy that fails (out of memory, device lost) must leave the
    /// layer where it was, which the bus already does by keeping the mapping.
    fn bring(&self, layer: usize, bytes: &[u8]);

    /// Drop layer `n` from the device. Called when the layer leaves the window.
    fn release(&self, layer: usize);

    /// Bytes currently held on the device, for the startup line.
    fn held_bytes(&self) -> usize;
}
