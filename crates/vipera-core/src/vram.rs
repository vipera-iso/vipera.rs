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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// Mock VramSink that tracks calls for testing edge cases.
    ///
    /// Uses atomic types so the struct is `Send + Sync`, matching the `VramSink` bound.
    struct TestVramSink {
        /// Bytes held across all layers, tracked via atomic.
        held: AtomicUsize,
        /// Whether bring was called with empty bytes.
        brought_empty: AtomicBool,
        /// Whether bring was called with oversized bytes (> 1 MiB).
        brought_oversized: AtomicBool,
    }

    impl TestVramSink {
        fn new() -> Self {
            Self {
                held: AtomicUsize::new(0),
                brought_empty: AtomicBool::new(false),
                brought_oversized: AtomicBool::new(false),
            }
        }
    }

    impl VramSink for TestVramSink {
        fn bring(&self, _layer: usize, bytes: &[u8]) {
            if bytes.is_empty() {
                self.brought_empty
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            if bytes.len() > 1024 * 1024 {
                self.brought_oversized
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            // Silently ignore: out-of-memory is handled by the bus keeping the mapping.
            // We do not update `held` here; tests that need exact counts use a different sink.
        }

        fn release(&self, _layer: usize) {
            // No-op for tests.
        }

        fn held_bytes(&self) -> usize {
            self.held.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[test]
    fn bring_with_empty_bytes_is_noop() {
        let sink = TestVramSink::new();
        sink.bring(0, &[]);
        assert!(sink.brought_empty.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn bring_with_oversized_bytes_is_noop() {
        let sink = TestVramSink::new();
        let big = vec![0u8; 1024 * 1024 + 1];
        sink.bring(0, &big);
        assert!(
            sink.brought_oversized
                .load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    #[test]
    fn held_bytes_returns_zero_when_no_layers_held() {
        let sink = TestVramSink::new();
        assert_eq!(sink.held_bytes(), 0);
    }

    #[test]
    fn held_bytes_reflects_tracking() {
        let sink = TestVramSink::new();
        // held_bytes starts at 0
        assert_eq!(sink.held_bytes(), 0);
        // Simulate holding some bytes via the atomic
        sink.held.store(2048, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(sink.held_bytes(), 2048);
    }

    #[test]
    fn release_is_noop_when_layer_not_held() {
        let sink = TestVramSink::new();
        // Releasing a layer that was never brought should not panic.
        sink.release(42);
    }
}
