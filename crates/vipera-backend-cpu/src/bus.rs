//! Where a model's weights live, layer by layer.
//!
//! A model file is mapped read-only, so the bytes are served from the page cache and a layer nobody
//! has touched is not in memory at all — it is still on disk and faults in when the pass reaches it.
//! On top of that mapping the bus keeps a *window* of layers marked resident (`madvise(WILLNEED)`)
//! and hands the ones that fell behind back to the kernel (`madvise(DONTNEED)`), so a model larger
//! than RAM can run with only the working window in memory.
//!
//! Three tiers are named because the request that shaped this was "VRAM for the layers in use, RAM
//! as the reserve, SSD for the rest":
//!
//! | Tier | What it means here |
//! |---|---|
//! | [`Tier::Vram`] | Not available: the workspace has no GPU backend, so nothing can compute on weights copied there. Reported, never assigned. |
//! | [`Tier::Ram`] | Inside the resident window: pre-faulted, and the kernel is asked to keep it. |
//! | [`Tier::Ssd`] | Mapped but not resident: the pages come back from the file on demand. |
//!
//! The window defaults to the whole model, which is exactly the old behaviour of reading the file
//! into a heap buffer, minus the copy. `VIPERA_RAM_LAYERS=<n>` narrows it to `n` layers, which is the
//! switch that lets a model larger than RAM run: the pass walks layers in order, so the window
//! follows it.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vipera_core::vram::VramSink;

use vipera_core::gguf::GgufFile;
use vipera_errors::ViperaError;

/// Which memory holds a layer's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Device memory. Assigned only by a GPU backend, and there is none yet.
    Vram,
    /// Resident in RAM, pre-faulted ahead of the pass.
    Ram,
    /// Backed by the model file: read from disk when the pass gets there.
    Ssd,
}

impl Tier {
    /// The name used in the startup line and in the profile.
    pub fn name(self) -> &'static str {
        match self {
            Self::Vram => "vram",
            Self::Ram => "ram",
            Self::Ssd => "ssd",
        }
    }
}

/// The two `unsafe` calls mapping needs, in one place.
///
/// The crate denies `unsafe_code` and sanctions it in `simd.rs`; mapping a file and handing pages
/// back need the same opt-out, so it lives here rather than on the whole module (EXC-005).
#[allow(unsafe_code)]
mod mapped {
    use std::fs::File;

    /// Map the model file read-only.
    ///
    /// # Safety
    ///
    /// A mapping is undefined behaviour if the file is truncated or changed while it is mapped. The
    /// engine opens the model once, read-only, and treats it as immutable for the life of the process;
    /// writing a model file under a running engine has always been outside the contract.
    pub(super) fn open(handle: &File) -> std::io::Result<memmap2::Mmap> {
        // SAFETY: see above — the handle is read-only and the mapping outlives every borrow of it.
        unsafe { memmap2::Mmap::map(handle) }
    }

    /// Hand a range of a read-only mapping back to the kernel.
    ///
    /// `DONTNEED` on a file mapping drops the pages without losing data: the next read faults them
    /// in again from the file. That is why the advice is unchecked in `memmap2` and wrapping it is
    /// sound here — nothing writes through the mapping.
    pub(super) fn release(map: &memmap2::Mmap, start: usize, len: usize) {
        // SAFETY: the mapping is read-only and stays alive; dropping pages cannot change its bytes.
        let _ =
            unsafe { map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, len) };
    }
}

/// A byte range of the model file that belongs to one layer.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: usize,
    len: usize,
}

/// The weights of a model, either mapped from a file or already in memory.
///
/// [`WeightSource::from_bytes`] exists for callers that hand over bytes they already read (the
/// tokenizer path, the tests); [`WeightSource::open`] maps instead of copying.
pub struct WeightSource {
    backing: Backing,
    /// One entry per transformer layer, in file order.
    spans: Vec<Span>,
    /// Layers kept resident from the one in use. Zero means "the whole model".
    window: usize,
    /// Start of the window the last [`WeightSource::advance`] installed, so a layer is advised once.
    placed: AtomicUsize,
    /// Where the window's layers also go, when the process has a device to put them on.
    vram: Option<Arc<dyn VramSink>>,
}

enum Backing {
    Heap(Vec<u8>),
    Mapped(memmap2::Mmap),
}

/// How many layers stay resident, from `VIPERA_RAM_LAYERS`.
///
/// Unset or zero keeps the whole model resident, which is what a model that fits in RAM wants.
fn window_from_env() -> usize {
    std::env::var("VIPERA_RAM_LAYERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

impl WeightSource {
    /// Map `path` read-only and index its layers.
    ///
    /// The mapping *is* the file: no read, no copy, and the caller parses the GGUF header from it.
    ///
    /// A mapping is undefined behaviour if another process truncates the file mid-read, which is the
    /// contract every mmap-based inference engine takes; the model file is opened read-only here.
    pub fn open(path: &Path) -> Result<(Self, GgufFile), ViperaError> {
        let handle = std::fs::File::open(path)?;
        let map = mapped::open(&handle)?;
        let file = GgufFile::parse(&map)?;
        let source = Self {
            spans: layer_spans(&file),
            backing: Backing::Mapped(map),
            window: window_from_env(),
            placed: AtomicUsize::new(usize::MAX),
            vram: None,
        };
        source.report();
        Ok((source, file))
    }

    /// Build from bytes a caller already has.
    pub fn from_bytes(bytes: Vec<u8>, file: &GgufFile) -> Self {
        let source = Self {
            spans: layer_spans(file),
            backing: Backing::Heap(bytes),
            window: window_from_env(),
            placed: AtomicUsize::new(usize::MAX),
            vram: None,
        };
        source.report();
        source
    }

    /// The whole model, borrowed for a pass.
    pub fn bytes(&self) -> &[u8] {
        match &self.backing {
            Backing::Heap(bytes) => bytes,
            Backing::Mapped(map) => map,
        }
    }

    /// Number of transformer layers the bus indexed.
    pub fn layers(&self) -> usize {
        self.spans.len()
    }

    /// Whether the weights came from a file mapping.
    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, Backing::Mapped(_))
    }

    /// The tier a layer is in under the window the pass last installed.
    pub fn tier_of(&self, layer: usize) -> Tier {
        let start = self.placed.load(Ordering::Relaxed);
        if self.vram.is_some() && layer >= start && layer < start + self.window {
            return Tier::Vram;
        }
        if self.window == 0 || start == usize::MAX || layer < start + self.window {
            Tier::Ram
        } else {
            Tier::Ssd
        }
    }

    /// Give the bus a device to copy the window's layers to.
    ///
    /// Called once, before the first pass, by whoever opened the device. Without it the bus is the
    /// mmap-and-page-cache path it was before; with it, every layer that enters the window is also
    /// handed to the sink, and the ones that leave are released.
    pub fn install_vram(&mut self, sink: Arc<dyn VramSink>) {
        self.vram = Some(sink);
        // Nothing has been placed yet, so the first `advance` decides the window from scratch.
        self.placed.store(usize::MAX, Ordering::Relaxed);
    }

    /// Bytes the device holds, or zero when there is no device.
    pub fn vram_bytes(&self) -> usize {
        self.vram.as_ref().map_or(0, |sink| sink.held_bytes())
    }

    /// Layer `layer`'s bytes, for a sink that wants its own copy.
    fn layer_bytes(&self, layer: usize) -> Option<&[u8]> {
        let span = self.spans.get(layer)?;
        self.bytes().get(span.start..span.start + span.len)
    }

    /// Move the resident window so it covers `layer` and the `window` layers after it.
    ///
    /// The forward pass calls this once per layer. It is a no-op for a heap source or a
    /// full-model window — there is nothing to page out when the model fits — and cheap otherwise
    /// because it only advises when the window actually moved.
    pub fn advance(&self, layer: usize) {
        let Backing::Mapped(map) = &self.backing else {
            return;
        };
        if self.window == 0 {
            return;
        }
        let previous = self.placed.swap(layer, Ordering::Relaxed);
        if previous == layer {
            return;
        }
        let end = (layer + self.window).min(self.spans.len());
        for span in &self.spans[layer.min(end)..end] {
            let _ = map.advise_range(memmap2::Advice::WillNeed, span.start, span.len);
        }
        if let Some(sink) = &self.vram {
            // Two windows: the one that was resident and the one that replaces it. A layer is
            // brought when it is in the new one and was not in the old, released the other way
            // round. Both loops are over the window, not the model.
            let len = self.spans.len();
            let new_start = layer.min(len);
            let new_end = (layer + self.window).min(len);
            let old_start = previous.min(len);
            let old_end = if previous == usize::MAX {
                new_start
            } else {
                (previous + self.window).min(len)
            };
            for index in new_start..new_end {
                if (index < old_start || index >= old_end)
                    && let Some(bytes) = self.layer_bytes(index)
                {
                    sink.bring(index, bytes);
                }
            }
            for index in old_start..old_end {
                if index < new_start || index >= new_end {
                    sink.release(index);
                }
            }
        }
        // Only the layers that left the window are handed back: advising a layer the pass is about
        // to touch again would just cost a fault.
        if previous != usize::MAX {
            for span in &self.spans[previous.min(layer)..layer.min(self.spans.len())] {
                mapped::release(map, span.start, span.len);
            }
        }
    }

    /// Where the weights ended up, when `VIPERA_RESIDENCY=1`.
    fn report(&self) {
        if std::env::var_os("VIPERA_RESIDENCY").is_none() {
            return;
        }
        let ram = if self.window == 0 {
            self.spans.len()
        } else {
            self.window.min(self.spans.len())
        };
        eprintln!(
            "weights: {} layers, {} ram, {} ssd, {:.0} MB via {}, vram unavailable (no gpu backend)",
            self.spans.len(),
            ram,
            self.spans.len() - ram,
            self.bytes().len() as f64 / (1024.0 * 1024.0),
            if self.is_mapped() { "mmap" } else { "heap" },
        );
    }
}

/// The tier plan a window would produce, for the tests and for [`WeightSource::tier_of`].
pub fn plan(layers: usize, window: usize, layer: usize) -> Vec<Tier> {
    (0..layers)
        .map(|index| {
            if window == 0 || index < layer + window {
                Tier::Ram
            } else {
                Tier::Ssd
            }
        })
        .collect()
}

/// Byte ranges of each layer's tensors, from the file's tensor table.
///
/// A layer owns every tensor whose name starts with `blk.{n}.`; the embeddings and the output
/// projection belong to no layer. A layer's span is the hull of its tensors, which for the shapes
/// GGUF writes is contiguous.
fn layer_spans(file: &GgufFile) -> Vec<Span> {
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let data = file.data_offset as usize;
    for tensor in &file.tensors {
        let Some(rest) = tensor.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((index, _)) = rest.split_once('.') else {
            continue;
        };
        let (Ok(index), Some(size)) = (index.parse::<usize>(), tensor.byte_len()) else {
            continue;
        };
        let start = data + tensor.offset as usize;
        let end = start + size as usize;
        if bounds.len() <= index {
            bounds.resize(index + 1, (usize::MAX, 0));
        }
        let entry = &mut bounds[index];
        entry.0 = entry.0.min(start);
        entry.1 = entry.1.max(end);
    }
    bounds
        .into_iter()
        .filter(|(start, end)| *start != usize::MAX && end > start)
        .map(|(start, end)| Span {
            start,
            len: end - start,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_window_is_ram_and_a_narrow_one_leaves_the_tail_on_disk() {
        assert_eq!(plan(4, 0, 2), vec![Tier::Ram; 4]);
        assert_eq!(
            plan(4, 2, 1),
            vec![Tier::Ram, Tier::Ram, Tier::Ram, Tier::Ssd]
        );
        assert_eq!(plan(3, 1, 2), vec![Tier::Ram, Tier::Ram, Tier::Ram]);
    }

    #[test]
    fn a_heap_source_keeps_its_bytes_and_advancing_costs_nothing() {
        let bytes = vec![7u8; 4096];
        let source = WeightSource {
            backing: Backing::Heap(bytes.clone()),
            spans: Vec::new(),
            window: 1,
            placed: AtomicUsize::new(usize::MAX),
            vram: None,
        };
        source.advance(0);
        assert_eq!(source.bytes(), &bytes[..]);
        assert!(!source.is_mapped());
        assert_eq!(source.tier_of(0), Tier::Ram);
    }
}
