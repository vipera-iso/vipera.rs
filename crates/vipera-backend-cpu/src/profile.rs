//! Per-phase timing of a forward pass, off unless `VIPERA_PROFILE_STEP` is set.
//!
//! A decode step is one pass over every layer, and its cost splits into phases that are fixed by
//! the model: the Q4_K matrix products, the Q6_K output projection, attention, and the normalisations
//! between them. This module accumulates the nanosecond totals of those phases and prints them once
//! per step, so a claim about where a step's time goes can be checked instead of argued.
//!
//! The timers are `std::time::Instant`s around whole phase calls, so the overhead is two clock reads
//! per call: under 200 per step, against a step measured in tens of milliseconds. Nothing is
//! recorded when the switch is off, which is every run that does not set it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The phases a forward pass is split into. `Other` is the remainder: rope, the residual adds, the
/// copy into the output buffer, and anything else not named.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Q4_K matrix products: attention projections and the feed-forward block.
    Q4K,
    /// Q6_K matrix products, which is the output projection in this model.
    Q6K,
    /// Every other weight format, which this model does not use.
    OtherDot,
    /// Causal attention over the positions so far, including the KV cache reads.
    Attention,
    /// RMS normalisation, including the rotary embeddings applied next to it.
    Norm,
}

/// The counters, one per [`Phase`].
static COUNTERS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Whole-pass time, so the phases can be reported as a share of it.
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// How many passes have been timed, and therefore how many lines were printed.
static STEPS: AtomicU64 = AtomicU64::new(0);
/// The same two counters for prompt chunks, which are counted apart from decode steps: a chunk
/// computes several positions per weight read, so mixing them into one average describes neither.
static PREFILL_COUNTERS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static PREFILL_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFILL_STEPS: AtomicU64 = AtomicU64::new(0);

/// Whether the phase timers are on.
pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VIPERA_PROFILE_STEP").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    })
}

// Whether the pass being timed is a prompt chunk, on the thread that started it.
thread_local! {
    static IN_CHUNK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Start a whole pass, or return `None` when the timers are off so the caller does no work.
///
/// `prefill` says whether this pass is a prompt chunk, which keeps the chunk and decode averages from
/// describing neither. The phases of the pass are counted against the same answer.
pub fn begin(prefill: bool) -> Option<Instant> {
    if !enabled() {
        return None;
    }
    IN_CHUNK.with(|flag| flag.set(prefill));
    Some(Instant::now())
}

/// Start a phase of the current pass, or return `None` when the timers are off.
pub fn stamp() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Add one finished phase to its counter.
pub fn note(phase: Phase, start: Option<Instant>) {
    if let Some(start) = start {
        let index = phase as usize;
        let nanos = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let counters = if IN_CHUNK.with(|flag| flag.get()) {
            &PREFILL_COUNTERS
        } else {
            &COUNTERS
        };
        counters[index].fetch_add(nanos, Ordering::Relaxed);
    }
}

/// One shape's totals, for the per-call breakdown printed once a decode run is warm.
struct Shape {
    label: &'static str,
    batch: usize,
    rows: usize,
    cols: usize,
    calls: u64,
    nanos: u64,
}

/// Every shape seen so far, in first-seen order. A mutex is fine here: this only runs when the
/// phase timers are on, which is a debugging switch, and the lock is uncontended.
static SHAPES: std::sync::Mutex<Vec<Shape>> = std::sync::Mutex::new(Vec::new());

/// Add one matvec call to its shape's total.
///
/// `label` is the weight format (`Q4_K`, ...), `batch` is the number of activations the call
/// carried (1 for a decode step), and `rows` × `cols` is the tensor. The report is per pass: the
/// counters are divided by the number of decode passes when printed, so a call's cost is comparable
/// with the `rest` line of the phase split.
pub fn record_shape(
    label: &'static str,
    batch: usize,
    rows: usize,
    cols: usize,
    elapsed: std::time::Duration,
) {
    let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
    let mut shapes = SHAPES.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(slot) = shapes.iter_mut().find(|slot| {
        slot.label == label && slot.batch == batch && slot.rows == rows && slot.cols == cols
    }) {
        slot.calls += 1;
        slot.nanos += nanos;
    } else if shapes.len() < 32 {
        shapes.push(Shape {
            label,
            batch,
            rows,
            cols,
            calls: 1,
            nanos,
        });
    }
}

/// Bits per weight for the formats this backend runs, so the table can report a rate.
fn bits_per_weight(label: &str) -> f64 {
    match label {
        "Q4_K" => 4.5,
        "Q6_K" => 6.5625,
        "Q8_0" => 8.5,
        "F32" => 32.0,
        "F16" | "Bf16" => 16.0,
        _ => 4.5,
    }
}

/// Print the per-shape table, biggest total first, as an average over the decode passes timed.
fn print_shapes(steps: u64) {
    let shapes = SHAPES.lock().unwrap_or_else(|error| error.into_inner());
    if shapes.is_empty() {
        return;
    }
    let mut ordered: Vec<&Shape> = shapes.iter().collect();
    ordered.sort_by_key(|shape| std::cmp::Reverse(shape.nanos));
    eprintln!("SHAPES over {steps} decode passes (ms per pass, ms per call, GB/s):");
    for shape in ordered {
        if steps == 0 {
            continue;
        }
        let bytes = shape.rows as f64 * shape.cols as f64 * bits_per_weight(shape.label) / 8.0;
        let per_call = shape.nanos as f64 / shape.calls as f64 / 1e6;
        if shape.batch > 1 {
            // A chunk pass carries `batch` activations, so its cost is compute per activation, not
            // a stream of this tensor: per-pass and GB/s would both describe nothing.
            eprintln!(
                "SHAPE {:>6} b{:<3} {:>6}x{:<6} {:>4} calls  {:>6.3} ms/call  (batch)",
                shape.label, shape.batch, shape.rows, shape.cols, shape.calls, per_call
            );
            continue;
        }
        let per_pass = shape.nanos as f64 / steps as f64 / 1e6;
        let gbps = bytes / (per_call / 1e3) / 1e9;
        eprintln!(
            "SHAPE {:>6} b{:<3} {:>6}x{:<6} {:>4} calls  {:>7.2} ms/pass  {:>6.3} ms/call  {:>5.1} GB/s",
            shape.label, shape.batch, shape.rows, shape.cols, shape.calls, per_pass, per_call, gbps
        );
    }
}

/// Record a whole pass and print the running split.
///
/// The numbers are averages over every pass of that kind timed so far, which keeps a single noisy
/// step from being read as the answer. Prompt chunks and decode steps are counted apart.
pub fn finish_pass(start: Option<Instant>) {
    let Some(start) = start else { return };
    let total = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let prefill = IN_CHUNK.with(|flag| flag.get());
    let (counters, total_at, steps_at): (&[AtomicU64; 5], &AtomicU64, &AtomicU64) = if prefill {
        (&PREFILL_COUNTERS, &PREFILL_TOTAL, &PREFILL_STEPS)
    } else {
        (&COUNTERS, &TOTAL, &STEPS)
    };
    total_at.fetch_add(total, Ordering::Relaxed);
    let steps = steps_at.fetch_add(1, Ordering::Relaxed) + 1;
    let average = total_at.load(Ordering::Relaxed) / steps;
    let kind = if prefill { "chunk" } else { "decode" };
    let names = ["q4k", "q6k", "other_dot", "attention", "norm"];
    let mut line = format!("PROFILE {kind} {steps} {:.2} ms:", average as f64 / 1e6);
    let mut claimed = 0;
    for (index, name) in names.iter().enumerate() {
        let nanos = counters[index].load(Ordering::Relaxed) / steps;
        claimed += nanos;
        let share = nanos as f64 / average as f64 * 100.0;
        line.push_str(&format!(
            " {name} {:.2}ms ({share:.0}%)",
            nanos as f64 / 1e6
        ));
    }
    let rest = average.saturating_sub(claimed);
    line.push_str(&format!(
        " rest {:.2}ms ({:.0}%)",
        rest as f64 / 1e6,
        rest as f64 / average as f64 * 100.0
    ));
    eprintln!("{line}");
    // One table per run, once the caches and the pool are warm: the first passes carry the fault-in
    // cost of the weights, which is not what a steady-state question is about.
    if !prefill && steps == 20 {
        print_shapes(steps);
    }
}
