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
}
