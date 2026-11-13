//! E1 from `docs/design/ADR-002-weight-mapping.md`: does page alignment alone move the Q4_K
//! weight stream?
//!
//! Run with:
//!
//! ```text
//! cargo test -p vipera-backend-cpu --release --test alignment_probe -- --ignored --nocapture
//! ```
//!
//! Two 200 MB Q4_K tensors, one from a plain `Vec` (what `std::fs::read` gives), one sliced at a
//! 2 MiB boundary inside an over-allocated `Vec`. No `unsafe`, no dependency: over-allocate
//! `len + 2 MiB` and start at the next boundary. Calls alternate between the two, so the machine's
//! frequency drift hits both arms.
//!
//! The ADR's gate: a difference under 3% rejects the alignment hypothesis and stops the mmap work.

use std::time::{Duration, Instant};

use vipera_backend_cpu::quant::{matvec, row_bytes};
use vipera_core::gguf::GgmlType;

const COLS: usize = 1536;
const BYTES: usize = 200 << 20;
const ALIGN: usize = 2 << 20; // 2 MiB: the hugepage size on this machine.

fn fill(slice: &mut [u8], seed: u64) {
    let mut state = seed;
    for byte in slice.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state >> 24) as u8;
    }
    // The first four bytes of each block are two `f16` scales; keep them 1.0 so the results are
    // numbers rather than NaN, in case anything prints them.
    for block in (0..slice.len() - 4).step_by(144) {
        slice[block..block + 4].copy_from_slice(&[0x00, 0x3c, 0x00, 0x3c]);
    }
}

fn address_of(slice: &[u8]) -> usize {
    slice.as_ptr() as usize
}

#[test]
#[ignore = "measurement, not a correctness check"]
fn page_alignment_moves_the_q4k_stream_or_it_does_not() {
    let stride = row_bytes(GgmlType::Q4K, COLS).expect("Q4_K row");
    let rows = BYTES / stride;
    let bytes = rows * stride;

    // Plain: the allocator's own start, like `std::fs::read` hands back.
    let mut plain = vec![0u8; bytes];
    fill(&mut plain, 0x2545_f491_4f6c_dd1d);

    // Aligned: over-allocate and start at the next 2 MiB boundary. Safe Rust: the padding only
    // moves the slice start, and its length is unchanged.
    let mut backing = vec![0u8; bytes + ALIGN];
    let offset = (ALIGN - (address_of(&backing) % ALIGN)) % ALIGN;
    let aligned = &mut backing[offset..offset + bytes];
    fill(aligned, 0x2545_f491_4f6c_dd1d);

    let x = vec![0.0123_f32; COLS];
    let mut out = vec![0.0_f32; rows];

    // Warm-up call per buffer: the first pass also faults the pages in.
    matvec(GgmlType::Q4K, &plain, COLS, rows, &x, &mut out).expect("matvec");
    matvec(GgmlType::Q4K, aligned, COLS, rows, &x, &mut out).expect("matvec");

    let mut plain_times: Vec<Duration> = Vec::new();
    let mut aligned_times: Vec<Duration> = Vec::new();
    for round in 0..5 {
        let order: [bool; 2] = if round % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for plain_first in order {
            let at = Instant::now();
            if plain_first {
                matvec(GgmlType::Q4K, &plain, COLS, rows, &x, &mut out).expect("matvec");
            } else {
                matvec(GgmlType::Q4K, aligned, COLS, rows, &x, &mut out).expect("matvec");
            }
            let elapsed = at.elapsed();
            if plain_first {
                plain_times.push(elapsed);
            } else {
                aligned_times.push(elapsed);
            }
            std::hint::black_box(&out);
        }
    }

    let gbps = |times: &[Duration]| {
        let best = times.iter().min().expect("a run");
        (bytes as f64 / best.as_secs_f64()) / 1e9
    };
    let median_us = |times: &[Duration]| {
        let mut sorted: Vec<u128> = times.iter().map(|t| t.as_micros()).collect();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    };

    println!(
        "plain   : start % 2MiB = {:>7}, best {:>6.2} GB/s, median {:>5.1} ms",
        address_of(&plain) % ALIGN,
        gbps(&plain_times),
        median_us(&plain_times) as f64 / 1000.0
    );
    println!(
        "aligned : start % 2MiB = {:>7}, best {:>6.2} GB/s, median {:>5.1} ms",
        address_of(aligned) % ALIGN,
        gbps(&aligned_times),
        median_us(&aligned_times) as f64 / 1000.0
    );
    println!(
        "aligned / plain on best time: {:.3}x  (E1 gate: >= 1.03 to keep ADR-002 alive)",
        gbps(&aligned_times) / gbps(&plain_times)
    );
}
