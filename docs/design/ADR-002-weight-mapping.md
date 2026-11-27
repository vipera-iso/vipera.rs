# ADR-002: page alignment or mmap for the weight buffer

Status: **Rejected at gate E1 itself** — page alignment buys nothing on this machine (section 4). No
product line of code changed; `fs::read` stays. mmap's remaining motive (RSS, cold start, a model larger
than RAM) is a memory goal rather than a speed one, and needs its own decision.
Date: 2026-11-13 · Owner: *to be filled in* · Supersedes: none

## 1. Context — current state and numbers

The model is loaded with `std::fs::read` — a `Vec<u8>` from the allocator, **not page aligned, no
hugepages, no mmap** (`bins/vipera/src/main.rs:167, 224, 334, 630, 874, 1091`). All 1.111 GB of tensors
live in the process's private heap, plus the KV cache.

Measured on this machine (`docs/benchmarks.md`, 2026-11-13, i5-11300H, 4 cores / 8 threads):

| Quantity | Value |
|---|---|
| Read set per decode token | **0.980 GB** (1.111 GB of tensors − 125 MB `token_embd`) |
| Measured read ceiling (STREAM-class, 4 threads) | **42.8 GB/s**; Copy 39.7 |
| Vipera decode | median-of-5 **35.1 ms/step** = **27.9 GB/s** (65% of ceiling); best 31.3 ms = 31.3 GB/s |
| Vipera, large Q4_K tensor (8960×1536) | **32.6 GB/s** |
| Vipera, Q6_K output (191 MB) | **35.6 GB/s** |
| llama.cpp on the same machine (4 threads) | 39.04 tok/s = **38.3 GB/s** (89% of ceiling) |
| 4 KiB pages for 1.111 GB | **~271,000 pages** |

## 2. The hypothesised mechanism

1. **TLB.** 271k 4 KiB pages against Tiger Lake's ~2,000 L2 TLB entries: every sweep over the weights is
   a page-walk chain that cannot be reused. 2 MiB hugepages divide the page count by 512.
2. **Alignment / allocation path.** An external experiment (Zig/Rust/C, 1 GiB, 1 thread) found Zig ~20%
   faster than Rust with each language's default allocator, and the **gap disappeared** when all three
   were forced onto page-aligned memory. The conclusion there: the difference came from the allocation
   path, not the language. That experiment's absolute numbers (~10 GB/s, 1-vCPU VM) do **not** carry over
   to this machine.
3. **THP on this machine:** `enabled=[always]`, `defrag=madvise`, `hpage_pmd_size=2 MiB`. But
   **`CONFIG_READ_ONLY_THP_FOR_FS` is absent** from `/proc/config.gz`, meaning a **read-only file mapping
   gets no hugepages** on this kernel; THP applies only to **anonymous memory**. This is the easiest place
   to over-promise: mmapping the file (option C) brings **no** hugepages, only shared RSS and lazy paging.
   To get hugepages the buffer must be anonymous, 2 MiB aligned, with `MADV_HUGEPAGE` (option B).

This is a **hypothesis**, not yet a cause. This ADR is not allowed to turn it into a conclusion before E1.

## 3. What this ADR does NOT explain

Four small attention tensors (1.33 MB and 0.22 MB) run at 5.2–18.3 GB/s and take 16% of the time — but
that is **not** TLB, not alignment, and not handshake overhead either: one dispatch measured **8.4 µs**
(`docs/benchmarks/beating-llamacpp.md` §E) and all the non-dot work of a step is 0.9–1.5 ms out of
40–45 ms. Dispatch coalescing was implemented, measured (paired median **0.999x**) and **reverted** for
no gain (`docs/benchmarks.md`). If E1 wins, it wins on large tensors (≥1 MB) through bandwidth, not by
making fewer calls.

## 4. Three experiments to run before changing code

E1 needs **no `unsafe` and no dependency** — it is the cheapest gate and must run first. E2 needs
`unsafe`/`libc`, so it only runs once E1 shows a signal (≥ 3%).

- **E1 — page alignment, compared inside one process.** Allocate a `Vec<u8>` of `len + 2 MiB`, take the
  subslice starting at the next 2 MiB boundary (all safe Rust), copy the 200 MB Q4_K tensor into both
  buffers, run the same `matvec_pooled` 20 times, alternating A/B, and take the median. Report each
  buffer's GB/s. *If the gap is < 3%: the alignment hypothesis is rejected and this ADR stops.*
- **E2 — hugepages, on anonymous memory.** Repeat E1 with an anonymous 2 MiB-aligned buffer plus
  `MADV_HUGEPAGE` (needs `unsafe`/`libc`, so this one is conditional), then read `/proc/self/smaps` to
  confirm `AnonHugePages` actually appears. **Do not** expect hugepages from a file mmap on this machine
  (section 2.3); if `AnonHugePages` stays 0, record "hugepages not measured" and draw no inference.
- **E3 — cold start.** Time to first token: `fs::read` (a 1.111 GB copy into the heap) against mmap
  (page faults on demand). mmap usually wins cold start and loses tail latency on first touch; both must
  be measured.

Only if E1 or E2 gives **≥ 5%** on large tensors does section 6 apply.

### E1 result (2026-11-13)

`crates/vipera-backend-cpu/tests/alignment_probe.rs`, 200 MB Q4_K, two buffers with the same seed,
alternating order over 5 rounds, best of 5:

| threads | plain (start % 2 MiB = 16) | aligned (start % 2 MiB = 0) | aligned / plain |
|---|---|---|---|
| 8 | 42.67 GB/s | 42.22 GB/s | **0.989x** |
| 4 | 39.84 GB/s | 39.91 GB/s | **1.002x** |

Both are below the 1.03 threshold → **page alignment is rejected**. Two further notes:

- The *unaligned* buffer already reached **42.67 GB/s**, 96% of the machine's 42.8 GB/s read-only
  ceiling. The Q4_K kernel is not short of bandwidth when the data is one flat span; the in-situ gap
  (33 GB/s on large shapes, 5.6 GB/s on the 0.22 MB shape) comes from call granularity and latency, not
  from the allocation path.
- `transparent_hugepage/enabled = [always]` on this machine, so E1 actually compared *allocator offsets*
  rather than "hugepages vs 4 KiB": both buffers may have been backed by THP. Separating that variable
  would need a branch forcing 4 KiB (`madvise(MADV_NOHUGEPAGE)`) — that needs `unsafe`, and should only be
  done if there is another reason to open exception EXC-002.
- E2 (hugepages on anonymous memory) **did not run**: E1 gave no signal, so per section 4 it stops here.
  Recorded as "not measured" rather than inferred.

## 5. Options

| # | Option | Gains | Costs | Policy |
|---|---|---|---|---|
| A | Keep `fs::read` | no change, portable, no `unsafe` | 1.111 GB private RSS, no page alignment, no THP | none |
| B | Anonymous 2 MiB-aligned heap (`std::alloc::alloc` with `Layout::from_size_align`) + `MADV_HUGEPAGE` | **The only hugepage path on this machine**; removes TLB pressure | still private RSS, still a 1.1 GB copy at load; `unsafe` outside `simd.rs` | EXC-002 |
| C | **mmap read-only**, falling back to `fs::read` | lower RSS (shared page cache, evictable), better cold start, runs models larger than RAM | **No hugepages** (this kernel does not enable THP for read-only files); `unsafe` or the `memmap2` dependency; SIGBUS if the file is truncated; Windows needs its own path | EXC-002 + R12 |
| D | C + `MAP_POPULATE` | fewer page faults at run time | loses the cold-start advantage, RSS is full again, still no hugepages | same as C |

`GgufFile::parse(&[u8])` already takes a slice, and `Pool` only borrows `&'d [u8]` for the scope of one
pass, so an owner `Mapped` living in `main` is enough — **no API change in core, runtime or backend**.

## 6. Decision (if E1/E2 met the threshold)

The decision depends on **which one wins in E1/E2**, and is not made in advance:

**Result: E1 below threshold, E2 not run → keep A (`fs::read`).** EXC-002 is not opened, no `unsafe`, no
dependency. mmap comes back to the table only when the goal is memory (RSS, cold start, a model larger
than RAM), and then the promote criteria must be rewritten for that goal rather than borrowing a speed
threshold.

The remaining speed lever is not in the weight buffer: see section 4 (E1) and the per-shape part of
`docs/benchmarks.md` — 25% of Q4_K time sits in the 0.22–1.3 MB tensors, and that is call latency.

Every option keeps the `fs::read` fallback and a gate (`VIPERA_MMAP=0` or `VIPERA_ALIGNED=0`) so it can be
switched off in one commit.

Fix the policy before merging, exactly as in ADR-001:

| Rule broken | Recorded where | What to change |
|---|---|---|
| `unsafe` only in `simd.rs` | workspace `Cargo.toml` still `forbid` | `allow(unsafe_code)` only in a new `src/mapped.rs` of `bins/vipera` (or `vipera-core`), following the `simd.rs:59` precedent |
| Every dependency needs approval | `CONSTRAINTS.md` | **EXC-002** with an owner and an expiry; if `memmap2` is used, add item **R12** to `research.md` under R3's admission rule |
| No second platform | CI is ubuntu-only today | State clearly that Windows/macOS use the `fs::read` fallback; do not open the matrix in v1 |

## 7. Test oracle

- `vipera-conformance` and `a_batched_step_matches_every_sequence_run_alone` must be green and
  **bit-identical** — mmap must not change a single bit.
- Compare `argmax` and logits between `VIPERA_MMAP=0` and `=1` on ≥3 prompts.
- Dedicated test: a file truncated after mapping must return `ViperaError`, not kill the process with
  SIGBUS.
- `verify` (predicted vs actual RSS) must match at least as well as it does today, not worse.

## 8. Promote / rollback

Current baseline: **decode median-of-5 35.1 ms (28.5 tok/s), best 31.3 ms; pp128 44.4–45.0 tok/s;
RSS ≈ 1.1 GB of weights + KV**.

Promote (make the winning option from section 6 the default) when, measured median-of-5 within one
session:

| # | Criterion | Threshold |
|---|---|---|
| 1 | decode | faster by **≥ 5%** at the median (≤ 33.3 ms) |
| 2 | prefill | no more than 2% slower (`pp128 ≥ 43.5`) |
| 3 | cold start | to first token **no more than** 10% slower than `fs::read` |
| 4 | RSS | **lower** than baseline (mmap uses page cache) — a rise means the mechanism is misunderstood |
| 5 | correctness | conformance + bit-identical parity, in both modes |
| 6 | stability | p99 per token no worse than 5% |

Rollback: turn the gate off (`VIPERA_MMAP=0` / `VIPERA_ALIGNED=0`, the `fs::read` fallback already
exists) or revert one commit; parse and runtime are untouched.

## 9. Risks

- **SIGBUS**: the file is truncated or replaced while running. Keep the fd and/or check the length before
  mapping; test in section 7.
- **Tail latency on first touch**: a major fault per 2 MiB; without `MAP_POPULATE`, the first token into a
  new region can be slower. Measure with p99, not just the median.
- **THP is not guaranteed**: `defrag=madvise` can refuse hugepages under fragmentation, and on this kernel
  it **only applies to anonymous memory**. Report what actually happened, do not promise.
- **RSS that "drops" is really page cache**: the report must say shared vs private explicitly, and must
  not compare `free`'s total.
- **Windows/macOS**: `fs::read` fallback; do not let `libc` break the build.
- **This is more a memory optimisation than a speed one**: if E1 gives only 2–3%, the real value is RSS
  and models larger than RAM — then the promote goal must be rewritten, not the threshold lowered to fit.

## 10. Non-goals

- No change to the KV cache layout, `ForwardPass`, or the matvec algorithm.
- No `MAP_POPULATE` by default.
- No new dependency if a `libc`-free mmap via syscalls in one `unsafe` module is enough (decided at
  implementation time, after E1).
- No promises about hugepages: report only what is measured.

## 11. Open questions

1. Do E1/E2 reach ≥5%? (this blocks the entire ADR)
2. Does the `unsafe` mmap live in `bins/vipera` or `vipera-core`? `core` is what knows GGUF, but `bins`
   is what reads files — leaning towards `bins` to keep `core` free of `unsafe`.
3. Who owns EXC-002, and when does it expire?
4. Is `madvise(MADV_SEQUENTIAL)` needed for the weight region, or is the hardware prefetcher enough?
   Measure, do not guess.
