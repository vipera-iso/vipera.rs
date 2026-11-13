# vipera-ggml — ADR-001 prototype bind

Prototype for [`docs/design/ADR-001-ggml-q4k-bind.md`](../../docs/design/ADR-001-ggml-q4k-bind.md):
ggml's Q4_K × Q8_K kernels behind three functions, so the prefill gap against llama.cpp can
be attributed to the kernel rather than guessed at.

**This crate is not part of the engine and not part of the workspace.** It is excluded on
purpose: `cargo test --workspace` must never need a C toolchain or a ggml checkout. It exists
to take a measurement and to hold the parity test that a real integration will inherit.

## What is proven

- The kernel links and runs: `ggml-quants.c` and `ggml-cpu/arch/x86/quants.c` compile
  standalone with only `cc`/`ar`, and the shim supplies the five `ggml.c` symbols they
  reference (`ggml_abort`, `ggml_type_size`, `ggml_type_name`, `ggml_row_size`, the f16
  lookup table).
- Its output matches a Rust reference that follows ggml's block layouts by hand:
  `cargo test`.
- Single-threaded Q4_K GEMV, AVX-512 VNNI kernel, median of five:

  | shape | time | throughput |
  |---|---|---|
  | 8960 × 1536 | 0.54 ms | 14.26 GB/s |
  | 1536 × 8960 | 0.45 ms | 17.17 GB/s |
  | 151936 × 1536 | 9.9 ms | 13.25 GB/s |

## What is not proven

The 8×8 repacked GEMM that actually gives llama.cpp its prefill throughput lives in
`ggml-cpu/arch/x86/repack.cpp` (`ggml_gemm_q4_K_8x8_q8_K`), not in the files bound here. That
file also compiles standalone with the same flags, so the next step is to bind it and pair it
with `ggml_quantize_mat_q8_K`; only then does a pp128 number mean anything about prefill.

Until that runs, the honest statement is: the GEMV kernel is faster per core than this
engine's, the plumbing works, and the prefill number is still unexplained.

## Build and measure

```bash
VIPERA_GGML_SRC=/tmp/llamacpp cargo test                       # parity
VIPERA_GGML_SRC=/tmp/llamacpp cargo test --release --test throughput -- --ignored --nocapture
```

`build.rs` picks AVX-512 VNNI / AVX-512 / AVX2 from `/proc/cpuinfo` and prints which one it
chose; the same level is what `kernel_build()` reports at run time. The ggml checkout is not
vendored — point `VIPERA_GGML_SRC` at one. Vendoring is packaging work for the promote step.

## Rules this crate keeps

- EXC-001 scope is a C toolchain, nothing else: no new Rust dependency, no `cc` crate.
- Three `unsafe` functions, three one-line contracts, no state, no threads, no buffers owned.
- The unused ggml tables are zeroed, not fabricated: a path that reaches them fails the parity
  test instead of producing plausible numbers.
