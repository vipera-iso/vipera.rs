# Measurements

Every performance claim in this repository points at a number here. The rule is in
[`../../CONSTRAINTS.md`](../../CONSTRAINTS.md): a change that cannot show a number is reverted, and
a change that measures neutral is reverted too — but the number is written down first, so nobody
builds the same dead end twice.

## What is where

| File | What it is |
|---|---|
| [`../benchmarks.md`](../benchmarks.md) | the ledger: one dated section per experiment, including the ones that lost. Read it before writing a kernel. |
| [`cpu-algorithms.md`](./cpu-algorithms.md) | survey of CPU inference techniques with sources, and which of them fit this engine. |
| [`beating-llamacpp.md`](./beating-llamacpp.md) | what the gap to llama.cpp is made of, measured on this laptop rather than assumed. |

The ledger is the one that grows. The two surveys are dated reports that the plan was written from;
where a later measurement contradicts one of them, the ledger says so and the surface report is left
in place so the correction has something to point at.

## Adding an entry

Record the hardware (CPU model, physical cores, `VIPERA_THREADS`), the OS and toolchain, the GGUF
(arch, quant, size), the llama.cpp build when comparing, and the exact commands. The `## Template`
section at the top of the ledger is the shape to fill.

Two habits the ledger depends on:

- **Quote ratios, not milliseconds.** Absolute numbers here move by ~20% between runs, because the
  session's own tooling holds a core; the ratio of two arms measured back to back does not move by
  more than a point or two.
- **Compare arms inside one process** when the change only affects scheduling, a cache hint or an
  address. Two separate runs compare two different amounts of interference — that mistake is why
  the same head split once measured 20% and later 34%. `attention_combinations` and
  `kv_row_order_at_1024` are the working examples; the recipe is the "How to measure a switch on
  this machine" section of the ledger.

## What a claim has to carry

The checklist behind "Adding an entry", in one place. A ledger row records:

- the measurement command, exact and copy-pasteable;
- the date, and the machine (CPU, cores, RAM, OS);
- the model (arch, quant, size) and the thread count (`VIPERA_THREADS`);
- the baseline and the new arm, measured under the same conditions;
- the switch that gates the change (`VIPERA_*` variable, feature, or "none — default path").

The withdrawn prefill/decode claim in [`../../README.md`](../../README.md) is the failure this list
prevents: it was read off `PROFILE` running averages, which undercount by ~11%, and carried none of
these fields, so nothing in the entry could show the number was wrong.

## Reproducing

Every bench in the ledger is an `#[ignore]`d test, so `cargo test --workspace` stays fast and the
benchmarks stay buildable:

```sh
VIPERA_TEST_MODEL=~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  cargo test --release -p vipera-backend-cpu --test win_vs_llama -- --ignored --nocapture
```

A bench that needs no model is a plain unit test in the crate that owns the code, named after what
it measures:

```sh
taskset -c 2-7 env VIPERA_THREADS=6 \
  cargo test --release -p vipera-backend-cpu --lib attention_combinations -- --ignored --nocapture
```

## Q8_0_R16 activation repack — measurement pending

`VIPERA_Q8R16` (default off) packs each 16-row group of a wide Q8_0 prompt chunk into Q8_0
activations — 16 contiguous scales and transposed codes — so one pass over a weight row produces
16 outputs. It is the ik_llama.cpp `Q8_0_R16` layout ported to the one integer-activation format
this backend has; Q4_K and Q6_K already share a weight pass across the chunk. It is also a numeric
change (one scale per 32 activation values), so tokens can differ from the f32 path.

**No number is claimed for it.** `bench_q8r16_against_f32_batch` is `#[ignore]`d and has not been
run on the reference machine, so there is no before/after to record yet. The exact measurement —
release, because a debug build cannot time a 8960×1536 Q8_0 row pass at a usable speed:

```sh
cargo test --release -p vipera-backend-cpu --lib bench_q8r16_against_f32_batch -- --ignored --nocapture
```

The bench compares the f32 batch row kernel against the repack back to back in one process; the
result only becomes a claim once its ratio is written into a dated section of the ledger
[`../benchmarks.md`](../benchmarks.md). Until then the switch stays off.

## Round-2 correctness fixes — 2026-11-25

Three bugs found in round 2, recorded together so the next person does not have to re-derive the
repro. These are **correctness fixes, not performance results**: no speed number is claimed for any
of them, none changes a default switch, and all five regression tests are plain unit tests with no
model or hardware dependency.

| Fix | Commit | What was wrong | Regression test(s) |
|---|---|---|---|
| Divide-by-zero in Q4K8 | `97162b6` | `matvec` accepted a zero-column matrix (zero is a whole number of blocks, so `row_bytes` passed it), and with `VIPERA_Q4K8=1` the 8x8 Q4_K kernel divided the tensor length by that zero stride. The public boundary now rejects it with a "zero columns" error, as `dequantize` already did. | `quant::tests::zero_columns_is_an_error_not_a_division_by_zero`, `quant::tests::q4k_and_q6k_round_trip_within_limits_at_block_boundaries` |
| Malformed BPE merges | `154669e` | `BpeEncoder::from_vocab` turned every merge pair into a rank, so a pair with an empty side produced a rank keyed by the separator alone. Empty sides are now skipped. | `bpe::tests::empty_merges_produce_no_merges_without_panic` |
| Malformed HTTP input | `0203c46` | The request reader had to answer, not panic or hang, on a missing request line or path, a non-numeric/negative/overflowing `Content-Length`, conflicting `Transfer-Encoding`, duplicate `Authorization`, and an unterminated oversized head; invalid UTF-8 in the head had to decode lossily rather than panic on a UTF-8 boundary. | `http::tests::malformed_requests_get_a_structured_error_instead_of_a_panic`, `http::tests::invalid_utf8_in_a_head_is_lossy_not_fatal` |

Exact commands, from the repository root; each targeted test also runs inside the full-crate
command recorded in its fix commit:

```sh
# divide-by-zero in Q4K8 (vipera-backend-cpu)
cargo test -p vipera-backend-cpu --lib zero_columns_is_an_error_not_a_division_by_zero
cargo test -p vipera-backend-cpu --lib q4k_and_q6k_round_trip_within_limits_at_block_boundaries

# malformed BPE merges (vipera-tokenizer)
cargo test -p vipera-tokenizer --lib empty_merges_produce_no_merges_without_panic

# malformed HTTP input (vipera-server)
cargo test -p vipera-server --lib malformed_requests_get_a_structured_error_instead_of_a_panic
cargo test -p vipera-server --lib invalid_utf8_in_a_head_is_lossy_not_fatal
```

The full-crate commands the round-2 workers ran alongside `cargo fmt --check` and
`cargo clippy -p <crate> --all-targets` were `cargo test -p vipera-backend-cpu`,
`cargo test -p vipera-tokenizer` and `cargo test -p vipera-server`.
