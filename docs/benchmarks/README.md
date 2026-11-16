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
