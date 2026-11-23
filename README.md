# vipera.rs

[![ci](https://github.com/vipera-iso/vipera.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/vipera-iso/vipera.rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![engine dependencies](https://img.shields.io/badge/engine%20dependencies-0-brightgreen.svg)
![msrv](https://img.shields.io/badge/rustc-1.99-blue.svg)

**A CPU-first inference engine for GGUF models, written in Rust with no third-party crates in the
engine itself.** It parses the file, plans memory, tokenizes, samples, generates, and serves an
OpenAI-compatible chat endpoint. The engine links nothing outside `std`; the workspace's two
dependencies sit outside it and are recorded rather than quietly assumed: `rustls` for TLS in
`vipera serve` (`EXC-002`) and `cudarc` for the CUDA backend (`EXC-006`), in
[`CONSTRAINTS.md`](CONSTRAINTS.md).

The engine is the Cargo workspace in this repository. The specification, plan and contracts live
under [`specs/001-vipera-rs-engine/`](specs/001-vipera-rs-engine/).

## Where it stands

One laptop, not a claim about yours: i5-11300H (4 cores, 8 threads), `VIPERA_THREADS=8`, Qwen2.5-1.5B
Q4_K_M, f16 KV, against llama.cpp `e574a17` (build 466), CPU only, same threads and model. Every row is
a dated section in [`docs/benchmarks.md`](docs/benchmarks.md), including the ones that lost. The
prefill rows are the 2026-11-23 paired session; the decode rows are the 2026-11-15 one.

| | vipera | llama.cpp, same machine | vipera / llama.cpp |
|---|---:|---:|---:|
| prefill, 128 tokens | 194 tok/s | 202 | 0.96–0.97x |
| prefill, 512 tokens | 186 tok/s | 204–212 | 0.89–0.90x |
| prefill, 1024 tokens | 177 tok/s | 196–207 | 0.86–0.89x |
| prefill, 4096 tokens | 129 tok/s | 159–164 | 0.79–0.80x |
| decode, shallow | 34.6 tok/s | 33.9–36.7 | 0.94–1.02x |
| decode, 4k of history | 24.2 tok/s | 25.3 | 0.96x |
| decode, 32k of history, window 2048 | 28.0 tok/s (35.8 ms/step) | not measured | — (6.8 tok/s with the full history) |
| KV held for a windowed 32k run | 61 MB | not measured | — (896 MB with the full history) |

llama.cpp is ahead on prefill at every length and level on decode. The prefill gap is not a batch-width
artefact: at the engine's own 128-token width, the 2026-11-15 matched-ubatch arm still put llama.cpp
at 0.87x, 0.85x, 0.86x and 0.75x. It is the batched attention pass: the matvecs
cost the same per chunk at 4k as at 1k, while attention grows with the history, from 6% of the first
chunk to 41% of a 4k one. An earlier table here quoted a slower llama.cpp build from 2026-11-05
(143.6 / 150.0 tok/s) and showed the engine ahead; the ledger records the correction.

Against the hardware rather than another engine: the Q4_K and Q6_K kernels stream weights at
29–30 GB/s, 68–70% of the 43 GB/s this machine can read at all, and the output projection reaches
84% of it.

The shape behind those numbers: decode is a weight stream (Q4_K 19 ms and Q6_K 11 ms of a 35 ms
step), prefill is attention, and prefill's remaining cost at 4k is the attention kernel rather than
the matmuls. Each of those sentences is a measurement with a date, not a hypothesis.

## Quick start

Requires stable Rust (edition 2024, toolchain pinned by `rust-toolchain.toml`). No system libraries,
no network, no GPU.

```bash
cargo build --release

# Look at exactly what a model will cost before loading it
./target/release/vipera model-info model.gguf
./target/release/vipera plan      model.gguf --ctx 4096 --kv-type q8_0
./target/release/vipera verify    model.gguf --ctx 4096

# Generate, and serve
./target/release/vipera generate  model.gguf --prompt "The capital of France is" --max-tokens 12 --temperature 0
./target/release/vipera serve     model.gguf --port 8080 --api-key sk-local
```

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'authorization: Bearer sk-local' -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## What works today

- **GGUF** v3 parsing: metadata, tensor directory, quantized blocks (Q4_K, Q6_K, Q8_0, F32).
- **Memory planning** before load, and `verify` to compare the prediction with what the model
  actually costs.
- **Tokenizer** from the model's own vocabulary, with chat templates and generation prompts.
- **Sampling**: greedy, temperature, top-k, top-p, min-p, repetition, presence and frequency
  penalties, per-token `logit_bias`, seeded RNG.
- **Generation** on the CPU, with a KV cache in `f16` or `q8_0`, streamed to stdout token by token.
- **Long context**: an optional sliding window (`VIPERA_ATTN_WINDOW`, 0 by default) over the first
  `sinks` tokens plus the last `window` positions, which keeps a decode step flat at depth, and a
  ring KV that holds only the blocks that window can reach (61 MB at 32k instead of 896 MB). It is
  lossy and opt-in: the model cannot see the middle of its history.
- **Prompt-lookup speculation** (on for greedy turns; `VIPERA_SPECULATIVE=0` disables): verifies a
  longest-match n-gram draft in one weight read, growing the draft while it lands, and keeps the
  token stream bit-identical; measured 1.6–1.7x on repetitive prompts and 1.02–1.05x on quoted text.
  A turn only speculates when it is the only one decoding, so concurrent requests keep sharing
  weight reads through the batch.
- **Prefix cache** (off; `VIPERA_PREFIX_CACHE=1` or `Engine::set_prefix_cache(true)` turns it on): a
  request that finishes normally keeps the KV of its prompt, and a later request copies the longest
  prompt prefix it shares with one, at least a block long, instead of computing it. Kept blocks are
  given back to any allocation that needs them, so they never cause a refusal; at most 32 prompts are
  kept. Only prompt positions are offered, never tokens a request generated.
- **HTTP server** (`vipera serve`): `POST /v1/chat/completions` and `POST /v1/completions` (streaming
  SSE and non-streaming), request `stop` sequences, streamed usage (`stream_options.include_usage`),
  token logprobs with top alternatives, up to four `n` choices on the non-streaming path,
  `response_format: json_object` validation, `GET /v1/models`, `/health`, `/ready`, `/metrics`,
  bearer auth, error envelopes, and a per-key (or per-peer) request budget (600/min by default,
  `--rate-limit N`, and `--rate-limit 0` turns it off). `--tls-cert` and `--tls-key` serve HTTPS and
  are re-read when the files change. Every request writes one JSON line to stderr
  (`{"event":"request","peer":…,"method":…,"path":…,"status":…,"micros":…}`, the same shape
  `vipera generate` prints), and `VIPERA_ACCESS_LOG=0` silences it.
- **AVX2 and VNNI quantized kernels** (Q4_K, Q6_K), selected at runtime and checked against the
  scalar reference.

## How the repo is held to its numbers

This is the part that decides whether a change lands, and it is the reason the tables above are
trustworthy:

- **A gate, not an opinion.** Every change passes `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`, and
  `cargo test --workspace --locked`; CI runs the same three on every push.
- **A ledger.** A performance change ships with a before and after in
  [`docs/benchmarks.md`](docs/benchmarks.md). A change that cannot show a number is reverted, a change
  that measures neutral is reverted too, and the number is written down first so nobody builds the
  same dead end twice.
- **A written bar.** [`CONSTRAINTS.md`](CONSTRAINTS.md) is the quality floor, with every exception
  (a dependency, a toolchain) carrying an owner and an expiry instead of being assumed.
- **Exactness where it is promised.** The f16 attention kernel is checked bit for bit against
  decoding the layer first (`streaming_attention_is_bit_identical_to_decoding_the_layer_first`), so a
  faster path that changes the last bits cannot land by accident.

## Layout

```text
Cargo.toml                    workspace: 10 library crates, 1 binary
crates/
  vipera-errors/              error type and exit-code mapping
  vipera-traits/              the interfaces a backend and an algorithm implement
  vipera-core/                GGUF reader, hardware probe, memory planner
  vipera-conformance/         the suite every backend must pass
  vipera-tokenizer/           vocabulary, BPE, chat rendering
  vipera-sampler/             RNG and sampling policies
  vipera-runtime/             pipeline, sequences, KV cache, engine
  vipera-backend-cpu/         quantized kernels, scalar reference, AVX2
  vipera-backend-gpu/         CUDA backend behind the same `ForwardPass` (one device, f16 KV)
  vipera-server/              HTTP server and JSON, behind `vipera serve`
bins/vipera/                  the `vipera` command (argument parsing only)
scripts/                      parity checks against llama.cpp
specs/001-vipera-rs-engine/   spec, plan, tasks, CLI and HTTP contracts
docs/design/                  ADRs and the performance specification
docs/benchmarks.md            the measurement ledger, losing experiments included
docs/benchmarks/              the surveys the plan was written from
```

## Documentation

| Where | What |
|---|---|
| [`specs/001-vipera-rs-engine/`](specs/001-vipera-rs-engine/) | specification, plan, CLI and HTTP contracts, session checkpoints |
| [`docs/design/`](docs/design/) | ADR-001 (the ggml Q4_K bind), ADR-002 (weight mapping), continuous batching, the performance spec |
| [`docs/benchmarks.md`](docs/benchmarks.md) | every experiment with a number: what won, what measured neutral, what would have to be true for a proposal to work |
| [`docs/benchmarks/`](docs/benchmarks/) | the CPU-algorithm survey and the llama.cpp gap report, and how to add a measurement |
| [`CONSTRAINTS.md`](CONSTRAINTS.md) | the quality bar the gate enforces, with each exception recorded rather than assumed |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | the gate, commit conventions, and how a performance claim is made |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | how people are expected to behave here, and where to report it when they are not |
| [`SECURITY.md`](SECURITY.md) | what to report privately, and what is out of scope |

## Limitations

- **GPU**: `vipera-backend-gpu` implements `ForwardPass` on one CUDA device with `f16` KV, and
  `cargo test -p vipera-backend-gpu` holds it to the CPU backend through the runtime's KV buffer. No
  Metal or Vulkan backend, no batched (prefill) kernel yet, no sliding window (a ring table is
  refused), and the CLI has no `--backend cuda` yet, so a normal build still runs on the CPU. `verify`
  reports host memory only.
- **Prefill**: llama.cpp is faster at every prompt length measured here, from 3–4% at 128 tokens to
  20–21% at 4096; decode is level. Both are measured on the machine described above, not estimated.
- **Algorithm registry**: pipeline stages exist, but the algorithm crates are not written, so the
  planner has a single CPU path to choose from.
- **Server**: TLS needs a certificate; there is no ACME client, so issue or renew it out of band and
  write the files in place. Rate limiting is opt-in by design, and a request that cannot fit in
  memory is refused rather than served.
- **Parity checks**: the continuation test compares against a usage recording of llama.cpp checked
  into `bins/vipera/tests/fixtures`, so it runs on a machine with only the model; point
  `LLAMA_SERVER` at a `llama-server` binary to compare against a live one instead.
- **Activation precision**: Q4_K activations are multiplied as `i8` by default, so outputs agree with
  the `f32` path on the checked prompts but are not bit-identical to it; `VIPERA_Q4K_F32=1` runs
  Q4_K with `f32` activations, and its greedy continuation on the checked prompt is pinned by
  `bins/vipera/tests/fixtures/q4k_f32_continuation.json`.
- **No releases.** `main` is the only supported version; there are no versioned artifacts to pin, and
  the gate on the merge commit is the confirmation that a change is good.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first: the gate, the commit convention, and what a
performance claim has to carry. Work on a branch, merge through a pull request, and treat the CI run
for the merge commit as the confirmation. [`CONSTRAINTS.md`](CONSTRAINTS.md) is the bar the review
holds the change to. By participating you agree to the
[code of conduct](CODE_OF_CONDUCT.md).

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option, matching the `MIT OR Apache-2.0` declaration in `Cargo.toml`.
