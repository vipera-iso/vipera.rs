# vipera.rs

A CPU-first inference engine for GGUF models, written in Rust with no external crates in the engine
itself: it parses the file, plans memory, tokenizes, samples, generates, and serves an
OpenAI-compatible chat endpoint. The one dependency in the workspace is `rustls`, for TLS
termination in `vipera serve`, and it is recorded as `EXC-002` in [`CONSTRAINTS.md`](CONSTRAINTS.md)
rather than quietly breaking the rule.

The engine is the Cargo workspace in this repository. The specification, plan and contracts live
under [`specs/001-vipera-rs-engine/`](specs/001-vipera-rs-engine/).

## What works today

- **GGUF** v3 parsing: metadata, tensor directory, quantized blocks (Q4_K, Q6_K, Q8_0, F32).
- **Memory planning** before load, and `verify` to compare the prediction with what the model
  actually costs.
- **Tokenizer** from the model's own vocabulary, with chat templates and generation prompts.
- **Sampling**: greedy, temperature, top-k, top-p, min-p, repetition, presence and frequency
  penalties, per-token `logit_bias`, seeded RNG.
- **Generation** on the CPU, with a KV cache in `f16` or `q8_0`, streamed to stdout token by token.
- **Prompt-lookup speculation** (on for greedy turns; `VIPERA_SPECULATIVE=0` disables): verifies a
  longest-match n-gram draft in one weight read, growing the draft while it lands, and keeps the
  token stream bit-identical; measured 1.6-1.7x on repetitive prompts and 1.02-1.05x on quoted
  text. A turn only speculates when it is the only one decoding, so concurrent requests keep
  sharing weight reads through the batch.
- **Prefix cache** (off; `VIPERA_PREFIX_CACHE=1` or `Engine::set_prefix_cache(true)` turns it on):
  a request that finishes normally keeps the KV of its prompt, and a later request copies the
  longest prompt prefix it shares with one, at least a block long, instead of computing it. Kept
  blocks are given back to any allocation that needs them, so they never cause a refusal; at most
  32 prompts are kept. Only prompt positions are offered, never tokens a request generated.
- **HTTP server** (`vipera serve`): `POST /v1/chat/completions` and `POST /v1/completions`
  (streaming SSE and non-streaming), request `stop` sequences, streamed usage
  (`stream_options.include_usage`), token logprobs with top alternatives (chat streaming and
  non-streaming) and up to four `n` choices on the non-streaming path, `response_format: json_object` validation on non-streaming
  replies,
  `GET /v1/models`, `/health`, `/ready`, `/metrics`, bearer auth, error envelopes, opt-in per-key
  rate limiting.
- **AVX2 and VNNI quantized kernels** (Q4_K, Q6_K), selected at runtime and checked against the
  scalar reference.

## Layout

```text
Cargo.toml                    workspace: 9 library crates, 1 binary
crates/
  vipera-errors/              error type and exit-code mapping
  vipera-traits/              the interfaces a backend and an algorithm implement
  vipera-core/                GGUF reader, hardware probe, memory planner
  vipera-conformance/         the suite every backend must pass
  vipera-tokenizer/           vocabulary, BPE, chat rendering
  vipera-sampler/             RNG and sampling policies
  vipera-runtime/             pipeline, sequences, KV cache, engine
  vipera-backend-cpu/         quantized kernels, scalar reference, AVX2
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
| [`SECURITY.md`](SECURITY.md) | what to report privately, and what is out of scope |

## Build and test

Requires stable Rust (edition 2024). No system libraries, no network, no GPU.

```bash
cargo build --release
cargo test --workspace        # 315 tests; model-dependent ones skip when no model is present
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs format, lint and test on every push (`.github/workflows/ci.yml`).

## Run a model

Any GGUF the CPU can hold, for example a Qwen2.5-1.5B-Instruct Q4_K_M:

```bash
vipera model-info  model.gguf
vipera plan        model.gguf --ctx 2048 --kv-type q8_0
vipera verify      model.gguf --ctx 2048
vipera generate    model.gguf --prompt "The capital of France is" --max-tokens 12 --temperature 0
vipera generate    model.gguf --chat --system "Be brief." --prompt "Why is the sky blue?"
```

`generate --json` reports token ids, timings and memory in one object, so results can be diffed
between runs.

## Serve

```bash
vipera serve model.gguf --port 8080 --api-key sk-local &
curl -s localhost:8080/v1/chat/completions \
  -H 'authorization: Bearer sk-local' -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

`--tls-cert chain.pem --tls-key key.pem` serves HTTPS instead of HTTP (the pair is re-read when the
files change, so a renewed certificate is picked up without a restart). The server batches up to four
conversations into one weight read; requests carry a per-key budget of 600 a minute unless
`--rate-limit N` says otherwise, and `--rate-limit 0` turns the budget off. Every request writes one JSON line to stderr — `{"event":"request","peer":…,
"method":…,"path":…,"status":…,"micros":…}` — the same event shape `vipera generate` prints, and
`VIPERA_ACCESS_LOG=0` silences it.

## Limitations

- **GPU**: there is no CUDA, Metal or Vulkan backend. `verify` reports host memory only.
- **Algorithm registry**: pipeline stages exist, but the algorithm crates are not written, so the
  planner has a single CPU path to choose from.
- **Server**: TLS needs a certificate: `--tls-cert` and `--tls-key` (PEM; the chain file leaf first)
  serve HTTPS, and the pair is re-read when the files change, so a renewal does not need a restart.
  There is no ACME client — issue or renew the certificate out of band and write the files in place.
- **Parity checks**: the continuation test compares against a usage recording of llama.cpp checked
  into `bins/vipera/tests/fixtures`, so it runs on a machine with only the model; point
  `LLAMA_SERVER` at a `llama-server` binary to compare against a live one instead.
- **Speed**: the CPU path is usable for a single conversation, not a throughput claim. Q4_K
  activations are multiplied as `i8`, so outputs agree with the `f32` path on the checked prompts
  but are not bit-identical to it.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option, matching the `MIT OR Apache-2.0` declaration in `Cargo.toml`.
