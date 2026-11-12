# vipera.rs

A CPU-first inference engine for GGUF models, written in Rust with no external crates: it parses the
file, plans memory, tokenizes, samples, generates, and serves an OpenAI-compatible chat endpoint.

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
- **Prompt-lookup speculation** (`VIPERA_SPECULATIVE=1`, greedy): verifies a longest-match n-gram
  draft in one weight read, growing the draft while it lands, and keeps the token stream
  bit-identical; measured 1.6-1.7x on repetitive prompts and 1.02-1.05x on quoted text.
- **HTTP server** (`vipera serve`): `POST /v1/chat/completions` and `POST /v1/completions`
  (streaming SSE and non-streaming), request `stop` sequences, streamed usage
  (`stream_options.include_usage`), token logprobs with top alternatives (chat streaming and
  non-streaming) and up to four `n` choices on the non-streaming path,
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
```

## Build and test

Requires stable Rust (edition 2024). No system libraries, no network, no GPU.

```bash
cargo build --release
cargo test --workspace        # 285 tests; model-dependent ones skip when no model is present
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

The server batches up to four conversations into one weight read; `--rate-limit` adds a per-key
request budget, and there is no request logging.

## Limitations

- **GPU**: there is no CUDA, Metal or Vulkan backend. `verify` reports host memory only.
- **Algorithm registry**: pipeline stages exist, but the algorithm crates are not written, so the
  planner has a single CPU path to choose from.
- **Server**: no request logging and no TLS; put it behind a proxy for either. Rate limiting is
  opt-in (`--rate-limit`), not a default.
- **Parity checks**: the tokenizer and continuation tests skip unless `llama-server` is installed.
- **Speed**: the CPU path is usable for a single conversation, not a throughput claim. Q4_K
  activations are multiplied as `i8`, so outputs agree with the `f32` path on the checked prompts
  but are not bit-identical to it.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option, matching the `MIT OR Apache-2.0` declaration in `Cargo.toml`.
