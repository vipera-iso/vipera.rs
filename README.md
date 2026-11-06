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
- **Sampling**: greedy, temperature, top-k, top-p, min-p, repetition penalty, seeded RNG.
- **Generation** on the CPU, with a KV cache in `f16` or `q8_0`, streamed to stdout token by token.
- **HTTP server** (`vipera serve`): `POST /v1/chat/completions` (streaming SSE and non-streaming),
  `GET /v1/models`, `/health`, `/ready`, `/metrics`, bearer auth, error envelopes.
- **AVX2 Q4_K matmul**, selected at runtime and checked against the scalar reference.

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
cargo test --workspace        # 187 tests; model-dependent ones skip when no model is present
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs exactly those three checks on every push (`.github/workflows/ci.yml`).

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

The server is single-conversation and serialized behind one engine; there is no rate limiting yet.

## Measured

On the machine this was developed on, Qwen2.5-1.5B-Instruct Q4_K_M, at temperature 0:

| | |
|---|---|
| Compute for 12 tokens | 0.98-1.03 s (11.7-12.2 tok/s, three runs) |
| Compute for 24 tokens | 1.76-1.83 s (13.1-13.6 tok/s, three runs) |
| KV cache at `--ctx 2048`, `f16` | 58.7 MB |
| Same cache at `q8_0` | 31.2 MB (53%) |

Time to first token is not in this table because the streaming path does not stamp it; the earlier
1.64 s figure was read from a run that no longer exists and is not repeated here.

Numbers are machine-bound, not a benchmark claim; `generate --json` is how they were read.

## Not there yet

- **GPU**: there is no CUDA, Metal or Vulkan backend. `verify` reports host memory only.
- **Algorithm registry**: pipeline stages exist, but the algorithm crates are not written, so the
  planner has a single CPU path to choose from.
- **Server**: no rate limiting, no request logging, no concurrent conversations.
- **llama.cpp parity**: the tokenizer and continuation checks skip unless `llama-server` is
  installed; they are the intended proof of numeric agreement.
- **Speed**: 12-13 tok/s greedy on this class of CPU. Reusing a thread pool across a token's
  matvecs, instead of spawning four threads per matvec, was worth about 11%: 11.9-12.1 tok/s before
  it, 13.1-13.6 after, same 24-token run. Q4_K activations are multiplied as `i8`, so outputs agree
  with the `f32` path on the checked prompts but are not bit-identical to it.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option, matching the `MIT OR Apache-2.0` declaration in `Cargo.toml`.
