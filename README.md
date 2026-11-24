# vipera.rs

[![ci](https://github.com/vipera-iso/vipera.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/vipera-iso/vipera.rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![engine dependencies](https://img.shields.io/badge/engine%20dependencies-0-brightgreen.svg)
![msrv](https://img.shields.io/badge/rustc-1.99-blue.svg)

A CPU-first inference engine for GGUF models, written in Rust. It parses the file, plans memory,
tokenizes, samples, generates, and serves an OpenAI-compatible chat endpoint. No third-party crates
in the engine itself: the workspace's two dependencies sit outside it and are recorded rather than
quietly assumed — `rustls` for TLS in `vipera serve` (`EXC-002`), `cudarc` for the CUDA backend
(`EXC-006`) — in [`CONSTRAINTS.md`](CONSTRAINTS.md).

## Recent wins (all behind switches, default off)

| Direction | Mechanism | Switch | Numbers (see [`docs/benchmarks.md`](docs/benchmarks.md)) |
|---|---|---|---|
| H1 — tile-GEMM attention at depth (attention −15.5% chunk at ~3.7k) | Two AVX-512 GEMMs over a 32×64 tile instead of a dot per key | `VIPERA_ATTN_GEMM` | merged #65, ledger "H1: tile-GEMM attention path — 2026-11-24" |
| H2 — RefineKV port (decode attention ≈ neutral on this machine) | Top-N per head in `attention_streaming_heads` | `VIPERA_KV_REFINEKV` | ledger "H2: RefineKV port — 2026-11-24", branch `perf/kv-refinekv` |
| H3 — DisSparse port (prefill, neutral on this machine) | Top-N rows per query entry in `attend_rows_sparse` | `VIPERA_PREFILL_SPARSE` | ledger "H3: DisSparse port — 2026-11-24" |
| H4 — TriAttention phase 1 (fused prefill skip, measurement pending) | Skip prefill tiles below threshold in `attend_rows_shared` | `VIPERA_TRIATTENTION` | ledger "H4: TriAttention phase-1 — 2026-11-24", branch `perf/attn-triattention` |
| 8x8 decode repack (A1) | `block_q4_Kx8` decode row-kernel path, `vpdpbusd` | `VIPERA_Q4K8` | step −34.8%, decode +53% (Pha 1b) |
| W1a tiled-band activation | Per-256 activation scales (`VIPERA_TILED_A`, default off) | `VIPERA_TILED_A` | chunk 8T +2.6% (2026-11-24) |

## Where it stands

One laptop, not a claim about yours: i5-11300H (4 cores, 8 threads), `VIPERA_THREADS=8`,
Qwen2.5-1.5B Q4_K_M, f16 KV, against llama.cpp `e574a17` (build 466), CPU only, same threads and
model. Every row is a dated section in [`docs/benchmarks.md`](docs/benchmarks.md), including the ones
that lost.

| | vipera | llama.cpp, same machine | vipera / llama.cpp |
|---|---:|---:|---:|
| prefill, 128 tokens | 226 tok/s | 218 | 1.04x |
| prefill, 512 tokens | 210 tok/s | 210 | 1.00x |
| prefill, 1024 tokens | 198 tok/s | 198 | 1.00x |
| prefill, 4096 tokens | 162 tok/s | 157 | 1.03x |
| decode, shallow | 34.6 tok/s (33.83 ms, `VIPERA_TILED_A` on) | 33.9–36.7 | 0.94–1.02x |
| decode, 4k of history | 24.2 tok/s | 25.3 | 0.96x |
| decode, 32k of history, window 2048 | 28.0 tok/s (35.8 ms/step) | not measured | — (6.8 tok/s with the full history) |
| KV held for a windowed 32k run | 61 MB | not measured | — (896 MB with the full history) |

Prefill is level with llama.cpp to +4% across those four lengths, re-measured on 2026-11-24 after an
earlier table showed a 20–21% deficit at 4096; decode is level, and llama.cpp's remaining edge is at
depth (0.96x at 4k of history). Prefill cost is not batch width: the matvecs cost the same per chunk
at 4k as at 1k, while attention grows with the history, from 6% of the first chunk to 41% of a 4k
one.

Against the hardware rather than another engine: the Q4_K and Q6_K kernels stream weights at
29–30 GB/s (68–70% of the 43 GB/s this machine can read at all), and the output projection reaches
84% of it.

### Repacking the weights, and what it is worth

llama.cpp's prefill is measured here at 217–227 tok/s (512 tokens), against an engine that measured
172 in that session and 210 in the 2026-11-24 re-measurement, and ik_llama.cpp at 268.7 / 199.4
tok/s at 512 / 4096 tokens against 209.6 / 140.3.

- **Ubatch is not it.** Running ik_llama.cpp at `-ub 128`, 256 and 512 moves it by 3–8%, not 15–20%.
- **The per-dot broadcast is not it either.** The tiled kernel spends one `vpbroadcastd` and one
  scratch load per `vpdpbusd`, where ik_llama.cpp reads a packed `__m512i` of weights and broadcasts
  the activation instead. Both microkernels for the same 16×16 tile were timed against each other
  (`repacked_gemm_form_against_the_broadcast_form`): 496–507 GFLOPS broadcast, 366–508 repacked, a
  median of +0.3%, and the one −26% run was a load spike.

What is left of that idea is the **unpack**, which the ledger already priced at 8.8% of the Q4_K
phase: splitting the nibbles, unpacking twelve header bytes and converting four f16 values per
row-block on every call. A tensor repacked at load removes that work, at 2.2× the bytes (i8 codes
plus two f32 per sub-block, 320 against 144), so it is a trade the engine has not taken: it buys at
most 5–6% of a prefill chunk and costs 2.2× the model's resident weights. Both numbers are in
[`CONSTRAINTS.md`](CONSTRAINTS.md) under `Rejected` with the way they were measured.

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
- **RoPE scaling** (linear and YaRN, from the model's `rope.scaling.*` metadata): a model
  extended past its training length rotates the way llama.cpp rotates it; a model without the
  keys is bit-identical to before. CPU only (the CUDA backend refuses such a model). Held to the
  ggml formula in unit tests and to llama.cpp's logits on a file whose metadata was set to YaRN
  (same argmax at 64 and 6001 tokens, ledger 2026-11-24); not yet run on a model trained with YaRN.
- **Long context**: an optional sliding window (`VIPERA_ATTN_WINDOW`, 0 by default) over the first
  `sinks` tokens plus the last `window` positions, which keeps a decode step flat at depth, and a
  ring KV that holds only the blocks that window can reach (61 MB at 32k instead of 896 MB). It is
  lossy and opt-in: the model cannot see the middle of its history.
- **Prompt-lookup speculation** (on for greedy turns; `VIPERA_SPECULATIVE=0` disables): verifies a
  longest-match n-gram draft in one weight read, growing the draft while it lands, and keeps the
  token stream bit-identical; measured 1.6–1.7× on repetitive prompts and 1.02–1.05× on quoted text.
  A turn only speculates when it is the only one decoding, so concurrent requests keep sharing
  weight reads through the batch.
- **Prefix cache** (off; `VIPERA_PREFIX_CACHE=1` or `Engine::set_prefix_cache(true)` turns it on): a
  request that finishes normally keeps the KV of its prompt, and a later request copies the longest
  prompt prefix it shares with one, at least a block long, instead of computing it. Kept blocks are
  given back to any allocation that needs them, so they never cause a refusal; at most 32 prompts
  are kept. Only prompt positions are offered, never tokens a request generated.
- **HTTP server** (`vipera serve`): `POST /v1/chat/completions` and `POST /v1/completions`
  (streaming SSE and non-streaming), request `stop` sequences, streamed usage
  (`stream_options.include_usage`), token logprobs with top alternatives, up to four `n` choices on
  the non-streaming path, `response_format: json_object` validation, `GET /v1/models`, `/health`,
  `/ready`, `/metrics`, bearer auth, error envelopes, and a per-key (or per-peer) request budget
  (600/min by default, `--rate-limit N`, `--rate-limit 0` turns it off). `--tls-cert`/`--tls-key`
  serve HTTPS and re-read the files when they change; a build with `--features acme` can order and
  renew the certificate itself (`--acme-domain`, HTTP-01). Every request writes one JSON line to stderr
  (`{"event":"request","peer":…,"method":…,"path":…,"status":…,"micros":…}`, the same shape
  `vipera generate` prints), and `VIPERA_ACCESS_LOG=0` silences it.
- **CUDA generation** (`--features cuda`, `generate --backend cuda`): the same `ForwardPass` on one
  device, Q4_K/Q6_K weights and `f16` KV, one request at a time. Prefill is one position per weight
  read and the speculative draft is off; `--kv-type q8_0` and `VIPERA_ATTN_WINDOW` are refused with
  the reason instead of being ignored.
- **AVX2 and VNNI quantized kernels** (Q4_K, Q6_K), selected at runtime and checked against the
  scalar reference. `vipera profile` prints the level the engine will use (`cpu_simd`: `avx512-vnni`,
  `vnni256`, `avx2`, `neon` or `scalar`), so a mixed fleet can see which host fell back.
- **8x8 repacked decode** (Q4_K, `VIPERA_Q4K8`, default off per ADR-006 — `=1` turns it on): the
  decode row-kernel path packs 8 rows as the ggml `block_q4_Kx8` and dots with `vpdpbusd` — step
  −34.8%, decode +53%. It changes the activation granularity, which is why it is not the default.
- **W1a tiled-band activation** (`VIPERA_TILED_A`, default off): the prefill tiled band reads one
  scale per 256-value block (`TiledActivation`, `fill_rows_batch`) plus 16 raw-`x` sums per block.
- **Prefill-sparse attention** (`VIPERA_PREFILL_SPARSE`, default off) and **RefineKV**
  (`VIPERA_KV_REFINEKV`, default off) — the two experiments in the "Recent wins" table.

## How the repo is held to its numbers

This is the part that decides whether a change lands, and it is the reason the tables above are
trustworthy:

- **A gate, not an opinion.** Every change passes `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`, and `cargo test --workspace
  --locked`; CI runs the same on every push.
- **A ledger.** A performance change ships with a before and after in [`docs/benchmarks.md`](docs/benchmarks.md).
  A change that cannot show a number is reverted, a change that measures neutral is reverted too,
  and the number is written down first so nobody builds the same dead end twice.
- **A written bar.** [`CONSTRAINTS.md`](CONSTRAINTS.md) is the quality floor, with every exception
  (a dependency, a toolchain) carrying an owner and an expiry instead of being assumed.
- **Exactness where it is promised.** The f16 attention kernel is checked bit for bit against
  decoding the layer first (`streaming_attention_is_bit_identical_to_decoding_the_layer_first`), so
  a faster path that changes the last bits cannot land by accident.

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
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | the gate, commit conventions, and what a performance claim has to carry |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | how people are expected to behave here, and where to report it when they are not |
| [`SECURITY.md`](SECURITY.md) | what to report privately, and what is out of scope |

## Limitations

- **GPU**: `vipera-backend-gpu` implements `ForwardPass` on one CUDA device with `f16` KV, and
  `cargo test -p vipera-backend-gpu` holds it to the CPU backend through the runtime's KV buffer
  (passing on a GTX 1650: same argmax at both checkpoints, cosine 0.9997). `generate --backend cuda`
  reaches it, but only when the binary is built with `--features cuda`; a normal build is CPU only
  and refuses the flag by name. No Metal or Vulkan backend, no batched (prefill) kernel (device
  prefill is one position per weight read), no sliding window (a ring table is refused), no
  speculative draft (it reads through `logits_range`, which the device backend does not implement),
  and `verify` reports host memory only.
- **Prefill**: re-measured 2026-11-24, the engine is level with llama.cpp to +4% at every length
  from 128 to 4096 tokens; the 20–21% deficit at 4096 an earlier table recorded is not reproduced.
  Decode is level too, with llama.cpp up to 4–6% ahead at depth (4k of history). What moved prefill
  is not attributed to one switch — the A/B for that has not been run — and the residual cost is
  attention growing with the history (6% of the first chunk at 1k, 41% of a 4k chunk) rather than
  batch width.
- **Algorithm registry**: pipeline stages exist, but the algorithm crates are not written, so the
  planner has a single CPU path to choose from.
- **Server**: TLS needs a certificate. An ACME client exists but is not in the default build:
  compile with `--features acme` and pass `--acme-domain` (plus optional `--acme-contact`,
  `--acme-directory`, `--acme-http-address`) to order and renew it over HTTP-01, writing the files
  `--tls-cert`/`--tls-key` name; without the feature those flags do not exist and you issue or renew
  out of band. Rate limiting is on by default (600/min per key, or per peer without a key;
  `--rate-limit N` changes it, `--rate-limit 0` turns it off), and a request whose plan does not fit
  in memory is refused rather than served.
- **Parity checks**: the continuation test compares against a usage recording of llama.cpp checked
  into `bins/vipera/tests/fixtures`, so it runs on a machine with only the model; point
  `LLAMA_SERVER` at a `llama-server` binary to compare against a live one instead.
- **Hardware floor**: the AVX-512 VNNI kernels are the fast path, and a CPU without them is
  correct but slower. Measured 2026-11-24 on the reference machine (wall clock, interleaved, same
  binary), the AVX2-only arm (`VIPERA_NO_VNNI=1`, `pmaddubsw`) ran prefill at **29.4 vs 142.6 tok/s**
  (4096 tokens; 32.3 vs 204.8 at 512) and decode at **21.0 vs 31.0 tok/s** — a 4.9–6.3× prefill gap
  and −32% decode. The tiled layout is over half of it: with VNNI but `VIPERA_TILED=0`, prefill
  measured 44.9 tok/s. That arm also runs a 32-token prompt chunk instead of 128 (`prefill_chunk`,
  `VIPERA_PREFILL_CHUNK` overrides it), measured 9% faster there and 15% slower on the tiled arm.
  Greedy output is identical across the levels on a short prompt (32 tokens), but
  a long prefill sums in a different order, so a token can differ at temperature 0; the same build
  on the same machine is deterministic. Ledger: `docs/benchmarks.md`, 2026-11-24.
- **Activation precision**: Q4_K activations are multiplied as `i8` by default, so outputs agree
  with the `f32` path on the checked prompts but are not bit-identical to it; `VIPERA_Q4K_F32=1`
  runs Q4_K with `f32` activations, and its greedy continuation on the checked prompt is pinned by
  `bins/vipera/tests/fixtures/q4k_f32_continuation.json`.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first: the gate, the commit convention, and what a
performance claim has to carry. Work on a branch, merge through a pull request, and treat the CI
run for the merge commit as the confirmation. [`CONSTRAINTS.md`](CONSTRAINTS.md) is the bar the
review holds the change to. By participating you agree to the
[code of conduct](CODE_OF_CONDUCT.md).

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option, matching the `MIT OR Apache-2.0` declaration in `Cargo.toml`.
