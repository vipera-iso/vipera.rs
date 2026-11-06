# Quickstart — vipera.rs Phase 1

How to prove each slice works. Nothing here needs a GPU, and every command is expected to work on a
clean checkout of this repository.

## Prerequisites

- Rust stable (`cargo 1.99.1` or newer) — already installed on this machine
- No model download required for slices 1–2; a 1 KiB synthetic GGUF is enough to exercise the parser

## Slice 1 — model introspection (delivered)

```bash
cd vipera.rs
cargo test --workspace        # unit + integration; no GPU, no network
cargo clippy --workspace --all-targets
```

Expected: all tests pass and clippy reports no warnings.

Make a synthetic model and ask the engine about it:

```bash
python3 - <<'PY'
import struct
def s(x):
    b=x.encode(); return struct.pack('<Q',len(b))+b
out=b'GGUF'+struct.pack('<I',3)+struct.pack('<Q',1)+struct.pack('<Q',2)
out+=s('general.architecture')+struct.pack('<I',8)+s('qwen2')
out+=s('qwen2.context_length')+struct.pack('<I',4)+struct.pack('<I',32768)
out+=s('blk.0.attn_q.weight')+struct.pack('<I',2)+struct.pack('<Q',32)+struct.pack('<Q',32)+struct.pack('<I',8)+struct.pack('<Q',0)
out+=b'\0'*(-len(out)%32)+b'\0'*(34*32)
open('/tmp/tiny.gguf','wb').write(out)
PY

cargo run -q -p vipera -- model-info /tmp/tiny.gguf
cargo run -q -p vipera -- model-info /tmp/tiny.gguf --json | python3 -m json.tool
```

Expected outcome (real output from this build):

```text
name: qwen2
identity: fnv1a64:39394915aafcc4cc
gguf_version: 3
tensor_count: 1
parameter_count: 1024
quantization: Q8_0
context_limit: 32768
vocab_size: 2
tensor_types:
  Q8_0: 1
```

Negative checks, each with exit code 2 and a message naming the cause:

```bash
printf 'NOPE' > /tmp/bad.gguf && cargo run -q -p vipera -- model-info /tmp/bad.gguf   # wrong magic
head -c 40 /tmp/tiny.gguf > /tmp/short.gguf && cargo run -q -p vipera -- model-info /tmp/short.gguf
cargo run -q -p vipera -- model-info /tmp/does-not-exist.gguf
```

## Slice 2 — hardware profile and placement plan (delivered)

```bash
cd vipera.rs
cargo test --workspace
cargo run -q -p vipera -- plan /tmp/plan.gguf --ctx 512
cargo run -q -p vipera -- plan /tmp/plan.gguf --ctx 512 --kv-type q8_0 --json
```

`/tmp/plan.gguf` is any GGUF with `block_count`, `embedding_length`, `attention.head_count`, and
`attention.head_count_kv` in its metadata. Real output from this build (GTX 1650 detected):

```text
model_identity: fnv1a64:55ef3b2da15b1361
hardware_fingerprint: hw_a32f9fee0db4c97f
cpu_cores_physical: 4
ram_total_bytes: 16544649216
device: cuda NVIDIA GeForce GTX 1650 (3911188480 bytes free of 4294967296)
layers_on_device: 2
kv_cache_type: f16
context_length: 512
batch_size: 1
predicted_device_bytes: 139264
predicted_host_bytes: 68
feasible: true
```

(That fixture is a 2-layer toy, so its numbers are small; `device_reserve_bytes` is 16 MiB for any
plan with layers on the device and is included in `predicted_device_bytes`. The calibrated numbers
for a real model are in the section below.)

Infeasible requests exit 3 and carry a reason:

```bash
cargo run -q -p vipera -- plan /tmp/plan.gguf --ctx 9999 --json   # exit 3
# feasible: false, violations[0].constraint = "context_limit"
```

Hardware detection is testable without a GPU: `parse_meminfo_total`, `parse_physical_cores`, and
`parse_nvidia_smi` are pure functions covered by unit tests.

## Slice 2b — calibration against measured memory (done)

The planner is checked against a reference implementation on real hardware, not against its own
arithmetic. Method: load a model with `llama-server -ngl 99` at a known context, read `nvidia-smi`
used memory minus the idle baseline, and compare with `vipera plan` for the same model and context.

```bash
# terminal A
cd vipera.cpp/build-cuda/bin
LD_LIBRARY_PATH=. ./llama-server -m ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -c 2048 -ngl 99 --host 127.0.0.1 --port 18083

# terminal B
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd vipera.rs
cargo run -q -p vipera -- plan ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --ctx 2048 --json
```

Result (idle baseline 2 MiB):

| ctx | predicted device | observed device | error |
|-----|------------------|-----------------|-------|
| 2048 | 1131.9 MiB | 1128 MiB | +0.3% |
| 4096 | 1187.9 MiB | 1186 MiB | +0.2% |

Two things the measurement changed:

1. Full offload now means **every** tensor on the device. Before the fix, `--ctx 2048` predicted
   772 MiB of device use where the GPU actually held 1128 MiB — a 31% under-prediction that would
   have called a non-fitting placement feasible.
2. `DEVICE_RESERVE_BYTES` (16 MiB) is now part of the estimate: the measured residual was 12.1 MiB at
   ctx 2048 and 14.1 MiB at ctx 4096. A Q8_0 model over-predicted by 6.5%, which is the safe
   direction. Re-measure when the runtime lands and replace the constant with our own numbers.

## Slice 3 — tokenizer and sampler (delivered)

```bash
cd vipera.rs
cargo test -p vipera-tokenizer -p vipera-sampler
```

`vipera-tokenizer` reads `tokenizer.ggml.tokens`, `.scores`, `.token_type`, `.model`, `.pre`, and
`.merges`. It picks one of three schemes from what the file declares:

- **byte-level BPE** (`tokenizer.ggml.model = gpt2`, merges present): bytes are mapped to printable
  characters, the text is cut by the pre-tokenizer, and pairs are merged by rank. This is what Qwen,
  Hermes, and Llama 3 models use.
- **unigram**: Viterbi over piece scores when `.scores` is present.
- **greedy longest-match**: when neither is available.

Control, user-defined, and `<...>` shaped tokens are matched before segmentation, so `<|im_start|>`
stays one id. SentencePiece text is normalised (`▁` plus dummy prefix), and characters outside the
vocabulary use `<0xNN>` byte tokens. Decoding reverses the byte mapping and skips control tokens.

Parity is measured, not assumed: 33 strings (code, JSON, Vietnamese, Cyrillic, emoji, tabs, repeated
whitespace, random Unicode) tokenized identically to llama.cpp's `/tokenize` on
`qwen2.5-1.5b-instruct-q4_k_m.gguf`. Re-run the comparison after touching the tokenizer; a divergence
means prompts reach the model differently than the model was trained on.

```bash
scripts/parity-tokenize.sh ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The script starts its own llama-server, compares both sides on a fixed corpus, prints the first
differing index, and exits non-zero on any divergence or missing prerequisite (it needs
`llama-server` on `PATH` or `LLAMA_SERVER`). It does not skip silently.

`vipera-sampler` applies repetition penalty, temperature, top-k, top-p, min-p in that order, with
temperature `0` meaning greedy. All randomness comes from an xorshift64* generator seeded only by
`SamplerConfig::seed`, so `(seed, config, call sequence)` fully determines the output — asserted by
the `the_same_seed_replays_the_same_run` test.

Chat rendering lives in `vipera-tokenizer::chat` and is reachable from the CLI:

```bash
cargo run -q -p vipera -- tokenize MODEL.gguf --system "be brief" --user "Hi there"
cargo run -q -p vipera -- tokenize MODEL.gguf --text "hello world" --json
cargo run -q -p vipera -- tokenize MODEL.gguf --ids "9707 1879"
printf '0.1, 4.2, 0.3' | cargo run -q -p vipera -- sample --temperature 0 --count 3 --seed 7
```

For `qwen2.5-1.5b-instruct-q4_k_m.gguf`, the rendered ChatML prompt is byte-identical to llama.cpp's
`/apply-template`, and its 32-token encoding is identical to `/tokenize` of that prompt. Templates this
build does not recognise (including Gemma) exit 2 instead of being approximated.

## `vipera profile` (delivered)

```bash
cd vipera.rs && cargo run -q -p vipera -- profile --json
```

Prints `hardware_fingerprint`, `cpu_cores_physical`, `ram_total_bytes`, and one object per device.
The fingerprint ignores free memory and driver versions, so it stays stable across probes; the free
byte count does not.

## Slice 4 — runtime and `verify` (delivered)

```bash
cd vipera.rs
cargo test -p vipera-runtime
cargo run -q -p vipera -- verify ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --ctx 2048 --cpu
```

`vipera-runtime` owns the loop: paged KV blocks backed by a real zeroed buffer, a sequence that keeps
its context between turns, prefix reuse over the longest common prefix, snapshots that invalidate
later snapshots when restored, release on finish/cancel/failure, per-stage timings, and one structured
JSON log line per request. The crate tests drive all of that with a deterministic model, so they need
no GPU and no weights.

Real output from this machine (GTX 1650, CPU placement, `f16` KV, 2048 tokens):

```text
model_identity: fnv1a64:533921e7d4cb1ae4
hardware_fingerprint: hw_b8983b83d1b27070
placement: cpu
layers_on_device: 0
kv_cache_type: f16
context_length: 2048
weights_bytes: 1111370240
kv_cache_bytes: 58720256
kv_blocks: 128
predicted_device_bytes: 0
measured_device_bytes: 0
predicted_host_bytes: 1170090496
measured_host_bytes: 1200779264
host_deviation_percent: +2.62
device_deviation_percent: not_measured
verdict: within_target
```

The +2.62% is the file's metadata and alignment padding, which a load holds but the weight sum does
not count. Without `--cpu` the same command exits 2:

```bash
cargo run -q -p vipera -- verify MODEL.gguf --ctx 2048   # exit 2: no device backend in this build
```

That refusal is deliberate: allocating no device memory and reporting a device figure anyway would be
the decorative estimate FR-005 exists to prevent.

## Slice 4b — CPU backend and `generate` (delivered)

```bash
cargo build --release -p vipera      # release; a debug build runs the same loops without optimisation
cargo run --release -q -p vipera -- generate \
    ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --prompt "The capital of France is" --max-tokens 12 --temperature 0
```

Real output from this machine: 5 prompt tokens and 12 generated in 4.2 s wall, of which 3.6 s is the
compute stage — 3.24 tokens per second, first token visible at 1.6 s (22.7 s and 0.53 tokens per
second before this slice's matvec work):

```text
 Paris. The capital of Italy is Rome. The capital of
```

`vipera-backend-cpu` dequantizes one row at a time from the file's own format (Q4_K/Q6_K here) and
keeps the weights quantized, so loading does not multiply memory by four. Every shape it needs — 28
blocks, 1536 hidden, 12 query heads, 2 KV heads, FFN 8960, epsilon 1e-6, rotary base 1e6 — comes from
the file's metadata, and K/V go into the runtime's paged buffer as half precision with the block table
deciding the slot.

The matvec is blocked over the reduction (one 256-value block of activations and its precomputed
sub-block sums stay hot while the weights are consumed) and rows are split across threads when a
product is large enough to pay for the spawn. Each row's sum order is unchanged, so a threaded run is
bit-identical to a single-threaded one. This took the same prompt from 22.7 s to 4.6 s of compute on
this 8-thread i5-11300H.

On top of that, Q4_K — the format that dominates this model — has an AVX2 kernel in
`vipera-backend-cpu/src/simd.rs`, chosen at runtime with `is_x86_feature_detected!("avx2")`. It is
2.5× the scalar kernel over the same data (0.162 s to 0.065 s for 355M multiply-adds) and took the
twelve-token compute from 4.58 s to 3.56 s with identical tokens. `unsafe` is usually forbidden here,
so that one crate opts out of the workspace lint, denies `unsafe_code` itself, and allows it only in
that module; randomised blocks are run through both kernels and compared, and a CPU without AVX2
never enters the module. AVX2 versions of Q6_K, Q8_0 and F32 were written and measured at 0.29×,
0.38× and 0.83× of scalar on this CPU, so only Q4_K is wired up — a hand-written kernel that loses to
the compiler is dead weight.

`--kv-type q8_0` stores K and V as one byte plus a half-precision scale per 32 values. At `--ctx 2048`
the cache drops from 58.7 MB to 31.2 MB (53%), and the same twelve-token greedy run produced the same
token ids. The layout is derived from the bytes per position the runtime reserved, so a cache planned
as `q8_0` is written as `q8_0` and any other geometry is refused by name.

Text mode prints each token as it is produced (first token at 1.6 s, one roughly every 0.32 s);
`--json` needs the finished turn and prints once. `scripts/parity-tokenize.sh` covers tokenization;
`cargo test -p vipera --test llama_continuations` diffs a greedy continuation against llama.cpp's
`/completion` token by token, and prints why it skipped when the model or `llama-server` is absent.

Checks that were actually run on the real model:

- the same prompt twice with `--temperature 0` produced byte-identical text (the loop and the KV reuse
  are deterministic);
- 40 tokens at `--ctx 512` crossed the 16-token block boundary at positions 16 and 32 and stayed
  coherent: *"the atmosphere scatters sunlight in all directions, and blue light is scattered more
  than other colors"*;
- `--chat --system "You are terse."` answered *"The largest planet in our solar system is Jupiter."*;
- the save/load path, `KvView` slot mapping across blocks, rotary pairs, GQA weight sharing, and each
  dequant format have unit tests (`cargo test -p vipera-backend-cpu`).

What is **not** verified: a token-by-token diff against llama.cpp for generation. The `llama-server`
binary that earlier slices used is no longer on this machine, so the comparison could not be run; the
evidence above is coherent, factually correct output, not a numeric match. Two runs of the model can
agree with each other while both differ from the reference, so treat exact parity as unproven until
`scripts/parity-tokenize.sh` (and a completion diff) can run on a machine with llama.cpp.

The backend's boundaries are explicit: qwen2/llama blocks only, AVX2 only for Q4_K (everything else
is scalar), and `f16` or `q8_0` KV. A quant format or architecture outside that list exits 2 with its
name in the message.

## Slice 5 — HTTP chat

`vipera serve <model> [--host] [--port] [--api-key] [--model-id] [--ctx] [--kv-type]` speaks the
contract in `contracts/http-chat.md`: `POST /v1/chat/completions` (streaming and not, OpenAI bodies),
`GET /v1/models`, `/health`, `/ready`, `/metrics`, and the frozen error envelope. It is a hand-written
HTTP/1.1 and JSON reader — no new dependency.

```text
$ curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Authorization: Bearer k' -d '{
  "model":"qwen2","stream":true,"max_tokens":8,"temperature":0,
  "messages":[{"role":"user","content":"The capital of France is"}]}'
data: {"id":"chatcmpl-…","object":"chat.completion.chunk",…"delta":{"role":"assistant","content":"Paris"}…}
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","delta":{},"finish_reason":"stop"}
data: [DONE]
```

Bounds, stated rather than implied: one generation at a time (requests queue on a mutex), request
bodies must be `Content-Length` framed, `stop` sequences are parsed and ignored, `429` rate limiting is
not implemented, and a client that disconnects mid-stream still lets the turn finish. `--host` outside
loopback refuses to start without `--api-key`. `cargo test -p vipera --test http_server` starts the
real binary and checks health, the 401, a completion, the `[DONE]` terminator and the metrics text.

## Slice 6 — conformance (GGUF part delivered)

```bash
cd vipera.rs && cargo test -p vipera-core --test conformance
```

The shared suite lives in `crates/vipera-conformance`. Every future algorithm and backend runs its
cases through the same runner; the GGUF parser already passes the `gguf-v2-v3` contract.

## Slice 5 — service (not yet implemented)

Will be validated by pointing an unmodified OpenAI-compatible client at `http://127.0.0.1:<port>`
and streaming a completion, per `contracts/http-chat.md`.

## What "green" means

Per `CONSTRAINTS.md`: `cargo test --workspace` passes, `cargo clippy --workspace --all-targets` is
warning-free, and no test is skipped. A slice that cannot meet this does not merge into the next.
