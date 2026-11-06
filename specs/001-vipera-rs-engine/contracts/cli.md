# CLI Contract — `vipera` (Phase 1)

The CLI is the whole product surface for scripts. Machine output is frozen here: renaming or removing a
JSON key, or changing a key's type, is a breaking change (constitution Principle IV).

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Usage error (bad flags, missing argument) |
| 2 | Input error (model missing, unreadable, corrupt, unsupported version) |
| 3 | Infeasible plan (requested placement cannot fit) |
| 4 | Internal error |

Errors go to stderr; stdout carries only the requested output. `--json` output goes to stdout as a
single JSON document with no extra lines, so `vipera ... --json | jq` works.

## `vipera model-info <path> [--json]`

Reads the model header only; does not load weights and does not require a GPU.

Text output: one `key: value` line per field, in the order below, indented two spaces for nested values.

```json
{
  "name": "qwen2",
  "identity": "fnv1a64:39394915aafcc4cc",
  "gguf_version": 3,
  "tensor_count": 291,
  "parameter_count": 1543714304,
  "quantization": "Q4_K_M",
  "context_limit": 32768,
  "layer_count": 28,
  "vocab_size": 151936,
  "tensor_types": { "Q4_K": 197, "F32": 94 }
}
```

Fields that the file does not declare are omitted rather than emitted as `null`; `identity`,
`gguf_version`, `tensor_count`, and `tensor_types` are always present. `identity` is stable across
runs and machines for the same file content. It is a content identity (FNV-1a over metadata and the
tensor table), **not** a cryptographic digest and **not** an integrity check.

## `vipera plan <path> [--ctx N] [--kv-type T] [--json]`

Produces a `PlacementPlan` (see `data-model.md`). Requires no model load.

```json
{
  "model_identity": "fnv1a64:39394915aafcc4cc",
  "hardware_fingerprint": "hw_7d1a…",
  "layers_on_device": 20,
  "kv_cache_type": "q8_0",
  "context_length": 8192,
  "batch_size": 1,
  "predicted_device_bytes": 4123456789,
  "device_reserve_bytes": 16777216,
  "predicted_host_bytes": 1234567890,
  "feasible": true,
  "violations": []
}
```

When `feasible` is `false`, the process exits 3 and `violations` is non-empty.

Two rules about the numbers, both learned by measuring:

- `layers_on_device` equal to the model's layer count means **every** tensor is on the device,
  including token embeddings and the output projection. A partial split leaves all non-layer tensors
  on the host.
- `device_reserve_bytes` is measured overhead that is neither weights nor KV cache: CUDA context,
  allocator slack, and the compute buffer. It is `16777216` (16 MiB) for every plan with at least one
  layer on the device and `0` for a CPU-only plan, and it is already inside
  `predicted_device_bytes`.

Calibration record (GTX 1650, 4 GiB, llama.cpp b24, `-ngl 99`):

| Model | ctx | predicted device | observed device | error |
|-------|-----|------------------|-----------------|-------|
| qwen2.5-1.5b-instruct Q4_K_M | 2048 | 1131.9 MiB | 1128 MiB | +0.3% |
| qwen2.5-1.5b-instruct Q4_K_M | 4096 | 1187.9 MiB | 1186 MiB | +0.2% |
| qwen2.5-1.5b-instruct Q8_0 | 2048 | 1873.1 MiB | 1758 MiB | +6.5% |

Observed values are `nvidia-smi` used memory minus the 2 MiB idle baseline, taken after a completed
request. The last row over-predicts because the reference implementation keeps part of that model off
the device; erring high is deliberate.

## `vipera verify <path> [--ctx N] [--json]`

Loads the model with the planned placement, measures actual peak memory, and prints a
`VerificationRecord`. Requires the planned backend to be available.

## `vipera tokenize <path> [flags]`

Three modes, chosen by the flags:

| Mode | Flags | Output |
|------|-------|--------|
| encode text | `--text T`, `--text-file F`, or stdin | ids and pieces |
| encode chat | `--system S`, `--user U`, `--assistant A` (repeatable, in order) | the rendered prompt plus its ids |
| decode | `--ids "1 2 3"` | the text those ids produce |

`--no-generation-prompt` omits the assistant header. `--json` prints one document:

```json
{
  "count": 5,
  "ids": [151644, 872, 198, 9707, 151645],
  "pieces": ["<|im_start|>", "user", "\n", "Hi", "<|im_end|>"],
  "prompt": "<|im_start|>user\nHi<|im_end|>\n"
}
```

`pieces` appears only when encoding, `prompt` only when a chat was rendered, and `text` only when
decoding. Control and user-defined tokens are matched as single ids, so a chat marker never becomes
characters of its own.

Chat rendering recognises four template families by their markers: ChatML (`<|im_start|>`), Llama 3
(`<|start_header_id|>`), Llama 2 (`[INST]` with `<<SYS>>`), and Mistral (`[INST]`). Any other template
is refused with exit 2 rather than approximated; Gemma is named explicitly because its templates
differ between model versions. Mistral has no system slot, so a system message there is an error
instead of being silently merged.

## `vipera sample [flags]`

Samples token ids from a distribution supplied on the command line, a file, or stdin as comma or
whitespace separated logits. This exists so the sampling policies are inspectable without a model:

```bash
printf '0.1, 4.2, 0.3' | vipera sample --temperature 0 --count 3
vipera sample --logits "1,2,3" --model m.gguf --seed 7 --top-k 1 --json
```

Flags: `--logits CSV`, `--logits-file PATH`, `--model PATH` (decodes the sampled ids), `--count N`,
`--seed N`, `--temperature F`, `--top-k N`, `--top-p F`, `--min-p F`, `--repetition-penalty F`,
`--recent IDS`, `--json`.

```json
{ "count": 3, "tokens": [3, 3, 3], "seed": 7, "text": "..." }
```

`text` is present only with `--model`. The same seed, flags, and input reproduce the same ids.

## `vipera verify <path> [--ctx N] [--kv-type T] [--cpu] [--json]`

Loads a placement for real and compares measured peak memory with the plan's prediction. This is the
close of the prediction loop (FR-005, SC-002).

```json
{
  "model_identity": "fnv1a64:533921e7d4cb1ae4",
  "hardware_fingerprint": "hw_b8983b83d1b27070",
  "placement": "cpu",
  "layers_on_device": 0,
  "kv_cache_type": "f16",
  "context_length": 2048,
  "weights_bytes": 1111370240,
  "kv_cache_bytes": 58720256,
  "kv_blocks": 128,
  "predicted_device_bytes": 0,
  "measured_device_bytes": 0,
  "device_deviation_percent": null,
  "predicted_host_bytes": 1170090496,
  "measured_host_bytes": 1200779264,
  "host_deviation_percent": 2.62,
  "verdict": "within_target"
}
```

Rules this record obeys:

- `measured_host_bytes` is the process's peak resident set (`VmHWM`) after the file is read and the KV
  cache is allocated, minus the peak before. The KV cache is a real zeroed buffer, not a computed
  number, so the measured side moves when the code changes.
- A deviation is `null` when the predicted side is zero, because there is nothing to divide by. A
  device placement this build cannot allocate reports `null` there rather than a made-up figure.
- `verdict` is `within_target` when every measured deviation is within ±10%, `outside_target`
  otherwise, and `not_measured` when the platform gave no memory figure.
- `--cpu` plans against this machine with the accelerator removed from the profile; the record's
  `hardware_fingerprint` is therefore that profile's, not `vipera profile`'s.
- An infeasible plan exits 3 with the violated constraints and loads nothing. A placement that needs a
  device exits 2, because this build has no device allocator: refusing is the honest answer, and the
  message names `--cpu`.

## `vipera generate <path> --prompt TEXT [flags]`

Runs the model on the CPU and prints what it writes next. This is the first command that computes
anything: it loads the weights, tokenizes the prompt, drives the runtime loop, and decodes the result.

Flags: `--prompt TEXT`, `--chat`, `--system TEXT`, `--max-tokens N` (default 32), `--ctx N`,
`--kv-type f16|q8_0`, `--temperature F`, `--top-k N`, `--top-p F`, `--min-p F`,
`--repetition-penalty F`, `--seed N`, `--json`.

Without `--json` the completion text is printed alone, so it can be piped. With `--json`:

```json
{
  "prompt_tokens": 5,
  "reused_tokens": 0,
  "processed_tokens": 8,
  "generated_tokens": 4,
  "finish": "length",
  "tokens": [220, 16, 20066, 335],
  "text": " Paris",
  "tokens_per_second": 0.38,
  "micros": {"allocate": 0, "compress": 0, "speculative": 0, "sparse": 0, "compute": 10570470, "verify": 0, "sample": 43903},
  "log": "{\"event\":\"generation\",...}"
}
```

- `finish` is `stop` when a stop token was sampled, `length` when the budget ran out, and `error` or
  `cancelled` otherwise.
- `processed_tokens` can exceed `prompt_tokens`: decoded tokens are fed back through the model, and the
  last sampled token never is.
- `micros` is the same seven-stage breakdown as the log line, so a slow run can be attributed.
- `--chat` renders the model's own template and then encodes it; `--system` is ignored without `--chat`.
- `tokens_per_second` is wall-clock over the whole turn, including load-free prefill and sampling.
- Without `--json`, text is written and flushed as each token is decoded, so the output is watchable
  and pipe-friendly. With `--json` the document can only be written once, so nothing streams.
- `tokens` holds the sampled ids, which is what a continuation diff against another implementation
  needs.

Supported models, stated rather than guessed: the qwen2/llama block layout, F32/F16/Q8_0/Q4_K/Q6_K
weights, an `f16` or `q8_0` KV cache, and a vocabulary this build's tokenizer reads. Anything else
exits 2 with the reason — an architecture or format the backend cannot compute is refused, never
approximated.

`--kv-type q8_0` stores one byte plus a half-precision scale per 32 cached values: 53% of the `f16`
cache at the same context (31.2 MB against 58.7 MB at 2048 tokens on the 1.5B model), at the cost of
a quantisation step on K and V.

## `vipera serve <path> [--host ADDR] [--port N] [--api-key K] [--model-id NAME] [--ctx N] [--kv-type f16|q8_0]`

Starts the HTTP service described in `http-chat.md`. Binds to loopback by default; binding to a
non-loopback address without an API key is refused at startup rather than warned about.

## `vipera profile [--json]`

Hardware profile only; reads `/proc` and `nvidia-smi`, never a model file.

Text output is `key: value` lines: `hardware_fingerprint`, `cpu_cores_physical`, `ram_total_bytes`,
then one `device:` line per device and a `device_driver:` line when a driver version is known.

```json
{
  "hardware_fingerprint": "hw_a32f9fee0db4c97f",
  "cpu_cores_physical": 4,
  "ram_total_bytes": 16544649216,
  "devices": [
    {
      "kind": "cuda",
      "name": "NVIDIA GeForce GTX 1650",
      "memory_total_bytes": 4294967296,
      "memory_free_bytes": 3911188480,
      "driver_version": "615.78.08"
    }
  ]
}
```

Keys are now frozen. `memory_free_bytes` changes between runs; `hardware_fingerprint` deliberately
does not depend on it, so a plan cache keyed on the fingerprint stays valid. `driver_version` is
`null` when the platform does not report one, and `devices` is `[]` on a CPU-only machine.

## Compatibility rules

- New flags are additive and default to current behavior.
- Unknown flags exit 1 with a usage message on stderr.
- Human-readable text may change wording; JSON keys and types may not.
- Every command that reads a model accepts a directory path only when a single `.gguf` file is present.
