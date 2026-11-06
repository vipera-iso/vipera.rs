# Data Model — vipera.rs Phase 1

Entities the engine reasons about. Field names here are the names used in the public JSON output
(`--json`) and in the API, so changing one is a contract change.

## ModelDescriptor

Identity and shape of a loaded model.

| Field | Type | Notes |
|-------|------|-------|
| `name` | string | Architecture family from the file, e.g. `llama`, `qwen2` |
| `identity` | string | Stable hash of metadata + tensor table (research R9) |
| `gguf_version` | integer | 2 or 3 in Phase 1 |
| `tensor_count` | integer | Count declared in the header |
| `parameter_count` | integer | Derived from tensor dimensions |
| `quantization` | string | Derived from tensor types present |
| `context_limit` | integer | From metadata when declared, otherwise unknown |
| `layer_count` | integer | From metadata when declared, otherwise derived |
| `vocab_size` | integer | From tokenizer metadata when present |

Validation: `identity` must be stable across processes; `tensor_count` must equal the number of tensor
entries actually parsed; `context_limit` may be absent but never guessed.

## GgufHeader / MetadataEntry / TensorInfo

Parser-level view, one per file.

- **GgufHeader**: magic, version, declared tensor count, declared metadata count, offset of the data section.
- **MetadataEntry**: key, typed value (bool, integers of each width, floats, string, array of any of
  those). Malformed entries are an error naming the key and offset.
- **TensorInfo**: name, dimensions, element type, offset within the data section. Total bytes must stay
  inside the file; a tensor that overruns the file is an error.

## HardwareProfile

| Field | Type | Notes |
|-------|------|-------|
| `cpu_cores_physical` | integer | Never zero; falls back to logical count |
| `ram_total_bytes` | integer | |
| `devices` | list of Device | Empty on CPU-only machines |
| `fingerprint` | string | Stable across reboots for the same hardware |

**Device**: `kind` (cuda/cpu), `name`, `memory_total_bytes`, `memory_free_bytes`, `compute_capability`,
`driver_version`.

## PlacementPlan

| Field | Type | Notes |
|-------|------|-------|
| `model_identity` | string | Links back to the ModelDescriptor |
| `hardware_fingerprint` | string | |
| `layers_on_device` | integer | 0 means CPU-only |
| `kv_cache_type` | string | |
| `context_length` | integer | |
| `batch_size` | integer | |
| `predicted_device_bytes` | integer | Peak, not steady state |
| `predicted_host_bytes` | integer | |
| `feasible` | boolean | `false` when any constraint is violated |
| `violations` | list of Violation | Empty when feasible |

**Violation**: `constraint` (device_memory, host_memory, context_limit, unsupported_quantization),
`required_bytes`, `available_bytes`, `message`. A plan with `feasible: false` is never loaded.

## VerificationRecord

| Field | Type | Notes |
|-------|------|-------|
| `model_identity` | string | |
| `hardware_fingerprint` | string | |
| `predicted_device_bytes` / `measured_device_bytes` | integer | |
| `predicted_host_bytes` / `measured_host_bytes` | integer | |
| `device_deviation_percent` / `host_deviation_percent` | number | Signed |
| `tokens_per_second` | number | Optional, absent when not measured |
| `verdict` | enum | `within_target` (≤10%), `outside_target`, `not_measured` |

## Conversation

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Client-visible in responses |
| `state` | enum | `admitted`, `prefill`, `decode`, `finished`, `cancelled`, `failed` |
| `turns` | list of Turn | Ordered |
| `token_count` | integer | |
| `snapshots` | list of SnapshotRef | Restorable points |

Transitions: `admitted → prefill → decode → finished`; any state may go to `cancelled` or `failed`.
Memory for a conversation is released on `finished`, `cancelled`, `failed`, or client disconnect.

## ContextSnapshot

`id`, `conversation_id`, `token_position`, `label` (optional user-supplied name). Restoring a snapshot
rewinds the conversation to `token_position`; later snapshots of that conversation become invalid.

## AlgorithmRegistration / BackendRegistration

`name`, `version`, `stability` (`experimental` / `stable` / `deprecated`), `capabilities` (list of
strings), `conformance` (`passing` / `failing` / `not_run`). Selection refuses a component whose
conformance is `failing`; falling back to the documented safe alternative is recorded in the response.

## AccessKey

`id` (never the secret), `label`, `permissions` (list), `rate_limit` (requests per window),
`window_seconds`, `enabled`. The secret itself is compared in constant time and never logged.

## Relationships

```text
ModelDescriptor 1 ── * PlacementPlan      (same identity, different hardware)
PlacementPlan   1 ── 1 VerificationRecord (per run)
Conversation    1 ── * ContextSnapshot
Conversation    * ── 1 SequenceBlocks      (retained KV state, released on finish)
AlgorithmRegistration / BackendRegistration ── selected per request, recorded per response
```
