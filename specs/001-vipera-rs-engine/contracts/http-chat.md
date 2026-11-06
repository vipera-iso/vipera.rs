# HTTP Contract — chat completions (Phase 1)

Compatibility target: an existing OpenAI-compatible client works after changing only the base URL.
Unknown request fields are ignored, never rejected.

## Authentication

`Authorization: Bearer <key>` when an API key is configured. Missing or invalid key → `401` with the
error body below. When the server is bound to loopback and no key is configured, requests are allowed.

## `POST /v1/chat/completions`

Request (subset honored; everything else ignored):

```json
{
  "model": "qwen2",
  "messages": [
    { "role": "system", "content": "You are terse." },
    { "role": "user", "content": "Hello" }
  ],
  "stream": false,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "min_p": 0.0,
  "max_tokens": 256,
  "stop": ["\n\n"],
  "seed": 42,
  "repetition_penalty": 1.1
}
```

Response (`200`, `stream: false`):

```json
{
  "id": "chatcmpl-01J…",
  "object": "chat.completion",
  "created": 1762400000,
  "model": "qwen2",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Hi." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14 }
}
```

Streaming (`200`, `stream: true`, `Content-Type: text/event-stream`): one event per delta, then a
terminator. Every data frame is a JSON `chat.completion.chunk` with `choices[0].delta`; the final frame
carries `finish_reason` and is followed by `data: [DONE]`.

```
data: {"id":"chatcmpl-01J…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}
data: {"id":"chatcmpl-01J…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

`finish_reason` values: `stop`, `length`, `cancelled`, `error`.

## Error body

Every non-2xx response uses one shape:

```json
{ "error": { "message": "model is not loaded", "type": "invalid_request_error", "code": "model_not_loaded" } }
```

| Status | `code` | When |
|--------|--------|------|
| 400 | `invalid_request` | Malformed JSON, empty messages, unsupported role |
| 401 | `unauthorized` | Missing or wrong API key |
| 404 | `model_not_found` | `model` does not match a loaded model |
| 422 | `context_too_long` | Prompt exceeds the model's context limit |
| 429 | `rate_limited` | Per-key request limit exceeded; includes `Retry-After` |
| 500 | `internal_error` | Engine failure; the same request ID appears in logs |
| 503 | `model_not_loaded` | Server is starting or the model failed to load |

Client disconnect mid-stream releases the conversation's retained state; the partial answer is not
retried automatically.

## `GET /v1/models`

```json
{ "object": "list", "data": [ { "id": "qwen2", "object": "model", "owned_by": "local" } ] }
```

## `GET /health` and `GET /ready`

`/health` → `200 {"status":"ok"}` while the process is alive.
`/ready` → `200 {"status":"ready","model":"qwen2"}` only when a model is loaded and can serve;
otherwise `503 {"status":"not_ready","reason":"..."}`.

## `GET /metrics`

Prometheus text exposition. Required families in Phase 1: `vipera_requests_total`,
`vipera_tokens_generated_total`, `vipera_request_duration_seconds` (histogram),
`vipera_time_to_first_token_seconds` (histogram), `vipera_kv_blocks_in_use`, `vipera_oom_events_total`.

## Compatibility rules

- Response field names, types, and the `data: [DONE]` terminator are frozen.
- New optional response fields may be added; existing ones are never renamed or removed.
- Unknown request fields are ignored; unknown roles and malformed bodies are rejected.
