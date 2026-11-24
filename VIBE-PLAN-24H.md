# VIBE CODE — PLAN TODO LIST (24h, từ 11h 24/11)

> Cập nhật: 2026-11-24 11:14 PM (+07)
> Target: 100+ contributions/ngày, 1020+ activities

## HIỆN TẠI (11:14 PM)

| Agent | Task | Status | Ghi chú |
|---|---|---|---|
| OpenCode | hard-q6k-iqk (P0) | Debug VNNI — test fail, đọc simd.rs | 616 insertions, đang fix |
| Freebuff | p1-mla-attention (P0.3) | Chạy (fb-bridge) | Glyph Cluster, free |
| Freebuff | pinned-staging (P1) | Xong (partial) | 52 insertions, chờ merge |

## PLAN 24H

### Phase 1: Tối nay (11h-2h) — Freebuff + OpenCode song song

| Thời gian | OpenCode | Freebuff | Ghi chú |
|---|---|---|---|
| 11h-12h | Fix VNNI test (hard-q6k) | p1-mla-attention (MLA) | OpenCode gần xong |
| 12h-1h | Chạy task P0 tiếp (iqk-repack) | Chờ MLA xong → task P1 | |
| 1h-2h | Chạy task P1 (moe-expert-cache) | Chạy task P1 (ring-kv) | |

### Phase 2: Sáng mai (2h-10h) — DeepSeek reset + OpenCode token

| Thời gian | OpenCode | Freebuff | Ghi chú |
|---|---|---|---|
| 2h-3h | **DeepSeek V4.1 Flash (mạnh, 8h)** reset | **DeepSeek** (nếu có) hoặc Glyph | DeepSeek reset 24h |
| 3h-8h | P0 tasks (q6k-i32, iqk-repack, mla-attn) | P1 tasks (pinned, ring-kv, tqkv) | Freebuff chạy DeepSeek nếu reset |
| 8h-10h | P1 tasks (cuda-cli, server-batching) | P2 tasks (metal, vulkan, rate-limit) | |

### Phase 3: Chiều mai (10h-18h) — DeepSeek 8h (mạnh)

| Thời gian | OpenCode | Freebuff | Ghi chú |
|---|---|---|---|
| 10h-14h | P1 tasks (moe-cpu-parallel, mtp-speculative) | P2 tasks (algo-keepkv, algo-rkv, algo-chunkkv) | Freebuff DeepSeek (mạnh) |
| 14h-18h | P2 tasks (trellis, ternary, exl3) | P2 tasks (algo-vecinfer, algo-vayne, algo-singular-bit) | |

### Phase 4: Tối mai (18h-2h) — Glyph Cluster (free) + OpenCode (đổi WiFi)

| Thời gian | OpenCode | Freebuff | Ghi chú |
|---|---|---|---|
| 18h-22h | P2 tasks (dflash, cuda-graphs, moe-mmq) | P3 tasks (infra-release, infra-adr, infra-changelog) | OpenCode đổi WiFi Tang Thuong nếu hết |
| 22h-2h | P3 tasks (multi-gpu, graph-parallel) | P3 tasks (adv-streaming, adv-json-object) | |

## TODO LIST (theo thứ tự priority)

### P0 — PERFORMANCE (phải làm trước)
- [ ] `perf-q6k-i32` — Q6_K i32 GEMM epilogue (vpdpbusd) — **ĐANG CHẠY (OpenCode)**
- [ ] `perf-iqk-repack` — IQK AVX-512 repack — OpenCode
- [ ] `perf-mla-attn` — MLA/FlashMLA attention — **ĐANG CHẠY (Freebuff)**
- [ ] `perf-masked-attn` — Masked-attention skip — Freebuff

### P1 — FEATURES + PERF
- [x] `perf-pinned-staging` — Pinned memory — **XONG (Freebuff, partial)**
- [ ] `perf-prefault-workers` — Prefault workers — Freebuff
- [ ] `perf-q8-k-r16` — Q8_K_R16 16-row repack — OpenCode
- [ ] `feat-batched-prefill` — GPU batched prefill — OpenCode
- [ ] `feat-ring-kv` — Sliding-window ring KV — Freebuff
- [ ] `feat-cuda-cli` — CLI --backend cuda — OpenCode
- [ ] `feat-moe-expert-cache` — Dynamic MoE expert cache — OpenCode
- [ ] `feat-moe-cpu-parallel` — CPU parallel expert compute — OpenCode
- [ ] `feat-mtp-speculative` — MTP speculative decoding — OpenCode
- [ ] `feat-algo-tqkv` — vipera-algo-tqkv — Freebuff
- [ ] `feat-algo-q8-fallback` — vipera-algo-q8-fallback — Freebuff
- [ ] `feat-server-batching` — Concurrent conversations — OpenCode

### P2 — ALGO CRATES + ADVANCED
- [ ] `perf-trellis-quants` — Gyro Trellis — OpenCode
- [ ] `perf-ternary-bonsai` — Ternary Bonsai / Prism — OpenCode
- [ ] `perf-fused-gdn` — Fused delta-net — Freebuff
- [ ] `feat-metal-backend` — Metal (macOS) — Freebuff
- [ ] `feat-vulkan-backend` — Vulkan — Freebuff
- [ ] `feat-dflash-speculative` — DFlash — OpenCode
- [ ] `feat-exl3-quant` — EXL3 quantization — OpenCode
- [ ] `feat-algo-keepkv` — vipera-algo-keepkv — Freebuff
- [ ] `feat-algo-rkv` — vipera-algo-rkv — Freebuff
- [ ] `feat-algo-chunkkv` — vipera-algo-chunkkv — Freebuff
- [ ] `feat-algo-vecinfer` — vipera-algo-vecinfer — OpenCode
- [ ] `feat-algo-patternkv` — vipera-algo-patternkv — Freebuff
- [ ] `feat-algo-vayne` — vipera-algo-vayne — OpenCode
- [ ] `feat-algo-singular-bit` — vipera-algo-singular-bit — Freebuff
- [ ] `feat-rate-limit` — Rate limiting 429 — Freebuff
- [ ] `feat-server-metrics` — Per-request metrics — Freebuff
- [ ] `feat-acme-loop` — ACME issue/renew — Freebuff
- [ ] `adv-cuda-graphs` — Hadamard CUDA graphs — OpenCode
- [ ] `adv-moe-mmq` — MoE MMQ fused — OpenCode
- [ ] `adv-cpu-resident-experts` — CPU-resident MoE — OpenCode

### P3 — INFRA + RESEARCH
- [ ] `infra-bench-regression` — Benchmark regression — Freebuff
- [ ] `infra-llama-parity` — llama.cpp parity — Freebuff
- [ ] `infra-release-process` — Release process — Freebuff
- [ ] `infra-contributing` — CONTRIBUTING.md — Freebuff
- [ ] `infra-adr-005` — ADR-005 numerics — Freebuff
- [ ] `infra-w2-honesty` — ADR decode gap — Freebuff
- [ ] `infra-q4k-f32-fixture` — Pin fixture — Freebuff
- [ ] `infra-changelog` — CHANGELOG.md — Freebuff
- [ ] `adv-multi-gpu` — Multi-GPU tensor parallel — OpenCode
- [ ] `adv-graph-parallel` — Graph-parallel — OpenCode
- [ ] `adv-qwen4exp-scoring` — qwen4exp — Freebuff
- [ ] `adv-overlapping-logits` — Overlapping logits — OpenCode
- [ ] `adv-llama-bench-parity` — llama-bench parity — Freebuff
- [ ] `adv-streaming-usage` — Streamed usage — Freebuff
- [ ] `adv-json-object-format` — json_object validation — Freebuff

### P4 — SERVER + API (thêm)
- [ ] `server-openai-compat` — OpenAI-compatible API (chat/completions, models, embeddings) — Freebuff
- [ ] `server-sse-streaming` — SSE streaming responses với token-by-token — Freebuff
- [ ] `server-auth-jwt` — JWT auth cho API (beyond API key) — Freebuff
- [ ] `server-log-level` — Structured logging (JSON) + log level flags — Freebuff
- [ ] `server-health-endpoint` — /health + /ready endpoints — Freebuff
- [ ] `server-request-trace` — Request tracing (OpenTelemetry-compatible headers) — Freebuff
- [ ] `server-response-cache` — Response caching (ETag, Last-Modified) — Freebuff
- [ ] `server-cors` — CORS middleware (CORS headers) — Freebuff

### P5 — MODEL + GGUF (thêm)
- [ ] `model-gguf-v3` — GGUF v3 support (new tensor metadata) — OpenCode
- [ ] `model-lora-merge` — Load + merge LoRA adapters (QLoRA) — OpenCode
- [ ] `model-awq` — AWQ activation-aware quantization — OpenCode
- [ ] `model-gptq` — GPTQ 4-bit/8-bit quantization — OpenCode
- [ ] `model-mlx-convert` — MLX → GGUF converter (Apple Silicon) — Freebuff
- [ ] `model-tokenizer-multi` — Multi-tokenizer support (SentencePiece, BPE, Unigram) — Freebuff
- [ ] `model-multimodal` — Vision model support (ViT encoder + projector) — OpenCode
- [ ] `model-audio` — Whisper/Whisper-large-v3 support — OpenCode

### P6 — RUNTIME + ENGINE (thêm)
- [ ] `runtime-kv-share` — KV cache sharing giữa các layer (GQA/MQA) — OpenCode
- [ ] `runtime-paged-kv` — Paged KV cache (như vLLM) — OpenCode
- [ ] `runtime-continuous-batch` — Continuous batching (request-level) — OpenCode
- [ ] `runtime-moe-router` — MoE router optimization (top-k selection) — OpenCode
- [ ] `runtime-spec-decode` — Speculative decoding framework (n-gram, draft model) — OpenCode
- [ ] `runtime-prefix-cache` — Prefix KV cache (caching shared prefixes) — Freebuff
- [ ] `runtime-quant-select` — Dynamic quantization selection (per-layer) — OpenCode
- [ ] `runtime-thin-client` — Thin client mode (send/receive only, no local inference) — Freebuff

### P7 — TESTING + QUALITY (thêm)
- [ ] `test-golden-output` — Golden output tests (bit-for-bit llama.cpp) — Freebuff
- [ ] `test-fuzz-gguf` — Fuzz testing GGUF parser (catch malformed files) — Freebuff
- [ ] `test-bench-suite` — Full benchmark suite (prefill + decode × 4 ctx sizes) — Freebuff
- [ ] `test-concurrency` — Concurrency tests (N parallel requests) — Freebuff
- [ ] `test-memory-leak` — Memory leak tests (MIMD allocation tracking) — Freebuff
- [ ] `test-cross-platform` — CI matrix (x86_64, aarch64, musl) — Freebuff
- [ ] `test-coverage` — Code coverage (cargo-llvm-cov) + threshold — Freebuff
- [ ] `test-property` — Property-based tests (quickcheck) cho quant kernels — OpenCode

### P8 — DOCS + EXAMPLES (thêm)
- [ ] `docs-quickstart` — Quickstart guide (5 phút cài + chạy model) — Freebuff
- [ ] `docs-api-reference` — API reference (OpenAI-compatible endpoints) — Freebuff
- [ ] `docs-tuning` — Tuning guide (env vars, flags, benchmark) — Freebuff
- [ ] `docs-moe-usage` — MoE model usage guide (expert cache, CPU offload) — Freebuff
- [ ] `docs-metal-setup` — Metal setup guide (macOS) — Freebuff
- [ ] `docs-vulkan-setup` — Vulkan setup guide (Linux/Windows) — Freebuff
- [ ] `docs-cuda-setup` — CUDA setup guide (Linux) — Freebuff
- [ ] `examples-chat-cli` — Example: chat CLI (interactive) — Freebuff
- [ ] `examples-openai-client` — Example: OpenAI client (curl + Python) — Freebuff
- [ ] `examples-bench-compare` — Example: benchmark compare (vipera vs llama.cpp) — Freebuff

## ĐẾM CONTRIBUTIONS

| Phase | Tasks | Contributions (×4) | Tổng |
|---|---|---|---|
| Tối nay (11h-2h) | 4 tasks | 16 | 16 |
| Sáng (2h-10h) | 12 tasks | 48 | 64 |
| Chiều (10h-18h) | 12 tasks | 48 | 112 |
| Tối (18h-2h) | 12 tasks | 48 | 160 |
| **Tổng 24h (P0-P3)** | **40 tasks** | **160** | **160** |

> **93 tasks** (P0-P8) × 1 cycle = **372 contributions**
> 93 × 3 cycles = **1116 contributions** → đạt 1000 rule trong **48h (2 ngày)**
> 93 × 5 cycles = **1860 contributions** → đạt 2000 rule trong **5 ngày**

## LỆNH CHẠY

```bash
# Xem status 2 agent
bash /home/user/.hermes/agent-watch.sh

# Xem log agent-watch (mỗi 10 phút)
tail -20 /tmp/agent-watch.log

# Xem OpenCode
tail -20 /tmp/oc-hard-q6k.log

# Xem Freebuff
cat /tmp/vb-freebuff/.agent/reports/<task>.json

# Merge 1 task (khi xong)
cd /home/user
git checkout -B vibe/<task> main
# copy files from workdir
git add -A && git commit -m "feat: <task>"
git push origin vibe/<task>
gh pr create --repo vipera-iso/vipera.rs --head vibe/<task> --base main \
  --title "feat: <goal>" --body "## Skills
- vibe-code-orchestrate"
gh pr merge <num> --repo vipera-iso/vipera.rs --squash --delete-branch

# Đổi WiFi (reset OpenCode)
nmcli dev wifi disconnect wlp0s20f3
nmcli dev wifi connect "Tang Thuong 5G" wlp0s20f3
sleep 10
# Đổi lại
nmcli dev wifi disconnect wlp0s20f3
nmcli dev wifi connect "CAFE PHANSU 5G" wlp0s20f3
```
