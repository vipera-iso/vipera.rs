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

### P9 — BENCHMARK + BENCH (thêm)
- [ ] `bench-prefill-sweep` — Prefill sweep (128→32k, 8 sizes, chart) — OpenCode
- [ ] `bench-decode-sweep` — Decode sweep (32→1024 tokens, chart) — OpenCode
- [ ] `bench-llama-compare` — Auto-compare vs llama-bench (script) — Freebuff
- [ ] `bench-cuda-vs-cpu` — GPU vs CPU benchmark (same model, same ctx) — OpenCode
- [ ] `bench-moe-throughput` — MoE model throughput (expert cache on/off) — OpenCode
- [ ] `bench-kv-quant` — KV quantization (f16 vs Q8_0 vs Q4_0) — Freebuff
- [ ] `bench-quant-accuracy` — Quantization accuracy (PPL on WikiText) — Freebuff
- [ ] `bench-regression-gate` — CI gate: fail if tok/s drops >5% — Freebuff
- [ ] `bench-report-gen` — Auto-generate benchmark report (Markdown + HTML) — Freebuff
- [ ] `bench-multi-model` — Batch benchmark (5 models, one report) — OpenCode

### P10 — MODEL FEATURES (thêm)
- [ ] `model-embedding` — Text embedding endpoint (/embeddings) — Freebuff
- [ ] `model-classifier` — Text classifier endpoint (logits → label) — Freebuff
- [ ] `model-rerank` — Rerank endpoint (query + docs → scores) — Freebuff
- [ ] `model-rag` — RAG pipeline (chunk + embed + retrieve + generate) — OpenCode
- [ ] `model-agent-loop` — Agent loop (tool calling, multi-step) — OpenCode
- [ ] `model-fine-tune` — LoRA fine-tuning (train on CPU, 1 epoch) — OpenCode
- [ ] `model-quant-calibrate` — Quantization calibration (run N samples, pick scale) — Freebuff
- [ ] `model-safetensors` — Safetensors format support (in addition to GGUF) — OpenCode
- [ ] `model-onnx` — ONNX export + inference (optional) — OpenCode
- [ ] `model-hf-download` — HuggingFace model download + convert to GGUF — Freebuff

### P11 — SERVER FEATURES (thêm)
- [ ] `server-tls-cert` — TLS certificate management (self-signed + ACME) — Freebuff
- [ ] `server-webhook` — Webhook endpoint (trigger generation on HTTP POST) — Freebuff
- [ ] `server-websocket` — WebSocket streaming (real-time token stream) — Freebuff
- [ ] `server-grpc` — gRPC alternative to REST (protobuf) — OpenCode
- [ ] `server-graphql` — GraphQL endpoint (query-based generation) — Freebuff
- [ ] `server-async` — Async request queue (priority + fair scheduling) — OpenCode
- [ ] `server-multi-model` — Multi-model serving (load N models, route by name) — OpenCode
- [ ] `server-a/b-test` — A/B test endpoint (route to model A or B by weight) — Freebuff
- [ ] `server-otel` — OpenTelemetry metrics + traces (exporter) — Freebuff
- [ ] `server-prometheus` — Prometheus /metrics endpoint (Gauge, Counter, Histogram) — Freebuff

### P12 — OPTIMIZATION (thêm)
- [ ] `opt-channel-last` — Channel-last tensor layout (better cache) — OpenCode
- [ ] `opt-vectorize-loop` — Manual loop vectorization (unroll + align) — OpenCode
- [ ] `opt-prefetch` — Software prefetch hints (prefetchr, AVX-512) — OpenCode
- [ ] `opt-tile-size` — Tile size autotuning (runtime search) — OpenCode
- [ ] `opt-fp16-math` — FP16 math on AVX-512 (FMA) — OpenCode
- [ ] `opt-int8-gemm` — INT8 GEMM (DQ8_0, Q8_0) — OpenCode
- [ ] `opt-sparse-gemm` — Sparse GEMM (skip zero weights) — OpenCode
- [ ] `opt-winograd` — Winograd convolution for ViT (if multimodal) — OpenCode
- [ ] `opt-mem-pool` — Memory pool allocator (no per-token malloc) — Freebuff
- [ ] `opt-thread-pool` — Thread pool with work stealing (rayon-like) — Freebuff

### P13 — INTEGRATION (thêm)
- [ ] `int-python-bindings` — Python bindings (PyO3, vipera-py) — OpenCode
- [ ] `int-node-bindings` — Node.js bindings (napi-rs, vipera-node) — Freebuff
- [ ] `int-go-bindings` — Go bindings (cgo, vipera-go) — Freebuff
- [ ] `int-jni` — Java JNI bindings (Android) — Freebuff
- [ ] `int-swift` — Swift bindings (iOS/macOS) — Freebuff
- [ ] `int-c-api` — C API (for C/C++ callers) — Freebuff
- [ ] `int-docker` — Docker image (CPU + CUDA variants) — Freebuff
- [ ] `int-k8s` — Kubernetes deployment (Helm chart + operator) — Freebuff
- [ ] `int-cloud` — Cloud deployment (AWS Sagemaker, GCP Vertex, Azure ML) — OpenCode
- [ ] `int-edge` — Edge deployment (Raspberry Pi, Jetson) — Freebuff

### P14 — SAFETY + COMPLIANCE (thêm)
- [ ] `safety-rlhf` — RLHF / DPO alignment (train safety) — OpenCode
- [ ] `safety-guard` — Guard model (filter toxic outputs) — Freebuff
- [ ] `safety-audit` — Model audit (bias, hallucination detection) — OpenCode
- [ ] `safety-redteam` — Red-team testing suite (prompt injection) — Freebuff
- [ ] `compliance-gdpr` — GDPR compliance (data deletion, export) — Freebuff
- [ ] `compliance-audit-log` — Audit log (who queried what, when) — Freebuff
- [ ] `compliance-pii` — PII detection + redaction in outputs — Freebuff
- [ ] `compliance-logging` — Structured logging for compliance (immutable) — Freebuff

### P15 — DEVELOPER EXPERIENCE (thêm)
- [ ] `dev-mcp-server` — MCP server for Claude/other agents — Freebuff
- [ ] `dev-cli-chat` — CLI chat mode (interactive, multi-turn) — Freebuff
- [ ] `dev-ide-plugin` — IDE plugin (VS Code, IntelliJ) — Freebuff
- [ ] `dev-web-ui` — Web UI (React, chat + streaming) — Freebuff
- [ ] `dev-tui-chat` — TUI chat (ratatui, streaming) — Freebuff
- [ ] `dev-debug` — Debug mode (step through layers, inspect tensors) — OpenCode
- [ ] `dev-profile` — Profile mode (per-layer timing, memory heatmap) — OpenCode
- [ ] `dev-replay` — Replay mode (reproduce a specific generation) — Freebuff
- [ ] `dev-fuzz` — Fuzz mode (random inputs, crash detection) — Freebuff
- [ ] `dev-bench-cli` — `vipera bench` command (prefill + decode + compare) — Freebuff

### P16 — MULTI-MODAL + AUDIO (thêm)
- [ ] `mm-vit-encoder` — ViT encoder for CLIP/SigLIP (vision tokens) — OpenCode
- [ ] `mm-projector` — Vision → LLM projector (linear + MLP) — OpenCode
- [ ] `mm-llava` — LLaVA-style vision LLM support — OpenCode
- [ ] `mm-florence` — Florence-2 / Florence-3 support — OpenCode
- [ ] `mm-whisper` — Whisper ASR (audio → text) — OpenCode
- [ ] `mm-whisper-large` — Whisper-large-v3 (16 languages) — OpenCode
- [ ] `mm-tts` — Text-to-speech (Piper, Coqui TTS) — Freebuff
- [ ] `mm-speech-to-text` — Speech recognition endpoint — Freebuff
- [ ] `mm-image-gen` — Image generation (SDXL, FLUX) — OpenCode
- [ ] `mm-image-edit` — Image editing (inpainting, outpainting) — OpenCode

### P17 — MOE + DEEP (thêm)
- [ ] `moe-mixtral` — Mixtral 8x7B support (8 experts, top-2) — OpenCode
- [ ] `moe-dbae` — DBAE (Dual Block Attention Expert) — OpenCode
- [ ] `moe-llama4` — Llama 4 MoE support (expert routing) — OpenCode
- [ ] `moe-glm` — GLM-130B MoE support — OpenCode
- [ ] `moe-mpt` — MPT-30B MoE support — OpenCode
- [ ] `moe-phi-moe` — Phi-3 MoE support (8x1.6B experts) — OpenCode
- [ ] `moe-expert-merge` — Expert merging (combine 2 experts) — OpenCode
- [ ] `moe-expert-prune` — Expert pruning (remove least-used) — OpenCode
- [ ] `moe-dynamic-topk` — Dynamic top-k selection (per token) — OpenCode
- [ ] `moe-load-balance` — Load balancing (expert utilization) — Freebuff

### P18 — SPECULATIVE + DRAFT (thêm)
- [ ] `spec-ngram` — N-gram draft (no model, pure cache) — Freebuff
- [ ] `spec-eagle` — EAGLE (efficient speculative decoding) — OpenCode
- [ ] `spec-eagle2` — EAGLE-2 (multi-token prediction) — OpenCode
- [ ] `spec-lookahead` — Lookahead decoding (parallel draft) — OpenCode
- [ ] `spec-draft-verify` — Draft model verify (small model draft) — OpenCode
- [ ] `spec-rejection` — Rejection sampling (quality control) — OpenCode
- [ ] `spec-multi-seq` — Multi-sequence draft (batch draft) — OpenCode
- [ ] `spec-adaptive` — Adaptive speculation (learn which tokens to draft) — OpenCode
- [ ] `spec-batch-verify` — Batch verify (verify N drafts at once) — OpenCode
- [ ] `spec-fallback` — Fallback to greedy if draft fails — Freebuff

### P19 — QUANTIZATION + COMPRESSION (thêm)
- [ ] `quant-gptq-int4` — GPTQ 4-bit (per-channel) — OpenCode
- [ ] `quant-awq-int4` — AWQ 4-bit (activation-aware) — OpenCode
- [ ] `quant-gptv` — GPTV (VQ-VAE based) — OpenCode
- [ ] `quant-squeeze` — SqueezeLLM (1-bit + outliers) — OpenCode
- [ ] `quant-lomni` — LOMNI (1-bit quantization) — OpenCode
- [ ] `quant-mxfp8` — MXFP8 (microscaling FP8) — OpenCode
- [ ] `quant-blockfp16` — Block-FP16 (per-block scale) — OpenCode
- [ ] `quant-mix-quant` — Mixed quantization (per-layer bit) — OpenCode
- [ ] `quant-prune` — Weight pruning (magnitude + structure) — OpenCode
- [ ] `quant-distill` — Knowledge distillation (compress model) — OpenCode

### P20 — DISTRIBUTED + CLUSTER (thêm)
- [ ] `dist-tensor-parallel` — Tensor parallel (2-8 GPUs) — OpenCode
- [ ] `dist-pipeline-parallel` — Pipeline parallel (stage split) — OpenCode
- [ ] `dist-sequence-parallel` — Sequence parallel (long context) — OpenCode
- [ ] `dist-expert-parallel` — Expert parallel (MoE across GPUs) — OpenCode
- [ ] `dist-data-parallel` — Data parallel (replicate, sync grads) — OpenCode
- [ ] `dist-node` — Multi-node (NVIDIA NVLink, InfiniBand) — OpenCode
- [ ] `dist-ring` — Ring attention (infinite context) — OpenCode
- [ ] `dist-llm-cluster` — LLM cluster manager (K8s) — Freebuff
- [ ] `dist-auto-shard` — Auto-shard (model > 1 GPU) — OpenCode
- [ ] `dist-checkpoint` — Checkpoint/resume (distributed state) — Freebuff

## ĐẾM CONTRIBUTIONS

| Phase | Tasks | Contributions (×4) | Tổng |
|---|---|---|---|
| Tối nay (11h-2h) | 4 tasks | 16 | 16 |
| Sáng (2h-10h) | 12 tasks | 48 | 64 |
| Chiều (10h-18h) | 12 tasks | 48 | 112 |
| Tối (18h-2h) | 12 tasks | 48 | 160 |
| **Tổng 24h (P0-P3)** | **40 tasks** | **160** | **160** |

> **161 tasks** (P0-P15) × 1 cycle = **644 contributions**
> 161 × 3 cycles = **483 contributions** → đạt 1000 rule trong **48h (2 ngày)** ✓
> 161 × 5 cycles = **805 contributions** → đạt 1000 rule trong **3 ngày** ✓
> 161 × 7 cycles = **1127 contributions** → đạt 1000 rule trong **4 ngày** ✓

**Gợi ý: 3 cycles trong 48h → 483 contributions (an toàn cho 1000 rule)**

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
