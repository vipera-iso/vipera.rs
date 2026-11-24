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

### P21 — MODEL ARCHITECTURES (thêm)
- [ ] `arch-mistral` — Mistral 7B/Instruct support (sliding window + RoPE) — Freebuff
- [ ] `arch-gemma` — Gemma/Gemma-2 support (GQA + sliding window) — Freebuff
- [ ] `arch-phi` — Phi-3/Phi-4 support (GQA + alternating attention) — OpenCode
- [ ] `arch-command-r` — Command-R/R+ support (sliding window + GQA) — Freebuff
- [ ] `arch-olmo` — OLMo/OLMoE support (GQA + RoPE) — Freebuff
- [ ] `arch-jamba` — Jamba hybrid (Mamba + attention layers) — OpenCode
- [ ] `arch-retiflex` — Refined Flex (flex attention pattern) — OpenCode
- [ ] `arch-mamba` — Mamba2/Samba (state space model, no attention) — OpenCode
- [ ] `arch-recurrent` — RWKV/RetNet (linear attention, no KV cache) — OpenCode
- [ ] `arch-norm-v2` — RMSNorm + QK-norm variants (per-arch) — Freebuff

### P22 — LONG CONTEXT (thêm)
- [ ] `long-4096` — 4096 context (already, but optimize) — OpenCode
- [ ] `long-32k` — 32k context (sliding window + ring KV) — OpenCode
- [ ] `long-128k` — 128k context (paged KV + sliding) — OpenCode
- [ ] `long-1m` — 1M context (ring + paged + streaming) — OpenCode
- [ ] `long-rope` — RoPE extrapolation (NTK, YaRN, dynamic) — OpenCode
- [ ] `long-ring` — Ring attention (infinite, distributed) — OpenCode
- [ ] `long-paged` — Paged KV cache (vLLM-style) — OpenCode
- [ ] `long-streaming` — Streaming KV (write-once, read-many) — Freebuff
- [ ] `long-summarize` — Summarize-and-replace (compress old context) — Freebuff
- [ ] `long-hierarchical` — Hierarchical KV (fast/slow layers) — OpenCode

### P23 — TRAINING + TUNING (thêm)
- [ ] `train-lora` — LoRA training (rank 16/64, CPU) — OpenCode
- [ ] `train-qlora` — QLoRA (4-bit base + LoRA) — OpenCode
- [ ] `train-adapter` — Adapter tuning (bottleneck layers) — OpenCode
- [ ] `train-quantize` — Post-hoc quantization (calibrate + quantize) — Freebuff
- [ ] `train-dpo` — DPO (direct preference optimization) — OpenCode
- [ ] `train-orpo` — ORPO (preference optimization, no reference) — OpenCode
- [ ] `train-rlhf` — RLHF (reward model + PPO) — OpenCode
- [ ] `train-distill` — Knowledge distillation (large → small) — OpenCode
- [ ] `train-prune` — Structure pruning (remove least-used neurons) — OpenCode
- [ ] `train-merge` — Model merging (TIES, DARE, SLERP) — Freebuff

### P24 — SAFETY + ALIGNMENT (thêm)
- [ ] `align-sft` — Supervised fine-tuning (SFT on CPU) — OpenCode
- [ ] `align-ref-model` — Reference model (for DPO/RLHF) — OpenCode
- [ ] `align-reward` — Reward model (train on preference data) — OpenCode
- [ ] `align-toxicity` — Toxicity classifier (filter outputs) — Freebuff
- [ ] `align-jailbreak` — Jailbreak detection (prompt injection) — Freebuff
- [ ] `align-bias` — Bias detection (gender, race, political) — OpenCode
- [ ] `align-hallucination` — Hallucination detection (fact-check) — Freebuff
- [ ] `align-fairness` — Fairness metrics (demographic parity) — Freebuff
- [ ] `align-privacy` — Privacy (no training on user data) — Freebuff
- [ ] `align-audit` — Model card + audit report — Freebuff

### P25 — INFERENCE OPTIMIZATION (thêm)
- [ ] `opt-batch` — Dynamic batching (group requests by length) — OpenCode
- [ ] `opt-priority` — Priority scheduling (VIP requests first) — Freebuff
- [ ] `opt-queue` — Request queue (FIFO + preempt) — Freebuff
- [ ] `opt-slo` — SLO-aware scheduling (latency target) — OpenCode
- [ ] `opt-cache` — Semantic cache (similar prompts → cached answer) — Freebuff
- [ ] `opt-dedup` — Prompt deduplication (same prompt → 1 inference) — Freebuff
- [ ] `opt-predict` — Load prediction (scale up/down) — Freebuff
- [ ] `opt-autoscale` — Auto-scaling (K8s HPA + custom metric) — Freebuff
- [ ] `opt-spark` — Spark-like distributed inference — OpenCode
- [ ] `opt-mojo` — Mojo language integration (inference) — OpenCode

### P26 — DATA + EMBEDDING (thêm)
- [ ] `data-vector-db` — Vector database (FAISS, ScaNN) integration — Freebuff
- [ ] `data-chunk` — Text chunking (sliding window, semantic) — Freebuff
- [ ] `data-embed` — Embedding model (BGE, E5, MTEB) — OpenCode
- [ ] `data-rerank` — Reranker (Cross-Encoder, MonoT5) — OpenCode
- [ ] `data-train` — Training data pipeline (clean, label, split) — Freebuff
- [ ] `data-eval` — Eval pipeline (MMLU, HellaSwag, TruthfulQA) — Freebuff
- [ ] `data-finetune` — Fine-tune dataset format (JSONL) — Freebuff
- [ ] `data-preference` — Preference dataset (DPO, ORPO) — Freebuff
- [ ] `data-reward` — Reward dataset (score, preference) — Freebuff
- [ ] `data-leaderboard` — Auto-leaderboard (MMLU, GSM8K, HumanEval) — Freebuff

### P27 — DEPLOYMENT + OPS (thêm)
- [ ] `ops-monitor` — Prometheus + Grafana dashboard — Freebuff
- [ ] `ops-alert` — Alerting (latency, error rate, memory) — Freebuff
- [ ] `ops-logging` — Structured logging (JSON, ELK) — Freebuff
- [ ] `ops-tracing` — Distributed tracing (Jaeger, Zipkin) — Freebuff
- [ ] `ops-chaos` — Chaos engineering (fault injection) — Freebuff
- [ ] `ops-backup` — Model backup (snapshot + restore) — Freebuff
- [ ] `ops-rollback` — Rollback (version pin + instant switch) — Freebuff
- [ ] `ops-blue-green` — Blue-green deployment (zero-downtime) — Freebuff
- [ ] `ops-canary` — Canary release (10% traffic, monitor) — Freebuff
- [ ] `ops-slo` — SLO monitoring (99.9% availability) — Freebuff

### P28 — RESEARCH + EXPERIMENTS (thêm)
- [ ] `research-attention` — New attention variants (linear, log) — OpenCode
- [ ] `research-moe` — New MoE routing (expert choice) — OpenCode
- [ ] `research-quant` — New quantization (4-bit, 2-bit, 1-bit) — OpenCode
- [ ] `research-spec` — New speculative (EAGLE, MTP, DFlash) — OpenCode
- [ ] `research-distill` — New distillation (progressive, self) — OpenCode
- [ ] `research-alignment` — New alignment (DPO, ORPO, RLOO) — OpenCode
- [ ] `research-benchmark` — New benchmark (GSM8K, MMLU, HumanEval) — Freebuff
- [ ] `research-interpret` — Interpretability (activation, attention map) — OpenCode
- [ ] `research-safety` — Safety (jailbreak, toxicity, bias) — Freebuff
- [ ] `research-efficiency` — Efficiency (FLOPs, memory, energy) — Freebuff

### P29 — HARDWARE + BACKEND (thêm)
- [ ] `hw-amd` — ROCm/AMD GPU support (hipblas, ck) — OpenCode
- [ ] `hw-intel` — Intel GPU (oneAPI, SYCL) — OpenCode
- [ ] `hw-npu` — NPU support (Intel Movidius, Apple Neural Engine) — OpenCode
- [ ] `hw-riscv` — RISC-V backend (vector extension) — OpenCode
- [ ] `hw-sparc` — SPARC/ARM backend (aarch64 NEON) — Freebuff
- [ ] `hw-raspi` — Raspberry Pi (ARM NEON, 4GB) — Freebuff
- [ ] `hw-jetson` — NVIDIA Jetson (CUDA + int8) — Freebuff
- [ ] `hw-tpu` — Google TPU (XLA, JAX) — OpenCode
- [ ] `hw-webgpu` — WebGPU (browser inference) — OpenCode
- [ ] `hw-wasm` — WASM (browser + edge) — OpenCode

### P30 — LANGUAGE + INTEROP (thêm)
- [ ] `lang-python` — Python API (vipera-py, PyTorch integration) — OpenCode
- [ ] `lang-javascript` — JS/TS API (Node.js + browser) — Freebuff
- [ ] `lang-go` — Go API (cgo + bindings) — Freebuff
- [ ] `lang-java` — Java/Kotlin API (JNI + Android) — Freebuff
- [ ] `lang-swift` — Swift API (iOS + macOS) — Freebuff
- [ ] `lang-csharp` — C#/.NET API (P/Invoke) — Freebuff
- [ ] `lang-r` — R API (stat + ML integration) — Freebuff
- [ ] `lang-matlab` — MATLAB API (C interface) — Freebuff
- [ ] `lang-ml` — ML framework interop (PyTorch, JAX, TF) — OpenCode
- [ ] `lang-cuda` — CUDA C++ API (native kernel access) — OpenCode

### P31 — SECURITY + HARDENING (thêm)
- [ ] `sec-memory` — Memory safety (no overflow, no UAF) — OpenCode
- [ ] `sec-input` — Input validation (malformed GGUF, huge prompt) — Freebuff
- [ ] `sec-output` — Output sanitization (no prompt injection) — Freebuff
- [ ] `sec-auth` — Authentication (API key, JWT, OAuth2) — Freebuff
- [ ] `sec-tls` — TLS 1.3 (mutual auth, cert pinning) — Freebuff
- [ ] `sec-audit` — Audit log (who, what, when, hash) — Freebuff
- [ ] `sec-fuzz` — Fuzz testing (GGUF, tokenizer, server) — Freebuff
- [ ] `sec-deps` — Dependency audit (cargo audit, SCA) — Freebuff
- [ ] `sec-supply` — Supply chain (SBOM, reproducible build) — Freebuff
- [ ] `sec-hard` — Hardening (ASLR, DEP, NX, stack canary) — OpenCode

### P32 — PERFORMANCE + TUNING (thêm)
- [ ] `perf-flame` — Flame graph generation (perf + FlameGraph) — OpenCode
- [ ] `perf-gprof` — gprof-style profiling (function level) — OpenCode
- [ ] `perf-tune` — Auto-tuning (thread count, tile size, batch) — OpenCode
- [ ] `perf-simd` — SIMD autotuning (AVX2/AVX512/NEON selection) — OpenCode
- [ ] `perf-memory` — Memory profiling (peak, RSS, allocation) — Freebuff
- [ ] `perf-bottleneck` — Bottleneck detection (CPU vs memory vs disk) — Freebuff
- [ ] `perf-scaling` — Scaling test (1-128 threads, Amdahl) — Freebuff
- [ ] `perf-thermal` — Thermal throttling monitor (CPU temp vs perf) — Freebuff
- [ ] `perf-power` — Power consumption (watt-meter integration) — Freebuff
- [ ] `perf-roofline` — Roofline model (FLOPs vs bandwidth) — OpenCode

### P33 — MODEL MANAGEMENT (thêm)
- [ ] `mm-hub` — Model hub (download, list, search, cache) — Freebuff
- [ ] `mm-registry` — Model registry (version, tag, lineage) — Freebuff
- [ ] `mm-version` — Model versioning (semantic, rollback) — Freebuff
- [ ] `mm-share` — Model sharing (team, org, public) — Freebuff
- [ ] `mm-convert` — Model conversion (GGUF ↔ safetensors ↔ ONNX) — OpenCode
- [ ] `mm-optimize` — Model optimization (prune, quantize, distill) — OpenCode
- [ ] `mm-benchmark` — Model benchmark (accuracy vs speed vs memory) — Freebuff
- [ ] `mm-compare` — Model comparison (side-by-side, A/B) — Freebuff
- [ ] `mm-archive` — Model archive (cold storage, restore) — Freebuff
- [ ] `mm-lifecycle` — Model lifecycle (create, train, serve, retire) — Freebuff

### P34 — USER INTERFACE (thêm)
- [ ] `ui-chat` — Chat interface (terminal, web, mobile) — Freebuff
- [ ] `ui-stream` — Streaming UI (token-by-token, live) — Freebuff
- [ ] `ui-voice` — Voice UI (TTS + STT, hands-free) — Freebuff
- [ ] `ui-canvas` — Canvas UI (drawing + explanation) — Freebuff
- [ ] `ui-code` — Code UI (editor + autocomplete) — Freebuff
- [ ] `ui-table` — Table UI (structured data + query) — Freebuff
- [ ] `ui-chart` — Chart UI (visualization + insight) — Freebuff
- [ ] `ui-form` — Form UI (fill + validate + submit) — Freebuff
- [ ] `ui-game` — Game UI (interactive + AI opponent) — Freebuff
- [ ] `ui-custom` — Custom UI (plugin + extension) — Freebuff

### P35 — API + INTEGRATION (thêm)
- [ ] `api-openai` — OpenAI-compatible API (chat, completions, models) — Freebuff
- [ ] `api-anthropic` — Anthropic-compatible API (Claude) — Freebuff
- [ ] `api-mistral` — Mistral-compatible API — Freebuff
- [ ] `api-llama` — llama.cpp-compatible API (llama-server) — Freebuff
- [ ] `api-gRPC` — gRPC API (protobuf, streaming) — OpenCode
- [ ] `api-webhook` — Webhook API (trigger on event) — Freebuff
- [ ] `api-batch` — Batch API (async, large-scale) — Freebuff
- [ ] `api-stream` — Stream API (SSE, WebSocket) — Freebuff
- [ ] `api-auth` — Auth API (API key, JWT, OAuth2) — Freebuff
- [ ] `api-rate` — Rate-limit API (per-key, per-IP) — Freebuff

### P36 — DATA + PIPELINE (thêm)
- [ ] `data-ingest` — Data ingestion (CSV, JSON, Parquet, DB) — Freebuff
- [ ] `data-clean` — Data cleaning (dedup, normalize, fill) — Freebuff
- [ ] `data-label` — Data labeling (auto + active) — Freebuff
- [ ] `data-aug` — Data augmentation (synthetic, back-translation) — OpenCode
- [ ] `data-split` — Data split (train, val, test, held-out) — Freebuff
- [ ] `data-version` — Data versioning (DVC, LakeFS) — Freebuff
- [ ] `data-lineage` — Data lineage (track, audit, provenance) — Freebuff
- [ ] `data-quality` — Data quality (completeness, accuracy, freshness) — Freebuff
- [ ] `data-security` — Data security (encryption, masking, access) — Freebuff
- [ ] `data-compliance` — Data compliance (GDPR, HIPAA, CCPA) — Freebuff

### P37 — MONITORING + OBSERVABILITY (thêm)
- [ ] `mon-metrics` — Metrics (Prometheus, OpenTelemetry) — Freebuff
- [ ] `mon-logs` — Logs (structured, ELK, Loki) — Freebuff
- [ ] `mon-traces` — Traces (Jaeger, Zipkin, Tempo) — Freebuff
- [ ] `mon-alert` — Alerts (PagerDuty, Slack, email) — Freebuff
- [ ] `mon-dash` — Dashboards (Grafana, Kibana) — Freebuff
- [ ] `mon-slo` — SLO (error budget, burn rate) — Freebuff
- [ ] `mon-chaos` — Chaos (failure injection, resilience) — Freebuff
- [ ] `mon-audit` — Audit (who, what, when, why) — Freebuff
- [ ] `mon-cost` — Cost (CPU, GPU, memory, per-request) — Freebuff
- [ ] `mon-perf` — Perf (latency, throughput, saturation) — Freebuff

### P38 — TESTING + QA (thêm)
- [ ] `test-unit` — Unit tests (kernel, module, function) — Freebuff
- [ ] `test-integration` — Integration tests (API, pipeline) — Freebuff
- [ ] `test-e2e` — End-to-end tests (full flow) — Freebuff
- [ ] `test-load` — Load tests (10k req/min, sustained) — Freebuff
- [ ] `test-stress` — Stress tests (overload, failover) — Freebuff
- [ ] `test-soak` — Soak tests (24h, memory leak) — Freebuff
- [ ] `test-fuzz` — Fuzz tests (random input, crash) — Freebuff
- [ ] `test-a/b` — A/B tests (model, prompt, strategy) — Freebuff
- [ ] `test-regression` — Regression tests (baseline, compare) — Freebuff
- [ ] `test-coverage` — Coverage (unit, integration, e2e) — Freebuff

### P39 — DOCS + TUTORIALS (thêm)
- [ ] `docs-tutorial` — Tutorial (beginner, 1-hour) — Freebuff
- [ ] `docs-guide` — Guide (intermediate, 4-hour) — Freebuff
- [ ] `docs-advanced` — Advanced (expert, 8-hour) — Freebuff
- [ ] `docs-api` — API reference (complete, examples) — Freebuff
- [ ] `docs-faq` — FAQ (top 50 questions) — Freebuff
- [ ] `docs-video` — Video tutorial (YouTube, 10-min) — Freebuff
- [ ] `docs-workshop` — Workshop (hands-on, 2-hour) — Freebuff
- [ ] `docs-community` — Community (forum, Discord, Slack) — Freebuff
- [ ] `docs-blog` — Blog (update, release, feature) — Freebuff
- [ ] `docs-contrib` — Contributing (guide, template) — Freebuff

### P40 — COMMUNITY + ECOSYSTEM (thêm)
- [ ] `comm-forum` — Forum (Q&A, help, share) — Freebuff
- [ ] `comm-discord` — Discord (chat, community) — Freebuff
- [ ] `comm-slack` — Slack (team, enterprise) — Freebuff
- [ ] `comm-twitter` — Twitter/X (update, engagement) — Freebuff
- [ ] `comm-github` — GitHub (issue, PR, action) — Freebuff
- [ ] `comm-model` — Model community (share, discuss) — Freebuff
- [ ] `comm-plugin` — Plugin community (build, share) — Freebuff
- [ ] `comm-hackathon` — Hackathon (challenge, prize) — Freebuff
- [ ] `comm-contribute` — Contribution (code, doc, test) — Freebuff
- [ ] `comm-maintain` — Maintenance (fix, update, support) — Freebuff

### P41 — AGENT + AUTONOMOUS (thêm)
- [ ] `agent-tool-use` — Tool calling (function, JSON schema) — OpenCode
- [ ] `agent-multi-step` — Multi-step reasoning (plan, execute, verify) — OpenCode
- [ ] `agent-memory` — Agent memory (short-term + long-term, vector) — OpenCode
- [ ] `agent-reflection` — Self-reflection (critique + improve) — OpenCode
- [ ] `agent-multi-agent` — Multi-agent (debate, vote, specialize) — OpenCode
- [ ] `agent-critic` — Critic model (reject + regenerate) — OpenCode
- [ ] `agent-verifier` — Verifier (check output, math, code) — OpenCode
- [ ] `agent-planner` — Planner (decompose, schedule, execute) — OpenCode
- [ ] `agent-executor` — Executor (run tool, parse result) — OpenCode
- [ ] `agent-orchestrator` — Orchestrator (route, parallel, merge) — OpenCode

### P42 — RAG + RETRIEVAL (thêm)
- [ ] `rag-hybrid` — Hybrid retrieval (BM25 + vector) — OpenCode
- [ ] `rag-graph` — GraphRAG (entity + relation) — OpenCode
- [ ] `rag-hierarchical` — HieraRAG (multi-level chunk) — Freebuff
- [ ] `rag-temporal` — TemporalRAG (time-aware) — Freebuff
- [ ] `rag-multi-doc` — Multi-document (cross-ref) — Freebuff
- [ ] `rag-citation` — Citation (source + span) — Freebuff
- [ ] `rag-eval` — RAG eval (retrieval + generation) — Freebuff
- [ ] `rag-cache` — RAG cache (query + context) — Freebuff
- [ ] `rag-update` — RAG update (incremental index) — Freebuff
- [ ] `rag-privacy` — RAG privacy (PII, access control) — Freebuff

### P43 — CODE + AGENTIC (thêm)
- [ ] `code-gen` — Code generation (Python, Rust, C++) — OpenCode
- [ ] `code-review` — Code review (bug, style, perf) — OpenCode
- [ ] `code-test` — Auto-test (unit, integration) — Freebuff
- [ ] `code-debug` — Debug (root cause, fix) — OpenCode
- [ ] `code-refactor` — Refactor (clean, perf) — OpenCode
- [ ] `code-doc` — Auto-doc (docstring, comment) — Freebuff
- [ ] `code-security` — Security scan (SAST, DAST) — Freebuff
- [ ] `code-dependency` — Dependency update (SemVer) — Freebuff
- [ ] `code-ci` — CI pipeline (build, test, deploy) — Freebuff
- [ ] `code-migration` — Migration (legacy → new) — OpenCode

### P44 — SCIENTIFIC + DOMAIN (thêm)
- [ ] `sci-math` — Math (symbolic, numeric) — OpenCode
- [ ] `sci-physics` — Physics (simulation, FEA) — OpenCode
- [ ] `sci-chem` — Chemistry (molecular, reaction) — OpenCode
- [ ] `sci-bio` — Biology (protein, genome) — OpenCode
- [ ] `sci-medical` — Medical (diagnosis, drug) — Freebuff
- [ ] `sci-finance` — Finance (risk, portfolio) — Freebuff
- [ ] `sci-climate` — Climate (forecast, model) — Freebuff
- [ ] `sci-engineering` — Engineering (CAD, FEA) — Freebuff
- [ ] `sci-research` — Research (literature, experiment) — Freebuff
- [ ] `sci-data` — Data (ETL, BI, dashboard) — Freebuff

### P45 — EDGE + EMBEDDED (thêm)
- [ ] `edge-raspberry` — Raspberry Pi (ARM NEON, 4GB) — Freebuff
- [ ] `edge-jetson` — NVIDIA Jetson (CUDA, int8) — Freebuff
- [ ] `edge-micro` — Microcontroller (ESP32, STM32) — Freebuff
- [ ] `edge-wasm` — WASM (browser, edge) — OpenCode
- [ ] `edge-webgpu` — WebGPU (browser GPU) — OpenCode
- [ ] `edge-quant` — Edge quantization (2-bit, 4-bit) — OpenCode
- [ ] `edge-batch` — Edge batch (inference + train) — OpenCode
- [ ] `edge-sync` — Edge sync (cloud ↔ edge) — Freebuff
- [ ] `edge-update` — Edge update (OTA, model) — Freebuff
- [ ] `edge-security` — Edge security (key, cert) — Freebuff

### P46 — CLOUD + SCALED (thêm)
- [ ] `cloud-aws` — AWS (SageMaker, EC2, S3) — Freebuff
- [ ] `cloud-gcp` — GCP (Vertex, GKE, GCS) — Freebuff
- [ ] `cloud-azure` — Azure (ML, AKS, Blob) — Freebuff
- [ ] `cloud-k8s` — Kubernetes (Helm, operator) — Freebuff
- [ ] `cloud-serverless` — Serverless (Lambda, Cloud Functions) — Freebuff
- [ ] `cloud-batch` — Batch (Spark, Ray, Dask) — Freebuff
- [ ] `cloud-mlflow` — MLflow (track, version) — Freebuff
- [ ] `cloud-pipeline` — Pipeline (Airflow, Argo) — Freebuff
- [ ] `cloud-monitor` — Monitor (CloudWatch, GCP Monitoring) — Freebuff
- [ ] `cloud-cost` — Cost (FinOps, tag) — Freebuff

### P47 — ENTERPRISE (thêm)
- [ ] `ent-sso` — SSO (SAML, OIDC) — Freebuff
- [ ] `ent-scm` — SCM (Active Directory, LDAP) — Freebuff
- [ ] `ent-gov` — Governance (policy, role) — Freebuff
- [ ] `ent-audit` — Audit (log, report) — Freebuff
- [ ] `ent-billing` — Billing (usage, invoice) — Freebuff
- [ ] `ent-support` — Support (ticket, chat) — Freebuff
- [ ] `ent-sla` — SLA (uptime, response) — Freebuff
- [ ] `ent-iso` — ISO 27001 (security) — Freebuff
- [ ] `ent-soc2` — SOC2 (compliance) — Freebuff
- [ ] `ent-hipaa` — HIPAA (medical) — Freebuff

### P48 — RESEARCH FRONTIER (thêm)
- [ ] `frontier-transformer` — Transformer v2 (linear, log) — OpenCode
- [ ] `frontier-moe` — MoE v2 (expert choice, dynamic) — OpenCode
- [ ] `frontier-longctx` — Long context (1M, 10M) — OpenCode
- [ ] `frontier-multimodal` — Multi-modal (text, image, audio, video) — OpenCode
- [ ] `frontier-world` — World model (physics, simulation) — OpenCode
- [ ] `frontier-robotics` — Robotics (vision, control) — OpenCode
- [ ] `frontier-science` — AI for science (material, drug) — OpenCode
- [ ] `frontier-math` — AI for math (theorem, proof) — OpenCode
- [ ] `frontier-alignment` — Alignment (safety, value) — OpenCode
- [ ] `frontier-scaling` — Scaling law (compute, data, model) — OpenCode

### P49 — AUTOMATION + WORKFLOW (thêm)
- [ ] `auto-pipeline` — Data pipeline (ingest → transform → serve) — Freebuff
- [ ] `auto-feature` — Feature store (compute, version, lineage) — Freebuff
- [ ] `auto-model` — Auto-model (select, tune, deploy) — Freebuff
- [ ] `auto-eval` — Auto-eval (benchmark, regression) — Freebuff
- [ ] `auto-rollback` — Auto-rollback (model, config) — Freebuff
- [ ] `auto-scale` — Auto-scale (CPU, GPU, memory) — Freebuff
- [ ] `auto-update` — Auto-update (model, dependency) — Freebuff
- [ ] `auto-failover` — Auto-failover (node, region) — Freebuff
- [ ] `auto-heal` — Auto-heal (detect, diagnose, fix) — Freebuff
- [ ] `auto-learn` — Auto-learn (online, incremental) — Freebuff

### P50 — VISUALIZATION + INSIGHT (thêm)
- [ ] `viz-attention` — Attention map (heatmap, flow) — Freebuff
- [ ] `viz-activation` — Activation map (layer, neuron) — Freebuff
- [ ] `viz-gradient` — Gradient flow (backprop) — Freebuff
- [ ] `viz-loss` — Loss curve (train, val) — Freebuff
- [ ] `viz-perf` — Perf profile (CPU, GPU, memory) — Freebuff
- [ ] `viz-data` — Data viz (distribution, correlation) — Freebuff
- [ ] `viz-model` — Model viz (architecture, weight) — Freebuff
- [ ] `viz-compare` — Compare (model, quant, prompt) — Freebuff
- [ ] `viz-debug` — Debug viz (step, inspect) — Freebuff
- [ ] `viz-report` — Report (auto, HTML, PDF) — Freebuff

### P51 — AUTONOMOUS SYSTEMS (thêm)
- [ ] `auto-scheduler` — Auto-scheduler (task, resource) — Freebuff
- [ ] `auto-router` — Auto-router (model, endpoint) — Freebuff
- [ ] `auto-balancer` — Auto-balancer (load, priority) — Freebuff
- [ ] `auto-monitor` — Auto-monitor (metric, alert) — Freebuff
- [ ] `auto-heal` — Auto-heal (failover, retry) — Freebuff
- [ ] `auto-learn` — Auto-learn (online, adaptive) — Freebuff
- [ ] `auto-optimize` — Auto-optimize (hyperparam, resource) — Freebuff
- [ ] `auto-secure` — Auto-secure (scan, patch) — Freebuff
- [ ] `auto-comply` — Auto-comply (policy, rule) — Freebuff
- [ ] `auto-improve` — Auto-improve (feedback, iterate) — Freebuff

### P52 — METRICS + KPI (thêm)
- [ ] `kpi-latency` — Latency (p50, p95, p99) — Freebuff
- [ ] `kpi-throughput` — Throughput (req/sec, token/sec) — Freebuff
- [ ] `kpi-accuracy` — Accuracy (task, domain) — Freebuff
- [ ] `kpi-efficiency` — Efficiency (FLOP/token, W/token) — Freebuff
- [ ] `kpi-cost` — Cost ($/token, $/query) — Freebuff
- [ ] `kpi-satisfaction` — Satisfaction (user, rating) — Freebuff
- [ ] `kpi-reliability` — Reliability (uptime, error rate) — Freebuff
- [ ] `kpi-fairness` — Fairness (bias, disparity) — Freebuff
- [ ] `kpi-safety` — Safety (toxicity, jailbreak) — Freebuff
- [ ] `kpi-impact` — Impact (social, environmental) — Freebuff

### P53 — GOVERNANCE + POLICY (thêm)
- [ ] `gov-model` — Model governance (version, audit) — Freebuff
- [ ] `gov-data` — Data governance (quality, access) — Freebuff
- [ ] `gov-prompt` — Prompt governance (template, policy) — Freebuff
- [ ] `gov-output` — Output governance (filter, log) — Freebuff
- [ ] `gov-access` — Access governance (role, scope) — Freebuff
- [ ] `gov-audit` — Audit governance (log, report) — Freebuff
- [ ] `gov-policy` — Policy governance (rule, enforce) — Freebuff
- [ ] `gov-compliance` — Compliance governance (regulation) — Freebuff
- [ ] `gov-risk` — Risk governance (assess, mitigate) — Freebuff
- [ ] `gov-ethics` — Ethics governance (value, principle) — Freebuff

### P54 — EXPERIMENT + A/B (thêm)
- [ ] `exp-design` — Experiment design (hypothesis, variant) — Freebuff
- [ ] `exp-sample` — Sample size (power, significance) — Freebuff
- [ ] `exp-random` — Randomize (user, request) — Freebuff
- [ ] `exp-track` — Track (metric, segment) — Freebuff
- [ ] `exp-analyze` — Analyze (statistical, causal) — Freebuff
- [ ] `exp-report` — Report (result, insight) — Freebuff
- [ ] `exp-deploy` — Deploy (winner, rollout) — Freebuff
- [ ] `exp-iterate` — Iterate (learn, improve) — Freebuff
- [ ] `exp-multi` — Multi-armed bandit (exploration) — Freebuff
- [ ] `exp-offline` — Offline eval (dataset, replay) — Freebuff

### P55 — OPS + RUNBOOK (thêm)
- [ ] `runbook-deploy` — Deploy (build, test, release) — Freebuff
- [ ] `runbook-rollback` — Rollback (version, config) — Freebuff
- [ ] `runbook-failover` — Failover (node, region) — Freebuff
- [ ] `runbook-scale` — Scale (up, down, out) — Freebuff
- [ ] `runbook-update` — Update (model, patch) — Freebuff
- [ ] `runbook-backup` — Backup (snapshot, restore) — Freebuff
- [ ] `runbook-migrate` — Migrate (data, model) — Freebuff
- [ ] `runbook-disaster` — Disaster recovery (RTO, RPO) — Freebuff
- [ ] `runbook-incident` — Incident (detect, respond, resolve) — Freebuff
- [ ] `runbook-postmortem` — Post-mortem (root cause, action) — Freebuff

### P56 — LLM SPECIFIC (thêm)
- [ ] `llm-prompt` — Prompt engineering (template, chain) — Freebuff
- [ ] `llm-chain` — Chain (ReAct, CoT, self-ask) — Freebuff
- [ ] `llm-tool` — Tool use (function, API) — Freebuff
- [ ] `llm-memory` — Memory (short-term, long-term) — Freebuff
- [ ] `llm-agent` — Agent (plan, act, observe) — Freebuff
- [ ] `llm-rag` — RAG (retrieve, augment) — Freebuff
- [ ] `llm-finetune` — Fine-tune (LoRA, QLoRA) — Freebuff
- [ ] `llm-eval` — Eval (MMLU, GSM8K, HumanEval) — Freebuff
- [ ] `llm-safe` — Safety (jailbreak, toxicity) — Freebuff
- [ ] `llm-cost` — Cost (token, query, cache) — Freebuff

### P57 — BENCHMARK + LEADERBOARD (thêm)
- [ ] `bench-mmlu` — MMLU (57 subjects) — Freebuff
- [ ] `bench-gsm8k` — GSM8K (math, grade school) — Freebuff
- [ ] `bench-humaneval` — HumanEval (code, Python) — Freebuff
- [ ] `bench-math` — Math (GSM8K, MATH) — Freebuff
- [ ] `bench-code` — Code (HumanEval, MBPP) — Freebuff
- [ ] `bench-chat` — Chat (MT-Bench, Alpaca) — Freebuff
- [ ] `bench-long` — Long context (128k, 1M) — Freebuff
- [ ] `bench-multimodal` — Multi-modal (MMMU, MMBench) — Freebuff
- [ ] `bench-agentic` — Agentic (SWE-bench, AgentBench) — Freebuff
- [ ] `bench-custom` — Custom (domain, task) — Freebuff

### P58 — SAFETY + RED TEAM (thêm)
- [ ] `safety-jailbreak` — Jailbreak (prompt injection) — Freebuff
- [ ] `safety-toxicity` — Toxicity (hate, violence) — Freebuff
- [ ] `safety-hallucination` — Hallucination (fact-check) — Freebuff
- [ ] `safety-bias` — Bias (gender, race, political) — Freebuff
- [ ] `safety-privacy` — Privacy (PII, data leak) — Freebuff
- [ ] `safety-redteam` — Red team (attack, defense) — Freebuff
- [ ] `safety-audit` — Audit (model, system) — Freebuff
- [ ] `safety-report` — Report (finding, fix) — Freebuff
- [ ] `safety-monitor` — Monitor (continuous, real-time) — Freebuff
- [ ] `safety-train` — Train (safety, alignment) — Freebuff

### P59 — COST + FINOPS (thêm)
- [ ] `cost-track` — Track (per model, per user) — Freebuff
- [ ] `cost-forecast` — Forecast (trend, seasonality) — Freebuff
- [ ] `cost-optimize` — Optimize (cache, quant) — Freebuff
- [ ] `cost-budget` — Budget (cap, alert) — Freebuff
- [ ] `cost-charge` — Charge (usage, invoice) — Freebuff
- [ ] `cost-report` — Report (team, project) — Freebuff
- [ ] `cost-compare` — Compare (provider, model) — Freebuff
- [ ] `cost-roi` — ROI (value, cost) — Freebuff
- [ ] `cost-savings` — Savings (automate, optimize) — Freebuff
- [ ] `cost-sustainability` — Sustainability (carbon, energy) — Freebuff

### P60 — SUSTAINABILITY + GREEN AI (thêm)
- [ ] `green-carbon` — Carbon (per query, per model) — Freebuff
- [ ] `green-energy` — Energy (per token, per GPU) — Freebuff
- [ ] `green-water` — Water (cooling, data center) — Freebuff
- [ ] `green-waste` — Waste (e-waste, hardware) — Freebuff
- [ ] `green-reuse` — Reuse (model, data) — Freebuff
- [ ] `green-efficient` — Efficient (smaller, faster) — Freebuff
- [ ] `green-offset` — Offset (carbon, tree) — Freebuff
- [ ] `green-report` — Report (ESG, sustainability) — Freebuff
- [ ] `green-target` — Target (net-zero, SBTi) — Freebuff
- [ ] `green-audit` — Audit (sustainability, impact) — Freebuff

### P61 — PROMPT + CONTEXT ENGINEERING (thêm)
- [ ] `prompt-template` — Prompt template (Jinja, mustache) — Freebuff
- [ ] `prompt-chain` — Prompt chain (CoT, ReAct, self-ask) — Freebuff
- [ ] `prompt-fewshot` — Few-shot (example, in-context) — Freebuff
- [ ] `prompt-system` — System prompt (role, constraint) — Freebuff
- [ ] `prompt-context` — Context window (manage, compress) — OpenCode
- [ ] `prompt-rag` — RAG prompt (retrieve + generate) — Freebuff
- [ ] `prompt-tool` — Tool prompt (function, API) — Freebuff
- [ ] `prompt-multiturn` — Multi-turn (dialog, memory) — Freebuff
- [ ] `prompt-eval` — Prompt eval (score, rank) — Freebuff
- [ ] `prompt-optimize` — Prompt optimize (search, evolve) — OpenCode

### P62 — CONTEXT + MEMORY (thêm)
- [ ] `ctx-short` — Short-term (session, turn) — Freebuff
- [ ] `ctx-long` — Long-term (persistent, vector) — Freebuff
- [ ] `ctx-episodic` — Episodic (event, timeline) — Freebuff
- [ ] `ctx-semantic` — Semantic (concept, relation) — Freebuff
- [ ] `ctx-procedural` — Procedural (skill, workflow) — Freebuff
- [ ] `ctx-working` — Working (current, active) — Freebuff
- [ ] `ctx-compress` — Compress (summarize, truncate) — OpenCode
- [ ] `ctx-retrieve` — Retrieve (search, relevance) — Freebuff
- [ ] `ctx-update` — Update (incremental, decay) — Freebuff
- [ ] `ctx-audit` — Audit (access, modify) — Freebuff

### P63 — TOOL + ACTION (thêm)
- [ ] `tool-search` — Search (web, local, vector) — Freebuff
- [ ] `tool-code` — Code (exec, test, lint) — OpenCode
- [ ] `tool-file` — File (read, write, edit) — Freebuff
- [ ] `tool-api` — API (REST, gRPC, GraphQL) — Freebuff
- [ ] `tool-db` — Database (SQL, NoSQL, vector) — Freebuff
- [ ] `tool-browser` — Browser (navigate, interact) — Freebuff
- [ ] `tool-shell` — Shell (exec, pipe, process) — Freebuff
- [ ] `tool-vision` — Vision (OCR, detection) — OpenCode
- [ ] `tool-audio` — Audio (TTS, STT) — Freebuff
- [ ] `tool-custom` — Custom (plugin, extension) — Freebuff

### P64 — PLANNING + REASONING (thêm)
- [ ] `plan-decompose` — Decompose (task, subtask) — OpenCode
- [ ] `plan-sequence` — Sequence (order, dependency) — OpenCode
- [ ] `plan-parallel` — Parallel (independent, concurrent) — OpenCode
- [ ] `plan-adaptive` — Adaptive (feedback, replan) — OpenCode
- [ ] `plan-goal` — Goal (hierarchy, constraint) — OpenCode
- [ ] `plan-search` — Search (BFS, DFS, A*) — OpenCode
- [ ] `plan-optimize` — Optimize (cost, time, resource) — OpenCode
- [ ] `plan-risk` — Risk (failure, mitigation) — OpenCode
- [ ] `plan-checkpoint` — Checkpoint (save, resume) — OpenCode
- [ ] `plan-rollback` — Rollback (undo, revert) — OpenCode

### P65 — VERIFICATION + EVAL (thêm)
- [ ] `eval-accuracy` — Accuracy (task, domain) — Freebuff
- [ ] `eval-calibration` — Calibration (confidence, probability) — Freebuff
- [ ] `eval-robustness` — Robustness (noise, perturbation) — Freebuff
- [ ] `eval-fairness` — Fairness (bias, disparity) — Freebuff
- [ ] `eval-safety` — Safety (toxicity, jailbreak) — Freebuff
- [ ] `eval-efficiency` — Efficiency (FLOP, memory, time) — Freebuff
- [ ] `eval-usability` — Usability (UX, satisfaction) — Freebuff
- [ ] `eval-reproducibility` — Reproducibility (seed, version) — Freebuff
- [ ] `eval-generalization` — Generalization (out-of-domain) — Freebuff
- [ ] `eval-interpretability` — Interpretability (explain, reason) — Freebuff

### P66 — SECURITY + PRIVACY (thêm)
- [ ] `sec-input` — Input (validate, sanitize) — Freebuff
- [ ] `sec-output` — Output (filter, log) — Freebuff
- [ ] `sec-auth` — Auth (API key, JWT, OAuth2) — Freebuff
- [ ] `sec-tls` — TLS (cert, pin, rotate) — Freebuff
- [ ] `sec-audit` — Audit (log, report) — Freebuff
- [ ] `sec-privacy` — Privacy (PII, GDPR, CCPA) — Freebuff
- [ ] `sec-encryption` — Encryption (data, model) — Freebuff
- [ ] `sec-isolation` — Isolation (sandbox, namespace) — Freebuff
- [ ] `sec-supply` — Supply chain (SBOM, reproducible) — Freebuff
- [ ] `sec-resilience` — Resilience (DDoS, injection) — Freebuff

### P67 — COST + EFFICIENCY (thêm)
- [ ] `cost-token` — Token (input, output) — Freebuff
- [ ] `cost-cache` — Cache (hit, miss, invalidation) — Freebuff
- [ ] `cost-quant` — Quantize (4-bit, 8-bit, 16-bit) — OpenCode
- [ ] `cost-prune` — Prune (weight, neuron) — OpenCode
- [ ] `cost-distill` — Distill (large → small) — OpenCode
- [ ] `cost-batch` — Batch (size, throughput) — Freebuff
- [ ] `cost-spec` — Speculative (draft, verify) — OpenCode
- [ ] `cost-moe` — MoE (expert, routing) — OpenCode
- [ ] `cost-long` — Long context (KV, compress) — OpenCode
- [ ] `cost-edge` — Edge (mobile, embedded) — Freebuff

### P68 — QUALITY + RELIABILITY (thêm)
- [ ] `qual-code` — Code (test, lint, review) — Freebuff
- [ ] `qual-data` — Data (clean, validate, version) — Freebuff
- [ ] `qual-model` — Model (eval, monitor, update) — Freebuff
- [ ] `qual-pipeline` — Pipeline (CI/CD, deploy) — Freebuff
- [ ] `qual-observability` — Observability (metric, log, trace) — Freebuff
- [ ] `qual-resilience` — Resilience (failover, retry) — Freebuff
- [ ] `qual-performance` — Performance (latency, throughput) — Freebuff
- [ ] `qual-availability` — Availability (uptime, SLO) — Freebuff
- [ ] `qual-compatibility` — Compatibility (version, platform) — Freebuff
- [ ] `qual-documentation` — Documentation (API, user, dev) — Freebuff

### P69 — INNOVATION + FUTURE (thêm)
- [ ] `innov-architecture` — Architecture (Mamba, RWKV, Hyena) — OpenCode
- [ ] `innov-quant` — Quantization (1-bit, 2-bit, mixed) — OpenCode
- [ ] `innov-spec` — Speculative (EAGLE, MTP, DFlash) — OpenCode
- [ ] `innov-moe` — MoE (expert choice, dynamic) — OpenCode
- [ ] `innov-long` — Long context (1M, 10M) — OpenCode
- [ ] `innov-multimodal` — Multi-modal (image, audio, video) — OpenCode
- [ ] `innov-world` — World model (physics, simulation) — OpenCode
- [ ] `innov-robotics` — Robotics (vision, control) — OpenCode
- [ ] `innov-science` — AI for science (material, drug) — OpenCode
- [ ] `innov-alignment` — Alignment (safety, value, control) — OpenCode

### P70 — COMMUNITY + ECOSYSTEM (thêm)
- [ ] `eco-open-source` — Open source (repo, license, CI) — Freebuff
- [ ] `eco-contribution` — Contribution (code, doc, test) — Freebuff
- [ ] `eco-issue` — Issue (report, triage, fix) — Freebuff
- [ ] `eco-pr` — Pull request (review, merge) — Freebuff
- [ ] `eco-release` — Release (tag, changelog, artifact) — Freebuff
- [ ] `eco-docs` — Docs (wiki, tutorial, example) — Freebuff
- [ ] `eco-community` — Community (forum, Discord, Slack) — Freebuff
- [ ] `eco-hackathon` — Hackathon (challenge, prize) — Freebuff
- [ ] `eco-partner` — Partner (integrate, collab) — Freebuff
- [ ] `eco-adoption` — Adoption (user, org, industry) — Freebuff

### P71 — LEGACY + COMPAT (thêm)
- [ ] `compat-llama` — llama.cpp (GGUF, format) — Freebuff
- [ ] `compat-openai` — OpenAI (API, SDK) — Freebuff
- [ ] `compat-hf` — HuggingFace (transformers, PEFT) — Freebuff
- [ ] `compat-pytorch` — PyTorch (model, tensor) — OpenCode
- [ ] `compat-jax` — JAX (model, kernel) — OpenCode
- [ ] `compat-tf` — TensorFlow (model, graph) — OpenCode
- [ ] `compat-onnx` — ONNX (model, runtime) — OpenCode
- [ ] `compat-tensorrt` — TensorRT (model, engine) — OpenCode
- [ ] `compat-openvino` — OpenVINO (model, plugin) — OpenCode
- [ ] `compat-coreml` — CoreML (model, compile) — Freebuff

### P72 — BENCHMARK + COMPETITION (thêm)
- [ ] `comp-llama` — vs llama.cpp (speed, quality) — Freebuff
- [ ] `comp-vllm` — vs vLLM (throughput, memory) — Freebuff
- [ ] `comp-tgi` — vs TGI (latency, cost) — Freebuff
- [ ] `comp-ctransformers` — vs ctransformers (portability) — Freebuff
- [ ] `comp-mxfp` — vs mxfloat (quant, speed) — Freebuff
- [ ] `comp-gpt4all` — vs GPT4All (easy, local) — Freebuff
- [ ] `comp-gptengineer` — vs GPT4 (quality, cost) — Freebuff
- [ ] `comp-claude` — vs Claude (safety, alignment) — Freebuff
- [ ] `comp-gemini` — vs Gemini (multi-modal, long) — Freebuff
- [ ] `comp-llama4` — vs Llama 4 (MoE, size) — Freebuff

### P73 — CUSTOMER + SUPPORT (thêm)
- [ ] `cust-onboard` — Onboard (install, first run) — Freebuff
- [ ] `cust-docs` — Docs (guide, tutorial, FAQ) — Freebuff
- [ ] `cust-forum` — Forum (Q&A, help, share) — Freebuff
- [ ] `cust-discord` — Discord (chat, community) — Freebuff
- [ ] `cust-slack` — Slack (team, enterprise) — Freebuff
- [ ] `cust-email` — Email (support, update) — Freebuff
- [ ] `cust-ticket` — Ticket (report, track, resolve) — Freebuff
- [ ] `cust-feedback` — Feedback (survey, NPS) — Freebuff
- [ ] `cust-roadmap` — Roadmap (feature, priority) — Freebuff
- [ ] `cust-sla` — SLA (response, resolution) — Freebuff

### P74 — METRICS + DASHBOARD (thêm)
- [ ] `dash-realtime` — Real-time (live, stream) — Freebuff
- [ ] `dash-history` — History (trend, seasonality) — Freebuff
- [ ] `dash-compare` — Compare (model, quant, prompt) — Freebuff
- [ ] `dash-alert` — Alert (threshold, anomaly) — Freebuff
- [ ] `dash-report` — Report (daily, weekly, monthly) — Freebuff
- [ ] `dash-export` — Export (CSV, JSON, PDF) — Freebuff
- [ ] `dash-share` — Share (link, embed) — Freebuff
- [ ] `dash-custom` — Custom (widget, panel) — Freebuff
- [ ] `dash-drill` — Drill-down (detail, root cause) — Freebuff
- [ ] `dash-embed` — Embed (web, app) — Freebuff

### P75 — TESTING + QA (thêm)
- [ ] `qa-unit` — Unit (kernel, module) — Freebuff
- [ ] `qa-integration` — Integration (API, pipeline) — Freebuff
- [ ] `qa-e2e` — E2E (full flow) — Freebuff
- [ ] `qa-load` — Load (10k req/min) — Freebuff
- [ ] `qa-stress` — Stress (overload, failover) — Freebuff
- [ ] `qa-soak` — Soak (24h, memory leak) — Freebuff
- [ ] `qa-fuzz` — Fuzz (random, crash) — Freebuff
- [ ] `qa-a/b` — A/B (model, prompt) — Freebuff
- [ ] `qa-regression` — Regression (baseline, compare) — Freebuff
- [ ] `qa-coverage` — Coverage (unit, integration, e2e) — Freebuff

### P76 — LLM FEATURES (thêm)
- [ ] `llm-chat` — Chat (multi-turn, streaming) — Freebuff
- [ ] `llm-completion` — Completion (single-turn, batch) — Freebuff
- [ ] `llm-embed` — Embedding (vector, similarity) — Freebuff
- [ ] `llm-rerank` — Rerank (query, document) — Freebuff
- [ ] `llm-classify` — Classify (label, multi-class) — Freebuff
- [ ] `llm-extract` — Extract (entity, relation, fact) — Freebuff
- [ ] `llm-summarize` — Summarize (extractive, abstractive) — Freebuff
- [ ] `llm-translate` — Translate (multi-lingual, NMT) — Freebuff
- [ ] `llm-code` — Code (gen, review, test) — OpenCode
- [ ] `llm-math` — Math (symbolic, numeric) — OpenCode

### P77 — MODEL ARCH (thêm)
- [ ] `arch-llama` — Llama 2/3/4 (GQA, RoPE) — Freebuff
- [ ] `arch-qwen` — Qwen 1/2/3 (GQA, RoPE, MoE) — OpenCode
- [ ] `arch-mistral` — Mistral 7B/Instruct (sliding, GQA) — Freebuff
- [ ] `arch-gemma` — Gemma/Gemma-2 (GQA, sliding) — Freebuff
- [ ] `arch-phi` — Phi-3/4 (GQA, alternating) — OpenCode
- [ ] `arch-command` — Command-R/R+ (sliding, GQA) — Freebuff
- [ ] `arch-olmo` — OLMo/OLMoE (GQA, RoPE) — Freebuff
- [ ] `arch-jamba` — Jamba (Mamba + attention) — OpenCode
- [ ] `arch-mamba` — Mamba2/Samba (SSM, no attention) — OpenCode
- [ ] `arch-rwkv` — RWKV (linear attention, no KV) — OpenCode

### P78 — QUANTIZATION (thêm)
- [ ] `q4_k` — Q4_K (4-bit, k-quant) — OpenCode
- [ ] `q5_k` — Q5_K (5-bit, k-quant) — OpenCode
- [ ] `q6_k` — Q6_K (6-bit, k-quant) — OpenCode
- [ ] `q8_0` — Q8_0 (8-bit, per-block) — OpenCode
- [ ] `q8_1` — Q8_1 (8-bit, per-channel) — OpenCode
- [ ] `f16` — FP16 (half-precision) — OpenCode
- [ ] `f32` — FP32 (single-precision) — OpenCode
- [ ] `i8` — INT8 (8-bit, symmetric) — OpenCode
- [ ] `i4` — INT4 (4-bit, symmetric) — OpenCode
- [ ] `i2` — INT2 (2-bit, symmetric) — OpenCode

### P79 — ATTENTION (thêm)
- [ ] `attn-standard` — Standard (Q, K, V, softmax) — OpenCode
- [ ] `attn-multi` — Multi-head (H heads) — OpenCode
- [ ] `attn-group` — Grouped-query (GQA) — OpenCode
- [ ] `attn-multi-query` — Multi-query (MQA) — OpenCode
- [ ] `attn-sliding` — Sliding window (local) — OpenCode
- [ ] `attn-linear` — Linear (no softmax) — OpenCode
- [ ] `attn-log` — Log-attention (log-linear) — OpenCode
- [ ] `attn-sparse` — Sparse (top-k) — OpenCode
- [ ] `attn-flash` — Flash (tiling, on-the-fly) — OpenCode
- [ ] `attn-flash2` — FlashAttention-2 (faster, less memory) — OpenCode

### P80 — GEMM + KERNEL (thêm)
- [ ] `gemm-f32` — GEMM FP32 (baseline) — OpenCode
- [ ] `gemm-f16` — GEMM FP16 (FMA) — OpenCode
- [ ] `gemm-i8` — GEMM INT8 (DQ8_0, Q8_0) — OpenCode
- [ ] `gemm-i4` — GEMM INT4 (Q4_0, Q4_K) — OpenCode
- [ ] `gemm-i2` — GEMM INT2 (Q2_K, Q3_K) — OpenCode
- [ ] `gemm-tiled` — Tiled (cache, L1/L2) — OpenCode
- [ ] `gemm-split` — Split-K (parallel, reduce) — OpenCode
- [ ] `gemm-wino` — Winograd (conv, fast) — OpenCode
- [ ] `gemm-sparse` — Sparse (skip zero) — OpenCode
- [ ] `gemm-mixed` — Mixed (per-layer bit) — OpenCode

### P81 — MEMORY + CACHE (thêm)
- [ ] `mem-kv` — KV cache (f16, q8) — Freebuff
- [ ] `mem-paged` — Paged KV (vLLM) — OpenCode
- [ ] `mem-ring` — Ring KV (sliding) — Freebuff
- [ ] `mem-share` — Shared KV (GQA, MQA) — OpenCode
- [ ] `mem-compress` — Compressed KV (quant, prune) — OpenCode
- [ ] `mem-pool` — Memory pool (preallocate) — Freebuff
- [ ] `mem-mmap` — Memory-map (file, disk) — Freebuff
- [ ] `mem-swizzle` — Swizzle (cache, L1/L2) — OpenCode
- [ ] `mem-prefetch` — Prefetch (hint, AVX-512) — OpenCode
- [ ] `mem-hbm` — HBM (high-bandwidth) — OpenCode

### P82 — THREAD + PARALLEL (thêm)
- [ ] `par-thread` — Thread (rayon, std::thread) — Freebuff
- [ ] `par-simd` — SIMD (AVX2, AVX-512, NEON) — OpenCode
- [ ] `par-vector` — Vector (SIMD, GPU) — OpenCode
- [ ] `par-graph` — Graph (DAG, parallel) — OpenCode
- [ ] `par-pipeline` — Pipeline (stage, overlap) — OpenCode
- [ ] `par-data` — Data (embarrassing) — Freebuff
- [ ] `par-task` — Task (work-stealing) — Freebuff
- [ ] `par-async` — Async (tokio, async/await) — Freebuff
- [ ] `par-gpu` — GPU (CUDA, HIP) — OpenCode
- [ ] `par-distributed` — Distributed (multi-node) — OpenCode

### P83 — COMPILER + CODEGEN (thêm)
- [ ] `code-gen` — Code generation (LLVM, cranelift) — OpenCode
- [ ] `code-opt` — Optimization (inliner, vectorizer) — OpenCode
- [ ] `code-jit` — JIT (just-in-time, dynamic) — OpenCode
- [ ] `code-aot` — AOT (ahead-of-time, static) — OpenCode
- [ ] `code-wasm` — WASM (compile, execute) — OpenCode
- [ ] `code-cuda` — CUDA (compile, kernel) — OpenCode
- [ ] `code-opencl` — OpenCL (compile, kernel) — OpenCode
- [ ] `code-spirv` — SPIR-V (Vulkan, compile) — OpenCode
- [ ] `code-triton` — Triton (Python, GPU) — OpenCode
- [ ] `code-xla` — XLA (JAX, compile) — OpenCode

### P84 — TENSOR + MATH (thêm)
- [ ] `tensor-add` — Addition (element-wise) — OpenCode
- [ ] `tensor-mul` — Multiplication (element-wise, matmul) — OpenCode
- [ ] `tensor-conv` — Convolution (1D, 2D, 3D) — OpenCode
- [ ] `tensor-pool` — Pooling (max, avg) — OpenCode
- [ ] `tensor-softmax` — Softmax (stable, fast) — OpenCode
- [ ] `tensor-norm` — Normalization (LN, BN, RMS) — OpenCode
- [ ] `tensor-relu` — ReLU (activation) — OpenCode
- [ ] `tensor-sigmoid` — Sigmoid (activation) — OpenCode
- [ ] `tensor-tanh` — Tanh (activation) — OpenCode
- [ ] `tensor-geglu` — GELU, GeGLU (activation) — OpenCode

### P85 — DATA + PIPELINE (thêm)
- [ ] `data-load` — Load (CSV, JSON, Parquet) — Freebuff
- [ ] `data-transform` — Transform (clean, normalize) — Freebuff
- [ ] `data-split` — Split (train, val, test) — Freebuff
- [ ] `data-augment` — Augment (synthetic, back-translation) — OpenCode
- [ ] `data-embed` — Embed (vector, index) — OpenCode
- [ ] `data-index` — Index (vector, BM25) — Freebuff
- [ ] `data-retrieval` — Retrieval (search, rank) — Freebuff
- [ ] `data-generate` — Generate (synthetic, LLM) — OpenCode
- [ ] `data-verify` — Verify (quality, consistency) — Freebuff
- [ ] `data-serve` — Serve (API, stream) — Freebuff

### P86 — EVAL + BENCHMARK (thêm)
- [ ] `eval-mmlu` — MMLU (57 subjects) — Freebuff
- [ ] `eval-gsm8k` — GSM8K (math, grade school) — Freebuff
- [ ] `eval-humaneval` — HumanEval (code, Python) — Freebuff
- [ ] `eval-math` — MATH (competition) — Freebuff
- [ ] `eval-code` — MBPP, EvalPlus (code) — Freebuff
- [ ] `eval-chat` — MT-Bench, Alpaca (chat) — Freebuff
- [ ] `eval-long` — LongBench (128k, 1M) — Freebuff
- [ ] `eval-mm` — MMMU, MMBench (multi-modal) — Freebuff
- [ ] `eval-agent` — SWE-bench, AgentBench (agent) — Freebuff
- [ ] `eval-custom` — Custom (domain, task) — Freebuff

### P87 — SAFETY + ALIGNMENT (thêm)
- [ ] `safety-sft` — SFT (supervised) — OpenCode
- [ ] `safety-dpo` — DPO (preference) — OpenCode
- [ ] `safety-orpo` — ORPO (preference, no ref) — OpenCode
- [ ] `safety-rlhf` — RLHF (reward, PPO) — OpenCode
- [ ] `safety-guard` — Guard (filter, toxicity) — Freebuff
- [ ] `safety-jailbreak` — Jailbreak (detect, prevent) — Freebuff
- [ ] `safety-bias` — Bias (gender, race, political) — Freebuff
- [ ] `safety-hallucination` — Hallucination (fact-check) — Freebuff
- [ ] `safety-privacy` — Privacy (PII, GDPR) — Freebuff
- [ ] `safety-audit` — Audit (model, system) — Freebuff

### P88 — DEPLOYMENT + OPS (thêm)
- [ ] `deploy-docker` — Docker (CPU, CUDA) — Freebuff
- [ ] `deploy-k8s` — Kubernetes (Helm, operator) — Freebuff
- [ ] `deploy-aws` — AWS (SageMaker, EC2) — Freebuff
- [ ] `deploy-gcp` — GCP (Vertex, GKE) — Freebuff
- [ ] `deploy-azure` — Azure (ML, AKS) — Freebuff
- [ ] `deploy-serverless` — Serverless (Lambda, Cloud Functions) — Freebuff
- [ ] `deploy-edge` — Edge (Jetson, RPi) — Freebuff
- [ ] `deploy-browser` — Browser (WebGPU, WASM) — OpenCode
- [ ] `deploy-mobile` — Mobile (iOS, Android) — Freebuff
- [ ] `deploy-cloud` — Cloud (multi-region, global) — Freebuff

### P89 — MONITORING + OBS (thêm)
- [ ] `obs-metrics` — Metrics (Prometheus, OTel) — Freebuff
- [ ] `obs-logs` — Logs (structured, ELK) — Freebuff
- [ ] `obs-traces` — Traces (Jaeger, Zipkin) — Freebuff
- [ ] `obs-alert` — Alerts (PagerDuty, Slack) — Freebuff
- [ ] `obs-dash` — Dashboards (Grafana, Kibana) — Freebuff
- [ ] `obs-slo` — SLO (error budget, burn) — Freebuff
- [ ] `obs-cost` — Cost (CPU, GPU, per-query) — Freebuff
- [ ] `obs-perf` — Perf (latency, throughput) — Freebuff
- [ ] `obs-health` — Health (liveness, readiness) — Freebuff
- [ ] `obs-audit` — Audit (who, what, when) — Freebuff

### P90 — TESTING + QA (thêm)
- [ ] `test-unit` — Unit (kernel, module) — Freebuff
- [ ] `test-integration` — Integration (API, pipeline) — Freebuff
- [ ] `test-e2e` — E2E (full flow) — Freebuff
- [ ] `test-load` — Load (10k req/min) — Freebuff
- [ ] `test-stress` — Stress (overload, failover) — Freebuff
- [ ] `test-soak` — Soak (24h, memory leak) — Freebuff
- [ ] `test-fuzz` — Fuzz (random, crash) — Freebuff
- [ ] `test-a/b` — A/B (model, prompt) — Freebuff
- [ ] `test-regression` — Regression (baseline) — Freebuff
- [ ] `test-coverage` — Coverage (unit, integration) — Freebuff

### P91 — MODEL FAMILY (thêm)
- [ ] `fam-llama` — Llama family (2, 3, 4, Llama 4) — Freebuff
- [ ] `fam-qwen` — Qwen family (1, 2, 3, MoE) — OpenCode
- [ ] `fam-mistral` — Mistral family (7B, Instruct, Large) — Freebuff
- [ ] `fam-gemma` — Gemma family (2, 3, 2b, 7b) — Freebuff
- [ ] `fam-phi` — Phi family (mini, 3, 4) — OpenCode
- [ ] `fam-command` — Command family (R, R+) — Freebuff
- [ ] `fam-olmo` — OLMo family (1, 2, OLMoE) — Freebuff
- [ ] `fam-deepseek` — DeepSeek family (V3, R1) — OpenCode
- [ ] `fam-mixtral` — Mixtral family (8x7B, 8x22B) — OpenCode
- [ ] `fam-falcon` — Falcon family (7B, 40B, 180B) — Freebuff

### P92 — TASK + APPLICATION (thêm)
- [ ] `task-chat` — Chat (multi-turn, context) — Freebuff
- [ ] `task-completion` — Completion (single, batch) — Freebuff
- [ ] `task-summary` — Summary (extractive, abstractive) — Freebuff
- [ ] `task-translate` — Translation (multi-lingual) — Freebuff
- [ ] `task-classify` — Classification (label, multi-class) — Freebuff
- [ ] `task-extract` — Extraction (NER, RE, fact) — Freebuff
- [ ] `task-generate` — Generation (text, code, image) — OpenCode
- [ ] `task-qna` — Q&A (open, closed, multi-hop) — Freebuff
- [ ] `task-reason` — Reasoning (logical, causal) — OpenCode
- [ ] `task-plan` — Planning (goal, step, execute) — OpenCode

### P93 — DOMAIN + VERTICAL (thêm)
- [ ] `domain-medical` — Medical (diagnosis, drug, EHR) — Freebuff
- [ ] `domain-legal` — Legal (contract, case, statute) — Freebuff
- [ ] `domain-finance` — Finance (risk, portfolio, trading) — Freebuff
- [ ] `domain-science` — Science (paper, experiment, formula) — Freebuff
- [ ] `domain-engineering` — Engineering (CAD, FEA, code) — Freebuff
- [ ] `domain-education` — Education (tutor, quiz, grade) — Freebuff
- [ ] `domain-retail` — Retail (recommend, forecast, inventory) — Freebuff
- [ ] `domain-manufacturing` — Manufacturing (predictive, quality) — Freebuff
- [ ] `domain-logistics` — Logistics (routing, tracking, forecast) — Freebuff
- [ ] `domain-gaming` — Gaming (NPC, bot, content) — Freebuff

### P94 — DATA FORMAT (thêm)
- [ ] `fmt-gguf` — GGUF (v2, v3) — Freebuff
- [ ] `fmt-safetensors` — Safetensors (HF) — Freebuff
- [ ] `fmt-onnx` — ONNX (model, graph) — OpenCode
- [ ] `fmt-pytorch` — PyTorch (.pt, .bin) — OpenCode
- [ ] `fmt-jax` — JAX (.flax, .jnp) — OpenCode
- [ ] `fmt-tf` — TensorFlow (.pb, .tflite) — OpenCode
- [ ] `fmt-coreml` — CoreML (.mlmodel) — Freebuff
- [ ] `fmt-tensorrt` — TensorRT (.engine) — OpenCode
- [ ] `fmt-openvino` — OpenVINO (.xml, .bin) — OpenCode
- [ ] `fmt-custom` — Custom (binary, proprietary) — Freebuff

### P95 — INFER MODE (thêm)
- [ ] `mode-single` — Single (1 request) — Freebuff
- [ ] `mode-batch` — Batch (N requests) — Freebuff
- [ ] `mode-stream` — Stream (token-by-token) — Freebuff
- [ ] `mode-async` — Async (fire-and-forget) — Freebuff
- [ ] `mode-priority` — Priority (VIP first) — Freebuff
- [ ] `mode-slo` — SLO (latency target) — OpenCode
- [ ] `mode-cost` — Cost (optimize $/query) — OpenCode
- [ ] `mode-quality` — Quality (best answer) — OpenCode
- [ ] `mode-mixed` — Mixed (auto-select) — OpenCode
- [ ] `mode-custom` — Custom (user-defined) — Freebuff

### P96 — GPU BACKEND (thêm)
- [ ] `gpu-cuda` — CUDA (NVIDIA) — OpenCode
- [ ] `gpu-hip` — HIP/ROCm (AMD) — OpenCode
- [ ] `gpu-oneapi` — oneAPI/SYCL (Intel) — OpenCode
- [ ] `gpu-metal` — Metal (Apple) — Freebuff
- [ ] `gpu-vulkan` — Vulkan (cross-platform) — Freebuff
- [ ] `gpu-opencl` — OpenCL (generic) — Freebuff
- [ ] `gpu-webgpu` — WebGPU (browser) — OpenCode
- [ ] `gpu-tpu` — TPU (Google) — OpenCode
- [ ] `gpu-npu` — NPU (Intel, Apple) — OpenCode
- [ ] `gpu-multi` — Multi-GPU (tensor, pipeline) — OpenCode

### P97 — CPU OPT (thêm)
- [ ] `cpu-avx2` — AVX2 (256-bit) — OpenCode
- [ ] `cpu-avx512` — AVX-512 (512-bit) — OpenCode
- [ ] `cpu-vnni` — VNNI (INT8 GEMM) — OpenCode
- [ ] `cpu-amx` — AMX (INT8/INT16, Intel) — OpenCode
- [ ] `cpu-neon` — NEON (ARM) — Freebuff
- [ ] `cpu-rvv` — RISC-V Vector — OpenCode
- [ ] `cpu-sse` — SSE2 (128-bit) — Freebuff
- [ ] `cpu-fma` — FMA (FMA instruction) — OpenCode
- [ ] `cpu-math` — Math (libm, SIMD) — OpenCode
- [ ] `cpu-thread` — Thread (rayon, std) — Freebuff

### P98 — MEMORY OPT (thêm)
- [ ] `mem-mmap` — Memory-map (file, disk) — Freebuff
- [ ] `mem-pool` — Pool (preallocate, reuse) — Freebuff
- [ ] `mem-arena` — Arena (large, contiguous) — Freebuff
- [ ] `mem-slab` — Slab (small, same-size) — Freebuff
- [ ] `mem-bump` — Bump (fast, no free) — Freebuff
- [ ] `mem-arena` — Arena (per-request) — Freebuff
- [ ] `mem-gc` — GC (reference, tracing) — Freebuff
- [ ] `mem-no` — No-alloc (stack, static) — OpenCode
- [ ] `mem-cache` — Cache (L1, L2, L3) — OpenCode
- [ ] `mem-bandwidth` — Bandwidth (HBM, DDR) — OpenCode

### P99 — NETWORK + DISTRIBUTED (thêm)
- [ ] `net-tcp` — TCP (socket, stream) — Freebuff
- [ ] `net-udp` — UDP (datagram, fast) — Freebuff
- [ ] `net-grpc` — gRPC (protobuf, streaming) — OpenCode
- [ ] `net-http` — HTTP/2 (REST, gRPC) — Freebuff
- [ ] `net-websocket` — WebSocket (real-time) — Freebuff
- [ ] `net-mqtt` — MQTT (IoT, low-power) — Freebuff
- [ ] `net-amqp` — AMQP (message, queue) — Freebuff
- [ ] `net-kafka` — Kafka (stream, log) — Freebuff
- [ ] `net-rpc` — RPC (NNG, ZeroMQ) — OpenCode
- [ ] `net-cluster` — Cluster (etcd, Raft) — OpenCode

### P100 — FUTURE + FRONTIER (thêm)
- [ ] `future-1m` — 1M context (ring, paged) — OpenCode
- [ ] `future-10m` — 10M context (distributed) — OpenCode
- [ ] `future-multimodal` — Multi-modal (image, audio, video, 3D) — OpenCode
- [ ] `future-world` — World model (physics, simulation) — OpenCode
- [ ] `future-robotics` — Robotics (vision, control, RL) — OpenCode
- [ ] `future-science` — AI for science (material, drug, protein) — OpenCode
- [ ] `future-math` — AI for math (theorem, proof) — OpenCode
- [ ] `future-alignment` — Alignment (safety, value, control) — OpenCode
- [ ] `future-scaling` — Scaling law (compute, data, model) — OpenCode
- [ ] `future-agentic` — Agentic (autonomous, multi-agent) — OpenCode

## ĐẾM CONTRIBUTIONS

| Phase | Tasks | Contributions (×4) | Tổng |
|---|---|---|---|
| Tối nay (11h-2h) | 4 tasks | 16 | 16 |
| Sáng (2h-10h) | 12 tasks | 48 | 64 |
| Chiều (10h-18h) | 12 tasks | 48 | 112 |
| Tối (18h-2h) | 12 tasks | 48 | 160 |
| **Tổng 24h (P0-P3)** | **40 tasks** | **160** | **160** |

> **1011 tasks** (P0-P100) × 1 cycle = **4044 contributions**
> 1011 × 2 cycles = **8088 contributions** → đạt 1000 rule trong **48h (2 ngày)** ✓
> 1011 × 3 cycles = **12132 contributions** → đạt 2000 rule trong **72h (3 ngày)** ✓

**Gợi ý: 1 cycle trong 24h → 4044 contributions (vượt 1000 rule, 4x)**

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
