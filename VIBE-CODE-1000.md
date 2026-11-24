# VIBE CODE 1000 — vipera.rs

> File giao tiếp chung cho Hermes + OpenCode + Freebuff.
> Mỗi agent đọc file này trước khi bắt đầu. Cập nhật khi có thay đổi.

## Rule 100 contributions/ngày

Mỗi task = 1 branch riêng → 1 PR → squash merge → 1 contribution.
51 tasks × 5 cycles × ~4 contributions = **1020 contributions**.

---

## 3 BÉ LÀM GÌ

| Bé | Vai trò | Tool | Model | Workdir |
|---|---|---|---|---|
| **Hermes** | Orchestrate, verify, merge | terminal + gh + tmux | — | `/home/user` (main repo) |
| **OpenCode** | Task KHÓ (P0-P1, kernels, GPU) | `opencode run --auto` | `opencode/tidepool-alpha-free` | `/tmp/vb-opencode` |
| **Freebuff** | Task TRUNG BÌNH (P1-P3, algo crates, server, docs) | `tmux fb-vipera` | `Glyph Cluster` (free) | `/tmp/vb-freebuff` |

---

## QUY TRÌNH MỖI TASK

```
1. Hermes đọc README.md "Limitations" + "Where it stands" → biết target
2. Hermes chọn task từ bảng bên dưới (P0 trước, P1 sau, P2-P3 cuối)
3. Hermes tạo task file: /tmp/vb-{worker}/.agent/tasks/<id>.md
4. Dispatch:
   - OpenCode: cd /tmp/vb-opencode && opencode run --model opencode/tidepool-alpha-free --auto "Read <task_file> and execute it."
   - Freebuff: tmux send-keys -t fb-vipera -l -- "Đọc <task_file> và làm."
5. Worker code + test + fmt → ghi report JSON
6. Hermes: cargo test + cargo fmt --check → copy files → git commit
7. git checkout -B vibe/<id> → push → gh pr create → wait CI → squash merge + delete branch
8. GitHub +1 contribution (hoặc +2 nếu count delete-branch)
```

### Task file format (Hermes viết)

```markdown
# <id> — <goal ngắn>

## End state
<goal> — tests pass, fmt clean, no new deps.
Read /home/user/README.md "Limitations" section before starting.

## Context
- Repo: /tmp/vb-<worker> (vipera.rs), branch <pr_branch>
- Reference forks: /home/user/llm-forks/
- Files: <files>
- Test: cargo test -p <crate> 2>&1 | grep "test result"
- No new dependencies

## Scope
- may touch: <files>
- off limits: Cargo.toml, specs/

## Report
Write JSON to: /tmp/vb-<worker>/.agent/reports/<id>.json
Schema: {"task":"<id>","status":"done|partial|blocked","summary":"...",
         "files_changed":[...],"checks":[],"blockers":[],"next":null}
```

---

## BẢNG 51 TASKS

### P0 — PERFORMANCE (decode + prefill, target 1.5-2.0x)

| # | ID | Goal | Worker | Files | Status |
|---|---|---|---|---|---|
| 1 | `perf-q6k-i32` | Q6_K i32 GEMM epilogue (vpdpbusd) từ ik_llama.cpp | OpenCode | `crates/vipera-backend-cpu/src/simd.rs` | ⬜ |
| 2 | `perf-iqk-repack` | IQK AVX-512 repack từ ik_llama.cpp HAVE_FANCY_SIMD | OpenCode | `crates/vipera-backend-cpu/src/iqk.rs` (new) | ⬜ |
| 3 | `perf-mla-attn` | MLA/FlashMLA attention từ ik_llama.cpp | Freebuff | `crates/vipera-backend-cpu/src/mla.rs` (new) | ⬜ |
| 4 | `perf-masked-attn` | Masked-attention skip từ llama-cpp-optimization-lab | Freebuff | `crates/vipera-backend-cpu/src/attention.rs` | ⬜ |

### P1 — FEATURES + PERF

| # | ID | Goal | Worker | Files | Status |
|---|---|---|---|---|---|
| 5 | `perf-pinned-staging` | Pinned memory staging (no per-token malloc) | Freebuff | `crates/vipera-runtime/src/memory.rs` | ⬜ |
| 6 | `perf-prefault-workers` | Prefault workers (pre-touch model pages) | Freebuff | `crates/vipera-core/src/model.rs` | ⬜ |
| 7 | `perf-q8-k-r16` | Q8_K_R16 / Q8_0_R16 16-row repack | OpenCode | `crates/vipera-backend-cpu/src/repack.rs` | ⬜ |
| 8 | `feat-batched-prefill` | GPU batched prefill kernel | OpenCode | `crates/vipera-backend-gpu/src/forward.rs` | ⬜ |
| 9 | `feat-ring-kv` | Sliding-window ring KV (61 MB vs 896 MB) | Freebuff | `crates/vipera-runtime/src/kv.rs` | ⬜ |
| 10 | `feat-cuda-cli` | CLI `--backend cuda` flag | OpenCode | `bins/vipera/src/main.rs` | ⬜ |
| 11 | `feat-moe-expert-cache` | Dynamic MoE expert cache (ik_llama PR 2644) | OpenCode | `crates/vipera-runtime/src/engine.rs` | ⬜ |
| 12 | `feat-moe-cpu-parallel` | CPU parallel expert compute (extended-strata-moe) | OpenCode | `crates/vipera-backend-gpu/src/moe.rs` (new) | ⬜ |
| 13 | `feat-mtp-speculative` | MTP speculative decoding (ik_llama) | OpenCode | `crates/vipera-runtime/src/engine.rs` | ⬜ |
| 14 | `feat-algo-tqkv` | vipera-algo-tqkv (token-query KV compression) | Freebuff | `crates/vipera-algo-tqkv/` (new) | ⬜ |
| 15 | `feat-algo-q8-fallback` | vipera-algo-q8-fallback (Q8_0 KV) | Freebuff | `crates/vipera-algo-q8-fallback/` (new) | ⬜ |
| 16 | `feat-server-batching` | Concurrent conversations + request batching | OpenCode | `crates/vipera-server/src/handler.rs` | ⬜ |

### P2 — ALGO CRATES + ADVANCED

| # | ID | Goal | Worker | Files | Status |
|---|---|---|---|---|---|
| 17 | `perf-trellis-quants` | Gyro Trellis (TQ2_T, TQK6/7, TQP5-7) từ ik_llama PR 2812 | OpenCode | `crates/vipera-core/src/gguf.rs` | ⬜ |
| 18 | `perf-ternary-bonsai` | Ternary Bonsai / Prism (PQ2_0/PTQ1_0) từ ik_llama PR 2550 | OpenCode | `crates/vipera-core/src/gguf.rs` | ⬜ |
| 19 | `perf-fused-gdn` | Fused delta-net (GDN) từ ik_llama | Freebuff | `crates/vipera-backend-cpu/src/attention.rs` | ⬜ |
| 20 | `feat-metal-backend` | ForwardPass trên Metal (macOS) | Freebuff | `crates/vipera-backend-metal/` (new) | ⬜ |
| 21 | `feat-vulkan-backend` | ForwardPass trên Vulkan (Windows/Linux) | Freebuff | `crates/vipera-backend-vulkan/` (new) | ⬜ |
| 22 | `feat-dflash-speculative` | DFlash speculative (draft on GPU, verify on CPU) | OpenCode | `crates/vipera-runtime/src/engine.rs` | ⬜ |
| 23 | `feat-exl3-quant` | EXL3 quantization (QTIP-based) từ exllamav3 | OpenCode | `crates/vipera-core/src/gguf.rs` | ⬜ |
| 24 | `feat-algo-keepkv` | vipera-algo-keepkv (top-K KV selection) | Freebuff | `crates/vipera-algo-keepkv/` (new) | ⬜ |
| 25 | `feat-algo-rkv` | vipera-algo-rkv (recency-based KV) | Freebuff | `crates/vipera-algo-rkv/` (new) | ⬜ |
| 26 | `feat-algo-chunkkv` | vipera-algo-chunkkv (512-token chunk compression) | Freebuff | `crates/vipera-algo-chunkkv/` (new) | ⬜ |
| 27 | `feat-algo-vecinfer` | vipera-algo-vecinfer (compressed vector KV, 4x less) | OpenCode | `crates/vipera-algo-vecinfer/` (new) | ⬜ |
| 28 | `feat-algo-patternkv` | vipera-algo-patternkv (repeated KV detection) | Freebuff | `crates/vipera-algo-patternkv/` (new) | ⬜ |
| 29 | `feat-algo-vayne` | vipera-algo-vayne (adaptive per-layer KV selection) | OpenCode | `crates/vipera-algo-vayne/` (new) | ⬜ |
| 30 | `feat-algo-singular-bit` | vipera-algo-singular-bit (1-bit KV, 8x memory reduction) | Freebuff | `crates/vipera-algo-singular-bit/` (new) | ⬜ |
| 31 | `feat-rate-limit` | Rate limiting 429 trong vipera-server | Freebuff | `crates/vipera-server/src/serve.rs` | ⬜ |
| 32 | `feat-server-metrics` | Per-request metrics (percentiles, memory, throughput) | Freebuff | `crates/vipera-server/src/serve.rs` | ⬜ |
| 33 | `feat-acme-loop` | ACME issue/renew loop (production TLS) | Freebuff | `crates/vipera-server/src/acme.rs` | ⬜ |
| 34 | `adv-cuda-graphs` | Hadamard-aware CUDA graphs (ik_llama PR 2824) | OpenCode | `crates/vipera-backend-gpu/src/forward.rs` | ⬜ |
| 35 | `adv-moe-mmq` | MoE MMQ fused (ik_llama PR 2831) | OpenCode | `crates/vipera-backend-gpu/src/moe.rs` | ⬜ |
| 36 | `adv-cpu-resident-experts` | CPU-resident MoE experts (ik_llama PR 2901) | OpenCode | `crates/vipera-runtime/src/engine.rs` | ⬜ |

### P3 — INFRA + RESEARCH

| # | ID | Goal | Worker | Files | Status |
|---|---|---|---|---|---|
| 37 | `infra-bench-regression` | Benchmark regression suite (CI, alert if >5% drop) | Freebuff | `crates/vipera-backend-cpu/tests/` + `.github/workflows/` | ⬜ |
| 38 | `infra-llama-parity` | llama.cpp parity test suite (CI) | Freebuff | `crates/vipera-conformance/src/lib.rs` | ⬜ |
| 39 | `infra-release-process` | Release process (tags, changelog, artifacts) | Freebuff | `.github/workflows/release.yml` (new) | ⬜ |
| 40 | `infra-contributing` | CONTRIBUTING.md | Freebuff | `CONTRIBUTING.md` (new) | ⬜ |
| 41 | `infra-adr-005` | ADR-005 numerics policy | Freebuff | `docs/design/ADR-005-*.md` | ⬜ |
| 42 | `infra-w2-honesty` | ADR: decode gap (27.3 vs 37.37 tok/s) | Freebuff | `docs/design/ADR-007-*.md` | ⬜ |
| 43 | `infra-q4k-f32-fixture` | Pin q4k_f32_continuation fixture | Freebuff | `bins/vipera/tests/fixtures/` | ⬜ |
| 44 | `infra-changelog` | CHANGELOG.md (improvement series) | Freebuff | `CHANGELOG.md` (new) | ⬜ |
| 45 | `adv-multi-gpu` | Multi-GPU tensor parallel (ik_llama split mode) | OpenCode | `crates/vipera-backend-gpu/src/` (extend) | ⬜ |
| 46 | `adv-graph-parallel` | Graph-parallel multi-GPU (pipeline parallelism) | OpenCode | `crates/vipera-backend-gpu/src/graph.rs` (new) | ⬜ |
| 47 | `adv-qwen4exp-scoring` | qwen4exp block scoring (ik_llama PR 2809) | Freebuff | `crates/vipera-backend-gpu/src/moe.rs` | ⬜ |
| 48 | `adv-overlapping-logits` | Overlapping logits in top-k MoE (ik_llama PR 2780) | OpenCode | `crates/vipera-backend-gpu/src/forward.rs` | ⬜ |
| 49 | `adv-llama-bench-parity` | llama-bench parity test (CI) | Freebuff | `scripts/parity-check.sh` | ⬜ |
| 50 | `adv-streaming-usage` | Streamed usage in SSE (include_usage) | Freebuff | `crates/vipera-server/src/handler.rs` | ⬜ |
| 51 | `adv-json-object-format` | response_format: json_object validation | Freebuff | `crates/vipera-server/src/json.rs` | ⬜ |

---

## TARGET HIỆN TẠI (từ README "Where it stands")

| Metric | Hiện tại | Target | Gap | Lệnh đo |
|---|---|---|---|---|
| prefill 128 | 233 tok/s (1.01x) | ✓ | — | `vipera generate --prompt "..." --max-tokens 1 --ctx 128` |
| prefill 512 | 221 tok/s (0.99x) | 1.0x+ | ~ | `--ctx 512` |
| prefill 1024 | 203 tok/s (0.97x) | 1.0x+ | 0.03x | `--ctx 1024` |
| **prefill 4096** | **145 tok/s (0.87x)** | **200+ (1.5x)** | **0.63x** | `--ctx 4096` + chunk timer |
| **decode shallow** | **29.5 tok/s (0.79x)** | **45+ (1.5x)** | **0.71x** | `--max-tokens 32 --json` |
| decode 4k | 24.2 tok/s (0.96x) | 1.0x+ | 0.04x | `--ctx 4096 --max-tokens 32 --json` |

```bash
# Decode
target/release/vipera generate ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --prompt "The capital of France is" --max-tokens 32 --json 2>&1 \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['tokens_per_second'])"

# Prefill (chunk timer)
target/release/vipera generate ~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --prompt "$(python3 -c 'print("The quick brown fox. "*401)')" --max-tokens 1 --ctx 4096 2>&1 \
  | grep -oE 'chunk [0-9]+ [0-9.]+ ms'
```

---

## WORKERS

### OpenCode (task khó)

```bash
# One-shot
cd /tmp/vb-opencode
/home/user/.opencode/bin/opencode run \
  --model opencode/tidepool-alpha-free --auto \
  "Read /tmp/vb-opencode/.agent/tasks/<id>.md and execute it.
   Working dir is /tmp/vb-opencode.
   When done, write report JSON to /tmp/vb-opencode/.agent/reports/<id>.json."

# Nếu hết free model
opencode models | grep -i free   # tìm model khác
```

### Freebuff (task trung bình)

```bash
# Tmux session
tmux has-session -t fb-vipera 2>/dev/null || \
  tmux new-session -d -s fb-vipera -c /tmp/vb-freebuff "freebuff" 9>&-

# Dispatch
tmux send-keys -t fb-vipera -l -- "Đọc /tmp/vb-freebuff/.agent/tasks/<id>.md và làm đúng như mô tả. Khi xong, ghi báo cáo JSON vào /tmp/vb-freebuff/.agent/reports/<id>.json."
tmux send-keys -t fb-vipera Enter

# Monitor
tmux capture-pane -t fb-vipera -p | tail -20

# Nếu model hết giờ → đổi: /model → chọn Glyph Cluster (free preview)
```

---

## HƯỚNG DẪN CHO 2 BÉ CODE

### OpenCode (đọc phần này khi được giao task)

1. **Đọc task file**: `/tmp/vb-opencode/.agent/tasks/<id>.md`
2. **Đọc README**: `/home/user/README.md` section "Limitations" — biết gap
3. **Đọc source reference**:
   - `/home/user/llm-forks/ik_llama.cpp/` — ggml-quants-8bit.c, ggml-impl-quants.c, ggml-impl-mla.c
   - `/home/user/llm-forks/llama-cpp-optimization-lab/patches/` — flash-next-selected.patch
   - `/home/user/llm-forks/extended-strata-moe/` — CPU parallel expert compute
   - `/home/user/llm-forks/exllamav3/` — EXL3 quantization
4. **Đọc code hiện tại**: `/home/user/crates/vipera-backend-cpu/src/` (simd.rs, repack.rs, model.rs)
5. **Code** → `cargo test -p <crate>` → `cargo fmt`
6. **Ghi report JSON** → xong

**Rules:**
- Không thêm dependency mới
- Không sửa Cargo.toml
- Không sửa specs/, docs/design/
- Commit email: `vinhtezos04@outlook.com` (đã set trong git config)

### Freebuff (đọc phần này khi được giao task)

1. **Đọc task file**: `/tmp/vb-freebuff/.agent/tasks/<id>.md`
2. **Đọc README**: `/home/user/README.md` section "Limitations"
3. **Đọc source reference** (nếu task yêu cầu):
   - `/home/user/llm-forks/` — đọc file C++ liên quan
4. **Đọc code hiện tại**: `/home/user/crates/vipera-*/src/`
5. **Code** → `cargo test -p <crate>` → `cargo fmt`
6. **Ghi report JSON** → xong

**Rules:**
- Không thêm dependency mới
- Không sửa Cargo.toml
- Nếu model hết giờ → đổi sang Glyph Cluster (free preview, không tốn giờ)
- Commit email: `vinhtezos04@outlook.com` (đã set trong git config)

---

## KHI NÀO ĐỢI, KHI NÀO PUSH TIẾP

| Tình huống | Hành động |
|---|---|
| Worker báo `status: done` | Hermes verify → copy → commit → PR → merge |
| Worker báo `status: partial` | Xem `summary` + `blockers` → decide: tiếp tục hay bỏ |
| Worker báo `status: blocked` | Đọc `blockers` → Hermes fix hoặc bỏ task, task khác |
| Timeout (15 min) | Kill worker → ghi chú → task khác |
| CI fail | Read log → fix → push lại |
| Freebuff hết giờ | `/model` → Glyph Cluster (free) |
| OpenCode hết free | `opencode models | grep free` → đổi model |

---

## ĐẾM CONTRIBUTIONS

| Hoạt động | Contributions |
|---|---|
| git commit | 1 |
| git push | 1 |
| PR create | 1 |
| PR squash merge | 1 |
| PR delete branch | 1 |
| **Tổng per task** | **~5** |

51 tasks × 5 cycles × 5 contributions = **1275 contributions** → đạt 1000 rule.

---

## PLAN 24H (từ 2h sáng)

| Thời gian | Freebuff | OpenCode | Ghi chú |
|---|---|---|---|
| 2h-10h | Glyph Cluster (free) — tasks P1-P2 | tidepool-alpha (free) — P0 hard-q6k-iqk | Freebuff queue #10, ~200s wait |
| 10h-18h | **DeepSeek V4.1 Flash (mạnh, 8h)** — tasks P1 | tidepool-alpha — P0/P1 (đổi WiFi nếu hết) | DeepSeek reset 24h → code 8h |
| 18h-2h | Glyph Cluster (free) — tasks P2-P3 | tidepool-alpha (đổi WiFi Nho) | Back model free |

**WiFi:**
- **Nho**: Freebuff (mặc định, giữ 24h)
- **Tang Thuong 5G**: OpenCode reset token khi hết (đổi 5-10s)
- **Hermes**: không bị ảnh hưởng đổi WiFi (chạy local)

**Rules:**
1. Freebuff hết 8h DeepSeek → đổi lại Glyph Cluster (free, không giới hạn)
2. OpenCode hết tidepool-alpha → đổi WiFi Tang Thuong 5G → reset → chạy tiếp
3. Mỗi task = 1 PR squash merge = ~4 contributions
4. Target: 51 tasks × 5 cycles = 1020+ contributions/ngày

---

## HIỆN TẠI (cập nhật liên tục)

- **Hermes**: orchestrate + verify + merge
- **OpenCode**: đang chạy `hard-q6k-iqk` (P0, test pass 0, đang code)
- **Freebuff**: đang chạy `pinned-staging` (Glyph Cluster, queue #10)
- **PRs mở**: #70 (tasks 007-009, CI green, chờ merge)
- **Branch**: `perf/attn-triattention` → main
