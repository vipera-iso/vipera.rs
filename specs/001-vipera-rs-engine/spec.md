# Feature Specification: vipera.rs Phase 1 — Local Inference Engine

**Feature Branch**: `001-vipera-rs-engine`

**Created**: 2026-11-06

**Status**: Draft

**Input**: User description: "Read plan.md and turn it into a Spec Kit specification. plan.md is the full repository blueprint for vipera.rs — a production-grade local inference engine intended to compete with llama.cpp / vLLM / TensorRT-LLM / SGLang. This specification covers Phase 1 only (v0.1.0–v0.3.0): 8 core crates, 6 algorithms, 2 backends (CUDA + CPU), ~3 months."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Drop-in chat endpoint for existing clients (Priority: P1)

An application developer already speaks to an OpenAI-compatible chat API. They install the engine, point the server at a local model file, and change only the base URL. Their existing client — including streaming responses — works without code changes, and the request is served entirely on the developer's own machine.

**Why this priority**: This is the smallest slice that delivers the product's core value: replacing a hosted chat model with a local one without rewriting the application. Everything else can be missing and this still ships something useful.

**Independent Test**: Start the server with one local model, issue a chat completion through an OpenAI-compatible client, and confirm a complete streamed answer with no outbound network calls to a model provider.

**Acceptance Scenarios**:

1. **Given** a local model file and a running server, **When** the developer sends a non-streaming chat request, **Then** the full answer returns in the response body with token usage reported.
2. **Given** the same setup, **When** the developer requests a streamed answer, **Then** tokens arrive incrementally and the stream terminates cleanly with a finish reason.
3. **Given** a machine with no GPU, **When** the same request is sent, **Then** the CPU-only build answers successfully with identical response shape.
4. **Given** a malformed or oversized request, **When** it reaches the server, **Then** a structured error returns with a non-success status and no server crash.

---

### User Story 2 - Predict memory before loading, verify after (Priority: P1)

An operator has a model that fits "almost" in their GPU. Before loading, they ask the engine what placement it will use and how much device and host memory that placement needs. After loading, they ask it to compare prediction against the actual measured footprint, so the numbers stay trustworthy rather than decorative.

**Why this priority**: Loading a model that does not fit wastes minutes and fails mid-flight. A plan that can be checked against reality is what makes the engine usable on consumer hardware.

**Independent Test**: For a given model and device, produce a placement plan with predicted peak memory, measure the real footprint, and compare the two without involving the chat endpoint.

**Acceptance Scenarios**:

1. **Given** a model and a device with known capacity, **When** the operator requests a plan, **Then** the response names the chosen layer placement and predicted peak device and host memory.
2. **Given** a model larger than the available device memory, **When** a plan is requested, **Then** the engine either produces a feasible partially-offloaded plan or refuses with an error stating what would need to change.
3. **Given** a completed plan and load, **When** the operator requests verification, **Then** predicted and actual figures are returned side by side with the deviation between them.

---

### User Story 3 - Offline single-prompt generation from the command line (Priority: P2)

A researcher wants one answer from a model without running a server or writing code: a single command with a model path and a prompt, returning the generated text. They use it in scripts and pipes.

**Why this priority**: It is the fastest path to a first successful run and the natural target for scripting and benchmarking, but the server path already proves the engine works.

**Independent Test**: Run one command against a local model and receive generated text on standard output with a non-zero exit status on failure.

**Acceptance Scenarios**:

1. **Given** a model path and a prompt, **When** the command runs, **Then** generated text is printed to standard output and the process exits successfully.
2. **Given** a missing or corrupt model path, **When** the command runs, **Then** a clear diagnostic goes to standard error and the exit status is non-zero.
3. **Given** a fixed seed and deterministic settings, **When** the same prompt is run twice, **Then** the same output is produced.

---

### User Story 4 - Reuse conversation context across turns (Priority: P2)

A user holds a long conversation. Rather than reprocessing the whole history on every turn, the engine retains the processed context, and when the model's output goes wrong the user can rewind to an earlier point in the conversation and continue from there.

**Why this priority**: Multi-turn latency is the difference between a demo and a usable assistant, and rollback is what makes retained context safe to rely on.

**Independent Test**: Send a multi-turn conversation, measure the time to first token on a follow-up turn versus a cold turn with the same prefix, then rewind and continue from the earlier point.

**Acceptance Scenarios**:

1. **Given** an ongoing conversation, **When** a follow-up turn is sent, **Then** the retained context is reused and the answer stays consistent with earlier turns.
2. **Given** a conversation the user wants to rewind, **When** they restore an earlier point, **Then** subsequent output continues from that point without recomputing the retained prefix.
3. **Given** several concurrent conversations, **When** they run together, **Then** one conversation's context never leaks into another's output.
4. **Given** a conversation that ends normally, **When** it completes, **Then** its retained memory is released for other conversations.

---

### User Story 5 - Operate the service safely (Priority: P3)

An operator runs the engine for a small team. They need to know it is alive, watch its throughput and latency, and keep unauthenticated or abusive clients from consuming the machine.

**Why this priority**: Necessary for shared use, but a single developer's local run is already valuable without it.

**Independent Test**: Start the server with an access key and a request limit, then confirm health checks, metrics, rejection of unknown keys, and throttling all behave as configured.

**Acceptance Scenarios**:

1. **Given** a running server, **When** the operator checks readiness, **Then** the state reflects whether a model is loaded and able to serve.
2. **Given** a running server, **When** metrics are scraped, **Then** request counts, token throughput, and latency figures are present.
3. **Given** an access key is configured, **When** a request arrives without a valid key, **Then** it is rejected with an authentication error.
4. **Given** a configured request limit, **When** a client exceeds it, **Then** further requests are refused with an explicit limit error until the window resets.

---

### User Story 6 - Add an algorithm or backend without touching the core (Priority: P3)

A contributor implements a new optimization or a new hardware backend. They add it in its own module, register it, and it becomes selectable — without editing the pipeline, the scheduler, or any shared core code.

**Why this priority**: This is what keeps the engine viable for a decade, but it only matters after the engine works at all.

**Independent Test**: Add a sixth Phase 1 algorithm in an isolated module and confirm it is selectable at runtime with no diff in core module files.

**Acceptance Scenarios**:

1. **Given** the engine runs with a default algorithm set, **When** a new algorithm is registered, **Then** it can be selected by name without changing core code.
2. **Given** an algorithm that fails at runtime, **When** it is selected, **Then** the engine falls back to the documented safe alternative and reports the degradation.
3. **Given** a backend unavailable on the host, **When** the engine starts, **Then** it selects an available backend and states which one it chose.

---

### Edge Cases

- Model file is truncated, has an unknown version, or declares tensor shapes that disagree with its metadata.
- Requested context length exceeds what the model supports, or the remaining memory after loading.
- Device memory is exhausted mid-generation by another process on the machine.
- Two requests arrive that together exceed memory: admission must reject or queue predictably, never crash.
- A conversation is rewound while another turn on the same conversation is still generating.
- Generation produces an end-of-sequence immediately, or hits the output token limit.
- Client disconnects mid-stream; retained memory must be reclaimed.
- An algorithm is selected that is not built for the detected backend.
- Clock or filesystem changes make a cached model artifact unreadable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load model files from a local path and reject unsupported or corrupt files with a diagnostic naming the specific inconsistency.
- **FR-002**: System MUST detect available hardware capabilities at startup and select a usable compute backend without requiring user configuration.
- **FR-003**: System MUST produce a placement plan for a model on a target device, reporting the layer split and predicted peak device and host memory.
- **FR-004**: System MUST refuse to load a placement that cannot fit, and MUST explain which constraint was violated and by how much.
- **FR-005**: System MUST measure actual memory footprint after loading and report the deviation from its prediction.
- **FR-006**: System MUST expose a chat completion endpoint compatible with the OpenAI request and response shape, including token usage reporting.
- **FR-007**: System MUST stream incremental output over that endpoint as it is generated, and MUST terminate the stream with a finish reason.
- **FR-008**: System MUST serve multiple concurrent conversations without cross-contamination of retained state.
- **FR-009**: System MUST retain processed conversation context across turns and reuse it instead of reprocessing the full history.
- **FR-010**: System MUST support restoring a conversation to an earlier retained point and continuing from it.
- **FR-011**: System MUST release retained context when a conversation finishes, fails, or the client disconnects.
- **FR-012**: System MUST apply a sampling policy that supports at minimum greedy, temperature, top-k, top-p, min-p, and repetition penalty, and MUST allow a fixed seed for reproducible output.
- **FR-013**: System MUST tokenize prompts and detokenize output for the supported model families, including chat template rendering.
- **FR-014**: System MUST provide liveness and readiness endpoints whose state reflects model availability.
- **FR-015**: System MUST expose request counts, token throughput, and latency metrics in a scrapeable form.
- **FR-016**: System MUST authenticate API requests when an access key is configured, and MUST allow unauthenticated operation when it is explicitly disabled for local use.
- **FR-017**: System MUST enforce a configurable request-rate limit per client and return an explicit limit error when exceeded.
- **FR-018**: System MUST allow algorithms to be registered and selected at runtime without modifying core engine code.
- **FR-019**: System MUST fall back to a documented safe algorithm or backend when the selected one fails, and MUST report the degradation.
- **FR-020**: System MUST provide a command-line entry point for single-prompt generation with deterministic output given a fixed seed.
- **FR-021**: System MUST provide a diagnostic command that reports detected hardware, selected backend, and loaded model status.
- **FR-022**: System MUST emit structured logs and per-stage timing for each generation request.
- **FR-023**: System MUST ship at least six algorithms and two backends in Phase 1, and a conformance test suite that every algorithm and backend must pass.
- **FR-024**: System MUST keep its resource footprint bounded: no unbounded growth in retained context or request bookkeeping over a sustained run.

### Key Entities *(include if feature involves data)*

- **Model Descriptor**: A loaded model's identity — architecture family, quantization, parameter count, layer count, context limit, tokenizer vocabulary.
- **Hardware Profile**: The machine's capabilities — available devices, capacity and free memory, compute capability, host memory, CPU features.
- **Placement Plan**: Chosen device/host split for a model, predicted peak device and host memory, and the constraints that were checked.
- **Verification Record**: Predicted versus measured memory and throughput for one plan, with the deviation.
- **Conversation**: An ordered sequence of turns owning retained processed context, with a lifecycle from admission to completion or cancellation.
- **Context Snapshot**: A restorable point in a conversation's retained state.
- **Algorithm Registration**: A named capability with a stability level, its configuration, and its conformance status.
- **Backend Registration**: A named compute provider with detected availability, capabilities, and conformance status.
- **Access Key**: A credential identifying a client, with its permissions and rate-limit bucket.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can go from installation to a streamed chat answer using an existing OpenAI-compatible client in under 10 minutes, changing only the base URL.
- **SC-002**: On the Phase 1 reference hardware set (one consumer GPU and one CPU-only machine), predicted peak memory is within 10% of measured peak memory for every tested model.
- **SC-003**: No tested model-and-device combination fails with an out-of-memory error during load; every infeasible combination is refused before loading with an actionable message (100% of cases).
- **SC-004**: On a follow-up turn that reuses retained context, time to first token is at most 40% of the cold-start time for the same prefix at 4K context.
- **SC-005**: A conversation rewound to an earlier snapshot produces the same continuation as a fresh run from that point, for 100% of tested cases.
- **SC-006**: A new algorithm can be added without editing shared core engine code, verified by adding one algorithm whose change footprint is confined to its own module and a single registration.
- **SC-007**: Under a sustained 24-hour run at the target request rate, error rate stays below 0.1% and memory stays flat (under 5% growth between hour 1 and hour 24).
- **SC-008**: All six Phase 1 algorithms and both backends pass the shared conformance suite, with results published alongside the release.

## Assumptions

- **Source of intent**: `plan.md` is the authoritative blueprint. This specification covers Phase 1 only (v0.1.0–v0.3.0): eight core crates, six algorithms, two backends, roughly three months. Phase 2+ items referenced in the blueprint (Metal, ROCm, Vulkan, WASM backends, plugins, distributed serving, extra SDKs, gRPC-level RBAC) are out of scope here.
- **Project boundary**: the blueprint describes a new engine (`vipera.rs`) while the existing constitution in `.specify/memory/constitution.md` governs `vipera-opt` (the local Python planner at this repository root). Requirements here follow the same measurement-first philosophy; the constitution should be amended or a separate constitution created once this engine has its own project root.
- **Model format**: GGUF is the only model format in Phase 1; other formats from the blueprint are later phases.
- **Deployment shape**: single node, single process, models read from the local filesystem. No multi-tenant serving, no cluster orchestration, no cloud control plane.
- **Client compatibility**: "OpenAI-compatible" means the request/response contract of chat completions plus streaming, not every optional field; unsupported fields are ignored rather than rejected.
- **Hardware baseline**: Phase 1 targets one consumer-grade GPU with CUDA and one CPU-only machine; a machine with no supported accelerator must still work.
- **Defaults chosen where the blueprint is silent**: authentication is off by default for local binding and required when bound beyond loopback; rate limiting defaults to a per-key bucket configured by the operator; retained context is released on conversation end rather than on a timer.
- **Quality gate**: the shared conformance suite is the contract for algorithms and backends; a component that cannot pass it does not ship in Phase 1.
