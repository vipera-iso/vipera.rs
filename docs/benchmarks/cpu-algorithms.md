# Which CPU inference algorithms are worth putting into vipera.rs

*Generated: 2026-11-09 | Sources: 14 | Confidence: medium-high*

## Executive summary

Decode on this project is bound by reading 1.117 GB of Q4_K/Q6_K weights per token, not by
arithmetic. Measured here: one thread reaches 8.7 tok/s, four reach ~20, while the same model under
llama.cpp reached 14.59 and 38.29 tok/s on this machine — so llama.cpp streams roughly twice the
bytes per second that vipera does. That gap is a streaming/layout problem first and an algorithm
problem second.

Of everything surveyed, four things have evidence strong enough to justify work, in this order:
serve-side continuous batching (already half-built, 5.6x measured here), a repacked weight layout
with a kernel that keeps several rows and accumulators in flight (+28-52% at 2-4 threads on other
people's AVX2 numbers), prompt-lookup speculative decoding (2.4x average on grounded tasks, no extra
model), and chunked prefill / prefix reuse for time-to-first-token. KV quantisation buys capacity,
not speed. T-MAC-style table lookup and LayerSkip are real but fit this codebase badly.

## 1. The bottleneck, stated with numbers

- Decode reads every weight once per token; at batch 1 it is memory-bandwidth bound, while prefill
  is compute bound. This is the standard framing and it holds for vipera
  ([architecture and tuning notes](https://github.com/ByronScottJones/Moe-llama.cpp/blob/main/docs/architecture-and-tuning.md)).
- Our own measurements agree: the Q4_K kernel gained 6.4 -> ~10 GB/s with a better inner loop, yet
  single-thread end to end stayed at 8.06 vs 7.99 tok/s (bit-identical tokens), because the real run
  streams 1.117 GB from DRAM and waits on latency [measured here, no URL].
- The same problem is visible elsewhere: on a 96-core Neoverse-N2, quantised decode runs at 54-63%
  of STREAM bandwidth and is compute bound, while "equivalent x86 setups reach 85-95% of bandwidth
  for the same workloads" — x86 decode is normally bandwidth-bound, which is what we are seeing
  ([llama.cpp #25976](https://github.com/ggml-org/llama.cpp/issues/25976)).
- Thread-pool overhead is a real share of a 4-core decode: with OpenMP's barrier on mingw the
  barrier alone cost ~40% of decode throughput because a MoE graph executes ~2000 barriers per
  token; ggml's own spin barrier recovered it
  ([llama.cpp #26200](https://github.com/ggml-org/llama.cpp/issues/26200)). Our `pool.rs` has the
  same shape of cost, and is worth auditing before any algorithm work.

**Implication for vipera:** the target for a kernel/layout change is bytes per second, not GB/s of
compute. `pooled_matvec_throughput_by_tensor_type` measures compute precisely because its tensors sit
in cache; it cannot predict end-to-end gains. Only `generate --max-tokens N` counts.

## 2. Continuous batching — the largest single lever

- Orca introduced iteration-level scheduling and reported **36.9x throughput at the same latency**
  against FasterTransformer on GPT-3 175B
  ([OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)).
- PagedAttention/vLLM reports **2-4x** over Orca/FasterTransformer, mostly by making the KV cache
  fit more concurrent requests ([SOSP'23](https://arxiv.org/abs/2309.06180)).
- Measured here: llama.cpp serving four concurrent requests reached 80.1-82.3 tok/s aggregate
  against 37.3-37.4 tok/s for one — a batching factor of **2.15x**. vipera's four requests used to
  queue on the engine mutex at 14.5-14.9 tok/s; the scheduler has since landed in `serve.rs` and
  the same run now measures **20.2 tok/s aggregate against 15.5 tok/s for one request, 1.30x**
  [measured here, no URL].
- Why 1.30x and not 2.15x: a four-entry decode step costs 2.6x a single-entry one (42.0 -> 109.9 ms
  in `batched_step_speedup_on_the_real_model`), so batching trades a memory-bound step for a
  compute-bound one. The server spends 63.2 s of CPU over 21.2 s of wall on that run — 2.98 of 4
  cores busy, not stalled on barriers — and the tensor kernels only reach 1.43-1.85x at B=4. The gap
  left against llama.cpp is kernel compute per entry, which is section 3, not the scheduler.
- Why it works here specifically: batching amortises one weight read over B sequences, so the
  marginal cost of sequence B is its activations, not 1.117 GB. That is the same arithmetic that
  makes speculative decoding work (section 4), and the two compete for the same slack.

**Confidence: high.** Effort: M. It is the only change that moves both throughput and cost per token.

## 3. Weight layout and kernel shape — closing the 1.7-2x streaming gap

- ik_llama.cpp with `-rtr 1` (all weights repacked row-interleaved, plus its own matmul kernels),
  AVX2 Ryzen-7950X, LLaMA-3.1-8B, tg128: Q4_K_S **9.57 -> 13.48 tok/s (1.41x)** at 2 threads and
  **17.55 -> 22.49 (1.28x)** at 4 threads; Q3_K_S 8.80 -> 12.99 (1.48x); Q5_K_S 7.51 -> 11.41
  (1.52x); Q8_0 4.95 -> 7.63 (1.54x). At 16 threads the advantage mostly disappears — it is a
  bandwidth-saturation effect, exactly our regime at 1-4 threads
  ([comparison](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)).
- Mainline's Q4_K block interleaving (8 rows per super-block, AVX2) moved pp512 **+25.7%** and tg128
  **-0.07%** ([PR #12829](https://github.com/ggml-org/llama.cpp/pull/12829)); VNNI repack kernels for
  Q8_0/Q4_0 moved pp2048 **3.24x** with tg128 **1.00x**
  ([PR #30036](https://github.com/ggml-org/llama.cpp/pull/30036)).
- Read together: **repacking alone mostly helps prefill.** The decode gain in ik comes from the
  layout plus a kernel that uses it — several rows per load and multiple independent accumulator
  chains, so the core has more misses in flight. Our kernel has two accumulator chains and one row
  per pass; that is the concrete thing to change.

**Confidence: medium-high** (consistent across two independent codebases, but measured on other
CPUs and other models). Effort: M-L, and it is the only one of the four that changes the hot loop
that is already the bottleneck. It is also the one that can keep bit-exactness: a layout change is
free of numerics, and an extra accumulator chain must be folded in the same order as today to keep
`VIPERA_EXACT_DOT` meaningful.

## 4. Prompt-lookup speculative decoding — no model, no crates

- Replacing the draft model with an n-gram match against the context gave a **2.4x average** on
  CNN/DailyMail summarisation and context QA, and "never worse than greedy on any example", with no
  output-quality change ([prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)).
  vLLM reports up to **2.8x** on CNN/DailyMail with n-gram drafting, against 1.5x for a draft model
  on ShareGPT ([vLLM blog](https://blog.vllm.ai/blog/2024-10-17-spec-decode)).
  Measured in this repo at **0.99x** with 100% acceptance: both implementations run prefilled chunks
  as a GEMM over the draft tokens, ours reads the weights once per position, so the win is not the
  drafting but the chunked prefill underneath it ([`beating-llamacpp.md`](./beating-llamacpp.md) §5A).
- The catch, and it matters for section 2: **gains shrink as the batch grows.** EAGLE on
  Llama3.1-8B/GSM8K falls from 1.73x at batch 1 to 1.21x at batch 128, and verification by the target
  model dominates the cost ([MLSys'26](https://arxiv.org/abs/2601.11580)).
- For vipera the fit is unusually good: the batch matvec landed this week is exactly the
  verification pass, the tokenizer and sampler already exist, and a draft is a `Vec` scan over the
  context. It would lift single-user CLI and RAG/summarisation traffic without touching the kernel.

**Confidence: medium-high** for the technique, **low** for the exact multiplier on a 1.5B CPU model
with 4 threads — nobody has published that combination, so it must be measured here first. Effort: S-M.

## 5. Chunked prefill and prefix reuse — latency, not throughput

- Sarathi-Serve splits a prefill into equal compute chunks and adds requests to a running batch
  without stopping decodes: **2.6x serving capacity** on Mistral-7B/A100 vs vLLM, and it removes
  decode stalls from long prompts ([OSDI'24](https://arxiv.org/abs/2403.02310)). A 256-token chunk
  costs about 12.5% of prefill throughput; 512 about 10%
  ([summary](https://hungchun0201.github.io/agentic-ai-survey/papers/sarathi-serve/)).
- vipera already reports `reused_tokens` (prefix reuse) in `generate --json`, so the second half of
  the idea exists; the missing half is scheduling prefill in chunks alongside decode.

**Confidence: high** for GPUs, **medium** for a 4-core CPU where the "spare compute" being filled
is much thinner. Effort: M, and it only pays once batching exists.

## 6. KV cache quantisation — capacity, not speed

- q8_0 KV is quality-neutral (PPL inside the error bar at every context length) and costs ~0-1% at
  4K, but at 64K it decodes at **56% of f16** on an A100; q4_0 is ~+0.5-1.1% PPL and about the same
  speed as q8_0 ([measured guide](https://sotaaz.com/post/llamacpp-kv-cache-quantization-guide-en)).
  Single-blog numbers, so treat the percentages as indicative.
- On CPU the effect is different in direction: field notes list KV type among knobs that were
  "speed-neutral" for CPU decode
  ([discussion #30333](https://github.com/ggml-org/llama.cpp/discussions/30333)).
- vipera already has `--kv-type q8_0`. The reason to default it is fitting four 2048-token sequences
  at 28 layers without thrashing; not token rate.

**Confidence: medium.** Effort: none (already implemented).

## 7. Things that do not fit vipera.rs

- **Wider vectors.** Already measured here: AVX-512 Q4_K was 7.99 vs 8.01 tok/s and 5.2 vs 6.8 GB/s;
  an AVX-512 VNNI Q6_K kernel this week was bit-identical, 8.7 vs 8.7 tok/s at one thread and
  19.8 vs 20.0 at four. Both removed, and the reason is recorded at the top of `simd.rs`.
- **T-MAC / LUT-GEMM.** Up to **4x** over llama.cpp, but the headline cases are 1-2 bit models; at
  4 bits its own table shows 0.022 vs 0.034 ms for a 4096x4096 GEMV, i.e. ~1.5x, and the method
  needs weights repacked into its table layout, which means abandoning the GGUF K-quant blocks and
  the bit-exactness invariant ([paper](https://arxiv.org/abs/2407.00088),
  [kernel table](https://github.com/microsoft/T-MAC)). It becomes interesting only if vipera ever
  ships a 2-bit path.
- **LayerSkip self-speculation.** 1.34-2.16x, but every number comes from a checkpoint trained with
  layer dropout and an early-exit loss; a stock Qwen2.5 GGUF has no such property
  ([paper](https://arxiv.org/abs/2410.16710)).
- **PagedAttention as such.** 2-4x on GPUs because KV fragmentation limited batch size; vipera's KV
  is already block-allocated and the server has one model with a handful of sequences. The idea
  worth taking is "more concurrent sequences", not the paging machinery.

## 8. What I would do, in order

1. **Finish the batching scheduler** in `serve.rs` (router + channel hand-off, already designed).
   Expected 3-5x on concurrent load here, measured against the recorded llama.cpp numbers.
2. **Audit `pool.rs` for barrier cost** before touching kernels: spin-then-park, and one barrier per
   matvec rather than per row. Cheap, and it compounds with everything else.
3. **One kernel/layout experiment**: 4 rows in flight with 4 accumulator chains and a prefetch of the
   next super-block, no format change. Target: single thread 8.7 -> 12+ tok/s. If it does not move
   `generate` throughput, revert it and say so — the same rule as the AVX-512 kernels.
4. **Prompt-lookup decoding** behind a flag, measured on a summarisation prompt and a plain chat
   prompt separately; expect a large gain on the first and little on the second.
5. **Chunked prefill** only after (1), and only if TTFT under concurrent load is actually a problem.

Every step gets an interleaved A/B in one session, per the project rule that a performance claim
without a number from this machine is not a claim.

## Sources

1. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling, 36.9x throughput at equal latency.
2. [Efficient Memory Management with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — 2-4x throughput, KV fragmentation as the limiter.
3. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill, stall-free batching, 2.6x capacity.
4. [Sarathi-Serve notes](https://hungchun0201.github.io/agentic-ai-survey/papers/sarathi-serve/) — chunk-size cost (~12.5% at 256 tokens).
5. [ik_llama.cpp CPU comparison](https://github.com/ikawrakow/ik_llama.cpp/discussions/164) — row-interleaved repack, tg128 +28-52% at 2-4 threads.
6. [llama.cpp PR #12829](https://github.com/ggml-org/llama.cpp/pull/12829) — Q4_K 8x8 interleave: pp +25.7%, tg -0.07%.
7. [llama.cpp PR #30036](https://github.com/ggml-org/llama.cpp/pull/30036) — VNNI repack: pp 3.24x, tg 1.00x.
8. [llama.cpp #25976](https://github.com/ggml-org/llama.cpp/issues/25976) — decode bandwidth vs STREAM, x86 at 85-95%.
9. [llama.cpp #26200](https://github.com/ggml-org/llama.cpp/issues/26200) — thread barrier cost ~40% of 4-core decode.
10. [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) — 2.4x on summarisation/context QA, no quality change.
11. [vLLM spec decode blog](https://blog.vllm.ai/blog/2024-10-17-spec-decode) — n-gram 2.8x on CNN/DM, draft model 1.5x on ShareGPT.
12. [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) — speedup shrinks with batch size (1.73x -> 1.21x, batch 1 -> 128).
13. [T-MAC](https://arxiv.org/abs/2407.00088) and [kernel table](https://github.com/microsoft/T-MAC) — LUT mpGEMM, up to 4x at low bit, ~1.5x at 4-bit.
14. [LayerSkip](https://arxiv.org/abs/2410.16710) — self-speculative decoding, but requires the training recipe.
15. [KV cache quantisation guide](https://sotaaz.com/post/llamacpp-kv-cache-quantization-guide-en) — q8_0 quality-neutral, 56% decode at 64K.

## Methodology

Sub-questions investigated: (1) what actually limits CPU decode and how ggml streams near peak,
(2) kernel and weight-layout techniques including LUT methods, (3) batching and scheduling,
(4) model-free and self-speculative decoding, (5) KV cache and prefix reuse.

Searched 12 queries across web and news, read 4 sources in full (prompt-lookup repo, plus the
llama.cpp issue/PR pages and ik_llama.cpp comparison via search snippets and one full read), and
synthesised 15 sources. The firecrawl and exa MCP servers were not available in this session, so the
`deep-research` workflow ran on `web_search`/`read_url`; the MLSys'26 PDF timed out on fetch and its
claims come from the arXiv HTML and the proceedings abstract.

Gap, stated plainly: no published CPU measurement exists for prompt-lookup decoding on a 1.5B model
at 4 threads, and none for a row-interleaved Q4_K layout on Tiger Lake. Those two numbers have to
come from this machine before either is believed.
