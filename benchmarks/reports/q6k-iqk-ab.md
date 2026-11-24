# A/B — VIPERA_Q6K_IQK (bench-q6k-iqk, 2026-11-25)

Machine: Intel i5-11300H (4 core / 8 thread), AVX2 + F16C + AVX-512 VNNI (`vipera profile --json` → `cpu_simd: avx512-vnni`); pinned with `taskset -c 2-7`, `VIPERA_THREADS=8`.
Model: `~/.vipera/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (Qwen2.5-1.5B, Q4_K_M; 29 Q6_K tensors — the model has Q6_K tensors, so prefill and decode ran on it).
Prompt: `/tmp/q6k-bench-prompt.txt` — 4,141 tokens (`vipera tokenize`).
Method: `vipera generate <model> --prompt <file> --ctx 8192 --max-tokens 16 --temperature 0 --json` with `VIPERA_PROFILE_STEP=1`; 3 runs per arm, median of per-run values. Prefill ms = median of `PROFILE chunk` lines' "this pass" (sum per run); decode step ms = median of `PROFILE decode` lines' "this pass"; decode tok/s = 1000 / decode step ms.

## Prefill (4,141 tokens)

| Arm | Prefill ms (median) | Prefill tok/s |
|---|---:|---:|
| `VIPERA_Q6K_IQK=0` (baseline) | 33,036.79 | 125.3 |
| `VIPERA_Q6K_IQK=1` | 36,329.72 | 114.0 |
| change vs baseline | +10.0% | −9.1% |

## Decode

| Arm | Step ms (median) | Decode tok/s | q6k phase (median) |
|---|---:|---:|---:|
| `VIPERA_Q6K_IQK=0` (baseline) | 72.97 | 13.70 | 109.27 |
| `VIPERA_Q6K_IQK=1` | 247.18 | 4.05 | 194.83 |
| change vs baseline | +3.39x | −70.5% | +78.3% |

## MLA (both arms on `VIPERA_Q6K_IQK=1`)

| Arm | Step ms (median) | Decode tok/s |
|---|---:|---:|
| `VIPERA_MLA=1` | 251.15 | 3.98 |
| `VIPERA_MLA=0` | 247.78 | 4.04 |
| change | +1.4% | — |

`VIPERA_MLA` is not referenced anywhere in the tree (verified by grep), so the two arms are expected to be identical; the measured difference is within run-to-run noise.

## Honest read

On this machine the repack path is slower than the row kernels it replaces: prefill −9.1% and decode −70.5% (the q6k phase grows about 2x). The measured numbers above are the A/B result, not a claim that the path is faster.
