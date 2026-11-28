# Constraints

Last reviewed: 2026-11-06 · Owner: repo maintainer · Applies to: every human and every agent that edits this repo.

This file is the written bar for the `vipera.rs` engine in this workspace. It exists because the agent writes far
more code than anyone reads line by line, so "good enough" has to be written down and checked, not remembered.

**Do not weaken this file to make a change pass.** Tightening is silent; loosening needs a row in Exceptions.

## Floor (always enforced)

These five are review-checked on every change. No tooling required, no exceptions without a row below.

- **No new suppression comments**: `# noqa`, `# type: ignore`, `# pyright: ignore`, `// eslint-disable`, `@ts-ignore`.
- **No unimplemented work presented as done**: `raise NotImplementedError` stubs, `pass  # TODO`, empty `except: pass`,
  functions that return a hardcoded value where the caller expects a computed one.
- **No skipped or deleted tests** without the reason in the same change (`@pytest.mark.skip`, `xfail`, removed assertions).
- **No secrets in source**: API keys, tokens, connection strings. Report the location, never the value.
- **A performance change names the skills it loaded**: its PR body carries a `## Skills` section naming the
  phases it used (`vipera-perf-loop` §2, where phases 1, 3 and 6 are mandatory). The `skills` CI job is red
  without that section, so this is checked by a machine and not by memory.
- **This file is not edited in the same change that was failing a rule.**

## Rules for the agent — what "code linh tinh" means here

1. **Work from the request.** The conversation is the brief. A change does not need a spec requirement or a
   constitution principle behind it; if the request is ambiguous, pick the smallest defensible reading and say which.
2. **No unrequested features.** No extra flags, endpoints, config keys, abstractions, or "while I'm here" additions. A new
   file needs a reason the task stated; a new dependency needs explicit approval.
3. **No drive-by refactors.** Do not reformat, rename, reorder imports in, or "clean up" files the task did not name.
   Unrelated improvements are proposed in words, not committed as diff.
4. **Read before writing.** Reuse what exists under `src/vipera/` instead of adding a parallel helper. Never invent an API
   that is not in the tree — check the file, then call it.
5. **No speculative generality.** No interface with one implementation, no plugin hook with no plugin, no config knob with
   no consumer. The smallest change that fully solves the task wins.
6. **No weakened checks to reach green.** Fix the cause. Skipping a test, loosening an assertion, adding a suppression, or
   editing a threshold is a finding, not a fix — and is reported as such.
7. **State assumptions out loud.** When the request is ambiguous, pick the smallest defensible reading and say which one in
   the reply. Guess silently only when the guess is reversible and unobservable.
8. **Evidence before claims.** Run the exact check and quote its result. If a check could not run, say "not run" and why.
   Never describe compiled-but-unverified behavior as tested.
9. **One concern per change.** A bug fix contains no feature; a feature contains no refactor; a refactor changes no behavior.
10. **Design notes are context, not a gate.** `specs/` and `docs/design/` say how an area got here and are worth reading
    before changing it; they do not authorise or forbid work. Ask the user, not the document, when the two disagree.

## Enforced with numbers

Baselines are measured on 2026-11-06, not aspirational. A number may only move in the stricter direction.

| Dimension | Rule | Checked by | Runs at |
|-----------|------|------------|---------|
| Lint | ruff findings ≤ 154 (today's baseline) | `ruff check src/vipera tests` | task end |
| Types | `mypy src/vipera --ignore-missing-imports` output ≤ 8 lines (today's baseline) | `mypy src/vipera --ignore-missing-imports --no-error-summary` | task end |
| Tests | `pytest tests/` passes once the dev extras are installed; zero new skips | `python3 -m pytest tests/ -q` | task end |
| Rust tests | `cargo test --workspace` — 0 failures, 0 ignored except the release-mode measurements listed in EXC-004, which the gate runs with `--ignored` | `cargo test --workspace`, then the release step in §How to run the checks | task end |
| Rust lint | `cargo clippy --workspace --all-targets --features acme` — 0 warnings | `cargo clippy --workspace --all-targets --locked --features acme` | task end |
| Changed lines | New behavior ships with a test that fails without it | review of the diff | review |

Known debt at baseline, not to be added to: `src/vipera/planner/ladder.py` references an undefined `LadderStep`
(4 errors), `src/vipera/planner/cache.py:64` an undefined `Plan`, `src/vipera/gguf/reader.py:37` uses `any` where
`Any` is meant, and `src/vipera/runtime/verifier.py:26` assigns `None` to a `list[Any]`. Fixing them is allowed and
welcome; adding to the count is not.

## Measured, not yet enforced

| Metric | Today | Direction |
|--------|-------|-----------|
| Project coverage | unknown — `pytest` and `pytest-cov` are not installed in the system interpreter | must be measured before it is enforced |
| Planner unit tests | 0 (tests cover gguf, hardware, a risk gate only) | must increase with every planner change |
| Diff-scoped checks | available — this is a git repository, and every change lands through a PR whose CI run is the confirmation | every performance claim also carries a before/after in `docs/benchmarks.md` |

## Rejected performance directions (measured, do not re-run blind)

Measured on the reference machine (i5-11300H, 8 threads, Qwen2.5-1.5B Q4_K_M) against
llama.cpp `e574a17` (pp128 230.8 tok/s, tg64 37.7 tok/s). Baseline vipera: pp128 186 tok/s,
decode 30–33 tok/s. Full ledger in `HANDOFF.md` (local, gitignored).

| Direction | Measurement | Verdict |
|---|---|---|
| Prefetch the next two rows in `fill_rows_batch` | decode 33.5–34.2 → 37.5–37.7 ms/step; prefill 186 → 172 tok/s | rejected: the real 900 MB stream is slower; `ffn_gate`'s 7.7 MB tensor is L3-hot and hid it |
| `MAX_BATCH` 128 → 256 | `q4k_tiled_batch_sweep` GFLOPS/token 161 (B=16) → 191 (32) → 185 (64) → 203 (128); fit of `prefill n=64/128` puts per-chunk weight cost at ≈0 | rejected: prefill is compute-bound, a wider chunk buys <2% for 16 more monomorphized layer stacks |
| AVX-512 integer tile at `B = 1` (decode) | decode 32.7–34.6 → 112–115 ms/step | rejected: 15 of the tile's 16 `dpbusd` columns are zeros |
| Thread count 4 / 6 instead of 8 | decode 26.7 / 28.7 tok/s vs 33.4 | rejected: 8 logical CPUs are the best setting here |
| Port llama.cpp's `bsums`-based Q6_K kernel | `output` Q6_K already 5.2 ms/call = 36.5 GB/s, near the ~43 GB/s read ceiling | not attempted: the bytes, not the kernel, are the limit |
| Narrow the prefill batch 128 → 32 **for every level** | wall clock, 4096 tokens: 30.6 s → 35.1 s on the tiled arm (+14.9%) | rejected as a default: the tiled kernel wants the wide chunk. The off-tiled arm is the opposite (155.2/152.7 s → 139.6/140.1 s, −9.1%), so it is kept as a conditional `prefill_chunk()` — ledger 2026-11-24 |
| Repack Q4_K at 4.5 bpw (packed scale header, or an 8x8 GEMM that skips the `wq` tile) | gate bench `bench_repacked_q4k_scale_header`: 148-byte header 131.95 vs 115.86 ns/block (114%); probes put the whole unpack at -8.8% of q4k, of which the nibble split is -1.5%, the header 0 and the scratch memset 0 | rejected: what is left is `q4k_scale_min`, two f16 conversions and sixteen products per row-block — required math; removing it needs +33% f32 bytes or f16 precision loss |
| Hoist `dvec[row] * scales[k][row]` out of the Q6_K band loop into a per-window table | counted, not built: `dvec` and `scales` are refilled per memory block inside `for b in 0..blocks`, and each row belongs to exactly one band, so the product is already computed once per `(block, sub-block, row)` | rejected: the factors are not loop-invariant, and a table would add a load per use for the same multiply |
| Repack Q4_K yet again on the **broadcast** axis (ik-style i8 weight vectors, no per-dot `vpbroadcastd`) | gate bench `repacked_gemm_form_against_the_broadcast_form`: the two microkernels for one 16x16 tile, (A) broadcast a scratch dword per dot, (B) load a repacked 16-row weight zmm and broadcast the activation dword; three release runs gave A 496.3 / 506.6 / 500.0 GFLOPS against B 365.9 / 508.0 / 501.7, median **+0.3%** (the -26% run was a load spike) | rejected again, this time by direct measurement: once the intrinsics inline, the per-dot broadcast is not what bounds the kernel. The old `NO_WEIGHT` probe (+29% / +56%) priced compiler hoisting, not the broadcast, so ik_llama.cpp's 15-20% lead has to come from elsewhere — the unpack this table already priced at 8.8%, Q6_K, and thread scheduling. Two ways the instrument lied first, worth remembering: 16 accumulators fed the same inputs, and a test function without `target_feature` (13.9 GFLOPS, 30 cycles an instruction) |
| `vpermb` (VBMI) split of the Q4_K nibbles | skipping just the split (scales and f16 kept): q4k median 432.05 -> 425.42 ms (-1.5%), chunk 627.74 -> 615.72 (-1.9%), 5/5 pairs | rejected: the split is ~1% of a chunk, under the 3% gate; the rest of the unpack is `q4k_scale_min` and f16 |
| Tile row window 32 -> 64 in the Q4_K/Q6_K tiled kernel | five interleaved pairs of two binaries: chunk median 783.08 -> 782.35 ms (-0.09%), q4k 540.22 -> 542.43 (+0.4%) | rejected: neutral; 32 rows keeps `buf` at 16 KB inside L1 |
| `ffn_down` kernel rewrite, from the "24% of a chunk at 1 GB/s" lead | the skip-down probe is real (−23% chunk), but down is 29% of a layer's MACs and costs 27.6%: per MAC gate 0.569 ns, Q4_K down 0.621, Q6_K down 0.810 (and Q6_K carries 46% more bytes). Same-shape kernel bench single thread: 1536×8960 at 21.6 ms is *faster* than 8960×1536 at 24.2 ms; whole product at 1/4/8 threads 23.4 / 8.7 / 7.8 ms for both, so 4 threads equal 8 and the product is ALU-bound. Skipping the Q6_K 6-bit unpack outright: 11.149 → 11.159 ms | rejected: no deficit to recover. The "1.0 vs 22.5 GB/s" that started it compared a compute-bound `B=128` product with a bandwidth-bound `B=1` one, and the b128 `GB/s` column divides by a pass total (it prints ~0). Vectorising the unpack is also closed: the probe prices it at zero |
| Swap the tiled kernel's loops so one weight broadcast feeds all 8 halves | three no-op probes on `q4k_tiled_batch128_throughput` (single thread, 8960×1536 / 1536×8960): broadcast replaced by a constant +29% / +56%, epilogue reduced to one `add` +33% / +42%, both +100% / +127%, and the same words read from a stale buffer +11% / +8% | not pursued: the swap only trades 512 weight broadcasts for 512 activation loads per row-block (576 either way). Reusing both dimensions needs `rows × halves` accumulators and 4×8 is already 32 `zmm` before `facc`/`codes`. Only 2 rows × 4 halves (16 `zmm`) reaches 576 → 384 loads per row (−33%) |
| Two rows × four halves in the Q4_K tiled kernel (`VIPERA_Q4K_ROW2`, written and then removed) | bit-identical to the eight-row path (`the_two_row_path_is_bit_identical_to_the_eight_row_one`, four shapes incl. an odd row count); single thread 152.3 / 150.3 GFLOPS against 151.7 / 152.6 off (gate shape) and 169.9 / 168.7 against 175.6 / 176.9 (down shape); chunk 4, 5 interleaved pairs, median 786.81 vs 800.59 ms (−1.7%, 4 of 5 pairs) | rejected: under the 3% gate for ~100 dense lines. A third fewer loads moved nothing, so load ports are not what binds the kernel; the `NO_WEIGHT` probe's gain is the compiler hoisting one `set1` per group instead of eight. A first version with a runtime half index was 21% slower (accumulators spilled to stack) |
| Fold the Q4_K tiled kernel's per-row weight broadcast into `vpdpbusd`'s memory operand | counted, not built: `q4k_matvec_tiled_range::<128>` emits exactly one `vpbroadcastd` per `vpdpbusd` (64:64), and the `m32bcst` form is the *signed* source while the weight dword sits in the unsigned one; the activations are `i8` clamped to ±127, so the two operands cannot be swapped | rejected: no encoding puts the unsigned operand in memory, so the pair cannot become one instruction. LLVM 1.99 also never folds a load into VNNI on its own. The remaining gap to the `dpbusd` roof (64% measured, probes put ~25% on the broadcast and ~30% on the required epilogue) needs a transposed tile — weight vector across 16 rows, codes in the broadcast lane — plus hand-written `asm!`; that is a CPU-contract change, not a kernel tweak, and nobody should re-derive it from scratch |
| New tile-resident attention micro-GEMM, wider than the existing `VIPERA_ATTN_QBLOCK` block | at depth ~380 the existing query-blocked fused kernel already cuts attention 24.5% (95.77 -> 72.32 ms) while the chunk moves +1.2% (inside noise); the 6-head f16 score dot is ~85% of the AVX2 FMA peak, leaving <=1.3x on the arithmetic | **REOPENED AND REVERSED 2026-11-23 (PR #65)**: the row was measured at depth 380, where attention is 5-10% of a chunk; at 1.5k tokens it is 16% and at 3.7k it is 26%, so the "at most ~1% of a chunk" ceiling came from the depth, not from the kernel. The fused kernel's dot was not near the FMA roofline either: it ran the whole attention phase at ~1.4 GFLOPS. A 32x64 tile through `simd::gemm` (the layout llama.cpp's tiled CPU FA uses) measured attention 6637 -> 3464 ms (-48%, 5/5 pairs) and chunks -13.0% at 3.7k. Kept behind `VIPERA_ATTN_GEMM` because it flips a greedy token; the default path is untouched |

| Q4_K repack, **second look** — closed on the broadcast mechanism, not on unpack | ik_llama.cpp `1180df7` built on this machine (8T, same GGUF): pp512 **268.7 / 245.7** vs vipera **209.6**; pp4096 **199.4 / 191.6** vs **140.3** (167 with the attention tile); `-ub 128` still 237.2, so ubatch is only worth 3-8% | **REOPENED** (the repack row above was closed on the price of *unpack* — 8.8% of q4k, correctly). The real gap is **1 broadcast + 1 scalar load per `vpdpbusd`**: their hot loop loads weights as `__m512i` holding several row-contiguous i8 rows, with the activation as a vector load rather than a broadcast; our `NO_WEIGHT` probe priced exactly that at **+29-56%** of the kernel. Port = repack at load time (row-contiguous i8 weights + Q8 activations), GGUF as the fallback |
| Interleave the Q3_K band's four `dpbusd` rows and 32-byte align its `codes` buffer (`a7e886e`, reverted) | kernel A/B, same shape, two arms alternating in one process, best of 6: interleaved 182.10 vs row order 181.81 ns/row-block with the aligned buffer (0.998×), 173.02 vs 173.19 without it (1.001×); the aligned buffer is 5.0% **slower** than the plain `[u8; 256]` (181.81 vs 173.19), and its codegen shows 8 `vmovdqu` → `vmovdqa` plus 24 prologue instructions for the stack realign | rejected: the reorder is neutral because LLVM already schedules the four chains, and the alignment pays for itself in the prologue, not in the stores. End-to-end could not resolve 5% (single pair 6 minutes apart, host drifted 12%); ledger 2026-11-27 |
| Read a Q3_K band group's four code words with one 16-byte load (`VIPERA_Q3K_WIDE`, written and then removed) | band bench, H=2, four interleaved rounds in one process: 48.15 / 48.00 / 47.92 / 48.04 ns per half with four 4-byte loads vs 47.96 / 48.05 / 47.81 / 48.11 with the 16-byte load (median **+0.1%**, gate was 3%) | rejected: **neither formulation survives codegen.** `castsi128(loadu 16B)` + `vpermd` folded to `vpbroadcastd (mem)` per word, and the follow-up `loadu 16B` + `vpextrd` + `set1` did too: the H=2 kernel contains 0 `vpermd` and 0 `vpextrd` and 55-69 memory-source broadcasts. The four "narrow loads" are therefore already one fused broadcast-load instruction each, so there is nothing to remove and the 55% the weight probe prices is that instruction's issue cost, not store forwarding. A later probe that wants to split load from broadcast has to defeat the fold (an opaque 16-byte load), and the expected ceiling is then the 3 removed loads only |
| Repack Q3_K at load into `dpbusd`-ready 320-byte blocks (ADR-008, **deferred, not built**) | `bench_q3k_decode_codes` with a pre-decoded block built from the same tensor: 43.2–46.1 vs 46.4–50.5 ns per half at H=2, **+7.6% median** over eight round pairs; density 110 → 320 B/block = **2.91x** | deferred: the 6% gate cleared, but the probe is cache-hot and the real block is 2.91x the bytes streamed at pp512. A repacked 6.9 GiB model is ~20 GiB on a box with 15 GB and ~3 GB free, so the traffic cost cannot be measured here. Anyone re-running this needs a bigger box first, not another kernel; `q3k_band_rows_packed` and `ADR-008` are the starting points |

| Eight tile halves over two rows in the Q3_K band (H=8, N=2 — decode once per chunk), test-only bench arm | `bench_q3k_decode_codes`, H=2 arms interleaved: 52.93 / 53.20 / 52.93 / 53.01 ns per half against 48.30 / 47.90 / 47.95 / 47.91 for the shipped pair (**−9.6 to −11.1%, median −10.7%**) | rejected: the decode-once-per-chunk saving is ~5.7% of the band at most, and halving the rows per pass doubles the activation side per output and the activation traffic per pass — 16 accumulators were not the problem, the shape was |

Where the time actually goes (measured, `VIPERA_PROFILE_STEP=1`): prefill chunk q4_k 68%,
q6_k 19%, attention 5%, norm 2%, rest 6%; decode q4_k 61%, q6_k 28%, attention 7%. So attention
is not the prefill bottleneck, and any further work has to move Q4_K streaming (~30 GB/s against
llama.cpp's ~39 GB/s over the whole step).

That phase table is a short-prompt one and attention's share is not a constant: it grows with depth
(52 ms of the first chunk, 162 ms of the twelfth at 1.5k tokens, 26% of the total at 3.7k). Read the
q4_k/q6_k shares as depth-independent and attention's as a function of the prompt, per the reopened
row above.

## Exceptions

| ID | Rule | Path | Reason | Owner | Expires |
|----|------|------|--------|-------|---------|
| EXC-002 | Stdlib-only (`research.md` R3), dependency approval (agent rule 2) | `crates/vipera-server/Cargo.toml`, `crates/vipera-server/src/serve.rs`, `crates/vipera-server/src/tls.rs`, `bins/vipera/Cargo.toml` (dev-dependency, client side of the TLS test) | Serve HTTPS. Writing TLS by hand is not something to ship, and the workspace had no alternative that was not one; `rustls` 0.23 with `ring`, no default features, is the whole of it. Everything else in the engine stays stdlib-only, and `vipera serve` without `--tls-cert`/`--tls-key` links nothing new at run time. Rollback: drop the two flags, one revert, and the plain-HTTP path is untouched. Full record: the `vipera-server` crate doc comment and the README. | vipera-iso | 2027-02-11 |

| EXC-003 | Stdlib-only (`research.md` R3), dependency approval (agent rule 2) | `crates/vipera-server/Cargo.toml` (`acme` feature), `crates/vipera-server/src/acme.rs` | ACME issuance (RFC 8555) behind feature `acme`, **default off**: `instant-acme` with `hyper-rustls`/`rcgen`/`ring`, plus `tokio` and `serde_json` for the client and the stored account. Writing an ACME client by hand is not something to ship. Nothing in the engine links these; `vipera serve` without `--acme-domain` compiles them out. Rollback: feature off, one revert. | vipera-iso | 2027-02-11 |
| EXC-004 | Rust tests: 0 ignored | `crates/vipera-backend-cpu/tests/prefix_cache.rs`, `crates/vipera-backend-cpu/tests/alignment_probe.rs`, `crates/vipera-backend-cpu/src/{model,quant,simd,tiled}.rs`, `crates/vipera-backend-cpu/tests/{batch,matvec_throughput,win_vs_llama}.rs` | Measured on 2026-11-19: a default `cargo test --workspace` reports 52 ignored tests, and they are measurements rather than checks — `bench:`-style timings, plus model-dependent tests that time a prefill. A debug build cannot run them at a usable speed (measured: one 200 MB matvec takes over 300 s in debug), so they stay out of the default run. The two that do assert — prefix cache and the alignment probe — are run by the gate in release with `--ignored`; without a model they print the reason and pass. Tightening means giving the model-dependent ones a checked-in fixture model. | vipera-iso | 2027-02-11 |
| EXC-005 | Stdlib-only, and `unsafe` only in `crates/vipera-backend-cpu/src/simd.rs` | `Cargo.toml` (`memmap2`), `crates/vipera-backend-cpu/src/bus.rs` | Tiered weights: the model file is mapped read-only instead of read into a `Vec<u8>`, and the pass keeps a window of layers resident (`madvise(WILLNEED)`) while handing the rest back (`madvise(DONTNEED)`). Measured on the 1.1 GB Qwen2.5-1.5B Q4_K_M: same speed as the heap buffer (prefill 187 vs 186 tok/s, decode 30.6 vs 30.8 tok/s), max RSS 1977 MB → 553 MB at `VIPERA_RAM_LAYERS=4`, decode 26.5 tok/s at `VIPERA_RAM_LAYERS=1`. `memmap2` (0.9, no transitive deps worth the name) owns the `unsafe`; `bus.rs` keeps the two calls it still needs — mapping and `DONTNEED` — in one `#[allow(unsafe_code)]` module, with the read-only-immutable-file contract stated there. | vipera-iso | 2027-02-11 |
| EXC-006 | Stdlib-only, and `unsafe` only in `crates/vipera-backend-cpu/src/simd.rs` | `Cargo.toml` (`cudarc`), `crates/vipera-backend-gpu/` | A CUDA backend: `cudarc` 0.19 (driver + NVRTC + `fallback-dynamic-loading`, so a machine without a driver still builds and reports why) binding `libcuda`, and the crate-wide `unsafe_code = "allow"` it needs — the whole crate is the FFI boundary. First kernel measured on a GTX 1650 (compute 7.5): a 8960x1536 Q4_K matvec matches the CPU dequantizer to 1.04e-6 relative over 64 rows and runs at 29.6 GB/s with the weights resident, i.e. CPU parity, not a win yet. Nothing in the CPU engine or the server links this crate; since 2026-11-24 `bins/vipera` links it behind its non-default `cuda` feature (`generate --backend cuda`), and the default build still contains no device path. | vipera-iso | 2027-02-11 |

An exception needs an owner and an expiry; without both it is a removal, not an exception.

### Closed

| ID | Closed | Why |
|----|--------|-----|
| EXC-001 | 2026-11-20 | ADR-001 measured its own premise and lost: on the same shape, B=128 and one thread, the Rust tiled Q4_K GEMM ran at 153.7 GFLOPS (8960x1536) / 174.4 (1536x8960) against ggml's repacked 125.1 / 128.8, so binding ggml would have made prefill 23-35% slower. A Zig port of the same kernel measured 1.5% (inside noise), and the decode path is a bandwidth stream where a different compiler produces the same instructions. No C/C++ toolchain, no `vendor/ggml/`, no `cc` build-dependency, and the `vipera-ggml` prototype crate is deleted; the numbers stay in `docs/benchmarks.md`. Reopening needs a new ADR and a new exception, not this row. |

## How to run the checks

```bash
# vipera-opt (Python, the `src/` and `tests/` trees)
ruff check src/vipera tests
mypy src/vipera --ignore-missing-imports --no-error-summary
python3 -m pytest tests/ -q

# vipera.rs (Rust workspace) — both baselines are zero and must stay zero
scripts/gate.sh              # every stage, in order
scripts/gate.sh lint         # one stage
scripts/gate.sh --list       # fmt lint build test features measure gpu
```

The Rust side has one executable form, `scripts/gate.sh`. CI runs it one stage per job and nothing
else — apart from the `skills` job, which reads the PR body and therefore has no local form, because
the body exists only in the event — so a stage that is green on a laptop and a stage that is green on
CI are the same command; the
stages are `fmt`, `lint`, `build`, `test`, `features`, `measure` and `gpu`, and the `ci` job fails
unless every one of them passed. The commands behind the stages, which are the bar itself:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --features acme -- -D warnings
cargo build --workspace --release --locked
cargo test --workspace --locked
cargo test -p vipera-server --features acme --locked
# The two measurement tests that assert (EXC-004). Release because a debug build cannot time them.
cargo test -p vipera-backend-cpu --release --locked --test prefix_cache --test alignment_probe -- --ignored
cargo test -p vipera-backend-gpu --locked
```

The commands above are the canonical definition of "green". If `CONSTRAINTS.md` and a wrapper script disagree, this file wins.

### Before a tagged release

One more check, outside the gate because it cannot run on CI: it needs a model and the machine the
baseline was taken on.

```bash
scripts/q3k-band-drift.sh    # one 128-token chunk, other_dot against scripts/q3k-band-baseline.txt
```

The Q3_K band is the largest single phase of a prompt chunk (45% of a late one, 69% of the first), and
its absolute cost per chunk is flat with depth, so one chunk sees a regression. The script fails above
8% — the measured noise floor: back-to-back runs agree to 0.04%, a run taken under load reads +5.6% —
and skips rather than fails on another host or model, because absolute milliseconds do not travel. If a
release is *meant* to move that number, update the baseline in the same commit and say why in the
message; `docs/benchmarks.md`, "Q3_K at real depths", records what the phases should look like.
