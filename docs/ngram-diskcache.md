# N-gram disk cache: exllamav3 mechanism and vipera integration notes

Research note only — no code was changed. Sources: local checkout `exllamav3` v1.6.7 at
`/home/user/llm-engines/exllamav3` (identical `ngram.py` files in `/home/user/llm-forks/exllamav3`),
and the vipera worktree at `/home/user/.local/share/opencode/worktree/953041/ngram-diskcache`.

## TL;DR

Exllamav3's n-gram disk cache is a **frozen suffix automaton (SAM) bank**: an offline-built,
zstd-compressed, read-only graph of a token corpus, memory-mapped-by-deserialization at generator
startup and shared by per-job cursors. Drafting matches the current token suffix against both the
job's *live* SAM (its own context) and the frozen corpus, takes the longest match, and verifies the
continuation with the target model exactly like any other speculative draft. The artifact is bound
to tokenizer identities, not to model weights or library versions.

Vipera already has the verification half (greedy in-context n-gram drafts verified in one
`logits_range` pass). The missing piece is the disk-backed corpus source plus a stateful matching
cursor. That is a contained addition to `vipera-runtime`, but two project rules bite: a zstd
dependency needs an explicit exception (`CONSTRAINTS.md` rule 2), and the workspace forbids `unsafe`
while the only existing compression/mmap exception is `memmap2` in the CPU backend.

## 1. Disambiguation: three different things called "ngram" / "disk cache"

| Source | What it actually is | Relevant here? |
| --- | --- | --- |
| `exllamav3/generator/ngram.py` (+ `util/build_sam.py`, `exllamav3_ext/sam.cpp`) | Frozen SAM **draft corpus** on disk, used by n-gram speculation | **Yes — the subject of this note** |
| `exllamav3/conversion/ngram.py` | Trellis quantization of the *model's hashed n-gram embedding table*; `NgramTableReader` (`conversion/ngram.py:547-634`) does on-demand `pread` + GPU decode of embedding rows. This is a deferred-load weight table, not a speculation corpus | No (different problem) |
| `exllamav3/generator/disk_cache.py` (+ `tests/tools/disk_cache_restart_e2e.py`) | A **KV page cache tier** on disk: page images + recurrent checkpoints, sqlite index, fingerprint of model/config/format version (`disk_cache.py:31-45`) | No, unless "disk cache" meant KV restart caching; that is a different and much larger integration (`vipera-runtime/src/kv.rs`, `memory.rs`) |

If the request intended the third row, this document is the wrong map; say so and this note should be
rewritten. Everything below assumes the first row: a disk-backed corpus that feeds the existing
speculative stage.

## 2. Mechanism (exllamav3)

### 2.1 Offline bank build — `util/build_sam.py`, `util/build_sam.md`

1. A YAML recipe (`version: 1`, `sources`, `target_bytes`, `max_tokens`, `seed`, chat-template kwargs)
   selects HF `datasets` streams. Default mixture: 40% SWE-smith trajectories, 30% Magicoder
   OSS-Instruct, 20% OpenCoder package_instruct, 10% SmolTalk, ~100 MB rendered UTF-8. Whole
   documents are normalized, deduplicated, optionally chat-templated, then tokenized. `-1` is
   inserted between documents (`build_sam.md:43-90`, `160-161`).
2. One live `BC_SAM` is built over the joined token stream (`sam.cpp:13-100`, `237-279`). The
   automaton stores suffix links, `max_len`, a representative occurrence end (`min_end`), and
   CSR-style edges. `export_csr()` (`sam.cpp:18-57`) sorts edges by signed token id per state and
   emits seven `int32` arrays; token IDs and document ends are added as two more arrays.
3. `write_bank()` (`build_sam.py:247-284`) pads each section to a 64-byte offset, writes a
   `struct.Struct('<8sIIQ')` prefix (`EXL3SAM\0`, version, JSON length, decompressed payload
   length), the JSON metadata, and exactly one zstd frame with pledged size and checksum. The
   int32 words are stored in **byte-plane order** (all byte 0s, then all byte 1s, ...). Publication
   is atomic via a temp file + `os.replace`; a failed build leaves an earlier bank intact.

Reference build from `build_sam.md:92-96`: 100.1 MB text, Gemma 4 tokenizer → 29,083,418 tokens,
314 MB compressed bank, ~80 s total on a Threadripper 7960X, **4.3 GiB peak process memory**. That
RAM figure is a builder concern, not a runtime one.

### 2.2 Frozen format v1 — `build_sam.md:124-166`, `generator/ngram.py:14-63`

- File = 24-byte prefix + JSON metadata + one zstd frame. All section offsets address the
  *decompressed* payload.
- Metadata carries per-section descriptors and sizes, `corpus_sha256`, tokenizer/config/template
  file hashes and effective template, dependency versions, the recipe, resolved revisions and
  sampling statistics. It asserts `format=exllamav3.frozen_sam`, `version=1`, `layout=csr`,
  `compression=zstd`, `separator=-1`, and `payload_bytes == size` (`ngram.py:30-34`).
- Nine int32 sections in byte-plane order, 64-byte aligned: `link` (states), `max_len` (states),
  `min_end` (states), `edge_offsets` (states+1), `edge_token` (edges), `edge_to` (edges),
  `root` (dense root transitions, `-1` = missing), `corpus` (token stream with `-1` separators),
  `document_ends` (separator positions).
- No pointers, no hash maps. Loader validation (`ngram.py:78-117`) checks array counts, link/root
  ranges, monotonic `max_len`, sorted edge labels, CSR bounds, graph consistency at state
  boundaries, the corpus hash, and that separators are exactly the document ends.

### 2.3 Load and tokenizer binding — `generator/ngram.py:18-120`

`NgramCorpus.__init__` decompresses the frame into one `bytearray` (16 MiB chunks via
`ZstdDecompressor.readinto`, no unshuffle pass), then reinterprets each section slice as a `uint8`
torch tensor. `check_tokenizer()` (`ngram.py:65-76`) hashes `tokenizer.json`,
`tokenizer_config.json`, `added_tokens.json` and compares to the metadata — model config,
quantization and library version may differ, **token IDs may not**. The immutable sections are
shared; each job calls `corpus.cursor()` to get an independent `FrozenSAMCursor` that references the
same backing storage (tests confirm cursors outlive the bank object:
`tests/generator/ngram/test_ngram_corpus.py:88-90`). Loading is outside generation timing.

### 2.4 Per-job matching cursor — `exllamav3_ext/sam.cpp:281-349`

`FrozenSAMCursor::draft(history, minimum, length)` keeps a monotonic `(pos, state, matched)` per
job. It consumes only newly appended history tokens, so the walk is incremental:

- `word(section, i)` reassembles a signed int32 from four byte planes.
- `next(state, token)`: root IDs below `root_dense_limit` (1,048,576) use the dense `root` table;
  otherwise CSR edges are scanned linearly when degree < 16 and binary-searched when >= 16.
- On a miss, follow suffix links, capping `matched` at the linked state's `max_len`, until a
  transition exists or the root is reached. Negative/out-of-int32 tokens reset the match.
- If `matched >= minimum`, the draft is the corpus slice starting at `min_end[state] + 1`,
  extended at most `length` tokens but **clamped at the next `-1` separator** — never a separator,
  never across documents. Returns `(matched, tensor[1,count])`.
- If history shrinks (rewind), `pos > size` resets the cursor (`sam.cpp:324`).

### 2.5 Live vs corpus selection and verification

Each job also owns a live `BC_SAM` of its own context (`job.py:1365-1367`) and answers
`suffix_match(seq)` against it (`job.py:1984`). `Job.get_ngram_draft()` (`job.py:2018-2039`) takes
the live continuation when its match is long enough, then replaces it with the corpus continuation
only if the corpus has one **and** either the live draft is empty or the corpus match is strictly
longer — live history wins ties. The generator batches n-gram jobs via `iterate_ngram_gen`
(`generator.py:1485-1512`), trims all drafts to the shortest in the batch, and verifies them in the
normal speculative pass. There is no separate "corpus trust" path: a wrong draft is simply rejected
by the target model, so the output stream is unchanged.

### 2.6 Cost profile

- Startup: one decompression + validation, shared across jobs; no graph rebuild at runtime.
- Per token: one incremental SAM step (a handful of byte-plane reads, suffix-link walks on miss).
- Per verify round: same target-model pass as any speculative draft; acceptance determines payoff.
- Builder: 100 MB of text → ~314 MB bank, several GiB peak RAM; not a runtime cost.

## 3. Integration points in vipera

### 3.1 What vipera already has

`crates/vipera-runtime/src/engine.rs`:

- `EngineConfig::speculative` + `speculative_enabled()` with `VIPERA_SPECULATIVE` override
  (lines 46-49, 187-201).
- In-context draft generators `ngram_draft`, `longest_ngram_draft`, `longest_ngram_draft_from`
  (lines 219-281; brute-force longest needle of 2-4 tokens, capped at `max_draft`) and greedy
  `speculative_accept` (287-293).
- `Turn::draft_cap` / `spec_misses` adaptive window and `SPEC_MISS_LIMIT` (158-185), gating for
  greedy, non-logprobs, unbatched turns (694-739), and the batched dispatch (776-826).
- `step_speculative()` (987-1155): builds `all = context ++ draft`, calls `forward.logits_range`,
  accepts the matching greedy prefix, emits one extra target token, rewinds KV to the accepted
  prefix, and times the draft under `Stage::Speculative`.
- `Stage::Speculative` exists in `crates/vipera-runtime/src/pipeline.rs` (22, 39, 51, 64, 285).

`vipera-traits/src/lib.rs:149-163` defines a `SpeculativeDecoder` trait (`draft(&self, context,
count, out)`), but the engine currently hardcodes its own n-gram draft and never consults the
registry — the trait is a candidate seam, not an active one.

### 3.2 Smallest defensible change (read-only corpus, no new builder in this repo)

1. New module in `vipera-runtime` (e.g. `src/ngram_bank.rs`): parse the header, validate metadata,
   hold the decompressed payload, and implement a byte-plane cursor with the same walk as
   `FrozenSAMCursor::draft`. `u32::MAX`/document-end list replaces `-1` since vipera tokens are
   `u32`. Safe Rust can do all of this (indexed slices, no `unsafe`).
2. Add one path knob to `EngineConfig` (e.g. an `Option<PathBuf>`) following the
   `VIPERA_SPECULATIVE` env-override pattern. Wire `bins/vipera/src/main.rs:785-804` and
   `crates/vipera-server/src/serve.rs:618-639`. **A new flag/key needs an explicit reason in the
   request** (`CONSTRAINTS.md` rule 2); keep it to one knob, not a family of them.
3. Replace the draft source in `step_speculative` (engine.rs:1043) with a "best of" selection:
   existing `longest_ngram_draft_from(context, cap, min_needle)` versus the corpus cursor's
   continuation, live wins ties, both truncated to `room`. The cursor must be per sequence and be
   reset on `restore()` (engine.rs:1280) and on any context truncation, mirroring `sam.cpp:324`.
4. Reuse the existing verifier unchanged: acceptance must remain bit-identical greedy output. The
   `draft_cap`/`spec_misses` logic already handles a corpus that stops landing.
5. Tests: byte-plane decode and graph validation (including corrupt/truncated banks), cursor
   equivalence to brute-force corpus search, separator clamping, cursor independence, live-tie
   rule, restore resets cursor, and a `step_speculative` parity test asserting the token stream is
   unchanged with a bank loaded. Add acceptance-rate measurement as an ignored benchmark per the
   existing performance convention (`CONSTRAINTS.md` EXC-004).

### 3.3 Points that need a decision

- **Tokenizer identity.** Exllamav3 pins HF `tokenizer*.json` SHA-256s; vipera reads GGUF metadata
  (`vipera-tokenizer/src/lib.rs:45-80`) and has no such files. Options: (a) record a hash of the
  GGUF vocabulary/metadata in the bank and validate at load (new format field, breaks exllamav3
  bank compatibility); (b) document that the bank must be built with the same GGUF and rely on
  verification to keep output correct; (c) add an external manifest. Because verification makes a
  mismatch safe-but-useless, (b) is the smallest defensible reading, but (a) is the robust one.
- **Format compatibility.** Reusing exllamav3's exact bank format means vipera must decompress
  zstd (no crate in the workspace today) and mirror a Python-specific tokenizer identity scheme.
  A vipera-native, uncompressed or `memmap2`-backed bank avoids both but cannot read exllamav3
  artifacts and needs its own builder.
- **Builder.** `util/build_sam.py` is ~400 LOC of Python plus HF `datasets`, `transformers`,
  Jinja chat templates and `zstandard`. Porting it to Rust is the largest single work item; the
  alternative is to keep bank construction an external offline step and only consume the artifact.
- **Backend reach.** Speculation already requires `ForwardPass::logits_range`; the GPU backend does
  not implement it (`README.md:222`), so a corpus bank inherits the CPU-only restriction. No
  backend change is implied.
- **Batching.** The engine only speculates when a turn is the sole decoding turn; corpus drafts do
  not change that. Per-job cursor state is still needed for multi-sequence engines.

## 4. Effort

Estimates assume one engineer familiar with the runtime and exclude review/benchmark cycles.

| Work item | Approx. size | Time |
| --- | ---: | ---: |
| Bank reader: header/metadata validation, byte-plane decode, integrity checks | 350-600 LOC | 2-4 d |
| Matching cursor: incremental SAM walk, dense root + CSR search, separator clamp, rewind reset | 150-250 LOC | 1-2 d |
| Engine integration: candidate merge, per-sequence cursor state, config knob + env override | 100-200 LOC | 1-2 d |
| CLI + server wiring, README/docs | small | 0.5-1 d |
| Tests (unit + engine parity + malformed bank cases) | 250-400 LOC | 2-3 d |
| Acceptance/benchmark harness and A/B measurement | small | 1-2 d |
| **Subtotal: reader + integration (external builder)** | | **~1-1.5 weeks** |
| Offline builder port: token stream assembly, SAM construction, CSR export, atomic writer, recipe handling | 600-1000 LOC | 3-6 d + 2-3 d if chat templates/recipes are needed |
| **Total: full path including an in-repo builder** | | **~2-3 weeks** |

Dependencies: zstd support requires an explicit exception (`CONSTRAINTS.md` rule 2 and the EXC
register); avoiding it shortens the schedule but drops exllamav3 artifact compatibility. A Python
builder keeps the port small but adds a non-Rust tool to a workspace whose gates are
`cargo test/clippy/fmt`.

## 5. Risk

| Risk | Severity | Mitigation / note |
| --- | --- | --- |
| Low acceptance wastes verify positions; corpus may not match workload | High (perf) | Reuse adaptive `draft_cap` + 5-miss cutoff; measure before claiming a win. Exllamav3 reports ~1.6-1.7x only on repetitive prompts/agentic traces, not prose. |
| Tokenizer ID mismatch (GGUF vs HF bank) silently kills acceptance | Medium | Output stays correct because drafts are verified; still need an identity check or documented provenance. |
| `-1` separator encoding in a `u32` token world | Medium | Use an explicit boundary table or a validated sentinel; never emit a separator, clamp at document ends. |
| Cursor drift after rewind/snapshot restore | Medium | Reset cursor when history shrinks (`sam.cpp:324`); vipera `restore()` at engine.rs:1280 is the hook. |
| Bank memory footprint (314 MB compressed, ~1.2 GB decompressed for the reference) | Medium | Map on demand / keep byte planes; avoid a full heap decompress; `unsafe_code = "forbid"` and EXC-005's `memmap2` precedent constrain the approach. |
| New dependency (zstd, HF dataset tooling) | Medium | Needs an explicit exception in `CONSTRAINTS.md`; feature-gate or keep it in an offline tool. |
| New config knob violates "no unrequested features" | Medium | One path knob with an env override, only if the request asks for it; otherwise no flag and no wiring. |
| Verification parity regressions (logprobs, EOS, KV rewind) | Low | The existing `step_speculative` path is unchanged; parity tests assert identical greedy streams. |
| Wrong corpus content / license provenance | Low | Banks are data artifacts; record source hashes and recipe in metadata, as exllamav3 does. |
| Confusing this feature with exllamav3's KV disk cache | Low | Disambiguation table above; confirm intent before starting. |

## 6. Recommended next step

Confirm which "disk cache" is meant. If it is the n-gram corpus bank, pick one of:

- **Consume exllamav3 banks as-is** (requires zstd exception + an identity policy), or
- **Vipera-native uncompressed/mmap bank** (no new runtime dependency, no artifact compatibility).

Then implement the reader and cursor first, wire the existing `step_speculative` draft source, and
measure acceptance on the repetitive/agentic prompts that justify the feature. Do not port the
offline builder until the runtime path has a measured win.

### Source map

- exllamav3: `exllamav3/generator/ngram.py`, `exllamav3/generator/job.py:1365,1984-2039`,
  `exllamav3/generator/generator.py:289-308,848-849,1485-1512`, `exllamav3/model_init.py:135`,
  `util/build_sam.py:23-24,247-284,369`, `util/build_sam.md:43-96,124-166`,
  `exllamav3/exllamav3_ext/sam.cpp:13-57,281-349`, `exllamav3_ext/sam_bc.h:1-10`,
  `tests/generator/ngram/test_ngram_corpus.py`, `exllamav3/conversion/ngram.py:547-634`,
  `exllamav3/generator/disk_cache.py:31-45`.
- vipera: `crates/vipera-runtime/src/engine.rs:46-49,158-201,219-293,694-826,987-1155,1280`,
  `crates/vipera-runtime/src/pipeline.rs`, `crates/vipera-traits/src/lib.rs:149-163`,
  `crates/vipera-tokenizer/src/lib.rs:45-80`, `bins/vipera/src/main.rs:704,746,785-804`,
  `crates/vipera-server/src/serve.rs:618-639`, `Cargo.toml:24-49`, `CONSTRAINTS.md:24-39,118-123`,
  `README.md:123-127,222`.
