# Paged KV block tables for vipera.rs

Status: design note, no code changes. Describes the scheme in
`crates/vipera-runtime/src/kv.rs` as inspected on 2026-11-28, how the engine and the
backends use it, and the constraints a change must keep. `specs/` and `docs/design/`
are historical context, not a gate.

## Why a block table

The runtime owns the KV cache as one byte buffer. A sequence's KV state is not a
contiguous slice of it: blocks are handed out lowest-id first, parked sequences keep
theirs, and a ring cache deliberately aliases blocks. A position `p` therefore cannot
be turned into a byte offset by arithmetic on `p` alone; it needs one lookup:

```
block  = table[p / tokens_per_block]
offset = (block * tokens_per_block + p % tokens_per_block) * bytes_per_token
```

This is the only indirection on the attention hot path. It is documented on
`ForwardContext` / `BatchEntry` (`crates/vipera-runtime/src/pipeline.rs`) and
implemented by the CPU backend's `KvView` and the GPU backend's `KvRows::offsets`
(`crates/vipera-backend-gpu/src/paged.rs`). The cache owns the buffer and counts the
bytes; what is stored inside a slot is the backend's business.

## Block table design

### Block size

**Decision: 16 tokens per block (`DEFAULT_TOKENS_PER_BLOCK`), fixed for the lifetime
of a cache.** `KvConfig::new` validates the geometry and `blocks_for` rounds token
counts up. No code path renegotiates the block size.

Rationale, in priority order:

1. **Waste bound.** A partially filled tail block is the maximum internal
   fragmentation per sequence. At `--ctx 2048`, f16 on a 1.5B model, one 16-token
   block is ~0.46 MB, so eight concurrent sequences waste at most ~3.7 MB.
2. **Prefix reuse granularity.** `Engine::adopt_prefix` copies whole blocks and
   refuses a shared prefix shorter than one block. 16 keeps short shared prefixes
   useful while making a copy a handful of `copy_within` calls.
3. **Table size.** A 32k context needs 2048 entries = 8 KiB of table per sequence.
   A larger block halves the table but coarsens reuse and doubles the wasted tail;
   a smaller one doubles allocation churn.
4. **Ring alignment.** The window math uses `PREFILL_HEADROOM = 128` and one extra
   block for a straddling window, both in token/block units; 16 divides 128 and
   2048 evenly.

`bytes_per_token` is owned by the model geometry (K and V, all layers, f16 or q8_0);
the cache only multiplies it by `tokens_per_block`. The GPU backend additionally
checks that it equals `2 * layers * kv_heads * head_dim * 2` before accepting a table.

### Shape

```text
tables: BTreeMap<u64 /* sequence id */, Vec<u32> /* physical block ids */>
```

- One entry per logical block the sequence can address:
  `ceil(tokens / tokens_per_block)`.
- `u32` block id; `KvConfig::new` rejects more than `u32::MAX` blocks.
- Keyed by sequence id, not a dense index, so ids need not be contiguous. Nothing
  iterates `tables` today; LRU order lives in the separate `parked: VecDeque`.
  Deterministic iteration is not currently a requirement.

### Paged tables

Each entry names a distinct physical block. `reserve` appends free blocks until the
table covers `tokens`; `truncate` pops the tail and returns those blocks. Tables
never compact: block ids stay put across turns, which is what makes a parked copy
byte-for-byte meaningful to a later request.

### Ring tables

A ring still has `Ring::slots` entries — one per block of the *context* — but
entries past the sinks alias a small rolling set:

```
slots 0..pinned_slots                        -> sink blocks (distinct, never move)
slots pinned_slots..pinned_slots+rolling     -> rolling blocks (distinct)
slots pinned_slots+rolling..                 -> pinned + (slot - pinned) % rolling
```

So position 33000 and position 1000 can name the same physical block. This is the
one deliberate break of "one entry, one block", and every consumer that cares must
check for it: `KvCache::release` sorts and dedups before freeing, `KvRows::repeats`
refuses a repeated table on the GPU, and `park` / `copy_from_parked` are disabled for
rings. A ring's physical footprint is `(pinned_slots + rolling_slots) * sequences`
blocks, not `slots * sequences`. `for_run` only builds a ring when
`context > window + sinks + PREFILL_HEADROOM`.

### Per-forward-pass copy

A table cannot stay borrowed while the cache's storage is borrowed mutably, so
`Engine::forward` (and the speculative verify path) copies `kv.table(id)` into a
`Vec<u32>` per entry before the batched pass. That is 512 bytes at `--ctx 2048` and
is deliberately not optimized away: it buys one `&mut [u8]` for the whole batch
instead of `B` aliasing borrows, which the type system forbids without `unsafe`.

## Allocation

### Physical blocks

```text
storage: Vec<u8>   // total_bytes, allocated and zeroed up front
free:    Vec<u32>  // physical block ids, descending, so pop() yields the lowest id
```

- Allocation is eager: a capacity refusal is actionable instead of a first-use OOM.
- No bitmap and no refcount. Ownership is implicit in which table (or parked entry)
  names an id; the free list is the complement after deduplication.
- `used_blocks = config.blocks - free.len()`, so `used_bytes` and the runtime stats
  report physical bytes, not logical table length.

### Reserve

`reserve(sequence, tokens)` (via `reserve_keeping`):

1. Compute `needed = blocks_for(tokens)`.
2. Refuse if `needed > config.blocks` (paged) or `needed > ring.slots` (ring).
3. Grow the table by appending free blocks until it covers `needed`.
4. If the free list empties, evict parked sequences oldest-first, except a donor
   being copied into.
5. Ring variant: allocate `pinned + rolling` physical blocks once, then fill the
   remaining table entries with aliases.

The capacity pre-check runs before a table entry is created, so a refused reserve
leaves nothing behind (`a_failed_reserve_leaves_no_partial_allocation`); the
`free.pop()` failure path inside the loop does not clean up, and relies on that
pre-check. `copy_from_parked` removes the entry if its reserve fails.

### Release, truncate, park, copy

- **release**: remove the table or parked entry, sort + dedup (ring safety), return
  every physical block to `free`. Idempotent; a second release returns 0.
- **truncate**: pop tail entries, return them to `free`, remove an empty sequence.
  Floor of `pinned + rolling` for a ring. See the ring caveat under Integration.
- **park**: move a finished sequence's table into the `parked` deque (LRU, cap
  `PARKED_LIMIT = 32`). Parked blocks count as in use but are reclaimable by any
  reserve. Disabled for rings.
- **copy_from_parked**: allocate fresh blocks for the new sequence, `copy_within`
  whole blocks from the donor, keep the donor parked and make it MRU. Refuses rings
  and sequences that already hold blocks.
- **trim**: exposes the retained index set from the `KeepKv` policy (or the newest
  `capacity_tokens` when there is no policy). It deliberately does not free blocks:
  the attention kernels read a block without asking whether a position survived.
  Applying a mask is the caller's job.

### Prefix reuse (engine layer)

`Engine::adopt_prefix` compares a new prompt against retired parked donors and
copies the longest shared prefix, at least one block and at least one token short of
the prompt (so the turn samples from logits it computed itself). The copy is by
whole blocks, so bytes past the prefix in the last block come along; they are
positions not yet computed and are overwritten before they are read.
`park_for_reuse` offers only prompt positions, not decode positions.

## Integration

| Concern | Paged | Ring |
|---|---|---|
| Physical blocks per sequence | up to `ceil(tokens/tpb)` | `pinned + rolling` |
| Table entries | one per logical block, distinct | `slots`, may alias |
| `park` / `copy_from_parked` | yes | no (aliasing makes reuse unsound) |
| `release` dedup | harmless | required |
| GPU backend | usable | refused by `KvRows::repeats` |
| Prefix cache | yes | no (nothing can be parked) |

The ring is not a second allocator: it is the same `KvCache`, the same table, the
same free list, with aliasing as policy. `ModelShape::window_kv_bytes` in
`vipera-core` mirrors the ring's sizing (window + sinks + headroom + two blocks of
slack) as an upper bound for the planner; a change to ring geometry must keep that
mirror true or plans under-claim memory.

### What a change must not break

1. **The table is the address.** CPU attention, GPU `KvRows::offsets`, and keepkv
   all rely on the resolution arithmetic above. A new indirection layer must keep
   it equivalent or every backend silently reads the wrong bytes.
2. **Aliasing stays detectable.** A consumer must be able to ask "does this table
   repeat a block?" without walking bytes. The current sort+dedup check is O(n log
   n) at setup, not per token.
3. **One free list per cache.** A cache is either paged or ring for its whole
   lifetime (`KvConfig::for_run` decides). Mixing modes would need a tagged block id
   or two free lists and a separate design.
4. **Block size is fixed per cache.** Ring headroom and prefill chunking depend on
   it. Variable sizes need per-entry sizes — a different data structure.
5. **All mutation goes through `reserve` / `release` / `truncate` / `park`.** Direct
   writes to `tables` drift the accounting that runtime stats report.

### Verified integration gaps (observed, not fixed here)

Both were reproduced with standalone probes against this worktree; no repository
code was changed.

1. **Admission under-admits a ring.** `Engine::admit` compares
   `blocks_for(prompt.len())` with `config.blocks`, i.e. the *physical* total, even
   though a ring table can address `ring.slots` blocks. Probe: `KvConfig::for_run`
   with `ctx=4096, window=512, sinks=4, sequences=4` gives `slots=256` but
   `blocks=172`; a 4096-token prompt is refused at admission although
   `reserve_ring` accepts it. Fix direction: compare the table length against
   `config.slot_count()` and check physical availability separately against
   `free_blocks() + reclaimable parked >= pinned + rolling`.
2. **`truncate` double-frees ring aliases.** `KvCache::truncate` returns every
   popped entry to `free`, but for a ring the tail entries alias rolling blocks
   that are still in the table. Probe: `KvConfig::ring(64, 4, 256, 32, 4, 1)` has
   43 physical blocks and 64 table slots; `reserve(1, 256)` then
   `truncate(1, 8)` pushes 21 aliases into `free`, and `release(1)` adds 43 more —
   `free` ends at 64 for 43 physical blocks. Reachable from
   `Engine::append_prompt` and the rewind/snapshot path once a ring sequence is
   longer than its window. Fix direction: shrink a ring's table without returning
   alias entries to `free` (or dedup before freeing), with a regression test.

## Effort

Single engineer, assuming the 25 existing `kv.rs` tests stay green (verified: 25
passed, 0 failed):

| Work | Effort |
|---|---|
| Document the current scheme (this note) | done |
| Fix ring admission check + test | 0.5–1 day |
| Fix ring `truncate` alias double-free + test | 1–2 days |
| Block refcount / copy-on-write to share prefixes without copying bytes | 2–3 weeks |
| Unify ring and paged under one tagged allocator | ~1 week |
| Per-backend block-table validation beyond the GPU check | 2–3 days each |
| Regression suite for a new allocation policy | ~1 week |

The cheap, high-value items are the two ring fixes; refcount and COW only pay off
if prefix copying becomes a bottleneck, which the benchmarks have not shown.

## Risks

- **Double free on rings.** `release` dedups; `truncate` currently does not (gap 2
  above). Any new release path must go through `KvCache::release`, and any path
  that shrinks a ring table must not free aliases. Keep
  `a_ring_frees_each_physical_block_once` green.
- **Stale prefix bytes.** `copy_from_parked` copies whole blocks, including
  positions beyond the prefix. Keep the "write before read within a block"
  invariant; do not add read-before-write optimizations without revisiting it.
- **GPU ring mismatch.** The device backend assumes distinct blocks. A ring table
  reaching it would read overwritten rows, not crash. Keep `KvRows::repeats` as a
  hard refusal at setup and rings CPU-only until a device kernel handles aliasing.
- **Allocation accounting drift.** `used_blocks`, `parked_blocks`, and
  `free_blocks` are derived from the free list and parked deque. A path that
  mutates tables without updating them silently misreports memory.
- **Admission mismatch.** Gap 1 above refuses valid long prompts on a ring and can
  hide a real capacity shortage on a paged cache, since `admit` checks one
  sequence's blocks against the whole cache's total.
- **Golden token stability.** Block layout is not supposed to affect logits, but
  ring aliasing and keepkv trim change which positions survive. Any allocation
  change must rerun the 24-token greedy continuation tests in
  `bins/vipera/tests/` and the batching tests in `engine.rs`
  (`two_sequences_in_one_step_produce_the_tokens_they_produce_alone`,
  `a_request_that_arrives_mid_turn_joins_the_batch`,
  `a_long_prompt_does_not_starve_a_decode_turn`,
  `arrival_order_does_not_change_a_sequence_tokens`).
- **Planner / runtime drift.** `window_kv_bytes` hardcodes the block size and adds
  two blocks of slack; changing ring geometry in `kv.rs` without updating it makes
  `vipera verify` predict less memory than the run consumes.
- **Block size lock-in.** A future "dynamic block size" feature is a new data
  structure, not a config tweak; scope it as such.
