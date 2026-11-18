//! Paged KV cache: fixed-size blocks, one table per sequence.
//!
//! The cache owns a real zeroed buffer, so a runtime built on it consumes the memory it claims and
//! `vipera verify` measures the same bytes the plan predicted. Blocks are handed out lowest-id first
//! and returned on release; there is no compaction, which keeps block tables stable across turns.
//!
//! A finished sequence's blocks can be *parked* instead of released: kept, in least-recently-used
//! order, so a later request that starts with the same tokens can copy them instead of computing
//! them again. Parked blocks are free for every purpose except that copy: [`KvCache::reserve`]
//! takes them back, oldest first, whenever the free list runs out, so a parked sequence never
//! makes an allocation fail.
//!
//! A cache built with [`KvConfig::ring`] is the exception to "a table's blocks are distinct": there
//! the table still has one entry per block of the context, but the entries past the sinks repeat a
//! small ring, so position 33000's KV overwrites position 1000's. Reads of the dropped positions
//! return whatever the ring holds, which is why this only makes sense under a sliding window — the
//! only reader, [`vipera_backend_cpu`]'s attention, then never asks for a position the window has
//! dropped. Because table entries alias, a ring cannot park or copy blocks, and its blocks are the
//! window and the sinks rather than the context: at 32k with a 2048-token window an f16 1.5B model
//! keeps 64 MB instead of 939 MB.

use std::collections::{BTreeMap, VecDeque};

use vipera_errors::ViperaError;

/// Tokens per block. 16 is the paged-attention default and keeps waste under one block per sequence.
pub const DEFAULT_TOKENS_PER_BLOCK: u64 = 16;

/// Most sequences kept parked at once. Beyond this the oldest is freed, so the number of
/// candidates a new request is compared against stays small.
pub const PARKED_LIMIT: usize = 32;

/// Tokens of a prompt chunk a ring has to keep behind the window: the chunk's positions are all
/// written before the first of them reads its window, so the window has to reach back one chunk.
pub const PREFILL_HEADROOM: u64 = 128;

/// How a ring cache maps a context onto a few blocks.
///
/// Slots `0..pinned_slots` hold the sink positions and never move. Slots from `pinned_slots` on
/// cycle through `rolling_slots` blocks, so a slot is `pinned_slots + (slot - pinned_slots) %
/// rolling_slots`. The table itself still has [`Ring::slots`] entries, one per block of the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring {
    /// Table entries the ring can address: `ceil(context / tokens_per_block)`.
    pub slots: u64,
    /// Blocks the sink positions own, counted from the start.
    pub pinned_slots: u64,
    /// Blocks the rest of the context cycles through.
    pub rolling_slots: u64,
}

/// Geometry of a KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvConfig {
    /// Bytes one token occupies across every layer of the cache (keys plus values).
    pub bytes_per_token: u64,
    /// Tokens a block holds.
    pub tokens_per_block: u64,
    /// Blocks the cache owns, across every sequence it was sized for.
    pub blocks: u64,
    /// The ring a windowed cache maps its context onto, when it is one.
    pub ring: Option<Ring>,
}

impl KvConfig {
    /// Check a geometry and reject one that cannot describe a cache.
    pub fn new(
        bytes_per_token: u64,
        tokens_per_block: u64,
        blocks: u64,
    ) -> Result<Self, ViperaError> {
        if bytes_per_token == 0 {
            return Err(ViperaError::invalid_argument(
                "KV cache needs at least one byte per token",
            ));
        }
        if tokens_per_block == 0 {
            return Err(ViperaError::invalid_argument(
                "a KV block must hold at least one token",
            ));
        }
        if blocks == 0 {
            return Err(ViperaError::invalid_argument(
                "a KV cache needs at least one block",
            ));
        }
        if blocks > u64::from(u32::MAX) {
            return Err(ViperaError::invalid_argument(
                "KV cache has more blocks than the table can address",
            ));
        }
        if bytes_per_token.checked_mul(tokens_per_block).is_none() {
            return Err(ViperaError::invalid_argument(
                "KV cache geometry overflows a 64-bit byte count",
            ));
        }
        let config = Self {
            bytes_per_token,
            tokens_per_block,
            blocks,
            ring: None,
        };
        if config.total_bytes().is_none() {
            return Err(ViperaError::invalid_argument(
                "KV cache geometry overflows a 64-bit byte count",
            ));
        }
        Ok(config)
    }

    /// Cover `budget_bytes` with whole blocks, keeping at least one.
    pub fn from_bytes(
        budget_bytes: u64,
        bytes_per_token: u64,
        tokens_per_block: u64,
    ) -> Result<Self, ViperaError> {
        let block_bytes = bytes_per_token
            .checked_mul(tokens_per_block)
            .filter(|value| *value > 0)
            .ok_or_else(|| ViperaError::invalid_argument("KV block size is not representable"))?;
        let blocks = (budget_bytes / block_bytes).max(1);
        Self::new(bytes_per_token, tokens_per_block, blocks)
    }

    /// Geometry for a cache that only has to *hold* a window, but must still be *addressed* by
    /// position.
    ///
    /// The cache owns the sinks plus a rolling window, and `context_tokens` is what the table can
    /// address. `sequences` caches are laid side by side, each with its own ring. `tokens_per_block`
    /// rounds every part; the rolling part gets one extra block so a window that straddles a block
    /// boundary never wraps onto itself, plus [`PREFILL_HEADROOM`] tokens because a prompt chunk
    /// writes its whole chunk before the chunk's first position reads its window.
    pub fn ring(
        bytes_per_token: u64,
        tokens_per_block: u64,
        context_tokens: u64,
        window_tokens: u64,
        sink_tokens: u64,
        sequences: u64,
    ) -> Result<Self, ViperaError> {
        if window_tokens == 0 {
            return Err(ViperaError::invalid_argument(
                "a ring KV cache needs a window to keep",
            ));
        }
        if sequences == 0 {
            return Err(ViperaError::invalid_argument(
                "a ring KV cache needs at least one sequence",
            ));
        }
        let per_block = tokens_per_block.max(1);
        let pinned_slots = sink_tokens.div_ceil(per_block);
        let rolling_slots = (window_tokens + PREFILL_HEADROOM + per_block).div_ceil(per_block) + 1;
        let slots = context_tokens
            .div_ceil(per_block)
            .max(pinned_slots + rolling_slots);
        let blocks = (pinned_slots + rolling_slots)
            .checked_mul(sequences)
            .ok_or_else(|| {
                ViperaError::invalid_argument(
                    "ring KV cache geometry overflows a 64-bit block count",
                )
            })?;
        let mut config = Self::new(bytes_per_token, tokens_per_block, blocks)?;
        config.ring = Some(Ring {
            slots,
            pinned_slots,
            rolling_slots,
        });
        Ok(config)
    }

    /// The cache a run should allocate for `context_tokens` positions.
    ///
    /// A sliding window makes the positions the model can still reach far shorter than the context,
    /// so the cache becomes a [`KvConfig::ring`] of the sinks and the window. When the context is no
    /// longer than that, a ring would hold everything anyway, so the plain paged cache is used — it
    /// is what a run without a window gets.
    pub fn for_run(
        bytes_per_token: u64,
        tokens_per_block: u64,
        context_tokens: u64,
        window: Option<(u64, u64)>,
        sequences: u64,
    ) -> Result<Self, ViperaError> {
        let budget = context_tokens
            .checked_mul(bytes_per_token)
            .ok_or_else(|| ViperaError::invalid_argument("context is too large for this model"))?;
        match window {
            Some((tokens, sinks)) if context_tokens > tokens + sinks + PREFILL_HEADROOM => {
                Self::ring(
                    bytes_per_token,
                    tokens_per_block,
                    context_tokens,
                    tokens,
                    sinks,
                    sequences,
                )
            }
            Some(_) | None => Self::from_bytes(budget, bytes_per_token, tokens_per_block),
        }
    }

    /// Bytes a block occupies.
    pub fn block_bytes(&self) -> u64 {
        self.bytes_per_token.saturating_mul(self.tokens_per_block)
    }

    /// Total bytes the cache owns.
    pub fn total_bytes(&self) -> Option<u64> {
        self.blocks.checked_mul(self.block_bytes())
    }

    /// Tokens the table can address.
    pub fn capacity_tokens(&self) -> u64 {
        self.slot_count().saturating_mul(self.tokens_per_block)
    }

    /// Table entries the cache can hand out: one per block of the context.
    pub fn slot_count(&self) -> u64 {
        self.ring.map_or(self.blocks, |ring| ring.slots)
    }

    /// Blocks of the table needed to cover `tokens`.
    pub fn blocks_for(&self, tokens: u64) -> u64 {
        tokens.div_ceil(self.tokens_per_block)
    }
}

/// A KV cache with a block table per sequence.
#[derive(Debug)]
pub struct KvCache {
    config: KvConfig,
    storage: Vec<u8>,
    free: Vec<u32>,
    tables: BTreeMap<u64, Vec<u32>>,
    /// Sequences whose blocks are kept for reuse, oldest first.
    parked: VecDeque<(u64, Vec<u32>)>,
}

impl KvCache {
    /// Allocate the cache. The buffer is real and zeroed, so the memory is committed now rather than
    /// at first use, which is what makes a capacity refusal actionable.
    pub fn new(config: KvConfig) -> Result<Self, ViperaError> {
        let total = config
            .total_bytes()
            .ok_or_else(|| ViperaError::invalid_argument("KV cache size is not representable"))?;
        let total = usize::try_from(total).map_err(|_| {
            ViperaError::infeasible(format!(
                "{total} bytes of KV cache do not fit this machine's address space"
            ))
        })?;
        let mut storage: Vec<u8> = Vec::new();
        storage.try_reserve_exact(total).map_err(|_| {
            ViperaError::infeasible(format!(
                "cannot reserve {total} bytes for the KV cache; free host memory first"
            ))
        })?;
        storage.resize(total, 0);

        let mut free = Vec::with_capacity(config.blocks as usize);
        for block in (0..config.blocks as u32).rev() {
            free.push(block);
        }
        Ok(Self {
            config,
            storage,
            free,
            tables: BTreeMap::new(),
            parked: VecDeque::new(),
        })
    }

    /// Geometry of this cache.
    pub const fn config(&self) -> KvConfig {
        self.config
    }

    /// Bytes actually allocated for the cache.
    pub fn allocated_bytes(&self) -> usize {
        self.storage.len()
    }

    /// The backing store, for a backend that writes its own K/V layout into it.
    pub fn storage_mut(&mut self) -> &mut [u8] {
        &mut self.storage
    }

    /// Block table of a sequence, if it holds one.
    pub fn table(&self, sequence: u64) -> Option<&[u32]> {
        self.tables.get(&sequence).map(Vec::as_slice)
    }

    /// Free blocks.
    pub fn free_blocks(&self) -> u64 {
        self.free.len() as u64
    }

    /// Blocks in use.
    pub fn used_blocks(&self) -> u64 {
        self.config.blocks.saturating_sub(self.free_blocks())
    }

    /// Bytes in use.
    pub fn used_bytes(&self) -> u64 {
        self.used_blocks().saturating_mul(self.config.block_bytes())
    }

    /// Blocks held by parked sequences. They count as in use here, and are free to any
    /// [`KvCache::reserve`].
    pub fn parked_blocks(&self) -> u64 {
        self.parked
            .iter()
            .map(|(_, table)| table.len() as u64)
            .sum()
    }

    /// Whether `sequence` is parked.
    pub fn is_parked(&self, sequence: u64) -> bool {
        self.parked.iter().any(|(id, _)| *id == sequence)
    }

    /// Make sure `sequence` has blocks covering `tokens`, allocating the difference.
    ///
    /// When the free list runs out, parked sequences are given up, oldest first.
    pub fn reserve(&mut self, sequence: u64, tokens: u64) -> Result<(), ViperaError> {
        self.reserve_keeping(sequence, tokens, None)
    }

    /// [`KvCache::reserve`] that will not give up the parked sequence `keep`.
    fn reserve_keeping(
        &mut self,
        sequence: u64,
        tokens: u64,
        keep: Option<u64>,
    ) -> Result<(), ViperaError> {
        if let Some(ring) = self.config.ring {
            return self.reserve_ring(sequence, tokens, ring);
        }
        let needed = self.config.blocks_for(tokens);
        if needed > self.config.blocks {
            return Err(ViperaError::infeasible(format!(
                "{tokens} tokens need {needed} KV blocks but the cache has {}; shorten the context",
                self.config.blocks
            )));
        }
        let current = self
            .tables
            .get(&sequence)
            .map_or(0, |table| table.len() as u64);
        let deficit = needed.saturating_sub(current);
        let evictable: u64 = self
            .parked
            .iter()
            .filter(|(id, _)| Some(*id) != keep)
            .map(|(_, table)| table.len() as u64)
            .sum();
        let in_use = self.used_blocks().saturating_sub(self.parked_blocks());
        if (self.free.len() as u64).saturating_add(evictable) < deficit {
            return Err(ViperaError::infeasible(format!(
                "KV cache is full: {in_use} blocks in use, {needed} needed for sequence {sequence}"
            )));
        }
        self.tables.entry(sequence).or_default();
        while self
            .tables
            .get(&sequence)
            .map_or(0, |table| table.len() as u64)
            < needed
        {
            if self.free.is_empty() {
                self.evict_oldest_except(keep);
            }
            let Some(block) = self.free.pop() else {
                return Err(ViperaError::infeasible(format!(
                    "KV cache is full: {in_use} blocks in use, {needed} needed for sequence {sequence}"
                )));
            };
            self.tables.entry(sequence).or_default().push(block);
        }
        Ok(())
    }

    /// [`KvCache::reserve`] for a ring cache: the table grows to the position asked for, but only
    /// the sinks and the ring's own blocks are ever allocated, and the entries past those repeat.
    fn reserve_ring(&mut self, sequence: u64, tokens: u64, ring: Ring) -> Result<(), ViperaError> {
        let needed = self.config.blocks_for(tokens);
        if needed > ring.slots {
            return Err(ViperaError::infeasible(format!(
                "{tokens} tokens need {needed} KV blocks but the ring addresses {}; shorten the \
                 context",
                ring.slots
            )));
        }
        let pinned = ring.pinned_slots as usize;
        let rolling = ring.rolling_slots as usize;
        let per_sequence = pinned + rolling;
        let target = (needed as usize).max(per_sequence);
        let missing =
            per_sequence.saturating_sub(self.tables.get(&sequence).map_or(0, |table| table.len()));
        if (self.free.len() as u64) < missing as u64 {
            let in_use = self.used_blocks();
            return Err(ViperaError::infeasible(format!(
                "KV cache is full: {in_use} blocks in use, {per_sequence} needed for sequence \
                 {sequence}"
            )));
        }
        let table = self.tables.entry(sequence).or_default();
        // The sinks and the ring itself are real blocks, taken once.
        while table.len() < per_sequence {
            let block = self
                .free
                .pop()
                .ok_or_else(|| ViperaError::internal("ring free list ran dry"))?;
            table.push(block);
        }
        // Past that, an entry points at the block its position has wrapped onto.
        while table.len() < target {
            let slot = table.len();
            let block = table[pinned + (slot - pinned) % rolling];
            table.push(block);
        }
        Ok(())
    }

    /// Free the oldest parked sequence other than `keep`.
    fn evict_oldest_except(&mut self, keep: Option<u64>) {
        let Some(position) = self.parked.iter().position(|(id, _)| Some(*id) != keep) else {
            return;
        };
        if let Some((_, table)) = self.parked.remove(position) {
            self.free.extend(table);
        }
    }

    /// Keep `sequence`'s blocks instead of releasing them, so a later request can copy its
    /// leading positions. Returns `false` when it holds none.
    ///
    /// At most [`PARKED_LIMIT`] sequences stay parked; the oldest beyond that is freed.
    pub fn park(&mut self, sequence: u64) -> bool {
        if self.config.ring.is_some() {
            return false;
        }
        let Some(table) = self.tables.remove(&sequence) else {
            return false;
        };
        if table.is_empty() {
            return false;
        }
        self.parked.push_back((sequence, table));
        while self.parked.len() > PARKED_LIMIT {
            self.evict_oldest_except(None);
        }
        true
    }

    /// Give `sequence`, which holds no blocks yet, a copy of the blocks of the parked sequence
    /// `donor` that cover its first `tokens` positions.
    ///
    /// The copy takes whole blocks, so bytes past `tokens` in the last block come along; they are
    /// positions the new sequence has not computed and overwrites before it reads them. The donor
    /// is kept out of the eviction this copy may need, and becomes the most recently used. On an
    /// error `sequence` holds nothing.
    pub fn copy_from_parked(
        &mut self,
        donor: u64,
        sequence: u64,
        tokens: u64,
    ) -> Result<(), ViperaError> {
        if self.config.ring.is_some() {
            return Err(ViperaError::unsupported(
                "a ring KV cache aliases its blocks, so a parked sequence cannot be copied into it",
            ));
        }
        let needed = self.config.blocks_for(tokens) as usize;
        let Some(source) = self
            .parked
            .iter()
            .find(|(id, _)| *id == donor)
            .map(|(_, table)| table.clone())
        else {
            return Err(ViperaError::invalid_argument(format!(
                "sequence {donor} is not parked"
            )));
        };
        if source.len() < needed {
            return Err(ViperaError::invalid_argument(format!(
                "sequence {donor} holds {} blocks, fewer than the {needed} that {tokens} tokens need",
                source.len()
            )));
        }
        if self.tables.contains_key(&sequence) {
            return Err(ViperaError::invalid_argument(format!(
                "sequence {sequence} already holds blocks"
            )));
        }
        if let Err(error) = self.reserve_keeping(sequence, tokens, Some(donor)) {
            self.tables.remove(&sequence);
            return Err(error);
        }
        let block_bytes = self.config.block_bytes() as usize;
        let target = self.tables.get(&sequence).cloned().unwrap_or_default();
        for (from, to) in source.iter().zip(&target) {
            let from = *from as usize * block_bytes;
            self.storage
                .copy_within(from..from + block_bytes, *to as usize * block_bytes);
        }
        if let Some(position) = self.parked.iter().position(|(id, _)| *id == donor)
            && let Some(entry) = self.parked.remove(position)
        {
            self.parked.push_back(entry);
        }
        Ok(())
    }

    /// Drop blocks past `tokens`, keeping the sequence's earlier blocks.
    ///
    /// A ring has no per-position blocks to drop: it keeps its sinks and its ring, and the table
    /// shrinks to whatever still addresses `tokens` but never below one ring.
    pub fn truncate(&mut self, sequence: u64, tokens: u64) {
        let floor = self
            .config
            .ring
            .map_or(0, |ring| (ring.pinned_slots + ring.rolling_slots) as usize);
        let keep = (self.config.blocks_for(tokens) as usize).max(floor);
        if let Some(table) = self.tables.get_mut(&sequence) {
            while table.len() > keep {
                if let Some(block) = table.pop() {
                    self.free.push(block);
                }
            }
            if table.is_empty() {
                self.tables.remove(&sequence);
            }
        }
    }

    /// Release every block a sequence owns, parked or not. Returns the number of blocks freed.
    pub fn release(&mut self, sequence: u64) -> u64 {
        let mut table = match self.tables.remove(&sequence) {
            Some(table) => table,
            None => {
                let Some(position) = self.parked.iter().position(|(id, _)| *id == sequence) else {
                    return 0;
                };
                self.parked
                    .remove(position)
                    .map(|(_, table)| table)
                    .unwrap_or_default()
            }
        };
        // Ring tables repeat a block, so releasing an entry per slot would free the same block many
        // times over and hand the same memory to two sequences.
        table.sort_unstable();
        table.dedup();
        let freed = table.len() as u64;
        for block in table {
            self.free.push(block);
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(blocks: u64) -> KvCache {
        KvCache::new(KvConfig::new(64, 4, blocks).unwrap()).unwrap()
    }

    #[test]
    fn the_cache_allocates_the_bytes_it_claims() {
        let cache = cache(8);
        assert_eq!(cache.allocated_bytes(), 8 * 64 * 4);
        assert_eq!(cache.free_blocks(), 8);
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn blocks_cover_a_token_count() {
        let config = KvConfig::new(10, 16, 4).unwrap();
        assert_eq!(config.blocks_for(0), 0);
        assert_eq!(config.blocks_for(1), 1);
        assert_eq!(config.blocks_for(16), 1);
        assert_eq!(config.blocks_for(17), 2);
        assert_eq!(config.capacity_tokens(), 64);
    }

    #[test]
    fn reserving_grows_the_table_and_uses_low_blocks_first() {
        let mut cache = cache(8);
        cache.reserve(1, 9).unwrap();
        assert_eq!(cache.table(1), Some([0u32, 1, 2].as_slice()));
        assert_eq!(cache.used_blocks(), 3);
        cache.reserve(2, 4).unwrap();
        assert_eq!(cache.table(2), Some([3u32].as_slice()));
    }

    #[test]
    fn truncating_returns_the_tail_blocks() {
        let mut cache = cache(8);
        cache.reserve(1, 16).unwrap();
        assert_eq!(cache.used_blocks(), 4);
        cache.truncate(1, 5);
        assert_eq!(cache.table(1), Some([0u32, 1].as_slice()));
        assert_eq!(cache.used_blocks(), 2);
        cache.truncate(1, 0);
        assert_eq!(cache.table(1), None);
        assert_eq!(cache.used_blocks(), 0);
    }

    #[test]
    fn releasing_frees_every_block_of_a_sequence_only() {
        let mut cache = cache(4);
        cache.reserve(1, 8).unwrap();
        cache.reserve(2, 8).unwrap();
        assert_eq!(cache.free_blocks(), 0);

        assert_eq!(cache.release(1), 2);
        assert_eq!(cache.free_blocks(), 2);
        assert_eq!(cache.table(1), None);
        assert_eq!(cache.table(2), Some([2u32, 3].as_slice()));
        assert_eq!(cache.release(1), 0);
    }

    #[test]
    fn a_reservation_larger_than_the_cache_is_refused() {
        let mut cache = cache(2);
        let error = cache.reserve(1, 9).unwrap_err();
        assert_eq!(error.exit_code(), 3);
        assert!(error.to_string().contains("KV blocks"));
    }

    #[test]
    fn exhausting_the_cache_names_the_sequence() {
        let mut cache = cache(2);
        cache.reserve(1, 8).unwrap();
        let error = cache.reserve(2, 1).unwrap_err();
        assert!(error.to_string().contains("sequence 2"));
    }

    #[test]
    fn a_budget_is_rounded_down_to_whole_blocks() {
        let config = KvConfig::from_bytes(1000, 10, 16).unwrap();
        assert_eq!(config.blocks, 6);
        assert!(config.total_bytes().unwrap() <= 1000);
    }

    #[test]
    fn a_failed_reserve_leaves_no_partial_allocation() {
        let mut cache = cache(2);
        cache.reserve(1, 4).unwrap();
        assert_eq!(cache.free_blocks(), 1);
        let err = cache.reserve(2, 8).unwrap_err();
        assert!(err.to_string().contains("full"));
        assert_eq!(
            cache.free_blocks(),
            1,
            "failed reserve must not drain the free list"
        );
        assert_eq!(
            cache.table(2),
            None,
            "failed reserve must not leave a partial table"
        );
    }

    #[test]
    fn huge_geometry_is_rejected_not_wrapped() {
        assert!(KvConfig::new(1 << 40, 1 << 40, 1).is_err());
        assert!(KvConfig::new(2, 16, u32::MAX as u64 + 1).is_err());
    }

    #[test]
    fn nonsense_geometry_is_rejected() {
        assert!(KvConfig::new(0, 16, 1).is_err());
        assert!(KvConfig::new(2, 0, 1).is_err());
        assert!(KvConfig::new(2, 16, 0).is_err());
    }

    /// Write `marker` into every byte of each block of `sequence`.
    fn fill(cache: &mut KvCache, sequence: u64, marker: u8) {
        let block_bytes = cache.config().block_bytes() as usize;
        let table = cache.table(sequence).expect("table").to_vec();
        for block in table {
            let start = block as usize * block_bytes;
            cache.storage_mut()[start..start + block_bytes].fill(marker);
        }
    }

    #[test]
    fn a_parked_sequence_is_given_back_oldest_first_when_the_free_list_runs_out() {
        let mut cache = cache(4);
        cache.reserve(1, 8).unwrap();
        cache.reserve(2, 8).unwrap();
        assert!(cache.park(1));
        assert!(cache.park(2));
        assert_eq!(cache.free_blocks(), 0);
        assert_eq!(cache.parked_blocks(), 4);

        cache.reserve(3, 4).unwrap();
        assert!(!cache.is_parked(1), "the oldest parked sequence goes first");
        assert!(cache.is_parked(2));
        assert_eq!(cache.table(3).map(<[u32]>::len), Some(1));
        assert_eq!(cache.parked_blocks(), 2);
    }

    #[test]
    fn parked_blocks_never_make_a_reservation_fail() {
        let mut cache = cache(4);
        cache.reserve(1, 16).unwrap();
        assert!(cache.park(1));
        cache.reserve(2, 16).unwrap();
        assert_eq!(cache.parked_blocks(), 0);
        assert_eq!(cache.table(2).map(<[u32]>::len), Some(4));
        // A request larger than the whole cache is still refused, parked or not.
        assert!(cache.reserve(3, 17).is_err());
    }

    #[test]
    fn copying_from_a_parked_sequence_copies_its_bytes_and_keeps_the_donor_parked() {
        let mut cache = cache(8);
        cache.reserve(1, 8).unwrap();
        fill(&mut cache, 1, 0xAB);
        assert!(cache.park(1));

        cache.copy_from_parked(1, 2, 6).unwrap();
        let block_bytes = cache.config().block_bytes() as usize;
        let table = cache.table(2).expect("table").to_vec();
        assert_eq!(table.len(), 2, "6 tokens at 4 per block");
        for block in table {
            let start = block as usize * block_bytes;
            assert!(
                cache.storage_mut()[start..start + block_bytes]
                    .iter()
                    .all(|byte| *byte == 0xAB)
            );
        }
        assert!(cache.is_parked(1), "the donor stays available");
        // The copy is independent: writing the new sequence leaves the donor alone.
        fill(&mut cache, 2, 0x11);
        cache.copy_from_parked(1, 3, 4).unwrap();
        let first = cache.table(3).expect("table")[0] as usize * block_bytes;
        assert_eq!(cache.storage_mut()[first], 0xAB);
    }

    #[test]
    fn a_copy_never_gives_up_its_own_donor_to_make_room() {
        let mut cache = cache(4);
        cache.reserve(1, 8).unwrap();
        cache.reserve(2, 8).unwrap();
        fill(&mut cache, 2, 0x5A);
        assert!(cache.park(1));
        assert!(cache.park(2));
        // The cache is full of parked sequences; 2 is the donor, so 1 is the one to go.
        cache.copy_from_parked(2, 3, 8).unwrap();
        assert!(!cache.is_parked(1));
        assert!(cache.is_parked(2));
        let block_bytes = cache.config().block_bytes() as usize;
        let first = cache.table(3).expect("table")[0] as usize * block_bytes;
        assert_eq!(cache.storage_mut()[first], 0x5A);
    }

    #[test]
    fn a_copy_that_cannot_fit_leaves_nothing_behind() {
        let mut cache = cache(3);
        cache.reserve(1, 12).unwrap();
        assert!(cache.park(1));
        // The only parked sequence is the donor and it is kept, so nothing is free for the copy.
        let error = cache.copy_from_parked(1, 2, 12);
        assert!(error.is_err());
        assert_eq!(cache.table(2), None);
        assert!(cache.is_parked(1));
        assert!(cache.copy_from_parked(7, 2, 4).is_err(), "not parked");
    }

    #[test]
    fn releasing_a_parked_sequence_frees_it() {
        let mut cache = cache(4);
        cache.reserve(1, 8).unwrap();
        assert!(cache.park(1));
        assert_eq!(cache.release(1), 2);
        assert!(!cache.is_parked(1));
        assert_eq!(cache.free_blocks(), 4);
        assert_eq!(cache.release(1), 0);
    }

    /// A 32k cache whose table can address 32k positions but whose storage is a window plus sinks.
    fn ring() -> KvCache {
        let config = KvConfig::ring(64, 16, 32_768, 2048, 4, 1).unwrap();
        KvCache::new(config).unwrap()
    }

    #[test]
    fn a_run_only_gets_a_ring_when_the_window_is_shorter_than_its_context() {
        let window = Some((2048, 4));
        let ring = KvConfig::for_run(64, 16, 32_768, window, 1).unwrap();
        assert!(ring.ring.is_some());
        let paged = KvConfig::for_run(64, 16, 2048, window, 1).unwrap();
        assert!(
            paged.ring.is_none(),
            "a window that keeps the whole context must not pay for a ring"
        );
        assert_eq!(paged.blocks, 128);
        assert!(
            KvConfig::for_run(64, 16, 4096, None, 1)
                .unwrap()
                .ring
                .is_none()
        );
    }

    #[test]
    fn a_ring_owns_the_window_and_the_sinks_not_the_whole_context() {
        let cache = ring();
        let full = 32_768 * 64;
        assert!(
            cache.allocated_bytes() as u64 <= full / 10,
            "{} bytes for a 32k context is not a window",
            cache.allocated_bytes()
        );
        // The table still addresses every position of the context.
        assert_eq!(cache.config().capacity_tokens(), 32_768);
        assert_eq!(cache.config().blocks_for(32_768), 2048);
        assert_eq!(cache.config().blocks_for(1), 1);
    }

    #[test]
    fn a_ring_table_rewrites_dropped_positions_and_keeps_every_kept_slot_distinct() {
        let mut cache = ring();
        cache.reserve(1, 32_768).unwrap();
        let table = cache.table(1).expect("table").to_vec();
        assert_eq!(table.len(), 2048, "one entry per block of the context");

        let (pinned, rolling) = {
            let ring = cache.config().ring.expect("ring");
            (ring.pinned_slots as usize, ring.rolling_slots as usize)
        };
        // Sink slots never move, so the first tokens of the history stay readable.
        let sinks: Vec<u32> = table[..pinned].to_vec();
        let mut pinned_unique = sinks.clone();
        pinned_unique.sort_unstable();
        pinned_unique.dedup();
        assert_eq!(pinned_unique.len(), sinks.len(), "sink slots are distinct");
        // The ring region is a cycle, and the table repeats it once it wraps.
        assert_eq!(table[pinned + rolling], table[pinned]);
        assert_eq!(table[pinned + rolling + 3], table[pinned + 3]);
        assert_eq!(table[2047], table[pinned + (2047 - pinned) % rolling]);

        // What a decode at the end of the context actually reads: the last window, whose slots
        // must be distinct, so the window never reads a block it has already overwritten. A window
        // is 2048 tokens, which straddles one more block than 2048 / 16.
        let window_slots = (2048 + 16) / 16;
        let mut kept: Vec<u32> = table[table.len() - window_slots..].to_vec();
        let seen = kept.len();
        kept.sort_unstable();
        kept.dedup();
        assert_eq!(kept.len(), seen, "kept slots must not collide");
        // And a sink never shares a block with the ring, so evicting cannot eat the first tokens.
        assert!(
            kept.iter().all(|block| !sinks.contains(block)),
            "the ring must not overwrite a sink block"
        );
    }

    #[test]
    fn a_ring_frees_each_physical_block_once() {
        let mut cache = ring();
        let blocks = cache.config().blocks;
        cache.reserve(1, 32_768).unwrap();
        assert_eq!(cache.used_blocks(), blocks);
        assert_eq!(cache.release(1), blocks);
        assert_eq!(cache.free_blocks(), blocks, "aliases must not leak blocks");
    }

    #[test]
    fn a_ring_refuses_to_park_and_cannot_address_more_than_its_context() {
        let mut cache = ring();
        cache.reserve(1, 16).unwrap();
        assert!(
            !cache.park(1),
            "a ring's blocks are aliased, so they cannot be reused"
        );
        let error = cache.reserve(2, 33_000).unwrap_err();
        assert!(error.to_string().contains("KV blocks"), "{error}");
    }

    #[test]
    fn only_the_most_recent_sequences_stay_parked() {
        let mut cache = cache(40);
        for sequence in 0..PARKED_LIMIT as u64 + 3 {
            cache.reserve(sequence, 4).unwrap();
            assert!(cache.park(sequence));
        }
        assert!(!cache.is_parked(0));
        assert!(!cache.is_parked(2));
        assert!(cache.is_parked(3));
        assert!(cache.is_parked(PARKED_LIMIT as u64 + 2));
        assert_eq!(cache.parked_blocks(), PARKED_LIMIT as u64);
    }
}
