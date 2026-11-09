//! Paged KV cache: fixed-size blocks, one table per sequence.
//!
//! The cache owns a real zeroed buffer, so a runtime built on it consumes the memory it claims and
//! `vipera verify` measures the same bytes the plan predicted. Blocks are handed out lowest-id first
//! and returned on release; there is no compaction, which keeps block tables stable across turns.

use std::collections::BTreeMap;

use vipera_errors::ViperaError;

/// Tokens per block. 16 is the paged-attention default and keeps waste under one block per sequence.
pub const DEFAULT_TOKENS_PER_BLOCK: u64 = 16;

/// Geometry of a KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvConfig {
    /// Bytes one token occupies across every layer of the cache (keys plus values).
    pub bytes_per_token: u64,
    /// Tokens a block holds.
    pub tokens_per_block: u64,
    /// Blocks the cache owns.
    pub blocks: u64,
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

    /// Bytes a block occupies.
    pub fn block_bytes(&self) -> u64 {
        self.bytes_per_token.saturating_mul(self.tokens_per_block)
    }

    /// Total bytes the cache owns.
    pub fn total_bytes(&self) -> Option<u64> {
        self.blocks.checked_mul(self.block_bytes())
    }

    /// Tokens every block together can hold.
    pub fn capacity_tokens(&self) -> u64 {
        self.blocks.saturating_mul(self.tokens_per_block)
    }

    /// Blocks needed to cover `tokens`.
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

    /// Make sure `sequence` has blocks covering `tokens`, allocating the difference.
    pub fn reserve(&mut self, sequence: u64, tokens: u64) -> Result<(), ViperaError> {
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
        if (self.free.len() as u64) < deficit {
            return Err(ViperaError::infeasible(format!(
                "KV cache is full: {} blocks in use, {} needed for sequence {sequence}",
                self.used_blocks(),
                needed
            )));
        }
        let table = self.tables.entry(sequence).or_default();
        while (table.len() as u64) < needed {
            let Some(block) = self.free.pop() else {
                return Err(ViperaError::infeasible(format!(
                    "KV cache is full: {} blocks in use, {} needed for sequence {sequence}",
                    self.used_blocks(),
                    needed
                )));
            };
            table.push(block);
        }
        Ok(())
    }

    /// Drop blocks past `tokens`, keeping the sequence's earlier blocks.
    pub fn truncate(&mut self, sequence: u64, tokens: u64) {
        let keep = self.config.blocks_for(tokens) as usize;
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

    /// Release every block a sequence owns. Returns the number of blocks freed.
    pub fn release(&mut self, sequence: u64) -> u64 {
        let Some(table) = self.tables.remove(&sequence) else {
            return 0;
        };
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
}
