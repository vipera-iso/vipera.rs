//! Top-k retention for a KV cache that is over budget.
//!
//! A position's score is the mean attention mass it received over the last [`KeepKv::recent`] query
//! rows, so the entries that mattered to recent decoding survive and the ones no query looked at
//! are the first to go. Selection is by block: [`KeepKv::retain`] spends whole [`CacheGeometry`]
//! blocks, because a cache can only own whole blocks, and a budget too small for one block keeps
//! nothing.
//!
//! This crate decides *which* entries survive, it does not free anything. A paged cache maps a
//! position to a block, and the attention kernel reads that block without asking whether the
//! position was evicted; freeing a block while a query can still reach it would make that query
//! read whatever the next sequence wrote there. So [`KeepKv::retain`] is a pure function of a
//! geometry and a score vector, and the runtime exposes the decision (`KvCache::trim`) without
//! touching its table until the kernel can honour the retained set.

#![warn(missing_docs)]

/// What the policy needs to know about a cache: how much one position costs and how it is blocked.
pub trait CacheGeometry {
    /// Bytes one position occupies across every layer (keys plus values).
    fn bytes_per_token(&self) -> u64;

    /// Positions one block holds.
    fn tokens_per_block(&self) -> u64;
}

/// Retention policy: keep the highest-scoring positions that fit `budget_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepKv {
    /// Bytes the retained positions may occupy, in whole blocks.
    budget_bytes: u64,
    /// Query rows at the end of the step whose attention weights are averaged.
    recent: usize,
}

impl KeepKv {
    /// Query rows [`KeepKv::from_env`] averages when `VIPERA_KEEPKV_RECENT` is unset.
    pub const DEFAULT_RECENT: usize = 128;

    /// A policy that keeps `budget_bytes` worth of blocks, scored over the last `recent` queries.
    pub const fn new(budget_bytes: u64, recent: usize) -> Self {
        Self {
            budget_bytes,
            recent,
        }
    }

    /// The byte budget, as given.
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Query rows the score averages over.
    pub const fn recent(&self) -> usize {
        self.recent
    }

    /// The policy a run asked for, or `None` when the switch is off.
    ///
    /// `VIPERA_KEEPKV_BUDGET_KB` is the budget in kibibytes; unset, `0` or unparsable all mean off,
    /// matching the convention of the other `VIPERA_*` switches. `VIPERA_KEEPKV_RECENT` overrides
    /// [`KeepKv::DEFAULT_RECENT`].
    pub fn from_env() -> Option<Self> {
        Self::from_values(
            std::env::var("VIPERA_KEEPKV_BUDGET_KB").ok().as_deref(),
            std::env::var("VIPERA_KEEPKV_RECENT").ok().as_deref(),
        )
    }

    /// [`KeepKv::from_env`]'s rule over raw values, so it is testable without mutating the process
    /// environment.
    fn from_values(budget_kb: Option<&str>, recent: Option<&str>) -> Option<Self> {
        let budget_kb: u64 = budget_kb?.trim().parse().ok()?;
        if budget_kb == 0 {
            return None;
        }
        let recent = recent
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(Self::DEFAULT_RECENT);
        Some(Self {
            budget_bytes: budget_kb.saturating_mul(1024),
            recent,
        })
    }

    /// Mean attention weight every position received over the last [`KeepKv::recent`] rows.
    ///
    /// `weights` is row-major, one row per query, `positions` columns. Rows before the last
    /// `recent` are ignored, a position no row attended to stays at `0`, and a value that is not
    /// finite contributes nothing. Fewer rows than `recent` averages over the rows that are there,
    /// so a short batch is not diluted by rows that do not exist.
    pub fn score(&self, weights: &[f32], positions: usize) -> Vec<f32> {
        let mut scores = vec![0.0_f32; positions];
        if positions == 0 || self.recent == 0 {
            return scores;
        }
        let present = weights.chunks_exact(positions).count();
        let used = present.min(self.recent);
        if used == 0 {
            return scores;
        }
        for row in weights
            .chunks_exact(positions)
            .skip(present.saturating_sub(used))
        {
            for (total, weight) in scores.iter_mut().zip(row) {
                if weight.is_finite() {
                    *total += *weight;
                }
            }
        }
        let divisor = used as f32;
        for total in &mut scores {
            *total /= divisor;
        }
        scores
    }

    /// Indices of the positions to keep: the highest `scores`, lowest index first on a tie.
    ///
    /// The budget is spent in whole blocks of `cache`, so the retained positions are at most
    /// `floor(budget / block_bytes) * tokens_per_block` — the bytes the cache would actually own.
    /// The result is sorted by index, which makes it usable as a retained-set mask and keeps the
    /// tie-break independent of the sort implementation. A non-finite score ranks below every
    /// finite one, so a position whose weights were never recorded is never kept ahead of a real
    /// one.
    pub fn retain<C: CacheGeometry + ?Sized>(&self, cache: &C, scores: &[f32]) -> Vec<usize> {
        let per_block = cache.tokens_per_block().max(1);
        let block_bytes = cache.bytes_per_token().saturating_mul(per_block);
        if block_bytes == 0 {
            return Vec::new();
        }
        let capacity = (self.budget_bytes / block_bytes).saturating_mul(per_block);
        let keep = usize::try_from(capacity)
            .unwrap_or(usize::MAX)
            .min(scores.len());
        if keep == 0 {
            return Vec::new();
        }

        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&left, &right| {
            rank(scores[right])
                .total_cmp(&rank(scores[left]))
                .then_with(|| left.cmp(&right))
        });
        order.truncate(keep);
        order.sort_unstable();
        order
    }
}

/// A score's rank, with everything that is not a number treated as the worst score.
fn rank(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache of 256-byte blocks that hold four positions each: the budgets below are exact
    /// multiples of a block, so a test that keeps `n` positions is not fighting the rounding.
    struct Geometry {
        bytes_per_token: u64,
        tokens_per_block: u64,
    }

    impl CacheGeometry for Geometry {
        fn bytes_per_token(&self) -> u64 {
            self.bytes_per_token
        }

        fn tokens_per_block(&self) -> u64 {
            self.tokens_per_block
        }
    }

    fn blocks() -> Geometry {
        Geometry {
            bytes_per_token: 64,
            tokens_per_block: 4,
        }
    }

    #[test]
    fn a_zero_budget_keeps_nothing() {
        let policy = KeepKv::new(0, 8);
        assert!(policy.retain(&blocks(), &[0.9, 0.8, 0.7]).is_empty());
    }

    #[test]
    fn an_unbounded_budget_keeps_every_entry() {
        let policy = KeepKv::new(u64::MAX, 8);
        assert_eq!(policy.retain(&blocks(), &[0.1, 0.9, 0.5]), vec![0, 1, 2]);
    }

    #[test]
    fn the_budget_is_spent_in_whole_blocks() {
        // Two 256-byte blocks hold eight positions.
        let policy = KeepKv::new(512, 8);
        assert_eq!(
            policy.retain(&blocks(), &[0.0; 8]),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        // 300 bytes buy one block, not one and a bit.
        let policy = KeepKv::new(300, 8);
        assert_eq!(policy.retain(&blocks(), &[0.0; 8]), vec![0, 1, 2, 3]);
        // Below one block there is nothing a cache could own.
        let policy = KeepKv::new(255, 8);
        assert!(policy.retain(&blocks(), &[0.0; 8]).is_empty());
    }

    #[test]
    fn descending_scores_keep_the_highest_and_return_them_in_index_order() {
        // One 256-byte block holds four of the eight positions.
        let policy = KeepKv::new(256, 8);
        let scores = [0.1, 7.0, 0.2, 9.0, 0.3, 8.0, 0.4, 0.5];
        assert_eq!(policy.retain(&blocks(), &scores), vec![1, 3, 5, 7]);
    }

    #[test]
    fn a_tie_keeps_the_lower_indices() {
        // One position per block, so the budget of two positions is exact.
        let flat = Geometry {
            bytes_per_token: 1,
            tokens_per_block: 1,
        };
        let policy = KeepKv::new(2, 8);
        assert_eq!(policy.retain(&flat, &[1.0, 1.0, 1.0]), vec![0, 1]);
        // The same set regardless of where the tie sits.
        assert_eq!(policy.retain(&flat, &[1.0, 1.0, 1.0, 1.0]), vec![0, 1]);
    }

    #[test]
    fn a_missing_score_ranks_below_every_real_one() {
        let flat = Geometry {
            bytes_per_token: 1,
            tokens_per_block: 1,
        };
        let policy = KeepKv::new(2, 8);
        let scores = [f32::NAN, 0.0, -1.0];
        assert_eq!(policy.retain(&flat, &scores), vec![1, 2]);
    }

    #[test]
    fn a_geometry_without_bytes_keeps_nothing() {
        let empty = Geometry {
            bytes_per_token: 0,
            tokens_per_block: 4,
        };
        let policy = KeepKv::new(u64::MAX, 8);
        assert!(policy.retain(&empty, &[1.0, 2.0]).is_empty());
    }

    #[test]
    fn the_switch_is_off_unless_a_positive_budget_is_set() {
        assert!(KeepKv::from_values(None, None).is_none());
        assert!(KeepKv::from_values(Some("0"), None).is_none());
        assert!(KeepKv::from_values(Some("  "), None).is_none());
        assert!(KeepKv::from_values(Some("lots"), None).is_none());
        let policy = KeepKv::from_values(Some("16"), None).expect("a budget");
        assert_eq!(policy.budget_bytes(), 16 * 1024);
        assert_eq!(policy.recent(), KeepKv::DEFAULT_RECENT);
        let policy = KeepKv::from_values(Some("16"), Some("32")).expect("a budget");
        assert_eq!(policy.recent(), 32);
    }

    #[test]
    fn a_score_is_the_mean_over_the_recent_rows_only() {
        // Two positions, three rows; `recent = 2` ignores the first row.
        let policy = KeepKv::new(1024, 2);
        let weights = [9.0, 9.0, 0.2, 0.4, 0.4, 0.0];
        assert_eq!(policy.score(&weights, 2), vec![0.3, 0.2]);
        // `recent = 0` has no window to average over.
        let policy = KeepKv::new(1024, 0);
        assert_eq!(policy.score(&weights, 2), vec![0.0, 0.0]);
        // A position no row attended to scores zero, and a non-finite weight contributes nothing.
        let policy = KeepKv::new(1024, 8);
        let weights = [f32::NAN, 0.0, 1.0, 0.0];
        assert_eq!(policy.score(&weights, 2), vec![0.5, 0.0]);
        // Fewer rows than the window averages over the rows that exist, not over the window.
        let policy = KeepKv::new(1024, 8);
        assert_eq!(policy.score(&[0.5, 1.5], 2), vec![0.5, 1.5]);
    }

    #[test]
    fn retention_reads_geometry_through_the_trait() {
        // The same scores and budget, two layouts: block rounding follows the cache, not the policy.
        let narrow = Geometry {
            bytes_per_token: 8,
            tokens_per_block: 16,
        };
        let policy = KeepKv::new(112, 4);
        // 112 bytes / (8 * 16) = 0 blocks.
        assert!(policy.retain(&narrow, &[1.0; 32]).is_empty());
        let policy = KeepKv::new(128, 4);
        assert_eq!(policy.retain(&narrow, &[1.0; 32]).len(), 16);
    }
}
