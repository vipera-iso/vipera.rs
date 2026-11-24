//! Dynamic MoE expert cache: hot experts stay resident ("pinned" in the fast tier), cold ones
//! fall back to the CPU load path.
//!
//! [`MoEExpertCache::load_expert`] is the dispatch-side entry point a backend calls before
//! computing a token's experts: `true` = the expert was already resident, `false` = it had to be
//! loaded. Capacity comes from `VIPERA_MOE_CACHE_N`; unset or `0` leaves the cache disabled — the
//! baseline behavior before it was added.

use std::collections::HashMap;

/// Window of recent accesses kept per expert for [`MoEExpertCache::hotness`].
const FREQ_WINDOW: usize = 64;

/// An LRU cache of hot experts with per-expert rolling-window access frequency tracking.
///
/// Disabled when built with capacity `0`: [`load_expert`](MoEExpertCache::load_expert) then
/// reports every expert as a miss and nothing is held.
pub struct MoEExpertCache {
    /// How many experts the cache holds; `0` disables it.
    capacity: usize,
    /// Resident experts: index → tick of its last use (smallest ticks evicted first).
    resident: HashMap<u32, u64>,
    /// Recent access ticks per expert, trimmed to [`FREQ_WINDOW`] entries.
    history: HashMap<u32, Vec<u64>>,
    /// Monotonic clock the recency ticks come from.
    clock: u64,
}

impl MoEExpertCache {
    /// A cache that holds at most `capacity` experts. Capacity `0` disables the cache: every
    /// load is a miss and nothing is resident.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            resident: HashMap::new(),
            history: HashMap::new(),
            clock: 0,
        }
    }

    /// A cache sized by `VIPERA_MOE_CACHE_N`; unset or `0` disables it (the default).
    pub fn from_env() -> Self {
        Self::new(Self::env_capacity())
    }

    /// `VIPERA_MOE_CACHE_N` parsed as a capacity. Unset or unparseable means `0` (disabled).
    pub fn env_capacity() -> usize {
        Self::env_capacity_from(std::env::var("VIPERA_MOE_CACHE_N").ok())
    }

    /// Capacity for a raw `VIPERA_MOE_CACHE_N` value; `None` or unparseable means `0`.
    pub fn env_capacity_from(value: Option<String>) -> usize {
        value
            .as_deref()
            .map_or(0, |value| value.trim().parse::<usize>().unwrap_or(0))
    }

    /// How many experts the cache holds; `0` when disabled.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the cache holds experts at all.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// Whether `idx` is currently resident (no load needed).
    pub fn contains(&self, idx: usize) -> bool {
        self.resident.contains_key(&(idx as u32))
    }

    /// Accesses recorded for `idx` in the rolling window; `0` when never loaded.
    pub fn hotness(&self, idx: usize) -> usize {
        self.history.get(&(idx as u32)).map_or(0, Vec::len)
    }

    /// Load `idx` through the cache: `true` when it was already resident, `false` when a load
    /// was needed. On a miss with a full cache, the least recently used expert is evicted.
    pub fn load_expert(&mut self, idx: usize) -> bool {
        let expert = idx as u32;
        self.clock += 1;
        let now = self.clock;

        let history = self.history.entry(expert).or_default();
        history.push(now);
        if history.len() > FREQ_WINDOW {
            history.drain(..history.len() - FREQ_WINDOW);
        }

        if self.capacity == 0 {
            return false;
        }
        if let Some(tick) = self.resident.get_mut(&expert) {
            *tick = now;
            return true;
        }
        if self.resident.len() >= self.capacity {
            let victim = self
                .resident
                .iter()
                .min_by_key(|(_, tick)| **tick)
                .map(|(expert, _)| *expert);
            if let Some(victim) = victim {
                self.resident.remove(&victim);
            }
        }
        self.resident.insert(expert, now);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_then_hit_on_resident_expert() {
        let mut cache = MoEExpertCache::new(2);
        assert!(!cache.load_expert(0)); // first load: miss
        assert!(cache.contains(0));
        assert!(cache.load_expert(0)); // resident now: hit
        assert!(!cache.load_expert(7)); // a different expert is a miss
        assert!(cache.load_expert(7));
    }

    #[test]
    fn lru_evicts_least_recently_used_first() {
        let mut cache = MoEExpertCache::new(2);
        assert!(!cache.load_expert(0));
        assert!(!cache.load_expert(1));
        assert!(!cache.load_expert(2)); // capacity 2: 0 is the least recently used
        assert!(!cache.contains(0));
        assert!(cache.contains(1));
        assert!(cache.contains(2));

        assert!(!cache.load_expert(0)); // still a miss — it was evicted
        assert!(!cache.contains(1)); // touching 0 evicted 1, the next least recent
        assert!(cache.contains(0));
        assert!(cache.contains(2));
    }

    #[test]
    fn repeated_hits_refresh_recency_before_eviction() {
        let mut cache = MoEExpertCache::new(2);
        assert!(!cache.load_expert(10));
        assert!(!cache.load_expert(11));

        // Repeated hits on 10 make it the most recently used, even though 11 was loaded later.
        for _ in 0..8 {
            assert!(cache.load_expert(10));
        }

        assert!(!cache.load_expert(12));
        assert!(!cache.contains(11)); // 11 is now the least recently used
        assert!(cache.contains(10));
        assert!(cache.contains(12));

        // Hits on 12 refresh it; on the next miss 10, not 12, is the victim.
        for _ in 0..8 {
            assert!(cache.load_expert(12));
        }
        assert!(!cache.load_expert(13));
        assert!(!cache.contains(10));
        assert!(cache.contains(12));
        assert!(cache.contains(13));
    }

    #[test]
    fn hotness_tracks_accesses_in_rolling_window() {
        let mut cache = MoEExpertCache::new(4);
        for _ in 0..FREQ_WINDOW + 1 {
            cache.load_expert(3);
        }
        assert_eq!(cache.hotness(3), FREQ_WINDOW);
        cache.load_expert(0);
        assert_eq!(cache.hotness(0), 1);
        assert_eq!(cache.hotness(99), 0);
    }

    #[test]
    fn zero_capacity_disables_the_cache() {
        let mut cache = MoEExpertCache::new(0);
        assert!(!cache.is_enabled());
        for idx in 0..8 {
            assert!(!cache.load_expert(idx));
            assert!(!cache.contains(idx));
        }
        assert_eq!(cache.hotness(4), 1); // frequency tracking still counts
    }

    #[test]
    fn env_capacity_reads_vipera_moe_cache_n() {
        // A missing or unparseable value disables the cache; a number sizes it.
        assert_eq!(MoEExpertCache::env_capacity_from(None), 0);
        assert_eq!(MoEExpertCache::env_capacity_from(Some("0".into())), 0);
        assert_eq!(MoEExpertCache::env_capacity_from(Some("16".into())), 16);
        assert_eq!(MoEExpertCache::env_capacity_from(Some(" 7 ".into())), 7);
        assert_eq!(MoEExpertCache::env_capacity_from(Some("garbage".into())), 0);
        assert_eq!(
            MoEExpertCache::env_capacity(),
            MoEExpertCache::env_capacity_from(std::env::var("VIPERA_MOE_CACHE_N").ok())
        );
        assert_eq!(
            MoEExpertCache::from_env().capacity(),
            MoEExpertCache::env_capacity()
        );
    }
}
