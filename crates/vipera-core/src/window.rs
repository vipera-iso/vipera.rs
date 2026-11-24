//! The sliding-window attention setting, read once from the environment.
//!
//! `VIPERA_ATTN_WINDOW=<positions>` makes a decode step attend to the sinks and the last `positions`
//! of the history instead of all of it, and `VIPERA_ATTN_SINKS=<tokens>` (default 4) is how much of
//! the front stays visible. It lives here, rather than only in the backend that reads K/V, because
//! the cache a run allocates depends on it: with a window the KV only has to *hold* the window, so
//! planning and allocation have to agree on the same numbers the attention kernel will use.

use std::sync::OnceLock;

/// Tokens of a prompt chunk a window has to keep behind itself.
///
/// A chunk's positions are all written before the first of them reads its window, so a window that
/// kept exactly `window` positions could evict a row the chunk had not read yet.
pub const PREFILL_HEADROOM: u64 = 128;

/// Sliding-window attention: positions kept before the current one, and tokens kept at the front.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Positions attended to immediately before the current one.
    pub tokens: u64,
    /// Tokens at the front of the history that are always attended to.
    pub sinks: u64,
}

impl Window {
    /// Whether `positions` is longer than the history this window keeps, so dropping it changes
    /// anything.
    pub fn is_lossy_at(&self, positions: u64) -> bool {
        positions > self.tokens.saturating_add(self.sinks)
    }
}

/// The configured window, or `None` for full-history attention.
///
/// Read once: a run's KV layout is decided at allocation time, so changing the variable mid-run
/// cannot change what the cache holds.
pub fn attention_window() -> Option<Window> {
    static WINDOW: OnceLock<Option<Window>> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        let tokens: u64 = std::env::var("VIPERA_ATTN_WINDOW")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        if tokens == 0 {
            return None;
        }
        let sinks = std::env::var("VIPERA_ATTN_SINKS")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(4);
        Some(Window { tokens, sinks })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_is_lossy_only_past_the_sinks_and_the_window() {
        let window = Window {
            tokens: 2048,
            sinks: 4,
        };
        assert!(!window.is_lossy_at(2052));
        assert!(window.is_lossy_at(2053));
    }

    #[test]
    fn zero_length_window_is_lossy_past_sinks() {
        // With 0 tokens and 4 sinks, positions > 4 are lossy
        let window = Window {
            tokens: 0,
            sinks: 4,
        };
        assert!(!window.is_lossy_at(4)); // exactly at the boundary
        assert!(window.is_lossy_at(5)); // past the boundary
        assert!(!window.is_lossy_at(0)); // before the boundary
        assert!(!window.is_lossy_at(1)); // before the boundary
    }

    #[test]
    fn single_token_window_is_lossy_past_sinks() {
        // With 1 token and 4 sinks, positions > 5 are lossy
        let window = Window {
            tokens: 1,
            sinks: 4,
        };
        assert!(!window.is_lossy_at(5)); // exactly at the boundary (1 + 4)
        assert!(window.is_lossy_at(6)); // past the boundary
        assert!(!window.is_lossy_at(4)); // before the boundary
        assert!(!window.is_lossy_at(3)); // before the boundary
    }
}
