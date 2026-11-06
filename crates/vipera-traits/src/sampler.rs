//! The sampling interface, and the registry a build's strategies are looked up in.
//!
//! A strategy here is one policy for turning logits into a token id: greedy takes the argmax, a
//! temperature strategy filters and draws. The math lives with the strategy, not here, so this crate
//! stays free of decisions about distributions and only owns the shape of the call.
//!
//! ```text
//! vipera-traits::Sampler        the interface          (this file)
//! vipera-sampler::Greedy        an implementation      one per policy
//! vipera-sampler::SAMPLERS      the build's registry   one line listing them
//! ```
//!
//! Adding a strategy is a new type plus one entry in that list. Nothing that consumes a
//! [`Sampler`] changes, which is the property the test in `vipera-sampler` holds: a strategy that
//! exists only inside a test is found by the same lookup as the real ones.

use std::fmt;

/// Why a strategy could not pick a token.
///
/// Two cases rather than one because they are different failures: the first means the model produced
/// nothing usable, the second means the filters removed everything. A caller that turns these into
/// messages should keep them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleError {
    /// Every logit was NaN or infinite.
    NoFiniteLogits,
    /// Nothing was left, or the weights of what was left summed to no usable mass.
    NoMass,
}

/// A source of randomness, so a strategy never reaches for one itself.
///
/// Implemented outside this crate (the generator lives with the sampler), which is why it is a
/// trait object rather than a generic: [`Sampler`] has to stay object safe for the registry.
pub trait Random {
    /// A uniform value in `[0, 1)`.
    fn next_f64(&mut self) -> f64;
}

/// The adjustment a strategy applies to a logit before ranking it.
///
/// The repetition penalty is the only one today, and it depends on the tokens already in context,
/// so it arrives as an object instead of being recomputed per strategy.
pub trait LogitPenalty {
    /// The logit for `id`, after the penalty.
    fn apply(&self, id: usize, logit: f32) -> f32;
}

/// A penalty that changes nothing, for a caller with no context to penalize.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPenalty;

impl LogitPenalty for NoPenalty {
    fn apply(&self, _id: usize, logit: f32) -> f32 {
        logit
    }
}

/// One call to a strategy: the logits and the knobs it may consult.
///
/// Every strategy receives the whole set, and each reads only what its policy uses; a strategy that
/// ignores `top_p` is not obliged to explain why.
#[derive(Clone, Copy)]
pub struct SampleRequest<'a> {
    /// One logit per vocabulary entry, in id order. May contain non-finite values, which every
    /// strategy is expected to skip rather than propagate.
    pub logits: &'a [f32],
    /// Applied to a logit before it is ranked.
    pub penalty: &'a dyn LogitPenalty,
    /// `0` means the caller asked for greedy decoding.
    pub temperature: f32,
    /// Keep the `k` most likely tokens; `0` turns the cut off.
    pub top_k: usize,
    /// Keep the shortest head of the distribution whose mass reaches this share.
    pub top_p: f32,
    /// Drop candidates below this share of the best one's probability.
    pub min_p: f32,
}

/// A policy that picks one token from a [`SampleRequest`].
///
/// `Send + Sync` because a build's strategies live in a `static` registry and are shared by every
/// thread that samples; a strategy therefore holds no mutable state, and the generator is passed in
/// per call.
pub trait Sampler: Send + Sync {
    /// The name the registry, the CLI and the documentation use for this policy.
    fn name(&self) -> &'static str;

    /// The chosen token id, or why nothing could be chosen.
    fn draw(&self, request: &SampleRequest<'_>, rng: &mut dyn Random) -> Result<u32, SampleError>;
}

/// The strategies a build has, in the order they are listed.
///
/// A slice rather than a map: there are a handful of strategies, lookup happens once per generation
/// rather than once per token, and a fixed order makes [`Registry::names`] stable for help text.
#[derive(Clone, Copy)]
pub struct Registry(&'static [&'static dyn Sampler]);

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

impl Registry {
    /// Wrap the build's list. The list is a `static` because every entry is a shared unit value.
    pub const fn new(entries: &'static [&'static dyn Sampler]) -> Self {
        Self(entries)
    }

    /// The strategy named `name`, or `None` if this build does not have one.
    pub fn find(&self, name: &str) -> Option<&'static dyn Sampler> {
        self.0.iter().copied().find(|entry| entry.name() == name)
    }

    /// Every name in the registry, in list order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.0.iter().map(|entry| entry.name())
    }

    /// How many strategies the build has.
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the build has no strategies at all, which is a build that cannot sample.
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A strategy that always returns the same id, used by tests that need a third entry.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct Constant(u32);

#[cfg(test)]
impl Sampler for Constant {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn draw(
        &self,
        _request: &SampleRequest<'_>,
        _rng: &mut dyn Random,
    ) -> Result<u32, SampleError> {
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Zeroes;

    impl Random for Zeroes {
        fn next_f64(&mut self) -> f64 {
            0.0
        }
    }

    static ENTRIES: Registry = Registry::new(&[&Constant(7)]);

    #[test]
    fn a_registry_finds_an_entry_by_its_name() {
        let found = ENTRIES.find("constant").expect("the entry is in the list");
        assert_eq!(found.name(), "constant");
        assert_eq!(
            found.draw(
                &SampleRequest {
                    logits: &[1.0],
                    penalty: &NoPenalty,
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                },
                &mut Zeroes,
            ),
            Ok(7)
        );
    }

    #[test]
    fn an_unknown_name_is_not_an_error_but_an_absent_entry() {
        assert!(ENTRIES.find("does-not-exist").is_none());
        assert_eq!(ENTRIES.len(), 1);
        assert!(!ENTRIES.is_empty());
        assert_eq!(ENTRIES.names().collect::<Vec<_>>(), vec!["constant"]);
    }

    #[test]
    fn a_penalty_that_changes_nothing_returns_the_logit() {
        assert_eq!(NoPenalty.apply(3, -1.5), -1.5);
    }
}
