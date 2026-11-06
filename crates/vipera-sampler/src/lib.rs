//! Deterministic token sampling.
//!
//! Every source of randomness comes from [`Rng`], which is seeded only from `SamplerConfig::seed`.
//! Two samplers with the same seed, the same config and the same sequence of calls return the same
//! ids on every machine, which is what makes a failing generation reproducible.
//!
//! Policies apply in a fixed order: repetition penalty, temperature, top-k, top-p, min-p. A
//! temperature of `0` means greedy and skips every other filter, matching the CLI contract.

#![warn(missing_docs)]

pub mod rng;

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use vipera_errors::ViperaError;

pub use rng::Rng;

/// Sampling knobs. Defaults are the ones the CLI advertises.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// `0` selects greedy decoding.
    pub temperature: f32,
    /// Keep the `k` most likely tokens; `0` disables the cut.
    pub top_k: usize,
    /// Keep the shortest head of the distribution whose mass reaches `top_p`.
    pub top_p: f32,
    /// Drop tokens whose probability is below `min_p` times the best token's.
    pub min_p: f32,
    /// Penalty applied to tokens that already appear in the recent context; `1.0` disables it.
    pub repetition_penalty: f32,
    /// Seed for every random choice in the run.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// Reject values that cannot describe a distribution.
    pub fn validate(&self) -> Result<(), ViperaError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(ViperaError::invalid_argument(
                "temperature must be a finite number at or above 0",
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(ViperaError::invalid_argument(
                "top_p must be greater than 0 and at most 1",
            ));
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 || self.min_p >= 1.0 {
            return Err(ViperaError::invalid_argument(
                "min_p must be at least 0 and below 1",
            ));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(ViperaError::invalid_argument(
                "repetition_penalty must be a finite number above 0",
            ));
        }
        Ok(())
    }
}

/// Stateful sampler: holds the config and the generator.
#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng: Rng,
}

impl Sampler {
    /// Build a sampler, rejecting an invalid config.
    pub fn new(config: SamplerConfig) -> Result<Self, ViperaError> {
        config.validate()?;
        Ok(Self {
            rng: Rng::from_seed(config.seed),
            config,
        })
    }

    /// The config in use.
    pub const fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Restart the random stream at `seed`, keeping the rest of the config.
    pub fn reseed(&mut self, seed: u64) {
        self.config.seed = seed;
        self.rng.reseed(seed);
    }

    /// Pick the next token id.
    ///
    /// `recent` is the context used by the repetition penalty; pass the most recent ids in any
    /// order. Non-finite logits are skipped rather than poisoning the distribution.
    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> Result<u32, ViperaError> {
        if logits.is_empty() {
            return Err(ViperaError::invalid_argument(
                "cannot sample from empty logits",
            ));
        }

        let penalty = RepetitionPenalty::new(recent, self.config.repetition_penalty);

        // Greedy needs one pass and no allocation: the argmax over the penalized logits, ties to
        // the lowest id. Sorting 151936 entries to read the first one is what this avoids.
        if self.config.temperature == 0.0 {
            let mut best: Option<(u32, f32)> = None;
            for (id, logit) in logits.iter().enumerate() {
                if !logit.is_finite() {
                    continue;
                }
                let value = penalty.apply(id, *logit);
                let better = match best {
                    None => true,
                    Some((best_id, best_value)) => {
                        value > best_value || (value == best_value && (id as u32) < best_id)
                    }
                };
                if better {
                    best = Some((id as u32, value));
                }
            }
            let Some((best_id, _)) = best else {
                return Err(ViperaError::invalid_argument(
                    "every logit is NaN or infinite",
                ));
            };
            return Ok(best_id);
        }

        let scale = 1.0 / self.config.temperature;
        // `top_k` of 0 turns the cut off, and top-p alone can still keep a large share of the
        // vocabulary. Take a generous bounded head first and sort every logit only when the cut
        // really falls beyond it: for a peaked distribution the answer is inside a few hundred
        // entries, and sorting 151936 to find that is the whole cost of this stage.
        let uncut = self.config.top_k == 0;
        let bound = if uncut {
            NO_CUT_CANDIDATES
        } else {
            self.config.top_k
        };
        let mut weights = top_candidates(logits, &penalty, bound);
        if weights.is_empty() {
            return Err(ViperaError::invalid_argument(
                "every logit is NaN or infinite",
            ));
        }

        // The softmax denominator is still the mass of every finite logit, in logit order, which is
        // what the earlier version summed after it had allocated and sorted the whole vector. Only
        // the candidates come from the bounded selection now.
        let peak = weights[0].1 * scale;
        let mut total = 0.0_f32;
        for (id, logit) in logits.iter().enumerate() {
            if logit.is_finite() {
                total += (penalty.apply(id, *logit) * scale - peak).exp();
            }
        }
        // `NaN` compares false against everything, so it is caught by the finiteness check.
        if total <= 0.0 || !total.is_finite() {
            return Err(ViperaError::internal(
                "softmax produced no usable probability mass",
            ));
        }

        for (_, weight) in weights.iter_mut() {
            *weight = (*weight * scale - peak).exp();
        }

        // `top_p` is compared against a share of the full mass, which is what the old division by
        // `total` did; keeping the weights unnormalized saves a division per candidate.
        let target_mass = self.config.top_p * total;
        let mut cut = mass_cut(&weights, target_mass);

        // The bounded head did not reach the cut, so the exact answer needs the tail too. This is
        // the only case that still sorts every logit, and it is the only case where it cannot be
        // avoided: top-p is defined over the whole distribution.
        if uncut && cut.is_none() && weights.len() < logits.len() {
            weights = top_candidates(logits, &penalty, 0);
            for (_, weight) in weights.iter_mut() {
                *weight = (*weight * scale - peak).exp();
            }
            cut = mass_cut(&weights, target_mass);
        }

        weights.truncate(cut.unwrap_or(weights.len()));

        let threshold = self.config.min_p * weights[0].1;
        weights.retain(|(_, weight)| *weight >= threshold);

        let mass: f32 = weights.iter().map(|(_, weight)| *weight).sum();
        let target = self.rng.next_f64() * f64::from(mass);
        let mut running = 0.0f64;
        for (id, weight) in &weights {
            running += f64::from(*weight);
            if target < running {
                return Ok(*id);
            }
        }
        // Only reachable through floating-point rounding at the last candidate.
        Ok(weights[weights.len() - 1].0)
    }
}

/// How many of the leading candidates `target` mass needs, or `None` when the whole set falls short.
fn mass_cut(weights: &[(u32, f32)], target: f32) -> Option<usize> {
    let mut cumulative = 0.0;
    for (index, (_, weight)) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= target {
            return Some(index + 1);
        }
    }
    None
}

/// How many candidates the no-cut path keeps before deciding it has to sort the vocabulary.
///
/// 4096 is far more than top-p ever keeps on a language model — the head that reaches 0.95 of the
/// mass is usually under a hundred tokens — while staying small enough that the bounded selection
/// and the sort of the head together cost less than a millisecond.
const NO_CUT_CANDIDATES: usize = 4096;

/// Descending by logit, then ascending by id, so the order is total and reproducible.
fn by_probability(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// One logit with the ordering the sampler ranks by: higher first, ties to the lower id.
///
/// `Ord` rather than a comparator closure because the bounded selection below compares this key
/// once per logit, and a closure passed through a generic comes out cheaper than one through a
/// function pointer.
struct Ranked {
    value: f32,
    id: u32,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .total_cmp(&other.value)
            .then(other.id.cmp(&self.id))
    }
}

/// The `top_k` best finite logits, best first, with the repetition penalty already applied.
///
/// `top_k` of 0 means no cut and falls back to sorting every logit, which is the honest reading of
/// that setting. Otherwise a heap of exactly `k` entries walks the vector once: each logit costs
/// one comparison against the worst kept entry, and an entry leaves the heap only when a better one
/// arrives, so the heap operations are `O(k log n)` in total instead of `O(n)`.
fn top_candidates(logits: &[f32], penalty: &RepetitionPenalty, top_k: usize) -> Vec<(u32, f32)> {
    if top_k == 0 || top_k >= logits.len() {
        let mut all: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .map(|(id, logit)| (id as u32, penalty.apply(id, *logit)))
            .collect();
        all.sort_unstable_by(by_probability);
        return all;
    }

    let mut heap: BinaryHeap<Reverse<Ranked>> = BinaryHeap::with_capacity(top_k + 1);
    for (id, logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let ranked = Ranked {
            value: penalty.apply(id, *logit),
            id: id as u32,
        };
        if heap.len() < top_k {
            heap.push(Reverse(ranked));
        } else if heap.peek().is_some_and(|Reverse(worst)| ranked > *worst) {
            heap.pop();
            heap.push(Reverse(ranked));
        }
    }
    let mut best: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse(ranked)| (ranked.id, ranked.value))
        .collect();
    best.sort_unstable_by(by_probability);
    best
}

/// The repetition penalty, applied in one binary search per logit.
///
/// The obvious loop over `recent` costs a scan of the whole logit vector per recent id, which is
/// tens of scans per token once the context is long. Sorting the handful of recent ids once turns
/// that into a `log n` lookup on the single pass that already visits every logit.
struct RepetitionPenalty {
    /// The penalized ids, sorted and deduplicated; empty when the penalty is disabled.
    ids: Vec<u32>,
    factor: f32,
}

impl RepetitionPenalty {
    fn new(recent: &[u32], factor: f32) -> Self {
        if factor == 1.0 || recent.is_empty() {
            return Self {
                ids: Vec::new(),
                factor,
            };
        }
        let mut ids = recent.to_vec();
        ids.sort_unstable();
        ids.dedup();
        Self { ids, factor }
    }

    fn apply(&self, id: usize, logit: f32) -> f32 {
        if self.ids.is_empty() || self.ids.binary_search(&(id as u32)).is_err() {
            return logit;
        }
        if logit > 0.0 {
            logit / self.factor
        } else {
            logit * self.factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temperature: f32, top_k: usize, top_p: f32, min_p: f32, penalty: f32) -> Sampler {
        Sampler::new(SamplerConfig {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty: penalty,
            seed: 1234,
        })
        .expect("config is valid")
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let mut s = sampler(0.0, 0, 1.0, 0.0, 1.0);
        for _ in 0..50 {
            assert_eq!(s.sample(&[0.1, 5.0, 0.2], &[]).unwrap(), 1);
        }
    }

    #[test]
    fn greedy_breaks_ties_towards_the_lowest_id() {
        let mut s = sampler(0.0, 0, 1.0, 0.0, 1.0);
        assert_eq!(s.sample(&[2.0, 2.0, 2.0], &[]).unwrap(), 0);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut s = sampler(1.0, 1, 1.0, 0.0, 1.0);
        for _ in 0..50 {
            assert_eq!(s.sample(&[0.3, 0.9, 0.6], &[]).unwrap(), 1);
        }
    }

    #[test]
    fn top_p_keeps_only_the_head_of_the_distribution() {
        let mut s = sampler(1.0, 0, 0.5, 0.0, 1.0);
        for _ in 0..200 {
            assert_eq!(s.sample(&[10.0, 0.0, 0.0], &[]).unwrap(), 0);
        }
    }

    #[test]
    fn min_p_drops_the_tail() {
        // The second token keeps roughly 0.47 of the mass, so a 0.95 floor removes it.
        let mut s = sampler(1.0, 0, 1.0, 0.95, 1.0);
        for _ in 0..200 {
            assert_eq!(s.sample(&[5.0, 4.9, -5.0], &[]).unwrap(), 0);
        }
    }

    #[test]
    fn repetition_penalty_pushes_a_repeated_token_down() {
        let mut s = sampler(0.0, 0, 1.0, 0.0, 2.0);
        assert_eq!(s.sample(&[5.0, 4.0], &[]).unwrap(), 0);
        assert_eq!(s.sample(&[5.0, 4.0], &[0]).unwrap(), 1);
    }

    #[test]
    fn a_long_logit_vector_is_cut_to_its_head() {
        // 200000 entries is the size of a real vocabulary, and the size at which selecting the
        // candidates costs less than sorting all of them.
        let mut logits = vec![-10.0_f32; 200_000];
        logits[123_456] = 8.0;
        logits[7] = 7.0;

        let mut ranked = sampler(1.0, 2, 1.0, 0.0, 1.0);
        for _ in 0..100 {
            assert!(matches!(ranked.sample(&logits, &[]).unwrap(), 123_456 | 7));
        }

        // Greedy reads the same maximum in one pass, with no vector and no sort.
        let mut greedy = sampler(0.0, 0, 1.0, 0.0, 1.0);
        assert_eq!(greedy.sample(&logits, &[]).unwrap(), 123_456);
    }

    #[test]
    fn without_top_k_top_p_cuts_the_head_of_a_long_vector() {
        let mut logits = vec![-20.0_f32; 200_000];
        logits[1] = 10.0;
        logits[199_999] = 9.9;
        let mut s = sampler(1.0, 0, 0.5, 0.0, 1.0);
        for _ in 0..100 {
            assert_eq!(s.sample(&logits, &[]).unwrap(), 1);
        }
    }

    #[test]
    fn without_top_k_a_flat_distribution_reaches_past_the_bounded_head() {
        // Equal logits put half the mass in the first 100000 ids, far beyond the 4096 candidates the
        // bounded head holds, so the whole vocabulary has to be ordered before the cut.
        let logits = vec![0.0_f32; 200_000];
        let mut s = sampler(1.0, 0, 0.5, 0.0, 1.0);
        for _ in 0..50 {
            let id = s.sample(&logits, &[]).unwrap();
            assert!(id < 100_000, "id {id} came from beyond the top-p half");
        }
    }

    #[test]
    fn non_finite_logits_are_skipped() {
        let mut s = sampler(0.0, 0, 1.0, 0.0, 1.0);
        assert_eq!(s.sample(&[f32::NAN, 1.0, f32::INFINITY], &[]).unwrap(), 1);
    }

    #[test]
    fn empty_or_unusable_logits_are_rejected() {
        let mut s = sampler(1.0, 0, 1.0, 0.0, 1.0);
        assert!(s.sample(&[], &[]).is_err());
        assert!(s.sample(&[f32::NAN, f32::NEG_INFINITY], &[]).is_err());
    }

    #[test]
    fn the_same_seed_replays_the_same_run() {
        let mut a = sampler(0.9, 40, 0.95, 0.0, 1.0);
        let mut b = sampler(0.9, 40, 0.95, 0.0, 1.0);
        let logits = [0.5, 0.4, 0.3, 0.2, 0.1];
        for _ in 0..64 {
            assert_eq!(
                a.sample(&logits, &[]).unwrap(),
                b.sample(&logits, &[]).unwrap()
            );
        }
    }

    #[test]
    fn a_different_seed_produces_a_different_run() {
        let mut a = sampler(1.0, 0, 1.0, 0.0, 1.0);
        let mut b = sampler(1.0, 0, 1.0, 0.0, 1.0);
        b.reseed(4321);
        let logits = [0.0, 0.0, 0.0];
        let run_a: Vec<u32> = (0..40).map(|_| a.sample(&logits, &[]).unwrap()).collect();
        let run_b: Vec<u32> = (0..40).map(|_| b.sample(&logits, &[]).unwrap()).collect();
        assert_ne!(run_a, run_b);
    }

    #[test]
    fn sampling_follows_the_distribution() {
        // log(3) versus 0 gives 0.75 / 0.25 at temperature 1.
        let mut s = sampler(1.0, 0, 1.0, 0.0, 1.0);
        let logits = [3.0f32.ln(), 0.0];
        let draws = 4_000;
        let first = (0..draws)
            .filter(|_| s.sample(&logits, &[]).unwrap() == 0)
            .count();
        let ratio = first as f32 / draws as f32;
        assert!((ratio - 0.75).abs() < 0.05, "ratio was {ratio}");
    }

    #[test]
    fn an_invalid_config_is_rejected() {
        for config in [
            SamplerConfig {
                temperature: -0.1,
                ..SamplerConfig::default()
            },
            SamplerConfig {
                temperature: f32::NAN,
                ..SamplerConfig::default()
            },
            SamplerConfig {
                top_p: 0.0,
                ..SamplerConfig::default()
            },
            SamplerConfig {
                top_p: 1.5,
                ..SamplerConfig::default()
            },
            SamplerConfig {
                min_p: 1.0,
                ..SamplerConfig::default()
            },
            SamplerConfig {
                repetition_penalty: 0.0,
                ..SamplerConfig::default()
            },
        ] {
            assert!(Sampler::new(config).is_err(), "accepted {config:?}");
        }
        assert!(Sampler::new(SamplerConfig::default()).is_ok());
    }
}
