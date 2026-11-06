//! Deterministic token sampling.
//!
//! Every source of randomness comes from [`Rng`], which is seeded only from `SamplerConfig::seed`.
//! Two samplers with the same seed, the same config and the same sequence of calls return the same
//! ids on every machine, which is what makes a failing generation reproducible.
//!
//! Policies apply in a fixed order: repetition penalty, temperature, top-k, top-p, min-p. A
//! temperature of `0` means greedy and skips every other filter, matching the CLI contract.

pub mod rng;

use std::cmp::Ordering;

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

        let mut scored: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .map(|(id, logit)| (id as u32, *logit))
            .collect();
        if scored.is_empty() {
            return Err(ViperaError::invalid_argument(
                "every logit is NaN or infinite",
            ));
        }

        let penalty = self.config.repetition_penalty;
        if penalty != 1.0 {
            for &id in recent {
                if let Some(entry) = scored.iter_mut().find(|(candidate, _)| *candidate == id) {
                    entry.1 = if entry.1 > 0.0 {
                        entry.1 / penalty
                    } else {
                        entry.1 * penalty
                    };
                }
            }
        }

        sort_by_probability(&mut scored);
        if self.config.temperature == 0.0 {
            return Ok(scored[0].0);
        }

        let scale = 1.0 / self.config.temperature;
        let peak = scored[0].1 * scale;
        let mut candidates: Vec<(u32, f32)> = scored
            .iter()
            .map(|(id, logit)| (*id, (logit * scale - peak).exp()))
            .collect();
        let total: f32 = candidates.iter().map(|(_, weight)| weight).sum();
        // `NaN` compares false against everything, so it is caught by the finiteness check.
        if total <= 0.0 || !total.is_finite() {
            return Err(ViperaError::internal(
                "softmax produced no usable probability mass",
            ));
        }
        for (_, weight) in candidates.iter_mut() {
            *weight /= total;
        }

        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        let mut cumulative = 0.0;
        let mut keep = candidates.len();
        for (index, (_, weight)) in candidates.iter().enumerate() {
            cumulative += weight;
            if cumulative >= self.config.top_p {
                keep = index + 1;
                break;
            }
        }
        candidates.truncate(keep);

        let threshold = self.config.min_p * candidates[0].1;
        candidates.retain(|(_, weight)| *weight >= threshold);

        let total: f32 = candidates.iter().map(|(_, weight)| weight).sum();
        let target = self.rng.next_f64() * f64::from(total);
        let mut running = 0.0f64;
        for (id, weight) in &candidates {
            running += f64::from(*weight);
            if target < running {
                return Ok(*id);
            }
        }
        // Only reachable through floating-point rounding at the last candidate.
        Ok(candidates[candidates.len() - 1].0)
    }
}

fn sort_by_probability(scored: &mut [(u32, f32)]) {
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
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
