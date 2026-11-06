//! Small deterministic generator, so a seed fully determines a run without pulling in `rand`.

use vipera_errors::ViperaError;

/// Xorshift64* generator, seeded through SplitMix64 so that neighbouring seeds diverge.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Build a generator from a user-visible seed.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            state: splitmix64(seed ^ 0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Restore the generator to `seed`.
    pub fn reseed(&mut self, seed: u64) {
        self.state = splitmix64(seed ^ 0x9E37_79B9_7F4A_7C15);
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 bits is the full precision of an f64 mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, bound)`.
    pub fn next_bounded(&mut self, bound: u64) -> Result<u64, ViperaError> {
        if bound == 0 {
            return Err(ViperaError::invalid_argument(
                "cannot draw from an empty range",
            ));
        }
        Ok(self.next_u64() % bound)
    }
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let mut a = Rng::from_seed(7);
        let mut b = Rng::from_seed(7);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn neighbouring_seeds_do_not_collide() {
        let mut a = Rng::from_seed(1);
        let mut b = Rng::from_seed(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn floats_stay_inside_the_unit_interval() {
        let mut rng = Rng::from_seed(42);
        for _ in 0..10_000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "out of range: {value}");
        }
    }

    #[test]
    fn bounded_draws_respect_the_bound() {
        let mut rng = Rng::from_seed(3);
        for _ in 0..1_000 {
            assert!(rng.next_bounded(5).unwrap() < 5);
        }
        assert!(rng.next_bounded(0).is_err());
    }

    #[test]
    fn reseeding_replays_the_stream() {
        let mut rng = Rng::from_seed(11);
        let first: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        rng.reseed(11);
        let second: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn zero_seed_still_produces_a_stream() {
        let mut rng = Rng::from_seed(0);
        let values: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert!(values.iter().any(|&v| v != 0));
    }
}
