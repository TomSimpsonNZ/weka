//! Faithful port of Triangle's `randomnation()` linear congruential generator
//! (triangle.cpp:6684). Reproducing this RNG exactly — together with identical
//! seeding — is what keeps randomized point-location sampling and quality-meshing
//! Steiner-point insertion order aligned with the C library.

/// Triangle's linear congruential generator.
///
/// `randomseed` is initialized to `1` in Triangle's `triangleinit()`, so [`Rng::new`]
/// matches the C default. Use [`Rng::with_seed`] to reproduce a specific run.
#[derive(Debug, Clone)]
pub struct Rng {
    seed: u64,
}

impl Default for Rng {
    fn default() -> Self {
        Self::new()
    }
}

impl Rng {
    /// Matches Triangle's default (`randomseed = 1`).
    pub fn new() -> Self {
        Self { seed: 1 }
    }

    /// Seed the generator explicitly (for reproducible meshes / tests).
    pub fn with_seed(seed: u64) -> Self {
        // Keep the seed in the generator's residue range, mirroring the modulus.
        Self {
            seed: seed % 714025,
        }
    }

    /// The raw internal seed (exposed for cross-checking against the C generator).
    pub fn raw_seed(&self) -> u64 {
        self.seed
    }

    /// Generate a random number in `0..choices`.
    ///
    /// Verbatim port of:
    /// ```c
    /// randomseed = (randomseed * 1366l + 150889l) % 714025l;
    /// return randomseed / (714025l / choices + 1);
    /// ```
    ///
    /// # Panics
    /// Panics if `choices == 0` (matching the C division-by-zero, but defined).
    pub fn randomnation(&mut self, choices: u32) -> u64 {
        assert!(choices > 0, "randomnation requires choices > 0");
        // randomseed < 714025, so randomseed*1366 < 1e9 — no overflow in u64.
        self.seed = (self.seed * 1366 + 150889) % 714025;
        self.seed / (714025 / choices as u64 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_c_seed_sequence() {
        // Starting from the C default seed of 1, the internal `randomseed`
        // recurrence produces this deterministic sequence (hand-derived from the
        // exact recurrence in triangle.cpp:6692).
        let mut rng = Rng::new();
        let mut seeds = Vec::new();
        for _ in 0..3 {
            rng.randomnation(1000);
            seeds.push(rng.raw_seed());
        }
        assert_eq!(seeds, vec![152255, 349944, 491668]);
    }

    #[test]
    fn output_is_bounded_and_deterministic() {
        let mut a = Rng::new();
        let mut b = Rng::new();
        for choices in [2u32, 7, 100, 4096] {
            for _ in 0..1000 {
                let x = a.randomnation(choices);
                let y = b.randomnation(choices);
                assert_eq!(x, y, "RNG must be deterministic");
                assert!(x < choices as u64, "value {x} out of range for {choices}");
            }
        }
    }
}
