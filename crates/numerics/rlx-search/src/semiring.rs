// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two semirings sequence search runs over, on log-domain `f32` weights.
//!
//! A trellis recursion written once against [`Semiring`] computes different
//! quantities depending on the semiring: over [`Log`] the CTC/HMM forward
//! pass gives the total log-likelihood of all paths, over [`Tropical`] the
//! same recursion gives the Viterbi (best single path) score. That is the
//! OpenFST/k2 view of decoding, kept to the two instances decoders use.

use crate::select::log_add;

/// A commutative semiring on `f32` log-weights.
pub trait Semiring {
    /// Identity of [`Semiring::plus`]: the weight of no path.
    const ZERO: f32 = f32::NEG_INFINITY;
    /// Identity of [`Semiring::times`]: the weight of the empty path.
    const ONE: f32 = 0.0;
    /// Combine alternative paths.
    fn plus(a: f32, b: f32) -> f32;
    /// Extend a path by an arc.
    #[inline]
    fn times(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// `(log-add, +)`: sums path probabilities. Forward algorithm, CTC loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Log;

impl Semiring for Log {
    #[inline]
    fn plus(a: f32, b: f32) -> f32 {
        log_add(a, b)
    }
}

/// `(max, +)`: keeps the best path. Viterbi, forced alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tropical;

impl Semiring for Tropical {
    #[inline]
    fn plus(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_hold() {
        for x in [-3.5f32, 0.0, 2.0] {
            assert_eq!(Log::plus(x, Log::ZERO), x);
            assert_eq!(Tropical::plus(x, Tropical::ZERO), x);
            assert_eq!(Log::times(x, Log::ONE), x);
        }
    }

    #[test]
    fn log_plus_dominates_tropical_plus() {
        let (a, b) = (-1.0f32, -1.5);
        assert!(Log::plus(a, b) > Tropical::plus(a, b));
    }
}
