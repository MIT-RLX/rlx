// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SplitMix64 (Steele, Lea & Flood 2014; Vigna's reference `splitmix64.c`).
//!
//! Not a numpy/torch stream — it is the small, seedable 64-bit generator that ports
//! reach for to make deterministic synthetic data, shuffles and bootstrap draws.

/// Advance `state` by the golden-ratio increment and return the mixed output.
///
/// Matches Vigna's reference implementation: from `state = 0` the first outputs are
/// `0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f`.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_implementation() {
        let mut s = 0u64;
        let got: Vec<u64> = (0..5).map(|_| splitmix64(&mut s)).collect();
        assert_eq!(
            got,
            [
                0xe220_a839_7b1d_cdaf,
                0x6e78_9e6a_a1b9_65f4,
                0x06c4_5d18_8009_454f,
                0xf88b_b8a8_724c_81ec,
                0x1b39_896a_51a8_749b,
            ]
        );
    }
}
