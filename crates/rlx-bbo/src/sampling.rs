// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Low-discrepancy / space-filling samplers for design-of-experiments.
//!
//! Used to warm-start Bayesian optimisation; also useful for any sweep
//! where independent uniform samples leave clumps and voids.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Latin Hypercube Sampling: each of `d` dimensions is partitioned into
/// `n` equal intervals; one sample falls in each interval. The samples
/// returned are normalised to the unit cube `[0, 1]^d`. Apply your own
/// bounds by mapping `x_i ← lo_i + (hi_i − lo_i)·u_i`.
pub fn lhs(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Build per-dimension permutations of [0..n).
    let mut perms: Vec<Vec<usize>> = (0..d)
        .map(|_| {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                p.swap(i, j);
            }
            p
        })
        .collect();
    let inv_n = 1.0 / (n as f64);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = Vec::with_capacity(d);
        for dim in 0..d {
            let u: f64 = rng.gen_range(0.0..1.0);
            let stratum = perms[dim][i] as f64;
            x.push((stratum + u) * inv_n);
        }
        samples.push(x);
    }
    // suppress unused-warning if d==0
    let _ = &mut perms;
    samples
}

/// First `n` Halton points in `d` dimensions, skipping the first `skip`
/// (skipping at least a few hundred is standard practice to avoid
/// pathological correlations in low-prime bases).
pub fn halton(n: usize, d: usize, skip: usize) -> Vec<Vec<f64>> {
    let primes = first_primes(d.max(1));
    (0..n)
        .map(|i| {
            primes
                .iter()
                .take(d)
                .map(|&b| van_der_corput(i + skip + 1, b))
                .collect()
        })
        .collect()
}

/// Sobol sequence over the unit cube — currently delegates to [`halton`]
/// while the Joe & Kuo direction-number table is being filled in. Both
/// are low-discrepancy and either is a fine warm-start for BO.
pub fn sobol(n: usize, d: usize, skip: usize) -> Vec<Vec<f64>> {
    halton(n, d, skip)
}

fn van_der_corput(mut i: usize, base: usize) -> f64 {
    let mut f = 1.0 / base as f64;
    let mut r = 0.0;
    while i > 0 {
        r += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    r
}

fn first_primes(n: usize) -> Vec<usize> {
    let table = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113,
    ];
    table.iter().copied().take(n.max(1)).collect()
}

// Reserved for future use — Joe & Kuo direction-number table when sobol
// gets its own implementation (currently delegates to halton).
#[allow(dead_code)]
const SOBOL_DIRECTION_NUMBERS: &[(u32, u32, &[u32])] = &[
    (1, 0, &[]),
    (2, 1, &[1]),
    (3, 1, &[1, 3]),
    (3, 2, &[1, 3, 1]),
    (4, 1, &[1, 1, 1]),
    (4, 4, &[1, 1, 3, 3]),
    (5, 2, &[1, 3, 5, 13]),
    (5, 4, &[1, 1, 5, 5]),
    (5, 7, &[1, 3, 5, 5]),
    (5, 11, &[1, 1, 7, 11]),
    (5, 13, &[1, 1, 5, 7]),
    (5, 14, &[1, 3, 7, 11]),
    (6, 1, &[1, 3, 3, 5, 1]),
    (6, 13, &[1, 1, 3, 5, 5]),
    (6, 16, &[1, 1, 7, 13, 25]),
    (6, 19, &[1, 1, 1, 7, 9]),
];

#[allow(dead_code)]
fn sobol_directions(d: usize) -> Vec<[u64; 32]> {
    let mut dir: Vec<[u64; 32]> = Vec::with_capacity(d);
    // Dimension 0 — special case: m_i = 1 for all i.
    {
        let mut col = [0u64; 32];
        for i in 0..32 {
            col[i] = 1u64 << (31 - i);
        }
        dir.push(col);
    }
    for k in 1..d {
        let (deg, a, m_init) = SOBOL_DIRECTION_NUMBERS[k];
        let mut m: Vec<u64> = m_init.iter().map(|&v| v as u64).collect();
        // Extend m via recurrence m_i = 2^1 a_1 m_{i-1} XOR ... XOR 2^deg m_{i-deg} XOR m_{i-deg}.
        while m.len() < 32 {
            let i = m.len();
            let mut v = m[i - deg as usize];
            v ^= v << deg;
            for j in 1..(deg as usize) {
                let bit = (a >> (deg as usize - 1 - j)) & 1;
                if bit == 1 {
                    v ^= (1u64 << j) * m[i - j];
                }
            }
            m.push(v);
        }
        let mut col = [0u64; 32];
        for i in 0..32 {
            col[i] = m[i] << (31 - i);
        }
        dir.push(col);
    }
    dir
}

#[allow(dead_code)]
fn lowest_zero_bit(i: u64) -> usize {
    let mut c = 0usize;
    let mut v = i;
    while v & 1 != 0 {
        v >>= 1;
        c += 1;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lhs_one_per_stratum() {
        let n = 20;
        let d = 3;
        let pts = lhs(n, d, 42);
        for dim in 0..d {
            let mut bins = vec![0usize; n];
            for p in &pts {
                let b = ((p[dim] * n as f64).floor() as usize).min(n - 1);
                bins[b] += 1;
            }
            for &b in &bins {
                assert_eq!(b, 1, "LHS stratum violation at dim {dim}");
            }
        }
    }

    #[test]
    fn halton_in_unit_cube() {
        let pts = halton(64, 5, 100);
        for p in &pts {
            for &v in p {
                assert!(v >= 0.0 && v < 1.0);
            }
        }
    }

    #[test]
    fn sobol_in_unit_cube() {
        let pts = sobol(64, 3, 0);
        for p in &pts {
            for &v in p {
                assert!(v >= 0.0 && v <= 1.0);
            }
        }
    }
}
