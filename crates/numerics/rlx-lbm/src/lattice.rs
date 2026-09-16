// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! D2Q9 and D3Q27 lattice constants and Hermite tensors.
//!
//! Both lattices have `cs² = 1/3`. The Hermite tensors are written from their
//! definitions rather than tabulated, so a new lattice needs no new algebra:
//!
//! ```text
//! H²_αβ(c)   = c_α c_β − cs² δ_αβ
//! H³_αβγ(c)  = c_α c_β c_γ − cs² (c_α δ_βγ + c_β δ_αγ + c_γ δ_αβ)
//! ```

/// Speed of sound squared. Same for D2Q9 and D3Q27.
pub const CS2: f64 = 1.0 / 3.0;

/// D2Q9 lattice velocities, in the rest-first ordering used throughout.
pub const D2Q9_C: [[i32; 2]; 9] = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
];

/// D2Q9 weights, aligned with [`D2Q9_C`].
pub const D2Q9_W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Index of the opposite direction, aligned with [`D2Q9_C`].
pub const D2Q9_OPP: [usize; 9] = [0, 2, 1, 4, 3, 7, 8, 5, 6];

/// D3Q27 lattice velocities.
pub const D3Q27_C: [[i32; 3]; 27] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, -1],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, -1],
    [0, -1, 1],
    [1, 1, 1],
    [-1, -1, -1],
    [1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
];

/// D3Q27 weights, aligned with [`D3Q27_C`].
pub const D3Q27_W: [f64; 27] = {
    let mut w = [0.0; 27];
    w[0] = 8.0 / 27.0;
    let mut i = 1;
    while i < 7 {
        w[i] = 2.0 / 27.0;
        i += 1;
    }
    while i < 19 {
        w[i] = 1.0 / 54.0;
        i += 1;
    }
    while i < 27 {
        w[i] = 1.0 / 216.0;
        i += 1;
    }
    w
};

/// Second-order Hermite tensor component `H²_αβ(c)` for integer lattice
/// velocity `c` and axis indices `a`, `b`.
#[inline]
pub fn h2(c: &[i32], a: usize, b: usize) -> f64 {
    let d = if a == b { CS2 } else { 0.0 };
    (c[a] * c[b]) as f64 - d
}

/// Third-order Hermite tensor component `H³_αβγ(c)`.
#[inline]
pub fn h3(c: &[i32], a: usize, b: usize, g: usize) -> f64 {
    let dbg = if b == g { 1.0 } else { 0.0 };
    let dag = if a == g { 1.0 } else { 0.0 };
    let dab = if a == b { 1.0 } else { 0.0 };
    (c[a] * c[b] * c[g]) as f64 - CS2 * (c[a] as f64 * dbg + c[b] as f64 * dag + c[g] as f64 * dab)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights must sum to one and the lattice must be symmetric — the two
    /// facts every isotropy proof rests on.
    #[test]
    fn lattices_are_normalized_and_symmetric() {
        let s2: f64 = D2Q9_W.iter().sum();
        assert!((s2 - 1.0).abs() < 1e-15, "D2Q9 weights sum to {s2}");
        let s3: f64 = D3Q27_W.iter().sum();
        assert!((s3 - 1.0).abs() < 1e-15, "D3Q27 weights sum to {s3}");

        for c in D2Q9_C {
            let neg = [-c[0], -c[1]];
            assert!(D2Q9_C.contains(&neg), "D2Q9 missing {neg:?}");
        }
        for c in D3Q27_C {
            let neg = [-c[0], -c[1], -c[2]];
            assert!(D3Q27_C.contains(&neg), "D3Q27 missing {neg:?}");
        }
    }

    /// `D2Q9_OPP` must actually point at the negated velocity.
    #[test]
    fn opposite_table_is_correct() {
        for (i, c) in D2Q9_C.iter().enumerate() {
            let o = D2Q9_OPP[i];
            assert_eq!(D2Q9_C[o], [-c[0], -c[1]], "opposite of dir {i}");
        }
    }

    /// The lattice must reproduce the Gaussian moments up to fourth order —
    /// this is what makes the Hermite expansion exact on it, and it is the
    /// precondition for every reconstruction formula in [`crate::moment`].
    #[test]
    fn lattice_reproduces_gaussian_moments() {
        // Σ w = 1, Σ w c_a = 0, Σ w c_a c_b = cs² δ, Σ w c_a c_b c_g = 0.
        for a in 0..2 {
            let m1: f64 = D2Q9_W
                .iter()
                .zip(D2Q9_C.iter())
                .map(|(w, c)| w * c[a] as f64)
                .sum();
            assert!(m1.abs() < 1e-15);
            for b in 0..2 {
                let m2: f64 = D2Q9_W
                    .iter()
                    .zip(D2Q9_C.iter())
                    .map(|(w, c)| w * (c[a] * c[b]) as f64)
                    .sum();
                let want = if a == b { CS2 } else { 0.0 };
                assert!((m2 - want).abs() < 1e-15, "D2Q9 second moment {a}{b}");
                for g in 0..2 {
                    let m3: f64 = D2Q9_W
                        .iter()
                        .zip(D2Q9_C.iter())
                        .map(|(w, c)| w * (c[a] * c[b] * c[g]) as f64)
                        .sum();
                    assert!(m3.abs() < 1e-15, "D2Q9 third moment {a}{b}{g}");
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                let m2: f64 = D3Q27_W
                    .iter()
                    .zip(D3Q27_C.iter())
                    .map(|(w, c)| w * (c[a] * c[b]) as f64)
                    .sum();
                let want = if a == b { CS2 } else { 0.0 };
                assert!((m2 - want).abs() < 1e-15, "D3Q27 second moment {a}{b}");
            }
        }
    }
}
