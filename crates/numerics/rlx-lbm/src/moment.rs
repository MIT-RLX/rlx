// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Moment encoding: store `(ρ, u, S)` and reconstruct the populations in the
//! kernel, instead of storing the populations.
//!
//! This is the HOME-LBM idea (Li et al., *ACM TOG* 42(6), 2023). A D3Q27 solver
//! that stores populations needs ≥58 values per node; storing the first three
//! velocity moments needs **10** (×2 for double buffering). The populations
//! never exist in memory — they are rebuilt from the moments each time a
//! neighbour is read, so streaming becomes a gather of 10 values rather than 27.
//!
//! Same shape as [`rlx_ir::Op::SynthMatMul`] and [`rlx_ir::Op::DequantMatMul`]:
//! keep a compact representation and expand it inside the loop, paying ALU to
//! avoid DRAM.
//!
//! # The reconstruction must be third order
//!
//! Truncating the Hermite series at second order (`f_i = ρw_i(1 + c·u/cs² +
//! H²:S/2cs⁴)`) is the earlier MR-LBM scheme, and it is the reason MR-LBM cannot
//! hold `Re > 4000`. [`reconstruct_d2q9`] and [`reconstruct_d3q27`] therefore
//! carry the third-order terms (paper Eq. 29 and Eq. 17).
//!
//! # The invariant that keeps this honest
//!
//! Reconstruction is only meaningful if it *round-trips*: taking the moments of
//! the reconstructed populations must return the moments you started from.
//!
//! ```text
//! Σ f_i = ρ,   Σ f_i c = ρu,   Σ f_i c_α c_β = ρ(S_αβ + cs² δ_αβ)
//! ```
//!
//! Nothing else checks this. A reconstruction can be wrong in a way that is
//! perfectly self-consistent across every backend that implements it — and a
//! wrong coefficient here shows up as an incorrect effective viscosity, i.e. as
//! plausible-looking flow, not as a crash. [`crate::invariant`] asserts it.
//!
//! The round trip pins how the formula is *assembled*; it is deliberately blind
//! to truncation order, since the third-order tensors are orthogonal to the
//! first three moments. Truncation order is covered by the Taylor–Green decay
//! test instead. See [`crate::invariant`] for the split.

use crate::lattice::{CS2, D2Q9_C, D2Q9_W, D3Q27_C, D3Q27_W, h2, h3};

/// The stored state of one D2Q9 node: 6 values instead of 9 populations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Moments2d {
    /// Density.
    pub rho: f64,
    /// Velocity `(ux, uy)`.
    pub u: [f64; 2],
    /// Second-moment tensor `S_αβ = Π_αβ/ρ − cs² δ_αβ`, as `[Sxx, Syy, Sxy]`.
    ///
    /// Note this is the **full** second moment minus `cs²`, so at equilibrium
    /// `S = u ⊗ u` — not the off-equilibrium part alone. Mixing the two
    /// conventions is the single easiest way to get a reconstruction that looks
    /// right and silently rescales the viscous stress.
    pub s: [f64; 3],
}

impl Moments2d {
    /// Equilibrium state at the given density and velocity: `S = u ⊗ u`.
    pub fn equilibrium(rho: f64, u: [f64; 2]) -> Self {
        Self {
            rho,
            u,
            s: [u[0] * u[0], u[1] * u[1], u[0] * u[1]],
        }
    }
}

/// The stored state of one D3Q27 node: 10 values instead of 27 populations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Moments3d {
    /// Density.
    pub rho: f64,
    /// Velocity `(ux, uy, uz)`.
    pub u: [f64; 3],
    /// `[Sxx, Syy, Szz, Sxy, Sxz, Syz]`, same convention as [`Moments2d::s`].
    pub s: [f64; 6],
}

impl Moments3d {
    /// Equilibrium state: `S = u ⊗ u`.
    pub fn equilibrium(rho: f64, u: [f64; 3]) -> Self {
        Self {
            rho,
            u,
            s: [
                u[0] * u[0],
                u[1] * u[1],
                u[2] * u[2],
                u[0] * u[1],
                u[0] * u[2],
                u[1] * u[2],
            ],
        }
    }
}

/// Third-order Hermite reconstruction of the nine D2Q9 populations (Eq. 29).
///
/// ```text
/// f_i = ρ w_i [ 1 + (c_i·u)/cs² + H²(c_i):S / (2cs⁴)
///             + (1/2cs⁶)( H³_xxy(c_i)·A_xxy + H³_xyy(c_i)·A_xyy ) ]
/// A_xxy = S_xx u_y + 2 S_xy u_x − 2 u_x u_x u_y
/// A_xyy = S_yy u_x + 2 S_xy u_y − 2 u_x u_y u_y
/// ```
///
/// The `A` terms are the third-order Hermite coefficients expressed in the
/// stored moments; the factor 3 on each off-diagonal index permutation is
/// folded into the coefficient below.
pub fn reconstruct_d2q9(m: &Moments2d) -> [f64; 9] {
    let (ux, uy) = (m.u[0], m.u[1]);
    let (sxx, syy, sxy) = (m.s[0], m.s[1], m.s[2]);

    let a_xxy = sxx * uy + 2.0 * sxy * ux - 2.0 * ux * ux * uy;
    let a_xyy = syy * ux + 2.0 * sxy * uy - 2.0 * ux * uy * uy;

    let inv_cs2 = 1.0 / CS2;
    let inv_2cs4 = 1.0 / (2.0 * CS2 * CS2);
    // The xxy / xyy index groups each occur 3 times in the symmetric contraction
    // (xxy, xyx, yxx), so the 1/(2cs⁶) prefactor carries a factor of 3.
    let inv_2cs6 = 3.0 / (2.0 * CS2 * CS2 * CS2);

    let mut f = [0.0; 9];
    for i in 0..9 {
        let c = &D2Q9_C[i];
        let cu = (c[0] as f64) * ux + (c[1] as f64) * uy;
        let hs = h2(c, 0, 0) * sxx + h2(c, 1, 1) * syy + 2.0 * h2(c, 0, 1) * sxy;
        let h3s = h3(c, 0, 0, 1) * a_xxy + h3(c, 0, 1, 1) * a_xyy;
        f[i] = m.rho * D2Q9_W[i] * (1.0 + cu * inv_cs2 + hs * inv_2cs4 + h3s * inv_2cs6);
    }
    f
}

/// Third-order Hermite reconstruction of the 27 D3Q27 populations (Eq. 17).
///
/// Seven third-order index groups: the six `ααβ` ones (each with multiplicity 3
/// in the symmetric contraction) and `xyz` (multiplicity 6).
pub fn reconstruct_d3q27(m: &Moments3d) -> [f64; 27] {
    let (ux, uy, uz) = (m.u[0], m.u[1], m.u[2]);
    let (sxx, syy, szz) = (m.s[0], m.s[1], m.s[2]);
    let (sxy, sxz, syz) = (m.s[3], m.s[4], m.s[5]);

    let a_xxy = sxx * uy + 2.0 * sxy * ux - 2.0 * ux * ux * uy;
    let a_xyy = syy * ux + 2.0 * sxy * uy - 2.0 * ux * uy * uy;
    let a_xxz = sxx * uz + 2.0 * sxz * ux - 2.0 * ux * ux * uz;
    let a_xzz = szz * ux + 2.0 * sxz * uz - 2.0 * ux * uz * uz;
    let a_yyz = syy * uz + 2.0 * syz * uy - 2.0 * uy * uy * uz;
    let a_yzz = szz * uy + 2.0 * syz * uz - 2.0 * uy * uz * uz;
    let a_xyz = sxy * uz + syz * ux + sxz * uy - 2.0 * ux * uy * uz;

    let inv_cs2 = 1.0 / CS2;
    let inv_2cs4 = 1.0 / (2.0 * CS2 * CS2);
    let cs6 = CS2 * CS2 * CS2;
    // Multiplicity 3 for the ααβ groups, 6 for xyz.
    let k3 = 3.0 / (2.0 * cs6);
    let k6 = 6.0 / (2.0 * cs6);

    let mut f = [0.0; 27];
    for i in 0..27 {
        let c = &D3Q27_C[i];
        let cu = (c[0] as f64) * ux + (c[1] as f64) * uy + (c[2] as f64) * uz;
        let hs = h2(c, 0, 0) * sxx
            + h2(c, 1, 1) * syy
            + h2(c, 2, 2) * szz
            + 2.0 * (h2(c, 0, 1) * sxy + h2(c, 0, 2) * sxz + h2(c, 1, 2) * syz);
        let h3s = k3
            * (h3(c, 0, 0, 1) * a_xxy
                + h3(c, 0, 1, 1) * a_xyy
                + h3(c, 0, 0, 2) * a_xxz
                + h3(c, 0, 2, 2) * a_xzz
                + h3(c, 1, 1, 2) * a_yyz
                + h3(c, 1, 2, 2) * a_yzz)
            + k6 * h3(c, 0, 1, 2) * a_xyz;
        f[i] = m.rho * D3Q27_W[i] * (1.0 + cu * inv_cs2 + hs * inv_2cs4 + h3s);
    }
    f
}

/// Take the moments of a set of D2Q9 populations — the inverse of
/// [`reconstruct_d2q9`].
pub fn moments_d2q9(f: &[f64; 9]) -> Moments2d {
    let rho: f64 = f.iter().sum();
    let mut mom = [0.0; 2];
    let mut pi = [0.0; 3]; // xx, yy, xy
    for i in 0..9 {
        let c = &D2Q9_C[i];
        mom[0] += f[i] * c[0] as f64;
        mom[1] += f[i] * c[1] as f64;
        pi[0] += f[i] * (c[0] * c[0]) as f64;
        pi[1] += f[i] * (c[1] * c[1]) as f64;
        pi[2] += f[i] * (c[0] * c[1]) as f64;
    }
    let inv = 1.0 / rho;
    Moments2d {
        rho,
        u: [mom[0] * inv, mom[1] * inv],
        s: [pi[0] * inv - CS2, pi[1] * inv - CS2, pi[2] * inv],
    }
}

/// Take the moments of a set of D3Q27 populations.
pub fn moments_d3q27(f: &[f64; 27]) -> Moments3d {
    let rho: f64 = f.iter().sum();
    let mut mom = [0.0; 3];
    let mut pi = [0.0; 6]; // xx, yy, zz, xy, xz, yz
    for i in 0..27 {
        let c = &D3Q27_C[i];
        for a in 0..3 {
            mom[a] += f[i] * c[a] as f64;
        }
        pi[0] += f[i] * (c[0] * c[0]) as f64;
        pi[1] += f[i] * (c[1] * c[1]) as f64;
        pi[2] += f[i] * (c[2] * c[2]) as f64;
        pi[3] += f[i] * (c[0] * c[1]) as f64;
        pi[4] += f[i] * (c[0] * c[2]) as f64;
        pi[5] += f[i] * (c[1] * c[2]) as f64;
    }
    let inv = 1.0 / rho;
    Moments3d {
        rho,
        u: [mom[0] * inv, mom[1] * inv, mom[2] * inv],
        s: [
            pi[0] * inv - CS2,
            pi[1] * inv - CS2,
            pi[2] * inv - CS2,
            pi[3] * inv,
            pi[4] * inv,
            pi[5] * inv,
        ],
    }
}

/// Relaxation rate from kinematic viscosity: `Ω = 1/τ`, `τ = 3ν + 1/2`.
pub fn omega(viscosity: f64) -> f64 {
    1.0 / (3.0 * viscosity + 0.5)
}

/// D2Q9 central-moment collision in closed form (paper App. C), with an
/// optional body force `F`.
///
/// Operating directly on the stored moments is the second half of the speedup:
/// no transform matrix `M`, no inverse — just algebra on six numbers.
pub fn collide_d2q9(m: &Moments2d, viscosity: f64, force: [f64; 2]) -> Moments2d {
    let om = omega(viscosity);
    let rho = m.rho;
    let (fx, fy) = (force[0], force[1]);
    // Post-streaming velocity already carries the half-force correction.
    let ux = m.u[0];
    let uy = m.u[1];
    let (sxx, syy, sxy) = (m.s[0], m.s[1], m.s[2]);

    // Work in Π = ρ(S + cs²δ) so the force terms read as in the paper, then
    // convert back at the end.
    let pixx = rho * (sxx + CS2);
    let piyy = rho * (syy + CS2);
    let pixy = rho * sxy;

    let ru2 = rho * ux * ux;
    let rv2 = rho * uy * uy;
    let dev = (pixx - piyy) / 2.0;

    let pixy_n = (1.0 - om) * pixy + om * rho * ux * uy + (1.0 - 0.5 * om) * (fy * ux + fx * uy);
    let pixx_n = rho * CS2
        + dev * (1.0 - om)
        + (1.0 + om) / 2.0 * ru2
        + (1.0 - om) / 2.0 * rv2
        + fx * ux
        + (1.0 - om) / 2.0 * (fx * ux - fy * uy);
    let piyy_n = rho * CS2 - dev * (1.0 - om)
        + (1.0 + om) / 2.0 * rv2
        + (1.0 - om) / 2.0 * ru2
        + fy * uy
        + (1.0 - om) / 2.0 * (fy * uy - fx * ux);

    let inv = 1.0 / rho;
    Moments2d {
        rho,
        u: [ux + 0.5 * fx * inv, uy + 0.5 * fy * inv],
        s: [pixx_n * inv - CS2, piyy_n * inv - CS2, pixy_n * inv],
    }
}
