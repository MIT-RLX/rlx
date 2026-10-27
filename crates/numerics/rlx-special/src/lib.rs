// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-special — SciPy's special functions bit-for-bit: Boost.Math and cephes
//!
//! SciPy ≥ 1.15 evaluates `special.betainc`, `betaincc` and `stdtr` (so
//! `stats.t.cdf` / `stats.t.sf` / `ttest_*` p-values) with **Boost.Math**, compiled
//! by clang with its default `-ffp-contract=on`. This crate is a faithful port of
//! those code paths for `double`, so a Rust reimplementation of a SciPy
//! statistic can agree with it *exactly*, not to a tolerance.
//!
//! **Which SciPy functions are Boost — measured, not assumed** (SciPy 1.17.1,
//! 3000 random arguments each, bitwise against a clang `-ffp-contract=on` build
//! of the same Boost headers): `betainc`, `betaincc`, `stdtr` — 0 mismatches.
//! `gammainc`, `gammaincc`, `chdtrc` / `stats.chi2.sf`, `erf`, `erfc`, `gammaln`,
//! `gamma` — 10–84 % mismatches: those are SciPy's cephes-derived `xsf` code, not
//! Boost. Those are ported separately in [`mod@cephes`], from `scipy/xsf`'s headers
//! (verified the same way: 0 / 36 000 mismatches against SciPy). The Boost
//! [`mod@gamma`] family is Boost's own — it is what `ibeta` calls internally — so use
//! [`mod@cephes`] wherever SciPy's `gammainc` / `chi2.sf` / `erf` is the reference.
//!
//! What "faithful" means here, concretely (each measured — see the crate README):
//!
//! * **The same algorithms and branch thresholds** — `ibeta_imp`'s region
//!   dispatch, `gamma_incomplete_imp_final`'s eight evaluation methods, the
//!   Lanczos-13m53 sums, the 53-bit `erf` rationals, the 64-bit-tag
//!   `lgamma_small` rationals, the Temme expansion.
//! * **The same evaluation order** — Boost's second-order Horner scheme
//!   (`BOOST_MATH_POLY_METHOD 3` / `RATIONAL_METHOD 3`) for every fixed-size
//!   polynomial and rational, see [`tools`].
//! * **The same fused multiply-adds** — clang contracts `a*b + c` (and
//!   `a - b*c`, `a += b*c`) *within one C++ expression*, the left product of an
//!   addition first. Each such site is a `fmac` call here — fused on Apple targets,
//!   two roundings on the Linux x86-64 reference, whose SciPy wheel is built without
//!   FMA; everything else rounds as written. Without the contractions ~2-3% of `betainc` values differ
//!   by an ulp.
//! * **libm where Boost uses libm** — Boost's `double` overloads of `log1p` and
//!   `expm1` forward to the C library, so do [`f64::ln_1p`] / [`f64::exp_m1`].
//!
//! Only the *regularized* (`normalised = true`) incomplete functions are ported —
//! those are the ones SciPy exposes.

// Coefficients are Boost's decimal literals verbatim (so they diff against the
// headers); rustc rounds them exactly as clang does. And `!(a < b)` is kept where
// Boost's loop conditions depend on NaN comparing false.
#![allow(clippy::excessive_precision, clippy::neg_cmp_op_on_partial_ord)]

pub mod beta;
pub mod cephes;
mod cephes_tables;
pub mod dist;
pub mod erf;
mod factorial_table;
pub mod gamma;
pub mod lanczos;
pub mod quadpack;
pub mod tools;

pub use beta::{beta, ibeta, ibetac};

/// `a*b + c` at a site where the reference compiler contracts: clang fuses it on
/// Apple targets (SciPy's macOS arm64 wheel); SciPy's Linux x86-64 wheel builds Boost
/// and xsf for the SSE baseline, so there it rounds twice (measured: no FMA
/// instruction in `special/_ufuncs*.so`). Every contraction site of the port is a
/// `fmac` call.
pub(crate) trait Contract {
    fn fmac(self, b: f64, c: f64) -> f64;
}

impl Contract for f64 {
    #[inline(always)]
    fn fmac(self, b: f64, c: f64) -> f64 {
        if cfg!(target_vendor = "apple") {
            self.mul_add(b, c)
        } else {
            self * b + c
        }
    }
}
pub use dist::{normal_cdf, students_t_cdf, students_t_sf};
pub use erf::{erf, erfc};
pub use gamma::{gamma_p, gamma_q, lgamma, log1pmx, powm1, tgamma, tgamma_delta_ratio, tgamma1pm1};
