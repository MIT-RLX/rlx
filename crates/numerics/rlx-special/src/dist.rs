// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distribution functions in the spelling SciPy uses, built on [`mod@crate::beta`]
//! and [`mod@crate::gamma`].

use crate::beta::{ibeta, ibetac};
use crate::erf::erfc;
use crate::tools::EPSILON;

use std::f64::consts::SQRT_2;

/// Standard normal CDF, Boost's `cdf(normal_distribution(0, 1), x)`:
/// `erfc(-x / √2) / 2`.
pub fn normal_cdf(x: f64) -> f64 {
    erfc(-(x / SQRT_2)) / 2.0
}

/// Student-t CDF, Boost's `cdf(students_t_distribution(df), t)` — and SciPy's
/// `special.stdtr(df, t)` / `stats.t.cdf(t, df)`.
pub fn students_t_cdf(df: f64, t: f64) -> f64 {
    if !(df > 0.0) || t.is_nan() {
        return f64::NAN;
    }
    if t == 0.0 {
        return 0.5;
    }
    if t.is_infinite() {
        return if t < 0.0 { 0.0 } else { 1.0 };
    }
    if df > 1.0 / EPSILON {
        return normal_cdf(t);
    }
    let x2 = t * t;
    let probability = if df > 2.0 * x2 {
        let z = x2 / (df + x2);
        ibetac(0.5, df / 2.0, z) / 2.0
    } else {
        let z = df / (df + x2);
        ibeta(df / 2.0, 0.5, z) / 2.0
    };
    if t > 0.0 {
        1.0 - probability
    } else {
        probability
    }
}

/// Student-t survival function as SciPy spells it: `stats.t.sf(t, df)` is
/// `stdtr(df, -t)`, *not* `1 - cdf`.
pub fn students_t_sf(df: f64, t: f64) -> f64 {
    students_t_cdf(df, -t)
}
