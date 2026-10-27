// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `boost::math::erf` / `erfc` for `double` — `detail::erf_imp` with the
//! 53-bit tag. (Boost does *not* forward these to libm on the host.)

use crate::Contract;
use crate::tools::evaluate_polynomial;

/// `boost::math::erf(z)`.
pub fn erf(z: f64) -> f64 {
    erf_imp(z, false)
}

/// `boost::math::erfc(z)`.
pub fn erfc(z: f64) -> f64 {
    erf_imp(z, true)
}

/// `exp(-z*z)` with the rounding error of `z*z` folded back in, as the
/// `z >= 1.5` branches do: `hi` keeps the top 26 bits of `z`.
#[inline]
fn exp_neg_sq_over_z(z: f64) -> f64 {
    let (mant, expon) = frexp(z);
    let hi = ldexp(ldexp(mant, 26).floor(), expon - 26);
    let lo = z - hi;
    let sq = z * z;
    // `((hi*hi - sq) + 2*hi*lo) + lo*lo`: three fused steps.
    let err_sqr = lo.fmac(lo, (2.0 * hi).fmac(lo, hi.fmac(hi, -sq)));
    rlx_libm::reference::exp(-sq) * rlx_libm::reference::exp(-err_sqr) / z
}

pub(crate) fn erf_imp(mut z: f64, mut invert: bool) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    let mut prefix_multiplier: i32 = 1;
    let mut prefix_adder: i32 = 0;
    if z < 0.0 {
        z = -z;
        if !invert {
            prefix_multiplier = -1;
        } else if z > 0.5 {
            prefix_adder = 2;
            prefix_multiplier = -1;
        } else {
            invert = false;
            prefix_adder = 1;
        }
    }
    let result;
    if z < 0.5 {
        if z < 1e-10 {
            if z == 0.0 {
                result = 0.0;
            } else {
                const C: f64 = 0.003379167095512573896158903121545171688;
                // `z * 1.125f + z * c`: the left product fuses.
                result = z.fmac(1.125, z * C);
            }
        } else {
            const Y: f64 = 1.044948577880859375;
            const P: [f64; 5] = [
                0.0834305892146531832907,
                -0.338165134459360935041,
                -0.0509990735146777432841,
                -0.00772758345802133288487,
                -0.000322780120964605683831,
            ];
            const Q: [f64; 5] = [
                1.0,
                0.455004033050794024546,
                0.0875222600142252549554,
                0.00858571925074406212772,
                0.000370900071787748000569,
            ];
            let zz = z * z;
            result = z * (Y + evaluate_polynomial(&P, zz) / evaluate_polynomial(&Q, zz));
        }
    } else if if invert { z < 28.0 } else { z < 5.93 } {
        invert = !invert;
        if z < 1.5 {
            const Y: f64 = 0.405935764312744140625;
            const P: [f64; 6] = [
                -0.098090592216281240205,
                0.178114665841120341155,
                0.191003695796775433986,
                0.0888900368967884466578,
                0.0195049001251218801359,
                0.00180424538297014223957,
            ];
            const Q: [f64; 7] = [
                1.0,
                1.84759070983002217845,
                1.42628004845511324508,
                0.578052804889902404909,
                0.12385097467900864233,
                0.0113385233577001411017,
                0.337511472483094676155e-5,
            ];
            let mut r = Y + evaluate_polynomial(&P, z - 0.5) / evaluate_polynomial(&Q, z - 0.5);
            r *= rlx_libm::reference::exp(-z * z) / z;
            result = r;
        } else if z < 2.5 {
            const Y: f64 = 0.50672817230224609375;
            const P: [f64; 6] = [
                -0.0243500476207698441272,
                0.0386540375035707201728,
                0.04394818964209516296,
                0.0175679436311802092299,
                0.00323962406290842133584,
                0.000235839115596880717416,
            ];
            const Q: [f64; 6] = [
                1.0,
                1.53991494948552447182,
                0.982403709157920235114,
                0.325732924782444448493,
                0.0563921837420478160373,
                0.00410369723978904575884,
            ];
            let mut r = Y + evaluate_polynomial(&P, z - 1.5) / evaluate_polynomial(&Q, z - 1.5);
            r *= exp_neg_sq_over_z(z);
            result = r;
        } else if z < 4.5 {
            const Y: f64 = 0.5405750274658203125;
            const P: [f64; 6] = [
                0.00295276716530971662634,
                0.0137384425896355332126,
                0.00840807615555585383007,
                0.00212825620914618649141,
                0.000250269961544794627958,
                0.113212406648847561139e-4,
            ];
            const Q: [f64; 6] = [
                1.0,
                1.04217814166938418171,
                0.442597659481563127003,
                0.0958492726301061423444,
                0.0105982906484876531489,
                0.000479411269521714493907,
            ];
            let mut r = Y + evaluate_polynomial(&P, z - 3.5) / evaluate_polynomial(&Q, z - 3.5);
            r *= exp_neg_sq_over_z(z);
            result = r;
        } else {
            const Y: f64 = 0.5579090118408203125;
            const P: [f64; 7] = [
                0.00628057170626964891937,
                0.0175389834052493308818,
                -0.212652252872804219852,
                -0.687717681153649930619,
                -2.5518551727311523996,
                -3.22729451764143718517,
                -2.8175401114513378771,
            ];
            const Q: [f64; 7] = [
                1.0,
                2.79257750980575282228,
                11.0567237927800161565,
                15.930646027911794143,
                22.9367376522880577224,
                13.5064170191802889145,
                5.48409182238641741584,
            ];
            let mut r = Y + evaluate_polynomial(&P, 1.0 / z) / evaluate_polynomial(&Q, 1.0 / z);
            r *= exp_neg_sq_over_z(z);
            result = r;
        }
    } else {
        result = 0.0;
        invert = !invert;
    }
    if invert {
        prefix_adder += prefix_multiplier;
        prefix_multiplier = -prefix_multiplier;
    }
    // `prefix_adder + prefix_multiplier * result`, both ints promoted: one fma,
    // exact here since the multiplier is ±1.
    f64::from(prefix_multiplier).fmac(result, f64::from(prefix_adder))
}

/// `frexp`: `x = m · 2^e` with `0.5 <= |m| < 1` (finite, nonzero `x`).
pub(crate) fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // Subnormal: normalise first.
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e - 54);
    }
    let e = exp_field - 1022;
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022u64 << 52));
    (m, e)
}

/// `ldexp(x, e)`: `x · 2^e`, exact unless it under/overflows.
pub(crate) fn ldexp(x: f64, e: i32) -> f64 {
    // Split the scale so neither factor overflows or loses precision for |e| <= 2098.
    let mut x = x;
    let mut e = e;
    while e > 1023 {
        x *= f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
        e -= 1023;
    }
    while e < -1022 {
        x *= f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
        e += 1022;
    }
    x * f64::from_bits(((e + 1023) as u64) << 52)
}
