// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Boost.Math gamma family for `double`, Lanczos-13m53 throughout:
//! `tgamma`, `lgamma`, `tgamma1pm1`, `tgamma_delta_ratio`, and the regularized
//! incomplete gamma `gamma_p` / `gamma_q` (`detail::gamma_incomplete_imp_final`
//! with `normalised = true`, which is what SciPy's `gammainc` / `gammaincc` /
//! `chdtrc` call).

use crate::erf::erfc;
use crate::factorial_table::FACTORIALS;
use crate::lanczos::{self, G};
use crate::tools::{
    EPSILON, LOG_MAX_VALUE, LOG_MIN_VALUE, MAX_FACTORIAL, MAX_SERIES_ITERATIONS, MAX_VALUE,
    ROOT_EPSILON, continued_fraction_a, evaluate_polynomial, itrunc, sum_series,
};

use crate::Contract;
use std::f64::consts::{E, PI};

/// `constants::euler<double>()`.
const EULER: f64 = 0.577215664901532860606512090082402431;

/// `unchecked_factorial<double>(i)`.
#[inline]
pub fn unchecked_factorial(i: u32) -> f64 {
    FACTORIALS[i as usize]
}

/// `boost::math::log1pmx(x)` = `log1p(x) - x`.
pub fn log1pmx(x: f64) -> f64 {
    if x < -1.0 {
        return f64::NAN;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    let a = x.abs();
    if a > 0.95 {
        return rlx_libm::reference::log(1.0 + x) - x;
    }
    if a < EPSILON {
        return -x * x / 2.0;
    }
    // `log1p_series` with its first term (`x`) consumed before summing.
    let m_mult = -x;
    let mut m_prod = -1.0f64;
    let mut k = 0i32;
    let mut next = move || {
        m_prod *= m_mult;
        k += 1;
        m_prod / f64::from(k)
    };
    next();
    sum_series(next, EPSILON, MAX_SERIES_ITERATIONS, 0.0).0
}

/// `boost::math::powm1(x, y)` = `x^y - 1`, for `x > 0`.
pub fn powm1(x: f64, y: f64) -> f64 {
    if (y * (x - 1.0)).abs() < 0.5 || y.abs() < 0.2 {
        let l = y * rlx_libm::reference::log(x);
        if l < 0.5 {
            return rlx_libm::reference::expm1(l);
        }
        if l > LOG_MAX_VALUE {
            return f64::INFINITY;
        }
    }
    rlx_libm::reference::pow(x, y) - 1.0
}

/// `detail::sinpx(z)` = `z * sin(pi * z)`, reduced exactly.
fn sinpx(z: f64) -> f64 {
    let mut sign = 1.0f64;
    let z = z.abs();
    let mut fl = z.floor();
    let mut dist;
    // `is_odd(fl)` for a type convertible to int: `static_cast<int>(v) & 1`
    // (arm64 `fcvtzs` saturates, as Rust's `as` does).
    if (fl as i32) & 1 != 0 {
        fl += 1.0;
        dist = fl - z;
        sign = -sign;
    } else {
        dist = z - fl;
    }
    if dist > 0.5 {
        dist = 1.0 - dist;
    }
    let result = rlx_libm::reference::sin(dist * PI);
    sign * z * result
}

/// `detail::gamma_imp_final` (the `z > -20` part of `gamma_imp`).
fn gamma_imp_final(mut z: f64) -> f64 {
    let mut result = 1.0f64;
    if z <= 0.0 {
        while z < 0.0 {
            result /= z;
            z += 1.0;
        }
    }
    if z.floor() == z && z < f64::from(MAX_FACTORIAL) {
        result *= unchecked_factorial((itrunc(z) - 1) as u32);
    } else if z < ROOT_EPSILON {
        if z < 1.0 / MAX_VALUE {
            return f64::INFINITY;
        }
        result *= 1.0 / z - EULER;
    } else {
        result *= lanczos::lanczos_sum(z);
        let zgh = z + G - 0.5;
        let lzgh = rlx_libm::reference::log(zgh);
        if z * lzgh > LOG_MAX_VALUE {
            if lzgh * z / 2.0 > LOG_MAX_VALUE {
                return result.signum() * f64::INFINITY;
            }
            let hp = rlx_libm::reference::pow(zgh, (z / 2.0) - 0.25);
            result *= hp / rlx_libm::reference::exp(zgh);
            if MAX_VALUE / hp < result {
                return result.signum() * f64::INFINITY;
            }
            result *= hp;
        } else {
            result *= rlx_libm::reference::pow(zgh, z - 0.5) / rlx_libm::reference::exp(zgh);
        }
    }
    result
}

/// `boost::math::tgamma(z)`.
pub fn tgamma(z: f64) -> f64 {
    if z <= 0.0 {
        if z.floor() == z {
            return f64::NAN; // pole
        }
        if z <= -20.0 {
            let mut result = gamma_imp_final(-z) * sinpx(z);
            if result.is_infinite() {
                return 0.0; // Boost catches the overflow and reports underflow
            }
            result = -PI / result;
            return result;
        }
    }
    gamma_imp_final(z)
}

/// `detail::lgamma_small_imp` with the 64-bit tag (what a 53-bit `double` selects).
/// `zm1 = z - 1`, `zm2 = z - 2` are passed in so they can carry extra precision.
pub(crate) fn lgamma_small_imp(mut z: f64, mut zm1: f64, mut zm2: f64) -> f64 {
    let mut result = 0.0f64;
    if zm1 == 0.0 || zm2 == 0.0 {
        // lgamma(1) = lgamma(2) = 0
    } else if z > 2.0 {
        if z >= 3.0 {
            // Boost also decrements `zm2` here, then overwrites it below.
            loop {
                z -= 1.0;
                result += rlx_libm::reference::log(z);
                if !(z >= 3.0) {
                    break;
                }
            }
            zm2 = z - 2.0;
        }
        const P: [f64; 7] = [
            -0.180355685678449379109e-1,
            0.25126649619989678683e-1,
            0.494103151567532234274e-1,
            0.172491608709613993966e-1,
            -0.259453563205438108893e-3,
            -0.541009869215204396339e-3,
            -0.324588649825948492091e-4,
        ];
        const Q: [f64; 8] = [
            0.1e1,
            0.196202987197795200688e1,
            0.148019669424231326694e1,
            0.541391432071720958364e0,
            0.988504251128010129477e-1,
            0.82130967464889339326e-2,
            0.224936291922115757597e-3,
            -0.223352763208617092964e-6,
        ];
        const Y: f64 = 0.158963680267333984375e0;
        let r = zm2 * (z + 1.0);
        let mut rr = evaluate_polynomial(&P, zm2);
        rr /= evaluate_polynomial(&Q, zm2);
        result += r.fmac(Y, r * rr);
    } else {
        if z < 1.0 {
            result += -rlx_libm::reference::log(z);
            zm2 = zm1;
            zm1 = z;
            z += 1.0;
        }
        if z <= 1.5 {
            const Y: f64 = 0.52815341949462890625;
            const P: [f64; 7] = [
                0.490622454069039543534e-1,
                -0.969117530159521214579e-1,
                -0.414983358359495381969e0,
                -0.406567124211938417342e0,
                -0.158413586390692192217e0,
                -0.240149820648571559892e-1,
                -0.100346687696279557415e-2,
            ];
            const Q: [f64; 7] = [
                0.1e1,
                0.302349829846463038743e1,
                0.348739585360723852576e1,
                0.191415588274426679201e1,
                0.507137738614363510846e0,
                0.577039722690451849648e-1,
                0.195768102601107189171e-2,
            ];
            let r = evaluate_polynomial(&P, zm1) / evaluate_polynomial(&Q, zm1);
            let prefix = zm1 * zm2;
            result += prefix.fmac(Y, prefix * r);
        } else {
            const Y: f64 = 0.452017307281494140625;
            const P: [f64; 6] = [
                -0.292329721830270012337e-1,
                0.144216267757192309184e0,
                -0.142440390738631274135e0,
                0.542809694055053558157e-1,
                -0.850535976868336437746e-2,
                0.431171342679297331241e-3,
            ];
            const Q: [f64; 7] = [
                0.1e1,
                -0.150169356054485044494e1,
                0.846973248876495016101e0,
                -0.220095151814995745555e0,
                0.25582797155975869989e-1,
                -0.100666795539143372762e-2,
                -0.827193521891290553639e-6,
            ];
            let r = zm2 * zm1;
            let rr = evaluate_polynomial(&P, -zm2) / evaluate_polynomial(&Q, -zm2);
            result += r.fmac(Y, r * rr);
        }
    }
    result
}

/// `detail::lgamma_imp_final`.
fn lgamma_imp_final(z: f64) -> (f64, i32) {
    let mut sresult = 1;
    let result;
    if z < ROOT_EPSILON {
        if z == 0.0 {
            return (f64::INFINITY, 1);
        }
        if 4.0 * z.abs() < EPSILON {
            result = -rlx_libm::reference::log(z.abs());
        } else {
            result = rlx_libm::reference::log((1.0 / z - EULER).abs());
        }
        if z < 0.0 {
            sresult = -1;
        }
    } else if z < 15.0 {
        result = lgamma_small_imp(z, z - 1.0, z - 2.0);
    } else if z >= 3.0 && z < 100.0 {
        result = rlx_libm::reference::log(tgamma(z));
    } else {
        let zgh = z + G - 0.5;
        let mut r = rlx_libm::reference::log(zgh) - 1.0;
        r *= z - 0.5;
        if r * EPSILON < 20.0 {
            r += rlx_libm::reference::log(lanczos::lanczos_sum_exp_g_scaled(z));
        }
        result = r;
    }
    (result, sresult)
}

/// `boost::math::lgamma(z)`, with the sign of `Γ(z)`.
pub fn lgamma_with_sign(z: f64) -> (f64, i32) {
    if z <= -ROOT_EPSILON {
        if z.floor() == z {
            return (f64::INFINITY, 1); // pole
        }
        let mut t = sinpx(z);
        let mut sresult = 1;
        if t < 0.0 {
            t = -t;
        } else {
            sresult = -sresult;
        }
        let result =
            rlx_libm::reference::log(PI) - lgamma_imp_final(-z).0 - rlx_libm::reference::log(t);
        return (result, sresult);
    }
    lgamma_imp_final(z)
}

/// `boost::math::lgamma(z)`.
pub fn lgamma(z: f64) -> f64 {
    lgamma_with_sign(z).0
}

/// `boost::math::tgamma1pm1(dz)` = `Γ(1 + dz) - 1` (`detail::tgammap1m1_imp`).
pub fn tgamma1pm1(dz: f64) -> f64 {
    if dz < 0.0 {
        if dz < -0.5 {
            tgamma(1.0 + dz) - 1.0
        } else {
            rlx_libm::reference::expm1(
                -rlx_libm::reference::log1p(dz) + lgamma_small_imp(dz + 2.0, dz + 1.0, dz),
            )
        }
    } else if dz < 2.0 {
        rlx_libm::reference::expm1(lgamma_small_imp(dz + 1.0, dz, dz - 1.0))
    } else {
        tgamma(1.0 + dz) - 1.0
    }
}

/// `detail::tgamma_delta_ratio_imp_lanczos_final`.
fn tgamma_delta_ratio_lanczos_final(z: f64, delta: f64) -> f64 {
    let zgh = z + G - 0.5;
    let mut result;
    if z + delta == z {
        result = rlx_libm::reference::exp(-delta);
    } else {
        if delta.abs() < 10.0 {
            result = rlx_libm::reference::exp((0.5 - z) * rlx_libm::reference::log1p(delta / zgh));
        } else {
            result = rlx_libm::reference::pow(zgh / (zgh + delta), z - 0.5);
        }
        result *= lanczos::lanczos_sum(z) / lanczos::lanczos_sum(z + delta);
    }
    result *= rlx_libm::reference::pow(E / (zgh + delta), delta);
    result
}

/// `boost::math::tgamma_delta_ratio(z, delta)` = `Γ(z) / Γ(z + delta)`.
pub fn tgamma_delta_ratio(mut z: f64, mut delta: f64) -> f64 {
    if z <= 0.0 || z + delta <= 0.0 {
        return tgamma(z) / tgamma(z + delta);
    }
    let maxf = f64::from(MAX_FACTORIAL);
    if delta.floor() == delta {
        if z.floor() == z && z <= maxf && z + delta <= maxf {
            return unchecked_factorial((itrunc(z) - 1) as u32)
                / unchecked_factorial((itrunc(z + delta) - 1) as u32);
        }
        if delta.abs() < 20.0 {
            if delta == 0.0 {
                return 1.0;
            }
            if delta < 0.0 {
                z -= 1.0;
                let mut result = z;
                loop {
                    delta += 1.0;
                    if delta == 0.0 {
                        break;
                    }
                    z -= 1.0;
                    result *= z;
                }
                return result;
            } else {
                let mut result = 1.0 / z;
                loop {
                    delta -= 1.0;
                    if delta == 0.0 {
                        break;
                    }
                    z += 1.0;
                    result /= z;
                }
                return result;
            }
        }
    }
    // `tgamma_delta_ratio_imp_lanczos`
    if z < EPSILON {
        if maxf < delta {
            let mut ratio = tgamma_delta_ratio_lanczos_final(delta, maxf - delta);
            ratio *= z;
            ratio *= unchecked_factorial(MAX_FACTORIAL - 1);
            return 1.0 / ratio;
        }
        return 1.0 / (z * tgamma(z + delta));
    }
    tgamma_delta_ratio_lanczos_final(z, delta)
}

/// `detail::regularised_gamma_prefix(a, z)` = `z^a e^{-z} / Γ(a)`.
pub(crate) fn regularised_gamma_prefix(a: f64, z: f64) -> f64 {
    if z >= MAX_VALUE || (a > 0.0 && z == 0.0) {
        return 0.0;
    }
    let agh = a + G - 0.5;
    let mut prefix;
    let d = ((z - a) - G + 0.5) / agh;
    if a < 1.0 {
        if z <= LOG_MIN_VALUE || a < 1.0 / MAX_VALUE {
            // `a * log(z) - z - lgamma(a)`: the left product fuses.
            return rlx_libm::reference::exp(a.fmac(rlx_libm::reference::log(z), -z) - lgamma(a));
        }
        return rlx_libm::reference::pow(z, a) * rlx_libm::reference::exp(-z) / tgamma(a);
    } else if (d * d * a).abs() <= 100.0 && a > 150.0 {
        // `a * log1pmx(d) + z * (0.5 - g) / agh`
        prefix = a.fmac(log1pmx(d), z * (0.5 - G) / agh);
        prefix = rlx_libm::reference::exp(prefix);
    } else {
        let alz = a * rlx_libm::reference::log(z / agh);
        let amz = a - z;
        if alz.min(amz) <= LOG_MIN_VALUE || alz.max(amz) >= LOG_MAX_VALUE {
            let amza = amz / a;
            if alz.min(amz) / 2.0 > LOG_MIN_VALUE && alz.max(amz) / 2.0 < LOG_MAX_VALUE {
                let sq = rlx_libm::reference::pow(z / agh, a / 2.0)
                    * rlx_libm::reference::exp(amz / 2.0);
                prefix = sq * sq;
            } else if alz.min(amz) / 4.0 > LOG_MIN_VALUE
                && alz.max(amz) / 4.0 < LOG_MAX_VALUE
                && z > a
            {
                let sq = rlx_libm::reference::pow(z / agh, a / 4.0)
                    * rlx_libm::reference::exp(amz / 4.0);
                prefix = sq * sq;
                prefix *= prefix;
            } else if amza > LOG_MIN_VALUE && amza < LOG_MAX_VALUE {
                prefix = rlx_libm::reference::pow((z * rlx_libm::reference::exp(amza)) / agh, a);
            } else {
                prefix = rlx_libm::reference::exp(alz + amz);
            }
        } else {
            prefix = rlx_libm::reference::pow(z / agh, a) * rlx_libm::reference::exp(amz);
        }
    }
    prefix *= (agh / E).sqrt() / lanczos::lanczos_sum_exp_g_scaled(a);
    prefix
}

/// `detail::lower_gamma_series(a, z, init_value)`.
fn lower_gamma_series(a: f64, z: f64, init: f64) -> f64 {
    let mut a = a;
    let mut result = 1.0f64;
    let next = move || {
        let r = result;
        a += 1.0;
        result *= z / a;
        r
    };
    sum_series(next, EPSILON, MAX_SERIES_ITERATIONS, init).0
}

/// `detail::upper_gamma_fraction(a, z, eps)`.
fn upper_gamma_fraction(a: f64, z: f64, eps: f64) -> f64 {
    let mut zz = z - a + 1.0;
    let mut k = 0i32;
    let f = move || {
        k += 1;
        zz += 2.0;
        let kf = f64::from(k);
        (kf * (a - kf), zz)
    };
    1.0 / (z - a + 1.0 + continued_fraction_a(f, eps, u64::MAX))
}

/// `detail::tgamma_small_upper_part(a, x, &gam, invert)`; returns `(result, gam)`.
fn tgamma_small_upper_part(a: f64, x: f64, invert: bool) -> (f64, f64) {
    let mut result = tgamma1pm1(a);
    let pgam = (result + 1.0) / a;
    let mut p = powm1(x, a);
    result -= p;
    result /= a;
    // `small_gamma2_series`
    let mut sres = -x;
    let sx = -x;
    let mut apn = a + 1.0;
    let mut n = 1i32;
    let series = move || {
        let r = sres / apn;
        sres *= sx;
        n += 1;
        sres /= f64::from(n);
        apn += 1.0;
        r
    };
    let max_iter = MAX_SERIES_ITERATIONS - 10;
    p += 1.0;
    let init = if invert { pgam } else { 0.0 };
    result = -p * sum_series(series, EPSILON, max_iter, (init - result) / p).0;
    if invert {
        result = -result;
    }
    (result, pgam)
}

/// `detail::finite_gamma_q(a, x)` (integer `a`).
fn finite_gamma_q(a: f64, x: f64) -> f64 {
    let e = rlx_libm::reference::exp(-x);
    let mut sum = e;
    if sum != 0.0 {
        let mut term = sum;
        let mut n = 1u32;
        while f64::from(n) < a {
            term /= f64::from(n);
            term *= x;
            sum += term;
            n += 1;
        }
    }
    sum
}

/// `detail::finite_half_gamma_q(a, x)` (half-integer `a`).
fn finite_half_gamma_q(a: f64, x: f64) -> f64 {
    let mut e = erfc(x.sqrt());
    if e != 0.0 && a > 1.0 {
        let mut term = rlx_libm::reference::exp(-x) / (PI * x).sqrt();
        term *= x;
        const HALF: f64 = 0.5;
        term /= HALF;
        let mut sum = term;
        let mut n = 2u32;
        while f64::from(n) < a {
            term /= f64::from(n) - HALF;
            term *= x;
            sum += term;
            n += 1;
        }
        e += sum;
    }
    e
}

/// `detail::incomplete_tgamma_large_x(a, x)`.
fn incomplete_tgamma_large_x(a: f64, x: f64) -> f64 {
    let mut a_poch = a - 1.0;
    let mut term = 1.0f64;
    let next = move || {
        let r = term;
        term *= a_poch / x;
        a_poch -= 1.0;
        r
    };
    sum_series(next, EPSILON, MAX_SERIES_ITERATIONS, 0.0).0
}

/// `detail::igamma_temme_large(a, x)`, 53-bit tag.
fn igamma_temme_large(a: f64, x: f64) -> f64 {
    let sigma = (x - a) / a;
    let phi = -log1pmx(sigma);
    let y = a * phi;
    let mut z = (2.0 * phi).sqrt();
    if x < a {
        z = -z;
    }
    const C0: [f64; 15] = [
        -0.33333333333333333,
        0.083333333333333333,
        -0.014814814814814815,
        0.0011574074074074074,
        0.0003527336860670194,
        -0.00017875514403292181,
        0.39192631785224378e-4,
        -0.21854485106799922e-5,
        -0.185406221071516e-5,
        0.8296711340953086e-6,
        -0.17665952736826079e-6,
        0.67078535434014986e-8,
        0.10261809784240308e-7,
        -0.43820360184533532e-8,
        0.91476995822367902e-9,
    ];
    const C1: [f64; 13] = [
        -0.0018518518518518519,
        -0.0034722222222222222,
        0.0026455026455026455,
        -0.00099022633744855967,
        0.00020576131687242798,
        -0.40187757201646091e-6,
        -0.18098550334489978e-4,
        0.76491609160811101e-5,
        -0.16120900894563446e-5,
        0.46471278028074343e-8,
        0.1378633446915721e-6,
        -0.5752545603517705e-7,
        0.11951628599778147e-7,
    ];
    const C2: [f64; 11] = [
        0.0041335978835978836,
        -0.0026813271604938272,
        0.00077160493827160494,
        0.20093878600823045e-5,
        -0.00010736653226365161,
        0.52923448829120125e-4,
        -0.12760635188618728e-4,
        0.34235787340961381e-7,
        0.13721957309062933e-5,
        -0.6298992138380055e-6,
        0.14280614206064242e-6,
    ];
    const C3: [f64; 9] = [
        0.00064943415637860082,
        0.00022947209362139918,
        -0.00046918949439525571,
        0.00026772063206283885,
        -0.75618016718839764e-4,
        -0.23965051138672967e-6,
        0.11082654115347302e-4,
        -0.56749528269915966e-5,
        0.14230900732435884e-5,
    ];
    const C4: [f64; 7] = [
        -0.0008618882909167117,
        0.00078403922172006663,
        -0.00029907248030319018,
        -0.14638452578843418e-5,
        0.66414982154651222e-4,
        -0.39683650471794347e-4,
        0.11375726970678419e-4,
    ];
    const C5: [f64; 9] = [
        -0.00033679855336635815,
        -0.69728137583658578e-4,
        0.00027727532449593921,
        -0.00019932570516188848,
        0.67977804779372078e-4,
        0.1419062920643967e-6,
        -0.13594048189768693e-4,
        0.80184702563342015e-5,
        -0.22914811765080952e-5,
    ];
    const C6: [f64; 7] = [
        0.00053130793646399222,
        -0.00059216643735369388,
        0.00027087820967180448,
        0.79023532326603279e-6,
        -0.81539693675619688e-4,
        0.56116827531062497e-4,
        -0.18329116582843376e-4,
    ];
    const C7: [f64; 5] = [
        0.00034436760689237767,
        0.51717909082605922e-4,
        -0.00033493161081142236,
        0.0002812695154763237,
        -0.00010976582244684731,
    ];
    const C8: [f64; 3] = [
        -0.00065262391859530942,
        0.00083949872067208728,
        -0.00043829709854172101,
    ];
    let workspace = [
        evaluate_polynomial(&C0, z),
        evaluate_polynomial(&C1, z),
        evaluate_polynomial(&C2, z),
        evaluate_polynomial(&C3, z),
        evaluate_polynomial(&C4, z),
        evaluate_polynomial(&C5, z),
        evaluate_polynomial(&C6, z),
        evaluate_polynomial(&C7, z),
        evaluate_polynomial(&C8, z),
        -0.00059676129019274625,
    ];
    let mut result = evaluate_polynomial(&workspace, 1.0 / a);
    result *= rlx_libm::reference::exp(-y) / (2.0 * PI * a).sqrt();
    if x < a {
        result = -result;
    }
    result += erfc(y.sqrt()) / 2.0;
    result
}

/// `detail::gamma_incomplete_imp_final(a, x, normalised = true, invert)`.
fn gamma_incomplete_regularized(a: f64, x: f64, mut invert: bool) -> f64 {
    if a <= 0.0 || x < 0.0 || a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    let mut result;
    let is_small_a = a < 30.0 && a <= x + 1.0 && x < LOG_MAX_VALUE;
    let (is_int, is_half_int) = if is_small_a {
        let fa = a.floor();
        let is_int = fa == a;
        (is_int, if is_int { false } else { (fa - a).abs() == 0.5 })
    } else {
        (false, false)
    };
    let eval_method;
    if x == 0.0 {
        eval_method = 2;
    } else if is_int && x > 0.6 {
        invert = !invert;
        eval_method = 0;
    } else if is_half_int && x > 0.2 {
        invert = !invert;
        eval_method = 1;
    } else if x < ROOT_EPSILON && a > 1.0 {
        eval_method = 6;
    } else if x > 1000.0 && (a < x || (a - 50.0).abs() / x < 1.0) {
        invert = !invert;
        eval_method = 7;
    } else if x < 0.5 {
        eval_method = if -0.4 / rlx_libm::reference::log(x) < a {
            2
        } else {
            3
        };
    } else if x < 1.1 {
        eval_method = if x * 0.75 < a { 2 } else { 3 };
    } else {
        let mut use_temme = false;
        if a > 20.0 {
            let sigma = ((x - a) / a).abs();
            if a > 200.0 {
                if 20.0 / a > sigma * sigma {
                    use_temme = true;
                }
            } else if sigma < 0.4 {
                use_temme = true;
            }
        }
        if use_temme {
            eval_method = 5;
        } else if x - (1.0 / (3.0 * x)) < a {
            eval_method = 2;
        } else {
            eval_method = 4;
            invert = !invert;
        }
    }

    match eval_method {
        0 => result = finite_gamma_q(a, x),
        1 => result = finite_half_gamma_q(a, x),
        2 => {
            result = regularised_gamma_prefix(a, x);
            if result != 0.0 {
                let mut init_value = 0.0;
                let mut optimised_invert = false;
                if invert {
                    init_value = 1.0 / result;
                    init_value *= -a;
                    optimised_invert = true;
                }
                result *= lower_gamma_series(a, x, init_value) / a;
                if optimised_invert {
                    invert = false;
                    result = -result;
                }
            }
        }
        3 => {
            invert = !invert;
            let (r, g) = tgamma_small_upper_part(a, x, invert);
            invert = false;
            result = r / g;
        }
        4 => {
            result = regularised_gamma_prefix(a, x);
            if result != 0.0 {
                result *= upper_gamma_fraction(a, x, EPSILON);
            }
        }
        5 => {
            result = igamma_temme_large(a, x);
            if x >= a {
                invert = !invert;
            }
        }
        6 => {
            result = rlx_libm::reference::pow(x, a) / tgamma(a + 1.0);
            result *= 1.0 - a * x / (a + 1.0);
        }
        _ => {
            result = regularised_gamma_prefix(a, x);
            result /= x;
            if result != 0.0 {
                result *= incomplete_tgamma_large_x(a, x);
            }
        }
    }
    if result > 1.0 {
        result = 1.0;
    }
    if invert {
        result = 1.0 - result;
    }
    result
}

/// `boost::math::gamma_p(a, x)`: the regularized lower incomplete gamma
/// (SciPy `gammainc`).
pub fn gamma_p(a: f64, x: f64) -> f64 {
    gamma_incomplete_regularized(a, x, false)
}

/// `boost::math::gamma_q(a, x)`: the regularized upper incomplete gamma
/// (SciPy `gammaincc`).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    gamma_incomplete_regularized(a, x, true)
}
