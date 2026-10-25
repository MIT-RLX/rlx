// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Boost.Math beta family for `double`: `beta` and the regularized incomplete
//! beta `ibeta` / `ibetac` (`detail::ibeta_imp` with `normalised = true`, which is
//! what SciPy's `betainc` / `betaincc` and every Boost distribution CDF call).

use crate::erf::{erf, erfc};
use crate::gamma::{
    gamma_q, powm1, regularised_gamma_prefix, tgamma_delta_ratio, unchecked_factorial,
};
use crate::lanczos::{G, lanczos_sum_exp_g_scaled as lsum};
use crate::tools::{
    EPSILON, LOG_MAX_VALUE, LOG_MIN_VALUE, MAX_FACTORIAL, MAX_SERIES_ITERATIONS, MAX_VALUE,
    MIN_VALUE, continued_fraction_b, itrunc, sum_series,
};

use std::f64::consts::{E, FRAC_PI_2};

/// `boost::math::beta(a, b)` (`detail::beta_imp` with Lanczos).
pub fn beta(mut a: f64, mut b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let c = a + b;
    if c == a && b < EPSILON {
        return 1.0 / b;
    } else if c == b && a < EPSILON {
        return 1.0 / a;
    }
    if b == 1.0 {
        return 1.0 / a;
    } else if a == 1.0 {
        return 1.0 / b;
    } else if c < EPSILON {
        let mut result = c / a;
        result /= b;
        return result;
    }
    if a < b {
        std::mem::swap(&mut a, &mut b);
    }
    let agh = a + G - 0.5;
    let bgh = b + G - 0.5;
    let cgh = c + G - 0.5;
    let mut result = lsum(a) * (lsum(b) / lsum(c));
    let ambh = a - 0.5 - b;
    if (b * ambh).abs() < cgh * 100.0 && a > 100.0 {
        result *= (ambh * (-b / cgh).ln_1p()).exp();
    } else {
        result *= (agh / cgh).powf(a - 0.5 - b);
    }
    if cgh > 1e10 {
        result *= ((agh / cgh) * (bgh / cgh)).powf(b);
    } else {
        result *= ((agh * bgh) / (cgh * cgh)).powf(b);
    }
    result *= (E / bgh).sqrt();
    result
}

/// `detail::ibeta_power_terms(a, b, x, y, lanczos, normalised = true)`:
/// `x^a y^b / B(a, b)`.
fn ibeta_power_terms(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let c = a + b;
    let gh = G - 0.5;
    let agh = a + gh;
    let bgh = b + gh;
    let cgh = c + gh;
    let mut result = if a < MIN_VALUE || b < MIN_VALUE {
        0.0
    } else {
        lsum(c) / (lsum(a) * lsum(b))
    };
    result *= (bgh / E).sqrt();
    result *= (agh / cgh).sqrt();
    // `((x*b - y*a) - y*gh) / agh` — both subtractions of a product fuse.
    let l1 = (-y).mul_add(gh, x.mul_add(b, -(y * a))) / agh;
    let l2 = (-x).mul_add(gh, y.mul_add(a, -(x * b))) / bgh;
    if l1.abs().min(l2.abs()) < 0.2 {
        if l1 * l2 > 0.0 || a.min(b) < 1.0 {
            if l1.abs() < 0.1 {
                result *= (a * l1.ln_1p()).exp();
            } else {
                result *= ((x * cgh) / agh).powf(a);
            }
            if l2.abs() < 0.1 {
                result *= (b * l2.ln_1p()).exp();
            } else {
                result *= ((y * cgh) / bgh).powf(b);
            }
        } else if l1.abs().max(l2.abs()) < 0.5 {
            let small_a = a < b;
            let ratio = b / a;
            if (small_a && ratio * l2 < 0.1) || (!small_a && l1 / ratio > 0.1) {
                let mut l3 = (ratio * l2.ln_1p()).exp_m1();
                l3 = l3.mul_add(l1, l1 + l3); // `l1 + l3 + l3 * l1`
                l3 = a * l3.ln_1p();
                result *= l3.exp();
            } else {
                let mut l3 = (l1.ln_1p() / ratio).exp_m1();
                l3 = l3.mul_add(l2, l2 + l3); // `l2 + l3 + l3 * l2`
                l3 = b * l3.ln_1p();
                result *= l3.exp();
            }
        } else if l1.abs() < l2.abs() {
            let mut l = a.mul_add(l1.ln_1p(), b * ((y * cgh) / bgh).ln());
            if l <= LOG_MIN_VALUE || l >= LOG_MAX_VALUE {
                l += result.ln();
                if l >= LOG_MAX_VALUE {
                    return f64::INFINITY;
                }
                result = l.exp();
            } else {
                result *= l.exp();
            }
        } else {
            let mut l = b.mul_add(l2.ln_1p(), a * ((x * cgh) / agh).ln());
            if l <= LOG_MIN_VALUE || l >= LOG_MAX_VALUE {
                l += result.ln();
                if l >= LOG_MAX_VALUE {
                    return f64::INFINITY;
                }
                result = l.exp();
            } else {
                result *= l.exp();
            }
        }
    } else {
        let b1 = (x * cgh) / agh;
        let b2 = (y * cgh) / bgh;
        let l1 = a * b1.ln();
        let mut l2 = b * b2.ln();
        if l1 >= LOG_MAX_VALUE || l1 <= LOG_MIN_VALUE || l2 >= LOG_MAX_VALUE || l2 <= LOG_MIN_VALUE
        {
            if a < b {
                let p1 = b2.powf(b / a);
                let l3 = if b1 != 0.0 && p1 != 0.0 {
                    a * (b1.ln() + p1.ln())
                } else {
                    MAX_VALUE
                };
                if l3 < LOG_MAX_VALUE && l3 > LOG_MIN_VALUE {
                    result *= (p1 * b1).powf(a);
                } else {
                    l2 += l1 + result.ln();
                    if l2 >= LOG_MAX_VALUE {
                        return f64::INFINITY;
                    }
                    result = l2.exp();
                }
            } else {
                let p1 = if b1 < 1.0 && b < 1.0 && MAX_VALUE * b < a {
                    0.0
                } else {
                    b1.powf(a / b)
                };
                let l3 = if p1 != 0.0 && b2 != 0.0 {
                    (p1.ln() + b2.ln()) * b
                } else {
                    MAX_VALUE
                };
                if l3 < LOG_MAX_VALUE && l3 > LOG_MIN_VALUE {
                    result *= (p1 * b2).powf(b);
                } else if result != 0.0 {
                    l2 += l1 + result.ln();
                    if l2 >= LOG_MAX_VALUE {
                        return f64::INFINITY;
                    }
                    result = l2.exp();
                }
            }
        } else {
            result *= b1.powf(a) * b2.powf(b);
        }
    }
    // `result == 0` is a true zero or an underflow; both return 0 under SciPy's policy.
    result
}

/// `detail::ibeta_series(a, b, x, s0, lanczos, normalised = true)`.
fn ibeta_series(a: f64, b: f64, x: f64, s0: f64) -> f64 {
    let c = a + b;
    let agh = a + G - 0.5;
    let bgh = b + G - 0.5;
    let cgh = c + G - 0.5;
    let mut result;
    if a < MIN_VALUE || b < MIN_VALUE {
        result = 0.0;
    } else {
        let l1 = lsum(c);
        let l2 = lsum(a);
        let l3 = lsum(b);
        result = if l2 > 1.0 && l3 > 1.0 && MAX_VALUE / l2 < l3 {
            (l1 / l2) / l3
        } else {
            l1 / (l2 * l3)
        };
    }
    if !result.is_finite() {
        result = 0.0;
    }
    let l1 = (cgh / bgh).ln() * (b - 0.5);
    let l2 = (x * cgh / agh).ln() * a;
    if l1 > LOG_MIN_VALUE && l1 < LOG_MAX_VALUE && l2 > LOG_MIN_VALUE && l2 < LOG_MAX_VALUE {
        if a * b < bgh * 10.0 {
            result *= ((b - 0.5) * (a / bgh).ln_1p()).exp();
        } else {
            result *= (cgh / bgh).powf(b - 0.5);
        }
        result *= (x * cgh / agh).powf(a);
        result *= (agh / E).sqrt();
    } else if result != 0.0 {
        result = result.ln() + l1 + l2 + (agh.ln() - 1.0) / 2.0;
        result = result.exp();
    }
    if result < MIN_VALUE {
        return s0;
    }
    // `ibeta_series_t`
    let mut term = result;
    let mut apn = a;
    let mut poch = 1.0 - b;
    let mut n = 1i32;
    let next = move || {
        let r = term / apn;
        apn += 1.0;
        term *= poch * x / f64::from(n);
        n += 1;
        poch += 1.0;
        r
    };
    sum_series(next, EPSILON, MAX_SERIES_ITERATIONS, s0).0
}

/// `detail::ibeta_fraction2(a, b, x, y, normalised = true)`.
fn ibeta_fraction2(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let result = ibeta_power_terms(a, b, x, y);
    if result == 0.0 {
        return result;
    }
    let mut m = 0i32;
    let f = move || {
        let mf = f64::from(m);
        let two_m = f64::from(2 * m);
        let denom = a + two_m - 1.0;
        let an = (mf * (a + mf - 1.0) / denom) * ((a + b + mf - 1.0) / denom) * (b - mf) * x * x;
        let mut bn = mf;
        bn += (mf * (b - mf) * x) / (a + two_m - 1.0);
        // `a*y - b*x + 1 + m*(2 - x)`: `a*y - b*x` fuses, then `… + m*(2 - x)` fuses.
        let inner = mf.mul_add(2.0 - x, a.mul_add(y, -(b * x)) + 1.0);
        bn += ((a + mf) * inner) / (a + two_m + 1.0);
        m += 1;
        (an, bn)
    };
    let fract = continued_fraction_b(f, EPSILON, MAX_SERIES_ITERATIONS);
    result / fract
}

/// `detail::ibeta_a_step(a, b, x, y, k, normalised = true)`.
fn ibeta_a_step(a: f64, b: f64, x: f64, y: f64, k: i32) -> f64 {
    let mut prefix = ibeta_power_terms(a, b, x, y);
    prefix /= a;
    if prefix == 0.0 {
        return prefix;
    }
    let mut sum = 1.0;
    let mut term = 1.0;
    for i in 0..(k - 1) {
        let fi = f64::from(i);
        term *= (a + b + fi) * x / (a + fi + 1.0);
        sum += term;
    }
    prefix *= sum;
    prefix
}

/// `Pn_size<double>::value`.
const PN_SIZE: usize = 30;

/// `detail::beta_small_b_large_a_series(a, b, x, y, s0, mult, normalised = true)`.
fn beta_small_b_large_a_series(a: f64, b: f64, x: f64, y: f64, s0: f64, mult: f64) -> f64 {
    let bm1 = b - 1.0;
    let t = a + bm1 / 2.0;
    let lx = if y < 0.35 { (-y).ln_1p() } else { x.ln() };
    let u = -t * lx;
    let h = regularised_gamma_prefix(b, u);
    if h <= MIN_VALUE {
        return s0;
    }
    let mut prefix = h / tgamma_delta_ratio(a, b);
    prefix /= t.powf(b);
    prefix *= mult;
    let mut p = [0.0f64; PN_SIZE];
    p[0] = 1.0;
    let mut j = gamma_q(b, u) / h;
    let mut sum = prefix.mul_add(j, s0); // `s0 + prefix * j`
    let mut tnp1: u32 = 1;
    let mut lx2 = lx / 2.0;
    lx2 *= lx2;
    let mut lxp = 1.0;
    let t4 = 4.0 * t * t;
    let mut b2n = b;
    for n in 1..PN_SIZE {
        tnp1 += 2;
        p[n] = 0.0;
        let nf = n as f64;
        let mut tmp1: u32 = 3;
        for m in 1..n {
            let mbn = (m as f64).mul_add(b, -nf); // `m * b - n`
            p[n] += mbn * p[n - m] / unchecked_factorial(tmp1);
            tmp1 += 2;
        }
        p[n] /= nf;
        p[n] += bm1 / unchecked_factorial(tnp1);
        // `(b2n*(b2n+1)*j + (u+b2n+1)*lxp) / t4`
        j = (b2n * (b2n + 1.0)).mul_add(j, (u + b2n + 1.0) * lxp) / t4;
        lxp *= lx2;
        b2n += 2.0;
        let r = prefix * p[n] * j;
        sum += r;
        if (r / EPSILON).abs() < sum.abs() {
            break;
        }
    }
    sum
}

/// `boost::math::binomial_coefficient<double>(n, k)`.
fn binomial_coefficient(n: u32, k: u32) -> f64 {
    if k > n {
        return f64::NAN;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    if k == 1 || k == n - 1 {
        return f64::from(n);
    }
    let mut result;
    if n <= MAX_FACTORIAL {
        result = unchecked_factorial(n);
        result /= unchecked_factorial(n - k);
        result /= unchecked_factorial(k);
    } else {
        if k < n - k {
            result = f64::from(k) * beta(f64::from(k), f64::from(n - k + 1));
        } else {
            result = f64::from(n - k) * beta(f64::from(k + 1), f64::from(n - k));
        }
        if result == 0.0 {
            return f64::INFINITY;
        }
        result = 1.0 / result;
    }
    (result - 0.5).ceil()
}

/// `detail::binomial_ccdf(n, k, x, y)`.
fn binomial_ccdf(n: f64, k: f64, x: f64, y: f64) -> f64 {
    let mut result = x.powf(n);
    if result > MIN_VALUE {
        let mut term = result;
        let mut i = itrunc(n - 1.0) as u32;
        while f64::from(i) > k {
            term *= (f64::from(i + 1) * y) / ((n - f64::from(i)) * x);
            result += term;
            i -= 1;
        }
    } else {
        let mut start = itrunc(n * x) as i32;
        if f64::from(start) <= k + 1.0 {
            start = itrunc(k + 2.0) as i32;
        }
        let nn = itrunc(n) as u32;
        result = x.powf(f64::from(start))
            * y.powf(n - f64::from(start))
            * binomial_coefficient(nn, start as u32);
        if result == 0.0 {
            let mut i = (start - 1) as u32;
            while f64::from(i) > k {
                // `result += pow(x,i) * pow(y,n-i) * C(n,i)`: compound-assign of a product fuses.
                let fi = f64::from(i);
                result = (x.powf(fi) * y.powf(n - fi)).mul_add(binomial_coefficient(nn, i), result);
                i -= 1;
            }
        } else {
            let mut term = result;
            let start_term = result;
            let mut i = (start - 1) as u32;
            while f64::from(i) > k {
                term *= (f64::from(i + 1) * y) / ((n - f64::from(i)) * x);
                result += term;
                i -= 1;
            }
            term = start_term;
            let mut i = (start + 1) as u32;
            while f64::from(i) <= n {
                term *= (n - f64::from(i) + 1.0) * x / (f64::from(i) * y);
                result += term;
                i += 1;
            }
        }
    }
    result
}

/// `detail::ibeta_large_ab(a, b, x, y, invert, normalised = true)`.
fn ibeta_large_ab(a: f64, b: f64, x: f64, y: f64, invert: bool) -> f64 {
    let x0 = a / (a + b);
    let y0 = b / (a + b);
    let mut nu = x0.mul_add((x / x0).ln(), y0 * (y / y0).ln());
    if nu > 0.0 || x == x0 || y == y0 {
        nu = 0.0;
    }
    nu = (-2.0 * nu).sqrt();
    if nu != 0.0 && nu / (x - x0) < 0.0 {
        nu = -nu;
    }
    let arg = -nu * ((a + b) / 2.0).sqrt();
    if invert {
        (1.0 + erf(arg)) / 2.0
    } else {
        erfc(arg) / 2.0
    }
}

/// `detail::ibeta_imp(a, b, x, inv, normalised = true)`.
fn ibeta_imp(mut a: f64, mut b: f64, mut x: f64, inv: bool) -> f64 {
    let mut invert = inv;
    let mut y = 1.0 - x;
    let fract;
    if !a.is_finite() || !b.is_finite() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        if b == 0.0 {
            return f64::NAN;
        }
        if b > 0.0 {
            return if inv { 0.0 } else { 1.0 };
        }
    } else if b == 0.0 && a > 0.0 {
        return if inv { 1.0 } else { 0.0 };
    }
    if x == 0.0 {
        return if invert { 1.0 } else { 0.0 };
    }
    if x == 1.0 {
        return if !invert { 1.0 } else { 0.0 };
    }
    if a == 0.5 && b == 0.5 {
        return if invert {
            y.sqrt().asin() / FRAC_PI_2
        } else {
            x.sqrt().asin() / FRAC_PI_2
        };
    }
    if a == 1.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut x, &mut y);
        invert = !invert;
    }
    if b == 1.0 {
        if a == 1.0 {
            return if invert { y } else { x };
        }
        return if y < 0.5 {
            if invert {
                -(a * (-y).ln_1p()).exp_m1()
            } else {
                (a * (-y).ln_1p()).exp()
            }
        } else if invert {
            -powm1(x, a)
        } else {
            x.powf(a)
        };
    }

    // Reversing the arguments for the series: `swap(a,b); swap(x,y); invert = !invert`.
    macro_rules! flip {
        () => {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut x, &mut y);
            invert = !invert;
        };
    }
    // The "invert ? 1 - series : series" shape every series branch shares.
    macro_rules! series_or_complement {
        ($call:expr, $s0:ident) => {{
            if !invert {
                let $s0 = 0.0;
                $call
            } else {
                let $s0 = -1.0;
                invert = false;
                -$call
            }
        }};
    }

    if a.min(b) <= 1.0 {
        if x > 0.5 {
            flip!();
        }
        if a.max(b) <= 1.0 {
            if a >= 0.2f64.min(b) || x.powf(a) <= 0.9 {
                fract = series_or_complement!(ibeta_series(a, b, x, s0), s0);
            } else {
                flip!();
                if y >= 0.3 {
                    fract = series_or_complement!(ibeta_series(a, b, x, s0), s0);
                } else {
                    let mut f = ibeta_a_step(a, b, x, y, 20);
                    if !invert {
                        f = beta_small_b_large_a_series(a + 20.0, b, x, y, f, 1.0);
                    } else {
                        f -= 1.0;
                        invert = false;
                        f = -beta_small_b_large_a_series(a + 20.0, b, x, y, f, 1.0);
                    }
                    fract = f;
                }
            }
        } else if b <= 1.0 || (x < 0.1 && (b * x).powf(a) <= 0.7) {
            fract = series_or_complement!(ibeta_series(a, b, x, s0), s0);
        } else {
            flip!();
            if y >= 0.3 {
                fract = series_or_complement!(ibeta_series(a, b, x, s0), s0);
            } else if a >= 15.0 {
                fract = series_or_complement!(beta_small_b_large_a_series(a, b, x, y, s0, 1.0), s0);
            } else {
                let mut f = ibeta_a_step(a, b, x, y, 20);
                if !invert {
                    f = beta_small_b_large_a_series(a + 20.0, b, x, y, f, 1.0);
                } else {
                    f -= 1.0;
                    invert = false;
                    f = -beta_small_b_large_a_series(a + 20.0, b, x, y, f, 1.0);
                }
                fract = f;
            }
        }
    } else {
        let lambda = if a < b {
            a - (a + b) * x
        } else {
            (a + b) * y - b
        };
        if lambda < 0.0 {
            flip!();
        }
        if b < 40.0 {
            if a.floor() == a && b.floor() == b && a < f64::from(i32::MAX - 100) && y != 1.0 {
                let k = a - 1.0;
                let n = b + k;
                fract = binomial_ccdf(n, k, x, y);
            } else if b * x <= 0.7 {
                fract = series_or_complement!(ibeta_series(a, b, x, s0), s0);
            } else if a > 15.0 {
                let mut n = itrunc(b.floor()) as i32;
                if f64::from(n) == b {
                    n -= 1;
                }
                let bbar = b - f64::from(n);
                let mut f = ibeta_a_step(bbar, a, y, x, n);
                f = beta_small_b_large_a_series(a, bbar, x, y, f, 1.0);
                fract = f; // `/= prefix`, which is 1 when normalised
            } else {
                let mut n = itrunc(b.floor()) as i32;
                let mut bbar = b - f64::from(n);
                if bbar <= 0.0 {
                    n -= 1;
                    bbar += 1.0;
                }
                let mut f = ibeta_a_step(bbar, a, y, x, n);
                f += ibeta_a_step(a, bbar, x, y, 20);
                if invert {
                    f -= 1.0;
                }
                f = beta_small_b_large_a_series(a + 20.0, bbar, x, y, f, 1.0);
                if invert {
                    f = -f;
                    invert = false;
                }
                fract = f;
            }
        } else {
            let mut use_asym = false;
            let ma = a.max(b);
            let xa = if ma == a { x } else { y };
            let saddle = ma / (a + b);
            let mut powers = 0.0;
            // `1e-5f / tools::epsilon<T>()`: a *float* literal promoted to double.
            if ma > f64::from(1e-5f32) / EPSILON
                && ma / a.min(b) < if xa < saddle { 2.0 } else { 15.0 }
            {
                if a == b {
                    use_asym = true;
                } else {
                    powers = ((x / (a / (a + b))).ln())
                        .mul_add(a, (y / (b / (a + b))).ln() * b)
                        .exp();
                    if powers < EPSILON {
                        use_asym = true;
                    }
                }
            }
            if use_asym {
                let f = ibeta_large_ab(a, b, x, y, invert);
                if f * EPSILON < powers {
                    fract = ibeta_fraction2(a, b, x, y);
                } else {
                    invert = false;
                    fract = f;
                }
            } else {
                fract = ibeta_fraction2(a, b, x, y);
            }
        }
    }
    if invert { 1.0 - fract } else { fract }
}

/// `boost::math::ibeta(a, b, x)`: the regularized incomplete beta `I_x(a, b)`
/// (SciPy `betainc`).
pub fn ibeta(a: f64, b: f64, x: f64) -> f64 {
    ibeta_imp(a, b, x, false)
}

/// `boost::math::ibetac(a, b, x)` = `1 - I_x(a, b)`, computed directly
/// (SciPy `betaincc`).
pub fn ibetac(a: f64, b: f64, x: f64) -> f64 {
    ibeta_imp(a, b, x, true)
}
