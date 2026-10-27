// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SciPy's cephes functions (`scipy/xsf`, `include/xsf/cephes/*.h`) for `double` —
//! what `special.erf`, `erfc`, `ndtr`, `ndtri`, `gammainc`, `gammaincc`,
//! `gammaln`, `gamma` and `stats.chi2.sf` (`chdtrc`) evaluate.
//!
//! As with the Boost port, the arithmetic follows clang's `-ffp-contract=on`:
//! `a*b + c` (or `a - b*c`, `s -= c*b`) inside one C++ expression is a single
//! `fmac` call (fused on Apple, plain on the Linux reference); nothing fuses across
//! statements. The polynomial helpers are
//! cephes' own plain Horner loops (`polevl`, `p1evl`, `ratevl`), every step fused.
//! Note cephes carries its **own** `log1p`, `expm1`, `log1pmx` and Lanczos — they
//! are not libm's or Boost's.

use crate::cephes_tables::{IGAM_ASYMP_D, IGAM_ASYMP_K, IGAM_ASYMP_N};

use crate::Contract;
use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

// ── const.h ──
const MACHEP: f64 = 1.1102230246251565404236316680908203125E-16; // 2**-53
const MAXLOG: f64 = 7.097827128933839730962063185871E2;
const SQRT2PI: f64 = 2.506628274631000502415765284811045253007;
const LOGPI: f64 = 1.144729885849400174143427351353058711647;
const MAXGAM: f64 = 171.624376956302725;
const SCIPY_EULER: f64 = 0.577215664901532860606512090082402431;
const MAXITER: u64 = 500;

// ── polevl.h ──

/// `polevl(x, coef, N)`: `coef[0]` is the highest power; `N + 1` coefficients.
#[inline]
fn polevl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..=n] {
        ans = ans.fmac(x, c);
    }
    ans
}

/// `p1evl(x, coef, N)`: as [`polevl`] with an implied leading `1.0`; `N` coefficients.
#[inline]
fn p1evl(x: f64, coef: &[f64], n: usize) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..n] {
        ans = ans.fmac(x, c);
    }
    ans
}

/// `ratevl(x, num, M, denom, N)`: a rational evaluated in `x`, or in `1/x` when
/// `|x| > 1` (then scaled by `x^(M-N)`).
fn ratevl(x: f64, num: &[f64], m: usize, denom: &[f64], n: usize) -> f64 {
    let absx = x.abs();
    let (y, rev) = if absx > 1.0 {
        (1.0 / x, true)
    } else {
        (x, false)
    };
    let eval = |c: &[f64], k: usize| -> f64 {
        if rev {
            let mut ans = c[k];
            for i in (0..k).rev() {
                ans = ans.fmac(y, c[i]);
            }
            ans
        } else {
            let mut ans = c[0];
            for &ci in &c[1..=k] {
                ans = ans.fmac(y, ci);
            }
            ans
        }
    };
    let num_ans = eval(num, m);
    let denom_ans = eval(denom, n);
    if absx > 1.0 {
        let i = m as i32 - n as i32;
        // `pow(x, i)` with an int exponent promotes to `pow(double, double)`.
        rlx_libm::reference::pow(x, f64::from(i)) * num_ans / denom_ans
    } else {
        num_ans / denom_ans
    }
}

// ── ndtr.h ──

const NDTR_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
const NDTR_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];
const NDTR_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
const NDTR_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];
const NDTR_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
const NDTR_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];

/// cephes `erfc(a)` = SciPy `special.erfc`.
pub fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = if a < 0.0 { -a } else { a };
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let mut z = -a * a;
    if z < -MAXLOG {
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    z = rlx_libm::reference::exp(z);
    let (p, q) = if x < 8.0 {
        (polevl(x, &NDTR_P, 8), p1evl(x, &NDTR_Q, 8))
    } else {
        (polevl(x, &NDTR_R, 5), p1evl(x, &NDTR_S, 6))
    };
    let mut y = (z * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    if y != 0.0 {
        return y;
    }
    if a < 0.0 { 2.0 } else { 0.0 }
}

/// cephes `erf(x)` = SciPy `special.erf`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &NDTR_T, 4) / p1evl(z, &NDTR_U, 5)
}

/// cephes `ndtr(a)` = SciPy `special.ndtr` / `stats.norm.cdf`.
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < 1.0 {
        0.5f64.fmac(erf(x), 0.5) // `0.5 + 0.5 * erf(x)`
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

// ── ndtri.h ──

const NDTRI_P0: [f64; 5] = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];
const NDTRI_Q0: [f64; 8] = [
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];
const NDTRI_P1: [f64; 9] = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];
const NDTRI_Q1: [f64; 8] = [
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];
const NDTRI_P2: [f64; 9] = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];
const NDTRI_Q2: [f64; 8] = [
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
];

/// cephes `ndtri(y0)` = SciPy `special.ndtri` / `stats.norm.ppf`.
pub fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&y0) {
        return f64::NAN;
    }
    let mut code = 1;
    let mut y = y0;
    if y > (1.0 - 0.13533528323661269189) {
        y = 1.0 - y;
        code = 0;
    }
    if y > 0.13533528323661269189 {
        y -= 0.5;
        let y2 = y * y;
        // `y + y * (y2 * P / Q)`
        let mut x = y.fmac(y2 * polevl(y2, &NDTRI_P0, 4) / p1evl(y2, &NDTRI_Q0, 8), y);
        x *= SQRT2PI;
        return x;
    }
    let x = (-2.0 * rlx_libm::reference::log(y)).sqrt();
    let x0 = x - rlx_libm::reference::log(x) / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &NDTRI_P1, 8) / p1evl(z, &NDTRI_Q1, 8)
    } else {
        z * polevl(z, &NDTRI_P2, 8) / p1evl(z, &NDTRI_Q2, 8)
    };
    let mut x = x0 - x1;
    if code != 0 {
        x = -x;
    }
    x
}

// ── trig.h ──

fn sinpi(mut x: f64) -> f64 {
    let mut s = 1.0f64;
    if x.is_sign_negative() {
        x = -x;
        s = -1.0;
    }
    let r = x % 2.0;
    if r < 0.5 {
        s * rlx_libm::reference::sin(PI * r)
    } else if r > 1.5 {
        s * rlx_libm::reference::sin(PI * (r - 2.0))
    } else {
        -s * rlx_libm::reference::sin(PI * (r - 1.0))
    }
}

// ── gamma.h ──

const GAMMA_P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];
const GAMMA_Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];
const GAMMA_STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];
const MAXSTIR: f64 = 143.01608;

fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let mut w = 1.0 / x;
    w = w.fmac(polevl(w, &GAMMA_STIR, 4), 1.0); // `1.0 + w * polevl(...)`
    let mut y = rlx_libm::reference::exp(x);
    if x > MAXSTIR {
        let v = rlx_libm::reference::pow(x, 0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = rlx_libm::reference::pow(x, x - 0.5) / y;
    }
    SQRT2PI * y * w
}

/// cephes `Gamma(x)` = SciPy `special.gamma`.
pub fn gamma(mut x: f64) -> f64 {
    if !x.is_finite() {
        return if x > 0.0 { x } else { f64::NAN };
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    let mut sgngam = 1.0f64;
    let q = x.abs();
    if q > 33.0 {
        let z = if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                return f64::NAN;
            }
            let i = p as i32;
            if i & 1 == 0 {
                sgngam = -1.0;
            }
            let mut zz = q - p;
            if zz > 0.5 {
                p += 1.0;
                zz = q - p;
            }
            zz = q * sinpi(zz);
            if zz == 0.0 {
                return sgngam * f64::INFINITY;
            }
            zz = zz.abs();
            PI / (zz * stirf(q))
        } else {
            stirf(x)
        };
        return sgngam * z;
    }
    let mut z = 1.0f64;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1.0E-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0e-9 {
            return gamma_small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    let p = polevl(x, &GAMMA_P, 6);
    let q = polevl(x, &GAMMA_Q, 7);
    z * p / q
}

fn gamma_small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        f64::NAN
    } else {
        z / (0.5772156649015329f64.fmac(x, 1.0) * x)
    }
}

const GAMMA_A: [f64; 5] = [
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2,
];
const GAMMA_B: [f64; 6] = [
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5,
];
const GAMMA_C: [f64; 6] = [
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6,
];
const LS2PI: f64 = 0.91893853320467274178;
const MAXLGM: f64 = 2.556348e305;

fn lgam_large_x(x: f64) -> f64 {
    // `(x - 0.5) * log(x) - x + LS2PI`
    let q = (x - 0.5).fmac(rlx_libm::reference::log(x), -x) + LS2PI;
    if x > 1.0e8 {
        return q;
    }
    let mut p = 1.0 / (x * x);
    p = 7.9365079365079365079365e-4f64
        .fmac(p, -2.7777777777777777777778e-3)
        .fmac(p, 0.0833333333333333333333)
        / x;
    q + p
}

/// cephes `lgam_sgn(x, &sign)`: `ln|Γ(x)|` and the sign of `Γ(x)`.
pub fn lgam_sgn(x: f64) -> (f64, i32) {
    let mut sign = 1;
    if !x.is_finite() {
        return (x, sign);
    }
    if x < -34.0 {
        let q = -x;
        let (w, _) = lgam_sgn(q);
        let mut p = q.floor();
        if p == q {
            return (f64::INFINITY, sign);
        }
        let i = p as i32;
        sign = if i & 1 == 0 { -1 } else { 1 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * sinpi(z);
        if z == 0.0 {
            return (f64::INFINITY, sign);
        }
        return (LOGPI - rlx_libm::reference::log(z) - w, sign);
    }
    if x < 13.0 {
        let mut z = 1.0f64;
        let mut p = 0.0f64;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (f64::INFINITY, sign);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            sign = -1;
            z = -z;
        } else {
            sign = 1;
        }
        if u == 2.0 {
            return (rlx_libm::reference::log(z), sign);
        }
        p -= 2.0;
        let xx = x + p;
        let pp = xx * polevl(xx, &GAMMA_B, 5) / p1evl(xx, &GAMMA_C, 6);
        return (rlx_libm::reference::log(z) + pp, sign);
    }
    if x > MAXLGM {
        return (f64::from(sign) * f64::INFINITY, sign);
    }
    if x >= 1000.0 {
        return (lgam_large_x(x), sign);
    }
    let q = (x - 0.5).fmac(rlx_libm::reference::log(x), -x) + LS2PI;
    let p = 1.0 / (x * x);
    (q + polevl(p, &GAMMA_A, 4) / x, sign)
}

/// cephes `lgam(x)` = SciPy `special.gammaln`.
pub fn lgam(x: f64) -> f64 {
    lgam_sgn(x).0
}

// ── zeta.h ──

const ZETA_A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9,
    7.47242496e10,
    -2.950130727918164224e12,
    1.1646782814350067249e14,
    -4.5979787224074726105e15,
    1.8152105401943546773e17,
    -7.1661652561756670113e18,
];

/// cephes Hurwitz `zeta(x, q)` = SciPy `special.zeta(x, q)`.
pub fn zeta(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            return f64::NAN;
        }
    }
    if q > 1e8 {
        return (1.0 / (x - 1.0) + 1.0 / (2.0 * q)) * rlx_libm::reference::pow(q, 1.0 - x);
    }
    let mut s = rlx_libm::reference::pow(q, -x);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0f64;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = rlx_libm::reference::pow(a, -x);
        s += b;
        if (b / s).abs() < MACHEP {
            return s;
        }
    }
    let w = a;
    s += b * w / (x - 1.0);
    s = (-0.5f64).fmac(b, s); // `s -= 0.5 * b`
    a = 1.0;
    let mut k = 0.0f64;
    for za in ZETA_A {
        a *= x + k;
        b /= w;
        let t = a * b / za;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    s
}

// ── unity.h ──

const UNITY_LP: [f64; 7] = [
    4.5270000862445199635215E-5,
    4.9854102823193375972212E-1,
    6.5787325942061044846969E0,
    2.9911919328553073277375E1,
    6.0949667980987787057556E1,
    5.7112963590585538103336E1,
    2.0039553499201281259648E1,
];
const UNITY_LQ: [f64; 6] = [
    1.5062909083469192043167E1,
    8.3047565967967209469434E1,
    2.2176239823732856465394E2,
    3.0909872225312059774938E2,
    2.1642788614495947685003E2,
    6.0118660497603843919306E1,
];

/// cephes `log1p(x)` (its own rational, not libm's).
pub fn log1p(x: f64) -> f64 {
    let z = 1.0 + x;
    if z < FRAC_1_SQRT_2 || z > SQRT_2 {
        return rlx_libm::reference::log(z);
    }
    let z = x * x;
    // `-0.5 * z + x * (z * P / Q)`
    let z = (-0.5f64).fmac(
        z,
        x * (z * polevl(x, &UNITY_LP, 6) / p1evl(x, &UNITY_LQ, 6)),
    );
    x + z
}

/// cephes `log1pmx(x)` = `log(1 + x) - x`.
pub fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res = 0.0f64;
        let mut n: u64 = 2;
        while n < MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
            n += 1;
        }
        res
    } else {
        log1p(x) - x
    }
}

const UNITY_EP: [f64; 3] = [
    1.2617719307481059087798E-4,
    3.0299440770744196129956E-2,
    9.9999999999999999991025E-1,
];
const UNITY_EQ: [f64; 4] = [
    3.0019850513866445504159E-6,
    2.5244834034968410419224E-3,
    2.2726554820815502876593E-1,
    2.0000000000000000000897E0,
];

/// cephes `expm1(x)` (its own rational, not libm's).
pub fn expm1(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() || x > 0.0 {
            return x;
        }
        return -1.0;
    }
    if !(-0.5..=0.5).contains(&x) {
        return rlx_libm::reference::exp(x) - 1.0;
    }
    let xx = x * x;
    let mut r = x * polevl(xx, &UNITY_EP, 2);
    r /= polevl(xx, &UNITY_EQ, 3) - r;
    r + r
}

fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -SCIPY_EULER * x;
    let mut xfac = -x;
    for n in 2..42 {
        xfac *= -x;
        let coeff = zeta(f64::from(n), 1.0) * xfac / f64::from(n);
        res += coeff;
        if coeff.abs() < MACHEP * res.abs() {
            break;
        }
    }
    res
}

/// cephes `lgam1p(x)` = `lgam(x + 1)`, accurate near 0 and 1.
pub fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        rlx_libm::reference::log(x) + lgam1p_taylor(x - 1.0)
    } else {
        lgam(x + 1.0)
    }
}

// ── lanczos.h (cephes' copy; coefficients highest power first) ──

const LANCZOS_G: f64 = 6.024680040776729583740234375;
const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
];
const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];

fn lanczos_sum_expg_scaled(x: f64) -> f64 {
    ratevl(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        12,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
        12,
    )
}

// ── igam.h ──

const IGAM_MAXITER: i32 = 2000;
const IGAM_SMALL: f64 = 20.0;
const IGAM_LARGE: f64 = 200.0;
const IGAM_SMALLRATIO: f64 = 0.3;
const IGAM_LARGERATIO: f64 = 4.5;
const IGAM_BIG: f64 = 4.503599627370496e15;
const IGAM_BIGINV: f64 = 2.22044604925031308085e-16;

/// `x^a e^{-x} / Γ(a)`.
fn igam_fac(a: f64, x: f64) -> f64 {
    if (a - x).abs() > 0.4 * a.abs() {
        // `a * log(x) - x - lgam(a)`
        let ax = a.fmac(rlx_libm::reference::log(x), -x) - lgam(a);
        if ax < -MAXLOG {
            return 0.0;
        }
        return rlx_libm::reference::exp(ax);
    }
    let fac = a + LANCZOS_G - 0.5;
    let mut res = (fac / rlx_libm::reference::exp(1f64)).sqrt() / lanczos_sum_expg_scaled(a);
    if a < 200.0 && x < 200.0 {
        res *= rlx_libm::reference::exp(a - x) * rlx_libm::reference::pow(x / fac, a);
    } else {
        let num = x - a - LANCZOS_G + 0.5;
        // `a * log1pmx(num/fac) + x * (0.5 - g) / fac`
        res *= a
            .fmac(log1pmx(num / fac), x * (0.5 - LANCZOS_G) / fac)
            .exp();
    }
    res
}

fn igamc_continued_fraction(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0f64;
    let mut pkm2 = 1.0f64;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    for _ in 0..IGAM_MAXITER {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        let yc = y * c;
        let pk = pkm1.fmac(z, -(pkm2 * yc)); // `pkm1 * z - pkm2 * yc`
        let qk = qkm1.fmac(z, -(qkm2 * yc));
        let t;
        if qk != 0.0 {
            let r = pk / qk;
            t = ((ans - r) / r).abs();
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if pk.abs() > IGAM_BIG {
            pkm2 *= IGAM_BIGINV;
            pkm1 *= IGAM_BIGINV;
            qkm2 *= IGAM_BIGINV;
            qkm1 *= IGAM_BIGINV;
        }
        if t <= MACHEP {
            break;
        }
    }
    ans * ax
}

fn igam_series(a: f64, x: f64) -> f64 {
    let ax = igam_fac(a, x);
    if ax == 0.0 {
        return 0.0;
    }
    let mut r = a;
    let mut c = 1.0f64;
    let mut ans = 1.0f64;
    for _ in 0..IGAM_MAXITER {
        r += 1.0;
        c *= x / r;
        ans += c;
        if c <= MACHEP * ans {
            break;
        }
    }
    ans * ax / a
}

fn igamc_series(a: f64, x: f64) -> f64 {
    let mut fac = 1.0f64;
    let mut sum = 0.0f64;
    for n in 1..IGAM_MAXITER {
        fac *= -x / f64::from(n);
        let term = fac / (a + f64::from(n));
        sum += term;
        if term.abs() <= MACHEP * sum.abs() {
            break;
        }
    }
    let logx = rlx_libm::reference::log(x);
    let term = -expm1(a.fmac(logx, -lgam1p(a)));
    // `term - exp(a * logx - lgam(a)) * sum`
    (-(rlx_libm::reference::exp(a.fmac(logx, -lgam(a))))).fmac(sum, term)
}

fn asymptotic_series(a: f64, x: f64, igam: bool) -> f64 {
    let sgn: f64 = if igam { -1.0 } else { 1.0 };
    let lambda = x / a;
    let sigma = (x - a) / a;
    let eta = if lambda > 1.0 {
        (-2.0 * log1pmx(sigma)).sqrt()
    } else if lambda < 1.0 {
        -(-2.0 * log1pmx(sigma)).sqrt()
    } else {
        0.0
    };
    let mut res = 0.5 * erfc(sgn * eta * (a / 2.0).sqrt());
    let mut etapow = [0.0f64; IGAM_ASYMP_N];
    etapow[0] = 1.0;
    let mut maxpow = 0usize;
    let mut sum = 0.0f64;
    let mut afac = 1.0f64;
    let mut absoldterm = f64::INFINITY;
    for k in 0..IGAM_ASYMP_K {
        let mut ck = IGAM_ASYMP_D[k][0];
        for n in 1..IGAM_ASYMP_N {
            if n > maxpow {
                etapow[n] = eta * etapow[n - 1];
                maxpow += 1;
            }
            let ckterm = IGAM_ASYMP_D[k][n] * etapow[n];
            ck += ckterm;
            if ckterm.abs() < MACHEP * ck.abs() {
                break;
            }
        }
        let term = ck * afac;
        let absterm = term.abs();
        if absterm > absoldterm {
            break;
        }
        sum += term;
        if absterm < MACHEP * sum.abs() {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    res += sgn * rlx_libm::reference::exp(-0.5 * a * eta * eta) * sum / (2.0 * PI * a).sqrt();
    res
}

/// cephes `igam(a, x)` = SciPy `special.gammainc` (regularized lower).
pub fn igam(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    } else if a == 0.0 {
        return if x > 0.0 { 1.0 } else { f64::NAN };
    } else if x == 0.0 {
        return 0.0;
    } else if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 0.0 };
    } else if x.is_infinite() {
        return 1.0;
    }
    let absxma_a = (x - a).abs() / a;
    if (a > IGAM_SMALL && a < IGAM_LARGE && absxma_a < IGAM_SMALLRATIO)
        || (a > IGAM_LARGE && absxma_a < IGAM_LARGERATIO / a.sqrt())
    {
        return asymptotic_series(a, x, true);
    }
    if x > 1.0 && x > a {
        return 1.0 - igamc(a, x);
    }
    igam_series(a, x)
}

/// cephes `igamc(a, x)` = SciPy `special.gammaincc` (regularized upper).
pub fn igamc(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 || a < 0.0 {
        return f64::NAN;
    } else if a == 0.0 {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    } else if x == 0.0 {
        return 1.0;
    } else if a.is_infinite() {
        return if x.is_infinite() { f64::NAN } else { 1.0 };
    } else if x.is_infinite() {
        return 0.0;
    }
    let absxma_a = (x - a).abs() / a;
    if (a > IGAM_SMALL && a < IGAM_LARGE && absxma_a < IGAM_SMALLRATIO)
        || (a > IGAM_LARGE && absxma_a < IGAM_LARGERATIO / a.sqrt())
    {
        return asymptotic_series(a, x, false);
    }
    if x > 1.1 {
        if x < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_continued_fraction(a, x)
        }
    } else if x <= 0.5 {
        if -0.4 / rlx_libm::reference::log(x) < a {
            1.0 - igam_series(a, x)
        } else {
            igamc_series(a, x)
        }
    } else if x * 1.1 < a {
        1.0 - igam_series(a, x)
    } else {
        igamc_series(a, x)
    }
}

/// cephes `chdtrc(df, x)` = SciPy `stats.chi2.sf(x, df)`.
pub fn chdtrc(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    igamc(df / 2.0, x / 2.0)
}
