// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `erfc`: CORE-MATH's correctly rounded binary64 `erfc` (Sibidanov,
//! Zimmermann, Hubrecht, Jeannerod; MIT; `sysdeps/ieee754/dbl-64/s_erfc.c`,
//! `s_erf_common.c`), statement for statement. x86-64 builds it for the baseline
//! ISA (no FMA multiarch variant), so only the explicit `fma` calls fuse.

#[path = "erfc_data.rs"]
mod data;
use data::*;

#[inline]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let e = hi - a;
    (hi, b - e)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let aa = hi - b;
    let bb = hi - aa;
    let da = a - aa;
    let db = b - bb;
    (hi, da + db)
}

#[inline]
fn a_mul(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    (hi, fma(a, b, -hi))
}

/// `a·(bh + bl)`.
#[inline]
fn s_mul(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = a_mul(a, bh);
    (hi, fma(a, bl, lo))
}

/// `(ah + al)·(bh + bl) − al·bl`.
#[inline]
fn d_mul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, mut lo) = a_mul(ah, bh);
    lo = fma(ah, bl, lo);
    lo = fma(al, bh, lo);
    (hi, lo)
}

#[inline]
fn fast_sum(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = fast_two_sum(a, bh);
    (hi, lo + bl)
}

/// glibc `__ldexp` / `scalbn` (one rounding, also into the subnormals).
fn ldexp(x: f64, mut n: i32) -> f64 {
    let mut y = x;
    let p = |e: i32| f64::from_bits(((0x3ff + e) as u64) << 52);
    if n > 1023 {
        y *= p(1023);
        n -= 1023;
        if n > 1023 {
            y *= p(1023);
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // 2^-1022 · 2^53: stays normal, so the final multiply is the only rounding
        y *= p(-1022) * p(53);
        n += 1022 - 53;
        if n < -1022 {
            y *= p(-1022) * p(53);
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * p(n)
}

/// `__cr_erf_fast(z)` → `(h, l, err)` for `0 ≤ z ≤ 0x1.7afb48dc96626p+2`.
fn cr_erf_fast(z: f64) -> (f64, f64, f64) {
    if z < 0.0625 {
        let (z2h, z2l) = a_mul(z, z);
        let z4 = z2h * z2h;
        let c9 = fma(C0[7], z2h, C0[6]);
        let mut c5 = fma(C0[5], z2h, C0[4]);
        c5 = fma(c9, z4, c5);
        let (th, tl) = a_mul(z2h, c5);
        let (h, mut l) = fast_two_sum(C0[2], th);
        l += tl + C0[3];
        let h_copy = h;
        let (th, mut tl) = a_mul(z2h, h);
        tl += fma(z2h, l, C0[1]);
        let (h, mut l) = fast_two_sum(C0[0], th);
        l += fma(z2l, h_copy, tl);
        let (h, tl) = a_mul(h, z);
        let l = fma(l, z, tl);
        return (h, l, f64::from_bits(0x3ba7_8000_0000_0000)); // 0x1.78p-69
    }
    let v = (16.0 * z).floor();
    let i = (16.0 * z) as u32;
    let z = (z - 0.03125) - 0.0625 * v;
    let c = &C[(i - 1) as usize];
    let z2 = z * z;
    let z4 = z2 * z2;
    let c9 = fma(c[12], z, c[11]);
    let mut c7 = fma(c[10], z, c[9]);
    let c5 = fma(c[8], z, c[7]);
    let (mut c3h, mut c3l) = fast_two_sum(c[5], z * c[6]);
    c7 = fma(c9, z2, c7);
    let (h3, tl) = fast_two_sum(c3h, c5 * z2);
    c3h = h3;
    c3l += tl;
    let (h3, tl) = fast_two_sum(c3h, c7 * z4);
    c3h = h3;
    c3l += tl;
    let (th, tl) = a_mul(z, c3h);
    let (c2h, mut c2l) = fast_two_sum(c[4], th);
    c2l += fma(z, c3l, tl);
    let (th, tl) = a_mul(z, c2h);
    let (h, mut l) = fast_two_sum(c[2], th);
    l += tl + fma(z, c2l, c[3]);
    let (th, mut tl) = a_mul(z, h);
    tl = fma(z, l, tl);
    let (h, mut l) = fast_two_sum(c[0], th);
    l += tl + c[1];
    (h, l, f64::from_bits(0x3ba1_1000_0000_0000)) // 0x1.11p-69
}

/// `__cr_erf_accurate_tiny(z, false)` for `|z| < 1/8`.
fn cr_erf_accurate_tiny(z: f64) -> (f64, f64) {
    let z2 = z * z;
    let mut h = P[21 / 2 + 4];
    let mut a = 19;
    while a > 11 {
        h = fma(h, z2, P[a / 2 + 4]);
        a -= 2;
    }
    let mut l = 0.0f64;
    let mut a = 11;
    while a > 7 {
        let (th, mut tl) = a_mul(h, z);
        tl = fma(l, z, tl);
        let (hh, ll) = a_mul(th, z);
        h = hh;
        l = fma(tl, z, ll);
        let (hh, t) = fast_two_sum(P[a / 2 + 4], h);
        h = hh;
        l += t;
        a -= 2;
    }
    let mut a: i32 = 7;
    while a >= 1 {
        let au = a as usize;
        let (th, mut tl) = a_mul(h, z);
        tl = fma(l, z, tl);
        let (hh, ll) = a_mul(th, z);
        h = hh;
        l = fma(tl, z, ll);
        let (hh, t) = fast_two_sum(P[au - 1], h);
        h = hh;
        l += P[au] + t;
        a -= 2;
    }
    let (h, tl) = a_mul(h, z);
    (h, fma(l, z, tl))
}

/// `cr_erf_accurate(z)` for `0 ≤ z ≤ 0x1.7afb48dc96626p+2`.
fn cr_erf_accurate(z: f64) -> (f64, f64) {
    if z < 0.125 {
        return cr_erf_accurate_tiny(z);
    }
    let v = (8.0 * z).floor();
    let i = (8.0 * z) as u32;
    let z = (z - 0.0625) - 0.125 * v;
    let p = &C2[(i - 1) as usize];
    let mut h = p[26];
    for j in (11..=17).rev() {
        h = fma(h, z, p[8 + j]);
    }
    let mut l = 0.0f64;
    for j in (8..=10).rev() {
        let (th, mut tl) = a_mul(h, z);
        tl = fma(l, z, tl);
        let (hh, ll) = two_sum(p[8 + j], th);
        h = hh;
        l = ll + tl;
    }
    for j in (0..=7).rev() {
        let (th, mut tl) = a_mul(h, z);
        tl = fma(l, z, tl);
        let (hh, ll) = two_sum(p[2 * j], th);
        h = hh;
        l = ll + (p[2 * j + 1] + tl);
    }
    (h, l)
}

/// `q_1(zh, zl)`: `exp(zh + zl)` for tiny arguments.
fn q_1(zh: f64, zl: f64) -> (f64, f64) {
    let z = zh + zl;
    let mut q = fma(Q_1[4], zh, Q_1[3]);
    q = fma(q, z, Q_1[2]);
    let (hi, lo) = fast_two_sum(Q_1[1], q * z);
    let (hi, lo) = d_mul(zh, zl, hi, lo);
    fast_sum(Q_1[0], hi, lo)
}

/// `exp_1(xh, xl)`.
fn exp_1(xh: f64, xl: f64) -> (f64, f64) {
    const INVLOG2: f64 = f64::from_bits(0x40b7_1547_652b_82fe); // 0x1.71547652b82fep+12
    const LOG2H: f64 = f64::from_bits(0x3f26_2e42_fefa_39ef); // 0x1.62e42fefa39efp-13
    const LOG2L: f64 = f64::from_bits(0x3bba_bc9e_3b39_803f); // 0x1.abc9e3b39803fp-68
    let k = (xh * INVLOG2).round_ties_even();
    let (kh, kl) = s_mul(k, LOG2H, LOG2L);
    let (yh, mut yl) = fast_two_sum(xh - kh, xl);
    yl -= kl;
    let kk = k as i64;
    let m = (kk >> 12) + 0x3ff;
    let i2 = ((kk >> 6) & 0x3f) as usize;
    let i1 = (kk & 0x3f) as usize;
    let (hi, lo) = d_mul(T2[i1][0], T2[i1][1], T1[i2][0], T1[i2][1]);
    let (qh, ql) = q_1(yh, yl);
    let (hi, lo) = d_mul(hi, lo, qh, ql);
    let df = f64::from_bits((m as u64) << 52);
    (hi * df, lo * df)
}

/// `exp_accurate(xh, xl)` → `(h, l, e)` with `exp ≈ 2^e (h + l)`.
fn exp_accurate(xh: f64, xl: f64) -> (f64, f64, i32) {
    const INVLOG2ACC: f64 = f64::from_bits(0x3ff7_1547_652b_82fe); // 0x1.71547652b82fep+0
    const LOG2HACC: f64 = f64::from_bits(0x3fe6_2e42_fefa_39ef); // 0x1.62e42fefa39efp-1
    const LOG2LACC: f64 = f64::from_bits(0x3c7a_bc9e_3b39_8000); // 0x1.abc9e3b398p-56
    const LOG2TINY: f64 = f64::from_bits(0x398f_97b5_7a07_9a19); // 0x1.f97b57a079a19p-103
    let k = (xh * INVLOG2ACC).round_ties_even() as i32;
    let kf = -(k as f64);
    let yh = fma(kf, LOG2HACC, xh);
    let (th, tl) = two_sum(kf * LOG2LACC, xl);
    let (yh, yl) = fast_two_sum(yh, th);
    let yl = fma(kf, LOG2TINY, yl + tl);
    let mut h = E2[19 + 8];
    for i in (16..=18).rev() {
        h = fma(h, yh, E2[i + 8]);
    }
    let (th, mut tl) = a_mul(h, yh);
    tl = fma(h, yl, tl);
    let (mut h, mut l) = fast_two_sum(E2[15 + 8], th);
    l += tl;
    for i in (8..=14).rev() {
        let (th, mut tl) = a_mul(h, yh);
        tl = fma(h, yl, tl);
        tl = fma(l, yh, tl);
        let (hh, ll) = fast_two_sum(E2[i + 8], th);
        h = hh;
        l = ll + tl;
    }
    for i in (0..=7).rev() {
        let (th, mut tl) = a_mul(h, yh);
        tl = fma(h, yl, tl);
        tl = fma(l, yh, tl);
        let (hh, ll) = fast_two_sum(E2[2 * i], th);
        h = hh;
        l = ll + (tl + E2[2 * i + 1]);
    }
    (h, l, k)
}

/// `erfc_asympt_fast(x)` → `(h, l, err)`.
fn erfc_asympt_fast(x: f64) -> (f64, f64, f64) {
    if x >= f64::from_bits(0x4039_db1b_b14e_15ca) {
        return (0.0, 0.0, 1.0);
    }
    let (uh, ul) = a_mul(x, x);
    let (eh, el) = exp_1(-uh, -ul);
    let yh = 1.0 / x;
    let yl = yh * fma(-x, yh, 1.0);
    const THRESHOLD: [u64; 6] = [
        0x3fbd_5000_0000_0000,
        0x3fc5_9da6_ca29_1ba6,
        0x3fcb_c000_0000_0000,
        0x3fd0_c000_0000_0000,
        0x3fd3_8000_0000_0000,
        0x3fd6_3000_0000_0000,
    ];
    let mut i = 0;
    while yh > f64::from_bits(THRESHOLD[i]) {
        i += 1;
    }
    let p = &T[i];
    let (uh, mut ul) = a_mul(yh, yh);
    ul = fma(2.0 * yh, yl, ul);
    let mut zh = p[12];
    zh = fma(zh, uh, p[11]);
    zh = fma(zh, uh, p[10]);
    let (h, l) = s_mul(zh, uh, ul);
    let (mut zh, mut zl) = fast_two_sum(p[9], h);
    zl += l;
    let mut j = 15;
    while j >= 3 {
        let (h, l) = d_mul(zh, zl, uh, ul);
        let (a, b) = fast_two_sum(p[(j + 1) / 2], h);
        zh = a;
        zl = b + l;
        j -= 2;
    }
    let (h, l) = d_mul(zh, zl, uh, ul);
    let (zh, zl0) = fast_two_sum(p[0], h);
    let zl = zl0 + (l + p[1]);
    let (uh, ul) = d_mul(zh, zl, yh, yl);
    let (h, l) = d_mul(uh, ul, eh, el);
    let err = if h >= f64::from_bits(0x0441_51b9_a3fd_d5c9) {
        f64::from_bits(0x3bbd_9000_0000_0000) * h // 0x1.d9p-68
    } else {
        f64::from_bits(0x0010_0000_0000_0000)
    };
    (h, l, err)
}

/// `cr_erfc_fast(x)` → `(h, l, err)`.
fn cr_erfc_fast(x: f64) -> (f64, f64, f64) {
    const THRESHOLD1: f64 = f64::from_bits(0x4007_1378_6d9c_7c09); // 0x1.713786d9c7c09p+1
    if x < 0.0 {
        let (h, l, err) = cr_erf_fast(-x);
        let err = err * h;
        let (h, t) = fast_two_sum(1.0, h);
        return (h, t + l, err + f64::from_bits(0x3994_0000_0000_0000)); // 0x1.4p-102
    } else if x <= THRESHOLD1 {
        let (h, l, err) = cr_erf_fast(x);
        let err = err * h;
        let (h, t) = fast_two_sum(1.0, -h);
        let l = t - l;
        if x >= f64::from_bits(0x3fde_861f_bb24_c00a) {
            return (h, l, err);
        }
        return (h, l, err + f64::from_bits(0x3974_0000_0000_0000)); // 0x1.4p-104
    }
    erfc_asympt_fast(x)
}

/// `erfc_asympt_accurate(x)` for `1.70 < x < 27.3`.
fn erfc_asympt_accurate(x: f64) -> f64 {
    for e in EXCEPTIONS_ERFC.iter() {
        if x == e[0] {
            return e[1] + e[2];
        }
    }
    if x == f64::from_bits(0x403a_8f7b_fbd1_5495) {
        return fma(
            f64::from_bits(1),
            -0.25,
            f64::from_bits(0x0006_67bd_620f_d95b),
        );
    }
    let (uh, ul) = a_mul(x, x);
    let (eh, el, e) = exp_accurate(-uh, -ul);
    let yh = 1.0 / x;
    let yl = yh * fma(-x, yh, 1.0);
    const THRESHOLD: [u64; 10] = [
        0x3fb4_5000_0000_0000,
        0x3fbe_0000_0000_0000,
        0x3fc3_f000_0000_0000,
        0x3fc9_5000_0000_0000,
        0x3fcf_5000_0000_0000,
        0x3fd3_1000_0000_0000,
        0x3fd7_1000_0000_0000,
        0x3fdb_c000_0000_0000,
        0x3fe0_b000_0000_0000,
        0x3fe3_0000_0000_0000,
    ];
    let mut i = 0;
    while yh > f64::from_bits(THRESHOLD[i]) {
        i += 1;
    }
    let p = &TACC[i];
    let (uh, mut ul) = a_mul(yh, yh);
    ul = fma(2.0 * yh, yl, ul);
    let mut zh = p[14 + 6 + i];
    let mut zl = 0.0f64;
    let mut j = 27 + 2 * i;
    while j >= 13 {
        let (h, mut l) = a_mul(zh, uh);
        l = fma(zh, ul, l);
        l = fma(zl, uh, l);
        let (a, b) = two_sum(p[(j - 1) / 2 + 6], h);
        zh = a;
        zl = b + l;
        j -= 2;
    }
    let mut j: i32 = 11;
    while j >= 1 {
        let ju = j as usize;
        let (h, mut l) = a_mul(zh, uh);
        l = fma(zh, ul, l);
        l = fma(zl, uh, l);
        let (a, b) = two_sum(p[ju - 1], h);
        zh = a;
        zl = b + (l + p[ju]);
        j -= 2;
    }
    let (uh, mut ul) = a_mul(zh, yh);
    ul = fma(zh, yl, ul);
    ul = fma(zl, yh, ul);
    let (uh, ul) = fast_two_sum(uh, ul);
    let (h, mut l) = a_mul(uh, eh);
    l = fma(uh, el, l);
    l = fma(ul, eh, l);
    let mut res = ldexp(h + l, e);
    if res < f64::from_bits(0x0010_0000_0000_0000) {
        let mut corr = h - ldexp(res, -e);
        corr += l;
        res += ldexp(corr, e);
    }
    res
}

fn cr_erfc_accurate(x: f64) -> f64 {
    if x < 0.0 {
        for e in EXCEPTIONS_ACCURATE_ERFC.iter() {
            if x == e[0] {
                return e[1] + e[2];
            }
        }
        let (h, l) = cr_erf_accurate(-x);
        let (h, t) = fast_two_sum(1.0, h);
        return h + (t + l);
    } else if x <= f64::from_bits(0x3ffb_59ff_b450_828c) {
        for e in EXCEPTIONS_ACCURATE_ERFC_2.iter() {
            if x == e[0] {
                return e[1] + e[2];
            }
        }
        let (h, l) = cr_erf_accurate(x);
        let (h, t) = fast_two_sum(1.0, -h);
        return h + (t - l);
    }
    erfc_asympt_accurate(x)
}

/// glibc `erfc(x)` (≥ 2.42: correctly rounded).
pub fn erfc(x: f64) -> f64 {
    let t = x.to_bits();
    let at = t & 0x7fff_ffff_ffff_ffff;
    if t >= 0x8000_0000_0000_0000 {
        if t >= 0xc017_744f_8f74_e94b {
            if t >= 0xfff0_0000_0000_0000 {
                if t == 0xfff0_0000_0000_0000 {
                    return 2.0;
                }
                return x + x;
            }
            return 2.0 - f64::from_bits(0x3c90_0000_0000_0000); // 2 − 2^-54
        }
        if -f64::from_bits(0x3c9c_5bf8_91b4_ef6a) <= x {
            return fma(-x, f64::from_bits(0x3c90_0000_0000_0000), 1.0);
        }
    } else {
        if at >= 0x403b_39dc_41e4_8bfd {
            if at >= 0x7ff0_0000_0000_0000 {
                if at == 0x7ff0_0000_0000_0000 {
                    return 0.0;
                }
                return x + x;
            }
            return f64::from_bits(1) * 0.25;
        }
        if x <= f64::from_bits(0x3c8c_5bf8_91b4_ef6a) {
            return fma(-x, f64::from_bits(0x3c90_0000_0000_0000), 1.0);
        }
    }
    let (h, l, err) = cr_erfc_fast(x);
    let left = h + (l - err);
    let right = h + (l + err);
    if left == right {
        return left;
    }
    cr_erfc_accurate(x)
}

/// The hard-to-round inputs CORE-MATH tabulates (for tests of the slow path),
/// plus their neighbours.
#[doc(hidden)]
pub fn hard_cases() -> Vec<f64> {
    let mut v = vec![];
    for t in [
        &EXCEPTIONS_ERFC[..],
        &EXCEPTIONS_ACCURATE_ERFC[..],
        &EXCEPTIONS_ACCURATE_ERFC_2[..],
    ] {
        for e in t {
            for d in -2i64..=2 {
                v.push(f64::from_bits((e[0].to_bits() as i64 + d) as u64));
            }
        }
    }
    v
}
