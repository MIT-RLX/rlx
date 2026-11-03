// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `lgamma`: CORE-MATH's correctly rounded binary64 `lgamma`
//! (`sysdeps/ieee754/dbl-64/e_lgamma_r.c`, MIT), statement for statement.
//! [`lgamma_r`] also gives the sign of Γ(x), as glibc's `lgamma_r` does.

#[path = "lgamma_data.rs"]
mod data;
use crate::dd::*;
use crate::fma;
#[cfg(not(any(test, feature = "std")))]
#[allow(unused_imports)]
use crate::fp::Fp as _;
use data::*;

#[inline(always)]
fn d(b: u64) -> f64 {
    f64::from_bits(b)
}

/// glibc `lgamma`.
pub fn lgamma(x: f64) -> f64 {
    lgamma_r(x).0
}

/// glibc `lgamma_r`: `(log|Γ(x)|, sign of Γ(x))`.
pub fn lgamma_r(x: f64) -> (f64, i32) {
    let t = x.to_bits();
    let nx = t << 1;
    if nx >= 0xfeae_a9b2_4f16_a34c {
        if t == 0x7f57_54d9_278b_51a6 {
            return (d(0x7feffffffffffffe) - d(0x7c80000000000000), 1);
        }
        if t == 0x7f57_54d9_278b_51a7 {
            return (f64::MAX - d(0x7c80000000000000), 1);
        }
        if nx >= 0x7ff << 53 {
            if nx == 0x7ff << 53 {
                return (x.abs(), 1);
            }
            return (x + x, 1);
        }
        if t >> 63 != 0 {
            return (1.0 / 0.0, 1);
        }
        return (d(0x7fef000000000000) * d(0x7fef000000000000), 1);
    }
    let fx = x.floor();
    if fx == x {
        if x <= 0.0 {
            return (1.0 / 0.0, 1 - 2 * (t >> 63) as i32);
        }
        if x == 1.0 || x == 2.0 {
            return (0.0, 1);
        }
    }
    let mut au = (nx >> 38) as u32;
    let (fh, fl, mut eps);
    let mut sign = 1;
    if au < UBRD[0] {
        sign = 1 - 2 * (t >> 63) as i32;
        let (lh, ll) = logd(x.abs());
        if au < 0x1da_0000 {
            fh = -lh;
            fl = -ll;
            eps = 1.5e-22;
        } else if au < 0x1fd_0000 {
            let q = &SMALL_Q;
            let z = x;
            let z2 = z * z;
            let z4 = z2 * z2;
            let q0 = q[0] + z * q[1];
            let q2 = q[2] + z * q[3];
            let q4 = q[4] + z * q[5];
            let q6 = q[6] + z * q[7];
            let l = z * ((q0 + z2 * q2) + z4 * (q4 + z2 * q6));
            let (h, l) = polydddfst(z, 4, &SMALL_C0, l);
            let (h, l) = mulddd2(x, h, l);
            (fh, fl) = sumdd(-lh, -ll, h, l);
            eps = 1.5e-22;
        } else {
            let (tf, xl) = fasttwosum(1.0, x);
            au = (tf.to_bits() >> 37) as u32;
            let ou = (au - UBRD[0]) as u64;
            let mut j = ((0x1_57ce_d865u64.wrapping_sub(ou * 0x150d)).wrapping_mul(ou)
                + 0x1280_0000_0000)
                >> 45;
            j -= (au < UBRD[j as usize]) as u64;
            let j = j as usize;
            let z = (tf - OFFS[j]) + xl;
            let z2 = z * z;
            let z4 = z2 * z2;
            let q = &CL[j];
            let q0 = q[0] + z * q[1];
            let q2 = q[2] + z * q[3];
            let q4 = q[4] + z * q[5];
            let q6 = q[6] + z * q[7];
            let l = z * ((q0 + z2 * q2) + z4 * (q4 + z2 * q6));
            let (mut h, mut l) = polydddfst(z, 5, &CH[j], l);
            if j == 4 {
                (h, l) = mulddd2(-x, h, l);
            }
            eps = h.abs() * 8.3e-20;
            (fh, fl) = sumdd(-lh, -ll, h, l);
            eps += lh.abs() * 5e-22;
        }
    } else {
        let ax = x.abs();
        let (mut h, mut l);
        if au >= UBRD[19] {
            let (lh, mut ll) = logd(ax);
            let mut lh = lh - 1.0;
            if au >= 0x219_8000 {
                if au >= 0x3fa_baa6 {
                    (lh, ll) = fasttwosum(lh, ll);
                }
                let hlh = lh * 0.5;
                (lh, ll) = mulddd2(ax, lh, ll);
                ll -= hlh;
            } else {
                (lh, ll) = mulddd2(ax - 0.5, lh, ll);
            }
            let (c, q) = (&ASY_C, &ASY_Q);
            (lh, ll) = fastsum(lh, ll, c[0][0], c[0][1]);
            if ax < d(0x4630000000000000) {
                // 0x1p100
                let zh = 1.0 / ax;
                let zl = fma(zh, -ax, 1.0) * zh;
                let z2h = zh * zh;
                let z4h = z2h * z2h;
                let q0 = q[0] + z2h * q[1];
                let q2 = q[2] + z2h * q[3];
                let q4 = q[4];
                let f = z2h * (q0 + z4h * (q2 + z4h * q4));
                (h, l) = fasttwosum(c[1][0], f);
                l += c[1][1];
                (h, l) = muldd2(h, l, zh, zl);
            } else {
                h = 0.0;
                l = 0.0;
            }
            (h, l) = fastsum(lh, ll, h, l);
            eps = h.abs() * 4.5e-20;
        } else {
            let ou = (au - UBRD[0]) as u64;
            let mut j = ((0x1_57ce_d865u64.wrapping_sub(ou * 0x150d)).wrapping_mul(ou)
                + 0x1280_0000_0000)
                >> 45;
            j -= (au < UBRD[j as usize]) as u64;
            let j = j as usize;
            let z = ax - OFFS[j];
            let z2 = z * z;
            let z4 = z2 * z2;
            let q = &CL[j];
            let q0 = q[0] + z * q[1];
            let q2 = q[2] + z * q[3];
            let q4 = q[4] + z * q[5];
            let q6 = q[6] + z * q[7];
            l = z * ((q0 + z2 * q2) + z4 * (q4 + z2 * q6));
            (h, l) = polydddfst(z, 5, &CH[j], l);
            if j == 4 {
                (h, l) = mulddd2(1.0 - ax, h, l);
            }
            if j == 10 {
                (h, l) = mulddd2(ax - 2.0, h, l);
            }
            eps = h.abs() * 8.3e-20 + 1e-24;
        }
        if t >> 63 != 0 {
            let (sh, sl) = sinpipid(x - x.floor());
            let (sh, sl) = mulddd2(-x, sh, sl);
            let (lh, mut ll) = logd(sh);
            ll += sl / sh;
            let (nh, nl) = sumdd(h, l, lh, ll);
            h = -nh;
            l = -nl;
            eps += lh.abs() * 4e-22;
            let k = fx as i64;
            sign = 1 - 2 * (k & 1) as i32;
        }
        (fh, fl) = (h, l);
    }
    let (ub, lb) = (fh + (fl + eps), fh + (fl - eps));
    if ub != lb {
        return (lgamma_accurate(x), sign);
    }
    (ub, sign)
}

fn database(x: f64, mut f: f64) -> f64 {
    for row in DB.iter() {
        if row[0] == x {
            f = row[1] + row[2];
            break;
        }
    }
    f
}

/// `fasttwosum(fh, 2·fl)`'s error is zero: the tie nudge of the accurate path.
#[inline(always)]
fn tie(fh: f64, fl: f64) -> bool {
    fasttwosum(fh, 2.0 * fl).1 == 0.0
}

fn lgamma_accurate(x: f64) -> f64 {
    let sx = x;
    let mut x = x.abs();
    let (mut fh, mut fl, mut fll);
    let (p54, p52) = (d(0x3c90000000000000), d(0x3cb0000000000000));
    if x < d(0x39b0000000000000) {
        // 0x1p-100
        let (lh, ll, lll) = logd_accurate(x);
        (fh, fl) = fasttwosum(-lh, -ll);
        (fl, fll) = fasttwosum(fl, -lll);
        if tie(fh, fl) && 1.0 + p54 == 1.0 - p54 {
            fl *= 1.0 + p52.copysign(fl) * 1.0f64.copysign(fll);
        }
    } else if x < 0.25 {
        let (h, l) = polydddfst(sx, ACC_C0.len(), &ACC_C0, 0.0);
        let (h, l) = mulddd2(sx, h, l);
        let (lh, ll, lll) = logd_accurate(x);
        let (h, l) = sumdd(h, l, -ll, -lll);
        (fh, fll) = twosum(-lh, h);
        (fl, fll) = twosum(fll, l);
        if tie(fh, fl) && 1.0 + p54 == 1.0 - p54 {
            fl *= 1.0 + p52.copysign(fl) * 1.0f64.copysign(fll);
        }
    } else {
        fll = 0.0;
        let neg = sx < 0.0;
        // `fl = sumdd(fl, fll, a, b)` then `fh = twosum(fh, h)` then fold the
        // twosum error back in: the step every branch repeats.
        let add = |fh: f64, fl: f64, fll: f64, h: f64, l: f64, ll: f64| {
            let (fl, fll) = sumdd(fl, fll, l, ll);
            let (fh, e) = twosum(fh, h);
            let (fl, fll) = sumdd(fl, fll, e, 0.0);
            (fh, fl, fll)
        };
        if (x - 0.5).abs() < 0.25 {
            (fh, fl) = polydddfst(x - 0.5, ACC_B.len(), &ACC_B, 0.0);
            if sx > 0.0 {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, 0.0, -lh, -ll, -lll);
            }
        } else if (x - 2.5).abs() < 0.25 {
            (fh, fl) = polydddfst(x - 2.5, ACC_B.len(), &ACC_B, 0.0);
            let (lh, ll, lll) = logd_accurate(x - 1.0);
            (fh, fl, fll) = add(fh, fl, 0.0, lh, ll, lll);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, fll, lh, ll, lll);
            }
        } else if (x - 3.5).abs() < 0.25 {
            let (l2h, l2l, l2ll) = logd_accurate(x - 2.0);
            let (l1h, l1l, l1ll) = logd_accurate(x - 1.0);
            let (l1l, l1ll) = sumdd(l1l, l1ll, l2l, l2ll);
            let (l1h, l2h) = fasttwosum(l1h, l2h);
            let (l1l, l1ll) = sumdd(l1l, l1ll, l2h, 0.0);
            (fh, fl) = polydddfst(x - 3.5, ACC_B.len(), &ACC_B, 0.0);
            (fh, fl, fll) = add(fh, fl, 0.0, l1h, l1l, l1ll);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, fll, lh, ll, lll);
            }
        } else if (x - 1.0).abs() < 0.25 {
            (fh, fl) = polydddfst(x - 1.0, ACC_C0.len(), &ACC_C0, 0.0);
            (fh, fl) = mulddd2(x - 1.0, fh, fl);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, 0.0, lh, ll, lll);
            }
        } else if (x - 1.5).abs() < 0.25 {
            (fh, fl) = polydddfst(x - 1.5, ACC_B.len(), &ACC_B, 0.0);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, 0.0, lh, ll, lll);
            }
        } else if (x - 2.0).abs() < 0.25 {
            let (lh, ll, lll) = logd_accurate(x - 1.0);
            (fh, fl) = polydddfst(x - 2.0, ACC_C0.len(), &ACC_C0, 0.0);
            (fh, fl) = mulddd2(x - 2.0, fh, fl);
            (fh, fl, fll) = add(fh, fl, 0.0, lh, ll, lll);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, fll, lh, ll, lll);
            }
        } else if (x - 3.0).abs() < 0.25 {
            let (l2h, l2l, l2ll) = logd_accurate(x - 2.0);
            let (l1h, l1l, l1ll) = logd_accurate(x - 1.0);
            let (l1l, l1ll) = sumdd(l1l, l1ll, l2l, l2ll);
            let (l1h, l2h) = fasttwosum(l1h, l2h);
            let (l1l, l1ll) = sumdd(l1l, l1ll, l2h, 0.0);
            (fh, fl) = polydddfst(x - 3.0, ACC_C0.len(), &ACC_C0, 0.0);
            (fh, fl) = mulddd2(x - 3.0, fh, fl);
            (fh, fl, fll) = add(fh, fl, 0.0, l1h, l1l, l1ll);
            if neg {
                let (lh, ll, lll) = logd_accurate(x);
                (fh, fl, fll) = add(fh, fl, fll, lh, ll, lll);
            }
        } else {
            let mut xl = 0.0;
            if neg {
                (x, xl) = fasttwosum(x, 1.0);
            }
            (fh, fl, fll) = lgamma_asym_accurate(x, xl);
        }
        if neg {
            let phi = if sx < -0.5 { sx - sx.floor() } else { -sx };
            let (sh, sl) = sinpipid_accurate(phi);
            let (lh, mut ll, lll) = logd_accurate(sh);
            ll += sl / sh + lll;
            (fl, fll) = sumdd(fl, fll, ll, 0.0);
            let (h, e) = twosum(fh, lh);
            (fl, fll) = sumdd(fl, fll, e, 0.0);
            fh = -h;
            fl = -fl;
            fll = -fll;
        }
        (fh, fl) = fasttwosum(fh, fl);
        (fl, fll) = fasttwosum(fl, fll);
        (fh, fl) = fasttwosum(fh, fl);
        (fl, fll) = fasttwosum(fl, fll);
        if tie(fh, fl) {
            let p26 = d(0x3e50000000000000);
            fl *= 1.0 + p26.copysign(fl) * p26.copysign(fll);
        }
    }
    let _ = fll;
    if fh.abs() < 0.375 {
        // near the roots of lgamma below -2
        const SX_BND: f64 = f64::from_bits(0xc004e147ae147ae1);
        // (|fh| bound, sx window, scale, k) per root
        const ROOTS: [(u64, f64, f64, i32, usize); 17] = [
            (0x3fb7400000000000, SX_BND, -2.0, 3, 6),
            (0x3fb1680000000000, -3.0, SX_BND, 4, 6),
            (0x3fb2d40000000000, -3.5, -3.0, 6, 7),
            (0x3faef00000000000, -4.0, -3.5, 8, 7),
            (0x3fc0000000000000, -4.5, -4.0, 7, 7),
            (0x3fb0800000000000, -5.0, -4.5, 10, 7),
            (0x3fb1000000000000, -5.5, -5.0, 10, 6),
            (0x3fc1300000000000, -6.0, -5.5, 12, 6),
            (0x3fc1500000000000, -6.5, -6.0, 12, 6),
            (0x3fc3400000000000, -7.0, -6.5, 14, 6),
            (0x3fc1450000000000, -7.5, -7.0, 15, 6),
            (0x3fc76b0000000000, -8.0, -7.5, 17, 6),
            (
                0x3fc76c0000000000,
                f64::from_bits(0xc0210f5c28f5c28f),
                -8.0,
                17,
                6,
            ),
            (0x3fc9900000000000, -9.0, -8.5, 20, 6),
            (0x3fc9900000000000, -9.5, -9.0, 20, 6),
            (0x3fc76b0000000000, -10.0, -9.5, 24, 6),
            (0x3fc76b0000000000, -10.5, -10.0, 24, 6),
        ];
        for (r, &(bound, lo, hi, sc, k)) in ROOTS.iter().enumerate() {
            if fh.abs() < d(bound) && sx > lo && sx < hi {
                let x0 = &ROOT_X0[r];
                let c = ROOT_C[r];
                let sc = d(((1023 + sc) as u64) << 52);
                let (zh, mut zl) = fasttwosum(x0[0] + sx, x0[1]);
                zl += x0[2];
                let (sh, sl) = (zh * sc, zl * sc);
                let n = c.len();
                let l = sh * polyd(sh, k, &c[n - k..]);
                let (h, l) = polydd2(sh, sl, n - k, c, l);
                (fh, fl) = muldd2(zh, zl, h, l);
                break;
            }
        }
    }
    // `unsigned ft`: the low 32 bits only
    let ft = (fl.to_bits().wrapping_add(2) & (u64::MAX >> 12)) as u32;
    if ft <= 2 {
        return database(sx, fh + fl);
    }
    fh + fl
}

/// `as_logd`: `log(x)` as a double-double, `x` possibly subnormal.
fn logd(x: f64) -> (f64, f64) {
    let (mut t, e) = split(x);
    let ed = e as f64;
    let (i1, i2) = index(&LOGD_B, &mut t);
    let r = LOGD_R1[i1] * LOGD_R2[i2];
    let tf = d(t);
    let o = r * tf;
    let dxl = fma(r, tf, -o);
    let dxh = o - 1.0;
    let c = &LOGD_C;
    let dx = fma(r, tf, -1.0);
    let dx2 = dx * dx;
    let f = dx2 * ((c[0] + dx * c[1]) + dx2 * (c[2] + dx * c[3]));
    let lt = (LOGD_L1[i1][1] + LOGD_L2[i2][1]) + ed * d(0x3fe62e42fef80000);
    let lh = lt + dxh;
    let mut ll = (lt - lh) + dxh;
    ll += ((LOGD_L1[i1][0] + LOGD_L2[i2][0]) + d(0x3db1cf79abc9e3b4) * ed) + dxl;
    ll += f;
    (lh, ll)
}

/// `x`'s significand bits (subnormals normalized) and unbiased exponent.
fn split(x: f64) -> (u64, i32) {
    let mut t = x.to_bits();
    let mut ex = (t >> 52) as i32;
    if ex == 0 {
        let k = t.leading_zeros() as i32;
        t <<= k - 11;
        ex -= k - 12;
    }
    (t & (u64::MAX >> 12), ex - 0x3ff)
}

/// The two-level table index of `t` (significand bits), and `t` made `[1, 2)`.
fn index(b: &[(u16, i16)], t: &mut u64) -> (usize, usize) {
    let i = (*t >> (52 - 5)) as usize;
    let dd = (*t & (u64::MAX >> 17)) as i64;
    let j = (t
        .wrapping_add((b[i].0 as u64) << 33)
        .wrapping_add((b[i].1 as i64 * (dd >> 16)) as u64))
        >> (52 - 10);
    *t |= 0x3ff << 52;
    ((j >> 5) as usize, (j & 0x1f) as usize)
}

fn logd_accurate(x: f64) -> (f64, f64, f64) {
    let (mut t, e) = split(x);
    let ed = e as f64;
    let (i1, i2) = index(&LOGDA_B, &mut t);
    let r = LOGDA_R1[i1] * LOGDA_R2[i2];
    let tf = d(t);
    let o = r * tf;
    let dxl = fma(r, tf, -o);
    let dxh = o - 1.0;
    let c = &LOGDA_C;
    let (dxh, dxl) = fasttwosum(dxh, dxl);
    let l = dxh * (c[6][0] + dxh * (c[7][0] + dxh * c[8][0]));
    let (fh, fl) = polydd2(dxh, dxl, 6, c, l);
    let (fh, fl) = muldd2(dxh, dxl, fh, fl);
    let s2 = H1[i1][2] + H2[i2][2];
    let s1 = H1[i1][1] + H2[i2][1];
    let s0 = H1[i1][0] + H2[i2][0];
    let mut l0 = d(0x3fe62e42fefa3800) * ed;
    let l1 = d(0x3d2ef35793c76000) * ed;
    let l2 = d(0x3a8cc01f97b57a08) * ed;
    l0 += s2;
    let (l1, l2) = sumdd(l1, l2, s1, s0);
    let (l1, l2) = sumdd(l1, l2, fh, fl);
    let (l0, l1) = fasttwosum(l0, l1);
    let (l1, l2) = fasttwosum(l1, l2);
    (l0, l1, l2)
}

fn sinpipid(x: f64) -> (f64, f64) {
    let x = x - 0.5;
    let ax = x.abs();
    let sx = ax * 128.0;
    let ix = sx.round_ties_even();
    let ky = ix as i32;
    let kx = 64 - ky;
    if kx < 2 {
        let (c, cl) = (&SIN_KC, &SIN_KCL);
        let z = 0.5 - ax;
        let z2 = z * z;
        let z2l = fma(z, z, -z2);
        let f = z2 * (cl[0] + z2 * (cl[1] + z2 * cl[2]));
        let (fh, mut fl) = fasttwosum(c[0], f);
        fl += c[1];
        let (fh, fl) = muldd2(z2, z2l, fh, fl);
        let (fh, fl) = mulddd2(z, fh, fl);
        let (fh, e) = fasttwosum(z, fh);
        return (fh, fl + e);
    }
    let dd = ix - sx;
    let d2 = dd * dd;
    let (sh, sl) = (STPI[kx as usize][1], STPI[kx as usize][0]);
    let (ch, cl) = (STPI[ky as usize][1], STPI[ky as usize][0]);
    let c = &SIN_C;
    let c0 = d(0xbbd692b66e3cf6e8);
    let s = &SIN_S;
    let s0 = d(0x3c31a624b88c9448);
    let p = d2 * (c[1] + d2 * (c[2] + d2 * c[3]));
    let q = d2 * (s[1] + d2 * (s[2] + d2 * s[3]));
    let (qh, mut ql) = fasttwosum(s[0], q);
    ql += s0;
    let (ch, cl) = muldd2(qh, ql, ch, cl);
    let (th, mut tl) = fasttwosum(c[0], p);
    tl += c0;
    let (th, tl) = mulddd2(dd, th, tl);
    let (ph, pl) = muldd2(th, tl, sh, sl);
    let (ch, cl) = fastsum(ch, cl, ph, pl);
    let (ch, cl) = mulddd2(dd, ch, cl);
    fastsum(sh, sl, ch, cl)
}

fn sinpipid_accurate(x: f64) -> (f64, f64) {
    let x = (x - 0.5).abs() * 128.0;
    let ix = x.round_ties_even();
    let dd = ix - x;
    let ky = ix as i32;
    let kx = 64 - ky;
    let (sh, sl) = (STPI[kx as usize][1], STPI[kx as usize][0]);
    let (ch, cl) = (STPI[ky as usize][1], STPI[ky as usize][0]);
    let d2h = dd * dd;
    let d2l = fma(dd, dd, -d2h);
    let (ph, pl) = polydd2(d2h, d2l, 5, &SINA_C, 0.0);
    let (qh, ql) = polydd2(d2h, d2l, 6, &SINA_S, 0.0);
    let (ph, pl) = mulddd2(dd, ph, pl);
    let (ph, pl) = muldd2(sh, sl, ph, pl);
    let (qh, ql) = muldd2(ch, cl, qh, ql);
    let (ch, cl) = fastsum(qh, ql, ph, pl);
    let (ch, cl) = mulddd2(dd, ch, cl);
    fastsum(sh, sl, ch, cl)
}

/// `(l0x, *xl, *e)` of `as_lgamma_asym_accurate(xh, &xl, &e)`.
fn lgamma_asym_accurate(xh: f64, xl: f64) -> (f64, f64, f64) {
    let (mut l0, mut l1, mut l2) = logd_accurate(xh);
    let (l0x, l1x, l2x);
    if xh < d(0x4770000000000000) {
        // 0x1p120
        let zh = 1.0 / xh;
        let dz = xl * zh;
        let zl = (fma(zh, -xh, 1.0) - dz) * zh;
        if xl != 0.0 {
            let (dl1, mut dl2) = mulddd2(xl, zh, fma(zh, -xh, 1.0) * zh);
            dl2 -= dl1 * dl1 / 2.0;
            (l1, l2) = sumdd(l1, l2, dl1, dl2);
        }
        let (wh, wl) = if xh.to_bits() >> 52 > 0x3ff + 51 {
            (xh, xl - 0.5)
        } else {
            (xh - 0.5, xl)
        };
        l0 -= 1.0;
        let a0 = l0 * wh;
        let l0xl = fma(l0, wh, -a0);
        let a1 = l1 * wh;
        let l1xl = fma(l1, wh, -a1);
        let a2 = l2 * wh;
        let (a1, a2) = sumdd(a1, a2, l0xl, l1xl);
        let (mut a1, mut a2) = sumdd(a1, a2, l0 * wl, l1 * wl);
        let (z2h, z2l) = muldd2(zh, zl, zh, zl);
        let c: &[[f64; 2]] = if xh >= 48.0 {
            &ASYA_C48[..8]
        } else if xh >= 14.5 {
            &ASYA_C14[..12]
        } else {
            &ASYA_C
        };
        (a1, a2) = sumdd(a1, a2, c[0][0], c[0][1]);
        let (fh, fl) = polydd2(z2h, z2l, c.len() - 1, &c[1..], 0.0);
        let (fh, fl) = muldd2(zh, zl, fh, fl);
        (a1, a2) = sumdd(a1, a2, fh, fl);
        let (h, a1) = fasttwosum(a0, a1);
        let (a1, a2) = fasttwosum(a1, a2);
        (l0x, l1x, l2x) = (h, a1, a2);
    } else {
        let wl = xl - 0.5;
        l0 -= 1.0;
        let a0 = l0 * xh;
        let l0xl = fma(l0, xh, -a0);
        let a1 = l1 * xh;
        let l1xl = fma(l1, xh, -a1);
        let a2 = l2 * xh;
        let (a1, a2) = sumdd(a1, a2, l0xl, l1xl);
        let (a1, a2) = sumdd(a1, a2, l0 * wl, l1 * wl);
        (l0x, l1x, l2x) = (a0, a1, a2);
    }
    (l0x, l1x, l2x)
}
