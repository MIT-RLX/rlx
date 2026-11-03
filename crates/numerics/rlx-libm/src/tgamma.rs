// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `tgamma`: CORE-MATH's correctly rounded binary64 `tgamma`
//! (`sysdeps/ieee754/dbl-64/e_gamma_r.c`, MIT), statement for statement.
//! glibc's `tgamma` returns `__ieee754_gamma_r`'s value as is (the sign is in
//! it), which is [`tgamma`].

#[path = "tgamma_data.rs"]
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

fn splt(x: f64) -> (f64, f64) {
    const OFF: f64 = 201326592.0; // 0x1.8p27
    let xh = (x + OFF) - OFF;
    (xh, x - xh)
}

fn sprod(x: f64, x0: f64, x1: f64, x2: f64) -> (f64, f64, f64) {
    let z0 = x * x0;
    let z0l = fma(x, x0, -z0);
    let z1 = x * x1;
    let z2 = x * x2 + fma(x, x1, -z1);
    let (z0, e) = splt(z0);
    let (e, z0l) = fasttwosum(e, z0l);
    let (l1, e) = twosum(e, z1);
    (z0, l1, e + z0l + z2)
}

fn poly3(dd: f64, ch: &[f64], cl: &[[f64; 2]]) -> (f64, f64) {
    let (mut t0, mut t1, mut t2) = (1.0, 0.0, 0.0);
    let (mut s0, mut s1, mut s2) = (ch[0], 0.0, 0.0);
    for &c in &ch[1..] {
        (t0, t1, t2) = sprod(dd, t0, t1, t2);
        s0 += t0 * c;
        let (fh, fl) = mulddd2(c, t1, t2);
        (s1, s2) = sumdd(s1, s2, fh, fl);
    }
    let (fh, fl) = polyddd(dd, cl.len(), cl, 0.0);
    (s1, s2) = sumdd(s1, s2, fh, fl);
    let (s0, s1) = fasttwosum(s0, s1);
    let (s1, _) = fasttwosum(s1, s2);
    (s0, s1)
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

fn tgamma_accurate(x: f64) -> f64 {
    if x.abs() < 0.25 {
        let c = &ACC_C;
        let x2 = x * x;
        let x4 = x2 * x2;
        let mut c0 = c[0] + x * c[1] + x2 * (c[2] + x * c[3]);
        let c4 = c[4] + x * c[5] + x2 * (c[6] + x * c[7]);
        c0 += x4 * c4;
        let (ch, cl) = polyddd(x, 28, &ACC_CC, x * c0);
        let fh = 1.0 / x;
        let dh = fma(fh, -x, 1.0);
        let fl = dh * fh;
        let fll = fma(fl, -x, dh) * fh;
        let (fl, fll) = sumdd(fl, fll, ch, cl);
        let (fl, fll) = twosum(fl, fll);
        let (fh, fl) = fasttwosum(fh, fl);
        let (mut fl, fll) = fasttwosum(fl, fll);
        let (_, et) = fasttwosum(fh, 2.0 * fl);
        if et == 0.0 {
            if 1.0f64.copysign(fl) * 1.0f64.copysign(fll) > 0.0 {
                fl *= 1.0 + d(0x3cd0000000000000);
            } else {
                fl *= 1.0 - d(0x3cd0000000000000);
            }
        }
        return fh + fl;
    }
    let ix = x.floor();
    let dd = 2.0 * (x - (ix + 0.5));
    let i = ix as i32;
    let (ch, cl, jm, mut eoff): (&[f64], &[[f64; 2]], i32, i64) = if i > 159 {
        (&R159_CH, &R159_CL, 160 - i, 942)
    } else if i > 127 {
        (&R127_CH, &R127_CL, 128 - i, 713)
    } else if i > 95 {
        (&R95_CH, &R95_CL, 96 - i, 495)
    } else if i > 63 {
        (&R63_CH, &R63_CL, 64 - i, 293)
    } else if i > 31 {
        (&R31_CH, &R31_CL, 32 - i, 115)
    } else if i > 25 {
        (&R25_CH, &R25_CL, 26 - i, 86)
    } else if i > 15 {
        (&R15_CH, &R15_CL, 16 - i, 42)
    } else if i > 7 {
        (&R7_CH, &R7_CL, 8 - i, 14)
    } else {
        (&RLO_CH, &RLO_CL, 4 - i, 3)
    };
    let (mut fh, mut fl) = poly3(dd, ch, cl);
    let (mut wh, mut wl) = (1.0, 0.0);
    if jm > 0 {
        let (mut xph, mut xpl) = (x, 0.0);
        wh = xph;
        for _ in (1..jm).rev() {
            let l;
            if xph.abs() > 1.0 {
                (xph, l) = fasttwosum(xph, 1.0);
            } else {
                (xph, l) = fasttwosum(1.0, xph);
            }
            xpl += l;
            (xph, xpl) = fasttwosum(xph, xpl);
            (wh, wl) = muldd3(xph, xpl, wh, wl);
            if wh.abs() > d(0x6050000000000000) {
                // 0x1p518
                wh *= d(0x20b0000000000000); // 0x1p-500
                wl *= d(0x20b0000000000000);
                eoff -= 500;
            }
        }
    } else if jm < 0 {
        let (mut xph, mut xpl) = (x - 1.0, 0.0);
        wh = xph;
        for _ in 0..(-1 - jm) {
            let l;
            (xph, l) = fasttwosum(xph, -1.0);
            xpl += l;
            (xph, xpl) = fasttwosum(xph, xpl);
            (wh, wl) = muldd3(xph, xpl, wh, wl);
        }
    }
    if jm > 0 {
        let rh = 1.0 / wh;
        let rl = (fma(rh, -wh, 1.0) - wl * rh) * rh;
        (fh, fl) = muldd3(fh, fl, rh, rl);
    } else if jm < 0 {
        (fh, fl) = muldd3(fh, fl, wh, wl);
    }
    let mut crr = 0.0;
    let (p54, p107, p53) = (
        d(0x3c90000000000000),
        d(0x3940000000000000),
        d(0x3ca0000000000000),
    );
    if jm <= 0 {
        crr = ((p54 + p107) - p54) + ((p53 - p107) - p53);
        fl += fh * (jm as f64 * crr);
    } else {
        let op = p53 - p107;
        let om = -p53 + p107;
        if op == -om {
            crr = p53 - op;
        }
        fl -= fh * ((jm - 5) as f64 * crr * 1.04);
    }
    let eps = d(0x398e000000000000) * fh; // 0x1.ep-103
    let (ub, lb) = (fh + (fl + eps), fh + (fl - eps));
    let mut res = (fh + fl).to_bits();
    let re = ((res >> 52) & 0x7ff) as i64;
    if re + eoff <= 0 {
        res = res.wrapping_sub(((eoff + re - 1) as u64) << 52);
        res &= 0xfff << 52;
        let (h, l) = fasttwosum(d(res), fh);
        let fl = fl + l;
        let r = (h + fl).to_bits() & !(0x7ff << 52);
        return d(r);
    }
    res = res.wrapping_add((eoff as u64) << 52);
    if ub != lb {
        return database(x, d(res));
    }
    d(res)
}

/// glibc `tgamma` (`__ieee754_gamma_r`; CORE-MATH, correctly rounded).
pub fn tgamma(x: f64) -> f64 {
    let t = x.to_bits();
    let ax = t << 1;
    if ax >= 0x7ff << 53 {
        if ax == 0x7ff << 53 {
            if t >> 63 != 0 {
                return (x - x) / (x - x);
            }
            return x;
        }
        return x + x;
    }
    let z = x;
    if x.abs() < 0.25 {
        if ax < 0x71e0_0000_0000_0000 {
            if x == d(0x0004000000000000) {
                // 2^-1024
                return f64::MAX + d(0x7c90000000000000); // + 0x1p970
            }
            let mut r = 1.0 / x;
            if x == 0.0 {
                return r;
            }
            if fma(r, x, -1.0) == 0.0 {
                r -= 0.5;
            }
            return r;
        }
        let c = &TINY_C;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let mut c0 = c[0] + x * c[1] + x2 * (c[2] + x * c[3]);
        let c4 = c[4] + x * c[5] + x2 * (c[6] + x * c[7]);
        let mut c8 = c[8] + x * c[9] + x2 * (c[10] + x * c[11]);
        let c12 = c[12] + x * c[13] + x2 * (c[14] + x * c[15]);
        c0 += x4 * c4;
        c8 += x4 * c12;
        let (ch, cl) = polyddd(x, 5, &TINY_CC, x * (c0 + x8 * c8));
        let fh = 1.0 / z;
        let fl = fma(fh, -z, 1.0) * fh;
        let (fh, fl) = fastsum(fh, fl, ch, cl);
        let eps = fh * (3.5e-19 + (x2 * x4) * 4e-15);
        let (ub, lb) = (fh + (fl + eps), fh + (fl - eps));
        if ub != lb {
            return tgamma_accurate(x);
        }
        return ub;
    }
    if x >= d(0x406573fae561f648) {
        return d(0x7fef000000000000) + d(0x7fef000000000000); // 0x1.fp1023 twice
    }
    let fx = x.floor();
    let k = fx as i64;
    if fx == x {
        if x == 0.0 {
            return 1.0 / 0.0;
        }
        if x < 0.0 {
            return (x - x) / (x - x); // __math_invalid: a negative integer
        }
        let (mut t0h, mut t0l, mut x0) = (1.0, 0.0, 1.0);
        for _ in 1..k {
            (t0h, t0l) = mulddd2(x0, t0h, t0l);
            x0 += 1.0;
        }
        return t0h + t0l;
    }
    if x <= -184.0 {
        return d(0x0010000000000000) * SGN[(k & 1) as usize];
    }
    if x < -3.0 {
        let (lh, ll) = fasttwosum(-x, 1.0);
        let (lh, ll) = lgamma_asym(lh, ll);
        let (lh, ll, e) = expd(lh, ll);
        let ix = x.floor();
        let dx = x - ix;
        let ip = ix as i32;
        let (sh, sl) = sinpid(dx);
        let (lh, ll) = muldd2(sh, sl, lh, ll);
        const PIH: f64 = f64::from_bits(0x400921fb54442d18);
        const PIL: f64 = f64::from_bits(0x3ca1a62633145c07);
        let rcp = 1.0 / lh;
        let mut rh = rcp * PIH;
        let mut rl = rcp * (PIL - ll * rh - fma(rh, lh, -PIH));
        if ip & 1 != 0 {
            rh = -rh;
            rl = -rl;
        }
        let eps = rh * (d(0x3bbeb2049057bc61) - x * d(0x3b661019f74442b7));
        let th;
        if ip >= -170 {
            let (ub, lb) = (rh + (rl + eps), rh + (rl - eps));
            if ub != lb {
                return tgamma_accurate(x);
            }
            th = ub.to_bits().wrapping_sub(((e as i64) << 52) as u64);
        } else {
            let tb = rh.to_bits();
            let re = ((tb >> 52) & 0x7ff) as i32;
            if re - e <= 0 {
                let tb = tb.wrapping_add((((e - re + 1) as i64) << 52) as u64) & (0xfff << 52);
                let (h, l) = fasttwosum(d(tb), rh);
                let rl = rl + l;
                let (ub, lb) = (h + (rl + eps), h + (rl - eps));
                if ub != lb {
                    return tgamma_accurate(x);
                }
                return d(ub.to_bits() & !(0x7ff << 52));
            }
            let (ub, lb) = (rh + (rl + eps), rh + (rl - eps));
            if ub != lb {
                return tgamma_accurate(x);
            }
            th = ub.to_bits().wrapping_sub(((e as i64) << 52) as u64);
        }
        return d(th);
    }
    if x > 4.0 {
        let (lh, ll) = lgamma_asym(x, 0.0);
        let (lh, ll, e) = expd(lh, ll);
        let eps = lh * (d(0x3ba2e3b40a0e9b4f) + x * d(0x3b66aad80c11872c));
        let (ub, lb) = (lh + (ll + eps), lh + (ll - eps));
        if ub != lb {
            return tgamma_accurate(x);
        }
        return d(ub.to_bits().wrapping_add(((e as i64) << 52) as u64));
    }
    let c = &MID_C;
    let m = z - 3.5;
    let i = m.round_ties_even();
    let dd = z - (i + 3.5);
    let d2 = dd * dd;
    let d4 = d2 * d2;
    let fl = dd
        * ((c[10] + dd * c[11])
            + d2 * (c[12] + dd * c[13])
            + d4 * ((c[14] + dd * c[15]) + d2 * (c[16] + dd * c[17])));
    let (fh, fl) = polyddd(dd, 10, &MID_CC, fl);
    let jm = i.abs() as i32;
    let (mut wh, mut wl) = (1.0, 0.0);
    let (mut xph, mut xpl) = (z, 0.0);
    if jm != 0 {
        wh = xph;
        for _ in (1..jm).rev() {
            let l;
            if xph.abs() > 1.0 {
                (xph, l) = fasttwosum(xph, 1.0);
            } else {
                (xph, l) = fasttwosum(1.0, xph);
            }
            xpl += l;
            (wh, wl) = muldd2(xph, xpl, wh, wl);
        }
    }
    let rh = 1.0 / wh;
    let rl = (fma(rh, -wh, 1.0) - wl * rh) * rh;
    let (fh, fl) = muldd2(rh, rl, fh, fl);
    let eps = fh * d(0x3b92e3b40a0e9b4f);
    let (ub, lb) = (fh + (fl + eps), fh + (fl - eps));
    if ub != lb {
        return tgamma_accurate(x);
    }
    ub
}

fn logd(x: f64) -> (f64, f64) {
    let mut t = x.to_bits();
    let e = (t >> 52) as i32 - 0x3ff;
    t &= u64::MAX >> 12;
    let ed = e as f64;
    let i = (t >> (52 - 5)) as usize;
    let dd = (t & (u64::MAX >> 17)) as i64;
    let j = (t
        .wrapping_add((LOGD_B[i].0 as u64) << 33)
        .wrapping_add((LOGD_B[i].1 as i64 * (dd >> 16)) as u64))
        >> (52 - 10);
    t |= 0x3ff << 52;
    let (i1, i2) = ((j >> 5) as usize, (j & 0x1f) as usize);
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

fn sinpid(x: f64) -> (f64, f64) {
    let x = (x - 0.5).abs() * 128.0;
    let ix = x.round_ties_even();
    let dd = ix - x;
    let d2 = dd * dd;
    let ky = ix as i32;
    let kx = 64 - ky;
    let (sh, sl) = (ST[kx as usize][1], ST[kx as usize][0]);
    let (ch, cl) = (ST[ky as usize][1], ST[ky as usize][0]);
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

fn expd(x: f64, l: f64) -> (f64, f64, i32) {
    const LN2H: f64 = f64::from_bits(0x40971547652b82fe);
    const LN2L: f64 = f64::from_bits(0x3d1777d0ffda0d24);
    let (xh, xl) = muldd2(x, l, LN2H, LN2L);
    let ix = xh.round_ties_even();
    let (xh, xl) = fasttwosum(xh - ix, xl);
    let k = ix as i32;
    let (i0, i1) = (((k >> 5) & 31) as usize, (k & 31) as usize);
    let e = k >> 10;
    let (rh, rl) = muldd2(E0[i0][1], E0[i0][0], E1[i1][1], E1[i1][0]);
    let c = &EXP_C;
    let m = 1;
    let fl = xh * polyd(xh, 5 - m, &c[m..]);
    let (fh, fl) = polydd2(xh, xl, m, c, fl);
    let (fh, fl) = muldd2(xh, xl, fh, fl);
    let (fh, el) = fasttwosum(1.0, fh);
    let fl = fl + el;
    let (rh, rl) = muldd2(rh, rl, fh, fl);
    (rh, rl, e)
}

fn lgamma_asym(xh: f64, xl: f64) -> (f64, f64) {
    let zh = 1.0 / xh;
    let dz = xl * zh;
    let zl = (fma(zh, -xh, 1.0) - dz) * zh;
    let (lh, mut ll) = logd(xh);
    ll += dz;
    let (mut lh, mut ll) = muldd2(xh - 0.5, xl, lh - 1.0, ll);
    let (z2h, z2l) = muldd2(zh, zl, zh, zl);
    let x2 = z2h * z2h;
    let (fh, fl);
    if xh > 11.5 {
        let c = &ASYM_C1;
        (lh, ll) = fastsum(lh, ll, c[0][0], c[0][1]);
        let (b, q) = (&c[1..], &c[2..]);
        let q0 = q[0][0] + z2h * q[1][0];
        let q2 = q[2][0] + z2h * q[3][0];
        let q4 = q[4][0] + z2h * q[5][0];
        let l = z2h * (q0 + x2 * (q2 + x2 * q4));
        (fh, fl) = polydd2(z2h, z2l, 1, b, l);
    } else {
        let c = &ASYM_C2;
        (lh, ll) = fastsum(lh, ll, c[0][0], c[0][1]);
        let x4 = x2 * x2;
        let (b, q) = (&c[1..], &c[3..]);
        let mut q0 = q[0][0] + z2h * q[1][0];
        let q2 = q[2][0] + z2h * q[3][0];
        let mut q4 = q[4][0] + z2h * q[5][0];
        let q6 = q[6][0] + z2h * q[7][0];
        let q8 = q[8][0] + z2h * q[9][0];
        q4 += x2 * (q6 + x2 * q8);
        q0 += x2 * q2;
        q0 += x4 * q4;
        (fh, fl) = polydd2(z2h, z2l, 2, b, z2h * q0);
    }
    let (fh, fl) = muldd2(zh, zl, fh, fl);
    fastsum(lh, ll, fh, fl)
}
