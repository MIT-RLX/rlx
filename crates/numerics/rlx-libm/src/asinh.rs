// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `asinh`: CORE-MATH's correctly rounded binary64 `asinh`
//! (`sysdeps/ieee754/dbl-64/s_asinh.c`, MIT), statement for statement.

#[path = "asinh_local.rs"]
mod local;
#[path = "asincosh_data.rs"]
pub(crate) mod shared;
use crate::dd::*;
use crate::fma;
#[cfg(not(any(test, feature = "std")))]
#[allow(unused_imports)]
use crate::fp::Fp as _;
use local::*;
use shared::{B, C, L1, L2, R1, R2};

fn asinh_zero(x: f64, x2h: f64, x2l: f64) -> f64 {
    let c = &ZERO_CL;
    let y2 = x2h * (c[0] + x2h * (c[1] + x2h * (c[2] + x2h * (c[3] + x2h * c[4]))));
    let (y1, y2) = polydd(x2h, x2l, &ZERO_CH, y2);
    let (y1, y2) = muldd_acc(y1, y2, x2h, x2l);
    let (y1, y2) = mulddd(y1, y2, x);
    let (y0, y1) = fasttwosum(x, y1);
    let (y1, y2) = fasttwosum(y1, y2);
    y0 + nudge(y1, y2)
}

/// glibc `asinh`.
pub fn asinh(x: f64) -> f64 {
    let ax = x.abs();
    let u = ax.to_bits();
    if u < 0x3fbb_0000_0000_0000 {
        if u < 0x3e57_1374_4912_3ef7 {
            if u == 0 {
                return x;
            }
            return fma(f64::from_bits(0xbc30000000000000), x, x); // -0x1p-60
        }
        let x2h = x * x;
        let x2l = fma(x, x, -x2h);
        let x3h = x2h * x;
        let sl = if u < 0x3f93_0000_0000_0000 {
            if u < 0x3f30_0000_0000_0000 {
                if u < 0x3e5a_0000_0000_0000 {
                    x3h * CL1[0]
                } else {
                    x3h * (CL2[0] + x2h * CL2[1])
                }
            } else {
                let c = &CL4;
                x3h * (c[0] + x2h * (c[1] + x2h * (c[2] + x2h * c[3])))
            }
        } else {
            let c = &CL7;
            let c1 = c[1] + x2h * c[2];
            let c3 = c[3] + x2h * c[4];
            let c5 = c[5] + x2h * c[6];
            let x4 = x2h * x2h;
            x3h * (c[0] + x2h * (c1 + x4 * (c3 + x4 * c5)))
        };
        let eps = f64::from_bits(0x3ca7900000000000) * x3h; // 0x1.79p-53
        let (lb, ub) = (x + (sl - eps), x + (sl + eps));
        if lb == ub {
            return lb;
        }
        return asinh_zero(x, x2h, x2l);
    }
    let (mut x2h, mut x2l) = (0.0, 0.0);
    let (ah, al);
    let mut off = 0x3ff;
    if u < 0x4190_0000_0000_0000 {
        x2h = x * x;
        x2l = fma(x, x, -x2h);
        let (th, mut tl) = if u < 0x3ff0_0000_0000_0000 {
            fasttwosum(1.0, x2h)
        } else {
            fasttwosum(x2h, 1.0)
        };
        tl += x2l;
        let sq = th.sqrt();
        let rs = 0.5 / th;
        let a_l = (tl - fma(sq, sq, -th)) * (rs * sq);
        let (h, t) = fasttwosum(sq, ax);
        ah = h;
        al = a_l + t;
    } else if u < 0x4330_0000_0000_0000 {
        ah = 2.0 * ax;
        al = 0.5 / ax;
    } else {
        if u >= 0x7ff0_0000_0000_0000 {
            return x + x;
        }
        off = 0x3fe;
        ah = ax;
        al = 0.0;
    }
    let mut t = ah.to_bits();
    let e = (t >> 52) as i32 - off;
    t &= u64::MAX >> 12;
    let ed = e as f64;
    let i = (t >> (52 - 5)) as usize;
    let d = (t & (u64::MAX >> 17)) as i64;
    let j = (t
        .wrapping_add((B[i].0 as u64) << 33)
        .wrapping_add((B[i].1 as i64 * (d >> 16)) as u64))
        >> (52 - 10);
    t |= 0x3ff << 52;
    let (i1, i2) = ((j >> 5) as usize, (j & 0x1f) as usize);
    let r = R1[i1] * R2[i2];
    let dx = fma(r, f64::from_bits(t), -1.0);
    let dx2 = dx * dx;
    let f = dx2 * ((C[0] + dx * C[1]) + dx2 * ((C[2] + dx * C[3]) + dx2 * C[4]));
    const L2H: f64 = f64::from_bits(0x3fe62e42fefa3800); // 0x1.62e42fefa38p-1
    const L2L: f64 = f64::from_bits(0x3d2ef35793c76730); // 0x1.ef35793c7673p-45
    let lh = L2H * ed + (L1[i1][1] + L2[i2][1]);
    let mut ll = L2L * ed + L1[i1][0] + L2[i2][0] + al / ah + f;
    ll += dx;
    let s = 1.0f64.copysign(x);
    let (lh, ll) = (lh * s, ll * s);
    let eps = 1.63e-19;
    let (lb, ub) = (lh + (ll - eps), lh + (ll + eps));
    if lb == ub {
        return lb;
    }
    if ax < 0.25 {
        return asinh_zero(x, x2h, x2l);
    }
    asinh_refine(x, ah, al, f64::from_bits(0x3ff71547652b82fe) * lb.abs())
}

fn asinh_database(x: f64, mut f: f64) -> f64 {
    let ax = x.abs();
    for row in &DB {
        if row[0] == ax {
            let sgn = 1.0f64.copysign(x);
            f = sgn * row[1] + sgn * row[2];
            break;
        }
    }
    f
}

fn asinh_refine(x: f64, zh: f64, zl: f64, a: f64) -> f64 {
    let mut t = zh.to_bits();
    let e = (t >> 52) as i32 - 0x3ff + (zl == 0.0) as i32;
    t &= u64::MAX >> 12;
    t |= 0x3ff << 52;
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits(); // 0x1.00008p+0
    let i = (v.wrapping_sub(0x3ff << 52)) >> (52 - 16);
    let (i1, i2, i3, i4) = (
        ((i >> 12) & 0x1f) as usize,
        ((i >> 8) & 0xf) as usize,
        ((i >> 4) & 0xf) as usize,
        (i & 0xf) as usize,
    );
    const L20: f64 = f64::from_bits(0x3fd62e42fefa3800); // 0x1.62e42fefa38p-2
    const L21: f64 = f64::from_bits(0x3d1ef35793c76800); // 0x1.ef35793c768p-46
    const L22: f64 = f64::from_bits(0xba49ff0342542fc3); // -0x1.9ff0342542fc3p-91
    let (el2, el1, el0) = (L22 * ed, L21 * ed, L20 * ed);
    let ll = |k: usize, m: usize, c: usize| LL[k * 17 + m][c];
    let mut l0 = ll(0, i1, 0) + ll(1, i2, 0) + (ll(2, i3, 0) + ll(3, i4, 0));
    let l1 = ll(0, i1, 1) + ll(1, i2, 1) + (ll(2, i3, 1) + ll(3, i4, 1));
    let l2 = ll(0, i1, 2) + ll(1, i2, 2) + (ll(2, i3, 2) + ll(3, i4, 2));
    l0 += el0;
    let t12 = T1[i1] * T2[i2];
    let t34 = T3[i3] * T4[i4];
    let th = t12 * t34;
    let tl = fma(t12, t34, -th);
    let tf = f64::from_bits(t);
    let dh = th * tf;
    let dl = fma(th, tf, -dh);
    let sh = tl * tf;
    let sl = fma(tl, tf, -sh);
    let (xh, mut xl) = fasttwosum(dh - 1.0, dl);
    if zl != 0.0 {
        let tz = (zl.to_bits() as i64).wrapping_sub((e as i64) << 52) as u64;
        xl += th * f64::from_bits(tz);
    }
    let (xh, xl) = adddd(xh, xl, sh, sl);
    let c = &REF_CL;
    let sl = xh * (c[0] + xh * (c[1] + xh * c[2]));
    let (sh, sl) = polydd(xh, xl, &REF_CH, sl);
    let (sh, sl) = muldd_acc(xh, xl, sh, sl);
    let (sh, sl) = adddd(sh, sl, el1, el2);
    let (sh, sl) = adddd(sh, sl, l1, l2);
    let (v0, v2) = fasttwosum(l0, sh);
    let (v1, v2) = fasttwosum(v2, sl);
    let (v0, v1) = fasttwosum(v0, v1);
    let (v1, v2) = fasttwosum(v1, v2);
    let s = 2.0f64.copysign(x);
    let (v0, v1, v2) = (v0 * s, v1 * s, v2 * s);
    let v1 = nudge(v1, v2);
    let t = v1.to_bits();
    let t0 = v0.to_bits();
    let er = t.wrapping_add(41) & (u64::MAX >> 12);
    let de = ((t0 >> 52) & 0x7ff).wrapping_sub((t >> 52) & 0x7ff);
    let res = v0 + v1;
    if de > 99 || er < 80 {
        return asinh_database(x, res);
    }
    res
}
