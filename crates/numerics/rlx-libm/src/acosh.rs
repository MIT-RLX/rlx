// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `acosh`: CORE-MATH's correctly rounded binary64 `acosh`
//! (`sysdeps/ieee754/dbl-64/e_acosh.c`, MIT), statement for statement.

#[path = "acosh_local.rs"]
mod local;
use crate::asinh::shared::{B, C, L1, L2, R1, R2};
use crate::dd::*;
use crate::fma;
#[cfg(not(any(test, feature = "std")))]
#[allow(unused_imports)]
use crate::fp::Fp as _;
use local::*;

fn acosh_one(x: f64, sh: f64, sl: f64) -> f64 {
    let c = &ONE_CL;
    let y2 = x * (c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5])))));
    let (y1, y2) = polydd3(x, 0.0, &ONE_CH, y2);
    let (y1, y2) = mulddd(y1, y2, x);
    let (y0, mut y1) = fasttwosum(1.0, y1);
    y1 += y2;
    let (y0, y1) = muldd_acc2(y0, y1, sh, sl);
    y0 + y1
}

/// glibc `acosh`.
pub fn acosh(x: f64) -> f64 {
    let ix = x.to_bits();
    if ix >= 0x7ff0_0000_0000_0000 {
        let aix = ix << 1;
        if ix == 0x7ff0_0000_0000_0000 || aix > (0x7ff << 53) {
            return x + x;
        }
        return (x - x) / (x - x);
    }
    if ix <= 0x3ff0_0000_0000_0000 {
        if ix == 0x3ff0_0000_0000_0000 {
            return 0.0;
        }
        return (x - x) / (x - x);
    }
    let g;
    let mut off = 0x3fe;
    let mut t = ix;
    if ix < 0x3ff1_e83e_425a_ee63 {
        let z = x - 1.0;
        let iz = (-0.25) / z;
        let zt = 2.0 * z;
        let sh = zt.sqrt();
        let sl = fma(sh, sh, -zt) * (sh * iz);
        let c = &CL9;
        let z2 = z * z;
        let z4 = z2 * z2;
        let ds = fma(
            sh * z,
            c[0] + z
                * (((c[1] + z * c[2]) + z2 * (c[3] + z * c[4]))
                    + z4 * ((c[5] + z * c[6]) + z2 * (c[7] + z * c[8]))),
            sl,
        );
        let eps = ds * f64::from_bits(0x3ccfc00000000000) - f64::from_bits(0x3970000000000000) * sh;
        let (lb, ub) = (sh + (ds - eps), sh + (ds + eps));
        if lb == ub {
            return lb;
        }
        return acosh_one(z, sh, sl);
    } else if ix < 0x405b_f000_0000_0000 {
        off = 0x3ff;
        let x2h = x * x;
        let wh = x2h - 1.0;
        let wl = fma(x, x, -x2h);
        let sh = wh.sqrt();
        let ish = 0.5 / wh;
        let sl = (wl - fma(sh, sh, -wh)) * (sh * ish);
        let (th, mut tl) = fasttwosum(x, sh);
        tl += sl;
        t = th.to_bits();
        g = tl / th;
    } else if ix < 0x4087_1000_0000_0000 {
        let c = &CL4;
        let z = 1.0 / (x * x);
        g = c[0] + z * (c[1] + z * (c[2] + z * c[3]));
    } else if ix < 0x40e0_1000_0000_0000 {
        let c = &CL3;
        let z = 1.0 / (x * x);
        g = c[0] + z * (c[1] + z * c[2]);
    } else if ix < 0x41ea_0000_0000_0000 {
        let c = &CL2;
        let z = 1.0 / (x * x);
        g = c[0] + z * c[1];
    } else {
        g = 0.0;
    }
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
    let lh = (L1[i1][1] + L2[i2][1]) + L2H * ed;
    let mut ll = dx + L2L * ed;
    ll += g;
    ll += L1[i1][0] + L2[i2][0];
    ll += f;
    let eps = 2.8e-19;
    let (lb, ub) = (lh + (ll - eps), lh + (ll + eps));
    if lb == ub {
        return lb;
    }
    acosh_refine(x, f64::from_bits(0x3ff71547652b82fe) * lb)
}

fn acosh_database(x: f64, mut f: f64) -> f64 {
    for row in &DB {
        if row[0] == x {
            f = row[1] + row[2];
            break;
        }
    }
    f
}

fn acosh_refine(x: f64, a: f64) -> f64 {
    let ix = x.to_bits();
    let (zh, zl);
    if ix < 0x4190_0000_0000_0000 {
        let x2h = x * x;
        let x2l = fma(x, x, -x2h);
        let (wh, wl) = fasttwosum(x2h - 1.0, x2l);
        let sh = wh.sqrt();
        let ish = 0.5 / wh;
        let sl = (ish * sh) * (wl - fma(sh, sh, -wh));
        let (h, mut l) = fasttwosum(x, sh);
        l += sl;
        (zh, zl) = fasttwosum(h, l);
    } else if ix < 0x4330_0000_0000_0000 {
        zh = 2.0 * x;
        zl = -0.5 / x;
    } else {
        zh = x;
        zl = 0.0;
    }
    let mut t = zh.to_bits();
    let e = (t >> 52) as i32 - 0x3ff + (zl == 0.0) as i32;
    t &= u64::MAX >> 12;
    t |= 0x3ff << 52;
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits();
    let i = (v.wrapping_sub(0x3ff << 52)) >> (52 - 16);
    let (i1, i2, i3, i4) = (
        ((i >> 12) & 0x1f) as usize,
        ((i >> 8) & 0xf) as usize,
        ((i >> 4) & 0xf) as usize,
        (i & 0xf) as usize,
    );
    const L20: f64 = f64::from_bits(0x3fd62e42fefa3800);
    const L21: f64 = f64::from_bits(0x3d1ef35793c76800);
    const L22: f64 = f64::from_bits(0xba49ff0342542fc3);
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
    let (sh, sl) = polydd3(xh, xl, &REF_CH, sl);
    let (sh, sl) = muldd_acc2(xh, xl, sh, sl);
    let (sh, sl) = adddd(sh, sl, el1, el2);
    let (sh, sl) = adddd(sh, sl, l1, l2);
    let (v0, v2) = fasttwosum(l0, sh);
    let (v1, v2) = fasttwosum(v2, sl);
    let (v0, v1, v2) = (v0 * 2.0, v1 * 2.0, v2 * 2.0);
    let v1 = nudge(v1, v2);
    let t = v1.to_bits();
    let t0 = v0.to_bits();
    let er = t.wrapping_add(7) & (u64::MAX >> 12);
    let de = ((t0 >> 52) & 0x7ff).wrapping_sub((t >> 52) & 0x7ff);
    let res = v0 + v1;
    if de > 102 || er < 15 {
        return acosh_database(x, res);
    }
    res
}
