// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc 2.43's `atanh`: CORE-MATH's correctly rounded binary64 `atanh`
//! (`sysdeps/ieee754/dbl-64/e_atanh.c`, MIT), statement for statement.
//! Correctly rounded, so where gcc fuses does not reach the result.

#[path = "atanh_data.rs"]
mod data;
use crate::dd::*;
use crate::fma;
use data::*;

fn atanh_zero(x: f64) -> f64 {
    let x2 = x * x;
    let x2l = fma(x, x, -x2);
    let y2 = x2 * (CL[0] + x2 * (CL[1] + x2 * (CL[2] + x2 * (CL[3] + x2 * CL[4]))));
    let (y1, y2) = polydd3(x2, x2l, &CH, y2);
    let (y1, y2) = mulddd3(y1, y2, x);
    let (y1, y2) = muldd_acc2(x2, x2l, y1, y2);
    let (y0, y1) = fasttwosum(x, y1);
    let (y1, y2) = fasttwosum(y1, y2);
    y0 + nudge(y1, y2)
}

/// glibc `atanh`.
pub fn atanh(x: f64) -> f64 {
    let ax = x.abs();
    let aix = ax.to_bits();
    if aix >= 0x3ff0_0000_0000_0000 {
        if aix == 0x3ff0_0000_0000_0000 {
            // __math_divzero(x < 0)
            return (if x < 0.0 { -1.0 } else { 1.0 }) / 0.0;
        }
        if aix > 0x7ff0_0000_0000_0000 {
            return x + x;
        }
        return (x - x) / (x - x);
    }
    if aix < 0x3fd0_0000_0000_0000 {
        if aix < 0x3e4d_12ed_0af1_a27f {
            return fma(x, f64::from_bits(0x3c80_0000_0000_0000), x); // 0x1p-55
        }
        let x2 = x * x;
        const C: [f64; 9] = [
            f64::from_bits(0x3fc999999999999a),
            f64::from_bits(0x3fc2492492492244),
            f64::from_bits(0x3fbc71c71c79715f),
            f64::from_bits(0x3fb745d16f777723),
            f64::from_bits(0x3fb3b13ca4174634),
            f64::from_bits(0x3fb110c9724989bd),
            f64::from_bits(0x3fae2d17608a5b2e),
            f64::from_bits(0x3faa0b56308cba0b),
            f64::from_bits(0x3fafb6341208ad2e),
        ];
        let dx2 = fma(x, x, -x2);
        let x4 = x2 * x2;
        let x3 = x2 * x;
        let x8 = x4 * x4;
        let dx3 = fma(x2, x, -x3) + dx2 * x;
        let p = (C[0] + x2 * C[1])
            + x4 * (C[2] + x2 * C[3])
            + x8 * ((C[4] + x2 * C[5]) + x4 * (C[6] + x2 * C[7]) + x8 * C[8]);
        let t = fma(x2, p, f64::from_bits(0x3c75555555555555)); // 0x1.5555555555555p-56
        let (ph, pl) = fasttwosum(f64::from_bits(0x3fd5555555555555), t);
        let (ph, pl) = muldd_acc(ph, pl, x3, dx3);
        let (ph, tl) = fasttwosum(x, ph);
        let pl = pl + tl;
        let eps =
            x * (x4 * f64::from_bits(0x3cad000000000000) + f64::from_bits(0x3980000000000000));
        let (lb, ub) = (ph + (pl - eps), ph + (pl + eps));
        if lb == ub {
            return lb;
        }
        return atanh_zero(x);
    }
    let (ph, pl) = fasttwosum(1.0, ax);
    let (qh, ql) = fasttwosub(1.0, ax);
    let iqh = 1.0 / qh;
    let th = ph * iqh;
    let tl = fma(ph, iqh, -th) + (pl + ph * (fma(-qh, iqh, 1.0) - ql * iqh)) * iqh;
    const C: [f64; 5] = [
        -1.0,
        f64::from_bits(0x3ff5555555555530),
        f64::from_bits(0xbfffffffffffffa0),
        f64::from_bits(0x40099999e33a6366),
        f64::from_bits(0xc01555559ef9525f),
    ];
    let mut t = th.to_bits();
    let e = (t >> 52) as i32 - 0x3ff;
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
    let tf = f64::from_bits(t);
    let r = (0.5 * R1[i1]) * R2[i2];
    let dx = fma(r, tf, -0.5);
    let dx2 = dx * dx;
    let rx = r * tf;
    let dxl = fma(r, tf, -rx);
    let f = dx2 * ((C[0] + dx * C[1]) + dx2 * (C[2] + dx * C[3] + dx2 * C[4]));
    const L2H: f64 = f64::from_bits(0x3fd62e42fefa3a00);
    const L2L: f64 = f64::from_bits(0xbcd0ca86c3898d00);
    let lh = (L1[i1][1] + L2[i2][1]) + L2H * ed;
    let (lh, mut ll) = fasttwosum(lh, rx - 0.5);
    ll += L2L * ed + (L1[i1][0] + L2[i2][0]) + dxl + 0.5 * tl / th;
    ll += f;
    let s = 1.0f64.copysign(x);
    let (lh, ll) = (lh * s, ll * s);
    let eps = 38e-24 + dx2 * f64::from_bits(0x3ce0000000000000); // 0x1p-49
    let (lb, ub) = (lh + (ll - eps), lh + (ll + eps));
    if lb == ub {
        return lb;
    }
    let (th, tl) = fasttwosum(th, tl);
    atanh_refine(
        x,
        th,
        tl,
        f64::from_bits(0x40071547652b82fe) * (lh + ll).abs(),
    )
}

fn atanh_database(x: f64, mut f: f64) -> f64 {
    const DB: [[f64; 3]; 2] = [
        [
            f64::from_bits(0x3fd2dbb7b1c91363),
            f64::from_bits(0x3fd36f33d51c264d),
            f64::from_bits(0x3c70000000000000),
        ],
        [
            f64::from_bits(0x3fdc493dc899e4a5),
            f64::from_bits(0x3fde611aa58ab608),
            f64::from_bits(0xbc70000000000000),
        ],
    ];
    let ax = x.abs();
    let sgn = 1.0f64.copysign(x);
    for row in &DB {
        if row[0] == ax {
            f = sgn * row[1] + sgn * row[2];
            break;
        }
    }
    f
}

fn atanh_refine(x: f64, zh: f64, zl: f64, a: f64) -> f64 {
    const CLR: [f64; 3] = [
        f64::from_bits(0xbfc0000000000000),
        f64::from_bits(0x3fb9999999a0754f),
        f64::from_bits(0xbfb55555555c3157),
    ];
    let mut t = zh.to_bits();
    let e = (t >> 52) as i32 - 0x3ff;
    t &= u64::MAX >> 12;
    t |= 0x3ff << 52;
    let ed = e as f64;
    let v = (a - ed + f64::from_bits(0x3ff0000800000000)).to_bits(); // 0x1.00008p+0
    let i = (v.wrapping_sub(0x3ff << 52)) >> (52 - 16);
    let (i1, i2, i3, i4) = (
        (i >> 12) as usize,
        ((i >> 8) & 0xf) as usize,
        ((i >> 4) & 0xf) as usize,
        (i & 0xf) as usize,
    );
    const L20: f64 = f64::from_bits(0x3fd62e42fefa3a00);
    const L21: f64 = f64::from_bits(0xbcd0ca86c3898d00);
    const L22: f64 = f64::from_bits(0x397f97b57a079a00);
    let (el2, el1, el0) = (L22 * ed, L21 * ed, L20 * ed);
    let ll = |k: usize, m: usize, c: usize| LL[k * 17 + m][c];
    let l0 = ll(0, i1, 0) + ll(1, i2, 0) + (ll(2, i3, 0) + ll(3, i4, 0)) + el0;
    let l1 = ll(0, i1, 1) + ll(1, i2, 1) + (ll(2, i3, 1) + ll(3, i4, 1));
    let l2 = ll(0, i1, 2) + ll(1, i2, 2) + (ll(2, i3, 2) + ll(3, i4, 2));
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
    let tz = (zl.to_bits() as i64).wrapping_sub((e as i64) << 52) as u64;
    xl += th * f64::from_bits(tz);
    let (xh, xl) = adddd(xh, xl, sh, sl);
    let sl = xh * (CLR[0] + xh * (CLR[1] + xh * CLR[2]));
    let (sh, sl) = polydd(xh, xl, &CH2, sl);
    let (sh, sl) = muldd_acc(xh, xl, sh, sl);
    let (sh, sl) = adddd(sh, sl, el1, el2);
    let (sh, sl) = adddd(sh, sl, l1, l2);
    let (v0, v2) = fasttwosum(l0, sh);
    let (v1, v2) = fasttwosum(v2, sl);
    let v1 = nudge(v1, v2);
    let t = v1.to_bits();
    let t0 = v0.to_bits();
    let er = (t.wrapping_add(1)) & (u64::MAX >> 12);
    let de = ((t0 >> 52) & 0x7ff).wrapping_sub((t >> 52) & 0x7ff);
    let s = 1.0f64.copysign(x);
    let res = v0 * s + v1 * s;
    if de > 104 || er < 3 {
        return atanh_database(x, res);
    }
    res
}
