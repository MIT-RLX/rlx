// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The double-double helpers of glibc's `sysdeps/ieee754/dbl-64/ddcoremath.h`
//! (CORE-MATH, MIT), as used by the `atanh` / `asinh` / `acosh` ports. Each
//! returns `(hi, lo)` where the C writes `lo` through a pointer.

use crate::fma;

#[inline(always)]
pub(crate) fn fasttwosum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let z = s - x;
    (s, y - z)
}

#[inline(always)]
pub(crate) fn fasttwosub(x: f64, y: f64) -> (f64, f64) {
    let s = x - y;
    let z = x - s;
    (s, z - y)
}

#[inline(always)]
pub(crate) fn adddd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let s = xh + ch;
    let d = s - xh;
    (s, ((ch - d) + (xh + (d - s))) + (xl + cl))
}

#[inline(always)]
pub(crate) fn muldd_acc(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = fma(ch, xh, -ahhh);
    ahhl += alhh + ahlh;
    let h = ahhh + ahhl;
    (h, (ahhh - h) + ahhl)
}

#[inline(always)]
pub(crate) fn muldd_acc2(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = fma(ch, xh, -ahhh);
    ahhl += alhh + ahlh;
    fasttwosum(ahhh, ahhl)
}

#[inline(always)]
pub(crate) fn mulddd(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let ahlh = ch * xl;
    let ahhh = ch * xh;
    let mut ahhl = fma(ch, xh, -ahhh);
    ahhl += ahlh;
    let h = ahhh + ahhl;
    (h, (ahhh - h) + ahhl)
}

#[inline(always)]
pub(crate) fn mulddd3(xh: f64, xl: f64, ch: f64) -> (f64, f64) {
    let hh = xh * ch;
    (hh, fma(ch, xh, -hh) + xl * ch)
}

/// `polydd(xh + xl)`; `l` is the C's in/out `*l` (the low part to start from).
pub(crate) fn polydd(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = c.len() - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = muldd_acc(xh, xl, ch, cl);
        let th = ch + c[i][0];
        let tl = (c[i][0] - th) + ch;
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// `polydd3`: as [`polydd`] with `muldd_acc2` and a FastTwoSum per step.
pub(crate) fn polydd3(xh: f64, xl: f64, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = c.len() - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = muldd_acc2(xh, xl, ch, cl);
        let (th, tl) = fasttwosum(c[i][0], ch);
        ch = th;
        cl += tl + c[i][1];
    }
    (ch, cl)
}

/// The tie nudge every refine step ends with: when `v1` has no bits below
/// its leading 12, move it one ULP towards `v2`'s sign.
#[inline(always)]
pub(crate) fn nudge(v1: f64, v2: f64) -> f64 {
    let t = v1.to_bits();
    if t & (u64::MAX >> 12) != 0 {
        return v1;
    }
    let t = if (v2.to_bits() ^ t) >> 63 != 0 {
        t.wrapping_sub(1)
    } else {
        t.wrapping_add(1)
    };
    f64::from_bits(t)
}

#[inline(always)]
pub(crate) fn twosum(x: f64, y: f64) -> (f64, f64) {
    if x.abs() > y.abs() {
        fasttwosum(x, y)
    } else {
        fasttwosum(y, x)
    }
}

#[inline(always)]
pub(crate) fn fastsum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (sh, sl) = fasttwosum(xh, yh);
    (sh, (xl + yl) + sl)
}

#[inline(always)]
pub(crate) fn sumdd(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (sh, sl) = if xh.abs() > yh.abs() {
        fasttwosum(xh, yh)
    } else {
        fasttwosum(yh, xh)
    };
    (sh, (xl + yl) + sl)
}

#[inline(always)]
pub(crate) fn muldd2(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahhh = ch * xh;
    (ahhh, (ch * xl + cl * xh) + fma(ch, xh, -ahhh))
}

#[inline(always)]
pub(crate) fn muldd3(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let ch = xh * yh;
    let cl1 = fma(xh, yh, -ch);
    let tl0 = xl * yl;
    let tl1 = tl0 + xh * yl;
    let cl2 = tl1 + xl * yh;
    let cl3 = cl1 + cl2;
    fasttwosum(ch, cl3)
}

#[inline(always)]
pub(crate) fn mulddd2(x: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahhh = ch * x;
    (ahhh, cl * x + fma(ch, x, -ahhh))
}

/// `polyddd(x, n, c, l)` over the first `n` rows of `c`.
pub(crate) fn polyddd(x: f64, n: usize, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = n - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = mulddd2(x, ch, cl);
        (ch, cl) = sumdd(c[i][0], c[i][1], ch, cl);
    }
    (ch, cl)
}

/// `polydd2(xh + xl, n, c, l)` over the first `n` rows of `c`.
pub(crate) fn polydd2(xh: f64, xl: f64, n: usize, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = n - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = muldd2(xh, xl, ch, cl);
        (ch, cl) = fastsum(c[i][0], c[i][1], ch, cl);
    }
    (ch, cl)
}

/// `polyd(x, n, c)`: Horner over the high parts of the first `n` rows.
pub(crate) fn polyd(x: f64, n: usize, c: &[[f64; 2]]) -> f64 {
    let mut i = n - 1;
    let mut ch = c[i][0];
    while i > 0 {
        i -= 1;
        ch = c[i][0] + x * ch;
    }
    ch
}

/// `polydddfst(x, n, c, l)` over the first `n` rows of `c`.
pub(crate) fn polydddfst(x: f64, n: usize, c: &[[f64; 2]], l: f64) -> (f64, f64) {
    let mut i = n - 1;
    let (mut ch, mut cl) = fasttwosum(c[i][0], l);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        (ch, cl) = mulddd2(x, ch, cl);
        (ch, cl) = fastsum(c[i][0], c[i][1], ch, cl);
    }
    (ch, cl)
}
