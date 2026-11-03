// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's double `asin` / `acos` (IBM Accurate Mathematical Library,
//! `sysdeps/ieee754/dbl-64/e_asin.c`) as x86-64 runs them on an FMA CPU:
//! `__ieee754_asin_fma` / `__ieee754_acos_fma`, gcc fusing every product whose
//! uses are all adds or subtracts.

#[path = "asin_data.rs"]
mod data;
use crate::fma;
use data::{ASNCS, INROOT, POWTWO};

const F1: f64 = f64::from_bits(0x3fc55555555554f9);
const F2: f64 = f64::from_bits(0x3fb333333336127d);
const F3: f64 = f64::from_bits(0x3fa6db6dae42c0e4);
const F4: f64 = f64::from_bits(0x3f9f1c7e04f4ad99);
const F5: f64 = f64::from_bits(0x3f96e442c822d419);
const F6: f64 = f64::from_bits(0x3f9292d80f453c72);
const RT0: f64 = f64::from_bits(0x3fefffffffecc1dd);
const RT1: f64 = f64::from_bits(0x3fdfffffff757304);
const RT2: f64 = f64::from_bits(0x3fd800496769c91a);
const RT3: f64 = f64::from_bits(0x3fd4006318d1dab9);
const HP0: f64 = f64::from_bits(0x3ff921fb54442d18);
const HP1: f64 = f64::from_bits(0x3c91a62633145c07);
const T24: f64 = 16777216.0;
const T27: f64 = 134217728.0;

#[inline(always)]
fn a(n: usize) -> f64 {
    f64::from_bits(ASNCS[n])
}

/// `((((f6·z + f5)·z + f4)·z + f3)·z + f2)·z + f1`, fused.
#[inline(always)]
fn fpoly(z: f64) -> f64 {
    let mut q = fma(F6, z, F5);
    q = fma(q, z, F4);
    q = fma(q, z, F3);
    q = fma(q, z, F2);
    fma(q, z, F1)
}

/// The table arms: `xx = |x| − x0`, `t = c1·xx + (xx²·P(xx) + c_d)` with `P`
/// of `deg` coefficients from `n + 2`; returns `(t, the arm's last entry)`.
#[inline(always)]
fn table(n: usize, deg: usize, ax: f64) -> (f64, f64) {
    let xx = ax - a(n);
    let mut q = a(n + 1 + deg);
    for j in (2..1 + deg).rev() {
        q = fma(xx, q, a(n + j));
    }
    let p = fma(xx * xx, q, a(n + 2 + deg));
    (fma(a(n + 1), xx, p), a(n + 3 + deg))
}

/// Table index and polynomial degree of `|x|` (high word `k`) in
/// `[0.125, 0.96875)`.
#[inline(always)]
fn arm(k: u32) -> (usize, usize) {
    if k < 0x3fe0_0000 {
        let n = if k < 0x3fd0_0000 {
            11 * ((k & 0x000f_ffff) >> 15)
        } else {
            11 * ((k & 0x000f_ffff) >> 14) + 352
        };
        (n as usize, 5)
    } else if k < 0x3fe8_0000 {
        ((1056 + ((k & 0x000f_e000) >> 11) * 3) as usize, 6)
    } else if k < 0x3fed_8000 {
        ((992 + ((k & 0x000f_e000) >> 13) * 13) as usize, 7)
    } else if k < 0x3fee_8000 {
        ((884 + ((k & 0x000f_e000) >> 13) * 14) as usize, 8)
    } else {
        ((768 + ((k & 0x000f_e000) >> 13) * 15) as usize, 9)
    }
}

/// `√z`-based pieces of the arm near 1, `z = (1 − |x|)/2`: `(y, cc, p)`
/// with `y + cc ≈ √z` and `p = z·fpoly(z)`. `y` splits `c` at 2^24
/// (`asin`) or 2^27 (`acos`).
#[inline(always)]
fn near_one(z: f64, acos: bool) -> (f64, f64, f64) {
    let k = (z.to_bits() >> 32) as u32;
    let mut t = INROOT[((k & 0x001f_ffff) >> 14) as usize] * POWTWO[(511 - (k >> 21)) as usize];
    let r = fma(-(t * t), z, 1.0);
    t *= fma(r, fma(r, fma(r, RT3, RT2), RT1), RT0);
    let c = t * z;
    let w = fma(-(0.5 * t), c, 1.5);
    let y = if acos {
        fma(-T27, c, fma(T27, c, c))
    } else {
        (c + T24) - T24
    };
    // `t = c·w` is only added (`t + y`), so fused there
    let cc = fma(-y, y, z) / fma(c, w, y);
    (y, cc, fpoly(z) * z)
}

/// glibc `asin`.
pub fn asin(x: f64) -> f64 {
    let bits = x.to_bits();
    let m = (bits >> 32) as u32 as i32;
    let k = (m as u32) & 0x7fff_ffff;
    let pos = m > 0;
    if k < 0x3e50_0000 {
        return x;
    }
    if k < 0x3fc0_0000 {
        let x2 = x * x;
        return fma(fpoly(x2), x2 * x, x);
    }
    if k < 0x3fef_0000 {
        let (n, deg) = arm(k);
        let (t, last) = table(n, deg, x.abs());
        let res = last + t;
        return if pos { res } else { -res };
    }
    if k < 0x3ff0_0000 {
        let z = 0.5 * (if pos { 1.0 - x } else { 1.0 + x });
        let (y, cc, p) = near_one(z, false);
        let cor = fma(-(2.0 * (y + cc)), p, fma(-2.0, cc, HP1));
        let res = fma(-2.0, y, HP0) + cor;
        return if pos { res } else { -res };
    }
    if k == 0x3ff0_0000 && bits as u32 == 0 {
        return if pos { HP0 } else { -HP0 };
    }
    (x - x) / (x - x)
}

/// glibc `acos`.
pub fn acos(x: f64) -> f64 {
    let bits = x.to_bits();
    let m = (bits >> 32) as u32 as i32;
    let k = (m as u32) & 0x7fff_ffff;
    let pos = m > 0;
    if k < 0x3c88_0000 {
        return HP0;
    }
    if k < 0x3fc0_0000 {
        let x2 = x * x;
        let r = HP0 - x;
        let cor = fma(-fpoly(x2), x2 * x, ((HP0 - r) - x) + HP1);
        return r + cor;
    }
    if k < 0x3fef_0000 {
        let (n, deg) = arm(k);
        let (t, last) = table(n, deg, x.abs());
        let (y, t) = if pos {
            (HP0 - last, HP1 - t)
        } else {
            (HP0 + last, HP1 + t)
        };
        return y + t;
    }
    if k < 0x3ff0_0000 {
        let z = 0.5 * (if pos { 1.0 - x } else { 1.0 + x });
        let (y, cc, p) = near_one(z, true);
        // Here gcc leaves `(y + cc)·p` unfused in both arms (found against
        // glibc 2.43; fusing it moved ~2·10^-4 of the results by an ULP).
        let res = if m < 0 {
            let cor = (HP1 - cc) - (y + cc) * p;
            (HP0 - y) + cor
        } else {
            y + (cc + p * (y + cc))
        };
        return res + res;
    }
    if k == 0x3ff0_0000 && bits as u32 == 0 {
        return if pos { 0.0 } else { 2.0 * HP0 };
    }
    (x - x) / (x - x)
}
