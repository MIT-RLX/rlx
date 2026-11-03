// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's double `atan` (IBM Accurate Mathematical Library,
//! `sysdeps/ieee754/dbl-64/s_atan.c`) as x86-64 runs it on an FMA CPU:
//! `__atan_fma`, gcc fusing each product into the add that consumes it and
//! `EMULV` an exact `fma` product.

#[path = "atan_data.rs"]
mod data;
use crate::fma;
use data::CIJ;

const D3: f64 = f64::from_bits(0xbfd5555555555555);
const D5: f64 = f64::from_bits(0x3fc99999999997fd);
const D7: f64 = f64::from_bits(0xbfc24924923f7603);
const D9: f64 = f64::from_bits(0x3fbc71c6e5129a3b);
const D11: f64 = f64::from_bits(0xbfb7458022b13c25);
const D13: f64 = f64::from_bits(0x3fb375f08b31cbce);
const A: f64 = f64::from_bits(0x3e4bb67a00000000);
const B: f64 = f64::from_bits(0x3fb0000000000000);
const C: f64 = 1.0;
const D: f64 = 16.0;
const E: f64 = f64::from_bits(0x43349ff200000000);
const HPI: f64 = f64::from_bits(0x3ff921fb54442d18);
const MHPI: f64 = f64::from_bits(0xbff921fb54442d18);
const HPI1: f64 = f64::from_bits(0x3c91a62633145c07);
const TWO52: f64 = 4503599627370496.0;

#[inline(always)]
fn c(i: usize, j: usize) -> f64 {
    f64::from_bits(CIJ[i][j])
}

/// `d3·x³ + … + d13·x¹³` over `x·v`, `v = x²`: `yy·(x·v)`.
#[inline(always)]
fn odd_poly(v: f64) -> f64 {
    let mut yy = fma(v, D13, D11);
    yy = fma(v, yy, D9);
    yy = fma(v, yy, D7);
    yy = fma(v, yy, D5);
    fma(v, yy, D3)
}

/// Table interval of `u` in `[1/16, 1)`.
#[inline(always)]
fn interval(u: f64) -> usize {
    (fma(256.0, u, TWO52) - TWO52) as i32 as usize - 16
}

/// glibc `atan`.
pub fn atan(x: f64) -> f64 {
    let bits = x.to_bits();
    let (ux, dx) = ((bits >> 32) as u32, bits as u32);
    if ux & 0x7ff0_0000 == 0x7ff0_0000 && ((ux & 0x000f_ffff) | dx) != 0 {
        return x + x;
    }
    let u = x.abs();
    if u < C {
        if u < B {
            if u < A {
                return x;
            }
            let v = x * x;
            // `yy *= x * v; y = x + yy`: the product is fused into the add
            return fma(odd_poly(v), x * v, x);
        }
        let i = interval(u);
        let z = u - c(i, 0);
        let mut yy = fma(z, c(i, 6), c(i, 5));
        yy = fma(z, yy, c(i, 4));
        yy = fma(z, yy, c(i, 3));
        yy = fma(z, yy, c(i, 2));
        // `yy *= z; y = t1 + yy`, fused
        return fma(yy, z, c(i, 1)).copysign(x);
    }
    if u < D {
        let w = 1.0 / u;
        let t1 = w * u;
        let t2 = fma(w, u, -t1);
        // `ww = w·((1 − t1) − t2)` is only added, so fused into `z`
        let i = interval(w);
        let z = fma(w, (1.0 - t1) - t2, w - c(i, 0));
        let mut yy = fma(z, c(i, 6), c(i, 5));
        yy = fma(z, yy, c(i, 4));
        yy = fma(z, yy, c(i, 3));
        yy = fma(z, yy, c(i, 2));
        yy = fma(-z, yy, HPI1);
        let t1 = HPI - c(i, 1);
        return (t1 + yy).copysign(x);
    }
    if u < E {
        let w = 1.0 / u;
        let v = w * w;
        let t1 = w * u;
        let t2 = fma(w, u, -t1);
        // ESUB(HPI, w): HPI > w here
        let t3 = HPI - w;
        let cor = (HPI - t3) - w;
        // `((HPI1 + cor) − ww) − yy`, `ww` and `yy` products fused in turn
        let yy = fma(-w, (1.0 - t1) - t2, HPI1 + cor);
        let yy = fma(-odd_poly(v), w * v, yy);
        return (t3 + yy).copysign(x);
    }
    if x > 0.0 { HPI } else { MHPI }
}

const INV16: f64 = 0.0625;
const OPI: f64 = f64::from_bits(0x400921fb54442d18);
const OPI1: f64 = f64::from_bits(0x3ca1a62633145c07);
const MOPI: f64 = f64::from_bits(0xc00921fb54442d18);
const QPI: f64 = f64::from_bits(0x3fe921fb54442d18);
const MQPI: f64 = f64::from_bits(0xbfe921fb54442d18);
const TQPI: f64 = f64::from_bits(0x4002d97c7f3321d2);
const MTQPI: f64 = f64::from_bits(0xc002d97c7f3321d2);
const TWO500: f64 = f64::from_bits(0x5f30000000000000);
const TWOM500: f64 = f64::from_bits(0x20b0000000000000);

/// `c2 + v·(c3 + v·(c4 + v·(c5 + v·c6)))` of interval `i`, fused.
#[inline(always)]
fn tab_poly(i: usize, v: f64) -> f64 {
    let mut p = fma(v, c(i, 6), c(i, 5));
    p = fma(v, p, c(i, 4));
    p = fma(v, p, c(i, 3));
    fma(v, p, c(i, 2))
}

/// glibc `atan2` (`e_atan2.c`, `__ieee754_atan2_fma`). In the small-ratio
/// arms of (ii)–(iv) gcc leaves `zz = u·v·P` unfused at the end of the
/// `t3 = ((c + cor) ± du) ± zz` chain (found against glibc 2.43: fusing it
/// moved 7 in 6·10^7 results by an ULP).
pub fn atan2(y: f64, x: f64) -> f64 {
    const EP: i32 = 59768832; // 57·16^5
    let (bx, by) = (x.to_bits(), y.to_bits());
    let (ux, dx) = ((bx >> 32) as u32, bx as u32);
    let (uy, dy) = ((by >> 32) as u32, by as u32);
    if ux & 0x7ff0_0000 == 0x7ff0_0000 && ((ux & 0x000f_ffff) | dx) != 0 {
        return x + y;
    }
    if uy & 0x7ff0_0000 == 0x7ff0_0000 && ((uy & 0x000f_ffff) | dy) != 0 {
        return y + y;
    }
    let xneg = ux & 0x8000_0000 != 0;
    if dy == 0 && uy == 0 {
        return if xneg { OPI } else { 0.0 };
    }
    if dy == 0 && uy == 0x8000_0000 {
        return if xneg { MOPI } else { -0.0 };
    }
    if x == 0.0 {
        return if uy & 0x8000_0000 == 0 { HPI } else { MHPI };
    }
    if dx == 0 && (ux == 0x7ff0_0000 || ux == 0xfff0_0000) {
        let pos = ux == 0x7ff0_0000;
        if uy == 0x7ff0_0000 && dy == 0 {
            return if pos { QPI } else { TQPI };
        }
        if uy == 0xfff0_0000 && dy == 0 {
            return if pos { MQPI } else { MTQPI };
        }
        if uy != 0x7ff0_0000 && uy != 0xfff0_0000 {
            let ypos = uy & 0x8000_0000 == 0;
            return match (pos, ypos) {
                (true, true) => 0.0,
                (true, false) => -0.0,
                (false, true) => OPI,
                (false, false) => MOPI,
            };
        }
    }
    if dy == 0 && uy == 0x7ff0_0000 {
        return HPI;
    }
    if dy == 0 && uy == 0xfff0_0000 {
        return MHPI;
    }
    let mut ax = x.abs();
    let mut ay = y.abs();
    let de = (uy & 0x7ff0_0000) as i32 - (ux & 0x7ff0_0000) as i32;
    if de >= EP {
        return if y > 0.0 { HPI } else { MHPI };
    }
    if de <= -EP {
        if x > 0.0 {
            return (ay / ax).copysign(y);
        }
        return if y > 0.0 { OPI } else { MOPI };
    }
    if ax < TWOM500 || ay < TWOM500 {
        ax *= TWO500;
        ay *= TWO500;
    }
    if ax > TWO500 || ay > TWO500 {
        ax *= TWOM500;
        ay *= TWOM500;
    }
    // u = min/max, du its correction (EMULV an exact `fma` product)
    let (u, du) = if ay < ax {
        let u = ay / ax;
        let v = ax * u;
        let vv = fma(ax, u, -v);
        (u, ((ay - v) - vv) / ax)
    } else {
        let u = ax / ay;
        let v = ay * u;
        let vv = fma(ay, u, -v);
        (u, ((ax - v) - vv) / ay)
    };
    let small = |u: f64| {
        let v = u * u;
        (u * v, odd_poly(v))
    };
    let z = if x > 0.0 {
        if ay < ax {
            // (i) atan(ay/ax)
            if u < INV16 {
                let (uv, p) = small(u);
                u + fma(uv, p, du)
            } else {
                let i = interval(u);
                let t3 = u - c(i, 0);
                // EADD(t3, du)
                let v = t3 + du;
                let dv = if t3.abs() > du.abs() {
                    (t3 - v) + du
                } else {
                    (du - v) + t3
                };
                let (t1, t2) = (c(i, 1), c(i, 2));
                let q = fma(v, fma(v, fma(v, c(i, 6), c(i, 5)), c(i, 4)), c(i, 3));
                let zz = fma(v, t2, fma(dv, t2, (v * v) * q));
                t1 + zz
            }
        } else if u < INV16 {
            // (ii) π/2 − atan(ax/ay)
            let (uv, p) = small(u);
            let t2 = HPI - u;
            let cor = (HPI - t2) - u;
            t2 + (((HPI1 + cor) - du) - uv * p)
        } else {
            let i = interval(u);
            let v = (u - c(i, 0)) + du;
            (HPI - c(i, 1)) + fma(-v, tab_poly(i, v), HPI1)
        }
    } else if ax < ay {
        // (iii) π/2 + atan(ax/ay)
        if u < INV16 {
            let (uv, p) = small(u);
            let t2 = HPI + u;
            let cor = (HPI - t2) + u;
            t2 + (((HPI1 + cor) + du) + uv * p)
        } else {
            let i = interval(u);
            let v = (u - c(i, 0)) + du;
            (HPI + c(i, 1)) + fma(v, tab_poly(i, v), HPI1)
        }
    } else if u < INV16 {
        // (iv) π − atan(ay/ax)
        let (uv, p) = small(u);
        let t2 = OPI - u;
        let cor = (OPI - t2) - u;
        t2 + (((OPI1 + cor) - du) - uv * p)
    } else {
        let i = interval(u);
        let v = (u - c(i, 0)) + du;
        (OPI - c(i, 1)) + fma(-v, tab_poly(i, v), OPI1)
    };
    z.copysign(y)
}
