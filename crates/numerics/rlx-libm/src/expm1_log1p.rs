// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's double `expm1` / `log1p` (fdlibm-derived, `sysdeps/ieee754/dbl-64/
//! s_expm1.c`, `s_log1p.c`) as x86-64 runs them on an FMA CPU (`__expm1_fma`,
//! `__log1p_fma`): gcc fuses a product into its add/subtract when every use of the
//! product is one, in the product's own basic block — so a product reaching a
//! join (`lo` in `expm1`) or used across branches (`hfsq` in `log1p`) stays plain.

use crate::fma;

#[inline]
fn hi_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn with_hi_word(x: f64, hi: u32) -> f64 {
    f64::from_bits((u64::from(hi) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// glibc `expm1(x)`.
pub fn expm1(x: f64) -> f64 {
    const HUGE: f64 = 1.0e+300;
    const TINY: f64 = 1.0e-300;
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const Q: [f64; 6] = [
        1.0,
        -3.33333333333331316428e-02,
        1.58730158725481460165e-03,
        -7.93650757867487942473e-05,
        4.00821782732936239552e-06,
        -2.01099218183624371326e-07,
    ];
    let mut x = x;
    let mut hx = hi_word(x);
    let xsb = hx & 0x8000_0000;
    hx &= 0x7fff_ffff;
    if hx >= 0x4043_687A {
        if hx >= 0x4086_2E42 {
            if hx >= 0x7ff0_0000 {
                let low = x.to_bits() as u32;
                if ((hx & 0xfffff) | low) != 0 {
                    return x + x;
                }
                return if xsb == 0 { x } else { -1.0 };
            }
            if x > O_THRESHOLD {
                return HUGE * HUGE;
            }
        }
        if xsb != 0 {
            return TINY - 1.0;
        }
    }
    let k: i32;
    let mut c = 0.0f64;
    if hx > 0x3fd6_2e42 {
        let (hi, lo);
        if hx < 0x3FF0_A2B2 {
            if xsb == 0 {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = fma(INVLN2, x, if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi = fma(-t, LN2_HI, x);
            // reaches the join (a phi): not fused into its uses
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3c90_0000 {
        let t = HUGE + x;
        return x - (t - (HUGE + x));
    } else {
        k = 0;
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1c = fma(hxs, Q[1], 1.0);
    let h2 = hxs * hxs;
    let r2c = fma(hxs, Q[3], Q[2]);
    let h4 = h2 * h2;
    let r3c = fma(hxs, Q[5], Q[4]);
    let r1 = fma(h4, r3c, fma(h2, r2c, r1c));
    let t = fma(-r1, hfx, 3.0);
    let mut e = hxs * ((r1 - t) / fma(-x, t, 6.0));
    if k == 0 {
        return x - fma(x, e, -hxs);
    }
    e = fma(x, e - c, -c);
    e -= hxs;
    if k == -1 {
        return fma(0.5, x - e, -0.5);
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return fma(2.0, x - e, 1.0);
    }
    if k <= -2 || k > 56 {
        let y = 1.0 - (e - x);
        let y = with_hi_word(y, hi_word(y).wrapping_add((k << 20) as u32));
        return y - 1.0;
    }
    let y;
    if k < 20 {
        let t = with_hi_word(1.0, 0x3ff0_0000 - (0x20_0000 >> k));
        let yy = t - (e - x);
        y = with_hi_word(yy, hi_word(yy).wrapping_add((k << 20) as u32));
    } else {
        let t = with_hi_word(1.0, ((0x3ff - k) << 20) as u32);
        let mut yy = x - (e + t);
        yy += 1.0;
        y = with_hi_word(yy, hi_word(yy).wrapping_add((k << 20) as u32));
    }
    y
}

/// glibc `log1p(x)`.
pub fn log1p(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const TWO54: f64 = 1.80143985094819840000e+16;
    const LP: [f64; 8] = [
        0.0,
        6.666666666666735130e-01,
        3.999999999940941908e-01,
        2.857142874366239149e-01,
        2.222219843214978396e-01,
        1.818357216161805012e-01,
        1.531383769920937332e-01,
        1.479819860511658591e-01,
    ];
    let hx = hi_word(x) as i32;
    let ax = hx & 0x7fff_ffff;
    let mut k = 1i32;
    let mut f = 0.0f64;
    let mut hu = 0i32;
    let mut c = 0.0f64;
    if hx < 0x3FDA_827A {
        if ax >= 0x3ff0_0000 {
            if x == -1.0 {
                return -TWO54 / 0.0;
            }
            return (x - x) / (x - x);
        }
        if ax < 0x3e20_0000 {
            if ax < 0x3c90_0000 {
                return x;
            }
            return fma(-(x * x), 0.5, x);
        }
        if hx > 0 || hx <= 0xbfd2_bec3_u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    } else if hx >= 0x7ff0_0000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x4340_0000 {
            u = 1.0 + x;
            hu = hi_word(u) as i32;
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = hi_word(u) as i32;
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000f_ffff;
        if hu < 0x6a09e {
            u = with_hi_word(u, (hu | 0x3ff0_0000) as u32);
        } else {
            k += 1;
            u = with_hi_word(u, (hu | 0x3fe0_0000) as u32);
            hu = (0x0010_0000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    // used across branches (its blocks differ): plain products
    let hfsq = 0.5 * f * f;
    let kf = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c = fma(kf, LN2_LO, c);
            return fma(kf, LN2_HI, c);
        }
        let r = hfsq * fma(-0.66666666666666666, f, 1.0);
        if k == 0 {
            return f - r;
        }
        return fma(kf, LN2_HI, -((r - fma(kf, LN2_LO, c)) - f));
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let z2 = z * z;
    let r2 = fma(z, LP[3], LP[2]);
    let z4 = z2 * z2;
    let r3 = fma(z, LP[5], LP[4]);
    let z6 = z4 * z2;
    let r4 = fma(z, LP[7], LP[6]);
    // `R1 = z·Lp[1]` is the first product of `R1 + z2·R2`: it fuses, z2·R2 stays
    let r = fma(z6, r4, fma(z4, r3, fma(z, LP[1], z2 * r2)));
    // `s·(hfsq + R)` is common to both returns: gcc hoists it above the branch, out
    // of its uses' blocks, so it is not fused (measured: fusing it misses 0.7 %)
    let sr = s * (hfsq + r);
    if k == 0 {
        return f - (hfsq - sr);
    }
    fma(kf, LN2_HI, -((hfsq - (sr + fma(kf, LN2_LO, c))) - f))
}
