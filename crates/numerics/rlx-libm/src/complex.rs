// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's `hypot` (`sysdeps/ieee754/dbl-64/e_hypot.c`, the non-FMA kernel x86-64
//! builds) and the complex `cexp` / `csqrt` (`math/s_cexp_template.c`,
//! `s_csqrt_template.c`, generic: no contraction), on this crate's `exp`, `sin`,
//! `cos`. Complex values are `(re, im)`.

/// glibc `scalbn(x, n)`.
fn scalbn(x: f64, n: i32) -> f64 {
    let mut y = x;
    let mut n = n;
    let p = |e: i32| f64::from_bits(((0x3ff + e) as u64) << 52);
    if n > 1023 {
        y *= p(1023);
        n -= 1023;
        if n > 1023 {
            y *= p(1023);
            n -= 1023;
            n = n.min(1023);
        }
    } else if n < -1022 {
        y *= p(-1022) * p(53);
        n += 1022 - 53;
        if n < -1022 {
            y *= p(-1022) * p(53);
            n += 1022 - 53;
            n = n.max(-1022);
        }
    }
    y * p(n)
}

fn hypot_kernel(ax: f64, ay: f64) -> f64 {
    let mut h = (ax * ax + ay * ay).sqrt();
    let (t1, t2);
    if h <= 2.0 * ay {
        let delta = h - ay;
        t1 = ax * (2.0 * delta - ax);
        t2 = (delta - 2.0 * (ax - ay)) * delta;
    } else {
        let delta = h - ax;
        t1 = 2.0 * delta * (ax - 2.0 * ay);
        t2 = (4.0 * delta - ay) * ay + delta * delta;
    }
    h -= (t1 + t2) / (2.0 * h);
    h
}

/// glibc `hypot(x, y)`.
pub fn hypot(x: f64, y: f64) -> f64 {
    const SCALE: f64 = f64::from_bits(0x1a70_0000_0000_0000); // 0x1p-600
    const LARGE_VAL: f64 = f64::from_bits(0x5fe0_0000_0000_0000); // 0x1p+511
    const TINY_VAL: f64 = f64::from_bits(0x2340_0000_0000_0000); // 0x1p-459
    const EPS: f64 = f64::from_bits(0x3c90_0000_0000_0000); // 0x1p-54
    if !x.is_finite() || !y.is_finite() {
        if x.is_infinite() || y.is_infinite() {
            return f64::INFINITY;
        }
        return x + y;
    }
    let (x, y) = (x.abs(), y.abs());
    let ax = if x < y { y } else { x };
    let ay = if x < y { x } else { y };
    if ax > LARGE_VAL {
        if ay <= ax * EPS {
            return ax + ay;
        }
        return hypot_kernel(ax * SCALE, ay * SCALE) / SCALE;
    }
    if ay < TINY_VAL {
        if ax >= ay / EPS {
            return ax + ay;
        }
        return hypot_kernel(ax / SCALE, ay / SCALE) * SCALE;
    }
    if ay <= ax * EPS {
        return ax + ay;
    }
    hypot_kernel(ax, ay)
}

/// glibc `cexp(re + i·im)`.
pub fn cexp(re: f64, im: f64) -> (f64, f64) {
    let rfin = re.is_finite();
    let ifin = im.is_finite();
    if rfin {
        if ifin {
            const T: f64 = 709.0; // (int) ((DBL_MAX_EXP - 1) * M_LN2)
            let (mut sinix, mut cosix) = if im.abs() > f64::MIN_POSITIVE {
                (crate::sin(im), crate::cos(im))
            } else {
                (im, 1.0)
            };
            let mut re = re;
            if re > T {
                let exp_t = crate::exp(T);
                re -= T;
                sinix *= exp_t;
                cosix *= exp_t;
                if re > T {
                    re -= T;
                    sinix *= exp_t;
                    cosix *= exp_t;
                }
            }
            if re > T {
                return (f64::MAX * cosix, f64::MAX * sinix);
            }
            let exp_val = crate::exp(re);
            return (exp_val * cosix, exp_val * sinix);
        }
        return (f64::NAN, f64::NAN);
    }
    if re.is_infinite() {
        if ifin {
            let value = if re.is_sign_negative() {
                0.0
            } else {
                f64::INFINITY
            };
            if im == 0.0 {
                return (value, im);
            }
            let (sinix, cosix) = if im.abs() > f64::MIN_POSITIVE {
                (crate::sin(im), crate::cos(im))
            } else {
                (im, 1.0)
            };
            return (value.copysign(cosix), value.copysign(sinix));
        }
        if !re.is_sign_negative() {
            return (f64::INFINITY, im - im);
        }
        return (0.0, 0.0f64.copysign(im));
    }
    (f64::NAN, if im == 0.0 { im } else { f64::NAN })
}

/// glibc `csqrt(re + i·im)`.
pub fn csqrt(re: f64, im: f64) -> (f64, f64) {
    let (rcls_finite, icls_finite) = (re.is_finite(), im.is_finite());
    if !rcls_finite || !icls_finite {
        if im.is_infinite() {
            return (f64::INFINITY, im);
        }
        if re.is_infinite() {
            if re < 0.0 {
                return (
                    if im.is_nan() { f64::NAN } else { 0.0 },
                    f64::INFINITY.copysign(im),
                );
            }
            return (
                re,
                if im.is_nan() {
                    f64::NAN
                } else {
                    0.0f64.copysign(im)
                },
            );
        }
        return (f64::NAN, f64::NAN);
    }
    if im == 0.0 {
        if re < 0.0 {
            return (0.0, (-re).sqrt().copysign(im));
        }
        return (re.sqrt().abs(), 0.0f64.copysign(im));
    }
    if re == 0.0 {
        let r = if im.abs() >= 2.0 * f64::MIN_POSITIVE {
            (0.5 * im.abs()).sqrt()
        } else {
            0.5 * (2.0 * im.abs()).sqrt()
        };
        return (r, r.copysign(im));
    }
    let (mut xr, mut xi) = (re, im);
    let mut scale = 0i32;
    if xr.abs() > f64::MAX / 4.0 {
        scale = 1;
        xr = scalbn(xr, -2);
        xi = scalbn(xi, -2);
    } else if xi.abs() > f64::MAX / 4.0 {
        scale = 1;
        xr = if xr.abs() >= 4.0 * f64::MIN_POSITIVE {
            scalbn(xr, -2)
        } else {
            0.0
        };
        xi = scalbn(xi, -2);
    } else if xr.abs() < 2.0 * f64::MIN_POSITIVE && xi.abs() < 2.0 * f64::MIN_POSITIVE {
        scale = -((53 + 1) / 2);
        xr = scalbn(xr, -2 * scale);
        xi = scalbn(xi, -2 * scale);
    }
    let d = hypot(xr, xi);
    let (mut r, mut s);
    if xr > 0.0 {
        r = (0.5 * (d + xr)).sqrt();
        if scale == 1 && xi.abs() < 1.0 {
            s = xi / r;
            r = scalbn(r, scale);
            scale = 0;
        } else {
            s = 0.5 * (xi / r);
        }
    } else {
        s = (0.5 * (d - xr)).sqrt();
        if scale == 1 && xi.abs() < 1.0 {
            r = (xi / s).abs();
            s = scalbn(s, scale);
            scale = 0;
        } else {
            r = (0.5 * (xi / s)).abs();
        }
    }
    if scale != 0 {
        r = scalbn(r, scale);
        s = scalbn(s, scale);
    }
    (r, s.copysign(xi))
}
