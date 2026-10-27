// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference LAPACK 3.12's dqds (`DLASQ2`–`DLASQ6`): the eigenvalues of a
//! positive definite tridiagonal in qd form, as `DLARRE` calls it. The arrays
//! are **1-based** (`z[0]` unused) so the index arithmetic reads like the
//! Fortran.

use crate::lapack::{PREC as EPS, SFMIN};

/// The dqds state `DLASQ2` threads through `DLASQ3`/`DLASQ4`.
#[derive(Default)]
struct Dqds {
    dmin: f64,
    sigma: f64,
    desig: f64,
    qmax: f64,
    nfail: i64,
    iter: i64,
    ndiv: i64,
    ttype: i64,
    dmin1: f64,
    dmin2: f64,
    dn: f64,
    dn1: f64,
    dn2: f64,
    g: f64,
    tau: f64,
}

/// `DLASRT('D')` on `z[1..=n]`: only the values matter, so any sort does.
fn sort_desc(z: &mut [f64], n: usize) {
    z[1..=n].sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
}

/// `DLASQ2(N, Z, INFO)`. `z` is 1-based of length ≥ 4n+1.
pub(crate) fn dlasq2(n: usize, z: &mut [f64]) -> i32 {
    const CBIAS: f64 = 1.5;
    let eps = EPS;
    let safmin = SFMIN;
    let tol = eps * 100.0;
    let tol2 = tol * tol;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return if z[1] < 0.0 { -201 } else { 0 };
    }
    if n == 2 {
        if z[1] < 0.0 {
            return -201;
        } else if z[2] < 0.0 {
            return -202;
        } else if z[3] < 0.0 {
            return -203;
        } else if z[3] > z[1] {
            z.swap(1, 3);
        }
        z[5] = z[1] + z[2] + z[3];
        if z[2] > z[3] * tol2 {
            let t = 0.5 * ((z[1] - z[3]) + z[2]);
            let mut s = z[3] * (z[2] / t);
            if s <= t {
                s = z[3] * (z[2] / (t * (1.0 + (1.0 + s / t).sqrt())));
            } else {
                s = z[3] * (z[2] / (t + t.sqrt() * (t + s).sqrt()));
            }
            let t = z[1] + (s + z[2]);
            z[3] *= z[1] / t;
            z[1] = t;
        }
        z[2] = z[3];
        z[6] = z[2] + z[1];
        return 0;
    }
    z[2 * n] = 0.0;
    let mut emin = z[2];
    let mut qmax = 0.0f64;
    let mut zmax = 0.0f64;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    let mut k = 1;
    while k <= 2 * (n - 1) {
        if z[k] < 0.0 {
            return -(200 + k as i32);
        } else if z[k + 1] < 0.0 {
            return -(200 + k as i32 + 1);
        }
        d += z[k];
        e += z[k + 1];
        qmax = fmax(qmax, z[k]);
        emin = fmin(emin, z[k + 1]);
        zmax = fmax(fmax(qmax, zmax), z[k + 1]);
        k += 2;
    }
    if z[2 * n - 1] < 0.0 {
        return -(200 + 2 * n as i32 - 1);
    }
    d += z[2 * n - 1];
    qmax = fmax(qmax, z[2 * n - 1]);
    let _ = fmax(qmax, zmax);
    if e == 0.0 {
        for k in 2..=n {
            z[k] = z[2 * k - 1];
        }
        sort_desc(z, n);
        z[2 * n - 1] = d;
        return 0;
    }
    let trace = d + e;
    if trace == 0.0 {
        z[2 * n - 1] = 0.0;
        return 0;
    }
    let ieee = true;
    let mut k = 2 * n;
    while k >= 2 {
        z[2 * k] = 0.0;
        z[2 * k - 1] = z[k];
        z[2 * k - 2] = 0.0;
        z[2 * k - 3] = z[k - 1];
        k -= 2;
    }
    let mut i0 = 1usize;
    let mut n0 = n as isize;
    if CBIAS * z[4 * i0 - 3] < z[4 * n0 as usize - 3] {
        let ipn4 = 4 * (i0 + n0 as usize);
        let mut i4 = 4 * i0;
        while i4 <= 2 * (i0 + n0 as usize - 1) {
            z.swap(i4 - 3, ipn4 - i4 - 3);
            z.swap(i4 - 1, ipn4 - i4 - 5);
            i4 += 4;
        }
    }
    let mut pp = 0usize;
    for _ in 0..2 {
        let n0u = n0 as usize;
        let mut d = z[4 * n0u + pp - 3];
        let mut i4 = 4 * (n0u - 1) + pp;
        while i4 >= 4 * i0 + pp {
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -0.0;
                d = z[i4 - 3];
            } else {
                d = z[i4 - 3] * (d / (d + z[i4 - 1]));
            }
            i4 -= 4;
        }
        emin = z[4 * i0 + pp + 1];
        d = z[4 * i0 + pp - 3];
        let mut i4 = 4 * i0 + pp;
        while i4 <= 4 * (n0u - 1) + pp {
            z[i4 - 2 * pp - 2] = d + z[i4 - 1];
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -0.0;
                z[i4 - 2 * pp - 2] = d;
                z[i4 - 2 * pp] = 0.0;
                d = z[i4 + 1];
            } else if safmin * z[i4 + 1] < z[i4 - 2 * pp - 2]
                && safmin * z[i4 - 2 * pp - 2] < z[i4 + 1]
            {
                let temp = z[i4 + 1] / z[i4 - 2 * pp - 2];
                z[i4 - 2 * pp] = z[i4 - 1] * temp;
                d *= temp;
            } else {
                z[i4 - 2 * pp] = z[i4 + 1] * (z[i4 - 1] / z[i4 - 2 * pp - 2]);
                d = z[i4 + 1] * (d / z[i4 - 2 * pp - 2]);
            }
            emin = fmin(emin, z[i4 - 2 * pp]);
            i4 += 4;
        }
        z[4 * n0u - pp - 2] = d;
        qmax = z[4 * i0 - pp - 2];
        let mut i4 = 4 * i0 - pp + 2;
        while i4 <= 4 * n0u - pp - 2 {
            qmax = fmax(qmax, z[i4]);
            i4 += 4;
        }
        pp = 1 - pp;
    }
    let mut st = Dqds {
        iter: 2,
        ndiv: 2 * (n0 - i0 as isize) as i64,
        qmax,
        ..Default::default()
    };
    let _ = emin;
    for _iwhila in 1..=n + 1 {
        if n0 < 1 {
            // 170
            for k in 2..=n {
                z[k] = z[4 * k - 3];
            }
            sort_desc(z, n);
            let mut e = 0.0;
            for k in (1..=n).rev() {
                e += z[k];
            }
            z[2 * n + 1] = trace;
            z[2 * n + 2] = e;
            z[2 * n + 3] = st.iter as f64;
            z[2 * n + 4] = st.ndiv as f64 / (n * n) as f64;
            z[2 * n + 5] = 100.0 * st.nfail as f64 / st.iter as f64;
            return 0;
        }
        let n0u = n0 as usize;
        st.desig = 0.0;
        st.sigma = if n0u == n { 0.0 } else { -z[4 * n0u - 1] };
        if st.sigma < 0.0 {
            return 1;
        }
        let mut emax = 0.0f64;
        let mut emin = if n0u > i0 { z[4 * n0u - 5].abs() } else { 0.0 };
        let mut qmin = z[4 * n0u - 3];
        st.qmax = qmin;
        let mut i4 = 4 * n0u;
        let mut found = false;
        while i4 >= 8 {
            if z[i4 - 5] <= 0.0 {
                found = true;
                break;
            }
            if qmin >= 4.0 * emax {
                qmin = fmin(qmin, z[i4 - 3]);
                emax = fmax(emax, z[i4 - 5]);
            }
            st.qmax = fmax(st.qmax, z[i4 - 7] + z[i4 - 5]);
            emin = fmin(emin, z[i4 - 5]);
            i4 -= 4;
        }
        if !found {
            i4 = 4;
        }
        let _ = emin;
        i0 = i4 / 4;
        pp = 0;
        if n0u - i0 > 1 {
            let mut dee = z[4 * i0 - 3];
            let mut deemin = dee;
            let mut kmin = i0;
            let mut i4 = 4 * i0 + 1;
            while i4 <= 4 * n0u - 3 {
                dee = z[i4] * (dee / (dee + z[i4 - 2]));
                if dee <= deemin {
                    deemin = dee;
                    kmin = (i4 + 3) / 4;
                }
                i4 += 4;
            }
            if ((kmin - i0) * 2) < n0u - kmin && deemin <= 0.5 * z[4 * n0u - 3] {
                let ipn4 = 4 * (i0 + n0u);
                pp = 2;
                let mut i4 = 4 * i0;
                while i4 <= 2 * (i0 + n0u - 1) {
                    z.swap(i4 - 3, ipn4 - i4 - 3);
                    z.swap(i4 - 2, ipn4 - i4 - 2);
                    z.swap(i4 - 1, ipn4 - i4 - 5);
                    z.swap(i4, ipn4 - i4 - 4);
                    i4 += 4;
                }
            }
        }
        st.dmin = -fmax(0.0, qmin - 2.0 * qmin.sqrt() * emax.sqrt());
        let nbig = 100 * (n0u - i0 + 1);
        let mut converged = false;
        for _ in 0..nbig {
            if i0 as isize > n0 {
                converged = true;
                break;
            }
            dlasq3(i0, &mut n0, z, &mut pp, &mut st, ieee);
            // an early deflation return leaves PP = 2: Fortran's 1 − PP = −1 is
            // never read (the next pass exits), any non-zero value will do
            pp = if pp == 2 { 3 } else { 1 - pp };
            if pp == 0 && n0 - i0 as isize >= 3 {
                let n0u = n0 as usize;
                if z[4 * n0u] <= tol2 * st.qmax || z[4 * n0u - 1] <= tol2 * st.sigma {
                    let mut splt = i0 - 1;
                    st.qmax = z[4 * i0 - 3];
                    let mut emin = z[4 * i0 - 1];
                    let mut oldemn = z[4 * i0];
                    let mut i4 = 4 * i0;
                    while i4 <= 4 * (n0u - 3) {
                        if z[i4] <= tol2 * z[i4 - 3] || z[i4 - 1] <= tol2 * st.sigma {
                            z[i4 - 1] = -st.sigma;
                            splt = i4 / 4;
                            st.qmax = 0.0;
                            emin = z[i4 + 3];
                            oldemn = z[i4 + 4];
                        } else {
                            st.qmax = fmax(st.qmax, z[i4 + 1]);
                            emin = fmin(emin, z[i4 - 1]);
                            oldemn = fmin(oldemn, z[i4]);
                        }
                        i4 += 4;
                    }
                    z[4 * n0u - 1] = emin;
                    z[4 * n0u] = oldemn;
                    i0 = splt + 1;
                }
            }
        }
        if !converged {
            // INFO = 2: undo the shifts and return the partial result
            let mut i1 = i0;
            loop {
                let mut tempq = z[4 * i0 - 3];
                z[4 * i0 - 3] += st.sigma;
                // (the Fortran re-walks I0..N0, not I1..N1)
                for k in i0 + 1..=n0 as usize {
                    let tempe = z[4 * k - 5];
                    z[4 * k - 5] *= tempq / z[4 * k - 7];
                    tempq = z[4 * k - 3];
                    z[4 * k - 3] = z[4 * k - 3] + st.sigma + tempe - z[4 * k - 5];
                }
                if i1 > 1 {
                    let n1 = i1 - 1;
                    while i1 >= 2 && z[4 * i1 - 5] >= 0.0 {
                        i1 -= 1;
                    }
                    st.sigma = -z[4 * n1 - 1];
                } else {
                    break;
                }
            }
            for k in 1..=n {
                z[2 * k - 1] = z[4 * k - 3];
                z[2 * k] = if (k as isize) < n0 { z[4 * k - 1] } else { 0.0 };
            }
            return 2;
        }
    }
    3
}

/// gfortran's `MAX(a, b)`: `b` unless `a > b` (`maxsd`-style, NaN-insensitive here).
#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline]
fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `DLASQ3`.
fn dlasq3(i0: usize, n0: &mut isize, z: &mut [f64], pp: &mut usize, st: &mut Dqds, ieee: bool) {
    const CBIAS: f64 = 1.5;
    let n0in = *n0;
    let eps = EPS;
    let tol = eps * 100.0;
    let tol2 = tol * tol;
    // deflation (10 – 40)
    loop {
        if *n0 < i0 as isize {
            return;
        }
        let n0u = *n0 as usize;
        if n0u == i0 {
            z[4 * n0u - 3] = z[4 * n0u + *pp - 3] + st.sigma;
            *n0 -= 1;
            continue;
        }
        let nn = 4 * n0u + *pp;
        if n0u != i0 + 1 {
            if z[nn - 5] > tol2 * (st.sigma + z[nn - 3]) && z[nn - 2 * *pp - 4] > tol2 * z[nn - 7] {
                // 30
                if z[nn - 9] > tol2 * st.sigma && z[nn - 2 * *pp - 8] > tol2 * z[nn - 11] {
                    break; // 50
                }
            } else {
                z[4 * n0u - 3] = z[4 * n0u + *pp - 3] + st.sigma;
                *n0 -= 1;
                continue;
            }
        }
        // 40
        if z[nn - 3] > z[nn - 7] {
            z.swap(nn - 3, nn - 7);
        }
        let t = 0.5 * ((z[nn - 7] - z[nn - 3]) + z[nn - 5]);
        if z[nn - 5] > z[nn - 3] * tol2 && t != 0.0 {
            let mut s = z[nn - 3] * (z[nn - 5] / t);
            if s <= t {
                s = z[nn - 3] * (z[nn - 5] / (t * (1.0 + (1.0 + s / t).sqrt())));
            } else {
                s = z[nn - 3] * (z[nn - 5] / (t + t.sqrt() * (t + s).sqrt()));
            }
            let t = z[nn - 7] + (s + z[nn - 5]);
            z[nn - 3] *= z[nn - 7] / t;
            z[nn - 7] = t;
        }
        z[4 * n0u - 7] = z[nn - 7] + st.sigma;
        z[4 * n0u - 3] = z[nn - 3] + st.sigma;
        *n0 -= 2;
    }
    // 50
    let n0u = *n0 as usize;
    if *pp == 2 {
        *pp = 0;
    }
    let p = *pp;
    if (st.dmin <= 0.0 || *n0 < n0in) && CBIAS * z[4 * i0 + p - 3] < z[4 * n0u + p - 3] {
        let ipn4 = 4 * (i0 + n0u);
        let mut j4 = 4 * i0;
        while j4 <= 2 * (i0 + n0u - 1) {
            z.swap(j4 - 3, ipn4 - j4 - 3);
            z.swap(j4 - 2, ipn4 - j4 - 2);
            z.swap(j4 - 1, ipn4 - j4 - 5);
            z.swap(j4, ipn4 - j4 - 4);
            j4 += 4;
        }
        if n0u - i0 <= 4 {
            z[4 * n0u + p - 1] = z[4 * i0 + p - 1];
            z[4 * n0u - p] = z[4 * i0 - p];
        }
        st.dmin2 = fmin(st.dmin2, z[4 * n0u + p - 1]);
        z[4 * n0u + p - 1] = fmin(
            fmin(z[4 * n0u + p - 1], z[4 * i0 + p - 1]),
            z[4 * i0 + p + 3],
        );
        z[4 * n0u - p] = fmin(fmin(z[4 * n0u - p], z[4 * i0 - p]), z[4 * i0 - p + 4]);
        st.qmax = fmax(fmax(st.qmax, z[4 * i0 + p - 3]), z[4 * i0 + p + 1]);
        st.dmin = -0.0;
    }
    dlasq4(i0, n0u, z, p, n0in as usize, st);
    let mut go80 = false;
    loop {
        // 70
        dlasq5(i0, n0u, z, p, st, ieee, eps);
        st.ndiv += (n0u - i0 + 2) as i64;
        st.iter += 1;
        if st.dmin >= 0.0 && st.dmin1 >= 0.0 {
            break;
        } else if st.dmin < 0.0
            && st.dmin1 > 0.0
            && z[4 * (n0u - 1) - p] < tol * (st.sigma + st.dn1)
            && st.dn.abs() < tol * st.sigma
        {
            z[4 * (n0u - 1) - p + 2] = 0.0;
            st.dmin = 0.0;
            break;
        } else if st.dmin < 0.0 {
            st.nfail += 1;
            if st.ttype < -22 {
                st.tau = 0.0;
            } else if st.dmin1 > 0.0 {
                st.tau = (st.tau + st.dmin) * (1.0 - 2.0 * eps);
                st.ttype -= 11;
            } else {
                st.tau *= 0.25;
                st.ttype -= 12;
            }
            continue;
        } else if st.dmin.is_nan() {
            if st.tau == 0.0 {
                go80 = true;
                break;
            }
            st.tau = 0.0;
            continue;
        } else {
            go80 = true;
            break;
        }
    }
    if go80 {
        dlasq6(i0, n0u, z, p, st);
        st.ndiv += (n0u - i0 + 2) as i64;
        st.iter += 1;
        st.tau = 0.0;
    }
    // 90
    let t;
    if st.tau < st.sigma {
        st.desig += st.tau;
        t = st.sigma + st.desig;
        st.desig -= t - st.sigma;
    } else {
        t = st.sigma + st.tau;
        st.desig = st.sigma - (t - st.tau) + st.desig;
    }
    st.sigma = t;
}

/// `DLASQ4`: the shift.
fn dlasq4(i0: usize, n0: usize, z: &[f64], pp: usize, n0in: usize, st: &mut Dqds) {
    const CNST1: f64 = 0.5630;
    const CNST2: f64 = 1.010;
    const CNST3: f64 = 1.050;
    const QURTR: f64 = 0.250;
    const THIRD: f64 = 0.3330;
    const HUNDRD: f64 = 100.0;
    let (dmin, dmin1, dmin2, dn, dn1, dn2) = (st.dmin, st.dmin1, st.dmin2, st.dn, st.dn1, st.dn2);
    if dmin <= 0.0 {
        st.tau = -dmin;
        st.ttype = -1;
        return;
    }
    let nn = 4 * n0 + pp;
    let lo = 4 * i0 - 1 + pp; // the DO loops' lower bound
    let mut s = 0.0f64;
    // returns `Some(s)` for the final `TAU = S`, `None` for an early RETURN
    // (TAU unchanged)
    let r: Option<f64> = 'body: {
        if n0in == n0 {
            if dmin == dn || dmin == dn1 {
                let b1 = z[nn - 3].sqrt() * z[nn - 5].sqrt();
                let b2 = z[nn - 7].sqrt() * z[nn - 9].sqrt();
                let a2 = z[nn - 7] + z[nn - 5];
                if dmin == dn && dmin1 == dn1 {
                    let gap2 = dmin2 - a2 - dmin2 * QURTR;
                    let gap1 = if gap2 > 0.0 && gap2 > b2 {
                        a2 - dn - (b2 / gap2) * b2
                    } else {
                        a2 - dn - (b1 + b2)
                    };
                    if gap1 > 0.0 && gap1 > b1 {
                        s = fmax(dn - (b1 / gap1) * b1, 0.5 * dmin);
                        st.ttype = -2;
                    } else {
                        s = 0.0;
                        if dn > b1 {
                            s = dn - b1;
                        }
                        if a2 > (b1 + b2) {
                            s = fmin(s, a2 - (b1 + b2));
                        }
                        s = fmax(s, THIRD * dmin);
                        st.ttype = -3;
                    }
                } else {
                    st.ttype = -4;
                    s = QURTR * dmin;
                    let gam;
                    let mut a2;
                    let mut b2;
                    let np;
                    if dmin == dn {
                        gam = dn;
                        a2 = 0.0;
                        if z[nn - 5] > z[nn - 7] {
                            break 'body None;
                        }
                        b2 = z[nn - 5] / z[nn - 7];
                        np = nn - 9;
                    } else {
                        let npp = nn - 2 * pp;
                        gam = dn1;
                        if z[npp - 4] > z[npp - 2] {
                            break 'body None;
                        }
                        a2 = z[npp - 4] / z[npp - 2];
                        if z[nn - 9] > z[nn - 11] {
                            break 'body None;
                        }
                        b2 = z[nn - 9] / z[nn - 11];
                        np = nn - 13;
                    }
                    a2 += b2;
                    let mut i4 = np as isize;
                    while i4 >= lo as isize {
                        let i = i4 as usize;
                        if b2 == 0.0 {
                            break;
                        }
                        let b1 = b2;
                        if z[i] > z[i - 2] {
                            break 'body None;
                        }
                        b2 *= z[i] / z[i - 2];
                        a2 += b2;
                        if HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2 {
                            break;
                        }
                        i4 -= 4;
                    }
                    a2 *= CNST3;
                    if a2 < CNST1 {
                        s = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
                    }
                }
            } else if dmin == dn2 {
                st.ttype = -5;
                s = QURTR * dmin;
                let np = nn - 2 * pp;
                let mut b1 = z[np - 2];
                let mut b2 = z[np - 6];
                let gam = dn2;
                if z[np - 8] > b2 || z[np - 4] > b1 {
                    break 'body None;
                }
                let mut a2 = (z[np - 8] / b2) * (1.0 + z[np - 4] / b1);
                if n0 - i0 > 2 {
                    b2 = z[nn - 13] / z[nn - 15];
                    a2 += b2;
                    let mut i4 = nn as isize - 17;
                    while i4 >= lo as isize {
                        let i = i4 as usize;
                        if b2 == 0.0 {
                            break;
                        }
                        b1 = b2;
                        if z[i] > z[i - 2] {
                            break 'body None;
                        }
                        b2 *= z[i] / z[i - 2];
                        a2 += b2;
                        if HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2 {
                            break;
                        }
                        i4 -= 4;
                    }
                    a2 *= CNST3;
                }
                if a2 < CNST1 {
                    s = gam * (1.0 - a2.sqrt()) / (1.0 + a2);
                }
            } else {
                if st.ttype == -6 {
                    st.g += THIRD * (1.0 - st.g);
                } else if st.ttype == -18 {
                    st.g = QURTR * THIRD;
                } else {
                    st.g = QURTR;
                }
                s = st.g * dmin;
                st.ttype = -6;
            }
        } else if n0in == n0 + 1 {
            if dmin1 == dn1 && dmin2 == dn2 {
                st.ttype = -7;
                s = THIRD * dmin1;
                if z[nn - 5] > z[nn - 7] {
                    break 'body None;
                }
                let mut b1 = z[nn - 5] / z[nn - 7];
                let mut b2 = b1;
                if b2 != 0.0 {
                    let mut i4 = (4 * n0 + pp) as isize - 9;
                    while i4 >= lo as isize {
                        let i = i4 as usize;
                        let a2 = b1;
                        if z[i] > z[i - 2] {
                            break 'body None;
                        }
                        b1 *= z[i] / z[i - 2];
                        b2 += b1;
                        if HUNDRD * fmax(b1, a2) < b2 {
                            break;
                        }
                        i4 -= 4;
                    }
                }
                let b2 = (CNST3 * b2).sqrt();
                let a2 = dmin1 / (1.0 + b2 * b2);
                let gap2 = 0.5 * dmin2 - a2;
                if gap2 > 0.0 && gap2 > b2 * a2 {
                    s = fmax(s, a2 * (1.0 - CNST2 * a2 * (b2 / gap2) * b2));
                } else {
                    s = fmax(s, a2 * (1.0 - CNST2 * b2));
                    st.ttype = -8;
                }
            } else {
                s = QURTR * dmin1;
                if dmin1 == dn1 {
                    s = 0.5 * dmin1;
                }
                st.ttype = -9;
            }
        } else if n0in == n0 + 2 {
            if dmin2 == dn2 && 2.0 * z[nn - 5] < z[nn - 7] {
                st.ttype = -10;
                s = THIRD * dmin2;
                if z[nn - 5] > z[nn - 7] {
                    break 'body None;
                }
                let mut b1 = z[nn - 5] / z[nn - 7];
                let mut b2 = b1;
                if b2 != 0.0 {
                    let mut i4 = (4 * n0 + pp) as isize - 9;
                    while i4 >= lo as isize {
                        let i = i4 as usize;
                        if z[i] > z[i - 2] {
                            break 'body None;
                        }
                        b1 *= z[i] / z[i - 2];
                        b2 += b1;
                        if HUNDRD * b1 < b2 {
                            break;
                        }
                        i4 -= 4;
                    }
                }
                let b2 = (CNST3 * b2).sqrt();
                let a2 = dmin2 / (1.0 + b2 * b2);
                let gap2 = z[nn - 7] + z[nn - 9] - z[nn - 11].sqrt() * z[nn - 9].sqrt() - a2;
                if gap2 > 0.0 && gap2 > b2 * a2 {
                    s = fmax(s, a2 * (1.0 - CNST2 * a2 * (b2 / gap2) * b2));
                } else {
                    s = fmax(s, a2 * (1.0 - CNST2 * b2));
                }
            } else {
                s = QURTR * dmin2;
                st.ttype = -11;
            }
        } else if n0in > n0 + 2 {
            s = 0.0;
            st.ttype = -12;
        }
        Some(s)
    };
    if let Some(s) = r {
        st.tau = s;
    }
}

/// `DLASQ5`: one dqds transform with shift `tau`.
fn dlasq5(i0: usize, n0: usize, z: &mut [f64], pp: usize, st: &mut Dqds, ieee: bool, eps: f64) {
    if n0 as isize - i0 as isize - 1 <= 0 {
        return;
    }
    let sigma = st.sigma;
    let mut tau = st.tau;
    let dthresh = eps * (sigma + tau);
    if tau < dthresh * 0.5 {
        tau = 0.0;
    }
    st.tau = tau;
    let flush = tau == 0.0; // the `TAU.EQ.ZERO` branch flushes tiny d to zero
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4] - tau;
    let mut dmin = d;
    st.dmin1 = -z[j4];
    let last = 4 * (n0 - 3);
    // `ieee`: no sign checks; else return on a negative d
    macro_rules! early {
        () => {{
            st.dmin = dmin;
            return;
        }};
    }
    if pp == 0 {
        j4 = 4 * i0;
        while j4 <= last {
            z[j4 - 2] = d + z[j4 - 1];
            if ieee {
                let temp = z[j4 + 1] / z[j4 - 2];
                d = d * temp - tau;
                if flush && d < dthresh {
                    d = 0.0;
                }
                dmin = fmin(dmin, d);
                z[j4] = z[j4 - 1] * temp;
                emin = fmin(z[j4], emin);
            } else {
                if d < 0.0 {
                    early!();
                }
                z[j4] = z[j4 + 1] * (z[j4 - 1] / z[j4 - 2]);
                d = z[j4 + 1] * (d / z[j4 - 2]) - tau;
                if flush && d < dthresh {
                    d = 0.0;
                }
                dmin = fmin(dmin, d);
                emin = fmin(emin, z[j4]);
            }
            j4 += 4;
        }
    } else {
        j4 = 4 * i0;
        while j4 <= last {
            z[j4 - 3] = d + z[j4];
            if ieee {
                let temp = z[j4 + 2] / z[j4 - 3];
                d = d * temp - tau;
                if flush && d < dthresh {
                    d = 0.0;
                }
                dmin = fmin(dmin, d);
                z[j4 - 1] = z[j4] * temp;
                emin = fmin(z[j4 - 1], emin);
            } else {
                if d < 0.0 {
                    early!();
                }
                z[j4 - 1] = z[j4 + 2] * (z[j4] / z[j4 - 3]);
                d = z[j4 + 2] * (d / z[j4 - 3]) - tau;
                if flush && d < dthresh {
                    d = 0.0;
                }
                dmin = fmin(dmin, d);
                emin = fmin(emin, z[j4 - 1]);
            }
            j4 += 4;
        }
    }
    let dnm2 = d;
    st.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm2 + z[j4p2];
    if !ieee && dnm2 < 0.0 {
        early!();
    }
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dnm1 = z[j4p2 + 2] * (dnm2 / z[j4 - 2]) - tau;
    dmin = fmin(dmin, dnm1);
    st.dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm1 + z[j4p2];
    if !ieee && dnm1 < 0.0 {
        early!();
    }
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dn = z[j4p2 + 2] * (dnm1 / z[j4 - 2]) - tau;
    dmin = fmin(dmin, dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    st.dmin = dmin;
    st.dn = dn;
    st.dn1 = dnm1;
    st.dn2 = dnm2;
}

/// `DLASQ6`: one dqd transform (no shift), guarding against underflow.
fn dlasq6(i0: usize, n0: usize, z: &mut [f64], pp: usize, st: &mut Dqds) {
    if n0 as isize - i0 as isize - 1 <= 0 {
        return;
    }
    let safmin = SFMIN;
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4];
    let mut dmin = d;
    // one step: q at `a` (the new sum), e at `e_new`, from `e_old`, `q_next`
    let (o_s, o_e, o_q, o_w) = if pp == 0 {
        (2usize, 1usize, 1isize, 0usize)
    } else {
        (3, 0, 2, 1)
    };
    j4 = 4 * i0;
    while j4 <= 4 * (n0 - 3) {
        // pp=0: Z(J4-2)=D+Z(J4-1); Z(J4) from Z(J4+1)
        // pp=1: Z(J4-3)=D+Z(J4);   Z(J4-1) from Z(J4+2)
        let s = j4 - o_s;
        let e = j4 - o_e;
        let q = (j4 as isize + o_q) as usize;
        let w = j4 - o_w;
        z[s] = d + z[e];
        if z[s] == 0.0 {
            z[w] = 0.0;
            d = z[q];
            dmin = d;
            emin = 0.0;
        } else if safmin * z[q] < z[s] && safmin * z[s] < z[q] {
            let temp = z[q] / z[s];
            z[w] = z[e] * temp;
            d *= temp;
        } else {
            z[w] = z[q] * (z[e] / z[s]);
            d = z[q] * (d / z[s]);
        }
        dmin = fmin(dmin, d);
        emin = fmin(emin, z[w]);
        j4 += 4;
    }
    let dnm2 = d;
    st.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm2 + z[j4p2];
    let dnm1;
    if z[j4 - 2] == 0.0 {
        z[j4] = 0.0;
        dnm1 = z[j4p2 + 2];
        dmin = dnm1;
        emin = 0.0;
    } else if safmin * z[j4p2 + 2] < z[j4 - 2] && safmin * z[j4 - 2] < z[j4p2 + 2] {
        let temp = z[j4p2 + 2] / z[j4 - 2];
        z[j4] = z[j4p2] * temp;
        dnm1 = dnm2 * temp;
    } else {
        z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
        dnm1 = z[j4p2 + 2] * (dnm2 / z[j4 - 2]);
    }
    dmin = fmin(dmin, dnm1);
    st.dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm1 + z[j4p2];
    let dn;
    if z[j4 - 2] == 0.0 {
        z[j4] = 0.0;
        dn = z[j4p2 + 2];
        dmin = dn;
        emin = 0.0;
    } else if safmin * z[j4p2 + 2] < z[j4 - 2] && safmin * z[j4 - 2] < z[j4p2 + 2] {
        let temp = z[j4p2 + 2] / z[j4 - 2];
        z[j4] = z[j4p2] * temp;
        dn = dnm1 * temp;
    } else {
        z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
        dn = z[j4p2 + 2] * (dnm1 / z[j4 - 2]);
    }
    dmin = fmin(dmin, dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    st.dmin = dmin;
    st.dn = dn;
    st.dn1 = dnm1;
    st.dn2 = dnm2;
}
