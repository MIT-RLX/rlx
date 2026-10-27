// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The divide-and-conquer half of `DLALSD` (reference LAPACK 3.12, as OpenBLAS
//! 0.3.30 bundles it): `DLASDA` (bidiagonal SVD by divide and conquer, with the
//! factored representation), `DLALSA`/`DLALS0` (applying it), and the secular
//! equation solvers `DLASD4`/`DLASD5`/`DLAED6`, `DLASD6`/`DLASD7`/`DLASD8`. Translated
//! statement for statement (gfortran, no FMA: every operation rounds once, left to
//! right inside each parenthesis); BLAS through [`crate::openblas`].
//!
//! Index conventions follow the Fortran (1-based `i` ↦ `[i-1]`); a 2-D Fortran array
//! `X(LDX, *)` is a column-major slice with its leading dimension.

use crate::lapack::{EPS, SFMIN, dlamrg, dlapy2, dlascl, dlasdq, sign};
use crate::openblas as ob;

/// `DLAMC3(a, b)`: `a + b` (forced to memory).
fn dlamc3(a: f64, b: f64) -> f64 {
    a + b
}

/// `DLASDT(n, msub) → (lvl, nd, inode, ndiml, ndimr)` (1-based node indices).
pub(crate) fn dlasdt(n: usize, msub: usize) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<usize>) {
    let maxn = n.max(1);
    let temp = (maxn as f64 / (msub + 1) as f64).ln() / 2f64.ln();
    let lvl = (temp as i64 + 1) as usize;
    let mut inode = vec![0usize; n.max(1)];
    let mut ndiml = vec![0usize; n.max(1)];
    let mut ndimr = vec![0usize; n.max(1)];
    let i = n / 2;
    inode[0] = i + 1;
    ndiml[0] = i;
    ndimr[0] = n - i - 1;
    let (mut il, mut ir, mut llst) = (0usize, 1usize, 1usize);
    for _ in 1..lvl {
        for i in 0..llst {
            il += 2;
            ir += 2;
            let ncrnt = llst + i;
            ndiml[il - 1] = ndiml[ncrnt - 1] / 2;
            ndimr[il - 1] = ndiml[ncrnt - 1] - ndiml[il - 1] - 1;
            inode[il - 1] = inode[ncrnt - 1] - ndimr[il - 1] - 1;
            ndiml[ir - 1] = ndimr[ncrnt - 1] / 2;
            ndimr[ir - 1] = ndimr[ncrnt - 1] - ndiml[ir - 1] - 1;
            inode[ir - 1] = inode[ncrnt - 1] + ndiml[ir - 1] + 1;
        }
        llst *= 2;
    }
    let nd = llst * 2 - 1;
    (lvl, nd, inode, ndiml, ndimr)
}

/// `DLASD5(i, d, z, delta, rho) → (dsigma)`, writing `delta`, `work`.
fn dlasd5(i: usize, d: &[f64], z: &[f64], delta: &mut [f64], rho: f64, work: &mut [f64]) -> f64 {
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const FOUR: f64 = 4.0;
    let del = d[1] - d[0];
    let delsq = del * (d[1] + d[0]);
    let dsigma;
    if i == 1 {
        let w = 1.0
            + FOUR
                * rho
                * (z[1] * z[1] / (d[0] + THREE * d[1]) - z[0] * z[0] / (THREE * d[0] + d[1]))
                / del;
        if w > 0.0 {
            let b = delsq + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[0] * z[0] * delsq;
            let mut tau = TWO * c / (b + (b * b - FOUR * c).abs().sqrt());
            tau /= d[0] + (d[0] * d[0] + tau).sqrt();
            dsigma = d[0] + tau;
            delta[0] = -tau;
            delta[1] = del - tau;
            work[0] = TWO * d[0] + tau;
            work[1] = (d[0] + tau) + d[1];
        } else {
            let b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[1] * z[1] * delsq;
            let mut tau = if b > 0.0 {
                -(TWO * c / (b + (b * b + FOUR * c).sqrt()))
            } else {
                (b - (b * b + FOUR * c).sqrt()) / TWO
            };
            tau /= d[1] + (d[1] * d[1] + tau).abs().sqrt();
            dsigma = d[1] + tau;
            delta[0] = -(del + tau);
            delta[1] = -tau;
            work[0] = d[0] + tau + d[1];
            work[1] = TWO * d[1] + tau;
        }
    } else {
        let b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
        let c = rho * z[1] * z[1] * delsq;
        let mut tau = if b > 0.0 {
            (b + (b * b + FOUR * c).sqrt()) / TWO
        } else {
            TWO * c / (-b + (b * b + FOUR * c).sqrt())
        };
        tau /= d[1] + (d[1] * d[1] + tau).sqrt();
        dsigma = d[1] + tau;
        delta[0] = -(del + tau);
        delta[1] = -tau;
        work[0] = d[0] + tau + d[1];
        work[1] = TWO * d[1] + tau;
    }
    dsigma
}

/// `DLAED6(kniter, orgati, rho, d, z, finit) → (tau, info)`.
fn dlaed6(
    kniter: usize,
    orgati: bool,
    rho: f64,
    d: &[f64; 3],
    z: &[f64; 3],
    finit: f64,
) -> (f64, usize) {
    const MAXIT: usize = 40;
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const FOUR: f64 = 4.0;
    const EIGHT: f64 = 8.0;
    let mut info = 0;
    let (mut lbd, mut ubd) = if orgati { (d[1], d[2]) } else { (d[0], d[1]) };
    if finit < 0.0 {
        lbd = 0.0;
    } else {
        ubd = 0.0;
    }
    let mut niter = 1;
    let mut tau = 0.0f64;
    if kniter == 2 {
        let (mut a, mut b, mut c);
        if orgati {
            let temp = (d[2] - d[1]) / TWO;
            c = rho + z[0] / ((d[0] - d[1]) - temp);
            a = c * (d[1] + d[2]) + z[1] + z[2];
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
        } else {
            let temp = (d[0] - d[1]) / TWO;
            c = rho + z[2] / ((d[2] - d[1]) - temp);
            a = c * (d[0] + d[1]) + z[0] + z[1];
            b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
        }
        let temp = a.abs().max(b.abs()).max(c.abs());
        a /= temp;
        b /= temp;
        c /= temp;
        tau = if c == 0.0 {
            b / a
        } else if a <= 0.0 {
            (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
        } else {
            TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
        };
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / TWO;
        }
        if d[0] == tau || d[1] == tau || d[2] == tau {
            tau = 0.0;
        } else {
            let temp = finit
                + tau * z[0] / (d[0] * (d[0] - tau))
                + tau * z[1] / (d[1] * (d[1] - tau))
                + tau * z[2] / (d[2] * (d[2] - tau));
            if temp <= 0.0 {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if finit.abs() <= temp.abs() {
                tau = 0.0;
            }
        }
    }
    let eps = EPS;
    let base = 2.0f64;
    // BASE**( INT( LOG( SAFMIN ) / LOG( BASE ) / THREE ) )
    let e = (rlx_libm::log(SFMIN) / rlx_libm::log(base) / THREE) as i64;
    let small1 = 2f64.powi(e as i32);
    let sminv1 = 1.0 / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;
    let temp = if orgati {
        (d[1] - tau).abs().min((d[2] - tau).abs())
    } else {
        (d[0] - tau).abs().min((d[1] - tau).abs())
    };
    let mut scale = false;
    let (mut dscale, mut zscale) = (*d, *z);
    let mut sclinv = 1.0;
    if temp <= small1 {
        scale = true;
        let sclfac;
        if temp <= small2 {
            sclfac = sminv2;
            sclinv = small2;
        } else {
            sclfac = sminv1;
            sclinv = small1;
        }
        for i in 0..3 {
            dscale[i] = d[i] * sclfac;
            zscale[i] = z[i] * sclfac;
        }
        tau *= sclfac;
        lbd *= sclfac;
        ubd *= sclfac;
    }
    let (mut fc, mut df, mut ddf) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..3 {
        let temp = 1.0 / (dscale[i] - tau);
        let temp1 = zscale[i] * temp;
        let temp2 = temp1 * temp;
        let temp3 = temp2 * temp;
        fc += temp1 / dscale[i];
        df += temp2;
        ddf += temp3;
    }
    let mut f = finit + tau * fc;
    let mut converged = f.abs() <= 0.0;
    if !converged {
        if f <= 0.0 {
            lbd = tau;
        } else {
            ubd = tau;
        }
        let iter = niter + 1;
        niter = iter;
        let mut done = false;
        while niter <= MAXIT {
            let (temp1, temp2) = if orgati {
                (dscale[1] - tau, dscale[2] - tau)
            } else {
                (dscale[0] - tau, dscale[1] - tau)
            };
            let mut a = (temp1 + temp2) * f - temp1 * temp2 * df;
            let mut b = temp1 * temp2 * f;
            let mut c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
            let temp = a.abs().max(b.abs()).max(c.abs());
            a /= temp;
            b /= temp;
            c /= temp;
            let mut eta = if c == 0.0 {
                b / a
            } else if a <= 0.0 {
                (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
            } else {
                TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
            };
            if f * eta >= 0.0 {
                eta = -f / df;
            }
            tau += eta;
            if tau < lbd || tau > ubd {
                tau = (lbd + ubd) / TWO;
            }
            fc = 0.0;
            let mut erretm = 0.0f64;
            df = 0.0;
            ddf = 0.0;
            let mut hit_pole = false;
            for i in 0..3 {
                if (dscale[i] - tau) != 0.0 {
                    let temp = 1.0 / (dscale[i] - tau);
                    let temp1 = zscale[i] * temp;
                    let temp2 = temp1 * temp;
                    let temp3 = temp2 * temp;
                    let temp4 = temp1 / dscale[i];
                    fc += temp4;
                    erretm += temp4.abs();
                    df += temp2;
                    ddf += temp3;
                } else {
                    hit_pole = true;
                    break;
                }
            }
            if hit_pole {
                done = true;
                break;
            }
            f = finit + tau * fc;
            erretm = EIGHT * (finit.abs() + tau.abs() * erretm) + tau.abs() * df;
            if f.abs() <= FOUR * eps * erretm || (ubd - lbd) <= FOUR * eps * tau.abs() {
                done = true;
                break;
            }
            if f <= 0.0 {
                lbd = tau;
            } else {
                ubd = tau;
            }
            niter += 1;
        }
        if !done {
            info = 1;
        }
        converged = true;
    }
    let _ = converged;
    if scale {
        tau *= sclinv;
    }
    (tau, info)
}

/// `DLASD4(n, i, d, z, delta, rho) → (sigma, info)`, writing `delta` and `work`.
#[allow(clippy::too_many_lines)]
fn dlasd4(
    n: usize,
    i: usize,
    d: &[f64],
    z: &[f64],
    delta: &mut [f64],
    rho: f64,
    work: &mut [f64],
) -> (f64, usize) {
    const MAXIT: usize = 400;
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const FOUR: f64 = 4.0;
    const EIGHT: f64 = 8.0;
    const TEN: f64 = 10.0;
    let info;
    if n == 1 {
        delta[0] = 1.0;
        work[0] = 1.0;
        return ((d[0] * d[0] + rho * z[0] * z[0]).sqrt(), 0);
    }
    if n == 2 {
        return (dlasd5(i, d, z, delta, rho, work), 0);
    }
    let eps = EPS;
    let rhoinv = 1.0 / rho;
    let mut tau2;
    let mut sigma;
    // 1-based helpers
    let dd = |k: usize| d[k - 1];
    let zz = |k: usize| z[k - 1];
    if i == n {
        let ii = n - 1;
        let mut niter = 1;
        let temp = rho / TWO;
        let temp1 = temp / (dd(n) + (dd(n) * dd(n) + temp).sqrt());
        for j in 1..=n {
            work[j - 1] = dd(j) + dd(n) + temp1;
            delta[j - 1] = (dd(j) - dd(n)) - temp1;
        }
        let mut psi = 0.0f64;
        for j in 1..=n - 2 {
            psi += zz(j) * zz(j) / (delta[j - 1] * work[j - 1]);
        }
        let c = rhoinv + psi;
        let w = c
            + zz(ii) * zz(ii) / (delta[ii - 1] * work[ii - 1])
            + zz(n) * zz(n) / (delta[n - 1] * work[n - 1]);
        let tau;
        if w <= 0.0 {
            let temp1 = (dd(n) * dd(n) + rho).sqrt();
            let temp = zz(n - 1) * zz(n - 1)
                / ((dd(n - 1) + temp1) * (dd(n) - dd(n - 1) + rho / (dd(n) + temp1)))
                + zz(n) * zz(n) / rho;
            if c <= temp {
                tau = rho;
            } else {
                let delsq = (dd(n) - dd(n - 1)) * (dd(n) + dd(n - 1));
                let a = -c * delsq + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
                let b = zz(n) * zz(n) * delsq;
                tau2 = if a < 0.0 {
                    TWO * b / ((a * a + FOUR * b * c).sqrt() - a)
                } else {
                    (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c)
                };
                tau = tau2 / (dd(n) + (dd(n) * dd(n) + tau2).sqrt());
            }
        } else {
            let delsq = (dd(n) - dd(n - 1)) * (dd(n) + dd(n - 1));
            let a = -c * delsq + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
            let b = zz(n) * zz(n) * delsq;
            tau2 = if a < 0.0 {
                TWO * b / ((a * a + FOUR * b * c).sqrt() - a)
            } else {
                (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c)
            };
            tau = tau2 / (dd(n) + (dd(n) * dd(n) + tau2).sqrt());
        }
        let mut tau = tau;
        sigma = dd(n) + tau;
        for j in 1..=n {
            delta[j - 1] = (dd(j) - dd(n)) - tau;
            work[j - 1] = dd(j) + dd(n) + tau;
        }
        let (mut dpsi, mut psi, mut erretm) = (0.0f64, 0.0f64, 0.0f64);
        for j in 1..=ii {
            let temp = zz(j) / (delta[j - 1] * work[j - 1]);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let temp = zz(n) / (delta[n - 1] * work[n - 1]);
        let mut phi = zz(n) * temp;
        let mut dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;
        let mut w = rhoinv + phi + psi;
        if w.abs() <= eps * erretm {
            return (sigma, 0);
        }
        niter += 1;
        let dtnsq1 = work[n - 2] * delta[n - 2];
        let dtnsq = work[n - 1] * delta[n - 1];
        let mut c = w - dtnsq1 * dpsi - dtnsq * dphi;
        let a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
        let b = dtnsq * dtnsq1 * w;
        if c < 0.0 {
            c = c.abs();
        }
        let mut eta = if c == 0.0 {
            rho - sigma * sigma
        } else if a >= 0.0 {
            (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
        } else {
            TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt())
        };
        if w * eta > 0.0 {
            eta = -w / (dpsi + dphi);
        }
        let temp = eta - dtnsq;
        if temp > rho {
            eta = rho + dtnsq;
        }
        eta /= sigma + (eta + sigma * sigma).sqrt();
        tau += eta;
        sigma += eta;
        for j in 1..=n {
            delta[j - 1] -= eta;
            work[j - 1] += eta;
        }
        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=ii {
            let temp = zz(j) / (work[j - 1] * delta[j - 1]);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        tau2 = work[n - 1] * delta[n - 1];
        let temp = zz(n) / tau2;
        phi = zz(n) * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;
        w = rhoinv + phi + psi;
        let iter = niter + 1;
        for _niter in iter..=MAXIT {
            if w.abs() <= eps * erretm {
                return (sigma, 0);
            }
            let dtnsq1 = work[n - 2] * delta[n - 2];
            let dtnsq = work[n - 1] * delta[n - 1];
            let c = w - dtnsq1 * dpsi - dtnsq * dphi;
            let a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
            let b = dtnsq1 * dtnsq * w;
            let mut eta = if a >= 0.0 {
                (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
            } else {
                TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt())
            };
            if w * eta > 0.0 {
                eta = -w / (dpsi + dphi);
            }
            let temp = eta - dtnsq;
            if temp <= 0.0 {
                eta /= TWO;
            }
            eta /= sigma + (eta + sigma * sigma).sqrt();
            tau += eta;
            sigma += eta;
            for j in 1..=n {
                delta[j - 1] -= eta;
                work[j - 1] += eta;
            }
            dpsi = 0.0;
            psi = 0.0;
            erretm = 0.0;
            for j in 1..=ii {
                let temp = zz(j) / (work[j - 1] * delta[j - 1]);
                psi += zz(j) * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = erretm.abs();
            tau2 = work[n - 1] * delta[n - 1];
            let temp = zz(n) / tau2;
            phi = zz(n) * temp;
            dphi = temp * temp;
            erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;
            w = rhoinv + phi + psi;
        }
        let _ = tau;
        info = 1;
        return (sigma, info);
    }
    // 0 < i < n
    let ip1 = i + 1;
    let delsq = (dd(ip1) - dd(i)) * (dd(ip1) + dd(i));
    let delsq2 = delsq / TWO;
    let sq2 = ((dd(i) * dd(i) + dd(ip1) * dd(ip1)) / TWO).sqrt();
    let temp = delsq2 / (dd(i) + sq2);
    for j in 1..=n {
        work[j - 1] = dd(j) + dd(i) + temp;
        delta[j - 1] = (dd(j) - dd(i)) - temp;
    }
    let mut psi = 0.0f64;
    for j in 1..i {
        psi += zz(j) * zz(j) / (work[j - 1] * delta[j - 1]);
    }
    let mut phi = 0.0f64;
    let mut j = n;
    while j >= i + 2 {
        phi += zz(j) * zz(j) / (work[j - 1] * delta[j - 1]);
        j -= 1;
    }
    let c = rhoinv + psi + phi;
    let w = c
        + zz(i) * zz(i) / (work[i - 1] * delta[i - 1])
        + zz(ip1) * zz(ip1) / (work[ip1 - 1] * delta[ip1 - 1]);
    let mut geomavg = false;
    let orgati;
    let ii;
    let (mut sglb, mut sgub);
    let mut tau;
    if w > 0.0 {
        orgati = true;
        ii = i;
        sglb = 0.0;
        sgub = delsq2 / (dd(i) + sq2);
        let a = c * delsq + zz(i) * zz(i) + zz(ip1) * zz(ip1);
        let b = zz(i) * zz(i) * delsq;
        tau2 = if a > 0.0 {
            TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
        } else {
            (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
        };
        tau = tau2 / (dd(i) + (dd(i) * dd(i) + tau2).sqrt());
        let temp = eps.sqrt();
        if dd(i) <= temp * dd(ip1) && zz(i).abs() <= temp && dd(i) > 0.0 {
            tau = (TEN * dd(i)).min(sgub);
            geomavg = true;
        }
    } else {
        orgati = false;
        ii = ip1;
        sglb = -delsq2 / (dd(ii) + sq2);
        sgub = 0.0;
        let a = c * delsq - zz(i) * zz(i) - zz(ip1) * zz(ip1);
        let b = zz(ip1) * zz(ip1) * delsq;
        tau2 = if a < 0.0 {
            TWO * b / (a - (a * a + FOUR * b * c).abs().sqrt())
        } else {
            -(a + (a * a + FOUR * b * c).abs().sqrt()) / (TWO * c)
        };
        tau = tau2 / (dd(ip1) + (dd(ip1) * dd(ip1) + tau2).abs().sqrt());
    }
    sigma = dd(ii) + tau;
    for j in 1..=n {
        work[j - 1] = dd(j) + dd(ii) + tau;
        delta[j - 1] = (dd(j) - dd(ii)) - tau;
    }
    let iim1 = ii - 1;
    let iip1 = ii + 1;
    // ψ, φ, their derivatives and the error bound at the current τ (loops 150/160)
    let sums = |work: &[f64], delta: &[f64]| -> (f64, f64, f64, f64, f64) {
        let (mut dpsi, mut psi, mut erretm) = (0.0f64, 0.0f64, 0.0f64);
        for j in 1..=iim1 {
            let temp = zz(j) / (work[j - 1] * delta[j - 1]);
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let (mut dphi, mut phi) = (0.0f64, 0.0f64);
        let mut j = n;
        while j >= iip1 {
            let temp = zz(j) / (work[j - 1] * delta[j - 1]);
            phi += zz(j) * temp;
            dphi += temp * temp;
            erretm += phi;
            j -= 1;
        }
        (dpsi, psi, dphi, phi, erretm)
    };
    let (mut dpsi, mut psi, mut dphi, phi1, mut erretm) = sums(work, delta);
    phi = phi1;
    let mut w = rhoinv + phi + psi;
    let mut swtch3 = if orgati { w < 0.0 } else { w > 0.0 };
    if ii == 1 || ii == n {
        swtch3 = false;
    }
    let temp = zz(ii) / (work[ii - 1] * delta[ii - 1]);
    let mut dw = dpsi + dphi + temp * temp;
    let temp = zz(ii) * temp;
    w += temp;
    erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs();
    if w.abs() <= eps * erretm {
        return (sigma, 0);
    }
    if w <= 0.0 {
        sglb = sglb.max(tau);
    } else {
        sgub = sgub.min(tau);
    }
    // The first step runs with NITER = 2 and SWTCH = .FALSE.; loop 230 then runs
    // NITER = 3 … MAXIT with the convergence test and bound update at its top.
    let mut swtch = false;
    let mut niter = 2usize;
    let mut first = true;
    loop {
        if !first {
            if w.abs() <= eps * erretm {
                return (sigma, 0);
            }
            if w <= 0.0 {
                sglb = sglb.max(tau);
            } else {
                sgub = sgub.min(tau);
            }
        }
        let mut eta;
        if !swtch3 {
            eta = eta_rational(
                work, delta, z, w, dw, &mut dpsi, &mut dphi, swtch, orgati, i, ip1, ii, delsq,
            );
        } else {
            match eta_three_pole(
                work, delta, d, z, w, rhoinv, psi, phi, dpsi, dphi, swtch, orgati, ii, iim1, iip1,
                niter,
            ) {
                Some(e) => eta = e,
                None => {
                    swtch3 = false;
                    eta = eta_rational(
                        work, delta, z, w, dw, &mut dpsi, &mut dphi, swtch, orgati, i, ip1, ii,
                        delsq,
                    );
                }
            }
        }
        if w * eta >= 0.0 {
            eta = -w / dw;
        }
        eta /= sigma + (sigma * sigma + eta).sqrt();
        let temp = tau + eta;
        if temp > sgub || temp < sglb {
            eta = if w < 0.0 {
                (sgub - tau) / TWO
            } else {
                (sglb - tau) / TWO
            };
            if geomavg {
                if w < 0.0 {
                    if tau > 0.0 {
                        eta = (sgub * tau).sqrt() - tau;
                    }
                } else if sglb > 0.0 {
                    eta = (sglb * tau).sqrt() - tau;
                }
            }
        }
        let prew = w;
        tau += eta;
        sigma += eta;
        for j in 1..=n {
            work[j - 1] += eta;
            delta[j - 1] -= eta;
        }
        let s = sums(work, delta);
        dpsi = s.0;
        psi = s.1;
        dphi = s.2;
        phi = s.3;
        erretm = s.4;
        tau2 = work[ii - 1] * delta[ii - 1];
        let temp = zz(ii) / tau2;
        dw = dpsi + dphi + temp * temp;
        let temp = zz(ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs();
        if first {
            swtch = if orgati {
                -w > prew.abs() / TEN
            } else {
                w > prew.abs() / TEN
            };
            first = false;
        } else if w * prew > 0.0 && w.abs() > prew.abs() / TEN {
            swtch = !swtch;
        }
        niter += 1;
        if niter > MAXIT {
            break;
        }
    }
    info = 1;
    (sigma, info)
}

/// `DLASD4`'s rational step (the `.NOT.SWTCH3` branch); with `SWTCH` it first adds
/// the pole term to DPSI or DPHI, as the Fortran does in place.
#[allow(clippy::too_many_arguments)]
fn eta_rational(
    work: &[f64],
    delta: &[f64],
    z: &[f64],
    w: f64,
    dw: f64,
    dpsi: &mut f64,
    dphi: &mut f64,
    swtch: bool,
    orgati: bool,
    i: usize,
    ip1: usize,
    ii: usize,
    delsq: f64,
) -> f64 {
    const TWO: f64 = 2.0;
    const FOUR: f64 = 4.0;
    let zz = |k: usize| z[k - 1];
    let dtipsq = work[ip1 - 1] * delta[ip1 - 1];
    let dtisq = work[i - 1] * delta[i - 1];
    let c = if !swtch {
        if orgati {
            let q = zz(i) / dtisq;
            w - dtipsq * dw + delsq * (q * q)
        } else {
            let q = zz(ip1) / dtipsq;
            w - dtisq * dw - delsq * (q * q)
        }
    } else {
        let temp = zz(ii) / (work[ii - 1] * delta[ii - 1]);
        if orgati {
            *dpsi += temp * temp;
        } else {
            *dphi += temp * temp;
        }
        w - dtisq * *dpsi - dtipsq * *dphi
    };
    let mut a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
    let b = dtipsq * dtisq * w;
    if c == 0.0 {
        if a == 0.0 {
            a = if !swtch {
                if orgati {
                    zz(i) * zz(i) + dtipsq * dtipsq * (*dpsi + *dphi)
                } else {
                    zz(ip1) * zz(ip1) + dtisq * dtisq * (*dpsi + *dphi)
                }
            } else {
                dtisq * dtisq * *dpsi + dtipsq * dtipsq * *dphi
            };
        }
        b / a
    } else if a <= 0.0 {
        (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
    } else {
        TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
    }
}

/// `DLASD4`'s three-pole step (`SWTCH3`, via `DLAED6`); `None` when `DLAED6` fails.
#[allow(clippy::too_many_arguments)]
fn eta_three_pole(
    work: &[f64],
    delta: &[f64],
    d: &[f64],
    z: &[f64],
    w: f64,
    rhoinv: f64,
    psi: f64,
    phi: f64,
    dpsi: f64,
    dphi: f64,
    swtch: bool,
    orgati: bool,
    ii: usize,
    iim1: usize,
    iip1: usize,
    niter: usize,
) -> Option<f64> {
    let (dd, zz) = (|k: usize| d[k - 1], |k: usize| z[k - 1]);
    let dtiim = work[iim1 - 1] * delta[iim1 - 1];
    let dtiip = work[iip1 - 1] * delta[iip1 - 1];
    let temp = rhoinv + psi + phi;
    let mut zzv = [0.0f64; 3];
    let c;
    if swtch {
        c = temp - dtiim * dpsi - dtiip * dphi;
        zzv[0] = dtiim * dtiim * dpsi;
        zzv[2] = dtiip * dtiip * dphi;
    } else if orgati {
        let mut temp1 = zz(iim1) / dtiim;
        temp1 *= temp1;
        let temp2 = (dd(iim1) - dd(iip1)) * (dd(iim1) + dd(iip1)) * temp1;
        c = temp - dtiip * (dpsi + dphi) - temp2;
        zzv[0] = zz(iim1) * zz(iim1);
        zzv[2] = if dpsi < temp1 {
            dtiip * dtiip * dphi
        } else {
            dtiip * dtiip * ((dpsi - temp1) + dphi)
        };
    } else {
        let mut temp1 = zz(iip1) / dtiip;
        temp1 *= temp1;
        let temp2 = (dd(iip1) - dd(iim1)) * (dd(iim1) + dd(iip1)) * temp1;
        c = temp - dtiim * (dpsi + dphi) - temp2;
        zzv[0] = if dphi < temp1 {
            dtiim * dtiim * dpsi
        } else {
            dtiim * dtiim * (dpsi + (dphi - temp1))
        };
        zzv[2] = zz(iip1) * zz(iip1);
    }
    zzv[1] = zz(ii) * zz(ii);
    let ddv = [dtiim, delta[ii - 1] * work[ii - 1], dtiip];
    let (eta, info) = dlaed6(niter, orgati, c, &ddv, &zzv, w);
    if info != 0 { None } else { Some(eta) }
}

/// `DLASD8(icompq=1, k, d, z, vf, vl, difl, difr(ldgnum,2), dsigma) → info`.
#[allow(clippy::too_many_arguments)]
fn dlasd8(
    k: usize,
    d: &mut [f64],
    z: &mut [f64],
    vf: &mut [f64],
    vl: &mut [f64],
    difl: &mut [f64],
    difr: &mut [f64],
    ldgnum: usize,
    dsigma: &[f64],
) -> usize {
    if k == 1 {
        d[0] = z[0].abs();
        difl[0] = d[0];
        difl[1] = 1.0;
        difr[ldgnum] = 1.0;
        return 0;
    }
    // WORK: IWK1 = 1 (k), IWK2 = k+1 (k), IWK3 = 2k+1 (k)
    let mut work = vec![0.0f64; 3 * k];
    let (iwk2i, iwk3i) = (k, 2 * k); // 0-based starts
    let rho0 = ob::dnrm2(k, z, 1);
    dlascl(rho0, 1.0, k, 1, z, k);
    let rho = rho0 * rho0;
    for v in &mut work[iwk3i..iwk3i + k] {
        *v = 1.0;
    }
    for j in 1..=k {
        let (sigma, info) = {
            let (w1, rest) = work.split_at_mut(iwk2i);
            dlasd4(k, j, dsigma, z, w1, rho, &mut rest[..k])
        };
        d[j - 1] = sigma;
        if info != 0 {
            return info;
        }
        work[iwk3i + j - 1] = work[iwk3i + j - 1] * work[j - 1] * work[iwk2i + j - 1];
        difl[j - 1] = -work[j - 1];
        difr[j - 1] = -work[j]; // WORK(J+1): for J = K this is WORK(IWK2)
        for i in 1..j {
            work[iwk3i + i - 1] = work[iwk3i + i - 1] * work[i - 1] * work[iwk2i + i - 1]
                / (dsigma[i - 1] - dsigma[j - 1])
                / (dsigma[i - 1] + dsigma[j - 1]);
        }
        for i in j + 1..=k {
            work[iwk3i + i - 1] = work[iwk3i + i - 1] * work[i - 1] * work[iwk2i + i - 1]
                / (dsigma[i - 1] - dsigma[j - 1])
                / (dsigma[i - 1] + dsigma[j - 1]);
        }
    }
    for i in 1..=k {
        z[i - 1] = sign(work[iwk3i + i - 1].abs().sqrt(), z[i - 1]);
    }
    for j in 1..=k {
        let diflj = difl[j - 1];
        let dj = d[j - 1];
        let dsigj = -dsigma[j - 1];
        let (mut difrj, mut dsigjp) = (0.0, 0.0);
        if j < k {
            difrj = -difr[j - 1];
            dsigjp = -dsigma[j];
        }
        work[j - 1] = -z[j - 1] / diflj / (dsigma[j - 1] + dj);
        for i in 1..j {
            work[i - 1] = z[i - 1] / (dlamc3(dsigma[i - 1], dsigj) - diflj) / (dsigma[i - 1] + dj);
        }
        for i in j + 1..=k {
            work[i - 1] = z[i - 1] / (dlamc3(dsigma[i - 1], dsigjp) + difrj) / (dsigma[i - 1] + dj);
        }
        let temp = ob::dnrm2(k, &work, 1);
        work[iwk2i + j - 1] = ob::ddot(k, &work[..k], 1, vf, 1) / temp;
        work[iwk3i + j - 1] = ob::ddot(k, &work[..k], 1, vl, 1) / temp;
        difr[j - 1 + ldgnum] = temp;
    }
    vf[..k].copy_from_slice(&work[iwk2i..iwk2i + k]);
    vl[..k].copy_from_slice(&work[iwk3i..iwk3i + k]);
    0
}

/// Outputs of [`dlasd7`] beyond the arrays it updates in place.
struct Deflation {
    k: usize,
    givptr: usize,
    c: f64,
    s: f64,
}

/// `DLASD7(icompq=1, nl, nr, sqre, ...)`.
#[allow(clippy::too_many_arguments)]
fn dlasd7(
    nl: usize,
    nr: usize,
    sqre: usize,
    d: &mut [f64],
    z: &mut [f64],
    zw: &mut [f64],
    vf: &mut [f64],
    vfw: &mut [f64],
    vl: &mut [f64],
    vlw: &mut [f64],
    alpha: f64,
    beta: f64,
    dsigma: &mut [f64],
    idx: &mut [usize],
    idxp: &mut [usize],
    idxq: &mut [usize],
    perm: &mut [usize],
    givcol: &mut [usize],
    ldgcol: usize,
    givnum: &mut [f64],
    ldgnum: usize,
) -> Deflation {
    let n = nl + nr + 1;
    let m = n + sqre;
    let nlp1 = nl + 1;
    let nlp2 = nl + 2;
    let mut givptr = 0usize;
    let z1 = alpha * vl[nlp1 - 1];
    vl[nlp1 - 1] = 0.0;
    let tau = vf[nlp1 - 1];
    for i in (1..=nl).rev() {
        z[i] = alpha * vl[i - 1];
        vl[i - 1] = 0.0;
        vf[i] = vf[i - 1];
        d[i] = d[i - 1];
        idxq[i] = idxq[i - 1] + 1;
    }
    vf[0] = tau;
    for i in nlp2..=m {
        z[i - 1] = beta * vf[i - 1];
        vf[i - 1] = 0.0;
    }
    for i in nlp2..=n {
        idxq[i - 1] += nlp1;
    }
    for i in 2..=n {
        let q = idxq[i - 1];
        dsigma[i - 1] = d[q - 1];
        zw[i - 1] = z[q - 1];
        vfw[i - 1] = vf[q - 1];
        vlw[i - 1] = vl[q - 1];
    }
    {
        let mut ix = vec![0usize; n - 1];
        dlamrg(nl, nr, &dsigma[1..n], 1, 1, &mut ix);
        idx[1..n].copy_from_slice(&ix);
    }
    for i in 2..=n {
        let idxi = 1 + idx[i - 1];
        d[i - 1] = dsigma[idxi - 1];
        z[i - 1] = zw[idxi - 1];
        vf[i - 1] = vfw[idxi - 1];
        vl[i - 1] = vlw[idxi - 1];
    }
    let eps = EPS;
    let tol0 = alpha.abs().max(beta.abs());
    let tol = 8.0 * 8.0 * eps * d[n - 1].abs().max(tol0);
    let mut k = 1usize;
    let mut k2 = n + 1;
    let mut jprev = 0usize;
    let mut all_deflated = false;
    let mut found = false;
    for j in 2..=n {
        if z[j - 1].abs() <= tol {
            k2 -= 1;
            idxp[k2 - 1] = j;
            if j == n {
                all_deflated = true;
                break;
            }
        } else {
            jprev = j;
            found = true;
            break;
        }
    }
    let (mut cc, mut ss) = (0.0, 0.0);
    if !all_deflated {
        debug_assert!(found);
        let mut j = jprev;
        loop {
            j += 1;
            if j > n {
                break;
            }
            if z[j - 1].abs() <= tol {
                k2 -= 1;
                idxp[k2 - 1] = j;
            } else if (d[j - 1] - d[jprev - 1]).abs() <= tol {
                let s0 = z[jprev - 1];
                let c0 = z[j - 1];
                let tau = dlapy2(c0, s0);
                z[j - 1] = tau;
                z[jprev - 1] = 0.0;
                let c = c0 / tau;
                let s = -s0 / tau;
                givptr += 1;
                let mut idxjp = idxq[idx[jprev - 1]];
                let mut idxj = idxq[idx[j - 1]];
                if idxjp <= nlp1 {
                    idxjp -= 1;
                }
                if idxj <= nlp1 {
                    idxj -= 1;
                }
                givcol[givptr - 1 + ldgcol] = idxjp;
                givcol[givptr - 1] = idxj;
                givnum[givptr - 1 + ldgnum] = c;
                givnum[givptr - 1] = s;
                rot1(vf, jprev - 1, j - 1, c, s);
                rot1(vl, jprev - 1, j - 1, c, s);
                k2 -= 1;
                idxp[k2 - 1] = jprev;
                jprev = j;
            } else {
                k += 1;
                zw[k - 1] = z[jprev - 1];
                dsigma[k - 1] = d[jprev - 1];
                idxp[k - 1] = jprev;
                jprev = j;
            }
        }
        // 90
        k += 1;
        zw[k - 1] = z[jprev - 1];
        dsigma[k - 1] = d[jprev - 1];
        idxp[k - 1] = jprev;
    }
    // 100
    for j in 2..=n {
        let jp = idxp[j - 1];
        dsigma[j - 1] = d[jp - 1];
        vfw[j - 1] = vf[jp - 1];
        vlw[j - 1] = vl[jp - 1];
    }
    for j in 2..=n {
        let jp = idxp[j - 1];
        perm[j - 1] = idxq[idx[jp - 1]];
        if perm[j - 1] <= nlp1 {
            perm[j - 1] -= 1;
        }
    }
    for t in 0..n - k {
        d[k + t] = dsigma[k + t];
    }
    dsigma[0] = 0.0;
    let hlftol = tol / 2.0;
    if dsigma[1].abs() <= hlftol {
        dsigma[1] = hlftol;
    }
    if m > n {
        z[0] = dlapy2(z1, z[m - 1]);
        if z[0] <= tol {
            cc = 1.0;
            ss = 0.0;
            z[0] = tol;
        } else {
            cc = z1 / z[0];
            ss = -z[m - 1] / z[0];
        }
        rot1(vf, m - 1, 0, cc, ss);
        rot1(vl, m - 1, 0, cc, ss);
    } else if z1.abs() <= tol {
        z[0] = tol;
    } else {
        z[0] = z1;
    }
    for t in 1..k {
        z[t] = zw[t];
    }
    for t in 1..n {
        vf[t] = vfw[t];
        vl[t] = vlw[t];
    }
    Deflation {
        k,
        givptr,
        c: cc,
        s: ss,
    }
}

/// `DROT(1, X(a), 1, X(b), 1, c, s)` (OpenBLAS Haswell, fused).
fn rot1(x: &mut [f64], a: usize, b: usize, c: f64, s: f64) {
    let (xa, xb) = (x[a], x[b]);
    x[a] = c.mul_add(xa, s * xb);
    x[b] = c.mul_add(xb, -(s * xa));
}

/// The factored-form outputs of `DLASDA` for one subproblem (all `ld = ldu` arrays
/// start at the subproblem's first row, as the Fortran passes `X(NLF, …)`).
pub(crate) struct Factors<'a> {
    pub u: &'a mut [f64],
    pub vt: &'a mut [f64],
    pub ldu: usize,
    pub k: &'a mut [usize],
    pub difl: &'a mut [f64],
    pub difr: &'a mut [f64],
    pub z: &'a mut [f64],
    pub poles: &'a mut [f64],
    pub givptr: &'a mut [usize],
    pub givcol: &'a mut [usize],
    pub ldgcol: usize,
    pub perm: &'a mut [usize],
    pub givnum: &'a mut [f64],
    pub c: &'a mut [f64],
    pub s: &'a mut [f64],
}

/// `DLASD6(icompq=1, nl, nr, sqre, d, vf, vl, alpha, beta, idxq, …)` for one merge.
/// Each `*_off` is the subproblem's row offset into the shared factor arrays.
#[allow(clippy::too_many_arguments)]
fn dlasd6(
    nl: usize,
    nr: usize,
    sqre: usize,
    d: &mut [f64],
    vf: &mut [f64],
    vl: &mut [f64],
    alpha: &mut f64,
    beta: &mut f64,
    idxq: &mut [usize],
    f: &mut Factors,
    nlf: usize,
    lvl: usize,
    lvl2: usize,
    j: usize,
) -> usize {
    let n = nl + nr + 1;
    let m = n + sqre;
    let (ldu, ldgcol) = (f.ldu, f.ldgcol);
    let mut orgnrm = alpha.abs().max(beta.abs());
    d[nl] = 0.0;
    for v in d.iter().take(n) {
        if v.abs() > orgnrm {
            orgnrm = v.abs();
        }
    }
    dlascl(orgnrm, 1.0, n, 1, d, n);
    *alpha /= orgnrm;
    *beta /= orgnrm;
    let mut zv = vec![0.0f64; m];
    let mut dsigma = vec![0.0f64; n];
    let mut zw = vec![0.0f64; m];
    let mut vfw = vec![0.0f64; m];
    let mut vlw = vec![0.0f64; m];
    let mut idx = vec![0usize; n];
    let mut idxp = vec![0usize; n];
    let p = nlf - 1;
    let perm = &mut f.perm[p + (lvl - 1) * ldgcol..];
    let givcol = &mut f.givcol[p + (lvl2 - 1) * ldgcol..];
    let givnum = &mut f.givnum[p + (lvl2 - 1) * ldu..];
    let defl = dlasd7(
        nl,
        nr,
        sqre,
        d,
        &mut zv,
        &mut zw,
        vf,
        &mut vfw,
        vl,
        &mut vlw,
        *alpha,
        *beta,
        &mut dsigma,
        &mut idx,
        &mut idxp,
        idxq,
        perm,
        givcol,
        ldgcol,
        givnum,
        ldu,
    );
    let k = defl.k;
    f.k[j - 1] = k;
    f.givptr[j - 1] = defl.givptr;
    f.c[j - 1] = defl.c;
    f.s[j - 1] = defl.s;
    let info = {
        let difl = &mut f.difl[p + (lvl - 1) * ldu..];
        let difr = &mut f.difr[p + (lvl2 - 1) * ldu..];
        dlasd8(k, d, &mut zv, vf, vl, difl, difr, ldu, &dsigma)
    };
    // Z(NLF, LVL) receives the (updated) z
    for t in 0..k {
        f.z[p + t + (lvl - 1) * ldu] = zv[t];
    }
    if info != 0 {
        return info;
    }
    for t in 0..k {
        f.poles[p + t + (lvl2 - 1) * ldu] = d[t];
        f.poles[p + t + lvl2 * ldu] = dsigma[t];
    }
    dlascl(1.0, orgnrm, n, 1, d, n);
    let n1 = k;
    let n2 = n - k;
    let mut ix = vec![0usize; n];
    dlamrg(n1, n2, d, 1, -1, &mut ix);
    idxq[..n].copy_from_slice(&ix);
    0
}

/// `DLASDA(icompq=1, smlsiz, n, sqre=0, d, e, …)`: returns `info`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dlasda(
    smlsiz: usize,
    n: usize,
    sqre: usize,
    d: &mut [f64],
    e: &mut [f64],
    f: &mut Factors,
) -> usize {
    let m = n + sqre;
    let ldu = f.ldu;
    if n <= smlsiz {
        return dlasdq(sqre, n, m, n, 0, d, e, f.vt, ldu, f.u, ldu, &mut [], 1);
    }
    let (nlvl, nd, inode, ndiml, ndimr) = dlasdt(n, smlsiz);
    let mut idxq = vec![0usize; n];
    let mut vfw = vec![0.0f64; m]; // VF
    let mut vlw = vec![0.0f64; m]; // VL
    let ndb1 = (nd + 1) / 2;
    for i in ndb1..=nd {
        let i1 = i - 1;
        let ic = inode[i1];
        let nl = ndiml[i1];
        let nlp1 = nl + 1;
        let nr = ndimr[i1];
        let nlf = ic - nl;
        let nrf = ic + 1;
        let mut idxqi = nlf - 2; // IDXQ + NLF - 2 (1-based offset into IDXQ) → 0-based base
        let mut vfi = nlf - 1;
        let mut vli = nlf - 1;
        let mut sqrei = 1;
        // U(NLF,1) = I(nl), VT(NLF,1) = I(nl+1)
        for jj in 0..nl {
            for ii in 0..nl {
                f.u[(nlf - 1 + ii) + jj * ldu] = if ii == jj { 1.0 } else { 0.0 };
            }
        }
        for jj in 0..nlp1 {
            for ii in 0..nlp1 {
                f.vt[(nlf - 1 + ii) + jj * ldu] = if ii == jj { 1.0 } else { 0.0 };
            }
        }
        let info = {
            let (vt, u) = (&mut f.vt[nlf - 1..], &mut f.u[nlf - 1..]);
            dlasdq(
                sqrei,
                nl,
                nlp1,
                nl,
                0,
                &mut d[nlf - 1..],
                &mut e[nlf - 1..],
                vt,
                ldu,
                u,
                ldu,
                &mut [],
                1,
            )
        };
        for t in 0..nlp1 {
            vfw[vfi + t] = f.vt[nlf - 1 + t];
            vlw[vli + t] = f.vt[(nlf - 1 + t) + (nlp1 - 1) * ldu];
        }
        if info != 0 {
            return info;
        }
        for jj in 1..=nl {
            idxq[idxqi + jj] = jj;
        }
        sqrei = if i == nd && sqre == 0 { 0 } else { 1 };
        idxqi += nlp1;
        vfi += nlp1;
        vli += nlp1;
        let nrp1 = nr + sqrei;
        for jj in 0..nr {
            for ii in 0..nr {
                f.u[(nrf - 1 + ii) + jj * ldu] = if ii == jj { 1.0 } else { 0.0 };
            }
        }
        for jj in 0..nrp1 {
            for ii in 0..nrp1 {
                f.vt[(nrf - 1 + ii) + jj * ldu] = if ii == jj { 1.0 } else { 0.0 };
            }
        }
        let info = {
            let (vt, u) = (&mut f.vt[nrf - 1..], &mut f.u[nrf - 1..]);
            dlasdq(
                sqrei,
                nr,
                nrp1,
                nr,
                0,
                &mut d[nrf - 1..],
                &mut e[nrf - 1..],
                vt,
                ldu,
                u,
                ldu,
                &mut [],
                1,
            )
        };
        for t in 0..nrp1 {
            vfw[vfi + t] = f.vt[nrf - 1 + t];
            vlw[vli + t] = f.vt[(nrf - 1 + t) + (nrp1 - 1) * ldu];
        }
        if info != 0 {
            return info;
        }
        for jj in 1..=nr {
            idxq[idxqi + jj] = jj;
        }
    }
    // conquer
    let mut j = 1usize << nlvl;
    for lvl in (1..=nlvl).rev() {
        let lvl2 = lvl * 2 - 1;
        let (lf, ll) = if lvl == 1 {
            (1, 1)
        } else {
            (1usize << (lvl - 1), (1usize << lvl) - 1)
        };
        for i in lf..=ll {
            let im1 = i - 1;
            let ic = inode[im1];
            let nl = ndiml[im1];
            let nr = ndimr[im1];
            let nlf = ic - nl;
            let sqrei = if i == ll { sqre } else { 1 };
            let vfi = nlf - 1;
            let vli = nlf - 1;
            let idxqi = nlf - 1;
            let mut alpha = d[ic - 1];
            let mut beta = e[ic - 1];
            j -= 1;
            let nn = nl + nr + 1;
            let info = dlasd6(
                nl,
                nr,
                sqrei,
                &mut d[nlf - 1..nlf - 1 + nn],
                &mut vfw[vfi..],
                &mut vlw[vli..],
                &mut alpha,
                &mut beta,
                &mut idxq[idxqi..],
                f,
                nlf,
                lvl,
                lvl2,
                j,
            );
            if info != 0 {
                return info;
            }
        }
    }
    0
}

/// `DLALS0(icompq, nl, nr, sqre, nrhs, B, BX, …)` for one node.
#[allow(clippy::too_many_arguments)]
fn dlals0(
    icompq: usize,
    nl: usize,
    nr: usize,
    sqre: usize,
    nrhs: usize,
    b: &mut [f64],
    ldb: usize,
    bx: &mut [f64],
    ldbx: usize,
    perm: &[usize],
    givptr: usize,
    givcol: &[usize],
    ldgcol: usize,
    givnum: &[f64],
    ldgnum: usize,
    poles: &[f64],
    difl: &[f64],
    difr: &[f64],
    z: &[f64],
    k: usize,
    c: f64,
    s: f64,
) {
    let n = nl + nr + 1;
    let m = n + sqre;
    let nlp1 = nl + 1;
    let pol = |j: usize, col: usize| poles[j - 1 + (col - 1) * ldgnum];
    let dr = |j: usize, col: usize| difr[j - 1 + (col - 1) * ldgnum];
    let mut work = vec![0.0f64; k];
    if icompq == 0 {
        for i in 1..=givptr {
            let (r2, r1) = (givcol[i - 1 + ldgcol], givcol[i - 1]);
            rows_rot(
                b,
                ldb,
                r2 - 1,
                r1 - 1,
                nrhs,
                givnum[i - 1 + ldgnum],
                givnum[i - 1],
            );
        }
        for col in 0..nrhs {
            bx[col * ldbx] = b[(nlp1 - 1) + col * ldb];
        }
        for i in 2..=n {
            for col in 0..nrhs {
                bx[(i - 1) + col * ldbx] = b[(perm[i - 1] - 1) + col * ldb];
            }
        }
        if k == 1 {
            for col in 0..nrhs {
                b[col * ldb] = bx[col * ldbx];
            }
            if z[0] < 0.0 {
                for col in 0..nrhs {
                    b[col * ldb] *= -1.0;
                }
            }
        } else {
            for j in 1..=k {
                let diflj = difl[j - 1];
                let dj = pol(j, 1);
                let dsigj = -pol(j, 2);
                let (mut difrj, mut dsigjp) = (0.0, 0.0);
                if j < k {
                    difrj = -dr(j, 1);
                    dsigjp = -pol(j + 1, 2);
                }
                work[j - 1] = if z[j - 1] == 0.0 || pol(j, 2) == 0.0 {
                    0.0
                } else {
                    -pol(j, 2) * z[j - 1] / diflj / (pol(j, 2) + dj)
                };
                for i in 1..j {
                    work[i - 1] = if z[i - 1] == 0.0 || pol(i, 2) == 0.0 {
                        0.0
                    } else {
                        pol(i, 2) * z[i - 1] / (dlamc3(pol(i, 2), dsigj) - diflj) / (pol(i, 2) + dj)
                    };
                }
                for i in j + 1..=k {
                    work[i - 1] = if z[i - 1] == 0.0 || pol(i, 2) == 0.0 {
                        0.0
                    } else {
                        pol(i, 2) * z[i - 1]
                            / (dlamc3(pol(i, 2), dsigjp) + difrj)
                            / (pol(i, 2) + dj)
                    };
                }
                work[0] = -1.0;
                let temp = ob::dnrm2(k, &work, 1);
                // y is B(J, 1) at stride LDB: OpenBLAS's `incy != 1` path
                write_row_gemv_t(b, ldb, j - 1, nrhs, k, bx, ldbx, &work);
                dlascl(temp, 1.0, 1, nrhs, &mut b[j - 1..], ldb);
            }
        }
        if k < m.max(n) {
            for col in 0..nrhs {
                for r in k..n {
                    b[r + col * ldb] = bx[r + col * ldbx];
                }
            }
        }
    } else {
        if k == 1 {
            for col in 0..nrhs {
                bx[col * ldbx] = b[col * ldb];
            }
        } else {
            for j in 1..=k {
                let dsigj = pol(j, 2);
                work[j - 1] = if z[j - 1] == 0.0 {
                    0.0
                } else {
                    -z[j - 1] / difl[j - 1] / (dsigj + pol(j, 1)) / dr(j, 2)
                };
                for i in 1..j {
                    work[i - 1] = if z[j - 1] == 0.0 {
                        0.0
                    } else {
                        z[j - 1]
                            / (dlamc3(dsigj, -pol(i + 1, 2)) - dr(i, 1))
                            / (dsigj + pol(i, 1))
                            / dr(i, 2)
                    };
                }
                for i in j + 1..=k {
                    work[i - 1] = if z[j - 1] == 0.0 {
                        0.0
                    } else {
                        z[j - 1]
                            / (dlamc3(dsigj, -pol(i, 2)) - difl[i - 1])
                            / (dsigj + pol(i, 1))
                            / dr(i, 2)
                    };
                }
                write_row_gemv_t(bx, ldbx, j - 1, nrhs, k, b, ldb, &work);
            }
        }
        if sqre == 1 {
            for col in 0..nrhs {
                bx[(m - 1) + col * ldbx] = b[(m - 1) + col * ldb];
            }
            rows_rot(bx, ldbx, 0, m - 1, nrhs, c, s);
        }
        if k < m.max(n) {
            for col in 0..nrhs {
                for r in k..n {
                    bx[r + col * ldbx] = b[r + col * ldb];
                }
            }
        }
        for col in 0..nrhs {
            b[(nlp1 - 1) + col * ldb] = bx[col * ldbx];
        }
        if sqre == 1 {
            for col in 0..nrhs {
                b[(m - 1) + col * ldb] = bx[(m - 1) + col * ldbx];
            }
        }
        for i in 2..=n {
            for col in 0..nrhs {
                b[(perm[i - 1] - 1) + col * ldb] = bx[(i - 1) + col * ldbx];
            }
        }
        for i in (1..=givptr).rev() {
            let (r2, r1) = (givcol[i - 1 + ldgcol], givcol[i - 1]);
            rows_rot(
                b,
                ldb,
                r2 - 1,
                r1 - 1,
                nrhs,
                givnum[i - 1 + ldgnum],
                -givnum[i - 1],
            );
        }
    }
}

/// `DGEMV('T', k, nrhs, 1, A, lda, x, 1, 0, Y(row, 1), ldy)`: the output row has
/// stride `ldy` (OpenBLAS's `incy != 1` path).
#[allow(clippy::too_many_arguments)]
fn write_row_gemv_t(
    out: &mut [f64],
    ldy: usize,
    row: usize,
    nrhs: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    x: &[f64],
) {
    let mut yy = vec![0.0; (nrhs.max(1) - 1) * ldy + 1];
    assert!(ob::dgemv_ab(
        102, 112, k, nrhs, 1.0, a, lda, x, 1, 0.0, &mut yy, ldy
    ));
    for col in 0..nrhs {
        out[row + col * ldy] = yy[col * ldy];
    }
}

/// `DROT(nrhs, B(r0,1), ldb, B(r1,1), ldb, c, s)`.
fn rows_rot(b: &mut [f64], ldb: usize, r0: usize, r1: usize, n: usize, c: f64, s: f64) {
    for col in 0..n {
        let (x, y) = (b[r0 + col * ldb], b[r1 + col * ldb]);
        b[r0 + col * ldb] = c.mul_add(x, s * y);
        b[r1 + col * ldb] = c.mul_add(y, -(s * x));
    }
}

/// `DLALSA(icompq, smlsiz, n, nrhs, B, BX, U, VT, K, DIFL, DIFR, Z, POLES, GIVPTR,
/// GIVCOL, PERM, GIVNUM, C, S)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dlalsa(
    icompq: usize,
    smlsiz: usize,
    n: usize,
    nrhs: usize,
    b: &mut [f64],
    ldb: usize,
    bx: &mut [f64],
    ldbx: usize,
    f: &Factors,
) {
    let (ldu, ldgcol) = (f.ldu, f.ldgcol);
    let (nlvl, nd, inode, ndiml, ndimr) = dlasdt(n, smlsiz);
    let gemm_tn = |rows: usize,
                   u: &[f64],
                   ldu: usize,
                   src: &[f64],
                   lds: usize,
                   dst: &mut [f64],
                   ldd: usize| {
        ob::dgemm_ab(
            102, 112, 111, rows, nrhs, rows, 1.0, u, ldu, src, lds, 0.0, dst, ldd,
        );
    };
    let node = |lvl: usize, nlf: usize, j: usize| {
        let p = nlf - 1;
        let lvl2 = 2 * lvl - 1;
        (
            &f.perm[p + (lvl - 1) * ldgcol..],
            f.givptr[j - 1],
            &f.givcol[p + (lvl2 - 1) * ldgcol..],
            &f.givnum[p + (lvl2 - 1) * ldu..],
            &f.poles[p + (lvl2 - 1) * ldu..],
            &f.difl[p + (lvl - 1) * ldu..],
            &f.difr[p + (lvl2 - 1) * ldu..],
            &f.z[p + (lvl - 1) * ldu..],
            f.k[j - 1],
            f.c[j - 1],
            f.s[j - 1],
        )
    };
    if icompq == 0 {
        let ndb1 = (nd + 1) / 2;
        for i in ndb1..=nd {
            let i1 = i - 1;
            let ic = inode[i1];
            let nl = ndiml[i1];
            let nr = ndimr[i1];
            let nlf = ic - nl;
            let nrf = ic + 1;
            let bsrc: Vec<f64> = b.to_vec();
            gemm_tn(
                nl,
                &f.u[nlf - 1..],
                ldu,
                &bsrc[nlf - 1..],
                ldb,
                &mut bx[nlf - 1..],
                ldbx,
            );
            gemm_tn(
                nr,
                &f.u[nrf - 1..],
                ldu,
                &bsrc[nrf - 1..],
                ldb,
                &mut bx[nrf - 1..],
                ldbx,
            );
        }
        for i in 1..=nd {
            let ic = inode[i - 1];
            for col in 0..nrhs {
                bx[(ic - 1) + col * ldbx] = b[(ic - 1) + col * ldb];
            }
        }
        let mut j = 1usize << nlvl;
        for lvl in (1..=nlvl).rev() {
            let (lf, ll) = if lvl == 1 {
                (1, 1)
            } else {
                (1usize << (lvl - 1), (1usize << lvl) - 1)
            };
            for i in lf..=ll {
                let im1 = i - 1;
                let ic = inode[im1];
                let nl = ndiml[im1];
                let nr = ndimr[im1];
                let nlf = ic - nl;
                j -= 1;
                let (perm, givptr, givcol, givnum, poles, difl, difr, z, k, c, s) =
                    node(lvl, nlf, j);
                dlals0(
                    0,
                    nl,
                    nr,
                    0,
                    nrhs,
                    &mut bx[nlf - 1..],
                    ldbx,
                    &mut b[nlf - 1..],
                    ldb,
                    perm,
                    givptr,
                    givcol,
                    ldgcol,
                    givnum,
                    ldu,
                    poles,
                    difl,
                    difr,
                    z,
                    k,
                    c,
                    s,
                );
            }
        }
    } else {
        let mut j = 0usize;
        for lvl in 1..=nlvl {
            let (lf, ll) = if lvl == 1 {
                (1, 1)
            } else {
                (1usize << (lvl - 1), (1usize << lvl) - 1)
            };
            for i in (lf..=ll).rev() {
                let im1 = i - 1;
                let ic = inode[im1];
                let nl = ndiml[im1];
                let nr = ndimr[im1];
                let nlf = ic - nl;
                let sqre = if i == ll { 0 } else { 1 };
                j += 1;
                let (perm, givptr, givcol, givnum, poles, difl, difr, z, k, c, s) =
                    node(lvl, nlf, j);
                dlals0(
                    1,
                    nl,
                    nr,
                    sqre,
                    nrhs,
                    &mut b[nlf - 1..],
                    ldb,
                    &mut bx[nlf - 1..],
                    ldbx,
                    perm,
                    givptr,
                    givcol,
                    ldgcol,
                    givnum,
                    ldu,
                    poles,
                    difl,
                    difr,
                    z,
                    k,
                    c,
                    s,
                );
            }
        }
        let ndb1 = (nd + 1) / 2;
        for i in ndb1..=nd {
            let i1 = i - 1;
            let ic = inode[i1];
            let nl = ndiml[i1];
            let nr = ndimr[i1];
            let nlp1 = nl + 1;
            let nrp1 = if i == nd { nr } else { nr + 1 };
            let nlf = ic - nl;
            let nrf = ic + 1;
            let bsrc: Vec<f64> = b.to_vec();
            gemm_tn(
                nlp1,
                &f.vt[nlf - 1..],
                ldu,
                &bsrc[nlf - 1..],
                ldb,
                &mut bx[nlf - 1..],
                ldbx,
            );
            gemm_tn(
                nrp1,
                &f.vt[nrf - 1..],
                ldu,
                &bsrc[nrf - 1..],
                ldb,
                &mut bx[nrf - 1..],
                ldbx,
            );
        }
    }
}
