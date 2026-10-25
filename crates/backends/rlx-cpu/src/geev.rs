// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Eigenvalues of a general (nonsymmetric) real matrix.
//!
//! [`dgeev_eigvals`] is LAPACK `dgeev` (`jobvl = jobvr = 'N'`) where a platform
//! LAPACK is linked, and [`hqr_eigvals`] otherwise. `hqr_eigvals` is always
//! compiled so it can be tested against LAPACK on the hosts that have one:
//! Householder reduction to upper Hessenberg form, then the Francis
//! double-shift QR iteration with the classic exceptional shifts at iterations
//! 10 and 20 (EISPACK `hqr`'s scheme). It does not balance the matrix first, so
//! it agrees with `dgeev` to rounding, not bit for bit.

#[cfg(rlx_cpu_blas)]
unsafe extern "C" {
    #[link_name = "dgeev_"]
    fn lapack_dgeev(
        jobvl: *const i8,
        jobvr: *const i8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Eigenvalues `(re, im)` of the row-major `n × n` matrix `a`, in LAPACK's
/// order (complex pairs adjacent, positive imaginary part first). A matrix and
/// its transpose share eigenvalues, so the row-major buffer is passed as is.
pub fn dgeev_eigvals(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if a.len() != n * n {
        return Err(format!(
            "eigvals: expected {} values for n={n}, got {}",
            n * n,
            a.len()
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(rlx_cpu_blas)]
    {
        let mut buf = a.to_vec();
        let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
        let nn = n as i32;
        let one = 1i32;
        let lwork = (4 * n).max(1) as i32;
        let mut work = vec![0.0; lwork as usize];
        let mut dummy = [0.0f64; 1];
        let mut info = 0i32;
        let no = b'N' as i8;
        unsafe {
            lapack_dgeev(
                &no,
                &no,
                &nn,
                buf.as_mut_ptr(),
                &nn,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                dummy.as_mut_ptr(),
                &one,
                dummy.as_mut_ptr(),
                &one,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("eigvals: dgeev info={info}"));
        }
        Ok((wr, wi))
    }
    #[cfg(not(rlx_cpu_blas))]
    {
        hqr_eigvals(a, n)
    }
}

/// Reduce row-major `h` (`n × n`) to upper Hessenberg form in place by
/// Householder similarity transforms.
fn hessenberg(h: &mut [f64], n: usize) {
    let at = |i: usize, j: usize| i * n + j;
    for k in 0..n.saturating_sub(2) {
        let alpha: f64 = (k + 1..n)
            .map(|i| h[at(i, k)] * h[at(i, k)])
            .sum::<f64>()
            .sqrt();
        if alpha == 0.0 {
            continue;
        }
        let x0 = h[at(k + 1, k)];
        let beta = if x0 > 0.0 { -alpha } else { alpha };
        let mut v = vec![0.0; n];
        v[k + 1] = x0 - beta;
        for i in k + 2..n {
            v[i] = h[at(i, k)];
        }
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        // H ← (I − 2vvᵀ/‖v‖²) H
        for j in 0..n {
            let s: f64 = (k + 1..n).map(|i| v[i] * h[at(i, j)]).sum::<f64>() * 2.0 / vnorm2;
            for i in k + 1..n {
                h[at(i, j)] -= s * v[i];
            }
        }
        // H ← H (I − 2vvᵀ/‖v‖²)
        for i in 0..n {
            let s: f64 = (k + 1..n).map(|j| h[at(i, j)] * v[j]).sum::<f64>() * 2.0 / vnorm2;
            for j in k + 1..n {
                h[at(i, j)] -= s * v[j];
            }
        }
        for i in k + 2..n {
            h[at(i, k)] = 0.0;
        }
    }
}

fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// Eigenvalues of a general real matrix without LAPACK (see the module docs).
/// Order: deflation order, complex pairs adjacent (positive imaginary first).
pub fn hqr_eigvals(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if a.len() != n * n {
        return Err(format!(
            "eigvals: expected {} values for n={n}, got {}",
            n * n,
            a.len()
        ));
    }
    let mut h = a.to_vec();
    hessenberg(&mut h, n);
    let idx = |i: usize, j: usize| i * n + j;
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let anorm: f64 = h.iter().map(|x| x.abs()).sum();
    let mut nn = n as isize - 1;
    let mut t = 0.0;
    while nn >= 0 {
        let mut its = 0;
        loop {
            let nu = nn as usize;
            // a negligible subdiagonal element splits the matrix
            let mut l = nu;
            while l >= 1 {
                let mut s = h[idx(l - 1, l - 1)].abs() + h[idx(l, l)].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if h[idx(l, l - 1)].abs() + s == s {
                    h[idx(l, l - 1)] = 0.0;
                    break;
                }
                l -= 1;
            }
            let mut x = h[idx(nu, nu)];
            if l == nu {
                wr[nu] = x + t;
                wi[nu] = 0.0;
                nn -= 1;
                break;
            }
            let mut y = h[idx(nu - 1, nu - 1)];
            let mut w = h[idx(nu, nu - 1)] * h[idx(nu - 1, nu)];
            if l == nu - 1 {
                // a 2 × 2 block
                let p = 0.5 * (y - x);
                let q = p * p + w;
                let z = q.abs().sqrt();
                x += t;
                if q >= 0.0 {
                    let z = p + sign(z, p);
                    wr[nu - 1] = x + z;
                    wr[nu] = if z != 0.0 { x - w / z } else { x + z };
                    wi[nu - 1] = 0.0;
                    wi[nu] = 0.0;
                } else {
                    wr[nu - 1] = x + p;
                    wr[nu] = x + p;
                    wi[nu - 1] = z;
                    wi[nu] = -z;
                }
                nn -= 2;
                break;
            }
            if its == 60 {
                return Err("eigvals: QR iteration did not converge".into());
            }
            if its == 10 || its == 20 {
                // exceptional shift
                t += x;
                for i in 0..=nu {
                    h[idx(i, i)] -= x;
                }
                let s = h[idx(nu, nu - 1)].abs() + h[idx(nu - 1, nu - 2)].abs();
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            its += 1;
            // two consecutive small subdiagonal elements
            let mut m = nu - 2;
            let (mut p, mut q, mut r);
            loop {
                let z = h[idx(m, m)];
                let rr = x - z;
                let ss = y - z;
                p = (rr * ss - w) / h[idx(m + 1, m)] + h[idx(m, m + 1)];
                q = h[idx(m + 1, m + 1)] - z - rr - ss;
                r = h[idx(m + 2, m + 1)];
                let s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let u = h[idx(m, m - 1)].abs() * (q.abs() + r.abs());
                let v =
                    p.abs() * (h[idx(m - 1, m - 1)].abs() + z.abs() + h[idx(m + 1, m + 1)].abs());
                if u + v == v {
                    break;
                }
                m -= 1;
            }
            for i in m + 2..=nu {
                h[idx(i, i - 2)] = 0.0;
                if i != m + 2 {
                    h[idx(i, i - 3)] = 0.0;
                }
            }
            // the double-shift QR step on rows l..=nu, columns m..=nu
            let mut xk = 0.0;
            for k in m..nu {
                if k != m {
                    p = h[idx(k, k - 1)];
                    q = h[idx(k + 1, k - 1)];
                    r = if k != nu - 1 {
                        h[idx(k + 2, k - 1)]
                    } else {
                        0.0
                    };
                    xk = p.abs() + q.abs() + r.abs();
                    if xk != 0.0 {
                        p /= xk;
                        q /= xk;
                        r /= xk;
                    }
                }
                let s = sign((p * p + q * q + r * r).sqrt(), p);
                if s == 0.0 {
                    continue;
                }
                if k == m {
                    if l != m {
                        h[idx(k, k - 1)] = -h[idx(k, k - 1)];
                    }
                } else {
                    h[idx(k, k - 1)] = -s * xk;
                }
                p += s;
                let (xx, yy, zz) = (p / s, q / s, r / s);
                q /= p;
                r /= p;
                for j in k..=nu {
                    let mut pp = h[idx(k, j)] + q * h[idx(k + 1, j)];
                    if k != nu - 1 {
                        pp += r * h[idx(k + 2, j)];
                        h[idx(k + 2, j)] -= pp * zz;
                    }
                    h[idx(k + 1, j)] -= pp * yy;
                    h[idx(k, j)] -= pp * xx;
                }
                let mmin = if nu < k + 3 { nu } else { k + 3 };
                for i in l..=mmin {
                    let mut pp = xx * h[idx(i, k)] + yy * h[idx(i, k + 1)];
                    if k != nu - 1 {
                        pp += zz * h[idx(i, k + 2)];
                        h[idx(i, k + 2)] -= pp * r;
                    }
                    h[idx(i, k + 1)] -= pp * q;
                    h[idx(i, k)] -= pp;
                }
            }
        }
    }
    Ok((wr, wi))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(wr: &[f64], wi: &[f64]) -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = wr.iter().copied().zip(wi.iter().copied()).collect();
        v.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        v
    }

    #[test]
    fn known_spectra() {
        // rotation-scaling block + a real eigenvalue: {1 ± 2i, 3}
        let a = [1.0, -2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 3.0];
        for f in [dgeev_eigvals, hqr_eigvals] {
            let (wr, wi) = f(&a, 3).unwrap();
            let s = sorted(&wr, &wi);
            let want = [(1.0, -2.0), (1.0, 2.0), (3.0, 0.0)];
            for (g, w) in s.iter().zip(want) {
                assert!(
                    (g.0 - w.0).abs() < 1e-12 && (g.1 - w.1).abs() < 1e-12,
                    "{s:?}"
                );
            }
        }
        // companion matrix of (x−1)(x−2)(x−3)(x−4)
        let c = [
            10.0, -35.0, 50.0, -24.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let (wr, wi) = hqr_eigvals(&c, 4).unwrap();
        let s = sorted(&wr, &wi);
        for (k, g) in s.iter().enumerate() {
            assert!(
                (g.0 - (k + 1) as f64).abs() < 1e-10 && g.1.abs() < 1e-10,
                "{s:?}"
            );
        }
    }

    /// The fallback agrees with LAPACK on dense random matrices.
    #[test]
    fn hqr_matches_dgeev() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for n in [1usize, 2, 3, 5, 10, 17, 30] {
            let a: Vec<f64> = (0..n * n).map(|_| rnd() * 4.0).collect();
            let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let (r1, i1) = dgeev_eigvals(&a, n).unwrap();
            let (r2, i2) = hqr_eigvals(&a, n).unwrap();
            let (s1, s2) = (sorted(&r1, &i1), sorted(&r2, &i2));
            for (x, y) in s1.iter().zip(&s2) {
                assert!(
                    (x.0 - y.0).abs() + (x.1 - y.1).abs() < 1e-10 * norm.max(1.0),
                    "n={n}: {s1:?} vs {s2:?}"
                );
            }
        }
    }
}
