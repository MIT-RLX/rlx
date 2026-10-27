// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! LAPACK `DSYEVD('V', 'L')` — the symmetric eigensolver behind `numpy.linalg.eigh`
//! — as the Linux x86-64 reference runs it (reference LAPACK 3.12 in OpenBLAS 0.3.30,
//! gfortran without FMA contraction; the BLAS calls are the OpenBLAS Haswell kernels
//! already ported in [`crate::openblas`]): `DSYTRD` → `DSTEDC('I')` (divide and conquer:
//! `DLAED0`–`DLAED6`, `DSTEQR` leaves) → `DORMTR` → `DLACPY`.
//!
//! Fortran index *values* (`INDXQ`, `INDX`, …) stay 1-based, as in the source, and are
//! converted only where they index an array; loop variables are 0-based.

use crate::lapack::{
    EPS, PREC, SFMIN, dlamrg, dlanst_max, dlapy2, dlartg, dlascl, dlasr, dormxx_left, ob_idamax,
};
use crate::lapack_dc::dlaed6;
use crate::lapack_eig::dlansy_max_lower;
use crate::lapack_mrrr::dlaev2;
use crate::openblas::{dgemm, dnrm2, drot, dscal};

/// `ILAENV(9, 'DSTEDC' / 'DLAED0')`: the largest subproblem solved by `DSTEQR`.
const SMLSIZ: usize = 25;
/// `ILAENV(1, 'DSYTRD')`, which sizes the workspace numpy queries.
const NB_SYTRD: usize = 32;

/// Fortran `SIGN(A, B)`: `|A|` with the sign of `B` (gfortran: `copysign`).
fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

/// `numpy.linalg.eigh(a)` (`UPLO='L'`) on the Linux reference: `a` is the column-major
/// `n × n` matrix (only its lower triangle is read). Returns the eigenvalues in ascending
/// order and the eigenvectors (column-major, column `j` for eigenvalue `j`), or `INFO`.
/// The workspace is numpy's: `DSYEVD`'s own optimum from a workspace query.
pub fn dsyevd_lower(n: usize, a: &[f64], lda: usize) -> Result<(Vec<f64>, Vec<f64>), i64> {
    let lwork = if n <= 1 {
        1
    } else {
        (1 + 6 * n + 2 * n * n).max(2 * n + n * NB_SYTRD)
    };
    let mut a: Vec<f64> = a.to_vec();
    let lda = lda.max(1);
    let mut w = vec![0.0f64; n];
    if n == 0 {
        return Ok((w, Vec::new()));
    }
    if n == 1 {
        w[0] = a[0];
        return Ok((w, vec![1.0]));
    }
    let safmin = SFMIN;
    let eps = PREC; // DLAMCH('Precision')
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();
    let anrm = dlansy_max_lower(n, &a, lda);
    let mut iscale = false;
    let mut sigma = 1.0;
    if anrm > 0.0 && anrm < rmin {
        iscale = true;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = true;
        sigma = rmax / anrm;
    }
    if iscale {
        // DLASCL('L'): the same per-element product; the strict upper triangle is never read
        dlascl(1.0, sigma, n, n, &mut a, lda);
    }
    // WORK layout: E (n), TAU (n), Z (n²) then the DSTEDC / DORMTR scratch.
    let llwork = lwork - 2 * n;
    let llwrk2 = lwork - (2 * n + n * n);
    let (d, mut e, tau) = crate::lapack_eig::dsytrd_lower_pub(n, &mut a, lda, llwork as i64);
    w.copy_from_slice(&d);
    let mut z = vec![0.0f64; n * n];
    let info = dstedc_i(n, &mut w, &mut e, &mut z, n);
    if info != 0 {
        return Err(info);
    }
    // DORMTR('L', 'L', 'N'): DORMQR on A(2,1) applied to Z(2,1), k = n − 1
    dormxx_left(
        false,
        true,
        n - 1,
        n,
        n - 1,
        &mut a[1..],
        lda,
        &tau,
        &mut z[1..],
        n,
        llwrk2 as i64,
    );
    if iscale {
        dscal(n, 1.0 / sigma, &mut w, 1);
    }
    Ok((w, z))
}

/// `DSTEDC('I', n, d, e, Z, ldz)`: eigenvalues ascending in `d`, eigenvectors in `Z`.
fn dstedc_i(n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64], ldz: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        z[0] = 1.0;
        return 0;
    }
    if n <= SMLSIZ {
        return dsteqr_i(n, d, e, z, ldz);
    }
    // DLASET('Full', 0, 1)
    for j in 0..n {
        for i in 0..n {
            z[i + j * ldz] = if i == j { 1.0 } else { 0.0 };
        }
    }
    let orgnrm = dlanst_max(n, d, e);
    if orgnrm == 0.0 {
        return 0;
    }
    let eps = EPS; // DLAMCH('Epsilon')
    let mut start = 1usize; // 1-based, as the source
    while start <= n {
        let mut finish = start;
        while finish < n {
            let tiny = eps * d[finish - 1].abs().sqrt() * d[finish].abs().sqrt();
            if e[finish - 1].abs() > tiny {
                finish += 1;
            } else {
                break;
            }
        }
        let m = finish - start + 1;
        if m == 1 {
            start = finish + 1;
            continue;
        }
        let s0 = start - 1;
        if m > SMLSIZ {
            let orgnrm = dlanst_max(m, &d[s0..], &e[s0..]);
            dlascl(orgnrm, 1.0, m, 1, &mut d[s0..], m);
            dlascl(orgnrm, 1.0, m - 1, 1, &mut e[s0..], m - 1);
            let info = dlaed0_2(m, &mut d[s0..], &mut e[s0..], &mut z[s0 + s0 * ldz..], ldz);
            if info != 0 {
                let (m1, info) = ((m + 1) as i64, info);
                return (info / m1 + start as i64 - 1) * (n as i64 + 1) + info % m1 + start as i64
                    - 1;
            }
            dlascl(1.0, orgnrm, m, 1, &mut d[s0..], m);
        } else {
            let info = dsteqr_i(m, &mut d[s0..], &mut e[s0..], &mut z[s0 + s0 * ldz..], ldz);
            if info != 0 {
                return start as i64 * (n as i64 + 1) + finish as i64;
            }
        }
        start = finish + 1;
    }
    selection_sort_with_vectors(n, d, z, ldz);
    0
}

/// The selection sort `DSTEDC` / `DSTEQR` end with: ascending `d`, `Z`'s columns swapped along.
fn selection_sort_with_vectors(n: usize, d: &mut [f64], z: &mut [f64], ldz: usize) {
    for ii in 1..n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().take(n).skip(ii) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for r in 0..n {
                z.swap(r + i * ldz, r + k * ldz);
            }
        }
    }
}

/// `DSTEQR('I', n, d, e, Z, ldz)`: implicit QL/QR; `Z` is set to the identity first.
fn dsteqr_i(n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64], ldz: usize) -> i64 {
    const MAXIT: usize = 30;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        z[0] = 1.0;
        return 0;
    }
    let eps = EPS;
    let eps2 = eps * eps;
    let safmin = SFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;
    for j in 0..n {
        for i in 0..n {
            z[i + j * ldz] = if i == j { 1.0 } else { 0.0 };
        }
    }
    let mut work = vec![0.0f64; 2 * (n - 1)];
    let nmaxit = n * MAXIT;
    let mut jtot = 0usize;
    // 1-based L1, M, L, LEND, as the source
    let mut l1 = 1usize;
    let nm1 = n - 1;
    loop {
        // label 10
        if l1 > n {
            break;
        }
        if l1 > 1 {
            e[l1 - 2] = 0.0;
        }
        let mut m = n;
        if l1 <= nm1 {
            for mm in l1..=nm1 {
                let tst = e[mm - 1].abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (d[mm - 1].abs().sqrt() * d[mm].abs().sqrt()) * eps {
                    e[mm - 1] = 0.0;
                    m = mm;
                    break;
                }
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue;
        }
        let anorm = dlanst_max(lend - l + 1, &d[l - 1..], &e[l - 1..]);
        let mut iscale = 0;
        if anorm == 0.0 {
            continue;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl(anorm, ssfmax, lend - l + 1, 1, &mut d[l - 1..], n);
            dlascl(anorm, ssfmax, lend - l, 1, &mut e[l - 1..], n);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl(anorm, ssfmin, lend - l + 1, 1, &mut d[l - 1..], n);
            dlascl(anorm, ssfmin, lend - l, 1, &mut e[l - 1..], n);
        }
        if d[lend - 1].abs() < d[l - 1].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            // QL iteration (label 40)
            loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let tst = e[mm - 1].abs() * e[mm - 1].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // label 80
                    d[l - 1] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 1], e[l - 1], d[l]);
                    work[l - 1] = c;
                    work[n - 1 + l - 1] = s;
                    let (cw, sw) = work.split_at(n - 1);
                    dlasr(
                        false,
                        false,
                        n,
                        2,
                        &cw[l - 1..],
                        &sw[l - 1..],
                        &mut z[(l - 1) * ldz..],
                        ldz,
                    );
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l] - p) / (2.0 * e[l - 1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 1] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let mm1 = m - 1;
                for i in (l..=mm1).rev() {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m - 1 {
                        e[i] = r;
                    }
                    g = d[i] - p;
                    r = (d[i - 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = -s;
                }
                let mm = m - l + 1;
                let (cw, sw) = work.split_at(n - 1);
                dlasr(
                    false,
                    false,
                    n,
                    mm,
                    &cw[l - 1..],
                    &sw[l - 1..],
                    &mut z[(l - 1) * ldz..],
                    ldz,
                );
                d[l - 1] -= p;
                e[l - 1] = g;
            }
        } else {
            // QR iteration (label 90)
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm > lend {
                        let tst = e[mm - 2].abs() * e[mm - 2].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm - 2].abs() + safmin {
                            m = mm;
                            break;
                        }
                        mm -= 1;
                    }
                }
                if m > lend {
                    e[m - 2] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // label 130
                    d[l - 1] = p;
                    if l > lend {
                        l -= 1;
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 2], e[l - 2], d[l - 1]);
                    work[m - 1] = c;
                    work[n - 1 + m - 1] = s;
                    let (cw, sw) = work.split_at(n - 1);
                    dlasr(
                        false,
                        true,
                        n,
                        2,
                        &cw[m - 1..],
                        &sw[m - 1..],
                        &mut z[(l - 2) * ldz..],
                        ldz,
                    );
                    d[l - 2] = rt1;
                    d[l - 1] = rt2;
                    e[l - 2] = 0.0;
                    if l >= lend + 2 {
                        l -= 2;
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l - 2] - p) / (2.0 * e[l - 2]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 2] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m {
                        e[i - 2] = r;
                    }
                    g = d[i - 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i - 1] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[n - 1 + i - 1] = s;
                }
                let mm = l - m + 1;
                let (cw, sw) = work.split_at(n - 1);
                dlasr(
                    false,
                    true,
                    n,
                    mm,
                    &cw[m - 1..],
                    &sw[m - 1..],
                    &mut z[(m - 1) * ldz..],
                    ldz,
                );
                d[l - 1] -= p;
                e[lm1 - 1] = g;
            }
        }
        // label 140: undo scaling
        if iscale == 1 {
            dlascl(ssfmax, anorm, lendsv - lsv + 1, 1, &mut d[lsv - 1..], n);
            dlascl(ssfmax, anorm, lendsv - lsv, 1, &mut e[lsv - 1..], n);
        } else if iscale == 2 {
            dlascl(ssfmin, anorm, lendsv - lsv + 1, 1, &mut d[lsv - 1..], n);
            dlascl(ssfmin, anorm, lendsv - lsv, 1, &mut e[lsv - 1..], n);
        }
        if jtot < nmaxit {
            continue;
        }
        return e[..n - 1].iter().filter(|&&v| v != 0.0).count() as i64;
    }
    selection_sort_with_vectors(n, d, z, ldz);
    0
}

/// `DLAED0(ICOMPQ=2, n, d, e, Q, ldq)`: divide and conquer on a tridiagonal matrix whose
/// off-diagonal `e` couples halves; `Q` receives the eigenvectors.
fn dlaed0_2(n: usize, d: &mut [f64], e: &mut [f64], q: &mut [f64], ldq: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    // IWORK(1..SUBPBS): the subproblem sizes, then their running sums (1-based)
    let mut part = vec![0usize; 2 * n + 2];
    part[1] = n;
    let mut subpbs = 1usize;
    while part[subpbs] > SMLSIZ {
        for j in (1..=subpbs).rev() {
            part[2 * j] = part[j].div_ceil(2);
            part[2 * j - 1] = part[j] / 2;
        }
        subpbs *= 2;
    }
    for j in 2..=subpbs {
        part[j] += part[j - 1];
    }
    let spm1 = subpbs - 1;
    for i in 1..=spm1 {
        let submat = part[i] + 1;
        let smm1 = submat - 1;
        d[smm1 - 1] -= e[smm1 - 1].abs();
        d[submat - 1] -= e[smm1 - 1].abs();
    }
    // INDXQ (1-based positions, 1-based values)
    let mut indxq = vec![0usize; n + 1];
    for i in 0..=spm1 {
        let (submat, matsiz) = if i == 0 {
            (1, part[1])
        } else {
            (part[i] + 1, part[i + 1] - part[i])
        };
        let s0 = submat - 1;
        let info = dsteqr_i(
            matsiz,
            &mut d[s0..],
            &mut e[s0..],
            &mut q[s0 + s0 * ldq..],
            ldq,
        );
        if info != 0 {
            return (submat * (n + 1) + submat + matsiz - 1) as i64;
        }
        for (k, j) in (submat..=part[i + 1]).enumerate() {
            indxq[j] = k + 1;
        }
    }
    while subpbs > 1 {
        let spm2 = subpbs - 2;
        let mut i = 0;
        while i <= spm2 {
            let (submat, matsiz, msd2) = if i == 0 {
                (1, part[2], part[1])
            } else {
                let matsiz = part[i + 2] - part[i];
                (part[i] + 1, matsiz, matsiz / 2)
            };
            let s0 = submat - 1;
            let rho = e[submat + msd2 - 2];
            let info = dlaed1(
                matsiz,
                &mut d[s0..],
                &mut q[s0 + s0 * ldq..],
                ldq,
                &mut indxq[submat..submat + matsiz],
                rho,
                msd2,
            );
            if info != 0 {
                return (submat * (n + 1) + submat + matsiz - 1) as i64;
            }
            part[i / 2 + 1] = part[i + 2];
            i += 2;
        }
        subpbs /= 2;
    }
    // sort by INDXQ: D and Q's columns
    let mut dw = vec![0.0f64; n];
    let mut qw = vec![0.0f64; n * n];
    for i in 1..=n {
        let j = indxq[i];
        dw[i - 1] = d[j - 1];
        for r in 0..n {
            qw[r + (i - 1) * n] = q[r + (j - 1) * ldq];
        }
    }
    d[..n].copy_from_slice(&dw);
    for c in 0..n {
        q[c * ldq..c * ldq + n].copy_from_slice(&qw[c * n..c * n + n]);
    }
    0
}

/// `DLAED1(n, d, Q, ldq, INDXQ, rho, cutpnt)`: merge two solved halves joined by a rank-one
/// update.
fn dlaed1(
    n: usize,
    d: &mut [f64],
    q: &mut [f64],
    ldq: usize,
    indxq: &mut [usize],
    rho: f64,
    cutpnt: usize,
) -> i64 {
    if n == 0 {
        return 0;
    }
    // Z: the last row of Q's first block, then the first row of its second block
    let mut z = vec![0.0f64; n];
    for j in 0..cutpnt {
        z[j] = q[(cutpnt - 1) + j * ldq];
    }
    for j in cutpnt..n {
        z[j] = q[cutpnt + j * ldq];
    }
    let mut dlamda = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let mut q2 = vec![0.0f64; n * n];
    let mut indx = vec![0usize; n];
    let mut indxc = vec![0usize; n];
    let mut indxp = vec![0usize; n];
    let mut coltyp = vec![0usize; n.max(4)];
    let mut rho = rho;
    let k = dlaed2(
        n,
        cutpnt,
        d,
        q,
        ldq,
        indxq,
        &mut rho,
        &mut z,
        &mut dlamda,
        &mut w,
        &mut q2,
        &mut indx,
        &mut indxc,
        &mut indxp,
        &mut coltyp,
    );
    if k != 0 {
        let ctot = [coltyp[0], coltyp[1], coltyp[2], coltyp[3]];
        let info = dlaed3(
            k,
            n,
            cutpnt,
            d,
            q,
            ldq,
            rho,
            &mut dlamda,
            &q2,
            &indxc,
            &ctot,
            &mut w,
        );
        if info != 0 {
            return info;
        }
        dlamrg(k, n - k, d, 1, -1, indxq);
    } else {
        for (i, v) in indxq.iter_mut().enumerate() {
            *v = i + 1;
        }
    }
    0
}

/// `DLAED2`: deflation. Returns `K`, the size of the non-deflated problem; `coltyp[0..4]`
/// ends as `CTOT`.
#[allow(clippy::too_many_arguments)]
fn dlaed2(
    n: usize,
    n1: usize,
    d: &mut [f64],
    q: &mut [f64],
    ldq: usize,
    indxq: &mut [usize],
    rho: &mut f64,
    z: &mut [f64],
    dlamda: &mut [f64],
    w: &mut [f64],
    q2: &mut [f64],
    indx: &mut [usize],
    indxc: &mut [usize],
    indxp: &mut [usize],
    coltyp: &mut [usize],
) -> usize {
    let n2 = n - n1;
    let n1p1 = n1 + 1;
    if *rho < 0.0 {
        dscal(n2, -1.0, &mut z[n1..], 1);
    }
    let t = 1.0 / 2f64.sqrt();
    dscal(n, t, z, 1);
    *rho = (2.0 * *rho).abs();
    for v in indxq.iter_mut().skip(n1) {
        *v += n1;
    }
    for i in 0..n {
        dlamda[i] = d[indxq[i] - 1];
    }
    dlamrg(n1, n2, dlamda, 1, 1, indxc);
    for i in 0..n {
        indx[i] = indxq[indxc[i] - 1];
    }
    let imax = ob_idamax(&z[..n]);
    let jmax = ob_idamax(&d[..n]);
    let eps = EPS;
    let tol = 8.0 * eps * d[jmax].abs().max(z[imax].abs());
    if *rho * z[imax].abs() <= tol {
        // every eigenpair deflates: reorder by INDX
        for j in 0..n {
            let i = indx[j];
            for r in 0..n {
                q2[r + j * n] = q[r + (i - 1) * ldq];
            }
            dlamda[j] = d[i - 1];
        }
        for c in 0..n {
            q[c * ldq..c * ldq + n].copy_from_slice(&q2[c * n..c * n + n]);
        }
        d[..n].copy_from_slice(&dlamda[..n]);
        return 0;
    }
    for c in coltyp.iter_mut().take(n1) {
        *c = 1;
    }
    for c in coltyp.iter_mut().take(n).skip(n1) {
        *c = 3;
    }
    let mut k = 0usize;
    let mut k2 = n + 1; // 1-based
    let mut j = 1usize; // 1-based
    let mut pj = 0usize; // 1-based
    let mut found = false;
    while j <= n {
        let nj = indx[j - 1];
        if *rho * z[nj - 1].abs() <= tol {
            k2 -= 1;
            coltyp[nj - 1] = 4;
            indxp[k2 - 1] = nj;
            if j == n {
                break;
            }
            j += 1;
        } else {
            pj = nj;
            found = true;
            break;
        }
    }
    if found {
        // label 80
        loop {
            j += 1;
            if j > n {
                break;
            }
            let nj = indx[j - 1];
            if *rho * z[nj - 1].abs() <= tol {
                k2 -= 1;
                coltyp[nj - 1] = 4;
                indxp[k2 - 1] = nj;
            } else {
                let mut s = z[pj - 1];
                let mut c = z[nj - 1];
                let tau = dlapy2(c, s);
                let t = d[nj - 1] - d[pj - 1];
                c /= tau;
                s = -s / tau;
                if (t * c * s).abs() <= tol {
                    z[nj - 1] = tau;
                    z[pj - 1] = 0.0;
                    if coltyp[nj - 1] != coltyp[pj - 1] {
                        coltyp[nj - 1] = 2;
                    }
                    coltyp[pj - 1] = 4;
                    {
                        let (lo, hi) = (pj.min(nj) - 1, pj.max(nj) - 1);
                        let (left, right) = q.split_at_mut(hi * ldq);
                        let (colp, coln) = if pj < nj {
                            (&mut left[lo * ldq..lo * ldq + n], &mut right[..n])
                        } else {
                            (&mut right[..n], &mut left[lo * ldq..lo * ldq + n])
                        };
                        drot(n, colp, 1, coln, 1, c, s);
                    }
                    let t = d[pj - 1] * (c * c) + d[nj - 1] * (s * s);
                    d[nj - 1] = d[pj - 1] * (s * s) + d[nj - 1] * (c * c);
                    d[pj - 1] = t;
                    k2 -= 1;
                    let mut i = 1usize;
                    loop {
                        if k2 + i <= n {
                            if d[pj - 1] < d[indxp[k2 + i - 1] - 1] {
                                indxp[k2 + i - 2] = indxp[k2 + i - 1];
                                indxp[k2 + i - 1] = pj;
                                i += 1;
                                continue;
                            } else {
                                indxp[k2 + i - 2] = pj;
                            }
                        } else {
                            indxp[k2 + i - 2] = pj;
                        }
                        break;
                    }
                    pj = nj;
                } else {
                    k += 1;
                    dlamda[k - 1] = d[pj - 1];
                    w[k - 1] = z[pj - 1];
                    indxp[k - 1] = pj;
                    pj = nj;
                }
            }
        }
        // label 100
        k += 1;
        dlamda[k - 1] = d[pj - 1];
        w[k - 1] = z[pj - 1];
        indxp[k - 1] = pj;
    }
    let mut ctot = [0usize; 4];
    for &ct in coltyp.iter().take(n) {
        ctot[ct - 1] += 1;
    }
    let mut psm = [
        1usize,
        1 + ctot[0],
        1 + ctot[0] + ctot[1],
        1 + ctot[0] + ctot[1] + ctot[2],
    ];
    let k = n - ctot[3];
    for jj in 1..=n {
        let js = indxp[jj - 1];
        let ct = coltyp[js - 1];
        indx[psm[ct - 1] - 1] = js;
        indxc[psm[ct - 1] - 1] = jj;
        psm[ct - 1] += 1;
    }
    let mut i = 1usize;
    let mut iq1 = 0usize; // 0-based offsets into Q2
    let mut iq2 = (ctot[0] + ctot[1]) * n1;
    for _ in 0..ctot[0] {
        let js = indx[i - 1];
        q2[iq1..iq1 + n1].copy_from_slice(&q[(js - 1) * ldq..(js - 1) * ldq + n1]);
        z[i - 1] = d[js - 1];
        i += 1;
        iq1 += n1;
    }
    for _ in 0..ctot[1] {
        let js = indx[i - 1];
        q2[iq1..iq1 + n1].copy_from_slice(&q[(js - 1) * ldq..(js - 1) * ldq + n1]);
        q2[iq2..iq2 + n2].copy_from_slice(&q[n1 + (js - 1) * ldq..n1 + (js - 1) * ldq + n2]);
        z[i - 1] = d[js - 1];
        i += 1;
        iq1 += n1;
        iq2 += n2;
    }
    for _ in 0..ctot[2] {
        let js = indx[i - 1];
        q2[iq2..iq2 + n2].copy_from_slice(&q[n1 + (js - 1) * ldq..n1 + (js - 1) * ldq + n2]);
        z[i - 1] = d[js - 1];
        i += 1;
        iq2 += n2;
    }
    let iq1 = iq2;
    for _ in 0..ctot[3] {
        let js = indx[i - 1];
        q2[iq2..iq2 + n].copy_from_slice(&q[(js - 1) * ldq..(js - 1) * ldq + n]);
        iq2 += n;
        z[i - 1] = d[js - 1];
        i += 1;
    }
    if k < n {
        for c in 0..ctot[3] {
            let dst = (k + c) * ldq;
            q[dst..dst + n].copy_from_slice(&q2[iq1 + c * n..iq1 + c * n + n]);
        }
        d[k..n].copy_from_slice(&z[k..n]);
    }
    coltyp[..4].copy_from_slice(&ctot);
    let _ = n1p1;
    k
}

/// `DLAED3`: solve the secular equation for the `K` non-deflated roots and form the
/// merged eigenvectors (`Q2 · S` through `DGEMM`).
#[allow(clippy::too_many_arguments)]
fn dlaed3(
    k: usize,
    n: usize,
    n1: usize,
    d: &mut [f64],
    q: &mut [f64],
    ldq: usize,
    rho: f64,
    dlamda: &mut [f64],
    q2: &[f64],
    indx: &[usize],
    ctot: &[usize; 4],
    w: &mut [f64],
) -> i64 {
    if k == 0 {
        return 0;
    }
    for j in 1..=k {
        let (dlam, info) = dlaed4(
            k,
            j,
            &dlamda[..k],
            &w[..k],
            &mut q[(j - 1) * ldq..(j - 1) * ldq + k],
            rho,
        );
        d[j - 1] = dlam;
        if info != 0 {
            return info;
        }
    }
    let mut s = vec![0.0f64; n * k.max(1)];
    if k == 2 {
        for j in 0..2 {
            w[0] = q[j * ldq];
            w[1] = q[1 + j * ldq];
            q[j * ldq] = w[indx[0] - 1];
            q[1 + j * ldq] = w[indx[1] - 1];
        }
    } else if k > 2 {
        s[..k].copy_from_slice(&w[..k]);
        for i in 0..k {
            w[i] = q[i + i * ldq];
        }
        for j in 0..k {
            for i in 0..j {
                w[i] *= q[i + j * ldq] / (dlamda[i] - dlamda[j]);
            }
            for i in j + 1..k {
                w[i] *= q[i + j * ldq] / (dlamda[i] - dlamda[j]);
            }
        }
        for i in 0..k {
            w[i] = sign((-w[i]).sqrt(), s[i]);
        }
        for j in 0..k {
            for i in 0..k {
                s[i] = w[i] / q[i + j * ldq];
            }
            let temp = dnrm2(k, &s[..k], 1);
            for i in 0..k {
                let ii = indx[i];
                q[i + j * ldq] = s[ii - 1] / temp;
            }
        }
    }
    // label 110: back-transform with the deflated structure
    let n2 = n - n1;
    let n12 = ctot[0] + ctot[1];
    let n23 = ctot[1] + ctot[2];
    for c in 0..k {
        for r in 0..n23 {
            s[r + c * n23] = q[ctot[0] + r + c * ldq];
        }
    }
    let iq2 = n1 * n12;
    if n23 != 0 {
        dgemm(
            crate::COL_MAJOR,
            crate::NO_TRANS,
            crate::NO_TRANS,
            n2,
            k,
            n23,
            &q2[iq2..],
            n2.max(1),
            &s,
            n23,
            &mut q[n1..],
            ldq,
        );
    } else {
        for c in 0..k {
            for r in 0..n2 {
                q[n1 + r + c * ldq] = 0.0;
            }
        }
    }
    for c in 0..k {
        for r in 0..n12 {
            s[r + c * n12] = q[r + c * ldq];
        }
    }
    if n12 != 0 {
        dgemm(
            crate::COL_MAJOR,
            crate::NO_TRANS,
            crate::NO_TRANS,
            n1,
            k,
            n12,
            q2,
            n1.max(1),
            &s,
            n12,
            q,
            ldq,
        );
    } else {
        for c in 0..k {
            for r in 0..n1 {
                q[r + c * ldq] = 0.0;
            }
        }
    }
    0
}

/// `DLAED4(n, i, d, z, delta, rho) → (dlam, info)`: the `i`-th (1-based) root of the
/// secular equation `1 + rho·Σ z_j² / (d_j − λ) = 0`, with `delta_j = d_j − λ`.
fn dlaed4(n: usize, i: usize, d: &[f64], z: &[f64], delta: &mut [f64], rho: f64) -> (f64, i64) {
    const MAXIT: usize = 30;
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const FOUR: f64 = 4.0;
    const EIGHT: f64 = 8.0;
    const TEN: f64 = 10.0;
    // 1-based views: D(j) = d[j-1]
    let dd = |j: usize| d[j - 1];
    let zz = |j: usize| z[j - 1];
    if n == 1 {
        delta[0] = 1.0;
        return (d[0] + rho * z[0] * z[0], 0);
    }
    if n == 2 {
        let (dlam, del) = dlaed5(i, [d[0], d[1]], [z[0], z[1]], rho);
        delta[0] = del[0];
        delta[1] = del[1];
        return (dlam, 0);
    }
    let eps = EPS;
    let rhoinv = 1.0 / rho;
    let dlam;
    if i == n {
        let ii = n - 1;
        let mut niter = 1usize;
        let midpt = rho / TWO;
        for j in 1..=n {
            delta[j - 1] = (dd(j) - dd(i)) - midpt;
        }
        let mut psi = 0.0;
        for j in 1..=n - 2 {
            psi += zz(j) * zz(j) / delta[j - 1];
        }
        let mut c = rhoinv + psi;
        let mut w = c + zz(ii) * zz(ii) / delta[ii - 1] + zz(n) * zz(n) / delta[n - 1];
        let mut tau;
        let (mut dltlb, mut dltub);
        if w <= 0.0 {
            let temp = zz(n - 1) * zz(n - 1) / (dd(n) - dd(n - 1) + rho) + zz(n) * zz(n) / rho;
            if c <= temp {
                tau = rho;
            } else {
                let del = dd(n) - dd(n - 1);
                let a = -c * del + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
                let b = zz(n) * zz(n) * del;
                tau = if a < 0.0 {
                    TWO * b / ((a * a + FOUR * b * c).sqrt() - a)
                } else {
                    (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c)
                };
            }
            dltlb = midpt;
            dltub = rho;
        } else {
            let del = dd(n) - dd(n - 1);
            let a = -c * del + zz(n - 1) * zz(n - 1) + zz(n) * zz(n);
            let b = zz(n) * zz(n) * del;
            tau = if a < 0.0 {
                TWO * b / ((a * a + FOUR * b * c).sqrt() - a)
            } else {
                (a + (a * a + FOUR * b * c).sqrt()) / (TWO * c)
            };
            dltlb = 0.0;
            dltub = midpt;
        }
        for j in 1..=n {
            delta[j - 1] = (dd(j) - dd(i)) - tau;
        }
        let (mut dpsi, mut erretm) = (0.0f64, 0.0f64);
        psi = 0.0;
        for j in 1..=ii {
            let temp = zz(j) / delta[j - 1];
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let temp = zz(n) / delta[n - 1];
        let mut phi = zz(n) * temp;
        let mut dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
        w = rhoinv + phi + psi;
        if w.abs() <= eps * erretm {
            return (dd(i) + tau, 0);
        }
        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }
        niter += 1;
        c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
        let a = (delta[n - 2] + delta[n - 1]) * w - delta[n - 2] * delta[n - 1] * (dpsi + dphi);
        let b = delta[n - 2] * delta[n - 1] * w;
        if c < 0.0 {
            c = c.abs();
        }
        let mut eta = if c == 0.0 {
            -w / (dpsi + dphi)
        } else if a >= 0.0 {
            (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
        } else {
            TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt())
        };
        if w * eta > 0.0 {
            eta = -w / (dpsi + dphi);
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            eta = if w < 0.0 {
                (dltub - tau) / TWO
            } else {
                (dltlb - tau) / TWO
            };
        }
        for j in 1..=n {
            delta[j - 1] -= eta;
        }
        tau += eta;
        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=ii {
            let temp = zz(j) / delta[j - 1];
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        let temp = zz(n) / delta[n - 1];
        phi = zz(n) * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
        w = rhoinv + phi + psi;
        let iter = niter + 1;
        for _niter in iter..=MAXIT {
            if w.abs() <= eps * erretm {
                return (dd(i) + tau, 0);
            }
            if w <= 0.0 {
                dltlb = dltlb.max(tau);
            } else {
                dltub = dltub.min(tau);
            }
            let c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
            let a = (delta[n - 2] + delta[n - 1]) * w - delta[n - 2] * delta[n - 1] * (dpsi + dphi);
            let b = delta[n - 2] * delta[n - 1] * w;
            let mut eta = if a >= 0.0 {
                (a + (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
            } else {
                TWO * b / (a - (a * a - FOUR * b * c).abs().sqrt())
            };
            if w * eta > 0.0 {
                eta = -w / (dpsi + dphi);
            }
            let temp = tau + eta;
            if temp > dltub || temp < dltlb {
                eta = if w < 0.0 {
                    (dltub - tau) / TWO
                } else {
                    (dltlb - tau) / TWO
                };
            }
            for j in 1..=n {
                delta[j - 1] -= eta;
            }
            tau += eta;
            dpsi = 0.0;
            psi = 0.0;
            erretm = 0.0;
            for j in 1..=ii {
                let temp = zz(j) / delta[j - 1];
                psi += zz(j) * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = erretm.abs();
            let temp = zz(n) / delta[n - 1];
            phi = zz(n) * temp;
            dphi = temp * temp;
            erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv + tau.abs() * (dpsi + dphi);
            w = rhoinv + phi + psi;
        }
        return (dd(i) + tau, 1);
    }
    // 1 <= i < n
    let mut niter = 1usize;
    let ip1 = i + 1;
    let del = dd(ip1) - dd(i);
    let midpt = del / TWO;
    for j in 1..=n {
        delta[j - 1] = (dd(j) - dd(i)) - midpt;
    }
    let mut psi = 0.0;
    for j in 1..i {
        psi += zz(j) * zz(j) / delta[j - 1];
    }
    let mut phi = 0.0;
    for j in (i + 2..=n).rev() {
        phi += zz(j) * zz(j) / delta[j - 1];
    }
    let c = rhoinv + psi + phi;
    let w0 = c + zz(i) * zz(i) / delta[i - 1] + zz(ip1) * zz(ip1) / delta[ip1 - 1];
    let orgati;
    let mut tau;
    let (mut dltlb, mut dltub);
    if w0 > 0.0 {
        orgati = true;
        let a = c * del + zz(i) * zz(i) + zz(ip1) * zz(ip1);
        let b = zz(i) * zz(i) * del;
        tau = if a > 0.0 {
            TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt())
        } else {
            (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c)
        };
        dltlb = 0.0;
        dltub = midpt;
    } else {
        orgati = false;
        let a = c * del - zz(i) * zz(i) - zz(ip1) * zz(ip1);
        let b = zz(ip1) * zz(ip1) * del;
        tau = if a < 0.0 {
            TWO * b / (a - (a * a + FOUR * b * c).abs().sqrt())
        } else {
            -(a + (a * a + FOUR * b * c).abs().sqrt()) / (TWO * c)
        };
        dltlb = -midpt;
        dltub = 0.0;
    }
    if orgati {
        for j in 1..=n {
            delta[j - 1] = (dd(j) - dd(i)) - tau;
        }
    } else {
        for j in 1..=n {
            delta[j - 1] = (dd(j) - dd(ip1)) - tau;
        }
    }
    let ii = if orgati { i } else { i + 1 };
    let iim1 = ii - 1;
    let iip1 = ii + 1;
    let (mut dpsi, mut erretm) = (0.0f64, 0.0f64);
    psi = 0.0;
    for j in 1..=iim1 {
        let temp = zz(j) / delta[j - 1];
        psi += zz(j) * temp;
        dpsi += temp * temp;
        erretm += psi;
    }
    erretm = erretm.abs();
    let mut dphi = 0.0f64;
    phi = 0.0;
    for j in (iip1..=n).rev() {
        let temp = zz(j) / delta[j - 1];
        phi += zz(j) * temp;
        dphi += temp * temp;
        erretm += phi;
    }
    let mut w = rhoinv + phi + psi;
    let mut swtch3 = if orgati { w < 0.0 } else { w > 0.0 };
    if ii == 1 || ii == n {
        swtch3 = false;
    }
    let temp = zz(ii) / delta[ii - 1];
    let mut dw = dpsi + dphi + temp * temp;
    let temp = zz(ii) * temp;
    w += temp;
    erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + tau.abs() * dw;
    let finish = |tau: f64| if orgati { dd(i) + tau } else { dd(ip1) + tau };
    if w.abs() <= eps * erretm {
        return (finish(tau), 0);
    }
    if w <= 0.0 {
        dltlb = dltlb.max(tau);
    } else {
        dltub = dltub.min(tau);
    }
    niter += 1;
    let mut eta;
    if !swtch3 {
        let c = if orgati {
            w - delta[ip1 - 1] * dw
                - (dd(i) - dd(ip1)) * ((zz(i) / delta[i - 1]) * (zz(i) / delta[i - 1]))
        } else {
            w - delta[i - 1] * dw
                - (dd(ip1) - dd(i)) * ((zz(ip1) / delta[ip1 - 1]) * (zz(ip1) / delta[ip1 - 1]))
        };
        let mut a = (delta[i - 1] + delta[ip1 - 1]) * w - delta[i - 1] * delta[ip1 - 1] * dw;
        let b = delta[i - 1] * delta[ip1 - 1] * w;
        if c == 0.0 {
            if a == 0.0 {
                a = if orgati {
                    zz(i) * zz(i) + delta[ip1 - 1] * delta[ip1 - 1] * (dpsi + dphi)
                } else {
                    zz(ip1) * zz(ip1) + delta[i - 1] * delta[i - 1] * (dpsi + dphi)
                };
            }
            eta = b / a;
        } else if a <= 0.0 {
            eta = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
        } else {
            eta = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
        }
    } else {
        let temp = rhoinv + psi + phi;
        let (c, z1, z3);
        if orgati {
            let mut temp1 = zz(iim1) / delta[iim1 - 1];
            temp1 *= temp1;
            c = temp - delta[iip1 - 1] * (dpsi + dphi) - (dd(iim1) - dd(iip1)) * temp1;
            z1 = zz(iim1) * zz(iim1);
            z3 = delta[iip1 - 1] * delta[iip1 - 1] * ((dpsi - temp1) + dphi);
        } else {
            let mut temp1 = zz(iip1) / delta[iip1 - 1];
            temp1 *= temp1;
            c = temp - delta[iim1 - 1] * (dpsi + dphi) - (dd(iip1) - dd(iim1)) * temp1;
            z1 = delta[iim1 - 1] * delta[iim1 - 1] * (dpsi + (dphi - temp1));
            z3 = zz(iip1) * zz(iip1);
        }
        let zzv = [z1, zz(ii) * zz(ii), z3];
        let dv = [delta[iim1 - 1], delta[ii - 1], delta[iip1 - 1]];
        let (e, info) = dlaed6(niter, orgati, c, &dv, &zzv, w);
        eta = e;
        if info != 0 {
            return (finish(tau), info as i64);
        }
    }
    if w * eta >= 0.0 {
        eta = -w / dw;
    }
    let temp = tau + eta;
    if temp > dltub || temp < dltlb {
        eta = if w < 0.0 {
            (dltub - tau) / TWO
        } else {
            (dltlb - tau) / TWO
        };
    }
    let mut prew = w;
    for j in 1..=n {
        delta[j - 1] -= eta;
    }
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for j in 1..=iim1 {
        let temp = zz(j) / delta[j - 1];
        psi += zz(j) * temp;
        dpsi += temp * temp;
        erretm += psi;
    }
    erretm = erretm.abs();
    dphi = 0.0;
    phi = 0.0;
    for j in (iip1..=n).rev() {
        let temp = zz(j) / delta[j - 1];
        phi += zz(j) * temp;
        dphi += temp * temp;
        erretm += phi;
    }
    let temp = zz(ii) / delta[ii - 1];
    dw = dpsi + dphi + temp * temp;
    let temp = zz(ii) * temp;
    w = rhoinv + phi + psi + temp;
    erretm =
        EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + (tau + eta).abs() * dw;
    let mut swtch = if orgati {
        -w > prew.abs() / TEN
    } else {
        w > prew.abs() / TEN
    };
    tau += eta;
    let iter = niter + 1;
    for niter in iter..=MAXIT {
        if w.abs() <= eps * erretm {
            return (finish(tau), 0);
        }
        if w <= 0.0 {
            dltlb = dltlb.max(tau);
        } else {
            dltub = dltub.min(tau);
        }
        if !swtch3 {
            let c = if !swtch {
                if orgati {
                    w - delta[ip1 - 1] * dw
                        - (dd(i) - dd(ip1)) * ((zz(i) / delta[i - 1]) * (zz(i) / delta[i - 1]))
                } else {
                    w - delta[i - 1] * dw
                        - (dd(ip1) - dd(i))
                            * ((zz(ip1) / delta[ip1 - 1]) * (zz(ip1) / delta[ip1 - 1]))
                }
            } else {
                let temp = zz(ii) / delta[ii - 1];
                if orgati {
                    dpsi += temp * temp;
                } else {
                    dphi += temp * temp;
                }
                w - delta[i - 1] * dpsi - delta[ip1 - 1] * dphi
            };
            let mut a = (delta[i - 1] + delta[ip1 - 1]) * w - delta[i - 1] * delta[ip1 - 1] * dw;
            let b = delta[i - 1] * delta[ip1 - 1] * w;
            if c == 0.0 {
                if a == 0.0 {
                    a = if !swtch {
                        if orgati {
                            zz(i) * zz(i) + delta[ip1 - 1] * delta[ip1 - 1] * (dpsi + dphi)
                        } else {
                            zz(ip1) * zz(ip1) + delta[i - 1] * delta[i - 1] * (dpsi + dphi)
                        }
                    } else {
                        delta[i - 1] * delta[i - 1] * dpsi + delta[ip1 - 1] * delta[ip1 - 1] * dphi
                    };
                }
                eta = b / a;
            } else if a <= 0.0 {
                eta = (a - (a * a - FOUR * b * c).abs().sqrt()) / (TWO * c);
            } else {
                eta = TWO * b / (a + (a * a - FOUR * b * c).abs().sqrt());
            }
        } else {
            let temp = rhoinv + psi + phi;
            let (c, z1, z3);
            if swtch {
                c = temp - delta[iim1 - 1] * dpsi - delta[iip1 - 1] * dphi;
                z1 = delta[iim1 - 1] * delta[iim1 - 1] * dpsi;
                z3 = delta[iip1 - 1] * delta[iip1 - 1] * dphi;
            } else if orgati {
                let mut temp1 = zz(iim1) / delta[iim1 - 1];
                temp1 *= temp1;
                c = temp - delta[iip1 - 1] * (dpsi + dphi) - (dd(iim1) - dd(iip1)) * temp1;
                z1 = zz(iim1) * zz(iim1);
                z3 = delta[iip1 - 1] * delta[iip1 - 1] * ((dpsi - temp1) + dphi);
            } else {
                let mut temp1 = zz(iip1) / delta[iip1 - 1];
                temp1 *= temp1;
                c = temp - delta[iim1 - 1] * (dpsi + dphi) - (dd(iip1) - dd(iim1)) * temp1;
                z1 = delta[iim1 - 1] * delta[iim1 - 1] * (dpsi + (dphi - temp1));
                z3 = zz(iip1) * zz(iip1);
            }
            let zzv = [z1, zz(ii) * zz(ii), z3];
            let dv = [delta[iim1 - 1], delta[ii - 1], delta[iip1 - 1]];
            let (e, info) = dlaed6(niter, orgati, c, &dv, &zzv, w);
            eta = e;
            if info != 0 {
                return (finish(tau), info as i64);
            }
        }
        if w * eta >= 0.0 {
            eta = -w / dw;
        }
        let temp = tau + eta;
        if temp > dltub || temp < dltlb {
            eta = if w < 0.0 {
                (dltub - tau) / TWO
            } else {
                (dltlb - tau) / TWO
            };
        }
        for j in 1..=n {
            delta[j - 1] -= eta;
        }
        tau += eta;
        prew = w;
        dpsi = 0.0;
        psi = 0.0;
        erretm = 0.0;
        for j in 1..=iim1 {
            let temp = zz(j) / delta[j - 1];
            psi += zz(j) * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = erretm.abs();
        dphi = 0.0;
        phi = 0.0;
        for j in (iip1..=n).rev() {
            let temp = zz(j) / delta[j - 1];
            phi += zz(j) * temp;
            dphi += temp * temp;
            erretm += phi;
        }
        let temp = zz(ii) / delta[ii - 1];
        dw = dpsi + dphi + temp * temp;
        let temp = zz(ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv + THREE * temp.abs() + tau.abs() * dw;
        if w * prew > 0.0 && w.abs() > prew.abs() / TEN {
            swtch = !swtch;
        }
    }
    dlam = finish(tau);
    (dlam, 1)
}

/// `DLAED5(i, d, z, rho) → (dlam, delta)`: the 2 × 2 secular equation.
fn dlaed5(i: usize, d: [f64; 2], z: [f64; 2], rho: f64) -> (f64, [f64; 2]) {
    const TWO: f64 = 2.0;
    const FOUR: f64 = 4.0;
    let del = d[1] - d[0];
    let mut delta = [0.0f64; 2];
    let dlam;
    if i == 1 {
        let w = 1.0 + TWO * rho * (z[1] * z[1] - z[0] * z[0]) / del;
        if w > 0.0 {
            let b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[0] * z[0] * del;
            let tau = TWO * c / (b + (b * b - FOUR * c).abs().sqrt());
            dlam = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        } else {
            let b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[1] * z[1] * del;
            let tau = if b > 0.0 {
                -TWO * c / (b + (b * b + FOUR * c).sqrt())
            } else {
                (b - (b * b + FOUR * c).sqrt()) / TWO
            };
            dlam = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }
        let temp = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        delta[0] /= temp;
        delta[1] /= temp;
    } else {
        let b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        let c = rho * z[1] * z[1] * del;
        let tau = if b > 0.0 {
            (b + (b * b + FOUR * c).sqrt()) / TWO
        } else {
            TWO * c / (-b + (b * b + FOUR * c).sqrt())
        };
        dlam = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
        let temp = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        delta[0] /= temp;
        delta[1] /= temp;
    }
    (dlam, delta)
}

/// `DSTEVD('V', n, d, e)` — `scipy.linalg.lapack.dstevd`: scale into range, then
/// `DSTEDC('I')` — for checking the divide and conquer apart from the reduction.
#[doc(hidden)]
pub fn dstevd_v(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i64> {
    let n = d.len();
    let (mut d, mut e) = (d.to_vec(), e.to_vec());
    e.resize(n.max(1), 0.0);
    let mut z = vec![0.0f64; n * n];
    if n == 0 {
        return Ok((d, z));
    }
    if n == 1 {
        z[0] = 1.0;
        return Ok((d, z));
    }
    let safmin = SFMIN;
    let eps = PREC;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();
    let tnrm = dlanst_max(n, &d, &e);
    let mut sigma = 1.0;
    let mut iscale = false;
    if tnrm > 0.0 && tnrm < rmin {
        iscale = true;
        sigma = rmin / tnrm;
    } else if tnrm > rmax {
        iscale = true;
        sigma = rmax / tnrm;
    }
    if iscale {
        dscal(n, sigma, &mut d, 1);
        dscal(n - 1, sigma, &mut e, 1);
    }
    let info = dstedc_i(n, &mut d, &mut e, &mut z, n);
    if info != 0 {
        return Err(info);
    }
    if iscale {
        dscal(n, 1.0 / sigma, &mut d, 1);
    }
    Ok((d, z))
}
