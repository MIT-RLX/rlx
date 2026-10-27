// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference LAPACK 3.12's MRRR tridiagonal eigensolver `DSTEMR` (`JOBZ = 'V'`,
//! `RANGE = 'A'`, the path `DSYEVR` takes): `DLARRE` (root representations,
//! dqds eigenvalues), `DLARRV` (the representation tree, `DLAR1V` twisted
//! factorisations), `DLARRJ` (bisection refinement when `TRYRAC`). Translated
//! statement for statement from the Fortran (BSD-3-Clause); `LOG` is glibc's.
//!
//! Index convention: a routine's array arguments are slices whose element `0` is
//! the Fortran element `1` (`X(I)` = `x[i - 1]`); local work arrays are 1-based
//! (`w[k]` = `WORK(K)`, element 0 unused).

use crate::lapack::{PREC as EPS, SFMIN};
use crate::lapack_dqds::dlasq2;

fn ln(x: f64) -> f64 {
    rlx_libm::log(x)
}

#[inline]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline]
fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `DLARUV`'s multiplier table `MM(128, 4)`.
pub(crate) const DLARUV_MM: [[i64; 4]; 128] = [
    [494, 322, 2508, 2549],
    [2637, 789, 3754, 1145],
    [255, 1440, 1766, 2253],
    [2008, 752, 3572, 305],
    [1253, 2859, 2893, 3301],
    [3344, 123, 307, 1065],
    [4084, 1848, 1297, 3133],
    [1739, 643, 3966, 2913],
    [3143, 2405, 758, 3285],
    [3468, 2638, 2598, 1241],
    [688, 2344, 3406, 1197],
    [1657, 46, 2922, 3729],
    [1238, 3814, 1038, 2501],
    [3166, 913, 2934, 1673],
    [1292, 3649, 2091, 541],
    [3422, 339, 2451, 2753],
    [1270, 3808, 1580, 949],
    [2016, 822, 1958, 2361],
    [154, 2832, 2055, 1165],
    [2862, 3078, 1507, 4081],
    [697, 3633, 1078, 2725],
    [1706, 2970, 3273, 3305],
    [491, 637, 17, 3069],
    [931, 2249, 854, 3617],
    [1444, 2081, 2916, 3733],
    [444, 4019, 3971, 409],
    [3577, 1478, 2889, 2157],
    [3944, 242, 3831, 1361],
    [2184, 481, 2621, 3973],
    [1661, 2075, 1541, 1865],
    [3482, 4058, 893, 2525],
    [657, 622, 736, 1409],
    [3023, 3376, 3992, 3445],
    [3618, 812, 787, 3577],
    [1267, 234, 2125, 77],
    [1828, 641, 2364, 3761],
    [164, 4005, 2460, 2149],
    [3798, 1122, 257, 1449],
    [3087, 3135, 1574, 3005],
    [2400, 2640, 3912, 225],
    [2870, 2302, 1216, 85],
    [3876, 40, 3248, 3673],
    [1905, 1832, 3401, 3117],
    [1593, 2247, 2124, 3089],
    [1797, 2034, 2762, 1349],
    [1234, 2637, 149, 2057],
    [3460, 1287, 2245, 413],
    [328, 1691, 166, 65],
    [2861, 496, 466, 1845],
    [1950, 1597, 4018, 697],
    [617, 2394, 1399, 3085],
    [2070, 2584, 190, 3441],
    [3331, 1843, 2879, 1573],
    [769, 336, 153, 3689],
    [1558, 1472, 2320, 2941],
    [2412, 2407, 18, 929],
    [2800, 433, 712, 533],
    [189, 2096, 2159, 2841],
    [287, 1761, 2318, 4077],
    [2045, 2810, 2091, 721],
    [1227, 566, 3443, 2821],
    [2838, 442, 1510, 2249],
    [209, 41, 449, 2397],
    [2770, 1238, 1956, 2817],
    [3654, 1086, 2201, 245],
    [3993, 603, 3137, 1913],
    [192, 840, 3399, 1997],
    [2253, 3168, 1321, 3121],
    [3491, 1499, 2271, 997],
    [2889, 1084, 3667, 1833],
    [2857, 3438, 2703, 2877],
    [2094, 2408, 629, 1633],
    [1818, 1589, 2365, 981],
    [688, 2391, 2431, 2009],
    [1407, 288, 1113, 941],
    [634, 26, 3922, 2449],
    [3231, 512, 2554, 197],
    [815, 1456, 184, 2441],
    [3524, 171, 2099, 285],
    [1914, 1677, 3228, 1473],
    [516, 2657, 4012, 2741],
    [164, 2270, 1921, 3129],
    [303, 2587, 3452, 909],
    [2144, 2961, 3901, 2801],
    [3480, 1970, 572, 421],
    [119, 1817, 3309, 4073],
    [3357, 676, 3171, 2813],
    [837, 1410, 817, 2337],
    [2826, 3723, 3039, 1429],
    [2332, 2803, 1696, 1177],
    [2089, 3185, 1256, 1901],
    [3780, 184, 3715, 81],
    [1700, 663, 2077, 1669],
    [3712, 499, 3019, 2633],
    [150, 3784, 1497, 2269],
    [2000, 1631, 1101, 129],
    [3375, 1925, 717, 1141],
    [1621, 3912, 51, 249],
    [3090, 1398, 981, 3917],
    [3765, 1349, 1978, 2481],
    [1149, 1441, 1813, 3941],
    [3146, 2224, 3881, 2217],
    [33, 2411, 76, 2749],
    [3082, 1907, 3846, 3041],
    [2741, 3192, 3694, 1877],
    [359, 2786, 1682, 345],
    [3316, 382, 124, 2861],
    [1749, 37, 1660, 1809],
    [185, 759, 3997, 3141],
    [2784, 2948, 479, 2825],
    [2202, 1862, 1141, 157],
    [2199, 3802, 886, 2881],
    [1364, 2423, 3514, 3637],
    [1244, 2051, 1301, 1465],
    [2020, 2295, 3604, 2829],
    [3160, 1332, 1888, 2161],
    [2785, 1832, 1836, 3365],
    [2772, 2405, 1990, 361],
    [1217, 3638, 2058, 2685],
    [1822, 3661, 692, 3745],
    [1245, 327, 1194, 2325],
    [2252, 3660, 20, 3609],
    [3904, 716, 3285, 3821],
    [2774, 1842, 2046, 3537],
    [997, 3987, 2107, 517],
    [2573, 1368, 3508, 3017],
    [1148, 1848, 3525, 2141],
    [545, 2366, 3801, 1537],
];

/// `DLARUV(ISEED, N, X)`: `n ≤ 128` uniform (0, 1) numbers, the seed advanced.
fn dlaruv(iseed: &mut [i64; 4], n: usize, x: &mut [f64]) {
    const IPW2: i64 = 4096;
    const R: f64 = 1.0 / IPW2 as f64;
    if n < 1 {
        return;
    }
    let (mut i1, mut i2, mut i3, mut i4) = (iseed[0], iseed[1], iseed[2], iseed[3]);
    let (mut it1, mut it2, mut it3, mut it4) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n.min(128) {
        let mm = DLARUV_MM[i];
        loop {
            it4 = i4 * mm[3];
            it3 = it4 / IPW2;
            it4 -= IPW2 * it3;
            it3 = it3 + i3 * mm[3] + i4 * mm[2];
            it2 = it3 / IPW2;
            it3 -= IPW2 * it2;
            it2 = it2 + i2 * mm[3] + i3 * mm[2] + i4 * mm[1];
            it1 = it2 / IPW2;
            it2 -= IPW2 * it1;
            it1 = it1 + i1 * mm[3] + i2 * mm[2] + i3 * mm[1] + i4 * mm[0];
            it1 %= IPW2;
            x[i] = R * (it1 as f64 + R * (it2 as f64 + R * (it3 as f64 + R * it4 as f64)));
            if x[i] == 1.0 {
                i1 += 2;
                i2 += 2;
                i3 += 2;
                i4 += 2;
                continue;
            }
            break;
        }
    }
    *iseed = [it1, it2, it3, it4];
}

/// `DLARNV(2, ISEED, N, X)`: uniform (−1, 1).
fn dlarnv_2(iseed: &mut [i64; 4], n: usize, x: &mut [f64]) {
    let mut u = [0.0f64; 128];
    let mut iv = 0;
    while iv < n {
        let il = 64.min(n - iv);
        dlaruv(iseed, il, &mut u);
        for i in 0..il {
            x[iv + i] = 2.0 * u[i] - 1.0;
        }
        iv += 64;
    }
}

/// `DLARRR`: whether the tridiagonal warrants relative accuracy (`INFO = 0`).
fn dlarrr(n: usize, d: &[f64], e: &[f64]) -> bool {
    const RELCOND: f64 = 0.999;
    if n == 0 {
        return true;
    }
    let smlnum = SFMIN / EPS;
    let rmin = smlnum.sqrt();
    let mut offdig = 0.0f64;
    let mut tmp = d[0].abs().sqrt();
    if tmp < rmin {
        return false;
    }
    for i in 2..=n {
        let tmp2 = d[i - 1].abs().sqrt();
        if tmp2 < rmin {
            return false;
        }
        let offdig2 = e[i - 2].abs() / (tmp * tmp2);
        if offdig + offdig2 >= RELCOND {
            return false;
        }
        tmp = tmp2;
        offdig = offdig2;
    }
    true
}

/// `DLARRA`: split where the off-diagonal is negligible. Returns `ISPLIT(1..NSPLIT)`.
fn dlarra(
    n: usize,
    d: &[f64],
    e: &mut [f64],
    e2: &mut [f64],
    spltol: f64,
    tnrm: f64,
) -> Vec<usize> {
    let mut isplit = vec![];
    if n == 0 {
        return isplit;
    }
    if spltol < 0.0 {
        let tmp1 = spltol.abs() * tnrm;
        for i in 1..n {
            if e[i - 1].abs() <= tmp1 {
                e[i - 1] = 0.0;
                e2[i - 1] = 0.0;
                isplit.push(i);
            }
        }
    } else {
        for i in 1..n {
            if e[i - 1].abs() <= spltol * d[i - 1].abs().sqrt() * d[i].abs().sqrt() {
                e[i - 1] = 0.0;
                e2[i - 1] = 0.0;
                isplit.push(i);
            }
        }
    }
    isplit.push(n);
    isplit
}

/// `DLARRK`: the `iw`-th eigenvalue by bisection. Returns `(w, werr, info)`.
#[allow(clippy::too_many_arguments)]
fn dlarrk(
    n: usize,
    iw: usize,
    gl: f64,
    gu: f64,
    d: &[f64],
    e2: &[f64],
    pivmin: f64,
    reltol: f64,
) -> (f64, f64, i32) {
    const FUDGE: f64 = 2.0;
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let tnorm = fmax(gl.abs(), gu.abs());
    let rtoli = reltol;
    let atoli = FUDGE * 2.0 * pivmin;
    let itmax = ((ln(tnorm + pivmin) - ln(pivmin)) / ln(2.0)) as i64 + 2;
    let mut info = -1;
    let mut left = gl - FUDGE * tnorm * EPS * n as f64 - FUDGE * 2.0 * pivmin;
    let mut right = gu + FUDGE * tnorm * EPS * n as f64 + FUDGE * 2.0 * pivmin;
    let mut it = 0i64;
    loop {
        let tmp1 = (right - left).abs();
        let tmp2 = fmax(right.abs(), left.abs());
        if tmp1 < fmax(fmax(atoli, pivmin), rtoli * tmp2) {
            info = 0;
            break;
        }
        if it > itmax {
            break;
        }
        it += 1;
        let mid = 0.5 * (left + right);
        let mut negcnt = 0usize;
        let mut t = d[0] - mid;
        if t.abs() < pivmin {
            t = -pivmin;
        }
        if t <= 0.0 {
            negcnt += 1;
        }
        for i in 2..=n {
            t = d[i - 1] - e2[i - 2] / t - mid;
            if t.abs() < pivmin {
                t = -pivmin;
            }
            if t <= 0.0 {
                negcnt += 1;
            }
        }
        if negcnt >= iw {
            right = mid;
        } else {
            left = mid;
        }
    }
    (0.5 * (left + right), 0.5 * (right - left).abs(), info)
}

/// `DLARRC('T', ...)`: Sturm counts at `vl`, `vu`. Returns `(eigcnt, lcnt, rcnt)`.
fn dlarrc_t(n: usize, vl: f64, vu: f64, d: &[f64], e: &[f64]) -> (i64, i64, i64) {
    let (mut lcnt, mut rcnt) = (0i64, 0i64);
    if n == 0 {
        return (0, 0, 0);
    }
    let mut lpivot = d[0] - vl;
    let mut rpivot = d[0] - vu;
    if lpivot <= 0.0 {
        lcnt += 1;
    }
    if rpivot <= 0.0 {
        rcnt += 1;
    }
    for i in 1..n {
        let tmp = e[i - 1] * e[i - 1];
        lpivot = (d[i] - vl) - tmp / lpivot;
        rpivot = (d[i] - vu) - tmp / rpivot;
        if lpivot <= 0.0 {
            lcnt += 1;
        }
        if rpivot <= 0.0 {
            rcnt += 1;
        }
    }
    (rcnt - lcnt, lcnt, rcnt)
}

/// `DLANEG`: the Sturm count of `L D Lᵀ − σ` through the twisted factorisation
/// at `r` (blocks of 128, NaN-safe re-run).
fn dlaneg(n: usize, d: &[f64], lld: &[f64], sigma: f64, r: usize) -> usize {
    const BLKLEN: usize = 128;
    let mut negcnt = 0usize;
    let mut t = -sigma;
    let mut bj = 1;
    while bj < r {
        let mut neg1 = 0;
        let bsav = t;
        let hi = (bj + BLKLEN - 1).min(r - 1);
        for j in bj..=hi {
            let dplus = d[j - 1] + t;
            if dplus < 0.0 {
                neg1 += 1;
            }
            let tmp = t / dplus;
            t = tmp * lld[j - 1] - sigma;
        }
        if t.is_nan() {
            neg1 = 0;
            t = bsav;
            for j in bj..=hi {
                let dplus = d[j - 1] + t;
                if dplus < 0.0 {
                    neg1 += 1;
                }
                let mut tmp = t / dplus;
                if tmp.is_nan() {
                    tmp = 1.0;
                }
                t = tmp * lld[j - 1] - sigma;
            }
        }
        negcnt += neg1;
        bj += BLKLEN;
    }
    let mut p = d[n - 1] - sigma;
    let mut bj = n as isize - 1;
    while bj >= r as isize {
        let mut neg2 = 0;
        let bsav = p;
        let lo = (bj - BLKLEN as isize + 1).max(r as isize);
        let mut j = bj;
        while j >= lo {
            let ju = j as usize;
            let dminus = lld[ju - 1] + p;
            if dminus < 0.0 {
                neg2 += 1;
            }
            let tmp = p / dminus;
            p = tmp * d[ju - 1] - sigma;
            j -= 1;
        }
        if p.is_nan() {
            neg2 = 0;
            p = bsav;
            let mut j = bj;
            while j >= lo {
                let ju = j as usize;
                let dminus = lld[ju - 1] + p;
                if dminus < 0.0 {
                    neg2 += 1;
                }
                let mut tmp = p / dminus;
                if tmp.is_nan() {
                    tmp = 1.0;
                }
                p = tmp * d[ju - 1] - sigma;
                j -= 1;
            }
        }
        negcnt += neg2;
        bj -= BLKLEN as isize;
    }
    let gamma = (t + sigma) + p;
    if gamma < 0.0 {
        negcnt += 1;
    }
    negcnt
}

/// `DLARRB`: refine eigenvalues `ifirst..=ilast` of `L D Lᵀ` by bisection.
/// `w`, `wgap`, `werr` are indexed `I − OFFSET` (element 0 = index 1).
#[allow(clippy::too_many_arguments)]
fn dlarrb(
    n: usize,
    d: &[f64],
    lld: &[f64],
    ifirst: usize,
    ilast: usize,
    rtol1: f64,
    rtol2: f64,
    offset: usize,
    w: &mut [f64],
    wgap: &mut [f64],
    werr: &mut [f64],
    pivmin: f64,
    spdiam: f64,
    twist: usize,
) {
    if n == 0 {
        return;
    }
    let mut work = vec![0.0f64; 2 * n + 2];
    let mut iwork = vec![0i64; 2 * n + 2];
    let maxitr = ((ln(spdiam + pivmin) - ln(pivmin)) / ln(2.0)) as i64 + 2;
    let mnwdth = 2.0 * pivmin;
    let r = if twist < 1 || twist > n { n } else { twist };
    let mut i1 = ifirst;
    let mut nint = 0i64;
    let mut prev = 0usize;
    let mut rgap = wgap[i1 - offset - 1];
    for i in i1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = w[ii - 1] - werr[ii - 1];
        let mut right = w[ii - 1] + werr[ii - 1];
        let lgap = rgap;
        rgap = wgap[ii - 1];
        let gap = fmin(lgap, rgap);
        let mut back = werr[ii - 1];
        loop {
            let negcnt = dlaneg(n, d, lld, left, r);
            if negcnt > i - 1 {
                left -= back;
                back *= 2.0;
            } else {
                break;
            }
        }
        back = werr[ii - 1];
        let mut negcnt;
        loop {
            negcnt = dlaneg(n, d, lld, right, r);
            if negcnt < i {
                right += back;
                back *= 2.0;
            } else {
                break;
            }
        }
        let width = 0.5 * (left - right).abs();
        let tmp = fmax(left.abs(), right.abs());
        let cvrgd = fmax(rtol1 * gap, rtol2 * tmp);
        if width <= cvrgd || width <= mnwdth {
            iwork[k - 1] = -1;
            if i == i1 && i < ilast {
                i1 = i + 1;
            }
            if prev >= i1 && i <= ilast {
                iwork[2 * prev - 1] = i as i64 + 1;
            }
        } else {
            prev = i;
            nint += 1;
            iwork[k - 1] = i as i64 + 1;
            iwork[k] = negcnt as i64;
        }
        work[k - 1] = left;
        work[k] = right;
    }
    let mut iter = 0i64;
    loop {
        let mut prev = i1 as i64 - 1;
        let mut i = i1;
        let olnint = nint;
        for _ip in 1..=olnint {
            let k = 2 * i;
            let ii = i - offset;
            let rgap = wgap[ii - 1];
            let mut lgap = rgap;
            if ii > 1 {
                lgap = wgap[ii - 2];
            }
            let gap = fmin(lgap, rgap);
            let next = iwork[k - 1] as usize;
            let left = work[k - 1];
            let right = work[k];
            let mid = 0.5 * (left + right);
            let width = right - mid;
            let tmp = fmax(left.abs(), right.abs());
            let cvrgd = fmax(rtol1 * gap, rtol2 * tmp);
            if width <= cvrgd || width <= mnwdth || iter == maxitr {
                nint -= 1;
                iwork[k - 1] = 0;
                if i1 == i {
                    i1 = next;
                } else if prev >= i1 as i64 {
                    iwork[2 * prev as usize - 1] = next as i64;
                }
                i = next;
                continue;
            }
            prev = i as i64;
            let negcnt = dlaneg(n, d, lld, mid, r);
            if negcnt < i {
                work[k - 1] = mid;
            } else {
                work[k] = mid;
            }
            i = next;
        }
        iter += 1;
        if !(nint > 0 && iter <= maxitr) {
            break;
        }
    }
    for i in ifirst..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        if iwork[k - 1] == 0 {
            w[ii - 1] = 0.5 * (work[k - 1] + work[k]);
            werr[ii - 1] = work[k] - w[ii - 1];
        }
    }
    for i in ifirst + 1..=ilast {
        let ii = i - offset;
        wgap[ii - 2] = fmax(0.0, w[ii - 1] - werr[ii - 1] - w[ii - 2] - werr[ii - 2]);
    }
}

/// `DLARRJ`: refine eigenvalues `ifirst..=ilast` of the tridiagonal `(d, e2)`
/// by bisection.
#[allow(clippy::too_many_arguments)]
fn dlarrj(
    n: usize,
    d: &[f64],
    e2: &[f64],
    ifirst: usize,
    ilast: usize,
    rtol: f64,
    offset: usize,
    w: &mut [f64],
    werr: &mut [f64],
    pivmin: f64,
    spdiam: f64,
) {
    if n == 0 {
        return;
    }
    let mut work = vec![0.0f64; 2 * n + 2];
    let mut iwork = vec![0i64; 2 * n + 2];
    let maxitr = ((ln(spdiam + pivmin) - ln(pivmin)) / ln(2.0)) as i64 + 2;
    let mut i1 = ifirst;
    let i2 = ilast;
    let mut nint = 0i64;
    let mut prev = 0usize;
    let count = |s: f64| {
        let mut cnt = 0usize;
        let mut dplus = d[0] - s;
        if dplus < 0.0 {
            cnt += 1;
        }
        for j in 2..=n {
            dplus = d[j - 1] - s - e2[j - 2] / dplus;
            if dplus < 0.0 {
                cnt += 1;
            }
        }
        cnt
    };
    for i in i1..=i2 {
        let k = 2 * i;
        let ii = i - offset;
        let mut left = w[ii - 1] - werr[ii - 1];
        let mid = w[ii - 1];
        let mut right = w[ii - 1] + werr[ii - 1];
        let width = right - mid;
        let tmp = fmax(left.abs(), right.abs());
        if width < rtol * tmp {
            iwork[k - 1] = -1;
            if i == i1 && i < i2 {
                i1 = i + 1;
            }
            if prev >= i1 && i <= i2 {
                iwork[2 * prev - 1] = i as i64 + 1;
            }
        } else {
            prev = i;
            let mut fac = 1.0f64;
            loop {
                let cnt = count(left);
                if cnt > i - 1 {
                    left -= werr[ii - 1] * fac;
                    fac *= 2.0;
                } else {
                    break;
                }
            }
            fac = 1.0;
            let mut cnt;
            loop {
                cnt = count(right);
                if cnt < i {
                    right += werr[ii - 1] * fac;
                    fac *= 2.0;
                } else {
                    break;
                }
            }
            nint += 1;
            iwork[k - 1] = i as i64 + 1;
            iwork[k] = cnt as i64;
        }
        work[k - 1] = left;
        work[k] = right;
    }
    let savi1 = i1;
    let mut iter = 0i64;
    loop {
        let mut prev = i1 as i64 - 1;
        let mut i = i1;
        let olnint = nint;
        for _p in 1..=olnint {
            let k = 2 * i;
            let next = iwork[k - 1] as usize;
            let left = work[k - 1];
            let right = work[k];
            let mid = 0.5 * (left + right);
            let width = right - mid;
            let tmp = fmax(left.abs(), right.abs());
            if width < rtol * tmp || iter == maxitr {
                nint -= 1;
                iwork[k - 1] = 0;
                if i1 == i {
                    i1 = next;
                } else if prev >= i1 as i64 {
                    iwork[2 * prev as usize - 1] = next as i64;
                }
                i = next;
                continue;
            }
            prev = i as i64;
            let cnt = count(mid);
            if cnt < i {
                work[k - 1] = mid;
            } else {
                work[k] = mid;
            }
            i = next;
        }
        iter += 1;
        if !(nint > 0 && iter <= maxitr) {
            break;
        }
    }
    for i in savi1..=ilast {
        let k = 2 * i;
        let ii = i - offset;
        if iwork[k - 1] == 0 {
            w[ii - 1] = 0.5 * (work[k - 1] + work[k]);
            werr[ii - 1] = work[k] - w[ii - 1];
        }
    }
}

/// `DLAR1V`'s outputs.
struct Twisted {
    negcnt: i64,
    nrminv: f64,
    resid: f64,
    rqcorr: f64,
}

/// `DLAR1V(N, B1, BN, LAMBDA, D, L, LD, LLD, PIVMIN, GAPTOL, Z, WANTNC, …)`: the
/// eigenvector of `L D Lᵀ − λ` from the twisted factorisation at `r` (in/out;
/// 0 = choose), written into `z[b1-1..bn]`; `isuppz` its support.
#[allow(clippy::too_many_arguments)]
fn dlar1v(
    n: usize,
    b1: usize,
    bn: usize,
    lambda: f64,
    d: &[f64],
    l: &[f64],
    ld: &[f64],
    lld: &[f64],
    pivmin: f64,
    gaptol: f64,
    z: &mut [f64],
    wantnc: bool,
    r: &mut usize,
    isuppz: &mut [usize; 2],
) -> Twisted {
    let mut work = vec![0.0f64; 4 * n + 2];
    let (r1, r2) = if *r == 0 { (b1, bn) } else { (*r, *r) };
    let indlpl = 0;
    let indumn = n;
    let inds = 2 * n + 1;
    let indp = 3 * n + 1;
    work[inds + b1 - 1] = if b1 == 1 { 0.0 } else { lld[b1 - 2] };
    let mut neg1 = 0i64;
    let mut s = work[inds + b1 - 1] - lambda;
    for i in b1..r1 {
        let dplus = d[i - 1] + s;
        work[indlpl + i] = ld[i - 1] / dplus;
        if dplus < 0.0 {
            neg1 += 1;
        }
        work[inds + i] = s * work[indlpl + i] * l[i - 1];
        s = work[inds + i] - lambda;
    }
    let mut sawnan1 = s.is_nan();
    if !sawnan1 {
        for i in r1..r2 {
            let dplus = d[i - 1] + s;
            work[indlpl + i] = ld[i - 1] / dplus;
            work[inds + i] = s * work[indlpl + i] * l[i - 1];
            s = work[inds + i] - lambda;
        }
        sawnan1 = s.is_nan();
    }
    if sawnan1 {
        neg1 = 0;
        s = work[inds + b1 - 1] - lambda;
        for i in b1..r1 {
            let mut dplus = d[i - 1] + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            work[indlpl + i] = ld[i - 1] / dplus;
            if dplus < 0.0 {
                neg1 += 1;
            }
            work[inds + i] = s * work[indlpl + i] * l[i - 1];
            if work[indlpl + i] == 0.0 {
                work[inds + i] = lld[i - 1];
            }
            s = work[inds + i] - lambda;
        }
        for i in r1..r2 {
            let mut dplus = d[i - 1] + s;
            if dplus.abs() < pivmin {
                dplus = -pivmin;
            }
            work[indlpl + i] = ld[i - 1] / dplus;
            work[inds + i] = s * work[indlpl + i] * l[i - 1];
            if work[indlpl + i] == 0.0 {
                work[inds + i] = lld[i - 1];
            }
            s = work[inds + i] - lambda;
        }
    }
    let mut neg2 = 0i64;
    work[indp + bn - 1] = d[bn - 1] - lambda;
    for i in (r1..bn).rev() {
        let dminus = lld[i - 1] + work[indp + i];
        let tmp = d[i - 1] / dminus;
        if dminus < 0.0 {
            neg2 += 1;
        }
        work[indumn + i] = l[i - 1] * tmp;
        work[indp + i - 1] = work[indp + i] * tmp - lambda;
    }
    let tmp = work[indp + r1 - 1];
    let sawnan2 = tmp.is_nan();
    if sawnan2 {
        neg2 = 0;
        for i in (r1..bn).rev() {
            let mut dminus = lld[i - 1] + work[indp + i];
            if dminus.abs() < pivmin {
                dminus = -pivmin;
            }
            let tmp = d[i - 1] / dminus;
            if dminus < 0.0 {
                neg2 += 1;
            }
            work[indumn + i] = l[i - 1] * tmp;
            work[indp + i - 1] = work[indp + i] * tmp - lambda;
            if tmp == 0.0 {
                work[indp + i - 1] = d[i - 1] - lambda;
            }
        }
    }
    let mut mingma = work[inds + r1 - 1] + work[indp + r1 - 1];
    if mingma < 0.0 {
        neg1 += 1;
    }
    let negcnt = if wantnc { neg1 + neg2 } else { -1 };
    if mingma.abs() == 0.0 {
        mingma = EPS * work[inds + r1 - 1];
    }
    *r = r1;
    for i in r1..r2 {
        let mut tmp = work[inds + i] + work[indp + i];
        if tmp == 0.0 {
            tmp = EPS * work[inds + i];
        }
        if tmp.abs() <= mingma.abs() {
            mingma = tmp;
            *r = i + 1;
        }
    }
    let r = *r;
    isuppz[0] = b1;
    isuppz[1] = bn;
    z[r - 1] = 1.0;
    let mut ztz = 1.0f64;
    let zz = |z: &[f64], i: usize| z[i - 1];
    if !sawnan1 && !sawnan2 {
        for i in (b1..r).rev() {
            z[i - 1] = -(work[indlpl + i] * zz(z, i + 1));
            if (zz(z, i).abs() + zz(z, i + 1).abs()) * ld[i - 1].abs() < gaptol {
                z[i - 1] = 0.0;
                isuppz[0] = i + 1;
                break;
            }
            ztz += zz(z, i) * zz(z, i);
        }
    } else {
        for i in (b1..r).rev() {
            if zz(z, i + 1) == 0.0 {
                z[i - 1] = -(ld[i] / ld[i - 1]) * zz(z, i + 2);
            } else {
                z[i - 1] = -(work[indlpl + i] * zz(z, i + 1));
            }
            if (zz(z, i).abs() + zz(z, i + 1).abs()) * ld[i - 1].abs() < gaptol {
                z[i - 1] = 0.0;
                isuppz[0] = i + 1;
                break;
            }
            ztz += zz(z, i) * zz(z, i);
        }
    }
    if !sawnan1 && !sawnan2 {
        for i in r..bn {
            z[i] = -(work[indumn + i] * zz(z, i));
            if (zz(z, i).abs() + zz(z, i + 1).abs()) * ld[i - 1].abs() < gaptol {
                z[i] = 0.0;
                isuppz[1] = i;
                break;
            }
            ztz += zz(z, i + 1) * zz(z, i + 1);
        }
    } else {
        for i in r..bn {
            if zz(z, i) == 0.0 {
                z[i] = -(ld[i - 2] / ld[i - 1]) * zz(z, i - 1);
            } else {
                z[i] = -(work[indumn + i] * zz(z, i));
            }
            if (zz(z, i).abs() + zz(z, i + 1).abs()) * ld[i - 1].abs() < gaptol {
                z[i] = 0.0;
                isuppz[1] = i;
                break;
            }
            ztz += zz(z, i + 1) * zz(z, i + 1);
        }
    }
    let tmp = 1.0 / ztz;
    let nrminv = tmp.sqrt();
    Twisted {
        negcnt,
        nrminv,
        resid: mingma.abs() * nrminv,
        rqcorr: mingma * tmp,
    }
}

/// `DLARRF`: a new relatively robust representation `L₊ D₊ L₊ᵀ = L D Lᵀ − σ` for
/// the cluster `clstrt..=clend`. Returns `(σ, D₊, L₊, info)`.
#[allow(clippy::too_many_arguments)]
fn dlarrf(
    n: usize,
    d: &[f64],
    l: &[f64],
    ld: &[f64],
    clstrt: usize,
    clend: usize,
    w: &[f64],
    wgap: &[f64],
    werr: &[f64],
    spdiam: f64,
    clgapl: f64,
    clgapr: f64,
    pivmin: f64,
) -> (f64, Vec<f64>, Vec<f64>, i32) {
    const MAXGROWTH1: f64 = 8.0;
    const MAXGROWTH2: f64 = 8.0;
    const KTRYMAX: i32 = 1;
    let mut dplus = vec![0.0f64; n + 1];
    let mut lplus = vec![0.0f64; n + 1];
    let mut work = vec![0.0f64; 2 * n + 2];
    if n == 0 {
        return (0.0, dplus, lplus, 0);
    }
    let fact = (1i64 << KTRYMAX) as f64;
    let shift;
    let mut forcer = false;
    let nofail = false;
    let clwdth = (w[clend - 1] - w[clstrt - 1]).abs() + werr[clend - 1] + werr[clstrt - 1];
    let avgap = clwdth / (clend - clstrt) as f64;
    let mingap = fmin(clgapl, clgapr);
    let mut lsigma = fmin(w[clstrt - 1], w[clend - 1]) - werr[clstrt - 1];
    let mut rsigma = fmax(w[clstrt - 1], w[clend - 1]) + werr[clend - 1];
    lsigma -= lsigma.abs() * 4.0 * EPS;
    rsigma += rsigma.abs() * 4.0 * EPS;
    let ldmax = 0.25 * mingap + 2.0 * pivmin;
    let rdmax = 0.25 * mingap + 2.0 * pivmin;
    let mut ldelta = fmax(avgap, wgap[clstrt - 1]) / fact;
    let mut rdelta = fmax(avgap, wgap[clend - 2]) / fact;
    let s0 = SFMIN;
    let mut smlgrowth = 1.0 / s0;
    let fail = (n - 1) as f64 * mingap / (spdiam * EPS);
    let fail2 = (n - 1) as f64 * mingap / (spdiam * EPS.sqrt());
    let mut bestshift = lsigma;
    let mut ktry = 0;
    let growthbound = MAXGROWTH1 * spdiam;
    // 1-based views
    let dp = |v: &Vec<f64>, i: usize| v[i];
    let sigma;
    'outer: loop {
        // 5
        let mut sawnan1 = false;
        let mut sawnan2 = false;
        ldelta = fmin(ldmax, ldelta);
        rdelta = fmin(rdmax, rdelta);
        let mut s = -lsigma;
        dplus[1] = d[0] + s;
        if dplus[1].abs() < pivmin {
            dplus[1] = -pivmin;
            sawnan1 = true;
        }
        let mut max1 = dplus[1].abs();
        for i in 1..n {
            lplus[i] = ld[i - 1] / dplus[i];
            s = s * lplus[i] * l[i - 1] - lsigma;
            dplus[i + 1] = d[i] + s;
            if dplus[i + 1].abs() < pivmin {
                dplus[i + 1] = -pivmin;
                sawnan1 = true;
            }
            max1 = fmax(max1, dplus[i + 1].abs());
        }
        sawnan1 = sawnan1 || max1.is_nan();
        if forcer || (max1 <= growthbound && !sawnan1) {
            sigma = lsigma;
            shift = 1;
            break 'outer;
        }
        s = -rsigma;
        work[1] = d[0] + s;
        if work[1].abs() < pivmin {
            work[1] = -pivmin;
            sawnan2 = true;
        }
        let mut max2 = work[1].abs();
        for i in 1..n {
            work[n + i] = ld[i - 1] / work[i];
            s = s * work[n + i] * l[i - 1] - rsigma;
            work[i + 1] = d[i] + s;
            if work[i + 1].abs() < pivmin {
                work[i + 1] = -pivmin;
                sawnan2 = true;
            }
            max2 = fmax(max2, work[i + 1].abs());
        }
        sawnan2 = sawnan2 || max2.is_nan();
        if forcer || (max2 <= growthbound && !sawnan2) {
            sigma = rsigma;
            shift = 2;
            break 'outer;
        }
        let mut goto50 = false;
        let mut indx = 0;
        if sawnan1 && sawnan2 {
            goto50 = true;
        } else {
            if !sawnan1 {
                indx = 1;
                if max1 <= smlgrowth {
                    smlgrowth = max1;
                    bestshift = lsigma;
                }
            }
            if !sawnan2 {
                if sawnan1 || max2 <= max1 {
                    indx = 2;
                }
                if max2 <= smlgrowth {
                    smlgrowth = max2;
                    bestshift = rsigma;
                }
            }
        }
        if !goto50 {
            let dorrr1 =
                clwdth < mingap / 128.0 && fmin(max1, max2) < fail2 && !sawnan1 && !sawnan2;
            if dorrr1 {
                if indx == 1 {
                    let mut tmp = dp(&dplus, n).abs();
                    let mut znm2 = 1.0f64;
                    let mut prod = 1.0f64;
                    let mut oldp = 1.0f64;
                    for i in (1..n).rev() {
                        if prod <= EPS {
                            prod = ((dplus[i + 1] * work[n + i + 1]) / (dplus[i] * work[n + i]))
                                * oldp;
                        } else {
                            prod *= work[n + i].abs();
                        }
                        oldp = prod;
                        znm2 += prod * prod;
                        tmp = fmax(tmp, (dplus[i] * prod).abs());
                    }
                    let rrr1 = tmp / (spdiam * znm2.sqrt());
                    if rrr1 <= MAXGROWTH2 {
                        sigma = lsigma;
                        shift = 1;
                        break 'outer;
                    }
                } else if indx == 2 {
                    let mut tmp = work[n].abs();
                    let mut znm2 = 1.0f64;
                    let mut prod = 1.0f64;
                    let mut oldp = 1.0f64;
                    for i in (1..n).rev() {
                        if prod <= EPS {
                            prod = ((work[i + 1] * lplus[i + 1]) / (work[i] * lplus[i])) * oldp;
                        } else {
                            prod *= lplus[i].abs();
                        }
                        oldp = prod;
                        znm2 += prod * prod;
                        tmp = fmax(tmp, (work[i] * prod).abs());
                    }
                    let rrr2 = tmp / (spdiam * znm2.sqrt());
                    if rrr2 <= MAXGROWTH2 {
                        sigma = rsigma;
                        shift = 2;
                        break 'outer;
                    }
                }
            }
        }
        // 50
        if ktry < KTRYMAX {
            lsigma = fmax(lsigma - ldelta, lsigma - ldmax);
            rsigma = fmin(rsigma + rdelta, rsigma + rdmax);
            ldelta *= 2.0;
            rdelta *= 2.0;
            ktry += 1;
        } else if smlgrowth < fail || nofail {
            lsigma = bestshift;
            rsigma = bestshift;
            forcer = true;
        } else {
            return (0.0, dplus, lplus, 1);
        }
    }
    if shift == 2 {
        dplus[1..=n].copy_from_slice(&work[1..=n]);
        lplus[1..n].copy_from_slice(&work[n + 1..2 * n]);
    }
    // back to element-0 = index-1 slices
    dplus.remove(0);
    lplus.remove(0);
    (sigma, dplus, lplus, 0)
}

/// `DLARRE`'s outputs (1-based vectors, element 0 unused).
struct Roots {
    /// `VL`, `VU` as `RANGE = 'A'` resets them (the Gerschgorin bounds).
    vl: f64,
    vu: f64,
    isplit: Vec<usize>,
    m: usize,
    w: Vec<f64>,
    werr: Vec<f64>,
    wgap: Vec<f64>,
    iblock: Vec<usize>,
    indexw: Vec<usize>,
    gers: Vec<f64>,
    pivmin: f64,
}

/// `DLARRE('A', N, VL, VU, …)`: split, find each block's root representation
/// `L D Lᵀ = T − σ` (stored over `d`, and `e` with `σ` in its last entry) and its
/// eigenvalues by dqds. `d`, `e`, `e2` are 1-based (`e[n]` is workspace).
#[allow(clippy::too_many_arguments)]
fn dlarre_all(
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    e2: &mut [f64],
    spltol: f64,
) -> Result<Roots, i32> {
    const FAC: f64 = 0.5;
    const MAXGROWTH: f64 = 64.0;
    const FUDGE: f64 = 2.0;
    const MAXTRY: usize = 6;
    const PERT: f64 = 8.0;
    const HNDRD: f64 = 100.0;
    let _ = FAC;
    let eps = EPS;
    let safmin = SFMIN;
    let rtl = eps.sqrt();
    let mut out = Roots {
        vl: 0.0,
        vu: 0.0,
        isplit: vec![0],
        m: 0,
        w: vec![0.0; n + 1],
        werr: vec![0.0; n + 1],
        wgap: vec![0.0; n + 1],
        iblock: vec![0; n + 1],
        indexw: vec![0; n + 1],
        gers: vec![0.0; 2 * n + 1],
        pivmin: 0.0,
    };
    assert!(n >= 2, "DLARRE with N = 1 is handled by DSTEMR's callers");
    let mut gl = d[1];
    let mut gu = d[1];
    let mut eold = 0.0f64;
    let mut emax = 0.0f64;
    e[n] = 0.0;
    for i in 1..=n {
        out.werr[i] = 0.0;
        out.wgap[i] = 0.0;
        let eabs = e[i].abs();
        if eabs >= emax {
            emax = eabs;
        }
        let tmp1 = eabs + eold;
        out.gers[2 * i - 1] = d[i] - tmp1;
        gl = fmin(gl, out.gers[2 * i - 1]);
        out.gers[2 * i] = d[i] + tmp1;
        gu = fmax(gu, out.gers[2 * i]);
        eold = eabs;
    }
    let pivmin = safmin * fmax(1.0, emax * emax);
    out.pivmin = pivmin;
    let spdiam0 = gu - gl;
    let isplit = dlarra(n, &d[1..], &mut e[1..], &mut e2[1..], spltol, spdiam0);
    out.isplit.extend_from_slice(&isplit);
    let nsplit = isplit.len();
    let (vl, vu) = (gl, gu);
    out.vl = vl;
    out.vu = vu;
    let mut ibegin = 1usize;
    let mut wbegin = 1usize;
    let mut m = 0usize;
    for jblk in 1..=nsplit {
        let iend = out.isplit[jblk];
        let inn = iend - ibegin + 1;
        if inn == 1 {
            m += 1;
            out.w[m] = d[ibegin];
            out.werr[m] = 0.0;
            out.wgap[m] = 0.0;
            out.iblock[m] = jblk;
            out.indexw[m] = 1;
            wbegin += 1;
            e[iend] = 0.0;
            ibegin = iend + 1;
            continue;
        }
        e[iend] = 0.0;
        let mut gl = d[ibegin];
        let mut gu = d[ibegin];
        for i in ibegin..=iend {
            gl = fmin(out.gers[2 * i - 1], gl);
            gu = fmax(out.gers[2 * i], gu);
        }
        let (tmp, tmp1, info) = dlarrk(inn, 1, gl, gu, &d[ibegin..], &e2[ibegin..], pivmin, rtl);
        if info != 0 {
            return Err(-1);
        }
        let isleft = fmax(gl, tmp - tmp1 - HNDRD * eps * (tmp - tmp1).abs());
        let (tmp, tmp1, info) = dlarrk(inn, inn, gl, gu, &d[ibegin..], &e2[ibegin..], pivmin, rtl);
        if info != 0 {
            return Err(-1);
        }
        let isrght = fmin(gu, tmp + tmp1 + HNDRD * eps * (tmp + tmp1).abs());
        let spdiam = isrght - isleft;
        // RANGE = 'A': the whole block by dqds
        let usedqd = true;
        let indl = 1usize;
        let indu = inn;
        let mb = inn;
        let wend = wbegin + mb - 1;
        let s1 = isleft + 0.25 * spdiam;
        let s2 = isrght - 0.25 * spdiam;
        let (mut cnt1, mut cnt2) = (0i64, 0i64);
        if mb > 1 {
            let (_, c1, c2) = dlarrc_t(inn, s1, s2, &d[ibegin..], &e[ibegin..]);
            cnt1 = c1;
            cnt2 = c2;
        }
        let (mut sigma, sgndef);
        if mb == 1 {
            sigma = gl;
            sgndef = 1.0;
        } else if cnt1 - indl as i64 >= indu as i64 - cnt2 {
            sigma = fmax(isleft, gl);
            sgndef = 1.0;
        } else {
            sigma = fmin(isrght, gu);
            sgndef = -1.0;
        }
        let mut tau = spdiam * eps * n as f64 + 2.0 * pivmin;
        tau = fmax(tau, 2.0 * eps * sigma.abs());
        let mut work = vec![0.0f64; 3 * inn + 1];
        let mut found = false;
        for idum in 1..=MAXTRY {
            let mut dpivot = d[ibegin] - sigma;
            work[1] = dpivot;
            let mut dmax = work[1].abs();
            let mut j = ibegin;
            for i in 1..inn {
                work[2 * inn + i] = 1.0 / work[i];
                let tmp = e[j] * work[2 * inn + i];
                work[inn + i] = tmp;
                dpivot = (d[j + 1] - sigma) - tmp * e[j];
                work[i + 1] = dpivot;
                dmax = fmax(dmax, dpivot.abs());
                j += 1;
            }
            let mut norep = dmax > MAXGROWTH * spdiam;
            if usedqd && !norep {
                for i in 1..=inn {
                    if sgndef * work[i] < 0.0 {
                        norep = true;
                    }
                }
            }
            if norep {
                if idum == MAXTRY - 1 {
                    if sgndef == 1.0 {
                        sigma = gl - FUDGE * spdiam * eps * n as f64 - FUDGE * 2.0 * pivmin;
                    } else {
                        sigma = gu + FUDGE * spdiam * eps * n as f64 + FUDGE * 2.0 * pivmin;
                    }
                } else {
                    sigma -= sgndef * tau;
                    tau *= 2.0;
                }
            } else {
                found = true;
                break;
            }
        }
        if !found {
            return Err(2);
        }
        e[iend] = sigma;
        d[ibegin..ibegin + inn].copy_from_slice(&work[1..=inn]);
        e[ibegin..ibegin + inn - 1].copy_from_slice(&work[inn + 1..2 * inn]);
        if mb > 1 {
            let mut iseed = [1i64; 4];
            let mut r = vec![0.0f64; 2 * inn];
            dlarnv_2(&mut iseed, 2 * inn - 1, &mut r);
            for i in 1..inn {
                d[ibegin + i - 1] *= 1.0 + eps * PERT * r[i - 1];
                e[ibegin + i - 1] *= 1.0 + eps * PERT * r[inn + i - 1];
            }
            d[iend] *= 1.0 + eps * 4.0 * r[inn - 1];
        }
        // dqds on the root representation
        let rtol = ln(inn as f64) * 4.0 * eps;
        let mut z = vec![0.0f64; 4 * inn + 6];
        let mut j = ibegin;
        for i in 1..inn {
            z[2 * i - 1] = d[j].abs();
            z[2 * i] = e[j] * e[j] * z[2 * i - 1];
            j += 1;
        }
        z[2 * inn - 1] = d[iend].abs();
        z[2 * inn] = 0.0;
        let info = dlasq2(inn, &mut z);
        if info != 0 {
            return Err(-5);
        }
        if z[1..=inn].iter().any(|&v| v < 0.0) {
            return Err(-6);
        }
        if sgndef > 0.0 {
            for i in indl..=indu {
                m += 1;
                out.w[m] = z[inn - i + 1];
                out.iblock[m] = jblk;
                out.indexw[m] = i;
            }
        } else {
            for i in indl..=indu {
                m += 1;
                out.w[m] = -z[i];
                out.iblock[m] = jblk;
                out.indexw[m] = i;
            }
        }
        for i in m - mb + 1..=m {
            out.werr[i] = rtol * out.w[i].abs();
        }
        for i in m - mb + 1..m {
            out.wgap[i] = fmax(
                0.0,
                out.w[i + 1] - out.werr[i + 1] - (out.w[i] + out.werr[i]),
            );
        }
        out.wgap[m] = fmax(0.0, (vu - sigma) - (out.w[m] + out.werr[m]));
        ibegin = iend + 1;
        wbegin = wend + 1;
    }
    out.m = m;
    Ok(out)
}

/// `DLARRV(N, VL, VU, D, L, PIVMIN, ISPLIT, M, 1, M, MINRGP, RTOL1, RTOL2, W, WERR,
/// WGAP, IBLOCK, INDEXW, GERS, Z, LDZ, ISUPPZ, …)`: the eigenvectors. All vectors
/// 1-based; `z` column-major `n × m` at `ldz` (0-based storage).
#[allow(clippy::too_many_arguments)]
fn dlarrv(
    n: usize,
    vl: f64,
    d: &mut [f64],
    l: &mut [f64],
    pivmin: f64,
    isplit: &[usize],
    m: usize,
    minrgp: f64,
    rtol1: f64,
    rtol2: f64,
    w: &mut [f64],
    werr: &mut [f64],
    wgap: &mut [f64],
    iblock: &[usize],
    indexw: &[usize],
    gers: &[f64],
    z: &mut [f64],
    ldz: usize,
    isuppz: &mut [usize],
) -> i32 {
    const MAXITR: i64 = 10;
    if n == 0 || m == 0 {
        return 0;
    }
    let zi = |i: usize, j: usize| (i - 1) + (j - 1) * ldz;
    // WORK(1..N): the eigenvalues relative to the current shift; LD, LLD
    let mut wk = vec![0.0f64; n + 1];
    let mut ldv = vec![0.0f64; n + 1];
    let mut lldv = vec![0.0f64; n + 1];
    // IWORK(IINDR+·): twist indices; IINDC1/2: cluster lists
    let mut twist = vec![0usize; n + 1];
    let mut cls = [vec![0usize; 2 * n + 3], vec![0usize; 2 * n + 3]];
    for j in 1..=m {
        for i in 1..=n {
            z[zi(i, j)] = 0.0;
        }
    }
    let eps = EPS;
    let rqtol = 2.0 * eps;
    let tryrqc = true;
    let mut ibegin = 1usize;
    let mut wbegin = 1usize;
    for jblk in 1..=iblock[m] {
        let iend = isplit[jblk];
        let mut sigma = l[iend];
        let mut wend = wbegin - 1;
        while wend < m && iblock[wend + 1] == jblk {
            wend += 1;
        }
        if wend < wbegin {
            ibegin = iend + 1;
            continue;
        }
        let mut gl = gers[2 * ibegin - 1];
        let mut gu = gers[2 * ibegin];
        for i in ibegin + 1..=iend {
            gl = fmin(gers[2 * i - 1], gl);
            gu = fmax(gers[2 * i], gu);
        }
        let spdiam = gu - gl;
        let oldien = ibegin - 1;
        let inn = iend - ibegin + 1;
        let im = wend - wbegin + 1;
        if ibegin == iend {
            z[zi(ibegin, wbegin)] = 1.0;
            isuppz[2 * wbegin - 1] = ibegin;
            isuppz[2 * wbegin] = ibegin;
            w[wbegin] += sigma;
            wk[wbegin] = w[wbegin];
            ibegin = iend + 1;
            wbegin += 1;
            continue;
        }
        wk[wbegin..wbegin + im].copy_from_slice(&w[wbegin..wbegin + im]);
        for i in 1..=im {
            w[wbegin + i - 1] += sigma;
        }
        let mut ndepth = 0usize;
        let mut parity = 1usize;
        let mut nclus = 1usize;
        cls[0][1] = 1;
        cls[0][2] = im;
        let mut idone = 0usize;
        while idone < im {
            if ndepth > m {
                return -2;
            }
            let oldncl = nclus;
            nclus = 0;
            parity = 1 - parity;
            let (oldc, newc) = if parity == 0 { (0, 1) } else { (1, 0) };
            for i in 1..=oldncl {
                let oldfst = cls[oldc][2 * i - 1];
                let oldlst = cls[oldc][2 * i];
                if ndepth > 0 {
                    // the representation stored in Z's columns j, j+1
                    let j = wbegin + oldfst - 1;
                    for r in 0..inn {
                        d[ibegin + r] = z[zi(ibegin + r, j)];
                    }
                    for r in 0..inn - 1 {
                        l[ibegin + r] = z[zi(ibegin + r, j + 1)];
                    }
                    sigma = z[zi(iend, j + 1)];
                    for c in j..=j + 1 {
                        for r in 0..inn {
                            z[zi(ibegin + r, c)] = 0.0;
                        }
                    }
                }
                for j in ibegin..iend {
                    let tmp = d[j] * l[j];
                    ldv[j] = tmp;
                    lldv[j] = tmp * l[j];
                }
                if ndepth > 0 {
                    let p = indexw[wbegin - 1 + oldfst];
                    let q = indexw[wbegin - 1 + oldlst];
                    let offset = indexw[wbegin] - 1;
                    dlarrb(
                        inn,
                        &d[ibegin..],
                        &lldv[ibegin..],
                        p,
                        q,
                        rtol1,
                        rtol2,
                        offset,
                        &mut wk[wbegin..],
                        &mut wgap[wbegin..],
                        &mut werr[wbegin..],
                        pivmin,
                        spdiam,
                        inn,
                    );
                    if oldfst > 1 {
                        let a = wbegin + oldfst - 2;
                        wgap[a] = fmax(wgap[a], w[a + 1] - werr[a + 1] - w[a] - werr[a]);
                    }
                    if wbegin + oldlst - 1 < wend {
                        let a = wbegin + oldlst - 1;
                        wgap[a] = fmax(wgap[a], w[a + 1] - werr[a + 1] - w[a] - werr[a]);
                    }
                    for j in oldfst..=oldlst {
                        w[wbegin + j - 1] = wk[wbegin + j - 1] + sigma;
                    }
                }
                let mut newfst = oldfst;
                for j in oldfst..=oldlst {
                    let newlst;
                    if j == oldlst {
                        newlst = j;
                    } else if wgap[wbegin + j - 1] >= minrgp * wk[wbegin + j - 1].abs() {
                        newlst = j;
                    } else {
                        continue;
                    }
                    let newsiz = newlst - newfst + 1;
                    let newftt = wbegin + newfst - 1;
                    if newsiz > 1 {
                        let lgap = if newfst == 1 {
                            fmax(0.0, w[wbegin] - werr[wbegin] - vl)
                        } else {
                            wgap[wbegin + newfst - 2]
                        };
                        let rgap = wgap[wbegin + newlst - 1];
                        for k in 1..=2 {
                            let p = if k == 1 {
                                indexw[wbegin - 1 + newfst]
                            } else {
                                indexw[wbegin - 1 + newlst]
                            };
                            let offset = indexw[wbegin] - 1;
                            dlarrb(
                                inn,
                                &d[ibegin..],
                                &lldv[ibegin..],
                                p,
                                p,
                                rqtol,
                                rqtol,
                                offset,
                                &mut wk[wbegin..],
                                &mut wgap[wbegin..],
                                &mut werr[wbegin..],
                                pivmin,
                                spdiam,
                                inn,
                            );
                        }
                        let (tau, dplus, lplus, info) = dlarrf(
                            inn,
                            &d[ibegin..],
                            &l[ibegin..],
                            &ldv[ibegin..],
                            newfst,
                            newlst,
                            &wk[wbegin..],
                            &wgap[wbegin..],
                            &werr[wbegin..],
                            spdiam,
                            lgap,
                            rgap,
                            pivmin,
                        );
                        if info != 0 {
                            return -2;
                        }
                        for r in 0..inn {
                            z[zi(ibegin + r, newftt)] = dplus[r];
                        }
                        for r in 0..inn - 1 {
                            z[zi(ibegin + r, newftt + 1)] = lplus[r];
                        }
                        let ssigma = sigma + tau;
                        z[zi(iend, newftt + 1)] = ssigma;
                        for k in newfst..=newlst {
                            let a = wbegin + k - 1;
                            let mut fudge = 3.0 * eps * wk[a].abs();
                            wk[a] -= tau;
                            fudge += 4.0 * eps * wk[a].abs();
                            werr[a] += fudge;
                        }
                        nclus += 1;
                        cls[newc][2 * nclus - 1] = newfst;
                        cls[newc][2 * nclus] = newlst;
                    } else {
                        // a singleton: Rayleigh-quotient iteration / bisection
                        let mut iter = 0i64;
                        let tol = 4.0 * ln(inn as f64) * eps;
                        let k = newfst;
                        let windex = wbegin + k - 1;
                        let windmn = (windex - 1).max(1);
                        let windpl = (windex + 1).min(m);
                        let mut lambda = wk[windex];
                        let mut left = wk[windex] - werr[windex];
                        let mut right = wk[windex] + werr[windex];
                        let indeig = indexw[windex];
                        let lgap = if k == 1 {
                            eps * fmax(left.abs(), right.abs())
                        } else {
                            wgap[windmn]
                        };
                        let rgap = if k == im {
                            eps * fmax(left.abs(), right.abs())
                        } else {
                            wgap[windex]
                        };
                        let gap = fmin(lgap, rgap);
                        let gaptol = if k == 1 || k == im { 0.0 } else { gap * eps };
                        let mut isupmn = inn;
                        let mut isupmx = 1usize;
                        let savgap = wgap[windex];
                        wgap[windex] = gap;
                        let mut usedbs = false;
                        let mut usedrq = false;
                        let mut needbs = !tryrqc;
                        let (mut bstres, mut bstw) = (0.0f64, 0.0f64);
                        let mut isz = [0usize; 2];
                        let mut tw;
                        let zc = zi(ibegin, windex);
                        loop {
                            // 120
                            if needbs {
                                usedbs = true;
                                let itmp1 = twist[windex];
                                let offset = indexw[wbegin] - 1;
                                dlarrb(
                                    inn,
                                    &d[ibegin..],
                                    &lldv[ibegin..],
                                    indeig,
                                    indeig,
                                    0.0,
                                    2.0 * eps,
                                    offset,
                                    &mut wk[wbegin..],
                                    &mut wgap[wbegin..],
                                    &mut werr[wbegin..],
                                    pivmin,
                                    spdiam,
                                    itmp1,
                                );
                                lambda = wk[windex];
                                twist[windex] = 0;
                            }
                            tw = dlar1v(
                                inn,
                                1,
                                inn,
                                lambda,
                                &d[ibegin..],
                                &l[ibegin..],
                                &ldv[ibegin..],
                                &lldv[ibegin..],
                                pivmin,
                                gaptol,
                                &mut z[zc..zc + inn],
                                !usedbs,
                                &mut twist[windex],
                                &mut isz,
                            );
                            if iter == 0 || tw.resid < bstres {
                                bstres = tw.resid;
                                bstw = lambda;
                            }
                            isupmn = isupmn.min(isz[0]);
                            isupmx = isupmx.max(isz[1]);
                            iter += 1;
                            if tw.resid > tol * gap
                                && tw.rqcorr.abs() > rqtol * lambda.abs()
                                && !usedbs
                            {
                                let sgndef = if indeig as i64 <= tw.negcnt {
                                    -1.0
                                } else {
                                    1.0
                                };
                                if tw.rqcorr * sgndef >= 0.0
                                    && lambda + tw.rqcorr <= right
                                    && lambda + tw.rqcorr >= left
                                {
                                    usedrq = true;
                                    if sgndef == 1.0 {
                                        left = lambda;
                                    } else {
                                        right = lambda;
                                    }
                                    wk[windex] = 0.5 * (right + left);
                                    lambda += tw.rqcorr;
                                    werr[windex] = 0.5 * (right - left);
                                } else {
                                    needbs = true;
                                }
                                if right - left < rqtol * lambda.abs() {
                                    usedbs = true;
                                    continue;
                                } else if iter < MAXITR {
                                    continue;
                                } else if iter == MAXITR {
                                    needbs = true;
                                    continue;
                                } else {
                                    return 5;
                                }
                            } else {
                                if usedrq && usedbs && bstres <= tw.resid {
                                    lambda = bstw;
                                    tw = dlar1v(
                                        inn,
                                        1,
                                        inn,
                                        lambda,
                                        &d[ibegin..],
                                        &l[ibegin..],
                                        &ldv[ibegin..],
                                        &lldv[ibegin..],
                                        pivmin,
                                        gaptol,
                                        &mut z[zc..zc + inn],
                                        !usedbs,
                                        &mut twist[windex],
                                        &mut isz,
                                    );
                                }
                                wk[windex] = lambda;
                                break;
                            }
                        }
                        let zfrom = isz[0] + oldien;
                        let zto = isz[1] + oldien;
                        isuppz[2 * windex - 1] = zfrom;
                        isuppz[2 * windex] = zto;
                        isupmn += oldien;
                        isupmx += oldien;
                        for ii in isupmn..zfrom {
                            z[zi(ii, windex)] = 0.0;
                        }
                        for ii in zto + 1..=isupmx {
                            z[zi(ii, windex)] = 0.0;
                        }
                        crate::openblas::dscal(
                            zto - zfrom + 1,
                            tw.nrminv,
                            &mut z[zi(zfrom, windex)..],
                            1,
                        );
                        // 125
                        w[windex] = lambda + sigma;
                        if k > 1 {
                            wgap[windmn] = fmax(
                                wgap[windmn],
                                w[windex] - werr[windex] - w[windmn] - werr[windmn],
                            );
                        }
                        if windex < wend {
                            wgap[windex] =
                                fmax(savgap, w[windpl] - werr[windpl] - w[windex] - werr[windex]);
                        }
                        idone += 1;
                    }
                    newfst = j + 1;
                }
            }
            ndepth += 1;
        }
        ibegin = iend + 1;
        wbegin = wend + 1;
    }
    0
}

/// `DSTEMR('V', 'A', N, D, E, …, TRYRAC)` as `DSYEVR` calls it: all eigenvalues
/// (ascending) and eigenvectors of the symmetric tridiagonal `(d, e)` (0-based,
/// `e` of length ≥ n, overwritten). `z` is `n × n` column-major at `ldz`.
/// Returns `INFO`.
pub(crate) fn dstemr_all(
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    w: &mut [f64],
    z: &mut [f64],
    ldz: usize,
    tryrac: bool,
) -> i32 {
    const MINRGP: f64 = 1.0e-3;
    let eps = EPS;
    let safmin = SFMIN;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = fmin(bignum.sqrt(), 1.0 / safmin.sqrt().sqrt());
    if n == 0 {
        return 0;
    }
    if n == 1 {
        w[0] = d[0];
        z[0] = 1.0;
        return 0;
    }
    if n == 2 {
        let (r1, r2, cs, sn) = dlaev2(d[0], e[0], d[1]);
        // R1 ≥ R2 always (DLAEV2's RT1 is the larger)
        let (mut r1, mut r2, mut laeswap) = (r1, r2, false);
        if r1 < r2 {
            e[1] = r1;
            r1 = r2;
            r2 = e[1];
            laeswap = true;
        }
        w[0] = r2;
        if laeswap {
            z[0] = cs;
            z[1] = sn;
        } else {
            z[0] = -sn;
            z[1] = cs;
        }
        w[1] = r1;
        if laeswap {
            z[ldz] = -sn;
            z[ldz + 1] = cs;
        } else {
            z[ldz] = cs;
            z[ldz + 1] = sn;
        }
        // NSPLIT.GT.1 .OR. N.EQ.2: the selection sort below
        sort_pairs(n, 2, w, z, ldz);
        return 0;
    }
    // 1-based copies
    let mut d1: Vec<f64> = std::iter::once(0.0).chain(d[..n].iter().copied()).collect();
    let mut e1: Vec<f64> = std::iter::once(0.0).chain(e[..n].iter().copied()).collect();
    let mut scale = 1.0f64;
    let mut tnrm = crate::lapack::dlanst_max(n, d, e);
    if tnrm > 0.0 && tnrm < rmin {
        scale = rmin / tnrm;
    } else if tnrm > rmax {
        scale = rmax / tnrm;
    }
    if scale != 1.0 {
        for v in d1[1..=n].iter_mut() {
            *v *= scale;
        }
        for v in e1[1..n].iter_mut() {
            *v *= scale;
        }
        tnrm *= scale;
    }
    let mut tryrac = tryrac;
    let thresh;
    if tryrac && !dlarrr(n, &d1[1..], &e1[1..]) {
        thresh = -eps;
        tryrac = false;
    } else if tryrac {
        thresh = eps;
    } else {
        // not TRYRAC: IINFO = -1
        thresh = -eps;
    }
    let dd = if tryrac { d1.clone() } else { vec![] };
    let mut e2 = vec![0.0f64; n + 1];
    for j in 1..n {
        e2[j] = e1[j] * e1[j];
    }
    let rtol1 = eps.sqrt();
    let rtol2 = fmax(eps.sqrt() * 5.0e-3, 4.0 * eps);
    let mut roots = match dlarre_all(n, &mut d1, &mut e1, &mut e2, thresh) {
        Ok(r) => r,
        Err(i) => return 10 + i.abs(),
    };
    let m = roots.m;
    let mut w1 = roots.w.clone();
    let mut isuppz = vec![0usize; 2 * n + 1];
    let info = dlarrv(
        n,
        roots.vl,
        &mut d1,
        &mut e1,
        roots.pivmin,
        &roots.isplit,
        m,
        MINRGP,
        rtol1,
        rtol2,
        &mut w1,
        &mut roots.werr,
        &mut roots.wgap,
        &roots.iblock,
        &roots.indexw,
        &roots.gers,
        z,
        ldz,
        &mut isuppz,
    );
    if info != 0 {
        return 20 + info.abs();
    }
    if tryrac {
        let mut ibegin = 1usize;
        let mut wbegin = 1usize;
        for jblk in 1..=roots.iblock[m] {
            let iend = roots.isplit[jblk];
            let inn = iend - ibegin + 1;
            let mut wend = wbegin - 1;
            while wend < m && roots.iblock[wend + 1] == jblk {
                wend += 1;
            }
            if wend < wbegin {
                ibegin = iend + 1;
                continue;
            }
            let offset = roots.indexw[wbegin] - 1;
            let ifirst = roots.indexw[wbegin];
            let ilast = roots.indexw[wend];
            let rtol2 = 4.0 * eps;
            dlarrj(
                inn,
                &dd[ibegin..],
                &e2[ibegin..],
                ifirst,
                ilast,
                rtol2,
                offset,
                &mut w1[wbegin..],
                &mut roots.werr[wbegin..],
                roots.pivmin,
                tnrm,
            );
            ibegin = iend + 1;
            wbegin = wend + 1;
        }
    }
    if scale != 1.0 {
        for v in w1[1..=m].iter_mut() {
            *v *= 1.0 / scale;
        }
    }
    w[..m].copy_from_slice(&w1[1..=m]);
    let nsplit = roots.isplit.len() - 1;
    if nsplit > 1 {
        sort_pairs(n, m, w, z, ldz);
    }
    0
}

/// `DSTEMR`/`DSYEVR`'s final selection sort, swapping eigenvector columns.
pub(crate) fn sort_pairs(n: usize, m: usize, w: &mut [f64], z: &mut [f64], ldz: usize) {
    for j in 0..m.saturating_sub(1) {
        let mut i = usize::MAX;
        let mut tmp = w[j];
        for jj in j + 1..m {
            if w[jj] < tmp {
                i = jj;
                tmp = w[jj];
            }
        }
        if i != usize::MAX {
            w[i] = w[j];
            w[j] = tmp;
            for r in 0..n {
                z.swap(r + i * ldz, r + j * ldz);
            }
        }
    }
}

/// `DLAEV2(A, B, C) → (RT1, RT2, CS1, SN1)`: the eigen-decomposition of
/// `[[a, b], [b, c]]`.
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };
    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1;
    }
    let (cs, sgn2) = if df >= 0.0 {
        (df + rt, 1)
    } else {
        (df - rt, -1)
    };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}
