//! `numpy_openblas` against NumPy 2.3.5 on OpenBLAS 0.3.30 Haswell (msi, 20 threads):
//! `tests/fixtures/openblas_haswell_*.txt`, from `tools/openblas_ref_*.py`. Zero tolerance.

fn stream(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// Whether `got` equals the fixture's values (hex bits, or `fnv:<hash>` of the bits).
fn same(want: &[&str], got: &[f64]) -> bool {
    if let Some(h) = want.first().and_then(|w| w.strip_prefix("fnv:")) {
        let mut x: u64 = 0xcbf29ce484222325;
        for v in got {
            for b in v.to_bits().to_le_bytes() {
                x = (x ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        return u64::from_str_radix(h, 16).unwrap() == x;
    }
    want.len() == got.len()
        && want
            .iter()
            .zip(got)
            .all(|(w, g)| u64::from_str_radix(w, 16).unwrap() == g.to_bits())
}

#[test]
fn ddot_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_ddot.txt");
    let mut bad = vec![];
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let (n, inc): (usize, usize) = (t[0].parse().unwrap(), t[1].parse().unwrap());
        let want = u64::from_str_radix(t[2], 16).unwrap();
        let x = stream(0x1234 + n as u64 * 7 + inc as u64, n * inc);
        let y = stream(0x9876 + n as u64 * 11 + inc as u64, n * inc);
        let got = rlx_linux_ref::openblas::ddot(n, &x, inc, &y, inc);
        if got.to_bits() != want {
            bad.push((n, inc));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ: {:?}",
        bad.len(),
        text.lines().count(),
        &bad[..bad.len().min(30)]
    );
}

#[test]
fn dsyrk_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dsyrk.txt");
    let (mut bad, mut tot, mut cases) = (vec![], 0usize, vec![]);
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let (n, k, tr): (usize, usize, u64) = (
            t[0].parse().unwrap(),
            t[1].parse().unwrap(),
            t[2].parse().unwrap(),
        );
        let a = stream(0x5151 + n as u64 * 131 + k as u64 * 7 + tr, n * k);
        let (trans, lda) = if tr == 0 { (111, k) } else { (112, n) };
        let mut c = vec![0.0; n * n];
        if n == 1 {
            // NumPy's matmul takes a 1×k·k×1 product as a dot product (`cblas_ddot`)
            c[0] = rlx_linux_ref::openblas::ddot(k, &a, 1, &a, 1);
        } else {
            rlx_linux_ref::openblas::dsyrk_upper(101, trans, n, k, &a, lda, &mut c, n);
        }
        let up: Vec<f64> = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .map(|(i, j)| c[i * n + j])
            .collect();
        tot += up.len();
        if !same(&t[3..], &up) {
            bad.push((n, k, tr, 1));
        }
        cases.push(());
    }
    assert!(
        bad.is_empty(),
        "{} of {} cases differ ({tot} elements): {:?}",
        bad.len(),
        cases.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dgesv_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgesv.txt");
    let mut bad = vec![];
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let (n, kind, seed): (usize, u64, u64) = (
            t[0].parse().unwrap(),
            t[1].parse().unwrap(),
            t[2].parse().unwrap(),
        );
        let u = stream(0x7777 + n as u64 * 1009 + kind * 31 + seed, n * n + n);
        let (mut a, mut b) = (u[..n * n].to_vec(), u[n * n..].to_vec());
        if kind == 1 {
            a.iter_mut().for_each(|v| *v = (*v * 3.0).round_ties_even());
            b.iter_mut().for_each(|v| *v = (*v * 5.0).round_ties_even());
        }
        if kind == 2 {
            // np.logspace(-8, 0, n) = 10 ** np.linspace(-8, 0, n)
            for i in 0..n {
                let e = if n == 1 {
                    -8.0
                } else {
                    -8.0 + i as f64 * (8.0 / (n - 1) as f64)
                };
                let s = 10f64.powf(e);
                a[i * n..(i + 1) * n].iter_mut().for_each(|v| *v *= s);
            }
        }
        // np.linalg.solve hands LAPACK a Fortran-order copy
        let mut col = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                col[i + j * n] = a[i * n + j];
            }
        }
        let info = rlx_linux_ref::openblas::dgesv(n, &mut col, &mut b).unwrap();
        let ok = if t[3] == "singular" {
            info != 0
        } else {
            info == 0
                && t[3..]
                    .iter()
                    .zip(&b)
                    .all(|(w, g)| u64::from_str_radix(w, 16).unwrap() == g.to_bits())
        };
        if !ok {
            bad.push((n, kind, seed));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} differ: {:?}",
        bad.len(),
        text.lines().count(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dgemm_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgemm.txt");
    let mut bad = vec![];
    let mut checked = 0;
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let p = |i: usize| t[i].parse::<usize>().unwrap();
        let (m, n, k, ta, tb) = (p(0), p(1), p(2), p(3), p(4));
        let u = stream(
            0x3131 + (m * 7 + n * 131 + k * 1031 + ta * 2 + tb) as u64,
            m * k + k * n,
        );
        let (a, b) = (&u[..m * k], &u[m * k..]);
        let lda = if ta == 1 { m } else { k };
        let ldb = if tb == 1 { k } else { n };
        let mut c = vec![0.0; m * n];
        let tr = |x: usize| if x == 1 { 112 } else { 111 };
        rlx_linux_ref::openblas::dgemm(101, tr(ta), tr(tb), m, n, k, a, lda, b, ldb, &mut c, n);
        checked += 1;
        if !same(&t[5..], &c) {
            bad.push((m, n, k, ta, tb));
        }
    }
    assert!(checked > 500, "{checked}");
    assert!(
        bad.is_empty(),
        "{} of {checked} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dgemv_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgemv.txt");
    let (mut bad, mut checked) = (vec![], 0);
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let p = |i: usize| t[i].parse::<usize>().unwrap();
        let (m, n, trans, incx) = (p(0), p(1), p(2), p(3));
        let lx = if trans == 0 { n } else { m };
        let u = stream(
            0x4242 + (m * 3 + n * 101 + trans * 7 + incx) as u64,
            m * n + lx * incx,
        );
        let (a, x) = (&u[..m * n], &u[m * n..]);
        let ly = if trans == 0 { m } else { n };
        let mut y = vec![0.0; ly];
        if !rlx_linux_ref::openblas::dgemv(
            101,
            if trans == 0 { 111 } else { 112 },
            m,
            n,
            a,
            n,
            x,
            incx,
            &mut y,
            1,
        ) {
            continue;
        }
        checked += 1;
        if !same(&t[4..], &y) {
            bad.push((m, n, trans, incx));
        }
    }
    assert!(checked > 400, "{checked}");
    assert!(
        bad.is_empty(),
        "{} of {checked} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dnrm2_and_drot_match_scipy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_l1.txt");
    let (mut bad, mut n_cases) = (vec![], 0);
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        n_cases += 1;
        match t[0] {
            "nrm2" => {
                let (n, inc, k): (usize, usize, i32) = (
                    t[1].parse().unwrap(),
                    t[2].parse().unwrap(),
                    t[3].parse().unwrap(),
                );
                let x: Vec<f64> = stream(
                    ((0x1717 + n * 13 + inc) as i64 + i64::from(k)) as u64,
                    (n * inc).max(1),
                )
                .iter()
                .map(|v| v * 2f64.powi(k))
                .collect();
                let got = rlx_linux_ref::openblas::dnrm2(n, &x, inc);
                if got.to_bits() != u64::from_str_radix(t[4], 16).unwrap() {
                    bad.push(line.chars().take(40).collect::<String>());
                }
            }
            "rot" => {
                let n: usize = t[1].parse().unwrap();
                let (mut x, mut y) = (stream(0x2323 + n as u64, n), stream(0x4545 + n as u64, n));
                rlx_linux_ref::openblas::drot(n, &mut x, 1, &mut y, 1, 0.6, 0.8 + 1e-9);
                let bar = t.iter().position(|s| *s == "|").unwrap();
                if !same(&t[2..bar], &x) || !same(&t[bar + 1..], &y) {
                    bad.push(format!("rot {n}"));
                }
            }
            op => panic!("{op}"),
        }
    }
    assert!(n_cases > 300);
    assert!(
        bad.is_empty(),
        "{} of {n_cases} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(20)]
    );
}

#[test]
fn dgelsd_matches_numpy_lstsq_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgelsd.txt");
    let (mut bad, mut checked, mut skipped) = (vec![], 0, 0);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(" | ").collect();
        let h: Vec<usize> = parts[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let (m, n, nrhs, kind, rc) = (h[0], h[1], h[2], h[3], h[4]);
        let u = stream(
            0x5a5a + (m * 977 + n * 131 + nrhs * 7 + kind) as u64,
            m * n + m * nrhs,
        );
        let mut a = u[..m * n].to_vec(); // row-major m×n
        let bm = &u[m * n..];
        if kind == 1 && n > 1 {
            for i in 0..m {
                a[i * n + n - 1] = a[i * n];
            }
        }
        if kind == 2 {
            // np.vander(x, n) via multiply.accumulate, then A /= sqrt((A*A).sum(axis=0))
            for i in 0..m {
                let x = (i + 1) as f64;
                let mut p = 1.0;
                for k in 0..n {
                    a[i * n + (n - 1 - k)] = p;
                    p *= x;
                }
            }
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += a[i * n + j] * a[i * n + j];
                }
                let s = s.sqrt();
                for i in 0..m {
                    a[i * n + j] /= s;
                }
            }
        }
        let rcond = if rc == 0 {
            f64::EPSILON * m.max(n) as f64
        } else {
            1e-3
        };
        let ldb = m.max(n);
        let mut ac = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                ac[i + j * m] = a[i * n + j];
            }
        }
        let mut b = vec![0.0; ldb * nrhs];
        for i in 0..m {
            for j in 0..nrhs {
                b[i + j * ldb] = bm[i * nrhs + j];
            }
        }
        let Some(r) = rlx_linux_ref::lapack::dgelsd(m, n, nrhs, &mut ac, &mut b, ldb, rcond) else {
            skipped += 1;
            continue;
        };
        checked += 1;
        let mut x = vec![0.0; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                x[i * nrhs + j] = b[i + j * ldb];
            }
        }
        let want_x: Vec<&str> = parts[1].split(' ').collect();
        let want_rank: usize = parts[2].parse().unwrap();
        let want_s: Vec<&str> = parts[3].split(' ').collect();
        if r.info != 0 || r.rank != want_rank || !same(&want_x, &x) || !same(&want_s, &r.s) {
            bad.push((m, n, nrhs, kind, rc));
        }
    }
    eprintln!("dgelsd: {checked} checked, {skipped} outside the ported branches");
    assert!(checked > 900, "{checked}");
    assert!(
        bad.is_empty(),
        "{} of {checked} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dgelsd_blocked_matches_numpy_lstsq_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgelsd_big.txt");
    let (mut bad, mut checked, mut skipped) = (vec![], 0, 0);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(" | ").collect();
        let h: Vec<usize> = parts[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let (m, n, nrhs, kind, rc) = (h[0], h[1], h[2], h[3], h[4]);
        let u = stream(
            0x5a5a + (m * 977 + n * 131 + nrhs * 7 + kind) as u64,
            m * n + m * nrhs,
        );
        let mut a = u[..m * n].to_vec(); // row-major m×n
        let bm = &u[m * n..];
        if kind == 1 && n > 1 {
            for i in 0..m {
                a[i * n + n - 1] = a[i * n];
            }
        }
        if kind == 2 {
            // np.vander(x, n) via multiply.accumulate, then A /= sqrt((A*A).sum(axis=0))
            for i in 0..m {
                let x = (i + 1) as f64;
                let mut p = 1.0;
                for k in 0..n {
                    a[i * n + (n - 1 - k)] = p;
                    p *= x;
                }
            }
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += a[i * n + j] * a[i * n + j];
                }
                let s = s.sqrt();
                for i in 0..m {
                    a[i * n + j] /= s;
                }
            }
        }
        let rcond = if rc == 0 {
            f64::EPSILON * m.max(n) as f64
        } else {
            1e-3
        };
        let ldb = m.max(n);
        let mut ac = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                ac[i + j * m] = a[i * n + j];
            }
        }
        let mut b = vec![0.0; ldb * nrhs];
        for i in 0..m {
            for j in 0..nrhs {
                b[i + j * ldb] = bm[i * nrhs + j];
            }
        }
        let Some(r) = rlx_linux_ref::lapack::dgelsd(m, n, nrhs, &mut ac, &mut b, ldb, rcond) else {
            skipped += 1;
            continue;
        };
        checked += 1;
        let mut x = vec![0.0; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                x[i * nrhs + j] = b[i + j * ldb];
            }
        }
        let want_x: Vec<&str> = parts[1].split(' ').collect();
        let want_rank: usize = parts[2].parse().unwrap();
        let want_s: Vec<&str> = parts[3].split(' ').collect();
        if r.info != 0 || r.rank != want_rank || !same(&want_x, &x) || !same(&want_s, &r.s) {
            bad.push((m, n, nrhs, kind, rc));
        }
    }
    eprintln!("dgelsd: {checked} checked, {skipped} outside the ported branches");
    assert!(checked > 150, "{checked}");
    assert!(
        bad.is_empty(),
        "{} of {checked} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn dgelsd_divide_and_conquer_matches_numpy_lstsq_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgelsd_dc.txt");
    let (mut bad, mut checked, mut skipped) = (vec![], 0, 0);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(" | ").collect();
        let h: Vec<usize> = parts[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let (m, n, nrhs, kind, rc) = (h[0], h[1], h[2], h[3], h[4]);
        let u = stream(
            0x5a5a + (m * 977 + n * 131 + nrhs * 7 + kind) as u64,
            m * n + m * nrhs,
        );
        let mut a = u[..m * n].to_vec(); // row-major m×n
        let bm = &u[m * n..];
        if kind == 1 && n > 1 {
            for i in 0..m {
                a[i * n + n - 1] = a[i * n];
            }
        }
        if kind == 2 {
            // np.vander(x, n) via multiply.accumulate, then A /= sqrt((A*A).sum(axis=0))
            for i in 0..m {
                let x = (i + 1) as f64;
                let mut p = 1.0;
                for k in 0..n {
                    a[i * n + (n - 1 - k)] = p;
                    p *= x;
                }
            }
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += a[i * n + j] * a[i * n + j];
                }
                let s = s.sqrt();
                for i in 0..m {
                    a[i * n + j] /= s;
                }
            }
        }
        let rcond = if rc == 0 {
            f64::EPSILON * m.max(n) as f64
        } else {
            1e-3
        };
        let ldb = m.max(n);
        let mut ac = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                ac[i + j * m] = a[i * n + j];
            }
        }
        let mut b = vec![0.0; ldb * nrhs];
        for i in 0..m {
            for j in 0..nrhs {
                b[i + j * ldb] = bm[i * nrhs + j];
            }
        }
        let Some(r) = rlx_linux_ref::lapack::dgelsd(m, n, nrhs, &mut ac, &mut b, ldb, rcond) else {
            skipped += 1;
            continue;
        };
        checked += 1;
        let mut x = vec![0.0; n * nrhs];
        for i in 0..n {
            for j in 0..nrhs {
                x[i * nrhs + j] = b[i + j * ldb];
            }
        }
        let want_x: Vec<&str> = parts[1].split(' ').collect();
        let want_rank: usize = parts[2].parse().unwrap();
        let want_s: Vec<&str> = parts[3].split(' ').collect();
        if r.info != 0 || r.rank != want_rank || !same(&want_x, &x) || !same(&want_s, &r.s) {
            bad.push((m, n, nrhs, kind, rc));
        }
    }
    eprintln!("dgelsd: {checked} checked, {skipped} outside the ported branches");
    assert!(checked > 200, "{checked}");
    assert!(
        bad.is_empty(),
        "{} of {checked} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(40)]
    );
}

#[test]
fn trmm_gemm_gemv_alpha_beta_match_scipy_on_openblas_haswell() {
    use rlx_linux_ref::openblas as ob;
    let text = include_str!("fixtures/openblas_haswell_l3ab.txt");
    const ALPHA: [f64; 3] = [1.0, -1.0, 0.7];
    const BETA: [f64; 3] = [0.0, 1.0, -0.3];
    let (mut bad, mut n_cases) = (std::collections::BTreeMap::<String, Vec<String>>::new(), 0);
    for line in text.lines() {
        let (head, want) = line.split_once(" | ").unwrap();
        let want: Vec<&str> = want.split(' ').collect();
        let h: Vec<&str> = head.split(' ').collect();
        let p = |i: usize| h[i].parse::<usize>().unwrap();
        n_cases += 1;
        let (kind, got) = match h[0] {
            "trmm" => {
                let (m, n, side, lower, trans, unit, ai) =
                    (p(1), p(2), p(3), p(4), p(5), p(6), p(7));
                let k = if side == 0 { m } else { n };
                let seed = 0x6161
                    + m * 101
                    + n * 7
                    + side * 3
                    + lower * 5
                    + trans * 11
                    + unit * 13
                    + ai * 17;
                let u = stream(seed as u64, k * k + m * n);
                let a = u[..k * k].to_vec();
                let mut b = u[k * k..].to_vec();
                ob::dtrmm(
                    side == 0,
                    lower == 1,
                    trans == 1,
                    unit == 1,
                    m,
                    n,
                    ALPHA[ai],
                    &a,
                    k,
                    &mut b,
                    m,
                );
                (
                    format!("trmm side{side} lower{lower} trans{trans} unit{unit}"),
                    b,
                )
            }
            "gemm" => {
                let (m, n, k, ta, tb, ai, bi) = (p(1), p(2), p(3), p(4), p(5), p(6), p(7));
                let seed = 0x7272 + m * 3 + n * 5 + k * 7 + ta + tb * 2 + ai * 9 + bi * 27;
                let u = stream(seed as u64, m * k + k * n + m * n);
                let a = u[..m * k].to_vec();
                let b = u[m * k..m * k + k * n].to_vec();
                let mut c = u[m * k + k * n..].to_vec();
                let (lda, ldb) = (if ta == 1 { k } else { m }, if tb == 1 { n } else { k });
                ob::dgemm_ab(
                    102,
                    if ta == 1 { 112 } else { 111 },
                    if tb == 1 { 112 } else { 111 },
                    m,
                    n,
                    k,
                    ALPHA[ai],
                    &a,
                    lda,
                    &b,
                    ldb,
                    BETA[bi],
                    &mut c,
                    m,
                );
                ("gemm".to_string(), c)
            }
            "gemv" => {
                let (m, n, trans, ai, bi) = (p(1), p(2), p(3), p(4), p(5));
                let seed = 0x8383 + m * 3 + n * 5 + trans + ai * 9 + bi * 27;
                let (lx, ly) = if trans == 0 { (n, m) } else { (m, n) };
                let u = stream(seed as u64, m * n + lx + ly);
                let a = u[..m * n].to_vec();
                let x = u[m * n..m * n + lx].to_vec();
                let mut y = u[m * n + lx..].to_vec();
                assert!(ob::dgemv_ab(
                    102,
                    if trans == 0 { 111 } else { 112 },
                    m,
                    n,
                    ALPHA[ai],
                    &a,
                    m,
                    &x,
                    1,
                    BETA[bi],
                    &mut y,
                    1
                ));
                ("gemv".to_string(), y)
            }
            k => panic!("{k}"),
        };
        if !same(&want, &got) {
            bad.entry(kind).or_default().push(head.to_string());
        }
    }
    assert!(n_cases > 5000);
    let total: usize = bad.values().map(Vec::len).sum();
    assert!(
        total == 0,
        "{total} of {n_cases} differ: {:#?}",
        bad.iter()
            .map(|(k, v)| (k, v.len(), &v[..v.len().min(3)]))
            .collect::<Vec<_>>()
    );
}

/// The input of `tools/openblas_ref_dsyevr.py`'s case `(n, kind)`, column-major.
fn dsyevr_matrix(n: usize, kind: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    match kind {
        7 => {
            return dsyevr_matrix(n, 0)
                .iter()
                .map(|v| v * 2f64.powi(-500))
                .collect();
        }
        4 => {
            for i in 0..n {
                a[i + i * n] = ((n - 1) as f64 / 2.0 - i as f64).abs();
                if i + 1 < n {
                    a[i + (i + 1) * n] = 1.0;
                    a[i + 1 + i * n] = 1.0;
                }
            }
            return a;
        }
        5 => {
            for i in 0..n {
                for j in 0..n {
                    a[i + j * n] = 1.0 + if i == j { 2.0 } else { 0.0 };
                }
            }
            return a;
        }
        6 => {
            let v = stream(0x7c11 + n as u64, n);
            let mut vtv = 0.0f64;
            for x in &v {
                vtv += x * x;
            }
            let q =
                |i: usize, j: usize| (if i == j { 1.0 } else { 0.0 }) - (2.0 * v[i] * v[j]) / vtv;
            let d: Vec<f64> = (0..n)
                .map(|i| match i % 3 {
                    0 => 1.0 + i as f64 * 1e-9,
                    1 => 2.0 + i as f64 * 1e-13,
                    _ => 3.0 + i as f64,
                })
                .collect();
            for i in 0..n {
                for j in 0..=i {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += q(i, k) * d[k] * q(j, k);
                    }
                    a[i + j * n] = s;
                    a[j + i * n] = s;
                }
            }
            return a;
        }
        _ => {}
    }
    if kind == 0 || kind == 3 {
        let u = stream(0x51e7 + (n * 31 + kind) as u64, n * n);
        for i in 0..n {
            for j in 0..n {
                let mut v = (u[i * n + j] + u[j * n + i]) * 0.5;
                if kind == 3 {
                    // Python's `round` returns an int: a rounded −0.4 is +0
                    v = (v * 4.0).round_ties_even() / 4.0 + 0.0;
                }
                a[i + j * n] = v;
            }
        }
        return a;
    }
    let f = if kind == 1 { n / 2 + 1 } else { 2 * n };
    let u = stream(0x6a3d + (n * 17 + kind) as u64, n * f);
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0f64;
            for l in 0..f {
                s += u[i * f + l] * u[j * f + l];
            }
            a[i + j * n] = s + 1.0;
            a[j + i * n] = s + 1.0;
        }
    }
    a
}

#[test]
fn dsyevr_matches_scipy_eigh_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dsyevr.txt");
    let (mut bad_trd, mut bad_w, mut bad_v, mut checked) = (vec![], vec![], vec![], 0);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(" | ").collect();
        let h: Vec<usize> = parts[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let (n, kind) = (h[0], h[1]);
        let f = |k: usize| parts[k].split(' ').collect::<Vec<&str>>();
        let a0 = dsyevr_matrix(n, kind);
        // the reduction alone
        let mut a = a0.clone();
        let (d, e, tau) = rlx_linux_ref::dsytrd_lower(n, &mut a, n, 28 * n as i64);
        if !(same(&f(3), &a)
            && same(&f(4), &d)
            && same(&f(5), &e[..n - 1])
            && same(&f(6), &tau[..n - 1]))
        {
            bad_trd.push((n, kind));
        }
        let mut a = a0.clone();
        match rlx_linux_ref::dsyevr_lower(n, &mut a, n, 33 * n as i64) {
            Ok((w, v)) => {
                if !same(&f(1), &w) {
                    bad_w.push((n, kind));
                }
                if !same(&f(2), &v) {
                    bad_v.push((n, kind));
                }
            }
            Err(info) => bad_w.push((n, 100 + info as usize)),
        }
        checked += 1;
    }
    assert!(
        bad_trd.is_empty() && bad_w.is_empty() && bad_v.is_empty(),
        "{checked} cases: dsytrd {bad_trd:?}, w {bad_w:?}, v {bad_v:?}"
    );
}

#[test]
fn dgesdd_matches_scipy_svd_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dgesdd.txt");
    let (mut bad, mut checked) = (vec![], 0);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(" | ").collect();
        let h: Vec<usize> = parts[0].split(' ').map(|v| v.parse().unwrap()).collect();
        let (m, n, kind) = (h[0], h[1], h[2]);
        let u = stream(0x5d5d + (m * 101 + n * 7 + kind) as u64, m * n);
        // row-major m×n as numpy builds it, then the Fortran copy LAPACK sees
        let mut a = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut v = u[i * n + j];
                if kind == 1 && n > 1 && j == n - 1 {
                    v = u[i * n];
                }
                if kind == 2 {
                    v *= rlx_libm::pow(10.0, -(j as f64));
                }
                a[i + j * m] = v;
            }
        }
        let Some(r) = rlx_linux_ref::lapack::dgesdd_s(m, n, &mut a, m) else {
            continue;
        };
        let (uu, s, vt) = r.unwrap();
        let f = |k: usize| parts[k].split(' ').collect::<Vec<&str>>();
        checked += 1;
        if !(same(&f(1), &uu) && same(&f(2), &s) && same(&f(3), &vt)) {
            bad.push((m, n, kind));
        }
    }
    assert!(checked >= 40, "{checked}");
    assert!(bad.is_empty(), "{checked} checked: {bad:?}");
}
