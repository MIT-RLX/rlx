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
        let got = rlx_cpu::numpy_openblas::ddot(n, &x, inc, &y, inc);
        if got.to_bits() != want {
            bad.push((n, inc));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ: {:?}", bad.len(), text.lines().count(), &bad[..bad.len().min(30)]);
}

#[test]
fn dsyrk_matches_numpy_on_openblas_haswell() {
    let text = include_str!("fixtures/openblas_haswell_dsyrk.txt");
    let (mut bad, mut tot, mut cases) = (vec![], 0usize, vec![]);
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let (n, k, tr): (usize, usize, u64) = (t[0].parse().unwrap(), t[1].parse().unwrap(), t[2].parse().unwrap());
        let a = stream(0x5151 + n as u64 * 131 + k as u64 * 7 + tr, n * k);
        let (trans, lda) = if tr == 0 { (111, k) } else { (112, n) };
        let mut c = vec![0.0; n * n];
        if n == 1 {
            // NumPy's matmul takes a 1×k·k×1 product as a dot product (`cblas_ddot`)
            c[0] = rlx_cpu::numpy_openblas::ddot(k, &a, 1, &a, 1);
        } else {
            rlx_cpu::numpy_openblas::dsyrk_upper(101, trans, n, k, &a, lda, &mut c, n);
        }
        let mut w = t[3..].iter();
        let mut case_bad = 0;
        for i in 0..n {
            for j in i..n {
                let want = u64::from_str_radix(w.next().unwrap(), 16).unwrap();
                tot += 1;
                if c[i * n + j].to_bits() != want {
                    case_bad += 1;
                }
            }
        }
        if case_bad > 0 {
            bad.push((n, k, tr, case_bad));
        }
        cases.push(());
    }
    assert!(bad.is_empty(), "{} of {} cases differ ({tot} elements): {:?}", bad.len(), cases.len(), &bad[..bad.len().min(40)]);
}
