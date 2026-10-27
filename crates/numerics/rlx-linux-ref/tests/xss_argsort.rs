//! `rlx_linux_ref::xss` against NumPy 2.3.5's `np.argsort` on AVX2 (msi):
//! `tests/fixtures/xss_argsort.txt` from `tools/xss_ref_argsort.py`.

fn stream(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

fn same(want: &[&str], got: &[usize]) -> bool {
    if let Some(h) = want.first().and_then(|w| w.strip_prefix("fnv:")) {
        let mut x: u64 = 0xcbf29ce484222325;
        for v in got {
            for b in (*v as u64).to_le_bytes() {
                x = (x ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        return u64::from_str_radix(h, 16).unwrap() == x;
    }
    want.len() == got.len()
        && want
            .iter()
            .zip(got)
            .all(|(w, g)| w.parse::<usize>().unwrap() == *g)
}

#[test]
fn argsort_matches_numpy_avx2() {
    let text = include_str!("fixtures/xss_argsort.txt");
    let mut bad = vec![];
    let mut n_cases = 0;
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let (n, r): (usize, u64) = (t[1].parse().unwrap(), t[2].parse().unwrap());
        let u = stream(0x9e37 + n as u64 * 31 + r, n);
        let got = if t[0] == "i64" {
            let v: Vec<i64> = u.iter().map(|x| (x * r as f64).floor() as i64).collect();
            rlx_linux_ref::xss::argsort_i64(&v)
        } else {
            let v: Vec<f64> = u
                .iter()
                .map(|x| (x * r as f64).floor() / 4.0 - 1.0)
                .collect();
            rlx_linux_ref::xss::argsort_f64(&v)
        };
        n_cases += 1;
        if !same(&t[3..], &got) {
            bad.push(format!("{} n={n} r={r}", t[0]));
        }
    }
    assert!(n_cases > 900);
    assert!(
        bad.is_empty(),
        "{} of {n_cases} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(30)]
    );
}
