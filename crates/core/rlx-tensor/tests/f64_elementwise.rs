// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The f64 CPU kernels against a plain reference, bit for bit, on both sides
//! of the parallel threshold.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn data(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64 + seed) * 0.37).sin() * 3.0 + 3.5)
        .collect()
}

/// Row-major strides of `dims` padded to `rank`, 0 on broadcast axes.
fn strides(dims: &[usize], rank: usize) -> Vec<usize> {
    let mut full = vec![1; rank - dims.len()];
    full.extend_from_slice(dims);
    let mut s = vec![0; rank];
    let mut acc = 1;
    for ax in (0..rank).rev() {
        s[ax] = if full[ax] == 1 { 0 } else { acc };
        acc *= full[ax];
    }
    s
}

fn broadcast(
    a: &[f64],
    da: &[usize],
    b: &[f64],
    db: &[usize],
    out: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let (sa, sb) = (strides(da, out.len()), strides(db, out.len()));
    (0..out.iter().product::<usize>())
        .map(|mut i| {
            let (mut ia, mut ib) = (0, 0);
            for ax in (0..out.len()).rev() {
                let c = i % out[ax];
                i /= out[ax];
                ia += c * sa[ax];
                ib += c * sb[ax];
            }
            f(a[ia], b[ib])
        })
        .collect()
}

#[test]
fn binary_broadcasts_match_the_reference() {
    #[rustfmt::skip]
    let shapes: [(&[usize], &[usize], &[usize]); 9] = [
        (&[7], &[7], &[7]),
        (&[1500, 900], &[1500, 900], &[1500, 900]),
        (&[1500, 1], &[1, 900], &[1500, 900]),
        (&[1, 900], &[1500, 1], &[1500, 900]),
        (&[1500, 900], &[900], &[1500, 900]),
        (&[1500, 900], &[1500, 1], &[1500, 900]),
        (&[30, 1, 7], &[1, 40, 7], &[30, 40, 7]),
        (&[30, 1, 700], &[1, 40, 1], &[30, 40, 700]),
        (&[3, 1], &[1, 4], &[3, 4]),
    ];
    for (da, db, out) in shapes {
        let (a, b) = (
            data(da.iter().product(), 1.0),
            data(db.iter().product(), 2.0),
        );
        let (ta, tb) = (
            Tensor::from_f64(a.clone(), da.to_vec()),
            Tensor::from_f64(b.clone(), db.to_vec()),
        );
        let cases: [(&str, Tensor, fn(f64, f64) -> f64); 6] = [
            ("add", &ta + &tb, |x, y| x + y),
            ("sub", &ta - &tb, |x, y| x - y),
            ("mul", &ta * &tb, |x, y| x * y),
            ("div", &ta / &tb, |x, y| x / y),
            ("max", ta.maximum(&tb), f64::max),
            ("pow", ta.pow(&tb), f64::powf),
        ];
        for (name, got, f) in cases {
            let want = broadcast(&a, da, &b, db, out, f);
            assert!(got.to_vec_f64() == want, "{name} {da:?} {db:?}");
        }
    }
}

#[test]
fn transpose_and_expand_match_the_reference() {
    for (d, perm) in [
        ([3usize, 4, 5], [2usize, 0, 1]),
        ([120, 90, 110], [2, 1, 0]),
        ([120, 90, 110], [1, 0, 2]),
    ] {
        let x = data(d.iter().product(), 3.0);
        let got = Tensor::from_f64(x.clone(), d.to_vec())
            .transpose(perm)
            .to_vec_f64();
        let (od, st) = (perm.map(|p| d[p]), [d[1] * d[2], d[2], 1]);
        let mut want = Vec::with_capacity(x.len());
        for i in 0..od[0] {
            for j in 0..od[1] {
                for k in 0..od[2] {
                    want.push(x[i * st[perm[0]] + j * st[perm[1]] + k * st[perm[2]]]);
                }
            }
        }
        assert!(got == want, "{d:?} {perm:?}");
    }
    let c = data(1500, 4.0);
    let zero = Tensor::from_f64(vec![0.0; 1500 * 900], [1500, 900]);
    let got = (&zero + &Tensor::from_f64(c.clone(), [1500, 1])).to_vec_f64();
    assert!(
        got == c
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, 900))
            .collect::<Vec<_>>()
    );
}

#[test]
fn activations_match_the_reference() {
    for n in [100, 1_500_000] {
        let x = data(n, 5.0);
        let t = Tensor::from_f64(x.clone(), [n]);
        assert!(
            t.exp().to_vec_f64() == x.iter().map(|v| v.exp()).collect::<Vec<_>>(),
            "exp {n}"
        );
        assert!(
            t.log().to_vec_f64() == x.iter().map(|v| v.ln()).collect::<Vec<_>>(),
            "log {n}"
        );
        assert!(
            t.abs().to_vec_f64() == x.iter().map(|v| v.abs()).collect::<Vec<_>>(),
            "abs {n}"
        );
    }
}

#[test]
fn f32_row_broadcasts_match_the_reference() {
    for (m, n) in [(3usize, 4usize), (1500, 900)] {
        let (a, b): (Vec<f32>, Vec<f32>) = (
            (0..m).map(|i| 1.5 + i as f32 * 0.25).collect(),
            (0..n).map(|i| 2.5 + i as f32 * 0.125).collect(),
        );
        let (ta, tb) = (
            Tensor::from_vec(a.clone(), [m, 1]),
            Tensor::from_vec(b.clone(), [1, n]),
        );
        let each = |f: fn(f32, f32) -> f32| {
            a.iter()
                .flat_map(|&x| b.iter().map(move |&y| f(x, y)))
                .collect::<Vec<_>>()
        };
        assert!((&ta * &tb).to_vec() == each(|x, y| x * y), "mul {m}");
        assert!((&ta - &tb).to_vec() == each(|x, y| x - y), "sub {m}");
        assert!(
            (&tb / &ta).to_vec()
                == a.iter()
                    .flat_map(|&x| b.iter().map(move |&y| y / x))
                    .collect::<Vec<_>>(),
            "div {m}"
        );
    }
}
