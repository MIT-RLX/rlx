// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_linalg::numpy::{dot, dot_mv, dot_vv, matmul}` against NumPy 2.x on
//! Accelerate, bit-for-bit, for every routing branch (`tools/gen_numpy_routing.py`).

use rlx_linalg::numpy::{self, View};

fn hx(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .map(|t| f64::from_bits(u64::from_str_radix(t, 16).unwrap()))
        .collect()
}

/// A buffer holding the logical row-major `x` (r×c) in the requested order, and its view.
fn layout(x: &[f64], r: usize, c: usize, order: &str) -> Vec<f64> {
    match order {
        "C" => x.to_vec(),
        "F" => {
            let mut f = vec![0.0; r * c];
            for i in 0..r {
                for j in 0..c {
                    f[i + j * r] = x[i * c + j];
                }
            }
            f
        }
        o => panic!("layout {o}"),
    }
}
fn view<'a>(buf: &'a [f64], r: usize, c: usize, order: &str) -> View<'a> {
    match order {
        "C" => View::c(buf, r, c),
        _ => View {
            data: buf,
            offset: 0,
            rows: r,
            cols: c,
            rs: 1,
            cs: r,
        },
    }
}

#[test]
fn numpy_routing_is_bit_exact() {
    // this target's reference: macOS (Accelerate) on Apple, the Linux x86-64 wheel
    // (OpenBLAS Haswell, msi) elsewhere
    #[cfg(target_vendor = "apple")]
    let text = include_str!("vectors/numpy_routing.txt");
    #[cfg(not(target_vendor = "apple"))]
    let text = include_str!("vectors/linux-x86_64/numpy_routing.txt");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let (mut n_cases, mut bad) = (0usize, Vec::new());
    while let Some(head) = lines.next() {
        let t: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(t[0], "case");
        let (fname, m, k, n, la, lb) = (
            t[1],
            t[2].parse::<usize>().unwrap(),
            t[3].parse::<usize>().unwrap(),
            t[4].parse::<usize>().unwrap(),
            t[5],
            t[6],
        );
        let a = hx(&lines.next().unwrap()[2..]);
        let b = if lb != "AT" {
            hx(&lines.next().unwrap()[2..])
        } else {
            Vec::new()
        };
        let want = hx(&lines.next().unwrap()[2..]);
        let got: Vec<f64> = match fname {
            "dot_vv" => vec![numpy::dot_vv(&a, 1, &b, 1, k)],
            "dot_mv" => {
                let ab = layout(&a, m, k, la);
                numpy::dot_mv(view(&ab, m, k, la), &b, 1)
            }
            "dot" | "matmul" => {
                let (ar, ac) = (if fname == "matmul" && m == 1 { 1 } else { m }, k);
                let ab = layout(&a, ar, ac, la);
                let av = view(&ab, ar, ac, la);
                let bb;
                let bv = if lb == "AT" {
                    av.t()
                } else {
                    let (br, bc) = (k, n);
                    bb = layout(&b, br, bc, lb);
                    view(&bb, br, bc, lb)
                };
                if fname == "dot" {
                    numpy::dot(av, bv)
                } else {
                    numpy::matmul(av, bv)
                }
            }
            other => panic!("{other}"),
        };
        n_cases += 1;
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        if got.len() != want.len() || diff > 0 {
            bad.push(format!("{head}: {diff} of {} differ", want.len()));
        }
    }
    assert!(n_cases > 1000, "only {n_cases} cases read");
    assert!(
        bad.is_empty(),
        "{} of {n_cases} cases differ:\n{}",
        bad.len(),
        bad[..bad.len().min(12)].join("\n")
    );
}
