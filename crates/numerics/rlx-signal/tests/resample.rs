// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `resample_poly` / `upfirdn` / `kaiser` / `i0` against SciPy 1.17.1 on macOS arm64
//! (`tests/vectors/scipy_resample.txt`, from `tools/gen_resample_vectors.py`).
//! Reports how many values are bit-identical and the worst ULP distance; the gate is
//! bit-exactness.

#![cfg(target_vendor = "apple")]

use rlx_signal::resample::{i0, kaiser, resample_poly, upfirdn};

const VECTORS: &str = include_str!("vectors/scipy_resample.txt");

fn arr(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
        .collect()
}

fn ulps(a: f64, b: f64) -> u64 {
    let (a, b) = (a.to_bits() as i64, b.to_bits() as i64);
    let o = |v: i64| if v < 0 { i64::MIN - v } else { v };
    (o(a) - o(b)).unsigned_abs()
}

/// (bit-identical count, total, worst ulp)
fn cmp(got: &[f64], want: &[f64]) -> (usize, usize, u64) {
    assert_eq!(got.len(), want.len(), "length");
    let same = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| ulps(*a, *b))
        .max()
        .unwrap_or(0);
    (same, got.len(), worst)
}

#[test]
fn matches_scipy_bit_for_bit() {
    let mut failures = Vec::new();
    for line in VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let mut parts = line.splitn(3, '|');
        let (name, args, want) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            arr(parts.next().unwrap()),
        );
        let (got, tag) = match name {
            "i0" => (
                arr(args).into_iter().map(i0).collect::<Vec<_>>(),
                "i0".to_string(),
            ),
            "kaiser" => {
                let (m, b) = args.split_once(',').unwrap();
                (
                    kaiser(m.parse().unwrap(), b.parse().unwrap()),
                    format!("kaiser({args})"),
                )
            }
            "resample_poly" => {
                let mut it = args.splitn(3, ',');
                let up: usize = it.next().unwrap().parse().unwrap();
                let down: usize = it.next().unwrap().parse().unwrap();
                let x = arr(it.next().unwrap());
                (
                    resample_poly(&x, up, down),
                    format!("resample_poly(n={}, {up}/{down})", x.len()),
                )
            }
            "upfirdn" => {
                let mut it = args.splitn(4, ',');
                let up: usize = it.next().unwrap().parse().unwrap();
                let down: usize = it.next().unwrap().parse().unwrap();
                let nh: usize = it.next().unwrap().parse().unwrap();
                let rest = arr(it.next().unwrap());
                let (h, x) = rest.split_at(nh);
                (
                    upfirdn(h, x, up, down),
                    format!("upfirdn(nh={nh}, {up}/{down})"),
                )
            }
            other => panic!("unknown record {other}"),
        };
        let (same, n, worst) = cmp(&got, &want);
        eprintln!("{tag:40} {same}/{n} bit-identical, worst {worst} ulp");
        if same != n {
            failures.push(tag);
        }
    }
    assert!(failures.is_empty(), "not bit-exact: {failures:?}");
}
