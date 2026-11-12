// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `sgemm_auto` on a mostly-zero `A` (one-hot rows, zero-padded batches, a
//! few dense rows mixed in) against an f64 reference written out here. These
//! shapes take the sparse-LHS path; its result must match a dense GEMM's to
//! f32 rounding, including rows it hands back to BLAS and rows that are all
//! zero.

use rlx_cpu::blas::sgemm_auto;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let v = a[i * k + p] as f64;
            for j in 0..n {
                c[i * n + j] += v * b[p * n + j] as f64;
            }
        }
    }
    c
}

/// `m` rows of `k` with `nnz` nonzeros each; rows in `dense` are full; rows
/// at or past `live` are zero (batch padding).
fn lhs(m: usize, k: usize, nnz: usize, dense: &[usize], live: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut a = vec![0f32; m * k];
    for i in 0..live.min(m) {
        if dense.contains(&i) {
            for p in 0..k {
                a[i * k + p] = (lcg(&mut s) % 1000) as f32 / 500.0 - 1.0;
            }
        } else {
            for _ in 0..nnz {
                let p = (lcg(&mut s) as usize) % k;
                a[i * k + p] = if lcg(&mut s).is_multiple_of(4) {
                    0.5
                } else {
                    1.0
                };
            }
        }
    }
    a
}

#[test]
fn sparse_lhs_matches_dense_reference() {
    for &(m, k, n, nnz, ref dense, live) in &[
        (512usize, 1890usize, 256usize, 40usize, vec![], 512usize),
        (512, 1890, 256, 40, vec![3, 200], 512),
        (300, 1024, 64, 8, vec![], 120),
        (64, 4096, 16, 1, vec![0], 64),
        (17, 2048, 33, 5, vec![], 17),
    ] {
        let a = lhs(m, k, nnz, dense, live, 7 + m as u64);
        let mut s = 99u64;
        let b: Vec<f32> = (0..k * n)
            .map(|_| (lcg(&mut s) % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        // Stale output contents must be overwritten, not accumulated.
        let mut c = vec![123.0f32; m * n];
        sgemm_auto(&a, &b, &mut c, m, k, n);
        let r = reference(&a, &b, m, k, n);
        for (i, (&got, &want)) in c.iter().zip(&r).enumerate() {
            let tol = 1e-4 * (1.0 + want.abs()) * (k as f64).sqrt();
            assert!(
                (got as f64 - want).abs() <= tol,
                "m={m} k={k} n={n} nnz={nnz}: c[{i}] = {got}, want {want}"
            );
        }
    }
}
