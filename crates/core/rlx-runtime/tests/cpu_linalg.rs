// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Dense linalg integration tests against the `rlx-linalg` package.
//!
//! Each test builds a graph containing a linalg op, runs it through
//! `Session::new(Device::Cpu)`, and validates the output against a
//! direct reference (matrix-multiply check, eigenvalue equation, etc.).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

#[allow(dead_code)]
fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn const_f64(g: &mut Graph, xs: &[f64], shape: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(shape, DType::F64),
    )
}

/// Dense row-major matmul reference (small sizes only).
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// 4×4 symmetric positive-definite matrix.
fn build_spd_4() -> Vec<f64> {
    // A = Mᵀ M + ε I  for some random-ish M; SPD by construction.
    let m = [
        1.0, 0.5, -0.25, 2.0, -1.5, 3.0, 0.75, -0.5, 0.0, 1.0, 4.0, 0.5, 2.0, -0.25, 1.0, 3.0_f64,
    ];
    let mut a = matmul(&m, &m, 4, 4, 4);
    // Make sure it's symmetric; matmul of MᵀM would be — but we did
    // M·M not Mᵀ·M, so explicitly symmetrize and add diag bump.
    for i in 0..4 {
        for j in (i + 1)..4 {
            let v = (a[i * 4 + j] + a[j * 4 + i]) * 0.5;
            a[i * 4 + j] = v;
            a[j * 4 + i] = v;
        }
    }
    for i in 0..4 {
        a[i * 4 + i] += 5.0;
    }
    a
}

#[test]
fn cholesky_factorizes_spd() {
    rlx_linalg::register();
    let a = build_spd_4();
    let n = 4;

    let mut g = Graph::new("cholesky");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let l = rlx_linalg::cholesky(&mut g, a_n, /*lower=*/ true);
    g.set_outputs(vec![l]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let l = bytes_to_f64s(&outs[0].0);
    assert_eq!(l.len(), n * n);

    // Verify upper triangle is zero.
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                l[i * n + j].abs() < 1e-12,
                "L[{i},{j}] should be zero (upper triangle): {}",
                l[i * n + j]
            );
        }
    }
    // Verify L · Lᵀ ≈ A.
    let mut lt = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            lt[i * n + j] = l[j * n + i];
        }
    }
    let llt = matmul(&l, &lt, n, n, n);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (llt[i * n + j] - a[i * n + j]).abs() < 1e-10,
                "(L·Lᵀ)[{i},{j}] = {} vs A = {}",
                llt[i * n + j],
                a[i * n + j]
            );
        }
    }
}

#[test]
fn solve_triangular_solves_lower_system() {
    rlx_linalg::register();
    // L · X = B with L lower triangular.
    let n = 4;
    let l: Vec<f64> = vec![
        2.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, -0.5, 1.0, 4.0, 0.0, 0.25, -1.5, 0.5, 2.0,
    ];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut g = Graph::new("trsm");
    let l_n = const_f64(&mut g, &l, &[n, n]);
    let b_n = const_f64(&mut g, &b, &[n, 1]);
    let x = rlx_linalg::solve_triangular(
        &mut g, l_n, b_n, /*lower=*/ true, /*transpose=*/ false,
    );
    g.set_outputs(vec![x]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let x = bytes_to_f64s(&c.run_typed(&[])[0].0);
    assert_eq!(x.len(), n);

    // Verify L · x ≈ b.
    let lx = matmul(&l, &x, n, n, 1);
    for i in 0..n {
        assert!(
            (lx[i] - b[i]).abs() < 1e-12,
            "L·x[{i}] = {} vs b = {}",
            lx[i],
            b[i]
        );
    }
}

#[test]
fn eigh_diagonalizes_symmetric_matrix() {
    rlx_linalg::register();
    let a = build_spd_4();
    let n = 4;

    let mut g = Graph::new("eigh");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let (eigvals, eigvecs) = rlx_linalg::eigh(&mut g, a_n);
    g.set_outputs(vec![eigvals, eigvecs]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let lambda = bytes_to_f64s(&outs[0].0);
    let v = bytes_to_f64s(&outs[1].0);
    assert_eq!(lambda.len(), n);
    assert_eq!(v.len(), n * n);

    // Eigenvalues must be ascending.
    for i in 1..n {
        assert!(
            lambda[i] >= lambda[i - 1] - 1e-12,
            "eigvals not ascending: λ[{}]={}, λ[{}]={}",
            i - 1,
            lambda[i - 1],
            i,
            lambda[i]
        );
    }
    // Eigenvalues must be > 0 (SPD).
    for &l in &lambda {
        assert!(l > 0.0, "SPD matrix has non-positive eigenvalue: {l}");
    }
    // Verify A · v_i ≈ λ_i · v_i for each row of `eigvecs` (which the
    // module doc says holds eigenvectors as rows in row-major view —
    // i.e., the col-major from LAPACK transposed).
    for i in 0..n {
        let mut av = vec![0f64; n];
        for r in 0..n {
            let mut s = 0f64;
            for c in 0..n {
                s += a[r * n + c] * v[i * n + c];
            }
            av[r] = s;
        }
        // Compare against λ_i · v_i.
        for r in 0..n {
            let want = lambda[i] * v[i * n + r];
            assert!(
                (av[r] - want).abs() < 1e-9,
                "A·v_{i}[{r}] = {} vs λ·v = {}",
                av[r],
                want
            );
        }
    }
}

#[test]
fn qr_factorizes_tall_matrix() {
    rlx_linalg::register();
    // 5×3 matrix; k = min(5, 3) = 3.
    let m: usize = 5;
    let n: usize = 3;
    let a: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        13.0, // <- last entry tweaked so A is full-rank
        14.0, 16.0, 19.0,
    ];

    let mut g = Graph::new("qr");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let (q, r) = rlx_linalg::qr(&mut g, a_n);
    g.set_outputs(vec![q, r]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let q = bytes_to_f64s(&outs[0].0);
    let r = bytes_to_f64s(&outs[1].0);
    assert_eq!(q.len(), m * n);
    assert_eq!(r.len(), n * n);

    // Q · R ≈ A.
    let qr = matmul(&q, &r, m, n, n);
    for i in 0..m {
        for j in 0..n {
            assert!(
                (qr[i * n + j] - a[i * n + j]).abs() < 1e-9,
                "(Q·R)[{i},{j}] = {} vs A = {}",
                qr[i * n + j],
                a[i * n + j]
            );
        }
    }
    // Qᵀ · Q ≈ I (orthonormality of columns).
    let mut qt = vec![0f64; n * m];
    for i in 0..m {
        for j in 0..n {
            qt[j * m + i] = q[i * n + j];
        }
    }
    let qtq = matmul(&qt, &q, n, m, n);
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq[i * n + j] - want).abs() < 1e-9,
                "(Qᵀ·Q)[{i},{j}] = {} vs {want}",
                qtq[i * n + j]
            );
        }
    }
}

#[test]
fn svd_decomposes_matrix() {
    rlx_linalg::register();
    // 3×4 matrix; k = min(3, 4) = 3.
    let m: usize = 3;
    let n: usize = 4;
    let a: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
    ];

    let mut g = Graph::new("svd");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let (u, s, vt) = rlx_linalg::svd(&mut g, a_n);
    g.set_outputs(vec![u, s, vt]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let u = bytes_to_f64s(&outs[0].0);
    let s = bytes_to_f64s(&outs[1].0);
    let vt = bytes_to_f64s(&outs[2].0);
    let k = m.min(n);
    assert_eq!(u.len(), m * k);
    assert_eq!(s.len(), k);
    assert_eq!(vt.len(), k * n);

    // Singular values descending.
    for i in 1..k {
        assert!(
            s[i] <= s[i - 1] + 1e-12,
            "S not descending: s[{}]={}, s[{}]={}",
            i - 1,
            s[i - 1],
            i,
            s[i]
        );
    }
    // U·diag(S)·Vᵀ ≈ A.
    let mut us = vec![0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] = u[i * k + j] * s[j];
        }
    }
    let usvt = matmul(&us, &vt, m, k, n);
    for i in 0..m {
        for j in 0..n {
            assert!(
                (usvt[i * n + j] - a[i * n + j]).abs() < 1e-9,
                "(U·diag(S)·Vᵀ)[{i},{j}] = {} vs A = {}",
                usvt[i * n + j],
                a[i * n + j]
            );
        }
    }
}
