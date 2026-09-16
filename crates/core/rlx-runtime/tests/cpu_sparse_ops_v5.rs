// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse Cholesky (direct, SPD) + LSQR (least-squares).

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use rlx_sparse::SparseTensor;

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
fn const_i32(g: &mut Graph, xs: &[i32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::I32),
    )
}
fn const_f64(g: &mut Graph, xs: &[f64]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F64),
    )
}

fn build_spd_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    // SPD tridiagonal.
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

fn densify(values: &[f64], col_idx: &[i32], row_ptr: &[i32], n: usize) -> Vec<f64> {
    let mut a = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a[r * n + col_idx[k] as usize] = values[k];
        }
    }
    a
}

// ── Sparse Cholesky ───────────────────────────────────────────────

/// One factorisation, several right-hand sides.
///
/// A direct method earns its `O(n³)` factorisation by amortising it: every
/// solve after the first costs `O(n²)`. A caller that has `k` right-hand
/// sides for one matrix — a batch of loads, a time series, a reconstruction
/// solving the same system per slice — should pay for one factorisation, not
/// `k` of them.
///
/// This checks the two routes agree exactly, since they run the same LAPACK
/// on the same factor: a block solve must be indistinguishable from solving
/// its columns one at a time.
#[test]
fn cholesky_solves_a_block_of_right_hand_sides_from_one_factorisation() {
    rlx_sparse::register();
    // A small SPD matrix: 4×4, diagonally dominant.
    let n = 4;
    let values = vec![4.0, 1.0, 1.0, 5.0, 1.0, 1.0, 6.0, 1.0, 1.0, 7.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];

    let solve = |b: &[f64]| -> Vec<f64> {
        let mut g = Graph::new("chol_block");
        let v_n = const_f64(&mut g, &values);
        let ci_n = const_i32(&mut g, &col_idx);
        let rp_n = const_i32(&mut g, &row_ptr);
        let b_n = const_f64(&mut g, b);
        let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
        let x = st.cholesky_solve(&mut g, b_n);
        g.set_outputs(vec![x]);
        let mut c = Session::new(Device::Cpu).compile(g);
        bytes_to_f64s(&c.run_typed(&[])[0].0)
    };

    // Three right-hand sides, as an n × 3 row-major block.
    let cols: [[f64; 4]; 3] = [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, -1.0],
        [2.5, -1.5, 0.5, 1.0],
    ];
    let mut block = vec![0f64; n * 3];
    for (j, c) in cols.iter().enumerate() {
        for (i, v) in c.iter().enumerate() {
            block[i * 3 + j] = *v;
        }
    }
    let got = solve(&block);
    assert_eq!(got.len(), n * 3);

    for (j, c) in cols.iter().enumerate() {
        let alone = solve(c);
        for i in 0..n {
            assert!(
                (got[i * 3 + j] - alone[i]).abs() < 1e-12,
                "column {j} of the block differs from solving it alone: \
                 {} vs {}",
                got[i * 3 + j],
                alone[i]
            );
        }
    }

    // And the answers actually solve the system.
    for (j, c) in cols.iter().enumerate() {
        for r in 0..n {
            let mut acc = 0f64;
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                acc += values[k] * got[col_idx[k] as usize * 3 + j];
            }
            assert!(
                (acc - c[r]).abs() < 1e-10,
                "row {r} of column {j}: {acc} vs {}",
                c[r]
            );
        }
    }
}

#[test]
fn cholesky_solves_spd_correctly() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut g = Graph::new("chol_fwd");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.cholesky_solve(&mut g, b_n);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);

    let a = densify(&values, &col_idx, &row_ptr, n);
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[i * n + j] * x_got[j];
        }
        assert!((acc - b[i]).abs() < 1e-10, "A·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn cholesky_vjp_db_matches_fd() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let build = || {
        let mut g = Graph::new("chol_grad");
        let v_n = const_f64(&mut g, &values);
        let ci_n = const_i32(&mut g, &col_idx);
        let rp_n = const_i32(&mut g, &row_ptr);
        let b_n = g.input("b", Shape::new(&[n], DType::F64));
        let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
        let x = st.cholesky_solve(&mut g, b_n);
        let loss = g.sum(x, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, b_n)
    };
    let (g, b_n) = build();
    let bwd = grad_with_loss(&g, &[b_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("b", &f64s_to_bytes(&b0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut fd = vec![0f64; n];
    for i in 0..n {
        let mut bp = b0.clone();
        bp[i] += h;
        let mut bm = b0.clone();
        bm[i] -= h;
        let lp = run_chol_loss(&values, &col_idx, &row_ptr, &bp, n);
        let lm = run_chol_loss(&values, &col_idx, &row_ptr, &bm, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db[i] - fd[i]).abs() < 1e-7,
            "chol dL/db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

fn run_chol_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64], n: usize) -> f64 {
    rlx_sparse::register();
    let mut g = Graph::new("chol_loss");
    let v_n = const_f64(&mut g, values);
    let ci_n = const_i32(&mut g, col_idx);
    let rp_n = const_i32(&mut g, row_ptr);
    let b_n = g.input("b", Shape::new(&[n], DType::F64));
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.cholesky_solve(&mut g, b_n);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("b", &f64s_to_bytes(b), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── LSQR ──────────────────────────────────────────────────────────

/// LSQR must stop when it has converged, not when it runs out of iterations.
///
/// The residual `‖A·x − b‖` reaches zero only for a *consistent* system. A
/// least-squares problem is overdetermined and inconsistent by construction —
/// that is what makes it a least-squares problem — so its residual converges
/// to a nonzero minimum, and a stopping test on `‖r‖` alone can never fire.
/// The solve then runs to `max_iter` every time: correct answer, arbitrary
/// cost.
///
/// The test that does fire is on `‖Aᵀ·r‖`, the gradient of `½‖A·x − b‖²`.
/// This checks it by raising the cap far beyond what the problem needs: a
/// solver that stops on convergence takes the same time either way, and one
/// that does not takes a hundred times longer.
#[test]
fn lsqr_stops_on_convergence_rather_than_on_the_iteration_cap() {
    rlx_sparse::register();
    // Deliberately inconsistent: 5 equations, 3 unknowns, no exact solution.
    let (m, n) = (5, 3);
    let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
    let row_ptr = vec![0, 2, 4, 6, 8, 10];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 0.5, 1.5];

    let run = |cap: u32| {
        let mut g = Graph::new("lsqr_cap");
        let v_n = const_f64(&mut g, &values);
        let ci_n = const_i32(&mut g, &col_idx);
        let rp_n = const_i32(&mut g, &row_ptr);
        let b_n = const_f64(&mut g, &b);
        let st = SparseTensor::from_csr(v_n, ci_n, rp_n, m, n);
        let x = st.lsqr_solve(&mut g, b_n, cap, 1e-12);
        g.set_outputs(vec![x]);
        let mut c = Session::new(Device::Cpu).compile(g);
        let t0 = std::time::Instant::now();
        let outs = c.run_typed(&[]);
        (bytes_to_f64s(&outs[0].0), t0.elapsed())
    };

    let (small, t_small) = run(50);
    let (large, t_large) = run(50_000);

    // Same answer from both, so the extra budget bought nothing.
    for (a, c) in small.iter().zip(&large) {
        assert!((a - c).abs() < 1e-9, "{small:?} vs {large:?}");
    }
    // And cost nothing: a thousandfold cap must not mean a thousandfold wait.
    // The bound is loose because it is a timing test; the failure it catches
    // is three orders of magnitude, not three per cent.
    assert!(
        t_large < t_small * 20 + std::time::Duration::from_millis(50),
        "1000x the iteration cap took {t_large:?} against {t_small:?}, so the \
         solve is running to the cap instead of stopping when it converges"
    );
}

#[test]
fn lsqr_solves_overdetermined_least_squares() {
    rlx_sparse::register();
    // 5×3 system, full column rank. Reference solution computed via
    // normal equations on the dense matrix.
    let m = 5;
    let n = 3;
    // Sparse pattern: entries in arbitrary positions.
    let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
    let row_ptr = vec![0, 2, 4, 6, 8, 10];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 0.5, 1.5];

    let mut g = Graph::new("lsqr_fwd");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, m, n);
    let x = st.lsqr_solve(&mut g, b_n, /*max_iter=*/ 200, /*tol=*/ 1e-12);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(x_got.len(), n);

    // Reference: solve normal equations directly.
    let a = {
        let mut a = vec![0f64; m * n];
        for r in 0..m {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                a[r * n + col_idx[k] as usize] = values[k];
            }
        }
        a
    };
    // Aᵀ·A and Aᵀ·b
    let mut ata = vec![0f64; n * n];
    let mut atb = vec![0f64; n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..m {
                ata[i * n + j] += a[l * n + i] * a[l * n + j];
            }
        }
        for l in 0..m {
            atb[i] += a[l * n + i] * b[l];
        }
    }
    // Verify Aᵀ·A·x = Aᵀ·b (normal equations).
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += ata[i * n + j] * x_got[j];
        }
        assert!(
            (acc - atb[i]).abs() < 1e-8,
            "(AᵀA·x)[{i}]={} (Aᵀb)[{i}]={}",
            acc,
            atb[i]
        );
    }
}

#[test]
fn lsqr_solves_square_system_consistent_b() {
    // Square SPD-ish system; LSQR should converge to A⁻¹·b.
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut g = Graph::new("lsqr_sq");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.lsqr_solve(&mut g, b_n, 500, 1e-12);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);

    let a = densify(&values, &col_idx, &row_ptr, n);
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[i * n + j] * x_got[j];
        }
        assert!((acc - b[i]).abs() < 1e-7, "A·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn lsqr_handles_zero_rhs() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b: Vec<f64> = vec![0.0; n];

    let mut g = Graph::new("lsqr_zero");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.lsqr_solve(&mut g, b_n, 100, 1e-12);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);
    for i in 0..n {
        assert!(x_got[i].abs() < 1e-12, "lsqr(0)[{i}]={}", x_got[i]);
    }
}
