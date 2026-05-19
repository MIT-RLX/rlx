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

//! Sparse-ops integration tests against the `rlx-sparse` package.
//!
//! `rlx-sparse` defines `SparseTensor` plus three custom ops
//! (`sparse_lu_solve`, `sparse_mat_vec`, `sparse_cg_solve`) all
//! registered against rlx's framework-level extension scaffold. This
//! file imports the package and exercises forward + autodiff +
//! composition through the public CPU pipeline (`Session::new(Cpu)`).
//!
//! The point isn't to test rlx-sparse internals — that crate has
//! its own tests — but to validate that the JAX-shaped contract
//! works end-to-end: a downstream package registers its ops, the
//! framework dispatches them, autodiff routes through the registered
//! VJP rules, all without rlx core edits.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use rlx_sparse::SparseTensor;

// ── Test scaffolding ──────────────────────────────────────────────

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Embed an i32 1D tensor as `Op::Constant`. The CSR sparsity pattern
/// is fixed per graph, so it lives in the graph rather than crossing
/// the runtime I/O surface — matches the pattern documented in
/// `rlx-sparse`'s usage guide.
fn const_i32(g: &mut Graph, xs: &[i32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let shape = Shape::new(&[xs.len()], DType::I32);
    g.add_node(Op::Constant { data: bytes }, vec![], shape)
}

/// 4×4 SPD tridiagonal matrix (CSR). Symmetric → v1 VJPs (which
/// assume symmetry) are correct.
///
///   [ 4 -1  0  0 ]
///   [-1  4 -1  0 ]
///   [ 0 -1  4 -1 ]
///   [ 0  0 -1  4 ]
fn build_tridiag_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

// ── Sparse-LU tests ───────────────────────────────────────────────

#[test]
fn sparse_lu_forward_solves_tridiagonal_system() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("sparse_lu_fwd");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, b);
    g.set_outputs(vec![x]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);
    assert_eq!(outs.len(), 1);
    let x_got = bytes_to_f64s(&outs[0].0);

    // Verify A·x ≈ b.
    let mut a_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a_dense[r * n + col_idx[k] as usize] = values[k];
        }
    }
    for i in 0..n {
        let mut s = 0f64;
        for j in 0..n {
            s += a_dense[i * n + j] * x_got[j];
        }
        assert!(
            (s - b_data[i]).abs() < 1e-12,
            "residual at row {i}: A·x = {s}, b = {}",
            b_data[i]
        );
    }
}

#[test]
fn sparse_lu_vjp_db_matches_finite_differences() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("sparse_lu_grad");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, b);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[b]);
    assert_eq!(bwd.outputs.len(), 2);

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);
    let fd_db = finite_difference_db_lu(&values, &col_idx, &row_ptr, &b_data, 1e-7);

    for i in 0..n {
        assert!(
            (db[i] - fd_db[i]).abs() < 5e-6,
            "db[{i}] (VJP) = {}, db[{i}] (FD) = {} — diff {}",
            db[i],
            fd_db[i],
            (db[i] - fd_db[i]).abs()
        );
    }
}

fn finite_difference_db_lu(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    b: &[f64],
    h: f64,
) -> Vec<f64> {
    let n = b.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let mut bp = b.to_vec();
        bp[i] += h;
        let mut bm = b.to_vec();
        bm[i] -= h;
        out[i] = (run_lu_loss(values, col_idx, row_ptr, &bp)
            - run_lu_loss(values, col_idx, row_ptr, &bm))
            / (2.0 * h);
    }
    out
}

fn run_lu_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64]) -> f64 {
    rlx_sparse::register();
    let n = b.len();
    let nnz = values.len();
    let mut g = Graph::new("lu_fwd_only");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let bn = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, bn);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── Mat-vec tests ─────────────────────────────────────────────────

#[test]
fn sparse_mat_vec_forward_matches_dense_reference() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("sparse_matvec");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let y = a.mat_vec(&mut g, x);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let x_data = [1.0_f64, 0.5, -2.0, 3.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("x", &f64s_to_bytes(&x_data), DType::F64),
    ]);
    let y_got = bytes_to_f64s(&outs[0].0);

    let mut a_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a_dense[r * n + col_idx[k] as usize] = values[k];
        }
    }
    for i in 0..n {
        let mut s = 0f64;
        for j in 0..n {
            s += a_dense[i * n + j] * x_data[j];
        }
        assert!((y_got[i] - s).abs() < 1e-12);
    }
}

#[test]
fn sparse_tensor_solve_then_matvec_recovers_input() {
    // Composition test: y = A · A⁻¹ · b ≈ b. Two custom-op call sites
    // sharing input nodes through the SparseTensor wrapper.
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("sparse_roundtrip");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);

    let x_solve = a.solve(&mut g, b);
    let y_back = a.mat_vec(&mut g, x_solve);
    g.set_outputs(vec![y_back]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let b_data = [3.0_f64, -1.0, 2.5, 0.5];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);
    let y_back_got = bytes_to_f64s(&outs[0].0);

    for i in 0..n {
        assert!(
            (y_back_got[i] - b_data[i]).abs() < 1e-10,
            "round-trip[{i}]: y_back = {}, b = {}",
            y_back_got[i],
            b_data[i]
        );
    }
}

#[test]
fn sparse_mat_vec_vjp_matches_finite_differences() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("matvec_grad");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let y = a.mat_vec(&mut g, x);
    let loss = g.sum(y, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let x_data = [1.0_f64, 0.5, -2.0, 3.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("x", &f64s_to_bytes(&x_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dx_vjp = bytes_to_f64s(&outs[1].0);

    let mut a_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a_dense[r * n + col_idx[k] as usize] = values[k];
        }
    }
    let dx_expected: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| a_dense[i * n + j]).sum())
        .collect();

    for i in 0..n {
        assert!((dx_vjp[i] - dx_expected[i]).abs() < 1e-10);
    }
}

// ── CG tests ──────────────────────────────────────────────────────

#[test]
fn cg_solve_forward_matches_lu_solve_within_tolerance() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("cg_vs_lu");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);

    let x_lu = a.solve(&mut g, b);
    let x_cg = a.cg_solve(&mut g, b, 100, 1e-12);
    g.set_outputs(vec![x_lu, x_cg]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);
    let x_lu = bytes_to_f64s(&outs[0].0);
    let x_cg = bytes_to_f64s(&outs[1].0);

    for i in 0..n {
        assert!(
            (x_lu[i] - x_cg[i]).abs() < 1e-9,
            "x_lu[{i}] = {}, x_cg[{i}] = {}",
            x_lu[i],
            x_cg[i]
        );
    }
}

#[test]
fn cg_solve_vjp_db_matches_finite_differences() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("cg_grad");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.cg_solve(&mut g, b, 200, 1e-14);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[b]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db_vjp = bytes_to_f64s(&outs[1].0);

    let h = 1e-7;
    let mut db_fd = vec![0f64; n];
    for i in 0..n {
        let mut bp = b_data.to_vec();
        bp[i] += h;
        let mut bm = b_data.to_vec();
        bm[i] -= h;
        db_fd[i] = (run_cg_loss(&values, &col_idx, &row_ptr, &bp)
            - run_cg_loss(&values, &col_idx, &row_ptr, &bm))
            / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db_vjp[i] - db_fd[i]).abs() < 1e-5,
            "db[{i}] (CG VJP) = {}, db[{i}] (FD) = {}",
            db_vjp[i],
            db_fd[i]
        );
    }
}

fn run_cg_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64]) -> f64 {
    let n = b.len();
    let nnz = values.len();
    let mut g = Graph::new("cg_fwd_only");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let bn = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.cg_solve(&mut g, bn, 200, 1e-14);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}
