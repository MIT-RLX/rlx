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

//! New sparse ops: values gradient, LU general (non-symmetric), GMRES.
//!
//! Each test mirrors the pattern in `cpu_sparse_ops.rs`: build a
//! graph, run forward, run autodiff, compare gradient against
//! finite differences.

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
fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
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

/// 4×4 SPD tridiagonal (same as cpu_sparse_ops.rs).
fn build_tridiag_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

/// Same sparsity pattern, asymmetric values. Diagonally dominant
/// so direct LU + GMRES both converge cleanly.
///
///   [ 5  -1   0   0 ]
///   [-2   4  -1   0 ]
///   [ 0  -2   4  -1 ]
///   [ 0   0  -2   3 ]
fn build_nonsym_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![5.0, -1.0, -2.0, 4.0, -1.0, -2.0, 4.0, -1.0, -2.0, 3.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

/// Compute (values_T, col_idx_T, row_ptr_T) — CSR of `Aᵀ`.
/// `A` is given via its CSR triplet; `Aᵀ` is the same matrix with
/// rows and columns swapped, encoded in CSR. For square matrices
/// this is the standard CSR↔CSC conversion: walk the input's
/// triplet, for each non-zero `(r, c, v)` push it into the output
/// keyed by `c` instead of `r`.
fn transpose_csr(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    n: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let nnz = values.len();
    let mut t_count = vec![0i32; n]; // count per output-row (= input column)
    for &c in col_idx {
        t_count[c as usize] += 1;
    }
    let mut t_row_ptr = vec![0i32; n + 1];
    for r in 0..n {
        t_row_ptr[r + 1] = t_row_ptr[r] + t_count[r];
    }
    let mut t_col_idx = vec![0i32; nnz];
    let mut t_values = vec![0f64; nnz];
    let mut cursor = t_row_ptr.clone();
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            let c = col_idx[k] as usize;
            let pos = cursor[c] as usize;
            t_col_idx[pos] = r as i32;
            t_values[pos] = values[k];
            cursor[c] += 1;
        }
    }
    (t_values, t_col_idx, t_row_ptr)
}

// ── Values-gradient parity ────────────────────────────────────────

/// dL/dvalues for sparse_lu_solve, FD parity.
/// L = Σ x where x = A⁻¹ b.  Closed-form per non-zero k:
///   dL/dvalues[k] = -[solve(Aᵀ, [1,...,1])][row(k)] · x[col(k)]
/// We compare the autodiff output against central-difference
/// gradient of the same loss w.r.t. each value entry independently.
#[test]
fn sparse_lu_vjp_dvalues_matches_finite_differences() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("lu_dvalues");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, b);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    // Differentiate w.r.t. values + b. The `values` gradient is the
    // new VJP path under test; `b` gradient was already covered by
    // the v1 test.
    let bwd = grad_with_loss(&g, &[v, b]);
    assert_eq!(bwd.outputs.len(), 3, "[loss, dL/dvalues, dL/db]");

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_values = bytes_to_f64s(&outs[1].0);
    assert_eq!(d_values.len(), nnz);

    // FD reference: perturb each values[k] by ±h, re-solve, central-diff.
    let h = 1e-7;
    for k in 0..nnz {
        let mut vp = values.clone();
        vp[k] += h;
        let mut vm = values.clone();
        vm[k] -= h;
        let lp = run_lu_loss(&vp, &col_idx, &row_ptr, &b_data);
        let lm = run_lu_loss(&vm, &col_idx, &row_ptr, &b_data);
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (d_values[k] - fd).abs() < 5e-6,
            "d_values[{k}] (VJP) = {}, FD = {}, diff {}",
            d_values[k],
            fd,
            (d_values[k] - fd).abs()
        );
    }
}

fn run_lu_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64]) -> f64 {
    rlx_sparse::register();
    let n = b.len();
    let nnz = values.len();
    let mut g = Graph::new("lu_fwd");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let bn = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, bn);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

/// dL/dvalues for sparse_mat_vec, FD parity.
#[test]
fn sparse_mat_vec_vjp_dvalues_matches_finite_differences() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let mut g = Graph::new("matvec_dvalues");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let y = a.mat_vec(&mut g, x);
    let loss = g.sum(y, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[v, x]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let x_data = [1.0_f64, 0.5, -2.0, 3.0];
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("x", &f64s_to_bytes(&x_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_values = bytes_to_f64s(&outs[1].0);
    assert_eq!(d_values.len(), nnz);

    let h = 1e-7;
    for k in 0..nnz {
        let mut vp = values.clone();
        vp[k] += h;
        let mut vm = values.clone();
        vm[k] -= h;
        let lp = run_matvec_loss(&vp, &col_idx, &row_ptr, &x_data);
        let lm = run_matvec_loss(&vm, &col_idx, &row_ptr, &x_data);
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (d_values[k] - fd).abs() < 1e-6,
            "matvec d_values[{k}] (VJP) = {}, FD = {}",
            d_values[k],
            fd
        );
    }
}

fn run_matvec_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], x: &[f64]) -> f64 {
    rlx_sparse::register();
    let n = x.len();
    let nnz = values.len();
    let mut g = Graph::new("matvec_fwd");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let xn = g.input("x", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let y = a.mat_vec(&mut g, xn);
    let loss = g.sum(y, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("x", &f64s_to_bytes(x), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── Non-symmetric LU general ──────────────────────────────────────

#[test]
fn sparse_lu_general_forward_solves_nonsymmetric_system() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let nnz = values.len();
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);

    let mut g = Graph::new("lu_general_fwd");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", &vt);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let b = g.input("b", Shape::new(&[n], DType::F64));

    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
    let x = a.solve_general(&mut g, b, &at);
    g.set_outputs(vec![x]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("values_t", &f64s_to_bytes(&vt), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);
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
            "lu_general residual at row {i}: {s} vs {}",
            b_data[i]
        );
    }
}

#[test]
fn sparse_lu_general_vjp_db_uses_transpose_correctly() {
    // dL/db = solve(Aᵀ, dL/dy). For a non-symmetric A, this WOULD
    // be wrong if we reused A's triplet for the adjoint solve —
    // verify that lu_solve_general's VJP picks up the transpose
    // triplet by comparing against an independent FD reference.
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let nnz = values.len();
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);

    let mut g = Graph::new("lu_general_grad");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", &vt);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let b = g.input("b", Shape::new(&[n], DType::F64));

    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
    let x = a.solve_general(&mut g, b, &at);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[b]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("values_t", &f64s_to_bytes(&vt), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    // FD reference.
    let h = 1e-7;
    for i in 0..n {
        let mut bp = b_data.to_vec();
        bp[i] += h;
        let mut bm = b_data.to_vec();
        bm[i] -= h;
        let lp = run_lu_general_loss(&values, &col_idx, &row_ptr, &vt, &cit, &rpt, &bp);
        let lm = run_lu_general_loss(&values, &col_idx, &row_ptr, &vt, &cit, &rpt, &bm);
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (db[i] - fd).abs() < 5e-6,
            "lu_general db[{i}] (VJP) = {}, FD = {}",
            db[i],
            fd
        );
    }
}

fn run_lu_general_loss(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    vt: &[f64],
    cit: &[i32],
    rpt: &[i32],
    b: &[f64],
) -> f64 {
    rlx_sparse::register();
    let n = b.len();
    let nnz = values.len();
    let mut g = Graph::new("lu_general_fwd_only");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", vt);
    let cit_n = const_i32(&mut g, cit);
    let rpt_n = const_i32(&mut g, rpt);
    let bn = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
    let x = a.solve_general(&mut g, bn, &at);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("values_t", &f64s_to_bytes(vt), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── GMRES ─────────────────────────────────────────────────────────

#[test]
fn sparse_gmres_forward_matches_lu_on_nonsymmetric() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let nnz = values.len();
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);

    let mut g = Graph::new("gmres_vs_lu_general");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", &vt);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let b = g.input("b", Shape::new(&[n], DType::F64));

    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);

    let x_lu = a.solve_general(&mut g, b, &at);
    let x_gmres = a.gmres_solve(&mut g, b, /*max_iter=*/ 100, /*tol=*/ 1e-12, &at);
    g.set_outputs(vec![x_lu, x_gmres]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("values_t", &f64s_to_bytes(&vt), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);
    let x_lu_got = bytes_to_f64s(&outs[0].0);
    let x_gmres_got = bytes_to_f64s(&outs[1].0);

    for i in 0..n {
        // GMRES converges to within ~tol·κ in forward error; for
        // this 4×4 well-conditioned matrix the gap is well under 1e-9.
        assert!(
            (x_lu_got[i] - x_gmres_got[i]).abs() < 1e-9,
            "gmres vs lu[{i}]: lu={} gmres={}",
            x_lu_got[i],
            x_gmres_got[i]
        );
    }
}

#[test]
fn sparse_gmres_vjp_db_matches_finite_differences() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let nnz = values.len();
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);

    let mut g = Graph::new("gmres_grad");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", &vt);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let b = g.input("b", Shape::new(&[n], DType::F64));

    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
    let x = a.gmres_solve(&mut g, b, 200, 1e-14, &at);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[b]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("values_t", &f64s_to_bytes(&vt), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    let h = 1e-7;
    for i in 0..n {
        let mut bp = b_data.to_vec();
        bp[i] += h;
        let mut bm = b_data.to_vec();
        bm[i] -= h;
        let lp = run_gmres_loss(&values, &col_idx, &row_ptr, &vt, &cit, &rpt, &bp);
        let lm = run_gmres_loss(&values, &col_idx, &row_ptr, &vt, &cit, &rpt, &bm);
        let fd = (lp - lm) / (2.0 * h);
        // GMRES adjoint compounds residual error from forward + adjoint
        // solves; loosen tolerance vs direct LU.
        assert!(
            (db[i] - fd).abs() < 1e-4,
            "gmres db[{i}] (VJP) = {}, FD = {}",
            db[i],
            fd
        );
    }
}

fn run_gmres_loss(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    vt: &[f64],
    cit: &[i32],
    rpt: &[i32],
    b: &[f64],
) -> f64 {
    rlx_sparse::register();
    let n = b.len();
    let nnz = values.len();
    let mut g = Graph::new("gmres_fwd_only");
    let v = g.input("values", Shape::new(&[nnz], DType::F64));
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let vt_n = const_f64_input(&mut g, "values_t", vt);
    let cit_n = const_i32(&mut g, cit);
    let rpt_n = const_i32(&mut g, rpt);
    let bn = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
    let x = a.gmres_solve(&mut g, bn, 200, 1e-14, &at);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("values", &f64s_to_bytes(values), DType::F64),
        ("values_t", &f64s_to_bytes(vt), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

/// F64 host input (graph Input, not Constant) — values_T needs to
/// vary at run time alongside values for FD perturbation, so it
/// can't be baked as a Constant.
fn const_f64_input(g: &mut Graph, name: &str, xs: &[f64]) -> NodeId {
    let _ = xs; // length comes from caller; xs is only metadata here
    g.input(name, Shape::new(&[xs.len()], DType::F64))
}
