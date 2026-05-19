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

//! Sparse improvements: BiCGSTAB, ILU(0)-PCG, SpGEMM.

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

fn build_nonsym_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    // 4×4 non-symmetric, well-conditioned. Diagonally dominant.
    // [ 4 -1  0  2]
    // [-1  5 -1  0]
    // [ 0  3  6 -2]
    // [ 1  0 -1  4]
    let values = vec![
        4.0, -1.0, 2.0, -1.0, 5.0, -1.0, 3.0, 6.0, -2.0, 1.0, -1.0, 4.0,
    ];
    let col_idx = vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3];
    let row_ptr = vec![0, 3, 6, 9, 12];
    (values, col_idx, row_ptr)
}

fn build_spd_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    // 4×4 SPD tridiagonal
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

// ── BiCGSTAB ──────────────────────────────────────────────────────

#[test]
fn bicgstab_solves_nonsymmetric_correctly() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let b: Vec<f64> = vec![3.0, 1.0, 2.0, -1.0];

    let mut g = Graph::new("bicgstab_fwd");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.bicgstab_solve(&mut g, b_n, /*max_iter=*/ 200, /*tol=*/ 1e-12);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);

    // Verify A·x ≈ b (densify A first).
    let mut a = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a[r * n + col_idx[k] as usize] = values[k];
        }
    }
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[i * n + j] * x_got[j];
        }
        assert!((acc - b[i]).abs() < 1e-9, "A·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn bicgstab_transpose_solves_a_transpose() {
    // Use the backend kernel directly via a graph with attrs.transpose=true.
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut g = Graph::new("bicgstab_t");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let mut attrs = Vec::with_capacity(13);
    attrs.extend_from_slice(&200u32.to_le_bytes());
    attrs.extend_from_slice(&1e-12f64.to_le_bytes());
    attrs.push(1); // transpose_a
    let x = g.custom_op(
        rlx_sparse::SPARSE_BICGSTAB_SOLVE,
        attrs,
        vec![v_n, ci_n, rp_n, b_n],
    );
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);

    // Reference: Aᵀ·x = b.
    let mut a = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a[r * n + col_idx[k] as usize] = values[k];
        }
    }
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[j * n + i] * x_got[j];
        } // Aᵀ
        assert!((acc - b[i]).abs() < 1e-9, "Aᵀ·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn bicgstab_vjp_b_matches_fd() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let b0: Vec<f64> = vec![3.0, 1.0, 2.0, -1.0];

    let build = || {
        let mut g = Graph::new("bicgstab_grad");
        let v_n = const_f64(&mut g, &values);
        let ci_n = const_i32(&mut g, &col_idx);
        let rp_n = const_i32(&mut g, &row_ptr);
        let b_n = g.input("b", Shape::new(&[n], DType::F64));
        let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
        let x = st.bicgstab_solve(&mut g, b_n, 200, 1e-12);
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
        let lp = run_bicgstab_loss(&values, &col_idx, &row_ptr, &bp, n);
        let lm = run_bicgstab_loss(&values, &col_idx, &row_ptr, &bm, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db[i] - fd[i]).abs() < 1e-6,
            "bicgstab dL/db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

fn run_bicgstab_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64], n: usize) -> f64 {
    rlx_sparse::register();
    let mut g = Graph::new("bicgstab_loss");
    let v_n = const_f64(&mut g, values);
    let ci_n = const_i32(&mut g, col_idx);
    let rp_n = const_i32(&mut g, row_ptr);
    let b_n = g.input("b", Shape::new(&[n], DType::F64));
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.bicgstab_solve(&mut g, b_n, 200, 1e-12);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("b", &f64s_to_bytes(b), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── ILU(0)-PCG ────────────────────────────────────────────────────

#[test]
fn ilu_pcg_solves_spd_correctly() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let mut g = Graph::new("ilu_pcg_fwd");
    let v_n = const_f64(&mut g, &values);
    let ci_n = const_i32(&mut g, &col_idx);
    let rp_n = const_i32(&mut g, &row_ptr);
    let b_n = const_f64(&mut g, &b);
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.ilu_pcg_solve(&mut g, b_n, 100, 1e-12);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);

    let mut a = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a[r * n + col_idx[k] as usize] = values[k];
        }
    }
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[i * n + j] * x_got[j];
        }
        assert!((acc - b[i]).abs() < 1e-9, "A·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn ilu_pcg_vjp_db_matches_fd() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_spd_4();
    let n = 4;
    let b0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let build = || {
        let mut g = Graph::new("ilu_pcg_grad");
        let v_n = const_f64(&mut g, &values);
        let ci_n = const_i32(&mut g, &col_idx);
        let rp_n = const_i32(&mut g, &row_ptr);
        let b_n = g.input("b", Shape::new(&[n], DType::F64));
        let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
        let x = st.ilu_pcg_solve(&mut g, b_n, 100, 1e-12);
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
        let lp = run_ilu_pcg_loss(&values, &col_idx, &row_ptr, &bp, n);
        let lm = run_ilu_pcg_loss(&values, &col_idx, &row_ptr, &bm, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db[i] - fd[i]).abs() < 1e-6,
            "ilu_pcg dL/db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

fn run_ilu_pcg_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64], n: usize) -> f64 {
    rlx_sparse::register();
    let mut g = Graph::new("ilu_pcg_loss");
    let v_n = const_f64(&mut g, values);
    let ci_n = const_i32(&mut g, col_idx);
    let rp_n = const_i32(&mut g, row_ptr);
    let b_n = g.input("b", Shape::new(&[n], DType::F64));
    let st = SparseTensor::from_csr(v_n, ci_n, rp_n, n, n);
    let x = st.ilu_pcg_solve(&mut g, b_n, 100, 1e-12);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("b", &f64s_to_bytes(b), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── SpGEMM (pure-Rust, no IR integration) ─────────────────────────

#[test]
fn spgemm_csr_matches_dense_matmul() {
    // A: 3×4 (row-major dense, then sparsified)
    // B: 4×2
    // C = A·B should be 3×2.
    let m = 3;
    let k = 4;
    let n = 2;
    let a_dense: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
    let b_dense: Vec<f64> = vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0];

    fn dense_to_csr(d: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        let mut vals = Vec::new();
        let mut cidx = Vec::new();
        let mut rptr = vec![0i32; rows + 1];
        for i in 0..rows {
            for j in 0..cols {
                let v = d[i * cols + j];
                if v != 0.0 {
                    vals.push(v);
                    cidx.push(j as i32);
                }
            }
            rptr[i + 1] = vals.len() as i32;
        }
        (vals, cidx, rptr)
    }
    let (av, ac, ar) = dense_to_csr(&a_dense, m, k);
    let (bv, bc, br) = dense_to_csr(&b_dense, k, n);

    let (cv, cc, cr) = rlx_sparse::spgemm_csr(&av, &ac, &ar, &bv, &bc, &br, m, k, n).unwrap();

    // Compute reference dense C.
    let mut c_ref = vec![0f64; m * n];
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                c_ref[i * n + j] += a_dense[i * k + l] * b_dense[l * n + j];
            }
        }
    }
    // Densify spgemm output.
    let mut c_got = vec![0f64; m * n];
    for i in 0..m {
        for kk in cr[i] as usize..cr[i + 1] as usize {
            c_got[i * n + cc[kk] as usize] = cv[kk];
        }
    }
    for i in 0..(m * n) {
        assert!(
            (c_got[i] - c_ref[i]).abs() < 1e-12,
            "spgemm[{i}]={} ref={}",
            c_got[i],
            c_ref[i]
        );
    }
}

#[test]
fn spgemm_csr_handles_zero_product_rows() {
    // A has a zero row. C should also have zero row.
    let m = 3;
    let k = 3;
    let n = 2;
    let a_dense: Vec<f64> = vec![
        1.0, 2.0, 0.0, 0.0, 0.0, 0.0, // zero row
        0.0, 3.0, 4.0,
    ];
    let b_dense: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    fn dense_to_csr(d: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        let mut vals = Vec::new();
        let mut cidx = Vec::new();
        let mut rptr = vec![0i32; rows + 1];
        for i in 0..rows {
            for j in 0..cols {
                let v = d[i * cols + j];
                if v != 0.0 {
                    vals.push(v);
                    cidx.push(j as i32);
                }
            }
            rptr[i + 1] = vals.len() as i32;
        }
        (vals, cidx, rptr)
    }
    let (av, ac, ar) = dense_to_csr(&a_dense, m, k);
    let (bv, bc, br) = dense_to_csr(&b_dense, k, n);
    let (_cv, _cc, cr) = rlx_sparse::spgemm_csr(&av, &ac, &ar, &bv, &bc, &br, m, k, n).unwrap();
    // Row 1 should have zero nnz.
    assert_eq!(cr[1], cr[2], "row 1 should be empty");
}
