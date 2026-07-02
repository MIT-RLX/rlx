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

//! Metal end-to-end tests for the v2 sparse ops:
//! values gradient, non-symmetric LU general, GMRES.
//!
//! Same shape as `metal_sparse_ops.rs` — bit-exact CPU↔Metal parity
//! is the strictest correctness signal because both paths run the
//! same f64 host code under different dispatchers.

#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_sparse::SparseTensor;

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
    let values = vec![5.0, -1.0, -2.0, 4.0, -1.0, -2.0, 4.0, -1.0, -2.0, 3.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

fn transpose_csr(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    n: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let nnz = values.len();
    let mut t_count = vec![0i32; n];
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

#[test]
fn lu_general_runs_on_metal_and_matches_cpu() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];

    let build = || {
        let mut g = Graph::new("lu_general");
        let v = const_f64(&mut g, &values);
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let vt_n = const_f64(&mut g, &vt);
        let cit_n = const_i32(&mut g, &cit);
        let rpt_n = const_i32(&mut g, &rpt);
        let b = const_f64(&mut g, &b_data);
        let a = SparseTensor::from_csr(v, ci, rp, n, n);
        let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
        let x = a.solve_general(&mut g, b, &at);
        g.set_outputs(vec![x]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let mut mtl = Session::new(Device::Metal).compile(build());
    let cpu_x = bytes_to_f64s(&cpu.run_typed(&[])[0].0);
    let mtl_x = bytes_to_f64s(&mtl.run_typed(&[])[0].0);
    for i in 0..n {
        assert_eq!(
            cpu_x[i], mtl_x[i],
            "lu_general[{i}]: cpu={} metal={}",
            cpu_x[i], mtl_x[i]
        );
    }
}

#[test]
fn gmres_runs_on_metal_and_matches_cpu() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let (vt, cit, rpt) = transpose_csr(&values, &col_idx, &row_ptr, n);
    let b_data = [1.0_f64, 2.5, -1.0, 3.0];

    let build = || {
        let mut g = Graph::new("gmres");
        let v = const_f64(&mut g, &values);
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let vt_n = const_f64(&mut g, &vt);
        let cit_n = const_i32(&mut g, &cit);
        let rpt_n = const_i32(&mut g, &rpt);
        let b = const_f64(&mut g, &b_data);
        let a = SparseTensor::from_csr(v, ci, rp, n, n);
        let at = SparseTensor::from_csr(vt_n, cit_n, rpt_n, n, n);
        let x = a.gmres_solve(&mut g, b, 100, 1e-12, &at);
        g.set_outputs(vec![x]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let mut mtl = Session::new(Device::Metal).compile(build());
    let cpu_x = bytes_to_f64s(&cpu.run_typed(&[])[0].0);
    let mtl_x = bytes_to_f64s(&mtl.run_typed(&[])[0].0);
    for i in 0..n {
        // Bit-exact: same f64 GMRES algorithm running on host CPU
        // under both dispatchers.
        assert_eq!(
            cpu_x[i], mtl_x[i],
            "gmres[{i}]: cpu={} metal={}",
            cpu_x[i], mtl_x[i]
        );
    }
}

#[test]
fn values_grad_runs_on_metal_and_matches_cpu() {
    // Direct unit test for the values_grad op (rather than through
    // autodiff) — sidesteps Metal's F64-host-input gap which would
    // be required if we differentiated w.r.t. an Input. Both CPU
    // and Metal receive the same Constant inputs and should produce
    // bit-exact identical outputs.
    rlx_sparse::register();

    // CSR pattern (the same nnz=10 tridiagonal mask).
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    // Arbitrary u and v vectors (length n=4).
    let u = [0.5_f64, -1.5, 2.0, -0.25];
    let v = [1.0_f64, 3.0, -0.5, 2.5];
    let n = 4;
    let nnz = col_idx.len();

    let build = || {
        let mut g = Graph::new("values_grad_direct");
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let u_n = const_f64(&mut g, &u);
        let v_n = const_f64(&mut g, &v);
        // Direct call to the registered op; bypasses SparseTensor
        // since values_grad isn't on the public API (it's a VJP
        // building block).
        let out = g.custom_op(
            rlx_sparse::SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![ci, rp, u_n, v_n],
        );
        g.set_outputs(vec![out]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let mut mtl = Session::new(Device::Metal).compile(build());
    let cpu_out = bytes_to_f64s(&cpu.run_typed(&[])[0].0);
    let mtl_out = bytes_to_f64s(&mtl.run_typed(&[])[0].0);

    assert_eq!(cpu_out.len(), nnz);
    assert_eq!(mtl_out.len(), nnz);
    for k in 0..nnz {
        assert_eq!(
            cpu_out[k], mtl_out[k],
            "values_grad[{k}]: cpu={} metal={}",
            cpu_out[k], mtl_out[k]
        );
    }
    // Sanity: compute the expected per-row answers manually so the
    // whole thing isn't just "agreeing with itself."
    // out[k] = u[row(k)] * v[col_idx[k]]
    let row_of: Vec<usize> = {
        let mut r_of_k = vec![0usize; nnz];
        for r in 0..n {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                r_of_k[k] = r;
            }
        }
        r_of_k
    };
    for k in 0..nnz {
        let want = u[row_of[k]] * v[col_idx[k] as usize];
        assert!(
            (cpu_out[k] - want).abs() < 1e-12,
            "values_grad[{k}]: got {}, expected {}",
            cpu_out[k],
            want
        );
    }
}
