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

//! End-to-end MLX dispatch test for `Op::Custom` via `rlx-sparse`.
//!
//! Validates the chain:
//!   1. `rlx_sparse::register()` registers `MlxKernel` impls when
//!      built with the `mlx` feature on macOS.
//!   2. The graph compiles for `Device::Mlx` because `Custom` is in
//!      `MLX_SUPPORTED_OPS`.
//!   3. `rlx-mlx`'s `lower_with_env` resolves the registered
//!      `MlxKernel` via `lookup_mlx_kernel` and calls its `execute`
//!      method to produce the lazy `Array` for this `Op::Custom`
//!      node.
//!   4. The kernel reads input `Array` bytes (`Array::to_bytes`),
//!      runs the host f64 algorithm, and rebuilds the output as a
//!      fresh `Array` via `Array::from_bytes` — the lazy MLX graph
//!      absorbs it as just another operand.
//!
//! Same SPD tridiagonal system as `cpu_sparse_ops.rs` and
//! `metal_sparse_ops.rs` so MLX and CPU results must match
//! element-wise (both run the same f64 host code under different
//! dispatchers; bit-exact agreement is the strictest correctness
//! signal).

#![cfg(all(feature = "cpu", feature = "mlx", target_os = "macos"))]

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
    // MLX's host I/O surface widens non-F32 inputs through `Array::
    // from_bytes`, so F64 inputs *do* round-trip correctly via
    // `run_typed`. We use Op::Constant here to keep the test
    // self-contained and parallel to `metal_sparse_ops.rs`.
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

fn build_tridiag_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

#[test]
fn sparse_lu_solve_runs_on_mlx_via_lazy_array_dispatch() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];

    let mut g = Graph::new("mlx_sparse_lu");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = const_f64(&mut g, &b_data);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, b);
    g.set_outputs(vec![x]);

    let mut compiled = Session::new(Device::Mlx).compile(g);
    let outs = compiled.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F64);
    let x_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(x_got.len(), n);

    // A·x ≈ b — same correctness check as the CPU + Metal tests.
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
            "mlx residual at row {i}: A·x = {s}, b = {}",
            b_data[i]
        );
    }
}

#[test]
fn sparse_mat_vec_runs_on_mlx_and_matches_cpu() {
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let x_data = [1.5_f64, -0.25, 2.0, 0.75];

    let build = || {
        let mut g = Graph::new("mlx_matvec");
        let v = const_f64(&mut g, &values);
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let x = const_f64(&mut g, &x_data);
        let a = SparseTensor::from_csr(v, ci, rp, n, n);
        let y = a.mat_vec(&mut g, x);
        g.set_outputs(vec![y]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let mut mlx = Session::new(Device::Mlx).compile(build());
    let cpu_out = cpu.run_typed(&[]);
    let mlx_out = mlx.run_typed(&[]);

    let cpu_y = bytes_to_f64s(&cpu_out[0].0);
    let mlx_y = bytes_to_f64s(&mlx_out[0].0);
    assert_eq!(cpu_y.len(), mlx_y.len());
    for i in 0..cpu_y.len() {
        // Both paths run the same f64 scalar `algos::mat_vec` body —
        // bit-exact agreement is the contract.
        assert_eq!(
            cpu_y[i], mlx_y[i],
            "matvec[{i}]: cpu={} mlx={}",
            cpu_y[i], mlx_y[i]
        );
    }
}
