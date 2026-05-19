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

//! End-to-end Metal dispatch test for `Op::Custom` via `rlx-sparse`.
//!
//! Validates the chain:
//!   1. `rlx_sparse::register()` registers `MetalKernel` impls when
//!      built with the `metal` feature on Apple Silicon.
//!   2. The graph compiles for `Device::Metal` because `Custom` is in
//!      `METAL_SUPPORTED_OPS`.
//!   3. `compile_thunks` resolves the registered `MetalKernel` and
//!      packs it into `Thunk::CustomOp`.
//!   4. The Metal executor's owned-encoder refactor lets it commit
//!      the current cmd_buf, wait, run the kernel against the
//!      unified-memory arena, and rebind cmd_buf for any subsequent
//!      thunks. No silent-no-op fallback; if the dispatch breaks, the
//!      test panics or returns wrong values.
//!
//! Same SPD tridiagonal system as `cpu_sparse_ops.rs` so the Metal
//! result must match the CPU result element-wise to f64 precision.

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
    // Metal's `run_typed` widens host inputs to f32 today (see
    // backend.rs::widen_bytes_to_f32). F64 inputs through the host
    // I/O surface need a dedicated direct-byte path that's an
    // orthogonal extension. For this test we encode F64 values as
    // `Op::Constant` so the whole graph is self-contained — the
    // unit under test is the Metal Op::Custom dispatch, not the
    // Metal I/O surface.
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
fn sparse_lu_solve_runs_on_metal_via_owned_encoder_path() {
    // The owned-encoder + cmd_buf-rebind path is what makes this
    // test possible. If that refactor regresses, this test would
    // either fail to compile (borrow error in encode_commit) or
    // panic at run time with the old "executor refactor needed"
    // message.
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();

    let _ = nnz; // values are baked as a constant for this test
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];

    let mut g = Graph::new("metal_sparse_lu");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = const_f64(&mut g, &b_data);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.solve(&mut g, b);
    g.set_outputs(vec![x]);

    let mut compiled = Session::new(Device::Metal).compile(g);
    let outs = compiled.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F64);
    let x_got = bytes_to_f64s(&outs[0].0);

    // Verify A·x ≈ b — same correctness check as the CPU test.
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
            "metal residual at row {i}: A·x = {s}, b = {}",
            b_data[i]
        );
    }
}

#[test]
fn sparse_mat_vec_runs_on_metal_and_matches_cpu() {
    // Cross-backend parity: build the same graph for CPU and Metal,
    // run both, demand element-wise agreement to f64 precision. This
    // is the strictest possible check that Metal dispatch is correct.
    rlx_sparse::register();

    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let _nnz = values.len();
    let x_data = [1.5_f64, -0.25, 2.0, 0.75];

    let build = || {
        let mut g = Graph::new("matvec");
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
    let mut mtl = Session::new(Device::Metal).compile(build());
    let cpu_out = cpu.run_typed(&[]);
    let mtl_out = mtl.run_typed(&[]);

    let cpu_y = bytes_to_f64s(&cpu_out[0].0);
    let mtl_y = bytes_to_f64s(&mtl_out[0].0);
    assert_eq!(cpu_y.len(), mtl_y.len());
    for i in 0..cpu_y.len() {
        // f64 arithmetic is bit-exact across CPU/Metal — both paths
        // run the same scalar f64 algorithm in `algos::mat_vec`.
        assert_eq!(
            cpu_y[i], mtl_y[i],
            "matvec[{i}]: cpu={} metal={}",
            cpu_y[i], mtl_y[i]
        );
    }
}
