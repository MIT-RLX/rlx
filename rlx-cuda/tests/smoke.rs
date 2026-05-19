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

//! Smoke tests for the CUDA backend.
//!
//! Every test starts with `if !rlx_cuda::is_available() { return; }` —
//! the crate compiles fine on Mac (and any other CUDA-less host) via
//! cudarc's dynamic-loading, so unit-test runs on those machines just
//! no-op. On a real CUDA box the same tests dispatch and assert on
//! actual GPU output.

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn binary_add_matches_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[
        ("x", &[1.0_f32, 2.0, 3.0, 4.0]),
        ("y", &[10.0_f32, 20.0, 30.0, 40.0]),
    ]);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn relu_clamps_negatives_to_zero() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("relu");
    let x = g.input("x", Shape::new(&[5], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[5], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[("x", &[-2.0_f32, -0.5, 0.0, 1.0, 3.0])]);
    assert_eq!(out[0], vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn matmul_2x3x2_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param("w", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let xv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outs = exe.run(&[("x", &xv)]);
    // Reference: row-major matmul.
    let mut want = vec![0.0_f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                want[i * 2 + j] += xv[i * 3 + k] * [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][k * 2 + j];
            }
        }
    }
    assert!(
        close(&outs[0], &want, 1e-4),
        "matmul mismatch: got {:?} want {want:?}",
        outs[0]
    );
}
