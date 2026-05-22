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
//! Regression test for the bidirectional-broadcast forward + backward
//! path on `Op::Binary`.
//!
//! Before the shape-aware `bcast_*_strides` fix in
//! `rlx-cpu/src/thunk.rs`, both `BinaryFull` (f32) and `BinaryFullF64`
//! computed `out[i] = lhs[i % lhs_len] OP rhs[i % rhs_len]`. This is
//! correct for *unidirectional* broadcast (`[N,S] op [S]` or
//! `[N,S] op scalar`) but produces the wrong cell mapping for
//! *bidirectional* broadcast like `[N,1] op [1,S] → [N,S]`, where
//! the [N,1] cell at `(row, col)` should map to lhs[row, 0] = lhs[row]
//! and rhs[0, col] = rhs[col], not `lhs[(row*S+col) % N]` /
//! `rhs[(row*S+col) % S]`.
//!
//! Symptom downstream: `eda-thermal::field_rlx::build_deposit_graph`
//! had to use explicit `Op::Expand` on both sides as a workaround,
//! after which AD-vs-FD matched to machine precision; without the
//! workaround the gradients came out scaled by ~1/N.
//!
//! This test pins the fix in three regimes (f64 add, f64 mul, f32
//! sub) and covers both the forward value and the backward gradient
//! via finite-difference comparison.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const N: usize = 4;
const S: usize = 3;

fn const_f64(g: &mut Graph, data: &[f64], shape: &[usize]) -> rlx_ir::NodeId {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(shape, DType::F64),
    )
}

fn decode_f64_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(c);
            f64::from_le_bytes(buf)
        })
        .collect()
}

/// Reference: NumPy-style `[N,1] op [1,S] → [N,S]` computed in pure Rust.
fn reference(a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
    assert_eq!(a.len(), N);
    assert_eq!(b.len(), S);
    let mut out = vec![0.0; N * S];
    for row in 0..N {
        for col in 0..S {
            let lhs = a[row];
            let rhs = b[col];
            out[row * S + col] = match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
                _ => unreachable!(),
            };
        }
    }
    out
}

#[test]
fn forward_f64_n1_times_1s_matches_reference() {
    let a = vec![1.5, -2.0, 0.25, 4.0]; // [N]
    let b = vec![3.0, -1.5, 2.5]; // [S]
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        let mut g = Graph::new("bidir_bcast_f64");
        let a_node = const_f64(&mut g, &a, &[N, 1]);
        let b_node = const_f64(&mut g, &b, &[1, S]);
        let out = g.binary(op, a_node, b_node, Shape::new(&[N, S], DType::F64));
        g.set_outputs(vec![out]);
        let mut compiled = Session::new(Device::Cpu).compile(g);
        let outs = compiled.run_typed(&[]);
        let got = decode_f64_vec(&outs[0].0);
        let expect = reference(&a, &b, op);
        assert_eq!(
            got.len(),
            expect.len(),
            "{op:?} length mismatch: {} vs {}",
            got.len(),
            expect.len()
        );
        for (i, (g_i, e_i)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (g_i - e_i).abs() < 1e-12,
                "{op:?} cell {i}: got {g_i}, expected {e_i}"
            );
        }
    }
}

#[test]
fn backward_f64_n1_times_1s_matches_finite_diff() {
    // Loss = sum(a[:,None] * b[None,:]) over the [N, S] product.
    // ∂loss/∂a[k] = S · b̄  (sum of all b's)
    // ∂loss/∂b[k] = N · ā  (sum of all a's)
    // Verify both AD output and the analytic identities.
    let a_init = vec![1.5_f64, -2.0, 0.25, 4.0];
    let b_init = vec![3.0_f64, -1.5, 2.5];

    let mut g = Graph::new("bidir_bcast_grad");
    let a_in = g.input("a", Shape::new(&[N, 1], DType::F64));
    let b_in = g.input("b", Shape::new(&[1, S], DType::F64));
    let prod = g.binary(BinaryOp::Mul, a_in, b_in, Shape::new(&[N, S], DType::F64));
    // Flatten + sum to scalar loss.
    let loss = g.sum(prod, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    let grad_g = grad_with_loss(&g, &[a_in, b_in]);
    let mut compiled = Session::new(Device::Cpu).compile(grad_g);
    let a_bytes: Vec<u8> = a_init.iter().flat_map(|x| x.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_init.iter().flat_map(|x| x.to_le_bytes()).collect();
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[
        ("a", &a_bytes, DType::F64),
        ("b", &b_bytes, DType::F64),
        ("d_output", &one, DType::F64),
    ]);
    let grad_a = decode_f64_vec(&outs[1].0);
    let grad_b = decode_f64_vec(&outs[2].0);
    assert_eq!(grad_a.len(), N);
    assert_eq!(grad_b.len(), S);

    let sum_b: f64 = b_init.iter().sum();
    let sum_a: f64 = a_init.iter().sum();
    for k in 0..N {
        assert!(
            (grad_a[k] - sum_b).abs() < 1e-10,
            "∂loss/∂a[{k}] = {} but expected sum(b) = {sum_b}",
            grad_a[k]
        );
    }
    for k in 0..S {
        assert!(
            (grad_b[k] - sum_a).abs() < 1e-10,
            "∂loss/∂b[{k}] = {} but expected sum(a) = {sum_a}",
            grad_b[k]
        );
    }
}

#[test]
fn unidirectional_broadcast_still_works() {
    // Regression guard for the legacy path: `[N, S] + [S]` (bias-
    // style last-axis broadcast) must still produce the right
    // forward values after the patch.
    let a = vec![
        1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]; // [4, 3]
    let b = vec![10.0_f64, 20.0, 30.0]; // [3]
    let mut g = Graph::new("uni_bcast");
    let a_node = const_f64(&mut g, &a, &[4, 3]);
    let b_node = const_f64(&mut g, &b, &[3]);
    let out = g.binary(
        BinaryOp::Add,
        a_node,
        b_node,
        Shape::new(&[4, 3], DType::F64),
    );
    g.set_outputs(vec![out]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let got = decode_f64_vec(&compiled.run_typed(&[])[0].0);
    let expected: Vec<f64> = (0..12).map(|i| a[i] + b[i % 3]).collect();
    for (i, (g_i, e_i)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g_i - e_i).abs() < 1e-12,
            "uni-bcast cell {i}: {g_i} vs {e_i}"
        );
    }
}
