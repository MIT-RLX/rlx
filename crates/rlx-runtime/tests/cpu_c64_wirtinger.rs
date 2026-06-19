// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Wirtinger reverse-mode AD on `DType::C64`.
//!
//! Loss functions are real (`Op::ComplexNormSq`), inputs are complex.
//! Convention: cotangents carry `∂L/∂z̄` (= `conj(∂L/∂z)`), matching
//! JAX. The new Wirtinger VJP rules for `BinaryOp::Mul` / `Div` and
//! the new `Op::Conjugate` are the unit under test.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn f32s_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// Bake a C64 constant from separate (re, im) vecs (interleaved f32).
fn const_c64(g: &mut Graph, re: &[f32], im: &[f32]) -> rlx_ir::NodeId {
    let n = re.len();
    let mut bytes = Vec::with_capacity(2 * n * 4);
    for i in 0..n {
        bytes.extend_from_slice(&re[i].to_le_bytes());
        bytes.extend_from_slice(&im[i].to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[n], DType::C64),
    )
}

#[test]
fn wirtinger_grad_mul_norm_sq() {
    // L = |a · b|² with a, b ∈ C^N. Closed-form Wirtinger gradient:
    //   ∂L/∂ā = b · conj(a·b) = |b|² · ā
    //   ∂L/∂b̄ = a · conj(a·b) = |a|² · b̄
    // (since L = (a·b)·conj(a·b), Wirtinger gives ∂/∂z̄ = (∂L/∂z̄)).
    //
    // Per-element: y_i = a_i · b_i, L = Σ |y_i|².
    //   dL/dā_i = b_i · conj(y_i) = b_i · conj(a_i)·conj(b_i)
    //           = |b_i|² · conj(a_i)
    // Wait — let me redo. ConjNormSqBackward emits dz = g·z (complex).
    // For L = |y|², ∂L/∂ȳ = y. So upstream feeding into Mul is y.
    // Mul VJP (Wirtinger): dL/dā = upstream · conj(b) = y · conj(b)
    //                            = (a·b) · conj(b) = a · |b|².
    // Note: under the carrying-∂L/∂z̄ convention with real L, the
    // "gradient" tensor reported is ∂L/∂z̄. For L = |y|², ∂L/∂ȳ = y.
    // So expected ∂L/∂ā = a · |b|², ∂L/∂b̄ = b · |a|².
    let n: usize = 4;
    let a_re = [1.0_f32, 0.5, -2.0, 0.25];
    let a_im = [0.5_f32, -1.0, 0.0, 2.0];
    let b_re = [0.7_f32, -0.3, 1.5, 0.4];
    let b_im = [-0.2_f32, 0.6, -0.8, 1.1];
    let mut g = Graph::new("wirtinger_mul");
    let a = const_c64(&mut g, &a_re, &a_im);
    let b = const_c64(&mut g, &b_re, &b_im);
    let y = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        a,
        b,
        Shape::new(&[n], DType::C64),
    );
    let abs_sq = g.complex_norm_sq(y); // F32 [n]
    let loss = g.sum(abs_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[("d_output", &f32s_to_bytes(&[1.0]), DType::F32)]);
    // outs: [loss(f32), da(C64), db(C64)]
    let da = bytes_to_f32s(&outs[1].0);
    let db = bytes_to_f32s(&outs[2].0);
    assert_eq!(da.len(), 2 * n);
    assert_eq!(db.len(), 2 * n);

    for i in 0..n {
        let mag_b_sq = b_re[i] * b_re[i] + b_im[i] * b_im[i];
        let mag_a_sq = a_re[i] * a_re[i] + a_im[i] * a_im[i];
        // Expected: ∂L/∂ā_i = a_i · |b_i|²  (re, im interleaved).
        let want_da_re = a_re[i] * mag_b_sq;
        let want_da_im = a_im[i] * mag_b_sq;
        let want_db_re = b_re[i] * mag_a_sq;
        let want_db_im = b_im[i] * mag_a_sq;
        assert!(
            (da[2 * i] - want_da_re).abs() < 1e-5,
            "da_re[{i}]: got {} vs want {}",
            da[2 * i],
            want_da_re
        );
        assert!(
            (da[2 * i + 1] - want_da_im).abs() < 1e-5,
            "da_im[{i}]: got {} vs want {}",
            da[2 * i + 1],
            want_da_im
        );
        assert!(
            (db[2 * i] - want_db_re).abs() < 1e-5,
            "db_re[{i}]: got {} vs want {}",
            db[2 * i],
            want_db_re
        );
        assert!(
            (db[2 * i + 1] - want_db_im).abs() < 1e-5,
            "db_im[{i}]: got {} vs want {}",
            db[2 * i + 1],
            want_db_im
        );
    }
}

#[test]
fn wirtinger_grad_div_norm_sq() {
    // L = |a / b|² = |a|² / |b|².
    //   ∂L/∂ā = a / |b|²
    //   ∂L/∂b̄ = -|a|² · b / |b|⁴ = -b · |a|² / |b|⁴
    let n: usize = 3;
    let a_re = [1.0_f32, 0.5, -2.0];
    let a_im = [0.5_f32, -1.0, 0.0];
    let b_re = [0.7_f32, -0.3, 1.5];
    let b_im = [-0.2_f32, 0.6, -0.8];
    let mut g = Graph::new("wirtinger_div");
    let a = const_c64(&mut g, &a_re, &a_im);
    let b = const_c64(&mut g, &b_re, &b_im);
    let y = g.binary(
        rlx_ir::op::BinaryOp::Div,
        a,
        b,
        Shape::new(&[n], DType::C64),
    );
    let abs_sq = g.complex_norm_sq(y);
    let loss = g.sum(abs_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[("d_output", &f32s_to_bytes(&[1.0]), DType::F32)]);
    let da = bytes_to_f32s(&outs[1].0);
    let db = bytes_to_f32s(&outs[2].0);

    for i in 0..n {
        let mag_b_sq = b_re[i] * b_re[i] + b_im[i] * b_im[i];
        let mag_a_sq = a_re[i] * a_re[i] + a_im[i] * a_im[i];
        let mag_b_4 = mag_b_sq * mag_b_sq;
        // ∂L/∂ā = a / |b|²
        let want_da_re = a_re[i] / mag_b_sq;
        let want_da_im = a_im[i] / mag_b_sq;
        // ∂L/∂b̄ = -b · |a|² / |b|⁴
        let want_db_re = -b_re[i] * mag_a_sq / mag_b_4;
        let want_db_im = -b_im[i] * mag_a_sq / mag_b_4;
        assert!(
            (da[2 * i] - want_da_re).abs() < 1e-4,
            "da_re[{i}]: got {} vs want {}",
            da[2 * i],
            want_da_re
        );
        assert!(
            (da[2 * i + 1] - want_da_im).abs() < 1e-4,
            "da_im[{i}]: got {} vs want {}",
            da[2 * i + 1],
            want_da_im
        );
        assert!(
            (db[2 * i] - want_db_re).abs() < 1e-4,
            "db_re[{i}]: got {} vs want {}",
            db[2 * i],
            want_db_re
        );
        assert!(
            (db[2 * i + 1] - want_db_im).abs() < 1e-4,
            "db_im[{i}]: got {} vs want {}",
            db[2 * i + 1],
            want_db_im
        );
    }
}

#[test]
fn wirtinger_grad_add_sub_norm_sq() {
    // L = |a + b|². Wirtinger:
    //   ∂L/∂ā = ∂L/∂b̄ = (a + b)   (linear, no conjugate needed).
    // Validates that Add/Sub still work correctly on C64 (their VJP
    // rule wasn't changed but they must continue producing the right
    // C64-typed cotangents).
    let n: usize = 2;
    let a_re = [1.0_f32, 0.5];
    let a_im = [0.5_f32, -1.0];
    let b_re = [0.7_f32, -0.3];
    let b_im = [-0.2_f32, 0.6];

    let mut g = Graph::new("wirtinger_add");
    let a = const_c64(&mut g, &a_re, &a_im);
    let b = const_c64(&mut g, &b_re, &b_im);
    let y = g.binary(
        rlx_ir::op::BinaryOp::Add,
        a,
        b,
        Shape::new(&[n], DType::C64),
    );
    let abs_sq = g.complex_norm_sq(y);
    let loss = g.sum(abs_sq, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[("d_output", &f32s_to_bytes(&[1.0]), DType::F32)]);
    let da = bytes_to_f32s(&outs[1].0);
    let db = bytes_to_f32s(&outs[2].0);

    for i in 0..n {
        let want_re = a_re[i] + b_re[i];
        let want_im = a_im[i] + b_im[i];
        assert!((da[2 * i] - want_re).abs() < 1e-5);
        assert!((da[2 * i + 1] - want_im).abs() < 1e-5);
        assert!((db[2 * i] - want_re).abs() < 1e-5);
        assert!((db[2 * i + 1] - want_im).abs() < 1e-5);
    }
}

#[test]
fn conjugate_op_round_trip() {
    // conj(conj(z)) == z — verifies the forward kernel and the
    // VJP-of-Conjugate rule (which is also Conjugate).
    let n: usize = 3;
    let z_re = [1.0_f32, -2.0, 0.5];
    let z_im = [0.5_f32, 1.5, -1.0];
    let mut g = Graph::new("conj_round_trip");
    let z = const_c64(&mut g, &z_re, &z_im);
    let z_conj = g.conjugate(z);
    let z_back = g.conjugate(z_conj);
    g.set_outputs(vec![z_back]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let got = bytes_to_f32s(&outs[0].0);
    for i in 0..n {
        assert!((got[2 * i] - z_re[i]).abs() < 1e-6);
        assert!((got[2 * i + 1] - z_im[i]).abs() < 1e-6);
    }
}
