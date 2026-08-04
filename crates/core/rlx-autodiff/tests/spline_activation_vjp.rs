// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::SplineActivation` (KAN Gaussian-RBF spline): native CPU forward vs a
//! hand-written reference, and its VJP (differentiable in BOTH `x` and `coeff`)
//! vs finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const ROWS: usize = 4;
const CH: usize = 3;
const NB: u32 = 5;
const GMIN: f32 = -2.0;
const GMAX: f32 = 2.0;

fn x_data() -> Vec<f32> {
    (0..ROWS * CH)
        .map(|i| (i as f32 * 0.7).sin() * 1.2)
        .collect()
}
fn coeff_data() -> Vec<f32> {
    (0..CH * NB as usize)
        .map(|i| (i as f32 * 0.5).cos() * 0.8)
        .collect()
}

// Reference: y[r,c] = Σ_g coeff[c,g]·exp(-((x[r,c]-center_g)·inv_h)²)
fn reference(x: &[f32], coeff: &[f32]) -> Vec<f32> {
    let nb = NB as usize;
    let step = (GMAX - GMIN) / (nb as f32 - 1.0);
    let inv_h = 1.0 / step;
    let mut out = vec![0f32; ROWS * CH];
    for r in 0..ROWS {
        for c in 0..CH {
            let xv = x[r * CH + c];
            let mut acc = 0f32;
            for gi in 0..nb {
                let center = GMIN + gi as f32 * step;
                let z = (xv - center) * inv_h;
                acc += coeff[c * nb + gi] * (-(z * z)).exp();
            }
            out[r * CH + c] = acc;
        }
    }
    out
}

fn spline(g: &mut Graph, x: NodeId, coeff: NodeId) -> NodeId {
    g.spline_activation(x, coeff, NB, GMIN, GMAX)
}

#[test]
fn spline_forward_matches_reference() {
    let (x, coeff) = (x_data(), coeff_data());
    let mut g = Graph::new("spline_fwd");
    let xin = g.input("x", Shape::new(&[ROWS, CH], DType::F32));
    let cin = g.input("coeff", Shape::new(&[CH, NB as usize], DType::F32));
    let y = spline(&mut g, xin, cin);
    g.set_outputs(vec![y]);
    let out = rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("x", &x), ("coeff", &coeff)])
        .pop()
        .unwrap();
    let want = reference(&x, &coeff);
    assert_eq!(out.len(), want.len());
    for i in 0..out.len() {
        assert!(
            (out[i] - want[i]).abs() < 1e-5,
            "spline[{i}]: {} vs {}",
            out[i],
            want[i]
        );
    }
}

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let sh = g.node(y).shape.clone();
    let n: usize = (0..sh.rank()).map(|i| sh.dim(i).unwrap_static()).product();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], sh);
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64],
        },
        vec![y2],
        Shape::new(&[n], DType::F32),
    );
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    )
}

fn loss_at(xv: &[f32], cv: &[f32]) -> f32 {
    let mut g = Graph::new("spline_loss");
    let xin = g.input("x", Shape::new(&[ROWS, CH], DType::F32));
    let cin = g.input("coeff", Shape::new(&[CH, NB as usize], DType::F32));
    let y = spline(&mut g, xin, cin);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("x", xv), ("coeff", cv)])
        .pop()
        .unwrap()[0]
}

#[test]
fn spline_vjp_matches_fd() {
    let (x, coeff) = (x_data(), coeff_data());
    let mut g = Graph::new("spline_grad");
    let xp = g.param("x", Shape::new(&[ROWS, CH], DType::F32));
    let cp = g.param("coeff", Shape::new(&[CH, NB as usize], DType::F32));
    let y = spline(&mut g, xp, cp);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[xp, cp]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("x", &x);
    c.set_param("coeff", &coeff);
    let outs = c.run(&[("d_output", &[1.0f32])]);
    let (dx, dcoeff) = (&outs[1], &outs[2]);
    assert_eq!(dx.len(), x.len());
    assert_eq!(dcoeff.len(), coeff.len());

    let eps = 1e-3f32;
    // ∂loss/∂x
    for i in 0..x.len() {
        let (mut xp_, mut xm_) = (x.clone(), x.clone());
        xp_[i] += eps;
        xm_[i] -= eps;
        let fd = (loss_at(&xp_, &coeff) - loss_at(&xm_, &coeff)) / (2.0 * eps);
        assert!(
            (fd - dx[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "d/dx[{i}]: analytic {} vs FD {fd}",
            dx[i]
        );
    }
    // ∂loss/∂coeff
    for i in 0..coeff.len() {
        let (mut cp_, mut cm_) = (coeff.clone(), coeff.clone());
        cp_[i] += eps;
        cm_[i] -= eps;
        let fd = (loss_at(&x, &cp_) - loss_at(&x, &cm_)) / (2.0 * eps);
        assert!(
            (fd - dcoeff[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "d/dcoeff[{i}]: analytic {} vs FD {fd}",
            dcoeff[i]
        );
    }
}
