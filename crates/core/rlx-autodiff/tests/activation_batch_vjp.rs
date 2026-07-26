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

//! VJP of the scalar activation batch vs finite differences. Softplus/Elu have
//! smooth gradients (`sigmoid` / `min(eˣ,1)`); Floor/Ceil/Sign are
//! piecewise-constant → exactly zero gradient.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let shape = g.node(y).shape.clone();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], shape);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![y2],
        Shape::from_dims(&[], DType::F32),
    )
}

fn grad(a: Activation, x_init: &[f32]) -> Vec<f32> {
    let n = x_init.len();
    let mut g = Graph::new("act_grad");
    let x = g.param("x", Shape::new(&[n], DType::F32));
    let y = g.add_node(Op::Activation(a), vec![x], Shape::new(&[n], DType::F32));
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[x]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("x", x_init);
    c.run(&[("d_output", &[1.0f32])])[1].clone()
}

fn loss_at(a: Activation, xv: &[f32]) -> f32 {
    let n = xv.len();
    let mut g = Graph::new("fwd");
    let xi = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.add_node(Op::Activation(a), vec![xi], Shape::new(&[n], DType::F32));
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("x", xv)])
        .pop()
        .unwrap()[0]
}

#[test]
fn softplus_elu_vjp_matches_fd() {
    // Avoid the exact-zero kink at x=0 where FD is ill-conditioned.
    let x_init: Vec<f32> = vec![-2.1, -0.7, 0.3, 1.4, 2.9, -1.6];
    for a in [Activation::Softplus, Activation::Elu] {
        let d = grad(a, &x_init);
        let eps = 1e-3f32;
        for i in 0..x_init.len() {
            let mut xp = x_init.clone();
            let mut xm = x_init.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd = (loss_at(a, &xp) - loss_at(a, &xm)) / (2.0 * eps);
            assert!(
                (fd - d[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
                "{a:?} grad[{i}]: analytic {} vs FD {fd}",
                d[i]
            );
        }
    }
}

#[test]
fn extra_activations_vjp_matches_fd() {
    // Values away from the HardSwish/HardSigmoid kinks at ±3 (and 0) where FD
    // is ill-conditioned.
    let x_init: Vec<f32> = vec![-2.0, -0.7, 0.4, 1.5, 4.5, -4.5];
    for a in [
        Activation::Erf,
        Activation::HardSwish,
        Activation::HardSigmoid,
        Activation::Mish,
        Activation::Softsign,
        Activation::LogSigmoid,
    ] {
        let d = grad(a, &x_init);
        let eps = 1e-3f32;
        for i in 0..x_init.len() {
            let mut xp = x_init.clone();
            let mut xm = x_init.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd = (loss_at(a, &xp) - loss_at(a, &xm)) / (2.0 * eps);
            assert!(
                (fd - d[i]).abs() <= 3e-2 * (1.0 + fd.abs()),
                "{a:?} grad[{i}]: analytic {} vs FD {fd}",
                d[i]
            );
        }
    }
}

#[test]
fn floor_ceil_sign_have_zero_grad() {
    let x_init: Vec<f32> = vec![-2.1, -0.7, 0.3, 1.4, 2.9, -1.6];
    for a in [Activation::Floor, Activation::Ceil, Activation::Sign] {
        let d = grad(a, &x_init);
        for (i, &g) in d.iter().enumerate() {
            assert_eq!(g, 0.0, "{a:?} grad[{i}] must be 0, got {g}");
        }
    }
}
