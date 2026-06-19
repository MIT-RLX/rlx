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

//! BatchNorm inference autodiff vs finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};

#[test]
fn batch_norm_inference_gamma_grad_matches_fd() {
    let b = 3usize;
    let c = 4usize;
    let eps = 1e-5f32;

    let mut g = Graph::new("bn");
    let x = g.input("x", Shape::new(&[b, c], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let mean = g.param("mean", Shape::new(&[c], DType::F32));
    let var = g.param("var", Shape::new(&[c], DType::F32));
    let y = g.batch_norm_inference(x, gamma, beta, mean, var, eps);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[gamma]);
    let x_data: Vec<f32> = (0..b * c).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let gamma_init = vec![1.0f32; c];
    let beta_init = vec![0.0f32; c];
    let mean_init = vec![0.1f32; c];
    let var_init = vec![0.5f32; c];

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    for (name, data) in [
        ("gamma", &gamma_init),
        ("beta", &beta_init),
        ("mean", &mean_init),
        ("var", &var_init),
    ] {
        compiled.set_param(name, data);
    }
    let outs = compiled.run(&[("x", &x_data), ("d_output", &[1.0f32])]);
    let dgamma = &outs[1];
    assert_eq!(dgamma.len(), c);
    assert!(dgamma.iter().all(|v| v.is_finite()));

    let h = 1e-3f32;
    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(g.clone());
    for (name, data) in [
        ("gamma", &gamma_init),
        ("beta", &beta_init),
        ("mean", &mean_init),
        ("var", &var_init),
    ] {
        fwd.set_param(name, data);
    }
    let loss0 = fwd.run(&[("x", &x_data)])[0][0];

    let mut max_err = 0.0f32;
    for i in 0..c {
        let mut gp = gamma_init.clone();
        gp[i] += h;
        fwd.set_param("gamma", &gp);
        let lp = fwd.run(&[("x", &x_data)])[0][0];
        let fd = (lp - loss0) / h;
        max_err = max_err.max((fd - dgamma[i]).abs());
    }
    assert!(max_err < 0.05, "max gamma grad error {max_err}");
}
