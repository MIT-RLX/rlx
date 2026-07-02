// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! KdV soliton PINN residual using third-order spatial derivatives.
//!
//! Residual `u_t + 6 u u_x + u_xxx` on a coarse `(t, x)` grid with a
//! tanh-shaped ansatz. Run:
//!
//! ```sh
//! cargo run -p rlx-runtime --example pinn_kdv_train --features cpu
//! ```

use rlx_autodiff::nth_order_grad;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f32s(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_out(b: &[u8]) -> f32 {
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

fn kdv_ansatz_graph() -> Graph {
    let mut g = Graph::new("kdv_ansatz");
    let t = g.input("t", Shape::scalar(DType::F32));
    let x = g.input("x", Shape::scalar(DType::F32));
    let c = g.add_node(
        rlx_ir::Op::Constant {
            data: 0.5f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::scalar(DType::F32),
    );
    let xi = g.binary(rlx_ir::op::BinaryOp::Sub, x, c, Shape::scalar(DType::F32));
    let phase = g.binary(rlx_ir::op::BinaryOp::Sub, xi, t, Shape::scalar(DType::F32));
    let u = g.activation(Activation::Tanh, phase, Shape::scalar(DType::F32));
    g.set_outputs(vec![u]);
    g
}

fn eval(g: &Graph, t: f32, x: f32) -> f32 {
    f32_out(
        &Session::new(Device::Cpu).compile(g.clone()).run_typed(&[
            ("t", &f32s(&[t]), DType::F32),
            ("x", &f32s(&[x]), DType::F32),
        ])[0]
            .0,
    )
}

fn main() {
    let forward = kdv_ansatz_graph();
    let u_t = nth_order_grad(&forward, "t", 1);
    let u_x = nth_order_grad(&forward, "x", 1);
    let u_xxx = nth_order_grad(&forward, "x", 3);

    let grid: &[(f32, f32)] = &[(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 1.5)];
    let mut worst = 0.0f32;
    for &(t, x) in grid {
        let u = eval(&forward, t, x);
        let ut = eval(&u_t, t, x);
        let ux = eval(&u_x, t, x);
        let uxxx = eval(&u_xxx, t, x);
        let r = ut + 6.0 * u * ux + uxxx;
        worst = worst.max(r.abs());
    }
    println!("KdV residual worst |r| on {grid:?}: {worst:.3e}");
    println!("(demo ansatz — not a trained PINN; shows third-order spatial AD runs end-to-end)");
}
