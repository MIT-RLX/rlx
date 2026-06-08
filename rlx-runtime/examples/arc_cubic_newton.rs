// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Arc-length cubic Newton: use `f'''` to stabilize steps on `f(x)=x³`.
//!
//! ```sh
//! cargo run -p rlx-runtime --example arc_cubic_newton --features cpu
//! ```

use rlx_autodiff::nth_order_grad;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f64s(x: f64) -> Vec<u8> {
    x.to_le_bytes().to_vec()
}
fn f64_out(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

fn cubic_graph() -> Graph {
    let mut g = Graph::new("x3");
    let x = g.input("x", Shape::scalar(DType::F64));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![x3]);
    g
}

fn main() {
    let g = cubic_graph();
    let mut x = 2.0;
    for iter in 0..5 {
        let g1 = nth_order_grad(&g, "x", 1);
        let g2 = nth_order_grad(&g, "x", 2);
        let g3 = nth_order_grad(&g, "x", 3);
        let f1 = f64_out(
            &Session::new(Device::Cpu)
                .compile(g1)
                .run_typed(&[("x", &f64s(x), DType::F64)])[0]
                .0,
        );
        let f2 = f64_out(
            &Session::new(Device::Cpu)
                .compile(g2)
                .run_typed(&[("x", &f64s(x), DType::F64)])[0]
                .0,
        );
        let f3 = f64_out(
            &Session::new(Device::Cpu)
                .compile(g3)
                .run_typed(&[("x", &f64s(x), DType::F64)])[0]
                .0,
        );
        let step = -f1 / (f2 + 1e-3 * f3.abs().max(1.0));
        println!("iter {iter}: x={x:.6}  f'={f1:.6}  step={step:.6}");
        x += step;
    }
    println!("final x ≈ {x:.6} (root near 0)");
}
