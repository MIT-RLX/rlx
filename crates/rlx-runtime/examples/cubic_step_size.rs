// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Cubic-regularized Newton step using third-order derivatives.
//!
//! For `f(x) = x⁴/4`, the cubic model at `x₀` uses `f'''(x₀)` to pick a
//! stable step. Run:
//!
//! ```sh
//! cargo run -p rlx-runtime --example cubic_step_size --features cpu
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

fn main() {
    let mut g = Graph::new("x4");
    let x = g.input("x", Shape::scalar(DType::F64));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
    let x4 = g.binary(BinaryOp::Mul, x2, x2, Shape::scalar(DType::F64));
    let quarter = g.add_node(
        rlx_ir::Op::Constant {
            data: (0.25f64).to_le_bytes().to_vec(),
        },
        vec![],
        Shape::scalar(DType::F64),
    );
    let quart = g.binary(BinaryOp::Mul, x4, quarter, Shape::scalar(DType::F64));
    g.set_outputs(vec![quart]);

    let x0 = 1.5;
    let g1 = nth_order_grad(&g, "x", 1);
    let g2 = nth_order_grad(&g, "x", 2);
    let g3 = nth_order_grad(&g, "x", 3);

    let f1 = f64_out(
        &Session::new(Device::Cpu)
            .compile(g1)
            .run_typed(&[("x", &f64s(x0), DType::F64)])[0]
            .0,
    );
    let f2 = f64_out(
        &Session::new(Device::Cpu)
            .compile(g2)
            .run_typed(&[("x", &f64s(x0), DType::F64)])[0]
            .0,
    );
    let f3 = f64_out(
        &Session::new(Device::Cpu)
            .compile(g3)
            .run_typed(&[("x", &f64s(x0), DType::F64)])[0]
            .0,
    );

    // f(x)=x⁴/4 → f'(x)=x³, f''(x)=3x², f'''(x)=6x
    let h = -f1 / (f2 + 0.5 * f3.abs());
    println!("x0={x0}  f'={f1:.6}  f''={f2:.6}  f'''={f3:.6}");
    println!("cubic-regularized step h ≈ {h:.6}  →  x1 ≈ {:.6}", x0 + h);
}
