// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal repro of the Voxtral MLP BLOCK on Metal vs CPU:
//!   out = x + down( silu(gate(rmsnorm(x))) * up(rmsnorm(x)) )
//!
//! Each piece passes parity alone (rmsnorm, full swiglu) but the BLOCK — which
//! keeps the residual `x` live across the wide (8192) SwiGLU intermediates —
//! diverges in the full model. Suspected residual/arena buffer aliasing: the
//! saved residual is overwritten by a SwiGLU intermediate before the add.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_mlp_block(rows: usize, h: usize, inter: usize, eps: f32) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mlp_block");
    let x = g.input("x", Shape::new(&[rows, h], f));
    let gamma = g.input("gamma", Shape::new(&[h], f));
    let beta = g.input("beta", Shape::new(&[h], f));
    let wg = g.input("wg", Shape::new(&[h, inter], f));
    let wu = g.input("wu", Shape::new(&[h, inter], f));
    let wd = g.input("wd", Shape::new(&[inter, h], f));

    let normed = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps },
        vec![x, gamma, beta],
        Shape::new(&[rows, h], f),
    );
    let gate = g.add_node(
        rlx_ir::Op::MatMul,
        vec![normed, wg],
        Shape::new(&[rows, inter], f),
    );
    let up = g.add_node(
        rlx_ir::Op::MatMul,
        vec![normed, wu],
        Shape::new(&[rows, inter], f),
    );
    let act = g.add_node(
        rlx_ir::Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[rows, inter], f),
    );
    let prod = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Mul),
        vec![act, up],
        Shape::new(&[rows, inter], f),
    );
    let down = g.add_node(
        rlx_ir::Op::MatMul,
        vec![prod, wd],
        Shape::new(&[rows, h], f),
    );
    // Residual add: x must stay live across the whole SwiGLU above.
    let out = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Add),
        vec![x, down],
        Shape::new(&[rows, h], f),
    );
    g.set_outputs(vec![out]);
    g
}

#[test]
fn metal_mlp_residual_block_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (rows, h, inter, eps) = (96, 3072, 8192, 1e-5);
    let mk = |seed: f32, n: usize, sc: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * seed).sin() * sc).collect()
    };
    let x = mk(0.0007, rows * h, 1.0);
    let gamma = mk(0.0011, h, 0.1)
        .iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let beta = vec![0.0f32; h];
    let wg = mk(0.0003, h * inter, 0.05);
    let wu = mk(0.0005, h * inter, 0.05);
    let wd = mk(0.0002, inter * h, 0.05);

    let inputs: [(&str, &[f32]); 6] = [
        ("x", &x),
        ("gamma", &gamma),
        ("beta", &beta),
        ("wg", &wg),
        ("wu", &wu),
        ("wd", &wd),
    ];
    let g = build_mlp_block(rows, h, inter, eps);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&inputs).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&inputs).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!("mlp block: max_abs={max_abs:.6} cpu_sum={cpu_sum:.5} metal_sum={metal_sum:.5}");
    assert!(max_abs < 1e-4, "mlp residual block max_abs={max_abs}");
}
