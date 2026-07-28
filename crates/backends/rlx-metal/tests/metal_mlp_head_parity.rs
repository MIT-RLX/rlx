// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MLP block + final rmsnorm (SHARED beta) + lm_head matmul, forced through
//! MPSGraph, Metal vs CPU. The MLP block alone passes under MPSGraph, but in the
//! full Voxtral model — where one `zero_beta` param is shared across rmsnorms and
//! a huge lm_head matmul follows — the MLP block's logits go garbage on Metal.
//! This adds exactly that context to find the MPSGraph DAG-lowering trigger.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[allow(clippy::too_many_arguments)]
fn build(rows: usize, h: usize, inter: usize, vocab: usize, eps: f32) -> Graph {
    // 3D batched shapes [1, rows, .] to mirror the model's [batch, seq, hidden].
    let f = DType::F32;
    let b = 1usize;
    let mut g = Graph::new("mlp_head");
    let x = g.input("x", Shape::new(&[b, rows, h], f));
    let gamma1 = g.input("gamma1", Shape::new(&[h], f));
    let gamma2 = g.input("gamma2", Shape::new(&[h], f));
    let beta = g.input("beta", Shape::new(&[h], f)); // SHARED across both rmsnorms
    let wg = g.input("wg", Shape::new(&[h, inter], f));
    let wu = g.input("wu", Shape::new(&[h, inter], f));
    let wd = g.input("wd", Shape::new(&[inter, h], f));
    let whead = g.input("whead", Shape::new(&[h, vocab], f));

    let n1 = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps },
        vec![x, gamma1, beta],
        Shape::new(&[b, rows, h], f),
    );
    let gate = g.add_node(
        rlx_ir::Op::MatMul,
        vec![n1, wg],
        Shape::new(&[b, rows, inter], f),
    );
    let up = g.add_node(
        rlx_ir::Op::MatMul,
        vec![n1, wu],
        Shape::new(&[b, rows, inter], f),
    );
    let act = g.add_node(
        rlx_ir::Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[b, rows, inter], f),
    );
    let prod = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Mul),
        vec![act, up],
        Shape::new(&[b, rows, inter], f),
    );
    let down = g.add_node(
        rlx_ir::Op::MatMul,
        vec![prod, wd],
        Shape::new(&[b, rows, h], f),
    );
    let resid = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Add),
        vec![x, down],
        Shape::new(&[b, rows, h], f),
    );
    // final norm reuses `beta`, then the wide lm_head matmul.
    let n2 = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps },
        vec![resid, gamma2, beta],
        Shape::new(&[b, rows, h], f),
    );
    let logits = g.add_node(
        rlx_ir::Op::MatMul,
        vec![n2, whead],
        Shape::new(&[b, rows, vocab], f),
    );
    g.set_outputs(vec![logits]);
    g
}

#[test]
fn metal_mlp_head_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (rows, h, inter, vocab, eps) = (380, 3072, 8192, 131072, 1e-5);
    let mk = |seed: f32, n: usize, sc: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * seed).sin() * sc).collect()
    };
    let x = mk(0.0007, rows * h, 1.0);
    let gamma1: Vec<f32> = mk(0.0011, h, 0.1).iter().map(|v| 1.0 + v).collect();
    let gamma2: Vec<f32> = mk(0.0013, h, 0.1).iter().map(|v| 1.0 + v).collect();
    let beta = vec![0.0f32; h];
    let wg = mk(0.0003, h * inter, 0.05);
    let wu = mk(0.0005, h * inter, 0.05);
    let wd = mk(0.0002, inter * h, 0.05);
    let whead = mk(0.00001, h * vocab, 0.02);

    let inputs: [(&str, &[f32]); 8] = [
        ("x", &x),
        ("gamma1", &gamma1),
        ("gamma2", &gamma2),
        ("beta", &beta),
        ("wg", &wg),
        ("wu", &wu),
        ("wd", &wd),
        ("whead", &whead),
    ];
    let g = build(rows, h, inter, vocab, eps);
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
    eprintln!("mlp+head: max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}");
    assert!(max_abs < 1e-3, "mlp+head max_abs={max_abs}");
}
