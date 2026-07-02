// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Minimal INT8 quantization-aware training (QAT) loop on the CPU backend.
//!
//! A one-layer linear model `y = x @ W` is fit to a synthetic target with
//! ordinary SGD — except the weight `W` is passed through `Op::FakeQuantize`
//! (symmetric, per-tensor, 8-bit) before every matmul. The forward pass
//! therefore *sees* the int8 rounding it will face at deployment, and the
//! straight-through estimator (STE) lets the gradient flow back to the f32
//! master weights so the optimizer can compensate.
//!
//! Reverse-mode autodiff (`grad_with_loss`) builds the backward graph; the
//! whole thing compiles to one CPU schedule that returns `[loss, dW]` each
//! step. The host loop just does `W -= lr * dW`.
//!
//! After training we emit the deployment int8 codes the way inference would
//! (`scale = max|W| / 127`, `q = round(W / scale)`), i.e. exactly what
//! `Op::Quantize` / `Op::QMatMul` consume on the inference path.
//!
//! Run:
//! ```sh
//! cargo run -p rlx-runtime --example int8_qat_train --features cpu
//! ```

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp, ScaleMode, SteKind};
use rlx_ir::{DType, Graph, Op, Philox4x32, Shape};
use rlx_runtime::{Device, Session};

const BATCH: usize = 8;
const IN: usize = 4;
const OUT: usize = 3;
const BITS: u8 = 8;
const STEPS: usize = 200;
const LR: f32 = 0.05;

fn f32s(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Forward graph: `loss = mean((fake_quant8(W) @ x - target)^2)`.
/// Returns the graph and the `NodeId` of the f32 master weight `W`
/// (the tensor we differentiate w.r.t. and update).
fn qat_linear_graph() -> (Graph, rlx_ir::NodeId) {
    let mut g = Graph::new("int8_qat_linear");

    let x = g.input("x", Shape::new(&[BATCH, IN], DType::F32));
    let w = g.input("w", Shape::new(&[IN, OUT], DType::F32));
    let target = g.input("target", Shape::new(&[BATCH, OUT], DType::F32));

    // QAT core: simulate int8 rounding of the weights in the forward pass.
    // Per-tensor (axis = None), symmetric, identity STE — the gradient
    // passes straight through the round during backward.
    let wq = g.add_node(
        Op::FakeQuantize {
            bits: BITS,
            axis: None,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![w],
        Shape::new(&[IN, OUT], DType::F32),
    );

    let out = g.matmul(x, wq, Shape::new(&[BATCH, OUT], DType::F32));
    let diff = g.binary(
        BinaryOp::Sub,
        out,
        target,
        Shape::new(&[BATCH, OUT], DType::F32),
    );
    let sq = g.binary(
        BinaryOp::Mul,
        diff,
        diff,
        Shape::new(&[BATCH, OUT], DType::F32),
    );
    let loss = g.reduce(
        sq,
        ReduceOp::Mean,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );

    g.set_outputs(vec![loss]);
    (g, w)
}

/// Emit the int8 deployment codes for a trained f32 weight, the same way
/// the inference path quantizes (symmetric, per-tensor max-abs scale).
fn quantize_int8(w: &[f32]) -> (f32, Vec<i8>) {
    let max_abs = w.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let scale = max_abs / 127.0;
    let codes = w
        .iter()
        .map(|&v| ((v / scale).round() as i32).clamp(-128, 127) as i8)
        .collect();
    (scale, codes)
}

fn main() {
    // ── Synthetic dataset: target = X @ W_true ──
    let mut rng = Philox4x32::new(7);
    let mut x = vec![0f32; BATCH * IN];
    rng.fill_normal(&mut x);
    let mut w_true = vec![0f32; IN * OUT];
    rng.fill_normal(&mut w_true);

    let mut target = vec![0f32; BATCH * OUT];
    for b in 0..BATCH {
        for o in 0..OUT {
            let mut acc = 0.0f32;
            for i in 0..IN {
                acc += x[b * IN + i] * w_true[i * OUT + o];
            }
            target[b * OUT + o] = acc;
        }
    }

    // ── Build forward + backward (reverse-mode AD) and compile once ──
    let (forward, w_id) = qat_linear_graph();
    let backward = grad_with_loss(&forward, &[w_id]);
    let mut compiled = Session::new(Device::Cpu).compile(backward);

    // ── Master weights start small and random ──
    let mut w = vec![0f32; IN * OUT];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        *v *= 0.1;
    }

    let x_bytes = f32s(&x);
    let target_bytes = f32s(&target);
    let seed = f32s(&[1.0]); // d(loss)/d(loss) = 1

    println!("step    loss");
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    for step in 0..STEPS {
        // backward graph returns [loss (scalar), dW (IN*OUT)]
        let outs = compiled.run_typed(&[
            ("x", &x_bytes, DType::F32),
            ("w", &f32s(&w), DType::F32),
            ("target", &target_bytes, DType::F32),
            ("d_output", &seed, DType::F32),
        ]);
        let loss = to_f32(&outs[0].0)[0];
        let dw = to_f32(&outs[1].0);

        // Plain SGD on the f32 master weights (STE routed the grad here).
        for (wv, &g) in w.iter_mut().zip(&dw) {
            *wv -= LR * g;
        }

        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
        if step % 20 == 0 || step == STEPS - 1 {
            println!("{step:>4}    {loss:.6}");
        }
    }

    println!();
    println!("loss {first_loss:.6} -> {last_loss:.6}  ({BITS}-bit QAT, {STEPS} SGD steps)");

    // ── Deployment-side int8 codes (what Op::Quantize / Op::QMatMul take) ──
    let (scale, codes) = quantize_int8(&w);
    println!("deploy int8 scale = {scale:.6}");
    println!("deploy int8 codes = {codes:?}");
}
