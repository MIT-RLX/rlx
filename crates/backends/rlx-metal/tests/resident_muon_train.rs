// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full GPU-resident Muon training loop, end-to-end (gradient-roundtrip layers
//! 2 + 3). A 2-D weight `W` (so Muon's Newton-Schulz orthogonalization actually
//! fires) is trained on a linear regression `y = x·W`, `loss = Σ(y − target)²`:
//!   - W lives RESIDENT in the unified-memory arena (`bind_gpu_handle`).
//!   - each step reads back ONLY the loss (`run_read_outputs(Some(&[0]))`) — the
//!     gradient stays resident in the arena (layer 2, no D2H of grads).
//!   - `Muon::step` runs on the param+grad slices ALIASED into the arena
//!     (`optimizer_step_resident`) and writes W in place (layer 3).
//!   - the next forward reads the updated W with no host re-upload.
//! No GPU→host→optimizer→host→GPU roundtrip anywhere in the loop.

#![cfg(target_os = "macos")]

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_metal::backend::MetalExecutable;
use rlx_optim::{Muon, Optimizer};

const B: usize = 8;
const IN: usize = 4;
const OUT: usize = 3;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn resident_muon_training_loop_converges() {
    if rlx_metal::device::metal_device().is_none() {
        return;
    }
    // Ground-truth linear map + data; target = x·W_true (exactly learnable).
    let w_true: Vec<f32> = (0..IN * OUT).map(|i| (i as f32 * 0.37).sin()).collect();
    let x: Vec<f32> = (0..B * IN).map(|i| (i as f32 * 0.11).cos() + 0.3).collect();
    let mut target = vec![0f32; B * OUT];
    for b in 0..B {
        for o in 0..OUT {
            let mut s = 0f32;
            for k in 0..IN {
                s += x[b * IN + k] * w_true[k * OUT + o];
            }
            target[b * OUT + o] = s;
        }
    }

    // Forward: y = x·W ; loss = Σ(y − target)². W is a 2-D Param; x/target baked.
    let mut g = Graph::new("linreg");
    let w = g.param("W", Shape::new(&[IN, OUT], DType::F32));
    let xn = g.add_node(
        Op::Constant {
            data: f32_bytes(&x),
        },
        vec![],
        Shape::new(&[B, IN], DType::F32),
    );
    let tn = g.add_node(
        Op::Constant {
            data: f32_bytes(&target),
        },
        vec![],
        Shape::new(&[B, OUT], DType::F32),
    );
    let y = g.matmul(xn, w, Shape::new(&[B, OUT], DType::F32));
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![y, tn],
        Shape::new(&[B, OUT], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, diff],
        Shape::new(&[B, OUT], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(B * OUT) as i64],
        },
        vec![sq],
        Shape::new(&[B * OUT], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);

    // Backward, then rebind trainable W: Param → Input so it can be resident.
    let mut backward = rlx_opt::grad_with_loss(&g, &[w]);
    for node in backward.nodes_mut() {
        if let Op::Param { name } = &node.op
            && name == "W"
        {
            node.op = Op::Input { name: name.clone() };
        }
    }

    let mut exe = MetalExecutable::compile(backward);
    assert!(
        exe.bind_gpu_handle("W", &[0f32; IN * OUT]),
        "W should bind resident"
    );

    let mut muon = Muon::new(0.05);
    let trainable = vec![("W".to_string(), vec![IN, OUT])];
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..300 {
        // Layer 2: read ONLY the loss (index 0); grad_W (index 1) stays resident.
        let outs = exe.run_read_outputs(&[("d_output", &[1.0f32])], Some(&[0]));
        let loss_val = outs[0][0];
        if step == 0 {
            first = loss_val;
        }
        last = loss_val;
        // Layer 3: Muon steps on arena-aliased param+grad slices, in place.
        exe.optimizer_step_resident(&trainable, |name, shape, p, grd| {
            muon.step(name, shape, p, grd);
        });
    }

    let w_final = exe.read_gpu_handle("W").expect("read resident W");
    let err = w_final
        .iter()
        .zip(&w_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("resident Muon: loss {first:.4} → {last:.6}, max |W-W_true| = {err:.4}");
    assert!(first.is_finite() && last.is_finite());
    assert!(w_final.iter().all(|v| v.is_finite()));
    assert!(
        last < first * 0.1,
        "resident Muon loop should converge: loss {first} → {last}"
    );
}
