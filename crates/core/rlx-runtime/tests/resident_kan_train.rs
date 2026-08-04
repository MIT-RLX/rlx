// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A REAL `Op::SplineActivation` (KAN) layer trained end-to-end, GPU-resident on
//! Metal, through the first-class runtime API (`CompiledGraph::optimizer_step_resident`
//! + `bind_gpu_handle` + `run_read_outputs(Some(&[0]))`) with real `Muon`. The
//! spline coefficients `[C, num_basis]` are 2-D → Muon orthogonalizes them. No
//! GPU→host→GPU roundtrip: the loop never drops to `MetalExecutable`.

#![cfg(all(target_os = "macos", feature = "metal", feature = "training"))]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_optim::{Muon, Optimizer};
use rlx_runtime::{Device, Session};

const B: usize = 8;
const C: usize = 3;
const NB: usize = 5;
const GMIN: f32 = -2.0;
const GMAX: f32 = 2.0;

fn spline_ref(x: &[f32], coeff: &[f32]) -> Vec<f32> {
    let step = (GMAX - GMIN) / (NB as f32 - 1.0);
    let inv_h = 1.0 / step;
    let mut out = vec![0f32; B * C];
    for r in 0..B {
        for c in 0..C {
            let xv = x[r * C + c];
            let mut acc = 0f32;
            for gi in 0..NB {
                let center = GMIN + gi as f32 * step;
                let z = (xv - center) * inv_h;
                acc += coeff[c * NB + gi] * (-(z * z)).exp();
            }
            out[r * C + c] = acc;
        }
    }
    out
}

#[test]
fn resident_kan_spline_layer_trains_on_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // Data + a ground-truth spline whose output is the (exactly fittable) target.
    let x: Vec<f32> = (0..B * C).map(|i| (i as f32 * 0.23).sin() * 1.4).collect();
    let coeff_true: Vec<f32> = (0..C * NB).map(|i| (i as f32 * 0.41).cos() * 0.7).collect();
    let target = spline_ref(&x, &coeff_true);

    // Forward: h = spline(x, coeff) ; loss = Σ(h − target)². coeff is a 2-D Param.
    let mut g = Graph::new("kan_layer");
    let xn = g.add_node(
        Op::Constant {
            data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, C], DType::F32),
    );
    let coeff = g.param("coeff", Shape::new(&[C, NB], DType::F32));
    let h = g.spline_activation(xn, coeff, NB as u32, GMIN, GMAX);
    let tn = g.add_node(
        Op::Constant {
            data: target.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, C], DType::F32),
    );
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![h, tn],
        Shape::new(&[B, C], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, diff],
        Shape::new(&[B, C], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(B * C) as i64],
        },
        vec![sq],
        Shape::new(&[B * C], DType::F32),
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

    // Backward; rebind the trainable coeff Param → Input so it can be resident.
    let mut backward = grad_with_loss(&g, &[coeff]);
    for node in backward.nodes_mut() {
        if let Op::Param { name } = &node.op
            && name == "coeff"
        {
            node.op = Op::Input { name: name.clone() };
        }
    }

    // First-class runtime API — Session → CompiledGraph, no MetalExecutable.
    let mut c = Session::new(Device::Metal).compile(backward);
    assert!(
        c.bind_gpu_handle("coeff", &[0f32; C * NB]),
        "coeff should bind resident"
    );

    let mut muon = Muon::new(0.05);
    let trainable = vec![("coeff".to_string(), vec![C, NB])];
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let mut stepped = false;
    for step in 0..300 {
        // Layer 2: read only the loss; grad_coeff stays resident in the arena.
        let loss_val = c.run_read_outputs(&[("d_output", &[1.0f32])], Some(&[0]))[0][0];
        if step == 0 {
            first = loss_val;
        }
        last = loss_val;
        // Layers 1+3: Muon steps on the arena-aliased param+grad, in place.
        stepped =
            c.optimizer_step_resident(&trainable, &mut |n, s, p, grad| muon.step(n, s, p, grad));
    }

    assert!(
        stepped,
        "Metal must support optimizer_step_resident (not the no-op default)"
    );
    assert!(first.is_finite() && last.is_finite());
    eprintln!("resident KAN(Muon) on Metal: loss {first:.4} → {last:.6}");
    assert!(
        last < first * 0.1,
        "resident KAN spline layer should train: loss {first} → {last}"
    );
}
