// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-block DiT train step: stacked adaLN → linear → gated residual blocks
//! (FLUX-style attn + MLP sublayers) → scalar sum loss → one CPU backward step.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const D: usize = 8;
const EPS: f32 = 1e-5;
const BLOCKS: usize = 3;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.11 * (i as f32) - 0.07 * ((i % 3) as f32)))
        .collect()
}

/// One DiT sublayer: adaLN → matmul → gated residual on the saved pre-norm `x`.
fn dit_sublayer(g: &mut Graph, x: rlx_ir::NodeId, prefix: &str, w_name: &str) -> rlx_ir::NodeId {
    let f = DType::F32;
    let scale = g.input(format!("{prefix}.scale"), Shape::new(&[B, 1, D], f));
    let shift = g.input(format!("{prefix}.shift"), Shape::new(&[B, 1, D], f));
    let gate = g.input(format!("{prefix}.gate"), Shape::new(&[B, 1, D], f));
    let w = g.param(w_name, Shape::new(&[D, D], f));

    let n = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, EPS);
    let flat = g.reshape_(n, vec![(B * S) as i64, D as i64]);
    let y2 = g.mm(flat, w);
    let y = g.reshape_(y2, vec![B as i64, S as i64, D as i64]);
    g.gated_residual(x, y, gate)
}

/// FLUX-like block: attn sublayer then MLP sublayer, both adaLN + linear + gate.
fn dit_block(g: &mut Graph, x: rlx_ir::NodeId, blk: usize) -> rlx_ir::NodeId {
    let attn_prefix = format!("blk{blk}.attn");
    let mlp_prefix = format!("blk{blk}.mlp");
    let h = dit_sublayer(g, x, &attn_prefix, &format!("blk{blk}.attn_w"));
    dit_sublayer(g, h, &mlp_prefix, &format!("blk{blk}.mlp_w"))
}

fn dit_multiblock_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("dit_multiblock");
    let mut x = g.param("x", Shape::new(&[B, S, D], f));
    for blk in 0..BLOCKS {
        x = dit_block(&mut g, x, blk);
    }
    let loss = g.sum(x, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    g
}

#[test]
fn dit_multiblock_one_train_step_cpu() {
    let g = dit_multiblock_loss();
    let x_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Param { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let w_ids: Vec<_> = g
        .nodes()
        .iter()
        .filter(|n| matches!(&n.op, Op::Param { name } if name.ends_with("_w")))
        .map(|n| n.id)
        .collect();
    assert_eq!(w_ids.len(), BLOCKS * 2, "attn + mlp weights per block");

    let mut param_ids = vec![x_id];
    param_ids.extend(w_ids.iter().copied());
    let bwd = grad_with_loss(&g, &param_ids);

    let ada_bwd = bwd
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::AdaLayerNormBackward { .. }))
        .count();
    let gate_bwd = bwd
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::GatedResidualBackward))
        .count();
    assert_eq!(
        ada_bwd,
        BLOCKS * 2,
        "expected AdaLayerNormBackward per adaLN sublayer"
    );
    assert_eq!(
        gate_bwd,
        BLOCKS * 2,
        "expected GatedResidualBackward per gated sublayer"
    );

    let x0 = fill(B * S * D, 1.0);
    let scale = fill(B * D, 0.2);
    let shift = fill(B * D, -0.1);
    let gate = fill(B * D, 0.3);
    let d_output = [1.0f32];

    let mut feeds: Vec<(&str, &[f32])> = vec![("d_output", &d_output)];
    let mod_feeds: [(&str, &[f32]); 6] = [
        ("attn.scale", &scale),
        ("attn.shift", &shift),
        ("attn.gate", &gate),
        ("mlp.scale", &scale),
        ("mlp.shift", &shift),
        ("mlp.gate", &gate),
    ];
    for blk in 0..BLOCKS {
        for (suffix, data) in mod_feeds {
            let name: &'static str = Box::leak(format!("blk{blk}.{suffix}").into_boxed_str());
            feeds.push((name, data));
        }
    }

    let mut c = Session::new(Device::Cpu).compile(bwd);
    c.set_param("x", &x0);
    for blk in 0..BLOCKS {
        let w = fill(D * D, 0.05 + 0.01 * blk as f32);
        c.set_param(&format!("blk{blk}.attn_w"), &w);
        c.set_param(&format!("blk{blk}.mlp_w"), &w);
    }
    let outs = c.run(&feeds);
    let loss = outs[0][0];
    assert!(loss.is_finite(), "loss={loss}");
    for out in &outs[1..] {
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|v| v.abs() > 1e-8),
            "expected non-zero grad entries"
        );
    }
}
