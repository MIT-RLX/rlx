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

//! Flow `dit_ada_gated_linear` → Session train step (CPU).

use rlx_flow::{MapWeights, ModelFlow, dit_ada_gated_linear};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 3;
const D: usize = 4;
const EPS: f32 = 1e-5;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.11 * (i as f32) - 0.07 * ((i % 3) as f32)))
        .collect()
}

#[test]
fn flow_dit_ada_gated_linear_train_step_cpu() {
    let f = DType::F32;
    let mut weights = MapWeights::default();
    weights.insert("w", fill(D * D, 0.05), vec![D, D]);

    let built = ModelFlow::new("flow_dit")
        .input("x", Shape::new(&[B, S, D], f))
        .input("blk.scale", Shape::new(&[B, 1, D], f))
        .input("blk.shift", Shape::new(&[B, 1, D], f))
        .input("blk.gate", Shape::new(&[B, 1, D], f))
        .stage(dit_ada_gated_linear(
            "blk",
            "w",
            AdaNormKind::LayerNorm,
            EPS,
        ))
        .build(&mut weights)
        .expect("flow build");

    let mut g = built.into_graph().expect("into_graph");
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNorm { .. }))
    );
    assert!(g.nodes().iter().any(|n| matches!(n.op, Op::GatedResidual)));

    let out = g.outputs[0];
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);

    let x_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let w_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Param { name } if name == "w"))
        .map(|n| n.id)
        .unwrap();
    let bwd = grad_with_loss(&g, &[x_id, w_id]);
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. }))
    );
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidualBackward))
    );

    let x0 = fill(B * S * D, 1.0);
    let w0 = fill(D * D, 0.05);
    let scale = fill(B * D, 0.2);
    let shift = fill(B * D, -0.1);
    let gate = fill(B * D, 0.3);
    let d_output = [1.0f32];

    let mut c = Session::new(Device::Cpu).compile(bwd);
    c.set_param("w", &w0);
    let outs = c.run(&[
        ("x", &x0),
        ("blk.scale", &scale),
        ("blk.shift", &shift),
        ("blk.gate", &gate),
        ("d_output", &d_output),
    ]);
    let loss_v = outs[0][0];
    let dx = &outs[1];
    let dw = &outs[2];
    assert!(loss_v.is_finite(), "loss={loss_v}");
    assert!(dx.iter().all(|v| v.is_finite()));
    assert!(dw.iter().all(|v| v.is_finite()));
    assert!(
        dx.iter().any(|v| v.abs() > 1e-8) || dw.iter().any(|v| v.abs() > 1e-8),
        "expected non-zero grads"
    );
}
