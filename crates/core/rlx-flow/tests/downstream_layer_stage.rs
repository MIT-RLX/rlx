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

//! Downstream extension seam: a [`LayerStage`] defined *outside* `rlx-flow`
//! composes primitives and runs through a flow via `ModelFlow::layer` — no
//! `FlowStage` enum variant, no core edit. As an integration test it is compiled
//! as an external crate, so it sees only rlx-flow's *public* API — exactly the
//! surface a model crate in `rlx-models` has. This is the proof that a novel
//! architecture block can live fully downstream.

use anyhow::Result;
use rlx_flow::MapWeights;
use rlx_flow::prelude::*;
use rlx_ir::{DType, Op, Shape};
use rlx_runtime::{Device, Session};

/// A model-crate-style linear block (`y = x @ w`), defined here as if it lived
/// in a downstream crate. It composes primitives via the curated [`FlowCtx`]
/// builders (`ctx.linear`) — no `rlx_ir::hir` import, no `HirMut` — so the
/// optimizer sees through it and it fuses like any built-in block.
struct DownstreamLinear {
    weight_key: String,
}

impl LayerStage for DownstreamLinear {
    fn name(&self) -> &str {
        "downstream_linear"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let out = ctx.linear(&input, &self.weight_key, false)?;
        Ok((
            out.clone(),
            StageArtifacts::hidden_only(out.shape().clone()),
        ))
    }
}

/// A block that publishes a side output (aux head): returns `relu(x @ w)` as the
/// hidden stream and also publishes the pre-activation `x @ w` as a named
/// auxiliary graph output — proving `ctx.publish_side_output` auto-wiring.
struct DownstreamAuxHead {
    weight_key: String,
}

impl LayerStage for DownstreamAuxHead {
    fn name(&self) -> &str {
        "downstream_aux_head"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let proj = ctx.linear(&input, &self.weight_key, false)?;
        ctx.publish_side_output("aux", &proj);
        let hidden = ctx.relu(&proj);
        let arts = StageArtifacts::hidden_only(hidden.shape().clone())
            .with_side("aux", proj.shape().clone());
        Ok((hidden, arts))
    }
}

#[test]
fn downstream_layer_stage_runs_through_flow_on_cpu() {
    const B: usize = 2;
    const D: usize = 3;
    let f = DType::F32;

    // w1 = I, w2 = 2·I  →  y = x @ w1 @ w2 = 2·x.
    let ident = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let two_ident = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
    let mut w = MapWeights::default();
    w.insert("w1", ident.clone(), vec![D, D]);
    w.insert("w2", two_ident.clone(), vec![D, D]);

    // Two downstream blocks chained through the flow — proves `.layer(..)`
    // threads the tensor and composes, with no `FlowStage` variant for either.
    let built = ModelFlow::new("downstream")
        .input("x", Shape::new(&[B, D], f))
        .layer_stage(DownstreamLinear {
            weight_key: "w1".into(),
        })
        .layer_stage(DownstreamLinear {
            weight_key: "w2".into(),
        })
        .build(&mut w)
        .expect("flow build");

    let g = built.into_graph().expect("into_graph");

    // The block lowered to real primitives, NOT an opaque custom call — this is
    // the whole point: it stays visible to fusion / the optimizer.
    assert!(
        !g.nodes().iter().any(|n| matches!(n.op, Op::Custom { .. })),
        "downstream block must compose primitives, not emit Op::Custom"
    );
    assert!(
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::MatMul))
            .count()
            >= 2,
        "expected the two chained matmuls to survive as MatMul nodes"
    );

    // End-to-end: compile + run on CPU and check the numerics (2·x).
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w1", &ident);
    c.set_param("w2", &two_ident);
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outs = c.run(&[("x", &x)]);
    let y = &outs[0];

    let expect: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
    assert_eq!(y.len(), expect.len(), "y={y:?}");
    for (a, b) in y.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-5, "y={y:?} expect={expect:?}");
    }
}

#[test]
fn downstream_block_publishes_side_output() {
    const B: usize = 2;
    const D: usize = 3;
    let f = DType::F32;

    let ident = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut w = MapWeights::default();
    w.insert("w", ident.clone(), vec![D, D]);

    let built = ModelFlow::new("aux")
        .input("x", Shape::new(&[B, D], f))
        .layer_stage(DownstreamAuxHead {
            weight_key: "w".into(),
        })
        .build(&mut w)
        .expect("flow build");

    // The published side output became a second graph output, named "aux".
    assert!(
        built.output_names().iter().any(|n| n == "aux"),
        "side output name should be registered: {:?}",
        built.output_names()
    );

    let g = built.into_graph().expect("into_graph");
    assert_eq!(g.outputs.len(), 2, "primary + one published side output");

    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w", &ident);
    let x = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let outs = c.run(&[("x", &x)]);
    // Primary = relu(x @ I) = relu(x); side "aux" = x @ I = x.
    let hidden = &outs[0];
    let aux = &outs[1];
    let expect_hidden: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
    for (a, b) in hidden.iter().zip(expect_hidden.iter()) {
        assert!((a - b).abs() < 1e-5, "hidden={hidden:?}");
    }
    for (a, b) in aux.iter().zip(x.iter()) {
        assert!((a - b).abs() < 1e-5, "aux={aux:?} expect={x:?}");
    }
}
