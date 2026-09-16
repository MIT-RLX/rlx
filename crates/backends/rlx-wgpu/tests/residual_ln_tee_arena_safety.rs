// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Step::FusedResidualLnTee` must not read operands whose arena slot has
//! already been reused — wgpu vs CPU.
//!
//! The tee collapses `s = h + delta; n = norm(s)` into the NORM's step, so `h`
//! and `delta` are read at the norm's position instead of the add's. That edge
//! is invisible to the memory planner: the fold happens during lowering, after
//! planning, so liveness has both operands dying at the add. Any node emitted
//! in between can then be given their bytes — including the norm's own output.
//!
//! `rlx-eegdm` was the field repro: the norm's output slot and `h` shared
//! offset 526336, so the residual read the norm's own partly written output.
//! It presented as cos=0.994 rather than as garbage, because every magnitude
//! survived and only the row means moved.
//!
//! The shape below is the one that triggers it: `h`/`delta` are dead after the
//! add (nothing else reads them), while the sum has a second consumer AFTER
//! the norm — so the tee fires but the operands' slots are free to be reused.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

fn build(rows: usize, cols: usize) -> Graph {
    let mut g = Graph::new("res_ln_tee_arena");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let w = g.input("w", Shape::new(&[cols, cols], DType::F32));

    // Two matmuls give `h` and `delta` their own arena slots and make them
    // genuinely dead after the add (a graph input would be pinned for the whole
    // run and could never be reused).
    let h = g.mm(x, w);
    let delta = g.mm(h, w);
    let s = g.add(h, delta);

    // The node that does the damage: a SMALL tensor emitted after the add and
    // before the norm. Both operands are dead at the add, so best-fit drops
    // this 512-byte result at the front of `h`'s freed 128 KiB slot and
    // overwrites exactly `h`'s first row. That is `rlx-eegdm`'s NodeId(71) —
    // a 128-element MatMul landing on offset 526336, the head of `h`'s
    // [526336, 657408) — and it is why the corruption showed up as a shifted
    // row mean with every magnitude intact instead of as obvious garbage.
    // It must also DIE before the norm — that is what leaves `h`'s slot free
    // over exactly the window the tee needs it in. (`small` is consumed
    // immediately by `small2`; only `small2` survives to the output.)
    let small_a = g.input("small_a", Shape::new(&[1, cols], DType::F32));
    let small = g.mm(small_a, w);
    let small2 = g.mm(small, w);

    let gamma = g.param("gamma", Shape::new(&[cols], DType::F32));
    let beta = g.param("beta", Shape::new(&[cols], DType::F32));
    let n = g.add_node(
        Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![s, gamma, beta],
        Shape::new(&[rows, cols], DType::F32),
    );

    // Second consumer of the sum, emitted AFTER the norm. Two consumers are
    // what makes the upstream `FuseResidualLN` decline and this tee fire; being
    // after the norm is what keeps the tee's own early-reader guard happy.
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![n, s],
        Shape::new(&[rows, cols], DType::F32),
    );
    // Keep `small` live so it is really materialised into that slot.
    let small_b = g.tile_(small2, vec![rows, 1]);
    let out = g.add(out, small_b);
    g.set_outputs(vec![out]);
    g
}

#[test]
fn tee_operands_survive_slot_reuse() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let (rows, cols) = (256usize, 128usize);
    let g = build(rows, cols);

    let x: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let w: Vec<f32> = (0..cols * cols)
        .map(|i| ((i as f32) * 0.011).cos() * 0.1)
        .collect();
    let small_a: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.023).sin()).collect();
    let gamma = vec![1.0f32; cols];
    let beta = vec![0.0f32; cols];

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param("gamma", &gamma);
        c.set_param("beta", &beta);
        c.run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("small_a", small_a.as_slice()),
        ])
        .remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);

    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // A clobbered residual keeps the magnitudes and moves the row means, so
    // compare elementwise rather than on any summary statistic.
    eprintln!("residual-ln tee arena safety: max_abs={max_abs:.6e}");
    assert!(
        max_abs <= 1e-4,
        "wgpu residual-LN tee diverged from CPU: max_abs {max_abs} > 1e-4 \
         (tee read an operand whose slot was reused)"
    );
}
