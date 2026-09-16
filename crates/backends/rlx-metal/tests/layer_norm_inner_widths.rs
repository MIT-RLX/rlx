// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::LayerNorm` vs CPU across last-axis widths — rlx-metal.
//!
//! Surfaced by `rlx-tribev2-audio` (W2v-BERT): its feature projection norms a
//! 160-wide last axis, and metal disagreed with CPU from that very first op
//! (max 14.39 vs 19.41), which then carried through every downstream stage
//! (cos 0.64 on all five). 160 is not a multiple of a typical threadgroup
//! width, so widths are swept either side of the power-of-two boundaries.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn max_abs_diff(inner: usize, outer: usize) -> Option<f32> {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        eprintln!("skip: metal unavailable");
        return None;
    }
    let mut g = Graph::new("ln_widths");
    let x = g.input("x", Shape::new(&[1, outer, inner], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[inner], DType::F32));
    let beta = g.param("beta", Shape::new(&[inner], DType::F32));
    let y = g.add_node(
        Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![x, gamma, beta],
        Shape::new(&[1, outer, inner], DType::F32),
    );
    g.set_outputs(vec![y]);

    // A large DC offset per row is what makes a one-pass variance cancel; the
    // real model's feature projection has exactly that.
    let xs: Vec<f32> = (0..outer * inner)
        .map(|i| {
            let r = (i / inner) as f32;
            ((i as f32) * 0.013).sin() * 2.0 + 10.0 + r
        })
        .collect();
    let gs: Vec<f32> = (0..inner).map(|i| 1.0 + 0.01 * i as f32).collect();
    let bs: Vec<f32> = (0..inner).map(|i| 0.05 * (i as f32).cos()).collect();

    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param("gamma", &gs);
        c.set_param("beta", &bs);
        c.run(&[("x", xs.as_slice())]).remove(0)
    };
    let cpu = run(Device::Cpu);
    let met = run(Device::Metal);
    Some(
        cpu.iter()
            .zip(&met)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max),
    )
}

#[test]
fn layer_norm_matches_cpu_across_last_axis_widths() {
    let widths = [
        32usize, 64, 96, 127, 128, 160, 192, 255, 256, 320, 512, 1024,
    ];
    let mut bad = Vec::new();
    for w in widths {
        let Some(d) = max_abs_diff(w, 16) else { return };
        eprintln!("layer_norm inner={w:5}: max_abs={d:.6e}");
        if d > 1e-4 {
            bad.push((w, d));
        }
    }
    assert!(
        bad.is_empty(),
        "metal LayerNorm disagrees with CPU at last-axis widths {bad:?}"
    );
}
