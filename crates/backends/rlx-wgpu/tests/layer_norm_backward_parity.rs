// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-wgpu parity for `LayerNormBackwardInput` / `LayerNormBackwardGamma`
//! at the shapes that exposed two kernel defects (found porting CLEF):
//!
//! * a race in the workgroup reduction — called back-to-back with no barrier
//!   after the read, so a 64×64 input failed while 65×64 passed;
//! * dgamma only covered the first 64 columns, so any hidden size > 64 (every
//!   transformer) had wrong gamma gradients past column 63.

use rlx::prelude::*;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn vals(n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 7 + k * 13) % 23) as f32) - 11.0) * 0.09)
        .collect()
}

fn check(rows: usize, h: usize) {
    let mut g = Graph::new("ln_bwd");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.add_node(
        Op::LayerNormBackwardInput {
            axis: -1,
            eps: 1e-5,
        },
        vec![x, gamma, dy],
        Shape::new(&[rows, h], DType::F32),
    );
    let dg = g.add_node(
        Op::LayerNormBackwardGamma {
            axis: -1,
            eps: 1e-5,
        },
        vec![x, dy],
        Shape::new(&[h], DType::F32),
    );
    g.set_outputs(vec![dx, dg]);
    let (xv, gv, dyv) = (vals(rows * h, 1), vals(h, 2), vals(rows * h, 3));
    let feed = [
        ("x", xv.as_slice()),
        ("gamma", gv.as_slice()),
        ("dy", dyv.as_slice()),
    ];
    let want = Session::new(Device::Cpu).compile(g.clone()).run(&feed);
    let got = WgpuExecutable::compile(g).run(&feed);
    for (name, (a, b)) in ["dx", "dgamma"].iter().zip(got.iter().zip(&want)) {
        let worst = a
            .iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        let scale = b.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
        assert!(
            worst <= 1e-4 * scale.max(1.0),
            "{rows}×{h} {name}: max |wgpu − cpu| = {worst:e} (scale {scale:e})"
        );
    }
}

#[test]
fn layer_norm_backward_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for (rows, h) in [(64, 64), (65, 64), (192, 64), (4, 256), (3, 512), (17, 100)] {
        check(rows, h);
    }
}
