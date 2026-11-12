// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-CUDA parity for `LayerNormBackwardInput` / `LayerNormBackwardGamma`.
//! The gamma kernel is column-tiled (32 columns per block, 32 warps over rows),
//! so the shapes cover rows not divisible by 32, hidden sizes not divisible by
//! 32, and a transformer-sized 8192×384 (the size that made the old serial
//! kernel cost ~85 ms per call).

use rlx_cuda::CudaExecutable;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

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
    let got = CudaExecutable::compile(g).run(&feed);
    for (name, (a, b)) in ["dx", "dgamma"].iter().zip(got.iter().zip(&want)) {
        assert_eq!(a.len(), b.len(), "{rows}×{h} {name}: length");
        let worst = a
            .iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        let scale = b.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
        assert!(
            worst <= 1e-4 * scale.max(1.0),
            "{rows}×{h} {name}: max |cuda − cpu| = {worst:e} (scale {scale:e})"
        );
    }
}

#[test]
fn layer_norm_backward_matches_cpu() {
    if rlx_ir::env::skip_unless_device("cuda", true, rlx_cuda::is_available()) {
        return;
    }
    for (rows, h) in [
        (1, 1),
        (64, 64),
        (65, 64),
        (7, 33),
        (17, 100),
        (3, 512),
        (8192, 384),
    ] {
        check(rows, h);
    }
}
