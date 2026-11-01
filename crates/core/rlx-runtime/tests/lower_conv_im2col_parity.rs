// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `LowerConvIm2col` (Conv → Im2Col + matmul, for backends with a GEMM but no
//! convolution kernel, such as the XDNA NPU) must agree with the native
//! kernel. CPU claims both `Conv` and `Im2Col`, so both forms run side by
//! side here, over plain, strided, padded, dilated, depthwise (with a channel
//! multiplier) and grouped convolutions — the forms the Zipformer2 conv
//! embed and conv modules use.

use rlx_fusion::LowerConvIm2col;
use rlx_fusion::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 7.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build(
    n: usize,
    c: usize,
    o: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    groups: usize,
) -> Graph {
    let mut g = Graph::new("conv");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let wt = g.input("w", Shape::new(&[o, c / groups, k[0], k[1]], DType::F32));
    let y = g.conv2d(x, wt, k, s, p, d, groups);
    g.set_outputs(vec![y]);
    g
}

fn run(graph: Graph, x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(graph);
    c.run(&[("x", x), ("w", w)]).remove(0)
}

#[test]
fn lowering_matches_the_native_kernel() {
    // (name, n, c, o, h, w, k, s, p, d, groups)
    #[allow(clippy::type_complexity)]
    let cases: &[(
        &str,
        usize,
        usize,
        usize,
        usize,
        usize,
        [usize; 2],
        [usize; 2],
        [usize; 2],
        [usize; 2],
        usize,
    )] = &[
        (
            "plain 3x3",
            2,
            3,
            5,
            7,
            6,
            [3, 3],
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        ),
        (
            "pad 1 freq only",
            1,
            1,
            8,
            9,
            10,
            [3, 3],
            [1, 1],
            [0, 1],
            [1, 1],
            1,
        ),
        (
            "stride 2x2",
            2,
            8,
            6,
            9,
            11,
            [3, 3],
            [2, 2],
            [0, 0],
            [1, 1],
            1,
        ),
        (
            "stride 1x2",
            1,
            4,
            6,
            7,
            9,
            [3, 3],
            [1, 2],
            [0, 0],
            [1, 1],
            1,
        ),
        (
            "dilation 2",
            1,
            3,
            4,
            9,
            9,
            [3, 3],
            [1, 1],
            [2, 2],
            [2, 2],
            1,
        ),
        (
            "depthwise 7x7 pad 0,3",
            2,
            6,
            6,
            10,
            8,
            [7, 7],
            [1, 1],
            [0, 3],
            [1, 1],
            6,
        ),
        (
            "depthwise 1xk (conv1d)",
            2,
            5,
            5,
            1,
            12,
            [1, 5],
            [1, 1],
            [0, 2],
            [1, 1],
            5,
        ),
        (
            "depthwise multiplier 2",
            1,
            3,
            6,
            6,
            6,
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            3,
        ),
        ("groups 2", 2, 4, 6, 6, 7, [3, 3], [1, 1], [1, 1], [1, 1], 2),
        (
            "pointwise 1x1",
            2,
            6,
            4,
            5,
            5,
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        ),
    ];
    for &(name, n, c, o, h, w, k, s, p, d, groups) in cases {
        let g = build(n, c, o, h, w, k, s, p, d, groups);
        let x = ramp(n * c * h * w, 1);
        let wt = ramp(o * (c / groups) * k[0] * k[1], 2);
        let native = run(g.clone(), &x, &wt);
        let lowered_graph = LowerConvIm2col.run(g);
        assert!(
            !lowered_graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Conv { .. })),
            "{name}: Conv left in the graph"
        );
        let lowered = run(lowered_graph, &x, &wt);
        assert_eq!(native.len(), lowered.len(), "{name}: length");
        let scale = native.iter().fold(1f32, |m, v| m.max(v.abs()));
        let worst = native
            .iter()
            .zip(&lowered)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-4 * scale,
            "{name}: max |Δ| {worst} (scale {scale})"
        );
    }
}
