// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `LowerConvTranspose2d` must agree with the native kernel, exactly.
//!
//! The pass exists so backends without a transposed-convolution kernel can run
//! one — which means it is only useful if it is indistinguishable from the
//! backends that do have one. CPU claims `OpKind::ConvTranspose2d`, so both
//! forms can be executed side by side here.
//!
//! The configurations are the ones Real-CUGAN actually uses (`k2 s2 p0` for the
//! U-Net upsamplers, `k4 s2 p3` for the ×2/×4 tail, `k5 s3 p2` for the ×3 tail)
//! plus the cases that exercise the parts easiest to get wrong: `output_padding`
//! (asymmetric), `dilation`, and `groups`.

use rlx_fusion::LowerConvTranspose2d;
use rlx_fusion::pass::Pass;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Deterministic, sign-varied fill — a constant would hide a mirrored kernel.
fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 7.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build(
    n: usize,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    op: [usize; 2],
    groups: usize,
) -> Graph {
    let mut g = Graph::new("ct");
    let x = g.input("x", Shape::new(&[n, cin, h, w], DType::F32));
    let wt = g.input(
        "w",
        Shape::new(&[cin, cout / groups, k[0], k[1]], DType::F32),
    );
    let out_h = (h - 1) * s[0] + d[0] * (k[0] - 1) + 1 - 2 * p[0] + op[0];
    let out_w = (w - 1) * s[1] + d[1] * (k[1] - 1) + 1 - 2 * p[1] + op[1];
    let y = g.add_node(
        Op::ConvTranspose2d {
            kernel_size: k.to_vec(),
            stride: s.to_vec(),
            padding: p.to_vec(),
            dilation: d.to_vec(),
            output_padding: op.to_vec(),
            groups,
        },
        vec![x, wt],
        Shape::new(&[n, cout, out_h, out_w], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(graph: Graph, x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(graph);
    c.run(&[("x", x), ("w", w)]).remove(0)
}

#[test]
fn lowering_matches_the_native_kernel() {
    // (name, cin, cout, h, w, k, s, p, dilation, output_padding, groups)
    let cases: &[(
        &str,
        usize,
        usize,
        usize,
        usize,
        [usize; 2],
        [usize; 2],
        [usize; 2],
        [usize; 2],
        [usize; 2],
        usize,
    )] = &[
        (
            "real-cugan up  k2 s2 p0",
            3,
            4,
            5,
            6,
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
            [0, 0],
            1,
        ),
        (
            "real-cugan x2  k4 s2 p3",
            3,
            2,
            6,
            5,
            [4, 4],
            [2, 2],
            [3, 3],
            [1, 1],
            [0, 0],
            1,
        ),
        (
            "real-cugan x3  k5 s3 p2",
            2,
            3,
            4,
            4,
            [5, 5],
            [3, 3],
            [2, 2],
            [1, 1],
            [0, 0],
            1,
        ),
        (
            "stride 1",
            2,
            2,
            5,
            5,
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 0],
            1,
        ),
        (
            "output_padding",
            2,
            2,
            4,
            4,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            [1, 1],
            1,
        ),
        (
            "dilation",
            2,
            2,
            4,
            5,
            [3, 3],
            [2, 2],
            [1, 1],
            [2, 2],
            [0, 0],
            1,
        ),
        (
            "grouped",
            4,
            4,
            4,
            4,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            [0, 0],
            2,
        ),
        (
            "asymmetric k/s/p",
            2,
            3,
            5,
            4,
            [3, 2],
            [2, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            1,
        ),
    ];

    for &(name, cin, cout, h, w, k, s, p, d, op, groups) in cases {
        let g = build(1, cin, cout, h, w, k, s, p, d, op, groups);
        let xs = ramp(cin * h * w, 1);
        let ws = ramp(cin * (cout / groups) * k[0] * k[1], 2);

        let native = run(g.clone(), &xs, &ws);
        let lowered_graph = LowerConvTranspose2d.run(g);
        assert!(
            !lowered_graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ConvTranspose2d { .. })),
            "{name}: the pass left a ConvTranspose2d behind"
        );
        let lowered = run(lowered_graph, &xs, &ws);

        assert_eq!(native.len(), lowered.len(), "{name}: output size differs");
        for (i, (a, b)) in native.iter().zip(&lowered).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "{name}: element {i} native {a} vs lowered {b}"
            );
        }
    }
}

/// `output_padding` is added to the trailing edge only. A symmetric version
/// produces a correctly *sized* tensor whose contents are shifted, so the size
/// check above cannot catch it — this pins the values at the border.
#[test]
fn output_padding_is_trailing_only() {
    let g = build(1, 1, 1, 3, 3, [3, 3], [2, 2], [1, 1], [1, 1], [1, 1], 1);
    let xs = ramp(9, 3);
    let ws = ramp(9, 4);
    let native = run(g.clone(), &xs, &ws);
    let lowered = run(LowerConvTranspose2d.run(g), &xs, &ws);
    // 6×6 output: the last row and column are the ones `output_padding` adds.
    assert_eq!(native.len(), 36);
    for (i, (a, b)) in native.iter().zip(&lowered).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "element {i}: native {a} vs lowered {b}"
        );
    }
}
