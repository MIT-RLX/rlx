// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `conv2d_tiled` (16×16 output tiles × 8 channels, input patch in shared
//! memory) vs CPU, plus the shapes that must stay on the direct kernel.
//!
//! The cases are the tiling edges: output sizes that leave partial tiles,
//! output channels that are not a multiple of the 8-channel block, the
//! asymmetric (2, 1) stride a page encoder halves only the height with, stride
//! 2 with dilation 2 (the widest input patch), 1×1 and 7×7 kernels, a bias
//! input, a batch, and a depthwise conv (groups = C, the direct kernel).
//!
//! The bias rides as `Op::Conv`'s optional third input, which the Vulkan
//! kernels apply and the CPU backend does not: the reference adds it on the
//! host.

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;

mod common;

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 2001) as f32 / 1000.0) - 1.0)
        .collect()
}

#[derive(Clone, Copy)]
struct Case {
    n: usize,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    k: [usize; 2],
    stride: [usize; 2],
    pad: [usize; 2],
    dil: [usize; 2],
    groups: usize,
    bias: bool,
}

fn build(c: &Case) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv");
    let x = g.input("x", Shape::new(&[c.n, c.cin, c.h, c.w], f));
    let w = g.input(
        "w",
        Shape::new(&[c.cout, c.cin / c.groups, c.k[0], c.k[1]], f),
    );
    let y = g.conv2d(x, w, c.k, c.stride, c.pad, c.dil, c.groups);
    if c.bias {
        // Same op with the optional third (bias) input.
        let b = g.input("b", Shape::new(&[c.cout], f));
        let shape = g.node(y).shape.clone();
        let op = g.node(y).op.clone();
        let yb = g.add_node(op, vec![x, w, b], shape);
        g.set_outputs(vec![yb]);
    } else {
        g.set_outputs(vec![y]);
    }
    g
}

fn check(c: Case) {
    let x = data(c.n * c.cin * c.h * c.w, 1);
    let w = data(c.cout * (c.cin / c.groups) * c.k[0] * c.k[1], 2);
    let b = data(c.cout, 3);
    let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w)];
    if c.bias {
        inputs.push(("b", &b));
    }
    // The CPU backend runs `Op::Conv` without its optional bias input, so the
    // reference is the bias-free CPU conv plus the per-channel bias.
    let mut want = {
        use rlx::prelude::*;
        let plain = Case { bias: false, ..c };
        Session::new(Device::Cpu)
            .compile(build(&plain))
            .run(&inputs[..2])
            .remove(0)
    };
    if c.bias {
        let plane = want.len() / (c.n * c.cout);
        for (i, v) in want.iter_mut().enumerate() {
            *v += b[(i / plane) % c.cout];
        }
    }
    let got = VulkanExecutable::compile(build(&c)).run(&inputs).remove(0);
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-3,
        "conv n={} {}→{} {}x{} k={:?} s={:?} d={:?} g={} bias={}: max |vk − cpu| = {worst:e}",
        c.n,
        c.cin,
        c.cout,
        c.h,
        c.w,
        c.k,
        c.stride,
        c.dil,
        c.groups,
        c.bias
    );
}

#[test]
fn dense_convs_take_the_tiled_kernel_and_match() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    let base = |cin, cout, h, w, k, stride, pad, dil, bias| Case {
        n: 1,
        cin,
        cout,
        h,
        w,
        k,
        stride,
        pad,
        dil,
        groups: 1,
        bias,
    };
    // 3×3 same-padding, partial tiles, 13 output channels.
    check(base(3, 13, 50, 37, [3, 3], [1, 1], [1, 1], [1, 1], true));
    // DAN's height-only downsampling.
    check(base(16, 16, 41, 33, [3, 3], [2, 1], [1, 1], [1, 1], true));
    // Widest input patch: stride 2, dilation 2.
    check(base(5, 9, 40, 40, [3, 3], [2, 2], [2, 2], [2, 2], false));
    // 1×1 and 7×7.
    check(base(32, 24, 19, 21, [1, 1], [1, 1], [0, 0], [1, 1], true));
    check(base(4, 8, 30, 30, [7, 7], [1, 1], [3, 3], [1, 1], false));
    // Batch of two.
    check(Case {
        n: 2,
        ..base(6, 10, 17, 18, [3, 3], [1, 1], [1, 1], [1, 1], true)
    });
}

#[test]
fn depthwise_stays_direct_and_matches() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        n: 1,
        cin: 12,
        cout: 12,
        h: 20,
        w: 23,
        k: [3, 3],
        stride: [1, 1],
        pad: [1, 1],
        dil: [1, 1],
        groups: 12,
        bias: true,
    });
}
