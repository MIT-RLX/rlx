// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Depthwise `Op::Conv` vs CPU when the kernel is WIDER than the sequence —
//! rlx-metal.
//!
//! W2v-BERT's Conformer conv module uses `conv_depthwise_kernel_size = 31` over
//! 1024 channels, and `rlx-tribev2-audio`'s parity input is only 16 frames
//! long. "Same" padding then reaches past both ends of the input at once, which
//! is a regime a conv kernel can easily get wrong while still being exact for
//! the usual seq >> k case.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn case(channels: usize, seq: usize, k: usize) -> Option<f32> {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        eprintln!("skip: metal unavailable");
        return None;
    }
    let pad = k / 2;
    let mut g = Graph::new("dw_conv");
    // NCHW with a height of 1: [B, C, 1, S], kernel [C, 1, 1, k], groups = C.
    let x = g.input("x", Shape::new(&[1, channels, 1, seq], DType::F32));
    let w = g.param("w", Shape::new(&[channels, 1, 1, k], DType::F32));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![1, k],
            stride: vec![1, 1],
            padding: vec![0, pad],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![x, w],
        Shape::new(&[1, channels, 1, seq + 2 * pad - k + 1], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs: Vec<f32> = (0..channels * seq)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let ws: Vec<f32> = (0..channels * k)
        .map(|i| ((i as f32) * 0.031).cos())
        .collect();
    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param("w", &ws);
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
fn depthwise_conv_matches_cpu_when_kernel_exceeds_seq() {
    // (channels, seq, k): the last is W2v-BERT's real shape.
    let cases = [(8usize, 64usize, 3usize), (8, 16, 31), (1024, 16, 31)];
    let mut bad = Vec::new();
    for (c, s, k) in cases {
        let Some(d) = case(c, s, k) else { return };
        eprintln!("depthwise C={c:5} S={s:3} k={k:3}: max_abs={d:.6e}");
        if d > 1e-4 {
            bad.push((c, s, k, d));
        }
    }
    assert!(
        bad.is_empty(),
        "metal depthwise conv disagrees with CPU: {bad:?}"
    );
}
