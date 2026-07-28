// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::FakeQuantizeLSQ` (learned-step-size QAT) — the full forward + backward
//! kernels already existed in rlx-cpu but were claimed in no `supported_ops`,
//! so LSQ graphs failed legalization. Validates the now-claimed forward.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn qmax(bits: u8) -> f32 {
    match bits {
        8 => 127.0,
        4 => 7.0,
        2 => 1.0,
        _ => unreachable!(),
    }
}

/// Reference: `clamp(round(x/s), -qmax, qmax) * s` per channel.
fn lsq_ref(x: &[f32], scale: &[f32], chan_dim: usize, inner: usize, bits: u8) -> Vec<f32> {
    let qm = qmax(bits);
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let c = if chan_dim == 1 {
                0
            } else {
                (i / inner) % chan_dim
            };
            let s = scale[c].max(1e-12);
            (xi / s).round().clamp(-qm, qm) * s
        })
        .collect()
}

#[test]
fn lsq_per_tensor_forward() {
    let (n, c) = (4usize, 3usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 - 6.0) * 0.3).collect();
    let scale = vec![0.25f32];
    let mut g = Graph::new("lsq");
    let xi = g.input("x", Shape::new(&[n, c], DType::F32));
    let si = g.input("scale", Shape::new(&[1], DType::F32));
    let y = g.add_node(
        Op::FakeQuantizeLSQ {
            bits: 8,
            axis: None,
        },
        vec![xi, si],
        Shape::new(&[n, c], DType::F32),
    );
    g.set_outputs(vec![y]);
    let got = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", x.as_slice()), ("scale", scale.as_slice())])
        .pop()
        .unwrap();
    let want = lsq_ref(&x, &scale, 1, 1, 8);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "per-tensor LSQ: {a} vs {b}");
    }
}

#[test]
fn lsq_per_channel_forward() {
    let (n, c) = (4usize, 3usize);
    let x: Vec<f32> = (0..n * c)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.2)
        .collect();
    let scale = vec![0.1f32, 0.25, 0.5]; // per-channel step sizes
    let mut g = Graph::new("lsq_pc");
    let xi = g.input("x", Shape::new(&[n, c], DType::F32));
    let si = g.input("scale", Shape::new(&[c], DType::F32));
    let y = g.add_node(
        Op::FakeQuantizeLSQ {
            bits: 4,
            axis: Some(1),
        },
        vec![xi, si],
        Shape::new(&[n, c], DType::F32),
    );
    g.set_outputs(vec![y]);
    let got = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", x.as_slice()), ("scale", scale.as_slice())])
        .pop()
        .unwrap();
    // [n,c] with axis=1 → chan_dim=c, inner=1.
    let want = lsq_ref(&x, &scale, c, 1, 4);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "per-channel LSQ: {a} vs {b}");
    }
}
