// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native WGSL kernels for `Op::Gru`/`Op::Rnn`/`Op::Mamba2` on wgpu
//! (single-layer / unidir / no-carry → native path; `state_size ≤ 256` for
//! Mamba2, `hidden ≤ 256` for Gru/Rnn) vs CPU. With the ops claimed in
//! `WGPU_SUPPORTED_OPS` the legalizer keeps the fused op (no decomposition).

#![cfg(feature = "gpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn ramp(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 23) as f32 - 11.0) * scale)
        .collect()
}

fn close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    let m = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(m <= 1e-3, "{what}: max abs diff {m:e} > 1e-3");
    eprintln!("{what}: max abs diff {m:.2e}");
}

#[test]
fn gru_native_wgpu_matches_cpu() {
    let (b, s, inp, h) = (2usize, 5, 6, 8);
    let inputs = [
        ("x", ramp(b * s * inp, 1, 0.1)),
        ("wih", ramp(3 * h * inp, 2, 0.05)),
        ("whh", ramp(3 * h * h, 3, 0.05)),
        ("bih", ramp(3 * h, 4, 0.02)),
        ("bhh", ramp(3 * h, 5, 0.02)),
    ];
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], DType::F32));
        let wih = g.input("wih", Shape::new(&[3 * h, inp], DType::F32));
        let whh = g.input("whh", Shape::new(&[3 * h, h], DType::F32));
        let bih = g.input("bih", Shape::new(&[3 * h], DType::F32));
        let bhh = g.input("bhh", Shape::new(&[3 * h], DType::F32));
        let y = g.gru(
            x,
            wih,
            whh,
            bih,
            bhh,
            None,
            h,
            1,
            false,
            Shape::new(&[b, s, h], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |d| {
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Session::new(d).compile(build()).run(&refs).pop().unwrap()
    };
    close("gru native wgpu", &run(Device::Gpu), &run(Device::Cpu));
}

#[test]
fn rnn_native_wgpu_matches_cpu() {
    let (b, s, inp, h) = (2usize, 5, 6, 8);
    for relu in [false, true] {
        let inputs = [
            ("x", ramp(b * s * inp, 1, 0.1)),
            ("wih", ramp(h * inp, 2, 0.05)),
            ("whh", ramp(h * h, 3, 0.05)),
            ("bias", ramp(h, 4, 0.02)),
        ];
        let build = || {
            let mut g = Graph::new("rnn");
            let x = g.input("x", Shape::new(&[b, s, inp], DType::F32));
            let wih = g.input("wih", Shape::new(&[h, inp], DType::F32));
            let whh = g.input("whh", Shape::new(&[h, h], DType::F32));
            let bias = g.input("bias", Shape::new(&[h], DType::F32));
            let y = g.rnn(
                x,
                wih,
                whh,
                bias,
                None,
                h,
                1,
                false,
                relu,
                Shape::new(&[b, s, h], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        };
        let run = |d| {
            let refs: Vec<(&str, &[f32])> =
                inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
            Session::new(d).compile(build()).run(&refs).pop().unwrap()
        };
        close(
            &format!("rnn native wgpu relu={relu}"),
            &run(Device::Gpu),
            &run(Device::Cpu),
        );
    }
}

#[test]
fn gru_bidir_wgpu_host_fallback_matches_cpu() {
    // Bidirectional → not the native single-layer/unidir path → GruHost.
    let (b, s, inp, h) = (2usize, 4, 5, 6);
    let inputs = [
        ("x", ramp(b * s * inp, 1, 0.1)),
        // 2 directions packed: [2 * 3h, in] then [2 * 3h, h] etc.
        ("wih", ramp(2 * 3 * h * inp, 2, 0.05)),
        ("whh", ramp(2 * 3 * h * h, 3, 0.05)),
        ("bih", ramp(2 * 3 * h, 4, 0.02)),
        ("bhh", ramp(2 * 3 * h, 5, 0.02)),
    ];
    let build = || {
        let mut g = Graph::new("gru_bidir");
        let x = g.input("x", Shape::new(&[b, s, inp], DType::F32));
        let wih = g.input("wih", Shape::new(&[2 * 3 * h, inp], DType::F32));
        let whh = g.input("whh", Shape::new(&[2 * 3 * h, h], DType::F32));
        let bih = g.input("bih", Shape::new(&[2 * 3 * h], DType::F32));
        let bhh = g.input("bhh", Shape::new(&[2 * 3 * h], DType::F32));
        let y = g.gru(
            x,
            wih,
            whh,
            bih,
            bhh,
            None,
            h,
            1,
            true,
            Shape::new(&[b, s, 2 * h], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |d| {
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Session::new(d).compile(build()).run(&refs).pop().unwrap()
    };
    close(
        "gru bidir wgpu (host fallback)",
        &run(Device::Gpu),
        &run(Device::Cpu),
    );
}

#[test]
fn mamba2_native_wgpu_matches_cpu() {
    let (b, s, h, p, n) = (2usize, 4, 3, 8, 16);
    let inputs = [
        ("x", ramp(b * s * h * p, 1, 0.06)),
        (
            "dt",
            (0..b * s * h)
                .map(|i| 0.2 + 0.05 * (i % 5) as f32)
                .collect(),
        ),
        ("a", (0..h).map(|i| -0.5 - 0.3 * i as f32).collect()),
        ("b", ramp(b * s * h * n, 3, 0.05)),
        ("c", ramp(b * s * h * n, 7, 0.05)),
    ];
    let build = || {
        let mut g = Graph::new("mamba2");
        let x = g.input("x", Shape::new(&[b, s, h, p], DType::F32));
        let dt = g.input("dt", Shape::new(&[b, s, h], DType::F32));
        let a = g.input("a", Shape::new(&[h], DType::F32));
        let bb = g.input("b", Shape::new(&[b, s, h, n], DType::F32));
        let cc = g.input("c", Shape::new(&[b, s, h, n], DType::F32));
        let y = g.mamba2(
            x,
            dt,
            a,
            bb,
            cc,
            p,
            n,
            Shape::new(&[b, s, h, p], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |d| {
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Session::new(d).compile(build()).run(&refs).pop().unwrap()
    };
    close("mamba2 native wgpu", &run(Device::Gpu), &run(Device::Cpu));
}
