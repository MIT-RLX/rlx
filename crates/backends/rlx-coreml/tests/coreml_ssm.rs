// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// State-space ops (Mamba selective scan, Qwen3.5 gated delta-net), lowered
// by unrolling over the sequence. Verified against the CPU backend — the
// reference executor — through the public Session API.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n ref {b:?}"
    );
}

fn seeded(n: usize, seed: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.137 + seed).sin()) * amp)
        .collect()
}

#[test]
fn selective_scan_vs_cpu() {
    let (b, s, h, n) = (1usize, 3usize, 4usize, 2usize);
    let build = || {
        let mut g = Graph::new("ssm");
        let x = g.input("x", Shape::new(&[b, s, h], DType::F32));
        let delta = g.input("delta", Shape::new(&[b, s, h], DType::F32));
        let a = g.input("a", Shape::new(&[h, n], DType::F32));
        let bb = g.input("b", Shape::new(&[b, s, n], DType::F32));
        let c = g.input("c", Shape::new(&[b, s, n], DType::F32));
        let y = g.selective_scan(x, delta, a, bb, c, n, Shape::new(&[b, s, h], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let xv = seeded(b * s * h, 0.0, 0.5);
    // Δ small & positive, A negative ⇒ stable exp(Δ·A) decay.
    let dv: Vec<f32> = (0..b * s * h)
        .map(|i| 0.1 + 0.05 * ((i as f32).sin().abs()))
        .collect();
    let av: Vec<f32> = (0..h * n).map(|i| -0.5 - 0.1 * (i as f32)).collect();
    let bvv = seeded(b * s * n, 1.0, 0.4);
    let cv = seeded(b * s * n, 2.0, 0.4);
    let feed: Vec<(&str, &[f32])> = vec![
        ("x", &xv),
        ("delta", &dv),
        ("a", &av),
        ("b", &bvv),
        ("c", &cv),
    ];

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let cpu_out = cpu.run(&feed).remove(0);
    let mut ane = Session::new(Device::Ane).compile(build());
    let ane_out = ane.run(&feed).remove(0);
    approx(&ane_out, &cpu_out, 1e-3);
}

#[test]
fn gated_delta_net_vs_cpu() {
    let (b, s, hh, n) = (1usize, 3usize, 2usize, 4usize);
    let build = || {
        let mut g = Graph::new("gdn");
        let q = g.input("q", Shape::new(&[b, s, hh, n], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, hh, n], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, hh, n], DType::F32));
        let gg = g.input("g", Shape::new(&[b, s, hh], DType::F32));
        let beta = g.input("beta", Shape::new(&[b, s, hh], DType::F32));
        let y = g.gated_delta_net(q, k, v, gg, beta, n, Shape::new(&[b, s, hh, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let qv = seeded(b * s * hh * n, 0.0, 0.3);
    let kv = seeded(b * s * hh * n, 1.0, 0.3);
    let vv = seeded(b * s * hh * n, 2.0, 0.5);
    // g log-scale, negative ⇒ exp(g) < 1 decay; beta in (0,1).
    let gv: Vec<f32> = (0..b * s * hh).map(|i| -0.2 - 0.05 * (i as f32)).collect();
    let bv: Vec<f32> = (0..b * s * hh)
        .map(|i| 0.5 + 0.1 * ((i as f32).cos()))
        .collect();
    let feed: Vec<(&str, &[f32])> = vec![
        ("q", &qv),
        ("k", &kv),
        ("v", &vv),
        ("g", &gv),
        ("beta", &bv),
    ];

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let cpu_out = cpu.run(&feed).remove(0);
    let mut ane = Session::new(Device::Ane).compile(build());
    let ane_out = ane.run(&feed).remove(0);
    approx(&ane_out, &cpu_out, 1e-3);
}
