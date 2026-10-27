// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! A compiled graph's later runs must match its first when a skipped
//! static-once weight pack sits between a host op and a GPU matmul.
//!
//! Found through `rlx-montageflow` on Linux/Vulkan (discrete NVIDIA hosts
//! elementwise ops and norms). A DiT final layer: `silu(c)` feeds two matmuls
//! that `FuseSharedInputMatMul` merges, so their weights become a host
//! `Concat` marked static-once. On run 2 that Concat is skipped. The flush
//! lookahead after the host Silu saw "next step is the Concat, a host step —
//! keep deferring", the loop then jumped to the GPU matmul, and that pass
//! cleared the host mirror with the deferred outputs still unflushed. The
//! LayerNorm after it read its input slot's PREVIOUS tenant from the device:
//! run 1 exact, every later run wrong (cosine −0.12).
//!
//! `RLX_WGPU_FORCE_HOST=1` hosts the same ops on any adapter, so this
//! reproduces on Metal as well as discrete Vulkan.
//!
//!     cargo test -p rlx-wgpu --test static_pack_skip_rerun -- --nocapture

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn rel_max(a: &[f32], b: &[f32]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
        / scale
}

/// `proj(LN((x + pe) · pe2) * (1 + scale) + shift)` with `[shift | scale]` from two
/// matmuls on the shared input `silu(c)` — the DiT final layer.
fn final_layer(b: usize, s: usize, d: usize, out: usize) -> Graph {
    let mut g = Graph::new("static_pack_rerun");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let c = g.input("c", Shape::new(&[b, d], DType::F32));
    let pe = g.param("pe", Shape::new(&[b, s, d], DType::F32));
    let pe2 = g.param("pe2", Shape::new(&[b, s, d], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[d], DType::F32));
    let beta = g.param("beta", Shape::new(&[d], DType::F32));
    let wa = g.param("wa", Shape::new(&[d, d], DType::F32));
    let wb = g.param("wb", Shape::new(&[d, d], DType::F32));
    let wo = g.param("wo", Shape::new(&[d, out], DType::F32));
    // Two host ops BEFORE the Silu, so their outputs are still deferred when
    // the skipped Concat comes up. The Mul keeps the LN from fusing with the
    // Add (a FusedResidualLN would be scheduled after the matmul instead).
    let h = g.add(x, pe);
    let h = g.mul(h, pe2);
    let sc = g.silu(c);
    let shift = g.mm(sc, wa);
    let scale = g.mm(sc, wb);
    let h = g.ln(h, gamma, beta, 1e-5);
    let scale = g.reshape_(scale, vec![b as i64, 1, d as i64]);
    let shift = g.reshape_(shift, vec![b as i64, 1, d as i64]);
    let one = g.full(&[1, 1, d], 1.0, DType::F32);
    let scale = g.add(scale, one);
    let h = g.mul(h, scale);
    let h = g.add(h, shift);
    let y = g.mm(h, wo);
    g.set_outputs(vec![y]);
    g
}

fn wave(n: usize, f: f32, a: f32) -> Vec<f32> {
    (0..n).map(|i| a * ((i as f32) * f).sin()).collect()
}

#[test]
fn skipped_weight_pack_does_not_drop_deferred_host_writes() {
    // Pin f32 matmul: the f16 opt-in (`RLX_WGPU_COOP_F16_VK_ENABLE`) gives up
    // the 1e-5 agreement with CPU this file checks.
    rlx_ir::env::remove("RLX_WGPU_COOP_F16_VK_ENABLE");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    let (b, s, d, out) = (2usize, 64usize, 128usize, 32usize);
    let g = final_layer(b, s, d, out);
    let params = [
        ("pe", wave(b * s * d, 0.031, 1.0)),
        (
            "pe2",
            wave(b * s * d, 0.029, 1.0)
                .iter()
                .map(|v| 1.0 + 0.5 * v)
                .collect(),
        ),
        (
            "gamma",
            wave(d, 0.7, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect(),
        ),
        ("beta", wave(d, 0.3, 0.1)),
        ("wa", wave(d * d, 0.013, 0.1)),
        ("wb", wave(d * d, 0.017, 0.1)),
        ("wo", wave(d * out, 0.011, 0.1)),
    ];
    let compile = |dev: Device| {
        let mut c = Session::new(dev).compile(g.clone());
        for (k, v) in &params {
            c.set_param(k, v);
        }
        c
    };
    let (mut cpu, mut gpu) = (compile(Device::Cpu), compile(Device::Gpu));
    let mut bad = Vec::new();
    // Same inputs every run: a correct run 2 is then bit-identical to run 1,
    // and a wrong one cannot hide behind input-dependent noise.
    let x = wave(b * s * d, 0.019, 2.0);
    let c = wave(b * d, 0.23, 3.0);
    let mut first: Option<Vec<f32>> = None;
    for r in 1..=3 {
        let want = cpu.run(&[("x", &x), ("c", &c)]).remove(0);
        let got = gpu.run(&[("x", &x), ("c", &c)]).remove(0);
        let e = rel_max(&got, &want);
        eprintln!("run {r}: rel_max vs CPU {e:.3e}");
        if e.is_nan() || e >= 1e-5 {
            bad.push(format!("run {r}: {e:.3e}"));
        }
        match &first {
            None => first = Some(got),
            Some(f) if f != &got => bad.push(format!("run {r} differs bitwise from run 1")),
            _ => {}
        }
    }
    rlx_ir::env::unset("RLX_WGPU_FORCE_HOST");
    assert!(
        bad.is_empty(),
        "runs of one compiled graph disagree: {bad:?}"
    );
}
