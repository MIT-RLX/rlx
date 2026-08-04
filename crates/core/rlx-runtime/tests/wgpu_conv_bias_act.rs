// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! wgpu native `FusedConvBiasAct` (`fused_conv_bias_act.wgsl`) parity vs CPU.
//!
//! wgpu keeps `Op::FusedConvBiasAct` fused — one kernel that runs the conv and
//! applies the `act(conv + bias[c] + residual)` epilogue in-register — instead
//! of the host round-trip it used before. CPU expands the op back to
//! `Conv → Add(bias) → [residual] → Activation` primitives, so it is the oracle.
//! This pins the native kernel against CPU for the relu / identity / residual
//! epilogues (act ids 0 and 0xFFFF, `has_residual` on and off).

#![cfg(feature = "gpu")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

struct Cfg {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
}

/// A graph whose only compute node is a directly-constructed
/// `FusedConvBiasAct` (bypasses the fusion matcher so the native arm is
/// exercised regardless of the fuser's shape gates), inputs `[x, w, b, (res)]`.
fn fused_graph(cfg: &Cfg, act: Option<Activation>, has_residual: bool) -> Graph {
    let mut g = Graph::new("conv_bias_act");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], F));
    let weight = g.input("w", Shape::new(&[cfg.c_out, cfg.c_in, cfg.k, cfg.k], F));
    let bias = g.input("b", Shape::new(&[cfg.c_out], F));
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], F);
    let mut inputs = vec![x, weight, bias];
    if has_residual {
        inputs.push(g.input("r", out_s.clone()));
    }
    let y = g.add_node(
        Op::FusedConvBiasAct {
            kernel_size: vec![cfg.k, cfg.k],
            stride: vec![1, 1],
            padding: vec![cfg.p, cfg.p],
            dilation: vec![1, 1],
            groups: 1,
            activation: act,
            has_residual,
        },
        inputs,
        out_s,
    );
    g.set_outputs(vec![y]);
    g
}

fn feeds(cfg: &Cfg) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.c_in * cfg.h * cfg.w)
        .map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * cfg.c_in * cfg.k * cfg.k)
        .map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5)
        .collect();
    // Negative-leaning bias so relu actually clamps some outputs.
    let b: Vec<f32> = (0..cfg.c_out).map(|i| i as f32 * 0.05 - 0.3).collect();
    let r: Vec<f32> = (0..cfg.c_out * cfg.h * cfg.w)
        .map(|i| (i * 3 % 19) as f32 / 19.0 - 0.5)
        .collect();
    (x, w, b, r)
}

#[test]
fn wgpu_fused_conv_bias_act_matches_cpu() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_fused_conv_bias_act (wgpu unavailable)");
        return;
    }
    let cfg = Cfg {
        c_in: 3,
        c_out: 5,
        h: 7,
        w: 6,
        k: 3,
        p: 1,
    };
    let (x, w, b, r) = feeds(&cfg);

    for (act, has_residual) in [
        (Some(Activation::Relu), false),
        (None, false),
        (Some(Activation::Relu), true),
    ] {
        let mut feed: Vec<(&str, &[f32])> = vec![
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("b", b.as_slice()),
        ];
        if has_residual {
            feed.push(("r", r.as_slice()));
        }

        let cpu = Session::new(Device::Cpu)
            .compile(fused_graph(&cfg, act, has_residual))
            .run(&feed)
            .remove(0);
        let gpu = Session::new(Device::Gpu)
            .compile(fused_graph(&cfg, act, has_residual))
            .run(&feed)
            .remove(0);

        assert_eq!(
            cpu.len(),
            gpu.len(),
            "len mismatch act={act:?} res={has_residual}"
        );
        let max = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "wgpu FusedConvBiasAct act={act:?} residual={has_residual} max_abs={max:.3e}"
        );
    }
}
