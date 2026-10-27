// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A CoopF16Vk matmul scheduled after host steps, on both of its paths.
//!
//! * The wide-f32 fallback's bind groups were stored by schedule index but
//!   looked up by `gpu_bi`, which host steps do not advance. On discrete
//!   NVIDIA, where every elementwise op is a host step, one ReLU anywhere
//!   earlier made the fallback bind another step's group: wgpu "Validation
//!   Error", a panic.
//! * The coop path reads A's f16-window offset from a per-step map, also keyed
//!   by schedule index, so it must stay right after host steps too.
//!
//! The ReLUs run on a side branch so `A` is still a graph input (a host-op
//! output is never routed to the coop path). The process opts into the coop
//! path itself; this file holds one test so the flags cannot leak.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn rel_err(pre_relus: usize) -> f32 {
    let (m, k, n) = (64usize, 64usize, 64usize);
    let mut g = Graph::new("coop_wide_after_host");
    let z = g.input("z", Shape::new(&[256], DType::F32));
    let mut side = z;
    for _ in 0..pre_relus {
        side = g.relu(side);
    }
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.mm(x, w);
    g.set_outputs(vec![side, y]);
    let zv: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin()).collect();
    let xv: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
    let wv: Vec<f32> = (0..k * n).map(|i| 0.02 * ((i % 13) as f32 - 6.0)).collect();
    let run = |d: Device| {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param("w", &wv);
        c.run(&[("z", zv.as_slice()), ("x", xv.as_slice())])
            .remove(1)
    };
    let (gpu, cpu) = (run(Device::Gpu), run(Device::Cpu));
    let max = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
    gpu.iter()
        .zip(&cpu)
        .fold(0f32, |a, (g, c)| a.max((g - c).abs()))
        / max
}

#[test]
fn coop_matmul_after_host_steps_on_both_paths() {
    rlx_ir::env::set("RLX_WGPU_COOP_F16_VK_ENABLE", "1");
    let Some(d) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no wgpu adapter, skipping");
        return;
    };
    if !rlx_wgpu::device::coop_f16_16x16_supported()
        || rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&d.device).is_none()
    {
        eprintln!("no 16×16 f16 cooperative matmul on this adapter, skipping");
        return;
    }
    let mut bad = Vec::new();
    // Wide f32 fallback: f32 arithmetic, so CPU-close.
    rlx_ir::env::set("RLX_WGPU_COOP_F16_VK_FORCE_WIDE", "1");
    for pre in 0..3 {
        let e = rel_err(pre);
        eprintln!("wide, {pre} host step(s) before: {e:.2e}");
        if e.is_nan() || e >= 1e-5 {
            bad.push(format!("wide after {pre}: {e:.2e}"));
        }
    }
    rlx_ir::env::unset("RLX_WGPU_COOP_F16_VK_FORCE_WIDE");
    // Coop: f16 operands, so f16-close (a misplaced operand is ~100% off).
    rlx_ir::env::set("RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE", "1");
    for pre in 0..3 {
        let e = rel_err(pre);
        eprintln!("coop, {pre} host step(s) before: {e:.2e}");
        if e.is_nan() || e >= 5e-3 {
            bad.push(format!("coop after {pre}: {e:.2e}"));
        }
    }
    rlx_ir::env::unset("RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE");
    assert!(bad.is_empty(), "{bad:?}");
}
