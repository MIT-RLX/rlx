// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu vs CPU for matmuls that now take CoopF32 through
//! `matmul_coop_f32_edge.wgsl` on Metal: K not a multiple of 8, N not a
//! multiple of 32, and B an ACTIVATION (not a Param) — multi-head attention at
//! head_dim 25 (`[S,25]·[25,S]`, `[S,S]·[S,25]`) and d_model-200 projections.
//! Before, every one of these fell to the scalar tiled kernel; the edge kernel
//! zero-pads both operands on ragged tiles, so a wrong pad shows up here as a
//! large error rather than a rounding-level one.

use rlx_ir::infer::GraphExt as _;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len(), "output length differs");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale
}

fn both(build: impl Fn() -> Graph, inputs: &[(&str, &[f32])]) -> (Vec<f32>, Vec<f32>) {
    let cpu = Session::new(Device::Cpu).compile(build()).run(inputs);
    let gpu = Session::new(Device::Gpu).compile(build()).run(inputs);
    (
        cpu.into_iter().next().unwrap(),
        gpu.into_iter().next().unwrap(),
    )
}

fn bmm(batch: usize, m: usize, k: usize, n: usize) {
    let a = noise(batch * m * k, 1);
    let b = noise(batch * k * n, 2);
    let (cpu, gpu) = both(
        || {
            let mut g = Graph::new("bmm-act");
            let x = g.input("a", Shape::new(&[batch, m, k], F));
            let y = g.input("b", Shape::new(&[batch, k, n], F));
            let z = g.mm(x, y);
            g.set_outputs(vec![z]);
            g
        },
        &[("a", &a), ("b", &b)],
    );
    let e = rel_err(&cpu, &gpu);
    assert!(e < 1e-5, "bmm {batch}x{m}x{k}x{n}: rel err {e:.3e}");
}

#[test]
fn activation_b_ragged_k_and_n() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    bmm(16, 191, 25, 191); // scores: K % 8 != 0, N % 32 != 0
    bmm(16, 191, 191, 25); // context: both ragged
    bmm(3, 37, 13, 41);
}

fn linear(m: usize, k: usize, n: usize, act: Option<Activation>) {
    let x = noise(m * k, 3);
    let w = noise(k * n, 4);
    let bias = noise(n, 5);
    // Weight and bias as Params — the case CoopF32 always took when aligned.
    let build = || {
        let mut g = Graph::new("linear");
        let xi = g.input("x", Shape::new(&[m, k], F));
        let wi = g.param("w", Shape::new(&[k, n], F));
        let bi = g.param("bias", Shape::new(&[n], F));
        let y = g.linear_fused(xi, wi, bi, act, Shape::new(&[m, n], F));
        g.set_outputs(vec![y]);
        g
    };
    let run = |dev: Device| {
        let mut exe = Session::new(dev).compile(build());
        exe.set_param("w", &w);
        exe.set_param("bias", &bias);
        exe.run(&[("x", x.as_slice())]).into_iter().next().unwrap()
    };
    let (cpu, gpu) = (run(Device::Cpu), run(Device::Gpu));
    let e = rel_err(&cpu, &gpu);
    assert!(e < 1e-5, "linear {m}x{k}x{n} act={act:?}: rel err {e:.3e}");
}

#[test]
fn unaligned_param_linear() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for act in [None, Some(Activation::Gelu)] {
        linear(191, 200, 600, act); // N % 32 != 0
        linear(191, 200, 200, act);
        linear(191, 21, 77, act); // K % 8 != 0 too
    }
}
