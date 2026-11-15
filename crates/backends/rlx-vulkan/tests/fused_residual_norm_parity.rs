// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `fused_residual_norm_row` (residual add + optional bias + LayerNorm or
//! RMSNorm, one workgroup per row) vs CPU.
//!
//! The cases are the kernel's edges: a single row (a decode step), rows
//! narrower than the 256-thread workgroup, rows wider than it (threads stride
//! the row) and of no particular size, many rows, and both with and without
//! the bias input.

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

fn check(rows: usize, h: usize, bias: bool, rms: bool) {
    let f = DType::F32;
    let build = || {
        let mut g = Graph::new("residual-norm");
        let x = g.input("x", Shape::new(&[rows, h], f));
        let res = g.input("res", Shape::new(&[rows, h], f));
        let b = bias.then(|| g.input("bias", Shape::new(&[h], f)));
        let gamma = g.input("g", Shape::new(&[h], f));
        let beta = g.input("b", Shape::new(&[h], f));
        let shape = Shape::new(&[rows, h], f);
        let o = if rms {
            g.fused_residual_rms_norm(x, res, b, gamma, beta, 1e-6, shape)
        } else {
            g.fused_residual_ln(x, res, b, gamma, beta, 1e-5, shape)
        };
        g.set_outputs(vec![o]);
        g
    };
    let (x, res, bv, gv, betav) = (
        data(rows * h, 1),
        data(rows * h, 2),
        data(h, 3),
        data(h, 4),
        data(h, 5),
    );
    let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("res", &res), ("g", &gv), ("b", &betav)];
    if bias {
        inputs.push(("bias", &bv));
    }
    let want = {
        use rlx::prelude::*;
        Session::new(Device::Cpu)
            .compile(build())
            .run(&inputs)
            .remove(0)
    };
    let got = VulkanExecutable::compile(build()).run(&inputs).remove(0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "residual {} rows={rows} h={h} bias={bias}: max |vk − cpu| = {worst:e}",
        if rms { "rms" } else { "ln" }
    );
}

#[test]
fn residual_layernorm_rows() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(1, 256, true, false);
    check(1, 1000, false, false);
    check(7, 96, true, false);
    check(577, 384, false, false);
}

#[test]
fn residual_rmsnorm_rows() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(1, 512, false, true);
    check(3, 1000, true, true);
    check(300, 64, false, true);
}
