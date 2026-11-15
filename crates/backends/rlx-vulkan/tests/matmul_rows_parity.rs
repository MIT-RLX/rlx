// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_rows` (matmuls with m ≤ 8: a workgroup per 32 columns, K split
//! over 8 lane groups) vs CPU.
//!
//! Covers what a K-split, column-tiled kernel can get wrong: m from 1 to the
//! 8-row cap, K that is not a multiple of the 8 slices, N that leaves a
//! partial 32-column tile, a batch with a broadcast weight, and the weight in
//! the separate weight buffer (a `Param`) as well as in the activation arena.

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

/// `a [batch, m, k] · w [k, n]`, the weight a `Param` or an input.
fn build(batch: usize, m: usize, k: usize, n: usize, weight_param: bool) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mm");
    let a = g.input("a", Shape::new(&[batch, m, k], f));
    let w = if weight_param {
        g.param("w", Shape::new(&[k, n], f))
    } else {
        g.input("w", Shape::new(&[k, n], f))
    };
    let y = g.matmul(a, w, Shape::new(&[batch, m, n], f));
    g.set_outputs(vec![y]);
    g
}

fn check(batch: usize, m: usize, k: usize, n: usize, weight_param: bool) {
    let a = data(batch * m * k, 1);
    let w = data(k * n, 2);
    let want: Vec<f32> = (0..batch * m * n)
        .map(|i| {
            let (bm, c) = (i / n, i % n);
            (0..k)
                .map(|kk| f64::from(a[bm * k + kk]) * f64::from(w[kk * n + c]))
                .sum::<f64>() as f32
        })
        .collect();
    let mut exe = VulkanExecutable::compile(build(batch, m, k, n, weight_param));
    let got = if weight_param {
        exe.set_param("w", &w);
        exe.run(&[("a", &a)]).remove(0)
    } else {
        exe.run(&[("a", &a), ("w", &w)]).remove(0)
    };
    let worst = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "matmul [{batch},{m},{k}]·[{k},{n}] (param={weight_param}): max |vk − ref| = {worst:e}"
    );
}

#[test]
fn few_rows_against_a_weight_param() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    for m in [1, 3, 8] {
        check(1, m, 256, 1024, true);
    }
    // K not a multiple of the 8 slices, N a partial tile.
    check(1, 2, 37, 50, true);
}

#[test]
fn few_rows_activations_and_batch() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(1, 1, 64, 33, false);
    check(3, 4, 129, 70, false);
}

/// `FusedMatMulBiasAct` with `matmul_rows`' epilogue (bias + activation in
/// the same dispatch) vs the CPU backend running the same graph.
fn check_fused(m: usize, k: usize, n: usize, act: Option<rlx_ir::op::Activation>) {
    let f = DType::F32;
    let build = || {
        let mut g = Graph::new("linear");
        let a = g.input("a", Shape::new(&[1, m, k], f));
        let w = g.param("w", Shape::new(&[k, n], f));
        let b = g.param("b", Shape::new(&[n], f));
        let y = g.add_node(
            rlx_ir::op::Op::FusedMatMulBiasAct { activation: act },
            vec![a, w, b],
            Shape::new(&[1, m, n], f),
        );
        g.set_outputs(vec![y]);
        g
    };
    let (a, w, b) = (data(m * k, 4), data(k * n, 5), data(n, 6));
    let want = {
        use rlx::prelude::*;
        let mut exe = Session::new(Device::Cpu).compile(build());
        exe.set_param("w", &w);
        exe.set_param("b", &b);
        exe.run(&[("a", &a)]).remove(0)
    };
    let mut exe = VulkanExecutable::compile(build());
    exe.set_param("w", &w);
    exe.set_param("b", &b);
    let got = exe.run(&[("a", &a)]).remove(0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "fused linear m={m} k={k} n={n} {act:?}: max |vk − cpu| = {worst:e}"
    );
}

#[test]
fn fused_linear_epilogue() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    use rlx_ir::op::Activation;
    check_fused(1, 256, 256, Some(Activation::Relu));
    check_fused(5, 96, 70, Some(Activation::Gelu));
    check_fused(1, 64, 300, None);
}
