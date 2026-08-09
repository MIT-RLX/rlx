// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hot-on-GPU / cold-on-CPU MoE split, ROCm vs CPU. The CPU backend computes the
//! full grouped matmul (lossless ground truth); ROCm runs the split with a partial
//! residency mask. No-ops when ROCm is unavailable.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn assert_close(cpu: &[f32], gpu: &[f32], label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label} len");
    let max_abs = cpu
        .iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cmax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = (1e-4 * (1.0 + cmax)).max(2e-3);
    assert!(max_abs < tol, "{label}: max_abs={max_abs} tol={tol}");
}

fn build(m: usize, k: usize, n: usize, num_experts: usize) -> (Graph, Vec<f32>) {
    let mut g = Graph::new("moe_gmm_hotcold");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[num_experts, k, n], DType::F32));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let out = g.add_node(
        Op::GroupedMatMul,
        vec![x_in, w, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![out]);

    let mut w_data = vec![0f32; num_experts * k * n];
    for e in 0..num_experts {
        for i in 0..k * n {
            w_data[e * k * n + i] = ((e as f32 + 1.0) * 0.13 + i as f32 * 0.017).sin();
        }
    }
    (g, w_data)
}

fn run_case(
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    x: &[f32],
    expert_idx: &[f32],
    resident: &[bool],
    label: &str,
) {
    if !rlx_rocm::is_available() {
        return;
    }
    let (g, w_data) = build(m, k, n, num_experts);

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    cpu.set_param("w", &w_data);
    let out_cpu = cpu.run(&[("x", x), ("expert_idx", expert_idx)])[0].clone();

    let mut gpu = Session::new(Device::Rocm).compile(g);
    gpu.set_param("w", &w_data);
    gpu.set_moe_resident_experts_per_layer(&[resident]);
    let out_gpu = gpu.run(&[("x", x), ("expert_idx", expert_idx)])[0].clone();

    assert_close(&out_cpu, &out_gpu, label);
}

#[test]
fn rocm_moe_split_sorted_batch() {
    let (m, k, n, ne) = (6usize, 4usize, 3usize, 3usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.031).cos()).collect();
    let expert_idx = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &expert_idx,
        &[true, false, true],
        "sorted expert1-cold",
    );
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &expert_idx,
        &[false, true, false],
        "sorted expert0,2-cold",
    );
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &expert_idx,
        &[false, false, false],
        "sorted all-cold",
    );
}

#[test]
fn rocm_moe_split_decode_single_token() {
    let (m, k, n, ne) = (1usize, 8usize, 5usize, 4usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin()).collect();
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &[3.0],
        &[true, true, true, false],
        "decode cold-expert",
    );
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &[0.0],
        &[true, true, true, false],
        "decode hot-expert",
    );
}

#[test]
fn rocm_moe_split_unsorted_batch() {
    let (m, k, n, ne) = (8usize, 5usize, 4usize, 4usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.019).cos()).collect();
    let expert_idx = vec![0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 1.0];
    run_case(
        m,
        k,
        n,
        ne,
        &x,
        &expert_idx,
        &[true, false, true, false],
        "unsorted mixed",
    );
}
