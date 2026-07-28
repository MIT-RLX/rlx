// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native low-precision `Op::ScaledMatMul` on Vulkan with a parameterized
//! `ScaledFormat::Custom` minifloat (`f4e3m0`) and a named FP8 format.
//!
//! Vulkan has no FP8/FP4 matrix path, so the four scaled ops run the CPU
//! decode-and-accumulate reference against the host-visible mapped arena (the
//! same `rlx-cpu` oracle the CPU backend uses) — so Vulkan must match a plain
//! f32 matmul for grid-aligned inputs and track it (cosine) otherwise. Skips
//! (no-op) on hosts with no Vulkan driver.

use rlx_ir::{DType, Graph, Op, ScaleLayout, ScaledFormat, Shape};
use rlx_vulkan::backend::VulkanExecutable;

fn build_scaled_mm_graph(
    fmt: ScaledFormat,
    layout: ScaleLayout,
    m: usize,
    k: usize,
    n: usize,
) -> Graph {
    let (ls_shape, rs_shape) = match layout {
        ScaleLayout::PerTensor => (Shape::new(&[1], DType::F32), Shape::new(&[1], DType::F32)),
        _ => {
            let nb = k.div_ceil(layout.block() as usize);
            (
                Shape::new(&[m, nb], DType::U8),
                Shape::new(&[n, nb], DType::U8),
            )
        }
    };
    let mut g = Graph::new("vk_scaled_mm");
    let lhs_in = g.input("lhs", Shape::new(&[m, k], DType::F32));
    let rhs_in = g.input("rhs", Shape::new(&[n, k], DType::F32));
    let ls = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_in],
        ls_shape,
    );
    let lq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![lhs_in, ls],
        Shape::new(&[m, k], DType::U8),
    );
    let rs = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_in],
        rs_shape,
    );
    let rq = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![rhs_in, rs],
        Shape::new(&[n, k], DType::U8),
    );
    let y = g.add_node(
        Op::ScaledMatMul {
            lhs_format: fmt,
            rhs_format: fmt,
            scale_layout: layout,
            has_bias: false,
        },
        vec![lq, rq, ls, rs],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn f32_matmul_tn(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// f4e3m0 grid-aligned inputs (amax 2 → per-tensor scale 0.125, exact power of
/// two) reconstruct bit-exactly, so the Vulkan host-fallback GEMM must equal a
/// plain f32 matmul. This exercises the full chain (ScaledQuantScale →
/// ScaledQuantize [U8 output] → ScaledMatMul) through the mapped arena.
#[test]
fn vulkan_scaled_f4e3m0_grid_is_exact() {
    if !rlx_vulkan::is_available() {
        eprintln!("skip: no Vulkan device");
        return;
    }
    eprintln!("[rlx-vulkan] f4e3m0 on {:?}", rlx_vulkan::device_name());
    let fmt = ScaledFormat::custom(3, 0);
    assert_eq!(fmt.to_string(), "f4e3m0");
    let (m, k, n) = (4usize, 32usize, 6usize);
    let grid = [2.0f32, -1.0, 0.5, -0.25, 1.0, -2.0, 0.25, -0.5];
    let lhs: Vec<f32> = (0..m * k).map(|i| grid[i % grid.len()]).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| grid[(i * 3 + 1) % grid.len()]).collect();

    let mut exe =
        VulkanExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::PerTensor, m, k, n));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    let reference = f32_matmul_tn(&lhs, &rhs, m, k, n);
    assert_eq!(out.len(), m * n);
    let max_abs = out
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[rlx-vulkan] f4e3m0 grid: max_abs_vs_f32={max_abs:.3e}");
    assert!(max_abs <= 5e-3, "f4e3m0 grid GEMM max_abs {max_abs}");
}

/// f4e3m0 on smooth data with block-MX scaling tracks the f32 matmul.
#[test]
fn vulkan_scaled_f4e3m0_tracks_f32() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let fmt = ScaledFormat::custom(3, 0);
    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();
    let mut exe = VulkanExecutable::compile(build_scaled_mm_graph(fmt, ScaleLayout::mx(), m, k, n));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    assert!(out.iter().all(|v| v.is_finite()), "f4e3m0 non-finite");
    let cos = cosine(&out, &f32_matmul_tn(&lhs, &rhs, m, k, n));
    eprintln!("[rlx-vulkan] f4e3m0 mx-block: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.7, "f4e3m0 mx-block cosine {cos} < 0.7");
}

/// Named E4M3 through the same host-fallback path is high-fidelity.
#[test]
fn vulkan_scaled_named_e4m3() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();
    let mut exe = VulkanExecutable::compile(build_scaled_mm_graph(
        ScaledFormat::F8E4M3,
        ScaleLayout::mx(),
        m,
        k,
        n,
    ));
    let out = exe
        .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
        .remove(0);
    let cos = cosine(&out, &f32_matmul_tn(&lhs, &rhs, m, k, n));
    eprintln!("[rlx-vulkan] e4m3 mx-block: cosine_vs_f32={cos:.5}");
    assert!(cos >= 0.99, "e4m3 mx-block cosine {cos} < 0.99");
}
