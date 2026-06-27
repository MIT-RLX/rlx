// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Native low-precision `Op::ScaledMatMul` on Metal. Apple GPUs have no FP8
//! matrix units, so Metal runs the decode-and-accumulate host fallback over
//! unified memory — the SAME rlx-cpu oracle the CPU backend uses. So Metal must
//! match CPU bit-for-bit, and both must track a plain f32 matmul (cosine).
//! Layout is TN: lhs [m,k], rhs [n,k], out = lhs·rhsᵀ.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, ScaleLayout, ScaledFormat, Shape};
use rlx_runtime::{Device, Session};

fn run_case(fmt: ScaledFormat, layout: ScaleLayout, m: usize, k: usize, n: usize, cos_thresh: f32) {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();

    // Scale tensor shapes per layout: per-tensor → [1] f32; block → [rows, k/block] u8.
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

    let mut g = Graph::new("scaled_mm");
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

    let run = |device: Device| -> Vec<f32> {
        Session::new(device)
            .compile(g.clone())
            .run(&[("lhs", lhs.as_slice()), ("rhs", rhs.as_slice())])
            .remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), m * n);

    // Metal host fallback uses the same oracle as CPU → bit-identical.
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-6, "{fmt} Metal vs CPU max_abs={max_abs}");

    // Both track the f32 matmul (cosine).
    let mut reference = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[j * k + p];
            }
            reference[i * n + j] = acc;
        }
    }
    let dot: f32 = metal.iter().zip(&reference).map(|(a, b)| a * b).sum();
    let na = metal.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    eprintln!("scaled_matmul {fmt}: metal-vs-cpu max_abs={max_abs:.2e} cosine-vs-f32={cos:.5}");
    assert!(cos >= cos_thresh, "{fmt} cosine {cos} < {cos_thresh}");
}

#[test]
fn metal_scaled_matmul_e4m3_matches_cpu_and_f32() {
    run_case(
        ScaledFormat::F8E4M3,
        ScaleLayout::PerTensor,
        4,
        64,
        8,
        0.999,
    );
}

#[test]
fn metal_scaled_matmul_e5m2_matches_cpu_and_f32() {
    run_case(ScaledFormat::F8E5M2, ScaleLayout::PerTensor, 4, 64, 8, 0.99);
}

#[test]
fn metal_scaled_matmul_gemv() {
    // m = 1 decode shape.
    run_case(
        ScaledFormat::F8E4M3,
        ScaleLayout::PerTensor,
        1,
        32,
        12,
        0.999,
    );
}

#[test]
fn metal_scaled_matmul_mxfp8_block() {
    // Block-scaled MXFP8 (E8M0 scales) — finer than per-tensor.
    run_case(ScaledFormat::F8E4M3, ScaleLayout::mx(), 4, 64, 8, 0.999);
}

#[test]
fn metal_scaled_matmul_nvfp4_block() {
    // FP4 with per-16 E4M3 block scales.
    run_case(ScaledFormat::F4E2M1, ScaleLayout::nvfp4(), 4, 64, 8, 0.95);
}
