// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verifies `Op::MatMul` with an **F32 left-hand** `[M,K]`, a **BF16
//! right-hand weight** `[K,N]`, and an **F32 output** `[M,N]` runs on the
//! portable wgpu backend and matches (a) the CPU backend and (b) a host
//! f32 reference computed from the bf16-rounded weights.
//!
//! The bf16 weight is uploaded via `set_param_typed(name, bytes, BF16)`.
//! The wgpu backend now keeps that weight **PACKED** (2 bytes/elem, two
//! bf16 per u32) in a side buffer and unpacks it **in-shader**
//! (`matmul_bf16w`, `bitcast<f32>(bits << 16)`) instead of widening it to
//! f32 in the arena — so the matmul reads HALF the weight bytes while
//! staying bit-exact to an f32 matmul over bf16-rounded weights.
//!
//! Each case asserts the process-wide packed-dispatch counter advanced,
//! proving the packed path (not the old f32-widen path) was taken.

#![cfg(all(feature = "gpu", target_os = "macos"))]

use half::bf16;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn build_matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("bf16_weight_matmul");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    // BF16-resident weight — legal rhs for a MatMul whose lhs (and thus
    // output, per matmul_shape) is F32.
    let w = g.param("w", Shape::new(&[k, n], DType::BF16));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn run_case(m: usize, k: usize, n: usize, tol: f32) {
    // Deterministic A (f32) and W (f32 → bf16).
    let a: Vec<f32> = (0..(m * k))
        .map(|i| ((i % 17) as f32) * 0.1 - 0.7)
        .collect();
    let w_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i % 13) as f32) * 0.05 - 0.3)
        .collect();
    let w_bf16: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_back: Vec<f32> = w_bf16.iter().map(|h| h.to_f32()).collect();
    let w_bytes: Vec<u8> = w_bf16.iter().flat_map(|h| h.to_le_bytes()).collect();

    // Host reference using bf16-rounded weights (what the packed buffer holds).
    let mut c_ref = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0.0f32;
            for ki in 0..k {
                s += a[mi * k + ki] * w_back[ki * n + ni];
            }
            c_ref[mi * n + ni] = s;
        }
    }

    // Run on wgpu (Device::Gpu) with the bf16 weight uploaded as typed bytes.
    // Snapshot the packed-dispatch counter around the run to prove the packed
    // matmul_bf16w kernel — not the old f32-widen path — actually executed.
    let packed_before = rlx_wgpu::bf16_packed_dispatch_count();
    let gpu_out = {
        let session = Session::new(Device::Gpu);
        let mut compiled =
            session.compile_with(build_matmul_graph(m, k, n), &CompileOptions::default());
        compiled.set_param_typed("w", &w_bytes, DType::BF16);
        compiled.run(&[("x", &a)]).pop().unwrap()
    };
    let packed_after = rlx_wgpu::bf16_packed_dispatch_count();

    // Run the identical graph on CPU as an apples-to-apples parity anchor.
    let cpu_out = {
        let session = Session::new(Device::Cpu);
        let mut compiled =
            session.compile_with(build_matmul_graph(m, k, n), &CompileOptions::default());
        compiled.set_param_typed("w", &w_bytes, DType::BF16);
        compiled.run(&[("x", &a)]).pop().unwrap()
    };

    assert_eq!(gpu_out.len(), m * n, "wgpu output element count");
    assert!(
        packed_after > packed_before,
        "PACKED bf16 matmul path was NOT taken (dispatch counter did not \
         advance: {packed_before} → {packed_after}) for [{m}x{k}]@[{k}x{n}]"
    );

    let max_gpu_vs_ref = c_ref
        .iter()
        .zip(&gpu_out)
        .map(|(r, g)| (r - g).abs())
        .fold(0f32, f32::max);
    let max_gpu_vs_cpu = cpu_out
        .iter()
        .zip(&gpu_out)
        .map(|(c, g)| (c - g).abs())
        .fold(0f32, f32::max);

    eprintln!(
        "bf16-PACKED matmul [{m}x{k}]@[{k}x{n}]: max|gpu-ref|={max_gpu_vs_ref:.3e}, \
         max|gpu-cpu|={max_gpu_vs_cpu:.3e}, tol={tol:.1e}, \
         packed_dispatches +{}",
        packed_after - packed_before
    );

    assert!(
        max_gpu_vs_ref < tol,
        "wgpu bf16-packed matmul drifted from host ref: max|Δ|={max_gpu_vs_ref}"
    );
    assert!(
        max_gpu_vs_cpu < tol,
        "wgpu bf16-packed matmul disagrees with CPU backend: max|Δ|={max_gpu_vs_cpu}"
    );
}

#[test]
fn wgpu_bf16_weight_matmul_unaligned() {
    // Shapes unaligned to the coop-matrix tiling — the packed tiled kernel
    // bounds-checks, so no alignment requirement.
    run_case(8, 16, 12, 1e-3);
}

#[test]
fn wgpu_bf16_weight_matmul_aligned_shapes() {
    // Coop-eligible shapes (m%32, k%8, n%32). The packed path is bf16-exact
    // (never routes through the lossy f16 shadow), so a tight tol holds.
    run_case(32, 64, 64, 1e-3);
}

#[test]
fn wgpu_bf16_weight_matmul_m1_decode() {
    // Decode-shaped GEMV: m=1 × unaligned N. Exercises the small/skinny
    // dispatch grid path for the packed weight.
    run_case(1, 128, 200, 1e-3);
}
