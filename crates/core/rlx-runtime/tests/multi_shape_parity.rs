// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multiple-shape support across ALL backends.
//!
//! Two mechanisms, both must hold on every backend:
//!  * **recompile-per-shape** (universal): the same logical model compiled at
//!    several input shapes runs and matches CPU. This is what lets e.g. an OCR
//!    detection U-Net run at a different aspect ratio.
//!  * **dynamic dims** (`Dim::Dynamic`): one graph, several input shapes, the
//!    backend specializes per run. wgpu/MLX do this internally; CPU/Metal must at
//!    least not crash (they recompile/replan under the hood).
//!
//! The model is conv2d → relu → conv_transpose2d (the OCR decoder shape), so the
//! per-shape arena plan and the shape-dependent conv/upsample kernels are covered.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

// ----------------------------- recompile per shape -----------------------------

/// conv2d(3→8, k3s1p1) → relu → conv_transpose2d(8→4, k2s2) upsample.
fn build(n: usize, h: usize, w: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("ms");
    let x = g.input("x", Shape::new(&[n, 3, h, w], f));
    let cw = g.input("cw", Shape::new(&[8, 3, 3, 3], f));
    let y = g.conv2d(x, cw, [3, 3], [1, 1], [1, 1], [1, 1], 1); // [n,8,h,w]
    let y = g.relu(y);
    let tw = g.input("tw", Shape::new(&[8, 4, 2, 2], f)); // ConvT [in=8,out=4,k2]
    let up = g.conv_transpose2d(y, tw, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1); // [n,4,2h,2w]
    g.set_outputs(vec![up]);
    g
}

fn weights() -> (Vec<f32>, Vec<f32>) {
    let cw: Vec<f32> = (0..8 * 3 * 3 * 3)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.03)
        .collect();
    let tw: Vec<f32> = (0..8 * 4 * 2 * 2)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.04)
        .collect();
    (cw, tw)
}

fn xdata(n: usize, h: usize, w: usize) -> Vec<f32> {
    (0..n * 3 * h * w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect()
}

fn run_static(device: Device, n: usize, h: usize, w: usize) -> Vec<f32> {
    let (cw, tw) = weights();
    let x = xdata(n, h, w);
    let mut c = Session::new(device).compile(build(n, h, w));
    c.run(&[
        ("x", x.as_slice()),
        ("cw", cw.as_slice()),
        ("tw", tw.as_slice()),
    ])
    .pop()
    .unwrap()
}

/// Shapes a detection-style model might see: square, wide, tall, batched.
fn shapes() -> Vec<(&'static str, usize, usize, usize)> {
    vec![
        ("16x16", 1, 16, 16),
        ("24x32-wide", 1, 24, 32),
        ("40x28-tall", 1, 40, 28),
        ("batch3-20x20", 3, 20, 20),
    ]
}

fn assert_close(what: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{what}: length {} vs {}",
        actual.len(),
        reference.len()
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

/// One compiled graph per shape on CPU must just run and stay finite.
#[test]
fn multi_shape_cpu_runs() {
    for (name, n, h, w) in shapes() {
        let out = run_static(Device::Cpu, n, h, w);
        assert_eq!(
            out.len(),
            n * 4 * (2 * h) * (2 * w),
            "cpu {name}: wrong output len"
        );
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
        eprintln!("cpu {name}: {} elems ok", out.len());
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn multi_shape_metal_matches_cpu() {
    for (name, n, h, w) in shapes() {
        assert_close(
            &format!("metal {name}"),
            &run_static(Device::Metal, n, h, w),
            &run_static(Device::Cpu, n, h, w),
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn multi_shape_mlx_matches_cpu() {
    for (name, n, h, w) in shapes() {
        assert_close(
            &format!("mlx {name}"),
            &run_static(Device::Mlx, n, h, w),
            &run_static(Device::Cpu, n, h, w),
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn multi_shape_wgpu_matches_cpu() {
    for (name, n, h, w) in shapes() {
        assert_close(
            &format!("wgpu {name}"),
            &run_static(Device::Gpu, n, h, w),
            &run_static(Device::Cpu, n, h, w),
        );
    }
}

// ------------------------------- dynamic batch ---------------------------------

/// Same model but with a DYNAMIC batch dim — one compiled graph, run at several
/// batch sizes. The backend must infer N from the input length and specialize.
fn build_dyn() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("ms_dyn");
    let x = g.input(
        "x",
        Shape::from_dims(
            &[
                Dim::Dynamic(0),
                Dim::Static(3),
                Dim::Static(8),
                Dim::Static(8),
            ],
            f,
        ),
    );
    let cw = g.input("cw", Shape::new(&[8, 3, 3, 3], f));
    let y = g.conv2d(x, cw, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let y = g.relu(y);
    let tw = g.input("tw", Shape::new(&[8, 4, 2, 2], f));
    let up = g.conv_transpose2d(y, tw, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1);
    g.set_outputs(vec![up]);
    g
}

/// Run a single compiled (dynamic) graph at every batch in `batches`, comparing
/// each to a fresh static compile at that batch.
fn dynamic_matches_static(device: Device, label: &str) {
    let (cw, tw) = weights();
    let mut c = Session::new(device).compile(build_dyn());
    for n in [1usize, 2, 4] {
        let x = xdata(n, 8, 8);
        let out = c
            .run(&[
                ("x", x.as_slice()),
                ("cw", cw.as_slice()),
                ("tw", tw.as_slice()),
            ])
            .pop()
            .unwrap();
        let reference = run_static(Device::Cpu, n, 8, 8);
        assert_close(&format!("{label} dyn-batch={n}"), &out, &reference);
    }
}

#[test]
fn dynamic_batch_cpu() {
    dynamic_matches_static(Device::Cpu, "cpu");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dynamic_batch_metal() {
    dynamic_matches_static(Device::Metal, "metal");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn dynamic_batch_mlx() {
    dynamic_matches_static(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn dynamic_batch_wgpu() {
    dynamic_matches_static(Device::Gpu, "wgpu");
}
