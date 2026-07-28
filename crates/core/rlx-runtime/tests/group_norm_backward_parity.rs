// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity for the `GroupNormBackward{Input,Gamma,Beta}` ops — full kernels
//! already existed on CPU (`training_bwd`) and MLX (`lower.rs`) but were never
//! claimed in `supported_ops`, so GroupNorm training was silently rejected.
//! CPU is the reference.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

fn data(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + seed) % 19) as f32 - 9.0) * 0.07)
        .collect()
}

/// Builds dx, dgamma, dbeta for one GroupNorm config; returns the three flat
/// outputs concatenated for an easy backend-vs-backend compare.
fn run_backward(device: Device, n: usize, c: usize, h: usize, w: usize, groups: usize) -> Vec<f32> {
    let mut g = Graph::new("gn_bwd");
    let x = g.input("x", nchw(n, c, h, w));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let dy = g.input("dy", nchw(n, c, h, w));
    let dx = g.group_norm_backward_input(x, gamma, beta, dy, groups, 1e-5);
    let dgamma = g.group_norm_backward_gamma(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    let dbeta = g.group_norm_backward_beta(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    g.set_outputs(vec![dx, dgamma, dbeta]);

    let xv = data(n * c * h * w, 1);
    let gv = data(c, 3);
    let bv = data(c, 5);
    let dyv = data(n * c * h * w, 7);
    let mut exe = Session::new(device).compile(g);
    let outs = exe.run(&[
        ("x", xv.as_slice()),
        ("gamma", gv.as_slice()),
        ("beta", bv.as_slice()),
        ("dy", dyv.as_slice()),
    ]);
    outs.into_iter().flatten().collect()
}

#[allow(dead_code)] // used only by the backend-gated parity tests below
fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max:e} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", a.len());
}

fn cfgs() -> Vec<(&'static str, usize, usize, usize, usize, usize)> {
    vec![
        ("tiny", 1, 8, 4, 4, 2),
        ("batch2", 2, 32, 8, 8, 8),
        ("vision", 1, 64, 16, 16, 32),
    ]
}

#[test]
fn group_norm_backward_cpu_runs() {
    // Was rejected at legalization before being claimed; now must compile+run.
    for (name, n, c, h, w, gr) in cfgs() {
        let out = run_backward(Device::Cpu, n, c, h, w, gr);
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn group_norm_backward_mlx_matches_cpu() {
    for (name, n, c, h, w, gr) in cfgs() {
        assert_close(
            &format!("gn-bwd mlx {name}"),
            &run_backward(Device::Mlx, n, c, h, w, gr),
            &run_backward(Device::Cpu, n, c, h, w, gr),
        );
    }
}
