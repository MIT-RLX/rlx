// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! MPSGraph erf-based `Activation::Gelu` vs CPU `scalar_gelu`.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape, op::Activation};
use rlx_metal::mps_graph::mps_graph_supported;
use rlx_metal::mps_graph_lower::try_lower;
use rlx_runtime::{Device, Session};

#[inline]
fn scalar_gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + scalar_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
fn scalar_erf(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - y * (-xa * xa).exp())
}

#[test]
fn mps_gelu_erf_matches_cpu_scalar() {
    if !mps_graph_supported() {
        eprintln!("skip: MPSGraph unavailable");
        return;
    }

    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.5).collect();

    let mut g = Graph::new("gelu");
    let x_in = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.activation(Activation::Gelu, x_in, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![y]);

    assert!(try_lower(&g).is_some(), "Gelu must lower via MPSGraph");

    let mut compiled = Session::new(Device::Metal).compile(g);
    let metal = compiled.run(&[("x", &x)]).remove(0);
    let cpu: Vec<f32> = x.iter().map(|&v| scalar_gelu(v)).collect();

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("gelu erf parity max_abs={max_abs:.6}");
    assert!(max_abs < 5e-4, "MPSGraph gelu max_abs={max_abs}");
}
