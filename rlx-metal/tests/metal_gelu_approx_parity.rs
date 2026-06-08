// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! MPSGraph `gelu_approx` vs CPU `scalar_gelu_approx`.

#![cfg(target_os = "macos")]

#[inline]
fn scalar_gelu_approx(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    const A: f32 = 0.044_715;
    0.5 * x * (1.0 + (C * (x + A * x * x * x)).tanh())
}
use rlx_ir::{DType, Graph, Shape, op::Activation};
use rlx_metal::mps_graph::mps_graph_supported;
use rlx_metal::mps_graph_lower::try_lower;
use rlx_runtime::{Device, Session};

#[test]
fn mps_gelu_approx_matches_cpu_scalar() {
    if !mps_graph_supported() {
        eprintln!("skip: MPSGraph unavailable");
        return;
    }

    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.5).collect();

    let mut g = Graph::new("gelu_approx");
    let x_in = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.activation(Activation::GeluApprox, x_in, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![y]);

    assert!(
        try_lower(&g).is_some(),
        "GeluApprox must lower via gelu_approx"
    );

    let mut compiled = Session::new(Device::Metal).compile(g);
    let metal = compiled.run(&[("x", &x)]).remove(0);

    let cpu: Vec<f32> = x.iter().map(|&v| scalar_gelu_approx(v)).collect();

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("gelu_approx parity max_abs={max_abs:.6}");
    assert!(max_abs < 5e-4, "MPSGraph gelu_approx max_abs={max_abs}");
}
