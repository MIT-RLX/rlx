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

//! `Op::RmsNorm` with an explicit (zero) beta on Metal vs CPU.
//!
//! The Voxtral LM prefill feeds every RMSNorm a `voxtral.zero_beta.hidden` beta
//! input (`zero_beta_named`), so the 3-input form `[x, gamma, beta]` is exercised
//! — unlike plain LLMs that use the 2-input form. Suspect for the Metal garbage.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_rmsnorm(b: usize, s: usize, h: usize, eps: f32) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("rmsnorm_beta");
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let gamma = g.input("gamma", Shape::new(&[h], f));
    let beta = g.input("beta", Shape::new(&[h], f));
    let y = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps },
        vec![x, gamma, beta],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_rmsnorm_beta_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, s, h, eps) = (1, 8, 3072, 1e-5);
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i as f32) * 0.013).sin() * 2.0)
        .collect();
    let gamma: Vec<f32> = (0..h)
        .map(|i| 1.0 + ((i as f32) * 0.001).cos() * 0.1)
        .collect();
    let beta = vec![0.0f32; h]; // zero beta, as Voxtral supplies

    let g = build_rmsnorm(b, s, h, eps);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta)])
        .remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta)])
        .remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!("rmsnorm+beta: max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}");
    assert!(max_abs < 1e-4, "rmsnorm+beta max_abs={max_abs}");
}
