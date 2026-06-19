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

//! `Op::Rope` execution on Metal vs CPU, packed `[B, S, H*D]`, head_dim 128
//! (Voxtral LM dims). Other tests cover attention/rmsnorm/swiglu but RoPE-apply
//! was only checked by formula inspection — this actually runs the kernel.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_rope(b: usize, s: usize, h: usize, d: usize) -> Graph {
    let f = DType::F32;
    let w = h * d;
    let half = d / 2;
    let mut g = Graph::new("rope");
    let x = g.input("x", Shape::new(&[b, s, w], f));
    // cos/sin tables: one row of `head_dim/2` per sequence position.
    let cos = g.input("cos", Shape::new(&[s, half], f));
    let sin = g.input("sin", Shape::new(&[s, half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim: d,
            n_rot: d,
        },
        vec![x, cos, sin],
        Shape::new(&[b, s, w], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_rope_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, s, h, d) = (1, 64, 32, 128);
    let half = d / 2;
    let x: Vec<f32> = (0..b * s * h * d)
        .map(|i| ((i as f32) * 0.0007).sin())
        .collect();
    // Realistic rotary tables: theta_j = pos * base^(-2j/d).
    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for p in 0..s {
        for j in 0..half {
            let freq = 1.0f32 / (100_000_000.0f32).powf((2 * j) as f32 / d as f32);
            let ang = p as f32 * freq;
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }

    let g = build_rope(b, s, h, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!("rope (hd=128): max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}");
    assert!(max_abs < 1e-4, "rope max_abs={max_abs}");
}
