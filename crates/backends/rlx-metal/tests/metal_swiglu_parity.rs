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

//! SwiGLU core `silu(gate) * up` on Metal vs CPU at the Voxtral intermediate
//! width (8192). The LLM MLP uses this; the Whisper encoder uses GELU. One of
//! the few per-layer ops without an existing Metal parity guard.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_swiglu(b: usize, s: usize, inter: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("swiglu");
    let gate = g.input("gate", Shape::new(&[b, s, inter], f));
    let up = g.input("up", Shape::new(&[b, s, inter], f));
    let act = g.add_node(
        rlx_ir::Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[b, s, inter], f),
    );
    let y = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Mul),
        vec![act, up],
        Shape::new(&[b, s, inter], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_swiglu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, s, inter) = (1, 16, 8192);
    let n = b * s * inter;
    let gate: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0009).sin() * 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0013).cos() * 2.0).collect();

    let g = build_swiglu(b, s, inter);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("gate", &gate), ("up", &up)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("gate", &gate), ("up", &up)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("swiglu silu*up: max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "swiglu max_abs={max_abs}");
}
