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

//! FULL SwiGLU MLP (gate/up matmul → silu → mul → down matmul) on Metal vs CPU,
//! Voxtral widths (H=3072, I=8192). Matches `SwiGluStage::emit`: gate and up both
//! read the same input, gate+silu fuse into an in-place `FusedMatMulBiasAct`.
//! The Voxtral MLP block diverges on Metal while the attention block matches, and
//! plain silu*mul passes — so the suspect is this matmul+silu fusion / aliasing.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_full_swiglu(rows: usize, h: usize, inter: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("full_swiglu");
    let x = g.input("x", Shape::new(&[rows, h], f));
    let wg = g.input("wg", Shape::new(&[h, inter], f));
    let wu = g.input("wu", Shape::new(&[h, inter], f));
    let wd = g.input("wd", Shape::new(&[inter, h], f));
    let gate = g.add_node(
        rlx_ir::Op::MatMul,
        vec![x, wg],
        Shape::new(&[rows, inter], f),
    );
    let up = g.add_node(
        rlx_ir::Op::MatMul,
        vec![x, wu],
        Shape::new(&[rows, inter], f),
    );
    let gate_act = g.add_node(
        rlx_ir::Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[rows, inter], f),
    );
    let prod = g.add_node(
        rlx_ir::Op::Binary(BinaryOp::Mul),
        vec![gate_act, up],
        Shape::new(&[rows, inter], f),
    );
    let down = g.add_node(
        rlx_ir::Op::MatMul,
        vec![prod, wd],
        Shape::new(&[rows, h], f),
    );
    g.set_outputs(vec![down]);
    g
}

#[test]
fn metal_full_swiglu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (rows, h, inter) = (64, 3072, 8192);
    let mk = |seed: f32, n: usize| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * seed).sin() * 0.05).collect()
    };
    let x = mk(0.0007, rows * h);
    let wg = mk(0.0003, h * inter);
    let wu = mk(0.0005, h * inter);
    let wd = mk(0.0002, inter * h);

    let g = build_full_swiglu(rows, h, inter);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m
        .run(&[("x", &x), ("wg", &wg), ("wu", &wu), ("wd", &wd)])
        .remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c
        .run(&[("x", &x), ("wg", &wg), ("wu", &wu), ("wd", &wd)])
        .remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!("full swiglu: max_abs={max_abs:.6} cpu_sum={cpu_sum:.5} metal_sum={metal_sum:.5}");
    assert!(max_abs < 1e-4, "full swiglu max_abs={max_abs}");
}
