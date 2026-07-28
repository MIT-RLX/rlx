// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Expand` cross-backend parity at various spatial sizes — the detection
//! U-Net's bias broadcast (`[1,C,1,1] → [N,C,H,W]`) diverged on Metal only at
//! full resolution, implicating a size-dependent Expand kernel bug.

#![allow(dead_code)]

use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn build(c: usize, h: usize, w: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("expand");
    let x = g.input("x", Shape::new(&[1, c, 1, 1], f));
    let y = g.add_node(
        Op::Expand {
            target_shape: vec![1, c as i64, h as i64, w as i64],
        },
        vec![x],
        Shape::new(&[1, c, h, w], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run(c: usize, h: usize, w: usize, device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..c).map(|i| (i as f32 - c as f32 / 2.0) * 0.1).collect();
    let mut comp = Session::new(device).compile(build(c, h, w));
    comp.run(&[("x", x.as_slice())]).pop().unwrap()
}

fn check(name: &str, c: usize, h: usize, w: usize) {
    let cpu = run(c, h, w, Device::Cpu);
    let dev = run(c, h, w, Device::Metal);
    let maxd = cpu
        .iter()
        .zip(&dev)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[expand] {name} c={c} {h}x{w}: max abs diff {maxd:.5}");
    assert!(maxd <= 1e-5, "{name}: Expand metal vs cpu diff {maxd}");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn expand_metal_matches_cpu() {
    check("small", 8, 6, 6);
    check("med", 8, 64, 64);
    check("hires", 8, 800, 600);
    check("hires-wide", 32, 800, 600);
}
