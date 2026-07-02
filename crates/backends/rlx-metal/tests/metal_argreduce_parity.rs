// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `Op::ArgMax` / `Op::ArgMin` on Metal vs CPU. These previously ran as a host
//! fallback over unified memory; they now have a native MSL kernel
//! (`argreduce`). The indices must match the CPU reference exactly.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn build(outer: usize, reduced: usize, inner: usize, is_max: bool) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("argreduce");
    let x = g.input("x", Shape::new(&[outer, reduced, inner], f));
    let op = if is_max {
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        }
    } else {
        Op::ArgMin {
            axis: 1,
            keep_dim: false,
        }
    };
    let y = g.add_node(op, vec![x], Shape::new(&[outer, inner], f));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_argreduce_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (outer, reduced, inner) = (3usize, 7usize, 4usize);
    // Pseudo-random but deterministic; spread so max/min per column is unique.
    let x: Vec<f32> = (0..outer * reduced * inner)
        .map(|i| (((i * 131 + 7) % 97) as f32) * 0.011 - 0.5)
        .collect();

    for is_max in [true, false] {
        let g = build(outer, reduced, inner, is_max);
        let mut m = Session::new(Device::Metal).compile(g.clone());
        let metal = m.run(&[("x", &x)]).remove(0);
        let mut c = Session::new(Device::Cpu).compile(g);
        let cpu = c.run(&[("x", &x)]).remove(0);

        assert_eq!(metal.len(), outer * inner);
        for (j, (a, b)) in metal.iter().zip(&cpu).enumerate() {
            assert_eq!(
                *a as i64, *b as i64,
                "argreduce[{j}] mismatch (is_max={is_max}): metal {a} vs cpu {b}"
            );
        }
    }
}

#[test]
fn metal_argreduce_first_best_tiebreak() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // Ties must resolve to the lowest index on both backends.
    let (outer, reduced, inner) = (1usize, 5usize, 2usize);
    // column 0: all 1.0 (tie → idx 0); column 1: ramps then peaks at idx 3.
    let x = vec![
        1.0, 0.1, // r0
        1.0, 0.2, // r1
        1.0, 0.3, // r2
        1.0, 0.9, // r3
        1.0, 0.4, // r4
    ];
    let g = build(outer, reduced, inner, true);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x)]).remove(0);
    assert_eq!(metal[0] as i64, 0, "tie → first index");
    assert_eq!(metal[1] as i64, 3, "peak at idx 3");
    assert_eq!(metal, cpu);
}
