// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu `ArgMax` / `ArgMin` / `Reverse` run as GPU kernels and match rlx-cpu
//! bit for bit.
//!
//! They used to be host steps, which make a run wait on the GPU mid-graph: the
//! browser cannot, so any WebGPU model with one fell back to the CPU there.
//! Argreduce is pinned on ties (first index wins), a NaN in the first slot
//! (keeps index 0), NaNs later in the row (never win) and infinities, along
//! every axis; Reverse on several axis subsets and ranks.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn run(g: &Graph, x: &[f32], d: Device) -> Vec<f32> {
    Session::new(d)
        .compile(g.clone())
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn host_steps(g: &Graph) -> Option<String> {
    Session::new(Device::Gpu)
        .compile(g.clone())
        .host_execution_summary()
}

#[test]
fn argreduce_and_reverse_match_cpu_on_the_gpu() {
    let Some(_gpu) = common::serialize_gpu() else {
        return;
    };
    let dims = [5usize, 7, 6];
    let n: usize = dims.iter().product();
    // Few distinct values, so every axis has exact ties: along axis 2 the
    // value cycles with `i % 3`, along axes 0 and 1 (strides 42 and 6, both
    // multiples of 3) it changes only through `(i / 6) % 2` and `i / 42`.
    // (An earlier `(i·37) % 11` had none on any axis, and a last-wins kernel
    // passed.)
    let mut x: Vec<f32> = (0..n)
        .map(|i| ((i % 3) + (i / 6) % 2 + (i / 84)) as f32)
        .collect();
    x[0] = f32::NAN; // first slot of axis 0/1/2 rows starting at 0
    x[13] = f32::NAN;
    x[40] = f32::INFINITY;
    x[41] = f32::NEG_INFINITY;
    let mut bad = Vec::new();
    for axis in 0..3 {
        for is_max in [true, false] {
            let mut g = Graph::new("argreduce");
            let i = g.input("x", Shape::new(&dims, DType::F32));
            let mut out = dims.to_vec();
            out[axis] = 1;
            let op = if is_max {
                Op::ArgMax {
                    axis,
                    keep_dim: true,
                }
            } else {
                Op::ArgMin {
                    axis,
                    keep_dim: true,
                }
            };
            let y = g.add_node(op, vec![i], Shape::new(&out, DType::F32));
            g.set_outputs(vec![y]);
            assert_eq!(
                host_steps(&g),
                None,
                "argreduce axis {axis} still has host steps"
            );
            let (a, b) = (run(&g, &x, Device::Gpu), run(&g, &x, Device::Cpu));
            if a.iter().zip(&b).any(|(p, q)| p.to_bits() != q.to_bits()) {
                bad.push(format!(
                    "arg{} axis {axis}: gpu {a:?} cpu {b:?}",
                    if is_max { "max" } else { "min" }
                ));
            }
        }
    }
    for (shape, axes) in [
        (vec![5usize, 7, 6], vec![0usize]),
        (vec![5, 7, 6], vec![1, 2]),
        (vec![5, 7, 6], vec![0, 1, 2]),
        (vec![210], vec![0]),
        (vec![2, 3, 5, 7], vec![1, 3]),
    ] {
        let mut g = Graph::new("reverse");
        let i = g.input("x", Shape::new(&shape, DType::F32));
        let y = g.add_node(
            Op::Reverse { axes: axes.clone() },
            vec![i],
            Shape::new(&shape, DType::F32),
        );
        g.set_outputs(vec![y]);
        assert_eq!(
            host_steps(&g),
            None,
            "reverse {shape:?}/{axes:?} still has host steps"
        );
        let (a, b) = (run(&g, &x, Device::Gpu), run(&g, &x, Device::Cpu));
        if a.iter().zip(&b).any(|(p, q)| p.to_bits() != q.to_bits()) {
            bad.push(format!("reverse {shape:?} axes {axes:?}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
