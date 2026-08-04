// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! wgpu native FusedSwiGLU (`fused_swiglu.wgsl`) parity vs CPU. wgpu keeps the
//! op fused (one kernel) instead of the Narrow+Silu+Mul decompose; this pins it
//! against the CPU result for both concat orders.

#![cfg(feature = "gpu")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

fn swiglu_graph(rows: usize, n_half: usize, gate_first: bool) -> Graph {
    let mut g = Graph::new("swiglu");
    let x = g.input("x", Shape::new(&[rows, 2 * n_half], F));
    let y = g.add_node(
        Op::FusedSwiGLU {
            cast_to: None,
            gate_first,
        },
        vec![x],
        Shape::new(&[rows, n_half], F),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn wgpu_fused_swiglu_matches_cpu() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_fused_swiglu (wgpu unavailable)");
        return;
    }
    // n_half > 64 → more than one dispatch workgroup per row's worth of output.
    let (rows, n_half) = (3usize, 200usize);
    let x: Vec<f32> = (0..rows * 2 * n_half)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.2)
        .collect();
    for gate_first in [false, true] {
        let cpu = Session::new(Device::Cpu)
            .compile(swiglu_graph(rows, n_half, gate_first))
            .run(&[("x", &x)])
            .remove(0);
        let gpu = Session::new(Device::Gpu)
            .compile(swiglu_graph(rows, n_half, gate_first))
            .run(&[("x", &x)])
            .remove(0);
        let max = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "wgpu SwiGLU gate_first={gate_first} max_abs={max:.3e}"
        );
    }
}
