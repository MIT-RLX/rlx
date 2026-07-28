// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::FusedSwiGLU` (canonical `gate_first=false`: [up || gate]) on Metal
//! MPSGraph vs CPU. Regression guard for the MPSGraph half-swap bug that
//! garbled every LLM SwiGLU MLP on Metal — the lowering computed silu(up)*gate
//! instead of up*silu(gate). Forced through MPSGraph (RLX_MPSGRAPH_FORCE).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_fused_swiglu_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // Force MPSGraph so this exercises the mps_graph_lower path, not thunks.
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");

    let f = DType::F32;
    let (rows, n) = (380usize, 8192usize); // last axis = 2n = [up || gate]
    let mut g = Graph::new("fused_swiglu");
    let x = g.input("x", Shape::new(&[1, rows, 2 * n], f));
    let y = g.add_node(
        rlx_ir::Op::FusedSwiGLU {
            cast_to: None,
            gate_first: false,
        },
        vec![x],
        Shape::new(&[1, rows, n], f),
    );
    g.set_outputs(vec![y]);

    let xv: Vec<f32> = (0..rows * 2 * n)
        .map(|i| ((i as f32) * 0.0007).sin() * 2.0)
        .collect();
    let inputs: [(&str, &[f32]); 1] = [("x", &xv)];

    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&inputs).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&inputs).remove(0);

    rlx_ir::env::unset("RLX_MPSGRAPH_FORCE");

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("fused swiglu (gate_first=false): max_abs={max_abs:.6}");
    assert!(
        max_abs < 1e-4,
        "fused swiglu MPSGraph vs CPU max_abs={max_abs}"
    );
}
