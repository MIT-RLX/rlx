// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-vs-CPU parity for the `Op::SynthMatMul` VJP **dx path** (the
//! gather-reconstruct Wᵀ → MatMul that produces the input gradient). The
//! existing `synth_codebook_train` test only differentiates w.r.t. the codebook
//! (the scatter path); nothing exercised dx — which is what feeds the KAN
//! spline backward in rlx-tiny and explodes on Metal.

#![cfg(all(target_os = "macos", feature = "metal", feature = "training"))]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape, SynthKind};
use rlx_runtime::{Device, Session};

const B: usize = 128; // rows (tokens)
const D: usize = 4; // entry_dim
const NE: usize = 256; // num_entries
const KB: usize = 192; // blocks; K = KB*D = 768 (ff)
const K: usize = KB * D;
const N: usize = 192; // output dim (hidden)

fn build() -> (Graph, Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut g = Graph::new("synth_dx");
    let x = g.param("x", Shape::new(&[B, K], DType::F32));
    // Model bakes indices as a U8 *Constant* (see rlx-tiny bake_idx), NOT a param.
    let idx: Vec<u8> = (0..N * KB).map(|i| ((i * 7 + 3) % NE) as u8).collect();
    let indices = g.add_node(
        Op::Constant { data: idx.clone() },
        vec![],
        Shape::new(&[N, KB], DType::U8),
    );
    let codebook = g.param("codebook", Shape::new(&[NE, D], DType::F32));
    let y = g.synth_matmul(
        x,
        indices,
        codebook,
        SynthKind::Codebook {
            entry_dim: D as u32,
            num_entries: NE as u32,
        },
        Shape::new(&[B, N], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![y, y],
        Shape::new(&[B, N], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(B * N) as i64],
        },
        vec![sq],
        Shape::new(&[B * N], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);

    // Differentiate BOTH x (dx path) and codebook (scatter path).
    let mut backward = grad_with_loss(&g, &[x, codebook]);
    for node in backward.nodes_mut() {
        if let Op::Param { name } = &node.op {
            if name == "x" || name == "codebook" {
                node.op = Op::Input { name: name.clone() };
            }
        }
    }

    let xv: Vec<f32> = (0..B * K).map(|i| (i as f32 * 0.011).sin() * 0.8).collect();
    let cv: Vec<f32> = (0..NE * D)
        .map(|i| (i as f32 * 0.019).cos() * 0.5)
        .collect();
    (backward, xv, idx, cv)
}

fn run_on(dev: Device, backward: &Graph, xv: &[f32], _idx: &[u8], cv: &[f32]) -> Vec<Vec<f32>> {
    let mut m = Session::new(dev).compile(backward.clone());
    m.run(&[("x", xv), ("codebook", cv), ("d_output", &[1.0f32])])
}

#[test]
fn metal_synth_vjp_dx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let (backward, xv, idx, cv) = build();
    let cpu = run_on(Device::Cpu, &backward, &xv, &idx, &cv);
    let met = run_on(Device::Metal, &backward, &xv, &idx, &cv);
    assert_eq!(cpu.len(), met.len());
    // outputs: [loss, dx (len B*K), dcodebook (len NE*D)]
    let labels = ["loss", "dx", "dcodebook"];
    for (oi, (c, m)) in cpu.iter().zip(met.iter()).enumerate() {
        let cpu_max = c.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let met_max = m.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let fin = m.iter().all(|v| v.is_finite());
        let lab = labels.get(oi).copied().unwrap_or("?");
        eprintln!(
            "output[{oi}={lab}] len={} cpu_max={cpu_max:.4e} metal_max={met_max:.4e} metal_finite={fin}",
            c.len()
        );
        assert!(fin, "output[{oi}={lab}] non-finite on Metal");
        let mut max_err = 0f32;
        for (a, b) in c.iter().zip(m.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        let scale = cpu_max.max(1.0);
        assert!(
            max_err / scale < 1e-3,
            "output[{oi}={lab}] Metal != CPU: max_err={max_err:.3e} rel={:.3e}",
            max_err / scale
        );
    }
}
