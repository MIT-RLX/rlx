// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::GatedDeltaNetBackward` must decompose to primitives for a backend that
//! has no kernel for it.
//!
//! Three backends claim the fused backward (CPU, Metal, MLX); eight run
//! `Op::GatedDeltaNet` forward. The other five reach the backward op through
//! the ordinary gradient walk and cannot execute it, so
//! `decompose_backward_ops_except` — the same mechanism that already covers
//! `AdaLayerNormBackward`, `ScanBackward` and the conv/norm backward kernels —
//! has to cover this one too.
//!
//! `preserved = []` is what a backend claiming none of them passes, so it is
//! also the strongest form of the check. The oracle is the fused kernel itself,
//! which `gated_delta_net_fused_backward.rs` pins to the unrolled path and
//! `gated_delta_net_vjp_fd.rs` pins to finite differences.
//!
//! Both gate modes and both state modes run: `carry_state` adds a sixth
//! gradient (`dstate`) and shifts nothing else, which is exactly the kind of
//! packing detail an off-by-one in the layout would hide.

use rlx_autodiff::{GradWithLossOptions, Wrt, decompose_backward_ops_except, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Op, Shape};

const B: usize = 2;
const S: usize = 5;
const H: usize = 3;
const N: usize = 8;

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

fn gdn_graph(gate_per_channel: bool, carry_state: bool) -> Graph {
    let f = DType::F32;
    let bshn = Shape::new(&[B, S, H, N], f);
    let gate_shape = if gate_per_channel {
        bshn.clone()
    } else {
        Shape::new(&[B, S, H], f)
    };
    let mut g = Graph::new("gdn");
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let gate = g.input("g", gate_shape);
    let beta = g.input("beta", Shape::new(&[B, S, H], f));
    let mut inputs = vec![q, k, v, gate, beta];
    if carry_state {
        inputs.push(g.input("state", Shape::new(&[B, H, N, N], f)));
    }
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: N,
            carry_state,
            gate_per_channel,
        },
        inputs,
        bshn,
    );
    g.set_outputs(vec![y]);
    g
}

fn backward_graph(gate_per_channel: bool, carry_state: bool) -> Graph {
    let mut wrt = vec![
        Wrt::Leaf("q".into()),
        Wrt::Leaf("k".into()),
        Wrt::Leaf("v".into()),
        Wrt::Leaf("g".into()),
        Wrt::Leaf("beta".into()),
    ];
    if carry_state {
        wrt.push(Wrt::Leaf("state".into()));
    }
    grad_with_loss_wrt(
        &gdn_graph(gate_per_channel, carry_state),
        &wrt,
        GradWithLossOptions::STRICT.with_aux(false),
    )
}

fn run(graph: Graph, gate_per_channel: bool, carry_state: bool) -> Vec<Vec<f32>> {
    let qkv = B * S * H * N;
    let bsh = B * S * H;
    let gl = if gate_per_channel { qkv } else { bsh };

    let q: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(1, i)).collect();
    let k: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(2, i)).collect();
    let v: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(3, i)).collect();
    // Decay below 1, so the log-gate is negative.
    let gate: Vec<f32> = (0..gl).map(|i| -0.4 + 0.15 * hashed(4, i)).collect();
    let beta: Vec<f32> = (0..bsh).map(|i| 0.5 + 0.2 * hashed(5, i)).collect();
    let cot: Vec<f32> = (0..qkv).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();
    let state: Vec<f32> = (0..B * H * N * N).map(|i| 0.2 * hashed(6, i)).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph);
    let mut binds: Vec<(&str, &[f32])> = vec![
        ("q", &q[..]),
        ("k", &k[..]),
        ("v", &v[..]),
        ("g", &gate[..]),
        ("beta", &beta[..]),
        ("d_output", &cot[..]),
    ];
    if carry_state {
        binds.push(("state", &state[..]));
    }
    compiled.run(&binds)
}

#[test]
fn decomposed_backward_matches_the_fused_kernel() {
    for carry_state in [false, true] {
        for gate_per_channel in [false, true] {
            let case = format!("gate_per_channel={gate_per_channel} carry_state={carry_state}");

            let fused_graph = backward_graph(gate_per_channel, carry_state);
            assert!(
                fused_graph
                    .nodes()
                    .iter()
                    .any(|n| matches!(n.op, Op::GatedDeltaNetBackward { .. })),
                "{case}: expected the fused op before decomposition — \
                 otherwise this compares one decomposition against another"
            );

            // `preserved = []` is what a backend claiming no fused backward
            // kernel passes.
            let decomposed = decompose_backward_ops_except(fused_graph.clone(), &[]);
            assert!(
                !decomposed
                    .nodes()
                    .iter()
                    .any(|n| matches!(n.op, Op::GatedDeltaNetBackward { .. })),
                "{case}: Op::GatedDeltaNetBackward survived decomposition"
            );

            let fused = run(fused_graph, gate_per_channel, carry_state);
            let primitive = run(decomposed, gate_per_channel, carry_state);

            assert_eq!(fused.len(), primitive.len(), "{case}: output count differs");
            let labels = ["y", "dq", "dk", "dv", "dg", "dbeta", "dstate"];
            let mut worst = 0.0f32;
            for (i, (a, b)) in fused.iter().zip(&primitive).enumerate() {
                let label = labels.get(i).copied().unwrap_or("?");
                assert_eq!(a.len(), b.len(), "{case}: {label} length differs");
                for (j, (x, y)) in a.iter().zip(b).enumerate() {
                    let d = (x - y).abs();
                    worst = worst.max(d);
                    assert!(d < 2e-4, "{case} {label}[{j}]: fused {x} vs decomposed {y}");
                }
                let magnitude = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(
                    magnitude > 1e-4,
                    "{case}: {label} is ~zero (max {magnitude}) — \
                     the comparison is vacuous"
                );
            }
            eprintln!("{case}: worst |fused - decomposed| = {worst:.2e}");
        }
    }
}

/// A backend that *does* claim the kernel must keep it — the decomposition is
/// ~32× slower, so leaking it onto CPU/Metal/MLX would be a silent regression.
#[test]
fn preserved_backends_keep_the_fused_op() {
    let bwd = backward_graph(false, false);
    let kept = decompose_backward_ops_except(bwd, &[rlx_ir::OpKind::GatedDeltaNetBackward]);
    assert!(
        kept.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedDeltaNetBackward { .. })),
        "a backend listing GatedDeltaNetBackward in supported_ops must keep it fused"
    );
}
