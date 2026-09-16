// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fused `Op::GatedDeltaNetBackward` must reproduce the unrolled path.
//!
//! `Op::GatedDeltaNet` used to be unfused for autodiff — its time loop
//! expanded into per-timestep primitives whose existing VJPs the gradient walk
//! could reach. That is correct but ~32× slower than the fused kernel, because
//! in SSA form every timestep materializes a fresh `[B·H, N, N]` state while
//! the kernel updates one working set in place. It now has a dedicated VJP.
//!
//! The unrolled path is independently verified against finite differences
//! (`gated_delta_net_vjp_fd.rs`), which makes it the natural oracle for the new
//! kernel: same graph, same inputs, gradients compared entry by entry. That is
//! a far tighter check than finite differences alone, and it covers the packing
//! and slicing of the bundled gradient output as well as the scan arithmetic.
//!
//! `RLX_GDN_UNFUSE_FOR_AD=1` selects the old path, so both run here. Everything
//! lives in one test: the switch is process-global, so splitting it would let
//! the two halves race.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Shape};

const B: usize = 2;
const S: usize = 6;
const H: usize = 3;
const N: usize = 8;

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

fn gdn_graph(gate_per_channel: bool) -> Graph {
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
    let y = g.add_node(
        rlx_ir::Op::GatedDeltaNet {
            state_size: N,
            carry_state: false,
            gate_per_channel,
        },
        vec![q, k, v, gate, beta],
        bshn,
    );
    g.set_outputs(vec![y]);
    g
}

/// `[dy_mirror, dq, dk, dv, dg, dbeta]`.
fn grads(gate_per_channel: bool) -> Vec<Vec<f32>> {
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

    let bwd = grad_with_loss_wrt(
        &gdn_graph(gate_per_channel),
        &[
            Wrt::Leaf("q".into()),
            Wrt::Leaf("k".into()),
            Wrt::Leaf("v".into()),
            Wrt::Leaf("g".into()),
            Wrt::Leaf("beta".into()),
        ],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.run(&[
        ("q", &q[..]),
        ("k", &k[..]),
        ("v", &v[..]),
        ("g", &gate[..]),
        ("beta", &beta[..]),
        ("d_output", &cot[..]),
    ])
}

#[test]
fn fused_backward_matches_the_unrolled_path() {
    for gate_per_channel in [false, true] {
        rlx_ir::env::unset("RLX_GDN_UNFUSE_FOR_AD");
        let fused = grads(gate_per_channel);

        rlx_ir::env::set("RLX_GDN_UNFUSE_FOR_AD", "1");
        let unrolled = grads(gate_per_channel);
        rlx_ir::env::unset("RLX_GDN_UNFUSE_FOR_AD");

        assert_eq!(
            fused.len(),
            unrolled.len(),
            "gate_per_channel={gate_per_channel}: output count differs"
        );
        let labels = ["y", "dq", "dk", "dv", "dg", "dbeta"];
        let mut worst = 0.0f32;
        for (i, (a, b)) in fused.iter().zip(&unrolled).enumerate() {
            let label = labels.get(i).copied().unwrap_or("?");
            assert_eq!(
                a.len(),
                b.len(),
                "gate_per_channel={gate_per_channel}: {label} length differs"
            );
            for (j, (x, y)) in a.iter().zip(b).enumerate() {
                let d = (x - y).abs();
                worst = worst.max(d);
                assert!(
                    d < 2e-4,
                    "gate_per_channel={gate_per_channel} {label}[{j}]: \
                     fused {x} vs unrolled {y}"
                );
            }
            let magnitude = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                magnitude > 1e-4,
                "gate_per_channel={gate_per_channel}: {label} is ~zero \
                 (max {magnitude}) — the comparison is vacuous"
            );
        }
        eprintln!("gate_per_channel={gate_per_channel}: worst |fused - unrolled| = {worst:.2e}");
    }
}

/// The fused path must actually be in use by default — otherwise the test above
/// compares the unrolled path against itself.
#[test]
fn fused_backward_is_the_default() {
    rlx_ir::env::unset("RLX_GDN_UNFUSE_FOR_AD");
    let bwd = grad_with_loss_wrt(
        &gdn_graph(false),
        &[Wrt::Leaf("q".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let has_backward_op = bwd
        .nodes()
        .iter()
        .any(|n| matches!(n.op, rlx_ir::Op::GatedDeltaNetBackward { .. }));
    assert!(
        has_backward_op,
        "expected Op::GatedDeltaNetBackward in the backward graph"
    );
}
