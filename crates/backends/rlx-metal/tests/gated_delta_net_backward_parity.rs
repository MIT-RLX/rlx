// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal `Op::GatedDeltaNetBackward` against the CPU kernel.
//!
//! The CPU kernel is verified against finite differences for both gate modes
//! (`rlx-cpu`, `gdn::backward_tests`) and against the unrolled decomposition it
//! replaced (`rlx-autodiff/tests/gated_delta_net_fused_backward.rs`), so it is
//! the right oracle here: same graph, same inputs, gradients compared entry by
//! entry.
//!
//! Both gate modes and both carry settings are covered — the per-channel gate
//! decays each key row separately and the carried state adds a sixth gradient,
//! and each takes a different path through the kernel.

// rlx-metal is itself the Metal backend; no feature gate needed here.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 5;
const H: usize = 3;
const N: usize = 16;

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

fn run(device: Device, gate_per_channel: bool, carry_state: bool) -> Vec<Vec<f32>> {
    let qkv = B * S * H * N;
    let bsh = B * S * H;
    let gl = if gate_per_channel { qkv } else { bsh };
    let state_len = B * H * N * N;

    let q: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(1, i)).collect();
    let k: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(2, i)).collect();
    let v: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(3, i)).collect();
    // Decay below 1 ⇒ negative log-gate.
    let gate: Vec<f32> = (0..gl).map(|i| -0.4 + 0.15 * hashed(4, i)).collect();
    let beta: Vec<f32> = (0..bsh).map(|i| 0.5 + 0.2 * hashed(5, i)).collect();
    let state: Vec<f32> = (0..state_len).map(|i| 0.1 * hashed(6, i)).collect();
    let cot: Vec<f32> = (0..qkv).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();

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
    let bwd = grad_with_loss_wrt(
        &gdn_graph(gate_per_channel, carry_state),
        &wrt,
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = Session::new(device).compile(bwd);
    let mut feed: Vec<(&str, &[f32])> = vec![
        ("q", &q[..]),
        ("k", &k[..]),
        ("v", &v[..]),
        ("g", &gate[..]),
        ("beta", &beta[..]),
        ("d_output", &cot[..]),
    ];
    if carry_state {
        feed.push(("state", &state[..]));
    }
    compiled.run(&feed)
}

fn check(gate_per_channel: bool, carry_state: bool) {
    // Skip where there is no Metal device. `rlx-metal` builds on Linux (the op
    // claim in `supported_ops` is a const and is checked below without a
    // device), but running `Device::Metal` there cannot work — these three tests
    // were the only ones in the crate without this guard, and they failed on both
    // Linux rigs the moment a workspace `cargo test` stopped skipping this crate.
    //
    // The skip is announced rather than silent: a parity test that quietly does
    // nothing is worse than one that fails, because it reads as coverage.
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!(
            "skip gated_delta_net_backward_parity (gate_per_channel={gate_per_channel}, \
             carry_state={carry_state}): Metal unavailable"
        );
        return;
    }
    let cpu = run(Device::Cpu, gate_per_channel, carry_state);
    let metal = run(Device::Metal, gate_per_channel, carry_state);

    assert_eq!(cpu.len(), metal.len(), "output count differs");
    let labels = ["y", "dq", "dk", "dv", "dg", "dbeta", "dstate"];
    let mut worst = 0.0f32;
    for (i, (a, b)) in cpu.iter().zip(&metal).enumerate() {
        let label = labels.get(i).copied().unwrap_or("?");
        assert_eq!(a.len(), b.len(), "{label}: length differs");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            worst = worst.max(d);
            assert!(
                d < 2e-4,
                "gate_per_channel={gate_per_channel} carry={carry_state} \
                 {label}[{j}]: cpu {x} vs metal {y}"
            );
        }
        let magnitude = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            magnitude > 1e-4,
            "{label} is ~zero (max {magnitude}) — the comparison is vacuous"
        );
    }
    eprintln!(
        "gate_per_channel={gate_per_channel} carry={carry_state}: \
         worst |cpu - metal| = {worst:.2e}"
    );
}

#[test]
fn per_head_gate_matches_cpu() {
    check(false, false);
}

#[test]
fn per_channel_gate_matches_cpu() {
    check(true, false);
}

#[test]
fn carried_state_matches_cpu() {
    check(false, true);
}

/// Metal must actually claim the op — otherwise the tests above silently
/// compare the CPU fallback against itself.
#[test]
fn metal_claims_the_backward_op() {
    assert!(
        rlx_metal::supported_ops::SUPPORTED_OPS.contains(&rlx_ir::OpKind::GatedDeltaNetBackward),
        "Metal should claim GatedDeltaNetBackward"
    );
}
