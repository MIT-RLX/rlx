// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fused ggml `L2_NORM` parity — Metal fused vs unfused vs CPU.
//!
//! Builds the op chain `rlx_qwen35::builder::l2_norm` emits —
//!   `mul(x,x) → sum(last) → sqrt → max(·, eps) → div(x, ·)`
//! — and runs it three ways:
//!   * Metal with `fuse_l2_norm` ON  (`RLX_METAL_FUSE_L2NORM` unset)
//!   * Metal with `fuse_l2_norm` OFF (`RLX_METAL_FUSE_L2NORM=0`)
//!   * CPU reference
//!
//! The fused path collapses the chain into one `L2NormLastDim` dispatch.
//! Gated-DeltaNet runs this twice per linear layer, so it is 36×/token on a
//! 24-layer model. `fused_l2_norm_chains()` proves the fused thunk was actually
//! emitted rather than the test silently measuring the unfused path twice.
//!
//! The fused kernel deliberately keeps the `sqrt` → `max` → *divide* sequence
//! rather than `rsqrt` of a clamped sum: those are algebraically equal but round
//! differently and diverge outright in the clamped branch.
//!
//! Fused vs unfused is NOT bit-identical (measured ~4.5e-8, i.e. sub-`f32::EPSILON`):
//! the fused kernel sums the row with a threadgroup power-of-2 tree, while the
//! unfused chain goes through the generic `Reduce` kernel, whose accumulation
//! order differs. Only the summation order differs, so the bound is tight.
//!
//! One `#[test]`: the off-switch is a process-global env var.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const ROWS: usize = 12;
const H: usize = 128; // GDN state dim

fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("l2_norm");
    let x = g.input("x", Shape::new(&[ROWS, H], f));
    let eps = g.input("eps", Shape::new(&[1], f));

    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![x, x],
        Shape::new(&[ROWS, H], f),
    );
    let sumsq = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![1],
            keep_dim: true,
        },
        vec![sq],
        Shape::new(&[ROWS, 1], f),
    );
    let rms = g.add_node(
        Op::Activation(Activation::Sqrt),
        vec![sumsq],
        Shape::new(&[ROWS, 1], f),
    );
    let denom = g.add_node(
        Op::Binary(BinaryOp::Max),
        vec![rms, eps],
        Shape::new(&[ROWS, 1], f),
    );
    let out = g.add_node(
        Op::Binary(BinaryOp::Div),
        vec![x, denom],
        Shape::new(&[ROWS, H], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn fused_l2_norm_matches_unfused_and_cpu() {
    let g = build();
    // Mixed magnitudes so the reduction is not trivially conditioned, plus one
    // all-but-zero row that exercises the `max(·, eps)` clamp branch — the one
    // place `rsqrt(max(sum, eps²))` would disagree with `max(sqrt(sum), eps)`.
    let mut x: Vec<f32> = (0..ROWS * H)
        .map(|i| ((i as f32) * 0.017).sin() * ((i % 7) as f32 + 0.5))
        .collect();
    for v in x.iter_mut().take(H) {
        *v = 0.0;
    }
    let eps = vec![1e-6f32];

    let run = |device: Device| -> Vec<f32> {
        let mut s = Session::new(device).compile(g.clone());
        s.run(&[("x", &x), ("eps", &eps)]).remove(0)
    };

    let cpu = run(Device::Cpu);

    rlx_ir::env::unset("RLX_METAL_FUSE_L2NORM"); // default = ON
    let before = rlx_metal::thunk::fused_l2_norm_chains();
    let fused = run(Device::Metal);
    let fired = rlx_metal::thunk::fused_l2_norm_chains() - before;
    assert_eq!(fired, 1, "expected exactly one fused L2_NORM chain");

    rlx_ir::env::set("RLX_METAL_FUSE_L2NORM", "0");
    let before_off = rlx_metal::thunk::fused_l2_norm_chains();
    let unfused = run(Device::Metal);
    assert_eq!(
        rlx_metal::thunk::fused_l2_norm_chains(),
        before_off,
        "off-switch must NOT fuse"
    );
    rlx_ir::env::unset("RLX_METAL_FUSE_L2NORM");

    let fu = max_abs(&fused, &unfused);
    let fc = max_abs(&fused, &cpu);
    eprintln!("[l2_norm] fused-vs-unfused={fu:.3e} fused-vs-cpu={fc:.3e}");

    // Summation order is the only difference (threadgroup tree vs the generic
    // `Reduce` kernel), so hold this well under `f32::EPSILON` (1.19e-7).
    assert!(
        fu < 1e-7,
        "fused vs unfused {fu} — larger than reduction-order noise"
    );
    assert!(fc < 1e-7, "fused vs cpu {fc}");

    // The clamped row must be exactly zero, not NaN: sqrt(0) = 0 is clamped to
    // eps before the divide.
    assert!(
        fused[..H].iter().all(|v| *v == 0.0),
        "all-zero row must normalise to zeros, got {:?}",
        &fused[..4]
    );
}
