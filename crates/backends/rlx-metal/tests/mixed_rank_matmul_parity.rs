// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `[M,K] · [B,K,N]` — a rank-2 left operand broadcast over a batched right one.
//!
//! This shape took the 2-D flatten path in `thunk::compile`, which derives
//! `k = A.numel() / (B·M)`. That is correct only when the rank-2 operand is on
//! the **right**: flattening `[B,M,K] · [K,N]` to `[B·M, K] · [K, N]` is exact.
//! With the ranks the other way round the same arithmetic yields `k = K/B`, and
//! Metal returned a tensor of exactly the right shape full of the wrong numbers
//! — 1.3e3 away from the CPU, cosine ≈ 0.
//!
//! It surfaced through an SPD congruence `Wᵀ·X·W`, where `W` is a rank-2 filter
//! bank and `X` a batch of covariances: the first product is the safe direction
//! and the second is not, so half of the expression was right.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;
const B: usize = 4;
const N: usize = 6;
const M: usize = 3;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn compare(build: impl Fn() -> Graph, inputs: &[(&str, &[f32])]) -> f32 {
    let cpu = Session::new(Device::Cpu).compile(build()).run(inputs);
    let metal = Session::new(Device::Metal).compile(build()).run(inputs);
    cpu[0]
        .iter()
        .zip(&metal[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

#[test]
fn rank2_lhs_times_rank3_rhs() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        return;
    }
    let w = noise(M * N, 1);
    let x = noise(B * N * N, 2);
    let worst = compare(
        || {
            let mut g = Graph::new("mixed-rank");
            let wt = g.input("w", Shape::new(&[M, N], F));
            let xs = g.input("x", Shape::new(&[B, N, N], F));
            let y = g.mm(wt, xs); // [M,N] · [B,N,N] → [B,M,N]
            g.set_outputs(vec![y]);
            g
        },
        &[("w", &w), ("x", &x)],
    );
    assert!(
        worst < 1e-4,
        "rank-2 lhs × rank-3 rhs diverged: {worst:.3e}"
    );
}

#[test]
fn rank3_lhs_times_rank2_rhs_is_unaffected() {
    // The direction the flatten path *is* correct for — kept so a repair of the
    // broken direction cannot quietly break the working one.
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        return;
    }
    let x = noise(B * N * N, 3);
    let w = noise(N * M, 4);
    let worst = compare(
        || {
            let mut g = Graph::new("mixed-rank-rhs");
            let xs = g.input("x", Shape::new(&[B, N, N], F));
            let wt = g.input("w", Shape::new(&[N, M], F));
            let y = g.mm(xs, wt);
            g.set_outputs(vec![y]);
            g
        },
        &[("x", &x), ("w", &w)],
    );
    assert!(
        worst < 1e-4,
        "rank-3 lhs × rank-2 rhs diverged: {worst:.3e}"
    );
}

#[test]
fn the_full_congruence_that_surfaced_it() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        return;
    }
    let w = noise(N * M, 5);
    let x = noise(B * N * N, 6);
    let worst = compare(
        || {
            let mut g = Graph::new("congruence");
            let wt = g.input("w", Shape::new(&[N, M], F));
            let xs = g.input("x", Shape::new(&[B, N, N], F));
            let t = g.transpose_(wt, vec![1, 0]); // [M,N]
            let a = g.mm(xs, wt); // [B,N,M]  — safe direction
            let y = g.mm(t, a); // [M,N]·[B,N,M] — the broken one
            g.set_outputs(vec![y]);
            g
        },
        &[("w", &w), ("x", &x)],
    );
    assert!(worst < 1e-4, "Wᵀ·X·W diverged: {worst:.3e}");
}
