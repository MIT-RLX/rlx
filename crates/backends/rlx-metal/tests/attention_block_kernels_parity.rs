// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal vs CPU for the kernels a multi-head encoder block (EEG-DINO:
//! d_model 200, 8 heads of 25, 191 tokens) routes through:
//!
//! * `bmm_simd_4x4` — f32 `BatchedSgemm` as ONE dispatch (was one MPS matmul
//!   per matrix). Ragged M/K/N (head_dim 25) and a broadcast operand.
//! * `softmax_lastax_out` — out-of-place last-axis softmax, register-cached
//!   when a row fits (191) and the streaming fallback when it doesn't (3000).
//! * `transpose_swap12_trail_rows_f32` — `[B,S,H,Dh] <-> [B,H,S,Dh]`; a copy,
//!   so it must be EXACT.
//! * `FusedMmBiasAct` at ragged M/N with bias + {none, relu, silu}, on the
//!   thunk path and (separately) the full-MPSGraph path.
//!
//! Every shape here is deliberately NOT a multiple of the kernels' tiles: the
//! edge staging and masked stores are what is under test.

#![cfg(target_os = "macos")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

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

fn metal_ok() -> bool {
    !rlx_ir::env::skip_unless_device(
        "metal",
        cfg!(target_os = "macos"),
        rlx_metal::is_available(),
    )
}

/// Env overrides are process-wide and tests run on parallel threads.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// (cpu, metal) outputs of the same graph, Metal on the THUNK path. A graph
/// this small lowers whole to MPSGraph by default, which would bypass every
/// kernel this file is about.
fn both(build: impl Fn() -> Graph, inputs: &[(&str, &[f32])]) -> (Vec<f32>, Vec<f32>) {
    both_with(build, inputs, "RLX_DISABLE_MPSGRAPH")
}

fn both_with(
    build: impl Fn() -> Graph,
    inputs: &[(&str, &[f32])],
    metal_flag: &str,
) -> (Vec<f32>, Vec<f32>) {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cpu = Session::new(Device::Cpu).compile(build()).run(inputs);
    let metal = {
        let _flag = rlx_ir::env::scoped(metal_flag, "1");
        Session::new(Device::Metal).compile(build()).run(inputs)
    };
    assert_eq!(cpu[0].len(), metal[0].len(), "output length differs");
    (
        cpu.into_iter().next().unwrap(),
        metal.into_iter().next().unwrap(),
    )
}

/// Max |Δ| relative to the reference's max |value| (scale-free).
fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
        / scale
}

fn bmm_case(batch: usize, m: usize, k: usize, n: usize, b_bcast: bool) {
    let a = noise(batch * m * k, 11);
    let bn = if b_bcast { k * n } else { batch * k * n };
    let b = noise(bn, 12);
    let (cpu, metal) = both(
        || {
            let mut g = Graph::new("bmm");
            let x = g.input("a", Shape::new(&[batch, m, k], F));
            let y = if b_bcast {
                g.input("b", Shape::new(&[1, k, n], F))
            } else {
                g.input("b", Shape::new(&[batch, k, n], F))
            };
            let z = g.mm(x, y);
            g.set_outputs(vec![z]);
            g
        },
        &[("a", &a), ("b", &b)],
    );
    let e = rel_err(&cpu, &metal);
    assert!(
        e < 1e-5,
        "bmm {batch}x{m}x{k}x{n} bcast={b_bcast}: rel err {e:.3e}"
    );
}

#[test]
fn batched_matmul_ragged_one_dispatch() {
    if !metal_ok() {
        return;
    }
    // scores [S,dh]·[dh,S] and context [S,S]·[S,dh] at head_dim 25.
    bmm_case(16, 191, 25, 191, false);
    bmm_case(16, 191, 191, 25, false);
    // all three dims ragged, and a broadcast right operand
    bmm_case(5, 37, 13, 41, false);
    bmm_case(5, 37, 13, 41, true);
}

fn softmax_case(rows: usize, cols: usize) {
    // Softmax of a matmul output: the planner gives it its own slot, so this
    // is the out-of-place path.
    let k = 7;
    let a = noise(rows * k, 21);
    let b = noise(k * cols, 22);
    let (cpu, metal) = both(
        || {
            let mut g = Graph::new("softmax");
            let x = g.input("a", Shape::new(&[rows, k], F));
            let w = g.input("b", Shape::new(&[k, cols], F));
            let s = g.mm(x, w);
            let p = g.softmax(s, -1, Shape::new(&[rows, cols], F));
            g.set_outputs(vec![p]);
            g
        },
        &[("a", &a), ("b", &b)],
    );
    let e = rel_err(&cpu, &metal);
    assert!(e < 1e-5, "softmax {rows}x{cols}: rel err {e:.3e}");
    for r in 0..rows {
        let s: f32 = metal[r * cols..(r + 1) * cols].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "softmax row {r} sums to {s}");
    }
}

#[test]
fn softmax_out_of_place_cached_and_streaming() {
    if !metal_ok() {
        return;
    }
    softmax_case(64, 191); // fits registers (cols <= 8 * tg width)
    softmax_case(9, 3000); // streaming fallback
}

#[test]
fn swap12_trailing_transpose_is_exact() {
    if !metal_ok() {
        return;
    }
    for &(b, s, h, d) in &[(2usize, 191usize, 8usize, 25usize), (3, 7, 5, 4)] {
        let x = noise(b * s * h * d, 31);
        // [B,S,H,D] -> [B,H,S,D] -> back: both directions are swap12.
        let (cpu, metal) = both(
            || {
                let mut g = Graph::new("swap12");
                let i = g.input("x", Shape::new(&[b, s, h, d], F));
                let t = g.transpose_(i, vec![0, 2, 1, 3]);
                let u = g.add(t, t);
                let back = g.transpose_(u, vec![0, 2, 1, 3]);
                g.set_outputs(vec![back]);
                g
            },
            &[("x", &x)],
        );
        assert_eq!(cpu, metal, "swap12 transpose {b}x{s}x{h}x{d} not exact");
    }
}

fn linear_case(m: usize, k: usize, n: usize, act: Option<Activation>, metal_flag: &str) {
    let x = noise(m * k, 41);
    let w = noise(k * n, 42);
    let bias = noise(n, 43);
    let (cpu, metal) = both_with(
        || {
            let mut g = Graph::new("linear");
            let xi = g.input("x", Shape::new(&[m, k], F));
            let wi = g.input("w", Shape::new(&[k, n], F));
            let bi = g.input("bias", Shape::new(&[n], F));
            let y = g.linear_fused(xi, wi, bi, act, Shape::new(&[m, n], F));
            g.set_outputs(vec![y]);
            g
        },
        &[("x", &x), ("w", &w), ("bias", &bias)],
        metal_flag,
    );
    // Independent host reference, so a disagreement says WHICH side is wrong.
    let mut host = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for t in 0..k {
                acc += x[i * k + t] as f64 * w[t * n + j] as f64;
            }
            let v = (acc + bias[j] as f64) as f32;
            host[i * n + j] = match act {
                Some(Activation::Relu) => v.max(0.0),
                Some(Activation::Silu) => v / (1.0 + (-v).exp()),
                _ => v,
            };
        }
    }
    let e_cpu = rel_err(&host, &cpu);
    let e_metal = rel_err(&host, &metal);
    assert!(
        e_cpu < 1e-5 && e_metal < 1e-5,
        "linear {m}x{k}x{n} act={act:?} [{metal_flag}]: cpu vs host {e_cpu:.3e}, metal vs host {e_metal:.3e}\n  \
         host {:?}\n  cpu  {:?}\n  metal {:?}",
        &host[..4],
        &cpu[..4],
        &metal[..4]
    );
}

#[test]
fn fused_linear_ragged_shapes() {
    if !metal_ok() {
        return;
    }
    for act in [None, Some(Activation::Relu), Some(Activation::Silu)] {
        let f = "RLX_DISABLE_MPSGRAPH";
        linear_case(191, 200, 600, act, f); // ragged M and N, 3x10 tiles
        linear_case(1528, 200, 200, act, f); // ragged N, 24x4 tiles
        linear_case(130, 64, 1000, act, f); // M straddles one tile row, ragged N
    }
}

/// The full-graph MPSGraph lowering mapped `FusedMatMulBiasAct { Relu }` to the
/// `None` arm and returned matmul + bias with the negatives intact (Metal-only;
/// found by `fused_linear_ragged_shapes` before it pinned the thunk path).
#[test]
fn mpsgraph_fused_linear_applies_relu() {
    if !metal_ok() {
        return;
    }
    for act in [Some(Activation::Relu), Some(Activation::Silu), None] {
        linear_case(191, 200, 600, act, "RLX_MPSGRAPH_FORCE");
    }
}
