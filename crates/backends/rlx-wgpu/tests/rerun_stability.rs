// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! wgpu must agree with CPU on a transformer block — op by op, and across
//! repeated runs of one compiled graph.
//!
//! These started as an investigation and are kept as the **ruled-out list** for
//! an open bug: `rlx-stackedlora` (an EEG transformer with LoRA adapters) embeds
//! ~50% off against CPU on wgpu at 22ch x 1000, deterministically, while Metal,
//! MLX and CoreML are all cos 1.0. The divergence appears after the patch-embed
//! stage and is input-dependent at a fixed shape.
//!
//! Everything below passes, so the bug is **not** in any of these on their own,
//! at the shapes that model actually builds:
//!
//!   * a hand-rolled rank-4 softmax (`max`/`exp`/`sum`/`div`, last axis, keepdim)
//!   * a low-rank matmul chain (`[m,k] @ [k,r] @ [r,n]`, r as small as 1)
//!   * batched rank-4 attention matmuls (`[B,H,S,D] @ [B,H,D,S] @ [B,H,S,D]`)
//!   * `ln`, `gelu`, `mm` and the softmax chain at `[1, 880, 200]`
//!   * re-running any of the above on the same compiled session
//!
//! Which points at an interaction — fusion, scheduling, or arena reuse in the
//! full graph — rather than a single kernel. The end-to-end reproducer lives in
//! `exg`'s `rlx-stackedlora/tests/xbackend_stages.rs`.
//!
//!     cargo test -p rlx-wgpu --test rerun_stability -- --nocapture

// The tolerance checks below are written `!(rel < tol)` rather than
// `rel >= tol`, and the negation is load-bearing: every comparison against NaN
// is false, so `NaN >= tol` would pass and silently report a diverged run as
// clean. `!(NaN < tol)` is true and fails, which is the behaviour a parity test
// needs. Clippy's rewrite is correct in general and wrong here.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            (z >> 40) as f32 / 8_388_608.0 - 0.5
        })
        .collect()
}

/// A hand-rolled softmax over the last axis of a rank-4 tensor, the shape an
/// attention block builds: `max(keepdim) → exp(x - m) → sum(keepdim) → div`.
fn softmax4_graph(b: usize, h: usize, s: usize) -> (Graph, Vec<usize>) {
    let dims = vec![b, h, s, s];
    let keep = vec![b, h, s, 1];
    let mut g = Graph::new("softmax4");
    let x = g.input("x", Shape::new(&dims, F32));
    let m = g.reduce(x, ReduceOp::Max, vec![3], true, Shape::new(&keep, F32));
    let d = g.sub(x, m);
    let e = g.exp(d);
    let sum = g.reduce(e, ReduceOp::Sum, vec![3], true, Shape::new(&keep, F32));
    let y = g.div(e, sum);
    g.set_outputs(vec![y]);
    (g, dims)
}

/// A LoRA-shaped chain: `[m,k] @ [k,r] @ [r,n]` with a *small* rank `r`. A tiled
/// matmul whose accumulator tile is not cleared per launch is correct on the
/// first run (the buffer starts zeroed) and wrong afterwards, and a rank far
/// below the tile width is where a partial tile shows up.
fn lora_graph(m: usize, k: usize, r: usize, n: usize) -> Graph {
    let mut g = Graph::new("lora");
    let x = g.input("x", Shape::new(&[m, k], F32));
    let a = g.param("a", Shape::new(&[k, r], F32));
    let b = g.param("b", Shape::new(&[r, n], F32));
    let lo = g.mm(x, a);
    let y = g.mm(lo, b);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn repeated_runs_of_a_low_rank_matmul_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("wgpu not available in this build — skipped");
        return;
    }
    let mut failures = Vec::new();
    for &(m, k, r, n) in &[
        (40usize, 200usize, 4usize, 200usize),
        (880, 200, 8, 200),
        (40, 200, 1, 200),
    ] {
        let mut cpu = Session::new(Device::Cpu).compile(lora_graph(m, k, r, n));
        let mut gpu = Session::new(Device::Gpu).compile(lora_graph(m, k, r, n));
        for (s, sess) in [(&mut cpu, 0u8), (&mut gpu, 1u8)]
            .iter()
            .map(|_| (0, 0))
            .take(0)
        {
            let _ = (s, sess);
        }
        for name in ["a", "b"] {
            let len = if name == "a" { k * r } else { r * n };
            let seed = if name == "a" { 11 } else { 12 };
            cpu.set_param(name, &fill(len, seed));
            gpu.set_param(name, &fill(len, seed));
        }
        cpu.finalize_params();
        gpu.finalize_params();
        for rep in 0..3 {
            let x = fill(m * k, 20 + rep as u64);
            let want = cpu.run(&[("x", &x)]).remove(0);
            let got = gpu.run(&[("x", &x)]).remove(0);
            let max = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
            eprintln!(
                "  m={m} k={k} r={r:2} n={n} run {rep}: max|Δ| = {max:.3e} ({:.2e} rel)",
                max / scale
            );
            if !(max / scale < 1e-4) {
                failures.push(format!(
                    "m={m} k={k} r={r} n={n} run {rep}: {:.2e} rel",
                    max / scale
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "wgpu low-rank matmul diverges:\n  {}",
        failures.join("\n  ")
    );
}

/// Batched rank-4 matmul, the shape an attention block builds:
/// `[B,H,S,D] @ [B,H,D,S] -> [B,H,S,S]` then `[B,H,S,S] @ [B,H,S,D]`.
fn batched_attn_graph(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let mut g = Graph::new("battn");
    let q = g.input("q", Shape::new(&[b, h, s, d], F32));
    let kt = g.input("kt", Shape::new(&[b, h, d, s], F32));
    let v = g.input("v", Shape::new(&[b, h, s, d], F32));
    let energy = g.mm(q, kt);
    let ctx = g.mm(energy, v);
    g.set_outputs(vec![ctx]);
    g
}

#[test]
fn repeated_runs_of_a_batched_matmul_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("wgpu not available in this build — skipped");
        return;
    }
    let mut failures = Vec::new();
    // 8 tokens is the size that stayed correct downstream, 40 the one that did not.
    for &(b, h, s, d) in &[
        (1usize, 4usize, 8usize, 25usize),
        (1, 4, 40, 25),
        (1, 8, 40, 64),
    ] {
        let mut cpu = Session::new(Device::Cpu).compile(batched_attn_graph(b, h, s, d));
        let mut gpu = Session::new(Device::Gpu).compile(batched_attn_graph(b, h, s, d));
        for rep in 0..3 {
            let q = fill(b * h * s * d, 30 + rep as u64);
            let kt = fill(b * h * d * s, 40 + rep as u64);
            let v = fill(b * h * s * d, 50 + rep as u64);
            let ins: Vec<(&str, &[f32])> = vec![("q", &q), ("kt", &kt), ("v", &v)];
            let want = cpu.run(&ins).remove(0);
            let got = gpu.run(&ins).remove(0);
            let max = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |acc, x| acc.max(x.abs())).max(1e-6);
            eprintln!(
                "  b={b} h={h} s={s:3} d={d:3} run {rep}: max|Δ| = {max:.3e} ({:.2e} rel)",
                max / scale
            );
            if !(max / scale < 1e-4) {
                failures.push(format!(
                    "b={b} h={h} s={s} d={d} run {rep}: {:.2e} rel",
                    max / scale
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "wgpu batched matmul diverges:\n  {}",
        failures.join("\n  ")
    );
}

/// Each op the failing transformer block uses, on its own, at the real shapes.
#[test]
fn per_op_at_transformer_shapes_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("wgpu not available in this build — skipped");
        return;
    }
    // [B, tokens, d_model] as `rlx-stackedlora` builds for 22ch x 1000.
    let (b, n, d) = (1usize, 40usize * 22, 200usize);
    let mut failures = Vec::new();
    for op in ["ln", "gelu", "mm", "softmax_last"] {
        let mut g = Graph::new(op);
        let x = g.input("x", Shape::new(&[b, n, d], F32));
        let y = match op {
            "ln" => {
                let gw = g.param("gw", Shape::new(&[d], F32));
                let gb = g.param("gb", Shape::new(&[d], F32));
                g.ln(x, gw, gb, 1e-5)
            }
            "gelu" => g.gelu(x),
            "mm" => {
                let w = g.param("w", Shape::new(&[d, d], F32));
                g.mm(x, w)
            }
            _ => {
                let keep = Shape::new(&[b, n, 1], F32);
                let m = g.reduce(x, ReduceOp::Max, vec![2], true, keep.clone());
                let e0 = g.sub(x, m);
                let e = g.exp(e0);
                let s = g.reduce(e, ReduceOp::Sum, vec![2], true, keep);
                g.div(e, s)
            }
        };
        g.set_outputs(vec![y]);
        let mut g2 = Graph::new(op);
        {
            // Rebuild an identical graph for the second session.
            let x = g2.input("x", Shape::new(&[b, n, d], F32));
            let y = match op {
                "ln" => {
                    let gw = g2.param("gw", Shape::new(&[d], F32));
                    let gb = g2.param("gb", Shape::new(&[d], F32));
                    g2.ln(x, gw, gb, 1e-5)
                }
                "gelu" => g2.gelu(x),
                "mm" => {
                    let w = g2.param("w", Shape::new(&[d, d], F32));
                    g2.mm(x, w)
                }
                _ => {
                    let keep = Shape::new(&[b, n, 1], F32);
                    let m = g2.reduce(x, ReduceOp::Max, vec![2], true, keep.clone());
                    let e0 = g2.sub(x, m);
                    let e = g2.exp(e0);
                    let s = g2.reduce(e, ReduceOp::Sum, vec![2], true, keep);
                    g2.div(e, s)
                }
            };
            g2.set_outputs(vec![y]);
        }
        let mut cpu = Session::new(Device::Cpu).compile(g);
        let mut gpu = Session::new(Device::Gpu).compile(g2);
        for (name, len) in [("gw", d), ("gb", d), ("w", d * d)] {
            if op == "ln" && (name == "gw" || name == "gb") || op == "mm" && name == "w" {
                cpu.set_param(name, &fill(len, 61));
                gpu.set_param(name, &fill(len, 61));
            }
        }
        cpu.finalize_params();
        gpu.finalize_params();
        for rep in 0..2 {
            let xs: Vec<f32> = fill(b * n * d, 70)
                .iter()
                .map(|v| v + rep as f32 * 0.25)
                .collect();
            let want = cpu.run(&[("x", &xs)]).remove(0);
            let got = gpu.run(&[("x", &xs)]).remove(0);
            let max = want
                .iter()
                .zip(&got)
                .map(|(a, c)| (a - c).abs())
                .fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
            eprintln!(
                "  {op:12} run {rep}: max|Δ| = {max:.3e} ({:.2e} rel)",
                max / scale
            );
            if !(max / scale < 1e-4) {
                failures.push(format!("{op} run {rep}: {:.2e} rel", max / scale));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "wgpu per-op diverges:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn repeated_runs_of_one_graph_match_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("wgpu not available in this build — skipped");
        return;
    }
    let mut failures = Vec::new();
    // 8 is the size that passed downstream, 40 the one that failed.
    for &(b, h, s) in &[(1usize, 4usize, 8usize), (1, 4, 40), (1, 8, 40), (2, 4, 64)] {
        let (g_cpu, dims) = softmax4_graph(b, h, s);
        let (g_gpu, _) = softmax4_graph(b, h, s);
        let n: usize = dims.iter().product();
        let mut cpu = Session::new(Device::Cpu).compile(g_cpu);
        let mut gpu = Session::new(Device::Gpu).compile(g_gpu);

        for rep in 0..3 {
            let x = fill(n, 7 + rep as u64);
            let want = cpu.run(&[("x", &x)]).remove(0);
            let got = gpu.run(&[("x", &x)]).remove(0);
            let max = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("  b={b} h={h} s={s:3} run {rep}: max|Δ| = {max:.3e}");
            if !(max < 1e-5) {
                failures.push(format!("b={b} h={h} s={s} run {rep}: max|Δ|={max:.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "wgpu diverges from cpu:\n  {}",
        failures.join("\n  ")
    );
}
