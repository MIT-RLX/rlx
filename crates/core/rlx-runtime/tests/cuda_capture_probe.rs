// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! CUDA graph-capture **segmentation opportunity** probe (Tier-3).
//!
//! rlx-cuda graph capture is all-or-nothing: a single host step anywhere in the
//! schedule (LSTM / Mamba2 / Scan / Sort / linalg …) forces the whole graph —
//! often an entire transformer — onto eager per-launch dispatch. This builds a
//! realistic *hybrid* graph — GPU compute blocks (matmul → SiLU → matmul →
//! residual) separated by host `Op::Sort` steps that stand in for an SSM/scan
//! fallback — and runs it under `RLX_CUDA_EXEC_MODE=graph` +
//! `RLX_CUDA_CAPTURE_DEBUG=1`. The capture-safety gate then prints, per run,
//! what SEGMENTED capture would recover ("would replay X/N steps across K
//! graph(s)"). No-ops without a real CUDA GPU (`is_available` guard); run on the
//! msi rig with `--nocapture` to read the diagnostic.
//!
//!   RLX_CUDA_EXEC_MODE=graph RLX_CUDA_CAPTURE_DEBUG=1 \
//!     cargo test -p rlx-runtime --features cuda --test cuda_capture_probe -- --nocapture

use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    }
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 0.1 - 0.05
        })
        .collect()
}

/// Hybrid graph: `layers` pre-norm FFN blocks (`RmsNorm → matmul → SiLU →
/// matmul → residual` — exercises reductions + cuBLAS in capture) with a host
/// `Sort` after every `host_every`-th block (never trailing) to mimic an SSM
/// host fallback.
fn hybrid_graph(m: usize, d: usize, hidden: usize, layers: usize, host_every: usize) -> Graph {
    let mut g = Graph::new("hybrid_ssm");
    let x = g.input("x", Shape::new(&[m, d], F));
    let w1 = g.input("w1", Shape::new(&[d, hidden], F));
    let w2 = g.input("w2", Shape::new(&[hidden, d], F));
    let gamma = g.input("gamma", Shape::new(&[d], F));
    let beta = g.input("beta", Shape::new(&[d], F));
    let mut h = x;
    for l in 0..layers {
        let n = g.add_node(
            rlx_ir::Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![h, gamma, beta],
            Shape::new(&[m, d], F),
        );
        let up = g.matmul(n, w1, Shape::new(&[m, hidden], F));
        let act = g.activation(Activation::Silu, up, Shape::new(&[m, hidden], F));
        let down = g.matmul(act, w2, Shape::new(&[m, d], F));
        h = g.binary(rlx_ir::op::BinaryOp::Add, h, down, Shape::new(&[m, d], F));
        // Host step between blocks (not after the last layer).
        if (l + 1) % host_every == 0 && l + 1 < layers {
            h = g.sort(h, 1, false, Shape::new(&[m, d], F));
        }
    }
    g.set_outputs(vec![h]);
    g
}

/// gamma≈1, beta≈0 norm params of length `d`.
fn norm_params(d: usize) -> (Vec<f32>, Vec<f32>) {
    let gamma: Vec<f32> = (0..d)
        .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let beta: Vec<f32> = (0..d).map(|i| ((i % 5) as f32 - 2.0) * 0.005).collect();
    (gamma, beta)
}

/// `layers` chained causal self-attention ops (`[1,H,S,Dh]`), each feeding its
/// output as the next query — exercises CUDA's FUSED attention kernel (softmax +
/// online reductions, the riskiest op for graph capture) inside one capture.
fn attention_graph(heads: usize, seq: usize, head_dim: usize, layers: usize) -> Graph {
    let mut g = Graph::new("attn_chain");
    let s = Shape::new(&[1, heads, seq, head_dim], F);
    let q = g.input("q", s.clone());
    let k = g.input("k", s.clone());
    let v = g.input("v", s.clone());
    let mut cur = q;
    for _ in 0..layers {
        cur = g.attention_kind(cur, k, v, heads, head_dim, MaskKind::Causal, s.clone());
    }
    g.set_outputs(vec![cur]);
    g
}

/// Whole-graph capture of a FUSED-ATTENTION chain — the highest-risk op for CUDA
/// graph capture (softmax + reductions). Run with RLX_CUDA_WHOLE_GRAPH_CAPTURE=1.
#[test]
fn cuda_whole_graph_attention_capture_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip cuda_whole_graph_attention ({dev:?} unavailable)");
        return;
    }
    let (heads, seq, head_dim, layers) = (4, 16, 32, 3);
    let n = heads * seq * head_dim;
    let q = seeded(n, 5);
    let k = seeded(n, 6);
    let v = seeded(n, 7);
    let feed: [(&str, &[f32]); 3] = [("q", &q), ("k", &k), ("v", &v)];
    let mk = || attention_graph(heads, seq, head_dim, layers);

    let cpu = Session::new(Device::Cpu).compile(mk()).run(&feed).remove(0);
    let mut cuda = Session::new(dev).compile(mk());
    let o1 = cuda.run(&feed).remove(0);
    let o2 = cuda.run(&feed).remove(0);
    let o3 = cuda.run(&feed).remove(0);

    let maxd = |o: &[f32]| {
        o.iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let (d1, d2, d3) = (maxd(&o1), maxd(&o2), maxd(&o3));
    eprintln!("[attn] max|CUDA-CPU| warmup={d1:.3e} capture={d2:.3e} replay={d3:.3e}");
    assert!(
        d1 < 1e-3 && d2 < 1e-3 && d3 < 1e-3,
        "attention diverged from CPU"
    );
    assert_eq!(o1, o2, "warm-up != capture");
    assert_eq!(
        o2, o3,
        "capture != replay (attention whole-graph determinism)"
    );
}

#[test]
fn cuda_segmentation_opportunity_probe() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip cuda_capture_probe ({dev:?} unavailable)");
        return;
    }
    let (m, d, hidden, layers, host_every) = (64, 256, 1024, 9, 3);
    let g = hybrid_graph(m, d, hidden, layers, host_every);
    let x = seeded(m * d, 1);
    let w1 = seeded(d * hidden, 2);
    let w2 = seeded(hidden * d, 3);

    eprintln!(
        "[probe] hybrid graph: {layers} GPU FFN blocks, host Sort every {host_every} \
         → {} host steps. Set RLX_CUDA_EXEC_MODE=graph RLX_CUDA_CAPTURE_DEBUG=1 \
         [RLX_CUDA_SEGMENTED_CAPTURE=1] to see / exercise segmented capture.",
        (1..layers).filter(|l| (l % host_every) == 0).count()
    );

    let (gamma, beta) = norm_params(d);
    let feed: [(&str, &[f32]); 5] = [
        ("x", &x),
        ("w1", &w1),
        ("w2", &w2),
        ("gamma", &gamma),
        ("beta", &beta),
    ];

    // CPU reference (deterministic oracle).
    let cpu = Session::new(Device::Cpu)
        .compile(hybrid_graph(m, d, hidden, layers, host_every))
        .run(&feed)
        .remove(0);

    // CUDA: run the SAME compiled session THREE times. Under
    // RLX_CUDA_SEGMENTED_CAPTURE=1 + RLX_CUDA_EXEC_MODE=graph these exercise, in
    // order: (1) the eager WARM-UP (loads kernel modules + cuBLAS workspace,
    // which can't be allocated mid-capture), (2) the CAPTURE run (records +
    // launches each per-segment graph), (3) the REPLAY run (launches the stored
    // graphs). Off the flag, all three are plain eager. All must match the CPU
    // oracle and each other bit-exactly.
    let mut cuda = Session::new(dev).compile(g);
    let o1 = cuda.run(&feed).remove(0);
    let o2 = cuda.run(&feed).remove(0);
    let o3 = cuda.run(&feed).remove(0);

    assert_eq!(o1.len(), m * d, "output shape");
    let max_vs_cpu = |o: &[f32]| {
        o.iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let (d1, d2, d3) = (max_vs_cpu(&o1), max_vs_cpu(&o2), max_vs_cpu(&o3));
    eprintln!("[probe] max|CUDA-CPU| warmup={d1:.3e} capture={d2:.3e} replay={d3:.3e}");
    // cuBLAS vs CPU BLAS accumulation differs; the runs just have to track the
    // oracle. The strong invariant is warmup == capture == replay (below).
    assert!(d1 < 1e-2, "CUDA warm-up run diverged from CPU: {d1:.3e}");
    assert!(d2 < 1e-2, "CUDA capture run diverged from CPU: {d2:.3e}");
    assert!(d3 < 1e-2, "CUDA replay run diverged from CPU: {d3:.3e}");
    // Segmented capture must be deterministic across warm-up, capture + replay.
    assert_eq!(o1, o2, "CUDA warm-up != capture (segmented determinism)");
    assert_eq!(o2, o3, "CUDA capture != replay (segmented determinism)");
}

/// WHOLE-GRAPH capture (`ExecMode::Graph`, no host steps → the whole schedule is
/// capture-safe). This is rlx's main graph-mode feature that silently ran eager
/// on the uncapturable null stream; the non-null-stream + disable_event_tracking
/// fix makes it actually capture. Run:
///   RLX_CUDA_EXEC_MODE=graph cargo test -p rlx-runtime --features cuda \
///     --test cuda_capture_probe cuda_whole_graph_capture -- --nocapture
#[test]
fn cuda_whole_graph_capture_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip cuda_whole_graph_capture ({dev:?} unavailable)");
        return;
    }
    // host_every > layers → no Sort → whole schedule is capture-safe.
    let layers: usize = std::env::var("RLX_WG_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let (m, d, hidden) = (64, 256, 1024);
    let mk = || hybrid_graph(m, d, hidden, layers, 1000);
    let x = seeded(m * d, 1);
    let w1 = seeded(d * hidden, 2);
    let w2 = seeded(hidden * d, 3);
    let (gamma, beta) = norm_params(d);
    let feed: [(&str, &[f32]); 5] = [
        ("x", &x),
        ("w1", &w1),
        ("w2", &w2),
        ("gamma", &gamma),
        ("beta", &beta),
    ];

    let cpu = Session::new(Device::Cpu).compile(mk()).run(&feed).remove(0);
    // warm-up (eager) → capture → replay under RLX_CUDA_EXEC_MODE=graph.
    let mut cuda = Session::new(dev).compile(mk());
    let o1 = cuda.run(&feed).remove(0);
    let o2 = cuda.run(&feed).remove(0);
    let o3 = cuda.run(&feed).remove(0);

    let maxd = |o: &[f32]| {
        o.iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let (d1, d2, d3) = (maxd(&o1), maxd(&o2), maxd(&o3));
    eprintln!("[whole-graph] max|CUDA-CPU| warmup={d1:.3e} capture={d2:.3e} replay={d3:.3e}");
    assert!(
        d1 < 1e-2 && d2 < 1e-2 && d3 < 1e-2,
        "whole-graph diverged from CPU"
    );
    assert_eq!(o1, o2, "warm-up != capture");
    assert_eq!(o2, o3, "capture != replay (whole-graph determinism)");
}
