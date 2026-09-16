// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **What is the static-weight-pack skip worth on ROCm?**
//!
//! The matmul-fusion passes fuse Q/K/V and gate/up into one GEMM by emitting a
//! `Concat` over the weight `Param`s. ROCm lowers that to **one step per
//! input**, so on a 28-layer Llama it is 5 launches per layer — 140 per token —
//! plus ~1.9 GB of constant copying. Those steps are now materialised once and
//! skipped after; `RLX_STATIC_WEIGHT_PACK=0` opts out, which is the A/B here.
//!
//! Metal measures the same change at −15.2% of total GPU device time. ROCm
//! should do better, because it pays the cost in launches as well as bytes.
//!
//! Decode-shaped by construction: batch 1, **seq 1**, so every GEMM is a GEMV
//! and the per-token weight traffic is the whole story. Synthetic weights — the
//! graph SHAPE decides the split, not the values.
//!
//! ```sh
//! cargo run --release -p rlx-rocm --example rocm_static_pack_bench
//! cargo run --release -p rlx-rocm --example rocm_static_pack_bench -- --layers 28 --iters 50
//! ```
//!
//! Run it on an IDLE GPU. It reports wall time per iteration, which on a
//! contended device measures the contention. `rocm-smi --showuse` first.

fn main() {
    if !rlx_rocm::is_available() {
        println!("no ROCm device — nothing to measure");
        return;
    }
    use rlx_ir::{DType, Graph, GraphExt, Shape};
    use rlx_rocm::backend::RocmExecutable;

    const F: DType = DType::F32;
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    // Carbon-500M geometry: hidden 1024, 16 heads / 8 KV heads, d_ff 3072.
    let layers = arg("--layers", 28);
    let iters = arg("--iters", 50);
    let (h, kv, ff) = (1024usize, 512usize, 3072usize);

    // Per layer: `concat([q,k,v]) -> [h, h+2kv]` and `concat([gate,up]) -> [h, 2ff]`,
    // each feeding one GEMV. That is the exact pair the fusion passes build.
    let mut g = Graph::new("decode_packs");
    let x = g.input("x", Shape::new(&[1, h], F));
    let mut cur = x;
    let mut names: Vec<(String, usize)> = Vec::new();
    for l in 0..layers {
        let (qn, kn, vn) = (format!("q{l}"), format!("k{l}"), format!("v{l}"));
        let (gn, un, dn) = (format!("g{l}"), format!("u{l}"), format!("d{l}"));
        let q = g.param(&qn, Shape::new(&[h, h], F));
        let k = g.param(&kn, Shape::new(&[h, kv], F));
        let v = g.param(&vn, Shape::new(&[h, kv], F));
        let gate = g.param(&gn, Shape::new(&[h, ff], F));
        let up = g.param(&un, Shape::new(&[h, ff], F));
        let down = g.param(&dn, Shape::new(&[ff, h], F));
        names.extend([
            (qn, h * h),
            (kn, h * kv),
            (vn, h * kv),
            (gn, h * ff),
            (un, h * ff),
            (dn, ff * h),
        ]);

        let qkv = g.concat_(vec![q, k, v], 1);
        let proj = g.matmul(cur, qkv, Shape::new(&[1, h + 2 * kv], F));
        let head = g.narrow_(proj, 1, 0, h);

        let gu = g.concat_(vec![gate, up], 1);
        let mlp = g.matmul(head, gu, Shape::new(&[1, 2 * ff], F));
        let half = g.narrow_(mlp, 1, 0, ff);
        let out = g.matmul(half, down, Shape::new(&[1, h], F));
        cur = g.add(cur, out);
    }
    g.set_outputs(vec![cur]);

    let skip_on = rlx_ir::env::flag_or("RLX_STATIC_WEIGHT_PACK", true);
    println!("static weight packs on ROCm — {layers} layers, decode shape (seq=1)");
    println!("  skip: {}", if skip_on { "ON (default)" } else { "OFF" });

    let mut exe = RocmExecutable::compile(g);
    for (n, len) in &names {
        exe.set_param(n, &vec![0.0001f32; *len]);
    }
    let xv = vec![1.0f32; h];

    // Warm: compile, first-touch, and (when enabled) materialise the packs so
    // the timed loop is steady state rather than a mix of both regimes.
    for _ in 0..3 {
        let _ = exe.run(&[("x", &xv)]);
    }
    let (marked, armed) = exe.static_once_report();
    println!("  packs: {marked} steps marked, skip armed = {armed}");
    if skip_on && !armed {
        println!("  WARNING: skip did not arm — this arm is not measuring what it claims");
    }

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = exe.run(&[("x", &xv)]);
    }
    let per = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("  {per:.3} ms/iter over {iters} iters");
}
