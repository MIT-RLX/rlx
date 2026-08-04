// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-shapes` — Tier 3 workload mining: per-op FLOPs / bytes / arithmetic
//! intensity → roofline classification + hot-GEMM-shape histogram (the shapes a
//! specialized/autotuned kernel + dispatch table should target).
//!
//! Usage: `opscope-shapes [layers] [mlp|transformer|moe]`  (default: 6 transformer)

use rlx_opscope::demo::build;
use rlx_opscope::shapes::{DEFAULT_RIDGE, gemm_shape_histogram, op_costs, roofline_class};

fn main() {
    let layers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let kind = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "transformer".into());
    let g = build(&kind, layers);
    let costs = op_costs(&g);

    let total_flops: u64 = costs.iter().map(|c| c.flops).sum();
    let total_bytes: u64 = costs.iter().map(|c| c.bytes).sum();
    println!(
        "graph: {kind}, {layers} blocks — {:.2} GFLOP, {:.2} MB, overall {:.1} FLOP/byte (ridge {DEFAULT_RIDGE})\n",
        total_flops as f64 / 1e9,
        total_bytes as f64 / 1e6,
        total_flops as f64 / total_bytes.max(1) as f64,
    );

    // Roofline breakdown by op kind.
    println!("Roofline (share of FLOPs by boundedness):");
    let mut mem = 0u64;
    let mut comp = 0u64;
    for c in &costs {
        match roofline_class(c, DEFAULT_RIDGE) {
            "memory-bound" => mem += c.flops,
            "compute-bound" => comp += c.flops,
            _ => {}
        }
    }
    let t = (mem + comp).max(1) as f64;
    println!(
        "  compute-bound: {:>5.1}%   (fuse epilogues, tile for cache)",
        comp as f64 / t * 100.0
    );
    println!(
        "  memory-bound : {:>5.1}%   (aggressive fusion — the win is bytes)\n",
        mem as f64 / t * 100.0
    );

    // Hot GEMM shapes → specialize/autotune these.
    println!("Hot GEMM shapes (M×K×N → count, %FLOPs) — dispatch-table candidates:");
    let hist = gemm_shape_histogram(&costs);
    let gemm_flops: u64 = hist.iter().map(|(_, (_, f))| *f).sum::<u64>().max(1);
    for ((m, k, n), (count, flops)) in hist.iter().take(10) {
        println!(
            "  {m:>4}×{k:<4}×{n:<4}  ×{count:<3}  {:>5.1}%   ({:.2} GFLOP)",
            *flops as f64 / gemm_flops as f64 * 100.0,
            *flops as f64 / 1e9
        );
    }
}
