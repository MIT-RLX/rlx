// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-flow` — mine repeated **dataflow** sub-DAGs (branching input cones)
//! and report them as decomposition / fusion candidates. Demonstrated on three
//! architectures whose repeated blocks the miner should recover:
//!   `mlp`         — residual `x + relu(x·W + b)`
//!   `transformer` — attention + FFN blocks (→ FusedAttentionBlock + SwiGLU)
//!   `moe`         — attention + top-k MoE expert blocks (→ grouped-matmul fuse)
//!
//! Usage: `opscope-flow [layers] [mlp|transformer|moe]`  (default: 6 mlp)

use rlx_opscope::dataflow::{decomposition_hint, repeated_flow_patterns};
use rlx_opscope::demo::build;

fn main() {
    let layers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let kind = std::env::args().nth(2).unwrap_or_else(|| "mlp".into());
    let g = build(&kind, layers);
    println!(
        "graph: {kind}, {layers} blocks, {} nodes\n",
        g.nodes().len()
    );

    let patterns = repeated_flow_patterns(&g, 2, 6, 2);
    println!(
        "{:>5} {:>4} {:>5}   repeated dataflow cone → decomposition candidate",
        "score", "×", "depth"
    );
    println!("{}", "-".repeat(100));
    for p in patterns.iter().take(12) {
        println!(
            "{:>5} {:>4} {:>5}   {}",
            p.score(),
            p.count,
            p.depth,
            p.tree
        );
        println!("{:>17}   {}", "", decomposition_hint(p));
    }
}
