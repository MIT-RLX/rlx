// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-graph` — run dataflow repeated-pattern mining on a graph **dumped
//! from another workspace** (e.g. a real rlx-models model). The dump is a plain
//! edge-list — one line per node: `idx op_name in0 in1 …` — so no shared build
//! or cross-workspace dependency is needed; the model repo just walks its graph
//! and prints it (see `rlx-vision-bench/examples/opscope_dump.rs`).
//!
//! Usage: `opscope-graph <dump.txt>`

use rlx_opscope::dataflow::{decomposition_hint, repeated_flow_patterns_on};

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: opscope-graph <dump.txt>");
    let text = std::fs::read_to_string(&path)?;

    let mut title = format!("graph {path}");
    let mut ops: Vec<String> = Vec::new();
    let mut inputs: Vec<Vec<usize>> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            title = rest.trim().to_string(); // header comment
            continue;
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        // idx op_name in0 in1 …  (idx == file order; used only as a sanity index)
        ops.push(tok[1].to_string());
        inputs.push(tok[2..].iter().filter_map(|s| s.parse().ok()).collect());
    }

    println!("{title}\n{} nodes\n", ops.len());
    let patterns = repeated_flow_patterns_on(&ops, &inputs, 2, 6, 2);
    println!(
        "{:>5} {:>4} {:>5}   repeated dataflow cone → decomposition candidate",
        "score", "×", "depth"
    );
    println!("{}", "-".repeat(100));
    for p in patterns.iter().take(14) {
        println!(
            "{:>5} {:>4} {:>5}   {}",
            p.score(),
            p.count,
            p.depth,
            p.tree
        );
        println!("{:>17}   {}", "", decomposition_hint(p));
    }
    Ok(())
}
