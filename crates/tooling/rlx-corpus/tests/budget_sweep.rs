// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the corpus costs, per architecture.
//!
//! The rest of the corpus asks whether a compiler change kept the answers
//! right. This asks what the machine spends to produce them — arena bytes,
//! host-boundary traffic, and the precision churn of a graph that is already
//! correct. `cargo test -p rlx-corpus --test budget_sweep -- --nocapture`
//! prints the table.

use rlx_opt::rlx_compile::fusion_pipeline::FusionTarget;
use rlx_runtime::check::{CheckOptions, check_graph};

const TARGETS: [FusionTarget; 4] = [
    FusionTarget::Cpu,
    FusionTarget::Metal,
    FusionTarget::Cuda,
    FusionTarget::Wgpu,
];

fn kb(n: usize) -> String {
    if n >= 1 << 20 {
        format!("{:.1}M", n as f64 / (1 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.0}K", n as f64 / (1 << 10) as f64)
    } else {
        format!("{n}")
    }
}

#[test]
fn budget_across_the_corpus() {
    let mut worst_width: Vec<(String, usize, usize)> = Vec::new();
    let mut off_path: Vec<(String, &'static str, usize, usize)> = Vec::new();

    let mut traffic: Vec<(String, usize, usize, usize)> = Vec::new();

    println!(
        "\n{:<38} {:>9} {:>8} {:>8} {:>5} {:>9} {:>6}",
        "case / backend", "arena", "params", "host-io", "disp", "moves", "x-arena"
    );
    for case in rlx_corpus::cases() {
        for t in TARGETS {
            let opts = CheckOptions {
                backends: vec![t],
                ..Default::default()
            };
            let report = check_graph(&case.graph, &opts);
            let Some(b) = report.backends.first().and_then(|s| s.budget.clone()) else {
                continue;
            };
            let label = format!("{}/{} [{:?}]", case.family, case.name, t);
            if b.width_overhead_bytes > 0 {
                worst_width.push((label.clone(), b.width_overhead_bytes, b.arena_bytes));
            }
            if b.off_fast_path_ops > 0 {
                off_path.push((
                    format!("{}/{}", case.family, case.name),
                    b.width_policy,
                    b.off_fast_path_ops,
                    b.off_fast_path_bytes,
                ));
            }
            let ratio = b.dram_bytes as f64 / b.arena_bytes.max(1) as f64;
            traffic.push((label.clone(), b.dispatches, b.dram_bytes, b.arena_bytes));
            println!(
                "{label:<38} {:>9} {:>8} {:>8} {:>5} {:>9} {:>5.1}x",
                kb(b.arena_bytes),
                kb(b.param_bytes),
                kb(b.host_io_bytes),
                b.dispatches,
                kb(b.dram_bytes),
                ratio,
            );
        }
    }

    worst_width.sort_by_key(|(_, o, _)| std::cmp::Reverse(*o));
    println!("\n-- largest width-policy overheads --");
    for (label, over, arena) in worst_width.iter().take(10) {
        println!(
            "  {label:<40} +{} of {} ({:.0}%)",
            kb(*over),
            kb(*arena),
            100.0 * *over as f64 / (*arena).max(1) as f64
        );
    }
    if worst_width.is_empty() {
        println!("  (none — the corpus is entirely f32, where width policy is free)");
    }

    println!("\n-- most memory traffic per byte of arena --");
    traffic.sort_by(|a, b| {
        let ra = a.2 as f64 / a.3.max(1) as f64;
        let rb = b.2 as f64 / b.3.max(1) as f64;
        rb.partial_cmp(&ra).unwrap()
    });
    for (label, disp, dram, arena) in traffic.iter().take(8) {
        println!(
            "  {label:<42} {disp:>3} disp  {:>8} over {:>8} arena  ({:.1}x)",
            kb(*dram),
            kb(*arena),
            *dram as f64 / (*arena).max(1) as f64
        );
    }

    println!("\n-- ops off the native fast path --");
    off_path.sort_by_key(|(_, _, _, bytes)| std::cmp::Reverse(*bytes));
    for (label, policy, ops, bytes) in off_path.iter().take(10) {
        println!("  {label:<40} {policy:<12} {ops} ops / {}", kb(*bytes));
    }
    if off_path.is_empty() {
        println!("  (none)");
    }
}
