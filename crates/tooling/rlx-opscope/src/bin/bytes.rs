// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-bytes` — print the static memory-traffic ledger for a demo graph.
//!
//! Answers "how many bytes does this graph *have* to move, and how much of that
//! is weights?" without running anything. See [`rlx_opscope::bytes`] for what the
//! number does and does not include — it is a lower bound and a diagnostic, not
//! an optimizer input.
//!
//! Usage: `opscope-bytes [kind] [layers] [--ridge <flop/byte>]`
//!
//! `kind` is any graph [`rlx_opscope::demo::build`] knows. The ridge point
//! defaults to 100 flop/byte, roughly an f32 GPU (e.g. ~40 TFLOP/s over
//! ~400 GB/s); pass the ratio for the device you care about.

use rlx_opscope::bytes::{analyze, human_bytes};
use rlx_opscope::demo;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut ridge = 100.0f64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ridge" => {
                ridge = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("--ridge needs a number"));
            }
            _ => positional.push(a.clone()),
        }
    }
    let kind = positional
        .first()
        .map(String::as_str)
        .unwrap_or("transformer");
    let layers: usize = positional.get(1).and_then(|v| v.parse().ok()).unwrap_or(4);

    let graph = demo::build(kind, layers);
    let cost = analyze(&graph);

    println!("graph: {kind} x{layers} ({} nodes)\n", graph.nodes().len());
    println!("{}", cost.report(12));

    let (mem, comp) = cost.roofline_split(ridge);
    let total = mem + comp;
    if total > 0 {
        println!(
            "roofline @ {ridge:.0} flop/byte: {} memory-bound ({:.1}%), {} compute-bound ({:.1}%)",
            human_bytes(mem),
            100.0 * mem as f64 / total as f64,
            human_bytes(comp),
            100.0 * comp as f64 / total as f64,
        );
    }
}
