// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Import-feasibility probe: import an arbitrary ONNX file at a fixed sequence
// length and print the lowering report (lowered / stubbed / unsupported).
//
//   cargo run -q -p rlx-onnx-import --example import_probe -- <file.onnx> [seq]

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: import_probe <file.onnx> [seq]"));
    let seq: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);

    // Optional distinct second length (e.g. ChatterBox decoder `feature_dim` =
    // the reference-mel length, independent of `num_speech_tokens`).
    let feature_dim: usize = std::env::var("RLX_PROBE_FEATURE_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(seq);
    let mut named = HashMap::new();
    named.insert("sequence_length".to_string(), seq);
    named.insert("total_sequence_length".to_string(), seq);
    named.insert("past_sequence_length".to_string(), 0);
    named.insert("batch_size".to_string(), 1);
    named.insert("num_speech_tokens".to_string(), seq);
    named.insert("feature_dim".to_string(), feature_dim);

    let dynamic = std::env::var_os("RLX_PROBE_DYNAMIC").is_some();
    let opts = ImportOptions {
        sequence_length: seq,
        named_lengths: named,
        strict: false,
        dynamic_sequence: dynamic,
        ..Default::default()
    };
    eprintln!("[probe] dynamic_sequence={dynamic}");

    eprintln!("[probe] importing {} at seq={seq}", path.display());
    let (hir, params, report, manifest) = build_hir_from_onnx_file(&path, opts)?;
    println!("=== import report ===");
    println!("  lowered   : {}", report.lowered);
    println!("  skipped   : {}", report.skipped);
    println!("  stubbed   : {}", report.stubbed);
    if !report.unsupported.is_empty() {
        println!("  UNSUPPORTED ops:");
        let mut v: Vec<_> = report.unsupported.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        for (op, c) in v {
            println!("    {c:4} {op}");
        }
    }
    if !report.stubbed_nodes.is_empty() {
        println!("  stubbed nodes (first {}):", report.stubbed_nodes.len());
        for n in report.stubbed_nodes.iter().take(20) {
            println!("    {n}");
        }
    }
    println!("  HIR nodes : {}", hir.nodes().len());
    println!("  params    : {}", params.len());
    println!("  graph outputs: {}", manifest.outputs.len());

    // Lower HIR → graph and histogram the native ops (proves the fused-op
    // decompositions produced the expected primitives — Attention / Rope /
    // RmsNorm for GroupQueryAttention + Skip/SimplifiedLayerNormalization).
    match rlx_ir::hir_to_graph(hir) {
        Ok(graph) => {
            let mut hist: HashMap<String, usize> = HashMap::new();
            for n in graph.nodes() {
                *hist.entry(format!("{:?}", n.op.kind())).or_insert(0) += 1;
            }
            let mut v: Vec<_> = hist.into_iter().collect();
            v.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            println!("=== native op histogram (hir_to_graph) ===");
            for (op, c) in v.iter().take(24) {
                println!("  {c:4} {op}");
            }
        }
        Err(e) => println!("  hir_to_graph FAILED: {e}"),
    }
    Ok(())
}
