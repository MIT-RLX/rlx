// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;
use std::process::exit;

use anyhow::Result;
use rlx_onnx_import::coverage::op_is_supported;
use rlx_onnx_import::ops::format_bundle_category_report;
use rlx_onnx_import::{ImportOptions, build_hir_from_bundle, load_bundle};

fn main() -> Result<()> {
    let args = env::args().skip(1);
    let mut strict = false;
    let mut lower = false;
    let mut quantize_bundle = false;
    let mut bundle_dir: Option<PathBuf> = None;
    for arg in args {
        match arg.as_str() {
            "--strict" => strict = true,
            "--lower" => lower = true,
            "--quantize-bundle" => quantize_bundle = true,
            path if !path.starts_with('-') => bundle_dir = Some(path.into()),
            other => {
                eprintln!("unknown flag: {other}");
                eprintln!(
                    "usage: rlx-onnx-import-report [--strict] [--lower] [--quantize-bundle] <bundle-dir>"
                );
                exit(2);
            }
        }
    }
    let bundle_dir = bundle_dir.unwrap_or_else(|| {
        eprintln!(
            "usage: rlx-onnx-import-report [--strict] [--lower] [--quantize-bundle] <bundle-dir>"
        );
        exit(2);
    });

    let bundle = load_bundle(&bundle_dir)?;

    let mut supported = 0usize;
    let mut unsupported = 0usize;
    let mut remaining: Vec<_> = Vec::new();
    for (op, count) in &bundle.manifest.op_histogram {
        if op_is_supported(op) {
            supported += count;
        } else {
            unsupported += count;
            remaining.push((op.as_str(), *count));
        }
    }

    println!("bundle: {}", bundle_dir.display());
    println!("  source: {}", bundle.manifest.source_onnx);
    println!("  nodes: {}", bundle.manifest.node_count);
    println!("  initializers: {}", bundle.manifest.initializer_count);
    println!(
        "  registry coverage (by op count): {supported}/{} ({:.1}%)",
        supported + unsupported,
        100.0 * supported as f64 / (supported + unsupported) as f64
    );
    if unsupported > 0 {
        println!("  remaining op kinds:");
        remaining.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        for (op, c) in remaining.iter().take(25) {
            println!("    {op}: {c}");
        }
    }

    let hist: Vec<(String, usize)> = bundle
        .manifest
        .op_histogram
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    println!("  by category:");
    for line in format_bundle_category_report(&hist).lines() {
        println!("    {line}");
    }

    if lower {
        let opts = if quantize_bundle && !strict {
            ImportOptions::quant_bundle()
        } else {
            ImportOptions {
                strict,
                quantize_bundle_rewrites: quantize_bundle,
                ..ImportOptions::default()
            }
        };
        let (_hir, _params, _typed, report) = build_hir_from_bundle(&bundle, opts)?;
        println!(
            "  HIR lower: {} lowered, {} skipped, {} stubbed",
            report.lowered, report.skipped, report.stubbed
        );
        if !report.unsupported.is_empty() {
            println!("  unsupported: {:?}", report.unsupported);
        }
        if strict && (report.stubbed > 0 || !report.unsupported.is_empty()) {
            exit(1);
        }
    } else if strict && unsupported > 0 {
        exit(1);
    }

    Ok(())
}
