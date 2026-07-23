// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Train a tiny MNIST MLP, then **bake + optimize + encrypt** into one `*.rlx`.
//!
//! The point of bake is not “zip model.json + weights.bin”. It is to produce an
//! already-optimized deploy artifact: params specialized, weight-only work
//! folded, zero matmuls skipped, ternary weights packed (TQ2_0), remaining
//! dense weights quantized (Q8_0) — **fewer ops and fewer bytes to load**, then
//! sealed with a password.
//!
//! ```bash
//! export RLX_BAKE_PASSWORD='demo-secret'
//! cargo run -p rlx-bake --example mnist_train_bake --features encrypt,runtime
//! # writes examples/out/mnist.rlx (or $RLX_BAKE_OUT)
//! ```

#[path = "common/mnist.rs"]
mod mnist_common;

use anyhow::{Context, Result};
use mnist_common::*;
use rlx_bake::{BakeOptions, BakeProfile, bake, write_rlx_encrypted};
use std::collections::HashMap;

fn main() -> Result<()> {
    let password = password_from_env()?;
    let out = default_out_path();
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let n = 8_192;
    let (images, labels) = match try_load_mnist(n) {
        Some(d) => {
            eprintln!("loaded {n} real MNIST train samples");
            d
        }
        None => {
            eprintln!("MNIST raw not found — using synthetic digit-ish data ({n} samples)");
            make_synthetic(n, 42)
        }
    };

    eprintln!("training MLP {IN}→{HIDDEN}→{OUT} …");
    let (mut weights, acc) = train_sgd(&images, &labels, n, /*epochs*/ 6, 0.05);
    eprintln!("train accuracy ≈ {:.1}%", acc * 100.0);

    // Deploy-time weight transform: first layer → exact ternary {−1,0,+1}.
    // Bake will pack it as GGUF TQ2_0 + DequantMatMul (add/sub/skip, no dense mul).
    let w1_bytes_f32 = weights.w1.len() * 4;
    weights.w1 = ternarize(&weights.w1);
    let ternary_nz = weights.w1.iter().filter(|&&v| v != 0.0).count();
    eprintln!(
        "ternarized w1: {}/{} non-zero (±1), rest exact 0 (eligible for skip + TQ2 pack)",
        ternary_nz,
        weights.w1.len()
    );
    eprintln!("fine-tuning head with frozen ternary w1 …");
    let (weights, acc2) = train_sgd_ex(&images, &labels, n, 4, 0.1, Some(weights));
    eprintln!("post-ternary accuracy ≈ {:.1}%", acc2 * 100.0);

    let graph = build_infer_graph();
    let mut bindings = HashMap::new();
    bindings.insert("w1".into(), weights.w1.clone());
    bindings.insert("b1".into(), weights.b1.clone());
    bindings.insert("w2".into(), weights.w2.clone());
    bindings.insert("b2".into(), weights.b2.clone());

    let nodes_before = graph.len();
    let f32_weight_bytes = weights.f32_bytes();

    // `size` = lossless skip/ternary + Q8_0 for remaining dense matmuls (w2).
    let opts = BakeOptions::from_profile(BakeProfile::Size);

    let (file, report) = bake(&graph, &bindings, &opts);

    eprintln!();
    eprintln!(
        "── bake optimization [{}] ({}) ──",
        opts.profile,
        opts.profile.description()
    );
    eprintln!("  graph nodes:     {nodes_before} → {}", report.nodes_after);
    eprintln!(
        "  weight table:    {} tensors, {} bytes (was ~{} bytes raw f32)",
        report.weight_count, report.weight_bytes, f32_weight_bytes
    );
    eprintln!(
        "  skipped zero MM: {}",
        report.optimize.skipped_zero_matmuls
    );
    eprintln!("  ternary packed:  {}", report.optimize.ternary_packed);
    eprintln!("  quant packed:    {}", report.optimize.quant_packed);
    eprintln!(
        "  memory mode:     {} (stripped graph {}B, table {}B, deduped consts {}, dropped folded {})",
        report.memory.mode,
        report.memory.graph_bytes_stripped,
        report.memory.table_bytes_stripped,
        report.memory.constants_deduped,
        report.memory.folded_bindings_dropped
    );
    eprintln!(
        "  w1 f32 alone was {w1_bytes_f32} bytes; after ternarize+TQ2 the artifact stores packed trits"
    );
    eprintln!(
        "  → less data to load, fewer runtime muls (DequantMatMul / skipped zeros), constants fused in"
    );
    for w in &file.weights {
        eprintln!(
            "    • {}  {}  shape={:?}  {}B  ({})",
            w.name,
            w.encoding,
            w.shape,
            w.data.len(),
            w.note
        );
    }

    write_rlx_encrypted(&out, &file, &password).context("writing encrypted *.rlx")?;
    eprintln!();
    eprintln!(
        "wrote encrypted artifact {} ({} bytes) — password from RLX_BAKE_PASSWORD",
        out.display(),
        std::fs::metadata(&out)?.len()
    );
    eprintln!(
        "next: cargo run -p rlx-bake --example mnist_run_encrypted --features encrypt,runtime"
    );
    Ok(())
}
