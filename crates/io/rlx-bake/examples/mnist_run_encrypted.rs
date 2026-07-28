// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load an encrypted baked MNIST `*.rlx` and run inference.
//!
//! Password comes **only** from `RLX_BAKE_PASSWORD` (no CLI secret). The
//! artifact already embeds optimized weights (ternary / quant / folded
//! constants) — no separate weights file and no `set_param`.
//!
//! ```bash
//! export RLX_BAKE_PASSWORD='demo-secret'
//! cargo run -p rlx-bake --example mnist_run_encrypted --features encrypt,runtime
//! ```

#[path = "common/mnist.rs"]
mod mnist_common;

use anyhow::{Context, Result, bail};
use mnist_common::*;
use rlx_bake::{is_encrypted, read_rlx_with_password};
use rlx_runtime::{Device, Session};
use std::fs;

fn main() -> Result<()> {
    let password = password_from_env()?;
    let path = default_out_path();
    if !path.is_file() {
        bail!(
            "missing {} — run mnist_train_bake first (same RLX_BAKE_PASSWORD)",
            path.display()
        );
    }

    let raw = fs::read(&path)?;
    anyhow::ensure!(
        is_encrypted(&raw),
        "{} is not an encrypted RLXENC01 artifact",
        path.display()
    );
    eprintln!(
        "loading encrypted {} ({} bytes) with RLX_BAKE_PASSWORD …",
        path.display(),
        raw.len()
    );

    let file = read_rlx_with_password(&path, &password).context("decrypt / parse *.rlx")?;
    eprintln!(
        "decrypted: graph {:?}  nodes={}  weights={} ({} bytes in table)",
        file.meta.name,
        file.graph.len(),
        file.meta.weight_count,
        file.meta.weight_bytes
    );
    eprintln!(
        "  bake stats: skip={} ternary={} quant={}",
        file.meta.skipped_zero_matmuls, file.meta.ternary_packed, file.meta.quant_packed
    );
    for w in &file.weights {
        eprintln!(
            "  weight {} encoding={} {}B",
            w.name,
            w.encoding,
            w.data.len()
        );
    }

    // Compact memory mode: table holds bytes; fill Constants before compile.
    let graph = file
        .into_runtime_graph()
        .context("materialize compact weights")?;
    let mut compiled = Session::new(Device::Cpu).compile(graph);
    eprintln!("compiled on CPU (no weight sidecar, no set_param)");

    let (images, labels) = match try_load_mnist(BATCH) {
        Some(d) => d,
        None => make_synthetic(BATCH, 7),
    };
    let x = &images[..BATCH * IN];
    let outs = compiled.run(&[("x", x)]);
    let logits = &outs[0];

    let mut correct = 0usize;
    for b in 0..BATCH {
        let row = &logits[b * OUT..(b + 1) * OUT];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .unwrap()
            .0;
        if pred == labels[b] {
            correct += 1;
        }
    }
    eprintln!(
        "inference batch={BATCH}: {correct}/{BATCH} correct ({:.0}%)",
        100.0 * correct as f32 / BATCH as f32
    );
    eprintln!("ok — encrypted optimized *.rlx ran end-to-end");
    Ok(())
}
