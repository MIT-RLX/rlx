// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Train a tiny MNIST MLP, bake → `.rlx` / `.rlxp`, and bench load + infer.
//!
//! ```bash
//! just throttle
//! RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bake --example mnist_bench_rlxp \
//!   --features runtime --release
//! ```

#[path = "common/mnist.rs"]
mod mnist_common;

use anyhow::Result;
use mnist_common::*;
use rlx_bake::{BakeOptions, BakeProfile, bake, read_rlx, write_rlx, write_rlxp};
use rlx_ir::Tick;
use rlx_pkg::{ContainerKind, Package};
use rlx_runtime::{Device, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const ITERS: usize = 40;
const WARMUP: usize = 5;

fn mean_std(xs: &[Duration]) -> (Duration, Duration) {
    let n = xs.len().max(1) as f64;
    let mean_ns: f64 = xs.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / n;
    let var: f64 = xs
        .iter()
        .map(|d| {
            let x = d.as_nanos() as f64 - mean_ns;
            x * x
        })
        .sum::<f64>()
        / n;
    (
        Duration::from_nanos(mean_ns as u64),
        Duration::from_nanos(var.sqrt() as u64),
    )
}

fn bench(label: &str, mut f: impl FnMut()) {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let (m, s) = mean_std(&samples);
    println!("{label:<48} {:>10.2?}  ±{:>8.2?}", m, s);
}

fn fmt_bytes(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{:.2} MiB", n as f64 / (1 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KiB", n as f64 / (1 << 10) as f64)
    } else {
        format!("{n} B")
    }
}

fn main() -> Result<()> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/out");
    std::fs::create_dir_all(&out_dir)?;
    let path_rlx = out_dir.join("mnist_bench.rlx");
    let path_flat = out_dir.join("mnist_bench.rlxp");
    let path_zip = out_dir.join("mnist_bench.zip");

    let n = 4_096;
    let (images, labels) = match try_load_mnist(n) {
        Some(d) => {
            eprintln!("loaded {n} real MNIST train samples");
            d
        }
        None => {
            eprintln!("MNIST raw not found — synthetic ({n})");
            make_synthetic(n, 7)
        }
    };

    eprintln!("training MLP {IN}→{HIDDEN}→{OUT} (3 epochs) …");
    let t_train = Instant::now();
    let (weights, acc) = train_sgd(&images, &labels, n, 3, 0.08);
    eprintln!(
        "train accuracy ≈ {:.1}%  ({:?})",
        acc * 100.0,
        t_train.elapsed()
    );

    let graph = build_infer_graph();
    let mut bindings = HashMap::new();
    bindings.insert("w1".into(), weights.w1.clone());
    bindings.insert("b1".into(), weights.b1.clone());
    bindings.insert("w2".into(), weights.w2.clone());
    bindings.insert("b2".into(), weights.b2.clone());

    let opts = BakeOptions::from_profile(BakeProfile::Exact);
    let (file, report) = bake(&graph, &bindings, &opts);
    eprintln!(
        "bake: {} nodes → {}, {} weight bytes",
        graph.len(),
        report.nodes_after,
        report.weight_bytes
    );

    write_rlx(&path_rlx, &file)?;
    write_rlxp(&path_flat, &file, Some(ContainerKind::Flat))?;
    write_rlxp(&path_zip, &file, Some(ContainerKind::Zip))?;

    let sz_rlx = std::fs::metadata(&path_rlx)?.len();
    let sz_flat = std::fs::metadata(&path_flat)?.len();
    let sz_zip = std::fs::metadata(&path_zip)?.len();

    println!();
    println!("=== MNIST MLP → format sizes ===");
    println!("format                         size");
    println!("----------------------------------------");
    println!("RLXBAKE1 (.rlx)           {:>10}", fmt_bytes(sz_rlx));
    println!("RLXP flat (.rlxp)         {:>10}", fmt_bytes(sz_flat));
    println!("RLXP zip  (.zip)          {:>10}", fmt_bytes(sz_zip));
    println!();

    // One batch of real/synthetic images for infer.
    let x: Vec<f32> = images[..BATCH * IN].to_vec();
    let session = Session::new(Device::Cpu);

    println!("=== load / compile / run (CPU, {ITERS} iters) ===");
    println!("{:<48} {:>10}  {:>8}", "operation", "mean", "±σ");
    println!("{}", "-".repeat(70));

    bench("open Package (.rlxp flat)", || {
        let _ = Package::open(&path_flat).unwrap();
    });
    bench("open Package (.rlxp zip)", || {
        let _ = Package::open(&path_zip).unwrap();
    });
    bench("read_rlx (.rlx)", || {
        let _ = read_rlx(&path_rlx).unwrap();
    });

    bench("flat: graph()+materialize", || {
        let p = Package::open(&path_flat).unwrap();
        let _ = p.graph().unwrap();
    });
    bench("rlx: into_runtime_graph", || {
        let f = read_rlx(&path_rlx).unwrap();
        let _ = f.clone().into_runtime_graph().unwrap();
    });

    bench("flat: open+compile", || {
        let p = Package::open(&path_flat).unwrap();
        let g = p.graph().unwrap();
        let _ = session.compile(g);
    });
    bench("rlx: open+compile", || {
        let f = read_rlx(&path_rlx).unwrap();
        let g = f.into_runtime_graph().unwrap();
        let _ = session.compile(g);
    });

    // Held-open infer: compile once, time run.
    let pack = Package::open(&path_flat)?;
    let mut compiled_flat = session.compile(pack.graph()?);
    let file = read_rlx(&path_rlx)?;
    let mut compiled_rlx = session.compile(file.into_runtime_graph()?);

    // Touch: first run may allocate; warm then time.
    let _ = compiled_flat.run(&[("x", x.as_slice())]);
    let _ = compiled_rlx.run(&[("x", x.as_slice())]);

    bench("flat: run batch=32 (held compiled)", || {
        let _ = compiled_flat.run(&[("x", x.as_slice())]);
    });
    bench("rlx:  run batch=32 (held compiled)", || {
        let _ = compiled_rlx.run(&[("x", x.as_slice())]);
    });

    // Accuracy check on the train batch.
    let out = compiled_flat.run(&[("x", x.as_slice())]);
    let logits = &out[0];
    let mut correct = 0usize;
    for b in 0..BATCH {
        let row = &logits[b * OUT..(b + 1) * OUT];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == labels[b] {
            correct += 1;
        }
    }
    println!();
    println!(
        "infer check: {correct}/{BATCH} correct on first train batch ({:.0}%)",
        100.0 * correct as f64 / BATCH as f64
    );

    // Sub-ms path with Tick for open flat alone.
    let mut ns_samples = Vec::new();
    for _ in 0..ITERS {
        let t0 = Tick::now();
        let _ = Package::open(&path_flat).unwrap();
        ns_samples.push(Tick::now().elapsed_ns(t0));
    }
    let mean_ns: f64 = ns_samples.iter().sum::<u64>() as f64 / ns_samples.len() as f64;
    println!("Tick open flat: {:.1} µs mean", mean_ns / 1000.0);

    println!();
    println!("artifacts:");
    println!("  {}", path_rlx.display());
    println!("  {}", path_flat.display());
    println!("  {}", path_zip.display());
    Ok(())
}
