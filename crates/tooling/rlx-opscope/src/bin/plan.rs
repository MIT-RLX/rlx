// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-plan` — Tier 4 actuation. Turns a mined sketch CSV into a **kernel
//! plan**: per site, the chosen exploit, the synthesized **runtime guard**
//! predicate that gates dispatch (with fallback), the guard's **stability**
//! (cross-step variance → false-positive risk), the **estimated** FLOP
//! reduction, and a **measured** A/B micro-bench (dense vs zero-skip) at the
//! observed density so the plan reports real, not theoretical, speedups.
//!
//! Usage: `opscope-plan <mined.csv>`

use std::collections::HashMap;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

/// Time an m×k×n matmul that consumes only `keep` fraction of the K dim — the
/// compute a zero-skipping kernel would actually do. Min of a few reps (ns).
fn bench_matmul(m: usize, k: usize, n: usize, keep: f32) -> f64 {
    let kk = ((k as f32 * keep).ceil() as usize).max(1).min(k);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0f32; m * n];
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..kk {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        black_box(&c);
        best = best.min(t.elapsed().as_nanos() as f64);
    }
    best
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}
fn std_(v: &[f32]) -> f32 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: opscope-plan <mined.csv>");
    let f = std::fs::File::open(&path)?;
    // (dist,site,role) → per-(run,step) density ; and l1 series for stationarity.
    let mut dens: HashMap<(String, String, String), Vec<f32>> = HashMap::new();
    let mut l1: HashMap<(String, String, String), Vec<f32>> = HashMap::new();

    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if i == 0 || line.is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != 13 {
            continue;
        }
        let numel: f32 = c[7].parse().unwrap_or(1.0);
        let key = (c[3].to_string(), c[8].to_string(), c[9].to_string()); // (dist,site,role)
        let val: f32 = c[12].parse().unwrap_or(0.0);
        match c[10] {
            "nnz" => dens.entry(key).or_default().push(val / numel.max(1.0)),
            "l1" => l1.entry(key).or_default().push(val),
            _ => {}
        }
    }

    println!("Kernel plan from {path}\n");
    println!(
        "{:<24} {:<26} {:>8} {:>8}   guard (stability)",
        "dist/site/role", "exploit", "est×", "meas×"
    );
    println!("{}", "-".repeat(104));

    let mut keys: Vec<_> = dens.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let d = mean(&dens[&key]);
        let dstd = std_(&dens[&key]);
        let (dist, site, role) = &key;
        if d < 0.5 {
            // Sparse exploit → guard on live density, measured zero-skip speedup.
            let est = 1.0 / d.max(1e-3);
            let meas = bench_matmul(128, 512, 128, 1.0) / bench_matmul(128, 512, 128, d);
            let stab = if dstd < 0.02 { "stable" } else { "watch drift" };
            println!(
                "{:<24} {:<26} {:>8.2} {:>8.2}   nnz/numel < 0.5  (σ={dstd:.3}, {stab})",
                format!("{dist}/{site}/{role}"),
                format!("sparse-GEMM ({:.0}% zeros)", (1.0 - d) * 100.0),
                est,
                meas,
            );
        }
    }

    // Weight stationarity → prepack once (a plan item, not a per-call guard).
    println!("\nStationary weights (prepack once, skip re-profiling):");
    let mut any = false;
    for (key, series) in &l1 {
        if key.2 == "rhs" && std_(series) / mean(series).max(1e-6) < 1e-4 {
            println!(
                "  {}/{}/{}: l1 constant across steps → prepack",
                key.0, key.1, key.2
            );
            any = true;
        }
    }
    if !any {
        println!("  (none detected — need a multi-step CSV from opscope-seq/mnist)");
    }
    Ok(())
}
