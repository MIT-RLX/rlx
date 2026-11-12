// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Train e-means (arXiv:2611.09501) on a raw little-endian f32 file.
//!
//! ```sh
//! cargo run --release -p rlx-vq --features emeans --example emeans_train -- \
//!     data.f32 <d> <k> [nepochs=5] [seed=1234] [centroids_out.f32]
//! ```
//!
//! Prints the recipe and one line per epoch, then writes the centroids and
//! last-epoch labels (`<out>.labels.u32`) when an output path is given.
//! `scripts/emeans_faiss_reference.py` compares this against faiss.

use std::time::Instant;

use rlx_vq::emeans::{Emeans, EmeansConfig};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: emeans_train <data.f32> <d> <k> [nepochs] [seed] [out.f32]");
        std::process::exit(2);
    }
    let bytes = std::fs::read(&args[0]).expect("read data");
    let d: usize = args[1].parse().expect("d");
    let k: usize = args[2].parse().expect("k");
    let nepochs: f64 = args.get(3).map_or(5.0, |s| s.parse().expect("nepochs"));
    let seed: u32 = args.get(4).map_or(1234, |s| s.parse().expect("seed"));
    let x: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let n = x.len() / d;

    let cfg = EmeansConfig {
        nepochs,
        seed,
        ..EmeansConfig::default()
    };
    let mut em = Emeans::new(d, k, cfg);
    let start = Instant::now();
    let (centroids, labels) = em.train(&x, None);
    let secs = start.elapsed().as_secs_f64();
    let r = em.recipe().unwrap();
    println!(
        "n={n} d={d} k={k} batch={} alpha0={:.6} steps={} law={:?}",
        r.batch_size, r.alpha0, r.total_steps, r.batch_law
    );
    for (i, s) in em.epoch_stats().iter().enumerate() {
        println!(
            "epoch {i}: obj={:.6e} imbalance={:.4} nsplit={}",
            s.objective, s.imbalance_factor, s.nsplit
        );
    }
    let (dist, _) = em.assign(&x);
    let mse = dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    println!(
        "final mse={mse:.6e} splits={} time={secs:.3}s",
        em.n_splits()
    );

    if let Some(out) = args.get(5) {
        let c: Vec<u8> = centroids.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, c).expect("write centroids");
        if let Some(l) = labels {
            let l: Vec<u8> = l.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out}.labels.u32"), l).expect("write labels");
        }
    }
}
