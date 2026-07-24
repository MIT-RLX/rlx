// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Wall-clock timings for hybrid RLXP load paths.
//!
//! ```text
//! cargo run -p rlx-pkg --example bench_hybrid --release
//! ```

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{
    ContainerKind, Package, PackedWeight, StorageTier, WriteOptions, write_package,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

const HOT_BYTES: usize = 64 << 20; // 64 MiB
const WARM_BYTES: usize = 64 << 20;
const WARM_BLOCK: u32 = 1 << 20; // 1 MiB
const ITERS: u32 = 20;

fn tiny_graph() -> Graph {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("bench_hybrid");
    let x = g.input("x", s.clone());
    let w = g.param("w_hot", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);
    g
}

fn patterned(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

fn warm_payload(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for i in (0..len).step_by(64) {
        v[i] = (i % 255) as u8;
    }
    v
}

fn mean_std(samples: &[Duration]) -> (Duration, Duration) {
    let n = samples.len() as f64;
    let mean_ns: f64 = samples.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / n;
    let var = samples
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

fn time_it(iters: u32, mut f: impl FnMut()) -> (Duration, Duration) {
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..3 {
        f();
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    mean_std(&samples)
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 {
        format!("{us:.1} µs")
    } else {
        format!("{:.2} ms", us / 1000.0)
    }
}

fn fmt_mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn checksum_touch(buf: &[u8]) -> u64 {
    let mut acc = 0u64;
    let step = 4096usize;
    let mut i = 0usize;
    while i < buf.len() {
        acc = acc.wrapping_add(buf[i] as u64);
        i += step;
    }
    if let Some(&b) = buf.last() {
        acc = acc.wrapping_add(b as u64);
    }
    acc
}

fn row(name: &str, mean: Duration, std: Duration) {
    println!("{:<36} {:>12} {:>12}", name, fmt_dur(mean), fmt_dur(std));
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let flat_path = dir.path().join("hybrid.rlxp");
    let zip_path = dir.path().join("hybrid.zip");

    let hot = patterned(HOT_BYTES, 7);
    let warm = warm_payload(WARM_BYTES);
    let mut tok = String::from("{\"vocab\":[");
    for i in 0..50_000 {
        tok.push_str(&format!("\"t{i}\","));
    }
    tok.push_str("\"end\"]}");
    let tok = tok.into_bytes();

    let weights = [
        PackedWeight {
            name: "w_hot".into(),
            shape: vec![HOT_BYTES / 4],
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: hot,
            rank: None,
            tier: StorageTier::Hot,
        },
        PackedWeight {
            name: "w_warm".into(),
            shape: vec![WARM_BYTES],
            scheme: "u8".into(),
            layout: "row_major".into(),
            data: warm,
            rank: None,
            tier: StorageTier::Warm,
        },
    ];
    let g = tiny_graph();
    let mut opts = WriteOptions {
        container: ContainerKind::Flat,
        warm_block_size: WARM_BLOCK,
        compress_sidecars: true,
        ..WriteOptions::default()
    };
    opts.sidecars
        .push(("tokenizer".into(), "application/json".into(), tok.clone()));

    eprintln!("writing flat hybrid pack…");
    let t0 = Instant::now();
    write_package(&flat_path, &g, &weights, &opts).expect("write flat");
    let write_flat = t0.elapsed();

    let mut zip_weights = weights.clone();
    zip_weights[1].tier = StorageTier::Hot;
    opts.container = ContainerKind::Zip;
    eprintln!("writing zip pack (both tensors STORE)…");
    let t0 = Instant::now();
    write_package(&zip_path, &g, &zip_weights, &opts).expect("write zip");
    let write_zip = t0.elapsed();

    let flat_sz = std::fs::metadata(&flat_path).unwrap().len();
    let zip_sz = std::fs::metadata(&zip_path).unwrap().len();
    let pack = Package::open(&flat_path).unwrap();
    let hot_stored = pack.weight_entry("w_hot").unwrap().length;
    let warm_stored = pack.weight_entry("w_warm").unwrap().length;

    println!();
    println!("=== RLXP hybrid pack bench ===");
    println!(
        "payload: hot raw {} | warm raw {}",
        fmt_mib(HOT_BYTES as u64),
        fmt_mib(WARM_BYTES as u64)
    );
    println!(
        "files:   flat {} | zip {} | warm on-disk {} ({:.2}× smaller)",
        fmt_mib(flat_sz),
        fmt_mib(zip_sz),
        fmt_mib(warm_stored),
        WARM_BYTES as f64 / warm_stored as f64,
    );
    println!(
        "write:   flat {} | zip {}",
        fmt_dur(write_flat),
        fmt_dur(write_zip)
    );
    println!();
    println!("{:<36} {:>12} {:>12}", "operation", "mean", "±σ");
    println!("{}", "-".repeat(62));

    let path = flat_path.clone();
    let (m, s) = time_it(ITERS, || {
        let _p = Package::open(&path).unwrap();
    });
    row("open flat (TOC+mmap)", m, s);

    let path_zip = zip_path.clone();
    let (m, s) = time_it(ITERS, || {
        let _p = Package::open(&path_zip).unwrap();
    });
    row("open zip (CD+mmap)", m, s);

    let pack = Package::open(&flat_path).unwrap();
    let (m, s) = time_it(ITERS, || {
        let buf = pack.tensor_mmap("w_hot").unwrap();
        black_box(checksum_touch(buf));
    });
    row("hot mmap + touch pages (64MiB)", m, s);

    let (m, s) = time_it(ITERS.min(10), || {
        let buf = pack.tensor_bytes("w_warm").unwrap();
        black_box(checksum_touch(&buf));
    });
    row("warm inflate all (64MiB raw)", m, s);

    let (m, s) = time_it(ITERS, || {
        let buf = pack.tensor_warm_block("w_warm", 0).unwrap();
        black_box(checksum_touch(&buf));
    });
    row("warm inflate 1 block (1MiB)", m, s);

    let (m, s) = time_it(ITERS, || {
        let buf = pack.sidecar("tokenizer").unwrap();
        black_box(checksum_touch(&buf));
    });
    row("cold sidecar zstd inflate", m, s);

    let (m, s) = time_it(ITERS, || {
        let _g = pack.graph().unwrap();
    });
    row("graph() deserialize+materialize", m, s);

    println!();
    println!(
        "stored lengths: hot {} | warm {}",
        fmt_mib(hot_stored),
        fmt_mib(warm_stored)
    );
}
