// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Head-to-head load timing: RLXP flat vs GGUF vs safetensors vs ZIP(DDUF-like).
//!
//! ```text
//! cargo run -p rlx-pkg --example bench_compare --release
//! ```
//!
//! ONNX is summarized qualitatively (protobuf graph parse; not a mmap weight
//! container in the same sense).

use rlx_gguf::{GgmlType, GgufFile, GgufWriter};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{
    ContainerKind, Package, PackedWeight, StorageTier, WriteOptions, write_package,
};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const N_BYTES: usize = 64 << 20; // 64 MiB f32-ish payload
const ITERS: u32 = 25;

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
    for _ in 0..3 {
        f();
    }
    let mut samples = Vec::with_capacity(iters as usize);
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

fn fmt_mib(b: u64) -> String {
    format!("{:.2} MiB", b as f64 / (1024.0 * 1024.0))
}

fn touch(buf: &[u8]) -> u64 {
    let mut a = 0u64;
    for i in (0..buf.len()).step_by(4096) {
        a = a.wrapping_add(buf[i] as u64);
    }
    if let Some(&x) = buf.last() {
        a = a.wrapping_add(x as u64);
    }
    a
}

fn write_safetensors(path: &std::path::Path, name: &str, data: &[u8]) {
    // Minimal safetensors: [u64 header_len][JSON][data]
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len() / 4,
        data.len()
    );
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    std::fs::write(path, out).unwrap();
}

fn open_safetensors_mmap(path: &std::path::Path) -> (memmap2::Mmap, usize, usize) {
    let f = std::fs::File::open(path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
    let data_start = 8 + hlen;
    let header: serde_json::Value = serde_json::from_slice(&map[8..data_start]).unwrap();
    let off = &header["w"]["data_offsets"];
    let begin = off[0].as_u64().unwrap() as usize;
    let end = off[1].as_u64().unwrap() as usize;
    (map, data_start + begin, data_start + end)
}

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let payload: Vec<u8> = (0..N_BYTES).map(|i| (i % 251) as u8).collect();

    // --- RLXP flat ---
    let rlxp = dir.path().join("w.rlxp");
    let g = {
        let s = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("cmp");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);
        g
    };
    write_package(
        &rlxp,
        &g,
        &[PackedWeight {
            name: "w".into(),
            shape: vec![N_BYTES / 4],
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: payload.clone(),
            rank: None,
            tier: StorageTier::Hot,
        }],
        &WriteOptions {
            container: ContainerKind::Flat,
            compress_sidecars: false,
            ..WriteOptions::default()
        },
    )
    .unwrap();

    // --- RLXP zip (DDUF-like outer container) ---
    let rlxp_zip = dir.path().join("w.zip");
    write_package(
        &rlxp_zip,
        &g,
        &[PackedWeight {
            name: "w".into(),
            shape: vec![N_BYTES / 4],
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: payload.clone(),
            rank: None,
            tier: StorageTier::Hot,
        }],
        &WriteOptions {
            container: ContainerKind::Zip,
            compress_sidecars: false,
            ..WriteOptions::default()
        },
    )
    .unwrap();

    // --- GGUF ---
    let gguf = dir.path().join("w.gguf");
    {
        let mut w = GgufWriter::new();
        w.add_tensor_bytes("w", vec![N_BYTES / 4], GgmlType::F32, payload.clone())
            .unwrap();
        w.write_to_path(&gguf).unwrap();
    }

    // --- safetensors ---
    let st = dir.path().join("w.safetensors");
    write_safetensors(&st, "w", &payload);

    let sizes: HashMap<&str, u64> = [
        ("rlxp flat", std::fs::metadata(&rlxp).unwrap().len()),
        ("rlxp zip", std::fs::metadata(&rlxp_zip).unwrap().len()),
        ("gguf", std::fs::metadata(&gguf).unwrap().len()),
        ("safetensors", std::fs::metadata(&st).unwrap().len()),
    ]
    .into_iter()
    .collect();

    println!();
    println!("=== Format compare ({} payload, {} iters) ===", fmt_mib(N_BYTES as u64), ITERS);
    println!();
    println!("{:<22} {:>10}  {}", "format", "file size", "notes");
    println!("{}", "-".repeat(72));
    for k in ["rlxp flat", "gguf", "safetensors", "rlxp zip"] {
        println!("{:<22} {:>10}", k, fmt_mib(sizes[k]));
    }
    println!();
    println!("{:<42} {:>12} {:>12}", "operation", "mean", "±σ");
    println!("{}", "-".repeat(68));

    let path = rlxp.clone();
    let (m, s) = time_it(ITERS, || {
        let p = Package::open(&path).unwrap();
        let buf = p.tensor_mmap("w").unwrap();
        black_box(touch(buf));
    });
    println!("{:<42} {:>12} {:>12}", "RLXP flat: open+mmap+touch", fmt_dur(m), fmt_dur(s));

    let path = gguf.clone();
    let (m, s) = time_it(ITERS, || {
        let f = GgufFile::from_path_mmap(&path).unwrap();
        let t = f.get("w").unwrap();
        let buf = f.tensor_bytes(t).unwrap();
        black_box(touch(buf));
    });
    println!("{:<42} {:>12} {:>12}", "GGUF: open_mmap+tensor+touch", fmt_dur(m), fmt_dur(s));

    let path = st.clone();
    let (m, s) = time_it(ITERS, || {
        let (map, begin, end) = open_safetensors_mmap(&path);
        black_box(touch(&map[begin..end]));
    });
    println!("{:<42} {:>12} {:>12}", "safetensors: mmap+hdr+touch", fmt_dur(m), fmt_dur(s));

    let path = rlxp_zip.clone();
    let (m, s) = time_it(ITERS, || {
        let p = Package::open(&path).unwrap();
        let buf = p.tensor_mmap("w").unwrap();
        black_box(touch(buf));
    });
    println!(
        "{:<42} {:>12} {:>12}",
        "RLXP zip≈DDUF: open+mmap+touch",
        fmt_dur(m),
        fmt_dur(s)
    );

    // Already-open reuse (steady-state)
    let pack = Package::open(&rlxp).unwrap();
    let gg = GgufFile::from_path_mmap(&gguf).unwrap();
    let (st_map, st_b, st_e) = open_safetensors_mmap(&st);

    let (m, s) = time_it(ITERS, || {
        black_box(touch(pack.tensor_mmap("w").unwrap()));
    });
    println!("{:<42} {:>12} {:>12}", "RLXP flat: mmap touch (held open)", fmt_dur(m), fmt_dur(s));

    let (m, s) = time_it(ITERS, || {
        let t = gg.get("w").unwrap();
        black_box(touch(gg.tensor_bytes(t).unwrap()));
    });
    println!("{:<42} {:>12} {:>12}", "GGUF: mmap touch (held open)", fmt_dur(m), fmt_dur(s));

    let (m, s) = time_it(ITERS, || {
        black_box(touch(&st_map[st_b..st_e]));
    });
    println!(
        "{:<42} {:>12} {:>12}",
        "safetensors: mmap touch (held open)",
        fmt_dur(m),
        fmt_dur(s)
    );

    println!();
    println!("ONNX (not timed here):");
    println!("  - Full ModelProto protobuf parse + graph walk (ms–tens of ms typical)");
    println!("  - Weights often as initializers inside protobuf or external data");
    println!("  - Not a GGUF/RLXP-class single mmap weight window unless external data + custom loader");
    println!("  - Usually larger on disk when weights are f32/f16 vs GGUF/RLXP packed quants");
    println!();
    println!("DDUF = ZIP64 STORE of safetensors (+ json). Closest timed proxy: RLXP zip row above.");
}
