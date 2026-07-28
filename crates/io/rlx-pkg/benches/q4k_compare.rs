// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compare Q4_K-packed RLXP flat size / open vs a GGUF with the same payload.
//!
//! ```text
//! RLX_ALLOW_THROTTLE=1 cargo bench -p rlx-pkg --bench q4k_compare
//! ```
//!
//! Note: warm zstd is **not** a substitute for Q4_K — this bench keeps weights
//! hot/raw so the comparison is fair against GGUF mmap.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_gguf::{GgmlType, GgufWriter, quantize};
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{ContainerKind, Package, PackedWeight, WriteOptions, write_package};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

const N: usize = 256 * 256; // one Q4_K-friendly size (multiple of 256)

struct Fixtures {
    rlxp: PathBuf,
    gguf: PathBuf,
    _dir: tempfile::TempDir,
    q4k_bytes: usize,
}

fn fixtures() -> &'static Fixtures {
    static F: OnceLock<Fixtures> = OnceLock::new();
    F.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let floats: Vec<f32> = (0..N).map(|i| (i as f32 * 0.001).sin()).collect();
        let q = quantize(&floats, GgmlType::Q4K).expect("quantize Q4_K");
        let q4k_bytes = q.len();

        let gguf_path = dir.path().join("w.gguf");
        let mut w = GgufWriter::new();
        w.add_tensor_bytes("w", vec![256, 256], GgmlType::Q4K, q.clone())
            .unwrap();
        w.write_to_path(&gguf_path).unwrap();

        let rlxp = dir.path().join("w.rlxp");
        let mut g = Graph::new("q4k");
        let s = Shape::new(&[1], DType::F32);
        let x = g.input("x", s);
        g.set_outputs(vec![x]);
        let weights = vec![PackedWeight::hot(
            "w",
            vec![256, 256],
            "gguf_q4_k",
            "bt_nk",
            q,
        )];
        write_package(
            &rlxp,
            &g,
            &weights,
            &WriteOptions {
                name: "q4k".into(),
                container: ContainerKind::Flat,
                include_graph: false,
                compress_sidecars: false,
                write_checksums: true,
                ..WriteOptions::default()
            },
        )
        .unwrap();

        Fixtures {
            rlxp,
            gguf: gguf_path,
            _dir: dir,
            q4k_bytes,
        }
    })
}

fn bench_q4k(c: &mut Criterion) {
    let f = fixtures();
    let rlxp_sz = std::fs::metadata(&f.rlxp).unwrap().len();
    let gguf_sz = std::fs::metadata(&f.gguf).unwrap().len();
    eprintln!(
        "q4k_compare sizes: payload={} rlxp={} gguf={} (rlxp/gguf={:.3})",
        f.q4k_bytes,
        rlxp_sz,
        gguf_sz,
        rlxp_sz as f64 / gguf_sz as f64
    );

    c.bench_function("open_rlxp_q4k", |b| {
        b.iter(|| {
            let p = Package::open(&f.rlxp).unwrap();
            let _ = black_box(p.tensor_mmap("w").unwrap().len());
        })
    });

    c.bench_function("open_gguf_q4k", |b| {
        b.iter(|| {
            let g = rlx_gguf::GgufFile::from_path_mmap(&f.gguf).unwrap();
            let t = g.get("w").unwrap();
            let bytes = g.tensor_bytes(t).unwrap();
            let _ = black_box(bytes.len());
        })
    });

    // One-shot wall times for the eprintln summary path.
    let t0 = Instant::now();
    let p = Package::open(&f.rlxp).unwrap();
    let _ = p.tensor_mmap("w").unwrap();
    let rlxp_open = t0.elapsed();
    let t1 = Instant::now();
    let g = rlx_gguf::GgufFile::from_path_mmap(&f.gguf).unwrap();
    let t = g.get("w").unwrap();
    let _ = g.tensor_bytes(t).unwrap();
    let gguf_open = t1.elapsed();
    eprintln!("q4k_compare open once: rlxp={rlxp_open:?} gguf={gguf_open:?}");
}

criterion_group!(benches, bench_q4k);
criterion_main!(benches);
