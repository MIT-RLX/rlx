// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Criterion bench for hybrid RLXP open / mmap / warm / cold paths.
//!
//! ```text
//! RLX_ALLOW_THROTTLE=1 cargo bench -p rlx-pkg --bench hybrid_load
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{
    ContainerKind, Package, PackedWeight, StorageTier, WriteOptions, write_package,
};
use std::path::PathBuf;
use std::sync::OnceLock;

const HOT_BYTES: usize = 32 << 20;
const WARM_BYTES: usize = 32 << 20;

struct Fixtures {
    flat: PathBuf,
    zip: PathBuf,
    _dir: tempfile::TempDir,
}

fn fixtures() -> &'static Fixtures {
    static F: OnceLock<Fixtures> = OnceLock::new();
    F.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let flat = dir.path().join("h.rlxp");
        let zip = dir.path().join("h.zip");
        let g = {
            let s = Shape::new(&[4], DType::F32);
            let mut g = Graph::new("b");
            let x = g.input("x", s.clone());
            let w = g.param("w_hot", s.clone());
            let y = g.binary(BinaryOp::Mul, x, w, s);
            g.set_outputs(vec![y]);
            g
        };
        let mut warm = vec![0u8; WARM_BYTES];
        for i in (0..WARM_BYTES).step_by(64) {
            warm[i] = (i % 255) as u8;
        }
        let weights = [
            PackedWeight {
                name: "w_hot".into(),
                shape: vec![HOT_BYTES / 4],
                scheme: "f32".into(),
                layout: "row_major".into(),
                data: (0..HOT_BYTES).map(|i| (i % 251) as u8).collect(),
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
        let mut opts = WriteOptions {
            container: ContainerKind::Flat,
            warm_block_size: 1 << 20,
            compress_sidecars: true,
            ..WriteOptions::default()
        };
        opts.sidecars.push((
            "tokenizer".into(),
            "application/json".into(),
            br#"{"a":[1,2,3],"pad":"xxxxxxxxxxxxxxxxxxxxxxxx"}"#.repeat(2000),
        ));
        write_package(&flat, &g, &weights, &opts).unwrap();
        let mut zw = weights.clone();
        zw[1].tier = StorageTier::Hot;
        opts.container = ContainerKind::Zip;
        write_package(&zip, &g, &zw, &opts).unwrap();
        Fixtures {
            flat,
            zip,
            _dir: dir,
        }
    })
}

fn touch(buf: &[u8]) -> u64 {
    let mut a = 0u64;
    for i in (0..buf.len()).step_by(4096) {
        a = a.wrapping_add(buf[i] as u64);
    }
    a
}

fn bench_hybrid(c: &mut Criterion) {
    let fx = fixtures();
    let mut group = c.benchmark_group("rlxp_hybrid");
    group.sample_size(30);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(2));

    group.bench_function("open_flat", |b| {
        b.iter(|| {
            let p = Package::open(black_box(&fx.flat)).unwrap();
            black_box(p.manifest().name.len());
        });
    });
    group.bench_function("open_zip", |b| {
        b.iter(|| {
            let p = Package::open(black_box(&fx.zip)).unwrap();
            black_box(p.manifest().name.len());
        });
    });

    let pack = Package::open(&fx.flat).unwrap();
    group.bench_function("hot_mmap_touch", |b| {
        b.iter(|| {
            let buf = pack.tensor_mmap("w_hot").unwrap();
            black_box(touch(buf));
        });
    });
    group.bench_function("warm_inflate_all", |b| {
        b.iter(|| {
            let buf = pack.tensor_bytes("w_warm").unwrap();
            black_box(touch(&buf));
        });
    });
    group.bench_function("warm_inflate_block0", |b| {
        b.iter(|| {
            let buf = pack.tensor_warm_block("w_warm", 0).unwrap();
            black_box(touch(&buf));
        });
    });
    group.bench_function("cold_sidecar", |b| {
        b.iter(|| {
            let buf = pack.sidecar("tokenizer").unwrap();
            black_box(touch(&buf));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_hybrid);
criterion_main!(benches);
