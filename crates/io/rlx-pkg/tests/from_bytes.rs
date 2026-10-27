// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A package opened from its bytes (`Package::from_bytes` — a browser has no
//! file to map) reads exactly as the same file opened from disk, every tier.

#![cfg(feature = "write")]
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{ContainerKind, Package, PackedWeight, StorageTier, WriteOptions, write_package};

/// `tempfile::tempdir()`, under `/tmp` on WASI (where `std::env::temp_dir` panics).
fn scratch() -> tempfile::TempDir {
    if cfg!(target_os = "wasi") {
        tempfile::tempdir_in("/tmp").unwrap()
    } else {
        tempfile::tempdir().unwrap()
    }
}

fn graph() -> Graph {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("from_bytes");
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);
    g
}

fn weight(name: &str, data: Vec<u8>, tier: StorageTier) -> PackedWeight {
    PackedWeight {
        name: name.into(),
        shape: vec![data.len()],
        scheme: "f32".into(),
        layout: "row_major".into(),
        data,
        rank: None,
        tier,
    }
}

#[test]
fn bytes_read_as_the_file_does() {
    let dir = scratch();
    let hot: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let warm: Vec<u8> = (0..48 * 1024).map(|i| (i % 13) as u8).collect();
    let cold: Vec<u8> = (0..9000).map(|i| (i % 5) as u8).collect();
    for (container, file) in [
        (ContainerKind::Flat, "p.rlxp"),
        (ContainerKind::Zip, "p.zip"),
    ] {
        let out = dir.path().join(file);
        let mut opts = WriteOptions {
            container,
            warm_block_size: 16 * 1024,
            compress_sidecars: true,
            ..WriteOptions::default()
        };
        opts.sidecars
            .push(("notes".into(), "text/plain".into(), cold.clone()));
        let ws = [
            weight("w", hot.clone(), StorageTier::Hot),
            weight("warm", warm.clone(), StorageTier::Warm),
        ];
        write_package(&out, &graph(), &ws, &opts).unwrap();

        let disk = Package::open(&out).unwrap();
        let mem = Package::from_bytes(std::fs::read(&out).unwrap()).unwrap();
        for name in ["w", "warm"] {
            assert_eq!(
                mem.tensor_bytes(name).unwrap(),
                disk.tensor_bytes(name).unwrap(),
                "{file}: {name}"
            );
        }
        assert_eq!(mem.tensor_bytes("warm").unwrap(), warm, "{file}");
        assert_eq!(mem.tensor_mmap("w").unwrap(), &hot[..], "{file}");
        assert_eq!(
            mem.sidecar("notes").unwrap(),
            disk.sidecar("notes").unwrap(),
            "{file}"
        );
        assert_eq!(mem.sidecar("notes").unwrap(), cold, "{file}");
        let (a, b) = (mem.graph().unwrap(), disk.graph().unwrap());
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "{file}: graph");
    }
}

#[test]
fn bytes_that_are_not_a_package_are_refused_by_name() {
    let e = Package::from_bytes(b"not a package at all".to_vec())
        .err()
        .expect("refused");
    assert!(
        e.to_string()
            .contains("not a flat RLXPFLAT, ZIP, or package directory"),
        "{e}"
    );
}
