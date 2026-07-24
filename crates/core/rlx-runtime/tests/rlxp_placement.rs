// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Placement helpers for `.rlxp` packages.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_pkg::{BakeWeight, Package, Placement, TensorShard, WriteOptions, package_from_bake};
use rlx_runtime::pkg::{tensors_for_rank, weight_names_for_rank};
use std::collections::BTreeMap;

#[test]
fn tensors_for_rank_filters_by_placement() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "a".into(),
        TensorShard {
            dim: 0,
            ranks: vec![0],
        },
    );
    tensors.insert(
        "b".into(),
        TensorShard {
            dim: 0,
            ranks: vec![1],
        },
    );
    let pl = Placement {
        parallelism: vec!["tp".into()],
        world_size: Some(2),
        topology: None,
        tensors,
        experts: BTreeMap::new(),
    };
    let all = ["a", "b", "c"];
    assert_eq!(tensors_for_rank(&pl, 0, &all), vec!["a", "c"]);
    assert_eq!(tensors_for_rank(&pl, 1, &all), vec!["b", "c"]);
}

#[test]
fn weight_names_for_rank_from_pack() {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("place");
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);
    let weights = [BakeWeight {
        name: "w".into(),
        shape: vec![4],
        encoding: "f32".into(),
        data: vec![0u8; 16],
    }];
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "w".into(),
        TensorShard {
            dim: 0,
            ranks: vec![1],
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("p");
    package_from_bake(
        &out,
        &g,
        &weights,
        WriteOptions {
            container: rlx_pkg::ContainerKind::Dir,
            placement: Some(Placement {
                parallelism: vec!["tp".into()],
                world_size: Some(2),
                topology: Some("mesh".into()),
                tensors,
                experts: BTreeMap::new(),
            }),
            ..WriteOptions::default()
        },
    )
    .unwrap();
    let pack = Package::open(&out).unwrap();
    assert!(weight_names_for_rank(&pack, 0).is_empty());
    assert_eq!(weight_names_for_rank(&pack, 1), vec!["w".to_string()]);
}
