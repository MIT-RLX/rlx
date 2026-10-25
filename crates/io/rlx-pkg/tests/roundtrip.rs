// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flat / directory / ZIP package round-trips.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_pkg::{
    BakeWeight, ContainerKind, Package, Placement, TensorShard, WriteOptions, package_from_bake,
};
use std::collections::BTreeMap;

fn tiny_graph() -> Graph {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("pkg_tiny");
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);
    g
}

fn weight_f32() -> BakeWeight {
    let vals = [1.0f32, 2.0, 3.0, 4.0];
    BakeWeight {
        name: "w".into(),
        shape: vec![4],
        encoding: "f32".into(),
        data: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

#[test]
fn roundtrip_flat_pack_default() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model.rlxp");
    let g = tiny_graph();
    let weights = vec![weight_f32()];
    let mut opts = WriteOptions {
        container: ContainerKind::Flat,
        name: "pkg_flat".into(),
        ..WriteOptions::default()
    };
    opts.sidecars.push((
        "tokenizer".into(),
        "application/json".into(),
        br#"{"tok":true}"#.to_vec(),
    ));
    package_from_bake(&out, &g, &weights, opts).expect("write flat");

    let raw = std::fs::read(&out).unwrap();
    assert_eq!(&raw[..8], b"RLXPFLAT");

    let pack = Package::open(&out).expect("open flat");
    assert_eq!(pack.tensor_f32("w").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(pack.sidecar("tokenizer").unwrap(), br#"{"tok":true}"#);
    let graph = pack.graph().unwrap();
    // Stripped on disk, materialized on load.
    let w_node = graph
        .nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some("w"))
        .expect("w node");
    // Tiny graph uses Param; bake exports use Constant. Either is fine — tensors
    // come from the data region either way.
    assert!(matches!(&w_node.op, Op::Constant { .. } | Op::Param { .. }));
}

#[test]
fn flat_smaller_than_zip_no_dup() {
    let dir = tempfile::tempdir().unwrap();
    let g = tiny_graph();
    // Larger weight to make overhead differences visible.
    let mut data = Vec::new();
    for i in 0..4096u32 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }
    let weights = vec![BakeWeight {
        name: "w".into(),
        shape: vec![4096],
        encoding: "f32".into(),
        data,
    }];
    let flat_path = dir.path().join("a.rlxp");
    let zip_path = dir.path().join("a.zip");
    package_from_bake(
        &flat_path,
        &g,
        &weights,
        WriteOptions {
            container: ContainerKind::Flat,
            strip_graph_weights: true,
            ..WriteOptions::default()
        },
    )
    .unwrap();
    package_from_bake(
        &zip_path,
        &g,
        &weights,
        WriteOptions {
            container: ContainerKind::Zip,
            strip_graph_weights: true,
            ..WriteOptions::default()
        },
    )
    .unwrap();
    let flat_sz = std::fs::metadata(&flat_path).unwrap().len();
    let zip_sz = std::fs::metadata(&zip_path).unwrap().len();
    assert!(
        flat_sz < zip_sz,
        "flat {flat_sz} should be smaller than zip {zip_sz}"
    );
}

#[test]
fn roundtrip_directory_pack() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model_pkg");
    let g = tiny_graph();
    let weights = vec![weight_f32()];
    let mut opts = WriteOptions {
        container: ContainerKind::Dir,
        name: "pkg_tiny".into(),
        producer: Some("test".into()),
        ..WriteOptions::default()
    };
    opts.sidecars.push((
        "tokenizer".into(),
        "application/json".into(),
        br#"{"tok":true}"#.to_vec(),
    ));
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "w".into(),
        TensorShard {
            dim: 0,
            ranks: vec![0, 1],
        },
    );
    opts.placement = Some(Placement {
        parallelism: vec!["tp".into()],
        world_size: Some(2),
        topology: Some("mesh".into()),
        tensors,
        experts: BTreeMap::new(),
    });

    package_from_bake(&out, &g, &weights, opts).expect("write dir");
    let pack = Package::open(&out).expect("open dir");
    assert_eq!(pack.tensor_f32("w").expect("f32"), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(pack.placement().unwrap().world_size, Some(2));
}

#[test]
fn roundtrip_zip_pack() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("model.zip");
    let g = tiny_graph();
    let weights = vec![weight_f32()];
    package_from_bake(
        &out,
        &g,
        &weights,
        WriteOptions {
            container: ContainerKind::Zip,
            name: "pkg_zip".into(),
            ..WriteOptions::default()
        },
    )
    .expect("write zip");
    let pack = Package::open(&out).expect("open zip");
    assert_eq!(pack.tensor_f32("w").unwrap(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn hybrid_hot_warm_cold() {
    use rlx_pkg::{PackedWeight, StorageTier, write_package};

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("hybrid.rlxp");
    let g = tiny_graph();

    // Hot: small mmap weight
    let hot = PackedWeight {
        name: "w".into(),
        shape: vec![4],
        scheme: "f32".into(),
        layout: "row_major".into(),
        data: [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        rank: None,
        tier: StorageTier::Hot,
    };
    // Warm: compressible blob (zeros + pattern) with small blocks
    let mut warm_raw = vec![0u8; 64 * 1024];
    for (i, b) in warm_raw.iter_mut().enumerate() {
        *b = (i % 17) as u8;
    }
    let warm = PackedWeight {
        name: "expert_cold".into(),
        shape: vec![warm_raw.len()],
        scheme: "f32".into(), // opaque bytes for this test
        layout: "row_major".into(),
        data: warm_raw.clone(),
        rank: None,
        tier: StorageTier::Warm,
    };

    let mut opts = WriteOptions {
        container: ContainerKind::Flat,
        warm_block_size: 16 * 1024,
        compress_sidecars: true,
        ..WriteOptions::default()
    };
    // Highly compressible sidecar
    let tok = {
        let mut s = String::from("{\"vocab\":[");
        s.push_str(&"\"x\",".repeat(2000));
        s.push_str("]}");
        s
    };
    opts.sidecars.push((
        "tokenizer".into(),
        "application/json".into(),
        tok.as_bytes().to_vec(),
    ));

    write_package(&out, &g, &[hot, warm], &opts).unwrap();
    let pack = Package::open(&out).unwrap();

    // Hot mmap works
    assert_eq!(pack.tensor_mmap("w").unwrap().len(), 16);
    assert_eq!(pack.tensor_f32("w").unwrap(), [1.0, 2.0, 3.0, 4.0]);

    // Warm: mmap refused; bytes round-trip; block API works
    assert!(pack.tensor_mmap("expert_cold").is_err());
    assert_eq!(pack.tensor_bytes("expert_cold").unwrap(), warm_raw);
    let b0 = pack.tensor_warm_block("expert_cold", 0).unwrap();
    assert_eq!(b0.len(), 16 * 1024);
    assert_eq!(b0, &warm_raw[..16 * 1024]);

    // Cold sidecar decompresses
    assert_eq!(pack.sidecar("tokenizer").unwrap(), tok.as_bytes());
    assert!(
        pack.manifest()
            .features
            .iter()
            .any(|f| f == "hybrid_storage")
    );

    // Warm stored smaller than raw
    let we = pack.weight_entry("expert_cold").unwrap();
    assert!(we.length < warm_raw.len() as u64);
    assert_eq!(we.tier, StorageTier::Warm);
}

#[test]
fn verify_checksums_and_bincode_toc() {
    use rlx_pkg::{PackedWeight, verify_package, write_package};

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("chk.rlxp");
    let g = tiny_graph();
    let hot = PackedWeight::hot(
        "w",
        vec![4],
        "f32",
        "row_major",
        [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    );
    write_package(
        &out,
        &g,
        &[hot],
        &WriteOptions {
            container: ContainerKind::Flat,
            bincode_toc: true,
            intern_strings: true,
            write_checksums: true,
            ..WriteOptions::default()
        },
    )
    .unwrap();
    let pack = Package::open(&out).unwrap();
    let report = verify_package(&pack).unwrap();
    assert_eq!(report.tensors_ok, 1);
    assert!(pack.manifest().features.iter().any(|f| f == "toc_bincode"));
}

#[test]
fn weight_only_and_zip_hybrid() {
    use rlx_pkg::{PackedWeight, StorageTier, write_package};

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("wo.zip");
    let g = tiny_graph();
    let mut warm_raw = vec![0u8; 32 * 1024];
    for (i, b) in warm_raw.iter_mut().enumerate() {
        *b = (i % 9) as u8;
    }
    let warm = PackedWeight {
        name: "blob".into(),
        shape: vec![warm_raw.len()],
        scheme: "f32".into(),
        layout: "row_major".into(),
        data: warm_raw.clone(),
        rank: None,
        tier: StorageTier::Warm,
    };
    write_package(
        &out,
        &g,
        &[warm],
        &WriteOptions {
            container: ContainerKind::Zip,
            include_graph: false,
            warm_block_size: 8 * 1024,
            ..WriteOptions::default()
        },
    )
    .unwrap();
    let pack = Package::open(&out).unwrap();
    assert!(!pack.has_graph());
    assert!(pack.graph().is_err());
    assert_eq!(pack.tensor_bytes("blob").unwrap(), warm_raw);
}

/// New packages store the graph as `.rlxmodel` text, readable in a
/// directory pack; packages from before it (bincode `graph/mir.bin`) still
/// open.
#[test]
fn graphs_are_rlxmodel_text_and_bincode_packages_still_open() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("pkg");
    let g = tiny_graph();
    let opts = WriteOptions {
        container: ContainerKind::Dir,
        name: "pkg_tiny".into(),
        ..WriteOptions::default()
    };
    package_from_bake(&out, &g, &[weight_f32()], opts).expect("write dir");
    let text = std::fs::read_to_string(out.join("graph/model.rlxmodel")).unwrap();
    assert!(text.starts_with("rlxmodel 1\nopset 1\n"), "{text}");
    assert!(text.contains("Binary(Mul) %x, %w : f32[4]"), "{text}");
    let fresh = Package::open(&out).unwrap().graph().unwrap();
    assert_eq!(fresh.len(), g.len());

    // The same package as an older rlx wrote it.
    std::fs::remove_file(out.join("graph/model.rlxmodel")).unwrap();
    std::fs::write(out.join("graph/mir.bin"), bincode::serialize(&g).unwrap()).unwrap();
    let manifest = out.join("rlx.json");
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    m["graph"] = serde_json::json!({"path": "graph/mir.bin", "encoding": "bincode_graph_v1"});
    std::fs::write(&manifest, serde_json::to_vec(&m).unwrap()).unwrap();
    let old = Package::open(&out).unwrap().graph().unwrap();
    assert_eq!(old.len(), g.len());
    assert_eq!(rlx_ir::rlxmodel::to_text(&old).unwrap(), text);
}

/// A package holds a whole model: several graphs and the `.rlxmodel` header
/// with the model's name and version, in every container.
#[test]
fn a_package_holds_a_module_with_its_model_block() {
    let dir = tempfile::tempdir().unwrap();
    let mut second = tiny_graph();
    second.name = "decode".into();
    for (container, file) in [
        (ContainerKind::Flat, "m.rlxp"),
        (ContainerKind::Dir, "m_dir"),
        (ContainerKind::Zip, "m.zip"),
    ] {
        let out = dir.path().join(file);
        let opts = WriteOptions {
            container,
            name: "pkg_tiny".into(),
            extra_graphs: vec![second.clone()],
            model: Some(serde_json::json!({"name": "tiny", "version": "1.2.0", "task": "graph"})),
            requires_rlx: Some("0.2.0".into()),
            ..WriteOptions::default()
        };
        package_from_bake(&out, &tiny_graph(), &[weight_f32()], opts).expect("write");
        let pack = Package::open(&out).expect("open");
        let m = pack.module().expect("module");
        let names: Vec<&str> = m.graphs.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["pkg_tiny", "decode"], "{file}");
        assert_eq!(
            (m.model_name(), m.model_version()),
            (Some("tiny"), Some("1.2.0"))
        );
        assert_eq!(m.requires.as_deref(), Some("0.2.0"));
        assert_eq!(m.rlx.as_deref(), Some(rlx_ir::rlxmodel::RLX_VERSION));
        // `graph()` is still the main one.
        assert_eq!(pack.graph().unwrap().name, "pkg_tiny");
    }
}
