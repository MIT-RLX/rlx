// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bake merges graph + weights; round-trip `*.rlx` keeps the weight table.

use rlx_bake::{BakeOptions, BakeProfile, bake, read_rlx, write_rlx};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use std::collections::HashMap;

#[test]
fn bake_mul_param_includes_weights() {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("mul_bake");
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);

    let mut bindings = HashMap::new();
    bindings.insert("w".into(), vec![1.0, 2.0, 3.0, 4.0]);

    let (file, report) = bake(&g, &bindings, &BakeOptions::default());
    assert!(report.params_baked >= 1);
    assert!(report.params_remaining.is_empty());
    assert_eq!(file.weights.len(), 1);
    assert_eq!(file.weights[0].name, "w");
    assert_eq!(file.weights[0].encoding, "f32");
    assert_eq!(file.weights[0].data.len(), 16);
    assert!(
        !file
            .graph
            .nodes()
            .iter()
            .any(|n| matches!(&n.op, Op::Param { name } if name == "w"))
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.rlx");
    write_rlx(&path, &file).expect("write_rlx");
    let loaded = read_rlx(&path).expect("read_rlx");

    assert_eq!(loaded.meta.name, "mul_bake");
    assert_eq!(loaded.meta.weight_count, 1);
    assert_eq!(loaded.weights.len(), 1);
    assert_eq!(loaded.weights[0].name, "w");
    assert_eq!(loaded.weights[0].data, file.weights[0].data);
}

#[test]
fn bake_skips_zero_matmul() {
    let s_x = Shape::new(&[2, 4], DType::F32);
    let s_w = Shape::new(&[4, 3], DType::F32);
    let s_y = Shape::new(&[2, 3], DType::F32);
    let mut g = Graph::new("zero_mm");
    let x = g.input("x", s_x);
    let w = g.param("w", s_w);
    let y = g.add_node(Op::MatMul, vec![x, w], s_y);
    g.set_outputs(vec![y]);

    let mut bindings = HashMap::new();
    bindings.insert("w".into(), vec![0.0; 12]);
    let (file, report) = bake(&g, &bindings, &BakeOptions::default());
    assert_eq!(report.optimize.skipped_zero_matmuls, 1);
    assert!(matches!(
        file.graph.node(file.graph.outputs[0]).op,
        Op::Constant { .. }
    ));
    // Default profiles drop folded-away bindings; the zero weight is gone from the table.
    assert!(
        report.memory.folded_bindings_dropped >= 1 || file.weights.is_empty(),
        "zero matmul weight should be dropped from the table when not kept"
    );
}

#[test]
fn bake_packs_ternary_matmul() {
    // 256 ternary weights → one TQ2_0 block.
    let k = 256;
    let n = 1;
    let s_x = Shape::new(&[2, k], DType::F32);
    let s_w = Shape::new(&[k, n], DType::F32);
    let s_y = Shape::new(&[2, n], DType::F32);
    let mut g = Graph::new("ternary_mm");
    let x = g.input("x", s_x);
    let w = g.param("w", s_w);
    let y = g.add_node(Op::MatMul, vec![x, w], s_y);
    g.set_outputs(vec![y]);

    let mut vals = vec![0.0f32; k * n];
    for (i, v) in vals.iter_mut().enumerate() {
        *v = match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
    }
    let mut bindings = HashMap::new();
    bindings.insert("w".into(), vals);

    let (file, report) = bake(&g, &bindings, &BakeOptions::default());
    assert_eq!(report.optimize.ternary_packed, 1);
    assert!(
        file.graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::DequantMatMul { .. })),
        "expected DequantMatMul after ternary pack"
    );
    assert!(
        file.weights
            .iter()
            .any(|w| w.name == "w" && w.encoding == "gguf_tq2_0"),
        "weight table should list TQ2_0 encoding: {:?}",
        file.weights
            .iter()
            .map(|w| (&w.name, &w.encoding))
            .collect::<Vec<_>>()
    );
}

/// Same ternary graph under each profile — packing only with `exact` / `size`.
#[test]
fn bake_profiles_change_rewrites() {
    let k = 256;
    let n = 1;
    let s_x = Shape::new(&[2, k], DType::F32);
    let s_w = Shape::new(&[k, n], DType::F32);
    let s_y = Shape::new(&[2, n], DType::F32);
    let mut g = Graph::new("profiles");
    let x = g.input("x", s_x);
    let w = g.param("w", s_w);
    let y = g.add_node(Op::MatMul, vec![x, w], s_y);
    g.set_outputs(vec![y]);

    let mut vals = vec![0.0f32; k];
    for (i, v) in vals.iter_mut().enumerate() {
        *v = match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
    }
    let mut bindings = HashMap::new();
    bindings.insert("w".into(), vals);

    let (_, merge) = bake(&g, &bindings, &BakeProfile::Merge.options());
    assert_eq!(merge.optimize.ternary_packed, 0);
    assert_eq!(merge.optimize.skipped_zero_matmuls, 0);

    let (_, fold) = bake(&g, &bindings, &BakeProfile::Fold.options());
    assert_eq!(fold.optimize.ternary_packed, 0);

    let (_, exact) = bake(&g, &bindings, &BakeProfile::Exact.options());
    assert_eq!(exact.optimize.ternary_packed, 1);
    assert_eq!(exact.optimize.quant_packed, 0);

    let (_, size) = bake(&g, &bindings, &BakeProfile::Size.options());
    assert_eq!(size.optimize.ternary_packed, 1);
    // Ternary already claimed the matmul; Q8 should not double-pack it.
    assert_eq!(size.optimize.quant_packed, 0);

    // Override: size profile but disable ternary → remaining weight eligible for Q8.
    let mut opts = BakeOptions::from_profile(BakeProfile::Size);
    opts.ternary = false;
    let (_, size_q) = bake(&g, &bindings, &opts);
    assert_eq!(size_q.optimize.ternary_packed, 0);
    assert_eq!(size_q.optimize.quant_packed, 1);
}

#[test]
fn convert_rlx_to_rlxp_roundtrip() {
    use rlx_bake::{convert_rlx_to_rlxp, write_rlx};
    use rlx_pkg::Package;

    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new("to_rlxp");
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.binary(BinaryOp::Mul, x, w, s);
    g.set_outputs(vec![y]);

    let mut bindings = HashMap::new();
    bindings.insert("w".into(), vec![1.0, 2.0, 3.0, 4.0]);
    let (file, _) = bake(&g, &bindings, &BakeOptions::default());

    let dir = tempfile::tempdir().unwrap();
    let rlx = dir.path().join("model.rlx");
    let rlxp = dir.path().join("model.rlxp");
    write_rlx(&rlx, &file).unwrap();
    convert_rlx_to_rlxp(&rlx, &rlxp, Some(rlx_pkg::ContainerKind::Flat)).unwrap();

    let pack = Package::open(&rlxp).unwrap();
    assert_eq!(pack.tensor_f32("w").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(pack.graph().unwrap().name, "to_rlxp");
}
