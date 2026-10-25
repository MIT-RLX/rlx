// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: ONNX `ArgMax` / `ArgMin` / `MaxPool` indices import as valid IR.
//!
//! The importer declared `Op::ArgMax` itself `i64`, ONNX's index type, while
//! the IR op returns f32-encoded indices; the verifier rejected the graph, so
//! a model with an argmax (the conformance `gqa_layernorm` fixture) could
//! not compile in a debug build. The importer now emits the f32 op and casts
//! its result to i64.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_onnx_import::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file, build_hir_from_parts};
use rlx_runtime::{Device, Session};
use serde_json::json;

fn argmax_rows(x: &[f32], width: usize) -> Vec<f32> {
    x.chunks(width)
        .map(|r| {
            r.iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0 as f32
        })
        .collect()
}

#[test]
fn the_gqa_fixture_compiles_and_its_argmax_is_the_logits_argmax() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-onnx-conformance/tests/fixtures/gqa_layernorm.onnx");
    let (hir, params, _, _) =
        build_hir_from_onnx_file(&path, ImportOptions::default()).expect("import");
    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let feed: Vec<(String, Vec<f32>)> = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => {
                let len = n.shape.num_elements().unwrap();
                let v = if n.shape.dtype().is_float() {
                    (0..len).map(|i| ((i as f32) * 0.37).sin()).collect()
                } else {
                    vec![0.0; len]
                };
                Some((name.clone(), v))
            }
            _ => None,
        })
        .collect();
    let mut c = Session::new(Device::Cpu)
        .try_compile(graph)
        .expect("compile");
    for (k, v) in &params {
        c.set_param(k, v);
    }
    let feed: Vec<(&str, &[f32])> = feed.iter().map(|(n, v)| (n.as_str(), &v[..])).collect();
    let out = c.run(&feed);
    let logits = &out[c.output_index("logits").unwrap()];
    let top = &out[c.output_index("top_f").unwrap()];
    assert_eq!(top, &argmax_rows(logits, 5));
    assert!(top.iter().any(|&t| t != 0.0), "all-zero indices: {top:?}");
}

fn io(name: &str, shape: &[usize], dtype: &str) -> IoMeta {
    IoMeta {
        name: name.into(),
        meta: TensorMeta {
            shape: shape.iter().map(|&d| json!(d)).collect(),
            dtype: dtype.into(),
        },
    }
}

/// One node reading `x`; `outputs` are `(name, shape, ONNX dtype)`.
fn node(op: &str, outputs: &[(&str, &[usize], &str)], attrs: serde_json::Value) -> BundleNode {
    BundleNode {
        name: op.into(),
        op: op.into(),
        inputs: vec!["x".into()],
        outputs: outputs.iter().map(|o| o.0.to_string()).collect(),
        attrs: serde_json::from_value(attrs).unwrap(),
        output_meta: outputs
            .iter()
            .map(|o| json!({"shape": o.1, "dtype": o.2}))
            .collect(),
    }
}

fn import(node: BundleNode, x: IoMeta, outputs: Vec<IoMeta>) -> rlx_ir::Graph {
    let manifest = BundleManifest {
        source_onnx: "test".into(),
        inputs: vec![x],
        outputs,
        node_count: 1,
        initializer_count: 0,
        op_histogram: HashMap::from([(node.op.clone(), 1)]),
    };
    let (hir, _, _, report) = build_hir_from_parts(
        &manifest,
        vec![node],
        HashMap::new(),
        TypedParams::new(),
        HashMap::new(),
        &HashMap::new(),
        ImportOptions::default(),
    )
    .expect("lower");
    assert_eq!(report.stubbed, 0, "{:?}", report.stubbed_nodes);
    rlx_ir::hir_to_graph(hir).expect("hir_to_graph")
}

#[test]
fn argmin_of_a_half_precision_input_is_exact_int64() {
    let g = import(
        node(
            "ArgMin",
            &[("i", &[2], "int64")],
            json!({"axis": 1, "keepdims": 0}),
        ),
        io("x", &[2, 3], "float16"),
        vec![io("i", &[2], "int64")],
    );
    assert_eq!(g.shape(g.outputs[0]).dtype(), rlx_ir::DType::I64);
    assert!(
        rlx_ir::verify_all(&g).is_empty(),
        "{:?}",
        rlx_ir::verify_all(&g)
    );
    let mut c = Session::new(Device::Cpu).try_compile(g).expect("compile");
    let x: Vec<u8> = [3.0f32, 1.0, 2.0, 0.5, 0.25, 4.0]
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let out = c.run_typed(&[("x", &x, rlx_ir::DType::F16)]);
    let idx: Vec<i64> = out[0]
        .0
        .as_chunks::<8>()
        .0
        .iter()
        .take(2)
        .map(|b| i64::from_le_bytes(*b))
        .collect();
    assert_eq!(idx, [1, 1]);
}

#[test]
fn max_pool_indices_are_valid_int64() {
    // x [1, 1, 2, 4], 2×2 windows: two windows, maxima at flat 5 and 3.
    let g = import(
        node(
            "MaxPool",
            &[
                ("y", &[1, 1, 1, 2], "float32"),
                ("idx", &[1, 1, 1, 2], "int64"),
            ],
            json!({"kernel_shape": [2, 2], "strides": [2, 2]}),
        ),
        io("x", &[1, 1, 2, 4], "float32"),
        vec![
            io("y", &[1, 1, 1, 2], "float32"),
            io("idx", &[1, 1, 1, 2], "int64"),
        ],
    );
    assert!(
        rlx_ir::verify_all(&g).is_empty(),
        "{:?}",
        rlx_ir::verify_all(&g)
    );
    let mut c = Session::new(Device::Cpu).try_compile(g).expect("compile");
    let x = [0.0f32, 1.0, 2.0, 9.0, 4.0, 8.0, 6.0, 7.0];
    let out = c.run(&[("x", &x)]);
    assert_eq!(out[0], [8.0, 9.0]);
    assert_eq!(out[1], [5.0, 3.0]);
}
