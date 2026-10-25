// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a negative ONNX `Pad` amount crops.
//!
//! VITS' relative attention pads its position embeddings by
//! `len − (window + 1)`: below zero for a short sequence (Piper at three to
//! five phonemes). The importer stubbed such a `Pad` with zeros, so every
//! attention layer read zero embeddings and the model ran wrong, silently.

use std::collections::HashMap;

use rlx_onnx_import::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, build_hir_from_parts};

#[test]
fn a_negative_pad_crops_and_the_rest_pads() {
    // x [2, 6]: crop one column before, pad two zero columns after.
    let pads = vec![0i64, -1, 0, 2];
    let nodes = vec![BundleNode {
        name: "attn/Pad".into(),
        op: "Pad".into(),
        inputs: vec!["x".into(), "pads".into()],
        outputs: vec!["y".into()],
        attrs: HashMap::from([("mode".into(), serde_json::json!("constant"))]),
        output_meta: vec![serde_json::json!({"shape": [], "dtype": "float32"})],
    }];
    let io = |name: &str, shape: Vec<serde_json::Value>| IoMeta {
        name: name.into(),
        meta: TensorMeta {
            shape,
            dtype: "float32".into(),
        },
    };
    let manifest = BundleManifest {
        source_onnx: "test".into(),
        inputs: vec![io("x", vec![serde_json::json!(2), serde_json::json!(6)])],
        outputs: vec![io("y", vec![])],
        node_count: 1,
        initializer_count: 1,
        op_histogram: HashMap::from([("Pad".into(), 1)]),
    };
    let (hir, _, _, report) = build_hir_from_parts(
        &manifest,
        nodes,
        HashMap::new(),
        TypedParams::new(),
        HashMap::from([("pads".to_string(), pads)]),
        &HashMap::from([("pads".to_string(), vec![4usize])]),
        ImportOptions::default(),
    )
    .expect("lower Pad");
    assert_eq!(report.stubbed, 0, "{:?}", report.stubbed_nodes);
    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(graph);
    c.finalize_params();
    let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let y = c.run(&[("x", &x)]).remove(0);
    assert_eq!(
        y,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, //
            7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0
        ]
    );
}
