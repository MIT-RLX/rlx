// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a nearest `Resize` that scales the **height** axis must lower to real ops.
//!
//! `lower_resize`'s nearest branch covered exactly two shapes — a 2×2 upsample, and a
//! *width*-only resize gated on `h_in == h_out == 1`. A height upsample matched neither and
//! fell through to the zero-filled `__resize__/…` stub at the end of the function, so the
//! model imported, ran, and returned wrong numbers with no signal.
//!
//! KittenTTS's vocoder is one: `f0_upsamp` is `[1,1,1,F] → [1,1,300,F]` (nearest ×300 on H),
//! which meant its NSF f0 source was zeros on every backend.

use std::collections::HashMap;

use rlx_onnx_import::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, build_hir_from_parts};

/// Import a single nearest `Resize` over `[1,1,h_in,w_in]` with the given per-axis scales,
/// returning `(output dims, stub count, ops present)`.
fn import_nearest_resize(
    h_in: usize,
    w_in: usize,
    scales: [f32; 4],
) -> (Vec<usize>, usize, Vec<String>) {
    let mut params = HashMap::new();
    params.insert("scales".into(), scales.to_vec());
    let mut init_shapes = HashMap::new();
    init_shapes.insert("scales".into(), vec![4usize]);

    let mut attrs = HashMap::new();
    attrs.insert("mode".into(), serde_json::json!("nearest"));
    attrs.insert(
        "coordinate_transformation_mode".into(),
        serde_json::json!("asymmetric"),
    );

    let nodes = vec![BundleNode {
        name: "f0_upsamp/Resize".into(),
        op: "Resize".into(),
        // ONNX Resize(X, roi, scales): roi is the empty optional.
        inputs: vec!["x".into(), String::new(), "scales".into()],
        outputs: vec!["y".into()],
        attrs,
        output_meta: vec![serde_json::json!({"shape": [], "dtype": "float32"})],
    }];

    let manifest = BundleManifest {
        source_onnx: "test".into(),
        inputs: vec![IoMeta {
            name: "x".into(),
            meta: TensorMeta {
                shape: vec![
                    serde_json::json!(1),
                    serde_json::json!(1),
                    serde_json::json!(h_in),
                    serde_json::json!(w_in),
                ],
                dtype: "float32".into(),
            },
        }],
        outputs: vec![IoMeta {
            name: "y".into(),
            meta: TensorMeta {
                shape: vec![],
                dtype: "float32".into(),
            },
        }],
        node_count: 1,
        initializer_count: 1,
        op_histogram: HashMap::from([("Resize".into(), 1)]),
    };

    let (hir, _params, _typed, report) = build_hir_from_parts(
        &manifest,
        nodes,
        params,
        TypedParams::new(),
        HashMap::new(),
        &init_shapes,
        ImportOptions::default(),
    )
    .expect("lower Resize");

    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let out = graph.nodes().last().expect("an output node");
    let dims = out
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect::<Vec<_>>();
    let ops = graph
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op))
        .collect::<Vec<_>>();
    (dims, report.stubbed, ops)
}

#[test]
fn nearest_height_upsample_lowers_without_a_stub() {
    // The exact KittenTTS `f0_upsamp` case, at a test-sized frame count.
    let (dims, stubbed, ops) = import_nearest_resize(1, 9, [1.0, 1.0, 300.0, 1.0]);
    assert_eq!(stubbed, 0, "height resize fell back to a zero stub");
    assert_eq!(dims, vec![1, 1, 300, 9], "wrong output shape: {dims:?}");
    assert!(
        ops.iter().any(|o| o.contains("Expand")),
        "expected the broadcast lowering, got {ops:?}"
    );
    assert!(
        !ops.iter().any(|o| o.contains("__resize__")),
        "a zero-filled resize param survived: {ops:?}"
    );
}

#[test]
fn nearest_both_axes_integral_upsample_lowers_without_a_stub() {
    // Non-square integral scales exercise the same identity on both axes at once; the
    // pre-existing branch only handled the 2×2 case.
    let (dims, stubbed, _) = import_nearest_resize(3, 5, [1.0, 1.0, 4.0, 2.0]);
    assert_eq!(stubbed, 0);
    assert_eq!(dims, vec![1, 1, 12, 10], "wrong output shape: {dims:?}");
}

#[test]
fn nearest_identity_resize_is_still_a_passthrough() {
    // All scales 1: the new branch must not claim this — it stays on the reshape path.
    let (dims, stubbed, ops) = import_nearest_resize(4, 7, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(stubbed, 0);
    assert_eq!(dims, vec![1, 1, 4, 7], "wrong output shape: {dims:?}");
    assert!(
        !ops.iter().any(|o| o.contains("Expand")),
        "identity resize should not emit a broadcast: {ops:?}"
    );
}
