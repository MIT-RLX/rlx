// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a genuine NCL depthwise-conv input `[1, C, L]` whose LENGTH `L`
//! happens to be a power-of-2 "typical channel" value (64, 80, 128, 256, …) must
//! stay NCL and keep `c_in == C`. The rank-3→4 helper's `is_vocoder_blc` guess
//! (`is_typical_channel(last) && mid > last`) misfired here and transposed
//! `[1,256,64] → [1,64,256]`, so at the conv `c_in/groups = 64/256 = 0`, the
//! im2col was empty, and the depthwise conv degenerated to a bias-only constant
//! output — the supertonic / luxtts ConvNeXt text-encoder babble (the dilation-2
//! "same" pad grows the length 56 → 64, right onto the collision). The conv site
//! now uses the weight's concrete in_channels to disambiguate.

use std::collections::HashMap;

use rlx_ir::Op;
use rlx_onnx_import::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, build_hir_from_parts};

/// Build a single depthwise 1-D Conv `[1, c, l] → [1, c, l_out]` (groups = c,
/// weight `[c, 1, k]`, given dilation, VALID pad) and return the input shape the
/// compiled Conv node actually sees (post-lowering, NCHW).
fn conv_input_dims(c: usize, l: usize, k: usize, dilation: usize) -> Vec<usize> {
    let w_elems = c * k;
    let mut params = HashMap::new();
    params.insert("w".into(), vec![0.02f32; w_elems]);
    params.insert("b".into(), vec![0.0f32; c]);

    let mut init_shapes = HashMap::new();
    init_shapes.insert("w".into(), vec![c, 1, k]);
    init_shapes.insert("b".into(), vec![c]);

    let mut attrs = HashMap::new();
    attrs.insert("kernel_shape".into(), serde_json::json!([k]));
    attrs.insert("strides".into(), serde_json::json!([1]));
    attrs.insert("pads".into(), serde_json::json!([0, 0]));
    attrs.insert("dilations".into(), serde_json::json!([dilation]));
    attrs.insert("group".into(), serde_json::json!(c));

    let nodes = vec![BundleNode {
        name: "dwconv/Conv".into(),
        op: "Conv".into(),
        inputs: vec!["x".into(), "w".into(), "b".into()],
        outputs: vec!["y".into()],
        attrs,
        output_meta: vec![serde_json::json!({"shape": [], "dtype": "float32"})],
    }];

    let manifest = BundleManifest {
        source_onnx: "test".into(),
        inputs: vec![IoMeta {
            name: "x".into(),
            meta: TensorMeta {
                // Genuine NCL: [1, C, L].
                shape: vec![
                    serde_json::json!(1),
                    serde_json::json!(c),
                    serde_json::json!(l),
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
        initializer_count: 2,
        op_histogram: HashMap::from([("Conv".into(), 1)]),
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
    .expect("lower Conv");
    assert_eq!(
        report.stubbed, 0,
        "unexpected stubs: {:?}",
        report.stubbed_nodes
    );

    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let conv = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Conv { .. }))
        .expect("a Conv node");
    graph
        .node(conv.inputs[0])
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

#[test]
fn ncl_depthwise_conv_keeps_channels_when_length_is_typical() {
    // The exact supertonic case: C=256, dilation-2 "same" pad → length 64.
    // Pre-fix this came out `[1, 64, 1, 256]` (c_in=64, c_in/groups=0).
    let dims = conv_input_dims(256, 64, 5, 2);
    assert_eq!(
        dims,
        vec![1, 256, 1, 64],
        "NCL [1,256,64] must stay channels-on-axis-1, got {dims:?}"
    );
}

#[test]
fn ncl_depthwise_conv_typical_length_variants() {
    // Every padded length that collides with a power-of-2 "typical channel" value
    // must be treated as NCL, not transposed. Channels C=256 > length L in all.
    for &l in &[16usize, 24, 32, 48, 64, 80, 96, 128, 160, 192] {
        let dims = conv_input_dims(256, l, 3, 1);
        assert_eq!(
            dims,
            vec![1, 256, 1, l],
            "NCL [1,256,{l}] must stay NCL (length {l} is a typical-channel value), got {dims:?}"
        );
    }
}
