// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Regression: 1-D ONNX Conv on `[N,C,L,1]` (length on H) with a rank-3 weight
//! must canonicalize to `[N,C,1,L]`, convolve along W, and collapse to NCL with
//! the real `L_out` — not collapse length to 1 (Kokoro/StyleTTS2 `noise_convs`).

use std::collections::HashMap;

use rlx_ir::DType;
use rlx_onnx_import::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use rlx_onnx_import::tensor_data::TypedParams;
use rlx_onnx_import::{ImportOptions, build_hir_from_parts};

#[test]
fn rank4_length_on_h_1d_conv_collapses_to_ncl_with_time() {
    // Input `[1,4,61,1]`, k=12, stride=6, pads=[3,3]:
    // L_out = (61 + 3 + 3 − 11 − 1) / 6 + 1 = 10.
    let l_in = 61usize;
    let cin = 4usize;
    let cout = 8usize;
    let k = 12usize;
    let l_out = (l_in + 3 + 3 - (k - 1) - 1) / 6 + 1;
    assert_eq!(l_out, 10);

    let w_elems = cout * cin * k;
    let mut params = HashMap::new();
    params.insert("w".into(), vec![0.01f32; w_elems]);
    params.insert("b".into(), vec![0.0f32; cout]);

    let mut init_shapes = HashMap::new();
    init_shapes.insert("w".into(), vec![cout, cin, k]);
    init_shapes.insert("b".into(), vec![cout]);

    let mut attrs = HashMap::new();
    attrs.insert("kernel_shape".into(), serde_json::json!([k]));
    attrs.insert("strides".into(), serde_json::json!([6]));
    attrs.insert("pads".into(), serde_json::json!([3, 3]));
    attrs.insert("dilations".into(), serde_json::json!([1]));
    attrs.insert("group".into(), serde_json::json!(1));

    let nodes = vec![BundleNode {
        name: "noise_convs/Conv".into(),
        op: "Conv".into(),
        inputs: vec!["x".into(), "w".into(), "b".into()],
        outputs: vec!["y".into()],
        attrs,
        // Empty meta forces recompute from operands (same as Kokoro noise_convs).
        output_meta: vec![serde_json::json!({"shape": [], "dtype": "float32"})],
    }];

    let manifest = BundleManifest {
        source_onnx: "test".into(),
        inputs: vec![IoMeta {
            name: "x".into(),
            meta: TensorMeta {
                shape: vec![
                    serde_json::json!(1),
                    serde_json::json!(cin),
                    serde_json::json!(l_in),
                    serde_json::json!(1),
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

    let out_id = *hir.outputs.last().expect("output");
    let shape = hir.node(out_id).shape.clone();
    assert_eq!(shape.dtype(), DType::F32);
    assert_eq!(
        shape.dims(),
        &[
            rlx_ir::Dim::Static(1),
            rlx_ir::Dim::Static(cout),
            rlx_ir::Dim::Static(l_out)
        ],
        "expected NCL [1,{cout},{l_out}], got {shape:?}"
    );
}
