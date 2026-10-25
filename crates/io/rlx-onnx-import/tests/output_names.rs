// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An imported model's outputs keep their ONNX names, through HIR and MIR
//! into the compiled graph, so callers find `logits` by name instead of by
//! position.

use std::path::PathBuf;

use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

#[test]
fn onnx_output_names_reach_the_compiled_graph() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-onnx-conformance/tests/fixtures/dit_adaln.onnx");
    let (hir, _, _, manifest) =
        build_hir_from_onnx_file(&path, ImportOptions::default()).expect("import");
    let want: Vec<String> = manifest.outputs.iter().map(|o| o.name.clone()).collect();
    assert!(!want.is_empty());
    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    for (i, name) in want.iter().enumerate() {
        assert_eq!(graph.output_name(i), Some(name.as_str()));
    }
    let c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(graph);
    let got: Vec<String> = c.outputs().into_iter().map(|o| o.name).collect();
    assert_eq!(got, want);
    assert_eq!(c.output_index(&want[0]), Ok(0));
}
