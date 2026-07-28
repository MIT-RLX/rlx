// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX → RLXP with optional executable graph (`--features onnx`).

use rlx_bake::{OnnxImportOptions, onnx_to_rlxp};
use rlx_pkg::Package;
use rlx_runtime::{Device, Session};
use std::path::PathBuf;

fn dit_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../rlx-onnx-conformance/tests/fixtures/dit_adaln.onnx")
}

#[test]
fn onnx_rlxp_embeds_executable_graph_by_default() {
    let onnx = dit_fixture();
    assert!(onnx.is_file(), "missing {}", onnx.display());
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dit.rlxp");
    onnx_to_rlxp(&onnx, &out, &OnnxImportOptions::default()).expect("onnx_to_rlxp");

    let pack = Package::open(&out).unwrap();
    assert!(pack.has_graph());
    assert!(!pack.weights_index().unwrap().tensors.is_empty());
    assert!(pack.sidecar("io").is_ok());

    let g = pack.graph().expect("graph");
    let _ = Session::new(Device::Cpu).compile(g);
}

#[test]
fn onnx_rlxp_graph_optional() {
    let onnx = dit_fixture();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("dit_w.rlxp");
    onnx_to_rlxp(
        &onnx,
        &out,
        &OnnxImportOptions {
            include_graph: false,
            ..OnnxImportOptions::default()
        },
    )
    .unwrap();
    let pack = Package::open(&out).unwrap();
    assert!(!pack.has_graph());
    assert!(pack.graph().is_err());
    assert!(!pack.weights_index().unwrap().tensors.is_empty());
}
