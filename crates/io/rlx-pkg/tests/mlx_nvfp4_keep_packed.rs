// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `import-mlx --keep-packed` for mlx-lm `nvfp4` → `MlxMxfp4` packs.

use std::collections::HashMap;
use std::fs;

use rlx_pkg::{MlxImportOptions, Package, mlx_to_rlxp};
use safetensors::tensor::{Dtype, TensorView};

#[test]
fn import_mlx_nvfp4_keep_packed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"quantization":{"group_size":16,"bits":4,"mode":"nvfp4"}}"#,
    )
    .unwrap();
    let n = 4usize;
    let k = 16usize;
    let w: Vec<u8> = (0..n * (k / 2)).map(|i| ((i * 3) % 256) as u8).collect();
    let scales: Vec<u8> = (0..n).map(|i| 64 + i as u8).collect();
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "lin.weight".into(),
        TensorView::new(Dtype::U8, vec![w.len()], &w).unwrap(),
    );
    tensors.insert(
        "lin.scales".into(),
        TensorView::new(Dtype::U8, vec![n, 1], &scales).unwrap(),
    );
    safetensors::serialize_to_file(&tensors, None, &dir.path().join("model.safetensors")).unwrap();

    let out = dir.path().join("out.rlxp");
    let opts = MlxImportOptions {
        dequant_to_f32: false,
        include_graph: false,
        ..Default::default()
    };
    mlx_to_rlxp(dir.path(), &out, &opts).unwrap();

    let pack = Package::open(&out).unwrap();
    let idx = pack.weights_index().expect("weights");
    let names: Vec<&str> = idx.names().collect();
    assert!(names.contains(&"lin.weight"));
    assert!(names.contains(&"lin.scales"));
    let w_entry = idx.tensors.iter().find(|t| t.name == "lin.weight").unwrap();
    assert!(
        w_entry.scheme.contains("mlx_mxfp4") || w_entry.scheme.contains("mxfp4"),
        "expected MlxMxfp4 scheme, got {}",
        w_entry.scheme
    );
}
