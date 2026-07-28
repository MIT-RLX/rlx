// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `import-mlx --keep-packed` retains MLX scheme triples.

use std::collections::HashMap;
use std::fs;

use rlx_pkg::{MlxImportOptions, mlx_to_rlxp};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn import_mlx_keep_packed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#,
    )
    .unwrap();
    let n = 4usize;
    let k = 64usize;
    let w: Vec<u8> = (0..n * (k / 2)).map(|i| ((i * 3) % 256) as u8).collect();
    let scales = f32_bytes(&(0..n).map(|i| 0.01 * (i + 1) as f32).collect::<Vec<_>>());
    let biases = f32_bytes(&vec![0.0f32; n]);
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "lin.weight".into(),
        TensorView::new(Dtype::U8, vec![w.len()], &w).unwrap(),
    );
    tensors.insert(
        "lin.scales".into(),
        TensorView::new(Dtype::F32, vec![n, 1], &scales).unwrap(),
    );
    tensors.insert(
        "lin.biases".into(),
        TensorView::new(Dtype::F32, vec![n, 1], &biases).unwrap(),
    );
    safetensors::serialize_to_file(&tensors, None, &dir.path().join("model.safetensors")).unwrap();

    let out = dir.path().join("out.rlxp");
    let opts = MlxImportOptions {
        dequant_to_f32: false,
        include_graph: true,
        ..Default::default()
    };
    mlx_to_rlxp(dir.path(), &out, &opts).unwrap();

    let pkg = rlx_pkg::Package::open(&out).unwrap();
    let idx = pkg.weights_index().expect("weights index");
    let names: Vec<_> = idx.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"lin.weight"), "names={names:?}");
    assert!(names.contains(&"lin.scales"));
    let wmeta = pkg.weight_entry("lin.weight").unwrap();
    assert!(
        wmeta.scheme.starts_with("mlx_affine"),
        "scheme={}",
        wmeta.scheme
    );
}
