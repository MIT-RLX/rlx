// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Synthetic mlx-community dir → `load_path` → packed affine Linear.

use std::collections::HashMap;
use std::fs;

use safetensors::tensor::{Dtype, TensorView};

fn write_synth_affine_dir(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
}"#,
    )
    .unwrap();

    // n=8 out, k=64 in → 1 group, 32 packed bytes/row (4-bit, 2 codes/byte).
    let n = 8usize;
    let k = 64usize;
    let n_groups = 1usize;
    let packs_in_group = k / 2;
    let w_bytes: Vec<u8> = (0..n * n_groups * packs_in_group)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * n_groups)
        .map(|i| 0.02 + 0.001 * (i % 7) as f32)
        .collect();
    let biases: Vec<f32> = (0..n * n_groups)
        .map(|i| -0.05 + 0.001 * (i % 5) as f32)
        .collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|x| x.to_le_bytes()).collect();
    let bias_bytes: Vec<u8> = biases.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "layers.0.weight".into(),
        TensorView::new(Dtype::U8, vec![w_bytes.len()], &w_bytes).unwrap(),
    );
    tensors.insert(
        "layers.0.scales".into(),
        TensorView::new(Dtype::F32, vec![n, n_groups], &scale_bytes).unwrap(),
    );
    tensors.insert(
        "layers.0.biases".into(),
        TensorView::new(Dtype::F32, vec![n, n_groups], &bias_bytes).unwrap(),
    );
    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn synth_affine_dir_load_and_pack() {
    let dir = tempfile::tempdir().unwrap();
    write_synth_affine_dir(dir.path());

    let mut w = rlx_mlx_io::load_path(dir.path()).unwrap();
    assert!(w.quant_scheme().is_some());
    let packed = w
        .take_packed_linear("layers.0.weight")
        .unwrap()
        .expect("affine packed");
    assert_eq!(packed.out_shape, vec![8, 64]);
    assert_eq!(packed.w_q.len(), 8 * 32);
    assert_eq!(packed.scales.len(), 8 * 4); // f32 LE
    assert_eq!(packed.biases.len(), 8 * 4);

    // Dense dequant path also works from a fresh load.
    let dense = rlx_mlx_io::load_path(dir.path())
        .unwrap()
        .into_f32_map()
        .unwrap();
    let layer = dense.get("layers.0").expect("dequantized layers.0");
    assert_eq!(layer.len(), 8 * 64);
    assert!(layer.iter().any(|v| *v != 0.0));
}
