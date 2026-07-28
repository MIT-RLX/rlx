// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! mlx-lm-shaped directory → pack → graph builder → CPU.

use std::collections::HashMap;
use std::fs;

use rlx_mlx_io::{build_mlp_chain_graph, collect_packed_linears, load_path, param_bindings_for};
use rlx_runtime::{Device, Session};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_mlx_lm_shaped(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    // group_size 32 so both layers (k=64 and k=32) are valid.
    fs::write(
        dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "quantization": { "group_size": 32, "bits": 4, "mode": "affine" }
}"#,
    )
    .unwrap();

    // Names sort as fc1 → fc2 so collect_packed_linears chains correctly.
    let n0 = 32usize;
    let k0 = 64usize;
    let n1 = 16usize;
    let k1 = 32usize;
    let ng0 = k0 / 32;
    let ng1 = k1 / 32;
    let w0: Vec<u8> = (0..n0 * (k0 / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let s0: Vec<f32> = (0..n0 * ng0)
        .map(|i| 0.02 + 0.001 * (i % 7) as f32)
        .collect();
    let b0: Vec<f32> = (0..n0 * ng0)
        .map(|i| -0.05 + 0.001 * (i % 5) as f32)
        .collect();
    let w1: Vec<u8> = (0..n1 * (k1 / 2))
        .map(|i| ((i * 41 + 7) % 256) as u8)
        .collect();
    let s1: Vec<f32> = (0..n1 * ng1)
        .map(|i| 0.03 + 0.001 * (i % 7) as f32)
        .collect();
    let b1: Vec<f32> = (0..n1 * ng1)
        .map(|i| -0.04 + 0.001 * (i % 5) as f32)
        .collect();

    let s0b = f32_bytes(&s0);
    let b0b = f32_bytes(&b0);
    let s1b = f32_bytes(&s1);
    let b1b = f32_bytes(&b1);

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "model.layers.0.mlp.fc1.weight".into(),
        TensorView::new(Dtype::U8, vec![w0.len()], &w0).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc1.scales".into(),
        TensorView::new(Dtype::F32, vec![n0, ng0], &s0b).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc1.biases".into(),
        TensorView::new(Dtype::F32, vec![n0, ng0], &b0b).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.weight".into(),
        TensorView::new(Dtype::U8, vec![w1.len()], &w1).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.scales".into(),
        TensorView::new(Dtype::F32, vec![n1, ng1], &s1b).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.biases".into(),
        TensorView::new(Dtype::F32, vec![n1, ng1], &b1b).unwrap(),
    );
    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn mlx_lm_shaped_chain_cpu() {
    let dir = tempfile::tempdir().unwrap();
    write_mlx_lm_shaped(dir.path());

    let mut w = load_path(dir.path()).unwrap();
    let linears = collect_packed_linears(&mut w).unwrap();
    assert_eq!(linears.len(), 2);
    assert!(linears[0].name.ends_with("fc1"));
    assert!(linears[1].name.ends_with("fc2"));
    assert_eq!(linears[0].packed.out_shape, vec![32, 64]);
    assert_eq!(linears[1].packed.out_shape, vec![16, 32]);

    let g = build_mlp_chain_graph("mlx_lm_mlp", &linears, 2).unwrap();
    let x: Vec<f32> = (0..2 * 64).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut c = Session::new(Device::Cpu).compile(g);
    for b in &linears {
        for (name, bytes, dt) in param_bindings_for(b) {
            c.set_param_typed(&name, &bytes, dt);
        }
    }
    let out = c.run(&[("x", x.as_slice())]).remove(0);
    assert_eq!(out.len(), 2 * 16);
    assert!(out.iter().any(|v| *v != 0.0));
}
