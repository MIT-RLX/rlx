//! mlx-lm-shaped dir → packed Linear chain → Metal vs CPU.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::fs;

use rlx_mlx_io::{build_mlp_chain_graph, collect_packed_linears, load_path, param_bindings_for};
use rlx_runtime::{Device, Session};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_dir(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{"quantization":{"group_size":32,"bits":4,"mode":"affine"}}"#,
    )
    .unwrap();
    let n0 = 32usize;
    let k0 = 64usize;
    let n1 = 16usize;
    let k1 = 32usize;
    let ng0 = k0 / 32;
    let ng1 = k1 / 32;
    let w0: Vec<u8> = (0..n0 * (k0 / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let s0 = f32_bytes(
        &(0..n0 * ng0)
            .map(|i| 0.02 + 0.001 * (i % 7) as f32)
            .collect::<Vec<_>>(),
    );
    let b0 = f32_bytes(
        &(0..n0 * ng0)
            .map(|i| -0.05 + 0.001 * (i % 5) as f32)
            .collect::<Vec<_>>(),
    );
    let w1: Vec<u8> = (0..n1 * (k1 / 2))
        .map(|i| ((i * 41 + 7) % 256) as u8)
        .collect();
    let s1 = f32_bytes(
        &(0..n1 * ng1)
            .map(|i| 0.03 + 0.001 * (i % 7) as f32)
            .collect::<Vec<_>>(),
    );
    let b1 = f32_bytes(
        &(0..n1 * ng1)
            .map(|i| -0.04 + 0.001 * (i % 5) as f32)
            .collect::<Vec<_>>(),
    );
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "model.layers.0.mlp.fc1.weight".into(),
        TensorView::new(Dtype::U8, vec![w0.len()], &w0).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc1.scales".into(),
        TensorView::new(Dtype::F32, vec![n0, ng0], &s0).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc1.biases".into(),
        TensorView::new(Dtype::F32, vec![n0, ng0], &b0).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.weight".into(),
        TensorView::new(Dtype::U8, vec![w1.len()], &w1).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.scales".into(),
        TensorView::new(Dtype::F32, vec![n1, ng1], &s1).unwrap(),
    );
    tensors.insert(
        "model.layers.0.mlp.fc2.biases".into(),
        TensorView::new(Dtype::F32, vec![n1, ng1], &b1).unwrap(),
    );
    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn metal_mlx_lm_shaped_chain() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    write_dir(dir.path());
    let mut w = load_path(dir.path()).unwrap();
    let linears = collect_packed_linears(&mut w).unwrap();
    let g = build_mlp_chain_graph("mlx_lm_metal", &linears, 2).unwrap();
    let x: Vec<f32> = (0..2 * 64).map(|i| ((i as f32) * 0.017).sin()).collect();

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        for b in &linears {
            for (name, bytes, dt) in param_bindings_for(b) {
                c.set_param_typed(&name, &bytes, dt);
            }
        }
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let cpu = run(Device::Cpu);
    let metal = run(Device::Metal);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "metal mlx-lm chain max_abs={max_abs}");
}
