//! Synthetic mlx-community dir → load → `Op::DequantMatMul` on CPU (+ Metal).

use std::collections::HashMap;
use std::fs;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_synth_affine_dir(dir: &std::path::Path) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
}"#,
    )
    .unwrap();

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
    let scale_bytes = f32_bytes(&scales);
    let bias_bytes = f32_bytes(&biases);

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
    (w_bytes, scales, biases)
}

#[test]
fn synth_affine_dir_session_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let _ = write_synth_affine_dir(dir.path());

    let mut w = rlx_mlx_io::load_path(dir.path()).unwrap();
    let packed = w
        .take_packed_linear("layers.0.weight")
        .unwrap()
        .expect("affine packed");
    let (n, k) = (packed.out_shape[0], packed.out_shape[1]);
    let m = 2usize;
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("mlx_e2e");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_p = g.param("w", Shape::new(&[packed.w_q.len()], DType::U8));
    let s_p = g.param("scale", Shape::new(&[n, k / 64], DType::F32));
    let z_p = g.param("zp", Shape::new(&[n, k / 64], DType::F32));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: packed.scheme,
        },
        vec![x_in, w_p, s_p, z_p],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param_typed("w", &packed.w_q, DType::U8);
    c.set_param_typed("scale", &packed.scales, packed.scale_dtype());
    c.set_param_typed("zp", &packed.biases, packed.bias_dtype());
    let out = c.run(&[("x", x.as_slice())]).remove(0);
    assert_eq!(out.len(), m * n);
    assert!(out.iter().any(|v| *v != 0.0));
}
