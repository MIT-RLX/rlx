//! `import-mlx --keep-packed` → `.rlxp` → `compile_rlxp_bind_params` → run.

use std::collections::HashMap;
use std::fs;

use rlx_pkg::{MlxImportOptions, mlx_to_rlxp};
use rlx_runtime::{Device, Session, compile_rlxp_bind_params};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn keep_packed_rlxp_bind_and_run() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#,
    )
    .unwrap();
    let n = 8usize;
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

    let session = Session::new(Device::Cpu);
    let mut compiled = compile_rlxp_bind_params(&session, &out).unwrap();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let y = compiled.run(&[("lin_x", x.as_slice())]).remove(0);
    assert_eq!(y.len(), n);
    assert!(y.iter().any(|v| *v != 0.0));
}
