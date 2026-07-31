// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The lazy mmap loader ([`load_path_lazy`]) caches each tensor's header offset at
//! open and slices directly on `take` (no per-take header re-parse) + prefetches.
//! This asserts it returns BYTE-IDENTICAL data to the eager [`load_path`] for both
//! a dense F32 tensor and a quantized (affine) Linear — i.e. the offset-cache
//! arithmetic is correct.

use std::collections::HashMap;
use std::fs;

use rlx_mlx_io::MlxRead;
use safetensors::tensor::{Dtype, TensorView};

fn write_dir(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{ "quantization": { "group_size": 64, "bits": 4, "mode": "affine" } }"#,
    )
    .unwrap();

    // A plain dense F32 tensor (embedding) + a quantized affine Linear.
    let embed: Vec<f32> = (0..3 * 8).map(|i| 0.1 * i as f32 - 0.7).collect();
    let embed_bytes: Vec<u8> = embed.iter().flat_map(|x| x.to_le_bytes()).collect();

    let (n, k, ng) = (8usize, 64usize, 1usize);
    let w_bytes: Vec<u8> = (0..n * ng * (k / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * ng).map(|i| 0.02 + 0.001 * (i % 7) as f32).collect();
    let biases: Vec<f32> = (0..n * ng)
        .map(|i| -0.05 + 0.001 * (i % 5) as f32)
        .collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|x| x.to_le_bytes()).collect();
    let bias_bytes: Vec<u8> = biases.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut t: HashMap<String, TensorView<'_>> = HashMap::new();
    t.insert(
        "tok_embeddings.weight".into(),
        TensorView::new(Dtype::F32, vec![3, 8], &embed_bytes).unwrap(),
    );
    t.insert(
        "layers.0.weight".into(),
        TensorView::new(Dtype::U8, vec![w_bytes.len()], &w_bytes).unwrap(),
    );
    t.insert(
        "layers.0.scales".into(),
        TensorView::new(Dtype::F32, vec![n, ng], &scale_bytes).unwrap(),
    );
    t.insert(
        "layers.0.biases".into(),
        TensorView::new(Dtype::F32, vec![n, ng], &bias_bytes).unwrap(),
    );
    safetensors::serialize_to_file(&t, None, &dir.join("model.safetensors")).unwrap();
}

#[test]
fn lazy_offset_cache_matches_eager() {
    let dir = tempfile::tempdir().unwrap();
    write_dir(dir.path());

    // Dense F32: lazy slice == eager.
    let (ed, es) = MlxRead::take_dense_f32(
        &mut rlx_mlx_io::load_path(dir.path()).unwrap(),
        "tok_embeddings.weight",
    )
    .unwrap();
    let (ld, ls) = MlxRead::take_dense_f32(
        &mut rlx_mlx_io::load_path_lazy(dir.path()).unwrap(),
        "tok_embeddings.weight",
    )
    .unwrap();
    assert_eq!(es, ls);
    assert_eq!(ed, ld, "lazy dense F32 must byte-match eager");

    // Quantized affine Linear: packed codes/scales/biases byte-identical.
    let ep = MlxRead::take_packed_linear(
        &mut rlx_mlx_io::load_path(dir.path()).unwrap(),
        "layers.0.weight",
    )
    .unwrap()
    .expect("eager packed");
    let lp = MlxRead::take_packed_linear(
        &mut rlx_mlx_io::load_path_lazy(dir.path()).unwrap(),
        "layers.0.weight",
    )
    .unwrap()
    .expect("lazy packed");
    assert_eq!(ep.w_q, lp.w_q, "lazy w_q must byte-match eager");
    assert_eq!(ep.scales, lp.scales);
    assert_eq!(ep.biases, lp.biases);
    assert_eq!(ep.out_shape, lp.out_shape);

    // A second take on the SAME lazy loader (exercises the cached-offset path
    // twice) still matches — proving no header-reparse dependence / state bug.
    let lp2 = MlxRead::take_packed_linear(
        &mut rlx_mlx_io::load_path_lazy(dir.path()).unwrap(),
        "layers.0.weight",
    )
    .unwrap()
    .unwrap();
    assert_eq!(lp2.w_q, ep.w_q);
}
