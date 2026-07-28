// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end: synthesize a `.safetensors` file in a tempdir, convert
//! it to GGUF at Q4_K, parse the GGUF back, and verify the
//! reconstruction stays close to the originals.

#![cfg(feature = "safetensors")]

use std::collections::HashMap;

use rlx_gguf_convert::{Converter, Scheme};
use safetensors::tensor::{Dtype, TensorView};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn synth(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f32 / 2.147e9) * 2.0
        })
        .collect()
}

#[test]
fn safetensors_to_gguf_q4_k() {
    let w = synth(512, 1);
    let b: [f32; 4] = [0.1, -0.2, 0.3, -0.4];

    let w_bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "w".into(),
        TensorView::new(Dtype::F32, vec![2, 256], &w_bytes).unwrap(),
    );
    tensors.insert(
        "b".into(),
        TensorView::new(Dtype::F32, vec![4], &b_bytes).unwrap(),
    );

    let st_path = tempfile::NamedTempFile::new().unwrap();
    safetensors::serialize_to_file(&tensors, None, st_path.path()).unwrap();

    let gguf_path = tempfile::NamedTempFile::new().unwrap();
    let report = Converter::from_safetensors(st_path.path())
        .unwrap()
        .default_scheme(Scheme::Q4_K)
        .skip_quant_for(|_, shape| shape.len() < 2)
        .architecture("test")
        .write_gguf(gguf_path.path())
        .unwrap();
    assert_eq!(report.tensors, 2);
    assert!(report.output_bytes > 0);

    let parsed = rlx_gguf::GgufFile::from_path(gguf_path.path()).unwrap();
    let (w_out, w_shape) = parsed.dequant_f32("w").unwrap();
    assert_eq!(w_shape, vec![2, 256]);
    assert!(cosine(&w, &w_out) > 0.99, "w cosine {}", cosine(&w, &w_out));

    let (b_out, b_shape) = parsed.dequant_f32("b").unwrap();
    assert_eq!(b_shape, vec![4]);
    // Bias was kept at native precision (F16 fallback since
    // shape.len() < 2), so reconstruction should be near-exact.
    for (a, b) in b.iter().zip(&b_out) {
        assert!((a - b).abs() < 1e-3, "bias mismatch {a} vs {b}");
    }
}

#[test]
fn per_tensor_scheme_overrides() {
    let w1 = synth(512, 2);
    let w2 = synth(512, 3);
    let w1_bytes: Vec<u8> = w1.iter().flat_map(|f| f.to_le_bytes()).collect();
    let w2_bytes: Vec<u8> = w2.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "a.weight".into(),
        TensorView::new(Dtype::F32, vec![2, 256], &w1_bytes).unwrap(),
    );
    tensors.insert(
        "b.weight".into(),
        TensorView::new(Dtype::F32, vec![2, 256], &w2_bytes).unwrap(),
    );
    let st_path = tempfile::NamedTempFile::new().unwrap();
    safetensors::serialize_to_file(&tensors, None, st_path.path()).unwrap();

    let gguf_path = tempfile::NamedTempFile::new().unwrap();
    let _report = Converter::from_safetensors(st_path.path())
        .unwrap()
        .default_scheme(Scheme::Q4_K)
        .scheme_for_name("a.weight", Scheme::Q6_K)
        .write_gguf(gguf_path.path())
        .unwrap();

    let parsed = rlx_gguf::GgufFile::from_path(gguf_path.path()).unwrap();
    assert_eq!(
        parsed.get("a.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q6K
    );
    assert_eq!(
        parsed.get("b.weight").unwrap().dtype,
        rlx_gguf::GgmlType::Q4K
    );
}
