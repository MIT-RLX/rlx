// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bindings' `Trainer` on a package (`Trainer.from_package`): LoRA on a
//! small model's matmuls through the binding's own optimizer, its loss
//! falling, and `save_package` writing an adapter that opens as the trained
//! model — what every language binding reaches.
#![cfg(all(feature = "training", feature = "models"))]

use rlx_bind::runtime::model::Model;
use rlx_bind::runtime::train::Trainer;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::model::{Tensor, rlxp};
use serde_json::json;

#[test]
fn a_package_trains_through_the_bindings_and_saves_its_adapter() {
    let (b, d, c) = (6usize, 5usize, 3usize);
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("forward");
    let x = g.input("x", f32s(&[b, d]));
    let w = g.param("w", f32s(&[d, c]));
    let y = g.mm(x, w);
    g.node_mut(y).name = Some("logits".into());
    g.set_outputs(vec![y]);
    let w0: Vec<f32> = (0..d * c)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0)
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("model.rlxp");
    rlxp::save_package(
        &base,
        &g,
        &[Tensor::f32("w", vec![d, c], &w0)],
        &json!({"name": "m"}),
    )
    .unwrap();

    let xs: Vec<f32> = (0..b * d)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) / 6.0)
        .collect();
    let labels: Vec<f32> = (0..b).map(|r| (r % c) as f32).collect();
    let mut t = Trainer::from_package(
        base.to_str().unwrap(),
        &json!({"function": "forward",
                "loss": {"cross_entropy": {"logits": "logits", "labels": "labels"}},
                "train": {"lora": {"targets": ["w"], "rank": 2, "alpha": 2}},
                "optimizer": {"kind": "adamw", "lr": 0.05}}),
    )
    .unwrap();
    let batch = [("x", xs.as_slice()), ("labels", labels.as_slice())];
    let first = t.evaluate(&batch).unwrap();
    for _ in 0..40 {
        t.step(&batch).unwrap();
    }
    let trained = t.evaluate(&batch).unwrap();
    eprintln!("loss {first} -> {trained}");
    assert!(trained < 0.7 * first, "{first} -> {trained}");

    let adapter = dir.path().join("model-lora.rlxp");
    t.save_package(adapter.to_str().unwrap(), Some("m-lora"))
        .unwrap();
    let mut m = Model::open(adapter.to_str().unwrap(), &json!({"device": "cpu"})).unwrap();
    let outs = m.run(&[("x", xs.as_slice())], &json!({})).unwrap();
    let logits = &outs[0];
    let mut loss = 0.0;
    for (r, &l) in labels.iter().enumerate() {
        let row = &logits[r * c..(r + 1) * c];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        loss += -(row[l as usize] - mx - z.ln());
    }
    let loss = loss / b as f32;
    eprintln!("adapter package: loss {loss}");
    assert!(
        (loss - trained).abs() < 1e-4 * trained.max(1.0),
        "{loss} vs {trained}"
    );
}
