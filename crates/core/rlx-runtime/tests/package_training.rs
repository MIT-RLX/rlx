// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A package trained as it is: a function, a loss, the params (all, or LoRA
//! factors on its matmuls), AdamW. Its loss falls; the LoRA trainer's
//! adapter package, opened, gives the trained model's outputs — the update
//! laid out as each tensor is stored, a referenced, transposed checkpoint
//! tensor included.
#![cfg(all(feature = "model-io", feature = "training", feature = "cpu"))]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::model::trainer::open_trainer;
use rlx_runtime::model::{Tensor, open_source, rlxp};
use serde_json::json;

const B: usize = 8;
const D: usize = 4;
const H: usize = 16;
const C: usize = 3;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `logits = relu(x·w1 + b1)·w2`.
fn mlp() -> Graph {
    let mut g = Graph::new("forward");
    let x = g.input("x", f32s(&[B, D]));
    let w1 = g.param("w1", f32s(&[D, H]));
    let b1 = g.param("b1", f32s(&[1, H]));
    let w2 = g.param("w2", f32s(&[H, C]));
    let h = g.mm(x, w1);
    let h = g.add(h, b1);
    let h = g.relu(h);
    g.node_mut(h).name = Some("hidden".into());
    let y = g.mm(h, w2);
    g.node_mut(y).name = Some("logits".into());
    g.set_outputs(vec![y]);
    g
}

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn batch() -> (Vec<f32>, Vec<f32>) {
    let x = rand(B * D, 3);
    // A label each row's largest feature names (mod the classes).
    let labels = (0..B)
        .map(|r| {
            let row = &x[r * D..(r + 1) * D];
            let best = (0..D).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            (best % C) as f32
        })
        .collect();
    (x, labels)
}

fn cross_entropy(logits: &[f32], labels: &[f32]) -> f32 {
    let mut total = 0.0;
    for (r, &l) in labels.iter().enumerate() {
        let row = &logits[r * C..(r + 1) * C];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|v| (v - m).exp()).sum();
        total += -(row[l as usize] - m - z.ln());
    }
    total / labels.len() as f32
}

#[test]
fn a_package_trains_and_its_lora_adapter_opens_as_the_trained_model() {
    let dir = tempfile::tempdir().unwrap();
    // The head is a checkpoint tensor stored `[out, in]`, referenced
    // transposed; the rest held.
    let head = rand(C * H, 11);
    let bytes: Vec<u8> = head.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![C, H], &bytes).unwrap();
    let st = safetensors::serialize(vec![("head.weight".to_string(), view)], None).unwrap();
    std::fs::write(dir.path().join("ckpt.safetensors"), st).unwrap();
    let base = dir.path().join("mlp.rlxp");
    rlxp::save_package(
        &base,
        &mlp(),
        &[
            Tensor::f32("w1", vec![D, H], &rand(D * H, 5)),
            Tensor::f32("b1", vec![1, H], &rand(H, 7)),
        ],
        &json!({
            "name": "mlp",
            "weights": [{"uri": "safetensors://ckpt.safetensors",
                         "map": {"w2": {"tensor": "head.weight", "transpose": true}}}],
            "train": {"function": "forward",
                      "loss": {"cross_entropy": {"logits": "logits", "labels": "labels"}}},
        }),
    )
    .unwrap();
    let (x, labels) = batch();
    let feed = || {
        vec![
            Tensor::f32("x", vec![B, D], &x),
            Tensor::f32("labels", vec![B], &labels),
        ]
    };

    // Every param: the loss falls.
    let mut full = open_trainer(
        base.to_str().unwrap(),
        &json!({"train": "all", "optimizer": {"lr": 0.05}}),
    )
    .unwrap();
    let first = full.loss(&feed()).unwrap();
    for _ in 0..150 {
        full.step(&feed()).unwrap();
    }
    let trained = full.loss(&feed()).unwrap();
    eprintln!("all params: loss {first} -> {trained}");
    assert!(trained < 0.5 * first, "{first} -> {trained}");
    // Saved, it opens as the trained model (its head, a reference before,
    // now held; the rest of the block as it was).
    let saved = dir.path().join("saved").join("mlp-trained.rlxp");
    std::fs::create_dir_all(saved.parent().unwrap()).unwrap();
    full.save(&saved).unwrap();
    let mut m = open_source(saved.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let logits = m
        .run(&[Tensor::f32("x", vec![B, D], &x)], &json!({}))
        .unwrap()[0]
        .to_f32()
        .unwrap();
    let reopened = cross_entropy(&logits, &labels);
    eprintln!("saved package: loss {reopened}");
    assert!((reopened - trained).abs() < 1e-4, "{reopened} vs {trained}");

    // LoRA on both matmuls: only the factors train, the base unchanged.
    let mut lora = open_trainer(
        base.to_str().unwrap(),
        &json!({"train": {"lora": {"targets": ["w*"], "rank": 2, "alpha": 4}},
                "optimizer": {"lr": 0.05}}),
    )
    .unwrap();
    let mut names = lora.trainable();
    names.sort();
    assert_eq!(
        names,
        ["lora.0.down", "lora.0.up", "lora.1.down", "lora.1.up"]
    );
    let start = lora.loss(&feed()).unwrap();
    assert!(
        (start - first).abs() < 1e-5,
        "LoRA changed the model at step 0"
    );
    // Few steps: the loss still large, so the adapter's check below sees a
    // wrongly laid-out update.
    for _ in 0..20 {
        lora.step(&feed()).unwrap();
    }
    let adapted = lora.loss(&feed()).unwrap();
    eprintln!("lora: loss {start} -> {adapted}");
    assert!(adapted < 0.8 * start, "{start} -> {adapted}");

    // The adapter package: the base with the factors merged.
    let adapter = dir.path().join("mlp-lora.rlxp");
    lora.save_adapter(&adapter, "mlp-lora").unwrap();
    let mut m = open_source(adapter.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let logits = m
        .run(&[Tensor::f32("x", vec![B, D], &x)], &json!({}))
        .unwrap()[0]
        .to_f32()
        .unwrap();
    let merged = cross_entropy(&logits, &labels);
    eprintln!("adapter package: loss {merged}");
    assert!(
        (merged - adapted).abs() < 1e-4 * adapted.max(1.0),
        "{merged} vs {adapted}"
    );
}

/// A sparse autoencoder on the hidden layer: it trains with the base frozen,
/// and its package encodes and reconstructs that activation.
#[test]
fn a_package_trains_a_sparse_autoencoder_on_an_activation() {
    let dir = tempfile::tempdir().unwrap();
    let (w1, b1) = (rand(D * H, 5), rand(H, 7));
    let base = dir.path().join("mlp.rlxp");
    rlxp::save_package(
        &base,
        &mlp(),
        &[
            Tensor::f32("w1", vec![D, H], &w1),
            Tensor::f32("b1", vec![1, H], &b1),
            Tensor::f32("w2", vec![H, C], &rand(H * C, 11)),
        ],
        &json!({"name": "mlp"}),
    )
    .unwrap();
    let (x, _) = batch();
    let feed = || vec![Tensor::f32("x", vec![B, D], &x)];
    let mut t = open_trainer(
        base.to_str().unwrap(),
        &json!({"function": "forward",
                "train": {"sae": {"tap": "hidden", "width": 32, "l1": 1e-3}},
                "optimizer": {"lr": 0.01}}),
    )
    .unwrap();
    let mut names = t.trainable();
    names.sort();
    assert_eq!(names, ["sae.b_dec", "sae.b_enc", "sae.w_dec", "sae.w_enc"]);
    let first = t.loss(&feed()).unwrap();
    for _ in 0..300 {
        t.step(&feed()).unwrap();
    }
    let trained = t.loss(&feed()).unwrap();
    eprintln!("sae: loss {first} -> {trained}");
    assert!(trained < 0.2 * first, "{first} -> {trained}");

    // The SAE package on the activation it was trained on.
    let out = dir.path().join("mlp-sae.rlxp");
    t.save_sae(&out, "mlp-sae").unwrap();
    let mut hidden = vec![0f32; B * H];
    for r in 0..B {
        for j in 0..H {
            let v: f32 = (0..D).map(|k| x[r * D + k] * w1[k * H + j]).sum::<f32>() + b1[j];
            hidden[r * H + j] = v.max(0.0);
        }
    }
    let mut m = open_source(out.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let outs = m
        .run(&[Tensor::f32("h", vec![B, H], &hidden)], &json!({}))
        .unwrap();
    let features = outs[0].to_f32().unwrap();
    let recon = outs[1].to_f32().unwrap();
    assert_eq!(features.len(), B * 32);
    assert!(features.iter().all(|v| *v >= 0.0));
    let mse: f32 = recon
        .iter()
        .zip(&hidden)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / hidden.len() as f32;
    eprintln!("sae package: reconstruction mse {mse}");
    assert!(mse <= trained, "{mse} > {trained}");

    // Spliced into the model it was trained on (open option `sae`).
    let run = |options: serde_json::Value| {
        let mut m = open_source(base.to_str().unwrap(), options).unwrap();
        m.run(&[Tensor::f32("x", vec![B, D], &x)], &json!({}))
            .unwrap()
            .iter()
            .map(|t| t.to_f32().unwrap())
            .collect::<Vec<_>>()
    };
    let plain = run(json!({"device": "cpu"}));
    let featured =
        run(json!({"device": "cpu", "sae": {"path": out.to_str().unwrap(), "mode": "features"}}));
    assert_eq!(featured.len(), 2, "the logits and the features");
    assert_eq!(
        featured[0], plain[0],
        "`features` leaves the model as it was"
    );
    let diff = featured[1]
        .iter()
        .zip(&features)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff < 1e-5,
        "the spliced SAE's features are its package's: {diff}"
    );
    let replaced = run(json!({"device": "cpu", "sae": out.to_str().unwrap()}));
    let moved = replaced[0]
        .iter()
        .zip(&plain[0])
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("replaced: logits moved by at most {moved}");
    assert!(moved > 0.0 && moved < 0.05, "{moved}");
}

/// A token model: its integer inputs (ids) fed as the trainer's f32 batch,
/// cast back in the graph; the next-token loss falls.
#[test]
fn a_package_with_token_inputs_trains() {
    let (v, h, n) = (12usize, 8usize, 16usize);
    let mut g = Graph::new("lm");
    let ids = g.input("ids", Shape::new(&[n], DType::I32));
    let emb = g.param("embed", f32s(&[v, h]));
    let x = g.gather_(emb, ids, 0);
    let w = g.param("head", f32s(&[h, v]));
    let y = g.mm(x, w);
    g.node_mut(y).name = Some("logits".into());
    g.set_outputs(vec![y]);
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("lm.rlxp");
    rlxp::save_package(
        &base,
        &g,
        &[
            Tensor::f32("embed", vec![v, h], &rand(v * h, 1)),
            Tensor::f32("head", vec![h, v], &rand(h * v, 2)),
        ],
        &json!({"name": "lm"}),
    )
    .unwrap();
    // A cycle: each token's successor is the next id.
    let seq: Vec<i32> = (0..n as i32).map(|i| i % v as i32).collect();
    let next: Vec<f32> = seq.iter().map(|&t| ((t + 1) % v as i32) as f32).collect();
    let feed = || {
        vec![
            Tensor::new(
                "ids",
                DType::I32,
                vec![n],
                seq.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            Tensor::f32("labels", vec![n], &next),
        ]
    };
    let mut t = open_trainer(
        base.to_str().unwrap(),
        &json!({"function": "lm", "train": "all", "optimizer": {"lr": 0.1},
                "loss": {"cross_entropy": {"logits": "logits", "labels": "labels"}}}),
    )
    .unwrap();
    let first = t.loss(&feed()).unwrap();
    for _ in 0..200 {
        t.step(&feed()).unwrap();
    }
    let trained = t.loss(&feed()).unwrap();
    eprintln!("token model: loss {first} -> {trained}");
    assert!(trained < 0.1 * first, "{first} -> {trained}");
}
