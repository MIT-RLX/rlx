// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Shared bits for the MNIST bake / encrypt demos.

#![allow(dead_code)] // each example only uses a subset

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use std::path::PathBuf;

pub const IN: usize = 784;
pub const HIDDEN: usize = 64;
pub const OUT: usize = 10;
pub const BATCH: usize = 32;

pub struct Weights {
    pub w1: Vec<f32>, // [IN, HIDDEN]
    pub b1: Vec<f32>, // [HIDDEN]
    pub w2: Vec<f32>, // [HIDDEN, OUT]
    pub b2: Vec<f32>, // [OUT]
}

impl Weights {
    pub fn f32_bytes(&self) -> usize {
        (self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()) * 4
    }
}

/// Inference MLP: `logits = relu(x @ w1 + b1) @ w2 + b2`.
pub fn build_infer_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mnist_mlp");
    let x = g.input("x", Shape::new(&[BATCH, IN], f));
    let w1 = g.param("w1", Shape::new(&[IN, HIDDEN], f));
    let b1 = g.param("b1", Shape::new(&[HIDDEN], f));
    let w2 = g.param("w2", Shape::new(&[HIDDEN, OUT], f));
    let b2 = g.param("b2", Shape::new(&[OUT], f));

    let h = g.add_node(Op::MatMul, vec![x, w1], Shape::new(&[BATCH, HIDDEN], f));
    let h = g.binary(BinaryOp::Add, h, b1, Shape::new(&[BATCH, HIDDEN], f));
    let h = g.activation(Activation::Relu, h, Shape::new(&[BATCH, HIDDEN], f));
    let y = g.add_node(Op::MatMul, vec![h, w2], Shape::new(&[BATCH, OUT], f));
    let y = g.binary(BinaryOp::Add, y, b2, Shape::new(&[BATCH, OUT], f));
    g.set_outputs(vec![y]);
    g
}

pub fn he_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (s >> 33) as f32 / (u32::MAX as f32);
            // Box-Muller-ish: two uniforms → approx normal via inverse CDF lite
            let u2 = {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 33) as f32 / (u32::MAX as f32)
            };
            let z = (-2.0 * (u.max(1e-6)).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * scale
        })
        .collect()
}

pub fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn add_bias_rows(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += bias[c];
        }
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn softmax_ce_grad(
    logits: &[f32],
    labels: &[usize],
    batch: usize,
    classes: usize,
) -> (f32, Vec<f32>) {
    let mut loss = 0.0;
    let mut dlogits = vec![0.0; batch * classes];
    for b in 0..batch {
        let row = &logits[b * classes..(b + 1) * classes];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = [0.0f32; OUT];
        let mut sum = 0.0;
        for c in 0..classes {
            exps[c] = (row[c] - max).exp();
            sum += exps[c];
        }
        let y = labels[b];
        loss -= (exps[y] / sum).ln();
        for c in 0..classes {
            let p = exps[c] / sum;
            dlogits[b * classes + c] = p - if c == y { 1.0 } else { 0.0 };
        }
    }
    (loss / batch as f32, dlogits)
}

/// Host SGD on a tiny MLP. Returns trained weights + last train accuracy.
pub fn train_sgd(
    images: &[f32],
    labels: &[usize],
    n: usize,
    epochs: usize,
    lr: f32,
) -> (Weights, f32) {
    train_sgd_ex(images, labels, n, epochs, lr, None)
}

/// Like [`train_sgd`], optionally freezing `w1` (e.g. after ternarize).
pub fn train_sgd_ex(
    images: &[f32],
    labels: &[usize],
    n: usize,
    epochs: usize,
    lr: f32,
    fixed: Option<Weights>,
) -> (Weights, f32) {
    let freeze_w1 = fixed.is_some();
    let (mut w1, mut b1, mut w2, mut b2) = if let Some(w) = fixed {
        (w.w1, w.b1, w.w2, w.b2)
    } else {
        (
            he_init(IN, HIDDEN, 1),
            vec![0.1; HIDDEN],
            he_init(HIDDEN, OUT, 2),
            zeros(OUT),
        )
    };

    let mut last_acc = 0.0;
    let mut order: Vec<usize> = (0..n / BATCH).collect();
    let mut rng = 12345u64;
    for epoch in 0..epochs {
        // Shuffle batch starts.
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut correct = 0usize;
        let mut seen = 0usize;
        let mut loss_sum = 0.0;
        let mut batches = 0usize;
        for &bi in &order {
            let off = bi * BATCH;
            let x = &images[off * IN..(off + BATCH) * IN];
            let y = &labels[off..off + BATCH];

            let mut h = matmul(x, &w1, BATCH, IN, HIDDEN);
            add_bias_rows(&mut h, &b1, BATCH, HIDDEN);
            let h_pre = h.clone();
            relu_inplace(&mut h);
            let mut logits = matmul(&h, &w2, BATCH, HIDDEN, OUT);
            add_bias_rows(&mut logits, &b2, BATCH, OUT);

            for b in 0..BATCH {
                let row = &logits[b * OUT..(b + 1) * OUT];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if pred == y[b] {
                    correct += 1;
                }
            }
            seen += BATCH;

            let (loss, dlogits) = softmax_ce_grad(&logits, y, BATCH, OUT);
            loss_sum += loss;
            batches += 1;

            let mut dw2 = vec![0.0; HIDDEN * OUT];
            let mut db2 = vec![0.0; OUT];
            for i in 0..HIDDEN {
                for j in 0..OUT {
                    let mut acc = 0.0;
                    for b in 0..BATCH {
                        acc += h[b * HIDDEN + i] * dlogits[b * OUT + j];
                    }
                    dw2[i * OUT + j] = acc / BATCH as f32;
                }
            }
            for j in 0..OUT {
                let mut acc = 0.0;
                for b in 0..BATCH {
                    acc += dlogits[b * OUT + j];
                }
                db2[j] = acc / BATCH as f32;
            }
            let mut dh = vec![0.0; BATCH * HIDDEN];
            for b in 0..BATCH {
                for i in 0..HIDDEN {
                    let mut acc = 0.0;
                    for j in 0..OUT {
                        acc += dlogits[b * OUT + j] * w2[i * OUT + j];
                    }
                    dh[b * HIDDEN + i] = if h_pre[b * HIDDEN + i] > 0.0 {
                        acc
                    } else {
                        0.0
                    };
                }
            }

            for (w, d) in w2.iter_mut().zip(dw2.iter()) {
                *w -= lr * *d;
            }
            for (w, d) in b2.iter_mut().zip(db2.iter()) {
                *w -= lr * *d;
            }
            if !freeze_w1 {
                let mut dw1 = vec![0.0; IN * HIDDEN];
                let mut db1 = vec![0.0; HIDDEN];
                for i in 0..IN {
                    for j in 0..HIDDEN {
                        let mut acc = 0.0;
                        for b in 0..BATCH {
                            acc += x[b * IN + i] * dh[b * HIDDEN + j];
                        }
                        dw1[i * HIDDEN + j] = acc / BATCH as f32;
                    }
                }
                for j in 0..HIDDEN {
                    let mut acc = 0.0;
                    for b in 0..BATCH {
                        acc += dh[b * HIDDEN + j];
                    }
                    db1[j] = acc / BATCH as f32;
                }
                for (w, d) in w1.iter_mut().zip(dw1.iter()) {
                    *w -= lr * *d;
                }
                for (w, d) in b1.iter_mut().zip(db1.iter()) {
                    *w -= lr * *d;
                }
            } else {
                let mut db1 = vec![0.0; HIDDEN];
                for j in 0..HIDDEN {
                    let mut acc = 0.0;
                    for b in 0..BATCH {
                        acc += dh[b * HIDDEN + j];
                    }
                    db1[j] = acc / BATCH as f32;
                }
                for (w, d) in b1.iter_mut().zip(db1.iter()) {
                    *w -= lr * *d;
                }
            }
        }
        last_acc = correct as f32 / seen.max(1) as f32;
        eprintln!(
            "  epoch {epoch}: loss={:.4} acc={:.1}%{}",
            loss_sum / batches.max(1) as f32,
            last_acc * 100.0,
            if freeze_w1 {
                " (w1 frozen ternary)"
            } else {
                ""
            }
        );
    }

    (Weights { w1, b1, w2, b2 }, last_acc)
}

/// Ternarize by magnitude: top ~⅓ of |w| → ±1 by sign, rest → 0 (BitNet-ish).
pub fn ternarize(w: &[f32]) -> Vec<f32> {
    let mut abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = abs[abs.len() * 2 / 3];
    w.iter()
        .map(|&v| {
            if v.abs() < thr {
                0.0
            } else if v > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Synthetic digit-ish blobs (no download required).
pub fn make_synthetic(n: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut s = seed;
    let mut images = Vec::with_capacity(n * IN);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let label = i % OUT;
        labels.push(label);
        for p in 0..IN {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (s >> 33) as f32 / (u32::MAX as f32) * 0.3;
            // Class-conditional mean on a stripe of pixels.
            let stripe = (p / 28) % OUT == label;
            let v = if stripe { 0.7 } else { 0.05 } + noise;
            images.push(v);
        }
    }
    (images, labels)
}

pub fn try_load_mnist(n: usize) -> Option<(Vec<f32>, Vec<usize>)> {
    let dir = mnist_raw_dir()?;
    let img_path = dir.join("train-images-idx3-ubyte");
    let lbl_path = dir.join("train-labels-idx1-ubyte");
    if !img_path.is_file() || !lbl_path.is_file() {
        return None;
    }
    let imgs = std::fs::read(&img_path).ok()?;
    let lbls = std::fs::read(&lbl_path).ok()?;
    let total = u32::from_be_bytes(imgs[4..8].try_into().ok()?) as usize;
    let take = n.min(total);
    let mut images = Vec::with_capacity(take * IN);
    for i in 0..take {
        let base = 16 + i * IN;
        for b in &imgs[base..base + IN] {
            images.push(*b as f32 / 255.0);
        }
    }
    let labels: Vec<usize> = lbls[8..8 + take].iter().map(|&b| b as usize).collect();
    Some((images, labels))
}

fn mnist_raw_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_MNIST_DIR") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(format!("{home}/.cache/torchvision-mnist/MNIST/raw"));
    if p.is_dir() { Some(p) } else { None }
}

pub fn default_out_path() -> PathBuf {
    std::env::var("RLX_BAKE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/out/mnist.rlx")
        })
}

pub fn password_from_env() -> anyhow::Result<String> {
    let pw = std::env::var("RLX_BAKE_PASSWORD")
        .map_err(|_| anyhow::anyhow!("set RLX_BAKE_PASSWORD in the environment"))?;
    if pw.is_empty() {
        anyhow::bail!("RLX_BAKE_PASSWORD is empty");
    }
    Ok(pw)
}
