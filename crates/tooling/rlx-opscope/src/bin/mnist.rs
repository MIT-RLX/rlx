// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-mnist` — run the harness on a **real MNIST model** with **real
//! MNIST images**. Builds the same MLP rlx-vision-bench uses
//! (784 → hidden → 10, `matmul → bias → ReLU → matmul → bias`), reads the real
//! `train-images-idx3-ubyte` from the local torchvision cache, and streams
//! batches through the fixed-weight network as a decode-like sequence.
//!
//! Weights are random (untrained) on purpose: we profile *data* structure, and
//! MNIST's input sparsity + post-ReLU activation sparsity appear regardless of
//! training. Pixels are scaled to `[0,1]` so background (0) stays a true zero →
//! the density sketch sees MNIST's border sparsity.
//!
//! Usage: `opscope-mnist [out.csv] [steps]`  (defaults: `opscope_mnist.csv`, 12)
//! then:  `opscope-mine <out.csv>`

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Philox4x32, Shape};
use rlx_opscope::{Recorder, StatConfig, inject_matmul_stats};
use rlx_runtime::{Device, Session};

const PIXELS: usize = 784;

/// Read the first `n` images from an IDX3 file, scaled to `[0,1]`.
fn load_idx_images(path: &str, n: usize) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let take = n.min(count);
    Ok(bytes[16..16 + take * PIXELS]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect())
}

/// MNIST MLP: `logits = (relu(x·W1 + b1))·W2 + b2`. Returns the graph with its
/// two matmul sites named `fc1`/`fc2`.
fn mnist_mlp(batch: usize, hidden: usize) -> Graph {
    let mut g = Graph::new("mnist_mlp");
    let x = g.input("x", Shape::new(&[batch, PIXELS], DType::F32));
    let w1 = g.param("fc1_w", Shape::new(&[PIXELS, hidden], DType::F32));
    let b1 = g.param("fc1_b", Shape::new(&[hidden], DType::F32));
    let w2 = g.param("fc2_w", Shape::new(&[hidden, 10], DType::F32));
    let b2 = g.param("fc2_b", Shape::new(&[10], DType::F32));
    let hs = Shape::new(&[batch, hidden], DType::F32);
    let os = Shape::new(&[batch, 10], DType::F32);

    let mm1 = g.matmul(x, w1, hs.clone());
    g.node_mut(mm1).name = Some("fc1".into());
    let h = g.add_node(Op::Binary(BinaryOp::Add), vec![mm1, b1], hs.clone());
    let h = g.activation(Activation::Relu, h, hs);
    let mm2 = g.matmul(h, w2, os.clone());
    g.node_mut(mm2).name = Some("fc2".into());
    let logits = g.add_node(Op::Binary(BinaryOp::Add), vec![mm2, b2], os);
    g.set_outputs(vec![logits]);
    g
}

/// He-initialized weight matrix `[rows, cols]` (fan_in = rows).
fn he_weight(rng: &mut Philox4x32, rows: usize, cols: usize) -> Vec<f32> {
    let mut w = vec![0f32; rows * cols];
    rng.fill_normal(&mut w);
    let scale = (2.0 / rows as f32).sqrt();
    for v in &mut w {
        *v *= scale;
    }
    w
}

fn home_mnist() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/macmini".into());
    format!("{home}/.cache/torchvision-mnist/MNIST/raw/train-images-idx3-ubyte")
}

fn main() -> std::io::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_mnist.csv".into());
    let steps: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let (batch, hidden) = (64usize, 128usize);

    let g = mnist_mlp(batch, hidden);
    let (ginj, specs) = inject_matmul_stats(&g, &StatConfig::default());
    let mut compiled = Session::new(Device::Cpu).compile(ginj);

    // Random He-scaled weights, zero biases; fixed across all steps.
    let mut rng = Philox4x32::new(0xC0FFEE);
    compiled.set_param("fc1_w", &he_weight(&mut rng, PIXELS, hidden));
    compiled.set_param("fc2_w", &he_weight(&mut rng, hidden, 10));
    compiled.set_param("fc1_b", &vec![0f32; hidden]);
    compiled.set_param("fc2_b", &[0f32; 10]);

    // Real MNIST pixels, [0,1].
    let path = home_mnist();
    let pixels = load_idx_images(&path, batch * steps as usize)?;
    let avail = pixels.len() / PIXELS;
    if avail < batch {
        eprintln!("[opscope] only {avail} images at {path}; need ≥{batch}");
        return Ok(());
    }
    let steps = (avail / batch).min(steps as usize) as u64;

    let mut rec = Recorder::create(&out)?;
    for step in 0..steps {
        let lo = step as usize * batch * PIXELS;
        let x = &pixels[lo..lo + batch * PIXELS];
        let outs = compiled.run(&[("x", x)]);
        rec.record(0, step, "cpu", "mnist", batch, PIXELS, 0, &specs, &outs)?;
    }
    rec.flush()?;
    eprintln!(
        "[opscope] real MNIST MLP (batch={batch}, hidden={hidden}, {} sketches/step) × {steps} steps → {out}",
        specs.len()
    );
    Ok(())
}
