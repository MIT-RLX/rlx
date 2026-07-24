// TinyConv-MNIST inference on the Apple Neural Engine (CoreML backend).
//
// The CNN matches the rlx-cortexm trainer / the other MNIST runners:
//   Conv(1->8,3x3 valid) -> +bias -> ReLU -> MaxPool2
//   Conv(8->16,3x3 valid) -> +bias -> ReLU -> MaxPool2
//   Flatten(400) -> FC(400->10) + bias
//
// CoreML/ANE is inference-only (no backward), so this is the deployment
// counterpart of the CPU *training* benchmark: we load the fp32 weights the
// trainer dumped (RLX_F32_DUMP), run the test set through the ANE, and report
// accuracy + throughput (images/s).
//
// Run (Apple silicon):
//   target/release/train-mnist --epochs 2 \
//       --out /tmp/tc.rs --val-set 0          # produces weights via RLX_F32_DUMP
//   cargo run -p rlx-coreml --release --example ane_mnist
//
// Env:
//   RLX_F32_DUMP  weights file (default /tmp/tinyconv_f32.bin)
//   MNIST_RAW     IDX dir (default ~/.cache/torchvision-mnist/MNIST/raw)
//   ANE_BATCH     batch size (default 100)
//   RLX_COREML_UNITS  gpu(default for fp32)/ane/cpu/all

#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::{ComputeUnits, CoremlExecutable, ane_available, chip_info};
use rlx_ir::op::*;
use rlx_ir::*;
use std::time::Instant;

// Conv weight sizes are written as out×in×kh×kw to mirror the tensor shape;
// the conv1 `*1` in-channels factor (grayscale MNIST) is kept for that parity.
#[allow(clippy::identity_op)]
const PARAMS: &[(&str, usize)] = &[
    ("conv1_w", 8 * 3 * 3),
    ("conv1_b", 8),
    ("conv2_w", 16 * 8 * 3 * 3),
    ("conv2_b", 16),
    ("fc_w", 400 * 10),
    ("fc_b", 10),
];

fn read_idx_images(p: &str) -> (Vec<f32>, usize) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let data = b[16..]
        .iter()
        .map(|&v| (v as f32 / 255.0 - 0.5) / 0.5)
        .collect();
    (data, n)
}

fn read_idx_labels(p: &str) -> Vec<u8> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    b[8..].to_vec()
}

/// Build the forward TinyConv graph for a fixed batch `b`.
fn build_graph(b: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("tinyconv_infer");
    let x = g.input("x", Shape::new(&[b, 1, 28, 28], f));

    let conv1_w = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[8], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[16], f));
    let fc_w = g.param("fc_w", Shape::new(&[400, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));

    let conv = |g: &mut Graph, x, w, c_out, h_out, w_out| {
        g.add_node(
            Op::Conv {
                kernel_size: vec![3, 3],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![x, w],
            Shape::new(&[b, c_out, h_out, w_out], f),
        )
    };
    let bias4d = |g: &mut Graph, x, bias, c, h, w| {
        let b4 = g.add_node(
            Op::Reshape {
                new_shape: vec![1, c as i64, 1, 1],
            },
            vec![bias],
            Shape::new(&[1, c, 1, 1], f),
        );
        g.binary(BinaryOp::Add, x, b4, Shape::new(&[b, c, h, w], f))
    };
    let pool = |g: &mut Graph, x, c, h_out, w_out| {
        g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![x],
            Shape::new(&[b, c, h_out, w_out], f),
        )
    };

    let c1 = conv(&mut g, x, conv1_w, 8, 26, 26);
    let c1 = bias4d(&mut g, c1, conv1_b, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[b, 8, 26, 26], f));
    let p1 = pool(&mut g, c1, 8, 13, 13);

    let c2 = conv(&mut g, p1, conv2_w, 16, 11, 11);
    let c2 = bias4d(&mut g, c2, conv2_b, 16, 11, 11);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[b, 16, 11, 11], f));
    let p2 = pool(&mut g, c2, 16, 5, 5);

    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, 400],
        },
        vec![p2],
        Shape::new(&[b, 400], f),
    );
    let mm = g.matmul(flat, fc_w, Shape::new(&[b, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[b, 10], f));
    g.set_outputs(vec![logits]);
    g
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let raw = std::env::var("MNIST_RAW")
        .unwrap_or_else(|_| format!("{home}/.cache/torchvision-mnist/MNIST/raw"));
    let wpath = std::env::var("RLX_F32_DUMP").unwrap_or_else(|_| "/tmp/tinyconv_f32.bin".into());
    let batch: usize = std::env::var("ANE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    eprintln!("chip: {:?}  ane_available={}", chip_info(), ane_available());

    // Weights.
    let wbytes = std::fs::read(&wpath).unwrap_or_else(|e| {
        panic!("read weights {wpath}: {e} (run the trainer with RLX_F32_DUMP={wpath})")
    });
    let all: Vec<f32> = wbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let total: usize = PARAMS.iter().map(|(_, n)| *n).sum();
    assert_eq!(
        all.len(),
        total,
        "weights file has {} f32, expected {total}",
        all.len()
    );

    // Test set.
    let (ximg, n) = read_idx_images(&format!("{raw}/t10k-images-idx3-ubyte"));
    let labels = read_idx_labels(&format!("{raw}/t10k-labels-idx1-ubyte"));
    let nb = n / batch;

    // Stage on the ANE (CPU+Neural Engine) and load weights.
    let mut exe =
        CoremlExecutable::compile_with_units(build_graph(batch), ComputeUnits::CpuAndNeuralEngine);
    let mut off = 0;
    for (name, len) in PARAMS {
        exe.set_param(name, &all[off..off + len]);
        off += len;
    }

    let img = 28 * 28;
    let getb = |bi: usize| ximg[bi * batch * img..(bi + 1) * batch * img].to_vec();

    // Warmup: first run finalizes (lowers -> .mlpackage -> loads MLModel).
    let t_compile = Instant::now();
    let _ = exe.run(&[("x", &getb(0))]).expect("ane run");
    let first_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    // Timed pass over the full test set.
    let mut steps = Vec::with_capacity(nb);
    let mut correct = 0usize;
    let t0 = Instant::now();
    for bi in 0..nb {
        let xb = getb(bi);
        let t = Instant::now();
        let out = exe.run(&[("x", &xb)]).expect("ane run").remove(0);
        steps.push(t.elapsed().as_secs_f64() * 1e3);
        for i in 0..batch {
            if argmax(&out[i * 10..(i + 1) * 10]) == labels[bi * batch + i] as usize {
                correct += 1;
            }
        }
    }
    let wall = t0.elapsed().as_secs_f64();

    let imgs = nb * batch;
    let acc = correct as f64 / imgs as f64;
    let p50 = median(steps.clone());
    let imgs_s = imgs as f64 / wall;
    eprintln!(
        "ane: acc={acc:.4} imgs={imgs} batch={batch} batch_p50={p50:.2}ms \
         first={first_ms:.0}ms imgs/s={imgs_s:.0}"
    );
    // Bench row: framework,device,test_acc,train_s,epoch_s,step_p50_ms,first_step_ms,imgs_per_s.
    // Inference-only backend: train_s / epoch_s left blank.
    println!("RLX_BENCH,rlx-ane,ane,{acc:.4},,,{p50:.1},{first_ms:.0},{imgs_s:.0}");
}
