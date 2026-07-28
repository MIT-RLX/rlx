// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// TinyConv-MNIST inference across RLX backends via the runtime Session.
//
// Same CNN as the trainer; loads the trained fp32 weights (RLX_F32_DUMP) and
// runs the test set through one Device, reporting accuracy + throughput. The
// point is to exercise the SAME IR on CPU / Metal / MLX / wgpu / ANE — the GPU
// backends keep element-wise regions fused and run their own parallel kernels,
// so this is RLX's "compile once, run anywhere" path.
//
// Build/run (Apple silicon):
//   DEVICE=metal cargo run -p rlx-runtime --release \
//       --features metal,mlx,gpu,coreml --example mnist_infer_backends
// DEVICE ∈ {cpu, metal, mlx, gpu, ane}.

use rlx_ir::op::*;
use rlx_ir::*;
use rlx_runtime::{Device, Session};
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
    let conv = |g: &mut Graph, x, w, co, ho, wo| {
        g.add_node(
            Op::Conv {
                kernel_size: vec![3, 3],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![x, w],
            Shape::new(&[b, co, ho, wo], f),
        )
    };
    let bias = |g: &mut Graph, x, bb, c, h, w| {
        let b4 = g.add_node(
            Op::Reshape {
                new_shape: vec![1, c as i64, 1, 1],
            },
            vec![bb],
            Shape::new(&[1, c, 1, 1], f),
        );
        g.binary(BinaryOp::Add, x, b4, Shape::new(&[b, c, h, w], f))
    };
    let pool = |g: &mut Graph, x, c, ho, wo| {
        g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![x],
            Shape::new(&[b, c, ho, wo], f),
        )
    };
    let c1 = conv(&mut g, x, conv1_w, 8, 26, 26);
    let c1 = bias(&mut g, c1, conv1_b, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[b, 8, 26, 26], f));
    let p1 = pool(&mut g, c1, 8, 13, 13);
    let c2 = conv(&mut g, p1, conv2_w, 16, 11, 11);
    let c2 = bias(&mut g, c2, conv2_b, 16, 11, 11);
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

fn argmax(r: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in r.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
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
    let devname = std::env::var("DEVICE").unwrap_or_else(|_| "cpu".into());
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let device = match devname.as_str() {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        "ane" => Device::Ane,
        _ => Device::Cpu,
    };

    let wbytes = std::fs::read(&wpath).unwrap_or_else(|e| panic!("weights {wpath}: {e}"));
    let all: Vec<f32> = wbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(all.len(), PARAMS.iter().map(|(_, n)| *n).sum::<usize>());

    let (ximg, n) = read_idx_images(&format!("{raw}/t10k-images-idx3-ubyte"));
    let labels = read_idx_labels(&format!("{raw}/t10k-labels-idx1-ubyte"));
    let nb = n / batch;
    let img = 28 * 28;

    let mut compiled = Session::new(device).compile(build_graph(batch));
    let mut off = 0;
    for (name, len) in PARAMS {
        compiled.set_param(name, &all[off..off + len]);
        off += len;
    }

    let getb = |bi: usize| ximg[bi * batch * img..(bi + 1) * batch * img].to_vec();
    let t0 = Instant::now();
    let _ = compiled.run(&[("x", &getb(0))]); // warmup / compile
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut steps = Vec::with_capacity(nb);
    let mut correct = 0usize;
    let tt = Instant::now();
    for bi in 0..nb {
        let xb = getb(bi);
        let t = Instant::now();
        let out = compiled.run(&[("x", &xb)]).remove(0);
        steps.push(t.elapsed().as_secs_f64() * 1e3);
        for i in 0..batch {
            if argmax(&out[i * 10..(i + 1) * 10]) == labels[bi * batch + i] as usize {
                correct += 1;
            }
        }
    }
    let wall = tt.elapsed().as_secs_f64();
    let imgs = nb * batch;
    let acc = correct as f64 / imgs as f64;
    let p50 = median(steps);
    let imgs_s = imgs as f64 / wall;
    eprintln!(
        "rlx[{devname}]: acc={acc:.4} imgs={imgs} batch={batch} batch_p50={p50:.2}ms \
         first={first_ms:.0}ms imgs/s={imgs_s:.0}"
    );
    println!(
        "RLX_BENCH,rlx-{devname}-infer,{devname},{acc:.4},,,{p50:.1},{first_ms:.0},{imgs_s:.0}"
    );
}
