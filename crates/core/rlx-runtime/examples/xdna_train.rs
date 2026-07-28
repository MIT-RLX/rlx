// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// BACKWARD + TRAINING on the AMD XDNA NPU. rlx-autodiff turns a forward graph into
// a backward (gradient) graph of `*Backward` ops; the compiler's
// `needs_backward_decompose` (training feature, on by default via rlx-opt) lowers
// every backward op the NPU doesn't claim into primitives, which then run on the
// NPU via the same chain path as forward inference. This validates the gradients
// vs the CPU backend and runs a full SGD training loop with the grad on the NPU.
//
//   AIECC=.. PEANO=.. RLX_XDNA_AIE_INCLUDE=.. RLX_XDNA_SHIM=.. \
//     cargo run -p rlx-runtime --features xdna --example xdna_train

use rlx_ir::op::{Activation, ReduceOp as IrRed};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad;
use rlx_runtime::{Device, Session};

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-6)
}

fn main() {
    println!("rlx backward + training → Device::Xdna (NPU)\n");
    let mut all = true;

    // (1) ELEMENTWISE BACKWARD: loss = sum(relu(x)); dL/dx = relu'(x) — bit-exact.
    // keep_dim keeps the loss rank-1 ([1]) so no rank-0 tensor hits the NPU arms.
    let n = 32usize;
    // Includes x=0 exactly (index 4): relu′ now decomposes to relu(sign(x)) (NaN-free
    // at the kink, H(0)=0) — the old relu(x)/x gave 0/0 = NaN here.
    let xdata: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect();
    let mut g1 = Graph::new("relu_loss");
    let sh = Shape::new(&[n], DType::F32);
    let x1 = g1.param("x", sh.clone());
    let y1 = g1.activation(Activation::Relu, x1, sh.clone());
    let l1 = g1.add_node(Op::Reduce { op: IrRed::Sum, axes: vec![0], keep_dim: true }, vec![y1], Shape::new(&[1], DType::F32));
    g1.set_outputs(vec![l1]);
    let bwd1 = grad(&g1, &[x1]);
    let run1 = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(bwd1.clone());
        c.set_param("x", &xdata);
        c.run(&[("d_output", &[1.0f32][..])])[0].clone()
    };
    let (cg1, ng1) = (run1(Device::Cpu), run1(Device::Xdna));
    let ok1 = cg1.len() == ng1.len() && cg1.iter().zip(&ng1).all(|(a, b)| (a - b).abs() < 1e-4);
    println!("  {:<16} {}  (dL/dx elementwise)", "grad·relu·sum", if ok1 { "PASS ✓" } else { "FAIL ✗" });
    all &= ok1;

    // (2) MATMUL BACKWARD: loss = sum(relu(x @ W)); dL/dW = xᵀ @ (relu'(h) ⊙ 1).
    // The backward matmul has TWO dynamic operands (no Param weight) → the
    // dynamic-weight NPU matmul path quantizes+tiles the weight per run.
    let (m, k, nn) = (16usize, 16usize, 16usize);
    let xin: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let win: Vec<f32> = (0..k * nn).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let mut g2 = Graph::new("mlp_loss");
    let x2 = g2.input("x", Shape::new(&[m, k], DType::F32));
    let w2 = g2.param("W", Shape::new(&[k, nn], DType::F32));
    let h2 = g2.matmul(x2, w2, Shape::new(&[m, nn], DType::F32));
    let a2 = g2.activation(Activation::Relu, h2, Shape::new(&[m, nn], DType::F32));
    // reduce both axes → [1,1] (keep_dim) so no rank-0
    let l2 = g2.add_node(Op::Reduce { op: IrRed::Sum, axes: vec![1], keep_dim: true }, vec![a2], Shape::new(&[m, 1], DType::F32));
    let l2b = g2.add_node(Op::Reduce { op: IrRed::Sum, axes: vec![0], keep_dim: true }, vec![l2], Shape::new(&[1, 1], DType::F32));
    g2.set_outputs(vec![l2b]);
    let bwd2 = grad(&g2, &[w2]);
    let run2 = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(bwd2.clone());
        c.set_param("W", &win);
        c.run(&[("x", &xin), ("d_output", &[1.0f32][..])])[0].clone()
    };
    let (cg2, ng2) = (run2(Device::Cpu), run2(Device::Xdna));
    let c2 = cos(&cg2, &ng2);
    let ok2 = c2 > 0.99;
    println!("  {:<16} {}  cos {:.4}  (dL/dW via NPU matmul)", "grad·mlp", if ok2 { "PASS ✓" } else { "FAIL ✗" }, c2);
    all &= ok2;

    // (3) TRAINING LOOP on the NPU: linear regression loss = Σ(x@W − t)². The grad
    // dL/dW runs on the NPU (forward matmul + backward dynamic matmul + elementwise);
    // the SGD step (W −= lr·grad) is on the host — the CoreML-training model. The
    // backward Session is compiled ONCE; each step just set_params W and re-runs.
    let (tm, tk, tn) = (8usize, 8usize, 4usize);
    let xt: Vec<f32> = (0..tm * tk).map(|i| ((i * 7 + 1) % 11) as f32 / 11.0 - 0.5).collect();
    let w_true: Vec<f32> = (0..tk * tn).map(|i| ((i * 5 + 2) % 9) as f32 / 9.0 - 0.4).collect();
    // target t = x @ w_true (host) — an achievable optimum so the loss can reach ~0.
    let mut tgt = vec![0f32; tm * tn];
    for i in 0..tm {
        for j in 0..tn {
            tgt[i * tn + j] = (0..tk).map(|p| xt[i * tk + p] * w_true[p * tn + j]).sum();
        }
    }
    let host_loss = |w: &[f32]| -> f32 {
        let mut s = 0.0f32;
        for i in 0..tm {
            for j in 0..tn {
                let h: f32 = (0..tk).map(|p| xt[i * tk + p] * w[p * tn + j]).sum();
                let d = h - tgt[i * tn + j];
                s += d * d;
            }
        }
        s
    };
    // forward loss graph: h = x@W; d = h − t; loss = Σ(d·d)  ([1,1] via keep_dim)
    let mut gt = Graph::new("reg_loss");
    let xtn = gt.input("x", Shape::new(&[tm, tk], DType::F32));
    let ttn = gt.input("t", Shape::new(&[tm, tn], DType::F32));
    let wtn = gt.param("W", Shape::new(&[tk, tn], DType::F32));
    let htn = gt.matmul(xtn, wtn, Shape::new(&[tm, tn], DType::F32));
    let dtn = gt.add_node(Op::Binary(rlx_ir::op::BinaryOp::Sub), vec![htn, ttn], Shape::new(&[tm, tn], DType::F32));
    let sqn = gt.add_node(Op::Binary(rlx_ir::op::BinaryOp::Mul), vec![dtn, dtn], Shape::new(&[tm, tn], DType::F32));
    let r1 = gt.add_node(Op::Reduce { op: IrRed::Sum, axes: vec![1], keep_dim: true }, vec![sqn], Shape::new(&[tm, 1], DType::F32));
    let lossn = gt.add_node(Op::Reduce { op: IrRed::Sum, axes: vec![0], keep_dim: true }, vec![r1], Shape::new(&[1, 1], DType::F32));
    gt.set_outputs(vec![lossn]);
    let bwdt = grad(&gt, &[wtn]);

    let mut w: Vec<f32> = vec![0.0; tk * tn]; // train from zero
    let lr = 0.15f32;
    let l0 = host_loss(&w);
    let mut cbt = Session::new(Device::Xdna).compile(bwdt.clone());
    let steps = 40usize;
    for s in 0..steps {
        cbt.set_param("W", &w);
        let g = cbt.run(&[("x", &xt), ("t", &tgt), ("d_output", &[1.0f32][..])])[0].clone();
        for (wi, gi) in w.iter_mut().zip(&g) {
            *wi -= lr * gi;
        }
        if s == 0 || s == steps / 2 - 1 || s == steps - 1 {
            println!("    step {:>2}: loss {:.5}", s + 1, host_loss(&w));
        }
    }
    let lf = host_loss(&w);
    let train_ok = lf < l0 * 0.1; // loss dropped by >10× on the NPU-computed grad
    println!("  {:<16} {}  loss {:.4} → {:.4}  (SGD on NPU grad, {} steps)", "train·regress", if train_ok { "PASS ✓" } else { "FAIL ✗" }, l0, lf, steps);
    all &= train_ok;

    println!("\n{}", if all { "NPU backward matches CPU ✓" } else { "MISMATCH" });
    if !all {
        std::process::exit(1);
    }
}
