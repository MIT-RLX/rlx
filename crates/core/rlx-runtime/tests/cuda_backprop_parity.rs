// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA ↔ CPU **gradient parity** for autodiff backward graphs.
//!
//! Builds real training (backward) graphs with `rlx_autodiff::grad_with_loss`
//! and checks that every gradient the CUDA backend produces matches the CPU
//! reference. This isolates CUDA backprop coverage from any transport/data
//! plumbing. No-ops on a CUDA-less host (`is_available` guard), so it's a
//! regression guard that only asserts on a real CUDA GPU box.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

/// Device under test — `RLX_PARITY_DEVICE` (cuda|metal|gpu|mlx|ane|rocm|vulkan…),
/// default CUDA. Lets the SAME gradient-parity suite run against every backend.
fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    }
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Run `bwd` (a backward graph, outputs `[loss, aux.., grad_0..]`) on `dev`
/// with the given params + inputs; returns all outputs.
fn run(
    bwd: &Graph,
    dev: Device,
    params: &[(&str, Vec<f32>)],
    inputs: &[(&str, &[f32])],
) -> Vec<Vec<f32>> {
    let mut sess = Session::new(dev).compile(bwd.clone());
    for (n, v) in params {
        sess.set_param(n, v);
    }
    sess.run(inputs)
}

fn softmax_ce_mean(g: &mut Graph, logits: NodeId, labels: NodeId) -> NodeId {
    let per = g.softmax_cross_entropy_with_logits(logits, labels);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![per],
        Shape::from_dims(&[], F),
    )
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Print per-output CUDA-vs-CPU errors for a backward graph; returns true if any
/// exceeds `tol`.
fn parity(
    name: &str,
    bwd: &Graph,
    params: &[(&str, Vec<f32>)],
    inputs: &[(&str, &[f32])],
    tol: f32,
) -> bool {
    let cpu = run(bwd, Device::Cpu, params, inputs);
    let cuda = run(bwd, target(), params, inputs);
    let mut bad = false;
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for (i, (c, g)) in cpu.iter().zip(&cuda).enumerate() {
        let e = max_err(c, g);
        let (nc, ng) = (l2(c), l2(g));
        // relative: is the CUDA output ~zero while CPU is not? (the zero-dW bug)
        let zero = ng < 1e-4 * nc.max(1e-9);
        eprintln!(
            "  {name}[{i}] max_err={e:.6} |cpu|={nc:.5} |cuda|={ng:.5}{} {}",
            if zero { " CUDA≈0!" } else { "" },
            if e < tol { "ok" } else { "MISMATCH" }
        );
        bad |= e >= tol;
    }
    bad
}

fn reduce(g: &mut Graph, op: ReduceOp, x: NodeId, axes: Vec<usize>, out: Shape) -> NodeId {
    g.add_node(
        Op::Reduce {
            op,
            axes,
            keep_dim: false,
        },
        vec![x],
        out,
    )
}

/// Isolate which backward *primitive* diverges on CUDA — matmul-VJP (both
/// transposes), reduce-sum/mean-VJP (seed broadcast), relu-VJP (mask),
/// bias-broadcast-VJP (reduce), softmax-CE-VJP.
#[test]
fn backward_primitives_isolation() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let mut bad = Vec::new();

    // matmul: loss = sum(x @ w); grad_x = 1·wᵀ, grad_w = xᵀ·1.
    {
        let mut g = Graph::new("mm");
        let x = g.param("x", Shape::new(&[4, 3], F));
        let w = g.param("w", Shape::new(&[3, 2], F));
        let y = g.matmul(x, w, Shape::new(&[4, 2], F));
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            y,
            vec![0, 1],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w]);
        let p = vec![("x", seeded(12, 1)), ("w", seeded(6, 2))];
        if parity("matmul", &bwd, &p, &[("d_output", &[1.0])], 1e-3) {
            bad.push("matmul");
        }
    }
    // reduce-mean: loss = mean(x); grad_x = 1/N.
    {
        let mut g = Graph::new("rm");
        let x = g.param("x", Shape::new(&[6], F));
        let loss = reduce(&mut g, ReduceOp::Mean, x, vec![0], Shape::from_dims(&[], F));
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "reduce_mean",
            &bwd,
            &[("x", seeded(6, 1))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("reduce_mean");
        }
    }
    // relu: loss = sum(relu(x)); grad_x = [x>0].
    {
        let mut g = Graph::new("relu");
        let x = g.param("x", Shape::new(&[6], F));
        let y = g.activation(Activation::Relu, x, Shape::new(&[6], F));
        let loss = reduce(&mut g, ReduceOp::Sum, y, vec![0], Shape::from_dims(&[], F));
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "relu",
            &bwd,
            &[("x", seeded(6, 1))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("relu");
        }
    }
    // bias broadcast: loss = sum(x + b); grad_b = sum over batch.
    {
        let mut g = Graph::new("bias");
        let x = g.param("x", Shape::new(&[4, 3], F));
        let b = g.param("b", Shape::new(&[3], F));
        let y = g.binary(BinaryOp::Add, x, b, Shape::new(&[4, 3], F));
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            y,
            vec![0, 1],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, b]);
        let p = vec![("x", seeded(12, 1)), ("b", seeded(3, 2))];
        if parity("bias_bcast", &bwd, &p, &[("d_output", &[1.0])], 1e-3) {
            bad.push("bias_bcast");
        }
    }
    // softmax-CE: loss = mean(ce(logits, labels)); grad_logits = (softmax−onehot)/N.
    {
        let mut g = Graph::new("ce");
        let logits = g.param("logits", Shape::new(&[4, 5], F));
        let labels = g.input("labels", Shape::new(&[4], F));
        let loss = softmax_ce_mean(&mut g, logits, labels);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[logits]);
        let y: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        if parity(
            "softmax_ce",
            &bwd,
            &[("logits", seeded(20, 1))],
            &[("labels", &y), ("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("softmax_ce");
        }
    }

    // maxpool + relu backward: loss = sum(maxpool(relu(x))); grad_x exercises
    // MaxPool2dBackward ∘ relu-backward (the conv-net path between grad_fw and
    // grad_cw — the actual suspect for the wgpu conv-net grad_cw divergence).
    {
        let mut g = Graph::new("poolbwd");
        let x = g.param("x", Shape::new(&[1, 1, 4, 4], F));
        let r = g.activation(Activation::Relu, x, Shape::new(&[1, 1, 4, 4], F));
        let p = g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![r],
            Shape::new(&[1, 1, 2, 2], F),
        );
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            p,
            vec![0, 1, 2, 3],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "pool_relu_bwd",
            &bwd,
            &[("x", seeded(16, 1))],
            &[("d_output", &[1.0])],
            1e-4,
        ) {
            bad.push("pool_relu_bwd");
        }
    }

    assert!(bad.is_empty(), "broken CUDA backward primitives: {bad:?}");
    eprintln!("all backward primitives match CPU ✓");
}

/// Isolate the conv-net `grad_cw` trigger: `v_fc_ce` (no bias) passes but the
/// full net fails, so add the bias add and multi-param grad one at a time.
#[test]
fn conv_bias_trigger() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let (n, cin, cout) = (4usize, 1usize, 4usize);
    let flat_dim = cout * 3 * 3;
    let xin = seeded(n * cin * 64, 1);
    let y: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
    let mut bad = Vec::new();

    // Exactly the full conv-net (x fed as INPUT, grad w.r.t. cw,fw,fb), toggling
    // whether `logits` is also a graph output (as conv_net_grads_match_cpu does).
    for extra_logits_output in [false, true] {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[n, cin, 8, 8], F));
        let w = g.param("w", Shape::new(&[cout, cin, 3, 3], F));
        let c = g.add_node(
            Op::Conv {
                kernel_size: vec![3, 3],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![x, w],
            Shape::new(&[n, cout, 6, 6], F),
        );
        let r = g.activation(Activation::Relu, c, Shape::new(&[n, cout, 6, 6], F));
        let p = g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![r],
            Shape::new(&[n, cout, 3, 3], F),
        );
        let flat = g.add_node(
            Op::Reshape {
                new_shape: vec![n as i64, flat_dim as i64],
            },
            vec![p],
            Shape::new(&[n, flat_dim], F),
        );
        let fw = g.param("fw", Shape::new(&[flat_dim, 10], F));
        let logits0 = g.matmul(flat, fw, Shape::new(&[n, 10], F));
        let fb = g.param("fb", Shape::new(&[10], F));
        let logits = g.binary(BinaryOp::Add, logits0, fb, Shape::new(&[n, 10], F));
        let labels = g.input("labels", Shape::new(&[n], F));
        let loss = softmax_ce_mean(&mut g, logits, labels);
        g.set_outputs(if extra_logits_output {
            vec![loss, logits]
        } else {
            vec![loss]
        });
        let bwd = rlx_autodiff::grad_with_loss(&g, &[w, fw, fb]);
        let params = vec![
            ("w", seeded(cout * cin * 9, 3)),
            ("fw", seeded(flat_dim * 10, 2)),
            ("fb", vec![0.5f32; 10]),
        ];
        let tag = if extra_logits_output {
            "with_logits_out"
        } else {
            "loss_only"
        };
        if parity(
            tag,
            &bwd,
            &params,
            &[
                ("x", xin.as_slice()),
                ("labels", y.as_slice()),
                ("d_output", &[1.0]),
            ],
            2e-3,
        ) {
            bad.push(tag);
        }
    }
    assert!(bad.is_empty(), "conv-net grad_cw trigger: {bad:?}");
}

/// Forward-op parity for the primitives the `SoftmaxCrossEntropyBackward`
/// decomposition rides — softmax, transpose[1,0], concat(axis 0), where/compare.
#[test]
fn softmax_ce_decomp_primitives() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let mut bad = Vec::new();

    // softmax over last axis.
    {
        let mut g = Graph::new("sm");
        let x = g.input("x", Shape::new(&[4, 5], F));
        let y = g.softmax(x, -1, Shape::new(&[4, 5], F));
        g.set_outputs(vec![y]);
        if parity("softmax", &g, &[], &[("x", &seeded(20, 1))], 1e-4) {
            bad.push("softmax");
        }
    }
    // transpose [5,4] → [4,5] (the one-hot [C,N]→[N,C] step).
    {
        let mut g = Graph::new("tr");
        let x = g.input("x", Shape::new(&[5, 4], F));
        let y = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![x],
            Shape::new(&[4, 5], F),
        );
        g.set_outputs(vec![y]);
        if parity("transpose", &g, &[], &[("x", &seeded(20, 1))], 1e-6) {
            bad.push("transpose");
        }
    }
    // im2col — the conv2d-weight-grad decomposition's patch extraction. Test
    // BATCH>1: batched M-ordering (N outer vs inner) must match CPU or the
    // downstream matmul contracts mismatched elements (the conv-net grad_cw bug).
    for (n, c) in [(1usize, 1usize), (4, 1), (2, 3)] {
        let mut g = Graph::new("im2col");
        let x = g.input("x", Shape::new(&[n, c, 4, 4], F));
        let y = g.im2col(x, [3, 3], [1, 1], [0, 0], [1, 1]);
        g.set_outputs(vec![y]);
        if parity(
            &format!("im2col_n{n}c{c}"),
            &g,
            &[],
            &[("x", &seeded(n * c * 16, 1))],
            1e-5,
        ) {
            bad.push("im2col");
        }
    }
    // single-element gather from a flat array — the static im2col decomposition
    // (`build_im2col_rows`) does this per patch element, then concats.
    {
        let mut g = Graph::new("gather1");
        let x = g.input("x", Shape::new(&[16], F));
        let idx = g.input("idx", Shape::new(&[1], F));
        let y = g.gather_(x, idx, 0);
        g.set_outputs(vec![y]);
        if parity(
            "gather_flat",
            &g,
            &[],
            &[("x", &seeded(16, 1)), ("idx", &[5.0])],
            1e-5,
        ) {
            bad.push("gather_flat");
        }
    }
    // concat of several [1] rows into a column (the other half of build_im2col_rows).
    {
        let mut g = Graph::new("concat1");
        let a = g.input("a", Shape::new(&[1], F));
        let b = g.input("b", Shape::new(&[1], F));
        let c = g.input("c", Shape::new(&[1], F));
        let y = g.concat_(vec![a, b, c], 0);
        g.set_outputs(vec![y]);
        if parity(
            "concat_rows",
            &g,
            &[],
            &[("a", &[1.0]), ("b", &[2.0]), ("c", &[3.0])],
            1e-6,
        ) {
            bad.push("concat_rows");
        }
    }
    assert!(
        bad.is_empty(),
        "broken CUDA decomposition primitives: {bad:?}"
    );
    eprintln!("all SCE-decomposition primitives match CPU ✓");
}

/// MLP `din→hidden→classes` with softmax-CE loss — the exact op mix distributed
/// MNIST training rides (matmul, bias add, relu, softmax-CE, reduce, and their
/// VJPs: matmul-transpose, reduce-broadcast, relu-mask).
#[test]
fn mlp_softmax_ce_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let (b, din, hidden, classes) = (8usize, 16usize, 12usize, 10usize);
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[b, din], F));
    let labels = g.input("labels", Shape::new(&[b], F));
    let w1 = g.param("w1", Shape::new(&[din, hidden], F));
    let b1 = g.param("b1", Shape::new(&[hidden], F));
    let w2 = g.param("w2", Shape::new(&[hidden, classes], F));
    let b2 = g.param("b2", Shape::new(&[classes], F));
    let h = g.matmul(x, w1, Shape::new(&[b, hidden], F));
    let h = g.binary(BinaryOp::Add, h, b1, Shape::new(&[b, hidden], F));
    let h = g.activation(Activation::Relu, h, Shape::new(&[b, hidden], F));
    let logits = g.matmul(h, w2, Shape::new(&[b, classes], F));
    let logits = g.binary(BinaryOp::Add, logits, b2, Shape::new(&[b, classes], F));
    let loss = softmax_ce_mean(&mut g, logits, labels);
    g.set_outputs(vec![loss, logits]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[w1, b1, w2, b2]);

    let params = vec![
        ("w1", seeded(din * hidden, 1)),
        ("b1", vec![0.0; hidden]),
        ("w2", seeded(hidden * classes, 2)),
        ("b2", vec![0.0; classes]),
    ];
    let x_in = seeded(b * din, 3);
    let y_in: Vec<f32> = (0..b).map(|i| (i % classes) as f32).collect();
    let inputs: Vec<(&str, &[f32])> =
        vec![("x", &x_in), ("labels", &y_in), ("d_output", &[1.0f32])];

    let cpu = run(&bwd, Device::Cpu, &params, &inputs);
    let cuda = run(&bwd, target(), &params, &inputs);
    assert_eq!(cpu.len(), cuda.len(), "output count mismatch");
    let tags = ["loss", "logits", "grad_w1", "grad_b1", "grad_w2", "grad_b2"];
    let mut bad = false;
    for (i, (c, g)) in cpu.iter().zip(&cuda).enumerate() {
        let e = max_err(c, g);
        let tag = tags.get(i).copied().unwrap_or("?");
        eprintln!(
            "  [{i}] {tag:8} max_err={e:.6}  {}",
            if e < 1e-3 { "ok" } else { "MISMATCH" }
        );
        bad |= e >= 1e-3;
    }
    assert!(
        !bad,
        "mlp: some CUDA gradients disagree with CPU (see above)"
    );
}

/// Conv → relu → maxpool → flatten → FC → softmax-CE — exercises the conv2d and
/// maxpool2d backward paths (cuDNN / host-fallback) in addition to the MLP mix.
#[test]
fn conv_net_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let (b, classes) = (4usize, 10usize);
    let mut g = Graph::new("cnn");
    let x = g.input("x", Shape::new(&[b, 1, 8, 8], F));
    let labels = g.input("labels", Shape::new(&[b], F));
    let cw = g.param("cw", Shape::new(&[4, 1, 3, 3], F));
    let flat_dim = 4 * 3 * 3; // conv 8→6, pool 6→3
    let fw = g.param("fw", Shape::new(&[flat_dim, classes], F));
    let fb = g.param("fb", Shape::new(&[classes], F));
    let c = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, cw],
        Shape::new(&[b, 4, 6, 6], F),
    );
    let c = g.activation(Activation::Relu, c, Shape::new(&[b, 4, 6, 6], F));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![c],
        Shape::new(&[b, 4, 3, 3], F),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, flat_dim as i64],
        },
        vec![p],
        Shape::new(&[b, flat_dim], F),
    );
    let logits = g.matmul(flat, fw, Shape::new(&[b, classes], F));
    let logits = g.binary(BinaryOp::Add, logits, fb, Shape::new(&[b, classes], F));
    let loss = softmax_ce_mean(&mut g, logits, labels);
    g.set_outputs(vec![loss, logits]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[cw, fw, fb]);

    let params = vec![
        ("cw", seeded(4 * 9, 1)),
        ("fw", seeded(flat_dim * classes, 2)),
        ("fb", vec![0.0; classes]),
    ];
    let x_in = seeded(b * 64, 3);
    let y_in: Vec<f32> = (0..b).map(|i| (i % classes) as f32).collect();
    let inputs: Vec<(&str, &[f32])> =
        vec![("x", &x_in), ("labels", &y_in), ("d_output", &[1.0f32])];

    let cpu = run(&bwd, Device::Cpu, &params, &inputs);
    let cuda = run(&bwd, target(), &params, &inputs);
    let tags = ["loss", "logits", "grad_cw", "grad_fw", "grad_fb"];
    let mut bad = false;
    for (i, (c, g)) in cpu.iter().zip(&cuda).enumerate() {
        let e = max_err(c, g);
        let tag = tags.get(i).copied().unwrap_or("?");
        eprintln!(
            "  [{i}] {tag:8} max_err={e:.6}  {}",
            if e < 2e-3 { "ok" } else { "MISMATCH" }
        );
        bad |= e >= 2e-3;
    }
    assert!(
        !bad,
        "conv: some CUDA gradients disagree with CPU (see above)"
    );
}

/// Two stacked convs — the grad w.r.t. conv1's weight backprops THROUGH conv2,
/// exercising `Conv2dBackwardInput`. Confirms a multi-conv CNN trains correctly.
#[test]
fn two_conv_grads_match_cpu() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let (b, classes) = (2usize, 10usize);
    let mut g = Graph::new("cnn2");
    let x = g.input("x", Shape::new(&[b, 1, 10, 10], F));
    let cw1 = g.param("cw1", Shape::new(&[4, 1, 3, 3], F));
    let cw2 = g.param("cw2", Shape::new(&[8, 4, 3, 3], F));
    let flat_dim = 8 * 3 * 3; // 10→8 (conv1), 8→6 (conv2), 6→3 (pool)
    let fw = g.param("fw", Shape::new(&[flat_dim, classes], F));
    let labels = g.input("labels", Shape::new(&[b], F));
    let conv = |g: &mut Graph, inp, w, co, ho, wo| {
        g.add_node(
            Op::Conv {
                kernel_size: vec![3, 3],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![inp, w],
            Shape::new(&[b, co, ho, wo], F),
        )
    };
    let c1 = conv(&mut g, x, cw1, 4, 8, 8);
    let r1 = g.activation(Activation::Relu, c1, Shape::new(&[b, 4, 8, 8], F));
    let c2 = conv(&mut g, r1, cw2, 8, 6, 6);
    let r2 = g.activation(Activation::Relu, c2, Shape::new(&[b, 8, 6, 6], F));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![r2],
        Shape::new(&[b, 8, 3, 3], F),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, flat_dim as i64],
        },
        vec![p],
        Shape::new(&[b, flat_dim], F),
    );
    let logits = g.matmul(flat, fw, Shape::new(&[b, classes], F));
    let loss = softmax_ce_mean(&mut g, logits, labels);
    g.set_outputs(vec![loss, logits]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[cw1, cw2, fw]);

    let params = vec![
        ("cw1", seeded(4 * 9, 1)),
        ("cw2", seeded(8 * 4 * 9, 2)),
        ("fw", seeded(flat_dim * classes, 3)),
    ];
    let y: Vec<f32> = (0..b).map(|i| (i % classes) as f32).collect();
    let x_in = seeded(b * 100, 4);
    let inputs: Vec<(&str, &[f32])> = vec![("x", &x_in), ("labels", &y), ("d_output", &[1.0f32])];
    let cpu = run(&bwd, Device::Cpu, &params, &inputs);
    let tgt = run(&bwd, target(), &params, &inputs);
    let tags = ["loss", "logits", "grad_cw1", "grad_cw2", "grad_fw"];
    let mut bad = false;
    for (i, (c, gg)) in cpu.iter().zip(&tgt).enumerate() {
        let e = max_err(c, gg);
        eprintln!(
            "  [{i}] {:9} max_err={e:.6}  {}",
            tags.get(i).copied().unwrap_or("?"),
            if e < 3e-3 { "ok" } else { "MISMATCH" }
        );
        bad |= e >= 3e-3;
    }
    assert!(
        !bad,
        "two-conv: some gradients disagree with CPU (see above)"
    );
}

// conv_transpose2d backward (dx, dw) — verifies the new `Op::ConvTranspose2d`
// VJP produces the SAME gradients on CUDA as on CPU (stride-2 upsampling case).
#[test]
fn conv_transpose2d_backward_parity() {
    if !is_available(target()) {
        eprintln!("backprop_parity: {:?} unavailable — skipping", target());
        return;
    }
    let mut g = Graph::new("ct");
    let x = g.param("x", Shape::new(&[1, 2, 3, 3], F));
    let w = g.param("w", Shape::new(&[2, 3, 2, 2], F));
    let y = g.conv_transpose2d(x, w, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1);
    let rank = g.node(y).shape.rank();
    let loss = reduce(
        &mut g,
        ReduceOp::Sum,
        y,
        (0..rank).collect(),
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w]);
    let p = vec![("x", seeded(18, 7)), ("w", seeded(24, 9))];
    let bad = parity("conv_transpose2d", &bwd, &p, &[("d_output", &[1.0])], 1e-3);
    assert!(
        !bad,
        "conv_transpose2d: gradients disagree CPU vs CUDA (see above)"
    );
}

/// Isolate which distinctive **codec** op's backward diverges on CUDA. The
/// hiphop conv-pyramid codec trains stably on CPU AND MLX at lr 3e-4 but
/// DIVERGES on CUDA (loss → thousands) — with bit-identical CUDA trajectories
/// across cuDNN/gather + TF32 on/off, i.e. a deterministic CUDA-vs-reference
/// gradient discrepancy, not precision. These exercise the ops the codec uses
/// that the primitive suite above does not: pixel-shuffle (6-D transpose),
/// tanh/gelu, exp (entropy `1/σ²`), concat + narrow (context entropy params),
/// and a masked conv (type-A causal). Square-loss so gradients are non-trivial.
#[test]
fn codec_ops_isolation() {
    if !is_available(target()) {
        eprintln!("codec_ops: {:?} unavailable — skipping", target());
        return;
    }
    // loss = sum(x^2) over a rank-`nd` tensor → grad exercises the op's routing.
    fn sqsum(g: &mut Graph, x: NodeId, nd: usize) -> NodeId {
        let sq = g.mul(x, x);
        reduce(
            g,
            ReduceOp::Sum,
            sq,
            (0..nd).collect(),
            Shape::from_dims(&[], F),
        )
    }
    let mut bad = Vec::new();

    // 1) pixel-shuffle 2× upsample: reshape → 6-D transpose → reshape (the up2 core).
    {
        let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
        let mut g = Graph::new("pshuf");
        let x = g.param("x", Shape::new(&[b, c * 4, h, w], F));
        let r = g.reshape_(x, vec![b as i64, c as i64, 2, 2, h as i64, w as i64]);
        let t = g.transpose_(r, vec![0, 1, 4, 2, 5, 3]);
        let o = g.reshape_(t, vec![b as i64, c as i64, (2 * h) as i64, (2 * w) as i64]);
        let loss = sqsum(&mut g, o, 4);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "pixel_shuffle",
            &bwd,
            &[("x", seeded(b * c * 4 * h * w, 1))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("pixel_shuffle");
        }
    }
    // 2) tanh (bound_ls / soft clamps).
    {
        let mut g = Graph::new("tanh");
        let x = g.param("x", Shape::new(&[64], F));
        let y = g.tanh(x);
        let loss = sqsum(&mut g, y, 1);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "tanh",
            &bwd,
            &[("x", seeded(64, 2))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("tanh");
        }
    }
    // 3) gelu (transforms).
    {
        let mut g = Graph::new("gelu");
        let x = g.param("x", Shape::new(&[64], F));
        let y = g.gelu(x);
        let loss = sqsum(&mut g, y, 1);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        // 3e-3: decomposed CUDA path uses the tanh-approx gelu derivative, CPU the
        // exact erf form; they agree to ~1-2e-3 (vs ~1.0 before the fix).
        if parity(
            "gelu",
            &bwd,
            &[("x", seeded(64, 3))],
            &[("d_output", &[1.0])],
            3e-3,
        ) {
            bad.push("gelu");
        }
    }
    // 4) exp entropy term: loss = sum((yq^2 * exp(-2*ls))) — the rate 1/σ² path.
    {
        let mut g = Graph::new("exprate");
        let ls = g.param("ls", Shape::new(&[64], F));
        let yq = g.param("yq", Shape::new(&[64], F));
        let neg2 = g.constant(-2.0, F);
        let ls2 = g.mul(ls, neg2);
        let iv = g.exp(ls2);
        let yq2 = g.mul(yq, yq);
        let q = g.mul(yq2, iv);
        let loss = reduce(&mut g, ReduceOp::Sum, q, vec![0], Shape::from_dims(&[], F));
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[ls, yq]);
        if parity(
            "exp_rate",
            &bwd,
            &[("ls", seeded(64, 4)), ("yq", seeded(64, 5))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("exp_rate");
        }
    }
    // 5) concat on channel axis (context entropy concats ctx + hyper log-σ).
    {
        let (b, c, h, w) = (2usize, 4usize, 3usize, 3usize);
        let mut g = Graph::new("concat");
        let a = g.param("a", Shape::new(&[b, c, h, w], F));
        let bb = g.param("bb", Shape::new(&[b, c, h, w], F));
        let cat = g.concat(vec![a, bb], 1, Shape::new(&[b, 2 * c, h, w], F));
        let loss = sqsum(&mut g, cat, 4);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[a, bb]);
        if parity(
            "concat",
            &bwd,
            &[
                ("a", seeded(b * c * h * w, 6)),
                ("bb", seeded(b * c * h * w, 7)),
            ],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("concat");
        }
    }
    // 6) narrow on channel axis (split entropy params into μ / log-σ).
    {
        let (b, c, h, w) = (2usize, 8usize, 3usize, 3usize);
        let mut g = Graph::new("narrow");
        let x = g.param("x", Shape::new(&[b, c, h, w], F));
        let mu = g.narrow_(x, 1, 0, c / 2);
        let sd = g.narrow_(x, 1, c / 2, c / 2);
        let s = g.add(mu, sd);
        let loss = sqsum(&mut g, s, 4);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "narrow_split",
            &bwd,
            &[("x", seeded(b * c * h * w, 8))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("narrow_split");
        }
    }
    // 7) masked conv: conv2d over (weight × constant causal mask) — Minnen context.
    {
        let (b, cin, cout, k, h, w) = (2usize, 3usize, 4usize, 3usize, 6usize, 6usize);
        let mut g = Graph::new("mconv");
        let x = g.param("x", Shape::new(&[b, cin, h, w], F));
        let wt = g.param("wt", Shape::new(&[cout, cin, k, k], F));
        // fixed 0/1 causal mask (raster-earlier positions only).
        let mut mask = vec![0f32; cout * cin * k * k];
        let center = k / 2;
        for co in 0..cout {
            for ci in 0..cin {
                for i in 0..k {
                    for j in 0..k {
                        if i < center || (i == center && j < center) {
                            mask[((co * cin + ci) * k + i) * k + j] = 1.0;
                        }
                    }
                }
            }
        }
        let bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mc = g.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[cout, cin, k, k], F),
        );
        let wm = g.mul(wt, mc);
        let y = g.conv2d(x, wm, [k, k], [1, 1], [center, center], [1, 1], 1);
        let loss = sqsum(&mut g, y, 4);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, wt]);
        if parity(
            "masked_conv",
            &bwd,
            &[
                ("x", seeded(b * cin * h * w, 9)),
                ("wt", seeded(cout * cin * k * k, 10)),
            ],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("masked_conv");
        }
    }

    eprintln!("codec_ops_isolation MISMATCHES: {bad:?}");
    assert!(bad.is_empty(), "codec ops disagree CPU vs CUDA: {bad:?}");
}

/// Round 2: codec ops the first pass didn't cover, at TRAINING-scale magnitudes.
/// After the gelu-backward fix the conv codec stops diverging on CUDA but hits a
/// deterministic NaN (a `+inf` gradient cascading through Conv2dBackwardWeight),
/// LR-independent → a second op whose CUDA backward overflows where CPU stays
/// finite. Suspects: STRIDED conv backward (analysis is k5/s2), 4-D mean-reduce
/// backward (the loss reduction), and the composed entropy path
/// `yq² · exp(−2·(5·tanh(z/5)))`. Inputs ×4 to probe magnitude sensitivity.
#[test]
fn codec_ops_isolation2() {
    if !is_available(target()) {
        eprintln!("codec_ops2: {:?} unavailable — skipping", target());
        return;
    }
    fn sqsum(g: &mut Graph, x: NodeId, nd: usize) -> NodeId {
        let sq = g.mul(x, x);
        reduce(
            g,
            ReduceOp::Sum,
            sq,
            (0..nd).collect(),
            Shape::from_dims(&[], F),
        )
    }
    let big = |n: usize, s: u64| -> Vec<f32> { seeded(n, s).iter().map(|v| v * 4.0).collect() };
    let mut bad = Vec::new();

    // 1) strided conv2d (k5, s2, pad2) — the codec's analysis transform. dW + dX.
    {
        let (b, cin, cout, h, w) = (2usize, 3usize, 5usize, 16usize, 16usize);
        let mut g = Graph::new("sconv");
        let x = g.param("x", Shape::new(&[b, cin, h, w], F));
        let wt = g.param("wt", Shape::new(&[cout, cin, 5, 5], F));
        let y = g.conv2d(x, wt, [5, 5], [2, 2], [2, 2], [1, 1], 1);
        let rank = g.node(y).shape.rank();
        let loss = sqsum(&mut g, y, rank);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, wt]);
        if parity(
            "strided_conv_s2",
            &bwd,
            &[
                ("x", big(b * cin * h * w, 11)),
                ("wt", big(cout * cin * 25, 12)),
            ],
            &[("d_output", &[1.0])],
            2e-3,
        ) {
            bad.push("strided_conv_s2");
        }
    }
    // 2) 4-D mean-reduce backward (loss = mean over [b,c,h,w]).
    {
        let (b, c, h, w) = (2usize, 4usize, 8usize, 8usize);
        let mut g = Graph::new("mean4d");
        let x = g.param("x", Shape::new(&[b, c, h, w], F));
        let loss = reduce(
            &mut g,
            ReduceOp::Mean,
            x,
            vec![0, 1, 2, 3],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        if parity(
            "mean_reduce_4d",
            &bwd,
            &[("x", big(b * c * h * w, 13))],
            &[("d_output", &[1.0])],
            1e-3,
        ) {
            bad.push("mean_reduce_4d");
        }
    }
    // 3) composed entropy: loss = sum( yq² · exp(−2·(5·tanh(z/5))) ) — the rate
    //    term's 1/σ² with the bounded log-σ. Overflow-prone if any op mis-scales.
    {
        let mut g = Graph::new("entropy");
        let z = g.param("z", Shape::new(&[128], F));
        let yq = g.param("yq", Shape::new(&[128], F));
        let inv5 = g.constant(0.2, F);
        let five = g.constant(5.0, F);
        let zt = g.mul(z, inv5);
        let th = g.tanh(zt);
        let ls = g.mul(th, five); // bounded log-σ ∈ [−5,5]
        let neg2 = g.constant(-2.0, F);
        let ls2 = g.mul(ls, neg2);
        let iv = g.exp(ls2); // 1/σ²
        let yq2 = g.mul(yq, yq);
        let q = g.mul(yq2, iv);
        let loss = reduce(&mut g, ReduceOp::Sum, q, vec![0], Shape::from_dims(&[], F));
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[z, yq]);
        if parity(
            "entropy_rate",
            &bwd,
            &[("z", big(128, 14)), ("yq", big(128, 15))],
            &[("d_output", &[1.0])],
            2e-3,
        ) {
            bad.push("entropy_rate");
        }
    }
    // 4) gelu with LARGE inputs (×8) — probe the tanh-approx-deriv for overflow.
    {
        let mut g = Graph::new("gelu_big");
        let x = g.param("x", Shape::new(&[64], F));
        let y = g.gelu(x);
        let loss = sqsum(&mut g, y, 1);
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        let xv: Vec<f32> = seeded(64, 16).iter().map(|v| v * 8.0).collect();
        if parity(
            "gelu_big",
            &bwd,
            &[("x", xv)],
            &[("d_output", &[1.0])],
            5e-3,
        ) {
            bad.push("gelu_big");
        }
    }

    eprintln!("codec_ops_isolation2 MISMATCHES: {bad:?}");
    assert!(
        bad.is_empty(),
        "codec ops (round 2) disagree CPU vs CUDA: {bad:?}"
    );
}

/// Lock default-on `RLX_CUDA_CONV_STABLE_BWD`: cuDNN v7 can pick Winograd/FFT
/// for backward-data/-filter; those transforms amplify intermediates and have
/// overflowed to `+inf` at codec scale (batch 32, ch 64) where CPU stayed
/// finite. The failsafe prefers IMPLICIT_GEMM (`ALGO_1`) unless explicitly
/// disabled with `RLX_CUDA_CONV_STABLE_BWD=0`.
///
/// Primary assert: every CUDA output is finite. Secondary: grads match CPU
/// (mean loss — a raw sum over ~1e5 elems makes abs-error unreadable).
/// Skips when libcudnn is unloadable so we don't silently lock the im2col path.
#[test]
fn cudnn_stable_conv_bwd_finite_matches_cpu() {
    if !is_available(target()) {
        eprintln!("cudnn_stable_bwd: {:?} unavailable — skipping", target());
        return;
    }
    #[cfg(feature = "cuda")]
    if matches!(target(), Device::Cuda) && rlx_cuda::device::cuda_dnn_handle().is_none() {
        eprintln!("cudnn_stable_bwd: libcudnn unloadable — skipping (would only exercise im2col)");
        return;
    }
    // Failsafe is default-on; clear any ambient opt-out for this process.
    rlx_ir::env::unset("RLX_CUDA_CONV_STABLE_BWD");
    #[cfg(feature = "cuda")]
    rlx_cuda::reload_runtime_config();

    // Codec-ish analysis shape: k5/s2/pad2 at the channel/batch that overflowed.
    let (b, cin, cout, h, w) = (32usize, 64usize, 64usize, 16usize, 16usize);
    let mut g = Graph::new("stable_bwd");
    let x = g.param("x", Shape::new(&[b, cin, h, w], F));
    let wt = g.param("wt", Shape::new(&[cout, cin, 5, 5], F));
    let y = g.conv2d(x, wt, [5, 5], [2, 2], [2, 2], [1, 1], 1);
    let rank = g.node(y).shape.rank();
    let sq = g.mul(y, y);
    let loss = reduce(
        &mut g,
        ReduceOp::Mean,
        sq,
        (0..rank).collect(),
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, wt]);

    let params = [
        ("x", seeded(b * cin * h * w, 21)),
        ("wt", seeded(cout * cin * 25, 22)),
    ];
    let inputs: &[(&str, &[f32])] = &[("d_output", &[1.0])];

    let cpu = run(&bwd, Device::Cpu, &params, inputs);
    let gpu = run(&bwd, target(), &params, inputs);
    assert_eq!(cpu.len(), gpu.len());

    for (i, g) in gpu.iter().enumerate() {
        let bad = g.iter().find(|v| !v.is_finite());
        assert!(
            bad.is_none(),
            "stable_bwd CUDA output[{i}] non-finite (first={:?}) — Winograd/FFT overflow?",
            bad
        );
    }

    let tags = ["loss", "grad_x", "grad_wt"];
    let tols = [5e-3f32, 5e-3, 2e-2];
    let mut bad = false;
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        let e = max_err(c, g);
        let tol = tols.get(i).copied().unwrap_or(5e-3);
        let tag = tags.get(i).copied().unwrap_or("?");
        eprintln!(
            "  stable_bwd[{i}] {tag:8} max_err={e:.6} tol={tol} {}",
            if e < tol { "ok" } else { "MISMATCH" }
        );
        bad |= e >= tol;
    }
    assert!(
        !bad,
        "cuDNN stable conv bwd: disagrees with CPU (see above)"
    );
}

/// Reproduce the codec's zero-dW bug in isolation: the residual-branch convs
/// (res.a2, res.s1, res.s2) get ~ZERO weight-gradient on CUDA while CPU/MLX
/// compute it correctly (measured via per-tensor param-update cosine). The
/// INPUT gradient flows (so upstream layers still train) but Conv2dBackwardWeight
/// yields ≈0. These exercise the exact failing shapes vs known-good controls.
#[test]
fn codec_conv_shapes_dw() {
    if !is_available(target()) {
        eprintln!("codec_conv_shapes: {:?} unavailable — skipping", target());
        return;
    }
    // (label, b, cin, cout, k, stride, pad, h, w) — grad_w is output index 2.
    let cases: &[(&str, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
        ("res_a1_ctl", 2, 1, 64, 5, 2, 2, 256, 256), // control — trains OK in codec
        ("base_a2_ctl", 2, 64, 12, 3, 1, 1, 64, 64), // control — trains OK
        ("res_a2_BAD", 2, 64, 12, 5, 2, 2, 128, 128), // codec res.a2 — ZERO dW on CUDA
        ("res_s1_BAD", 2, 12, 256, 3, 1, 1, 64, 64), // codec res.s1 (up2 conv) — ZERO
    ];
    let mut bad = Vec::new();
    for &(lbl, b, cin, cout, k, s, p, h, w) in cases {
        let mut g = Graph::new(lbl);
        let x = g.param("x", Shape::new(&[b, cin, h, w], F));
        let wt = g.param("wt", Shape::new(&[cout, cin, k, k], F));
        let y = g.conv2d(x, wt, [k, k], [s, s], [p, p], [1, 1], 1);
        let rank = g.node(y).shape.rank();
        let sq = g.mul(y, y);
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            sq,
            (0..rank).collect(),
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, wt]);
        let p_in = vec![
            ("x", seeded(b * cin * h * w, 1)),
            ("wt", seeded(cout * cin * k * k, 2)),
        ];
        // outputs: [loss, grad_x, grad_wt]; index 2 = dW is the one under suspicion.
        if parity(lbl, &bwd, &p_in, &[("d_output", &[1.0])], 2e-3) {
            bad.push(lbl);
        }
    }
    eprintln!("codec_conv_shapes_dw MISMATCHES: {bad:?}");
    assert!(bad.is_empty(), "conv dW disagrees CPU vs CUDA: {bad:?}");
}

/// Reproduce the codec's zero-dW in CONTEXT (single convs alone don't). Two
/// patterns from the residual branch: (A) conv -> pixel-shuffle (up2) -> loss,
/// and (B) conv -> gelu -> conv -> loss. Checks the conv WEIGHT gradients (dW)
/// CPU vs CUDA — the codec zeroes these on CUDA while dx stays correct.
#[test]
fn codec_context_dw() {
    if !is_available(target()) {
        eprintln!("codec_context_dw: skip");
        return;
    }
    let mut bad = Vec::new();
    // (A) conv -> pixel-shuffle (res.s1 pattern): cin=12, cout*4=256, then shuffle to 2h,2w.
    {
        let (b, cin, cout, h, w) = (2usize, 12usize, 64usize, 64usize, 64usize);
        let mut g = Graph::new("convPS");
        let x = g.param("x", Shape::new(&[b, cin, h, w], F));
        let wt = g.param("wt", Shape::new(&[cout * 4, cin, 3, 3], F));
        let y = g.conv2d(x, wt, [3, 3], [1, 1], [1, 1], [1, 1], 1);
        let r = g.reshape_(y, vec![b as i64, cout as i64, 2, 2, h as i64, w as i64]);
        let t = g.transpose_(r, vec![0, 1, 4, 2, 5, 3]);
        let o = g.reshape_(
            t,
            vec![b as i64, cout as i64, (2 * h) as i64, (2 * w) as i64],
        );
        let sq = g.mul(o, o);
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            sq,
            vec![0, 1, 2, 3],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, wt]);
        if parity(
            "convPS_dW",
            &bwd,
            &[
                ("x", seeded(b * cin * h * w, 1)),
                ("wt", seeded(cout * 4 * cin * 9, 2)),
            ],
            &[("d_output", &[1.0])],
            5e-3,
        ) {
            bad.push("convPS");
        }
    }
    // (B) conv -> gelu -> conv (res.a1->gelu->res.a2 pattern): check BOTH dW.
    {
        let (b, h, w) = (2usize, 128usize, 128usize);
        let mut g = Graph::new("convGeluConv");
        let x = g.param("x", Shape::new(&[b, 1, 256, 256], F));
        let w1 = g.param("w1", Shape::new(&[64, 1, 5, 5], F)); // res.a1
        let a = g.conv2d(x, w1, [5, 5], [2, 2], [2, 2], [1, 1], 1); // ->[b,64,128,128]
        let ga = g.gelu_approx(a);
        let w2 = g.param("w2", Shape::new(&[12, 64, 5, 5], F)); // res.a2
        let y = g.conv2d(ga, w2, [5, 5], [2, 2], [2, 2], [1, 1], 1); // ->[b,12,64,64]
        let sq = g.mul(y, y);
        let loss = reduce(
            &mut g,
            ReduceOp::Sum,
            sq,
            vec![0, 1, 2, 3],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let _ = (h, w);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w1, w2]);
        if parity(
            "convGeluConv_dW",
            &bwd,
            &[
                ("x", seeded(b * 256 * 256, 3)),
                ("w1", seeded(64 * 25, 4)),
                ("w2", seeded(12 * 64 * 25, 5)),
            ],
            &[("d_output", &[1.0])],
            1e-2,
        ) {
            bad.push("convGeluConv");
        }
    }
    eprintln!("codec_context_dw MISMATCHES: {bad:?}");
    assert!(bad.is_empty(), "context dW disagrees: {bad:?}");
}

/// Reproduce the codec's residual-synthesis `+inf` in the backward. up2 chain:
/// yq[b,12,64,64] -> conv+pixelshuffle -> [b,64,128,128] -> gelu -> conv+shuffle
/// -> [b,1,256,256] -> loss. The codec NaNs with a +inf at a backward Reshape
/// here (large-spatial up2 backward). Check CUDA gradients stay finite.
#[test]
fn codec_up2_chain_finite() {
    if !is_available(target()) {
        eprintln!("up2_chain: skip");
        return;
    }
    let (b, lat, c) = (2usize, 12usize, 64usize);
    let mut g = Graph::new("up2chain");
    let x = g.param("x", Shape::new(&[b, lat, 64, 64], F));
    let w1 = g.param("w1", Shape::new(&[c * 4, lat, 3, 3], F));
    let y1 = g.conv2d(x, w1, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let r1 = g.reshape_(y1, vec![b as i64, c as i64, 2, 2, 64, 64]);
    let t1 = g.transpose_(r1, vec![0, 1, 4, 2, 5, 3]);
    let s1 = g.reshape_(t1, vec![b as i64, c as i64, 128, 128]);
    let gs1 = g.gelu_approx(s1);
    let w2 = g.param("w2", Shape::new(&[4, c, 3, 3], F));
    let y2 = g.conv2d(gs1, w2, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let r2 = g.reshape_(y2, vec![b as i64, 1, 2, 2, 128, 128]);
    let t2 = g.transpose_(r2, vec![0, 1, 4, 2, 5, 3]);
    let o = g.reshape_(t2, vec![b as i64, 1, 256, 256]);
    let sq = g.mul(o, o);
    let loss = reduce(
        &mut g,
        ReduceOp::Sum,
        sq,
        vec![0, 1, 2, 3],
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w1, w2]);
    let outs = run(
        &bwd,
        target(),
        &[
            ("x", seeded(b * lat * 64 * 64, 1)),
            ("w1", seeded(c * 4 * lat * 9, 2)),
            ("w2", seeded(4 * c * 9, 3)),
        ],
        &[("d_output", &[1.0])],
    );
    let mut any_bad = false;
    for (i, o) in outs.iter().enumerate() {
        let nfin = o.iter().filter(|v| !v.is_finite()).count();
        if nfin > 0 {
            eprintln!("  up2chain out[{i}]: {nfin} non-finite of {}", o.len());
            any_bad = true;
        }
    }
    eprintln!("up2_chain reproduces +inf: {any_bad}");
    assert!(
        !any_bad,
        "CUDA up2-chain backward produced non-finite gradients (repro of codec +inf)"
    );
}

/// The codec's residual bias-gradient hits ~874670 on CUDA (vs CPU max node 8.5)
/// and grows to +inf. bias-grad = reduce_sum(dy) over batch×spatial (65536
/// elems). This checks the large-spatial bias-broadcast backward (the reduce)
/// at the exact codec shape [16,256,64,64] — is CUDA's d_bias correct?
#[test]
fn codec_bias_reduce_large() {
    if !is_available(target()) {
        eprintln!("bias_reduce: skip");
        return;
    }
    let (b, c, h, w) = (16usize, 256usize, 64usize, 64usize);
    let mut g = Graph::new("biasred");
    let x = g.param("x", Shape::new(&[b, c, h, w], F));
    let bias = g.param("bias", Shape::new(&[1, c, 1, 1], F));
    let y = g.add(x, bias); // broadcast add
    let sq = g.mul(y, y);
    let loss = reduce(
        &mut g,
        ReduceOp::Sum,
        sq,
        vec![0, 1, 2, 3],
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, bias]);
    // small inputs so d_bias is modestly-sized; a wrong reduce shows as huge/mismatch.
    let xv: Vec<f32> = seeded(b * c * h * w, 1).iter().map(|v| v * 0.1).collect();
    let bv: Vec<f32> = seeded(c, 2).iter().map(|v| v * 0.1).collect();
    let cpu = run(
        &bwd,
        Device::Cpu,
        &[("x", xv.clone()), ("bias", bv.clone())],
        &[("d_output", &[1.0])],
    );
    let cud = run(
        &bwd,
        target(),
        &[("x", xv), ("bias", bv)],
        &[("d_output", &[1.0])],
    );
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    // outputs: [loss, d_x, d_bias]; index 2 = d_bias
    for (i, (cp, cd)) in cpu.iter().zip(&cud).enumerate() {
        let me = max_err(cp, cd);
        eprintln!(
            "  biasred[{i}] |cpu|={:.4} |cuda|={:.4} max_err={me:.4}",
            l2(cp),
            l2(cd)
        );
    }
    let me = max_err(&cpu[2], &cud[2]);
    let rel = me / (l2(&cpu[2]).max(1e-9));
    assert!(
        rel < 1e-2,
        "d_bias (large-spatial reduce) disagrees CPU vs CUDA: rel={rel}"
    );
}

/// Faithful residual-synthesis repro with MEAN loss + conv BIASES (the codec's
/// up2 uses conv()+bias). yq[b,12,64,64] -> up2(conv12->256 +bias)+shuffle ->
/// [b,64,128,128] -> gelu -> up2(conv64->4 +bias)+shuffle -> [b,1,256,256] ->
/// MEAN((.)²). Codec's residual bias-grad is ~874670 on CUDA but should be ~1e6×
/// smaller under a mean loss. Compare ALL grads CPU vs CUDA (esp. biases).
#[test]
fn codec_residual_synth_meanloss() {
    if !is_available(target()) {
        eprintln!("resid_synth: skip");
        return;
    }
    let (b, lat, c) = (4usize, 12usize, 64usize);
    let mut g = Graph::new("residsynth");
    let x = g.param("x", Shape::new(&[b, lat, 64, 64], F));
    // up2 #1: conv lat->c*4 + bias, pixelshuffle to 128
    let w1 = g.param("w1", Shape::new(&[c * 4, lat, 3, 3], F));
    let b1 = g.param("b1", Shape::new(&[1, (c * 4) as i64 as usize, 1, 1], F));
    let y1 = g.conv2d(x, w1, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let y1 = g.add(y1, b1);
    let r1 = g.reshape_(y1, vec![b as i64, c as i64, 2, 2, 64, 64]);
    let t1 = g.transpose_(r1, vec![0, 1, 4, 2, 5, 3]);
    let s1 = g.reshape_(t1, vec![b as i64, c as i64, 128, 128]);
    let gs1 = g.gelu_approx(s1);
    // up2 #2: conv c->1*4 + bias, pixelshuffle to 256
    let w2 = g.param("w2", Shape::new(&[4, c, 3, 3], F));
    let b2 = g.param("b2", Shape::new(&[1, 4, 1, 1], F));
    let y2 = g.conv2d(gs1, w2, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let y2 = g.add(y2, b2);
    let r2 = g.reshape_(y2, vec![b as i64, 1, 2, 2, 128, 128]);
    let t2 = g.transpose_(r2, vec![0, 1, 4, 2, 5, 3]);
    let o = g.reshape_(t2, vec![b as i64, 1, 256, 256]);
    let sq = g.mul(o, o);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![sq],
        Shape::from_dims(&[], F),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, w1, b1, w2, b2]);
    let sm = |n, s| -> Vec<f32> { seeded(n, s).iter().map(|v| v * 0.2).collect() };
    let ins = [
        ("x", sm(b * lat * 64 * 64, 1)),
        ("w1", sm(c * 4 * lat * 9, 2)),
        ("b1", sm(c * 4, 3)),
        ("w2", sm(4 * c * 9, 4)),
        ("b2", sm(4, 5)),
    ];
    let cpu = run(&bwd, Device::Cpu, &ins, &[("d_output", &[1.0])]);
    let cud = run(&bwd, target(), &ins, &[("d_output", &[1.0])]);
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let lbl = ["loss", "d_x", "d_w1", "d_b1", "d_w2", "d_b2"];
    let mut bad = false;
    for (i, (cp, cd)) in cpu.iter().zip(&cud).enumerate() {
        let (nc, nd) = (l2(cp), l2(cd));
        let rel = (nc - nd).abs() / nc.max(1e-9);
        eprintln!(
            "  {} |cpu|={:.6} |cuda|={:.6} rel={:.4}{}",
            lbl.get(i).unwrap_or(&"?"),
            nc,
            nd,
            rel,
            if rel > 0.05 { "  MISMATCH" } else { "" }
        );
        if rel > 0.05 {
            bad = true;
        }
    }
    assert!(!bad, "residual synth grads disagree CPU vs CUDA");
}

/// Minimal test of the all-axis MEAN backward at the codec's exact size. The
/// codec's dist = mean(sq) over [16,1,256,256]=1M; on CUDA its gradients come
/// out ~1e6× too large (≈ N), i.e. the 1/N is lost in the full graph. Does the
/// all-axis mean backward apply 1/N at 1M elements? grad(mean(x²)) = 2x/N.
#[test]
fn mean_backward_large_1m() {
    if !is_available(target()) {
        eprintln!("mean_1m: skip");
        return;
    }
    for &(bb, cc, hh, ww) in &[
        (16usize, 1usize, 256usize, 256usize),
        (4, 1, 256, 256),
        (16, 64, 64, 64),
    ] {
        let n = (bb * cc * hh * ww) as f32;
        let mut g = Graph::new("mean1m");
        let x = g.param("x", Shape::new(&[bb, cc, hh, ww], F));
        let sq = g.mul(x, x);
        let loss = g.add_node(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes: vec![0, 1, 2, 3],
                keep_dim: false,
            },
            vec![sq],
            Shape::from_dims(&[], F),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[x]);
        let xv = seeded(bb * cc * hh * ww, 7);
        let cpu = run(
            &bwd,
            Device::Cpu,
            &[("x", xv.clone())],
            &[("d_output", &[1.0])],
        );
        let cud = run(&bwd, target(), &[("x", xv)], &[("d_output", &[1.0])]);
        let l2 = |v: &[f32]| v.iter().map(|z| z * z).sum::<f32>().sqrt();
        // grad_x = 2x/N ; index 1
        let rel = max_err(&cpu[1], &cud[1]) / l2(&cpu[1]).max(1e-12);
        eprintln!(
            "  mean1m [{bb},{cc},{hh},{ww}] N={n:.0}: |cpu_grad|={:.6} |cuda_grad|={:.6} rel_err={rel:.4}",
            l2(&cpu[1]),
            l2(&cud[1])
        );
        assert!(rel < 1e-2, "mean backward wrong 1/N at N={n}");
    }
}
