// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
    for (i, (c, g)) in cpu.iter().zip(&cuda).enumerate() {
        let e = max_err(c, g);
        eprintln!(
            "  {name}[{i}] max_err={e:.6} {}",
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
