// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity for Cumsum / Concat / Pool — lowered to existing kernels (triangular
// matmul, masked-gather blend, reduce-with-window-groups) — vs RLX's CPU
// backend.

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

#[test]
fn cumsum_inclusive_and_exclusive() {
    let xs = f32s(8, |i| i as f32 * 0.5 - 1.0);
    for exclusive in [false, true] {
        let mut g = Graph::new("cs");
        let x = g.input("x", sh(&[2, 4]));
        let y = g.cumsum(x, -1, exclusive, sh(&[2, 4]));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
}

#[test]
fn concat_axis1() {
    let a = f32s(2 * 2, |i| i as f32);
    let b = f32s(2 * 3, |i| 10.0 + i as f32);
    let mut g = Graph::new("cat");
    let ai = g.input("a", sh(&[2, 2]));
    let bi = g.input("b", sh(&[2, 3]));
    let y = g.concat(vec![ai, bi], 1, sh(&[2, 5]));
    g.set_outputs(vec![y]);
    parity(g, &[("a", &a), ("b", &b)]);
}

#[test]
fn concat_axis0_three_inputs() {
    let a = f32s(3, |i| i as f32);
    let b = f32s(6, |i| 10.0 + i as f32);
    let c = f32s(3, |i| 100.0 + i as f32);
    let mut g = Graph::new("cat3");
    let ai = g.input("a", sh(&[1, 3]));
    let bi = g.input("b", sh(&[2, 3]));
    let ci = g.input("c", sh(&[1, 3]));
    let y = g.concat(vec![ai, bi, ci], 0, sh(&[4, 3]));
    g.set_outputs(vec![y]);
    parity(g, &[("a", &a), ("b", &b), ("c", &c)]);
}

fn pool(kind: ReduceOp) -> (Graph, Vec<f32>) {
    // NCHW [1,1,4,4], 2x2 stride 2 → [1,1,2,2].
    let xs = f32s(16, |i| (i as f32 * 0.7).sin() + 0.1 * i as f32);
    let mut g = Graph::new("pool");
    let x = g.input("x", sh(&[1, 1, 4, 4]));
    let y = g.add_node(
        Op::Pool {
            kind,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        sh(&[1, 1, 2, 2]),
    );
    g.set_outputs(vec![y]);
    (g, xs)
}

#[test]
fn max_pool2d() {
    let (g, xs) = pool(ReduceOp::Max);
    parity(g, &[("x", &xs)]);
}

#[test]
fn avg_pool2d() {
    let (g, xs) = pool(ReduceOp::Mean);
    parity(g, &[("x", &xs)]);
}

fn conv(pad: usize) -> (Graph, Vec<f32>, Vec<f32>) {
    // input [1,2,4,4], weight [3,2,2,2], stride 1, dilation 1, groups 1.
    let (cin, h, w) = (2usize, 4usize, 4usize);
    let (cout, kh, kw) = (3usize, 2usize, 2usize);
    let xs = f32s(cin * h * w, |i| (i as f32 * 0.5).sin());
    let ws = f32s(cout * cin * kh * kw, |i| 0.1 * (i as f32 % 7.0) - 0.3);
    let mut g = Graph::new("conv");
    let x = g.input("x", sh(&[1, cin, h, w]));
    let weight = g.input("w", sh(&[cout, cin, kh, kw]));
    let y = g.conv2d(x, weight, [kh, kw], [1, 1], [pad, pad], [1, 1], 1);
    g.set_outputs(vec![y]);
    (g, xs, ws)
}

#[test]
fn conv2d_no_pad() {
    let (g, xs, ws) = conv(0);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

#[test]
fn conv2d_padded() {
    let (g, xs, ws) = conv(1);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

/// Grouped conv: 4 input channels in 2 groups, 6 outputs, padded.
#[test]
fn conv2d_grouped() {
    let (cin, h, w, cout, k, groups) = (4usize, 5usize, 5usize, 6usize, 3usize, 2usize);
    let xs = f32s(cin * h * w, |i| (i as f32 * 0.37).cos());
    let ws = f32s(cout * (cin / groups) * k * k, |i| {
        0.07 * (i as f32 % 11.0) - 0.35
    });
    let mut g = Graph::new("conv_grouped");
    let x = g.input("x", sh(&[1, cin, h, w]));
    let weight = g.input("w", sh(&[cout, cin / groups, k, k]));
    let y = g.conv2d(x, weight, [k, k], [1, 1], [1, 1], [1, 1], groups);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

/// Depthwise causal 1-D conv as LFM2's ShortConv builds it: length in H,
/// kernel `[k, 1]`, `groups = channels`, left padding already concatenated.
#[test]
fn conv1d_depthwise() {
    let (c, l, k) = (8usize, 7usize, 3usize);
    let xs = f32s(c * l, |i| (i as f32 * 0.21).sin());
    let ws = f32s(c * k, |i| 0.1 * (i as f32 % 5.0) - 0.2);
    let mut g = Graph::new("conv_depthwise");
    let x = g.input("x", sh(&[1, c, l, 1]));
    let weight = g.input("w", sh(&[c, 1, k, 1]));
    let y = g.conv2d(x, weight, [k, 1], [1, 1], [0, 0], [1, 1], c);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

/// `Op::Attention` has no WebGL kernel: legalization lowers it to matmul /
/// softmax. A decode step (1 query, 6 keys), causal, 4 query / 2 KV heads.
#[test]
fn attention_via_legalization() {
    let (h, kvh, d, sk) = (4usize, 2usize, 8usize, 6usize);
    let q = f32s(h * d, |i| (i as f32 * 0.3).sin());
    let k = f32s(sk * kvh * d, |i| (i as f32 * 0.17).cos());
    let v = f32s(sk * kvh * d, |i| (i as f32 * 0.11).sin());
    let mut g = Graph::new("attn");
    let qi = g.input("q", sh(&[1, 1, h * d]));
    let ki = g.input("k", sh(&[1, sk, kvh * d]));
    let vi = g.input("v", sh(&[1, sk, kvh * d]));
    let a = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: rlx_ir::op::MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        sh(&[1, 1, h * d]),
    );
    g.set_outputs(vec![a]);
    parity(g, &[("q", &q), ("k", &k), ("v", &v)]);
}

/// Partial rotation (`n_rot < head_dim`) with a table wider than `n_rot/2`,
/// in both pairing styles: NeoX `(i, i + n_rot/2)` and GPT-J `(2j, 2j + 1)`.
fn rope(style: rlx_ir::op::RopeStyle) {
    let (s, heads, d, n_rot, tab) = (5usize, 2usize, 8usize, 4usize, 4usize);
    let xs = f32s(s * heads * d, |i| (i as f32 * 0.29).sin());
    let cos = f32s(s * tab, |i| (i as f32 * 0.13).cos());
    let sin = f32s(s * tab, |i| (i as f32 * 0.13).sin());
    let mut g = Graph::new("rope");
    let x = g.input("x", sh(&[1, s, heads * d]));
    let c = g.input("cos", sh(&[s, tab]));
    let sn = g.input("sin", sh(&[s, tab]));
    let y = g.add_node(
        Op::Rope {
            head_dim: d,
            n_rot,
            style,
        },
        vec![x, c, sn],
        sh(&[1, s, heads * d]),
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("cos", &cos), ("sin", &sin)]);
}

#[test]
fn rope_neox_partial() {
    rope(rlx_ir::op::RopeStyle::NeoX);
}

#[test]
fn rope_gptj_partial() {
    rope(rlx_ir::op::RopeStyle::GptJ);
}

/// Activations added for the browser models (GELU in EmbeddingGemma and
/// Moonshine, softplus in Qwen3.5's gated delta-net), over a range wide
/// enough to reach both tails.
#[test]
fn gelu_and_softplus() {
    use rlx_ir::op::Activation;
    let xs = f32s(64, |i| (i as f32 - 32.0) * 0.4);
    for act in [
        Activation::Gelu,
        Activation::GeluApprox,
        Activation::Softplus,
    ] {
        let mut g = Graph::new("act");
        let x = g.input("x", sh(&[64]));
        let y = g.add_node(Op::Activation(act), vec![x], sh(&[64]));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
}
