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

fn conv3d_case(pad: usize, stride: usize, dil: usize, groups: usize) {
    // input [2,4,4,5,5], weight [6,4/g,2,3,3].
    let (n, cin, d, h, w) = (2usize, 4usize, 4usize, 5usize, 5usize);
    let (cout, kd, kh, kw) = (6usize, 2usize, 3usize, 3usize);
    let xs = f32s(n * cin * d * h * w, |i| (i as f32 * 0.29).sin());
    let ws = f32s(cout * (cin / groups) * kd * kh * kw, |i| {
        0.05 * (i as f32 % 13.0) - 0.3
    });
    let mut g = Graph::new("conv3d");
    let x = g.input("x", sh(&[n, cin, d, h, w]));
    let weight = g.input("w", sh(&[cout, cin / groups, kd, kh, kw]));
    let y = g.conv3d(
        x,
        weight,
        [stride, stride, stride],
        [pad, pad, pad],
        [dil, dil, dil],
        groups,
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

#[test]
fn conv3d_no_pad() {
    conv3d_case(0, 1, 1, 1);
}

#[test]
fn conv3d_padded_strided() {
    conv3d_case(1, 2, 1, 1);
}

#[test]
fn conv3d_dilated() {
    conv3d_case(1, 1, 2, 1);
}

#[test]
fn conv3d_grouped() {
    conv3d_case(1, 1, 1, 2);
}

/// ConvTranspose3d (also the Conv3d input-gradient after decomposition):
/// strided, padded, dilated, with output padding, plain and grouped.
fn conv_transpose3d_case(groups: usize, dil: usize) {
    let (n, cin, d, h, w) = (2usize, 4usize, 3usize, 3usize, 4usize);
    let (cout, k) = (6usize, 3usize);
    let xs = f32s(n * cin * d * h * w, |i| (i as f32 * 0.31).cos());
    let ws = f32s(cin * (cout / groups) * k * k * k, |i| {
        0.03 * (i as f32 % 17.0) - 0.25
    });
    let mut g = Graph::new("conv_transpose3d");
    let x = g.input("x", sh(&[n, cin, d, h, w]));
    let weight = g.input("w", sh(&[cin, cout / groups, k, k, k]));
    let y = g.conv_transpose3d(
        x,
        weight,
        [2, 1, 2],
        [1, 0, 1],
        [dil, dil, dil],
        [1, 0, 1],
        groups,
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

#[test]
fn conv_transpose3d() {
    conv_transpose3d_case(1, 1);
}

#[test]
fn conv_transpose3d_grouped_dilated() {
    conv_transpose3d_case(2, 2);
}

/// A segmentation head as rlx-segment builds it: a biased dilated 3-D conv
/// (`FusedConvBiasAct`, rank 5) + GELU, a group norm over channels, then a
/// biased 1×1×1 conv to classes and the argmax over them. Folded and fused,
/// the plan holds no index table the size of an activation, and the class
/// scores never get a slot of their own.
#[test]
fn conv3d_head_folds_and_fuses() {
    use rlx_ir::op::Activation;
    let (n, cin, d, h, w, mid, classes) = (1usize, 2usize, 5usize, 4usize, 6usize, 3usize, 4usize);
    let xs = f32s(n * cin * d * h * w, |i| (i as f32 * 0.37).sin());
    let w1 = f32s(mid * cin * 27, |i| 0.07 * (i as f32 % 11.0) - 0.3);
    let b1 = [0.1f32, -0.2, 0.05];
    let w2 = f32s(classes * mid, |i| 0.3 * (i as f32 % 5.0) - 0.6);
    let b2 = [0.0f32, 0.2, -0.1, 0.05];
    let (ones, zeros) = (vec![1.0f32; mid], vec![0.0f32; mid]);
    let mut g = Graph::new("head");
    let x = g.input("x", sh(&[n, cin, d, h, w]));
    let wa = g.input("w1", sh(&[mid, cin, 3, 3, 3]));
    let ba = g.input("b1", sh(&[mid]));
    let wb = g.input("w2", sh(&[classes, mid, 1, 1, 1]));
    let bb = g.input("b2", sh(&[classes]));
    let gm = g.input("g", sh(&[mid]));
    let bt = g.input("b", sh(&[mid]));
    let conv = |g: &mut Graph, x, wt, b, c, k: usize, p: usize, dil: usize| {
        g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![k; 3],
                stride: vec![1; 3],
                padding: vec![p; 3],
                dilation: vec![dil; 3],
                groups: 1,
                activation: None,
                has_residual: false,
            },
            vec![x, wt, b],
            sh(&[n, c, d, h, w]),
        )
    };
    let y = conv(&mut g, x, wa, ba, mid, 3, 2, 2);
    let y = g.activation(Activation::Gelu, y, sh(&[n, mid, d, h, w]));
    let flat = g.reshape(
        y,
        vec![n as i64, mid as i64, (d * h * w) as i64, 1],
        sh(&[n, mid, d * h * w, 1]),
    );
    let y = g.group_norm(flat, gm, bt, mid, 1e-5);
    let y = g.reshape(
        y,
        vec![n as i64, mid as i64, d as i64, h as i64, w as i64],
        sh(&[n, mid, d, h, w]),
    );
    let y = conv(&mut g, y, wb, bb, classes, 1, 0, 1);
    let lab = g.argmax(y, 1, false, sh(&[n, d, h, w]));
    g.set_outputs(vec![lab]);
    let inputs: [(&str, &[f32]); 7] = [
        ("x", &xs),
        ("w1", &w1),
        ("b1", &b1),
        ("w2", &w2),
        ("b2", &b2),
        ("g", &ones),
        ("b", &zeros),
    ];
    parity(g.clone(), &inputs);

    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let act = n * mid * d * h * w;
    for s in &plan.steps {
        if let rlx_webgl::plan::Step::Gather { idx, .. } = s {
            assert!(
                idx.len() < act,
                "an activation-sized index table: {}",
                idx.len()
            );
        }
    }
    assert!(
        plan.steps
            .iter()
            .any(|s| matches!(s, rlx_webgl::plan::Step::Conv1ArgMax { .. })),
        "head not fused"
    );
    assert!(
        !plan
            .steps
            .iter()
            .any(|s| matches!(s, rlx_webgl::plan::Step::ArgAxis { .. }))
    );
}

/// 3-D max and average pooling (SynthStrip's 2×2×2 downsampling, and a
/// padded, overlapping case), then a three-input concat on channels.
#[test]
fn pool3d_and_channel_concat() {
    let (n, c, d, h, w) = (2usize, 3usize, 4usize, 6usize, 4usize);
    let xs = f32s(n * c * d * h * w, |i| (i as f32 * 0.61).sin());
    for (kind, k, s, p) in [
        (ReduceOp::Max, 2, 2, 0),
        (ReduceOp::Max, 3, 1, 1),
        (ReduceOp::Mean, 3, 2, 1),
    ] {
        let o = |l: usize| (l + 2 * p - k) / s + 1;
        let mut g = Graph::new("pool3d");
        let x = g.input("x", sh(&[n, c, d, h, w]));
        let y = g.add_node(
            Op::Pool {
                kind,
                kernel_size: vec![k; 3],
                stride: vec![s; 3],
                padding: vec![p; 3],
            },
            vec![x],
            sh(&[n, c, o(d), o(h), o(w)]),
        );
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
    let a = f32s(n * 2 * d * h * w, |i| i as f32);
    let b = f32s(n * d * h * w, |i| -(i as f32));
    let mut g = Graph::new("cat3");
    let xa = g.input("a", sh(&[n, 2, d, h, w]));
    let xb = g.input("b", sh(&[n, 1, d, h, w]));
    let xc = g.input("c", sh(&[n, c, d, h, w]));
    let y = g.concat(vec![xa, xb, xc], 1, sh(&[n, 3 + c, d, h, w]));
    g.set_outputs(vec![y]);
    parity(g, &[("a", &a), ("b", &b), ("c", &xs)]);
}

/// A Reshape between two tall shapes (rows over `TALL_ROWS`) that widening
/// lays out alike needs no copy: the plan reads the source instead.
#[test]
fn reshape_between_widened_layouts_aliases() {
    use rlx_ir::op::Activation;
    let n = 18000 * 3;
    let xs = f32s(n, |i| (i as f32 * 0.013).sin());
    let mut g = Graph::new("reshape_wide");
    let x = g.input("x", sh(&[18000, 3]));
    let a = g.add_node(Op::Activation(Activation::Relu), vec![x], sh(&[18000, 3]));
    let r = g.reshape(a, vec![9000, 6], sh(&[9000, 6]));
    let y = g.add_node(Op::Activation(Activation::Sigmoid), vec![r], sh(&[9000, 6]));
    g.set_outputs(vec![y]);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    assert!(
        !plan
            .steps
            .iter()
            .any(|s| matches!(s, rlx_webgl::Step::Copy { .. })),
        "a copy between identical layouts survived"
    );
    parity(g, &[("x", &xs)]);
}

/// MeshNet's layers: a biased conv then an activation, and an instance norm
/// over the flattened volume, reshaped back, then an activation. Both
/// activations are applied by the step writing their input (`act`), so the
/// plan has no `Unary` left — unless the conv has a second reader. Tall
/// enough (over `TALL_ROWS` rows) that the flattened and the volume layouts
/// both widen alike, as a brain's do, and the reshapes are aliases.
#[test]
fn activations_fold_into_conv_and_group_norm() {
    use rlx_ir::op::Activation;
    let (ci, co, d) = (2usize, 3usize, [64usize, 48, 6]);
    let sp = d.iter().product::<usize>();
    let xs = f32s(ci * sp, |i| (i as f32 * 0.37).sin());
    let ws = f32s(co * ci * 27, |i| (i as f32 * 0.11).cos() * 0.3);
    let bs = f32s(co, |i| i as f32 * 0.1 - 0.1);
    let vol = sh(&[1, co, d[0], d[1], d[2]]);
    for both in [false, true] {
        let mut g = Graph::new("meshnet_layer");
        let x = g.input("x", sh(&[1, ci, d[0], d[1], d[2]]));
        let w = g.input("w", sh(&[co, ci, 3, 3, 3]));
        let b = g.input("b", sh(&[co]));
        let y = g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![3; 3],
                stride: vec![1; 3],
                padding: vec![1; 3],
                dilation: vec![1; 3],
                groups: 1,
                activation: None,
                has_residual: false,
            },
            vec![x, w, b],
            vol.clone(),
        );
        let a = g.add_node(Op::Activation(Activation::Elu), vec![y], vol.clone());
        let flat = sh(&[1, co, sp, 1]);
        let f = g.reshape(a, vec![1, co as i64, sp as i64, 1], flat.clone());
        let gamma = g.input("g", sh(&[co]));
        let beta = g.input("bt", sh(&[co]));
        let n = g.group_norm(f, gamma, beta, co, 1e-5);
        let r = g.reshape(
            n,
            vec![1, co as i64, d[0] as i64, d[1] as i64, d[2] as i64],
            vol.clone(),
        );
        let out = g.add_node(Op::Activation(Activation::Gelu), vec![r], vol.clone());
        g.set_outputs(if both { vec![out, y] } else { vec![out] });
        let plan = rlx_webgl::build_plan(&g).expect("plan");
        let unary = plan
            .steps
            .iter()
            .filter(|s| matches!(s, rlx_webgl::Step::Unary { .. }))
            .count();
        assert_eq!(
            unary,
            usize::from(both),
            "both={both}: {unary} Unary steps left"
        );
        let gs = f32s(co, |i| 1.0 + i as f32 * 0.2);
        let bts = f32s(co, |i| i as f32 * 0.05);
        parity(
            g,
            &[("x", &xs), ("w", &ws), ("b", &bs), ("g", &gs), ("bt", &bts)],
        );
    }
}
