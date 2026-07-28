// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// On-device parity tests for the second wave of ops: comparison, select,
// expand, cumsum, scatter-add, the norm family, LoRA, and the vision ops
// (conv / conv_transpose / pool). Each checks CoreML output against a hand
// reference.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::CoremlExecutable;
use rlx_ir::op::{CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n ref {b:?}"
    );
}

fn node(g: &mut Graph, op: Op, ins: Vec<rlx_ir::NodeId>, shape: Shape) -> rlx_ir::NodeId {
    g.append_node(op, ins, shape, None)
}

#[test]
fn compare_gt() {
    let mut g = Graph::new("cmp");
    let a = g.input("a", Shape::new(&[4], DType::F32));
    let b = g.input("b", Shape::new(&[4], DType::F32));
    let c = node(
        &mut g,
        Op::Compare(CmpOp::Gt),
        vec![a, b],
        Shape::new(&[4], DType::Bool),
    );
    let f = node(
        &mut g,
        Op::Cast { to: DType::F32 },
        vec![c],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![f]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[
            ("a", &[1.0f32, 5.0, 3.0, 2.0]),
            ("b", &[2.0f32, 4.0, 3.0, 1.0]),
        ])
        .unwrap()
        .remove(0);
    approx(&out, &[0.0, 1.0, 0.0, 1.0], 1e-6);
}

#[test]
fn cast_f64_demotes_to_fp32() {
    // CoreML has no f64 storage. A `Cast { to: F64 }` must demote to fp32 and
    // still compile + run end-to-end. Route through F16 so the casts are real
    // type changes (fp32→fp16→fp32) rather than identities; the test values are
    // all exactly representable in f16, so the round-trip is lossless.
    let mut g = Graph::new("cast_f64");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let h = node(
        &mut g,
        Op::Cast { to: DType::F16 },
        vec![x],
        Shape::new(&[4], DType::F16),
    );
    let y = node(
        &mut g,
        Op::Cast { to: DType::F64 },
        vec![h],
        Shape::new(&[4], DType::F64),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[("x", &[1.5f32, -2.0, 3.25, 4.0])])
        .unwrap()
        .remove(0);
    approx(&out, &[1.5, -2.0, 3.25, 4.0], 1e-6);
}

#[test]
fn where_select() {
    let mut g = Graph::new("where");
    let c = g.input("c", Shape::new(&[4], DType::F32));
    let a = g.input("a", Shape::new(&[4], DType::F32));
    let b = g.input("b", Shape::new(&[4], DType::F32));
    let y = node(
        &mut g,
        Op::Where,
        vec![c, a, b],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[
            ("c", &[1.0f32, 0.0, 1.0, 0.0]),
            ("a", &[10.0f32, 20.0, 30.0, 40.0]),
            ("b", &[-1.0f32, -2.0, -3.0, -4.0]),
        ])
        .unwrap()
        .remove(0);
    approx(&out, &[10.0, -2.0, 30.0, -4.0], 1e-6);
}

#[test]
fn expand_broadcast() {
    let mut g = Graph::new("expand");
    let x = g.input("x", Shape::new(&[1, 3], DType::F32));
    let y = node(
        &mut g,
        Op::Expand {
            target_shape: vec![2, 3],
        },
        vec![x],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e.run(&[("x", &[1.0f32, 2.0, 3.0])]).unwrap().remove(0);
    approx(&out, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn cumsum_inclusive() {
    let mut g = Graph::new("cumsum");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = node(
        &mut g,
        Op::Cumsum {
            axis: -1,
            exclusive: false,
        },
        vec![x],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[("x", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])])
        .unwrap()
        .remove(0);
    approx(&out, &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0], 1e-5);
}

#[test]
fn scatter_add() {
    // 3 updates of width 2 into rows {0,2,0} of a [3,2] output.
    let mut g = Graph::new("scatter");
    let upd = g.input("upd", Shape::new(&[3, 2], DType::F32));
    let idx = g.input("idx", Shape::new(&[3], DType::F32));
    let y = node(
        &mut g,
        Op::ScatterAdd,
        vec![upd, idx],
        Shape::new(&[3, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[
            ("upd", &[1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0]),
            ("idx", &[0.0f32, 2.0, 0.0]),
        ])
        .unwrap()
        .remove(0);
    // row0 = upd0+upd2 = [4,4], row1 = [0,0], row2 = upd1 = [2,2]
    approx(&out, &[4.0, 4.0, 0.0, 0.0, 2.0, 2.0], 1e-5);
}

#[test]
fn scatter_nd_update() {
    // data = [[1,1,1],[2,2,2]], replace row0 with [9,9,9] via indices [[0]].
    let mut g = Graph::new("scatter_nd");
    let data = g.input("data", Shape::new(&[2, 3], DType::F32));
    let indices = g.input("indices", Shape::new(&[1, 1], DType::F32));
    let updates = g.input("updates", Shape::new(&[1, 3], DType::F32));
    let y = node(
        &mut g,
        Op::ScatterNd {
            reduction: rlx_ir::ScatterNdReduction::None,
        },
        vec![data, indices, updates],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[
            ("data", &[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]),
            ("indices", &[0.0f32]),
            ("updates", &[9.0f32, 9.0, 9.0]),
        ])
        .unwrap()
        .remove(0);
    approx(&out, &[9.0, 9.0, 9.0, 2.0, 2.0, 2.0], 1e-5);
}

#[test]
fn batch_norm_inference() {
    let mut g = Graph::new("bn");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let gamma = g.param("g", Shape::new(&[3], DType::F32));
    let beta = g.param("b", Shape::new(&[3], DType::F32));
    let mean = g.param("m", Shape::new(&[3], DType::F32));
    let var = g.param("v", Shape::new(&[3], DType::F32));
    let y = node(
        &mut g,
        Op::BatchNormInference { eps: 1e-5 },
        vec![x, gamma, beta, mean, var],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (gv, bv, mv, vv) = (
        [2.0f32, 1.0, 0.5],
        [0.1f32, 0.2, 0.3],
        [1.0f32, 2.0, 3.0],
        [1.0f32, 4.0, 9.0],
    );
    let mut e = CoremlExecutable::compile(g);
    e.set_param("g", &gv);
    e.set_param("b", &bv);
    e.set_param("m", &mv);
    e.set_param("v", &vv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    let mut want = vec![0.0f32; 6];
    for i in 0..2 {
        for c in 0..3 {
            let xh = (xs[i * 3 + c] - mv[c]) / (vv[c] + 1e-5).sqrt();
            want[i * 3 + c] = gv[c] * xh + bv[c];
        }
    }
    approx(&out, &want, 1e-4);
}

#[test]
fn group_norm_nchw() {
    // N=1, C=4, H=2, W=2, groups=2.
    let (c, hw) = (4usize, 4usize);
    let mut g = Graph::new("gn");
    let x = g.input("x", Shape::new(&[1, 4, 2, 2], DType::F32));
    let gamma = g.param("g", Shape::new(&[4], DType::F32));
    let beta = g.param("b", Shape::new(&[4], DType::F32));
    let y = node(
        &mut g,
        Op::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
        },
        vec![x, gamma, beta],
        Shape::new(&[1, 4, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let gv = [1.0f32, 0.5, 2.0, 1.5];
    let bv = [0.0f32, 0.1, -0.1, 0.2];
    let mut e = CoremlExecutable::compile(g);
    e.set_param("g", &gv);
    e.set_param("b", &bv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    // reference: group of 2 channels each, over 2*hw=8 elems.
    let mut want = vec![0.0f32; c * hw];
    for grp in 0..2 {
        let chans = [grp * 2, grp * 2 + 1];
        let mut vals: Vec<f32> = Vec::new();
        for &ch in &chans {
            for p in 0..hw {
                vals.push(xs[ch * hw + p]);
            }
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for &ch in &chans {
            for p in 0..hw {
                want[ch * hw + p] = (xs[ch * hw + p] - mean) * inv * gv[ch] + bv[ch];
            }
        }
    }
    approx(&out, &want, 2e-3);
}

#[test]
fn layer_norm_2d_nchw() {
    let (c, hw) = (4usize, 4usize);
    let mut g = Graph::new("ln2d");
    let x = g.input("x", Shape::new(&[1, 4, 2, 2], DType::F32));
    let gamma = g.param("g", Shape::new(&[4], DType::F32));
    let beta = g.param("b", Shape::new(&[4], DType::F32));
    let y = node(
        &mut g,
        Op::LayerNorm2d { eps: 1e-5 },
        vec![x, gamma, beta],
        Shape::new(&[1, 4, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs: Vec<f32> = (0..16).map(|i| (i as f32 - 7.0) * 0.3).collect();
    let gv = [1.0f32, 0.5, 2.0, 1.5];
    let bv = [0.0f32, 0.1, -0.1, 0.2];
    let mut e = CoremlExecutable::compile(g);
    e.set_param("g", &gv);
    e.set_param("b", &bv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    // reference: per spatial position, normalize over channels.
    let mut want = vec![0.0f32; c * hw];
    for p in 0..hw {
        let vals: Vec<f32> = (0..c).map(|ch| xs[ch * hw + p]).collect();
        let mean = vals.iter().sum::<f32>() / c as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ch in 0..c {
            want[ch * hw + p] = (xs[ch * hw + p] - mean) * inv * gv[ch] + bv[ch];
        }
    }
    approx(&out, &want, 2e-3);
}

#[test]
fn lora_matmul() {
    // x[2,3] @ W[3,4] + scale * (x @ A[3,2]) @ B[2,4]
    let mut g = Graph::new("lora");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("W", Shape::new(&[3, 4], DType::F32));
    let a = g.param("A", Shape::new(&[3, 2], DType::F32));
    let b = g.param("B", Shape::new(&[2, 4], DType::F32));
    let y = node(
        &mut g,
        Op::LoraMatMul { scale: 0.5 },
        vec![x, w, a, b],
        Shape::new(&[2, 4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wv: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let av: Vec<f32> = (0..6).map(|i| i as f32 * 0.2 - 0.5).collect();
    let bv: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 - 0.3).collect();
    let mut e = CoremlExecutable::compile(g);
    e.set_param("W", &wv);
    e.set_param("A", &av);
    e.set_param("B", &bv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    let mm = |lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize| {
        let mut o = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    o[i * n + j] += lhs[i * k + kk] * rhs[kk * n + j];
                }
            }
        }
        o
    };
    let xw = mm(&xs, &wv, 2, 3, 4);
    let xa = mm(&xs, &av, 2, 3, 2);
    let xab = mm(&xa, &bv, 2, 2, 4);
    let want: Vec<f32> = xw.iter().zip(&xab).map(|(p, q)| p + 0.5 * q).collect();
    approx(&out, &want, 1e-4);
}

#[test]
fn conv2d_basic() {
    // x[1,1,4,4] * w[1,1,3,3], stride 1, no pad -> [1,1,2,2]
    let mut g = Graph::new("conv");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
    let w = g.param("W", Shape::new(&[1, 1, 3, 3], DType::F32));
    let y = node(
        &mut g,
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let wv: Vec<f32> = (0..9).map(|i| (i % 3) as f32 * 0.5).collect();
    let mut e = CoremlExecutable::compile(g);
    e.set_param("W", &wv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    let mut want = vec![0.0f32; 4];
    for oh in 0..2 {
        for ow in 0..2 {
            let mut acc = 0.0;
            for kh in 0..3 {
                for kw in 0..3 {
                    acc += xs[(oh + kh) * 4 + (ow + kw)] * wv[kh * 3 + kw];
                }
            }
            want[oh * 2 + ow] = acc;
        }
    }
    approx(&out, &want, 1e-3);
}

#[test]
fn max_pool_2x2() {
    let mut g = Graph::new("pool");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
    let y = node(
        &mut g,
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let xs: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut e = CoremlExecutable::compile(g);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);
    // max of each 2x2 block: [5,7,13,15]
    approx(&out, &[5.0, 7.0, 13.0, 15.0], 1e-5);
}

#[test]
fn axial_rope2d_vs_cpu() {
    use rlx_runtime::{Device, Session};
    // [B=1, seq=end_x*end_y*repeat, H*D], H=2, D=8, 2x2 grid, repeat=1.
    let (ex, ey, hd_, nh, rep) = (2usize, 2usize, 8usize, 2usize, 1usize);
    let seq = ex * ey * rep;
    let hid = nh * hd_;
    let build = || {
        let mut g = Graph::new("axrope");
        let x = g.input("x", Shape::new(&[1, seq, hid], DType::F32));
        let y = g.axial_rope2d(x, ex, ey, hd_, nh, 10000.0, rep);
        g.set_outputs(vec![y]);
        g
    };
    let xs: Vec<f32> = (0..seq * hid).map(|i| ((i as f32) * 0.1).sin()).collect();

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let cpu_out = cpu.run(&[("x", &xs)]).remove(0);
    let mut ane = Session::new(Device::Ane).compile(build());
    let ane_out = ane.run(&[("x", &xs)]).remove(0);
    approx(&ane_out, &cpu_out, 1e-3);
}

#[test]
fn resize_nearest_2x() {
    // NCHW [1,1,2,2] -> [1,1,4,4], each pixel duplicated 2x2.
    let mut g = Graph::new("resize");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let y = node(
        &mut g,
        Op::ResizeNearest2x,
        vec![x],
        Shape::new(&[1, 1, 4, 4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]).unwrap().remove(0);
    approx(
        &out,
        &[
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ],
        1e-5,
    );
}

#[test]
fn stop_gradient_identity() {
    let mut g = Graph::new("sg");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let r = g.activation(
        rlx_ir::op::Activation::Relu,
        x,
        Shape::new(&[4], DType::F32),
    );
    let y = node(
        &mut g,
        Op::StopGradient,
        vec![r],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[("x", &[-1.0f32, 2.0, -3.0, 4.0])])
        .unwrap()
        .remove(0);
    approx(&out, &[0.0, 2.0, 0.0, 4.0], 1e-6);
}

#[test]
fn grouped_matmul_moe() {
    // M=3 tokens, K=2, E=2 experts, N=2. expert_idx picks per token.
    let mut g = Graph::new("moe");
    let x = g.input("x", Shape::new(&[3, 2], DType::F32));
    let w = g.param("W", Shape::new(&[2, 2, 2], DType::F32)); // [E,K,N]
    let eidx = g.input("e", Shape::new(&[3], DType::F32));
    let y = node(
        &mut g,
        Op::GroupedMatMul,
        vec![x, w, eidx],
        Shape::new(&[3, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // expert0 = identity-ish, expert1 = [[0,1],[1,0]] swap
    let wv = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let experts = [0.0f32, 1.0, 0.0];
    let mut e = CoremlExecutable::compile(g);
    e.set_param("W", &wv);
    let out = e.run(&[("x", &xs), ("e", &experts)]).unwrap().remove(0);

    let mut want = vec![0.0f32; 6];
    for t in 0..3 {
        let ex = experts[t] as usize;
        for nn in 0..2 {
            let mut acc = 0.0;
            for kk in 0..2 {
                acc += xs[t * 2 + kk] * wv[ex * 4 + kk * 2 + nn];
            }
            want[t * 2 + nn] = acc;
        }
    }
    approx(&out, &want, 1e-4);
}

#[test]
fn topk_indices() {
    // top-2 indices of each row, descending by value.
    let mut g = Graph::new("topk");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = node(
        &mut g,
        Op::TopK { k: 2 },
        vec![x],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut e = CoremlExecutable::compile(g);
    let out = e
        .run(&[("x", &[1.0f32, 9.0, 3.0, 7.0, 5.0, 2.0, 8.0, 4.0])])
        .unwrap()
        .remove(0);
    // row0 [1,9,3,7] -> idx [1,3]; row1 [5,2,8,4] -> idx [2,0]
    approx(&out, &[1.0, 3.0, 2.0, 0.0], 1e-5);
}

#[test]
fn conv_transpose_2x2() {
    // x[1,1,2,2], w[1,1,2,2], stride 2 -> [1,1,4,4] (non-overlapping)
    let mut g = Graph::new("convt");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let w = g.param("W", Shape::new(&[1, 1, 2, 2], DType::F32));
    let y = node(
        &mut g,
        Op::ConvTranspose2d {
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            output_padding: vec![0, 0],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 1, 4, 4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0];
    let wv = [1.0f32, 0.5, 0.25, 2.0];
    let mut e = CoremlExecutable::compile(g);
    e.set_param("W", &wv);
    let out = e.run(&[("x", &xs)]).unwrap().remove(0);

    // each input pixel writes a 2x2 block scaled by the kernel.
    let mut want = vec![0.0f32; 16];
    for ih in 0..2 {
        for iw in 0..2 {
            for kh in 0..2 {
                for kw in 0..2 {
                    let oh = ih * 2 + kh;
                    let ow = iw * 2 + kw;
                    want[oh * 4 + ow] += xs[ih * 2 + iw] * wv[kh * 2 + kw];
                }
            }
        }
    }
    approx(&out, &want, 1e-3);
}
