// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native Vulkan backend ↔ CPU reference parity. Runs the same graph on
//! `Device::Cpu` (ground truth) and `Device::Vulkan` and asserts the outputs
//! match. Skips cleanly when no Vulkan device is reachable (CI without a
//! driver), so the suite is green everywhere; it does real cross-backend
//! validation when a device (MoltenVK on macOS, lavapipe / a real GPU on
//! Linux) is present.

#![cfg(all(feature = "vulkan", feature = "cpu"))]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp, RopeStyle};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

fn f(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// Deterministic pseudo-random data in [-1, 1) (no rng dep, reproducible).
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn run_on(device: Device, graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
    Session::new(device).compile(graph).run(inputs)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Compile + run `build()` on CPU and Vulkan; assert outputs agree within `tol`.
fn parity(name: &str, build: impl Fn() -> Graph, inputs: &[(&str, &[f32])], tol: f32) {
    if !is_available(Device::Vulkan) {
        eprintln!("[parity] {name}: SKIP (no Vulkan device)");
        return;
    }
    let cpu = run_on(Device::Cpu, build(), inputs);
    let vk = run_on(Device::Vulkan, build(), inputs);
    assert_eq!(cpu.len(), vk.len(), "{name}: output count");
    for (i, (c, v)) in cpu.iter().zip(&vk).enumerate() {
        assert_eq!(c.len(), v.len(), "{name} out{i}: length");
        let m = max_abs(c, v);
        assert!(
            m <= tol,
            "{name} out{i}: max|Δ| = {m} > {tol}\n cpu[..8]={:?}\n  vk[..8]={:?}",
            &c[..c.len().min(8)],
            &v[..v.len().min(8)]
        );
    }
    eprintln!("[parity] {name}: OK (max tol {tol})");
}

#[test]
fn elementwise_chain() {
    // (a + b) then relu, then * a  — binary + activation + binary.
    let build = || {
        let mut g = Graph::new("ew");
        let a = g.input("a", f(&[6]));
        let b = g.input("b", f(&[6]));
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f(&[6]));
        let r = g.add_node(Op::Activation(Activation::Relu), vec![s], f(&[6]));
        let o = g.add_node(Op::Binary(BinaryOp::Mul), vec![r, a], f(&[6]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "elementwise_chain",
        build,
        &[("a", &data(6, 1)), ("b", &data(6, 2))],
        1e-4,
    );
}

#[test]
fn matmul_batched() {
    let (bsz, m, k, n) = (2usize, 3, 4, 5);
    let build = || {
        let mut g = Graph::new("mm");
        let a = g.input("a", f(&[bsz, m, k]));
        let b = g.input("b", f(&[bsz, k, n]));
        let o = g.add_node(Op::MatMul, vec![a, b], f(&[bsz, m, n]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "matmul_batched",
        build,
        &[("a", &data(bsz * m * k, 3)), ("b", &data(bsz * k * n, 4))],
        1e-3,
    );
}

#[test]
fn softmax_last_axis() {
    let build = || {
        let mut g = Graph::new("sm");
        let x = g.input("x", f(&[4, 8]));
        let o = g.add_node(Op::Softmax { axis: -1 }, vec![x], f(&[4, 8]));
        g.set_outputs(vec![o]);
        g
    };
    parity("softmax_last_axis", build, &[("x", &data(32, 5))], 1e-4);
}

#[test]
fn rms_norm() {
    // Op::RmsNorm carries (x, gamma, beta).
    let build = || {
        let mut g = Graph::new("rms");
        let x = g.input("x", f(&[4, 8]));
        let w = g.input("w", f(&[8]));
        let b = g.input("b", f(&[8]));
        let o = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, w, b],
            f(&[4, 8]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "rms_norm",
        build,
        &[("x", &data(32, 6)), ("w", &data(8, 7)), ("b", &data(8, 15))],
        1e-3,
    );
}

#[test]
fn reduce_sum_last_axis() {
    let build = || {
        let mut g = Graph::new("red");
        let x = g.input("x", f(&[4, 8]));
        let o = g.add_node(
            Op::Reduce {
                op: rlx_ir::op::ReduceOp::Sum,
                axes: vec![1],
                keep_dim: false,
            },
            vec![x],
            f(&[4]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity("reduce_sum_last_axis", build, &[("x", &data(32, 8))], 1e-3);
}

#[test]
fn transpose_2d() {
    let build = || {
        let mut g = Graph::new("tr");
        let x = g.input("x", f(&[3, 5]));
        let o = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![x], f(&[5, 3]));
        g.set_outputs(vec![o]);
        g
    };
    parity("transpose_2d", build, &[("x", &data(15, 9))], 1e-6);
}

#[test]
fn gather_embedding() {
    // data [vocab=5, dim=4], idx [3] → [3, 4]
    let build = || {
        let mut g = Graph::new("gat");
        let table = g.input("t", f(&[5, 4]));
        let idx = g.input("i", f(&[3]));
        let o = g.add_node(Op::Gather { axis: 0 }, vec![table, idx], f(&[3, 4]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "gather_embedding",
        build,
        &[("t", &data(20, 10)), ("i", &[3.0, 0.0, 4.0])],
        1e-6,
    );
}

#[test]
fn rope_default_style() {
    // x [b=1, s=2, hidden=8], head_dim=4, nh=2; cos/sin [s, head_dim/2].
    let build = || {
        let mut g = Graph::new("rope");
        let x = g.input("x", f(&[1, 2, 8]));
        let cos = g.input("cos", f(&[2, 2]));
        let sin = g.input("sin", f(&[2, 2]));
        let o = g.rope_n(x, cos, sin, 4, 4);
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "rope_default_style",
        build,
        &[
            ("x", &data(16, 11)),
            ("cos", &[1.0, 0.95, 0.8, 0.6]),
            ("sin", &[0.0, 0.31, 0.6, 0.8]),
        ],
        1e-3,
    );
}

#[test]
fn layer_norm() {
    // Op::LayerNorm carries (x, gamma, beta).
    let build = || {
        let mut g = Graph::new("ln");
        let x = g.input("x", f(&[4, 8]));
        let w = g.input("w", f(&[8]));
        let b = g.input("b", f(&[8]));
        let o = g.add_node(
            Op::LayerNorm {
                axis: -1,
                eps: 1e-5,
            },
            vec![x, w, b],
            f(&[4, 8]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "layer_norm",
        build,
        &[
            ("x", &data(32, 16)),
            ("w", &data(8, 17)),
            ("b", &data(8, 18)),
        ],
        1e-3,
    );
}

#[test]
fn dit_packed_ada_layer_norm_backward() {
    use rlx_ir::op::AdaNormKind;
    let (b, s, d) = (2usize, 4usize, 8usize);
    let build = || {
        let mut g = Graph::new("ada_bwd");
        let x = g.input("x", f(&[b, s, d]));
        let scale = g.input("scale", f(&[b, 1, d]));
        let shift = g.input("shift", f(&[b, 1, d]));
        let dy = g.input("dy", f(&[b, s, d]));
        let o = g.ada_layer_norm_backward(x, scale, shift, dy, AdaNormKind::LayerNorm, 1e-5);
        g.set_outputs(vec![o]);
        g
    };
    let nx = b * s * d;
    let ns = b * d;
    parity(
        "dit_ada_layer_norm_backward",
        build,
        &[
            ("x", &data(nx, 21)),
            ("scale", &data(ns, 22)),
            ("shift", &data(ns, 23)),
            ("dy", &data(nx, 24)),
        ],
        2e-3,
    );
}

#[test]
fn dit_packed_gated_residual_backward() {
    let (b, s, d) = (2usize, 4usize, 8usize);
    let build = || {
        let mut g = Graph::new("gate_bwd");
        let x = g.input("x", f(&[b, s, d]));
        let y = g.input("y", f(&[b, s, d]));
        let gate = g.input("gate", f(&[b, 1, d]));
        let dy = g.input("dy", f(&[b, s, d]));
        let o = g.gated_residual_backward(x, y, gate, dy);
        g.set_outputs(vec![o]);
        g
    };
    let nx = b * s * d;
    let ng = b * d;
    parity(
        "dit_gated_residual_backward",
        build,
        &[
            ("x", &data(nx, 31)),
            ("y", &data(nx, 32)),
            ("gate", &data(ng, 33)),
            ("dy", &data(nx, 34)),
        ],
        1e-4,
    );
}

#[test]
fn narrow_axis1() {
    let build = || {
        let mut g = Graph::new("nar");
        let x = g.input("x", f(&[4, 6]));
        let o = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 1,
                len: 3,
            },
            vec![x],
            f(&[4, 3]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity("narrow_axis1", build, &[("x", &data(24, 19))], 1e-6);
}

#[test]
fn concat_axis1() {
    let build = || {
        let mut g = Graph::new("cat");
        let a = g.input("a", f(&[2, 3]));
        let b = g.input("b", f(&[2, 2]));
        let o = g.add_node(Op::Concat { axis: 1 }, vec![a, b], f(&[2, 5]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "concat_axis1",
        build,
        &[("a", &data(6, 20)), ("b", &data(4, 21))],
        1e-6,
    );
}

#[test]
fn expand_broadcast() {
    let build = || {
        let mut g = Graph::new("exp");
        let x = g.input("x", f(&[1, 4]));
        let o = g.add_node(
            Op::Expand {
                target_shape: vec![3, 4],
            },
            vec![x],
            f(&[3, 4]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity("expand_broadcast", build, &[("x", &data(4, 22))], 1e-6);
}

#[test]
fn where_compare() {
    let build = || {
        let mut g = Graph::new("wc");
        let a = g.input("a", f(&[5]));
        let b = g.input("b", f(&[5]));
        let t = g.input("t", f(&[5]));
        let cond = g.add_node(
            Op::Compare(rlx_ir::op::CmpOp::Lt),
            vec![a, b],
            Shape::new(&[5], DType::Bool),
        );
        let o = g.add_node(Op::Where, vec![cond, a, t], f(&[5]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "where_compare",
        build,
        &[
            ("a", &data(5, 23)),
            ("b", &data(5, 24)),
            ("t", &data(5, 25)),
        ],
        1e-6,
    );
}

#[test]
fn cumsum_last_axis() {
    let build = || {
        let mut g = Graph::new("cs");
        let x = g.input("x", f(&[2, 5]));
        let o = g.add_node(
            Op::Cumsum {
                axis: -1,
                exclusive: false,
            },
            vec![x],
            f(&[2, 5]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity("cumsum_last_axis", build, &[("x", &data(10, 26))], 1e-4);
}

#[test]
fn attention_causal_bshd() {
    let (b, s, nh, dh) = (1usize, 3, 2, 4);
    let build = || {
        let mut g = Graph::new("attn");
        let q = g.input("q", f(&[b, s, nh, dh]));
        let k = g.input("k", f(&[b, s, nh, dh]));
        let v = g.input("v", f(&[b, s, nh, dh]));
        let o = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            f(&[b, s, nh, dh]),
        );
        g.set_outputs(vec![o]);
        g
    };
    let n = b * s * nh * dh;
    parity(
        "attention_causal_bshd",
        build,
        &[
            ("q", &data(n, 12)),
            ("k", &data(n, 13)),
            ("v", &data(n, 14)),
        ],
        2e-3,
    );
}

// ── Exhaustive variant coverage ────────────────────────────────────────────

/// Strictly-positive data (for div / pow / log / sqrt / rsqrt).
fn pos(n: usize, seed: u64) -> Vec<f32> {
    data(n, seed).iter().map(|x| x.abs() + 0.5).collect()
}

#[test]
fn all_binary_ops() {
    let a = pos(8, 30);
    let b = pos(8, 31);
    for (op, tag, tol) in [
        (BinaryOp::Add, "add", 1e-4),
        (BinaryOp::Sub, "sub", 1e-4),
        (BinaryOp::Mul, "mul", 1e-4),
        (BinaryOp::Div, "div", 1e-3),
        (BinaryOp::Max, "max", 1e-6),
        (BinaryOp::Min, "min", 1e-6),
        (BinaryOp::Pow, "pow", 2e-3),
    ] {
        let build = move || {
            let mut g = Graph::new("bin");
            let x = g.input("a", f(&[8]));
            let y = g.input("b", f(&[8]));
            let o = g.add_node(Op::Binary(op), vec![x, y], f(&[8]));
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("binary_{tag}"),
            build,
            &[("a", &a), ("b", &b)],
            tol,
        );
    }
}

#[test]
fn all_activations() {
    // (activation, tag, needs_positive_input, tol)
    let cases = [
        (Activation::Gelu, "gelu", false, 5e-3),
        (Activation::GeluApprox, "gelu_approx", false, 5e-3),
        (Activation::Silu, "silu", false, 1e-4),
        (Activation::Sigmoid, "sigmoid", false, 1e-5),
        (Activation::Tanh, "tanh", false, 1e-5),
        (Activation::Exp, "exp", false, 1e-4),
        (Activation::Log, "log", true, 1e-4),
        (Activation::Sqrt, "sqrt", true, 1e-5),
        (Activation::Rsqrt, "rsqrt", true, 1e-4),
        (Activation::Neg, "neg", false, 1e-6),
        (Activation::Abs, "abs", false, 1e-6),
        (Activation::Sin, "sin", false, 1e-5),
        (Activation::Cos, "cos", false, 1e-5),
        (Activation::Tan, "tan", false, 1e-3),
        (Activation::Atan, "atan", false, 1e-5),
        (Activation::Recip, "recip", false, 1e-6),
        (Activation::Round, "round", false, 1e-6),
    ];
    for (act, tag, need_pos, tol) in cases {
        let x = if need_pos { pos(8, 40) } else { data(8, 40) };
        let build = move || {
            let mut g = Graph::new("act");
            let i = g.input("x", f(&[8]));
            let o = g.add_node(Op::Activation(act), vec![i], f(&[8]));
            g.set_outputs(vec![o]);
            g
        };
        parity(&format!("act_{tag}"), build, &[("x", &x)], tol);
    }
}

#[test]
fn all_compare_ops() {
    // Inputs share some equal elements so every comparison exercises both arms.
    let a = vec![1.0, 2.0, 3.0, 4.0, 2.0];
    let b = vec![1.0, 5.0, 2.0, 4.0, 3.0];
    for (op, tag) in [
        (CmpOp::Eq, "eq"),
        (CmpOp::Ne, "ne"),
        (CmpOp::Lt, "lt"),
        (CmpOp::Le, "le"),
        (CmpOp::Gt, "gt"),
        (CmpOp::Ge, "ge"),
    ] {
        let build = move || {
            let mut g = Graph::new("cmp");
            let x = g.input("a", f(&[5]));
            let y = g.input("b", f(&[5]));
            // Compare yields Bool; cast to F32 so the graph output is a clean
            // 1.0/0.0 stream (a Bool output read back through the f32 `run()`
            // path is byte-reinterpreted on the CPU side, not 0/1).
            let cmp = g.add_node(Op::Compare(op), vec![x, y], Shape::new(&[5], DType::Bool));
            let o = g.add_node(Op::Cast { to: DType::F32 }, vec![cmp], f(&[5]));
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("compare_{tag}"),
            build,
            &[("a", &a), ("b", &b)],
            1e-6,
        );
    }
}

#[test]
fn all_reduce_ops() {
    for (op, tag, tol) in [
        (ReduceOp::Sum, "sum", 1e-3),
        (ReduceOp::Mean, "mean", 1e-4),
        (ReduceOp::Max, "max", 1e-6),
        (ReduceOp::Min, "min", 1e-6),
        (ReduceOp::Prod, "prod", 1e-3),
    ] {
        let build = move || {
            let mut g = Graph::new("red");
            let x = g.input("x", f(&[4, 6]));
            let o = g.add_node(
                Op::Reduce {
                    op,
                    axes: vec![1],
                    keep_dim: false,
                },
                vec![x],
                f(&[4]),
            );
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("reduce_{tag}"),
            build,
            &[("x", &data(24, 50))],
            tol,
        );
    }
}

#[test]
fn attention_masks_and_layout() {
    let (b, s, nh, dh) = (1usize, 4, 2, 4);
    let n = b * s * nh * dh;
    let (q, k, v) = (data(n, 60), data(n, 61), data(n, 62));

    // [B, S, H, D] across mask kinds.
    for (mask, tag) in [
        (MaskKind::None, "none"),
        (MaskKind::Causal, "causal"),
        (MaskKind::SlidingWindow(2), "sliding"),
    ] {
        let build = move || {
            let mut g = Graph::new("attn");
            let q = g.input("q", f(&[b, s, nh, dh]));
            let k = g.input("k", f(&[b, s, nh, dh]));
            let v = g.input("v", f(&[b, s, nh, dh]));
            let o = g.add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: dh,
                    v_head_dim: None,
                    mask_kind: mask,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q, k, v],
                f(&[b, s, nh, dh]),
            );
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("attn_bshd_{tag}"),
            build,
            &[("q", &q), ("k", &k), ("v", &v)],
            2e-3,
        );
    }

    // [B, H, S, D] layout (heads at axis 1), causal.
    let build = move || {
        let mut g = Graph::new("attn_bhsd");
        let q = g.input("q", f(&[b, nh, s, dh]));
        let k = g.input("k", f(&[b, nh, s, dh]));
        let v = g.input("v", f(&[b, nh, s, dh]));
        let o = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: Some(0.5),
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            f(&[b, nh, s, dh]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "attn_bhsd_causal_scaled",
        build,
        &[("q", &q), ("k", &k), ("v", &v)],
        2e-3,
    );
}

#[test]
fn rope_styles_and_partial() {
    let x = data(16, 70);
    let cos = vec![1.0, 0.95, 0.8, 0.6];
    let sin = vec![0.0, 0.31, 0.6, 0.8];
    for (style, n_rot, tag) in [
        (RopeStyle::NeoX, 4usize, "neox_full"),
        (RopeStyle::GptJ, 4, "gptj_full"),
        (RopeStyle::NeoX, 2, "neox_partial"),
        (RopeStyle::GptJ, 2, "gptj_partial"),
    ] {
        let build = move || {
            let mut g = Graph::new("rope");
            let xi = g.input("x", f(&[1, 2, 8]));
            let c = g.input("cos", f(&[2, 2]));
            let sn = g.input("sin", f(&[2, 2]));
            let o = g.add_node(
                Op::Rope {
                    head_dim: 4,
                    n_rot,
                    style,
                },
                vec![xi, c, sn],
                f(&[1, 2, 8]),
            );
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("rope_{tag}"),
            build,
            &[("x", &x), ("cos", &cos), ("sin", &sin)],
            1e-3,
        );
    }
}

#[test]
fn softmax_ranks() {
    // Last-axis softmax across ranks. (Non-last-axis softmax is intentionally
    // not parity-checked: the CPU kernel is last-axis-only — `rows×cols` with
    // `cols = dim(axis)` — so it's not a valid oracle for an arbitrary axis,
    // whereas the Vulkan kernel handles any axis via strides.)
    for (shape, tag) in [
        (vec![7usize], "1d"),
        (vec![4, 8], "2d"),
        (vec![2, 3, 5], "3d"),
    ] {
        let n: usize = shape.iter().product();
        let sh = shape.clone();
        let build = move || {
            let mut g = Graph::new("sm");
            let x = g.input("x", f(&sh));
            let o = g.add_node(Op::Softmax { axis: -1 }, vec![x], f(&sh));
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("softmax_{tag}"),
            build,
            &[("x", &data(n, 80))],
            1e-4,
        );
    }
}

#[test]
fn binary_broadcasting() {
    // Trailing broadcast [2,3] + [3].
    let build1 = || {
        let mut g = Graph::new("bc1");
        let a = g.input("a", f(&[2, 3]));
        let b = g.input("b", f(&[3]));
        let o = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f(&[2, 3]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "binary_bcast_trailing",
        build1,
        &[("a", &data(6, 90)), ("b", &data(3, 91))],
        1e-4,
    );

    // Scalar broadcast [4] * [1].
    let build2 = || {
        let mut g = Graph::new("bc2");
        let a = g.input("a", f(&[4]));
        let b = g.input("b", f(&[1]));
        let o = g.add_node(Op::Binary(BinaryOp::Mul), vec![a, b], f(&[4]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "binary_bcast_scalar",
        build2,
        &[("a", &data(4, 92)), ("b", &[2.5f32])],
        1e-4,
    );
}

#[test]
fn reshape_passthrough() {
    // Reshape aliases the slot; a following op must read the right data.
    let build = || {
        let mut g = Graph::new("rc");
        let x = g.input("x", f(&[2, 6]));
        let r = g.add_node(
            Op::Reshape {
                new_shape: vec![3, 4],
            },
            vec![x],
            f(&[3, 4]),
        );
        let o = g.add_node(Op::Activation(Activation::Abs), vec![r], f(&[3, 4]));
        g.set_outputs(vec![o]);
        g
    };
    parity("reshape_passthrough", build, &[("x", &data(12, 95))], 1e-6);
}

#[test]
fn fma_decomposed() {
    // Op::Fma is not in the native set → LowerFma rewrites it to mul+add.
    let build = || {
        let mut g = Graph::new("fma");
        let a = g.input("a", f(&[5]));
        let b = g.input("b", f(&[5]));
        let c = g.input("c", f(&[5]));
        let o = g.add_node(Op::Fma, vec![a, b, c], f(&[5]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "fma_decomposed",
        build,
        &[
            ("a", &data(5, 96)),
            ("b", &data(5, 97)),
            ("c", &data(5, 98)),
        ],
        1e-4,
    );
}

// Note: a hand-built `Op::DotGeneral` is intentionally not covered — its IR
// shape inference is a stub (returns input 0's shape), so the verifier rejects
// a correctly-declared matmul output shape. The DotGeneral→MatMul rewrite is
// exercised indirectly by the (heavily covered) MatMul path; the decomposition
// machinery itself is covered by `fma_decomposed`.

// ── Fused-op decomposition parity ──────────────────────────────────────────
//
// The fusion pipeline emits these macro-ops for real models. The CPU backend
// runs them with native fused kernels; the Vulkan backend has no native fused
// kernel, so `legalize_or_rewrite_for_backend` must unfuse them into the
// primitive set. These tests prove unfuse(Vulkan) == native(CPU).

#[test]
fn fused_matmul_bias_act_parity() {
    let (m, k, n) = (3usize, 4, 5);
    let build = || {
        let mut g = Graph::new("fmba");
        let x = g.input("x", f(&[m, k]));
        let w = g.input("w", f(&[k, n]));
        let b = g.input("b", f(&[n]));
        let o = g.fused_matmul_bias_act(x, w, b, Some(Activation::Silu), f(&[m, n]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "fused_matmul_bias_act",
        build,
        &[
            ("x", &data(m * k, 200)),
            ("w", &data(k * n, 201)),
            ("b", &data(n, 202)),
        ],
        1e-3,
    );
}

#[test]
fn fused_residual_rms_norm_parity() {
    let (rows, h) = (4usize, 8);
    let build = || {
        let mut g = Graph::new("frms");
        let x = g.input("x", f(&[rows, h]));
        let res = g.input("res", f(&[rows, h]));
        let gamma = g.input("g", f(&[h]));
        let beta = g.input("b", f(&[h]));
        let o = g.fused_residual_rms_norm(x, res, None, gamma, beta, 1e-6, f(&[rows, h]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "fused_residual_rms_norm",
        build,
        &[
            ("x", &data(rows * h, 210)),
            ("res", &data(rows * h, 211)),
            ("g", &data(h, 212)),
            ("b", &data(h, 213)),
        ],
        1e-3,
    );
}

#[test]
fn fused_swiglu_ffn_parity() {
    let (m, k, ff) = (3usize, 4, 6);
    let build = || {
        let mut g = Graph::new("swiglu");
        let x = g.input("x", f(&[m, k]));
        let up = g.input("up", f(&[k, ff]));
        let gate = g.input("gate", f(&[k, ff]));
        let down = g.input("down", f(&[ff, k]));
        let o = g.fused_swiglu_ffn(x, up, gate, down, f(&[m, k]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "fused_swiglu_ffn",
        build,
        &[
            ("x", &data(m * k, 220)),
            ("up", &data(k * ff, 221)),
            ("gate", &data(k * ff, 222)),
            ("down", &data(ff * k, 223)),
        ],
        1e-3,
    );
}

/// A realistic attention sub-block composed from primitives, the way a model
/// builder emits it: rms_norm → QKV matmuls → RoPE → attention → out proj →
/// residual. Validates the whole chain (not just isolated ops) matches CPU.
#[test]
fn transformer_attention_block_parity() {
    let (b, s, nh, dh) = (1usize, 4, 2, 4);
    let h = nh * dh; // 8
    let r4 =
        |bb: usize, ss: usize, n: usize, d: usize| vec![bb as i64, ss as i64, n as i64, d as i64];
    let build = move || {
        let mut g = Graph::new("block");
        let x = g.input("x", f(&[b, s, h]));
        let g1 = g.input("g1", f(&[h]));
        let b1 = g.input("b1", f(&[h]));
        let wq = g.input("wq", f(&[h, h]));
        let wk = g.input("wk", f(&[h, h]));
        let wv = g.input("wv", f(&[h, h]));
        let wo = g.input("wo", f(&[h, h]));
        let cos = g.input("cos", f(&[s, dh / 2]));
        let sin = g.input("sin", f(&[s, dh / 2]));

        let xn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, g1, b1],
            f(&[b, s, h]),
        );
        let q = g.add_node(Op::MatMul, vec![xn, wq], f(&[b, s, h]));
        let k = g.add_node(Op::MatMul, vec![xn, wk], f(&[b, s, h]));
        let v = g.add_node(Op::MatMul, vec![xn, wv], f(&[b, s, h]));
        // RoPE on the rank-3 [b, s, h] tensors (before head reshape).
        let rope = |g: &mut Graph, t, c, sn| {
            g.add_node(
                Op::Rope {
                    head_dim: dh,
                    n_rot: dh,
                    style: RopeStyle::NeoX,
                },
                vec![t, c, sn],
                f(&[b, s, h]),
            )
        };
        let qr = rope(&mut g, q, cos, sin);
        let kr = rope(&mut g, k, cos, sin);
        let q4 = g.add_node(
            Op::Reshape {
                new_shape: r4(b, s, nh, dh),
            },
            vec![qr],
            f(&[b, s, nh, dh]),
        );
        let k4 = g.add_node(
            Op::Reshape {
                new_shape: r4(b, s, nh, dh),
            },
            vec![kr],
            f(&[b, s, nh, dh]),
        );
        let v4 = g.add_node(
            Op::Reshape {
                new_shape: r4(b, s, nh, dh),
            },
            vec![v],
            f(&[b, s, nh, dh]),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q4, k4, v4],
            f(&[b, s, nh, dh]),
        );
        let attn3 = g.add_node(
            Op::Reshape {
                new_shape: vec![b as i64, s as i64, h as i64],
            },
            vec![attn],
            f(&[b, s, h]),
        );
        let o = g.add_node(Op::MatMul, vec![attn3, wo], f(&[b, s, h]));
        let out = g.add_node(Op::Binary(BinaryOp::Add), vec![x, o], f(&[b, s, h]));
        g.set_outputs(vec![out]);
        g
    };
    let n = b * s * h;
    parity(
        "transformer_attention_block",
        build,
        &[
            ("x", &data(n, 230)),
            ("g1", &data(h, 231)),
            ("b1", &data(h, 232)),
            ("wq", &data(h * h, 233)),
            ("wk", &data(h * h, 234)),
            ("wv", &data(h * h, 235)),
            ("wo", &data(h * h, 236)),
            ("cos", &data(s * dh / 2, 237)),
            ("sin", &data(s * dh / 2, 238)),
        ],
        3e-3,
    );
}

// ── Extended op coverage (batch: shape / reduction / vision) ───────────────

#[test]
fn reverse_axes() {
    let build = || {
        let mut g = Graph::new("rev");
        let x = g.input("x", f(&[2, 4]));
        let o = g.add_node(Op::Reverse { axes: vec![1] }, vec![x], f(&[2, 4]));
        g.set_outputs(vec![o]);
        g
    };
    parity("reverse_axis1", build, &[("x", &data(8, 300))], 1e-6);

    let build2 = || {
        let mut g = Graph::new("rev2");
        let x = g.input("x", f(&[2, 3]));
        let o = g.add_node(Op::Reverse { axes: vec![0, 1] }, vec![x], f(&[2, 3]));
        g.set_outputs(vec![o]);
        g
    };
    parity("reverse_both", build2, &[("x", &data(6, 301))], 1e-6);
}

#[test]
fn argmax_argmin() {
    for (is_max, tag) in [(true, "argmax"), (false, "argmin")] {
        let build = move || {
            let mut g = Graph::new("arg");
            let x = g.input("x", f(&[4, 5]));
            let op = if is_max {
                Op::ArgMax {
                    axis: 1,
                    keep_dim: false,
                }
            } else {
                Op::ArgMin {
                    axis: 1,
                    keep_dim: false,
                }
            };
            let o = g.add_node(op, vec![x], f(&[4]));
            g.set_outputs(vec![o]);
            g
        };
        parity(tag, build, &[("x", &data(20, 310))], 1e-6);
    }
}

#[test]
fn layer_norm_2d() {
    let build = || {
        let mut g = Graph::new("ln2d");
        let x = g.input("x", f(&[1, 4, 2, 2]));
        let gm = g.input("g", f(&[4]));
        let bt = g.input("b", f(&[4]));
        let o = g.add_node(
            Op::LayerNorm2d { eps: 1e-6 },
            vec![x, gm, bt],
            f(&[1, 4, 2, 2]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "layer_norm_2d",
        build,
        &[
            ("x", &data(16, 320)),
            ("g", &data(4, 321)),
            ("b", &data(4, 322)),
        ],
        1e-3,
    );
}

#[test]
fn pool2d_max_avg() {
    for (kind, tag) in [(ReduceOp::Max, "max"), (ReduceOp::Mean, "avg")] {
        let build = move || {
            let mut g = Graph::new("pool");
            let x = g.input("x", f(&[1, 2, 4, 4]));
            let o = g.add_node(
                Op::Pool {
                    kind,
                    kernel_size: vec![2, 2],
                    stride: vec![2, 2],
                    padding: vec![0, 0],
                },
                vec![x],
                f(&[1, 2, 2, 2]),
            );
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("pool2d_{tag}"),
            build,
            &[("x", &data(32, 330))],
            1e-4,
        );
    }
}

// ── Decomposition confirmations (no native kernel; rewrite → primitives) ────

#[test]
fn group_norm_decomposed() {
    let build = || {
        let mut g = Graph::new("gn");
        let x = g.input("x", f(&[1, 4, 2, 2]));
        let gm = g.input("g", f(&[4]));
        let bt = g.input("b", f(&[4]));
        let o = g.add_node(
            Op::GroupNorm {
                num_groups: 2,
                eps: 1e-5,
            },
            vec![x, gm, bt],
            f(&[1, 4, 2, 2]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "group_norm",
        build,
        &[
            ("x", &data(16, 340)),
            ("g", &data(4, 341)),
            ("b", &data(4, 342)),
        ],
        1e-3,
    );
}

#[test]
fn batch_norm_inference_decomposed() {
    let build = || {
        let mut g = Graph::new("bn");
        let x = g.input("x", f(&[2, 4]));
        let gm = g.input("g", f(&[4]));
        let bt = g.input("b", f(&[4]));
        let mean = g.input("m", f(&[4]));
        let var = g.input("v", f(&[4]));
        let o = g.add_node(
            Op::BatchNormInference { eps: 1e-5 },
            vec![x, gm, bt, mean, var],
            f(&[2, 4]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "batch_norm_inference",
        build,
        &[
            ("x", &data(8, 350)),
            ("g", &data(4, 351)),
            ("b", &data(4, 352)),
            ("m", &data(4, 353)),
            ("v", &pos(4, 354)),
        ],
        1e-3,
    );
}

#[test]
fn resize_nearest_2x_decomposed() {
    let build = || {
        let mut g = Graph::new("rn2x");
        let x = g.input("x", f(&[1, 2, 2, 2]));
        let o = g.add_node(Op::ResizeNearest2x, vec![x], f(&[1, 2, 4, 4]));
        g.set_outputs(vec![o]);
        g
    };
    parity("resize_nearest_2x", build, &[("x", &data(8, 360))], 1e-6);
}

// ── Batch 2: MoE + vision convolution ──────────────────────────────────────

#[test]
fn grouped_matmul_moe() {
    let (m, k, n, e) = (4usize, 3, 5, 2);
    let build = || {
        let mut g = Graph::new("gmm");
        let input = g.input("in", f(&[m, k]));
        let weight = g.input("w", f(&[e, k, n]));
        let idx = g.input("idx", f(&[m]));
        let o = g.add_node(Op::GroupedMatMul, vec![input, weight, idx], f(&[m, n]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "grouped_matmul_moe",
        build,
        &[
            ("in", &data(m * k, 400)),
            ("w", &data(e * k * n, 401)),
            ("idx", &[0.0, 1.0, 0.0, 1.0]),
        ],
        1e-3,
    );
}

#[test]
fn conv2d_variants() {
    // (stride, pad, dilation, groups, out_h, out_w, tag)
    let cases = [
        (1usize, 1usize, 1usize, 1usize, 5usize, 5usize, "same_pad"),
        (2, 1, 1, 1, 3, 3, "stride2"),
        (1, 0, 1, 1, 3, 3, "valid"),
    ];
    for (st, pad, dil, grp, oh, ow, tag) in cases {
        let (cin, cout) = (2usize, 3usize);
        // Op::Conv takes [x, weight] (no bias input; bias is a separate Add).
        let build = move || {
            let mut g = Graph::new("conv");
            let x = g.input("x", f(&[1, cin, 5, 5]));
            let w = g.input("w", f(&[cout, cin, 3, 3]));
            let o = g.add_node(
                Op::Conv {
                    kernel_size: vec![3, 3],
                    stride: vec![st, st],
                    padding: vec![pad, pad],
                    dilation: vec![dil, dil],
                    groups: grp,
                },
                vec![x, w],
                f(&[1, cout, oh, ow]),
            );
            g.set_outputs(vec![o]);
            g
        };
        parity(
            &format!("conv2d_{tag}"),
            build,
            &[
                ("x", &data(cin * 25, 410)),
                ("w", &data(cout * cin * 9, 411)),
            ],
            1e-3,
        );
    }

    // Grouped conv (groups == cin == cout): depthwise.
    let build = || {
        let mut g = Graph::new("dwconv");
        let x = g.input("x", f(&[1, 4, 5, 5]));
        let w = g.input("w", f(&[4, 1, 3, 3]));
        let o = g.add_node(
            Op::Conv {
                kernel_size: vec![3, 3],
                stride: vec![1, 1],
                padding: vec![1, 1],
                dilation: vec![1, 1],
                groups: 4,
            },
            vec![x, w],
            f(&[1, 4, 5, 5]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "conv2d_depthwise",
        build,
        &[("x", &data(100, 413)), ("w", &data(36, 414))],
        1e-3,
    );
}

// ── Batch 3: SSM ───────────────────────────────────────────────────────────

#[test]
fn selective_scan_mamba() {
    let (b, s, h, n) = (1usize, 4, 3, 2);
    let build = || {
        let mut g = Graph::new("ssm");
        let x = g.input("x", f(&[b, s, h]));
        let delta = g.input("delta", f(&[b, s, h]));
        let a = g.input("a", f(&[h, n]));
        let bm = g.input("b", f(&[b, s, n]));
        let cm = g.input("c", f(&[b, s, n]));
        let o = g.add_node(
            Op::SelectiveScan { state_size: n },
            vec![x, delta, a, bm, cm],
            f(&[b, s, h]),
        );
        g.set_outputs(vec![o]);
        g
    };
    // Keep Δ and A modest so exp(Δ·A) stays well-conditioned.
    let small: Vec<f32> = data(b * s * h, 500).iter().map(|v| v * 0.1).collect();
    let neg_a: Vec<f32> = data(h * n, 502)
        .iter()
        .map(|v| -(v.abs() * 0.5 + 0.1))
        .collect();
    parity(
        "selective_scan_mamba",
        build,
        &[
            ("x", &data(b * s * h, 501)),
            ("delta", &small),
            ("a", &neg_a),
            ("b", &data(b * s * n, 503)),
            ("c", &data(b * s * n, 504)),
        ],
        2e-3,
    );
}

// ── Batch 4: vision unfold + scatter ───────────────────────────────────────

#[test]
fn im2col_unfold() {
    let build = || {
        let mut g = Graph::new("i2c");
        let x = g.input("x", f(&[1, 2, 4, 4]));
        let o = g.add_node(
            Op::Im2Col {
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
                dilation: vec![1, 1],
            },
            vec![x],
            f(&[4, 8]),
        );
        g.set_outputs(vec![o]);
        g
    };
    parity("im2col_unfold", build, &[("x", &data(32, 600))], 1e-6);
}

#[test]
fn scatter_add_axis0() {
    let build = || {
        let mut g = Graph::new("sca");
        let upd = g.input("u", f(&[3, 2]));
        let idx = g.input("i", f(&[3]));
        let o = g.add_node(Op::ScatterAdd, vec![upd, idx], f(&[4, 2]));
        g.set_outputs(vec![o]);
        g
    };
    parity(
        "scatter_add_axis0",
        build,
        &[("u", &data(6, 610)), ("i", &[0.0, 2.0, 0.0])],
        1e-5,
    );
}

#[test]
fn topk_indices() {
    let build = || {
        let mut g = Graph::new("topk");
        let x = g.input("x", f(&[2, 6]));
        let o = g.add_node(Op::TopK { k: 3 }, vec![x], f(&[2, 3]));
        g.set_outputs(vec![o]);
        g
    };
    parity("topk_indices", build, &[("x", &data(12, 700))], 1e-6);
}

// ── Host-fallback ops (RNN family / Mamba2 / conv-transpose / FFT) ──────────
// These run on the CPU reference against the host-visible arena (no native
// SPIR-V kernel yet), so parity is exact by construction.

#[test]
fn rnn_host() {
    for (relu, tag) in [(false, "tanh"), (true, "relu")] {
        let (b, s, inp, h) = (1usize, 3, 4, 5);
        let build = move || {
            let mut g = Graph::new("rnn");
            let x = g.input("x", f(&[b, s, inp]));
            let wih = g.input("wih", f(&[h, inp]));
            let whh = g.input("whh", f(&[h, h]));
            let bias = g.input("bias", f(&[h]));
            let y = g.rnn(x, wih, whh, bias, None, h, 1, false, relu, f(&[b, s, h]));
            g.set_outputs(vec![y]);
            g
        };
        parity(
            &format!("rnn_{tag}"),
            build,
            &[
                ("x", &data(b * s * inp, 800)),
                ("wih", &data(h * inp, 801)),
                ("whh", &data(h * h, 802)),
                ("bias", &data(h, 803)),
            ],
            2e-3,
        );
    }
}

#[test]
fn gru_host() {
    let (b, s, inp, h) = (1usize, 3, 4, 5);
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", f(&[b, s, inp]));
        let wih = g.input("wih", f(&[3 * h, inp]));
        let whh = g.input("whh", f(&[3 * h, h]));
        let bih = g.input("bih", f(&[3 * h]));
        let bhh = g.input("bhh", f(&[3 * h]));
        let y = g.gru(x, wih, whh, bih, bhh, None, h, 1, false, f(&[b, s, h]));
        g.set_outputs(vec![y]);
        g
    };
    parity(
        "gru_host",
        build,
        &[
            ("x", &data(b * s * inp, 810)),
            ("wih", &data(3 * h * inp, 811)),
            ("whh", &data(3 * h * h, 812)),
            ("bih", &data(3 * h, 813)),
            ("bhh", &data(3 * h, 814)),
        ],
        2e-3,
    );
}

#[test]
fn lstm_host() {
    let (b, s, inp, h) = (1usize, 3, 4, 5);
    let build = || {
        let mut g = Graph::new("lstm");
        let x = g.input("x", f(&[b, s, inp]));
        let wih = g.input("wih", f(&[4 * h, inp]));
        let whh = g.input("whh", f(&[4 * h, h]));
        let bias = g.input("bias", f(&[4 * h]));
        let y = g.lstm(x, wih, whh, bias, h, 1, false, f(&[b, s, h]));
        g.set_outputs(vec![y]);
        g
    };
    parity(
        "lstm_host",
        build,
        &[
            ("x", &data(b * s * inp, 820)),
            ("wih", &data(4 * h * inp, 821)),
            ("whh", &data(4 * h * h, 822)),
            ("bias", &data(4 * h, 823)),
        ],
        2e-3,
    );
}

#[test]
fn mamba2_host() {
    let (bb, ss, hh, pp, nn) = (1usize, 3, 2, 2, 2);
    let build = || {
        let mut g = Graph::new("mamba2");
        let x = g.input("x", f(&[bb, ss, hh, pp]));
        let dt = g.input("dt", f(&[bb, ss, hh]));
        let a = g.input("a", f(&[hh]));
        let bmat = g.input("b", f(&[bb, ss, hh, nn]));
        let cmat = g.input("c", f(&[bb, ss, hh, nn]));
        let y = g.mamba2(x, dt, a, bmat, cmat, pp, nn, f(&[bb, ss, hh, pp]));
        g.set_outputs(vec![y]);
        g
    };
    let dt: Vec<f32> = data(bb * ss * hh, 831)
        .iter()
        .map(|v| v.abs() * 0.1)
        .collect();
    let a: Vec<f32> = data(hh, 832)
        .iter()
        .map(|v| -(v.abs() * 0.5 + 0.1))
        .collect();
    parity(
        "mamba2_host",
        build,
        &[
            ("x", &data(bb * ss * hh * pp, 830)),
            ("dt", &dt),
            ("a", &a),
            ("b", &data(bb * ss * hh * nn, 833)),
            ("c", &data(bb * ss * hh * nn, 834)),
        ],
        2e-3,
    );
}

#[test]
fn conv_transpose2d_host() {
    let build = || {
        let mut g = Graph::new("convt");
        let x = g.input("x", f(&[1, 2, 3, 3]));
        let w = g.input("w", f(&[2, 3, 2, 2]));
        let y = g.conv_transpose2d(x, w, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1);
        g.set_outputs(vec![y]);
        g
    };
    parity(
        "conv_transpose2d_host",
        build,
        &[("x", &data(18, 840)), ("w", &data(24, 841))],
        2e-3,
    );
}

#[test]
fn fft_host() {
    // f32 forward FFT of a 4-point complex signal (2N=8 real-block layout).
    let build = || {
        let mut g = Graph::new("fft");
        let x = g.input("x", f(&[8]));
        let y = g.fft(x, false);
        g.set_outputs(vec![y]);
        g
    };
    parity("fft_host", build, &[("x", &data(8, 850))], 1e-3);
}

#[test]
fn dequant_matmul_host() {
    if !is_available(Device::Vulkan) {
        eprintln!("[parity] dequant_matmul: SKIP (no Vulkan device)");
        return;
    }
    use rlx_gguf::GgmlType;
    let (m, k, n) = (2usize, 64, 8); // k a multiple of the 32-element block
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.45)
        .collect();
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    for ggml in [GgmlType::Q8_0, GgmlType::Q4_0] {
        let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
        let scheme = rlx_cpu::quant_scheme_for_ggml(ggml).expect("scheme");
        let pn = packed.len();
        let build = || {
            let mut g = Graph::new("dq");
            let xi = g.input("x", f(&[m, k]));
            let wp = g.param("w_packed", Shape::new(&[pn], DType::U8));
            let y = g.add_node(Op::DequantMatMul { scheme }, vec![xi, wp], f(&[m, n]));
            g.set_outputs(vec![y]);
            g
        };
        let run_on_dev = |dev| {
            let mut c = Session::new(dev).compile(build());
            c.set_param_typed("w_packed", &packed, DType::U8);
            c.run(&[("x", x.as_slice())]).pop().unwrap()
        };
        let cpu = run_on_dev(Device::Cpu);
        let vk = run_on_dev(Device::Vulkan);
        let mx = max_abs(&cpu, &vk);
        assert!(
            mx < 1e-3,
            "dequant_matmul {ggml:?}: max|Δ| = {mx}\n cpu={:?}\n  vk={:?}",
            &cpu[..cpu.len().min(8)],
            &vk[..vk.len().min(8)]
        );
        eprintln!("[parity] dequant_matmul_{ggml:?}: OK");
    }
}

#[test]
fn rng_and_sample_host() {
    // RngNormal/RngUniform carry their own key+op_seed (deterministic per-op),
    // and Sample carries a fixed seed — so the host-fallback matches CPU.
    let rng_build = |normal: bool| {
        move || {
            let mut g = Graph::new("rng");
            let t = g.input("t", f(&[2, 3]));
            let op = if normal {
                Op::RngNormal {
                    mean: 0.0,
                    scale: 1.0,
                    key: 42,
                    op_seed: Some(1.0),
                }
            } else {
                Op::RngUniform {
                    low: -1.0,
                    high: 1.0,
                    key: 42,
                    op_seed: Some(1.0),
                }
            };
            let o = g.add_node(op, vec![t], f(&[2, 3]));
            g.set_outputs(vec![o]);
            g
        }
    };
    parity("rng_normal", rng_build(true), &[("t", &data(6, 900))], 1e-5);
    parity(
        "rng_uniform",
        rng_build(false),
        &[("t", &data(6, 901))],
        1e-5,
    );

    // Sample: logits [2, 10] → token indices [2], deterministic seed.
    let build = || {
        let mut g = Graph::new("sample");
        let logits = g.input("logits", f(&[2, 10]));
        let y = g.sample(logits, 0, 1.0, 1.0, 7, f(&[2]));
        g.set_outputs(vec![y]);
        g
    };
    parity("sample", build, &[("logits", &data(20, 902))], 1e-5);
}
