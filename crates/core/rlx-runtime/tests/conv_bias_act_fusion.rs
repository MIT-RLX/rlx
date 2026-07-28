// SPDX-License-Identifier: MIT OR Apache-2.0
//! `FuseConvBiasAct` correctness — the CUDA→cuDNN fused conv-bias-activation op.
//!
//! CUDA is the only backend that claims `Op::FusedConvBiasAct` (lowered to
//! `cudnnConvolutionBiasActivationForward`); every other backend decomposes it
//! back to `Conv → Reshape → Expand → Add → [Activation]` in `unfuse`. This
//! test validates the two GPU-free, correctness-critical halves on CPU:
//!
//!   1. Structural — `FuseConvBiasAct` matches the canonical conv-bias graph
//!      and collapses it to exactly one fused node; `unfuse_fused_for_autodiff`
//!      restores the primitives.
//!   2. Numeric — running `fuse → unfuse` is a bit-exact no-op vs the original
//!      graph on CPU, so the fusion never changes semantics.
//!
//! The cuDNN fused *dispatch* itself is validated on a CUDA rig
//! (`cuda_backprop_parity.rs`-style), not here.

use rlx_fusion::pass::Pass;
use rlx_fusion::{FuseConvAffineAct, FuseConvBiasAct, unfuse_fused_for_autodiff};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_runtime::{Device, Session};

#[derive(Clone, Copy)]
struct Cfg {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
    groups: usize,
    act: Option<Activation>,
}

/// Canonical conv-bias-[activation] graph, mirroring `conv_bias_parity.rs`:
/// `conv2d → reshape(bias→[1,C,1,1]) → Expand([1,C,H,W]) → Add → [Activation]`.
fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_bias_act");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let bias = g.input("b", Shape::new(&[cfg.c_out], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let bias4 = g.reshape(
        bias,
        vec![1, cfg.c_out as i64, 1, 1],
        Shape::new(&[1, cfg.c_out, 1, 1], f),
    );
    let exp = g.add_node(
        Op::Expand {
            target_shape: vec![1, cfg.c_out as i64, cfg.h as i64, cfg.w as i64],
        },
        vec![bias4],
        out_s.clone(),
    );
    let out = g.binary(BinaryOp::Add, y, exp, out_s.clone());
    let out = match cfg.act {
        Some(a) => g.add_node(Op::Activation(a), vec![out], out_s),
        None => out,
    };
    g.set_outputs(vec![out]);
    g
}

fn count<F: Fn(&Op) -> bool>(g: &Graph, pred: F) -> usize {
    g.nodes().iter().filter(|n| pred(&n.op)).count()
}

fn feeds(cfg: &Cfg) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.c_in * cfg.h * cfg.w)
        .map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k * cfg.k)
        .map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..cfg.c_out).map(|i| i as f32 * 0.05 - 0.2).collect();
    (x, w, b)
}

fn run_cpu(g: Graph, cfg: &Cfg) -> Vec<f32> {
    let (x, w, b) = feeds(cfg);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("b", b.as_slice()),
        ])
        .pop()
        .unwrap()
}

/// `FuseConvBiasAct` collapses conv+bias+ReLU to a single fused node, and
/// `unfuse` restores the primitives.
#[test]
fn fuse_then_unfuse_structure() {
    let cfg = Cfg {
        c_in: 3,
        c_out: 8,
        h: 10,
        w: 12,
        k: 3,
        p: 1,
        groups: 1,
        act: Some(Activation::Relu),
    };
    let g = build(&cfg);
    assert_eq!(count(&g, |o| matches!(o, Op::Conv { .. })), 1);

    let fused = FuseConvBiasAct.run(g);
    assert_eq!(
        count(&fused, |o| matches!(o, Op::FusedConvBiasAct { .. })),
        1,
        "conv+bias+relu should collapse to one FusedConvBiasAct"
    );
    assert_eq!(count(&fused, |o| matches!(o, Op::Conv { .. })), 0);
    assert_eq!(
        count(&fused, |o| matches!(o, Op::Activation(_))),
        0,
        "ReLU must be folded into the fused op"
    );
    // Fused node carries [input, weight, bias].
    let fnode = fused
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::FusedConvBiasAct { .. }))
        .unwrap();
    assert_eq!(fnode.inputs.len(), 3);
    assert!(matches!(
        fnode.op,
        Op::FusedConvBiasAct {
            activation: Some(Activation::Relu),
            ..
        }
    ));

    let decomp = unfuse_fused_for_autodiff(fused);
    assert_eq!(
        count(&decomp, |o| matches!(o, Op::FusedConvBiasAct { .. })),
        0,
        "unfuse must remove the fused op"
    );
    assert_eq!(count(&decomp, |o| matches!(o, Op::Conv { .. })), 1);
    assert_eq!(count(&decomp, |o| matches!(o, Op::Activation(_))), 1);
}

/// The fusion is narrow: only cuDNN-friendly shapes (groups=1, k>1) with a
/// Relu / no-op epilogue fuse — the combo that benchmarks faster than unfused.
/// Non-Relu activations and 1×1 / depthwise / grouped shapes are left entirely
/// on the conv + elementwise-region path (0 FusedConvBiasAct).
#[test]
fn only_cudnn_friendly_relu_fuses() {
    let base = Cfg {
        c_in: 4,
        c_out: 8,
        h: 8,
        w: 8,
        k: 3,
        p: 1,
        groups: 1,
        act: None,
    };

    // Fuses: friendly shape, Relu or identity epilogue.
    for act in [None, Some(Activation::Relu)] {
        let f = FuseConvBiasAct.run(build(&Cfg { act, ..base }));
        assert_eq!(
            count(&f, |o| matches!(o, Op::FusedConvBiasAct { .. })),
            1,
            "friendly conv + {act:?} should fuse"
        );
    }

    // Does NOT fuse: non-Relu activation (bails entirely — activation stays,
    // conv stays, no fused op), or non-friendly shape.
    let not_fused = [
        Cfg {
            act: Some(Activation::Gelu),
            ..base
        },
        Cfg {
            act: Some(Activation::Sigmoid),
            ..base
        },
        Cfg {
            act: Some(Activation::Silu),
            ..base
        },
        Cfg {
            k: 1,
            p: 0,
            act: Some(Activation::Relu),
            ..base
        }, // 1×1
        Cfg {
            c_in: 8,
            c_out: 8,
            groups: 8,
            act: Some(Activation::Relu),
            ..base
        }, // depthwise
    ];
    for cfg in not_fused {
        let f = FuseConvBiasAct.run(build(&cfg));
        assert_eq!(
            count(&f, |o| matches!(o, Op::FusedConvBiasAct { .. })),
            0,
            "should NOT fuse: k={} groups={} act={:?}",
            cfg.k,
            cfg.groups,
            cfg.act
        );
        assert_eq!(
            count(&f, |o| matches!(o, Op::Conv { .. })),
            1,
            "conv preserved"
        );
    }
}

/// `fuse → unfuse` is a bit-exact no-op vs the original graph on CPU. The
/// friendly+Relu/None cases actually fuse then decompose (exercising the
/// matcher + `unfuse_fused_conv_bias_act`); the rest don't fuse at all, so the
/// round-trip is trivially identity — both must leave CPU numerics unchanged.
#[test]
fn fuse_unfuse_is_numeric_noop() {
    let cases = [
        // Fuse then decompose (friendly shape, Relu / identity epilogue).
        Cfg {
            c_in: 3,
            c_out: 8,
            h: 10,
            w: 12,
            k: 3,
            p: 1,
            groups: 1,
            act: None,
        },
        Cfg {
            c_in: 3,
            c_out: 8,
            h: 10,
            w: 12,
            k: 3,
            p: 1,
            groups: 1,
            act: Some(Activation::Relu),
        },
        // Not fused — non-Relu activation on a friendly shape.
        Cfg {
            c_in: 4,
            c_out: 6,
            h: 9,
            w: 9,
            k: 3,
            p: 1,
            groups: 1,
            act: Some(Activation::Sigmoid),
        },
        Cfg {
            c_in: 4,
            c_out: 6,
            h: 9,
            w: 9,
            k: 3,
            p: 1,
            groups: 1,
            act: Some(Activation::Tanh),
        },
        // Not fused — 1×1 pointwise.
        Cfg {
            c_in: 5,
            c_out: 7,
            h: 8,
            w: 8,
            k: 1,
            p: 0,
            groups: 1,
            act: Some(Activation::Relu),
        },
        // Not fused — depthwise.
        Cfg {
            c_in: 8,
            c_out: 8,
            h: 8,
            w: 8,
            k: 3,
            p: 1,
            groups: 8,
            act: Some(Activation::Relu),
        },
    ];
    for (i, cfg) in cases.iter().enumerate() {
        let reference = run_cpu(build(cfg), cfg);
        let roundtrip = unfuse_fused_for_autodiff(FuseConvBiasAct.run(build(cfg)));
        let got = run_cpu(roundtrip, cfg);
        assert_eq!(
            reference.len(),
            got.len(),
            "case {i}: output length mismatch"
        );
        let maxd = reference
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            maxd <= 1e-6,
            "case {i}: fuse→unfuse changed CPU output by {maxd}"
        );
    }
}

// ── FuseConvAffineAct: host-pre-folded BatchNorm affine ────────────────────

/// funasr CAM++ frozen-BN block shape: `conv(no bias) → Mul(scale[1,C,1,1]) →
/// Add(shift[1,C,1,1]) → Relu`.
fn build_affine(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_affine");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let scale = g.input("s", Shape::new(&[cfg.c_out], f));
    let shift = g.input("b", Shape::new(&[cfg.c_out], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let cc = Shape::new(&[1, cfg.c_out, 1, 1], f);
    let scale_r = g.reshape(scale, vec![1, cfg.c_out as i64, 1, 1], cc.clone());
    let mul = g.binary(BinaryOp::Mul, y, scale_r, out_s.clone());
    let shift_r = g.reshape(shift, vec![1, cfg.c_out as i64, 1, 1], cc);
    let add = g.binary(BinaryOp::Add, mul, shift_r, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![add], out_s);
    g.set_outputs(vec![out]);
    g
}

fn run_cpu_affine(g: Graph, cfg: &Cfg) -> Vec<f32> {
    let (x, w, shift) = feeds(cfg);
    let s: Vec<f32> = (0..cfg.c_out).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("s", s.as_slice()),
            ("b", shift.as_slice()),
        ])
        .pop()
        .unwrap()
}

/// `FuseConvAffineAct` folds `conv→Mul→Add→relu` into one `FusedConvBiasAct`
/// (scale folded into the weights), leaving no standalone Conv/Relu.
#[test]
fn conv_affine_fuses() {
    let cfg = Cfg {
        c_in: 4,
        c_out: 8,
        h: 8,
        w: 8,
        k: 3,
        p: 1,
        groups: 1,
        act: None,
    };
    let f = FuseConvAffineAct.run(build_affine(&cfg));
    assert_eq!(
        count(&f, |o| matches!(
            o,
            Op::FusedConvBiasAct {
                activation: Some(Activation::Relu),
                ..
            }
        )),
        1,
        "conv+affine+relu should fold to one FusedConvBiasAct(Relu)"
    );
    assert_eq!(
        count(&f, |o| matches!(o, Op::Conv { .. })),
        0,
        "conv absorbed"
    );
    assert_eq!(
        count(&f, |o| matches!(o, Op::Activation(_))),
        0,
        "relu absorbed"
    );
    // Not friendly (1×1) → not fused.
    let f11 = FuseConvAffineAct.run(build_affine(&Cfg { k: 1, p: 0, ..cfg }));
    assert_eq!(count(&f11, |o| matches!(o, Op::FusedConvBiasAct { .. })), 0);
}

/// Folding per-channel scale into the weights is numerically equivalent (within
/// float-reorder tolerance) to the original activation-space affine on CPU.
#[test]
fn conv_affine_numeric_equivalent() {
    for cfg in [
        Cfg {
            c_in: 4,
            c_out: 8,
            h: 10,
            w: 12,
            k: 3,
            p: 1,
            groups: 1,
            act: None,
        },
        Cfg {
            c_in: 3,
            c_out: 6,
            h: 9,
            w: 9,
            k: 3,
            p: 1,
            groups: 1,
            act: None,
        },
    ] {
        let reference = run_cpu_affine(build_affine(&cfg), &cfg);
        let folded = unfuse_fused_for_autodiff(FuseConvAffineAct.run(build_affine(&cfg)));
        let got = run_cpu_affine(folded, &cfg);
        let maxd = reference
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(maxd <= 1e-4, "affine fold changed CPU output by {maxd}");
    }
}

/// funasr CAM++ residual block: `conv → Mul → Add(shift) → Add(residual) →
/// Relu` — the residual is a full `[1,C,H,W]` tensor added before the Relu.
fn build_affine_residual(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_affine_res");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let scale = g.input("s", Shape::new(&[cfg.c_out], f));
    let shift = g.input("b", Shape::new(&[cfg.c_out], f));
    let residual = g.input("r", Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let cc = Shape::new(&[1, cfg.c_out, 1, 1], f);
    let scale_r = g.reshape(scale, vec![1, cfg.c_out as i64, 1, 1], cc.clone());
    let mul = g.binary(BinaryOp::Mul, y, scale_r, out_s.clone());
    let shift_r = g.reshape(shift, vec![1, cfg.c_out as i64, 1, 1], cc);
    let add = g.binary(BinaryOp::Add, mul, shift_r, out_s.clone());
    let res_add = g.binary(BinaryOp::Add, add, residual, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![res_add], out_s);
    g.set_outputs(vec![out]);
    g
}

fn run_cpu_affine_residual(g: Graph, cfg: &Cfg) -> Vec<f32> {
    let (x, w, shift) = feeds(cfg);
    let s: Vec<f32> = (0..cfg.c_out).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    let r: Vec<f32> = (0..cfg.c_out * cfg.h * cfg.w)
        .map(|i| (i * 3 % 19) as f32 / 19.0 - 0.5)
        .collect();
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("s", s.as_slice()),
            ("b", shift.as_slice()),
            ("r", r.as_slice()),
        ])
        .pop()
        .unwrap()
}

/// `FuseConvAffineAct` folds the residual block into one
/// `FusedConvBiasAct{has_residual}` (4 inputs), and `fuse→unfuse` (residual
/// added before Relu) is numerically equivalent on CPU.
#[test]
fn conv_affine_residual_fuses_and_matches() {
    let cfg = Cfg {
        c_in: 8,
        c_out: 8,
        h: 10,
        w: 10,
        k: 3,
        p: 1,
        groups: 1,
        act: None,
    };

    let f = FuseConvAffineAct.run(build_affine_residual(&cfg));
    let fnode = f
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::FusedConvBiasAct { .. }))
        .expect("should fuse residual block");
    assert!(matches!(
        fnode.op,
        Op::FusedConvBiasAct {
            has_residual: true,
            activation: Some(Activation::Relu),
            ..
        }
    ));
    assert_eq!(fnode.inputs.len(), 4, "[x, w·scale, shift, residual]");
    assert_eq!(count(&f, |o| matches!(o, Op::Conv { .. })), 0);
    assert_eq!(count(&f, |o| matches!(o, Op::Activation(_))), 0);

    let reference = run_cpu_affine_residual(build_affine_residual(&cfg), &cfg);
    let folded = unfuse_fused_for_autodiff(FuseConvAffineAct.run(build_affine_residual(&cfg)));
    let got = run_cpu_affine_residual(folded, &cfg);
    let maxd = reference
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maxd <= 1e-4, "residual fold changed CPU output by {maxd}");
}

/// Addresses the weight-scale caveat: when weights are compile-time-bound
/// (`param_bindings`), `specialize_params` bakes them to Constants before
/// fusion, and post-fusion `ConstantFolding` folds the emitted `Mul(w, scale)`
/// into a single baked-constant weight `w·scale` — so it costs NOTHING per
/// forward. Mirrors the real pipeline (specialize → fuse → post-fusion fold).
#[test]
fn conv_affine_scale_fold_baked_with_bound_weights() {
    use rlx_compile::{ConstantFolding, specialize_params};
    use std::collections::HashMap;

    let (ci, co, k) = (4usize, 8usize, 3usize);
    let f = DType::F32;
    let mut g = Graph::new("conv_affine_param");
    let x = g.input("x", Shape::new(&[1, ci, 8, 8], f));
    // Weights as Op::Param (as a real model builder emits them).
    let weight = g.add_node(
        Op::Param { name: "w".into() },
        vec![],
        Shape::new(&[co, ci, k, k], f),
    );
    let scale = g.add_node(Op::Param { name: "s".into() }, vec![], Shape::new(&[co], f));
    let shift = g.add_node(Op::Param { name: "b".into() }, vec![], Shape::new(&[co], f));
    let y = g.conv2d(x, weight, [k, k], [1, 1], [1, 1], [1, 1], 1);
    let out_s = Shape::new(&[1, co, 8, 8], f);
    let cc = Shape::new(&[1, co, 1, 1], f);
    let scale_r = g.reshape(scale, vec![1, co as i64, 1, 1], cc.clone());
    let mul = g.binary(BinaryOp::Mul, y, scale_r, out_s.clone());
    let shift_r = g.reshape(shift, vec![1, co as i64, 1, 1], cc);
    let add = g.binary(BinaryOp::Add, mul, shift_r, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![add], out_s);
    g.set_outputs(vec![out]);

    // Uniform weights, but PER-CHANNEL-VARYING scale so an axis-misaligned
    // broadcast fold would produce visibly wrong values.
    let mut bindings: HashMap<String, Vec<f32>> = HashMap::new();
    bindings.insert("w".into(), vec![0.1; co * ci * k * k]);
    bindings.insert("s".into(), (0..co).map(|o| (o + 1) as f32).collect());
    bindings.insert("b".into(), vec![0.5; co]);

    // specialize (weights → Constants) → fuse (emits Mul(w,scale)) → fold.
    let g = specialize_params(&g, &bindings);
    let g = FuseConvAffineAct.run(g);
    let g = ConstantFolding.run(g);

    let fnode = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::FusedConvBiasAct { .. }))
        .expect("should fuse");
    // The scaled weight (input 1) is now a baked Constant — the Mul was folded.
    let Op::Constant { data } = &g.node(fnode.inputs[1]).op else {
        panic!(
            "w·scale must be a baked Constant, got {:?}",
            g.node(fnode.inputs[1]).op
        );
    };
    // Verify the folded VALUE: w[o,i,h,w]·scale[o] = 0.1·(o+1), per output channel.
    let vals: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(vals.len(), co * ci * k * k);
    let per_ch = ci * k * k;
    for (idx, v) in vals.iter().enumerate() {
        let o = idx / per_ch;
        let want = 0.1 * (o + 1) as f32;
        assert!(
            (v - want).abs() < 1e-6,
            "ch {o}: w·scale = {v}, want {want}"
        );
    }
}
