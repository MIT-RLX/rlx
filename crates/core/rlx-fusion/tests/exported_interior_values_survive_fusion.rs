// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A producer that is also a graph output must not be absorbed by a fusion.
//!
//! `FuseMatMulBiasAct`, `FuseConvBiasAct`/`FuseConvAffineAct` and
//! `FuseMatMulResidual` checked "one consumer" with `users().len() == 1`, which
//! says nothing about graph outputs. With the matmul exported as well, the
//! rewrite pointed that output at the fused node, so it returned the post-bias
//! value (rlx CPU: `x·w` came back as `x·w + b`). Found by the IREE round trip
//! in `rlx-stablehlo`, whose bisect exposes every op. The check is
//! `has_single_use`, which also excludes graph outputs.
//!
//! Each case: the pattern fuses when the interior value is private (so the
//! check below is not vacuous), and keeps that value's op when it is exported.

use rlx_fusion::fusion::{
    FuseConvAffineAct, FuseConvBiasAct, FuseMatMulBiasAct, FuseMatMulResidual,
};
use rlx_fusion::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, OpKind, Shape};

const F: DType = DType::F32;

fn kinds(g: &Graph) -> Vec<OpKind> {
    g.nodes().iter().map(|n| n.op.kind()).collect()
}

/// `relu(x·w + b)`; `export` = the interior node to also expose.
fn mm_bias_act(export: fn(&[NodeId; 2]) -> Option<NodeId>) -> Graph {
    let mut g = Graph::new("mm_bias_act");
    let x = g.input("x", Shape::new(&[2, 3], F));
    let w = g.param("w", Shape::new(&[3, 4], F));
    let b = g.param("b", Shape::new(&[4], F));
    let s = Shape::new(&[2, 4], F);
    let m = g.matmul(x, w, s.clone());
    let a = g.binary(BinaryOp::Add, m, b, s.clone());
    let r = g.add_node(Op::Activation(Activation::Relu), vec![a], s);
    let mut outs = vec![r];
    outs.extend(export(&[m, a]));
    g.set_outputs(outs);
    g
}

fn mm_residual(export_mm: bool) -> Graph {
    let mut g = Graph::new("mm_residual");
    let x = g.input("x", Shape::new(&[2, 3], F));
    let h = g.input("h", Shape::new(&[2, 4], F));
    let w = g.param("w", Shape::new(&[3, 4], F));
    let s = Shape::new(&[2, 4], F);
    let m = g.matmul(x, w, s.clone());
    let y = g.binary(BinaryOp::Add, m, h, s);
    g.set_outputs(if export_mm { vec![y, m] } else { vec![y] });
    g
}

fn conv(g: &mut Graph) -> (NodeId, Shape) {
    let x = g.input("x", Shape::new(&[1, 2, 6, 6], F));
    let w = g.param("w", Shape::new(&[4, 2, 3, 3], F));
    let s = Shape::new(&[1, 4, 6, 6], F);
    let c = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        s.clone(),
    );
    (c, s)
}

fn per_channel(g: &mut Graph, name: &str, s: &Shape) -> NodeId {
    let p = g.param(name, Shape::new(&[4], F));
    let r = g.add_node(
        Op::Reshape {
            new_shape: vec![1, 4, 1, 1],
        },
        vec![p],
        Shape::new(&[1, 4, 1, 1], F),
    );
    g.add_node(
        Op::Expand {
            target_shape: s.dims().iter().map(|d| d.unwrap_static() as i64).collect(),
        },
        vec![r],
        s.clone(),
    )
}

/// `relu(conv(x) + b)`.
fn conv_bias_act(export_conv: bool) -> Graph {
    let mut g = Graph::new("conv_bias_act");
    let (c, s) = conv(&mut g);
    let b = per_channel(&mut g, "b", &s);
    let a = g.binary(BinaryOp::Add, c, b, s.clone());
    let r = g.add_node(Op::Activation(Activation::Relu), vec![a], s);
    g.set_outputs(if export_conv { vec![r, c] } else { vec![r] });
    g
}

/// `relu(conv(x)·scale + shift)` (host-folded BatchNorm).
fn conv_affine_act(export: fn(&[NodeId; 2]) -> Option<NodeId>) -> Graph {
    let mut g = Graph::new("conv_affine_act");
    let (c, s) = conv(&mut g);
    let sc = per_channel(&mut g, "scale", &s);
    let sh = per_channel(&mut g, "shift", &s);
    let m = g.binary(BinaryOp::Mul, c, sc, s.clone());
    let a = g.binary(BinaryOp::Add, m, sh, s.clone());
    let r = g.add_node(Op::Activation(Activation::Relu), vec![a], s);
    let mut outs = vec![r];
    outs.extend(export(&[c, m]));
    g.set_outputs(outs);
    g
}

/// The output at `idx` is still produced by an op of `kind`.
fn output_kind(g: &Graph, idx: usize) -> OpKind {
    g.node(g.outputs[idx]).op.kind()
}

#[test]
fn matmul_bias_act() {
    let fused = FuseMatMulBiasAct.run(mm_bias_act(|_| None));
    assert!(
        kinds(&fused).contains(&OpKind::FusedMatMulBiasAct),
        "control: should fuse"
    );

    let g = FuseMatMulBiasAct.run(mm_bias_act(|ids| Some(ids[0])));
    assert_eq!(
        output_kind(&g, 1),
        OpKind::MatMul,
        "exported matmul was absorbed"
    );

    // The bias add exported: matmul+bias may fuse, the activation must not.
    let g = FuseMatMulBiasAct.run(mm_bias_act(|ids| Some(ids[1])));
    assert_ne!(
        output_kind(&g, 0),
        output_kind(&g, 1),
        "relu and add outputs merged"
    );
}

#[test]
fn matmul_residual() {
    let fused = FuseMatMulResidual.run(mm_residual(false));
    assert!(
        kinds(&fused).contains(&OpKind::FusedMatMulResidual),
        "control: should fuse"
    );
    let g = FuseMatMulResidual.run(mm_residual(true));
    assert_eq!(
        output_kind(&g, 1),
        OpKind::MatMul,
        "exported matmul was absorbed"
    );
}

#[test]
fn conv_bias_act_keeps_an_exported_conv() {
    let fused = FuseConvBiasAct.run(conv_bias_act(false));
    assert!(
        kinds(&fused).contains(&OpKind::FusedConvBiasAct),
        "control: should fuse"
    );
    let g = FuseConvBiasAct.run(conv_bias_act(true));
    assert_eq!(
        output_kind(&g, 1),
        OpKind::Conv,
        "exported conv was absorbed"
    );
}

#[test]
fn conv_affine_act_keeps_exported_interiors() {
    let fused = FuseConvAffineAct.run(conv_affine_act(|_| None));
    assert!(
        kinds(&fused).contains(&OpKind::FusedConvBiasAct),
        "control: should fuse"
    );
    let g = FuseConvAffineAct.run(conv_affine_act(|ids| Some(ids[0])));
    assert_eq!(
        output_kind(&g, 1),
        OpKind::Conv,
        "exported conv was absorbed"
    );
    let g = FuseConvAffineAct.run(conv_affine_act(|ids| Some(ids[1])));
    assert_eq!(
        output_kind(&g, 1),
        OpKind::Binary,
        "exported scale multiply was absorbed"
    );
}
