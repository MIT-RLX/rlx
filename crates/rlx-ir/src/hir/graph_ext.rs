// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Shape-inferred HIR builder — mirrors [`crate::infer::GraphExt`] for
//! [`super::HirModule`], emitting [`super::HirOp::Mir`] primitives.

use crate::hir::{HirModule, HirNodeId};
use crate::op::*;
use crate::shape;
use crate::{DType, Op, Shape};

/// Mutable HIR builder view — implements [`HirGraphExt`] without conflicting
/// with [`HirModule`]'s block-level `rms_norm` / `rope` methods.
pub struct HirMut<'a>(pub &'a mut HirModule);

impl<'a> HirMut<'a> {
    pub fn new(hir: &'a mut HirModule) -> Self {
        Self(hir)
    }

    pub fn inner(&mut self) -> &mut HirModule {
        self.0
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.0.input(name, shape)
    }

    pub fn param(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.0.param(name, shape)
    }

    pub fn set_outputs(&mut self, outputs: Vec<HirNodeId>) {
        self.0.set_outputs(outputs);
    }

    /// Scaled dot-product attention with a caller-supplied mask tensor
    /// (matches legacy [`crate::Graph::attention`]).
    pub fn attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            crate::ops::attention::attention_kind_op(
                num_heads,
                head_dim,
                MaskKind::Custom,
                None,
                None,
            ),
            vec![q, k, v, mask],
            shape,
        )
    }
}

/// Ergonomic shape-inferred building on [`HirMut`].
pub trait HirGraphExt {
    fn shape(&self, id: HirNodeId) -> &Shape;

    fn add_node(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId;

    fn mm(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn add(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn sub(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn mul(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn div(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;

    fn gelu(&mut self, x: HirNodeId) -> HirNodeId;
    fn gelu_approx(&mut self, x: HirNodeId) -> HirNodeId;
    fn silu(&mut self, x: HirNodeId) -> HirNodeId;
    fn relu(&mut self, x: HirNodeId) -> HirNodeId;
    fn exp(&mut self, x: HirNodeId) -> HirNodeId;
    fn sqrt(&mut self, x: HirNodeId) -> HirNodeId;
    fn neg(&mut self, x: HirNodeId) -> HirNodeId;
    fn tanh(&mut self, x: HirNodeId) -> HirNodeId;

    fn ln(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId;
    fn group_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        num_groups: usize,
        eps: f32,
    ) -> HirNodeId;
    fn layer_norm2d(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
    ) -> HirNodeId;
    fn resize_nearest_2x(&mut self, x: HirNodeId) -> HirNodeId;
    fn conv2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId;
    fn conv_transpose2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId;
    fn rms_norm(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId;

    fn sum(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId;
    fn mean(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId;
    fn sm(&mut self, x: HirNodeId, axis: i32) -> HirNodeId;

    fn reshape_(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId;
    fn transpose_(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId;
    fn narrow_(&mut self, x: HirNodeId, axis: usize, start: usize, len: usize) -> HirNodeId;
    fn concat_(&mut self, inputs: Vec<HirNodeId>, axis: usize) -> HirNodeId;
    fn gather_(&mut self, table: HirNodeId, indices: HirNodeId, axis: usize) -> HirNodeId;

    fn eq(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;
    fn lt(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId;

    fn attention_(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> HirNodeId;

    fn attention_kind(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> HirNodeId {
        self.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, shape, None, None)
    }

    fn attention_kind_opts(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> HirNodeId;

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, head_dim: usize) -> HirNodeId;
    fn rope_n(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> HirNodeId;

    fn cast(&mut self, x: HirNodeId, to: DType) -> HirNodeId;
    fn activation(&mut self, act: Activation, input: HirNodeId, shape: Shape) -> HirNodeId;

    fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId;

    fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId;
}

impl HirGraphExt for HirMut<'_> {
    fn shape(&self, id: HirNodeId) -> &Shape {
        &self.0.node(id).shape
    }

    fn add_node(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId {
        self.0.mir(op, inputs, shape)
    }

    fn mm(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s =
            shape::matmul_shape(self.shape(lhs), self.shape(rhs)).expect("matmul shape inference");
        self.0.mir(Op::MatMul, vec![lhs, rhs], s)
    }

    fn add(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("add shape inference");
        self.0.mir(Op::Binary(BinaryOp::Add), vec![lhs, rhs], s)
    }

    fn sub(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("sub shape inference");
        self.0.mir(Op::Binary(BinaryOp::Sub), vec![lhs, rhs], s)
    }

    fn mul(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("mul shape inference");
        self.0.mir(Op::Binary(BinaryOp::Mul), vec![lhs, rhs], s)
    }

    fn div(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("div shape inference");
        self.0.mir(Op::Binary(BinaryOp::Div), vec![lhs, rhs], s)
    }

    fn gelu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Gelu), vec![x], s)
    }

    fn gelu_approx(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::Activation(Activation::GeluApprox), vec![x], s)
    }

    fn silu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Silu), vec![x], s)
    }

    fn relu(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Relu), vec![x], s)
    }

    fn exp(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Exp), vec![x], s)
    }

    fn sqrt(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Sqrt), vec![x], s)
    }

    fn neg(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Neg), vec![x], s)
    }

    fn tanh(&mut self, x: HirNodeId) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::Activation(Activation::Tanh), vec![x], s)
    }

    fn ln(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::LayerNorm { axis: -1, eps }, vec![x, gamma, beta], s)
    }

    fn group_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        num_groups: usize,
        eps: f32,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::GroupNorm { num_groups, eps }, vec![x, gamma, beta], s)
    }

    fn layer_norm2d(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(Op::LayerNorm2d { eps }, vec![x, gamma, beta], s)
    }

    fn resize_nearest_2x(&mut self, x: HirNodeId) -> HirNodeId {
        let in_s = self.shape(x);
        let out = Shape::new(
            &[
                in_s.dim(0).unwrap_static(),
                in_s.dim(1).unwrap_static(),
                in_s.dim(2).unwrap_static() * 2,
                in_s.dim(3).unwrap_static() * 2,
            ],
            in_s.dtype(),
        );
        self.0.mir(Op::ResizeNearest2x, vec![x], out)
    }

    fn conv2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            Op::Conv {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: vec![1, 1],
                groups,
            },
            vec![x, weight],
            out_shape,
        )
    }

    fn conv_transpose2d(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.0.mir(
            Op::ConvTranspose2d {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: dilation.to_vec(),
                output_padding: output_padding.to_vec(),
                groups,
            },
            vec![x, weight],
            out_shape,
        )
    }

    fn rms_norm(&mut self, x: HirNodeId, gamma: HirNodeId, beta: HirNodeId, eps: f32) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0
            .mir(Op::RmsNorm { axis: -1, eps }, vec![x, gamma, beta], s)
    }

    fn sum(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.0.mir(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes,
                keep_dim,
            },
            vec![x],
            s,
        )
    }

    fn mean(&mut self, x: HirNodeId, axes: Vec<usize>, keep_dim: bool) -> HirNodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.0.mir(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes,
                keep_dim,
            },
            vec![x],
            s,
        )
    }

    fn sm(&mut self, x: HirNodeId, axis: i32) -> HirNodeId {
        let s = shape::softmax_shape(self.shape(x));
        self.0.mir(Op::Softmax { axis }, vec![x], s)
    }

    fn reshape_(&mut self, x: HirNodeId, new_shape: Vec<i64>) -> HirNodeId {
        let s = shape::reshape_shape(self.shape(x), &new_shape).expect("reshape shape inference");
        self.0.mir(Op::Reshape { new_shape }, vec![x], s)
    }

    fn transpose_(&mut self, x: HirNodeId, perm: Vec<usize>) -> HirNodeId {
        let s = shape::transpose_shape(self.shape(x), &perm).expect("transpose shape inference");
        self.0.mir(Op::Transpose { perm }, vec![x], s)
    }

    fn narrow_(&mut self, x: HirNodeId, axis: usize, start: usize, len: usize) -> HirNodeId {
        let s = shape::narrow_shape(self.shape(x), axis, len).expect("narrow shape inference");
        self.0.mir(Op::Narrow { axis, start, len }, vec![x], s)
    }

    fn concat_(&mut self, inputs: Vec<HirNodeId>, axis: usize) -> HirNodeId {
        let shapes: Vec<&Shape> = inputs.iter().map(|&id| self.shape(id)).collect();
        let s = shape::concat_shape(&shapes, axis).expect("concat shape inference");
        self.0.mir(Op::Concat { axis }, inputs, s)
    }

    fn gather_(&mut self, table: HirNodeId, indices: HirNodeId, axis: usize) -> HirNodeId {
        let s = shape::gather_shape(self.shape(table), self.shape(indices), axis)
            .expect("gather shape inference");
        self.0.mir(Op::Gather { axis }, vec![table, indices], s)
    }

    fn eq(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("eq shape inference");
        self.0.mir(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
    }

    fn lt(&mut self, lhs: HirNodeId, rhs: HirNodeId) -> HirNodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("lt shape inference");
        self.0.mir(Op::Compare(CmpOp::Lt), vec![lhs, rhs], s)
    }

    fn attention_(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: HirNodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> HirNodeId {
        let s = shape::attention_shape(self.shape(q));
        HirMut::attention(self, q, k, v, mask, num_heads, head_dim, s)
    }

    fn attention_kind(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> HirNodeId {
        self.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, shape, None, None)
    }

    fn attention_kind_opts(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> HirNodeId {
        self.0.mir(
            crate::ops::attention::attention_kind_op(
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            ),
            vec![q, k, v],
            shape,
        )
    }

    fn rope(&mut self, x: HirNodeId, cos: HirNodeId, sin: HirNodeId, head_dim: usize) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        self.0.mir(
            Op::Rope {
                head_dim,
                n_rot: head_dim,
            },
            vec![x, cos, sin],
            s,
        )
    }

    fn rope_n(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> HirNodeId {
        let s = shape::unary_shape(self.shape(x));
        HirModule::rope(self.0, x, cos, sin, head_dim, n_rot, s)
    }

    fn cast(&mut self, x: HirNodeId, to: DType) -> HirNodeId {
        let s = self.shape(x).clone().with_dtype(to);
        self.0.mir(Op::Cast { to }, vec![x], s)
    }

    fn activation(&mut self, act: Activation, input: HirNodeId, shape: Shape) -> HirNodeId {
        self.0.mir(Op::Activation(act), vec![input], shape)
    }

    fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId {
        HirModule::gated_delta_net(self.0, q, k, v, g, beta, state_size, shape)
    }

    fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        shape: Shape,
    ) -> HirNodeId {
        HirModule::gated_delta_net_carry(self.0, q, k, v, g, beta, state, state_size, shape)
    }
}
