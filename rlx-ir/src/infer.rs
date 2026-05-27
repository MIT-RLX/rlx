// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Shape-inferred graph builder — ergonomic API that auto-computes output shapes.
//!
//! Import [`GraphExt`] and call short-name methods instead of providing explicit shapes:
//! ```rust
//! use rlx_ir::*;
//! use rlx_ir::infer::GraphExt;
//!
//! let mut g = Graph::new("example");
//! let x = g.input("x", Shape::new(&[4, 384], DType::F32));
//! let w = g.param("w", Shape::new(&[384, 1536], DType::F32));
//! let b = g.param("b", Shape::new(&[1536], DType::F32));
//! let mm = g.mm(x, w);
//! let add = g.add(mm, b);
//! let out = g.gelu(add);
//! ```

use crate::op::*;
use crate::shape;
use crate::{DType, Graph, NodeId, Op, Shape};

/// Extension trait for shape-inferred graph building.
pub trait GraphExt {
    // ── Linear algebra ──────────────────────────────────────
    fn mm(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    // ── Binary ──────────────────────────────────────────────
    fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    // ── Activation ──────────────────────────────────────────
    fn gelu(&mut self, x: NodeId) -> NodeId;
    /// Tanh-approximation GELU (PyTorch's default `gelu` formula,
    /// also candle's `Tensor::gelu`). Use this when porting models
    /// whose reference implementations use the tanh form for
    /// numerical parity (e.g. DINOv2, many ViTs).
    fn gelu_approx(&mut self, x: NodeId) -> NodeId;
    fn silu(&mut self, x: NodeId) -> NodeId;
    fn relu(&mut self, x: NodeId) -> NodeId;
    fn exp(&mut self, x: NodeId) -> NodeId;
    fn sqrt(&mut self, x: NodeId) -> NodeId;
    fn neg(&mut self, x: NodeId) -> NodeId;
    fn tanh(&mut self, x: NodeId) -> NodeId;

    // ── Normalization ───────────────────────────────────────
    fn ln(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId;
    fn layer_norm2d(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId;
    fn group_norm(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        num_groups: usize,
        eps: f32,
    ) -> NodeId;
    fn rms_norm(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId;

    // ── Convolution (NCHW) ───────────────────────────────────
    fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> NodeId;
    fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
    ) -> NodeId;

    // ── Reduction ───────────────────────────────────────────
    fn sum(&mut self, x: NodeId, axes: Vec<usize>, keep_dim: bool) -> NodeId;
    fn mean(&mut self, x: NodeId, axes: Vec<usize>, keep_dim: bool) -> NodeId;
    fn sm(&mut self, x: NodeId, axis: i32) -> NodeId;

    // ── Shape manipulation ──────────────────────────────────
    fn reshape_(&mut self, x: NodeId, new_shape: Vec<i64>) -> NodeId;
    fn transpose_(&mut self, x: NodeId, perm: Vec<usize>) -> NodeId;
    fn narrow_(&mut self, x: NodeId, axis: usize, start: usize, len: usize) -> NodeId;
    fn concat_(&mut self, inputs: Vec<NodeId>, axis: usize) -> NodeId;
    fn gather_(&mut self, table: NodeId, indices: NodeId, axis: usize) -> NodeId;

    // ── Comparison ──────────────────────────────────────────
    fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId;

    // ── Attention ───────────────────────────────────────────
    fn attention_(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> NodeId;

    // ── RoPE ────────────────────────────────────────────────
    fn rope(&mut self, x: NodeId, cos: NodeId, sin: NodeId, head_dim: usize) -> NodeId;
    /// Partial RoPE: rotate the first `n_rot` dims (NeoX offset `n_rot/2`).
    fn rope_n(
        &mut self,
        x: NodeId,
        cos: NodeId,
        sin: NodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> NodeId;

    // ── Cast ────────────────────────────────────────────────
    fn cast(&mut self, x: NodeId, to: DType) -> NodeId;
}

impl GraphExt for Graph {
    fn mm(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s =
            shape::matmul_shape(self.shape(lhs), self.shape(rhs)).expect("matmul shape inference");
        self.matmul(lhs, rhs, s)
    }

    fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("add shape inference");
        self.binary(BinaryOp::Add, lhs, rhs, s)
    }

    fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("sub shape inference");
        self.binary(BinaryOp::Sub, lhs, rhs, s)
    }

    fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("mul shape inference");
        self.binary(BinaryOp::Mul, lhs, rhs, s)
    }

    fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::binary_shape(self.shape(lhs), self.shape(rhs)).expect("div shape inference");
        self.binary(BinaryOp::Div, lhs, rhs, s)
    }

    fn gelu(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Gelu, x, s)
    }

    fn gelu_approx(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::GeluApprox, x, s)
    }

    fn silu(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Silu, x, s)
    }

    fn relu(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Relu, x, s)
    }

    fn exp(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Exp, x, s)
    }

    fn sqrt(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Sqrt, x, s)
    }

    fn neg(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Neg, x, s)
    }

    fn tanh(&mut self, x: NodeId) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.activation(Activation::Tanh, x, s)
    }

    fn ln(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.layer_norm(x, gamma, beta, -1, eps, s)
    }

    fn layer_norm2d(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId {
        Graph::layer_norm2d(self, x, gamma, beta, eps)
    }

    fn group_norm(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        num_groups: usize,
        eps: f32,
    ) -> NodeId {
        Graph::group_norm(self, x, gamma, beta, num_groups, eps)
    }

    fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> NodeId {
        Graph::conv2d(
            self,
            input,
            weight,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
    }

    fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
    ) -> NodeId {
        Graph::conv_transpose2d(
            self,
            input,
            weight,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        )
    }

    fn rms_norm(&mut self, x: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId {
        let s = shape::unary_shape(self.shape(x));
        self.add_node(Op::RmsNorm { axis: -1, eps }, vec![x, gamma, beta], s)
    }

    fn sum(&mut self, x: NodeId, axes: Vec<usize>, keep_dim: bool) -> NodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.reduce(x, ReduceOp::Sum, axes, keep_dim, s)
    }

    fn mean(&mut self, x: NodeId, axes: Vec<usize>, keep_dim: bool) -> NodeId {
        let s =
            shape::reduce_shape(self.shape(x), &axes, keep_dim).expect("reduce shape inference");
        self.reduce(x, ReduceOp::Mean, axes, keep_dim, s)
    }

    fn sm(&mut self, x: NodeId, axis: i32) -> NodeId {
        let s = shape::softmax_shape(self.shape(x));
        self.softmax(x, axis, s)
    }

    fn reshape_(&mut self, x: NodeId, new_shape: Vec<i64>) -> NodeId {
        let s = shape::reshape_shape(self.shape(x), &new_shape).expect("reshape shape inference");
        self.reshape(x, new_shape, s)
    }

    fn transpose_(&mut self, x: NodeId, perm: Vec<usize>) -> NodeId {
        let s = shape::transpose_shape(self.shape(x), &perm).expect("transpose shape inference");
        self.add_node(Op::Transpose { perm }, vec![x], s)
    }

    fn narrow_(&mut self, x: NodeId, axis: usize, start: usize, len: usize) -> NodeId {
        let s = shape::narrow_shape(self.shape(x), axis, len).expect("narrow shape inference");
        self.add_node(Op::Narrow { axis, start, len }, vec![x], s)
    }

    fn concat_(&mut self, inputs: Vec<NodeId>, axis: usize) -> NodeId {
        let shapes: Vec<&Shape> = inputs.iter().map(|&id| self.shape(id)).collect();
        let s = shape::concat_shape(&shapes, axis).expect("concat shape inference");
        self.concat(inputs, axis, s)
    }

    fn gather_(&mut self, table: NodeId, indices: NodeId, axis: usize) -> NodeId {
        let s = shape::gather_shape(self.shape(table), self.shape(indices), axis)
            .expect("gather shape inference");
        self.gather(table, indices, axis, s)
    }

    fn eq(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::compare_shape(self.shape(lhs), self.shape(rhs))
            .expect("compare shape inference");
        self.add_node(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
    }

    fn lt(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let s = shape::compare_shape(self.shape(lhs), self.shape(rhs))
            .expect("compare shape inference");
        self.add_node(Op::Compare(CmpOp::Lt), vec![lhs, rhs], s)
    }

    fn attention_(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        num_heads: usize,
        head_dim: usize,
    ) -> NodeId {
        let s = shape::attention_shape(self.shape(q));
        self.attention(q, k, v, mask, num_heads, head_dim, s)
    }

    fn rope(&mut self, x: NodeId, cos: NodeId, sin: NodeId, head_dim: usize) -> NodeId {
        self.rope_n(x, cos, sin, head_dim, head_dim)
    }

    fn rope_n(
        &mut self,
        x: NodeId,
        cos: NodeId,
        sin: NodeId,
        head_dim: usize,
        n_rot: usize,
    ) -> NodeId {
        assert!(
            n_rot <= head_dim && n_rot.is_multiple_of(2),
            "rope_n: n_rot={n_rot} must be even and <= head_dim={head_dim}"
        );
        let s = shape::unary_shape(self.shape(x));
        self.add_node(Op::Rope { head_dim, n_rot }, vec![x, cos, sin], s)
    }

    fn cast(&mut self, x: NodeId, to: DType) -> NodeId {
        let s = shape::cast_shape(self.shape(x), to);
        self.add_node(Op::Cast { to }, vec![x], s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inferred_conv2d_and_conv_transpose2d() {
        let mut g = Graph::new("conv");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[1, 4, 8, 8], f));
        let w = g.param("w", Shape::new(&[8, 2, 3, 3], f));
        let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 2);
        assert_eq!(g.shape(y), &Shape::new(&[1, 8, 8, 8], f));

        let wt = g.param("wt", Shape::new(&[4, 8, 2, 2], f));
        let z = g.conv_transpose2d(x, wt, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1);
        assert_eq!(g.shape(z), &Shape::new(&[1, 8, 16, 16], f));
    }

    #[test]
    fn inferred_layer_norm2d() {
        let mut g = Graph::new("ln2d");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[1, 4, 8, 8], f));
        let gamma = g.param("g", Shape::new(&[4], f));
        let beta = g.param("b", Shape::new(&[4], f));
        let y = g.layer_norm2d(x, gamma, beta, 1e-6);
        assert_eq!(g.shape(y), &Shape::new(&[1, 4, 8, 8], f));
    }

    #[test]
    fn inferred_matmul_bias_gelu() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4, 15, 384], DType::F32));
        let w = g.param("w", Shape::new(&[384, 1536], DType::F32));
        let b = g.param("b", Shape::new(&[1536], DType::F32));

        // No explicit shapes needed!
        let mm = g.mm(x, w);
        let add = g.add(mm, b);
        let out = g.gelu(add);
        g.set_outputs(vec![out]);

        assert_eq!(g.shape(mm), &Shape::new(&[4, 15, 1536], DType::F32));
        assert_eq!(g.shape(add), &Shape::new(&[4, 15, 1536], DType::F32));
        assert_eq!(g.shape(out), &Shape::new(&[4, 15, 1536], DType::F32));
    }

    #[test]
    fn inferred_bert_ffn() {
        let mut g = Graph::new("bert_ffn");
        let f = DType::F32;
        let h = 384;
        let int = 1536;

        let x = g.input("x", Shape::new(&[4, 15, h], f));
        let int_w = g.param("int.w", Shape::new(&[h, int], f));
        let int_b = g.param("int.b", Shape::new(&[int], f));
        let out_w = g.param("out.w", Shape::new(&[int, h], f));
        let out_b = g.param("out.b", Shape::new(&[h], f));
        let gamma = g.param("g", Shape::new(&[h], f));
        let beta = g.param("b", Shape::new(&[h], f));

        let mm1 = g.mm(x, int_w);
        let a1 = g.add(mm1, int_b);
        let ffn = g.gelu(a1);
        let mm2 = g.mm(ffn, out_w);
        let out = g.add(mm2, out_b);
        let res = g.add(out, x);
        let normed = g.ln(res, gamma, beta, 1e-12);
        g.set_outputs(vec![normed]);

        assert_eq!(g.shape(ffn), &Shape::new(&[4, 15, int], f));
        assert_eq!(g.shape(out), &Shape::new(&[4, 15, h], f));
        assert_eq!(g.shape(normed), &Shape::new(&[4, 15, h], f));
    }

    #[test]
    fn inferred_gather_reshape() {
        let mut g = Graph::new("test");
        let table = g.param("emb", Shape::new(&[30522, 384], DType::F32));
        let ids = g.input("ids", Shape::new(&[4, 15], DType::I64));

        let gathered = g.gather_(table, ids, 0);
        assert_eq!(g.shape(gathered), &Shape::new(&[4, 15, 384], DType::F32));

        let reshaped = g.reshape_(gathered, vec![60, 384]);
        assert_eq!(g.shape(reshaped), &Shape::new(&[60, 384], DType::F32));

        let transposed = g.transpose_(reshaped, vec![1, 0]);
        assert_eq!(g.shape(transposed), &Shape::new(&[384, 60], DType::F32));
    }

    #[test]
    fn inferred_reduce_softmax() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4, 15, 384], DType::F32));

        let s = g.sm(x, -1);
        assert_eq!(g.shape(s), &Shape::new(&[4, 15, 384], DType::F32));

        let m = g.mean(x, vec![2], false);
        assert_eq!(g.shape(m), &Shape::new(&[4, 15], DType::F32));

        let mk = g.mean(x, vec![2], true);
        assert_eq!(g.shape(mk), &Shape::new(&[4, 15, 1], DType::F32));
    }
}
