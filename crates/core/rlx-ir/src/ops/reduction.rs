// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduction builders: reduce, softmax, cumsum, sample
//! (plan #53).

use crate::op::ReduceOp;
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Reduce.
    pub fn reduce(
        &mut self,
        input: NodeId,
        op: ReduceOp,
        axes: Vec<usize>,
        keep_dim: bool,
        shape: Shape,
    ) -> NodeId {
        self.push(Op::Reduce { op, axes, keep_dim }, vec![input], shape, None)
    }

    /// Softmax.
    pub fn softmax(&mut self, input: NodeId, axis: i32, shape: Shape) -> NodeId {
        self.push(Op::Softmax { axis }, vec![input], shape, None)
    }

    /// Histogram of `input` into `bins` equal-width buckets over `[min, max]`.
    /// Output is a 1-D f32 tensor `[bins]` of counts. Elements outside the
    /// range are dropped and `x == max` lands in the last bin (matches
    /// `numpy.histogram`). Non-differentiable.
    pub fn histogram(&mut self, input: NodeId, bins: usize, min: f32, max: f32) -> NodeId {
        let out = Shape::new(&[bins], crate::DType::F32);
        self.push(Op::Histogram { bins, min, max }, vec![input], out, None)
    }

    /// Cumulative sum along an axis (output shape == input shape).
    pub fn cumsum(&mut self, input: NodeId, axis: i32, exclusive: bool, shape: Shape) -> NodeId {
        self.push(Op::Cumsum { axis, exclusive }, vec![input], shape, None)
    }

    /// Cumulative product along an axis (output shape == input shape).
    pub fn cumprod(&mut self, input: NodeId, axis: i32, exclusive: bool, shape: Shape) -> NodeId {
        self.push(Op::CumProd { axis, exclusive }, vec![input], shape, None)
    }

    /// Cumulative maximum along an axis (output shape == input shape).
    pub fn cummax(&mut self, input: NodeId, axis: i32, exclusive: bool, shape: Shape) -> NodeId {
        self.push(Op::CumMax { axis, exclusive }, vec![input], shape, None)
    }

    /// Index of the max along `axis` (f32-encoded indices).
    pub fn argmax(&mut self, input: NodeId, axis: usize, keep_dim: bool, shape: Shape) -> NodeId {
        self.push(Op::ArgMax { axis, keep_dim }, vec![input], shape, None)
    }

    /// Index of the min along `axis` (f32-encoded indices).
    pub fn argmin(&mut self, input: NodeId, axis: usize, keep_dim: bool, shape: Shape) -> NodeId {
        self.push(Op::ArgMin { axis, keep_dim }, vec![input], shape, None)
    }

    /// Fused sample: logits → token id (one f32-encoded id per row).
    /// `output_shape` should be `[batch]` (one id per logit row).
    pub fn sample(
        &mut self,
        logits: NodeId,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
        output_shape: Shape,
    ) -> NodeId {
        self.push(
            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            },
            vec![logits],
            output_shape,
            None,
        )
    }
}
