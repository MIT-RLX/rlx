// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NCHW pooling builder (`Op::Pool`).

use crate::op::ReduceOp;
use crate::{Graph, NodeId, Op};

impl Graph {
    /// 2-D pooling over NCHW (`Op::Pool`). Channels pass through; only the two
    /// spatial axes shrink.
    ///
    /// `kind` selects the reduction — [`ReduceOp::Max`] and [`ReduceOp::Mean`]
    /// are the ones backends lower.
    pub fn pool2d(
        &mut self,
        input: NodeId,
        kind: ReduceOp,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let out = crate::shape::pool2d_output_shape(&in_s, kernel_size, stride, padding)
            .expect("pool2d shape inference");
        self.push(
            Op::Pool {
                kind,
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
            },
            vec![input],
            out,
            None,
        )
    }

    /// Max pooling over NCHW — [`Graph::pool2d`] with [`ReduceOp::Max`].
    pub fn max_pool2d(
        &mut self,
        input: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> NodeId {
        self.pool2d(input, ReduceOp::Max, kernel_size, stride, padding)
    }

    /// Average pooling over NCHW — [`Graph::pool2d`] with [`ReduceOp::Mean`].
    pub fn avg_pool2d(
        &mut self,
        input: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> NodeId {
        self.pool2d(input, ReduceOp::Mean, kernel_size, stride, padding)
    }
}
