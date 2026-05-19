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

    /// Cumulative sum along an axis (output shape == input shape).
    pub fn cumsum(&mut self, input: NodeId, axis: i32, exclusive: bool, shape: Shape) -> NodeId {
        self.push(Op::Cumsum { axis, exclusive }, vec![input], shape, None)
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
