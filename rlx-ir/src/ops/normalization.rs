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

//! Normalization builders: LayerNorm, fused residual+LN (plan #53).

use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// LayerNorm2d on NCHW (normalize across channels at each spatial position).
    pub fn layer_norm2d(&mut self, input: NodeId, gamma: NodeId, beta: NodeId, eps: f32) -> NodeId {
        let shape = self.node(input).shape.clone();
        self.push(
            Op::LayerNorm2d { eps },
            vec![input, gamma, beta],
            shape,
            None,
        )
    }

    /// Group normalization on NCHW.
    pub fn group_norm(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        num_groups: usize,
        eps: f32,
    ) -> NodeId {
        let shape = self.node(input).shape.clone();
        self.push(
            Op::GroupNorm { num_groups, eps },
            vec![input, gamma, beta],
            shape,
            None,
        )
    }

    /// Layer normalization.
    pub fn layer_norm(
        &mut self,
        input: NodeId,
        gamma: NodeId,
        beta: NodeId,
        axis: i32,
        eps: f32,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::LayerNorm { axis, eps },
            vec![input, gamma, beta],
            shape,
            None,
        )
    }

    /// Fused residual + bias + layer norm (created by optimization passes).
    pub fn fused_residual_ln(
        &mut self,
        x: NodeId,
        residual: NodeId,
        bias: Option<NodeId>,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        shape: Shape,
    ) -> NodeId {
        let has_bias = bias.is_some();
        let mut inputs = vec![x, residual];
        if let Some(b) = bias {
            inputs.push(b);
        }
        inputs.push(gamma);
        inputs.push(beta);
        self.push(Op::FusedResidualLN { has_bias, eps }, inputs, shape, None)
    }

    /// Fused residual + bias + RMS norm (created by optimization passes).
    pub fn fused_residual_rms_norm(
        &mut self,
        x: NodeId,
        residual: NodeId,
        bias: Option<NodeId>,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        shape: Shape,
    ) -> NodeId {
        let has_bias = bias.is_some();
        let mut inputs = vec![x, residual];
        if let Some(b) = bias {
            inputs.push(b);
        }
        inputs.push(gamma);
        inputs.push(beta);
        self.push(
            Op::FusedResidualRmsNorm { has_bias, eps },
            inputs,
            shape,
            None,
        )
    }
}
