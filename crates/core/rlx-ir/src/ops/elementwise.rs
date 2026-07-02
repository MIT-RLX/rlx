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

//! Element-wise builders: binary ops, activations (plan #53).

use crate::op::{Activation, BinaryOp};
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Binary element-wise operation.
    pub fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId, out_shape: Shape) -> NodeId {
        self.push(Op::Binary(op), vec![lhs, rhs], out_shape, None)
    }

    /// Unary activation.
    pub fn activation(&mut self, act: Activation, input: NodeId, shape: Shape) -> NodeId {
        self.push(Op::Activation(act), vec![input], shape, None)
    }

    /// Per-tensor INT8 quantization. Output dtype = `I8`, same shape
    /// otherwise. `scale` and `zero_point` apply uniformly to every
    /// element. Use `quantize_per_channel` when weights deserve
    /// per-channel scales (the standard PTQ improvement).
    pub fn quantize(&mut self, x: NodeId, scale: f32, zero_point: i32) -> NodeId {
        let shape = self.shape(x).clone().with_dtype(crate::DType::I8);
        self.push(
            Op::Quantize {
                axis: None,
                scales: vec![scale],
                zero_points: vec![zero_point],
            },
            vec![x],
            shape,
            None,
        )
    }

    /// Per-channel INT8 quantization. `scales` and `zero_points` must
    /// each have length `input.dim(axis)`; the kernel picks the i-th
    /// pair when quantizing the i-th slice along `axis`. The most
    /// common usage is `axis = 0` for a `[C_out, C_in, kH, kW]`
    /// conv weight (one scale per output channel).
    pub fn quantize_per_channel(
        &mut self,
        x: NodeId,
        axis: usize,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    ) -> NodeId {
        debug_assert_eq!(scales.len(), zero_points.len());
        let shape = self.shape(x).clone().with_dtype(crate::DType::I8);
        debug_assert_eq!(
            shape.dim(axis),
            crate::shape::Dim::Static(scales.len()),
            "quantize_per_channel: scales.len() must match input.dim(axis)"
        );
        self.push(
            Op::Quantize {
                axis: Some(axis),
                scales,
                zero_points,
            },
            vec![x],
            shape,
            None,
        )
    }

    /// Per-tensor INT8 dequantization (inverse of `quantize`). Output
    /// dtype is f32.
    pub fn dequantize(&mut self, x: NodeId, scale: f32, zero_point: i32) -> NodeId {
        let shape = self.shape(x).clone().with_dtype(crate::DType::F32);
        self.push(
            Op::Dequantize {
                axis: None,
                scales: vec![scale],
                zero_points: vec![zero_point],
            },
            vec![x],
            shape,
            None,
        )
    }

    /// Per-channel INT8 dequantization (inverse of
    /// `quantize_per_channel`).
    pub fn dequantize_per_channel(
        &mut self,
        x: NodeId,
        axis: usize,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    ) -> NodeId {
        debug_assert_eq!(scales.len(), zero_points.len());
        let shape = self.shape(x).clone().with_dtype(crate::DType::F32);
        debug_assert_eq!(
            shape.dim(axis),
            crate::shape::Dim::Static(scales.len()),
            "dequantize_per_channel: scales.len() must match input.dim(axis)"
        );
        self.push(
            Op::Dequantize {
                axis: Some(axis),
                scales,
                zero_points,
            },
            vec![x],
            shape,
            None,
        )
    }
}
