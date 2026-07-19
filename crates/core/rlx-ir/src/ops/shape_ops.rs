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

//! Shape-manipulation builders: reshape, gather, concat
//! (plan #53). Other shape ops (narrow, transpose, expand) live
//! on `GraphExt` in `infer.rs` since they need shape inference.

use crate::op::ScatterNdReduction;
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Reshape.
    pub fn reshape(&mut self, input: NodeId, new_shape: Vec<i64>, out_shape: Shape) -> NodeId {
        self.push(Op::Reshape { new_shape }, vec![input], out_shape, None)
    }

    /// Gather (embedding lookup).
    pub fn gather(&mut self, table: NodeId, indices: NodeId, axis: usize, shape: Shape) -> NodeId {
        self.push(Op::Gather { axis }, vec![table, indices], shape, None)
    }

    /// ONNX ScatterND: copy `data`, write `updates` at multi-index locations.
    pub fn scatter_nd(
        &mut self,
        data: NodeId,
        indices: NodeId,
        updates: NodeId,
        reduction: ScatterNdReduction,
    ) -> NodeId {
        let shape = self.node(data).shape.clone();
        self.push(
            Op::ScatterNd { reduction },
            vec![data, indices, updates],
            shape,
            None,
        )
    }

    /// ONNX ScatterElements along `axis`.
    pub fn scatter_elements(
        &mut self,
        data: NodeId,
        indices: NodeId,
        updates: NodeId,
        axis: i32,
        reduction: ScatterNdReduction,
    ) -> NodeId {
        let shape = self.node(data).shape.clone();
        self.push(
            Op::ScatterElements { axis, reduction },
            vec![data, indices, updates],
            shape,
            None,
        )
    }

    /// ONNX GatherND.
    pub fn gather_nd(
        &mut self,
        data: NodeId,
        indices: NodeId,
        batch_dims: i32,
        out_shape: Shape,
    ) -> NodeId {
        self.push(
            Op::GatherNd { batch_dims },
            vec![data, indices],
            out_shape,
            None,
        )
    }

    /// ONNX GatherElements / take_along_axis — output shape = indices shape.
    pub fn gather_elements(&mut self, data: NodeId, indices: NodeId, axis: i32) -> NodeId {
        let shape = self.node(indices).shape.clone();
        self.push(
            Op::GatherElements { axis },
            vec![data, indices],
            shape,
            None,
        )
    }

    /// Concatenate tensors along an axis.
    pub fn concat(&mut self, inputs: Vec<NodeId>, axis: usize, shape: Shape) -> NodeId {
        self.push(Op::Concat { axis }, inputs, shape, None)
    }

    /// Reverse (flip) element order along each axis in `axes`. Output shape is
    /// the same as the input; only the listed axes flip (batch-general).
    pub fn reverse(&mut self, input: NodeId, axes: Vec<usize>) -> NodeId {
        let shape = self.node(input).shape.clone();
        self.push(Op::Reverse { axes }, vec![input], shape, None)
    }
}
