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

    /// Concatenate tensors along an axis.
    pub fn concat(&mut self, inputs: Vec<NodeId>, axis: usize, shape: Shape) -> NodeId {
        self.push(Op::Concat { axis }, inputs, shape, None)
    }
}
