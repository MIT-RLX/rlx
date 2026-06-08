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

//! Tensor handle flowing through block stages — wraps internal HIR node id.

use rlx_ir::{HirNodeId, Shape};

/// Output of a block stage. Model authors see shape + opaque id only.
#[derive(Debug, Clone)]
pub struct FlowValue {
    pub(crate) id: HirNodeId,
    pub shape: Shape,
}

impl FlowValue {
    pub fn new(id: HirNodeId, shape: Shape) -> Self {
        Self { id, shape }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Tier-2 escape: read internal node id (prefer new blocks over this).
    pub fn hir_id(&self) -> HirNodeId {
        self.id
    }
}
