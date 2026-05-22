// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

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
