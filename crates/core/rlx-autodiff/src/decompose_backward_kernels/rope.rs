// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

//! `rope` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// RoPE backward = forward RoPE with negated sin table (NeoX).
pub fn compose_rope_backward(
    g: &mut Graph,
    dy: NodeId,
    cos: NodeId,
    sin: NodeId,
    head_dim: usize,
    n_rot: usize,
) -> NodeId {
    let sin_shape = g.node(sin).shape.clone();
    let neg = scalar_const(-1.0, &sin_shape, g);
    let neg_sin = g.mul(sin, neg);
    g.rope_n(dy, cos, neg_sin, head_dim, n_rot)
}
