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

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

//! `scan` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// Unrolled `ScanBackward` for static length (trajectory-cached scans).
pub fn compose_scan_backward(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    out_shape: &Shape,
) -> NodeId {
    let l = length as usize;
    assert!(
        l > 0 && l <= SCAN_DECOMPOSE_MAX_LENGTH as usize,
        "compose_scan_backward: length 1..={SCAN_DECOMPOSE_MAX_LENGTH}"
    );
    assert!(
        save_trajectory,
        "compose_scan_backward: save_trajectory=true only"
    );
    let dcarry = run_scan_backward_steps(
        g,
        init,
        trajectory,
        upstream,
        xs,
        body_vjp,
        forward_body,
        length,
        save_trajectory,
        num_checkpoints,
        out_shape,
        None,
    )
    .0;
    finalize_scan_backward_carry(g, dcarry, out_shape)
}


/// Unrolled `ScanBackwardXs` — stacks per-step `dxs` along axis 0.
pub fn compose_scan_backward_xs(
    g: &mut Graph,
    init: NodeId,
    trajectory: NodeId,
    upstream: NodeId,
    xs: &[NodeId],
    body_vjp: &Graph,
    forward_body: Option<&Graph>,
    length: u32,
    save_trajectory: bool,
    num_checkpoints: u32,
    xs_idx: u32,
    out_shape: &Shape,
) -> NodeId {
    let l = length as usize;
    assert!(
        l > 0 && l <= SCAN_DECOMPOSE_MAX_LENGTH as usize,
        "compose_scan_backward_xs: length 1..={SCAN_DECOMPOSE_MAX_LENGTH}"
    );
    assert!(
        save_trajectory,
        "compose_scan_backward_xs: save_trajectory=true only"
    );
    let out_idx = 1 + xs_idx as usize;
    assert!(
        out_idx < body_vjp.outputs.len(),
        "compose_scan_backward_xs: xs_idx out of range"
    );
    let carry_shape = g.node(trajectory).shape.clone();
    let mut carry_step_dims: Vec<usize> = carry_shape
        .dims()
        .iter()
        .skip(1)
        .map(|d| d.unwrap_static())
        .collect();
    if carry_step_dims.is_empty() {
        carry_step_dims.push(1);
    }
    let carry_step = Shape::new(&carry_step_dims, carry_shape.dtype());
    let (dcarry, dx_steps) = run_scan_backward_steps(
        g,
        init,
        trajectory,
        upstream,
        xs,
        body_vjp,
        forward_body,
        length,
        save_trajectory,
        num_checkpoints,
        &carry_step,
        Some(out_idx),
    );
    let mut dx_steps = dx_steps.expect("dx steps");
    dx_steps.reverse();
    let stacked = g.concat_(dx_steps, 0);
    reconcile_node_shape(g, stacked);
    let _ = (dcarry, out_shape);
    stacked
}

