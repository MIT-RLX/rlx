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

//! Shared one-op host-fallback plumbing (`Op::ScanBackward` / `ScanBackwardXs`
//! and similar nested-body ops) mirroring [`super::ScanHostDesc`].

use super::scan::{eval_single_op_f32, run_scan_packed_f32};
use rlx_ir::{Graph, NodeId, Op, Shape};

/// Compiled nested-body op + outer-arena byte layout — the fields formerly
/// duplicated as `Step::HostOp` / `Thunk::HostOp` across GPU backends.
#[derive(Clone, Debug)]
pub struct HostOpDesc {
    pub op: Op,
    pub out_byte_off: usize,
    pub out_shape: Shape,
    /// `(byte_offset, declared_shape)` per operand, in graph-input order.
    pub inputs: Vec<(usize, Shape)>,
}

/// Build a [`HostOpDesc`] from a graph node. `byte_off` returns the outer-arena
/// byte offset of a [`NodeId`] (operands and output).
pub fn host_op_desc_from_node(
    graph: &Graph,
    node: &rlx_ir::Node,
    mut byte_off: impl FnMut(NodeId) -> usize,
) -> HostOpDesc {
    HostOpDesc {
        op: node.op.clone(),
        out_byte_off: byte_off(node.id),
        out_shape: node.shape.clone(),
        inputs: node
            .inputs
            .iter()
            .map(|&id| (byte_off(id), graph.node(id).shape.clone()))
            .collect(),
    }
}

/// Run [`eval_single_op_f32`] against a raw byte arena from a [`HostOpDesc`].
///
/// # Safety
/// `base` must span every outer offset referenced by `desc` (inputs + output).
pub unsafe fn execute_host_op_on_bytes(base: *mut u8, desc: &HostOpDesc) {
    let staged: Vec<(Shape, Vec<f32>)> = desc
        .inputs
        .iter()
        .map(|(off, sh)| {
            let n = sh.num_elements().unwrap_or(0);
            let mut v = vec![0f32; n];
            unsafe {
                let src = base.add(*off) as *const f32;
                std::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), n);
            }
            (sh.clone(), v)
        })
        .collect();
    let refs: Vec<(Shape, &[f32])> = staged
        .iter()
        .map(|(sh, v)| (sh.clone(), v.as_slice()))
        .collect();
    let y = eval_single_op_f32(&desc.op, &desc.out_shape, &refs);
    unsafe {
        let dst = base.add(desc.out_byte_off) as *mut f32;
        std::ptr::copy_nonoverlapping(y.as_ptr(), dst, y.len());
    }
}

/// Evaluate `desc` against a host f32 arena (byte offsets → `/4` element indices).
/// Used by CUDA / ROCm after a full-arena D2H.
pub fn eval_host_op_on_f32_arena(host: &mut [f32], desc: &HostOpDesc) {
    debug_assert!(desc.out_byte_off.is_multiple_of(4));
    let staged: Vec<(Shape, Vec<f32>)> = desc
        .inputs
        .iter()
        .map(|(off, sh)| {
            debug_assert!(off.is_multiple_of(4));
            let off_f32 = *off / 4;
            let n = sh.num_elements().unwrap_or(0);
            let end = (off_f32 + n).min(host.len());
            (sh.clone(), host[off_f32..end].to_vec())
        })
        .collect();
    let refs: Vec<(Shape, &[f32])> = staged
        .iter()
        .map(|(sh, v)| (sh.clone(), v.as_slice()))
        .collect();
    let y = eval_single_op_f32(&desc.op, &desc.out_shape, &refs);
    let out_f32 = desc.out_byte_off / 4;
    let end = (out_f32 + y.len()).min(host.len());
    host[out_f32..end].copy_from_slice(&y[..(end - out_f32)]);
}

/// Value-map path: load `Op::Scan` inputs via `get` and run the packed host loop.
pub fn run_scan_node_f32(node: &rlx_ir::Node, mut get: impl FnMut(NodeId) -> Vec<f32>) -> Vec<f32> {
    let (body, length, save_trajectory, num_bcast, num_xs) = match &node.op {
        Op::Scan {
            body,
            length,
            save_trajectory,
            num_bcast,
            num_xs,
            ..
        } => (
            body,
            *length,
            *save_trajectory,
            *num_bcast as usize,
            *num_xs as usize,
        ),
        _ => panic!("run_scan_node_f32: expected Op::Scan"),
    };
    let init = get(node.inputs[0]);
    let mut bcasts = Vec::with_capacity(num_bcast);
    for i in 0..num_bcast {
        bcasts.push(get(node.inputs[1 + i]));
    }
    let mut xs = Vec::with_capacity(num_xs);
    for i in 0..num_xs {
        xs.push(get(node.inputs[1 + num_bcast + i]));
    }
    let out_len = node.shape.num_elements().unwrap_or(0);
    run_scan_packed_f32(
        body,
        length,
        save_trajectory,
        num_bcast,
        num_xs,
        &init,
        &bcasts,
        &xs,
        out_len,
    )
}

/// Value-map path: load inputs via `get` and evaluate a nested-body op
/// (`ScanBackward` / `ScanBackwardXs`, …) with [`eval_single_op_f32`].
pub fn run_host_op_node_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    mut get: impl FnMut(NodeId) -> Vec<f32>,
) -> Vec<f32> {
    let staged: Vec<(Shape, Vec<f32>)> = node
        .inputs
        .iter()
        .map(|&id| (graph.node(id).shape.clone(), get(id)))
        .collect();
    let refs: Vec<(Shape, &[f32])> = staged
        .iter()
        .map(|(sh, v)| (sh.clone(), v.as_slice()))
        .collect();
    eval_single_op_f32(&node.op, &node.shape, &refs)
}

/// Build a [`HostOpDesc`] from `(graph, node, |nid| byte_offset)`.
#[macro_export]
macro_rules! rlx_host_op_desc {
    ($graph:expr, $node:expr, $byte_off:expr) => {
        $crate::thunk::host_op_desc_from_node(&$graph, $node, $byte_off)
    };
}

/// Execute a host op against a raw byte arena from a [`HostOpDesc`].
#[macro_export]
macro_rules! rlx_execute_host_op_on_bytes {
    ($base:expr, $desc:expr) => {
        $crate::thunk::execute_host_op_on_bytes($base, $desc)
    };
}

/// D2H → CPU host-op → H2D staging for discrete-GPU f32 arenas.
///
/// Byte offsets inside `$desc` are converted via `/4` in
/// [`eval_host_op_on_f32_arena`].
#[macro_export]
macro_rules! rlx_host_op_stage_d2h {
    (
        arena_size_bytes = $nbytes:expr,
        desc = $desc:expr,
        sync = $sync:block,
        dtoh = |$host:ident| $dtoh:block,
        htod = |$host_back:ident| $htod:block $(,)?
    ) => {
        $crate::rlx_arena_stage_d2h! {
            arena_size_bytes = $nbytes,
            sync = $sync,
            dtoh = |$host| $dtoh,
            on_host = |host_mut| {
                $crate::thunk::eval_host_op_on_f32_arena(host_mut, $desc);
            },
            htod = |$host_back| $htod,
        }
    };
}

/// Contiguous host span covering every region a [`HostOpDesc`] touches, with
/// offsets rebased to the span start (wgpu readback mirror of [`super::ScanHostSpan`]).
#[derive(Clone)]
pub struct HostOpSpan {
    pub lo: usize,
    pub hi: usize,
    pub desc: HostOpDesc,
}

impl HostOpSpan {
    pub fn from_desc(desc: HostOpDesc) -> Self {
        let out_n = desc.out_shape.num_elements().unwrap_or(0) * 4;
        let mut lo = desc.out_byte_off;
        let mut hi = desc.out_byte_off + out_n;
        for (off, sh) in &desc.inputs {
            let n = sh.num_elements().unwrap_or(0) * 4;
            lo = lo.min(*off);
            hi = hi.max(*off + n);
        }
        let rebated = HostOpDesc {
            op: desc.op,
            out_byte_off: desc.out_byte_off - lo,
            out_shape: desc.out_shape,
            inputs: desc
                .inputs
                .into_iter()
                .map(|(o, sh)| (o - lo, sh))
                .collect(),
        };
        Self {
            lo,
            hi,
            desc: rebated,
        }
    }

    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
