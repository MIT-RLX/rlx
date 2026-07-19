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

//! Shared `Op::Scan` host-fallback plumbing for GPU backends.
//!
//! Backends compile a scan node into a [`ScanHostDesc`] once (via
//! [`scan_host_desc_from_node`] / [`rlx_scan_host_desc`]), then at run time
//! stage arena bytes (unified memory, full D2H, or a rebated span) and call
//! [`execute_scan_host_desc`]. Short Scans are preferentially unrolled by
//! [`maybe_unroll_scans`] / [`rlx_maybe_unroll_scans`] so the body runs as
//! ordinary device ops.

use crate::thunk::{ScanBodyPlan, compile_scan_body, execute_scan_host};
use rlx_ir::{Graph, NodeId, Op};
use std::sync::Arc;

/// Compiled scan + outer-arena layout — the fields formerly duplicated as
/// `Step::ScanHost` / `Thunk::ScanHost` across Metal / CUDA / ROCm / wgpu.
#[derive(Clone)]
pub struct ScanHostDesc {
    pub plan: Arc<ScanBodyPlan>,
    pub outer_init_off: usize,
    pub outer_final_off: usize,
    pub length: u32,
    pub save_trajectory: bool,
    /// `(outer_off, per_step_bytes)` for each per-step `xs` input.
    pub xs_outer: Vec<(usize, usize)>,
    /// `(outer_off, total_bytes)` for each broadcast input.
    pub bcast_outer: Vec<(usize, usize)>,
}

impl std::fmt::Debug for ScanHostDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanHostDesc")
            .field("carry_bytes", &self.plan.carry_bytes)
            .field("length", &self.length)
            .field("save_trajectory", &self.save_trajectory)
            .field("num_xs", &self.xs_outer.len())
            .field("num_bcast", &self.bcast_outer.len())
            .finish()
    }
}

/// Build a [`ScanHostDesc`] from an `Op::Scan` node. `byte_off` returns the
/// outer-arena byte offset of a [`NodeId`] (init / bcast / xs / output).
pub fn scan_host_desc_from_node(
    graph: &Graph,
    node: &rlx_ir::Node,
    mut byte_off: impl FnMut(NodeId) -> usize,
) -> ScanHostDesc {
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
        _ => panic!("scan_host_desc_from_node: expected Op::Scan"),
    };
    let plan = compile_scan_body(body, num_bcast, num_xs);
    let bcast_outer: Vec<(usize, usize)> = (0..num_bcast)
        .map(|i| {
            let id = node.inputs[1 + i];
            (byte_off(id), graph.node(id).shape.size_bytes().unwrap())
        })
        .collect();
    let xs_outer: Vec<(usize, usize)> = (0..num_xs)
        .map(|i| {
            let id = node.inputs[1 + num_bcast + i];
            let total = graph.node(id).shape.size_bytes().unwrap();
            (byte_off(id), total / length as usize)
        })
        .collect();
    ScanHostDesc {
        plan: Arc::new(plan),
        outer_init_off: byte_off(node.inputs[0]),
        outer_final_off: byte_off(node.id),
        length,
        save_trajectory,
        xs_outer,
        bcast_outer,
    }
}

/// Run [`execute_scan_host`] from a [`ScanHostDesc`].
///
/// # Safety
/// `base` must span every outer offset referenced by `desc`.
pub unsafe fn execute_scan_host_desc(base: *mut u8, desc: &ScanHostDesc) {
    unsafe {
        execute_scan_host(
            base,
            &desc.plan,
            desc.outer_init_off,
            desc.outer_final_off,
            desc.length,
            desc.save_trajectory,
            &desc.xs_outer,
            &desc.bcast_outer,
        );
    }
}

/// Contiguous host span covering every region a scan touches, with offsets
/// rebased to the span start. Used by wgpu (no host-mapped arena).
#[derive(Clone)]
pub struct ScanHostSpan {
    pub lo: usize,
    pub hi: usize,
    pub desc: ScanHostDesc,
}

impl ScanHostSpan {
    pub fn from_desc(desc: ScanHostDesc) -> Self {
        let cb = desc.plan.carry_bytes;
        let out_len = if desc.save_trajectory {
            desc.length as usize * cb
        } else {
            cb
        };
        let mut lo = desc.outer_init_off.min(desc.outer_final_off);
        let mut hi = (desc.outer_init_off + cb).max(desc.outer_final_off + out_len);
        for &(o, t) in &desc.bcast_outer {
            lo = lo.min(o);
            hi = hi.max(o + t);
        }
        for &(o, ps) in &desc.xs_outer {
            lo = lo.min(o);
            hi = hi.max(o + desc.length as usize * ps);
        }
        let rebated = ScanHostDesc {
            plan: desc.plan,
            outer_init_off: desc.outer_init_off - lo,
            outer_final_off: desc.outer_final_off - lo,
            length: desc.length,
            save_trajectory: desc.save_trajectory,
            xs_outer: desc.xs_outer.iter().map(|&(o, ps)| (o - lo, ps)).collect(),
            bcast_outer: desc.bcast_outer.iter().map(|&(o, t)| (o - lo, t)).collect(),
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

/// Evaluate a single op (including nested-body ops like `Op::ScanBackward`)
/// via a one-op CPU graph + thunk executor. Used by GPU host-fallbacks for
/// ops that are awkward to drive as raw `ScanHostDesc` layout.
pub fn eval_single_op_f32(
    op: &Op,
    out_shape: &rlx_ir::Shape,
    inputs: &[(rlx_ir::Shape, &[f32])],
) -> Vec<f32> {
    let mut g = Graph::new("scan_eval_single_op");
    let ids: Vec<NodeId> = inputs
        .iter()
        .enumerate()
        .map(|(i, (sh, _))| {
            g.append_node(
                Op::Input {
                    name: format!("in{i}"),
                },
                vec![],
                sh.clone(),
                None,
            )
        })
        .collect();
    let out = g.append_node(op.clone(), ids.clone(), out_shape.clone(), None);
    g.set_outputs(vec![out]);
    let plan = rlx_opt::memory::plan_memory_aligned(&g, 16);
    let mut arena = crate::arena::Arena::from_plan(plan);
    for (i, (_, vals)) in inputs.iter().enumerate() {
        let slot = arena.slice_mut(ids[i]);
        let n = slot.len().min(vals.len());
        slot[..n].copy_from_slice(&vals[..n]);
    }
    let schedule = crate::thunk::compile_thunks(&g, &arena);
    crate::thunk::execute_thunks(&schedule, arena.raw_buf_mut());
    let n = out_shape.num_elements().unwrap_or(0);
    arena.slice_mut(out)[..n].to_vec()
}

/// Selectively unroll short Scans (see [`rlx_opt::control_flow::maybe_unroll_scans`]).
pub fn maybe_unroll_scans(graph: Graph, max_length: u32) -> Graph {
    rlx_opt::control_flow::maybe_unroll_scans(graph, max_length)
}

/// Run `Op::Scan` against packed f32 host buffers (value-map / CoreML /
/// OneAPI paths). Packs `[init, bcasts…, xs…]` into a contiguous byte arena,
/// executes [`execute_scan_host`], and returns the output floats.
#[allow(clippy::too_many_arguments)]
pub fn run_scan_packed_f32(
    body: &Graph,
    length: u32,
    save_trajectory: bool,
    num_bcast: usize,
    num_xs: usize,
    init: &[f32],
    bcasts: &[Vec<f32>],
    xs: &[Vec<f32>],
    out_elems: usize,
) -> Vec<f32> {
    assert_eq!(bcasts.len(), num_bcast);
    assert_eq!(xs.len(), num_xs);
    let plan = compile_scan_body(body, num_bcast, num_xs);
    let mut arena: Vec<u8> = Vec::new();
    let init_off = arena.len();
    arena.extend(init.iter().flat_map(|f| f.to_le_bytes()));
    let mut bcast_outer: Vec<(usize, usize)> = Vec::with_capacity(num_bcast);
    for v in bcasts {
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        bcast_outer.push((off, v.len() * 4));
    }
    let mut xs_outer: Vec<(usize, usize)> = Vec::with_capacity(num_xs);
    for v in xs {
        let total = v.len() * 4;
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        xs_outer.push((off, total / length as usize));
    }
    let final_off = arena.len();
    arena.resize(final_off + out_elems * 4, 0);
    let desc = ScanHostDesc {
        plan: Arc::new(plan),
        outer_init_off: init_off,
        outer_final_off: final_off,
        length,
        save_trajectory,
        xs_outer,
        bcast_outer,
    };
    unsafe {
        execute_scan_host_desc(arena.as_mut_ptr(), &desc);
    }
    arena[final_off..final_off + out_elems * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build a [`ScanHostDesc`] from `(graph, node, |nid| byte_offset)`.
#[macro_export]
macro_rules! rlx_scan_host_desc {
    ($graph:expr, $node:expr, $byte_off:expr) => {
        $crate::thunk::scan_host_desc_from_node(&$graph, $node, $byte_off)
    };
}

/// Execute a scan against a raw byte arena from a [`ScanHostDesc`].
#[macro_export]
macro_rules! rlx_execute_scan_on_bytes {
    ($base:expr, $desc:expr) => {
        $crate::thunk::execute_scan_host_desc($base, $desc)
    };
}

/// Generic D2H → host-body → H2D staging for discrete-GPU f32 arenas.
///
/// Shared by Scan / HostOp / Spd host-fallbacks. `$on_host` receives
/// `&mut [f32]` covering the full arena (`arena_size_bytes / 4` elems).
#[macro_export]
macro_rules! rlx_arena_stage_d2h {
    (
        arena_size_bytes = $nbytes:expr,
        sync = $sync:block,
        dtoh = |$host:ident| $dtoh:block,
        on_host = |$host_mut:ident| $on_host:block,
        htod = |$host_back:ident| $htod:block $(,)?
    ) => {{
        let n_f32 = ($nbytes) / 4;
        $sync
        let mut $host = vec![0f32; n_f32];
        $dtoh
        {
            let $host_mut: &mut [f32] = &mut $host[..];
            $on_host
        }
        let $host_back = $host;
        $htod
    }};
}

/// D2H → CPU scan → H2D staging for discrete-GPU arenas.
#[macro_export]
macro_rules! rlx_scan_stage_d2h {
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
                unsafe {
                    $crate::thunk::execute_scan_host_desc(
                        host_mut.as_mut_ptr() as *mut u8,
                        $desc,
                    );
                }
            },
            htod = |$host_back| $htod,
        }
    };
}

/// Prefer short-Scan IR unroll; leave long / checkpointed Scans intact.
#[macro_export]
macro_rules! rlx_maybe_unroll_scans {
    ($graph:expr, $max_length:expr) => {
        $crate::thunk::maybe_unroll_scans($graph, $max_length)
    };
}
