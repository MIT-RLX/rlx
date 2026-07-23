// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Single-op CPU reference evaluation via `rlx-cpu`'s thunk executor (the same
//! kernels the CPU backend uses, so results are bit-for-bit the reference).
//!
//! Two callers:
//! - On a host with no Level Zero device (the macOS dev box / CI), `backend.rs`
//!   walks the whole legalized graph through this evaluator — every op is served
//!   by the CPU reference, so the backend is fully correct without Intel HW.
//! - On Intel hardware, ops with no native SPIR-V kernel yet route here (read
//!   from the USM arena, eval, write back) — exactly rlx-vulkan's host-fallback.

use rlx_ir::{DType, Graph, Op, Shape};

/// One host-eval input: f32 activations, or raw bytes for a packed / mask
/// buffer (`U8`/`I8`/`Bool`, e.g. `Compare` predicates feeding `Where`).
pub enum HostBuf {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

/// A host-eval op's output in its native dtype.
pub enum HostOut {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

/// Run a single op on the CPU reference and return its output in its native
/// dtype. `inputs[i]` is `(declared_shape, buffer)`.
///
/// `FusedConvBiasAct` / `PartitionedConv` are CPU Nops — expand to primitives
/// first (same as `unfuse::expand_cpu_nop_fused`) so the host path stays correct
/// when the native fused kernel is unavailable.
pub fn eval(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
    if matches!(op, Op::FusedConvBiasAct { .. } | Op::PartitionedConv { .. }) {
        return eval_expanded_fused(op, out_shape, inputs);
    }
    eval_direct(op, out_shape, inputs)
}

fn eval_expanded_fused(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
    let mut mini = Graph::new("oneapi_host_unfuse");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, (sh, _)) in inputs.iter().enumerate() {
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh.clone(),
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, out_shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    // Evaluate the expanded graph end-to-end on CPU with the same inputs.
    let plan = rlx_compile::memory::plan_memory_aligned(&expanded, 16);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op {
            if let Some(rest) = name.strip_prefix("in") {
                if let Ok(i) = rest.parse::<usize>() {
                    match &inputs[i].1 {
                        HostBuf::F32(vals) => {
                            let slot = arena.slice_mut(n.id);
                            let take = slot.len().min(vals.len());
                            slot[..take].copy_from_slice(&vals[..take]);
                        }
                        HostBuf::Bytes(bytes) => {
                            let off = arena.byte_offset(n.id);
                            let raw = arena.raw_buf_mut();
                            let take = bytes.len().min(raw.len().saturating_sub(off));
                            raw[off..off + take].copy_from_slice(&bytes[..take]);
                        }
                    }
                }
            }
        }
    }
    let schedule = rlx_cpu::thunk::compile_thunks(&expanded, &arena);
    rlx_cpu::thunk::execute_thunks(&schedule, arena.raw_buf_mut());
    let out = expanded.outputs[0];
    let n = out_shape.num_elements().unwrap_or(0);
    match out_shape.dtype() {
        DType::U8 | DType::I8 | DType::Bool => {
            let nbytes = n * out_shape.dtype().size_bytes().max(1);
            let off = arena.byte_offset(out);
            HostOut::Bytes(arena.raw_buf()[off..off + nbytes].to_vec())
        }
        _ => {
            let slot = arena.slice_mut(out);
            let take = n.min(slot.len());
            HostOut::F32(slot[..take].to_vec())
        }
    }
}

fn eval_direct(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
    let mut g = Graph::new("oneapi_host_eval");
    let ids: Vec<rlx_ir::NodeId> = inputs
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

    let plan = rlx_compile::memory::plan_memory_aligned(&g, 16);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);

    for (i, (sh, buf)) in inputs.iter().enumerate() {
        match buf {
            // I64 inputs arrive as ONE f32 lane per element (the f32-uniform
            // arena widens every integer to a single f32 lane, value-cast). But
            // rlx-cpu stores I64 NATIVELY as 8 bytes/elem and reads it back as
            // i64 (e.g. `exec_gather`'s `idx_i64` path takes `sl_i64`), so
            // copying the f32 lanes through the f32 `slice_mut` view leaves the
            // 8-byte slots half-filled with the wrong bit pattern — an i64
            // Gather then reads garbage rows (indices out of range → zeros).
            // Narrow each lane back to a native little-endian i64 first. (Other
            // integer dtypes rlx-cpu reads as f32 lanes, e.g. `exec_gather`'s
            // non-`idx_i64` branch, so they keep the plain f32 copy below.)
            HostBuf::F32(vals) if sh.dtype() == DType::I64 => {
                let off = arena.byte_offset(ids[i]);
                let raw = arena.raw_buf_mut();
                let cap = raw.len().saturating_sub(off) / 8;
                for (k, &v) in vals.iter().take(cap).enumerate() {
                    raw[off + k * 8..off + k * 8 + 8].copy_from_slice(&(v as i64).to_le_bytes());
                }
            }
            HostBuf::F32(vals) => {
                let slot = arena.slice_mut(ids[i]);
                let n = slot.len().min(vals.len());
                slot[..n].copy_from_slice(&vals[..n]);
            }
            HostBuf::Bytes(bytes) => {
                let off = arena.byte_offset(ids[i]);
                let raw = arena.raw_buf_mut();
                let n = bytes.len().min(raw.len().saturating_sub(off));
                raw[off..off + n].copy_from_slice(&bytes[..n]);
            }
        }
    }

    let schedule = rlx_cpu::thunk::compile_thunks(&g, &arena);
    rlx_cpu::thunk::execute_thunks(&schedule, arena.raw_buf_mut());

    let n = out_shape.num_elements().unwrap_or(0);
    match out_shape.dtype() {
        // Packed / mask outputs must not be reinterpreted as f32 — Compare→Bool
        // allocates 1 byte/elem, so `slice_mut` would be 4× too short.
        DType::U8 | DType::I8 | DType::Bool => {
            let nbytes = n * out_shape.dtype().size_bytes().max(1);
            let off = arena.byte_offset(out);
            HostOut::Bytes(arena.raw_buf()[off..off + nbytes].to_vec())
        }
        // Complex outputs occupy 2 (C64) / 4 (C128) f32 lanes per element:
        // rlx-cpu stores them in `size_bytes()`-granular slots (8 B / 16 B) and
        // its data-movement ops (Expand/Transpose/Concat/Narrow/…) copy whole
        // elements, so the `[re, im]` / df64 lanes stay paired. Reading back only
        // `num_elements` f32 would truncate the output to a fraction of its lanes
        // (dropping the imaginary / df64-lo lanes) — the same lane-count readback
        // that `backend.rs::read_outputs` applies via `arena_lane_count`.
        dt if dt.is_complex() => {
            let lanes = if dt == DType::C64 { 2 } else { 4 };
            let slot = arena.slice_mut(out);
            let take = (n * lanes).min(slot.len());
            HostOut::F32(slot[..take].to_vec())
        }
        _ => {
            let slot = arena.slice_mut(out);
            let take = n.min(slot.len());
            HostOut::F32(slot[..take].to_vec())
        }
    }
}
