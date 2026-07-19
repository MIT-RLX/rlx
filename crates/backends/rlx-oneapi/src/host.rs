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
pub fn eval(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
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

    for (i, (_, buf)) in inputs.iter().enumerate() {
        match buf {
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
        _ => {
            let slot = arena.slice_mut(out);
            let take = n.min(slot.len());
            HostOut::F32(slot[..take].to_vec())
        }
    }
}
