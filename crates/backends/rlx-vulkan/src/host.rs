// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! CPU host-fallback for ops that have no native SPIR-V kernel yet (the
//! sequential / specialized families: RNN, Mamba2, GatedDeltaNet,
//! ConvTranspose2d, FFT). Each fallback builds a one-op CPU graph, runs it
//! through `rlx-cpu`'s thunk executor (the same kernels the CPU backend uses,
//! so results are bit-for-bit the reference), and returns the f32 output.
//!
//! Because the Vulkan arena is HOST_VISIBLE + mapped, the executor reads the
//! op's inputs straight out of the arena and writes the result straight back —
//! no device↔host staging. The cost is one queue flush around the op.

use rlx_ir::{DType, Graph, Op, Shape};

/// One host-fallback input: f32 activations, or raw bytes for a packed quant
/// weight (U8/I8 operands such as the GGUF weight of `Op::DequantMatMul` or the
/// packed codes / block scales of `Op::ScaledMatMul`).
pub enum HostBuf {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

/// A host-fallback op's output, in its native dtype: f32 for most ops, or packed
/// bytes for ops that emit `DType::U8`/`I8` (e.g. `Op::ScaledQuantize` codes and
/// block-scale `Op::ScaledQuantScale`).
pub enum HostOut {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

/// Run a single op on the CPU reference and return its output in its native
/// dtype. `inputs[i]` is `(declared_shape, buffer)` read from the arena.
pub fn eval(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
    let mut g = Graph::new("vk_host_fallback");
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
        // Packed-byte outputs (quant codes / block scales) are read back raw so
        // the U8 bytes aren't reinterpreted as f32.
        DType::U8 | DType::I8 => {
            let nbytes = n * out_shape.dtype().size_bytes();
            let off = arena.byte_offset(out);
            HostOut::Bytes(arena.raw_buf()[off..off + nbytes].to_vec())
        }
        _ => HostOut::F32(arena.slice_mut(out)[..n].to_vec()),
    }
}
