// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU host-fallback for ops that have no native SPIR-V kernel yet (or that
//! exceed a native size cap): oversized Lstm/Gru/Rnn/Mamba2, FFT, GGUF
//! dequant, and specialized families. Each fallback builds a one-op CPU
//! graph, runs it through `rlx-cpu`'s thunk executor (the same kernels the
//! CPU backend uses, so results are bit-for-bit the reference), and returns
//! the f32 output.
//!
//! Because the Vulkan arena is HOST_VISIBLE + mapped, the executor reads the
//! op's inputs straight out of the arena and writes the result straight back —
//! no device↔host staging. The cost is one queue flush around the op.
//!
//! The Vulkan arena is **f32-uniform**: integer / bool tensors are stored as
//! `f32`-encoded values (one f32 word per element). The CPU arena uses native
//! dtype widths, so this module converts at the boundary.

use rlx_ir::{DType, Graph, Op, Shape};

/// One host-fallback input: f32 activations (including f32-encoded ints from
/// the Vulkan arena), or raw bytes for a packed quant weight (U8/I8).
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

/// Pack f32-encoded arena values into the CPU arena's native dtype bytes.
fn write_f32_encoded_as_native(raw: &mut [u8], off: usize, dtype: DType, vals: &[f32]) {
    match dtype {
        // C128 (complex f64) has no f32-uniform arena representation yet;
        // it is rejected upstream via `is_complex()`. Present only to keep
        // this exhaustive match compiling until the f32-sim path lands.
        DType::C128 => panic!("rlx-vulkan: C128 not representable on the f32-uniform arena"),
        DType::F32 | DType::C64 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = v.to_le_bytes();
                let dst = off + i * 4;
                if dst + 4 <= raw.len() {
                    raw[dst..dst + 4].copy_from_slice(&b);
                }
            }
        }
        DType::F64 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = (v as f64).to_le_bytes();
                let dst = off + i * 8;
                if dst + 8 <= raw.len() {
                    raw[dst..dst + 8].copy_from_slice(&b);
                }
            }
        }
        DType::I64 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = (v as i64).to_le_bytes();
                let dst = off + i * 8;
                if dst + 8 <= raw.len() {
                    raw[dst..dst + 8].copy_from_slice(&b);
                }
            }
        }
        DType::I32 | DType::U32 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = (v as i32).to_le_bytes();
                let dst = off + i * 4;
                if dst + 4 <= raw.len() {
                    raw[dst..dst + 4].copy_from_slice(&b);
                }
            }
        }
        DType::I16 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = (v as i16).to_le_bytes();
                let dst = off + i * 2;
                if dst + 2 <= raw.len() {
                    raw[dst..dst + 2].copy_from_slice(&b);
                }
            }
        }
        DType::I8 => {
            for (i, &v) in vals.iter().enumerate() {
                let dst = off + i;
                if dst < raw.len() {
                    raw[dst] = v as i8 as u8;
                }
            }
        }
        DType::U8 | DType::Bool => {
            for (i, &v) in vals.iter().enumerate() {
                let dst = off + i;
                if dst < raw.len() {
                    raw[dst] = v as u8;
                }
            }
        }
        DType::F16 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = half::f16::from_f32(v).to_le_bytes();
                let dst = off + i * 2;
                if dst + 2 <= raw.len() {
                    raw[dst..dst + 2].copy_from_slice(&b);
                }
            }
        }
        DType::BF16 => {
            for (i, &v) in vals.iter().enumerate() {
                let b = half::bf16::from_f32(v).to_le_bytes();
                let dst = off + i * 2;
                if dst + 2 <= raw.len() {
                    raw[dst..dst + 2].copy_from_slice(&b);
                }
            }
        }
    }
}

/// Unpack CPU-native bytes into f32-encoded values for the Vulkan arena.
fn read_native_as_f32_encoded(raw: &[u8], off: usize, dtype: DType, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    match dtype {
        // C128 (complex f64): rejected upstream via `is_complex()`; arm
        // present only to keep this exhaustive match compiling.
        DType::C128 => panic!("rlx-vulkan: C128 not representable on the f32-uniform arena"),
        DType::F32 | DType::C64 => {
            for i in 0..n {
                let s = off + i * 4;
                if s + 4 > raw.len() {
                    break;
                }
                out.push(f32::from_le_bytes([
                    raw[s],
                    raw[s + 1],
                    raw[s + 2],
                    raw[s + 3],
                ]));
            }
        }
        DType::F64 => {
            for i in 0..n {
                let s = off + i * 8;
                if s + 8 > raw.len() {
                    break;
                }
                let v = f64::from_le_bytes(raw[s..s + 8].try_into().unwrap());
                out.push(v as f32);
            }
        }
        DType::I64 => {
            for i in 0..n {
                let s = off + i * 8;
                if s + 8 > raw.len() {
                    break;
                }
                let v = i64::from_le_bytes(raw[s..s + 8].try_into().unwrap());
                out.push(v as f32);
            }
        }
        DType::I32 | DType::U32 => {
            for i in 0..n {
                let s = off + i * 4;
                if s + 4 > raw.len() {
                    break;
                }
                let v = i32::from_le_bytes([raw[s], raw[s + 1], raw[s + 2], raw[s + 3]]);
                out.push(v as f32);
            }
        }
        DType::I16 => {
            for i in 0..n {
                let s = off + i * 2;
                if s + 2 > raw.len() {
                    break;
                }
                out.push(i16::from_le_bytes([raw[s], raw[s + 1]]) as f32);
            }
        }
        DType::I8 => {
            for i in 0..n {
                if off + i >= raw.len() {
                    break;
                }
                out.push(raw[off + i] as i8 as f32);
            }
        }
        DType::U8 | DType::Bool => {
            for i in 0..n {
                if off + i >= raw.len() {
                    break;
                }
                out.push(raw[off + i] as f32);
            }
        }
        DType::F16 => {
            for i in 0..n {
                let s = off + i * 2;
                if s + 2 > raw.len() {
                    break;
                }
                out.push(half::f16::from_le_bytes([raw[s], raw[s + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let s = off + i * 2;
                if s + 2 > raw.len() {
                    break;
                }
                out.push(half::bf16::from_le_bytes([raw[s], raw[s + 1]]).to_f32());
            }
        }
    }
    out
}

/// Input indices an op writes back **in place**, which the host fallback has to
/// carry out of its private CPU arena by hand.
///
/// [`eval`] builds a throwaway graph on its own `rlx_cpu::arena::Arena` and
/// reads back only the output slot. That is right for a pure op and silently
/// wrong for one whose contract is to mutate an operand: the RNN family with
/// `carry: true` writes the final `hn`/`cn` over `h0`/`c0` so the next `run()`
/// continues the sequence, and those writes landed in the scratch arena and
/// were dropped. Symptom: four single-step runs diverged from one four-step run
/// from step 1 onward — state simply never advanced.
///
/// Vulkan is the only backend affected. CUDA, ROCm, Metal and wgpu run carry
/// natively and never take this path.
pub fn inplace_inputs(op: &Op) -> &'static [usize] {
    // Indices follow `Op::num_inputs`: Lstm(x, w_ih, w_hh, bias, h0, c0),
    // Gru(x, w_ih, w_hh, b_ih, b_hh, h0), Rnn(x, w_ih, w_hh, bias, h0).
    match op {
        Op::Lstm { carry: true, .. } => &[4, 5],
        Op::Gru { carry: true, .. } => &[5],
        Op::Rnn { carry: true, .. } => &[4],
        _ => &[],
    }
}

/// [`eval`]'s result plus any operands the op mutated in place.
pub struct HostEval {
    pub out: HostOut,
    /// `(input index, new contents)` for each entry of [`inplace_inputs`].
    pub inplace: Vec<(usize, HostOut)>,
}

/// Run a single op on the CPU reference and return its output as f32-encoded
/// values (or raw bytes for U8/I8 packed outputs) for the Vulkan arena.
pub fn eval(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostOut {
    eval_full(op, out_shape, inputs).out
}

/// [`eval`], also returning operands the op wrote back in place.
pub fn eval_full(op: &Op, out_shape: &Shape, inputs: &[(Shape, HostBuf)]) -> HostEval {
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

    for (i, (sh, buf)) in inputs.iter().enumerate() {
        match buf {
            HostBuf::F32(vals) => {
                let off = arena.byte_offset(ids[i]);
                write_f32_encoded_as_native(arena.raw_buf_mut(), off, sh.dtype(), vals);
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

    // Read a slot back in whatever encoding its dtype calls for.
    let read_slot = |arena: &rlx_cpu::arena::Arena, id: rlx_ir::NodeId, sh: &Shape| -> HostOut {
        let n = sh.num_elements().unwrap_or(0);
        let off = arena.byte_offset(id);
        match sh.dtype() {
            // Packed-byte slots (quant codes / block scales) are read back raw
            // so the U8 bytes aren't reinterpreted as f32.
            DType::U8 | DType::I8 => {
                let nbytes =
                    (n * sh.dtype().size_bytes()).min(arena.raw_buf().len().saturating_sub(off));
                HostOut::Bytes(arena.raw_buf()[off..off + nbytes].to_vec())
            }
            dt => HostOut::F32(read_native_as_f32_encoded(arena.raw_buf(), off, dt, n)),
        }
    };

    // Anything the op mutated in place has to come back too — see
    // `inplace_inputs`. Without this the RNN carry writeback died in this
    // function's private arena.
    let inplace: Vec<(usize, HostOut)> = inplace_inputs(op)
        .iter()
        .filter_map(|&i| {
            let (sh, _) = inputs.get(i)?;
            Some((i, read_slot(&arena, ids[i], sh)))
        })
        .collect();

    HostEval {
        out: read_slot(&arena, out, out_shape),
        inplace,
    }
}
