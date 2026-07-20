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

/// Run a single op on the CPU reference and return its output as f32-encoded
/// values (or raw bytes for U8/I8 packed outputs) for the Vulkan arena.
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

    let n = out_shape.num_elements().unwrap_or(0);
    match out_shape.dtype() {
        // Packed-byte outputs (quant codes / block scales) are read back raw so
        // the U8 bytes aren't reinterpreted as f32.
        DType::U8 | DType::I8 => {
            let nbytes = n * out_shape.dtype().size_bytes();
            let off = arena.byte_offset(out);
            let avail = arena.raw_buf().len().saturating_sub(off);
            let nbytes = nbytes.min(avail);
            HostOut::Bytes(arena.raw_buf()[off..off + nbytes].to_vec())
        }
        dt => {
            let off = arena.byte_offset(out);
            HostOut::F32(read_native_as_f32_encoded(arena.raw_buf(), off, dt, n))
        }
    }
}
