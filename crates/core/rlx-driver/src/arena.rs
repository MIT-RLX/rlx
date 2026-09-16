// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-agnostic arena trait — the contract every backend's memory
//! plan obeys.
//!
//! Lifted from CpuExecutable / MetalExecutable's previously duplicated
//! arena helpers. Each new backend (CUDA, ROCm, wgpu, WASM, TPU) implements
//! this trait once and gets:
//!   - typed input feed (`f32 → arena_dtype`)
//!   - typed output read (`arena_dtype → f32`)
//!   - per-node byte offset resolution
//!
//! The trait deliberately exposes raw pointers / byte offsets rather than
//! Rust slices so the same implementation works for host-resident memory
//! (CPU/WASM), unified memory (Apple Silicon Metal/MPSGraph), and
//! discrete-VRAM backends (CUDA/ROCm) where reading involves a copy.

use rlx_ir::{DType, NodeId};

/// Per-backend arena interface.
///
/// All concrete arenas — `rlx-cpu::Arena`, `rlx-metal::Arena`, future
/// `rlx-cuda::Arena`, `rlx-wgpu::Arena` — implement this trait so the
/// runtime can drive them uniformly. The actual byte layout is owned
/// by the backend; we only require offset-based access.
pub trait DeviceArena {
    /// Byte offset of `id`'s buffer slot in the arena. `usize::MAX` for
    /// nodes that don't have an arena slot (e.g. fused-away intermediates).
    fn byte_offset(&self, id: NodeId) -> usize;

    /// True if `id` has a real arena slot.
    fn has_buffer(&self, id: NodeId) -> bool;

    /// Total arena size in bytes.
    fn size_bytes(&self) -> usize;

    /// Write a host-side `f32` slice into `id`'s slot, casting to `dtype`
    /// if necessary. Truncates to the buffer's capacity (no panic on overflow).
    ///
    /// For discrete-memory backends this involves a host→device copy; for
    /// unified-memory backends (Apple Silicon, integrated GPUs) it's a
    /// direct write.
    fn write_input_f32(&mut self, id: NodeId, dtype: DType, data: &[f32]);

    /// Read `id`'s slot as a host-side `Vec<f32>`, casting from `dtype` if
    /// necessary. The number of elements is determined by the backend
    /// based on the memory plan (typically `shape.num_elements()`).
    fn read_output_f32(&self, id: NodeId, dtype: DType, n_elements: usize) -> Vec<f32>;
}

/// Helper: cast f32 input to bytes of `dtype` and write to `dst_ptr`.
/// Used by every CPU-resident-arena backend. GPU backends can call this
/// after staging into a host buffer, then upload.
///
/// `max_elems` is a count of `dtype` ELEMENTS, so every arm must write
/// `dtype.size_bytes()` per element. The F32 fall-through writes 4 B/elem,
/// which is only sound for the 4-byte dtypes (F32/I32/U32) — a narrower slot
/// takes it as a 2x (I16) or 4x (U8/I8/Bool) buffer overrun. The narrow arms
/// below convert numerically, the exact inverse of
/// `rlx_runtime::backend::widen_bytes_to_f32`.
pub unsafe fn write_typed_from_f32(dst_ptr: *mut u8, dtype: DType, src: &[f32], max_elems: usize) {
    let n = src.len().min(max_elems);
    match dtype {
        DType::F64 => unsafe {
            // F64 slots are 8 B/elem; widen the f32 input. (Values carry
            // f32 precision — this is the f32 entry path; `run_typed`'s
            // `all_f64` branch is the full-precision route.)
            let dst = dst_ptr as *mut f64;
            for i in 0..n {
                *dst.add(i) = src[i] as f64;
            }
        },
        DType::F16 => unsafe {
            let dst = dst_ptr as *mut half::f16;
            for i in 0..n {
                *dst.add(i) = half::f16::from_f32(src[i]);
            }
        },
        DType::BF16 => unsafe {
            let dst = dst_ptr as *mut half::bf16;
            for i in 0..n {
                *dst.add(i) = half::bf16::from_f32(src[i]);
            }
        },
        DType::C64 => unsafe {
            // Interleaved [re, im, re, im, ...]; `max_elems` is complex count.
            let dst = dst_ptr as *mut f32;
            let n = src.len().min(max_elems.saturating_mul(2));
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
        },
        DType::C128 => unsafe {
            // Interleaved [re, im, ...] f64 pairs (16 B/elem); `max_elems`
            // is complex count. This is the f32 entry path: widen each
            // interleaved f32 lane to f64 (values carry f32 precision).
            let dst = dst_ptr as *mut f64;
            let n = src.len().min(max_elems.saturating_mul(2));
            for i in 0..n {
                *dst.add(i) = src[i] as f64;
            }
        },
        // 1-byte slots. `as` casts on floats saturate (and map NaN to 0),
        // so an out-of-range value clamps rather than being UB.
        DType::U8 => unsafe {
            for i in 0..n {
                *dst_ptr.add(i) = src[i] as u8;
            }
        },
        DType::Bool => unsafe {
            for i in 0..n {
                *dst_ptr.add(i) = u8::from(src[i] != 0.0);
            }
        },
        DType::I8 => unsafe {
            let dst = dst_ptr as *mut i8;
            for i in 0..n {
                *dst.add(i) = src[i] as i8;
            }
        },
        // 2-byte slot.
        DType::I16 => unsafe {
            let dst = dst_ptr as *mut i16;
            for i in 0..n {
                *dst.add(i) = src[i] as i16;
            }
        },
        // F32 / I32 / U32: 4 B/elem, so the width matches. Integers keep the
        // long-standing f32-bit-aliased representation of the f32 arena here.
        _ => unsafe {
            let dst = dst_ptr as *mut f32;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
        },
    }
}

/// Helper: read `n_elems` of `dtype` from `src_ptr`, returning `Vec<f32>`.
pub unsafe fn read_typed_to_f32(src_ptr: *const u8, dtype: DType, n_elems: usize) -> Vec<f32> {
    match dtype {
        DType::F64 => {
            // F64 slots are 8 B/elem; narrow to f32 for the f32 read path.
            let mut out = Vec::with_capacity(n_elems);
            unsafe {
                let src = src_ptr as *const f64;
                for i in 0..n_elems {
                    out.push(*src.add(i) as f32);
                }
            }
            out
        }
        DType::F16 => {
            let mut out = Vec::with_capacity(n_elems);
            unsafe {
                let src = src_ptr as *const half::f16;
                for i in 0..n_elems {
                    out.push((*src.add(i)).to_f32());
                }
            }
            out
        }
        DType::BF16 => {
            let mut out = Vec::with_capacity(n_elems);
            unsafe {
                let src = src_ptr as *const half::bf16;
                for i in 0..n_elems {
                    out.push((*src.add(i)).to_f32());
                }
            }
            out
        }
        DType::C64 => unsafe {
            // Interleaved [re, im, re, im, ...]; `n_elems` is complex count.
            let src = src_ptr as *const f32;
            std::slice::from_raw_parts(src, n_elems.saturating_mul(2)).to_vec()
        },
        DType::C128 => unsafe {
            // Interleaved [re, im, ...] f64 pairs (16 B/elem); `n_elems` is
            // complex count. f32 read path: narrow each f64 lane to f32.
            let src = src_ptr as *const f64;
            let lanes = n_elems.saturating_mul(2);
            let mut out = Vec::with_capacity(lanes);
            for i in 0..lanes {
                out.push(*src.add(i) as f32);
            }
            out
        },
        // NOTE: narrow (U8/I8/Bool/I16) slots deliberately keep the f32 read
        // below. This is `read_output`'s path, and on the CPU backend — which
        // plans with `ArenaWidthPolicy::Native` — an integer/bool ACTIVATION is
        // written by its kernel as f32 ("widen at compute", see the
        // `plan_memory_hybrid` doc), so the f32 read is what matches it. That
        // read runs past a natively-sized slot: it is the "integer-overrun
        // risk" `plan_memory_hybrid` names, and closing it means settling
        // whether a native-width U8 slot carries codes or widened f32 — a
        // policy change, not a local fix. The WRITE path above is different:
        // it is host input, where a native slot has no room for f32 at all.
        _ => unsafe {
            let src = src_ptr as *const f32;
            std::slice::from_raw_parts(src, n_elems).to_vec()
        },
    }
}
