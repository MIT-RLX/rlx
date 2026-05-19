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
/// Currently supports F32 / F16 / BF16. Other dtypes fall through to F32.
pub unsafe fn write_typed_from_f32(dst_ptr: *mut u8, dtype: DType, src: &[f32], max_elems: usize) {
    let n = src.len().min(max_elems);
    match dtype {
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
        _ => unsafe {
            let dst = dst_ptr as *mut f32;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
        },
    }
}

/// Helper: read `n_elems` of `dtype` from `src_ptr`, returning `Vec<f32>`.
pub unsafe fn read_typed_to_f32(src_ptr: *const u8, dtype: DType, n_elems: usize) -> Vec<f32> {
    match dtype {
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
        _ => unsafe {
            let src = src_ptr as *const f32;
            std::slice::from_raw_parts(src, n_elems).to_vec()
        },
    }
}
