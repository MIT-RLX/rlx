// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! When the ND-indexing host fallback is **not** needed.
//!
//! The rest of this crate stages ops to the host. This module is its inverse
//! for the four ONNX indexing ops: given the [`IndexingThunk`] a backend was
//! about to hand to rlx-cpu, decide whether `indexing_nd.cu` (CUDA/ROCm) or the
//! equivalent WGSL can run it on-device instead, and if so produce every scalar,
//! offset and `meta` word the launch needs.
//!
//! It lives beside the fallback rather than in a backend because the decision is
//! identical on CUDA, ROCm and wgpu — all three have an f32-uniform arena, and
//! all three were paying a full D2H → CPU → H2D round trip mid-graph for these
//! ops. The shape arithmetic itself is in [`rlx_gpu_dispatch::indexing`], which
//! is dependency-free; this module is only the part that has to know what an
//! `IndexingThunk` is.
//!
//! **Returning `None` is always safe** — it means the caller keeps the existing
//! host route. Every guard below is a case where the on-device kernels do not
//! reproduce the CPU kernel exactly, so declining is the correct answer rather
//! than a missed optimisation to be chased later.

use rlx_cpu::onnx_indexing::scatter_nd_reduction_to_attrs;
use rlx_cpu::thunk::{IndexingThunk, Thunk};
use rlx_gpu_dispatch::indexing::{self as plan, KernelKind, Prologue};

/// Everything a backend needs to launch one on-device indexing op.
///
/// Offsets are **f32 element** indices (`byte_offset / 4`), matching the
/// `*_off: u32` convention the arena kernels use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexingLaunch {
    /// Which kernel, and its kernel-specific scalars.
    pub kind: KernelKind,
    /// Threads for the main kernel.
    pub n: u32,
    /// `data` operand.
    pub data_off: u32,
    /// `indices` operand (f32-encoded integers).
    pub idx_off: u32,
    /// `updates` operand; 0 for the gathers, which have none.
    pub upd_off: u32,
    /// Destination slot.
    pub dst_off: u32,
    /// Destination extent, for the kernels' bounds guard.
    pub dst_len: u32,
    /// `copy_sanitize_f32` prologue; `None` for the gathers.
    pub prologue: Option<Prologue>,
    /// u32 buffer of shapes/strides, uploaded once at schedule-build time.
    pub meta: Vec<u32>,
}

/// Byte offset → f32 element index, rejecting anything not 4-aligned.
///
/// A misaligned operand means the slot is not an f32 tensor in the arena, which
/// is exactly the situation these kernels must not guess at.
fn elem_off(byte_off: usize) -> Option<u32> {
    if !byte_off.is_multiple_of(4) {
        return None;
    }
    u32::try_from(byte_off / 4).ok()
}

/// The reduction, if the GPU kernels implement it.
fn reduction_of(r: rlx_ir::ScatterNdReduction) -> Option<plan::Reduction> {
    plan::Reduction::from_code(i32::from_le_bytes(scatter_nd_reduction_to_attrs(r)))
}

/// Plan an on-device launch for `thunk`, or `None` to keep the host route.
///
/// Declines, in order of how often they fire:
///
/// * **Genuine packed I64 indices** (`indices_i64 != 0`). On these arenas an I64
///   index tensor is normally stored as f32-*valued* slots and the thunk is
///   marked with `force_indices_f32`; when it is not, the buffer really is 8
///   bytes per element and the f32 readers here would consume float bits.
/// * **Non-f32 gathered elements** (GatherElements' `data_elem_bytes != 4`),
///   e.g. gathering i64 token ids, which must move at their true width.
/// * **`Mul` / `Max` / `Min` scatters**, which would need a float CAS loop.
/// * **Shapes the planner declines** — see [`rlx_gpu_dispatch::indexing`].
#[must_use]
pub fn plan_indexing(thunk: &IndexingThunk) -> Option<IndexingLaunch> {
    match thunk.inner() {
        Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len: _,
            indices_len: _,
            out_len,
            indices_i64,
            batch_dims,
        } => {
            if *indices_i64 != 0 {
                return None;
            }
            let p = plan::plan_gather_nd(data_shape, indices_shape, *batch_dims, *out_len)?;
            Some(IndexingLaunch {
                kind: p.kind(),
                n: p.n,
                data_off: elem_off(*data)?,
                idx_off: elem_off(*indices)?,
                upd_off: 0,
                dst_off: elem_off(*dst)?,
                dst_len: *out_len,
                prologue: None,
                meta: p.meta,
            })
        }

        Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len: _,
            out_len,
            indices_i64,
            data_elem_bytes,
            axis,
        } => {
            // GatherElements preserves dtype: a non-f32 element width means the
            // gathered payload is not what this f32 kernel would move.
            if *indices_i64 != 0 || *data_elem_bytes != 4 {
                return None;
            }
            let p = plan::plan_gather_elements(data_shape, indices_shape, *axis, *out_len)?;
            Some(IndexingLaunch {
                kind: p.kind(*data_len),
                n: p.n,
                data_off: elem_off(*data)?,
                idx_off: elem_off(*indices)?,
                upd_off: 0,
                dst_off: elem_off(*dst)?,
                dst_len: *out_len,
                prologue: None,
                meta: p.meta,
            })
        }

        Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            axis,
            reduction,
        } => {
            if *indices_i64 != 0 {
                return None;
            }
            let red = reduction_of(*reduction)?;
            let p = plan::plan_scatter_elements(
                data_shape,
                indices_shape,
                *axis,
                *indices_len,
                *updates_len,
            )?;
            Some(IndexingLaunch {
                kind: p.kind(red),
                n: p.n,
                data_off: elem_off(*data)?,
                idx_off: elem_off(*indices)?,
                upd_off: elem_off(*updates)?,
                dst_off: elem_off(*dst)?,
                dst_len: *data_len,
                prologue: Some(Prologue::scatter_elements(*data_len, data == dst)),
                meta: p.meta,
            })
        }

        Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len: _,
            indices_i64,
            reduction,
        } => {
            if *indices_i64 != 0 {
                return None;
            }
            let red = reduction_of(*reduction)?;
            let p = plan::plan_scatter_nd(data_shape, indices_shape, *data_len, *updates_len)?;
            Some(IndexingLaunch {
                kind: p.kind(red),
                n: p.n,
                data_off: elem_off(*data)?,
                idx_off: elem_off(*indices)?,
                upd_off: elem_off(*updates)?,
                dst_off: elem_off(*dst)?,
                dst_len: *data_len,
                prologue: Some(Prologue::scatter_nd(*data_len, data == dst)),
                meta: p.meta,
            })
        }

        // `IndexingThunk` is constructed only from the four kinds above, but the
        // inner `Thunk` enum is much wider. Anything else stays on the host.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::ScatterNdReduction;

    fn gather_nd_thunk(indices_i64: u8) -> IndexingThunk {
        IndexingThunk(Thunk::GatherNd {
            data: 0,
            indices: 64,
            dst: 128,
            data_shape: vec![2, 2, 2],
            indices_shape: vec![2, 1],
            data_len: 8,
            indices_len: 2,
            out_len: 8,
            indices_i64,
            batch_dims: 0,
        })
    }

    #[test]
    fn plans_a_gather_nd() {
        let l = plan_indexing(&gather_nd_thunk(0)).expect("planned");
        assert_eq!(l.n, 8);
        assert_eq!(l.data_off, 0);
        assert_eq!(l.idx_off, 16);
        assert_eq!(l.dst_off, 32);
        assert!(l.prologue.is_none());
        assert_eq!(
            l.kind,
            KernelKind::GatherNd {
                k: 1,
                slice: 4,
                tuples_per_batch: 2,
                batch_stride: 8,
            }
        );
    }

    #[test]
    fn declines_packed_i64_indices() {
        // The bug this whole path exists to avoid: reading float bits as i64.
        assert!(plan_indexing(&gather_nd_thunk(1)).is_none());
    }

    #[test]
    fn declines_non_f32_gather_elements() {
        let t = IndexingThunk(Thunk::GatherElements {
            data: 0,
            indices: 64,
            dst: 128,
            data_shape: vec![3, 3],
            indices_shape: vec![3, 3],
            data_len: 9,
            indices_len: 9,
            out_len: 9,
            indices_i64: 0,
            data_elem_bytes: 8,
            axis: 1,
        });
        assert!(plan_indexing(&t).is_none());
    }

    #[test]
    fn declines_misaligned_operands() {
        let t = IndexingThunk(Thunk::GatherNd {
            data: 2, // not 4-aligned → not an f32 slot
            indices: 64,
            dst: 128,
            data_shape: vec![2, 2, 2],
            indices_shape: vec![2, 1],
            data_len: 8,
            indices_len: 2,
            out_len: 8,
            indices_i64: 0,
            batch_dims: 0,
        });
        assert!(plan_indexing(&t).is_none());
    }

    fn scatter_nd_thunk(reduction: ScatterNdReduction, dst: usize) -> IndexingThunk {
        IndexingThunk(Thunk::ScatterNd {
            data: 0,
            indices: 64,
            updates: 128,
            dst,
            data_shape: vec![4, 3],
            indices_shape: vec![2, 1],
            data_len: 12,
            updates_len: 6,
            indices_len: 2,
            indices_i64: 0,
            reduction,
        })
    }

    #[test]
    fn plans_scatter_nd_and_its_prologue() {
        let l = plan_indexing(&scatter_nd_thunk(ScatterNdReduction::None, 256)).expect("planned");
        assert_eq!(l.n, 6);
        assert_eq!(l.dst_len, 12);
        // Not aliased → copy, and ScatterND never sanitises.
        assert_eq!(
            l.prologue,
            Some(Prologue {
                src_len: 12,
                do_copy: 1,
                do_sanitize: 0
            })
        );

        // Aliased dst → skip the copy, matching the CPU's `ptr::eq` check.
        let l = plan_indexing(&scatter_nd_thunk(ScatterNdReduction::None, 0)).expect("planned");
        assert_eq!(l.prologue.map(|p| p.do_copy), Some(0));
    }

    #[test]
    fn scatter_elements_prologue_sanitises() {
        let t = IndexingThunk(Thunk::ScatterElements {
            data: 0,
            indices: 64,
            updates: 128,
            dst: 256,
            data_shape: vec![3, 3],
            indices_shape: vec![3, 3],
            data_len: 9,
            updates_len: 9,
            indices_len: 9,
            indices_i64: 0,
            axis: 1,
            reduction: ScatterNdReduction::None,
        });
        let l = plan_indexing(&t).expect("planned");
        assert_eq!(l.prologue.map(|p| p.do_sanitize), Some(1));
    }

    #[test]
    fn declines_cas_reductions() {
        for r in [
            ScatterNdReduction::Mul,
            ScatterNdReduction::Max,
            ScatterNdReduction::Min,
        ] {
            assert!(
                plan_indexing(&scatter_nd_thunk(r, 256)).is_none(),
                "{r:?} must stay on the host"
            );
        }
        // Add is on-device.
        assert!(plan_indexing(&scatter_nd_thunk(ScatterNdReduction::Add, 256)).is_some());
    }
}
