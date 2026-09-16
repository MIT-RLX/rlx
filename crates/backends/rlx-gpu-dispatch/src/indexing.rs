// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Launch plans for the ONNX ND indexing ops on an f32-uniform arena.
//!
//! GatherND / GatherElements / ScatterElements / ScatterND all resolve to the
//! same three questions: *is this shape configuration one the GPU kernel
//! implements*, *how many threads*, and *what goes in the `meta` u32 buffer*.
//! None of those answers involve a device, and all four kernels are shared
//! between CUDA, ROCm and wgpu — so the answers live here rather than being
//! written out three times and drifting.
//!
//! The addressing reproduces `rlx_cpu::onnx_indexing` exactly. Where the CPU
//! kernel has several code paths, this planner implements the one the GPU
//! kernel implements and returns `None` for the rest, which keeps those cases
//! on the host instead of quietly computing something else. A `None` is always
//! a correctness-preserving fallback, never a wrong answer.
//!
//!
//! One divergence from the CPU kernels is inherent rather than a gap: with
//! `reduction=none`, ONNX leaves a scatter undefined when two index tuples name
//! the same destination. rlx-cpu resolves it sequentially (last write wins);
//! the GPU kernels write concurrently. The planner cannot rule this out — the
//! indices are runtime data — so it does not try. `Reduction::Add` is the
//! order-independent option.
//!
//! Kept dependency-free like the rest of this crate: reductions arrive as a
//! plain [`Reduction`] rather than `rlx_ir::ScatterNdReduction`, so `rlx-metal`
//! and `rlx-wgpu` can read this without linking the IR.

/// Scatter accumulation mode the GPU kernels implement.
///
/// `Mul`/`Max`/`Min` are deliberately absent: a float CAS loop would be slower
/// than the host round-trip it replaces at the sizes these ops appear at, and
/// harder to prove equal to the CPU result under duplicate indices. Callers map
/// those to `None` from [`Reduction::from_code`] returning `None` and stay on
/// the host.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Overwrite the destination slot.
    Overwrite,
    /// Atomic add into the destination slot.
    Add,
}

impl Reduction {
    /// Map the IR's reduction discriminant (as encoded by
    /// `rlx_cpu::onnx_indexing::scatter_nd_reduction_to_attrs`) onto the modes
    /// the GPU kernels implement. `None` means "stay on the host".
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Overwrite),
            1 => Some(Self::Add),
            _ => None,
        }
    }

    /// The `reduction` kernel argument.
    #[must_use]
    pub fn code(self) -> u32 {
        match self {
            Self::Overwrite => 0,
            Self::Add => 1,
        }
    }
}

/// Which `indexing_nd` kernel a scheduled step launches, with the scalars that
/// kernel takes beyond the offsets and `meta` every one of them shares.
///
/// Lives here rather than in a backend's `Step` enum so CUDA, ROCm and wgpu
/// schedule the same four kernels from the same description instead of three
/// parallel enums that have to be kept in agreement by hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    /// `gather_nd_f32`.
    GatherNd {
        k: u32,
        slice: u32,
        tuples_per_batch: u32,
        batch_stride: u32,
    },
    /// `gather_elements_f32`.
    GatherElements {
        rank: u32,
        axis: u32,
        axis_dim: i32,
        /// Bound for the `data.get(off)` guard the CPU kernel has.
        data_len: u32,
    },
    /// `scatter_elements_f32`.
    ScatterElements {
        rank: u32,
        axis: u32,
        reduction: Reduction,
    },
    /// `scatter_nd_reduce_f32`.
    ScatterNd {
        k: u32,
        slice: u32,
        reduction: Reduction,
    },
}

/// `copy_sanitize_f32` arguments — the prologue that seeds a scatter's `dst`
/// from `data`.
///
/// ScatterElements zeroes non-finite slots afterwards and ScatterND does not;
/// that asymmetry is in the CPU kernels, so it is carried explicitly rather
/// than inferred at the launch site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prologue {
    /// Elements of `data` available to copy.
    pub src_len: u32,
    /// 0 when `data` and `dst` already alias.
    pub do_copy: u32,
    /// 1 for ScatterElements, 0 for ScatterND.
    pub do_sanitize: u32,
}

impl Prologue {
    /// The prologue for a ScatterElements into `dst`.
    #[must_use]
    pub fn scatter_elements(src_len: u32, aliased: bool) -> Self {
        Self {
            src_len,
            do_copy: u32::from(!aliased),
            do_sanitize: 1,
        }
    }

    /// The prologue for a ScatterND into `dst`.
    #[must_use]
    pub fn scatter_nd(src_len: u32, aliased: bool) -> Self {
        Self {
            src_len,
            do_copy: u32::from(!aliased),
            do_sanitize: 0,
        }
    }
}

impl GatherNdPlan {
    /// The kernel description for this plan.
    #[must_use]
    pub fn kind(&self) -> KernelKind {
        KernelKind::GatherNd {
            k: self.k,
            slice: self.slice,
            tuples_per_batch: self.tuples_per_batch,
            batch_stride: self.batch_stride,
        }
    }
}

impl GatherElementsPlan {
    /// The kernel description for this plan.
    #[must_use]
    pub fn kind(&self, data_len: u32) -> KernelKind {
        KernelKind::GatherElements {
            rank: self.rank,
            axis: self.axis,
            axis_dim: self.axis_dim,
            data_len,
        }
    }
}

impl ScatterElementsPlan {
    /// The kernel description for this plan.
    #[must_use]
    pub fn kind(&self, reduction: Reduction) -> KernelKind {
        KernelKind::ScatterElements {
            rank: self.rank,
            axis: self.axis,
            reduction,
        }
    }
}

impl ScatterNdPlan {
    /// The kernel description for this plan.
    #[must_use]
    pub fn kind(&self, reduction: Reduction) -> KernelKind {
        KernelKind::ScatterNd {
            k: self.k,
            slice: self.slice,
            reduction,
        }
    }
}

/// Row-major element strides for `shape`.
fn row_major_strides(shape: &[u32]) -> Vec<u32> {
    let mut strides = vec![1u32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(shape[i + 1].max(1));
    }
    strides
}

fn product(shape: &[u32]) -> u64 {
    shape.iter().map(|&d| u64::from(d)).product::<u64>()
}

/// `rlx_cpu::onnx_indexing::normalize_axis`.
fn normalize_axis(axis: i32, rank: usize) -> usize {
    let a = if axis < 0 { axis + rank as i32 } else { axis };
    (a.max(0) as usize).min(rank.saturating_sub(1))
}

/// A 1-D launch over `n` threads with a `meta` u32 buffer.
///
/// `meta` is uploaded once when the schedule is built, so a plan carries no
/// per-launch allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherNdPlan {
    /// Total output elements (thread count).
    pub n: u32,
    /// Index-tuple width (`indices_shape.last()`).
    pub k: u32,
    /// Contiguous elements copied per index tuple.
    pub slice: u32,
    /// Index tuples per batch entry.
    pub tuples_per_batch: u32,
    /// Element stride between consecutive batch entries in `data`.
    pub batch_stride: u32,
    /// `[data_strides[b..b+k], data_dims[b..b+k]]` — already shifted by
    /// `batch_dims`, so the kernel never needs `b`.
    pub meta: Vec<u32>,
}

/// Plan an ONNX GatherND. Mirrors `gather_nd_src_offsets`.
#[must_use]
pub fn plan_gather_nd(
    data_shape: &[u32],
    indices_shape: &[u32],
    batch_dims: i32,
    out_len: u32,
) -> Option<GatherNdPlan> {
    if data_shape.is_empty() || indices_shape.is_empty() {
        return None;
    }
    let k = *indices_shape.last()? as usize;
    let b = (batch_dims.max(0) as usize)
        .min(data_shape.len())
        .min(indices_shape.len());
    // `gather_nd_src_offsets` indexes `data_strides[b + m]` directly.
    if k == 0 || b + k > data_shape.len() {
        return None;
    }

    let data_strides = row_major_strides(data_shape);
    let slice = product(&data_shape[b + k..]).max(1);
    let batch_count = product(&data_shape[..b]).max(1);
    let tuples_per_batch = product(&indices_shape[b..indices_shape.len() - 1]).max(1);
    let batch_stride = product(&data_shape[b..]).max(1);

    let n = batch_count
        .checked_mul(tuples_per_batch)?
        .checked_mul(slice)?;
    // A disagreement here means the caller's shapes and the output slot do not
    // describe the same tensor; the host path is shape-tolerant, so defer.
    if n != u64::from(out_len) {
        return None;
    }

    let mut meta = Vec::with_capacity(2 * k);
    meta.extend_from_slice(&data_strides[b..b + k]);
    meta.extend(data_shape[b..b + k].iter().copied());

    Some(GatherNdPlan {
        n: u32::try_from(n).ok()?,
        k: u32::try_from(k).ok()?,
        slice: u32::try_from(slice).ok()?,
        tuples_per_batch: u32::try_from(tuples_per_batch).ok()?,
        batch_stride: u32::try_from(batch_stride).ok()?,
        meta,
    })
}

/// Launch plan for GatherElements / take-along-axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherElementsPlan {
    /// Output elements (thread count).
    pub n: u32,
    /// Shared rank of `data` and `indices`.
    pub rank: u32,
    /// Normalised gather axis.
    pub axis: u32,
    /// `data_shape[axis].max(1)` — the wrap/clamp bound for index values.
    pub axis_dim: i32,
    /// `[ishape[rank], dstride[rank], ostride[rank], dshape[rank]]`.
    pub meta: Vec<u32>,
}

/// Plan an ONNX GatherElements. Mirrors `gather_elements_f32`.
///
/// Returns `None` unless `indices` has the same rank as `data`: the CPU kernel
/// indexes `indices_shape[k]` for every `k < rank`, so a shorter shape has no
/// defined meaning to reproduce.
#[must_use]
pub fn plan_gather_elements(
    data_shape: &[u32],
    indices_shape: &[u32],
    axis: i32,
    out_len: u32,
) -> Option<GatherElementsPlan> {
    if data_shape.is_empty() || indices_shape.is_empty() {
        return None;
    }
    let rank = data_shape.len();
    if indices_shape.len() != rank {
        return None;
    }
    let axis = normalize_axis(axis, rank);

    let mut dstride = vec![1u32; rank];
    let mut ostride = vec![1u32; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        dstride[k] = dstride[k + 1].saturating_mul(data_shape[k + 1].max(1));
        ostride[k] = ostride[k + 1].saturating_mul(indices_shape[k + 1].max(1));
    }

    let n = product(indices_shape).max(1).min(u64::from(out_len));
    if n == 0 {
        return None;
    }

    let mut meta = Vec::with_capacity(4 * rank);
    meta.extend_from_slice(indices_shape);
    meta.extend_from_slice(&dstride);
    meta.extend_from_slice(&ostride);
    meta.extend_from_slice(data_shape);

    Some(GatherElementsPlan {
        n: u32::try_from(n).ok()?,
        rank: u32::try_from(rank).ok()?,
        axis: u32::try_from(axis).ok()?,
        axis_dim: i32::try_from(data_shape[axis].max(1)).ok()?,
        meta,
    })
}

/// Launch plan for ScatterElements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScatterElementsPlan {
    /// Update elements (thread count).
    pub n: u32,
    /// Shared rank of `data` and `indices`.
    pub rank: u32,
    /// Normalised scatter axis.
    pub axis: u32,
    /// `[istride[rank], dstride[rank], dshape[rank]]`.
    pub meta: Vec<u32>,
}

/// Plan an ONNX ScatterElements. Mirrors the *exact* branch of
/// `scatter_elements_f32` — the one taken when `indices` has the same rank as
/// `data` and `prod(indices_shape) == n`, so a flat update position decomposes
/// by the indices' own strides.
///
/// Returns `None` for rank 1 and for the "dense" branch, both of which infer a
/// layout rather than being told one. Those stay on the host: reproducing a
/// heuristic on the GPU is how the two implementations drift.
#[must_use]
pub fn plan_scatter_elements(
    data_shape: &[u32],
    indices_shape: &[u32],
    axis: i32,
    indices_len: u32,
    updates_len: u32,
) -> Option<ScatterElementsPlan> {
    if data_shape.is_empty() {
        return None;
    }
    let rank = data_shape.len();
    // rank 1 has its own CPU branch (flat index, no stride decomposition).
    if rank < 2 || indices_shape.len() != rank {
        return None;
    }
    let n = u64::from(indices_len.min(updates_len));
    if n == 0 || product(indices_shape) != n {
        return None;
    }
    let axis = normalize_axis(axis, rank);

    // NOTE: plain products, no `.max(1)` — matching the exact branch, which
    // builds both stride vectors without clamping. A zero extent would make the
    // kernel divide by zero, but it also makes `n == 0` above, so it cannot
    // reach here.
    let mut istride = vec![1u32; rank];
    let mut dstride = vec![1u32; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        istride[d] = istride[d + 1].saturating_mul(indices_shape[d + 1]);
        dstride[d] = dstride[d + 1].saturating_mul(data_shape[d + 1]);
    }

    let mut meta = Vec::with_capacity(3 * rank);
    meta.extend_from_slice(&istride);
    meta.extend_from_slice(&dstride);
    meta.extend_from_slice(data_shape);

    Some(ScatterElementsPlan {
        n: u32::try_from(n).ok()?,
        rank: u32::try_from(rank).ok()?,
        axis: u32::try_from(axis).ok()?,
        meta,
    })
}

/// Launch plan for ScatterND.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScatterNdPlan {
    /// `num_updates * slice` (thread count).
    pub n: u32,
    /// Index-tuple width.
    pub k: u32,
    /// Contiguous elements written per index tuple.
    pub slice: u32,
    /// `[data_strides[..k], data_dims[..k]]`.
    pub meta: Vec<u32>,
}

/// Plan an ONNX ScatterND. Mirrors `scatter_nd_dst_offsets`.
///
/// Unlike the `k <= 4` scalar-argument kernel in `scatter_nd.cu`, this plan has
/// no rank cap: the strides ride in `meta`.
#[must_use]
pub fn plan_scatter_nd(
    data_shape: &[u32],
    indices_shape: &[u32],
    data_len: u32,
    updates_len: u32,
) -> Option<ScatterNdPlan> {
    if data_shape.is_empty() || indices_shape.is_empty() {
        return None;
    }
    let k = *indices_shape.last()? as usize;
    if k == 0 || k > data_shape.len() {
        return None;
    }
    if product(data_shape) != u64::from(data_len) {
        return None;
    }

    let data_strides = row_major_strides(data_shape);
    let slice = product(&data_shape[k..]).max(1);
    let num_updates = product(&indices_shape[..indices_shape.len() - 1]).max(1);
    let n = num_updates.checked_mul(slice)?;
    // The CPU kernel silently skips updates past the end of the buffer; rather
    // than reproduce that on-device, decline the shape.
    if n > u64::from(updates_len) {
        return None;
    }

    let mut meta = Vec::with_capacity(2 * k);
    meta.extend_from_slice(&data_strides[..k]);
    meta.extend_from_slice(&data_shape[..k]);

    Some(ScatterNdPlan {
        n: u32::try_from(n).ok()?,
        k: u32::try_from(k).ok()?,
        slice: u32::try_from(slice).ok()?,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_nd_slice_shape() {
        // data [2,2,2], indices [2,1] → 2 slices of 4.
        let p = plan_gather_nd(&[2, 2, 2], &[2, 1], 0, 8).expect("plan");
        assert_eq!(p.k, 1);
        assert_eq!(p.slice, 4);
        assert_eq!(p.tuples_per_batch, 2);
        assert_eq!(p.n, 8);
        // strides[0] = 4, dims[0] = 2
        assert_eq!(p.meta, vec![4, 2]);
    }

    #[test]
    fn gather_nd_batch_dims_shift_meta() {
        // batch_dims=1 drops axis 0 from the index arithmetic entirely.
        let p = plan_gather_nd(&[2, 3, 4], &[2, 1, 1], 1, 8).expect("plan");
        assert_eq!(p.k, 1);
        assert_eq!(p.slice, 4);
        assert_eq!(p.batch_stride, 12);
        // strides[1] = 4, dims[1] = 3
        assert_eq!(p.meta, vec![4, 3]);
    }

    #[test]
    fn gather_nd_declines_when_out_len_disagrees() {
        assert!(plan_gather_nd(&[2, 2, 2], &[2, 1], 0, 7).is_none());
    }

    #[test]
    fn gather_nd_declines_k_past_rank() {
        assert!(plan_gather_nd(&[4], &[1, 2], 0, 1).is_none());
    }

    #[test]
    fn gather_elements_meta_layout() {
        let p = plan_gather_elements(&[3, 3], &[2, 3], 0, 6).expect("plan");
        assert_eq!(p.rank, 2);
        assert_eq!(p.axis, 0);
        assert_eq!(p.axis_dim, 3);
        assert_eq!(p.n, 6);
        // ishape, dstride, ostride, dshape
        assert_eq!(p.meta, vec![2, 3, 3, 1, 3, 1, 3, 3]);
    }

    #[test]
    fn gather_elements_negative_axis_normalises() {
        let p = plan_gather_elements(&[2, 5], &[2, 3], -1, 6).expect("plan");
        assert_eq!(p.axis, 1);
        assert_eq!(p.axis_dim, 5);
    }

    #[test]
    fn gather_elements_declines_rank_mismatch() {
        assert!(plan_gather_elements(&[3, 3], &[3], 0, 3).is_none());
    }

    #[test]
    fn scatter_elements_exact_branch_only() {
        let p = plan_scatter_elements(&[3, 3], &[2, 3], 1, 6, 6).expect("plan");
        assert_eq!(p.n, 6);
        assert_eq!(p.axis, 1);
        // istride (from [2,3]), dstride (from [3,3]), dshape
        assert_eq!(p.meta, vec![3, 1, 3, 1, 3, 3]);

        // prod(indices_shape) != n → the CPU takes its "dense" guess; decline.
        assert!(plan_scatter_elements(&[3, 3], &[2, 3], 1, 4, 4).is_none());
        // rank 1 has its own CPU branch.
        assert!(plan_scatter_elements(&[4], &[2], 0, 2, 2).is_none());
    }

    #[test]
    fn scatter_nd_has_no_rank_cap() {
        // k = 5 — past the scalar-argument kernel's k <= 4 limit.
        let p = plan_scatter_nd(&[2, 2, 2, 2, 2], &[1, 5], 32, 1).expect("plan");
        assert_eq!(p.k, 5);
        assert_eq!(p.slice, 1);
        assert_eq!(p.n, 1);
        assert_eq!(p.meta, vec![16, 8, 4, 2, 1, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn scatter_nd_declines_short_updates() {
        // 2 tuples x slice 4 = 8 updates needed, only 4 supplied.
        assert!(plan_scatter_nd(&[2, 2, 2], &[2, 1], 8, 4).is_none());
    }

    #[test]
    fn scatter_nd_declines_inconsistent_data_len() {
        assert!(plan_scatter_nd(&[2, 2, 2], &[2, 1], 7, 8).is_none());
    }

    #[test]
    fn reduction_codes_round_trip() {
        assert_eq!(Reduction::from_code(0), Some(Reduction::Overwrite));
        assert_eq!(Reduction::from_code(1), Some(Reduction::Add));
        // Mul / Max / Min stay on the host.
        assert_eq!(Reduction::from_code(2), None);
        assert_eq!(Reduction::from_code(3), None);
        assert_eq!(Reduction::from_code(4), None);
        assert_eq!(Reduction::Add.code(), 1);
    }
}
