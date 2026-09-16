// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The GPU ND-indexing kernels must address memory exactly like rlx-cpu.**
//!
//! GatherND / GatherElements / ScatterElements / ScatterND run on the host on
//! every f32-uniform arena backend (CUDA, ROCm, wgpu) via `Step::CpuIndexing`.
//! Moving them on-device means re-deriving four pieces of index arithmetic that
//! already exist in `rlx_cpu::onnx_indexing` — negative-index wrap, clamp to the
//! last valid element, off-axis coordinate carry, and the batch-dims shift. Get
//! any of those subtly wrong and nothing panics: you get a plausible tensor.
//! That is the failure mode `Op::ScatterNd` already shipped once, when reading
//! an f32-encoded index slot as raw i64 bits silently drifted F5 DiT.
//!
//! A GPU parity test cannot catch it on a machine with no GPU, and the rigs are
//! not always up. So this checks the same thing structurally instead: it replays
//! the **kernel's own addressing** — the arithmetic written in
//! `rlx-gpu-kernels/kernels/indexing_nd.cu`, driven by the `meta` buffer and
//! scalars that `rlx_gpu_dispatch::indexing` plans — as scalar Rust, and
//! requires bit-equality with the CPU kernel on the same inputs.
//!
//! The replay functions below are a transliteration of the `.cu`, not a second
//! implementation: each mirrors one kernel body statement for statement. If
//! someone edits the kernel without editing these, the divergence is the point
//! — that is what a device-free guard can catch that a skipped GPU test cannot.
//!
//! What this does *not* prove: that the kernels compile, launch, or that the
//! atomics behave under contention. Those need a device. This proves the part
//! that was going to be wrong.

use rlx_cpu::onnx_indexing as cpu;
use rlx_gpu_dispatch::indexing as plan;

// ---------------------------------------------------------------------------
// Kernel replays — one per `extern "C" __global__` in indexing_nd.cu
// ---------------------------------------------------------------------------

/// `rlx_idx_f32` from the kernel.
fn rlx_idx_f32(raw: f32, dim: i32) -> u32 {
    let mut idx = raw.round_ties_even() as i32;
    if idx < 0 {
        idx += dim;
    }
    if idx < 0 {
        idx = 0;
    }
    if dim > 0 && idx >= dim {
        idx = dim - 1;
    }
    idx as u32
}

/// `gather_nd_f32`.
// The `if x != 0 { a / x } else { 0 }` shapes below are clippy's
// `manual_checked_ops`, and they stay: these functions are a statement-for-
// statement transliteration of the `.cu`, which is the only property that makes
// them a useful guard. `checked_div` would read better and correspond worse.
#[allow(clippy::manual_checked_ops)]
fn replay_gather_nd(data: &[f32], idx: &[f32], out: &mut [f32], p: &plan::GatherNdPlan) {
    let (strides, dims) = p.meta.split_at(p.k as usize);
    for i in 0..p.n {
        let t = i / p.slice;
        let j = i - t * p.slice;
        let bi = if p.tuples_per_batch != 0 {
            t / p.tuples_per_batch
        } else {
            0
        };
        let mut off = bi * p.batch_stride;
        let tuple = t * p.k;
        for m in 0..p.k as usize {
            off += rlx_idx_f32(idx[(tuple + m as u32) as usize], dims[m] as i32) * strides[m];
        }
        out[i as usize] = data[(off + j) as usize];
    }
}

/// `gather_elements_f32`.
fn replay_gather_elements(
    data: &[f32],
    idx: &[f32],
    out: &mut [f32],
    p: &plan::GatherElementsPlan,
) {
    let r = p.rank as usize;
    let ishape = &p.meta[..r];
    let dstride = &p.meta[r..2 * r];
    let ostride = &p.meta[2 * r..3 * r];
    let dshape = &p.meta[3 * r..4 * r];
    for lin in 0..p.n {
        let mut off = 0u32;
        for d in 0..r {
            if d as u32 == p.axis {
                off += rlx_idx_f32(idx[lin as usize], p.axis_dim) * dstride[d];
            } else {
                let dim = if ishape[d] != 0 { ishape[d] } else { 1 };
                let coord = (lin / ostride[d]) % dim;
                let last = dshape[d].saturating_sub(1);
                off += coord.min(last) * dstride[d];
            }
        }
        if (off as usize) < data.len() {
            out[lin as usize] = data[off as usize];
        }
    }
}

/// `copy_sanitize_f32`.
fn replay_copy_sanitize(src: &[f32], dst: &mut [f32], do_copy: bool, do_sanitize: bool) {
    for i in 0..dst.len() {
        let v = if do_copy && i < src.len() {
            src[i]
        } else {
            dst[i]
        };
        dst[i] = if do_sanitize && !v.is_finite() {
            0.0
        } else {
            v
        };
    }
}

/// `copy_sanitize_f32` (copy + sanitise) then `scatter_elements_f32`.
#[allow(clippy::manual_checked_ops)]
fn replay_scatter_elements(
    data: &[f32],
    idx: &[f32],
    updates: &[f32],
    out: &mut [f32],
    p: &plan::ScatterElementsPlan,
    reduction: plan::Reduction,
) {
    // ScatterElements' CPU prologue zeroes non-finite slots; ScatterND's does not.
    replay_copy_sanitize(data, out, true, true);

    let r = p.rank as usize;
    let istride = &p.meta[..r];
    let dstride = &p.meta[r..2 * r];
    let dshape = &p.meta[2 * r..3 * r];
    let dst_len = out.len() as u32;
    for f in 0..p.n {
        let mut row = idx[f as usize].round_ties_even() as i32;
        if row < 0 {
            row = 0;
        }
        let mut rem = f;
        let mut dst = 0u32;
        for d in 0..r {
            let mut c = if istride[d] != 0 { rem / istride[d] } else { 0 };
            if istride[d] != 0 {
                rem %= istride[d];
            }
            if d as u32 == p.axis {
                c = row as u32;
            }
            let last = dshape[d].saturating_sub(1);
            dst += c.min(last) * dstride[d];
        }
        if dst >= dst_len {
            continue;
        }
        let v = updates[f as usize];
        match reduction {
            plan::Reduction::Add => out[dst as usize] += v,
            plan::Reduction::Overwrite => out[dst as usize] = v,
        }
    }
}

/// `scatter_nd_reduce_f32` (destination pre-seeded with `data`).
fn replay_scatter_nd(
    idx: &[f32],
    updates: &[f32],
    out: &mut [f32],
    p: &plan::ScatterNdPlan,
    reduction: plan::Reduction,
) {
    let (strides, dims) = p.meta.split_at(p.k as usize);
    let dst_len = out.len() as u32;
    for i in 0..p.n {
        let u = i / p.slice;
        let j = i - u * p.slice;
        let mut off = 0u32;
        for m in 0..p.k as usize {
            off += rlx_idx_f32(idx[(u * p.k + m as u32) as usize], dims[m] as i32) * strides[m];
        }
        if off + j >= dst_len {
            continue;
        }
        let v = updates[(u * p.slice + j) as usize];
        match reduction {
            plan::Reduction::Add => out[(off + j) as usize] += v,
            plan::Reduction::Overwrite => out[(off + j) as usize] = v,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 1.0).collect()
}

fn usized(shape: &[u32]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

/// Indices as the f32-uniform arena stores them, and as i64 for the CPU kernel.
fn idx_pair(vals: &[i64]) -> (Vec<f32>, Vec<i64>) {
    (vals.iter().map(|&v| v as f32).collect(), vals.to_vec())
}

// ---------------------------------------------------------------------------
// GatherND
// ---------------------------------------------------------------------------

fn check_gather_nd(data_shape: &[u32], indices_shape: &[u32], batch_dims: i32, idx_vals: &[i64]) {
    let ds = usized(data_shape);
    let is = usized(indices_shape);
    let data = iota(ds.iter().product());
    let (idx_f32, idx_i64) = idx_pair(idx_vals);

    let (offsets, slice) = cpu::gather_nd_src_offsets(&ds, &idx_i64, &is, batch_dims as usize);
    let out_len = offsets.len() * slice;
    let mut expect = vec![0.0f32; out_len];
    for (t, &off) in offsets.iter().enumerate() {
        for j in 0..slice {
            expect[t * slice + j] = data[off + j];
        }
    }

    let p = plan::plan_gather_nd(data_shape, indices_shape, batch_dims, out_len as u32)
        .unwrap_or_else(|| panic!("planner declined GatherND {data_shape:?} / {indices_shape:?}"));
    let mut got = vec![0.0f32; out_len];
    replay_gather_nd(&data, &idx_f32, &mut got, &p);

    assert_eq!(
        got, expect,
        "GatherND data={data_shape:?} idx={indices_shape:?} batch_dims={batch_dims}"
    );
}

#[test]
fn gather_nd_matches_cpu() {
    // Full-coordinate gather: one scalar per tuple.
    check_gather_nd(&[2, 2], &[2, 2], 0, &[0, 0, 1, 1]);
    // Partial coordinate: each tuple names a contiguous slice.
    check_gather_nd(&[2, 2, 2], &[2, 1], 0, &[1, 0]);
    // Rank-3 index tensor over a rank-3 data tensor.
    check_gather_nd(
        &[3, 4, 5],
        &[3, 2, 2],
        0,
        &[0, 0, 2, 3, 1, 1, 0, 2, 2, 2, 1, 0],
    );
}

#[test]
fn gather_nd_batch_dims_match_cpu() {
    // batch_dims = 1: axis 0 is shared, not indexed.
    check_gather_nd(&[2, 3, 4], &[2, 1, 1], 1, &[2, 0]);
    check_gather_nd(&[3, 4, 2], &[3, 2, 1], 1, &[0, 3, 1, 2, 3, 0]);
}

#[test]
fn gather_nd_negative_and_out_of_range_indices_match_cpu() {
    // -1 wraps to the last row; 99 clamps to it. Both must agree with CPU.
    check_gather_nd(&[4, 3], &[3, 1], 0, &[-1, 99, -4]);
    check_gather_nd(&[2, 2, 2], &[3, 2], 0, &[-1, -1, 5, 0, 0, -2]);
}

// ---------------------------------------------------------------------------
// GatherElements
// ---------------------------------------------------------------------------

fn check_gather_elements(data_shape: &[u32], indices_shape: &[u32], axis: i32, idx_vals: &[i64]) {
    let ds = usized(data_shape);
    let is = usized(indices_shape);
    let data = iota(ds.iter().product());
    let (idx_f32, idx_i64) = idx_pair(idx_vals);
    let out_len: usize = is.iter().product();

    let mut expect = vec![0.0f32; out_len];
    cpu::gather_elements_f32(&data, &idx_i64, &mut expect, &ds, &is, axis);

    let p = plan::plan_gather_elements(data_shape, indices_shape, axis, out_len as u32)
        .unwrap_or_else(|| panic!("planner declined GatherElements {data_shape:?}"));
    let mut got = vec![0.0f32; out_len];
    replay_gather_elements(&data, &idx_f32, &mut got, &p);

    assert_eq!(
        got, expect,
        "GatherElements data={data_shape:?} idx={indices_shape:?} axis={axis}"
    );
}

#[test]
fn gather_elements_matches_cpu() {
    check_gather_elements(&[3, 3], &[2, 3], 0, &[1, 2, 0, 2, 0, 1]);
    check_gather_elements(&[3, 3], &[3, 2], 1, &[0, 2, 1, 1, 2, 0]);
    check_gather_elements(
        &[2, 3, 4],
        &[2, 3, 2],
        2,
        &[0, 3, 1, 2, 3, 0, 2, 2, 1, 3, 0, 1],
    );
}

#[test]
fn gather_elements_negative_axis_matches_cpu() {
    check_gather_elements(&[3, 4], &[3, 2], -1, &[3, 0, 1, 2, 2, 1]);
}

#[test]
fn gather_elements_negative_and_out_of_range_indices_match_cpu() {
    check_gather_elements(
        &[3, 4],
        &[3, 4],
        1,
        &[-1, 7, 0, -4, 2, -2, 9, 1, 0, 3, -3, 2],
    );
}

#[test]
fn gather_elements_smaller_off_axis_indices_match_cpu() {
    // ONNX allows the index tensor to be smaller than `data` off-axis; the
    // kernel must decompose by the INDICES' strides, not the data's. Getting
    // this backwards is right for row 0 and wrong for every other row.
    check_gather_elements(&[4, 5], &[2, 3], 1, &[0, 4, 2, 1, 3, 0]);
}

// ---------------------------------------------------------------------------
// ScatterElements
// ---------------------------------------------------------------------------

fn check_scatter_elements(
    data_shape: &[u32],
    indices_shape: &[u32],
    axis: i32,
    idx_vals: &[i64],
    reduction: rlx_ir::ScatterNdReduction,
) {
    let ds = usized(data_shape);
    let is = usized(indices_shape);
    let data = iota(ds.iter().product());
    let (idx_f32, idx_i64) = idx_pair(idx_vals);
    let n = idx_vals.len();
    let updates: Vec<f32> = (0..n).map(|i| -(i as f32) - 1.0).collect();

    let mut expect = vec![0.0f32; data.len()];
    cpu::scatter_elements_f32(
        &data,
        &updates,
        &idx_i64,
        &mut expect,
        &ds,
        &is,
        axis,
        reduction,
    );

    let code = i32::from_le_bytes(cpu::scatter_nd_reduction_to_attrs(reduction));
    let red = plan::Reduction::from_code(code).expect("reduction is GPU-eligible");
    let p = plan::plan_scatter_elements(data_shape, indices_shape, axis, n as u32, n as u32)
        .unwrap_or_else(|| panic!("planner declined ScatterElements {data_shape:?}"));
    let mut got = vec![0.0f32; data.len()];
    replay_scatter_elements(&data, &idx_f32, &updates, &mut got, &p, red);

    assert_eq!(
        got, expect,
        "ScatterElements data={data_shape:?} idx={indices_shape:?} axis={axis} red={reduction:?}"
    );
}

#[test]
fn scatter_elements_matches_cpu() {
    use rlx_ir::ScatterNdReduction::None;
    check_scatter_elements(&[3, 3], &[2, 3], 0, &[1, 0, 2, 0, 2, 1], None);
    check_scatter_elements(&[3, 3], &[3, 2], 1, &[0, 2, 1, 0, 2, 1], None);
    check_scatter_elements(
        &[2, 3, 4],
        &[2, 3, 2],
        2,
        &[0, 3, 1, 2, 3, 0, 2, 2, 1, 3, 0, 1],
        None,
    );
}

#[test]
fn scatter_elements_add_matches_cpu() {
    use rlx_ir::ScatterNdReduction::Add;
    // Distinct destinations: the atomic order cannot matter, so the CPU's
    // sequential result is the only correct answer and equality is meaningful.
    check_scatter_elements(&[3, 3], &[3, 3], 1, &[0, 1, 2, 0, 1, 2, 0, 1, 2], Add);
}

#[test]
fn scatter_elements_negative_axis_matches_cpu() {
    use rlx_ir::ScatterNdReduction::None;
    check_scatter_elements(&[3, 4], &[3, 2], -1, &[3, 0, 1, 2, 2, 1], None);
}

#[test]
fn scatter_elements_smaller_off_axis_indices_match_cpu() {
    use rlx_ir::ScatterNdReduction::None;
    check_scatter_elements(&[4, 5], &[2, 3], 1, &[0, 4, 2, 1, 3, 0], None);
}

// ---------------------------------------------------------------------------
// ScatterND
// ---------------------------------------------------------------------------

fn check_scatter_nd(
    data_shape: &[u32],
    indices_shape: &[u32],
    idx_vals: &[i64],
    reduction: rlx_ir::ScatterNdReduction,
) {
    let ds = usized(data_shape);
    let is = usized(indices_shape);
    let data = iota(ds.iter().product());
    let (idx_f32, idx_i64) = idx_pair(idx_vals);

    let (offsets, slice) = cpu::scatter_nd_dst_offsets(&ds, &idx_i64, &is);
    let updates: Vec<f32> = (0..offsets.len() * slice)
        .map(|i| -(i as f32) - 1.0)
        .collect();

    let mut expect = vec![0.0f32; data.len()];
    cpu::scatter_nd_into_f32(&data, &updates, &mut expect, &offsets, slice, reduction);

    let code = i32::from_le_bytes(cpu::scatter_nd_reduction_to_attrs(reduction));
    let red = plan::Reduction::from_code(code).expect("reduction is GPU-eligible");
    let p = plan::plan_scatter_nd(
        data_shape,
        indices_shape,
        data.len() as u32,
        updates.len() as u32,
    )
    .unwrap_or_else(|| panic!("planner declined ScatterND {data_shape:?}"));

    // The kernel writes into `dst` after `copy_sanitize_f32` seeds it from
    // `data` — with `do_sanitize = 0`, since `scatter_nd_into_f32` is a plain copy.
    let mut got = vec![0.0f32; data.len()];
    replay_copy_sanitize(&data, &mut got, true, false);
    replay_scatter_nd(&idx_f32, &updates, &mut got, &p, red);

    assert_eq!(
        got, expect,
        "ScatterND data={data_shape:?} idx={indices_shape:?} red={reduction:?}"
    );
}

#[test]
fn scatter_nd_matches_cpu() {
    use rlx_ir::ScatterNdReduction::None;
    check_scatter_nd(&[4, 3], &[2, 1], &[0, 3], None);
    check_scatter_nd(&[2, 2, 2], &[2, 2], &[0, 1, 1, 0], None);
    // k = 5 — past the `k <= 4` scalar-argument kernel in scatter_nd.cu, which
    // is the case that had to stay on the host before `meta` carried strides.
    check_scatter_nd(
        &[2, 2, 2, 2, 2],
        &[2, 5],
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        None,
    );
}

#[test]
fn scatter_nd_add_matches_cpu() {
    use rlx_ir::ScatterNdReduction::Add;
    // Distinct destinations — see the ScatterElements note.
    check_scatter_nd(&[4, 3], &[3, 1], &[0, 1, 2], Add);
}

#[test]
fn scatter_nd_negative_and_out_of_range_indices_match_cpu() {
    use rlx_ir::ScatterNdReduction::None;
    check_scatter_nd(&[4, 3], &[3, 1], &[-1, 9, -4], None);
}

// ---------------------------------------------------------------------------
// The planner declines what the kernels do not implement
// ---------------------------------------------------------------------------

#[test]
fn planner_declines_the_cpu_only_shapes() {
    // ScatterElements rank 1 and the "dense" guess branch stay on the host.
    assert!(plan::plan_scatter_elements(&[6], &[3], 0, 3, 3).is_none());
    assert!(plan::plan_scatter_elements(&[3, 3], &[2, 3], 1, 4, 4).is_none());
    // GatherElements with a shorter index rank has no CPU semantics to mirror.
    assert!(plan::plan_gather_elements(&[3, 3], &[3], 0, 3).is_none());
    // Mul / Max / Min are not GPU-eligible reductions.
    for r in [
        rlx_ir::ScatterNdReduction::Mul,
        rlx_ir::ScatterNdReduction::Max,
        rlx_ir::ScatterNdReduction::Min,
    ] {
        let code = i32::from_le_bytes(cpu::scatter_nd_reduction_to_attrs(r));
        assert_eq!(
            plan::Reduction::from_code(code),
            None,
            "{r:?} must stay on the host"
        );
    }
}
