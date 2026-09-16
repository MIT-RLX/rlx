// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// On-device ONNX ND indexing: GatherND, GatherElements, ScatterElements,
// ScatterND. WGSL twin of `rlx-gpu-kernels/kernels/indexing_nd.cu`, driven by
// the same launch plans from `rlx_gpu_dispatch::indexing`, so the two backends
// cannot drift. One difference, and it is deliberate: the scatters here are
// overwrite-only (no f32 atomics in core WGSL — see `reduction` below).
//
// These four were the last ops still taking `Step::CpuIndexing` on wgpu — a
// readback of data + indices + updates, a CPU pass, and an upload of the
// result, in the middle of a graph.
//
// Index slots hold f32-ENCODED integers: the wgpu arena is f32-uniform, so an
// I64 index tensor is widened to f32 on upload (`force_indices_f32`). Reading
// such a slot as raw integer bits is the bug that drifted F5 DiT, so every read
// below rounds the float instead.
//
// Shapes and strides ride in the `axis_meta` storage buffer (`meta` is a
// reserved WGSL keyword) (the 3-binding pattern
// `expand.wgsl` uses), so there is no rank cap and no per-dispatch allocation.
//
// Duplicate destinations diverge from CPU, by ONNX's own admission: with
// `reduction=none` it leaves ScatterND/ScatterElements undefined when two index
// tuples name the same slot. rlx-cpu resolves it sequentially (last update in
// flat order wins); these kernels write concurrently and the winner is whichever
// invocation the scheduler runs last. Both are legal; they are not equal.

struct Params {
    n: u32,           // thread count
    data_off: u32,
    idx_off: u32,
    upd_off: u32,     // scatter only
    dst_off: u32,
    dst_len: u32,     // scatter bounds guard / gather data bound
    rank: u32,        // GatherElements / ScatterElements
    axis: u32,        // GatherElements / ScatterElements
    axis_dim: i32,    // GatherElements
    k: u32,           // GatherND / ScatterND tuple width
    slice: u32,       // GatherND / ScatterND contiguous run
    tuples_per_batch: u32, // GatherND
    batch_stride: u32,     // GatherND
    // Always 0 (overwrite). Core WGSL has no f32 atomics, and a plain
    // read-modify-write would silently drop updates whenever two threads hit
    // the same destination — a wrong tensor with no error, which is the exact
    // failure mode this whole path exists to avoid. `Reduction::Add` therefore
    // keeps the CPU host route on wgpu; CUDA and ROCm do it with `atomicAdd`.
    // The field stays so the Params layout matches the .cu launch arguments.
    reduction: u32,
    src_len: u32,     // copy_sanitize
    flags: u32,       // copy_sanitize: bit0 = do_copy, bit1 = do_sanitize
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;
@group(0) @binding(2) var<storage, read>       axis_meta: array<u32>;

// Read an f32-encoded index slot, wrap negatives, clamp into [0, dim).
// Mirrors `rlx_idx_f32` in the .cu and the CPU kernels' wrap-then-clamp.
fn rlx_idx(raw: f32, dim: i32) -> u32 {
    var idx: i32 = i32(round(raw));
    if (idx < 0) { idx = idx + dim; }
    if (idx < 0) { idx = 0; }
    if (dim > 0 && idx >= dim) { idx = dim - 1; }
    return u32(idx);
}

fn tid(gid: vec3<u32>, ngs: vec3<u32>) -> u32 {
    return gid.x + gid.y * ngs.x * 64u;
}

// ---------------------------------------------------------------------------
// GatherND — axis_meta = [data_strides[k], data_dims[k]] (already shifted by
// batch_dims on the host). One thread per output element.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn gather_nd(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n) { return; }

    let t = i / params.slice;
    let j = i - t * params.slice;
    var bi: u32 = 0u;
    if (params.tuples_per_batch != 0u) { bi = t / params.tuples_per_batch; }

    var off: u32 = bi * params.batch_stride;
    let tuple = t * params.k;
    for (var m: u32 = 0u; m < params.k; m = m + 1u) {
        let dim = i32(axis_meta[params.k + m]);
        off = off + rlx_idx(arena[params.idx_off + tuple + m], dim) * axis_meta[m];
    }
    arena[params.dst_off + i] = arena[params.data_off + off + j];
}

// ---------------------------------------------------------------------------
// GatherElements — axis_meta = [ishape, dstride, ostride, dshape], each `rank` long.
// The flat output position decomposes by the INDICES' strides (ONNX lets the
// index tensor be smaller than `data` off-axis); coordinate `axis` comes from
// the index value. One thread per output element.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn gather_elements(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) ngs: vec3<u32>) {
    let lin = tid(gid, ngs);
    if (lin >= params.n) { return; }

    let r = params.rank;
    var off: u32 = 0u;
    for (var d: u32 = 0u; d < r; d = d + 1u) {
        let dstride = axis_meta[r + d];
        if (d == params.axis) {
            off = off + rlx_idx(arena[params.idx_off + lin], params.axis_dim) * dstride;
        } else {
            var dim = axis_meta[d];
            if (dim == 0u) { dim = 1u; }
            let coord = (lin / axis_meta[2u * r + d]) % dim;
            var last: u32 = 0u;
            let dshape = axis_meta[3u * r + d];
            if (dshape > 0u) { last = dshape - 1u; }
            off = off + min(coord, last) * dstride;
        }
    }
    // `data.get(off)` on the CPU side: an out-of-range read leaves the output
    // slot untouched. `dst_len` carries `data_len` for this kernel.
    if (off < params.dst_len) {
        arena[params.dst_off + lin] = arena[params.data_off + off];
    }
}

// ---------------------------------------------------------------------------
// ScatterElements — axis_meta = [istride, dstride, dshape], each `rank` long.
// Only the CPU kernel's *exact* branch (indices rank == data rank and
// prod(indices_shape) == n); the host declines the rest. One thread per update.
// `dst` must already hold the sanitised copy of `data` (see copy_sanitize).
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn scatter_elements(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(num_workgroups) ngs: vec3<u32>) {
    let f = tid(gid, ngs);
    if (f >= params.n) { return; }

    let r = params.rank;
    // CPU: `indices[flat_i].max(0)` — a bare floor at 0, NOT the wrap-and-clamp
    // `rlx_idx` applies elsewhere. Kept different on purpose.
    var row: i32 = i32(round(arena[params.idx_off + f]));
    if (row < 0) { row = 0; }

    var rem: u32 = f;
    var dst: u32 = 0u;
    for (var d: u32 = 0u; d < r; d = d + 1u) {
        let istride = axis_meta[d];
        var c: u32 = 0u;
        if (istride != 0u) {
            c = rem / istride;
            rem = rem % istride;
        }
        if (d == params.axis) { c = u32(row); }
        var last: u32 = 0u;
        let dshape = axis_meta[2u * r + d];
        if (dshape > 0u) { last = dshape - 1u; }
        dst = dst + min(c, last) * axis_meta[r + d];
    }
    if (dst >= params.dst_len) { return; }

    // Overwrite only — see the `reduction` note on `Params`.
    arena[params.dst_off + dst] = arena[params.upd_off + f];
}

// ---------------------------------------------------------------------------
// ScatterND — axis_meta = [data_strides[k], data_dims[k]].
// One thread per (update, slice element).
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn scatter_nd_reduce(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n) { return; }

    let u = i / params.slice;
    let j = i - u * params.slice;

    var off: u32 = 0u;
    for (var m: u32 = 0u; m < params.k; m = m + 1u) {
        let dim = i32(axis_meta[params.k + m]);
        off = off + rlx_idx(arena[params.idx_off + u * params.k + m], dim) * axis_meta[m];
    }
    if (off + j >= params.dst_len) { return; }

    // Overwrite only — see the `reduction` note on `Params`.
    arena[params.dst_off + off + j] = arena[params.upd_off + u * params.slice + j];
}

// ---------------------------------------------------------------------------
// Scatter prologue: seed `dst` from `data`, optionally zeroing non-finite slots.
//
// The zeroing reproduces `scatter_elements_f32`'s CPU prologue;
// `scatter_nd_into_f32` has none, so ScatterND clears bit1. bit0 is clear when
// `data` and `dst` already alias, matching the CPU's `ptr::eq` check.
//
// WGSL has no `isfinite`, and `v != v` alone misses the infinities. Comparing
// against f32::MAX catches both: `abs(NaN) <= MAX` is false because every NaN
// comparison is, and `abs(inf) <= MAX` is false because inf exceeds it. An
// `< inf` literal would be cleaner but 2^128 is not representable in f32.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn copy_sanitize(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n) { return; }

    let do_copy = (params.flags & 1u) != 0u;
    let do_sanitize = (params.flags & 2u) != 0u;

    var v: f32 = arena[params.dst_off + i];
    if (do_copy && i < params.src_len) { v = arena[params.data_off + i]; }
    if (do_sanitize && !(abs(v) <= 3.4028235e38)) { v = 0.0; }
    arena[params.dst_off + i] = v;
}
