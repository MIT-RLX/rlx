// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Per-row reduction along the LAST axis. Output shape: input shape
// minus its last dim (or with the last dim collapsed to 1 when
// keep_dim is set; the dispatcher handles the shape arithmetic).
//
// One thread per output row — sequential read across the inner axis.
// Slow by GPU standards but functional; tree-reduction with shared
// memory is the obvious optimization once the op set is broad enough
// to run real models.

struct Params {
    outer: u32,        // product of dims before the reduce axis
    reduce_dim: u32,   // size of the reduce axis
    inner: u32,        // product of dims after the reduce axis (=1 for last-axis)
    in_off: u32,
    out_off: u32,
    op: u32,           // 0=sum, 1=mean, 2=max, 3=min, 4=prod
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Per-workgroup scratch for parallel tree reduction within an output
// row when the reduce dim exceeds the small-dim fast path.
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let total_out = params.outer * params.inner;
    if (params.reduce_dim == 0u) { return; }

    // Two dispatch modes share this kernel:
    //   * Small reduce_dim (≤ 64): one thread per output cell does a
    //     sequential reduction. Dispatched on `total_out` threads.
    //   * Large reduce_dim (> 64): one workgroup per output cell does
    //     a 64-wide parallel reduction with a workgroup-shared scratch
    //     buffer. Dispatched on `total_out` workgroups (1 per cell).
    //
    // The host picks which mode via the dispatch grid size — see
    // backend.rs::Step::Reduce dispatch logic. Both forms read the
    // same params and write the same output cell.
    if (params.reduce_dim <= 64u) {
        let cell = gid.x + gid.y * ngs.x * 64u;
        if (cell >= total_out) { return; }
        let outer_idx = cell / params.inner;
        let inner_idx = cell % params.inner;
        let stride_outer = params.reduce_dim * params.inner;
        let base = params.in_off + outer_idx * stride_outer + inner_idx;

        var acc: f32 = arena[base];
        for (var r: u32 = 1u; r < params.reduce_dim; r = r + 1u) {
            let v = arena[base + r * params.inner];
            switch (params.op) {
                case 0u, 1u: { acc = acc + v; }
                case 2u:     { acc = max(acc, v); }
                case 3u:     { acc = min(acc, v); }
                case 4u:     { acc = acc * v; }
                default:     {}
            }
        }
        if (params.op == 1u) {
            acc = acc / f32(params.reduce_dim);
        }
        arena[params.out_off + outer_idx * params.inner + inner_idx] = acc;
        return;
    }

    // Large-dim path: each workgroup handles one output cell.
    let cell = wid.x + wid.y * ngs.x;
    if (cell >= total_out) { return; }
    let outer_idx = cell / params.inner;
    let inner_idx = cell % params.inner;
    let stride_outer = params.reduce_dim * params.inner;
    let base = params.in_off + outer_idx * stride_outer + inner_idx;
    let tid = lid.x;

    // Phase 1: each of 64 threads strides through the reduce dim,
    // accumulating its partial. Identity element depends on op.
    var acc: f32;
    if (tid < params.reduce_dim) {
        acc = arena[base + tid * params.inner];
    } else {
        switch (params.op) {
            case 0u, 1u: { acc = 0.0; }
            case 2u:     { acc = -1e30; }
            case 3u:     { acc =  1e30; }
            case 4u:     { acc = 1.0; }
            default:     { acc = 0.0; }
        }
    }
    var r: u32 = tid + 64u;
    loop {
        if (r >= params.reduce_dim) { break; }
        let v = arena[base + r * params.inner];
        switch (params.op) {
            case 0u, 1u: { acc = acc + v; }
            case 2u:     { acc = max(acc, v); }
            case 3u:     { acc = min(acc, v); }
            case 4u:     { acc = acc * v; }
            default:     {}
        }
        r = r + 64u;
    }
    scratch[tid] = acc;
    workgroupBarrier();

    // Phase 2: tree-reduce the 64 partials down to 1.
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let v = scratch[tid + stride];
            switch (params.op) {
                case 0u, 1u: { scratch[tid] = scratch[tid] + v; }
                case 2u:     { scratch[tid] = max(scratch[tid], v); }
                case 3u:     { scratch[tid] = min(scratch[tid], v); }
                case 4u:     { scratch[tid] = scratch[tid] * v; }
                default:     {}
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        var final_acc = scratch[0];
        if (params.op == 1u) {
            final_acc = final_acc / f32(params.reduce_dim);
        }
        arena[params.out_off + outer_idx * params.inner + inner_idx] = final_acc;
    }
}
