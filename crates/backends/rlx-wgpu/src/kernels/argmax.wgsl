// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Argmax along the last axis. Returns f32-encoded indices (rlx
// convention); ties go to the smaller index. One 256-thread workgroup per
// row: each thread scans a strided slice, then the workgroup reduces
// (value, index) pairs. One thread per row used to scan a whole LM-head row
// (248k logits for Qwen3.5) alone.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let row = wid.x + wid.y * ngs.x;
    // Uniform across the workgroup: returning before the barriers is allowed.
    if (row >= params.outer || params.inner == 0u) { return; }
    let base = params.in_off + row * params.inner;
    var bv: f32 = -3.4e38;
    var bi: u32 = 0xffffffffu;
    for (var i: u32 = t; i < params.inner; i = i + 256u) {
        let v = arena[base + i];
        // Strictly greater: within a thread, indices only grow.
        if (v > bv || bi == 0xffffffffu) {
            bv = v;
            bi = i;
        }
    }
    best_v[t] = bv;
    best_i[t] = bi;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            let ov = best_v[t + s];
            let oi = best_i[t + s];
            let take = oi != 0xffffffffu
                && (best_i[t] == 0xffffffffu || ov > best_v[t] || (ov == best_v[t] && oi < best_i[t]));
            if (take) {
                best_v[t] = ov;
                best_i[t] = oi;
            }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        arena[params.out_off + row] = f32(best_i[0]);
    }
}
