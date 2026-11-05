// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Top-K indices along the last axis. Output: f32-encoded indices,
// shape `[..., k]`. One workgroup per row: pass `step` picks, across the
// workgroup's threads, the best element ranked after pick `step - 1` in
// (value descending, index ascending) order — a strict total order, so ties
// go to the lower index (torch.topk(largest=True)) and nothing is tracked
// but the previous pick. O(k · inner / 64) per thread.
//
// (One thread per row, rescanning its previous picks for every element, took
// ~0.5 s for k = 50 over a 4096 vocabulary.)

struct Params {
    outer: u32,
    inner: u32,
    k: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const NONE: u32 = 0xffffffffu;
const NEG_INF: f32 = -3.4e38;

var<workgroup> best_v: array<f32, 64>;
var<workgroup> best_i: array<u32, 64>;
var<workgroup> prev_v: f32;
var<workgroup> prev_i: u32;

@compute @workgroup_size(64)
fn topk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.k;
    for (var step: u32 = 0u; step < params.k; step = step + 1u) {
        var bv: f32 = NEG_INF;
        var bi: u32 = NONE;
        for (var j: u32 = t; j < params.inner; j = j + 64u) {
            let v = arena[in_base + j];
            let after = step == 0u || v < prev_v || (v == prev_v && j > prev_i);
            if (after && (bi == NONE || v > bv)) {
                bv = v;
                bi = j;
            }
        }
        best_v[t] = bv;
        best_i[t] = bi;
        workgroupBarrier();
        for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
            if (t < s) {
                let ov = best_v[t + s];
                let oi = best_i[t + s];
                let better = oi != NONE && (best_i[t] == NONE || ov > best_v[t] || (ov == best_v[t] && oi < best_i[t]));
                if (better) {
                    best_v[t] = ov;
                    best_i[t] = oi;
                }
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            arena[out_base + step] = f32(best_i[0]);
            prev_v = best_v[0];
            prev_i = best_i[0];
        }
        workgroupBarrier();
    }
}
