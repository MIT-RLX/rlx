// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Numerically-stable softmax along the last axis, ONE WORKGROUP PER ROW (64
// threads) with a shared-memory reduction — replaces the prior scalar
// one-thread-per-row three-pass loop.
//
// Precision: the row max is a plain parallel tree (max is exact). The exp-sum
// preserves the compensated (Kahan) accumulation the scalar path used for F5
// DiT ODE integration (plain f32 sum drift compounds across the ODE — wgpu fox
// 0/6 at NFE=32): each thread Kahan-sums its strided lanes, then thread 0
// Kahan-merges the 64 partials sequentially (≈ scalar-Kahan accuracy; only 64
// serial adds). WGSL has no f64.

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> scratch: array<f32, 64>;

fn tree_max(tid: u32) {
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
}

@compute @workgroup_size(64)
fn softmax(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = wid.x + wid.y * ngs.x;
    if (row >= params.outer || params.inner == 0u) { return; }
    let tid = lid.x;
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;

    // ── Pass 1: row max (parallel tree, exact). ──
    var m: f32 = -3.40282347e38; // -f32::MAX (identity for max)
    var i: u32 = tid;
    loop {
        if (i >= params.inner) { break; }
        m = max(m, arena[in_base + i]);
        i = i + 64u;
    }
    scratch[tid] = m;
    workgroupBarrier();
    tree_max(tid);
    let row_max = scratch[0];
    workgroupBarrier(); // all read scratch[0] before Pass 2 overwrites it.

    // ── Pass 2: Σ exp(x − max), Kahan-compensated. ──
    var s: f32 = 0.0;
    var c: f32 = 0.0;
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        let e = exp(arena[in_base + i] - row_max);
        let y = e - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
        i = i + 64u;
    }
    scratch[tid] = s;
    workgroupBarrier();
    // Thread 0 Kahan-merges the 64 partials (serial, but only 64 adds), keeping
    // the compensated-sum accuracy the ODE needs.
    if (tid == 0u) {
        var acc: f32 = 0.0;
        var cc: f32 = 0.0;
        for (var k: u32 = 0u; k < 64u; k = k + 1u) {
            let y = scratch[k] - cc;
            let t = acc + y;
            cc = (t - acc) - y;
            acc = t;
        }
        scratch[0] = acc;
    }
    workgroupBarrier();
    let inv_s = 1.0 / scratch[0];

    // ── Pass 3: write normalized (recompute exp; keeps peak memory low). ──
    i = tid;
    loop {
        if (i >= params.inner) { break; }
        arena[out_base + i] = exp(arena[in_base + i] - row_max) * inv_s;
        i = i + 64u;
    }
}
