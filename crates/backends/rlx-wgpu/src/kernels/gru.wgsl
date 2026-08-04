// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// GRU (gate order r, z, n; linear_before_reset=1; separate b_ih/b_hh).
// Dispatched once per (layer, direction) by the lowering, which loops
// layers×dirs and ping-pongs intermediate layer outputs through an in-arena
// scratch region (x_off/out_off are absolute arena word offsets). One workgroup
// per batch item; thread `k` owns hidden unit `k`, shared hidden state. `h0_off`
// (0 → h0=0) seeds the state; `out_width`=dirs·hidden, `dir_off`=dir·hidden and
// `reverse` place this direction's output. hidden ≤ 256 (else host fallback).
// Single-layer/unidir/no-carry reduces to the original kernel. Barriers sit in
// uniform control flow (outside the active-thread guard).

struct Params {
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    x_off: u32,
    wih_off: u32,
    whh_off: u32,
    bih_off: u32,
    bhh_off: u32,
    out_off: u32,
    seq_stride: u32,
    h0_off: u32,
    out_width: u32,
    dir_off: u32,
    reverse: u32,
    _p5: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const MAX_H: u32 = 256u;
var<workgroup> h_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn gru(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wid.x;
    let k = lid.x;
    let h = params.hidden;
    let in_sz = params.input_size;
    let lane_on = (bi < params.batch) && (k < h) && (h <= MAX_H);

    if (k < MAX_H) {
        var seed: f32 = 0.0;
        if (params.h0_off != 0u && lane_on) {
            seed = arena[params.h0_off + bi * h + k];
        }
        h_sh[k] = seed;
    }
    workgroupBarrier();

    for (var step: u32 = 0u; step < params.seq; step = step + 1u) {
        var t = step;
        if (params.reverse != 0u) { t = params.seq - 1u - step; }
        var h_k: f32 = 0.0;
        if (lane_on) {
            let x_base = params.x_off + (bi * params.seq_stride + t) * in_sz;
            var xi: array<f32, 3>;
            var hipart: array<f32, 3>;
            for (var g: u32 = 0u; g < 3u; g = g + 1u) {
                let r = g * h + k;
                var ax = arena[params.bih_off + r];
                let wih_row = params.wih_off + r * in_sz;
                for (var j: u32 = 0u; j < in_sz; j = j + 1u) {
                    ax = ax + arena[wih_row + j] * arena[x_base + j];
                }
                var ah = arena[params.bhh_off + r];
                let whh_row = params.whh_off + r * h;
                for (var j: u32 = 0u; j < h; j = j + 1u) {
                    ah = ah + arena[whh_row + j] * h_sh[j];
                }
                xi[g] = ax;
                hipart[g] = ah;
            }
            let rg = 1.0 / (1.0 + exp(-(xi[0] + hipart[0])));
            let zg = 1.0 / (1.0 + exp(-(xi[1] + hipart[1])));
            let ng = tanh(xi[2] + rg * hipart[2]);
            h_k = (1.0 - zg) * ng + zg * h_sh[k];
        }
        // Uniform barrier: all threads finished reading the old h_sh.
        workgroupBarrier();
        if (lane_on) {
            h_sh[k] = h_k;
            arena[params.out_off + (bi * params.seq_stride + t) * params.out_width + params.dir_off + k] = h_k;
        }
        workgroupBarrier();
    }
}
