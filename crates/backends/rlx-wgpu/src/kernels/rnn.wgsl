// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Elman RNN (`relu`!=0 ? relu : tanh). Dispatched once per (layer, direction)
// by the lowering — same in-arena scratch ping-pong and h0/out_width/dir_off/
// reverse convention as gru.wgsl. One workgroup per batch item; thread `k` owns
// hidden unit `k`, shared hidden state. hidden ≤ 256 (else host fallback).
// Single-layer/unidir/no-carry reduces to the original kernel.

struct Params {
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    x_off: u32,
    wih_off: u32,
    whh_off: u32,
    bias_off: u32,
    out_off: u32,
    seq_stride: u32,
    relu: u32,
    h0_off: u32,
    out_width: u32,
    dir_off: u32,
    reverse: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const MAX_H: u32 = 256u;
var<workgroup> h_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn rnn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
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
            var acc = arena[params.bias_off + k];
            let wih_row = params.wih_off + k * in_sz;
            for (var j: u32 = 0u; j < in_sz; j = j + 1u) {
                acc = acc + arena[wih_row + j] * arena[x_base + j];
            }
            let whh_row = params.whh_off + k * h;
            for (var j: u32 = 0u; j < h; j = j + 1u) {
                acc = acc + arena[whh_row + j] * h_sh[j];
            }
            if (params.relu != 0u) {
                h_k = max(acc, 0.0);
            } else {
                h_k = tanh(acc);
            }
        }
        workgroupBarrier();
        if (lane_on) {
            h_sh[k] = h_k;
            arena[params.out_off + (bi * params.seq_stride + t) * params.out_width + params.dir_off + k] = h_k;
        }
        workgroupBarrier();
    }
}
