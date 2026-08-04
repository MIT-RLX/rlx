// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// LSTM (gate order i, f, g, o; single merged bias). Dispatched once per (layer,
// direction) by the lowering, which loops layers×dirs and ping-pongs
// intermediate layer outputs through an in-arena scratch region (x_off/out_off
// are absolute arena word offsets). One workgroup per batch item; thread `k`
// owns hidden unit `k`, keeps its cell state `c_k` in a register, and shares
// `h_prev` in workgroup memory for the w_hh matvec. `h0_off`/`c0_off` (0 → 0)
// seed the hidden/cell state; `out_width`=dirs·hidden, `dir_off`=dir·hidden and
// `reverse` place this direction's output. hidden ≤ 256 (else host fallback).
// Single-layer/unidir/no-carry reduces to the plain kernel. Bit-for-bit mirror
// of `execute_lstm_f32`. Barriers sit in uniform control flow.

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
    h0_off: u32,
    c0_off: u32,
    out_width: u32,
    dir_off: u32,
    reverse: u32,
    _p: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const MAX_H: u32 = 256u;
var<workgroup> h_sh: array<f32, 256>;

@compute @workgroup_size(256)
fn lstm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wid.x;
    let k = lid.x;
    let h = params.hidden;
    let in_sz = params.input_size;
    let lane_on = (bi < params.batch) && (k < h) && (h <= MAX_H);

    var c_k: f32 = 0.0;
    if (k < MAX_H) {
        var seed: f32 = 0.0;
        if (params.h0_off != 0u && lane_on) {
            seed = arena[params.h0_off + bi * h + k];
        }
        h_sh[k] = seed;
    }
    if (params.c0_off != 0u && lane_on) {
        c_k = arena[params.c0_off + bi * h + k];
    }
    workgroupBarrier();

    for (var step: u32 = 0u; step < params.seq; step = step + 1u) {
        var t = step;
        if (params.reverse != 0u) { t = params.seq - 1u - step; }
        var h_k: f32 = 0.0;
        if (lane_on) {
            let x_base = params.x_off + (bi * params.seq_stride + t) * in_sz;
            // Gate rows i=k, f=h+k, g=2h+k, o=3h+k.
            var z: array<f32, 4>;
            for (var gate: u32 = 0u; gate < 4u; gate = gate + 1u) {
                let r = gate * h + k;
                var acc = arena[params.bias_off + r];
                let wih_row = params.wih_off + r * in_sz;
                for (var j: u32 = 0u; j < in_sz; j = j + 1u) {
                    acc = acc + arena[wih_row + j] * arena[x_base + j];
                }
                let whh_row = params.whh_off + r * h;
                for (var j: u32 = 0u; j < h; j = j + 1u) {
                    acc = acc + arena[whh_row + j] * h_sh[j];
                }
                z[gate] = acc;
            }
            let i_g = 1.0 / (1.0 + exp(-z[0]));
            let f_g = 1.0 / (1.0 + exp(-z[1]));
            let g_g = tanh(z[2]);
            let o_g = 1.0 / (1.0 + exp(-z[3]));
            c_k = f_g * c_k + i_g * g_g;
            h_k = o_g * tanh(c_k);
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
