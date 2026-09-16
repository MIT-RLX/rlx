// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Real INT8 `Op::QConv2d`: x[N,Cin,H,W] i8 · w[Cout,Cin/g,KH,KW] i8 + bias i32
// → out[N,Cout,Ho,Wo] i8, with i32 accumulation and a float requantize.
// Mirrors `rlx_cpu::thunk::ops::quant::exec_q_conv2d` and the Vulkan
// `q_conv2d.comp`.
//
// Same three conventions as `q_matmul.wgsl`: i8 tensors are byte-packed so
// `x`/`w`/`out` are byte offsets; `bias` is an I32 tensor, which keeps a full
// f32-sized slot on this arena and holds the integer as an f32 *value*; and one
// invocation owns a whole output WORD, because WGSL has no byte store and a
// per-element launch would lose three of every four results.

struct Params {
    batch: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    /// Byte offsets for the packed i8 operands.
    x_off: u32,
    w_off: u32,
    out_off: u32,
    /// f32-element offset.
    bias_off: u32,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: f32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

fn rd_i8(byte_off: u32) -> i32 {
    let word = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    let b = (bitcast<u32>(arena[word]) >> shift) & 0xffu;
    return i32(b) - 256 * i32((b >> 7u) & 1u);
}

fn wr_word(byte_off: u32, bytes: vec4<u32>) {
    let packed = (bytes.x & 0xffu)
        | ((bytes.y & 0xffu) << 8u)
        | ((bytes.z & 0xffu) << 16u)
        | ((bytes.w & 0xffu) << 24u);
    arena[byte_off / 4u] = bitcast<f32>(packed);
}

fn round_half_away(x: f32) -> f32 {
    let sgn = f32(x > 0.0) - f32(x < 0.0);
    return sgn * floor(abs(x) + 0.5);
}

fn q_conv2d_one(i: u32) -> u32 {
    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let co = q2 % params.c_out;
    let nn = q2 / params.c_out;

    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    var acc = i32(trunc(arena[params.bias_off + co]));
    for (var ci_off = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        let in_chan = ((nn * params.c_in) + ci) * params.h * params.w;
        let wt_chan = ((co * c_in_per_g) + ci_off) * params.kh * params.kw;
        for (var ki = 0u; ki < params.kh; ki = ki + 1u) {
            for (var kj = 0u; kj < params.kw; kj = kj + 1u) {
                let ih = i32(ho * params.sh + ki * params.dh) - i32(params.ph);
                let iw = i32(wo * params.sw + kj * params.dw) - i32(params.pw);
                if (ih < 0 || iw < 0 || ih >= i32(params.h) || iw >= i32(params.w)) {
                    continue;
                }
                let xv = rd_i8(params.x_off + in_chan + u32(ih) * params.w + u32(iw))
                    - params.x_zp;
                let wv = rd_i8(params.w_off + wt_chan + ki * params.kw + kj) - params.w_zp;
                acc = acc + xv * wv;
            }
        }
    }
    var r = i32(round_half_away(f32(acc) * params.mult)) + params.out_zp;
    r = clamp(r, -128, 127);
    return u32(r) & 0xffu;
}

@compute @workgroup_size(64)
fn q_conv2d(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = gid.x + gid.y * ngs.x * 64u;
    let total = params.batch * params.c_out * params.h_out * params.w_out;
    let words = (total + 3u) / 4u;
    if (t >= words) { return; }
    let base = params.out_off + t * 4u;
    let existing = bitcast<u32>(arena[base / 4u]);
    var out = vec4<u32>(0u, 0u, 0u, 0u);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let i = t * 4u + lane;
        if (i >= total) {
            out[lane] = (existing >> (lane * 8u)) & 0xffu;
            continue;
        }
        out[lane] = q_conv2d_one(i);
    }
    wr_word(base, out);
}
