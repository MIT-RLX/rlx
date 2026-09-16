// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Real INT8 `Op::QMatMul`: x[M,K] i8 · w[K,N] i8 + bias[N] i32 → out[M,N] i8,
// with i32 accumulation and a float requantize. Mirrors
// `rlx_cpu::thunk::ops::quant::exec_q_mat_mul` and the Vulkan `q_matmul.comp`.
//
// i8 tensors are byte-packed in the f32-uniform arena, so `x`/`w`/`out` are byte
// offsets. `bias` is not: an I32 tensor keeps a full f32-sized slot on these
// arenas and holds the integer as an f32 *value* (the CPU arena holds raw i32
// bits instead — each is right for its own arena, which is why the CPU kernel
// reads `*const i32` here and this one truncates a float).
//
// One invocation owns a whole output WORD. WGSL has no byte store, so a
// per-element launch would have four invocations read-modify-writing the same
// u32 and three of every four results would be lost — which is exactly what the
// Vulkan twin did until this session.
//
// Requantize rounds half AWAY from zero (Rust's `f32::round`), not WGSL
// `round`'s ties-to-even.

struct Params {
    m: u32,
    k: u32,
    n: u32,
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
    _pad0: u32,
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

fn q_matmul_one(idx: u32) -> u32 {
    let mi = idx / params.n;
    let ni = idx - mi * params.n;
    var acc = i32(trunc(arena[params.bias_off + ni]));
    for (var ki = 0u; ki < params.k; ki = ki + 1u) {
        let xv = rd_i8(params.x_off + mi * params.k + ki) - params.x_zp;
        let wv = rd_i8(params.w_off + ki * params.n + ni) - params.w_zp;
        acc = acc + xv * wv;
    }
    var r = i32(round_half_away(f32(acc) * params.mult)) + params.out_zp;
    r = clamp(r, -128, 127);
    return u32(r) & 0xffu;
}

@compute @workgroup_size(64)
fn q_matmul(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = gid.x + gid.y * ngs.x * 64u;
    let total = params.m * params.n;
    let words = (total + 3u) / 4u;
    if (t >= words) { return; }
    let base = params.out_off + t * 4u;
    let existing = bitcast<u32>(arena[base / 4u]);
    var out = vec4<u32>(0u, 0u, 0u, 0u);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let idx = t * 4u + lane;
        if (idx >= total) {
            out[lane] = (existing >> (lane * 8u)) & 0xffu;
            continue;
        }
        out[lane] = q_matmul_one(idx);
    }
    wr_word(base, out);
}
