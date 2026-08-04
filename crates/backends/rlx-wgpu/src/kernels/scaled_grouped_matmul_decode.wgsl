// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Native low-precision *grouped* (MoE) decode-GEMM for Op::ScaledGroupedMatMul
// — the wgpu MXFP4 analogue of the CUDA scaled_grouped_matmul_decode kernel.
// One thread per output C[row=token, col=out]; the token's expert picks the
// weight slab, and only that routed expert's FP4 (E2M1) codes are decoded on
// the fly (no f32 weight materialization). TN per expert:
//   out[i,j] = Σ_p decode(input[i,p])·s_in · decode(weight[e,j,p])·s_w (+ bias)
// input codes [M,K], weight codes [E,N,K], input scale [M,nblk] (E8M0 bytes),
// weight scale [E·N,nblk], expert_idx [M] f32, bias [E·N] f32 (per-expert).
// scale_mode: 0 = per-tensor f32, 1 = block E8M0 (MXFP4 default).

struct Params {
    m: u32,
    k: u32,
    n: u32,
    num_experts: u32,
    input_byte_off: u32,
    weight_byte_off: u32,
    input_scale_byte_off: u32,
    weight_scale_byte_off: u32,
    idx_off: u32,   // f32 element offset
    out_off: u32,   // f32 element offset
    bias_off: u32,  // f32 element offset
    scale_mode: u32,
    block: u32,
    has_bias: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// One packed byte out of the f32-word arena.
fn rd_byte(byte_off: u32) -> u32 {
    let word = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    return (bitcast<u32>(arena[word]) >> shift) & 0xffu;
}

// FP4 E2M1 code → f32 (matches rlx_ir FP4_E2M1_LUT bit-for-bit).
fn decode_e2m1(code: u32) -> f32 {
    let c = code & 0xfu;
    let mag = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let v = mag[c & 0x7u];
    if ((c & 0x8u) != 0u) { return -v; }
    return v;
}

// E8M0 scale byte → 2^(byte-127). Byte 0xFF is the NaN sentinel (unused here).
fn decode_e8m0(s: u32) -> f32 {
    return bitcast<f32>(s << 23u);
}

fn in_scale(row: u32, blk: u32, nblk: u32) -> f32 {
    if (params.scale_mode == 0u) {
        return arena[params.input_scale_byte_off / 4u];
    }
    return decode_e8m0(rd_byte(params.input_scale_byte_off + row * nblk + blk));
}

fn w_scale(wrow: u32, blk: u32, nblk: u32) -> f32 {
    if (params.scale_mode == 0u) {
        return arena[params.weight_scale_byte_off / 4u];
    }
    return decode_e8m0(rd_byte(params.weight_scale_byte_off + wrow * nblk + blk));
}

@compute @workgroup_size(8, 8)
fn scaled_grouped_matmul_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; // token i
    let col = gid.x; // output j
    if (row >= params.m || col >= params.n) { return; }
    let e = u32(arena[params.idx_off + row]);
    if (e >= params.num_experts) { return; }

    let k = params.k;
    var nblk: u32 = 1u;
    if (params.scale_mode != 0u) { nblk = (k + params.block - 1u) / params.block; }
    let wrow = e * params.n + col;

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < k; p = p + 1u) {
        var blk: u32 = 0u;
        if (params.scale_mode != 0u) { blk = p / params.block; }
        let ls = in_scale(row, blk, nblk);
        let ws = w_scale(wrow, blk, nblk);
        let a = decode_e2m1(rd_byte(params.input_byte_off + row * k + p)) * ls;
        let b = decode_e2m1(rd_byte(params.weight_byte_off + wrow * k + p)) * ws;
        acc = acc + a * b;
    }
    if (params.has_bias != 0u) { acc = acc + arena[params.bias_off + wrow]; }
    arena[params.out_off + row * params.n + col] = acc;
}
