// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 2D NCHW conv fused with the bias + activation (+ optional residual) epilogue:
//   y = act(conv(x, w) + bias[c] + residual)
// Native replacement for the CPU host round-trip wgpu used for FusedConvBiasAct.
// The conv body is identical to `conv2d.wgsl` (bit-for-bit); only the write
// applies the epilogue. act ids match wgpu `activation_op_id` (opcode_relu_first):
//   0 relu · 1 sigmoid · 2 tanh · 5 sqrt · 7 neg · 8 abs · 9 gelu · 10 silu ·
//   11 gelu_approx · 0xFFFF = identity.

const TILE: u32 = 4u;

struct Params {
    n: u32, c_in: u32, c_out: u32,
    h: u32, w: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
    ph: u32, pw: u32,
    dh: u32, dw: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
    has_bias: u32, bias_off: u32,
    act_id: u32,
    has_residual: u32, residual_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn erf_approx(x: f32) -> f32 {
    // Abramowitz–Stegun 7.1.26 (matches conv_bias_act_epilogue.comp).
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
    return select(-y, y, x >= 0.0);
}

fn gelu_erf(v: f32) -> f32 {
    return 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476));
}

// Ids match wgpu `activation_op_id` (opcode_relu_first) + the matmul epilogue:
//   0 relu · 1 sigmoid · 2 tanh · 5 sqrt · 7 neg · 8 abs · 9 gelu · 10 silu ·
//   11 gelu_approx · 0xFFFF = identity.
fn apply_act(v_in: f32, id: u32) -> f32 {
    var v = v_in;
    if (id == 0xFFFFu) { return v; }
    switch (id) {
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(64)
fn fused_conv_bias_act(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let spatial = params.h_out * params.w_out;
    let sp_tiles = (spatial + TILE - 1u) / TILE;
    let total = params.n * params.c_out * sp_tiles;
    let tid = gid.x + gid.y * ngs.x * 64u;
    if (tid >= total) { return; }
    let sp_tile = tid % sp_tiles;
    let q2 = tid / sp_tiles;
    let co = q2 % params.c_out;
    let nn = q2 / params.c_out;
    let sp_base = sp_tile * TILE;

    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    var ho_a = array<u32, 4>(0u, 0u, 0u, 0u);
    var wo_a = array<u32, 4>(0u, 0u, 0u, 0u);
    var lanes: u32 = 0u;
    for (var t: u32 = 0u; t < TILE; t = t + 1u) {
        let sp = sp_base + t;
        if (sp >= spatial) { break; }
        ho_a[t] = sp / params.w_out;
        wo_a[t] = sp % params.w_out;
        lanes = lanes + 1u;
    }

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        let in_base_ch = (nn * params.c_in + ci) * params.h;
        for (var kr: u32 = 0u; kr < params.kh; kr = kr + 1u) {
            for (var kc: u32 = 0u; kc < params.kw; kc = kc + 1u) {
                let w_idx = ((co * c_in_per_g + ci_off) * params.kh + kr) * params.kw + kc;
                let wv = arena[params.w_off + w_idx];
                for (var t: u32 = 0u; t < lanes; t = t + 1u) {
                    let in_r_signed = i32(ho_a[t] * params.sh + kr * params.dh) - i32(params.ph);
                    let in_c_signed = i32(wo_a[t] * params.sw + kc * params.dw) - i32(params.pw);
                    if (in_r_signed < 0 || in_c_signed < 0
                        || in_r_signed >= i32(params.h)
                        || in_c_signed >= i32(params.w)) {
                        continue;
                    }
                    let in_idx = (in_base_ch + u32(in_r_signed)) * params.w + u32(in_c_signed);
                    acc[t] = acc[t] + arena[params.in_off + in_idx] * wv;
                }
            }
        }
    }

    // Epilogue: y = act(conv + bias[c] + residual).
    let out_base_ch = (nn * params.c_out + co) * spatial;
    let bias = select(0.0, arena[params.bias_off + co], params.has_bias != 0u);
    for (var t: u32 = 0u; t < lanes; t = t + 1u) {
        let out_idx = params.out_off + out_base_ch + sp_base + t;
        var v = acc[t] + bias;
        if (params.has_residual != 0u) {
            v = v + arena[params.residual_off + out_base_ch + sp_base + t];
        }
        arena[out_idx] = apply_act(v, params.act_id);
    }
}
