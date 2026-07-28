// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// GroupNorm (NCHW) backward. Matches CPU `training_bwd::group_norm_backward_*`.
// One thread per (batch, group) for dx; single-thread serial for dgamma/dbeta.

struct Params {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    num_groups: u32,
    eps_bits: u32,
    x_off: u32,
    gamma_off: u32,
    dy_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(1)
fn group_norm_bwd_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ng = gid.x;
    if (ng >= params.n * params.num_groups) { return; }
    let bn = ng / params.num_groups;
    let g = ng % params.num_groups;
    let spatial = params.h * params.w;
    let plane = params.c * spatial;
    let cpg = params.c / params.num_groups;
    let count = f32(cpg * spatial);
    let n_inv = 1.0 / count;
    let eps = bitcast<f32>(params.eps_bits);
    let c0 = g * cpg;
    let b_base = bn * plane;

    var mean: f32 = 0.0;
    for (var c: u32 = 0u; c < cpg; c = c + 1u) {
        let base = params.x_off + b_base + (c0 + c) * spatial;
        for (var s: u32 = 0u; s < spatial; s = s + 1u) {
            mean = mean + arena[base + s];
        }
    }
    mean = mean * n_inv;

    var var_: f32 = 0.0;
    for (var c: u32 = 0u; c < cpg; c = c + 1u) {
        let base = params.x_off + b_base + (c0 + c) * spatial;
        for (var s: u32 = 0u; s < spatial; s = s + 1u) {
            let d = arena[base + s] - mean;
            var_ = var_ + d * d;
        }
    }
    var_ = var_ * n_inv;
    let inv_std = inverseSqrt(var_ + eps);

    var s_sy: f32 = 0.0;
    var s_sxh: f32 = 0.0;
    for (var c: u32 = 0u; c < cpg; c = c + 1u) {
        let gi = c0 + c;
        let gamm = arena[params.gamma_off + gi];
        let x_base = params.x_off + b_base + gi * spatial;
        let dy_base = params.dy_off + b_base + gi * spatial;
        for (var s: u32 = 0u; s < spatial; s = s + 1u) {
            let xh = (arena[x_base + s] - mean) * inv_std;
            let sy = arena[dy_base + s] * gamm;
            s_sy = s_sy + sy;
            s_sxh = s_sxh + sy * xh;
        }
    }
    let m_sy = s_sy * n_inv;
    let m_sxh = s_sxh * n_inv;

    for (var c: u32 = 0u; c < cpg; c = c + 1u) {
        let gi = c0 + c;
        let gamm = arena[params.gamma_off + gi];
        let x_base = params.x_off + b_base + gi * spatial;
        let dy_base = params.dy_off + b_base + gi * spatial;
        let out_base = params.out_off + b_base + gi * spatial;
        for (var s: u32 = 0u; s < spatial; s = s + 1u) {
            let xh = (arena[x_base + s] - mean) * inv_std;
            let sy = arena[dy_base + s] * gamm;
            arena[out_base + s] = inv_std * (sy - m_sy - xh * m_sxh);
        }
    }
}

@compute @workgroup_size(1)
fn group_norm_bwd_gamma(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let spatial = params.h * params.w;
    let plane = params.c * spatial;
    let cpg = params.c / params.num_groups;
    let count = f32(cpg * spatial);
    let n_inv = 1.0 / count;
    let eps = bitcast<f32>(params.eps_bits);

    for (var ch: u32 = 0u; ch < params.c; ch = ch + 1u) {
        arena[params.out_off + ch] = 0.0;
    }

    for (var bn: u32 = 0u; bn < params.n; bn = bn + 1u) {
        let b_base = bn * plane;
        for (var g: u32 = 0u; g < params.num_groups; g = g + 1u) {
            let c0 = g * cpg;
            var mean: f32 = 0.0;
            for (var c: u32 = 0u; c < cpg; c = c + 1u) {
                let base = params.x_off + b_base + (c0 + c) * spatial;
                for (var s: u32 = 0u; s < spatial; s = s + 1u) {
                    mean = mean + arena[base + s];
                }
            }
            mean = mean * n_inv;
            var var_: f32 = 0.0;
            for (var c: u32 = 0u; c < cpg; c = c + 1u) {
                let base = params.x_off + b_base + (c0 + c) * spatial;
                for (var s: u32 = 0u; s < spatial; s = s + 1u) {
                    let d = arena[base + s] - mean;
                    var_ = var_ + d * d;
                }
            }
            var_ = var_ * n_inv;
            let inv_std = inverseSqrt(var_ + eps);
            for (var c: u32 = 0u; c < cpg; c = c + 1u) {
                let gi = c0 + c;
                let x_base = params.x_off + b_base + gi * spatial;
                let dy_base = params.dy_off + b_base + gi * spatial;
                var acc: f32 = arena[params.out_off + gi];
                for (var s: u32 = 0u; s < spatial; s = s + 1u) {
                    let xh = (arena[x_base + s] - mean) * inv_std;
                    acc = acc + arena[dy_base + s] * xh;
                }
                arena[params.out_off + gi] = acc;
            }
        }
    }
}

@compute @workgroup_size(1)
fn group_norm_bwd_beta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let spatial = params.h * params.w;
    let plane = params.c * spatial;
    for (var ch: u32 = 0u; ch < params.c; ch = ch + 1u) {
        arena[params.out_off + ch] = 0.0;
    }
    for (var bn: u32 = 0u; bn < params.n; bn = bn + 1u) {
        let b_base = bn * plane;
        for (var ch: u32 = 0u; ch < params.c; ch = ch + 1u) {
            let dy_base = params.dy_off + b_base + ch * spatial;
            var acc: f32 = arena[params.out_off + ch];
            for (var s: u32 = 0u; s < spatial; s = s + 1u) {
                acc = acc + arena[dy_base + s];
            }
            arena[params.out_off + ch] = acc;
        }
    }
}
