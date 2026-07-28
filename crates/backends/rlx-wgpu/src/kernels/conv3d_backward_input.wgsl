// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Conv3d backward-input (NCDHW gather). Weight [C_out, C_in/groups, kD, kH, kW].

struct Params {
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    dy_off: u32,
    w_off: u32,
    dx_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn conv3d_backward_input(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.c_in * params.d * params.h * params.w;
    if (i >= total) {
        return;
    }
    let iw = i % params.w;
    let q1 = i / params.w;
    let ih = q1 % params.h;
    let q2 = q1 / params.h;
    let id = q2 % params.d;
    let q3 = q2 / params.d;
    let ci = q3 % params.c_in;
    let nn = q3 / params.c_in;

    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = ci / c_in_per_g;
    let ci_off = ci - g * c_in_per_g;
    let co_start = g * c_out_per_g;

    var acc = 0.0;
    for (var kz = 0u; kz < params.kd; kz = kz + 1u) {
        let num_d = i32(id) + i32(params.pd) - i32(kz * params.dd);
        if (num_d < 0 || (num_d % i32(params.sd)) != 0) { continue; }
        let do_ = u32(num_d / i32(params.sd));
        if (do_ >= params.d_out) { continue; }
        for (var ki = 0u; ki < params.kh; ki = ki + 1u) {
            let num_h = i32(ih) + i32(params.ph) - i32(ki * params.dh);
            if (num_h < 0 || (num_h % i32(params.sh)) != 0) { continue; }
            let ho = u32(num_h / i32(params.sh));
            if (ho >= params.h_out) { continue; }
            for (var kj = 0u; kj < params.kw; kj = kj + 1u) {
                let num_w = i32(iw) + i32(params.pw) - i32(kj * params.dw);
                if (num_w < 0 || (num_w % i32(params.sw)) != 0) { continue; }
                let wo = u32(num_w / i32(params.sw));
                if (wo >= params.w_out) { continue; }
                for (var co_off = 0u; co_off < c_out_per_g; co_off = co_off + 1u) {
                    let co = co_start + co_off;
                    let dyv = arena[params.dy_off + ((((nn * params.c_out + co) * params.d_out + do_) * params.h_out + ho) * params.w_out + wo)];
                    let wv = arena[params.w_off + ((((co * c_in_per_g + ci_off) * params.kd + kz) * params.kh + ki) * params.kw + kj)];
                    acc = acc + dyv * wv;
                }
            }
        }
    }
    arena[params.dx_off + i] = acc;
}
