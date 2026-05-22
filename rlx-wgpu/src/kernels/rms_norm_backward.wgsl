// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
// RMSNorm backward (row = last-axis slice). wrt: 0=dx, 1=dgamma, 2=dbeta.

struct Params {
    outer: u32,
    inner: u32,
    x_off: u32,
    gamma_off: u32,
    beta_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    wrt: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(1)
fn rms_norm_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.wrt != 0u || params.inner == 0u) { return; }
    let row = gid.x;
    if (row >= params.outer) { return; }
    let x_base = params.x_off + row * params.inner;
    let dy_base = params.dy_off + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);

    var dot: f32 = 0.0;
    var sumsq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i++) {
        let xv = arena[x_base + i];
        let gv = arena[params.gamma_off + i];
        let dyv = arena[dy_base + i];
        dot = dot + dyv * gv * xv;
        sumsq = sumsq + xv * xv;
    }
    dot = dot * n_inv;
    let inv_r = inverseSqrt(sumsq * n_inv + eps);
    let inv_r3 = inv_r * inv_r * inv_r;
    for (var i: u32 = 0u; i < params.inner; i++) {
        let xv = arena[x_base + i];
        let gv = arena[params.gamma_off + i];
        let dyv = arena[dy_base + i];
        let term = gv * dyv - xv * dot * inv_r3;
        arena[out_base + i] = term * inv_r;
    }
}

@compute @workgroup_size(1)
fn rms_norm_bwd_param(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || params.inner == 0u) { return; }
    let eps = bitcast<f32>(params.eps_bits);
    let n_inv = 1.0 / f32(params.inner);
    for (var i: u32 = 0u; i < params.inner; i++) {
        arena[params.out_off + i] = 0.0;
    }
    for (var row: u32 = 0u; row < params.outer; row++) {
        let x_base = params.x_off + row * params.inner;
        let dy_base = params.dy_off + row * params.inner;
        var sumsq: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i++) {
            let xv = arena[x_base + i];
            sumsq = sumsq + xv * xv;
        }
        let inv_r = inverseSqrt(sumsq * n_inv + eps);
        if (params.wrt == 1u) {
            for (var i: u32 = 0u; i < params.inner; i++) {
                let xv = arena[x_base + i];
                let dyv = arena[dy_base + i];
                arena[params.out_off + i] = arena[params.out_off + i] + dyv * xv * inv_r;
            }
        } else {
            for (var i: u32 = 0u; i < params.inner; i++) {
                let dyv = arena[dy_base + i];
                arena[params.out_off + i] = arena[params.out_off + i] + dyv;
            }
        }
    }
}
