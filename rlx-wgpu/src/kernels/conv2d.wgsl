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

// 2D NCHW convolution. One thread per output element (n, c_out,
// h_out, w_out); the thread streams over the kernel window and
// input-channel axis to accumulate the convolution.
//
// Weight layout matches rlx-cpu: [c_out, c_in/groups, kh, kw].
// Bias is not applied here; downstream Add handles it.

struct Params {
    n: u32, c_in: u32, c_out: u32,
    h: u32, w: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
    ph: u32, pw: u32,
    dh: u32, dw: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn conv2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let co = q2 % params.c_out;
    let nn = q2 / params.c_out;

    let c_in_per_g = params.c_in  / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    var acc: f32 = 0.0;
    for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        for (var kr: u32 = 0u; kr < params.kh; kr = kr + 1u) {
            for (var kc: u32 = 0u; kc < params.kw; kc = kc + 1u) {
                let in_r_signed = i32(ho * params.sh + kr * params.dh) - i32(params.ph);
                let in_c_signed = i32(wo * params.sw + kc * params.dw) - i32(params.pw);
                if (in_r_signed < 0 || in_c_signed < 0
                    || in_r_signed >= i32(params.h)
                    || in_c_signed >= i32(params.w)) {
                    continue;
                }
                let in_r = u32(in_r_signed);
                let in_c = u32(in_c_signed);
                let in_idx = ((nn * params.c_in + ci) * params.h + in_r) * params.w + in_c;
                let w_idx = ((co * c_in_per_g + ci_off) * params.kh + kr) * params.kw + kc;
                acc = acc + arena[params.in_off + in_idx] * arena[params.w_off + w_idx];
            }
        }
    }
    arena[params.out_off + i] = acc;
}
