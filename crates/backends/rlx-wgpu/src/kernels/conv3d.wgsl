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

// 3D NCDHW conv. Weight: [c_out, c_in/groups, kd, kh, kw].

struct Params {
    n: u32, c_in: u32, c_out: u32,
    d: u32, h: u32, w: u32,
    d_out: u32, h_out: u32, w_out: u32,
    kd: u32, kh: u32, kw: u32,
    sd: u32, sh: u32, sw: u32,
    pd: u32, ph: u32, pw: u32,
    dd: u32, dh: u32, dw: u32,
    groups: u32,
    in_off: u32, w_off: u32, out_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn conv3d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c_out * params.d_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let co = q3 % params.c_out;
    let nn = q3 / params.c_out;

    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    var acc: f32 = 0.0;
    for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
          for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
                let in_d_s = i32(do_ * params.sd + kd * params.dd) - i32(params.pd);
                let in_h_s = i32(ho  * params.sh + kh * params.dh) - i32(params.ph);
                let in_w_s = i32(wo  * params.sw + kw * params.dw) - i32(params.pw);
                if (in_d_s < 0 || in_h_s < 0 || in_w_s < 0
                    || in_d_s >= i32(params.d)
                    || in_h_s >= i32(params.h)
                    || in_w_s >= i32(params.w)) {
                    continue;
                }
                let id = u32(in_d_s);
                let ih = u32(in_h_s);
                let iw = u32(in_w_s);
                let in_idx = (((nn * params.c_in + ci) * params.d + id)
                              * params.h + ih) * params.w + iw;
                let w_idx  = (((co * c_in_per_g + ci_off) * params.kd + kd)
                              * params.kh + kh) * params.kw + kw;
                acc = acc + arena[params.in_off + in_idx] * arena[params.w_off + w_idx];
            }
          }
        }
    }
    arena[params.out_off + i] = acc;
}
