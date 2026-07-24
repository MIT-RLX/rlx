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

// Conv3d backward-weight (NCDHW). One thread per dw element.

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
    x_off: u32,
    dy_off: u32,
    dw_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn conv3d_backward_weight(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c_in_per_g = params.c_in / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let total = params.c_out * c_in_per_g * params.kd * params.kh * params.kw;
    if (i >= total) {
        return;
    }
    let kj = i % params.kw;
    let q1 = i / params.kw;
    let ki = q1 % params.kh;
    let q2 = q1 / params.kh;
    let kz = q2 % params.kd;
    let q3 = q2 / params.kd;
    let ci_off = q3 % c_in_per_g;
    let co = q3 / c_in_per_g;
    let g = co / c_out_per_g;
    let ci = g * c_in_per_g + ci_off;

    var acc = 0.0;
    for (var nn = 0u; nn < params.n; nn = nn + 1u) {
        for (var do_ = 0u; do_ < params.d_out; do_ = do_ + 1u) {
            let id = i32(do_ * params.sd + kz * params.dd) - i32(params.pd);
            if (id < 0 || id >= i32(params.d)) { continue; }
            for (var ho = 0u; ho < params.h_out; ho = ho + 1u) {
                let ih = i32(ho * params.sh + ki * params.dh) - i32(params.ph);
                if (ih < 0 || ih >= i32(params.h)) { continue; }
                for (var wo = 0u; wo < params.w_out; wo = wo + 1u) {
                    let iw = i32(wo * params.sw + kj * params.dw) - i32(params.pw);
                    if (iw < 0 || iw >= i32(params.w)) { continue; }
                    let dyv = arena[params.dy_off + ((((nn * params.c_out + co) * params.d_out + do_) * params.h_out + ho) * params.w_out + wo)];
                    let xv = arena[params.x_off + ((((nn * params.c_in + ci) * params.d + u32(id)) * params.h + u32(ih)) * params.w + u32(iw))];
                    acc = acc + dyv * xv;
                }
            }
        }
    }
    arena[params.dw_off + i] = acc;
}
