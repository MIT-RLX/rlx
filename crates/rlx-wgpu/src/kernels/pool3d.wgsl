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

// 3D NCDHW pool. Op-kind selector matches Pool2D / Pool1D.

struct Params {
    n: u32, c: u32, d: u32, h: u32, w: u32,
    d_out: u32, h_out: u32, w_out: u32,
    kd: u32, kh: u32, kw: u32,
    sd: u32, sh: u32, sw: u32,
    pd: u32, ph: u32, pw: u32,
    op: u32,
    in_off: u32, out_off: u32,
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn pad_value(op: u32) -> f32 {
    switch (op) {
        case 2u: { return -3.4e38; }
        case 3u: { return  3.4e38; }
        case 4u: { return  1.0; }
        default: { return  0.0; }
    }
}

@compute @workgroup_size(64)
fn pool3d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.n * params.c * params.d_out * params.h_out * params.w_out;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let wo = i % params.w_out;
    let q1 = i / params.w_out;
    let ho = q1 % params.h_out;
    let q2 = q1 / params.h_out;
    let do_ = q2 % params.d_out;
    let q3 = q2 / params.d_out;
    let cc = q3 % params.c;
    let nn = q3 / params.c;

    var acc: f32 = pad_value(params.op);
    var have_init: bool = false;
    for (var kd: u32 = 0u; kd < params.kd; kd = kd + 1u) {
      for (var kh: u32 = 0u; kh < params.kh; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kw; kw = kw + 1u) {
            let in_d_s = i32(do_ * params.sd + kd) - i32(params.pd);
            let in_h_s = i32(ho  * params.sh + kh) - i32(params.ph);
            let in_w_s = i32(wo  * params.sw + kw) - i32(params.pw);
            var v: f32;
            if (in_d_s < 0 || in_h_s < 0 || in_w_s < 0
                || in_d_s >= i32(params.d)
                || in_h_s >= i32(params.h)
                || in_w_s >= i32(params.w)) {
                v = pad_value(params.op);
            } else {
                let id = u32(in_d_s);
                let ih = u32(in_h_s);
                let iw = u32(in_w_s);
                let idx = (((nn * params.c + cc) * params.d + id) * params.h + ih) * params.w + iw;
                v = arena[params.in_off + idx];
            }
            if (!have_init) { acc = v; have_init = true; }
            else {
                switch (params.op) {
                    case 0u, 1u: { acc = acc + v; }
                    case 2u:     { acc = max(acc, v); }
                    case 3u:     { acc = min(acc, v); }
                    case 4u:     { acc = acc * v; }
                    default:     {}
                }
            }
        }
      }
    }
    if (params.op == 1u) { acc = acc / f32(params.kd * params.kh * params.kw); }
    arena[params.out_off + i] = acc;
}
