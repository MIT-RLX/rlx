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

// 3D NCDHW transposed conv (output-centric gather).
// Weight: [c_in, c_out/groups, kd, kh, kw] (PyTorch ConvTranspose3d).

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
fn conv_transpose3d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
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
    let oc_off = co % c_out_per_g;
    let ci_start = g * c_in_per_g;

    var acc: f32 = 0.0;
    for (var kz: u32 = 0u; kz < params.kd; kz = kz + 1u) {
        let num_d = i32(do_) + i32(params.pd) - i32(kz * params.dd);
        if (num_d < 0 || (num_d % i32(params.sd)) != 0) { continue; }
        let id = u32(num_d / i32(params.sd));
        if (id >= params.d) { continue; }
        for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
            let num_h = i32(ho) + i32(params.ph) - i32(ky * params.dh);
            if (num_h < 0 || (num_h % i32(params.sh)) != 0) { continue; }
            let ih = u32(num_h / i32(params.sh));
            if (ih >= params.h) { continue; }
            for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
                let num_w = i32(wo) + i32(params.pw) - i32(kx * params.dw);
                if (num_w < 0 || (num_w % i32(params.sw)) != 0) { continue; }
                let iw = u32(num_w / i32(params.sw));
                if (iw >= params.w) { continue; }
                for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
                    let ci = ci_start + ci_off;
                    let in_idx = (((nn * params.c_in + ci) * params.d + id)
                                  * params.h + ih) * params.w + iw;
                    let w_idx = (((ci * c_out_per_g + oc_off) * params.kd + kz)
                                 * params.kh + ky) * params.kw + kx;
                    acc = acc + arena[params.in_off + in_idx] * arena[params.w_off + w_idx];
                }
            }
        }
    }
    arena[params.out_off + i] = acc;
}
