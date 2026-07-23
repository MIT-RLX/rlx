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

// MaxPool2d backward (NCHW). One thread per input spatial location (n,c,ih,iw);
// accumulates dy from every pool window whose argmax lands on this pixel.

struct Params {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    x_off: u32,
    dy_off: u32,
    dx_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(8, 8, 1)
fn maxpool2d_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iw = gid.x;
    let ih = gid.y;
    let nc = gid.z;
    if (nc >= params.n * params.c || ih >= params.h || iw >= params.w) {
        return;
    }

    let p_h = i32(ih) + i32(params.ph);
    let p_w = i32(iw) + i32(params.pw);
    var oh_max = p_h / i32(params.sh);
    var ow_max = p_w / i32(params.sw);
    if (oh_max >= i32(params.h_out)) { oh_max = i32(params.h_out) - 1; }
    if (ow_max >= i32(params.w_out)) { ow_max = i32(params.w_out) - 1; }
    var oh_min = 0;
    if (p_h - i32(params.kh) >= 0) {
        oh_min = (p_h - i32(params.kh)) / i32(params.sh) + 1;
    }
    var ow_min = 0;
    if (p_w - i32(params.kw) >= 0) {
        ow_min = (p_w - i32(params.kw)) / i32(params.sw) + 1;
    }

    let in_chan = nc * params.h * params.w;
    let out_chan = nc * params.h_out * params.w_out;
    var acc = 0.0;
    for (var oh = oh_min; oh <= oh_max; oh = oh + 1) {
        for (var ow = ow_min; ow <= ow_max; ow = ow + 1) {
            var best_v = -3.402823466e+38;
            var best_h = -1;
            var best_w = -1;
            for (var ki = 0u; ki < params.kh; ki = ki + 1u) {
                let hh = oh * i32(params.sh) + i32(ki) - i32(params.ph);
                if (hh < 0 || hh >= i32(params.h)) { continue; }
                for (var kj = 0u; kj < params.kw; kj = kj + 1u) {
                    let ww = ow * i32(params.sw) + i32(kj) - i32(params.pw);
                    if (ww < 0 || ww >= i32(params.w)) { continue; }
                    let v = arena[params.x_off + in_chan + u32(hh) * params.w + u32(ww)];
                    if (v > best_v) {
                        best_v = v;
                        best_h = hh;
                        best_w = ww;
                    }
                }
            }
            if (best_h == i32(ih) && best_w == i32(iw)) {
                acc = acc + arena[params.dy_off + out_chan + u32(oh) * params.w_out + u32(ow)];
            }
        }
    }
    arena[params.dx_off + in_chan + ih * params.w + iw] = acc;
}
