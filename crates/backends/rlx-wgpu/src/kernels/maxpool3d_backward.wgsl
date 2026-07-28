// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// MaxPool3d backward (NCDHW). One thread per input element.

struct Params {
    n: u32,
    c: u32,
    d: u32,
    h: u32,
    w: u32,
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
    x_off: u32,
    dy_off: u32,
    dx_off: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn maxpool3d_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n * params.c * params.d * params.h * params.w;
    if (idx >= total) {
        return;
    }

    let iw = idx % params.w;
    let q1 = idx / params.w;
    let ih = q1 % params.h;
    let q2 = q1 / params.h;
    let id = q2 % params.d;
    let q3 = q2 / params.d;
    let cc = q3 % params.c;
    let nn = q3 / params.c;
    let base_nc = (nn * params.c + cc) * params.d * params.h * params.w;

    var do_lo = i32(id) + i32(params.pd) - i32(params.kd) + 1;
    if (do_lo <= 0) {
        do_lo = 0;
    } else {
        do_lo = (do_lo + i32(params.sd) - 1) / i32(params.sd);
    }
    let do_hi = (i32(id) + i32(params.pd)) / i32(params.sd);
    var ho_lo = i32(ih) + i32(params.ph) - i32(params.kh) + 1;
    if (ho_lo <= 0) {
        ho_lo = 0;
    } else {
        ho_lo = (ho_lo + i32(params.sh) - 1) / i32(params.sh);
    }
    let ho_hi = (i32(ih) + i32(params.ph)) / i32(params.sh);
    var wo_lo = i32(iw) + i32(params.pw) - i32(params.kw) + 1;
    if (wo_lo <= 0) {
        wo_lo = 0;
    } else {
        wo_lo = (wo_lo + i32(params.sw) - 1) / i32(params.sw);
    }
    let wo_hi = (i32(iw) + i32(params.pw)) / i32(params.sw);

    var acc = 0.0;
    for (var do_ = do_lo; do_ <= do_hi && do_ < i32(params.d_out); do_ = do_ + 1) {
        let dstart = do_ * i32(params.sd) - i32(params.pd);
        for (var ho = ho_lo; ho <= ho_hi && ho < i32(params.h_out); ho = ho + 1) {
            let hstart = ho * i32(params.sh) - i32(params.ph);
            for (var wo = wo_lo; wo <= wo_hi && wo < i32(params.w_out); wo = wo + 1) {
                let wstart = wo * i32(params.sw) - i32(params.pw);
                var best = -3.402823466e+38;
                var best_idx = -1;
                for (var kz = 0u; kz < params.kd; kz = kz + 1u) {
                    let irz = dstart + i32(kz);
                    if (irz < 0 || irz >= i32(params.d)) { continue; }
                    for (var i = 0u; i < params.kh; i = i + 1u) {
                        let ir = hstart + i32(i);
                        if (ir < 0 || ir >= i32(params.h)) { continue; }
                        for (var j = 0u; j < params.kw; j = j + 1u) {
                            let ic = wstart + i32(j);
                            if (ic < 0 || ic >= i32(params.w)) { continue; }
                            let id3 = base_nc + (u32(irz) * params.h + u32(ir)) * params.w + u32(ic);
                            let v = arena[params.x_off + id3];
                            if (v > best) {
                                best = v;
                                best_idx = i32(id3);
                            }
                        }
                    }
                }
                if (best_idx == i32(idx)) {
                    let dy_i = ((((nn * params.c + cc) * params.d_out + u32(do_)) * params.h_out + u32(ho)) * params.w_out + u32(wo));
                    acc = acc + arena[params.dy_off + dy_i];
                }
            }
        }
    }
    arena[params.dx_off + idx] = acc;
}
