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

// GPU im2col for the NCHW convolution im2col+GEMM path (N == 1, groups == 1).
//
// Gathers the receptive field of every output position into a column matrix
// `col[k_total, spatial]` laid out **row-major** so it feeds the standard
// `matmul` kernel directly as the B operand:
//
//   A = weight[c_out, c_in*kh*kw]   (row-major, exactly as stored)
//   B = col[c_in*kh*kw, spatial]    (this kernel's output)
//   C = out[c_out, spatial]         (matmul writes NCHW output for N=1)
//
// The flattened reduction index `k = ((ci*kh)+kr)*kw + kc` matches the
// (ci, kr, kc) nesting of the direct conv kernel, so the subsequent GEMM
// accumulates in the same order → bit-identical f32 result.
//
// One thread per col element (`k_total * spatial` threads). Out-of-bounds
// receptive-field taps (padding / dilation past the edge) store 0.0.

struct Params {
    c_in: u32,
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
    dh: u32,
    dw: u32,
    in_off: u32,
    col_off: u32,
    k_total: u32,   // c_in * kh * kw
    spatial: u32,   // h_out * w_out
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn im2col2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let total = params.k_total * params.spatial;
    let idx = gid.x + gid.y * ngs.x * 64u;
    if (idx >= total) { return; }

    let s = idx % params.spatial;   // output spatial position (ho*w_out + wo)
    let k = idx / params.spatial;   // flattened (ci, kr, kc)

    let kc = k % params.kw;
    let tmp = k / params.kw;
    let kr = tmp % params.kh;
    let ci = tmp / params.kh;

    let ho = s / params.w_out;
    let wo = s % params.w_out;

    let in_r = i32(ho * params.sh + kr * params.dh) - i32(params.ph);
    let in_c = i32(wo * params.sw + kc * params.dw) - i32(params.pw);

    var v: f32 = 0.0;
    if (in_r >= 0 && in_c >= 0 && in_r < i32(params.h) && in_c < i32(params.w)) {
        // N == 1: element (0, ci, in_r, in_c) of [1, c_in, h, w].
        let in_idx = (ci * params.h + u32(in_r)) * params.w + u32(in_c);
        v = arena[params.in_off + in_idx];
    }
    arena[params.col_off + idx] = v;
}
