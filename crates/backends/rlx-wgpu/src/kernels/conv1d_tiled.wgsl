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

// Fast 1D convolution specialized for the `one_d` NCHW-relabeled case
// (weight `[c_out, c_in, kh, 1]`, input `[1, c_in, L, 1]`, `w == kw == 1`,
// `groups == 1`, `N == 1`).
//
// The vocoder-decoder convs reach `conv2d.wgsl` through the `one_d` relabel,
// but that general kernel carries 2D machinery the 1D case never needs: a
// per-lane `(ho, wo)` div/mod decode, a ragged-tile `lanes` loop, and an inner
// `kc` loop over the singleton width. This kernel drops all of it — one thread
// per output element, a tight `(ci, kr)` reduction with a single bounds test on
// the length axis — which lifts occupancy enough to run ~4-5× faster on the
// Apple GPU for the decoder's shapes.
//
// The reduction nests `ci` (outer) then `kr` (inner) exactly as `conv2d.wgsl`
// and rlx-cpu, so the accumulator sums in the same order → bit-identical f32.
//
// Reuses `Conv2dParams` (h == L, w == 1, h_out == L_out, kh == k, kw == 1).

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
fn conv1d_tiled(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let l_out = params.h_out;              // length axis (w_out == 1, kw == 1)
    let total = params.c_out * l_out;      // N == 1
    let idx = gid.x + gid.y * ngs.x * 64u;
    if (idx >= total) { return; }

    let lo = idx % l_out;                  // output length position
    let co = idx / l_out;                  // output channel

    let h_i = i32(params.h);
    let base_in = i32(lo * params.sh) - i32(params.ph);
    let w_row = co * params.c_in * params.kh;

    var acc: f32 = 0.0;
    for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
        let in_ch_base = params.in_off + ci * params.h;
        let w_base = params.w_off + w_row + ci * params.kh;
        for (var kr: u32 = 0u; kr < params.kh; kr = kr + 1u) {
            let in_pos = base_in + i32(kr * params.dh);
            if (in_pos >= 0 && in_pos < h_i) {
                acc = acc + arena[in_ch_base + u32(in_pos)] * arena[w_base + kr];
            }
        }
    }

    arena[params.out_off + idx] = acc;
}
