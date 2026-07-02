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

// 2D NCHW convolution. Each thread computes `TILE` consecutive output spatial
// positions (flattened `h_out*w_out`) for one `(n, c_out)`, accumulating in
// `TILE` registers. The kernel weight `w[co, ci, kr, kc]` is loaded **once** per
// (ci, kr, kc) and reused across all `TILE` outputs, cutting weight
// global-memory traffic by `TILE×` (the naive one-thread-per-output form re-read
// every weight `h_out*w_out` times). Each accumulator sums over (ci, kr, kc) in
// the same order as the scalar form, so results are bit-identical.
//
// The host dispatches `ceil(n*c_out*h_out*w_out / TILE)` threads; `TILE` MUST
// match `CONV2D_TILE` in backend.rs.
//
// Weight layout matches rlx-cpu: [c_out, c_in/groups, kh, kw].
// Bias is not applied here; downstream Add handles it.

const TILE: u32 = 4u;

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
    let spatial = params.h_out * params.w_out;
    let sp_tiles = (spatial + TILE - 1u) / TILE;
    let total = params.n * params.c_out * sp_tiles;
    let tid = gid.x + gid.y * ngs.x * 64u;
    if (tid >= total) { return; }
    let sp_tile = tid % sp_tiles;
    let q2 = tid / sp_tiles;
    let co = q2 % params.c_out;
    let nn = q2 / params.c_out;
    let sp_base = sp_tile * TILE;

    let c_in_per_g = params.c_in  / params.groups;
    let c_out_per_g = params.c_out / params.groups;
    let g = co / c_out_per_g;
    let ci_start = g * c_in_per_g;

    // Decode each tile lane's output (ho, wo) ONCE (the div/mod by w_out is the
    // expensive op — never put it inside the reduction loop). `lanes` ≤ TILE
    // handles the ragged last tile so the inner loop can run unconditionally.
    var ho_a = array<u32, 4>(0u, 0u, 0u, 0u);
    var wo_a = array<u32, 4>(0u, 0u, 0u, 0u);
    var lanes: u32 = 0u;
    for (var t: u32 = 0u; t < TILE; t = t + 1u) {
        let sp = sp_base + t;
        if (sp >= spatial) { break; }
        ho_a[t] = sp / params.w_out;
        wo_a[t] = sp % params.w_out;
        lanes = lanes + 1u;
    }

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    for (var ci_off: u32 = 0u; ci_off < c_in_per_g; ci_off = ci_off + 1u) {
        let ci = ci_start + ci_off;
        let in_base_ch = (nn * params.c_in + ci) * params.h;
        for (var kr: u32 = 0u; kr < params.kh; kr = kr + 1u) {
            for (var kc: u32 = 0u; kc < params.kw; kc = kc + 1u) {
                let w_idx = ((co * c_in_per_g + ci_off) * params.kh + kr) * params.kw + kc;
                let wv = arena[params.w_off + w_idx];        // loaded once, reused over lanes
                for (var t: u32 = 0u; t < lanes; t = t + 1u) {
                    let in_r_signed = i32(ho_a[t] * params.sh + kr * params.dh) - i32(params.ph);
                    let in_c_signed = i32(wo_a[t] * params.sw + kc * params.dw) - i32(params.pw);
                    if (in_r_signed < 0 || in_c_signed < 0
                        || in_r_signed >= i32(params.h)
                        || in_c_signed >= i32(params.w)) {
                        continue;
                    }
                    let in_idx = (in_base_ch + u32(in_r_signed)) * params.w + u32(in_c_signed);
                    acc[t] = acc[t] + arena[params.in_off + in_idx] * wv;
                }
            }
        }
    }

    let out_base_ch = (nn * params.c_out + co) * spatial;
    for (var t: u32 = 0u; t < lanes; t = t + 1u) {
        arena[params.out_off + out_base_ch + sp_base + t] = acc[t];
    }
}
