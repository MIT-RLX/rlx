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

// Ternary-pruned radix-2 butterfly stage (interleaved C64 [batch, n_fft, 2]).
// Mirrors CUDA `fft_butterfly_stage.cu` / CPU `execute_fft_butterfly_stage_f32`.
// One thread per (batch, butterfly); gate=0 copies the pair, else twiddle + optional rev.
// Offsets are f32-ELEMENT offsets into the arena.

struct Params {
    batch: u32,
    n_fft: u32,
    stage: u32,
    half: u32,
    state_off: u32,
    out_off: u32,
    gate_off: u32,
    rev_off: u32,
    tw_re_off: u32,
    tw_im_off: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn fft_butterfly_stage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let idx = gid.x + gid.y * ngs.x * 64u;
    let n = params.batch * params.half;
    if (idx >= n) { return; }
    let b = idx / params.half;
    let bf = idx % params.half;

    let stride = 1u << params.stage;
    let row_elems = params.n_fft * 2u;
    let inp_base = params.state_off + b * row_elems;
    let out_base = params.out_off + b * row_elems;

    let group = bf / stride;
    let k = bf % stride;
    let i0 = group * 2u * stride + k;
    let i1 = i0 + stride;

    if (arena[params.gate_off + bf] == 0.0) {
        arena[out_base + i0 * 2u]      = arena[inp_base + i0 * 2u];
        arena[out_base + i0 * 2u + 1u] = arena[inp_base + i0 * 2u + 1u];
        arena[out_base + i1 * 2u]      = arena[inp_base + i1 * 2u];
        arena[out_base + i1 * 2u + 1u] = arena[inp_base + i1 * 2u + 1u];
        return;
    }

    let w_re = arena[params.tw_re_off + bf];
    let w_im = arena[params.tw_im_off + bf];
    let in_a_re = arena[inp_base + i0 * 2u];
    let in_a_im = arena[inp_base + i0 * 2u + 1u];
    let in_b_re = arena[inp_base + i1 * 2u];
    let in_b_im = arena[inp_base + i1 * 2u + 1u];

    let b_re = in_b_re * w_re - in_b_im * w_im;
    let b_im = in_b_re * w_im + in_b_im * w_re;
    let top_re = in_a_re + b_re;
    let top_im = in_a_im + b_im;
    let bot_re = in_a_re - b_re;
    let bot_im = in_a_im - b_im;

    var oa_re: f32;
    var oa_im: f32;
    var ob_re: f32;
    var ob_im: f32;
    if (arena[params.rev_off + bf] >= 0.5) {
        oa_re = bot_re; oa_im = bot_im;
        ob_re = top_re; ob_im = top_im;
    } else {
        oa_re = top_re; oa_im = top_im;
        ob_re = bot_re; ob_im = bot_im;
    }
    arena[out_base + i0 * 2u]      = oa_re;
    arena[out_base + i0 * 2u + 1u] = oa_im;
    arena[out_base + i1 * 2u]      = ob_re;
    arena[out_base + i1 * 2u + 1u] = ob_im;
}
