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

// SAM2-style axial 2-D RoPE on `[batch, seq, num_heads * head_dim]`.
// Matches `rlx_ir::ops::axial_rope2d::apply_axial_rope2d` (per-batch plane).
// One thread per output element.

struct Params {
    batch: u32,
    seq: u32,
    hidden: u32,
    end_x: u32,
    end_y: u32,
    head_dim: u32,
    num_heads: u32,
    repeat_factor: u32,
    theta: f32,
    in_off: u32,
    out_off: u32,
    n_total: u32, // batch * seq * hidden
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn axial_rope2d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n_total) { return; }

    let d = i % params.hidden;
    let q1 = i / params.hidden;
    let tok = q1 % params.seq;
    let bi = q1 / params.seq;

    let head_dim = params.head_dim;
    let half = head_dim / 2u;
    let d_in_head = d % head_dim;
    let buf_idx = bi * params.seq * params.hidden + tok * params.hidden + d;
    let head_base = buf_idx - d_in_head;

    // Only even lanes of each rotated pair write both outputs (avoids races).
    if ((d_in_head & 1u) != 0u) {
        return;
    }

    let repeat = max(params.repeat_factor, 1u);
    let pos = tok / repeat;
    let tx = f32(pos % params.end_x);
    let ty = f32(pos / params.end_x);

    if (d_in_head < half) {
        // X-axis rotation on first half: pairs (2c, 2c+1).
        let c = d_in_head / 2u;
        let freq = 1.0 / pow(params.theta, f32(4u * c) / f32(head_dim));
        let ang = tx * freq;
        let co = cos(ang);
        let si = sin(ang);
        let ix0 = head_base + 2u * c;
        let ix1 = ix0 + 1u;
        let x0 = arena[params.in_off + ix0];
        let x1 = arena[params.in_off + ix1];
        arena[params.out_off + ix0] = x0 * co - x1 * si;
        arena[params.out_off + ix1] = x0 * si + x1 * co;
    } else {
        // Y-axis rotation on second half.
        let c = (d_in_head - half) / 2u;
        let freq = 1.0 / pow(params.theta, f32(4u * c) / f32(head_dim));
        let ang = ty * freq;
        let co = cos(ang);
        let si = sin(ang);
        let ix0 = head_base + half + 2u * c;
        let ix1 = ix0 + 1u;
        let x0 = arena[params.in_off + ix0];
        let x1 = arena[params.in_off + ix1];
        arena[params.out_off + ix0] = x0 * co - x1 * si;
        arena[params.out_off + ix1] = x0 * si + x1 * co;
    }
}
