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
struct Params {
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dy_off: u32,
    cos_off: u32,
    sin_off: u32,
    dx_off: u32,
    cos_len: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn rope_bwd(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    let total = params.batch * params.seq * params.hidden;
    if (i >= total) { return; }

    let nh = params.hidden / params.head_dim;
    let d = i % params.head_dim;
    let q1 = i / params.head_dim;
    let hi = q1 % nh;
    let q2 = q1 / nh;
    let si = q2 % params.seq;
    let bi = q2 / params.seq;
    let rot_half = params.n_rot / 2u;
    let half_dh = params.head_dim / 2u;
    let tab_off = (si * half_dh) % max(params.cos_len, 1u);

    let dy_base = params.dy_off + bi * params.seq * params.hidden + si * params.hidden + hi * params.head_dim;
    let dx_base = params.dx_off + bi * params.seq * params.hidden + si * params.hidden + hi * params.head_dim;

    if (d < rot_half) {
        let y1 = arena[dy_base + d];
        let y2 = arena[dy_base + rot_half + d];
        let c = arena[params.cos_off + tab_off + d];
        let s = arena[params.sin_off + tab_off + d];
        arena[dx_base + d] = y1 * c + y2 * s;
        arena[dx_base + rot_half + d] = -y1 * s + y2 * c;
    } else if (d >= params.n_rot) {
        arena[dx_base + d] = arena[dy_base + d];
    }
}
