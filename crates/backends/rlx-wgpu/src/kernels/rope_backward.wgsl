// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
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
    // The table's own last dimension; see the note at `tab_off`.
    cos_row_stride: u32,
    // GptJ pairing (adjacent lanes 2i / 2i+1) rather than NeoX rotate-half
    // (lane i with lane i + n_rot/2). Must match the forward `rope.wgsl`.
    interleaved: u32,
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
    // The cos/sin table stores exactly the rotation angles — `n_rot/2` per
    // token, NOT head_dim/2. For PARTIAL rope (n_rot < head_dim) a head_dim/2
    // stride overshoots into a later token's angles for every position ≥1.
    // Full rope has n_rot == head_dim, so rot_half == head_dim/2: unchanged.
    // The table's ACTUAL row width, passed in from its last dimension — not
    // head_dim/2 and not n_rot/2, both of which are wrong for half the models
    // in use. Hardcoding `rot_half` here matched every other backward kernel
    // and disagreed with the forward under partial rotation.
    let tab_off = (si * params.cos_row_stride) % max(params.cos_len, 1u);

    let dy_base = params.dy_off + bi * params.seq * params.hidden + si * params.hidden + hi * params.head_dim;
    let dx_base = params.dx_off + bi * params.seq * params.hidden + si * params.hidden + hi * params.head_dim;

    if (params.interleaved != 0u) {
        // GptJ: one invocation per PAIR — the even lane writes both halves.
        if (d < params.n_rot && (d & 1u) == 0u) {
            let j = d >> 1u;
            let y1 = arena[dy_base + 2u * j];
            let y2 = arena[dy_base + 2u * j + 1u];
            let c = arena[params.cos_off + tab_off + j];
            let s = arena[params.sin_off + tab_off + j];
            arena[dx_base + 2u * j] = y1 * c + y2 * s;
            arena[dx_base + 2u * j + 1u] = -y1 * s + y2 * c;
        } else if (d >= params.n_rot) {
            arena[dx_base + d] = arena[dy_base + d];
        }
    } else if (d < rot_half) {
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
