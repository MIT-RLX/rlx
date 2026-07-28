// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// DiT gated residual: out = x + gate * y  (gate broadcasts like adaLN scale).
// lead_pack: [lead_rank, x_lead[8], gate_lead[8]] packed as vec4s (uniform
// address space requires array stride multiple of 16 — plain `array<u32,N>`
// is illegal).

struct Params {
    outer: u32,
    inner: u32,
    x_off: u32,
    y_off: u32,
    gate_off: u32,
    out_off: u32,
    // Pad so lead_pack (vec4 array) starts at offset 32.
    _pre0: u32,
    _pre1: u32,
    lead_pack: array<vec4<u32>, 5>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

fn lead_at(i: u32) -> u32 {
    return params.lead_pack[i / 4u][i % 4u];
}

fn gate_base_for_row(row: u32, inner: u32) -> u32 {
    let lead_rank = lead_at(0u);
    var rem = row;
    var gate_base: u32 = 0u;
    var gate_stride: u32 = inner;
    var j: i32 = i32(lead_rank) - 1;
    loop {
        if (j < 0) { break; }
        var xd = lead_at(1u + u32(j));
        if (xd == 0u) { xd = 1u; }
        let xi = rem % xd;
        rem = rem / xd;
        var gd = lead_at(9u + u32(j));
        if (gd == 0u) { gd = 1u; }
        if (gd != 1u) {
            gate_base += xi * gate_stride;
        }
        gate_stride = gate_stride * gd;
        j = j - 1;
    }
    return gate_base;
}

@compute @workgroup_size(64)
fn gated_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let gid_flat = gid.x + gid.y * ngs.x * 64u;
    let total = params.outer * params.inner;
    if (gid_flat >= total || params.inner == 0u) { return; }
    let row = gid_flat / params.inner;
    let col = gid_flat % params.inner;
    let gate_base = gate_base_for_row(row, params.inner);
    let i = row * params.inner + col;
    arena[params.out_off + i] =
        arena[params.x_off + i]
        + arena[params.gate_off + gate_base + col] * arena[params.y_off + i];
}
