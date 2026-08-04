// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused SwiGLU: out = up * silu(gate), from a concatenated [.., 2*n_half] input.
// Keeps gate/up/silu(gate) in registers — replaces the Narrow+Silu+Mul decompose
// that wrote three intermediates to the arena per FFN. Matches the shared
// `fused_swiglu.cu` / Metal `fused_swiglu`.

struct Params {
    n_half: u32,
    outer: u32,       // rows; total output elements = outer * n_half
    gate_first: u32,  // 1 = gate in low half, up in high; 0 = up low, gate high
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn fused_swiglu(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let total = params.outer * params.n_half;
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= total) { return; }
    let row = i / params.n_half;
    let col = i % params.n_half;
    let base = params.in_off + row * (2u * params.n_half);
    var up: f32;
    var gate: f32;
    if (params.gate_first != 0u) {
        gate = arena[base + col];
        up = arena[base + params.n_half + col];
    } else {
        up = arena[base + col];
        gate = arena[base + params.n_half + col];
    }
    arena[params.out_off + i] = up * (gate / (1.0 + exp(-gate)));
}
