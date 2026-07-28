// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Fused (residual add + optional bias add) + LayerNorm.
//
//   y = layer_norm(x + residual + [bias])
//
// One thread per outer row, sequential reduction over the inner dim.
// Compared to running Add → [Add] → LayerNorm as three separate
// kernels, this saves 2 dispatches and reads/writes the [outer, inner]
// arena slot twice instead of four times.
//
// Inputs (offsets in f32 elements):
//   in_off:       [outer, inner]
//   residual_off: [outer, inner]
//   bias_off:     [inner]   (only read when has_bias != 0)
//   gamma_off:    [inner]
//   beta_off:     [inner]
// Output:
//   out_off:      [outer, inner]

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    _p0: u32, _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn fused_residual_ln(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let out_base = params.out_off      + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1: fold residual + bias into the OUTPUT slot and accumulate
    // the mean. (The summed value x+residual+bias is materialized in
    // out_base, so the variance pass reads it back cheaply.)
    var sum_x: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[out_base + i] = v;
        sum_x = sum_x + v;
    }
    let mean = sum_x * n_inv;
    // Pass 1b: STABLE TWO-PASS variance = mean((x − mean)²). The one-pass
    // identity E[x²] − (E[x])² catastrophically cancels in f32 when the
    // row carries a large DC offset (pre-norm transformer activations),
    // collapsing the variance to near-zero and corrupting the norm on
    // wgpu only. Two-pass matches CPU/Metal/MLX/CoreML; the extra read
    // over `inner` is worth correctness.
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let d = arena[out_base + i] - mean;
        sum_sq = sum_sq + d * d;
    }
    let var_ = sum_sq * n_inv;
    let inv_std = inverseSqrt(var_ + eps);

    // Pass 2: normalize, scale, shift in place. (Was Pass 3.)
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        arena[out_base + i] = (arena[out_base + i] - mean) * inv_std * g + b;
    }
}
