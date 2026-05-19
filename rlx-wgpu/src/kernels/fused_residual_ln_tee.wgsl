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

// FusedResidualLN with a "tee" output: writes BOTH (h + residual + [bias])
// AND LN(h + residual + [bias]) to two separate arena slots. Used when
// the sum has multiple consumers downstream (the LN AND a later residual
// add) and the regular `fused_residual_ln` can't fire because of its
// single-consumer guard.
//
// Replaces the (Add → LN) sequence with one dispatch when Add is multi-
// consumer. Vision pre-norm transformers hit this 23× per forward
// (every residual feeds both the next LN and the *next-next* residual);
// w/o this kernel the wgpu schedule has 35 standalone Binary + 36
// standalone LayerNorm. With this kernel each (Add → LN) collapses to
// one FusedResidualLnTee.
//
// Memory layout:
//   sum_off:      [outer, inner]   ← (h + residual + [bias])
//   ln_out_off:   [outer, inner]   ← LN of the sum
// Other consumers of the sum read sum_off (= the eliminated Add's old
// arena slot, which the planner has already allocated for them).

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    residual_off: u32,
    bias_off: u32,
    gamma_off: u32,
    beta_off: u32,
    sum_off: u32,
    ln_out_off: u32,
    eps_bits: u32,
    has_bias: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn fused_residual_ln_tee(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base  = params.in_off       + row * params.inner;
    let res_base = params.residual_off + row * params.inner;
    let sum_base = params.sum_off      + row * params.inner;
    let out_base = params.ln_out_off   + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);
    let with_bias = params.has_bias != 0u;

    // Pass 1: write the SUM to sum_base AND accumulate stats in the
    // same loop (E[x²] − (E[x])² identity for variance — same trick as
    // the standard FRL kernel).
    var sum_x:   f32 = 0.0;
    var sum_x2:  f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[sum_base + i] = v;
        sum_x  = sum_x  + v;
        sum_x2 = sum_x2 + v * v;
    }
    let mean = sum_x * n_inv;
    // Clamp negative variance from f32 cancellation; see
    // fused_residual_ln.wgsl for the full rationale (inverseSqrt of
    // a non-positive value is undefined and returns NaN on NVIDIA).
    let var_ = max(sum_x2 * n_inv - mean * mean, 0.0);
    let inv_std = inverseSqrt(var_ + eps);

    // Pass 2: read sum_base, write LN result to out_base. Two distinct
    // slots — the sum stays available for the other consumer.
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        arena[out_base + i] = (arena[sum_base + i] - mean) * inv_std * g + b;
    }
}
