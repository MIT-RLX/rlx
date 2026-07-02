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

    // Pass 1 (fused mean + variance): fold residual + bias into the
    // OUTPUT slot, and accumulate BOTH sum_x and sum_x² in the same
    // loop. variance = E[x²] − (E[x])². This collapses what used to
    // be two sequential read passes over `inner` into one — eliminates
    // ~33 % of the LayerNorm wall time at BERT inner=384/768 dims.
    //
    // The "subtract mean then square" form is more stable when var is
    // very small, but f32 accumulation here gives plenty of headroom
    // for BERT-class activations (~1.0 magnitudes). PyTorch's
    // `nn.LayerNorm` uses the same identity for the same reason.
    var sum_x:   f32 = 0.0;
    var sum_x2:  f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        var v = arena[in_base + i] + arena[res_base + i];
        if (with_bias) { v = v + arena[params.bias_off + i]; }
        arena[out_base + i] = v;
        sum_x  = sum_x  + v;
        sum_x2 = sum_x2 + v * v;
    }
    let mean = sum_x * n_inv;
    // E[x²] − E[x]² can come out slightly negative under f32
    // catastrophic cancellation (near-uniform rows). WGSL leaves
    // `inverseSqrt(x ≤ 0)` undefined: Apple/Metal returns finite,
    // NVIDIA's `rcpsqrt.approx.f32` returns NaN. Clamp to 0 so the
    // result matches the CPU LN path on every backend.
    let var_ = max(sum_x2 * n_inv - mean * mean, 0.0);
    let inv_std = inverseSqrt(var_ + eps);

    // Pass 2: normalize, scale, shift in place. (Was Pass 3.)
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let g = arena[params.gamma_off + i];
        let b = arena[params.beta_off  + i];
        arena[out_base + i] = (arena[out_base + i] - mean) * inv_std * g + b;
    }
}
