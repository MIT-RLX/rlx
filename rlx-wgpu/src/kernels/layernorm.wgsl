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

// LayerNorm and RmsNorm fused into one kernel via op flag. Both
// reduce along the last axis (feature dim).
//
//   LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
//   RmsNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
//
// Inputs (offsets in f32 elements):
//   in_off:    [outer, inner]
//   gamma_off: [inner]
//   beta_off:  [inner]   (LayerNorm only; RmsNorm ignores)
// Output:
//   out_off:   [outer, inner]

struct Params {
    outer: u32,
    inner: u32,
    in_off: u32,
    out_off: u32,
    gamma_off: u32,
    beta_off: u32,
    eps_bits: u32,    // bitcast-encoded f32 eps
    op: u32,          // 0=LayerNorm, 1=RmsNorm
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn norm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.inner;
    let n_inv = 1.0 / f32(params.inner);
    let eps = bitcast<f32>(params.eps_bits);

    if (params.op == 0u) {
        // LayerNorm: fused mean + variance pass via E[x²] − (E[x])²
        // identity. One read pass over `inner` instead of two —
        // halves memory traffic for the mean+variance phase. f32
        // accumulation gives plenty of headroom for BERT-class
        // activations; same identity PyTorch's nn.LayerNorm uses.
        var sum_x:  f32 = 0.0;
        var sum_x2: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let v = arena[in_base + i];
            sum_x  = sum_x  + v;
            sum_x2 = sum_x2 + v * v;
        }
        let mean = sum_x * n_inv;
        // Clamp negative variance from f32 cancellation; see
        // fused_residual_ln.wgsl for the full rationale.
        let var_ = max(sum_x2 * n_inv - mean * mean, 0.0);
        let inv_std = inverseSqrt(var_ + eps);
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let g = arena[params.gamma_off + i];
            let b = arena[params.beta_off + i];
            arena[out_base + i] = (arena[in_base + i] - mean) * inv_std * g + b;
        }
    } else {
        // RmsNorm: divide by sqrt(mean(x^2) + eps), apply scale.
        var ss: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let v = arena[in_base + i];
            ss = ss + v * v;
        }
        let inv_rms = inverseSqrt(ss * n_inv + eps);
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let g = arena[params.gamma_off + i];
            arena[out_base + i] = arena[in_base + i] * inv_rms * g;
        }
    }
}
