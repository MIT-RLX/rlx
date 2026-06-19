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

// Grouped (MoE) matmul. One workgroup per (M, N) tile; each thread
// computes one output element by reading the per-token expert id, then
// streaming K against weight[expert_id]. Naive correctness-first
// implementation — segmented/grouped GEMM optimization can come later.
//
// input       : [M, K]
// weight      : [num_experts, K, N]
// expert_idx  : [M]              (f32-encoded per-token expert id)
// output      : [M, N]
//   output[m, n] = input[m, :] · weight[expert_idx[m], :, n]

struct Params {
    m: u32,
    k: u32,
    n: u32,
    num_experts: u32,
    in_off: u32,
    w_off: u32,
    idx_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(8, 8)
fn grouped_matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }
    let e_f = arena[params.idx_off + row];
    let e   = u32(e_f);
    if (e >= params.num_experts) { return; }
    let w_base = params.w_off + e * params.k * params.n;
    let in_base = params.in_off + row * params.k;
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        acc = acc + arena[in_base + kk]
                  * arena[w_base + kk * params.n + col];
    }
    arena[params.out_off + row * params.n + col] = acc;
}
