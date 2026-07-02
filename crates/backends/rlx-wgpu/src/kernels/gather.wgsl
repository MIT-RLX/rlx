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

// Embedding-style gather along the leading axis.
//
// Input  shape [vocab, dim]  (i.e. axis_size = vocab, dim is "inner")
// Indices flat shape [n_idx]  (f32-encoded; we round to u32)
// Output shape [n_idx, dim]
//
// Each output element [i, d] = input[indices[i], d].

struct Params {
    n_out: u32,        // total output elements (n_idx * dim)
    n_idx: u32,        // number of indices
    dim: u32,          // inner axis size
    vocab: u32,        // input axis size (for clamping out-of-range indices)
    in_off: u32,       // offset of the embedding table
    idx_off: u32,      // offset of the indices array
    out_off: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let o = gid.x + gid.y * ngs.x * 64u;
    if (o >= params.n_out) { return; }
    let d = o % params.dim;
    let i = o / params.dim;
    let idx_f = arena[params.idx_off + i];
    var idx_u: u32 = u32(max(idx_f, 0.0));
    if (idx_u >= params.vocab) { idx_u = params.vocab - 1u; }
    arena[params.out_off + o] = arena[params.in_off + idx_u * params.dim + d];
}
