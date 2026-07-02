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

// Slice along an axis. Each output row maps to one input row at an
// offset within the chosen axis.
//
// Input shape  [..., outer, axis_size, inner]
// Output shape [..., outer, len,       inner]
//
// Index decomposition:
//   out_index = (outer_idx, off, inner_idx)
//   in_index  = (outer_idx, off + start, inner_idx)
// where start is the slice's starting position along the axis.

struct Params {
    out_total: u32,    // total output elements
    outer: u32,        // product of leading dims (above the narrow axis)
    inner: u32,        // product of trailing dims (below the narrow axis)
    axis_in_size: u32, // size of the input's narrow axis
    axis_out_size: u32,// size of the output's narrow axis (== len)
    start: u32,        // slice start along the axis
    in_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn narrow(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.out_total) { return; }
    // Decompose i into (outer_idx, axis_idx_out, inner_idx).
    let inner = params.inner;
    let axis_out = params.axis_out_size;
    let inner_idx = i % inner;
    let q1 = i / inner;
    let axis_idx = q1 % axis_out;
    let outer_idx = q1 / axis_out;
    // Recompose into the input's flat index using the input's axis size.
    let in_idx = (outer_idx * params.axis_in_size + (axis_idx + params.start)) * inner + inner_idx;
    arena[params.out_off + i] = arena[params.in_off + in_idx];
}
