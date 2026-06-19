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

// Concat along an axis. Same shape decomposition as Narrow but in
// reverse: copy a single input into the output at a specified
// offset along the concat axis. Multi-input concat dispatches this
// kernel once per input with each input's start position.

struct Params {
    n_in: u32,           // total input elements
    outer: u32,          // product of leading dims (above concat axis)
    inner: u32,          // product of trailing dims (below concat axis)
    axis_in_size: u32,   // this input's size along the concat axis
    axis_out_size: u32,  // total output size along the concat axis
    start: u32,          // where this input's first axis-element lands in output
    in_off: u32,
    out_off: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn concat(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n_in) { return; }
    // Decompose i into (outer_idx, axis_idx, inner_idx) for the input.
    let inner = params.inner;
    let axis_in = params.axis_in_size;
    let inner_idx = i % inner;
    let q1 = i / inner;
    let axis_idx = q1 % axis_in;
    let outer_idx = q1 / axis_in;
    // Recompose into the OUTPUT's flat index, shifting the axis position by start.
    let out_idx = (outer_idx * params.axis_out_size + (axis_idx + params.start)) * inner + inner_idx;
    arena[params.out_off + out_idx] = arena[params.in_off + i];
}
