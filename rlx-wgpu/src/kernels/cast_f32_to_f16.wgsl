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

enable f16;

// Pre-pass cast that mirrors a region of the f32 arena into the f16
// shadow buffer. Used before `matmul_coop16` to make matmul's A
// operand (a runtime activation, not a Param) readable as f16.
// One thread per element; arena_f16[off..off+len] gets the
// downcast of arena[off..off+len].

struct Params {
    src_off: u32,   // f32-element offset (also f16-element offset; packed densely)
    len: u32,       // element count
    _p0: u32, _p1: u32,
};

// Bind ordering matches `build_kernel_3`: 0=storage(rw), 1=uniform,
// 2=storage(ro). Source is read-only (arena), destination is the
// read-write target (arena_f16).
@group(0) @binding(0) var<storage, read_write> arena_f16: array<f16>;
@group(0) @binding(1) var<uniform>             params:    Params;
@group(0) @binding(2) var<storage, read>       arena:     array<f32>;

@compute @workgroup_size(64)
fn cast_f32_to_f16(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.len) { return; }
    arena_f16[params.src_off + i] = f16(arena[params.src_off + i]);
}
