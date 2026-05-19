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

// Two-phase scatter-add. wgpu's MSL backend in naga 22 does not yet
// emit atomicCompareExchangeWeak, so the float-CAS atomic-add we use
// in the Metal backend is unavailable here. We instead serialize the
// accumulate phase to a single thread (workgroup_size = 1, dispatch =
// 1). Correct, but bandwidth-bound; worth revisiting once naga's MSL
// emitter catches up.

struct Params {
    op: u32,            // 0 = zero, 1 = accumulate
    out_off: u32,
    upd_off: u32,
    idx_off: u32,
    out_total: u32,
    num_updates: u32,
    trailing: u32,
    out_dim: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn scatter_add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;

    if (params.op == 0u) {
        if (i >= params.out_total) { return; }
        arena[params.out_off + i] = 0.0;
        return;
    }

    // Accumulate: serialize to thread 0 of workgroup 0. We dispatch
    // a single workgroup for this phase so only one thread runs the
    // whole loop. Slow for large num_updates, but correct without
    // atomic float-add support.
    if (i != 0u) { return; }
    for (var k: u32 = 0u; k < params.num_updates; k = k + 1u) {
        let row_f = arena[params.idx_off + k];
        let row = u32(row_f);
        if (row >= params.out_dim) { continue; }
        for (var j: u32 = 0u; j < params.trailing; j = j + 1u) {
            let v = arena[params.upd_off + k * params.trailing + j];
            arena[params.out_off + row * params.trailing + j] =
                arena[params.out_off + row * params.trailing + j] + v;
        }
    }
}
