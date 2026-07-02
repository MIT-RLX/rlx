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

// Top-K indices along the last axis. Output: f32-encoded indices,
// shape `[..., k]`. One thread per outer row; each thread does k
// stepes of serial argmax over the inner dim, masking previously
// chosen entries with -infinity in a small per-thread tracking
// array.
//
// O(outer · k · inner) work — fine for the typical sampling shapes
// (k ≤ 64, inner = vocab). Larger workloads can revisit with a
// proper segmented sort once the rest of the IR coverage is in.
//
// Ties broken by smaller index, matching torch.topk(largest=True).

struct Params {
    outer: u32,
    inner: u32,
    k: u32,
    in_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const NEG_INF: f32 = -3.4e38;

@compute @workgroup_size(64)
fn topk(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer) { return; }
    let in_base  = params.in_off  + row * params.inner;
    let out_base = params.out_off + row * params.k;

    // Track which indices have been picked. We re-scan each step —
    // O(k * inner) per row, small for typical k.
    for (var step: u32 = 0u; step < params.k; step = step + 1u) {
        var best_v: f32 = NEG_INF;
        var best_i: u32 = 0u;
        for (var j: u32 = 0u; j < params.inner; j = j + 1u) {
            // Skip indices already picked in earlier passes.
            var taken: bool = false;
            for (var p: u32 = 0u; p < step; p = p + 1u) {
                let prev = u32(arena[out_base + p]);
                if (prev == j) { taken = true; break; }
            }
            if (taken) { continue; }
            let v = arena[in_base + j];
            if (v > best_v || (v == best_v && j < best_i)) {
                best_v = v;
                best_i = j;
            }
        }
        arena[out_base + step] = f32(best_i);
    }
}
