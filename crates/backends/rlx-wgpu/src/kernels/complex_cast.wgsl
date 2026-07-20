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

// Standalone complex `Op::Cast` on the f32-uniform arena, dispatched over the
// complex-element index `k ∈ [0, n)`. Representation:
//   C64  = 2 f32 lanes `[re, im]`           (8 B/elem)
//   C128 = 4 f32 lanes `[re_hi, re_lo, im_hi, im_lo]` df64  (16 B/elem)
//
// Every source of a real→complex cast comes from an f32 real (lo=0), so all
// six directions are pure lane MOVES — no compensated df64 arithmetic. The
// C128→C64 narrow drops the `lo` lanes (keeps `hi`), the widen sets them 0.
//
//   mode 0 real→C64 : out[2k]=in[k];   out[2k+1]=0
//   mode 1 C64→real : out[k]=in[2k]
//   mode 2 real→C128: out[4k]=in[k];   out[4k+1..3]=0
//   mode 3 C128→real: out[k]=in[4k]
//   mode 4 C64→C128 : out[4k]=in[2k]; out[4k+1]=0; out[4k+2]=in[2k+1]; out[4k+3]=0
//   mode 5 C128→C64 : out[2k]=in[4k]; out[2k+1]=in[4k+2]
//
// `in_off` / `out_off` are f32-element offsets (the start lane of each tensor).

struct Params {
    n: u32,        // number of complex elements
    in_off: u32,   // f32-element offset of the source
    out_off: u32,  // f32-element offset of the destination
    mode: u32,     // 0..5 (see table above)
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn complex_cast_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let k = gid.x + gid.y * ngs.x * 64u;
    if (k >= params.n) { return; }
    let i = params.in_off;
    let o = params.out_off;
    switch (params.mode) {
        case 0u: { // real → C64
            arena[o + 2u * k]      = arena[i + k];
            arena[o + 2u * k + 1u] = 0.0;
        }
        case 1u: { // C64 → real
            arena[o + k] = arena[i + 2u * k];
        }
        case 2u: { // real → C128
            arena[o + 4u * k]      = arena[i + k];
            arena[o + 4u * k + 1u] = 0.0;
            arena[o + 4u * k + 2u] = 0.0;
            arena[o + 4u * k + 3u] = 0.0;
        }
        case 3u: { // C128 → real
            arena[o + k] = arena[i + 4u * k];
        }
        case 4u: { // C64 → C128
            arena[o + 4u * k]      = arena[i + 2u * k];
            arena[o + 4u * k + 1u] = 0.0;
            arena[o + 4u * k + 2u] = arena[i + 2u * k + 1u];
            arena[o + 4u * k + 3u] = 0.0;
        }
        case 5u: { // C128 → C64
            arena[o + 2u * k]      = arena[i + 4u * k];
            arena[o + 2u * k + 1u] = arena[i + 4u * k + 2u];
        }
        default: {}
    }
}
