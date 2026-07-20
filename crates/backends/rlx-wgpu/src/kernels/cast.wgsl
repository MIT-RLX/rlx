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

// Numeric `Op::Cast` on the f32-uniform arena. Every tensor is stored as
// f32, so a cast is a per-element re-encode of the already-f32 value:
//
//   mode 0 = identity      — int→float / float→float / same-kind / Bool→num.
//                            The stored f32 already holds the correct value.
//   mode 1 = float → int   — truncate toward zero, then SATURATE to
//                            [lo, hi] (the destination int range). NaN → 0.
//                            Matches rlx-cpu (`x as iN`, which saturates).
//                            The int-valued result is stored back as f32.
//   mode 2 = → Bool        — store `1.0` when `value != 0`, else `0.0`.
//
// lo_bits / hi_bits carry the f32 clamp bounds for mode 1 (bit-reinterpreted
// so the host can pass exact endpoints such as i32::MIN).

struct Params {
    n: u32,
    in_off: u32,
    out_off: u32,
    mode: u32,
    lo_bits: u32,
    hi_bits: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn cast_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.n) { return; }
    let x = arena[params.in_off + i];
    var y: f32 = x;
    if (params.mode == 1u) {
        let lo = bitcast<f32>(params.lo_bits);
        let hi = bitcast<f32>(params.hi_bits);
        let t = clamp(trunc(x), lo, hi);
        y = select(t, 0.0, x != x); // NaN → 0
    } else if (params.mode == 2u) {
        y = select(0.0, 1.0, x != 0.0);
    }
    arena[params.out_off + i] = y;
}
