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

// Softmax cross-entropy backward (integer labels), one thread per row:
//   dlogits[n,c] = (softmax(logits[n])[c] - [c==label]) * d_loss[n]

struct Params {
    outer: u32,
    inner: u32,
    logits_off: u32,
    labels_off: u32,
    d_loss_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn softmax_cross_entropy_backward(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let lbase = params.logits_off + row * params.inner;
    let obase = params.out_off + row * params.inner;

    var m: f32 = arena[lbase];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        m = max(m, arena[lbase + i]);
    }

    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        s = s + exp(arena[lbase + i] - m);
    }
    let inv = 1.0 / s;
    let scale = arena[params.d_loss_off + row];
    let label = u32(arena[params.labels_off + row]);
    let label_c = min(label, params.inner - 1u);

    for (var k: u32 = 0u; k < params.inner; k = k + 1u) {
        let p = exp(arena[lbase + k] - m) * inv;
        let oh = select(0.0, 1.0, k == label_c);
        arena[obase + k] = (p - oh) * scale;
    }
}
