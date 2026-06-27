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

// Fused dense / soft-label softmax cross-entropy along the last axis.
// One thread per row computes, numerically stably,
//   loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]
// in a single pass over the class axis (one max pass + one fused
// sum-exp / dot pass). Slow-but-correct one-thread-per-row form,
// mirroring `softmax.wgsl`; a workgroup tree reduction is future work.

struct Params {
    outer: u32,        // N rows
    inner: u32,        // C classes
    logits_off: u32,
    targets_off: u32,
    out_off: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

@compute @workgroup_size(64)
fn softmax_cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let lbase = params.logits_off + row * params.inner;
    let tbase = params.targets_off + row * params.inner;

    // Pass 1: row max for numerical stability.
    var m: f32 = arena[lbase];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        m = max(m, arena[lbase + i]);
    }

    // Pass 2: Σ exp(x - max) and Σ targets·logits in one sweep.
    var s: f32 = 0.0;
    var dot: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[lbase + i];
        s = s + exp(v - m);
        dot = dot + arena[tbase + i] * v;
    }

    arena[params.out_off + row] = (m + log(s)) - dot;
}
