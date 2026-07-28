// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Softmax cross-entropy along the last axis (one thread per row).
//
// Dense / soft-label (`softmax_cross_entropy`):
//   loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]
// Integer labels (`softmax_cross_entropy_with_logits`):
//   loss[n] = logsumexp(logits[n]) - logits[n, label]

struct Params {
    outer: u32,        // N rows
    inner: u32,        // C classes
    logits_off: u32,
    targets_off: u32,  // dense targets [N,C], or labels [N] (f32-encoded)
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

    var m: f32 = arena[lbase];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        m = max(m, arena[lbase + i]);
    }

    var s: f32 = 0.0;
    var dot: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[lbase + i];
        s = s + exp(v - m);
        dot = dot + arena[tbase + i] * v;
    }

    arena[params.out_off + row] = (m + log(s)) - dot;
}

@compute @workgroup_size(64)
fn softmax_cross_entropy_with_logits(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer || params.inner == 0u) { return; }
    let lbase = params.logits_off + row * params.inner;

    var m: f32 = arena[lbase];
    for (var i: u32 = 1u; i < params.inner; i = i + 1u) {
        m = max(m, arena[lbase + i]);
    }

    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        s = s + exp(arena[lbase + i] - m);
    }

    let label = u32(arena[params.targets_off + row]);
    let label_c = min(label, params.inner - 1u);
    arena[params.out_off + row] = (m + log(s)) - arena[lbase + label_c];
}
