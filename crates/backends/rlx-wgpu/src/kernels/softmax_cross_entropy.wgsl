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
        s = s + epi_exp(v - m);
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
        s = s + epi_exp(arena[lbase + i] - m);
    }

    let label = u32(arena[params.targets_off + row]);
    let label_c = min(label, params.inner - 1u);
    arena[params.out_off + row] = (m + log(s)) - arena[lbase + label_c];
}

// `rlxsl::exact`'s correctly-rounded helpers, verbatim (`rlx_*` → `epi_*`): the
// builtin `/`, `inverseSqrt` and `exp` round by the compiler's math mode, so
// native wgpu (Apple fast math) and a browser (safe math) disagreed.
fn epi_exp(x: f32) -> f32 {
    let bx = bitcast<u32>(x);
    if ((bx & 0x7fffffffu) > 0x7f800000u) { return x; }
    let xc = clamp(x, -110.0, 89.0);
    let n = round(xc * 1.44269504);
    let r = fma(n, -1.428606765330187e-6, fma(n, -0.693145751953125, xc));
    var p = 1.9875691500e-4;
    p = fma(p, r, 1.3981999507e-3);
    p = fma(p, r, 8.3334519073e-3);
    p = fma(p, r, 4.1665795894e-2);
    p = fma(p, r, 1.6666665459e-1);
    p = fma(p, r, 5.0000001201e-1);
    return ldexp(1.0 + fma(p, r * r, r), i32(n));
}
