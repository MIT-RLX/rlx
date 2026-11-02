// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
        s = s + epi_exp(arena[lbase + i] - m);
    }
    let inv = epi_div_rn(1.0, s);
    let scale = arena[params.d_loss_off + row];
    let label = u32(arena[params.labels_off + row]);
    let label_c = min(label, params.inner - 1u);

    for (var k: u32 = 0u; k < params.inner; k = k + 1u) {
        let p = epi_exp(arena[lbase + k] - m) * inv;
        let oh = select(0.0, 1.0, k == label_c);
        arena[obase + k] = (p - oh) * scale;
    }
}

// `rlxsl::exact`'s correctly-rounded helpers, verbatim (`rlx_*` → `epi_*`): the
// builtin `/`, `inverseSqrt` and `exp` round by the compiler's math mode, so
// native wgpu (Apple fast math) and a browser (safe math) disagreed.
fn epi_div_rn(a: f32, b: f32) -> f32 {
    let k = abs(a / b) < 0x1p-100f;
    let a2 = select(a, a * 0x1p64f, k);
    let q = a2 / b;
    let ok = q != 0.0 && abs(q) <= 0x1.fffffep+127f;
    var c = select(q, q + fma(-b, q, a2) / b, ok);
    // The step rounds its correction before adding it, so near a midpoint
    // `c` depends on the first quotient. Move to the neighbour on the
    // quotient's side when the quotient lies past their midpoint (ties to
    // even): `r = a - b·c` is exact, and `r - b·u/2` has the sign of
    // `a - b·(c + n)/2`.
    if (ok && c != 0.0 && abs(c) < 0x1.fffffep+127f) {
        let r = fma(-b, c, a2);
        if (r != 0.0) {
            let up = (r > 0.0) == (b > 0.0);
            let cb = bitcast<u32>(c);
            let nb = select(cb - 1u, cb + 1u, up == (c > 0.0));
            let n = bitcast<f32>(nb);
            let t = fma(-b, (n - c) * 0.5, r);
            if ((t != 0.0 && ((t > 0.0) == (b > 0.0)) == up) || (t == 0.0 && (nb & 1u) == 0u)) {
                c = n;
            }
        }
    }
    return select(c, c * 0x1p-64f, k);
}

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
